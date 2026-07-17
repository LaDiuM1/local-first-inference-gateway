"""서비스별 OpenAI 방식 API 키의 발급·검증·목록·폐기와 영속 저장소."""

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, sleep

from gateway.file_io import atomic_write

STORE_VERSION = 1
API_KEY_PREFIX = "sk-oat-"
# 잠금 획득 상한 — CLI의 발급·폐기가 잡는 짧은 잠금은 충분히 기다리고, 멈춘
# 파일시스템은 무한 대기 대신 저장소 불가(503)로 끝낸다.
LOCK_TIMEOUT_SECONDS = 5.0
LOCK_RETRY_INTERVAL_SECONDS = 0.05
_KEY_ID_PATTERN = re.compile(r"^[0-9a-f]{16}$")
_KEY_PATTERN = re.compile(r"^sk-oat-([0-9a-f]{16})\.([A-Za-z0-9_-]{43})$")
_DUMMY_SALT = hashlib.sha256(b"openat-invalid-api-key-salt").digest()[:16]
_DUMMY_DIGEST = hashlib.sha256(b"openat-invalid-api-key-digest").digest()
_PROCESS_LOCKS: dict[Path, threading.RLock] = {}
_PROCESS_LOCKS_GUARD = threading.Lock()


class ApiKeyStoreError(Exception):
    """키 저장소를 안전하게 읽거나 쓸 수 없다."""


class ApiKeyNotFoundError(Exception):
    """폐기하려는 키 식별자가 저장소에 없다."""


@dataclass(frozen=True)
class IssuedApiKey:
    key_id: str
    client: str
    api_key: str = field(repr=False)
    created_at: str


@dataclass(frozen=True)
class ApiKeyIdentity:
    key_id: str
    client: str


@dataclass(frozen=True)
class ApiKeySummary:
    key_id: str
    client: str
    created_at: str
    revoked_at: str | None

    @property
    def status(self) -> str:
        if self.revoked_at is None:
            return "active"
        return "revoked"


class ApiKeyStore:
    """원본 키 없이 salt와 SHA-256 digest만 보관하는 파일 기반 키 저장소."""

    def __init__(self, path: Path) -> None:
        self._path = path.resolve()
        self._lock_path = self._path.with_suffix(f"{self._path.suffix}.lock")

    def validate(self) -> None:
        with self._locked():
            self._read_document()

    def issue(self, client: str) -> IssuedApiKey:
        normalized_client = _validate_client(client)
        with self._locked():
            document = self._read_document()
            existing_ids = {entry["key_id"] for entry in document["keys"]}
            key_id = _new_key_id(existing_ids)
            api_key = f"{API_KEY_PREFIX}{key_id}.{secrets.token_urlsafe(32)}"
            salt = secrets.token_bytes(16)
            created_at = _utc_now()
            document["keys"].append(
                {
                    "key_id": key_id,
                    "client": normalized_client,
                    "salt": _encode_bytes(salt),
                    "digest": _encode_bytes(_digest(api_key, salt)),
                    "created_at": created_at,
                    "revoked_at": None,
                }
            )
            self._write_document(document)
        return IssuedApiKey(key_id, normalized_client, api_key, created_at)

    def authenticate(self, api_key: str) -> ApiKeyIdentity | None:
        key_id = _extract_key_id(api_key)
        with self._locked():
            document = self._read_document()
            entry = next(
                (item for item in document["keys"] if item["key_id"] == key_id),
                None,
            )

        salt = _DUMMY_SALT
        expected_digest = _DUMMY_DIGEST
        if entry is not None:
            salt = _decode_bytes(entry["salt"])
            expected_digest = _decode_bytes(entry["digest"])
        candidate_digest = _digest(api_key, salt)
        digest_matches = hmac.compare_digest(candidate_digest, expected_digest)
        if entry is None or entry["revoked_at"] is not None or not digest_matches:
            return None
        return ApiKeyIdentity(entry["key_id"], entry["client"])

    def list_keys(self) -> list[ApiKeySummary]:
        with self._locked():
            document = self._read_document()
        return [
            ApiKeySummary(
                key_id=entry["key_id"],
                client=entry["client"],
                created_at=entry["created_at"],
                revoked_at=entry["revoked_at"],
            )
            for entry in document["keys"]
        ]

    def revoke(self, key_id: str) -> ApiKeySummary:
        if not _KEY_ID_PATTERN.fullmatch(key_id):
            raise ApiKeyNotFoundError("API key identifier was not found")
        with self._locked():
            document = self._read_document()
            entry = next(
                (item for item in document["keys"] if item["key_id"] == key_id),
                None,
            )
            if entry is None:
                raise ApiKeyNotFoundError("API key identifier was not found")
            if entry["revoked_at"] is None:
                entry["revoked_at"] = _utc_now()
                self._write_document(document)
            return ApiKeySummary(
                key_id=entry["key_id"],
                client=entry["client"],
                created_at=entry["created_at"],
                revoked_at=entry["revoked_at"],
            )

    def _read_document(self) -> dict:
        if not self._path.exists():
            return {"version": STORE_VERSION, "keys": []}
        try:
            document = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ApiKeyStoreError("API key store is unavailable") from error
        _validate_document(document)
        return document

    def _write_document(self, document: dict) -> None:
        encoded = (
            json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        try:
            atomic_write(self._path, encoded)
        except OSError as error:
            raise ApiKeyStoreError("API key store could not be updated") from error

    @contextmanager
    def _locked(self) -> Iterator[None]:
        # 잠금 대기에 상한을 둔다 — 저장소가 멈춰도 인증 요청이 무한 대기로 워커를
        # 붙들지 않고 fail-closed(503)로 끝난다.
        process_lock = _process_lock_for(self._lock_path)
        if not process_lock.acquire(timeout=LOCK_TIMEOUT_SECONDS):
            raise ApiKeyStoreError("API key store lock is unavailable")
        try:
            try:
                self._lock_path.parent.mkdir(parents=True, exist_ok=True)
                with self._lock_path.open("a+b") as handle:
                    if handle.tell() == 0:
                        handle.write(b"\0")
                        handle.flush()
                    handle.seek(0)
                    _lock_file(handle)
                    try:
                        yield
                    finally:
                        handle.seek(0)
                        _unlock_file(handle)
            except ApiKeyStoreError:
                raise
            except OSError as error:
                raise ApiKeyStoreError("API key store lock is unavailable") from error
        finally:
            process_lock.release()


def _process_lock_for(path: Path) -> threading.RLock:
    with _PROCESS_LOCKS_GUARD:
        return _PROCESS_LOCKS.setdefault(path, threading.RLock())


def _lock_file(handle: object) -> None:
    """파일 잠금을 상한 안에서 획득한다 — 비차단 시도를 반복하고 초과 시 OSError."""
    deadline = monotonic() + LOCK_TIMEOUT_SECONDS
    if os.name == "nt":
        import msvcrt

        while True:
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                if monotonic() >= deadline:
                    raise
                sleep(LOCK_RETRY_INTERVAL_SECONDS)
    import fcntl

    while True:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except OSError:
            if monotonic() >= deadline:
                raise
            sleep(LOCK_RETRY_INTERVAL_SECONDS)


def _unlock_file(handle: object) -> None:
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validate_document(document: object) -> None:
    if not isinstance(document, dict):
        raise ApiKeyStoreError("API key store has an invalid format")
    if document.get("version") != STORE_VERSION or not isinstance(
        document.get("keys"), list
    ):
        raise ApiKeyStoreError("API key store has an unsupported format")
    seen_ids: set[str] = set()
    for entry in document["keys"]:
        if not _is_valid_entry(entry) or entry["key_id"] in seen_ids:
            raise ApiKeyStoreError("API key store has an invalid entry")
        seen_ids.add(entry["key_id"])


def _is_valid_entry(entry: object) -> bool:
    if not isinstance(entry, dict):
        return False
    if not _KEY_ID_PATTERN.fullmatch(entry.get("key_id", "")):
        return False
    if not _is_valid_client(entry.get("client")):
        return False
    if not isinstance(entry.get("created_at"), str):
        return False
    if entry.get("revoked_at") is not None and not isinstance(
        entry.get("revoked_at"), str
    ):
        return False
    try:
        salt = _decode_bytes(entry.get("salt"))
        digest = _decode_bytes(entry.get("digest"))
    except ApiKeyStoreError:
        return False
    return len(salt) == 16 and len(digest) == hashlib.sha256().digest_size


def _validate_client(client: str) -> str:
    normalized = client.strip()
    if not _is_valid_client(normalized):
        raise ValueError("client must be 1-100 printable characters")
    return normalized


def _is_valid_client(client: object) -> bool:
    return (
        isinstance(client, str)
        and 1 <= len(client) <= 100
        and all(character.isprintable() for character in client)
    )


def _new_key_id(existing_ids: set[str]) -> str:
    while True:
        key_id = secrets.token_hex(8)
        if key_id not in existing_ids:
            return key_id


def _extract_key_id(api_key: str) -> str | None:
    if len(api_key) > 256:
        return None
    match = _KEY_PATTERN.fullmatch(api_key)
    if match is None:
        return None
    return match.group(1)


def _digest(api_key: str, salt: bytes) -> bytes:
    return hashlib.sha256(salt + api_key.encode("utf-8", "replace")).digest()


def _encode_bytes(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _decode_bytes(value: object) -> bytes:
    if not isinstance(value, str):
        raise ApiKeyStoreError("API key store contains invalid binary data")
    try:
        return base64.b64decode(value, validate=True)
    except ValueError as error:
        raise ApiKeyStoreError("API key store contains invalid binary data") from error


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
