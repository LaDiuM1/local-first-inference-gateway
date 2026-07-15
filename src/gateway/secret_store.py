"""Watchdog 운영 키를 Windows DPAPI로 암호화해 파일에 저장한다."""

import ctypes
import os
from ctypes import wintypes
from pathlib import Path

from gateway.file_io import atomic_write

_FILE_MAGIC = b"OAT-DPAPI-1\0"
_CRYPTPROTECT_UI_FORBIDDEN = 0x1
_CRYPTPROTECT_LOCAL_MACHINE = 0x4


class ProtectedSecretError(Exception):
    """운영 비밀을 보호하거나 복호화할 수 없다."""


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_byte))]


def protect_machine_secret(path: Path, secret: str) -> None:
    if not secret:
        raise ProtectedSecretError("secret must not be empty")
    encrypted = _crypt_protect(secret.encode("utf-8"))
    try:
        atomic_write(path, _FILE_MAGIC + encrypted)
    except OSError as error:
        raise ProtectedSecretError("protected secret could not be written") from error


def load_machine_secret(path: Path) -> str:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ProtectedSecretError("protected secret is unavailable") from error
    if not content.startswith(_FILE_MAGIC):
        raise ProtectedSecretError("protected secret has an invalid format")
    decrypted = _crypt_unprotect(content[len(_FILE_MAGIC) :])
    try:
        return decrypted.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ProtectedSecretError("protected secret has invalid text") from error


def _crypt_protect(content: bytes) -> bytes:
    _require_windows()
    input_buffer = ctypes.create_string_buffer(content)
    input_blob = _DataBlob(
        len(content), ctypes.cast(input_buffer, ctypes.POINTER(ctypes.c_byte))
    )
    output_blob = _DataBlob()
    flags = _CRYPTPROTECT_UI_FORBIDDEN | _CRYPTPROTECT_LOCAL_MACHINE
    succeeded = ctypes.windll.crypt32.CryptProtectData(
        ctypes.byref(input_blob),
        None,
        None,
        None,
        None,
        flags,
        ctypes.byref(output_blob),
    )
    if not succeeded:
        raise ProtectedSecretError("Windows could not protect the secret")
    return _copy_and_free(output_blob)


def _crypt_unprotect(content: bytes) -> bytes:
    _require_windows()
    input_buffer = ctypes.create_string_buffer(content)
    input_blob = _DataBlob(
        len(content), ctypes.cast(input_buffer, ctypes.POINTER(ctypes.c_byte))
    )
    output_blob = _DataBlob()
    succeeded = ctypes.windll.crypt32.CryptUnprotectData(
        ctypes.byref(input_blob),
        None,
        None,
        None,
        None,
        _CRYPTPROTECT_UI_FORBIDDEN,
        ctypes.byref(output_blob),
    )
    if not succeeded:
        raise ProtectedSecretError("Windows could not decrypt the secret")
    return _copy_and_free(output_blob)


def _copy_and_free(blob: _DataBlob) -> bytes:
    try:
        return ctypes.string_at(blob.pbData, blob.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(blob.pbData)


def _require_windows() -> None:
    if os.name != "nt":
        raise ProtectedSecretError("Windows DPAPI is required")
