"""외부 관리 API 없이 API 키를 발급·목록·폐기하는 로컬 CLI."""

import argparse
import sys
from pathlib import Path

from gateway.api_keys import ApiKeyNotFoundError, ApiKeyStore, ApiKeyStoreError
from gateway.paths import API_KEY_STORE_PATH
from gateway.secret_store import ProtectedSecretError, protect_machine_secret


def main(arguments: list[str] | None = None) -> int:
    parser = _build_parser()
    parsed = parser.parse_args(arguments)
    store_path = parsed.store or API_KEY_STORE_PATH
    store = ApiKeyStore(store_path)
    try:
        if parsed.command == "issue":
            return _issue(store, parsed.client, parsed.protected_output)
        if parsed.command == "list":
            return _list(store)
        if parsed.command == "revoke":
            return _revoke(store, parsed.key_id)
    except (
        ApiKeyStoreError,
        ApiKeyNotFoundError,
        ProtectedSecretError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    parser.error("a command is required")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage inference gateway API keys")
    # 관리자가 명시적으로 지정할 때만 다른 저장소를 쓴다 — 생략하면 운영 작업이 읽는 보호된 자리다.
    parser.add_argument(
        "--store", type=Path, help="use this API key store instead of the protected one"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    issue = commands.add_parser("issue", help="issue one service API key")
    issue.add_argument("--client", required=True, help="client service identifier")
    issue.add_argument(
        "--protected-output",
        type=Path,
        help="store the key with machine-scoped Windows DPAPI instead of displaying it",
    )

    commands.add_parser("list", help="list non-secret key metadata")
    revoke = commands.add_parser("revoke", help="revoke one key")
    revoke.add_argument("--key-id", required=True, help="non-secret key identifier")
    return parser


def _issue(store: ApiKeyStore, client: str, protected_output: Path | None) -> int:
    issued = store.issue(client)
    print(f"key_id: {issued.key_id}")
    print(f"client: {issued.client}")
    print(f"created_at: {issued.created_at}")
    if protected_output is not None:
        try:
            protect_machine_secret(protected_output, issued.api_key)
        except ProtectedSecretError:
            store.revoke(issued.key_id)
            raise
        print("api_key: stored with Windows DPAPI; plaintext was not displayed")
        return 0
    print(f"api_key: {issued.api_key}")
    print(
        "This plaintext key is shown once and cannot be recovered from the server store."
    )
    return 0


def _list(store: ApiKeyStore) -> int:
    summaries = store.list_keys()
    if not summaries:
        print("No API keys.")
        return 0
    print("KEY_ID\tCLIENT\tSTATUS\tCREATED_AT\tREVOKED_AT")
    for summary in summaries:
        revoked_at = summary.revoked_at or "-"
        print(
            f"{summary.key_id}\t{summary.client}\t{summary.status}\t"
            f"{summary.created_at}\t{revoked_at}"
        )
    return 0


def _revoke(store: ApiKeyStore, key_id: str) -> int:
    summary = store.revoke(key_id)
    print(f"revoked key_id: {summary.key_id}")
    print(f"client: {summary.client}")
    print(f"revoked_at: {summary.revoked_at}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
