"""서비스별 API 키 저장소와 로컬 관리 CLI 계약, 운영 저장소 경로·소유권 고정 테스트."""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from gateway import file_io, key_cli
from gateway.api_keys import ApiKeyStore, ApiKeyStoreError
from gateway.config import Settings
from gateway.file_io import atomic_write
from gateway.key_cli import main
from gateway.main import create_app
from gateway.paths import API_KEY_STORE_PATH, STATE_DIRECTORY, WATCHDOG_KEY_PATH
from gateway.secret_store import ProtectedSecretError, load_machine_secret


def test_issue_authenticate_list_and_revoke_without_storing_plaintext(
    tmp_path: Path,
) -> None:
    store_path = tmp_path / "api-keys.json"
    store = ApiKeyStore(store_path)

    issued = store.issue("catalog-service")

    assert issued.api_key.startswith(f"sk-oat-{issued.key_id}.")
    assert store.authenticate(issued.api_key).client == "catalog-service"
    assert issued.api_key not in store_path.read_text(encoding="utf-8")
    assert issued.api_key.split(".", 1)[1] not in store_path.read_text(encoding="utf-8")
    assert store.list_keys()[0].status == "active"

    revoked = store.revoke(issued.key_id)

    assert revoked.status == "revoked"
    assert store.authenticate(issued.api_key) is None


def test_clients_have_independent_keys_and_revocation(tmp_path: Path) -> None:
    store = ApiKeyStore(tmp_path / "api-keys.json")
    catalog = store.issue("catalog-service")
    chatbot = store.issue("chatbot-service")

    store.revoke(catalog.key_id)

    assert store.authenticate(catalog.api_key) is None
    assert store.authenticate(chatbot.api_key).client == "chatbot-service"


def test_authentication_uses_constant_time_digest_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ApiKeyStore(tmp_path / "api-keys.json")
    issued = store.issue("service")
    calls = 0

    from gateway import api_keys

    original = api_keys.hmac.compare_digest

    def tracking_compare(left: bytes, right: bytes) -> bool:
        nonlocal calls
        calls += 1
        return original(left, right)

    monkeypatch.setattr(api_keys.hmac, "compare_digest", tracking_compare)

    assert store.authenticate(issued.api_key) is not None
    assert store.authenticate("sk-oat-0000000000000000." + "A" * 43) is None
    assert store.authenticate("malformed") is None
    assert calls == 3


def test_concurrent_issuance_keeps_store_complete(tmp_path: Path) -> None:
    store_path = tmp_path / "api-keys.json"

    def issue(index: int) -> str:
        return ApiKeyStore(store_path).issue(f"service-{index}").api_key

    with ThreadPoolExecutor(max_workers=8) as executor:
        keys = list(executor.map(issue, range(24)))

    store = ApiKeyStore(store_path)
    assert len(store.list_keys()) == 24
    assert all(store.authenticate(api_key) is not None for api_key in keys)


def test_concurrent_processes_do_not_lose_key_records(tmp_path: Path) -> None:
    store_path = tmp_path / "api-keys.json"
    code = (
        "from pathlib import Path; from gateway.api_keys import ApiKeyStore; "
        f"ApiKeyStore(Path({str(store_path)!r})).issue('process-client')"
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(8)
    ]

    return_codes = [process.wait(timeout=30) for process in processes]

    assert return_codes == [0] * 8
    assert len(ApiKeyStore(store_path).list_keys()) == 8


def test_atomic_write_failure_preserves_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "state.json"
    target.write_bytes(b"complete")

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated"):
        atomic_write(target, b"partial replacement")

    assert target.read_bytes() == b"complete"
    assert list(tmp_path.glob("*.tmp")) == []


def test_corrupt_store_fails_closed(tmp_path: Path) -> None:
    store_path = tmp_path / "api-keys.json"
    store_path.write_text('{"version": 1, "keys": [', encoding="utf-8")

    with pytest.raises(ApiKeyStoreError):
        ApiKeyStore(store_path).authenticate("sk-oat-0000000000000000." + "A" * 43)


def test_cli_lists_and_revokes_without_printing_key_material(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store_path = tmp_path / "api-keys.json"
    assert (
        main(
            [
                "--store",
                str(store_path),
                "issue",
                "--client",
                "catalog-service",
            ]
        )
        == 0
    )
    issue_output = capsys.readouterr().out
    api_key = next(
        line.removeprefix("api_key: ")
        for line in issue_output.splitlines()
        if line.startswith("api_key: sk-oat-")
    )
    key_id = ApiKeyStore(store_path).list_keys()[0].key_id

    assert main(["--store", str(store_path), "list"]) == 0
    list_output = capsys.readouterr().out
    assert api_key not in list_output
    assert "digest" not in list_output.lower()
    assert "salt" not in list_output.lower()

    assert main(["--store", str(store_path), "revoke", "--key-id", key_id]) == 0
    revoke_output = capsys.readouterr().out
    assert api_key not in revoke_output
    assert ApiKeyStore(store_path).authenticate(api_key) is None


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI contract")
def test_cli_can_store_watchdog_key_with_dpapi_without_displaying_plaintext(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store_path = tmp_path / "api-keys.json"
    protected_path = tmp_path / "watchdog-key.dpapi"

    assert (
        main(
            [
                "--store",
                str(store_path),
                "issue",
                "--client",
                "watchdog",
                "--protected-output",
                str(protected_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    api_key = load_machine_secret(protected_path)

    assert api_key.startswith("sk-oat-")
    assert api_key not in output
    assert api_key.encode() not in protected_path.read_bytes()
    assert ApiKeyStore(store_path).authenticate(api_key).client == "watchdog"


# --- 운영 저장소 경로 고정: 설치기가 잠근 자리 밖을 인증 근거로 삼을 수 없다 ---


def test_gateway_reads_the_protected_store_and_env_cannot_move_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 배포 사본의 .env는 작업 트리에서 복사되므로 비관리자가 내용을 정할 수 있다. 저장소 자리가
    # 설정이면 그 한 줄이 사용자가 쓸 수 있는 파일을 인증 근거로 만들어 아무 키나 통과시킨다.
    planted_store = tmp_path / "planted-keys.json"
    monkeypatch.setenv("GATEWAY_API_KEY_STORE_PATH", str(planted_store))

    configuration = Settings(_env_file=None)
    app = create_app()

    assert not hasattr(configuration, "api_key_store_path")
    assert app.state.api_key_store_path == API_KEY_STORE_PATH
    assert API_KEY_STORE_PATH == STATE_DIRECTORY / "api-keys.json"


def test_env_file_cannot_redirect_the_gateway_store(tmp_path: Path) -> None:
    deployed_env = tmp_path / ".env"
    deployed_env.write_text(
        f"GATEWAY_API_KEY_STORE_PATH={tmp_path / 'planted-keys.json'}\n",
        encoding="utf-8",
    )

    configuration = Settings(_env_file=deployed_env)

    assert not hasattr(configuration, "api_key_store_path")


def test_cli_without_store_option_uses_the_protected_store_the_gateway_reads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 실제 %ProgramData% 저장소를 건드리지 않고 CLI가 고르는 자리만 확인한다. 관리자가 명시한
    # --store만 다른 자리를 쓴다 — 운영 작업과 분리된 도구 계약이다.
    monkeypatch.setenv("GATEWAY_API_KEY_STORE_PATH", str(tmp_path / "planted.json"))
    chosen: list[Path] = []

    def record_store(path: Path) -> ApiKeyStore:
        chosen.append(path)
        return ApiKeyStore(tmp_path / "api-keys.json")

    monkeypatch.setattr(key_cli, "ApiKeyStore", record_store)

    assert main(["list"]) == 0
    assert main(["--store", str(tmp_path / "explicit.json"), "list"]) == 0
    capsys.readouterr()

    assert chosen == [API_KEY_STORE_PATH, tmp_path / "explicit.json"]


def _state_paths_reported_by_a_new_process(program_data: str | None) -> list[str]:
    # 작업이 시작될 때와 같다 — 프로세스가 자기 환경으로 자리를 정한다.
    environment = dict(os.environ)
    if program_data is None:
        environment.pop("PROGRAMDATA", None)
    else:
        environment["PROGRAMDATA"] = program_data
    code = (
        "import json; from gateway import paths; "
        "print(json.dumps([str(paths.STATE_DIRECTORY), str(paths.API_KEY_STORE_PATH), "
        "str(paths.WATCHDOG_KEY_PATH)]))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return json.loads(completed.stdout)


@pytest.mark.skipif(os.name != "nt", reason="Windows 공용 데이터 폴더 계약")
@pytest.mark.parametrize(
    "program_data",
    [str(Path.home() / "planted"), "", None],
    ids=["redirected", "empty", "removed"],
)
def test_program_data_environment_cannot_move_the_operational_state_paths(
    program_data: str | None,
) -> None:
    # 설치기는 관리자 셸의 환경으로, 작업은 SYSTEM의 환경으로 실행된다. 자리를 환경에서 읽으면 두 환경이
    # 다를 때 잠근 자리와 인증 근거로 읽는 자리가 어긋나, 잠그지 않은 자리에 미리 놓인 저장소를 채택한다.
    expected = [str(STATE_DIRECTORY), str(API_KEY_STORE_PATH), str(WATCHDOG_KEY_PATH)]

    reported = _state_paths_reported_by_a_new_process(program_data)

    assert reported == expected


# --- 운영 상태 파일 소유권: 저장소를 바꿔도 Administrators 소유가 유지된다 ---


def test_protected_state_writes_take_ownership_before_replacing_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 소유자는 교체 전 임시 파일에 건다 — 교체한 뒤에 걸면 옮기지 못한 순간 운영 저장소가 이미 그 파일을
    # 쓴 개별 관리자 계정 소유로 남는다.
    target = tmp_path / "api-keys.json"
    monkeypatch.setattr(file_io, "PROTECTED_STATE_PATHS", frozenset({target.resolve()}))
    steps: list[str] = []
    monkeypatch.setattr(
        file_io,
        "_take_administrative_ownership",
        lambda path: steps.append("owner"),
    )
    original_replace = os.replace

    def record_replace(source: str, destination: str) -> None:
        steps.append("replace")
        original_replace(source, destination)

    monkeypatch.setattr(os, "replace", record_replace)

    atomic_write(target, b"content")

    assert steps == ["owner", "replace"]
    assert target.read_bytes() == b"content"


def test_ownership_failure_keeps_the_previous_store_instead_of_publishing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store_path = tmp_path / "api-keys.json"
    store_path.write_text('{"version": 1, "keys": []}', encoding="utf-8")
    monkeypatch.setattr(
        file_io, "PROTECTED_STATE_PATHS", frozenset({store_path.resolve()})
    )

    def fail_ownership(path: Path) -> None:
        raise OSError("simulated ownership failure")

    monkeypatch.setattr(file_io, "_take_administrative_ownership", fail_ownership)

    with pytest.raises(ApiKeyStoreError):
        ApiKeyStore(store_path).issue("catalog-service")

    assert json.loads(store_path.read_text(encoding="utf-8"))["keys"] == []
    assert list(tmp_path.glob("*.tmp")) == []


def test_documented_key_management_restores_ownership_of_the_operational_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # 발급·폐기는 저장소를 바꾸는 문서화된 일반 경로다. 설치기만 소유자를 되돌리면, 그 사이 실행한 키
    # 관리가 저장소를 실행 계정 소유로 남겨 그 계정의 권한 상승 없는 프로세스가 DACL을 다시 써 키를
    # 심을 수 있다.
    store_path = tmp_path / "api-keys.json"
    monkeypatch.setattr(
        file_io, "PROTECTED_STATE_PATHS", frozenset({store_path.resolve()})
    )
    owned: list[Path] = []
    monkeypatch.setattr(
        file_io, "_take_administrative_ownership", lambda path: owned.append(path)
    )

    assert main(["--store", str(store_path), "issue", "--client", "catalog"]) == 0
    key_id = ApiKeyStore(store_path).list_keys()[0].key_id
    assert main(["--store", str(store_path), "revoke", "--key-id", key_id]) == 0
    capsys.readouterr()

    assert len(owned) == 2
    assert all(path.parent == tmp_path for path in owned)


def test_failed_protected_output_revokes_the_unrecoverable_key(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_path = tmp_path / "api-keys.json"

    def fail_protection(path: Path, secret: str) -> None:
        raise ProtectedSecretError("simulated protection failure")

    monkeypatch.setattr("gateway.key_cli.protect_machine_secret", fail_protection)

    result = main(
        [
            "--store",
            str(store_path),
            "issue",
            "--client",
            "watchdog",
            "--protected-output",
            str(tmp_path / "secret.dpapi"),
        ]
    )
    output = capsys.readouterr()
    summaries = ApiKeyStore(store_path).list_keys()

    assert result == 1
    assert len(summaries) == 1
    assert summaries[0].status == "revoked"
    assert "sk-oat-" not in output.out
    assert "sk-oat-" not in output.err
