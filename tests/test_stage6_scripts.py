"""6단계 PowerShell 작업의 구문, 무변경 검증, loopback 실행 계약 테스트.

SYSTEM 실행 경로 검사는 실제 ACL을 바꾸지 않고 확인한다 — 안전한 쪽은 기본 설치가 관리 주체에게만
쓰기를 허용하는 Windows 시스템 경로를, 안전하지 않은 쪽은 pytest의 임시 디렉터리를 그대로 쓴다.
import 경로 격리는 임시 디렉터리에 배포 사본과 원본 트리를 흉내 내 실제 Python 실행 결과로 확인한다.
"""

import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from gateway.paths import API_KEY_STORE_PATH, STATE_DIRECTORY, WATCHDOG_KEY_PATH

POWERSHELL = shutil.which("powershell") or (
    r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
)
ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
# 기본 ACL이 TrustedInstaller 소유에 관리 주체 외 읽기·실행만 허용하는 경로.
ADMIN_ONLY_SYSTEM_FILE = Path(POWERSHELL)
ADMIN_ONLY_SYSTEM_TREE = Path(r"C:\Windows\System32\WindowsPowerShell")
UNSAFE_EXECUTION_PATH_MARKER = "UNSAFE SYSTEM 실행 경로"
EXTERNAL_IMPORT_PATH_MARKER = "UNSAFE SYSTEM import 경로"
UNSAFE_STATE_PATH_MARKER = "UNSAFE 운영 상태 파일"
UNADOPTABLE_STATE_DIRECTORY_MARKER = "UNSAFE 운영 상태 디렉터리"
STAGE6_TASK_NAMES = ["Gateway", "Chat Ollama", "Embedding Ollama", "Watchdog"]

pytestmark = pytest.mark.skipif(
    platform.system() != "Windows" or not Path(POWERSHELL).exists(),
    reason="Windows PowerShell 운영 계약",
)


def _run(command: str, timeout: float = 60) -> subprocess.CompletedProcess[bytes]:
    # 한국어 기본 콘솔(cp949)에서도 스크립트가 내는 한국어를 그대로 받아 본다.
    utf8_console = "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
    return subprocess.run(
        [
            POWERSHELL,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            utf8_console + command,
        ],
        cwd=ROOT,
        capture_output=True,
        timeout=timeout,
    )


def _array(arguments: list[str]) -> str:
    return "@(" + ",".join(f"'{value}'" for value in arguments) + ")"


def _text(completed: subprocess.CompletedProcess[bytes]) -> str:
    return completed.stdout.decode("utf-8", "replace")


def _dot_source(script: str, command: str) -> subprocess.CompletedProcess[bytes]:
    return _run(f". '{SCRIPTS / script}'; {command}")


def _plan(command: str) -> list[str]:
    # 계획은 이름의 순서 있는 목록이다 — 순서 자체가 계약이므로 그대로 받아 본다.
    completed = _dot_source("stage6_tasks.ps1", f"({command}) -join '|'")
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    reported = _text(completed).strip()
    if not reported:
        return []
    return reported.split("|")


def test_all_stage6_powershell_files_parse_without_errors() -> None:
    files = [
        SCRIPTS / name
        for name in [
            "deployment_swap.ps1",
            "execution_security.ps1",
            "install_stage6.ps1",
            "run_chat_ollama.ps1",
            "run_gateway.ps1",
            "run_watchdog.ps1",
            "show_stage6_status.ps1",
            "stage6_tasks.ps1",
            "supervisor_common.ps1",
            "task_control.ps1",
        ]
    ]
    joined = ",".join(f"'{path}'" for path in files)
    command = (
        f"$failed=$false; @({joined}) | ForEach-Object {{ "
        "$tokens=$null; $errors=$null; "
        "[System.Management.Automation.Language.Parser]::ParseFile($_,[ref]$tokens,[ref]$errors) | Out-Null; "
        "if($errors.Count -gt 0){$failed=$true} }; if($failed){exit 1}"
    )

    result = _run(command)

    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")


def test_task_control_reports_missing_targets_without_mutation() -> None:
    script = SCRIPTS / "task_control.ps1"
    task = _run(
        f"& '{script}' -Action State -Kind Task -Name 'OpenAt-Test-Missing-Task'"
    )
    service = _run(
        f"& '{script}' -Action State -Kind Service -Name 'OpenAt-Test-Missing-Service'"
    )

    assert task.returncode == 0
    assert task.stdout.decode("utf-8", "replace").strip() == "Missing"
    assert service.returncode == 0
    assert service.stdout.decode("utf-8", "replace").strip() == "Missing"


def test_task_control_refuses_to_stop_a_service_for_a_deployment_swap() -> None:
    # cloudflared 공식 서비스는 배포 사본을 쓰지 않는다 — 교체를 이유로 중지할 대상이 아니다.
    script = SCRIPTS / "task_control.ps1"

    result = _run(f"& '{script}' -Action Stop -Kind Service -Name 'cloudflared'")

    assert result.returncode != 0
    assert "서비스 중지는 이 스크립트가 다루지 않는다" in result.stderr.decode(
        "utf-8", "replace"
    )


def test_installer_validate_only_is_deterministic_and_non_mutating(
    tmp_path: Path,
) -> None:
    fake_ollama = tmp_path / "ollama.exe"
    fake_ollama.touch()
    models = tmp_path / "models"
    models.mkdir()
    script = SCRIPTS / "install_stage6.ps1"
    command = (
        f"& '{script}' -ValidateOnly -ProjectRoot '{ROOT}' "
        f"-OllamaPath '{fake_ollama}' -OllamaModelsPath '{models}'"
    )

    result = _run(command)

    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    output = result.stdout.decode("utf-8", "replace")
    assert "VALID project" in output
    assert "scheduled task" in output


def test_chat_ollama_runner_sets_loopback_gpu_and_cloud_contract(
    tmp_path: Path,
) -> None:
    output = tmp_path / "environment.json"
    fake_child = tmp_path / "write_environment.py"
    fake_child.write_text(
        """\
import json, os, sys
names = ["OLLAMA_HOST", "OLLAMA_CONTEXT_LENGTH", "OLLAMA_MAX_LOADED_MODELS", "OLLAMA_NUM_PARALLEL", "OLLAMA_NO_CLOUD", "OLLAMA_API_KEY", "CUDA_VISIBLE_DEVICES"]
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump({name: os.environ.get(name) for name in names}, handle)
""",
        encoding="utf-8",
    )
    serve = _array([sys.executable, str(fake_child), str(output)])
    script = SCRIPTS / "run_chat_ollama.ps1"
    command = f"& '{script}' -ServeCommand {serve} -MaxRestarts 1"

    result = _run(command)
    environment = json.loads(output.read_text(encoding="utf-8"))

    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert environment == {
        "OLLAMA_HOST": "127.0.0.1:11434",
        "OLLAMA_CONTEXT_LENGTH": "8192",
        "OLLAMA_MAX_LOADED_MODELS": "1",
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_NO_CLOUD": "1",
        "OLLAMA_API_KEY": None,
        "CUDA_VISIBLE_DEVICES": None,
    }


def test_gateway_runner_accepts_injected_child_and_stays_loopback_only(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "gateway-child.txt"
    fake_child = tmp_path / "child.py"
    fake_child.write_text(
        "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('ran')",
        encoding="utf-8",
    )
    child = _array([sys.executable, str(fake_child), str(marker)])
    script = SCRIPTS / "run_gateway.ps1"
    command = (
        f"& '{script}' -ProjectRoot '{ROOT}' -PythonPath '{sys.executable}' "
        f"-GatewayCommand {child} -MaxRestarts 1"
    )

    result = _run(command)

    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert marker.read_text() == "ran"
    assert "0.0.0.0" not in script.read_text(encoding="utf-8")


def test_installer_has_no_task_removal_path() -> None:
    content = (SCRIPTS / "install_stage6.ps1").read_text(encoding="utf-8")

    assert "Unregister-ScheduledTask" not in content
    assert "service install <TUNNEL_TOKEN>" in content


# --- SYSTEM 실행 경로 불변식: 관리 주체만 수정할 수 있는 경로인지 실제 ACL로 판정한다 ---


def test_admin_only_system_paths_are_accepted() -> None:
    checks = [
        f"Test-AdminOnlyExecutionPath -Path '{ADMIN_ONLY_SYSTEM_FILE}'",
        f"Test-AdminOnlyExecutionPath -Path '{ADMIN_ONLY_SYSTEM_TREE}' -Recurse",
    ]
    for check in checks:
        completed = _dot_source("execution_security.ps1", check)

        assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
        assert _text(completed).strip() == "True", check


def test_user_writable_directory_is_rejected_with_the_writable_principal(
    tmp_path: Path,
) -> None:
    payload = tmp_path / "run_gateway.ps1"
    payload.write_text("Write-Output 'payload'", encoding="utf-8")

    safe = _dot_source(
        "execution_security.ps1", f"Test-AdminOnlyExecutionPath -Path '{payload}'"
    )
    risks = _dot_source(
        "execution_security.ps1",
        f"(Get-ExecutionPathRisks -Path '{payload}') -join [Environment]::NewLine",
    )

    assert _text(safe).strip() == "False"
    # 소유자만으로도, 부여된 수정 권한만으로도 SYSTEM 실행 경로로 쓸 수 없다는 근거가 나와야 한다.
    assert str(payload) in _text(risks)
    assert "소유자" in _text(risks)


def test_user_writable_descendant_is_rejected_by_the_recursive_check(
    tmp_path: Path,
) -> None:
    package = tmp_path / "gateway"
    package.mkdir()
    (package / "main.py").write_text("print('imported by SYSTEM')", encoding="utf-8")

    completed = _dot_source(
        "execution_security.ps1",
        f"(Get-ExecutionPathRisks -Path '{package}' -Recurse | "
        f"Where-Object {{ $_ -like '*main.py*' }}).Count",
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert int(_text(completed).strip()) > 0


def test_assert_admin_only_execution_path_throws_for_user_writable_path(
    tmp_path: Path,
) -> None:
    completed = _dot_source(
        "execution_security.ps1",
        f"Assert-AdminOnlyExecutionPath -Path '{tmp_path}' -Purpose '테스트 경로'",
    )

    assert completed.returncode != 0
    assert UNSAFE_EXECUTION_PATH_MARKER in completed.stderr.decode("utf-8", "replace")


def test_missing_execution_path_is_reported_as_a_risk(tmp_path: Path) -> None:
    absent = tmp_path / "not-installed.exe"

    completed = _dot_source(
        "execution_security.ps1", f"Test-AdminOnlyExecutionPath -Path '{absent}'"
    )

    assert _text(completed).strip() == "False"


def test_installer_reports_unsafe_system_execution_paths_without_mutation(
    tmp_path: Path,
) -> None:
    fake_ollama = tmp_path / "ollama.exe"
    fake_ollama.touch()
    models = tmp_path / "models"
    models.mkdir()
    script = SCRIPTS / "install_stage6.ps1"

    result = _run(
        f"& '{script}' -ValidateOnly -ProjectRoot '{ROOT}' "
        f"-OllamaPath '{fake_ollama}' -OllamaModelsPath '{models}'"
    )

    output = _text(result)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    # 사용자가 쓸 수 있는 Ollama 실행 파일은 SYSTEM 작업이 실행할 수 없다고 보고해야 한다.
    assert (
        f"{UNSAFE_EXECUTION_PATH_MARKER} - 승인되지 않은 주체가 수정할 수 있다 (Ollama 실행 파일)"
        in output
    )
    # 스크립트는 action 실행 파일을 고정 경로로 적고 PATH 조회는 표기가 다를 수 있다 — Windows 경로
    # 비교이므로 대소문자를 구분하지 않는다.
    expected = f"SAFE SYSTEM 실행 경로: 작업 action 실행 파일 - {POWERSHELL}"
    assert os.path.normcase(expected) in os.path.normcase(output)


@pytest.mark.parametrize(
    ("command_line", "expected"),
    [
        (
            '"C:\\Program Files (x86)\\cloudflared\\cloudflared.exe" tunnel run --token X',
            "C:\\Program Files (x86)\\cloudflared\\cloudflared.exe",
        ),
        (
            "C:\\cloudflared\\cloudflared.exe tunnel run",
            "C:\\cloudflared\\cloudflared.exe",
        ),
        ("C:\\cloudflared\\cloudflared.exe", "C:\\cloudflared\\cloudflared.exe"),
    ],
    ids=["quoted-with-arguments", "unquoted-with-arguments", "no-arguments"],
)
def test_service_executable_is_extracted_from_the_registered_command_line(
    command_line: str, expected: str
) -> None:
    # 등록 정보는 인자까지 붙은 명령줄이다 — 실행 파일 자리를 정확히 뽑아야 그 자리를 판정할 수 있다.
    completed = _dot_source(
        "execution_security.ps1",
        f"Get-ServiceExecutablePath -CommandLine '{command_line}'",
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert _text(completed).strip() == expected


def test_service_executable_a_non_admin_can_replace_is_rejected(tmp_path: Path) -> None:
    # 명령줄에 cloudflared라는 이름이 들어 있는지 보는 것만으로는 사용자가 바꿀 수 있는 자리의 실행
    # 파일도 그대로 통과한다. SYSTEM으로 실행되는 서비스이므로 뽑아낸 자리를 같은 기준으로 판정한다.
    planted = tmp_path / "cloudflared.exe"
    planted.touch()

    completed = _dot_source(
        "execution_security.ps1",
        "Test-AdminOnlyExecutionPath -Path (Get-ServiceExecutablePath "
        f"-CommandLine '\"{planted}\" tunnel run --token FAKE')",
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert _text(completed).strip() == "False"


def test_import_paths_inside_the_deployment_or_python_distribution_are_contained() -> (
    None
):
    completed = _dot_source(
        "execution_security.ps1",
        "@(Get-ExternalImportPaths -Paths @("
        "'D:\\deploy', 'D:\\deploy\\src\\gateway\\__init__.py', "
        "'D:\\deploy\\.venv\\Lib\\site-packages', 'C:\\python\\Lib'"
        ") -ApprovedRoots @('D:\\deploy', 'C:\\python')).Count",
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert int(_text(completed).strip()) == 0


def test_editable_pointer_into_the_work_tree_is_reported_as_an_external_import_path() -> (
    None
):
    work_tree_source = "C:\\Users\\dev\\project\\src"

    completed = _dot_source(
        "execution_security.ps1",
        f"@(Get-ExternalImportPaths -Paths @('D:\\deploy\\src', '{work_tree_source}') "
        "-ApprovedRoots @('D:\\deploy')) -join '|'",
    )

    assert _text(completed).strip() == work_tree_source


def test_directory_sharing_a_name_prefix_with_the_deployment_is_not_contained() -> None:
    # 접두사만 겹치는 이웃 디렉터리를 사본 안으로 보면 격리 판정이 무너진다.
    completed = _dot_source(
        "execution_security.ps1",
        "@(Get-ExternalImportPaths -Paths @('D:\\deploy-evil\\src') "
        "-ApprovedRoots @('D:\\deploy')) -join '|'",
    )

    assert _text(completed).strip() == "D:\\deploy-evil\\src"


def _import_gate(working_directory: Path, approved_roots: list[str]) -> str:
    roots = ",".join(f"'{root}'" for root in approved_roots)
    return (
        f"Assert-ContainedImportPaths -PythonPath '{sys.executable}' "
        f"-WorkingDirectory '{working_directory}' -ApprovedRoots @({roots})"
    )


def test_import_gate_accepts_a_deployment_that_imports_only_its_own_copy(
    tmp_path: Path,
) -> None:
    deployment = tmp_path / "deployment"
    (deployment / "gateway").mkdir(parents=True)
    (deployment / "gateway" / "__init__.py").write_text("", encoding="utf-8")

    completed = _dot_source(
        "execution_security.ps1",
        _import_gate(deployment, [str(deployment), sys.base_prefix, str(ROOT)]),
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")


def test_import_gate_blocks_a_gateway_resolved_outside_the_deployment(
    tmp_path: Path,
) -> None:
    # editable 설치가 원본 트리를 sys.path에 얹는 것과 같은 상황을 PYTHONPATH로 만든다 — 사본의
    # ACL이 아니라 실제 실행 결과를 물어야 드러난다.
    work_tree = tmp_path / "work-tree" / "src"
    (work_tree / "gateway").mkdir(parents=True)
    (work_tree / "gateway" / "__init__.py").write_text("", encoding="utf-8")
    deployment = tmp_path / "deployment"
    deployment.mkdir()

    # PowerShell 오류 레코드 서식은 긴 경로를 콘솔 폭에서 줄바꿈하므로 메시지를 그대로 받아 본다.
    completed = _dot_source(
        "execution_security.ps1",
        f"$env:PYTHONPATH='{work_tree}'; try {{ "
        + _import_gate(deployment, [str(deployment), sys.base_prefix, str(ROOT)])
        + "; exit 0 } catch { Write-Output $_.Exception.Message; exit 1 }",
    )

    reported = _text(completed)
    assert completed.returncode == 1, "등록 전에 멈춰야 한다"
    assert EXTERNAL_IMPORT_PATH_MARKER in reported
    # 어느 코드를 실제로 import하게 되는지가 근거로 나와야 한다.
    assert str(work_tree / "gateway" / "__init__.py") in reported


def test_deployed_editable_pointer_is_retargeted_into_the_deployment_copy(
    tmp_path: Path,
) -> None:
    source = tmp_path / "work-tree"
    site_packages = source / ".venv" / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)
    (source / "src").mkdir()
    (site_packages / "local_first_inference_gateway.pth").write_text(
        f"{source / 'src'}\n", encoding="utf-8"
    )
    (site_packages / "_virtualenv.pth").write_text("import _virtualenv\n", "utf-8")
    deployment = tmp_path / "deployment"
    shutil.copytree(source, deployment)

    completed = _dot_source(
        "execution_security.ps1",
        f"Set-DeployedEditablePointers -Source '{source}' -Destination '{deployment}'",
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    deployed = deployment / ".venv" / "Lib" / "site-packages"
    # 사본의 venv는 사본의 소스만 import한다.
    assert (deployed / "local_first_inference_gateway.pth").read_text().strip() == str(
        deployment / "src"
    )
    # 경로가 아닌 실행 줄과 원본 트리의 포인터는 그대로 둔다.
    assert (deployed / "_virtualenv.pth").read_text().strip() == "import _virtualenv"
    assert (
        site_packages / "local_first_inference_gateway.pth"
    ).read_text().strip() == str(source / "src")


def test_installer_aborts_before_any_change_when_a_system_path_is_unsafe(
    tmp_path: Path,
) -> None:
    # 관리자 권한 확인보다 먼저 불변식을 검증한다 — 깨져 있으면 작업을 하나도 등록하지 않는다.
    fake_ollama = tmp_path / "ollama.exe"
    fake_ollama.touch()
    models = tmp_path / "models"
    models.mkdir()
    script = SCRIPTS / "install_stage6.ps1"

    result = _run(
        f"& '{script}' -ProjectRoot '{ROOT}' -StartTasks "
        f"-OllamaPath '{fake_ollama}' -OllamaModelsPath '{models}'"
    )

    errors = result.stderr.decode("utf-8", "replace")
    assert result.returncode != 0
    assert UNSAFE_EXECUTION_PATH_MARKER in errors
    assert "Register-ScheduledTask" not in errors


# --- 운영 상태 파일: 채택 여부는 잠그기 전 권한으로 정하고, 남은 파일은 잠근 뒤에 확인한다 ---


def test_state_directory_non_admins_could_write_is_not_adoptable(
    tmp_path: Path,
) -> None:
    # 비관리자가 미리 만들어 둔 키 저장소를 채택하면 그 사용자가 인증을 통과할 키를 심을 수 있다.
    completed = _dot_source(
        "execution_security.ps1", f"Test-StateDirectoryWasAdminOnly -Path '{tmp_path}'"
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert _text(completed).strip() == "False"


def test_admin_only_state_directory_is_adoptable_with_its_existing_files() -> None:
    completed = _dot_source(
        "execution_security.ps1",
        f"Test-StateDirectoryWasAdminOnly -Path '{ADMIN_ONLY_SYSTEM_TREE}'",
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert _text(completed).strip() == "True"


def test_state_directory_that_does_not_exist_yet_is_not_adoptable(
    tmp_path: Path,
) -> None:
    # 처음 만드는 자리는 만드는 순간 상위 디렉터리의 권한을 상속한다 — 잠글 때까지는 비관리자도 파일을
    # 만들 수 있으므로 거기서 발견한 파일을 채택해도 되는 자리로 보지 않는다.
    completed = _dot_source(
        "execution_security.ps1",
        f"Test-StateDirectoryWasAdminOnly -Path '{tmp_path / 'first-install'}'",
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert _text(completed).strip() == "False"


def test_key_store_planted_after_the_adoption_verdict_is_still_refused(
    tmp_path: Path,
) -> None:
    # 판정과 잠금 사이에 비관리자가 키 저장소를 심는 경합이다. 판정 시점에는 자리가 비어 있어도 잠근
    # 뒤에 확인하므로, 심어 둔 저장소가 소유권과 권한을 정리받고 인증 근거가 되는 일이 없어야 한다.
    completed = _dot_source(
        "execution_security.ps1",
        f"$adoptable = Test-StateDirectoryWasAdminOnly -Path '{tmp_path}'; "
        f"Set-Content -LiteralPath '{tmp_path / 'api-keys.json'}' -Value '{{}}'; "
        f"if (-not $adoptable) {{ Assert-NoPlantedStateFiles -Path '{tmp_path}' }}",
    )

    reported = completed.stderr.decode("utf-8", "replace")
    assert completed.returncode != 0, "채택하기 전에 멈춰야 한다"
    assert UNADOPTABLE_STATE_DIRECTORY_MARKER in reported
    assert "api-keys.json" in reported


def test_state_directory_with_nothing_left_after_the_lock_adopts_nothing_and_proceeds(
    tmp_path: Path,
) -> None:
    # 채택할 파일이 없으면 그대로 진행한다 — 이후 만드는 파일은 잠근 뒤의 제한을 상속한다.
    completed = _dot_source(
        "execution_security.ps1", f"Assert-NoPlantedStateFiles -Path '{tmp_path}'"
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")


def test_admin_only_state_file_is_accepted() -> None:
    completed = _dot_source(
        "execution_security.ps1",
        f"Assert-AdminOnlyStatePath -Path '{ADMIN_ONLY_SYSTEM_FILE}' "
        "-Purpose 'API 키 저장소'",
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")


def test_installer_locks_exactly_the_state_paths_the_gateway_and_watchdog_read() -> (
    None
):
    # 잠그는 자리와 SYSTEM이 인증 근거로 읽는 자리가 어긋나면 잠금은 아무것도 막지 못한다.
    completed = _dot_source(
        "execution_security.ps1",
        f"(Get-ProtectedStatePaths -PythonPath '{sys.executable}') | "
        "ConvertTo-Json -Compress",
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    reported = json.loads(_text(completed))
    assert reported == {
        "Directory": str(STATE_DIRECTORY),
        "KeyStore": str(API_KEY_STORE_PATH),
        "WatchdogKey": str(WATCHDOG_KEY_PATH),
    }


def test_installer_locks_the_same_state_paths_from_a_redirected_environment() -> None:
    # 설치기는 관리자 셸의 환경으로, 작업은 SYSTEM의 환경으로 실행된다. 자리를 환경에서 읽으면 잠그는
    # 자리가 SYSTEM이 읽는 자리와 어긋나 잠금이 아무것도 막지 못한다.
    completed = _dot_source(
        "execution_security.ps1",
        f"$env:PROGRAMDATA = '{Path.home() / 'planted'}'; "
        f"(Get-ProtectedStatePaths -PythonPath '{sys.executable}') | "
        "ConvertTo-Json -Compress",
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    reported = json.loads(_text(completed))
    assert reported == {
        "Directory": str(STATE_DIRECTORY),
        "KeyStore": str(API_KEY_STORE_PATH),
        "WatchdogKey": str(WATCHDOG_KEY_PATH),
    }


@pytest.mark.parametrize("arguments", ["", "-Recurse"], ids=["single", "recursive"])
def test_owner_change_failure_is_detected_instead_of_reported_as_success(
    tmp_path: Path, arguments: str
) -> None:
    # icacls는 /C를 주면 실패해도, /T는 대상이 없어도 종료 코드 0을 돌려준다. 배포 사본과 운영 키의
    # 소유자를 옮기지 못한 것을 성공으로 읽으면 만든 관리자 계정이 소유자로 남는다.
    absent = tmp_path / "never-created"

    completed = _dot_source(
        "execution_security.ps1", f"Set-ApprovedOwner -Path '{absent}' {arguments}"
    )

    reported = completed.stderr.decode("utf-8", "replace")
    assert completed.returncode != 0
    assert "소유자를 Administrators로 설정하지 못했다" in reported


def test_state_file_a_non_admin_can_rewrite_is_rejected_as_state_not_execution(
    tmp_path: Path,
) -> None:
    # 상태 디렉터리를 잠가도 이미 있던 파일의 소유자와 명시적 ACE는 남는다.
    key_store = tmp_path / "api-keys.json"
    key_store.write_text("{}", encoding="utf-8")

    completed = _dot_source(
        "execution_security.ps1",
        f"Assert-AdminOnlyStatePath -Path '{key_store}' -Purpose 'API 키 저장소'",
    )

    reported = completed.stderr.decode("utf-8", "replace")
    assert completed.returncode != 0
    assert UNSAFE_STATE_PATH_MARKER in reported
    assert UNSAFE_EXECUTION_PATH_MARKER not in reported


# --- 배포 사본 교체: 검증을 통과한 사본만 배포 자리에 놓고, 실패하면 직전 사본을 되돌린다 ---


def _deployment(path: Path, marker: str) -> Path:
    (path / "src").mkdir(parents=True)
    (path / "src" / "gateway.py").write_text(marker, encoding="utf-8")
    return path


def test_verified_copy_replaces_the_live_deployment_and_keeps_the_previous_one(
    tmp_path: Path,
) -> None:
    staging = _deployment(tmp_path / "app.staging", "verified")
    live = _deployment(tmp_path / "app", "live")
    previous = tmp_path / "app.previous"

    completed = _dot_source(
        "deployment_swap.ps1",
        f"Switch-Deployment -Staging '{staging}' -Live '{live}' -Previous '{previous}'",
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert (live / "src" / "gateway.py").read_text() == "verified"
    # 교체 뒤 확인이 실패하면 되돌려야 하므로 직전 배포본은 남긴다.
    assert (previous / "src" / "gateway.py").read_text() == "live"
    assert not staging.exists()


def test_live_deployment_survives_when_the_staged_copy_cannot_be_moved_into_place(
    tmp_path: Path,
) -> None:
    live = _deployment(tmp_path / "app", "live")
    previous = tmp_path / "app.previous"

    completed = _dot_source(
        "deployment_swap.ps1",
        f"Switch-Deployment -Staging '{tmp_path / 'app.staging'}' -Live '{live}' "
        f"-Previous '{previous}'",
    )

    assert completed.returncode != 0
    # 기존 작업이 가리키는 자리에는 직전 배포본이 그대로 있어야 한다.
    assert (live / "src" / "gateway.py").read_text() == "live"
    assert not previous.exists()


def test_failed_copy_is_replaced_by_the_previous_deployment_on_restore(
    tmp_path: Path,
) -> None:
    live = _deployment(tmp_path / "app", "unverified")
    previous = _deployment(tmp_path / "app.previous", "live")

    completed = _dot_source(
        "deployment_swap.ps1",
        f"Restore-Deployment -Live '{live}' -Previous '{previous}'",
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert (live / "src" / "gateway.py").read_text() == "live"
    assert not previous.exists()


def test_failed_first_install_leaves_no_unverified_copy_in_the_deployment_place(
    tmp_path: Path,
) -> None:
    # 되돌릴 직전 배포본이 없어도 확인을 통과하지 못한 사본을 그 자리에 두지 않는다.
    live = _deployment(tmp_path / "app", "unverified")

    completed = _dot_source(
        "deployment_swap.ps1",
        f"Restore-Deployment -Live '{live}' -Previous '{tmp_path / 'app.previous'}'",
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert not live.exists()


# --- 실행 중인 작업이 있는 배포 갱신: 가짜 작업 어댑터로 실제 교체 절차를 그대로 돌린다 ---

# Update-Deployment가 실제로 부르는 순서와 최종 상태를 본다. Windows 작업을 만들지 않으려고 조회·
# 중지·시작 경계만 가짜로 바꾸고, 배포 자리 교체는 임시 디렉터리에서 진짜로 일어나게 둔다.
# 가짜 상태는 이름이 겹치지 않는 자루 하나에 담고 그 안의 값만 바꾼다 — PowerShell은 스크립트 블록의
# 변수를 부르는 쪽 범위에서 찾으므로, 흔한 이름을 쓰면 진짜 함수의 지역 변수를 잘못 집는다.
_UPDATE_HARNESS = """
$live = '<LIVE>'
$staging = '<STAGING>'
$previous = '<PREVIOUS>'
$fake = @{
    Log = '<LOG>'
    StopFailure = '<STOP_FAILURE>'
    StartFailure = '<START_FAILURE>'
    StartFailuresLeft = <START_FAILURE_COUNT>
    Holder = $null
    Running = [ordered]@{
        'Gateway' = $false
        'Chat Ollama' = $false
        'Embedding Ollama' = $false
        'Watchdog' = $false
    }
}
foreach ($started in <RUNNING>) { $fake.Running[$started] = $true }
function Write-Call {
    param([string]$Text)
    Add-Content -LiteralPath $fake.Log -Value $Text -Encoding UTF8
}
$control = [pscustomobject]@{
    GetRunningNames = { $fake.Running.Keys | Where-Object { $fake.Running[$_] } }
    StopTask = {
        param([string]$Name)
        Write-Call "stop:$Name"
        if ($Name -eq $fake.StopFailure) {
            throw "작업이 제한 시간 안에 중지되지 않았다: $Name"
        }
        $fake.Running[$Name] = $false
    }
    StartTask = {
        param([string]$Name)
        Write-Call "start:$Name"
        if ($Name -eq $fake.StartFailure -and $fake.StartFailuresLeft -gt 0) {
            $fake.StartFailuresLeft = $fake.StartFailuresLeft - 1
            throw "작업을 시작하지 못했다: $Name"
        }
        $fake.Running[$Name] = $true
    }
}
try {
    Update-Deployment -Live $live -Staging $staging -Previous $previous `
        -TaskControl $control -Names <NAMES> -WatchdogName 'Watchdog' `
        -StartTasks:$<START_TASKS> `
        -StageAndVerify { Write-Call 'stage'; <STAGE_BODY> } `
        -AfterSwap { Write-Call 'after-swap'; <AFTER_SWAP_BODY> }
    Write-Call 'completed'
}
catch {
    Write-Call "failed: $($_.Exception.Message)"
}
<EPILOGUE>
Write-Output (@($fake.Running.Keys | Where-Object { $fake.Running[$_] }) -join '|')
"""

_STAGE_A_VERIFIED_COPY = (
    "$null = New-Item -ItemType Directory -Path $staging; "
    "Set-Content -LiteralPath \"$staging\\marker.txt\" -Value 'verified'"
)


@dataclass(frozen=True)
class UpdateResult:
    calls: list[str]
    running: list[str]
    live_marker: str


def _update_deployment(
    tmp_path: Path,
    *,
    running: list[str],
    start_tasks: bool = False,
    stage_body: str = _STAGE_A_VERIFIED_COPY,
    after_swap_body: str = "",
    epilogue: str = "",
    stop_failure: str = "",
    start_failure: str = "",
    start_failure_count: int = 1,
) -> UpdateResult:
    log = tmp_path / "calls.log"
    live = tmp_path / "app"
    substitutions = {
        "<LIVE>": str(live),
        "<STAGING>": str(tmp_path / "app.staging"),
        "<PREVIOUS>": str(tmp_path / "app.previous"),
        "<LOG>": str(log),
        "<STOP_FAILURE>": stop_failure,
        "<START_FAILURE>": start_failure,
        "<START_FAILURE_COUNT>": str(start_failure_count),
        "<RUNNING>": _array(running),
        "<NAMES>": _array(STAGE6_TASK_NAMES),
        "<START_TASKS>": str(start_tasks).lower(),
        "<STAGE_BODY>": stage_body,
        "<AFTER_SWAP_BODY>": after_swap_body,
        "<EPILOGUE>": epilogue,
    }
    script = _UPDATE_HARNESS
    for placeholder, value in substitutions.items():
        script = script.replace(placeholder, value)

    completed = _dot_source("deployment_swap.ps1", script)

    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    calls = []
    if log.exists():
        calls = log.read_text(encoding="utf-8-sig").splitlines()
    marker = live / "marker.txt"
    live_marker = ""
    if marker.exists():
        live_marker = marker.read_text(encoding="utf-8").strip()
    still_running = [name for name in _text(completed).strip().split("|") if name]
    return UpdateResult(calls, still_running, live_marker)


def _live_deployment(tmp_path: Path) -> None:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "marker.txt").write_text("live", encoding="utf-8")


def test_running_tasks_are_stopped_for_the_swap_and_restarted_with_the_new_key(
    tmp_path: Path,
) -> None:
    # scripts/install_stage6.ps1 -ReissueWatchdogKey -StartTasks의 실행 중 Watchdog 사례다.
    # Watchdog은 보호 키를 기동 시 한 번만 읽으므로, 키를 쓰기 전에 멈추고 쓴 뒤에 세워야 새 키를 읽는다.
    _live_deployment(tmp_path)

    result = _update_deployment(
        tmp_path,
        running=STAGE6_TASK_NAMES,
        start_tasks=True,
        after_swap_body="Write-Call 'issue-watchdog-key'; Write-Call 'register-tasks'",
    )

    assert result.calls == [
        "stage",
        "stop:Watchdog",
        "stop:Gateway",
        "stop:Chat Ollama",
        "stop:Embedding Ollama",
        "after-swap",
        "issue-watchdog-key",
        "register-tasks",
        "start:Gateway",
        "start:Chat Ollama",
        "start:Embedding Ollama",
        "start:Watchdog",
        "completed",
    ]
    assert result.live_marker == "verified"
    assert result.running == STAGE6_TASK_NAMES


def _installer_stage_commands(parameter: str) -> list[str]:
    # 설치기가 어느 단계에 무엇을 맡겼는지 구문 트리로 읽는다 — 문자열이 어딘가에 있다는 것만으로는
    # 그 호출이 되돌릴 수 있는 단계 안에 있는지 알 수 없다.
    script = SCRIPTS / "install_stage6.ps1"
    completed = _run(
        "$errors = $null; $tokens = $null; "
        "$ast = [System.Management.Automation.Language.Parser]::ParseFile("
        f"'{script}', [ref]$tokens, [ref]$errors); "
        "$call = @($ast.FindAll({ param($node) "
        "$node -is [System.Management.Automation.Language.CommandAst] -and "
        "$node.GetCommandName() -eq 'Update-Deployment' }, $true))[0]; "
        "$elements = @($call.CommandElements); "
        "$block = $null; "
        "for ($index = 0; $index -lt $elements.Count; $index++) { "
        "if ($elements[$index] -is "
        "[System.Management.Automation.Language.CommandParameterAst] -and "
        f"$elements[$index].ParameterName -eq '{parameter}') "
        "{ $block = $elements[$index + 1] } }; "
        "@($block.ScriptBlock.FindAll({ param($node) "
        "$node -is [System.Management.Automation.Language.CommandAst] }, $true) | "
        "ForEach-Object { $_.GetCommandName() }) -join '|'"
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    return [name for name in _text(completed).strip().split("|") if name]


def _installer_top_level_commands() -> list[str]:
    # 설치기가 무엇을 어느 순서로 하는지 구문 트리로 읽는다 — 함수 정의 안은 빼고 실제로 실행되는
    # 최상위 문장만 본다. 문자열이 파일 어딘가에 있다는 것만으로는 그 검사가 시스템을 바꾸기 전에
    # 도는지 알 수 없다.
    script = SCRIPTS / "install_stage6.ps1"
    completed = _run(
        "$errors = $null; $tokens = $null; "
        "$ast = [System.Management.Automation.Language.Parser]::ParseFile("
        f"'{script}', [ref]$tokens, [ref]$errors); "
        "@($ast.EndBlock.Statements | Where-Object { $_ -isnot "
        "[System.Management.Automation.Language.FunctionDefinitionAst] } | "
        "ForEach-Object { $_.FindAll({ param($node) $node -is "
        "[System.Management.Automation.Language.CommandAst] }, $true) } | "
        "ForEach-Object { $_.GetCommandName() }) -join '|'"
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    return [name for name in _text(completed).strip().split("|") if name]


def test_installer_validates_every_system_executed_path_before_it_changes_anything() -> (
    None
):
    order = _installer_top_level_commands()

    # cloudflared 서비스도 SYSTEM으로 실행된다 — 실행 파일 자리를 뽑아 검사 대상에 넣은 뒤에 검사한다.
    assert order.index("Resolve-CloudflaredExecutable") < order.index(
        "Get-SystemExecutedPaths"
    )
    assert order.index("Get-SystemExecutedPaths") < order.index(
        "Assert-AdminOnlyExecutionPath"
    )
    # 검사가 상태 디렉터리 ACL 변경과 배포 교체보다 먼저 끝나야 아무것도 바꾸지 않고 멈출 수 있다.
    assert order.index("Assert-AdminOnlyExecutionPath") < order.index(
        "Set-PrivateDirectory"
    )
    assert order.index("Assert-AdminOnlyExecutionPath") < order.index(
        "Update-Deployment"
    )


def test_installer_locks_the_state_directory_before_it_looks_for_planted_files() -> (
    None
):
    # 잠그기 전에 확인하면 확인과 잠금 사이에 심은 키 저장소를 그대로 인증 근거로 채택한다.
    order = _installer_top_level_commands()

    assert (
        order.index("Test-StateDirectoryWasAdminOnly")
        < order.index("Set-PrivateDirectory")
        < order.index("Assert-NoPlantedStateFiles")
        < order.index("Protect-StateFile")
    )


def test_installer_runs_key_issue_and_registration_inside_the_reversible_stage() -> (
    None
):
    # 교체 뒤 단계에 있어야 새 배포본의 키 CLI로 발급하고, 실패 시 배포와 실행 상태가 함께 되돌아간다.
    staged = _installer_stage_commands("StageAndVerify")
    after_swap = _installer_stage_commands("AfterSwap")

    # 사본 생성·검증은 기존 서비스가 실행 중인 채로 끝나야 한다.
    assert staged == ["Publish-Deployment", "Assert-DeploymentIsContained"]
    assert after_swap == [
        "Set-DeployedEditablePointers",
        "Assert-DeploymentIsContained",
        "Publish-WatchdogKey",
        "Ensure-TaskFolder",
        "Register-OwnedTask",
    ]


def test_staging_verification_failure_touches_neither_tasks_nor_the_live_deployment(
    tmp_path: Path,
) -> None:
    # 사본을 만들고 검증하는 동안에는 기존 서비스를 그대로 둔다 — 중단 시간은 교체 직전부터다.
    _live_deployment(tmp_path)

    result = _update_deployment(
        tmp_path,
        running=STAGE6_TASK_NAMES,
        start_tasks=True,
        stage_body="throw 'UNSAFE SYSTEM import 경로'",
    )

    assert result.calls == ["stage", "failed: UNSAFE SYSTEM import 경로"]
    assert result.live_marker == "live"
    assert result.running == STAGE6_TASK_NAMES


def test_stop_failure_aborts_before_the_swap_and_restarts_what_it_stopped(
    tmp_path: Path,
) -> None:
    # 배포 사본을 잡은 프로세스가 남았는데 교체를 밀어붙이면 반쯤 갱신된 자리가 남는다.
    _live_deployment(tmp_path)

    result = _update_deployment(
        tmp_path,
        running=STAGE6_TASK_NAMES,
        start_tasks=True,
        stop_failure="Chat Ollama",
    )

    assert result.calls == [
        "stage",
        "stop:Watchdog",
        "stop:Gateway",
        "stop:Chat Ollama",
        "start:Gateway",
        "start:Watchdog",
        "failed: 작업이 제한 시간 안에 중지되지 않았다: Chat Ollama",
    ]
    assert result.live_marker == "live"
    assert result.running == STAGE6_TASK_NAMES


def test_swap_failure_keeps_the_live_deployment_and_restores_running_tasks(
    tmp_path: Path,
) -> None:
    _live_deployment(tmp_path)

    # 검증은 통과했다고 하지만 옮길 사본이 없다 — 교체가 배포 자리를 비운 채 끝나면 안 된다.
    result = _update_deployment(
        tmp_path, running=["Gateway", "Watchdog"], stage_body="Write-Call 'no-copy'"
    )

    assert result.calls[:5] == [
        "stage",
        "no-copy",
        "stop:Watchdog",
        "stop:Gateway",
        "start:Gateway",
    ]
    assert result.calls[5] == "start:Watchdog"
    assert result.calls[6].startswith("failed:")
    assert result.live_marker == "live"
    assert result.running == ["Gateway", "Watchdog"]


def test_post_swap_failure_restores_the_previous_deployment_and_running_state(
    tmp_path: Path,
) -> None:
    _live_deployment(tmp_path)

    result = _update_deployment(
        tmp_path,
        running=["Gateway"],
        after_swap_body="throw 'UNSAFE SYSTEM 실행 경로'",
    )

    assert result.calls == [
        "stage",
        "stop:Gateway",
        "after-swap",
        "start:Gateway",
        "failed: UNSAFE SYSTEM 실행 경로",
    ]
    # 확인을 통과하지 못한 사본을 그 자리에 두면 다음 시작에서 SYSTEM이 그 코드를 실행한다.
    assert result.live_marker == "live"
    assert result.running == ["Gateway"]


def test_start_failure_stops_the_new_tasks_before_restoring_the_previous_deployment(
    tmp_path: Path,
) -> None:
    _live_deployment(tmp_path)

    result = _update_deployment(
        tmp_path,
        running=["Gateway", "Chat Ollama"],
        start_failure="Chat Ollama",
    )

    # 새 배포본으로 이미 올라온 Gateway를 멈춰야 직전 배포본을 되돌릴 수 있다.
    assert result.calls == [
        "stage",
        "stop:Gateway",
        "stop:Chat Ollama",
        "after-swap",
        "start:Gateway",
        "start:Chat Ollama",
        "stop:Gateway",
        "start:Gateway",
        "start:Chat Ollama",
        "failed: 작업을 시작하지 못했다: Chat Ollama",
    ]
    assert result.live_marker == "live"
    assert result.running == ["Gateway", "Chat Ollama"]


def test_a_deployment_that_cannot_be_restored_fails_loudly_with_both_causes(
    tmp_path: Path,
) -> None:
    # 배포 사본의 파일을 아직 잡고 있으면 되돌리기도 실패한다 — 조용히 넘어가면 SYSTEM이 검증하지
    # 않은 배포본을 실행하는데도 성공으로 보인다.
    _live_deployment(tmp_path)
    hold = (
        "$fake.Holder = [System.IO.File]::Open(\"$live\\held.txt\", 'Create', 'Write', 'None'); "
        "throw '교체 뒤 확인 실패'"
    )

    result = _update_deployment(
        tmp_path,
        running=[],
        after_swap_body=hold,
        epilogue="$fake.Holder.Dispose()",
    )

    assert result.calls[:2] == ["stage", "after-swap"]
    reported = "\n".join(result.calls[2:])
    assert "되돌리는 중에 다시 실패했다" in reported
    # 무엇 때문에 되돌리려 했는지와 왜 되돌리지 못했는지가 모두 남아야 한다.
    assert "원인: 교체 뒤 확인 실패" in reported
    assert "복구 실패: " in reported


def test_first_install_without_running_tasks_swaps_and_starts_everything(
    tmp_path: Path,
) -> None:
    # 되돌릴 배포본도 멈출 작업도 없는 최초 설치에서 같은 절차가 그대로 성립한다.
    result = _update_deployment(tmp_path, running=[], start_tasks=True)

    assert result.calls == [
        "stage",
        "after-swap",
        "start:Gateway",
        "start:Chat Ollama",
        "start:Embedding Ollama",
        "start:Watchdog",
        "completed",
    ]
    assert result.live_marker == "verified"
    assert result.running == STAGE6_TASK_NAMES


def test_repeated_update_without_start_tasks_leaves_stopped_tasks_stopped(
    tmp_path: Path,
) -> None:
    _live_deployment(tmp_path)

    result = _update_deployment(tmp_path, running=["Gateway"])

    assert result.calls == [
        "stage",
        "stop:Gateway",
        "after-swap",
        "start:Gateway",
        "completed",
    ]
    assert result.live_marker == "verified"
    assert result.running == ["Gateway"]


# --- 작업 소유권과 수명주기: 실제 작업을 만들지 않고 판정 경계를 확인한다 ---


def _owned_task(description: str, execute: str, arguments: str) -> str:
    return (
        f"[pscustomobject]@{{ Description = '{description}'; "
        f"Actions = @([pscustomobject]@{{ Execute = '{execute}'; "
        f"Arguments = '{arguments}' }}) }}"
    )


def test_owned_task_matches_only_on_description_executable_and_arguments() -> None:
    description = "Managed by local-first-inference-gateway Stage 6: Gateway"
    task = _owned_task(description, "C:\\PS.EXE", "-File run_gateway.ps1")
    expected = f"-Description '{description}' -Execute 'c:\\ps.exe' -Arguments '-File run_gateway.ps1'"

    same = _dot_source("stage6_tasks.ps1", f"Test-OwnedTask -Task ({task}) {expected}")
    foreign_description = _dot_source(
        "stage6_tasks.ps1",
        f"Test-OwnedTask -Task ({_owned_task('someone else', 'C:\\PS.EXE', '-File run_gateway.ps1')}) {expected}",
    )
    foreign_arguments = _dot_source(
        "stage6_tasks.ps1",
        f"Test-OwnedTask -Task ({_owned_task(description, 'C:\\PS.EXE', '-File other.ps1')}) {expected}",
    )
    foreign_executable = _dot_source(
        "stage6_tasks.ps1",
        f"Test-OwnedTask -Task ({_owned_task(description, 'C:\\evil.exe', '-File run_gateway.ps1')}) {expected}",
    )

    # 실행 파일 이름만 대소문자가 다른 것은 같은 작업이고, 설명·인자·실행 파일이 다르면 남의 작업이다.
    assert _text(same).strip() == "True"
    assert _text(foreign_description).strip() == "False"
    assert _text(foreign_arguments).strip() == "False"
    assert _text(foreign_executable).strip() == "False"


def test_watchdog_is_stopped_before_the_tasks_it_would_otherwise_recover() -> None:
    plan = _plan(
        f"Get-TaskStopPlan -Names {_array(STAGE6_TASK_NAMES)} "
        f"-RunningNames {_array(STAGE6_TASK_NAMES)} -WatchdogName 'Watchdog'"
    )

    # Watchdog을 나중에 세우면 정비 중에 멈춘 작업을 도로 복구해 교체를 막는다.
    assert plan == ["Watchdog", "Gateway", "Chat Ollama", "Embedding Ollama"]


def test_stop_plan_skips_tasks_that_are_not_running() -> None:
    plan = _plan(
        f"Get-TaskStopPlan -Names {_array(STAGE6_TASK_NAMES)} "
        "-RunningNames @('Chat Ollama') -WatchdogName 'Watchdog'"
    )

    assert plan == ["Chat Ollama"]


def test_first_install_has_nothing_to_stop() -> None:
    plan = _plan(
        f"Get-TaskStopPlan -Names {_array(STAGE6_TASK_NAMES)} "
        "-RunningNames @() -WatchdogName 'Watchdog'"
    )

    assert plan == []


def test_start_tasks_starts_every_task_after_the_swap() -> None:
    plan = _plan(
        f"Get-TaskStartPlan -Names {_array(STAGE6_TASK_NAMES)} "
        "-RunningBefore @() -StartTasks"
    )

    assert plan == STAGE6_TASK_NAMES


def test_install_without_start_tasks_restores_only_what_was_running() -> None:
    # 관리자가 일부러 멈춰 둔 작업은 갱신을 이유로 시작하지 않는다.
    plan = _plan(
        f"Get-TaskStartPlan -Names {_array(STAGE6_TASK_NAMES)} "
        "-RunningBefore @('Gateway', 'Watchdog')"
    )

    assert plan == ["Gateway", "Watchdog"]


def test_stopped_tasks_are_not_started_when_nothing_was_running() -> None:
    plan = _plan(
        f"Get-TaskStartPlan -Names {_array(STAGE6_TASK_NAMES)} -RunningBefore @()"
    )

    assert plan == []
