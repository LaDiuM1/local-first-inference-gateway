"""가짜 자식으로 임베딩 Ollama 감독의 재기동과 자식 정리를 검증한다."""

import ctypes
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

POWERSHELL = shutil.which("powershell") or (
    r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
)
SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_embedding_ollama.ps1"

pytestmark = pytest.mark.skipif(
    platform.system() != "Windows" or not Path(POWERSHELL).exists(),
    reason="감독 스크립트는 Windows PowerShell 환경에서만 검증한다",
)

FAKE_CHILD = """\
import os, sys, time
pid_log = sys.argv[1]
mode = sys.argv[2]
with open(pid_log, "a", encoding="utf-8") as handle:
    handle.write(f"{os.getpid()}\\n")
if mode == "long":
    time.sleep(60)
"""

STILL_ACTIVE = 259
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def _is_alive(pid: int) -> bool:
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        return exit_code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _serve_command(fake_child: Path, pid_log: Path, mode: str) -> str:
    parts = [str(sys.executable), str(fake_child), str(pid_log), mode]
    quoted = ",".join(f"'{part}'" for part in parts)
    return f"@({quoted})"


def _run_supervisor(command: str, timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        capture_output=True,
        timeout=timeout,
    )


def _wait_for_first_pid(pid_log: Path, timeout: float) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pids = pid_log.read_text().split()
        if pids:
            return int(pids[0])
        time.sleep(0.05)
    raise AssertionError("가짜 자식이 시작되지 않았다")


def test_supervisor_restarts_short_child_with_capped_backoff(tmp_path: Path) -> None:
    fake_child = tmp_path / "fake_ollama.py"
    fake_child.write_text(FAKE_CHILD, encoding="utf-8")
    pid_log = tmp_path / "pids.txt"
    pid_log.touch()

    max_restarts = 3
    serve = _serve_command(fake_child, pid_log, "short")
    command = (
        f"& '{SCRIPT}' -ServeCommand {serve} "
        f"-MaxRestarts {max_restarts} -BackoffSeconds 0.2 -MaxBackoffSeconds 1.0"
    )
    result = _run_supervisor(command, timeout=30)

    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")

    pids = [int(line) for line in pid_log.read_text().split()]
    # 임계값만큼 자식이 시작·종료·재기동됐다.
    assert len(pids) == max_restarts

    # 재기동 사이마다 backoff 대기가 있었다(마지막은 한도 도달로 대기 없이 종료).
    stdout = result.stdout.decode("utf-8", "replace")
    assert stdout.lower().count("backoff") == max_restarts - 1

    # 감독 종료 후 자식이 하나도 남지 않는다.
    assert all(not _is_alive(pid) for pid in pids)


def test_supervisor_stops_at_restart_limit(tmp_path: Path) -> None:
    fake_child = tmp_path / "fake_ollama.py"
    fake_child.write_text(FAKE_CHILD, encoding="utf-8")
    pid_log = tmp_path / "pids.txt"
    pid_log.touch()

    serve = _serve_command(fake_child, pid_log, "short")
    command = (
        f"& '{SCRIPT}' -ServeCommand {serve} "
        f"-MaxRestarts 1 -BackoffSeconds 0.1 -MaxBackoffSeconds 0.5"
    )
    result = _run_supervisor(command, timeout=30)

    assert result.returncode == 0
    # 한도가 1이면 정확히 한 번만 시작하고 backoff 없이 종료한다.
    assert len(pid_log.read_text().split()) == 1
    assert result.stdout.decode("utf-8", "replace").lower().count("backoff") == 0


def test_supervisor_kills_live_child_on_forced_termination(tmp_path: Path) -> None:
    fake_child = tmp_path / "fake_ollama.py"
    fake_child.write_text(FAKE_CHILD, encoding="utf-8")
    pid_log = tmp_path / "pids.txt"
    pid_log.touch()

    serve = _serve_command(fake_child, pid_log, "long")
    command = (
        f"& '{SCRIPT}' -ServeCommand {serve} -BackoffSeconds 0.1 -MaxBackoffSeconds 0.5"
    )
    supervisor = subprocess.Popen(
        [
            POWERSHELL,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    child_pid = None
    try:
        child_pid = _wait_for_first_pid(pid_log, timeout=25)
        assert _is_alive(child_pid)

        # finally가 실행되지 않는 강제 종료 — kill-on-close 잡 오브젝트만으로 자식이 정리돼야 한다.
        supervisor.kill()
        supervisor.wait(timeout=15)

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and _is_alive(child_pid):
            time.sleep(0.2)
        assert not _is_alive(child_pid)
    finally:
        if supervisor.poll() is None:
            supervisor.kill()
            supervisor.wait(timeout=10)
        if child_pid is not None and _is_alive(child_pid):
            subprocess.run(
                ["taskkill", "/F", "/PID", str(child_pid)], capture_output=True
            )
        if supervisor.stdout is not None:
            supervisor.stdout.close()
        if supervisor.stderr is not None:
            supervisor.stderr.close()
