"""추론 스택 구성요소의 실시간 상태 프로브 — 조회만 하고 비밀을 읽지 않는다.

Windows 예약 작업·서비스 상태는 로캘 의존적인 텍스트 출력 대신 PowerShell의
JSON 직렬화로 읽는다. 모든 프로브는 예외 대신 ok/detail로 실패를 보고한다.
"""

import json
import math
import shutil
import subprocess
from pathlib import Path

import httpx

SUBPROCESS_TIMEOUT_SECONDS = 10
_TASK_STATES = {0: "Unknown", 1: "Disabled", 2: "Queued", 3: "Ready", 4: "Running"}
_SERVICE_STATES = {
    1: "Stopped",
    2: "StartPending",
    3: "StopPending",
    4: "Running",
    5: "ContinuePending",
    6: "PausePending",
    7: "Paused",
}


def _failure(name: str, detail: str) -> dict:
    return {"name": name, "ok": False, "detail": detail, "data": {}}


async def probe_gateway(client: httpx.AsyncClient, health_url: str) -> dict:
    """게이트웨이 생존 프로브 — 401도 생존이다(키 없이 확인하는 watchdog과 같은 해석)."""
    name = "gateway"
    try:
        response = await client.get(health_url)
    except httpx.HTTPError as error:
        return _failure(name, f"{type(error).__name__}: {error}")
    if response.status_code == 200:
        return {"name": name, "ok": True, "detail": "응답함", "data": {}}
    if response.status_code == 401:
        return {"name": name, "ok": True, "detail": "응답함(인증 요구)", "data": {}}
    return _failure(name, f"HTTP {response.status_code}")


async def probe_ollama(client: httpx.AsyncClient, name: str, base_url: str) -> dict:
    """Ollama 생존·상주 모델 프로브 — 버전과 적재 모델의 VRAM 점유를 함께 본다."""
    try:
        version_response = await client.get(f"{base_url}/api/version")
        version_response.raise_for_status()
        version_body = version_response.json()
        ps_response = await client.get(f"{base_url}/api/ps")
        ps_response.raise_for_status()
        ps_body = ps_response.json()
    except (httpx.HTTPError, ValueError) as error:
        return _failure(name, f"{type(error).__name__}: {error}")
    if not isinstance(version_body, dict) or not isinstance(ps_body, dict):
        return _failure(name, "예상 밖 응답 형식")
    version = version_body.get("version")
    raw_models = ps_body.get("models")
    # 형태가 다른 응답을 정상으로 판정하지 않는다 — 빈 모델 목록은 정상(미적재)이다.
    if not isinstance(version, str) or not version:
        return _failure(name, "예상 밖 응답 형식 — version 누락")
    if not isinstance(raw_models, list):
        return _failure(name, "예상 밖 응답 형식 — models 목록 아님")
    models = [
        {
            "name": _clean_text(model.get("name")),
            "size_vram": _clean_number(model.get("size_vram")),
            "size": _clean_number(model.get("size")),
        }
        for model in raw_models
        if isinstance(model, dict)
    ]
    return {
        "name": name,
        "ok": True,
        "detail": f"v{version}",
        "data": {"version": version, "loaded_models": models},
    }


def _clean_text(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _clean_number(value: object) -> int | float | None:
    """상태 JSON에 싣는 수치 — 무한대·비수치가 직렬화를 깨지 않게 None으로 바꾼다."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    if not math.isfinite(value):
        return None
    return value


def _run_powershell_json(command: str) -> object:
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
        check=True,
    )
    output = completed.stdout.strip()
    if not output:
        return []
    return json.loads(output)


def probe_scheduled_tasks(task_path: str, task_names: tuple[str, ...]) -> dict:
    """저장소가 등록한 예약 작업들의 상태.

    SYSTEM으로 등록된 작업은 비관리자 셸에 보이지 않으므로, 조회 결과가 비면
    오류 대신 관리자 셸 안내로 보고한다.
    """
    name = "scheduled_tasks"
    # 비관리자 셸의 조회 오류는 PowerShell 종료 코드를 1로 만든다 — try/catch로
    # 항상 유효한 JSON을 내고, 단일 작업도 배열로 유지한다(InputObject 직렬화).
    command = (
        "try { $tasks = @(Get-ScheduledTask -TaskPath"
        f" '{task_path}' -ErrorAction Stop | Select-Object TaskName, State);"
        " ConvertTo-Json -InputObject $tasks -Compress } catch { '[]' }"
    )
    try:
        parsed = _run_powershell_json(command)
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        return _failure(name, f"{type(error).__name__}: {error}")
    rows = parsed if isinstance(parsed, list) else [parsed]
    states = {
        row.get("TaskName"): _TASK_STATES.get(row.get("State"), "Unknown")
        for row in rows
        if isinstance(row, dict)
    }
    if not states:
        return _failure(name, "작업이 보이지 않음 — 관리자 셸이 필요하거나 미등록")
    tasks = {task: states.get(task, "미등록") for task in task_names}
    running = sum(1 for state in tasks.values() if state == "Running")
    return {
        "name": name,
        "ok": running == len(task_names),
        "detail": f"실행 중 {running}/{len(task_names)}",
        "data": {"tasks": tasks},
    }


def probe_service(service_name: str) -> dict:
    """Windows 서비스 상태 프로브 — cloudflared 터널 서비스를 본다."""
    name = f"service:{service_name}"
    command = (
        f"try {{ (Get-Service -Name '{service_name}' -ErrorAction Stop)"
        ".Status.value__ } catch { '\"missing\"' }"
    )
    try:
        parsed = _run_powershell_json(command)
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        return _failure(name, f"{type(error).__name__}: {error}")
    if parsed == "missing":
        return _failure(name, "서비스가 보이지 않음 — 미설치이거나 조회 불가")
    state = _SERVICE_STATES.get(parsed if isinstance(parsed, int) else -1, "Unknown")
    return {
        "name": name,
        "ok": state == "Running",
        "detail": state,
        "data": {"state": state},
    }


def parse_gpu_output(output: str) -> dict:
    """nvidia-smi CSV 한 줄을 수치로 파싱한다 — 'N/A' 같은 비수치는 ValueError."""
    gpu_name, memory_used, memory_total, power, temperature = (
        value.strip() for value in output.split(",")
    )
    return {
        "gpu": gpu_name,
        "vram_used_mib": float(memory_used),
        "vram_total_mib": float(memory_total),
        "power_watt": float(power),
        "temperature_c": float(temperature),
    }


def probe_gpu() -> dict:
    """GPU 프로브 — nvidia-smi로 VRAM 사용량·전력·온도를 읽는다."""
    name = "gpu"
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=SUBPROCESS_TIMEOUT_SECONDS,
            check=True,
        ).stdout.strip()
        data = parse_gpu_output(output)
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        return _failure(name, f"{type(error).__name__}: {error}")
    detail = (
        f"{data['vram_used_mib']:.0f}/{data['vram_total_mib']:.0f}MiB"
        f" · {data['power_watt']:.1f}W · {data['temperature_c']:.0f}°C"
    )
    return {"name": name, "ok": True, "detail": detail, "data": data}


# 관측 로그 회전 상한(약 80MiB)과 배포 사본 갱신 여유를 감안한 운영 최소 여유 공간.
MINIMUM_FREE_DISK_GIB = 5.0


def probe_disk(target: Path) -> dict:
    """상태 디렉터리가 있는 볼륨의 디스크 여유 — 임계 미만이면 이상으로 판정한다."""
    name = "disk"
    try:
        usage = shutil.disk_usage(target.anchor or target)
    except OSError as error:
        return _failure(name, f"{type(error).__name__}: {error}")
    free_gib = usage.free / 1024**3
    total_gib = usage.total / 1024**3
    detail = f"여유 {free_gib:.1f}/{total_gib:.1f}GiB"
    if free_gib < MINIMUM_FREE_DISK_GIB:
        detail += f" — 임계 {MINIMUM_FREE_DISK_GIB:.0f}GiB 미만"
    return {
        "name": name,
        "ok": free_gib >= MINIMUM_FREE_DISK_GIB,
        "detail": detail,
        "data": {"free_gib": round(free_gib, 2), "total_gib": round(total_gib, 2)},
    }
