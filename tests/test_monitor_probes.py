"""운영 모니터 프로브 검증 — 이상 응답이 예외 전파 없이 구성요소 실패로 보고된다."""

import subprocess
from pathlib import Path

import httpx
import pytest
import respx

from gateway.monitor import probes
from gateway.monitor.probes import (
    parse_gpu_output,
    probe_disk,
    probe_gateway,
    probe_gpu,
    probe_ollama,
    probe_scheduled_tasks,
    probe_service,
)

pytestmark = pytest.mark.anyio

OLLAMA_BASE_URL = "http://ollama.test"
STACK_TASKS = ("Gateway", "Chat Ollama", "Embedding Ollama", "Watchdog")


def fake_powershell_run(stdout: str):
    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    return run


@respx.mock
async def test_probe_ollama_reports_unexpected_json_shape_as_failure(
    respx_mock: respx.Router,
) -> None:
    respx_mock.get(f"{OLLAMA_BASE_URL}/api/version").respond(200, json=["잘못된 형식"])
    respx_mock.get(f"{OLLAMA_BASE_URL}/api/ps").respond(200, json={"models": []})
    async with httpx.AsyncClient() as client:
        result = await probe_ollama(client, "chat_ollama", OLLAMA_BASE_URL)
    assert result["ok"] is False
    assert "예상 밖 응답 형식" in result["detail"]


@respx.mock
async def test_probe_ollama_rejects_missing_version(
    respx_mock: respx.Router,
) -> None:
    respx_mock.get(f"{OLLAMA_BASE_URL}/api/version").respond(200, json={})
    respx_mock.get(f"{OLLAMA_BASE_URL}/api/ps").respond(200, json={"models": []})
    async with httpx.AsyncClient() as client:
        result = await probe_ollama(client, "chat_ollama", OLLAMA_BASE_URL)
    assert result["ok"] is False
    assert "version 누락" in result["detail"]


@respx.mock
async def test_probe_ollama_rejects_non_list_models(
    respx_mock: respx.Router,
) -> None:
    respx_mock.get(f"{OLLAMA_BASE_URL}/api/version").respond(
        200, json={"version": "0.31.1"}
    )
    respx_mock.get(f"{OLLAMA_BASE_URL}/api/ps").respond(
        200, json={"models": "not-a-list"}
    )
    async with httpx.AsyncClient() as client:
        result = await probe_ollama(client, "chat_ollama", OLLAMA_BASE_URL)
    assert result["ok"] is False
    assert "models 목록 아님" in result["detail"]


@respx.mock
async def test_probe_ollama_sanitizes_non_finite_model_sizes(
    respx_mock: respx.Router,
) -> None:
    # 업스트림이 무한대·비수치 크기를 보내도 상태 JSON 직렬화가 깨지면 안 된다.
    respx_mock.get(f"{OLLAMA_BASE_URL}/api/version").respond(
        200, json={"version": "0.31.1"}
    )
    respx_mock.get(f"{OLLAMA_BASE_URL}/api/ps").respond(
        200,
        content=b'{"models": [{"name": 3, "size": 1e400, "size_vram": "big"}]}',
        headers={"content-type": "application/json"},
    )
    async with httpx.AsyncClient() as client:
        result = await probe_ollama(client, "chat_ollama", OLLAMA_BASE_URL)
    assert result["ok"] is True
    assert result["data"]["loaded_models"] == [
        {"name": None, "size_vram": None, "size": None}
    ]


@respx.mock
async def test_probe_ollama_skips_malformed_model_entries(
    respx_mock: respx.Router,
) -> None:
    respx_mock.get(f"{OLLAMA_BASE_URL}/api/version").respond(
        200, json={"version": "0.31.1"}
    )
    respx_mock.get(f"{OLLAMA_BASE_URL}/api/ps").respond(
        200, json={"models": [{"name": "정상"}, "문자열", 3]}
    )
    async with httpx.AsyncClient() as client:
        result = await probe_ollama(client, "chat_ollama", OLLAMA_BASE_URL)
    assert result["ok"] is True
    assert [model["name"] for model in result["data"]["loaded_models"]] == ["정상"]


@respx.mock
async def test_probe_gateway_treats_only_200_and_401_as_alive(
    respx_mock: respx.Router,
) -> None:
    respx_mock.get("http://gw.test/health").respond(503)
    async with httpx.AsyncClient() as client:
        result = await probe_gateway(client, "http://gw.test/health")
    assert result["ok"] is False
    assert "503" in result["detail"]


def test_parse_gpu_output_parses_numbers_and_rejects_not_available() -> None:
    parsed = parse_gpu_output("GTX 1080 Ti, 1048, 11264, 61.92, 57")
    assert parsed["vram_used_mib"] == 1048.0
    assert parsed["power_watt"] == pytest.approx(61.92)
    with pytest.raises(ValueError):
        parse_gpu_output("GTX 1080 Ti, 1048, 11264, N/A, 57")


def test_probe_gpu_reports_subprocess_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        raise subprocess.CalledProcessError(1, "nvidia-smi")

    monkeypatch.setattr(probes.subprocess, "run", failing_run)
    result = probe_gpu()
    assert result["ok"] is False
    assert "CalledProcessError" in result["detail"]


def test_probe_scheduled_tasks_maps_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = (
        '[{"TaskName":"Gateway","State":4},{"TaskName":"Chat Ollama","State":4},'
        '{"TaskName":"Embedding Ollama","State":4},{"TaskName":"Watchdog","State":3}]'
    )
    monkeypatch.setattr(probes.subprocess, "run", fake_powershell_run(stdout))
    result = probe_scheduled_tasks("\\LocalFirstInferenceGateway\\", STACK_TASKS)
    assert result["ok"] is False
    assert result["detail"] == "실행 중 3/4"
    assert result["data"]["tasks"]["Watchdog"] == "Ready"
    assert result["data"]["tasks"]["Gateway"] == "Running"


def test_probe_scheduled_tasks_marks_missing_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stdout = '[{"TaskName":"Gateway","State":4}]'
    monkeypatch.setattr(probes.subprocess, "run", fake_powershell_run(stdout))
    result = probe_scheduled_tasks("\\LocalFirstInferenceGateway\\", STACK_TASKS)
    assert result["ok"] is False
    assert result["data"]["tasks"]["Watchdog"] == "미등록"


def test_probe_scheduled_tasks_guides_when_nothing_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SYSTEM 작업은 비관리자 셸에 보이지 않는다 — 원시 오류 대신 안내를 낸다."""
    monkeypatch.setattr(probes.subprocess, "run", fake_powershell_run("[]"))
    result = probe_scheduled_tasks("\\LocalFirstInferenceGateway\\", STACK_TASKS)
    assert result["ok"] is False
    assert "관리자 셸" in result["detail"]


def test_probe_scheduled_tasks_reports_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timing_out_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        raise subprocess.TimeoutExpired("powershell", 10)

    monkeypatch.setattr(probes.subprocess, "run", timing_out_run)
    result = probe_scheduled_tasks("\\LocalFirstInferenceGateway\\", STACK_TASKS)
    assert result["ok"] is False
    assert "TimeoutExpired" in result["detail"]


def test_probe_service_maps_running_and_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(probes.subprocess, "run", fake_powershell_run("4"))
    running = probe_service("cloudflared")
    assert running["ok"] is True
    assert running["detail"] == "Running"

    monkeypatch.setattr(probes.subprocess, "run", fake_powershell_run('"missing"'))
    missing = probe_service("cloudflared")
    assert missing["ok"] is False
    assert "보이지 않음" in missing["detail"]

    monkeypatch.setattr(probes.subprocess, "run", fake_powershell_run("1"))
    stopped = probe_service("cloudflared")
    assert stopped["ok"] is False
    assert stopped["detail"] == "Stopped"


def test_probe_disk_flags_low_free_space(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gib = 1024**3

    def scarce_usage(path: object) -> object:
        return type("Usage", (), {"free": 2 * gib, "total": 100 * gib})()

    monkeypatch.setattr(probes.shutil, "disk_usage", scarce_usage)
    scarce = probe_disk(tmp_path)
    assert scarce["ok"] is False
    assert "임계" in scarce["detail"]

    def ample_usage(path: object) -> object:
        return type("Usage", (), {"free": 50 * gib, "total": 100 * gib})()

    monkeypatch.setattr(probes.shutil, "disk_usage", ample_usage)
    ample = probe_disk(tmp_path)
    assert ample["ok"] is True
