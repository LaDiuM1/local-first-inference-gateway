"""운영 모니터 앱 계약 검증 — 상태 JSON과 대시보드가 프로브·집계를 노출한다."""

import json
import time
from pathlib import Path

import httpx
import pytest
import respx

from gateway.monitor.app import MonitorSettings, create_monitor_app

pytestmark = pytest.mark.anyio

GATEWAY_HEALTH_URL = "http://gateway.test/health"
CHAT_BASE_URL = "http://chat.test"
EMBEDDING_BASE_URL = "http://embed.test"


def write_log(directory: Path) -> None:
    directory.mkdir(parents=True)
    records = [
        {
            "event": "request",
            "started_at": time.time() - 60,
            "path": "/v1/chat/completions",
            "alias": "chat",
            "stream": True,
            "status": 200,
            "provider": "local",
            "response_start_ms": 2000.0,
            "duration_ms": 3500.0,
            "completed": True,
        }
    ]
    # 마지막 줄도 개행으로 완결한다 — 개행 없는 꼬리는 손상이 아니라 쓰기 중 미완성으로 본다.
    lines = [json.dumps(record) for record in records] + ['{"broken":']
    (directory / "requests.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def mock_upstreams(router: respx.Router) -> None:
    router.get(GATEWAY_HEALTH_URL).respond(401, json={"error": "unauthorized"})
    for base in (CHAT_BASE_URL, EMBEDDING_BASE_URL):
        router.get(f"{base}/api/version").respond(200, json={"version": "0.31.1"})
        router.get(f"{base}/api/ps").respond(
            200,
            json={"models": [{"name": "fake-model", "size": 10, "size_vram": 10}]},
        )


@respx.mock
async def test_status_endpoint_reports_probes_and_request_summary(
    tmp_path: Path, respx_mock: respx.Router
) -> None:
    mock_upstreams(respx_mock)
    log_directory = tmp_path / "logs"
    write_log(log_directory)
    settings = MonitorSettings(
        log_directory=log_directory,
        gateway_health_url=GATEWAY_HEALTH_URL,
        chat_base_url=CHAT_BASE_URL,
        embedding_base_url=EMBEDDING_BASE_URL,
        include_system_probes=False,
        summary_windows_seconds=(3600,),
    )
    transport = httpx.ASGITransport(app=create_monitor_app(settings))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://monitor.test"
    ) as client:
        response = await client.get("/api/status")
    assert response.status_code == 200
    status = response.json()
    components = {row["name"]: row for row in status["components"]}
    assert components["gateway"]["ok"] is True
    assert "인증 요구" in components["gateway"]["detail"]
    assert components["chat_ollama"]["data"]["loaded_models"][0]["name"] == (
        "fake-model"
    )
    assert components["embedding_ollama"]["ok"] is True
    [summary] = status["requests"]
    assert summary["count"] == 1
    assert summary["providers"] == {"local": 1}
    assert status["log"]["readable"] is True
    assert status["log"]["invalid_lines"] == 1


@respx.mock
async def test_status_reports_unreadable_log_directory(
    tmp_path: Path, respx_mock: respx.Router
) -> None:
    """접근 불가 로그를 무트래픽 0건으로 오인하지 않도록 readable을 노출한다."""
    mock_upstreams(respx_mock)
    settings = MonitorSettings(
        log_directory=tmp_path / "없는-디렉터리",
        gateway_health_url=GATEWAY_HEALTH_URL,
        chat_base_url=CHAT_BASE_URL,
        embedding_base_url=EMBEDDING_BASE_URL,
        include_system_probes=False,
        summary_windows_seconds=(3600,),
    )
    transport = httpx.ASGITransport(app=create_monitor_app(settings))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://monitor.test"
    ) as client:
        response = await client.get("/api/status")
    assert response.status_code == 200
    assert response.json()["log"]["readable"] is False


@respx.mock
async def test_status_stays_200_with_corrupted_log_lines(
    tmp_path: Path, respx_mock: respx.Router
) -> None:
    """NaN·비 UTF-8 손상 줄이 있어도 상태 조회는 격리 집계로 200을 유지한다."""
    mock_upstreams(respx_mock)
    log_directory = tmp_path / "logs"
    log_directory.mkdir(parents=True)
    (log_directory / "requests.jsonl").write_bytes(
        b'{"event": "request", "duration_ms": NaN}\n'
        + b"\xff\xfe broken\n"
        + json.dumps(
            {
                "event": "request",
                "started_at": time.time() - 30,
                "path": "/v1/chat/completions",
                "alias": "chat",
                "status": 200,
                "provider": "local",
                "duration_ms": 100.0,
                "completed": True,
            }
        ).encode()
        + b"\n"
    )
    settings = MonitorSettings(
        log_directory=log_directory,
        gateway_health_url=GATEWAY_HEALTH_URL,
        chat_base_url=CHAT_BASE_URL,
        embedding_base_url=EMBEDDING_BASE_URL,
        include_system_probes=False,
        summary_windows_seconds=(3600,),
    )
    transport = httpx.ASGITransport(app=create_monitor_app(settings))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://monitor.test"
    ) as client:
        response = await client.get("/api/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["log"]["records"] == 1
    assert payload["log"]["invalid_lines"] == 2
    assert payload["requests"][0]["count"] == 1


@respx.mock
async def test_unserializable_probe_result_degrades_component_not_status(
    tmp_path: Path, respx_mock: respx.Router, monkeypatch: pytest.MonkeyPatch
) -> None:
    """무한대 수·고립 surrogate가 담긴 프로브 결과는 상태 500이 아니라 구성요소 실패다."""
    mock_upstreams(respx_mock)

    async def poisoned_probe(client: httpx.AsyncClient, health_url: str) -> dict:
        return {"name": "gateway", "ok": True, "detail": "ok", "data": {"v": "\ud800"}}

    monkeypatch.setattr("gateway.monitor.app.probe_gateway", poisoned_probe)
    settings = MonitorSettings(
        log_directory=tmp_path / "logs",
        gateway_health_url=GATEWAY_HEALTH_URL,
        chat_base_url=CHAT_BASE_URL,
        embedding_base_url=EMBEDDING_BASE_URL,
        include_system_probes=False,
        summary_windows_seconds=(3600,),
    )
    transport = httpx.ASGITransport(app=create_monitor_app(settings))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://monitor.test"
    ) as client:
        response = await client.get("/api/status")
    assert response.status_code == 200
    components = {row["name"]: row for row in response.json()["components"]}
    assert components["gateway"]["ok"] is False


@respx.mock
async def test_unencodable_probe_exception_message_degrades_component_not_status(
    tmp_path: Path, respx_mock: respx.Router, monkeypatch: pytest.MonkeyPatch
) -> None:
    """예외 메시지에 실린 인코딩 불가 문자도 상태 500이 아니라 구성요소 실패로 끝난다."""
    mock_upstreams(respx_mock)

    async def exploding_probe(client: httpx.AsyncClient, health_url: str) -> dict:
        raise RuntimeError("\ud800 손상 메시지")

    monkeypatch.setattr("gateway.monitor.app.probe_gateway", exploding_probe)
    settings = MonitorSettings(
        log_directory=tmp_path / "logs",
        gateway_health_url=GATEWAY_HEALTH_URL,
        chat_base_url=CHAT_BASE_URL,
        embedding_base_url=EMBEDDING_BASE_URL,
        include_system_probes=False,
        summary_windows_seconds=(3600,),
    )
    transport = httpx.ASGITransport(app=create_monitor_app(settings))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://monitor.test"
    ) as client:
        response = await client.get("/api/status")
    assert response.status_code == 200
    components = {row["name"]: row for row in response.json()["components"]}
    assert components["gateway"]["ok"] is False
    assert "RuntimeError" in components["gateway"]["detail"]


@respx.mock
async def test_probe_exception_degrades_single_component_not_whole_status(
    tmp_path: Path, respx_mock: respx.Router, monkeypatch: pytest.MonkeyPatch
) -> None:
    mock_upstreams(respx_mock)

    async def broken_probe(client: httpx.AsyncClient, health_url: str) -> dict:
        raise RuntimeError("프로브 내부 결함")

    monkeypatch.setattr("gateway.monitor.app.probe_gateway", broken_probe)
    settings = MonitorSettings(
        log_directory=tmp_path / "logs",
        gateway_health_url=GATEWAY_HEALTH_URL,
        chat_base_url=CHAT_BASE_URL,
        embedding_base_url=EMBEDDING_BASE_URL,
        include_system_probes=False,
        summary_windows_seconds=(3600,),
    )
    transport = httpx.ASGITransport(app=create_monitor_app(settings))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://monitor.test"
    ) as client:
        response = await client.get("/api/status")
    assert response.status_code == 200
    components = {row["name"]: row for row in response.json()["components"]}
    assert components["gateway"]["ok"] is False
    assert "RuntimeError" in components["gateway"]["detail"]
    assert components["chat_ollama"]["ok"] is True


async def test_dashboard_serves_inline_page_without_external_assets(
    tmp_path: Path,
) -> None:
    settings = MonitorSettings(log_directory=tmp_path, include_system_probes=False)
    transport = httpx.ASGITransport(app=create_monitor_app(settings))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://monitor.test"
    ) as client:
        response = await client.get("/")
    assert response.status_code == 200
    page = response.text
    assert "운영 모니터" in page
    assert "/api/status" in page
    assert "http://" not in page.replace("http://127.0.0.1", "")
    assert "https://" not in page
