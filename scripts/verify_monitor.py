"""운영 모니터 결정적 검증: 로그 후처리와 스택 프로브가 대시보드 계약대로 동작하는지 확인한다.

게이트웨이·두 Ollama를 로컬 fake로 돌리고 합성 관측 로그를 임시 디렉터리에 두어
실 운영 스택·비밀 없이 재실행할 수 있다. 시스템 프로브(예약 작업·서비스·GPU·디스크)는
머신 상태에 의존하므로 여기서는 끄고, 실측 확인은 운영 스택 대상 수동 실행으로 한다.

실행: uv run python scripts/verify_monitor.py
"""

import asyncio
import json
import socket
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse

import gateway.monitor.__main__ as monitor_main
from gateway.monitor.app import MonitorSettings, create_monitor_app

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")

HOST = "127.0.0.1"
STARTUP_TIMEOUT_SECONDS = 15

failures: list[str] = []


def check(label: str, passed: bool, detail: str) -> None:
    mark = "PASS"
    if not passed:
        mark = "FAIL"
        failures.append(label)
    print(f"[{mark}] {label} - {detail}")


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind((HOST, 0))
        return probe.getsockname()[1]


# chat과 embed fake는 경로 접두사와 표식(버전·모델명)을 다르게 둔다 — 모니터가
# 구성요소별로 올바른 대상 주소에 프로브했는지까지 단언하기 위해서다.
fake = FastAPI()


@fake.get("/health")
async def fake_gateway_health() -> JSONResponse:
    # 운영 게이트웨이는 키 없는 프로브에 401을 준다 — 모니터는 이를 생존으로 판정해야 한다.
    return JSONResponse(status_code=401, content={"error": "unauthorized"})


@fake.get("/chat/api/version")
async def fake_chat_version() -> JSONResponse:
    return JSONResponse({"version": "chat-0.31.1"})


@fake.get("/chat/api/ps")
async def fake_chat_ps() -> JSONResponse:
    return JSONResponse(
        {"models": [{"name": "fake-chat", "size": 100, "size_vram": 100}]}
    )


@fake.get("/embed/api/version")
async def fake_embed_version() -> JSONResponse:
    return JSONResponse({"version": "embed-0.31.1"})


@fake.get("/embed/api/ps")
async def fake_embed_ps() -> JSONResponse:
    return JSONResponse(
        {"models": [{"name": "fake-embed", "size": 50, "size_vram": 0}]}
    )


def start_fake(port: int) -> None:
    server = uvicorn.Server(
        uvicorn.Config(fake, host=HOST, port=port, log_level="critical")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if server.started:
            return
        time.sleep(0.05)
    raise RuntimeError("fake 업스트림이 기동하지 않았다")


def write_synthetic_log(directory: Path, now_epoch: float) -> None:
    """오류 분류·창 필터·회로 이벤트·손상 줄을 모두 덮는 합성 관측 로그."""
    directory.mkdir(parents=True)
    records = [
        {
            "event": "request",
            "started_at": now_epoch - 60,
            "path": "/v1/chat/completions",
            "alias": "chat",
            "stream": True,
            "status": 200,
            "provider": "local",
            "response_start_ms": 2000.0,
            "duration_ms": 4000.0,
            "completed": True,
        },
        {
            "event": "request",
            "started_at": now_epoch - 58,
            "path": "/v1/chat/completions",
            "alias": "chat",
            "stream": False,
            "status": 200,
            "provider": "local",
            "response_start_ms": 2500.0,
            "duration_ms": 3000.0,
            "completed": True,
        },
        {
            "event": "request",
            "started_at": now_epoch - 120,
            "path": "/v1/responses",
            "alias": "vision",
            "stream": False,
            "status": 200,
            "provider": "openai",
            "local_failure_reason": "local_error_status",
            "response_start_ms": 900.0,
            "duration_ms": 5000.0,
            "completed": True,
        },
        {
            "event": "request",
            "started_at": now_epoch - 90,
            "path": "/v1/embeddings",
            "alias": "embed",
            "stream": False,
            "status": 502,
            "provider": None,
            "response_start_ms": None,
            "duration_ms": 120.0,
            "completed": True,
        },
        {
            "event": "request",
            "started_at": now_epoch - 7200,
            "path": "/v1/chat/completions",
            "alias": "chat",
            "stream": False,
            "status": 200,
            "provider": "local",
            "response_start_ms": 1000.0,
            "duration_ms": 1500.0,
            "completed": True,
        },
        # 모니터·watchdog의 /health 폴링 — 추론 지표에서 제외되어야 한다.
        {
            "event": "request",
            "started_at": now_epoch - 30,
            "path": "/health",
            "alias": None,
            "stream": None,
            "status": 401,
            "provider": None,
            "response_start_ms": 0.4,
            "duration_ms": 0.9,
            "completed": True,
        },
        {
            "event": "request",
            "started_at": now_epoch - 25,
            "path": "/health",
            "alias": None,
            "stream": None,
            "status": 401,
            "provider": None,
            "response_start_ms": 0.4,
            "duration_ms": 0.9,
            "completed": True,
        },
        {
            "event": "circuit",
            "time": datetime.fromtimestamp(now_epoch - 100, tz=UTC).isoformat(),
            "alias": "chat",
            "state": "open",
        },
    ]
    # 마지막 줄도 개행으로 완결한다 — 개행 없는 꼬리는 손상이 아니라 쓰기 중 미완성으로 본다.
    lines = [json.dumps(record) for record in records] + ['{"잘린 줄":']
    (directory / "requests.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


async def fetch_status_and_page(settings: MonitorSettings) -> tuple[dict, str]:
    transport = httpx.ASGITransport(app=create_monitor_app(settings))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://monitor.verify"
    ) as client:
        status_response = await client.get("/api/status")
        page_response = await client.get("/")
    status_response.raise_for_status()
    page_response.raise_for_status()
    return status_response.json(), page_response.text


def run_verification() -> None:
    fake_port = free_port()
    start_fake(fake_port)
    fake_base = f"http://{HOST}:{fake_port}"
    with TemporaryDirectory(prefix="monitor-verify-") as directory:
        log_directory = Path(directory) / "logs"
        write_synthetic_log(log_directory, time.time())
        settings = MonitorSettings(
            log_directory=log_directory,
            gateway_health_url=f"{fake_base}/health",
            chat_base_url=f"{fake_base}/chat",
            embedding_base_url=f"{fake_base}/embed",
            include_system_probes=False,
            summary_windows_seconds=(3600, 86400),
        )
        status, page = asyncio.run(fetch_status_and_page(settings))

    components = {row["name"]: row for row in status["components"]}
    check(
        "게이트웨이 401을 생존으로 판정(비밀 미사용)",
        components["gateway"]["ok"] is True
        and "인증 요구" in components["gateway"]["detail"],
        components["gateway"]["detail"],
    )
    check(
        "두 Ollama 프로브가 각자 올바른 대상 주소를 조회",
        components["chat_ollama"]["ok"] is True
        and components["embedding_ollama"]["ok"] is True
        and components["chat_ollama"]["data"]["version"] == "chat-0.31.1"
        and components["chat_ollama"]["data"]["loaded_models"][0]["name"] == "fake-chat"
        and components["embedding_ollama"]["data"]["version"] == "embed-0.31.1"
        and components["embedding_ollama"]["data"]["loaded_models"][0]["name"]
        == "fake-embed",
        f"chat={components['chat_ollama']['detail']},"
        f" embed={components['embedding_ollama']['detail']}",
    )
    hour_summary, day_summary = status["requests"]
    check(
        "창 필터(1시간 4건 / 24시간 5건)",
        hour_summary["count"] == 4 and day_summary["count"] == 5,
        f"1h={hour_summary['count']}, 24h={day_summary['count']}",
    )
    check(
        "비추론 경로(/health) 지표 오염 차단",
        hour_summary["non_inference_count"] == 2 and hour_summary["client_errors"] == 0,
        f"non_inference={hour_summary['non_inference_count']},"
        f" 4xx={hour_summary['client_errors']}",
    )
    check(
        "오류 분류·provider 분포",
        hour_summary["operational_errors"] == 1
        and hour_summary["providers"] == {"local": 2, "openai": 1, "none": 1}
        and hour_summary["streaming"] == 1,
        f"providers={hour_summary['providers']}",
    )
    check(
        "지연 백분위(운영 오류 제외)",
        hour_summary["latency_ms"]["total"] == {"p50": 4000.0, "p95": 5000.0}
        and hour_summary["latency_ms"]["response_start"]
        == {"p50": 2000.0, "p95": 2500.0},
        json.dumps(hour_summary["latency_ms"], ensure_ascii=False),
    )
    check(
        "별칭 분해·회로 이벤트",
        hour_summary["aliases"]["chat"]["count"] == 2
        and hour_summary["aliases"]["embed"]["operational_errors"] == 1
        and [event["state"] for event in hour_summary["circuit_events"]] == ["open"],
        f"aliases={sorted(hour_summary['aliases'])}",
    )
    check(
        "로그 가용성·손상 줄 집계 노출",
        status["log"]["readable"] is True
        and status["log"]["invalid_lines"] == 1
        and status["log"]["records"] == 8,
        f"readable={status['log']['readable']}, records={status['log']['records']}",
    )
    check(
        "대시보드 페이지(외부 자산 없음)",
        "운영 모니터" in page and "/api/status" in page and "https://" not in page,
        f"{len(page.splitlines())}줄",
    )
    # 상수 확인이 아니라 실행 진입점을 실제로 호출해 바인딩 인자를 단언한다 —
    # uvicorn.run이 다른 host로 바뀌는 회귀를 잡는다.
    with (
        patch.object(monitor_main.uvicorn, "run") as fake_run,
        patch.object(sys, "argv", ["gateway.monitor"]),
    ):
        monitor_main.main()
    bound = fake_run.call_args.kwargs
    check(
        "실행 진입점이 loopback에만 바인딩",
        bound.get("host") == "127.0.0.1" and bound.get("port") == 29100,
        f"host={bound.get('host')}, port={bound.get('port')}",
    )

    if failures:
        print(f"\n운영 모니터 검증 실패 {len(failures)}건: {', '.join(failures)}")
        raise SystemExit(1)
    print("\n운영 모니터 자동 검증 통과 - 로그 후처리와 프로브가 계약대로 동작한다.")


if __name__ == "__main__":
    run_verification()
