"""7단계 결정적 검증: 동시 요청의 요청별 관측 기록이 사후 진단 가능한 형태로 남는지 확인한다.

게이트웨이의 업스트림을 모두 로컬 fake로 돌리고 실제 OpenAI·Ollama를 호출하지 않는다.
동시 6건의 처리 구간 겹침, 단계별 상대 시간의 순서, 스트리밍 완결, 폴백 provider와 실패 사유,
회로 차단기 전환 이벤트, 인증 실패 기록, x-request-id 대조, 프롬프트·키 미기록을 검사한다.

실행: uv run python scripts/verify_stage7.py
"""

import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from verification_auth import UVICORN_APPLICATION_ARGUMENTS, VerificationAuth

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")

HOST = "127.0.0.1"
STARTUP_TIMEOUT_SECONDS = 15
REQUEST_TIMEOUT_SECONDS = 15
LOG_WAIT_TIMEOUT_SECONDS = 10
CONCURRENT_REQUESTS = 6
LOCAL_CHAT_LATENCY_SECONDS = 0.3
FAILURE_THRESHOLD = 3
PROMPT_MARKER = "OBS-MARKER-비공개-프롬프트"


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind((HOST, 0))
        return probe.getsockname()[1]


FAKE_PORT = free_port()
GATEWAY_PORT = free_port()
FAKE_BASE_URL = f"http://{HOST}:{FAKE_PORT}"
GATEWAY_BASE_URL = f"http://{HOST}:{GATEWAY_PORT}"
AUTH = VerificationAuth.create("stage7-verifier")
AUTH_HEADERS = AUTH.headers
# apply_to가 게이트웨이에 지정하는 관측 로그 경로와 같은 위치를 읽는다.
REQUEST_LOG_PATH = AUTH.store_path.parent / "request-logs" / "requests.jsonl"

CONTROL = {"chat": "ok"}
failures: list[str] = []


def check(label: str, passed: bool, detail: str) -> None:
    mark = "PASS"
    if not passed:
        mark = "FAIL"
        failures.append(label)
    print(f"[{mark}] {label} - {detail}")


fake = FastAPI()


@fake.post("/v1/chat/completions")
async def fake_local_chat(request: Request) -> object:
    await request.body()
    if CONTROL["chat"] == "fail":
        return JSONResponse(status_code=503, content={"error": "local down"})
    if CONTROL["chat"] == "stream":
        return StreamingResponse(sse(), media_type="text/event-stream")
    await asyncio.sleep(LOCAL_CHAT_LATENCY_SECONDS)
    return JSONResponse(chat_completion("local"))


@fake.post("/openai-v1/chat/completions")
async def fake_openai_chat(request: Request) -> object:
    await request.body()
    return JSONResponse(chat_completion("openai"))


@fake.post("/api/embed")
async def fake_embed(request: Request) -> object:
    await request.body()
    return JSONResponse(
        {
            "model": "snowflake-arctic-embed2",
            "embeddings": [[0.1] * 1024],
            "prompt_eval_count": 1,
        }
    )


def chat_completion(marker: str) -> dict:
    return {
        "id": marker,
        "object": "chat.completion",
        "model": marker,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": marker},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


async def sse() -> AsyncIterator[bytes]:
    yield b'data: {"choices":[{"delta":{"content":"local"}}]}\n\n'
    yield b"data: [DONE]\n\n"


def start_fake() -> uvicorn.Server:
    server = uvicorn.Server(
        uvicorn.Config(fake, host=HOST, port=FAKE_PORT, log_level="critical")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if server.started:
            return server
        time.sleep(0.05)
    raise RuntimeError("fake upstream did not start")


def start_gateway() -> subprocess.Popen:
    environment = {
        **os.environ,
        "GATEWAY_OLLAMA_BASE_URL": FAKE_BASE_URL,
        "GATEWAY_EMBEDDING_OLLAMA_BASE_URL": FAKE_BASE_URL,
        "GATEWAY_OPENAI_BASE_URL": f"{FAKE_BASE_URL}/openai-v1",
        "GATEWAY_CIRCUIT_BREAKER_FAILURE_THRESHOLD": str(FAILURE_THRESHOLD),
        "GATEWAY_CIRCUIT_BREAKER_OPEN_SECONDS": "30",
        "OPENAI_API_KEY": "stage7-fake-openai-key",
    }
    AUTH.apply_to(environment)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            *UVICORN_APPLICATION_ARGUMENTS,
            "--host",
            HOST,
            "--port",
            str(GATEWAY_PORT),
            "--log-level",
            "critical",
        ],
        env=environment,
    )
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("gateway exited during startup")
        try:
            health = httpx.get(
                f"{GATEWAY_BASE_URL}/health", headers=AUTH_HEADERS, timeout=1
            )
            if health.status_code == 200:
                return process
        except httpx.TransportError:
            time.sleep(0.1)
    stop_gateway(process)
    raise RuntimeError("gateway did not start")


def stop_gateway(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def read_log_records() -> list[dict]:
    try:
        text = REQUEST_LOG_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    return [json.loads(line) for line in text.splitlines()]


def wait_for_records(predicate, expected_count: int, label: str) -> list[dict]:
    """조건에 맞는 요청 기록이 기대 건수만큼 쌓일 때까지 기다린다.

    기록은 요청 종료 직후 전용 스레드가 쓰므로 짧은 대기가 필요하고, 기동 확인용 /health
    폴링도 기록에 남으므로 절대 건수가 아니라 조건으로 대상을 고른다.
    """
    deadline = time.monotonic() + LOG_WAIT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        records = [
            record
            for record in read_log_records()
            if record["event"] == "request" and predicate(record)
        ]
        if len(records) >= expected_count:
            return records
        time.sleep(0.1)
    raise RuntimeError(f"기대한 관측 기록이 없다: {label}")


def concurrent_chat_calls(count: int) -> list[httpx.Response]:
    def call(index: int) -> httpx.Response:
        prompt = f"동시 요청 {index} {PROMPT_MARKER}"
        with httpx.Client(
            base_url=GATEWAY_BASE_URL,
            headers=AUTH_HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        ) as client:
            return client.post(
                "/v1/chat/completions",
                json={
                    "model": "chat",
                    "messages": [{"role": "user", "content": prompt}],
                },
            )

    with ThreadPoolExecutor(max_workers=count) as executor:
        return list(executor.map(call, range(count)))


def max_overlap(records: list[dict]) -> int:
    """기록의 시작 시각과 소요 시간만으로 동시에 처리 중이던 요청 수의 최댓값을 재구성한다."""
    boundaries: list[tuple[float, int]] = []
    for record in records:
        start = record["started_at"]
        end = start + record["duration_ms"] / 1000
        boundaries.append((start, 1))
        boundaries.append((end, -1))
    active = 0
    peak = 0
    for _, delta in sorted(boundaries):
        active += delta
        peak = max(peak, active)
    return peak


def phase_order_holds(record: dict) -> bool:
    return (
        0
        <= record["upstream_start_ms"]
        <= record["response_start_ms"]
        <= record["duration_ms"]
    )


def run_verification() -> None:
    start_fake()
    gateway = None
    client = None
    try:
        gateway = start_gateway()
        client = httpx.Client(
            base_url=GATEWAY_BASE_URL,
            headers=AUTH_HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        # 1. 동시 요청 — 요청별 기록과 겹침 재구성.
        responses = concurrent_chat_calls(CONCURRENT_REQUESTS)
        chat_records = wait_for_records(
            lambda record: record["path"] == "/v1/chat/completions",
            CONCURRENT_REQUESTS,
            "동시 chat 요청",
        )
        request_ids = {record["request_id"] for record in chat_records}
        check(
            "동시 요청 전건 성공·전건 기록",
            all(response.status_code == 200 for response in responses)
            and len(chat_records) == CONCURRENT_REQUESTS
            and len(request_ids) == CONCURRENT_REQUESTS,
            f"응답 {len(responses)}건, 기록 {len(chat_records)}건",
        )
        check(
            "요청별 단계 시간 순서(업스트림 시작 <= 응답 시작 <= 완료)",
            all(phase_order_holds(record) for record in chat_records),
            "upstream_start_ms/response_start_ms/duration_ms",
        )
        check(
            "기록만으로 동시 처리 구간 재구성",
            max_overlap(chat_records) >= 2,
            f"최대 동시 처리 {max_overlap(chat_records)}건",
        )
        check(
            "호출 주체·별칭·provider 기록",
            all(
                record["client"] == "stage7-verifier"
                and record["alias"] == "chat"
                and record["provider"] == "local"
                and record["completed"] is True
                for record in chat_records
            ),
            "client/alias/provider/completed",
        )

        # 2. x-request-id 헤더와 기록 대조.
        sampled = responses[0]
        header_id = sampled.headers.get("x-request-id", "")
        check(
            "x-request-id 헤더-기록 대조",
            bool(header_id) and header_id in request_ids,
            f"header={header_id[:8]}...",
        )

        # 3. 스트리밍 — 전송 종료 후 완결 기록.
        CONTROL["chat"] = "stream"
        streamed = client.post(
            "/v1/chat/completions",
            json={
                "model": "chat",
                "messages": [{"role": "user", "content": "x"}],
                "stream": True,
            },
        )
        [stream_record] = wait_for_records(
            lambda record: record["stream"] is True, 1, "스트리밍 chat 요청"
        )
        check(
            "스트리밍 요청 완결 기록",
            streamed.status_code == 200
            and stream_record["completed"] is True
            and stream_record["bytes_out"] > 0,
            f"HTTP {streamed.status_code}, bytes_out={stream_record['bytes_out']}",
        )

        # 4. 임베딩·Responses — 공개 별칭 기록.
        client.post("/v1/embeddings", json={"model": "embed", "input": "x"})
        CONTROL["chat"] = "ok"
        client.post(
            "/v1/responses",
            json={"model": "gpt-5.4-nano", "input": "이미지 특징 설명"},
        )
        [embed_record] = wait_for_records(
            lambda record: record["path"] == "/v1/embeddings", 1, "임베딩 요청"
        )
        [responses_record] = wait_for_records(
            lambda record: record["path"] == "/v1/responses", 1, "Responses 요청"
        )
        check(
            "임베딩·Responses 공개 별칭 기록",
            embed_record["alias"] == "embed"
            and responses_record["alias"] == "gpt-5.4-nano"
            and responses_record["provider"] == "local",
            f"embed={embed_record['alias']}, responses={responses_record['alias']}",
        )

        # 5. 인증 실패 — 신원 없는 기록.
        unauthorized = httpx.get(f"{GATEWAY_BASE_URL}/health", timeout=5)
        [unauthorized_record] = wait_for_records(
            lambda record: record["status"] == 401, 1, "인증 실패 요청"
        )
        check(
            "인증 실패 기록(신원 없음)",
            unauthorized.status_code == 401
            and unauthorized_record["client"] is None
            and unauthorized_record["key_id"] is None,
            f"HTTP {unauthorized.status_code}",
        )

        # 6. 폴백 — provider와 로컬 실패 사유.
        CONTROL["chat"] = "fail"
        fallback_response = client.post(
            "/v1/chat/completions",
            json={"model": "chat", "messages": [{"role": "user", "content": "x"}]},
        )
        [fallback_record] = wait_for_records(
            lambda record: record["local_failure_reason"] == "local_error_status",
            1,
            "폴백 요청",
        )
        check(
            "폴백 provider·실패 사유 기록",
            fallback_response.status_code == 200
            and fallback_record["provider"] == "openai",
            f"provider={fallback_record['provider']}, "
            f"reason={fallback_record['local_failure_reason']}",
        )

        # 7. 회로 열림 — 전환 이벤트와 로컬 생략 사유.
        for _ in range(FAILURE_THRESHOLD - 1):
            client.post(
                "/v1/chat/completions",
                json={"model": "chat", "messages": [{"role": "user", "content": "x"}]},
            )
        client.post(
            "/v1/chat/completions",
            json={"model": "chat", "messages": [{"role": "user", "content": "x"}]},
        )
        [skipped_record] = wait_for_records(
            lambda record: record["local_failure_reason"] == "circuit_open",
            1,
            "회로 열림 뒤 로컬 생략 요청",
        )
        circuit_events = [
            record for record in read_log_records() if record["event"] == "circuit"
        ]
        check(
            "회로 전환 이벤트·생략 사유 기록",
            any(
                event["alias"] == "chat" and event["state"] == "open"
                for event in circuit_events
            )
            and skipped_record["provider"] == "openai",
            f"circuit 이벤트 {len(circuit_events)}건, "
            f"reason={skipped_record['local_failure_reason']}",
        )

        # 8. 민감정보 미기록 — 프롬프트·키 원문이 로그에 없어야 한다.
        raw_log = REQUEST_LOG_PATH.read_text(encoding="utf-8")
        check(
            "프롬프트·API 키 미기록",
            PROMPT_MARKER not in raw_log and AUTH.api_key not in raw_log,
            f"로그 {len(raw_log.splitlines())}줄 검사",
        )
    finally:
        if client is not None:
            client.close()
        if gateway is not None:
            stop_gateway(gateway)
        AUTH.close()

    if failures:
        print(f"\n7단계 검증 실패 {len(failures)}건: {', '.join(failures)}")
        raise SystemExit(1)
    print("\n7단계 자동 검증 통과 - 요청별 관측 기록이 사후 진단 가능한 형태로 남는다.")


if __name__ == "__main__":
    run_verification()
