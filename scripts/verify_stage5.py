"""5단계 검증: 로컬 추론 장애 시 폴백과 회로 차단기, 임베딩 폴백 제외를 로컬 fake로 확인한다.

검증 기준(docs/ROADMAP.md 5단계):
- 로컬 추론이 중단된 상태에서 폴백 범위 내 요청이 정상 응답으로 반환되는지 확인
- 폴백 범위 밖의 요청이 잘못된 우회 없이 명시적 에러로 반환되는지 확인

확인 항목:
- chat·vision 폴백 — 로컬 장애 시 OpenAI(fake)로 우회하고 응답을 중계한다
- 로컬 성공 시 OpenAI 미호출, 200이지만 빈 본문인 경우도 폴백
- reasoning_effort 변환 — 로컬 none은 OpenAI에서 minimal, 명시 high는 그대로, 별칭 -> gpt-5-mini
- 회로 차단기 — 연속 실패로 open되면 로컬을 건너뛰고, open 시간 뒤 probe 성공으로 자동 복구한다
- 스트리밍 경계 — 첫 바이트 전 로컬 장애는 OpenAI 스트림으로 전환하고, 커밋 후 장애는 오류 종료한다
- embed 미폴백 — 임베딩 장애는 OpenAI를 호출하지 않고 OpenAI 규격 오류로 반환하며 별도 주소를 쓴다

실 OpenAI·실 Ollama를 호출하지 않는다. 게이트웨이가 바라보는 세 업스트림(로컬 chat·임베딩·OpenAI)을
모두 이 스크립트 안의 fake 서버로 돌려, 현재 노출된 키 없이 결정적으로 검증한다.
실행: uv run python scripts/verify_stage5.py
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

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

# 한국어 기본 콘솔(cp949)에서도 인코딩 불가 글리프로 죽지 않도록 출력 오류를 안전하게 대체한다.
# 라벨·메시지는 ASCII와 cp949로 표현 가능한 한글만 쓰지만, 안전장치로 대체 처리를 켜 둔다.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")


def free_port(host: str) -> int:
    # OS가 비어 있는 임시 포트를 고르게 해, 이미 떠 있는 게이트웨이 등과의 포트 충돌을 피한다.
    # 충돌 시 fake로 설정한 자식이 아니라 기존 프로세스를 검증 대상으로 오인해 실 업스트림을
    # 호출할 수 있으므로, 고정 포트 대신 매 실행마다 빈 포트를 잡는다.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        return probe.getsockname()[1]


GATEWAY_HOST = "127.0.0.1"
GATEWAY_PORT = free_port(GATEWAY_HOST)
GATEWAY_BASE_URL = f"http://{GATEWAY_HOST}:{GATEWAY_PORT}"
FAKE_HOST = "127.0.0.1"
FAKE_PORT = free_port(FAKE_HOST)
FAKE_BASE_URL = f"http://{FAKE_HOST}:{FAKE_PORT}"
FAKE_KEY = "sk-fake-verify-key-not-real"
FALLBACK_MODEL = "gpt-5-mini"
OPEN_SECONDS = 1.0
FAILURE_THRESHOLD = 3
STARTUP_TIMEOUT_SECONDS = 15
REQUEST_TIMEOUT_SECONDS = 30

# fake 업스트림 동작 제어와 관측 — 게이트웨이(자식 프로세스)는 HTTP로만 접근하고,
# 이 스크립트는 같은 프로세스의 전역을 직접 읽고 쓴다(요청 왕복이 happens-before를 만든다).
CONTROL = {"chat": "ok", "openai": "ok", "embed": "ok"}
STATS = {"local": 0, "openai": 0, "embed": 0, "last_openai_body": None}

failures: list[str] = []


def check(label: str, passed: bool, detail: str) -> None:
    mark = "FAIL"
    if passed:
        mark = "PASS"
    print(f"[{mark}] {label} - {detail}")
    if not passed:
        failures.append(label)


# --- fake 업스트림: 로컬 chat, OpenAI, 임베딩 세 역할을 경로로 구분한다 ---

fake = FastAPI()


def _chat_completion(marker: str) -> dict:
    return {
        "id": marker,
        "object": "chat.completion",
        "model": marker,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": f"{marker} answer"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


async def _sse(content: str) -> AsyncIterator[bytes]:
    yield f'data: {{"choices":[{{"delta":{{"content":"{content}"}}}}]}}\n\n'.encode()
    yield b"data: [DONE]\n\n"


async def _sse_midfail(content: str) -> AsyncIterator[bytes]:
    yield f'data: {{"choices":[{{"delta":{{"content":"{content}"}}}}]}}\n\n'.encode()
    # 첫 청크가 전달된 뒤에 끊어 provider 혼합 금지 경계를 만든다.
    await asyncio.sleep(0.2)
    raise RuntimeError("mid-stream drop")


@fake.post("/v1/chat/completions")
async def fake_local_chat(request: Request) -> object:
    STATS["local"] += 1
    await request.body()
    mode = CONTROL["chat"]
    if mode == "fail":
        return JSONResponse(status_code=503, content={"error": {"message": "down"}})
    if mode == "empty_ok_body":
        return Response(status_code=200, content=b"")
    if mode == "stream_ok":
        return StreamingResponse(_sse("loc"), media_type="text/event-stream")
    if mode == "stream_midfail":
        return StreamingResponse(
            _sse_midfail("loc-partial"), media_type="text/event-stream"
        )
    return JSONResponse(_chat_completion("local"))


@fake.post("/openai-v1/chat/completions")
async def fake_openai_chat(request: Request) -> object:
    STATS["openai"] += 1
    STATS["last_openai_body"] = json.loads(await request.body())
    if CONTROL["openai"] == "stream_ok":
        return StreamingResponse(_sse("open"), media_type="text/event-stream")
    return JSONResponse(_chat_completion("openai"))


@fake.post("/api/embed")
async def fake_embed(request: Request) -> object:
    STATS["embed"] += 1
    await request.body()
    mode = CONTROL["embed"]
    if mode == "fail":
        return JSONResponse(status_code=503, content={"error": "embed down"})
    if mode == "bad_body":
        # 200이지만 입력 1개에 벡터 0개 — 유효한 임베딩 응답이 아니다.
        return JSONResponse({"model": "snowflake-arctic-embed2", "embeddings": []})
    return JSONResponse(
        {
            "model": "snowflake-arctic-embed2",
            "embeddings": [[0.1, 0.2, 0.3]],
            "prompt_eval_count": 3,
        }
    )


def start_fake_server() -> uvicorn.Server:
    # 스트리밍 중간 장애 시나리오가 의도적으로 예외를 던지므로 fake 로그는 critical만 남긴다.
    config = uvicorn.Config(fake, host=FAKE_HOST, port=FAKE_PORT, log_level="critical")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if server.started:
            return server
        time.sleep(0.05)
    raise RuntimeError("fake 업스트림이 기동하지 않는다")


def start_gateway() -> subprocess.Popen:
    # 게이트웨이의 세 업스트림을 모두 fake로 돌리고, 폴백 키는 fake 값으로 덮어 실 OpenAI 호출을 막는다.
    env = {
        **os.environ,
        "GATEWAY_OLLAMA_BASE_URL": FAKE_BASE_URL,
        "GATEWAY_EMBEDDING_OLLAMA_BASE_URL": FAKE_BASE_URL,
        "GATEWAY_OPENAI_BASE_URL": f"{FAKE_BASE_URL}/openai-v1",
        "GATEWAY_CIRCUIT_BREAKER_FAILURE_THRESHOLD": str(FAILURE_THRESHOLD),
        "GATEWAY_CIRCUIT_BREAKER_OPEN_SECONDS": str(OPEN_SECONDS),
        "OPENAI_API_KEY": FAKE_KEY,
    }
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "gateway.main:app",
            "--host",
            GATEWAY_HOST,
            "--port",
            str(GATEWAY_PORT),
            "--log-level",
            # 커밋 후 장애 시 ASGI 예외는 의도한 오류 종료이므로 traceback 대신 검증 결과로 판정한다.
            "critical",
        ],
        env=env,
    )
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        # 자식이 조기 종료했으면(예: 포트 바인딩 실패) 다른 프로세스를 검증 대상으로 오인하지 않도록
        # /health를 확인하기 전에 자식 생존부터 확인한다.
        if process.poll() is not None:
            raise RuntimeError(
                f"게이트웨이 프로세스가 기동 중 종료했다 (exit {process.returncode})"
            )
        try:
            health = httpx.get(f"{GATEWAY_BASE_URL}/health")
            if health.json() == {"status": "ok"}:
                return process
        except httpx.TransportError:
            time.sleep(0.3)
    process.terminate()
    raise RuntimeError("게이트웨이가 기동하지 않는다")


def chat_request(client: httpx.Client, **extra: object) -> httpx.Response:
    body = {"model": "chat", "messages": [{"role": "user", "content": "x"}], **extra}
    return client.post(f"{GATEWAY_BASE_URL}/v1/chat/completions", json=body)


def reset_chat_breaker(client: httpx.Client) -> None:
    # 로컬 성공 한 번으로 chat 회로를 closed·연속 실패 0으로 되돌려, 시나리오 간 상태 누수를 막는다.
    CONTROL["chat"] = "ok"
    chat_request(client)


def read_stream(client: httpx.Client, **extra: object) -> str:
    body = {"model": "chat", "messages": [{"role": "user", "content": "x"}], **extra}
    received = b""
    with client.stream(
        "POST", f"{GATEWAY_BASE_URL}/v1/chat/completions", json=body
    ) as response:
        for chunk in response.iter_raw():
            received += chunk
    return received.decode("utf-8", "replace")


def read_aborted_stream(
    client: httpx.Client, **extra: object
) -> tuple[str, str | None]:
    body = {"model": "chat", "messages": [{"role": "user", "content": "x"}], **extra}
    received = bytearray()
    error_name = None
    try:
        with client.stream(
            "POST", f"{GATEWAY_BASE_URL}/v1/chat/completions", json=body
        ) as response:
            for chunk in response.iter_raw():
                received.extend(chunk)
    except httpx.TransportError as error:
        error_name = type(error).__name__
    return received.decode("utf-8", "replace"), error_name


fake_server = start_fake_server()
gateway = start_gateway()
try:
    client = httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS)

    # 1. 로컬 성공 — OpenAI를 호출하지 않는다.
    reset_chat_breaker(client)
    openai_before = STATS["openai"]
    CONTROL["chat"] = "ok"
    response = chat_request(client)
    check(
        "로컬 성공 시 OpenAI 미호출",
        response.status_code == 200
        and response.json()["id"] == "local"
        and STATS["openai"] == openai_before,
        f"id={response.json().get('id')}, openai 호출 {STATS['openai'] - openai_before}건",
    )

    # 2. chat 폴백 — 로컬 장애 시 OpenAI로 우회한다.
    reset_chat_breaker(client)
    CONTROL["chat"] = "fail"
    openai_before = STATS["openai"]
    response = chat_request(client)
    body = response.json()
    check(
        "chat 폴백",
        response.status_code == 200
        and body["id"] == "openai"
        and STATS["openai"] == openai_before + 1,
        f"id={body.get('id')}, 폴백 모델={STATS['last_openai_body']['model']}",
    )
    check(
        "폴백 모델 치환",
        STATS["last_openai_body"]["model"] == FALLBACK_MODEL,
        f"model={STATS['last_openai_body']['model']}",
    )

    # 3. vision 폴백 — 별칭도 gpt-5-mini로 우회한다.
    CONTROL["chat"] = "fail"
    response = client.post(
        f"{GATEWAY_BASE_URL}/v1/chat/completions",
        json={"model": "vision", "messages": [{"role": "user", "content": "x"}]},
    )
    check(
        "vision 폴백",
        response.status_code == 200
        and response.json()["id"] == "openai"
        and STATS["last_openai_body"]["model"] == FALLBACK_MODEL,
        f"id={response.json().get('id')}, model={STATS['last_openai_body']['model']}",
    )

    # 3b. 로컬이 200이지만 빈 본문을 주면 유효한 추론 응답이 아니므로 폴백한다.
    reset_chat_breaker(client)
    CONTROL["chat"] = "empty_ok_body"
    openai_before = STATS["openai"]
    response = chat_request(client)
    check(
        "빈 200 본문도 폴백",
        response.status_code == 200
        and response.json()["id"] == "openai"
        and STATS["openai"] == openai_before + 1,
        f"id={response.json().get('id')}, openai 호출 {STATS['openai'] - openai_before}건",
    )

    # 4. reasoning_effort 변환 — 로컬 none -> OpenAI minimal, 명시 high는 그대로.
    reset_chat_breaker(client)
    CONTROL["chat"] = "fail"
    chat_request(client)
    check(
        "none -> minimal 변환",
        STATS["last_openai_body"]["reasoning_effort"] == "minimal",
        f"reasoning_effort={STATS['last_openai_body']['reasoning_effort']}",
    )
    reset_chat_breaker(client)
    CONTROL["chat"] = "fail"
    chat_request(client, reasoning_effort="high")
    check(
        "명시 high 전달",
        STATS["last_openai_body"]["reasoning_effort"] == "high",
        f"reasoning_effort={STATS['last_openai_body']['reasoning_effort']}",
    )

    # 5. 스트리밍 경계 — 첫 바이트 전 로컬 장애는 OpenAI 스트림으로 전환한다.
    reset_chat_breaker(client)
    CONTROL["chat"] = "fail"
    CONTROL["openai"] = "stream_ok"
    received = read_stream(client, stream=True)
    check(
        "스트리밍 시작 전 OpenAI 전환",
        "open" in received and "loc" not in received,
        f"수신 본문에 open 포함={('open' in received)}, loc 포함={('loc' in received)}",
    )

    # 6. 스트리밍 중간 장애 — 첫 로컬 바이트 뒤 장애는 provider를 섞지 않는다.
    reset_chat_breaker(client)
    CONTROL["chat"] = "stream_midfail"
    CONTROL["openai"] = "stream_ok"
    openai_before = STATS["openai"]
    received, stream_error = read_aborted_stream(client, stream=True)
    check(
        "스트리밍 중간 장애 시 오류 종료·provider 미혼합",
        "loc-partial" in received
        and "open" not in received
        and stream_error == "RemoteProtocolError"
        and STATS["openai"] == openai_before,
        f"loc-partial 포함={('loc-partial' in received)}, 종료={stream_error}, "
        f"openai 호출 {STATS['openai'] - openai_before}건",
    )
    CONTROL["openai"] = "ok"

    # 7. 회로 차단기 — 연속 실패로 open되면 로컬을 건너뛴다.
    reset_chat_breaker(client)
    CONTROL["chat"] = "fail"
    for _ in range(FAILURE_THRESHOLD):
        chat_request(client)
    local_after_open = STATS["local"]
    openai_before = STATS["openai"]
    chat_request(client)
    check(
        "open 상태 로컬 생략",
        STATS["local"] == local_after_open and STATS["openai"] == openai_before + 1,
        f"open 이후 로컬 추가 호출 {STATS['local'] - local_after_open}건",
    )

    # 8. 자동 복구 — open 시간이 지나면 probe가 로컬을 다시 시도하고 성공 시 복구한다.
    time.sleep(OPEN_SECONDS + 0.3)
    CONTROL["chat"] = "ok"
    local_before = STATS["local"]
    response = chat_request(client)
    check(
        "open 시간 뒤 probe 자동 복구",
        response.status_code == 200
        and response.json()["id"] == "local"
        and STATS["local"] == local_before + 1,
        f"복구 응답 id={response.json().get('id')}",
    )

    # 9. embed 미폴백 — 임베딩 장애는 OpenAI를 호출하지 않고 오류로 반환한다.
    CONTROL["embed"] = "fail"
    openai_before = STATS["openai"]
    embed_before = STATS["embed"]
    response = client.post(
        f"{GATEWAY_BASE_URL}/v1/embeddings", json={"model": "embed", "input": "x"}
    )
    check(
        "embed 장애 미폴백",
        response.status_code >= 400
        and STATS["openai"] == openai_before
        and STATS["embed"] == embed_before + 1,
        f"status={response.status_code}, openai 호출 {STATS['openai'] - openai_before}건",
    )

    # 10. embed 정상 — 별도 임베딩 주소로 응답한다.
    CONTROL["embed"] = "ok"
    embed_before = STATS["embed"]
    response = client.post(
        f"{GATEWAY_BASE_URL}/v1/embeddings", json={"model": "embed", "input": "x"}
    )
    check(
        "embed 정상 응답(별도 주소)",
        response.status_code == 200
        and len(response.json()["data"][0]["embedding"]) == 3
        and STATS["embed"] == embed_before + 1,
        f"status={response.status_code}, 임베딩 호출 {STATS['embed'] - embed_before}건",
    )

    # 11. embed 무결성 — 200이지만 벡터 개수가 맞지 않으면 우회 없이 OpenAI 규격 502로 합성한다.
    CONTROL["embed"] = "bad_body"
    openai_before = STATS["openai"]
    embed_before = STATS["embed"]
    response = client.post(
        f"{GATEWAY_BASE_URL}/v1/embeddings", json={"model": "embed", "input": "x"}
    )
    check(
        "embed 무효 200 본문 502 합성(미폴백)",
        response.status_code == 502
        and response.json()["error"]["code"] == "upstream_invalid_response"
        and STATS["openai"] == openai_before
        and STATS["embed"] == embed_before + 1,
        f"status={response.status_code}, openai 호출 {STATS['openai'] - openai_before}건",
    )
finally:
    client.close()
    gateway.terminate()
    gateway.wait(timeout=10)
    fake_server.should_exit = True

if failures:
    print(f"\n5단계 검증 실패: {', '.join(failures)}")
    sys.exit(1)
print("\n5단계 검증 통과 - 폴백·회로 차단기·스트리밍 경계·임베딩 폴백 제외가 성립한다.")
