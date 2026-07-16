"""6단계 결정적 검증: 인증·본문 제한·캐시·기한·스트리밍·임베딩 미폴백·검색 호환 별칭을 fake로 확인한다.

기본 모드는 게이트웨이의 세 업스트림을 모두 로컬 fake 서버로 돌리며 실제 OpenAI, Ollama,
Cloudflare를 호출하지 않는다. `--public-host`를 명시하면 로컬 검증 뒤 공개 `/docs`와 인증된
`/health`만 추가 확인한다. 공개 키는 명령 인자가 아니라 `OPENAT_STAGE6_SMOKE_API_KEY` 환경변수에서
읽고 어떤 출력에도 포함하지 않는다.

실행: uv run python scripts/verify_stage6.py
선택: OPENAT_STAGE6_SMOKE_API_KEY=... uv run python scripts/verify_stage6.py --public-host api.example.com
"""

import argparse
import asyncio
import base64
import http.client
import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from collections.abc import AsyncIterator, Iterator

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from verification_auth import UVICORN_APPLICATION_ARGUMENTS, VerificationAuth

from gateway.api_keys import ApiKeyStore

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")

HOST = "127.0.0.1"
MAX_BODY_BYTES = 32 * 1024 * 1024
LOCAL_DEADLINE_SECONDS = 0.15
TOTAL_DEADLINE_SECONDS = 0.4
STARTUP_TIMEOUT_SECONDS = 15
REQUEST_TIMEOUT_SECONDS = 5
# embed 별칭은 native 1024차원 그대로, 검색 호환 별칭은 zero-padding으로 1536차원을 공개한다.
EMBED_DIMENSIONS = 1024
OPENAI_COMPAT_EMBED_DIMENSIONS = 1536


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((HOST, 0))
        return probe.getsockname()[1]


GATEWAY_PORT = free_port()
FAKE_PORT = free_port()
GATEWAY_BASE_URL = f"http://{HOST}:{GATEWAY_PORT}"
FAKE_BASE_URL = f"http://{HOST}:{FAKE_PORT}"
AUTH = VerificationAuth.create("stage6-verifier")
REVOKED_KEY = ApiKeyStore(AUTH.store_path).issue("revoked-verifier")
ApiKeyStore(AUTH.store_path).revoke(REVOKED_KEY.key_id)
AUTH_HEADERS = AUTH.headers

CONTROL = {"chat": "ok", "openai": "ok", "embed": "ok"}
STATS = {"local": 0, "openai": 0, "embed": 0}
failures: list[str] = []


def check(label: str, passed: bool, detail: str) -> None:
    mark = "PASS"
    if not passed:
        mark = "FAIL"
        failures.append(label)
    print(f"[{mark}] {label} - {detail}")


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


async def sse(marker: str, delay: float = 0.0) -> AsyncIterator[bytes]:
    if delay:
        await asyncio.sleep(delay)
    yield f'data: {{"choices":[{{"delta":{{"content":"{marker}"}}}}]}}\n\n'.encode()
    yield b"data: [DONE]\n\n"


async def sse_midfail() -> AsyncIterator[bytes]:
    yield b'data: {"choices":[{"delta":{"content":"local-partial"}}]}\n\n'
    await asyncio.sleep(0.05)
    raise RuntimeError("intentional stream failure")


fake = FastAPI()


async def slow_body(payload: bytes, delay: float) -> AsyncIterator[bytes]:
    # 상태 코드는 즉시, 본문은 로컬 기한을 넘겨 도착한다.
    await asyncio.sleep(delay)
    yield payload


@fake.post("/v1/chat/completions")
async def fake_local_chat(request: Request) -> object:
    STATS["local"] += 1
    await request.body()
    mode = CONTROL["chat"]
    if mode == "fail":
        return JSONResponse(status_code=503, content={"error": "local down"})
    if mode == "delayed_failure":
        await asyncio.sleep(0.1)
        return JSONResponse(status_code=503, content={"error": "local slow down"})
    if mode == "slow_stream":
        return StreamingResponse(sse("local", 0.25), media_type="text/event-stream")
    if mode == "midfail":
        return StreamingResponse(sse_midfail(), media_type="text/event-stream")
    if mode == "slow_4xx_body":
        return StreamingResponse(
            slow_body(b'{"error":{"message":"client mistake"}}', 0.3),
            status_code=400,
            media_type="application/json",
        )
    return JSONResponse(chat_completion("local"))


@fake.post("/openai-v1/chat/completions")
async def fake_openai_chat(request: Request) -> object:
    STATS["openai"] += 1
    await request.body()
    if CONTROL["openai"] == "slow":
        await asyncio.sleep(0.5)
    if CONTROL["openai"] == "stream":
        return StreamingResponse(sse("openai"), media_type="text/event-stream")
    return JSONResponse(chat_completion("openai"))


@fake.post("/api/embed")
async def fake_embed(request: Request) -> object:
    STATS["embed"] += 1
    payload = await request.json()
    if CONTROL["embed"] == "fail":
        return JSONResponse(status_code=503, content={"error": "embed down"})
    embedding_input = payload["input"]
    count = 1
    if isinstance(embedding_input, list):
        count = len(embedding_input)
    dimensions = EMBED_DIMENSIONS
    if CONTROL["embed"] == "wrong_dimensions":
        dimensions = EMBED_DIMENSIONS // 2
    vectors = [[float(index) / dimensions for index in range(dimensions)]]
    return JSONResponse(
        {
            "model": "snowflake-arctic-embed2",
            "embeddings": vectors * count,
            "prompt_eval_count": count,
        }
    )


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
        "GATEWAY_LOCAL_RESPONSE_START_TIMEOUT_SECONDS": str(LOCAL_DEADLINE_SECONDS),
        "GATEWAY_TOTAL_RESPONSE_START_TIMEOUT_SECONDS": str(TOTAL_DEADLINE_SECONDS),
        "GATEWAY_CIRCUIT_BREAKER_FAILURE_THRESHOLD": "100",
        "OPENAI_API_KEY": "stage6-fake-openai-key",
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
                f"{GATEWAY_BASE_URL}/health",
                headers=AUTH_HEADERS,
                timeout=1,
            )
            if health.status_code == 200:
                return process
        except httpx.TransportError:
            time.sleep(0.1)
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)
    raise RuntimeError("gateway did not start")


def chat(client: httpx.Client, **extra: object) -> httpx.Response:
    body = {"model": "chat", "messages": [{"role": "user", "content": "x"}], **extra}
    return client.post("/v1/chat/completions", json=body)


def oversized_content_length() -> tuple[int, dict]:
    connection = http.client.HTTPConnection(HOST, GATEWAY_PORT, timeout=5)
    connection.putrequest("POST", "/v1/chat/completions")
    connection.putheader("Authorization", AUTH_HEADERS["Authorization"])
    connection.putheader("Content-Type", "application/json")
    connection.putheader("Content-Length", str(MAX_BODY_BYTES + 1))
    connection.endheaders()
    response = connection.getresponse()
    body = json.loads(response.read())
    status = response.status
    connection.close()
    return status, body


def oversized_chunks() -> Iterator[bytes]:
    chunk_size = 4 * 1024 * 1024
    for _ in range(8):
        yield b" " * chunk_size
    yield b"x"


def decode_base64_dimensions(value: str) -> int:
    return len(base64.b64decode(value)) // struct.calcsize("f")


def run_local_verification() -> None:
    fake_server = start_fake()
    gateway = None
    client = None
    try:
        gateway = start_gateway()
        client = httpx.Client(
            base_url=GATEWAY_BASE_URL,
            headers=AUTH_HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        upstream_before = STATS.copy()
        for label, headers in [
            ("인증 없음", {}),
            ("잘못된 인증", {"Authorization": "Bearer invalid"}),
            (
                "폐기 키",
                {"Authorization": f"Bearer {REVOKED_KEY.api_key}"},
            ),
        ]:
            response = httpx.post(
                f"{GATEWAY_BASE_URL}/v1/chat/completions",
                json={"model": "chat", "messages": []},
                headers=headers,
            )
            check(
                label,
                response.status_code == 401
                and response.json()["error"]["code"] == "invalid_api_key",
                f"HTTP {response.status_code}",
            )
        check(
            "인증 실패 업스트림 미호출",
            upstream_before == STATS,
            f"호출 변화 local={STATS['local'] - upstream_before['local']}",
        )

        health = client.get("/health")
        docs = httpx.get(f"{GATEWAY_BASE_URL}/docs")
        api = chat(client)
        openapi = httpx.get(f"{GATEWAY_BASE_URL}/openapi.json")
        redoc = httpx.get(f"{GATEWAY_BASE_URL}/redoc")
        check(
            "정상 키 health",
            health.status_code == 200 and health.json() == {"status": "ok"},
            f"HTTP {health.status_code}",
        )
        check(
            "API와 health no-store",
            api.headers.get("cache-control") == "no-store"
            and health.headers.get("cache-control") == "no-store",
            f"api={api.headers.get('cache-control')}, health={health.headers.get('cache-control')}",
        )
        check(
            "공개 docs 5분 캐시",
            docs.status_code == 200
            and docs.headers.get("cache-control") == "public, max-age=300"
            and "1024" in docs.text
            and "1536" in docs.text,
            f"HTTP {docs.status_code}, cache={docs.headers.get('cache-control')}",
        )
        check(
            "자동 OpenAPI와 Redoc 비활성화",
            openapi.status_code == 404 and redoc.status_code == 404,
            f"openapi={openapi.status_code}, redoc={redoc.status_code}",
        )

        local_before = STATS["local"]
        length_status, length_body = oversized_content_length()
        check(
            "초과 Content-Length 선차단",
            length_status == 413
            and length_body["error"]["code"] == "request_too_large"
            and STATS["local"] == local_before,
            f"HTTP {length_status}",
        )
        chunked = client.post(
            "/v1/chat/completions",
            content=oversized_chunks(),
            headers={"Content-Type": "application/json"},
        )
        check(
            "Content-Length 없는 실제 바이트 초과 차단",
            chunked.status_code == 413 and STATS["local"] == local_before,
            f"HTTP {chunked.status_code}",
        )

        CONTROL["chat"] = "fail"
        fallback = chat(client)
        check(
            "로컬 장애 OpenAI fake 폴백",
            fallback.status_code == 200 and fallback.json()["id"] == "openai",
            f"HTTP {fallback.status_code}",
        )

        CONTROL["chat"] = "delayed_failure"
        CONTROL["openai"] = "slow"
        started = time.monotonic()
        timed_out = chat(client)
        elapsed = time.monotonic() - started
        check(
            "fallback 전체 기한 잔여 시간만 사용",
            timed_out.status_code == 504
            and timed_out.json()["error"]["code"] == "response_start_timeout"
            and elapsed < 0.6,
            f"HTTP {timed_out.status_code}, {elapsed:.2f}s",
        )

        CONTROL["chat"] = "slow_stream"
        CONTROL["openai"] = "stream"
        streamed = chat(client, stream=True)
        check(
            "첫 SSE 전 로컬 timeout 폴백",
            streamed.status_code == 200
            and "openai" in streamed.text
            and "local" not in streamed.text,
            f"HTTP {streamed.status_code}",
        )

        CONTROL["chat"] = "midfail"
        openai_before = STATS["openai"]
        received = ""
        stream_error = None
        try:
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={"model": "chat", "messages": [], "stream": True},
            ) as response:
                for chunk in response.iter_text():
                    received += chunk
        except httpx.TransportError as error:
            stream_error = type(error).__name__
        check(
            "첫 SSE 이후 provider 미혼합",
            "local-partial" in received
            and "openai" not in received
            and STATS["openai"] == openai_before
            and stream_error is not None,
            f"종료={stream_error}, OpenAI 추가 호출={STATS['openai'] - openai_before}",
        )

        CONTROL["chat"] = "ok"
        responses_success = client.post(
            "/v1/responses",
            json={
                "model": "gpt-5.4-nano",
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": "특징 설명"}],
                    }
                ],
            },
        )
        responses_body = responses_success.json()
        check(
            "Responses 호환 별칭 변환",
            responses_success.status_code == 200
            and responses_body.get("status") == "completed"
            and responses_body.get("model") == "gpt-5.4-nano"
            and responses_body["output"][0]["content"][0]["text"] == "local",
            f"HTTP {responses_success.status_code}",
        )
        wrong_endpoint_responses = client.post(
            "/v1/responses", json={"model": "chat", "input": "x"}
        )
        wrong_endpoint_chat = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-5.4-nano",
                "messages": [{"role": "user", "content": "x"}],
            },
        )
        check(
            "별칭-엔드포인트 경계 400",
            wrong_endpoint_responses.status_code == 400
            and wrong_endpoint_chat.status_code == 400,
            f"responses={wrong_endpoint_responses.status_code}, "
            f"chat={wrong_endpoint_chat.status_code}",
        )

        CONTROL["embed"] = "ok"
        single = client.post(
            "/v1/embeddings",
            json={"model": "embed", "input": "x", "encoding_format": "float"},
        )
        encoded = client.post(
            "/v1/embeddings",
            json={
                "model": "embed",
                "input": ["x", "y"],
                "encoding_format": "base64",
            },
        )
        encoded_body = encoded.json()
        check(
            "임베딩 1024차원 float",
            single.status_code == 200
            and len(single.json()["data"][0]["embedding"]) == EMBED_DIMENSIONS,
            f"HTTP {single.status_code}",
        )
        check(
            "임베딩 배열 base64 순서",
            encoded.status_code == 200
            and [item["index"] for item in encoded_body["data"]] == [0, 1]
            and all(
                decode_base64_dimensions(item["embedding"]) == EMBED_DIMENSIONS
                for item in encoded_body["data"]
            ),
            f"HTTP {encoded.status_code}, {len(encoded_body.get('data', []))}건",
        )
        compat = client.post(
            "/v1/embeddings",
            json={"model": "text-embedding-3-small", "input": "x"},
        )
        compat_vector = compat.json()["data"][0]["embedding"]
        check(
            "호환 별칭 1536차원 zero-padding",
            compat.status_code == 200
            and compat.json()["model"] == "text-embedding-3-small"
            and len(compat_vector) == OPENAI_COMPAT_EMBED_DIMENSIONS
            and compat_vector[EMBED_DIMENSIONS:]
            == [0.0] * (OPENAI_COMPAT_EMBED_DIMENSIONS - EMBED_DIMENSIONS),
            f"HTTP {compat.status_code}, {len(compat_vector)}차원",
        )

        CONTROL["embed"] = "wrong_dimensions"
        openai_before = STATS["openai"]
        wrong_dimensions = client.post(
            "/v1/embeddings", json={"model": "embed", "input": "x"}
        )
        check(
            "임베딩 계약 외 차원 502",
            wrong_dimensions.status_code == 502
            and wrong_dimensions.json()["error"]["code"] == "upstream_invalid_response"
            and STATS["openai"] == openai_before,
            f"HTTP {wrong_dimensions.status_code}, OpenAI 추가 호출={STATS['openai'] - openai_before}",
        )

        CONTROL["embed"] = "fail"
        CONTROL["openai"] = "ok"
        openai_before = STATS["openai"]
        failed_embed = client.post(
            "/v1/embeddings", json={"model": "embed", "input": "x"}
        )
        check(
            "임베딩 장애 미폴백",
            failed_embed.status_code == 503 and STATS["openai"] == openai_before,
            f"HTTP {failed_embed.status_code}, OpenAI 추가 호출={STATS['openai'] - openai_before}",
        )

        # 로컬이 확정한 폴백 대상 아닌 4xx는 본문이 로컬 기한을 넘겨 도착해도 외부로 나가지 않는다.
        CONTROL["chat"] = "slow_4xx_body"
        openai_before = STATS["openai"]
        for label, streaming in [("버퍼", False), ("스트리밍", True)]:
            slow_client_error = chat(client, stream=streaming)
            check(
                f"느린 4xx 본문 미폴백 - {label}",
                slow_client_error.status_code == 400
                and STATS["openai"] == openai_before,
                f"HTTP {slow_client_error.status_code}, "
                f"OpenAI 추가 호출={STATS['openai'] - openai_before}",
            )

        nested = b'{"model":"chat","messages":' + b"[" * 200_000 + b"]" * 200_000 + b"}"
        local_before = STATS["local"]
        too_deep = client.post(
            "/v1/chat/completions",
            content=nested,
            headers={"Content-Type": "application/json"},
        )
        check(
            "과도한 JSON 중첩 400",
            too_deep.status_code == 400
            and too_deep.json()["error"]["type"] == "invalid_request_error"
            and too_deep.headers.get("cache-control") == "no-store"
            and STATS["local"] == local_before,
            f"HTTP {too_deep.status_code}, cache={too_deep.headers.get('cache-control')}",
        )
    finally:
        if client is not None:
            client.close()
        if gateway is not None and gateway.poll() is None:
            gateway.terminate()
            try:
                gateway.wait(timeout=10)
            except subprocess.TimeoutExpired:
                gateway.kill()
                gateway.wait(timeout=10)
        fake_server.should_exit = True
        AUTH.close()


def run_public_smoke(public_host: str) -> None:
    api_key = os.getenv("OPENAT_STAGE6_SMOKE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAT_STAGE6_SMOKE_API_KEY is required when --public-host is used"
        )
    host = public_host.removeprefix("https://").rstrip("/")
    base_url = f"https://{host}"
    docs = httpx.get(f"{base_url}/docs", timeout=10)
    health = httpx.get(
        f"{base_url}/health",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
    )
    invalid = httpx.get(
        f"{base_url}/health",
        headers={"Authorization": "Bearer invalid"},
        timeout=10,
    )
    check(
        "공개 docs smoke",
        docs.status_code == 200,
        f"HTTP {docs.status_code}",
    )
    check(
        "공개 health 인증 smoke",
        health.status_code == 200 and invalid.status_code == 401,
        f"정상={health.status_code}, 잘못된 키={invalid.status_code}",
    )


parser = argparse.ArgumentParser()
parser.add_argument(
    "--public-host", help="optional public hostname for safe docs/health smoke"
)
arguments = parser.parse_args()

run_local_verification()
if arguments.public_host:
    run_public_smoke(arguments.public_host)

if failures:
    print(f"\n6단계 검증 실패: {', '.join(failures)}")
    sys.exit(1)
print(
    "\n6단계 자동 검증 통과 - 실제 외부망과 Windows 재부팅 검증은 별도 수동 확인이 필요하다."
)
