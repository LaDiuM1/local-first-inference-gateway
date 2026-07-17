"""요청 관측 계약 테스트 — 요청별 JSONL 기록, x-request-id, 회로 이벤트, 민감정보 제외."""

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

import anyio
import httpx
import pytest
import respx
from asgi_lifespan import LifespanManager
from pydantic import SecretStr
from starlette.requests import ClientDisconnect
from starlette.types import Message, Receive, Scope, Send

from gateway.config import settings
from gateway.main import create_app
from gateway.observability import ObservabilityMiddleware

pytestmark = pytest.mark.anyio

LOCAL_URL = f"{settings.ollama_base_url}/v1/chat/completions"
EMBED_URL = f"{settings.embedding_ollama_base_url}/api/embed"
OPENAI_URL = f"{settings.openai_base_url}/chat/completions"

PROMPT_MARKER = "관측-로그에-남으면-안-되는-프롬프트"
CHAT_REQUEST = {
    "model": "chat",
    "messages": [{"role": "user", "content": PROMPT_MARKER}],
}
CHAT_COMPLETION = {
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "응답"},
            "finish_reason": "stop",
        }
    ]
}
SSE_CHUNKS = [
    b'data: {"choices":[{"delta":{"content":"ob"}}]}\n\n',
    b"data: [DONE]\n\n",
]


class GatewayCredentials(Protocol):
    store_path: Path
    api_key: str

    @property
    def authorization_headers(self) -> dict[str, str]: ...


class _ByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], error: Exception | None = None) -> None:
        self._chunks = chunks
        self._error = error

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk
        if self._error is not None:
            raise self._error


@asynccontextmanager
async def running_client(
    credentials: GatewayCredentials,
) -> AsyncIterator[httpx.AsyncClient]:
    """요청을 보내고 lifespan을 닫아 로그 큐가 파일로 완전히 비워진 뒤에 읽게 한다."""
    app = create_app(api_key_store_path=credentials.store_path)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://gateway.test",
            headers=credentials.authorization_headers,
        ) as client:
            yield client


def read_log_lines() -> list[str]:
    path = settings.request_log_directory / "requests.jsonl"
    return path.read_text(encoding="utf-8").splitlines()


def read_records(event: str = "request") -> list[dict]:
    records = [json.loads(line) for line in read_log_lines()]
    return [record for record in records if record["event"] == event]


@respx.mock
async def test_chat_request_logs_phases_identity_and_request_id_header(
    gateway_credentials: GatewayCredentials,
) -> None:
    respx.post(LOCAL_URL).respond(200, json=CHAT_COMPLETION)

    async with running_client(gateway_credentials) as client:
        response = await client.post("/v1/chat/completions", json=CHAT_REQUEST)
    header_request_id = response.headers["x-request-id"]

    [record] = read_records()
    assert record["request_id"] == header_request_id
    assert record["client"] == "pytest-client"
    assert record["key_id"]
    assert record["method"] == "POST"
    assert record["path"] == "/v1/chat/completions"
    assert record["alias"] == "chat"
    assert record["stream"] is False
    assert record["status"] == 200
    assert record["provider"] == "local"
    assert record["local_failure_reason"] is None
    assert 0 <= record["upstream_start_ms"] <= record["response_start_ms"]
    assert record["response_start_ms"] <= record["duration_ms"]
    assert record["bytes_out"] > 0
    assert record["completed"] is True


@respx.mock
async def test_log_never_contains_prompt_or_api_key(
    gateway_credentials: GatewayCredentials,
) -> None:
    respx.post(LOCAL_URL).respond(200, json=CHAT_COMPLETION)

    async with running_client(gateway_credentials) as client:
        await client.post("/v1/chat/completions", json=CHAT_REQUEST)

    raw_log = "\n".join(read_log_lines())
    assert PROMPT_MARKER not in raw_log
    assert gateway_credentials.api_key not in raw_log
    assert CHAT_COMPLETION["choices"][0]["message"]["content"] not in raw_log


@respx.mock
async def test_fallback_request_logs_openai_provider_and_local_reason(
    gateway_credentials: GatewayCredentials, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "openai_api_key", SecretStr("sk-test-observe"))
    respx.post(LOCAL_URL).respond(503, json={"error": "down"})
    respx.post(OPENAI_URL).respond(200, json=CHAT_COMPLETION)

    async with running_client(gateway_credentials) as client:
        response = await client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert response.status_code == 200
    [record] = read_records()
    assert record["provider"] == "openai"
    assert record["local_failure_reason"] == "local_error_status"
    assert record["response_start_ms"] is not None


@respx.mock
async def test_circuit_transitions_are_logged_and_skip_reason_recorded(
    gateway_credentials: GatewayCredentials, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "openai_api_key", SecretStr("sk-test-observe"))
    threshold = settings.circuit_breaker_failure_threshold
    respx.post(LOCAL_URL).respond(503, json={"error": "down"})
    respx.post(OPENAI_URL).respond(200, json=CHAT_COMPLETION)

    async with running_client(gateway_credentials) as client:
        for _ in range(threshold + 1):
            await client.post("/v1/chat/completions", json=CHAT_REQUEST)

    circuit_events = read_records("circuit")
    assert {
        "alias": circuit_events[0]["alias"],
        "state": circuit_events[0]["state"],
    } == {"alias": "chat", "state": "open"}
    skip_records = [
        record
        for record in read_records()
        if record["local_failure_reason"] == "circuit_open"
    ]
    assert len(skip_records) == 1
    assert skip_records[0]["provider"] == "openai"


@respx.mock
async def test_streaming_request_logs_after_stream_end(
    gateway_credentials: GatewayCredentials,
) -> None:
    respx.post(LOCAL_URL).mock(
        return_value=httpx.Response(
            200,
            stream=_ByteStream(SSE_CHUNKS),
            headers={"content-type": "text/event-stream"},
        )
    )

    async with running_client(gateway_credentials) as client:
        response = await client.post(
            "/v1/chat/completions", json={**CHAT_REQUEST, "stream": True}
        )

    assert response.status_code == 200
    [record] = read_records()
    assert record["stream"] is True
    assert record["provider"] == "local"
    assert record["completed"] is True
    assert record["bytes_out"] == sum(len(chunk) for chunk in SSE_CHUNKS)
    assert record["response_start_ms"] <= record["duration_ms"]


@respx.mock
async def test_unhandled_exception_response_keeps_request_id_and_logs_status(
    gateway_credentials: GatewayCredentials,
) -> None:
    """처리되지 않은 예외도 관측을 거친 규격 500으로 나간다 — 장애 상황도 기록과 대조 가능해야 한다."""
    respx.post(LOCAL_URL).mock(side_effect=RuntimeError("boom"))

    app = create_app(api_key_store_path=gateway_credentials.store_path)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://gateway.test",
            headers=gateway_credentials.authorization_headers,
        ) as client:
            response = await client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert response.headers["cache-control"] == "no-store"
    [record] = read_records()
    assert record["request_id"] == response.headers["x-request-id"]
    assert record["status"] == 500
    assert record["completed"] is True


class _RecordingWriter:
    """파일 대신 메모리로 기록을 받는 writer — 미들웨어 집계를 단위로 검증한다."""

    def __init__(self) -> None:
        self.records: list[dict] = []

    def log_request(self, record: dict) -> None:
        self.records.append(record)


async def test_body_send_failure_is_not_counted_as_completed() -> None:
    """다운스트림 전송이 실패한 청크는 bytes_out에 더하지 않고 완결로도 남기지 않는다."""

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"chunk", "more_body": False})

    async def receive() -> Message:
        return {"type": "http.request"}

    async def failing_send(message: Message) -> None:
        if message["type"] == "http.response.body":
            raise RuntimeError("client disconnected")

    writer = _RecordingWriter()
    middleware = ObservabilityMiddleware(app, writer_provider=lambda: writer)
    scope: Scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions"}

    with pytest.raises(RuntimeError):
        await middleware(scope, receive, failing_send)

    [record] = writer.records
    assert record["status"] == 200
    assert record["bytes_out"] == 0
    assert record["completed"] is False


async def test_client_disconnect_during_body_read_logs_without_500() -> None:
    """본문 수신 중 클라이언트 연결 종료는 내부 오류가 아니다 — 500 합성 없이 미완결로 남긴다."""

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        raise ClientDisconnect()

    async def receive() -> Message:
        return {"type": "http.request"}

    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)

    writer = _RecordingWriter()
    middleware = ObservabilityMiddleware(app, writer_provider=lambda: writer)
    scope: Scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions"}

    await middleware(scope, receive, send)

    assert sent == []
    [record] = writer.records
    assert record["status"] is None
    assert record["completed"] is False


async def test_unauthenticated_request_is_logged_without_identity(
    gateway_credentials: GatewayCredentials,
) -> None:
    async with running_client(gateway_credentials) as client:
        response = await client.get(
            "/health", headers={"Authorization": "Bearer sk-oat-wrong"}
        )

    assert response.status_code == 401
    assert response.headers["x-request-id"]
    [record] = read_records()
    assert record["status"] == 401
    assert record["client"] is None
    assert record["alias"] is None
    assert record["provider"] is None


@respx.mock
async def test_embeddings_request_logs_public_alias(
    gateway_credentials: GatewayCredentials,
) -> None:
    respx.post(EMBED_URL).respond(
        200,
        json={
            "model": "snowflake-arctic-embed2",
            "embeddings": [[0.1] * 1024],
            "prompt_eval_count": 1,
        },
    )

    async with running_client(gateway_credentials) as client:
        await client.post("/v1/embeddings", json={"model": "embed", "input": "x"})

    [record] = read_records()
    assert record["alias"] == "embed"
    assert record["stream"] is False
    assert record["provider"] == "local"
    assert record["upstream_start_ms"] <= record["response_start_ms"]


@respx.mock
async def test_synthesized_embedding_502_logs_no_provider(
    gateway_credentials: GatewayCredentials,
) -> None:
    """합성 오류만 낸 요청에는 provider가 남지 않는다 — 무효 로컬 2xx 본문의 502가 그 경우다."""
    respx.post(EMBED_URL).respond(200, json={"unexpected": True})

    async with running_client(gateway_credentials) as client:
        response = await client.post(
            "/v1/embeddings", json={"model": "embed", "input": "x"}
        )

    assert response.status_code == 502
    [record] = read_records()
    assert record["provider"] is None
    assert record["local_failure_reason"] == "local_invalid_body"
    assert record["upstream_start_ms"] is not None
    assert record["response_start_ms"] is None


@respx.mock
async def test_confirmed_client_error_body_failure_logs_reason(
    gateway_credentials: GatewayCredentials,
) -> None:
    """폴백 비대상 4xx의 본문 읽기 실패도 실패 사유가 남는다 — 합성 오류의 원인을 재구성한다."""
    respx.post(LOCAL_URL).mock(
        return_value=httpx.Response(
            400,
            stream=_ByteStream([], error=httpx.ReadError("body read failed")),
            headers={"content-type": "application/json"},
        )
    )

    async with running_client(gateway_credentials) as client:
        response = await client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "upstream_response_unavailable"
    [record] = read_records()
    assert record["local_failure_reason"] == "local_body_unreadable"
    assert record["provider"] is None
    assert record["response_start_ms"] is None


@respx.mock
async def test_embedding_connection_failure_logs_local_reason(
    gateway_credentials: GatewayCredentials,
) -> None:
    """임베딩은 폴백하지 않지만 실패 사유는 chat과 같은 어휘로 남는다 — 재기동 구간 진단용이다."""
    respx.post(EMBED_URL).mock(side_effect=httpx.ConnectError("embedding down"))

    async with running_client(gateway_credentials) as client:
        response = await client.post(
            "/v1/embeddings", json={"model": "embed", "input": "x"}
        )

    assert response.status_code == 502
    [record] = read_records()
    assert record["local_failure_reason"] == "local_unreachable"
    assert record["provider"] is None
    assert record["response_start_ms"] is None


@respx.mock
async def test_forwarded_embedding_error_marks_response_start(
    gateway_credentials: GatewayCredentials,
) -> None:
    """전달 가능한 비-200 임베딩 응답은 로컬이 만든 응답이다 — provider와 응답 시작이 남는다."""
    respx.post(EMBED_URL).respond(404, json={"error": "model not found"})

    async with running_client(gateway_credentials) as client:
        response = await client.post(
            "/v1/embeddings", json={"model": "embed", "input": "x"}
        )

    assert response.status_code == 404
    [record] = read_records()
    assert record["provider"] == "local"
    assert record["local_failure_reason"] is None
    assert record["response_start_ms"] is not None


@respx.mock
async def test_forwarded_openai_error_on_stream_request_marks_response_start(
    gateway_credentials: GatewayCredentials, monkeypatch: pytest.MonkeyPatch
) -> None:
    """스트리밍 요청의 폴백이 비-2xx로 끝나 전달돼도 buffered 폴백과 같은 지점이 남는다."""
    monkeypatch.setattr(settings, "openai_api_key", SecretStr("sk-test-observe"))
    respx.post(LOCAL_URL).respond(503, json={"error": "down"})
    respx.post(OPENAI_URL).respond(
        429, json={"error": {"message": "rate limited", "type": "rate_limit_error"}}
    )

    async with running_client(gateway_credentials) as client:
        response = await client.post(
            "/v1/chat/completions", json={**CHAT_REQUEST, "stream": True}
        )

    assert response.status_code == 429
    [record] = read_records()
    assert record["provider"] == "openai"
    assert record["response_start_ms"] is not None


@respx.mock
async def test_responses_request_logs_public_alias_not_internal_vision(
    gateway_credentials: GatewayCredentials,
) -> None:
    respx.post(LOCAL_URL).respond(200, json=CHAT_COMPLETION)

    async with running_client(gateway_credentials) as client:
        await client.post(
            "/v1/responses",
            json={"model": "gpt-5.4-nano", "input": "이미지 특징을 설명해 줘."},
        )

    [record] = read_records()
    assert record["alias"] == "gpt-5.4-nano"
    assert record["provider"] == "local"


@respx.mock
async def test_concurrent_requests_record_overlapping_windows(
    gateway_credentials: GatewayCredentials,
) -> None:
    async def slow_local(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.15)
        return httpx.Response(200, json=CHAT_COMPLETION)

    respx.post(LOCAL_URL).mock(side_effect=slow_local)

    async with running_client(gateway_credentials) as client:

        async def call() -> None:
            response = await client.post("/v1/chat/completions", json=CHAT_REQUEST)
            assert response.status_code == 200

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(call)
            task_group.start_soon(call)

    records = read_records()
    assert len(records) == 2
    assert records[0]["request_id"] != records[1]["request_id"]
    intervals = [
        (record["started_at"], record["started_at"] + record["duration_ms"] / 1000)
        for record in records
    ]
    # 사후 진단 요건 — 기록만으로 두 요청이 같은 시간대에 처리 중이었음을 재구성할 수 있어야 한다.
    assert intervals[0][0] < intervals[1][1]
    assert intervals[1][0] < intervals[0][1]
