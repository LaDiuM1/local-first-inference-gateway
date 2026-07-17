"""로컬 장애 시 OpenAI 폴백 계약 테스트 — 로컬·OpenAI·임베딩 업스트림을 respx로 목킹한다.

회로 차단기 상태 전이 자체는 test_circuit_breaker.py가 가짜 clock으로 검증하고, 여기서는 게이트웨이가
폴백 대상 장애를 실제로 우회하는지, 클라이언트 오류와 임베딩은 우회하지 않는지, 스트리밍 전환 경계와
비밀 비노출을 검증한다. lifespan이 실제 키에 의존하지 않도록 폴백 키는 테스트 값으로 주입한다.
"""

import gzip
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol

import anyio
import httpx
import pytest
import respx
from asgi_lifespan import LifespanManager
from pydantic import SecretStr

from gateway.config import settings
from gateway.embeddings import NATIVE_EMBEDDING_VECTOR_DIMENSIONS
from gateway.main import create_app

pytestmark = pytest.mark.anyio

LOCAL_URL = f"{settings.ollama_base_url}/v1/chat/completions"
OPENAI_URL = f"{settings.openai_base_url}/chat/completions"
EMBED_URL = f"{settings.embedding_ollama_base_url}/api/embed"

CHAT_ALIAS = "chat"
VISION_ALIAS = "vision"
EMBED_ALIAS = "embed"
CHAT_MODEL = "gemma4:12b-it-qat"
FALLBACK_MODEL = "gpt-5-mini"
TEST_KEY = "test-openai-key-not-real"

CHAT_REQUEST = {"model": CHAT_ALIAS, "messages": [{"role": "user", "content": "안녕"}]}


class GatewayCredentials(Protocol):
    store_path: Path
    api_key: str


class TrackingStream(httpx.AsyncByteStream):
    """청크를 순서대로 흘리고, 지정 시 마지막에 오류를 낸다. 닫힘 여부로 자원 정리를 확인한다."""

    def __init__(self, chunks: list[bytes], error: Exception | None = None) -> None:
        self._chunks = chunks
        self._error = error
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk
        if self._error is not None:
            raise self._error

    async def aclose(self) -> None:
        self.closed = True


class SlowStream(httpx.AsyncByteStream):
    """응답 헤더는 즉시 오지만 본문은 늦게 도착하는 스트림 — 상태 확정 뒤의 본문 지연을 만든다.

    닫기는 실제 연결처럼 체크포인트를 지난다. 취소된 스코프 안에서 닫으면 곧바로 다시 취소되므로,
    기한 초과 경로가 닫기를 shield로 감싸지 않으면 이 스트림은 열린 채로 남는다.
    """

    def __init__(self, body: bytes, delay: float) -> None:
        self._body = body
        self._delay = delay
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        await anyio.sleep(self._delay)
        yield self._body

    async def aclose(self) -> None:
        await anyio.sleep(0)
        self.closed = True


async def _gateway_client(
    credentials: GatewayCredentials,
) -> AsyncIterator[httpx.AsyncClient]:
    # 폴백 키는 각 픽스처가 lifespan 진입 전에 주입한 뒤 여기서 인프로세스 클라이언트를 연다.
    # 임시 키 저장소는 앱 팩터리 인자로만 준다 — 운영 앱의 저장소 자리는 설정으로 열리지 않는다.
    app = create_app(api_key_store_path=credentials.store_path)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://gateway.test",
            headers={"Authorization": f"Bearer {credentials.api_key}"},
        ) as client:
            yield client


@pytest.fixture
async def fallback_client(
    monkeypatch: pytest.MonkeyPatch,
    gateway_credentials: GatewayCredentials,
) -> AsyncIterator[httpx.AsyncClient]:
    monkeypatch.setattr(settings, "openai_api_key", SecretStr(TEST_KEY))
    async for client in _gateway_client(gateway_credentials):
        yield client


@pytest.fixture
async def no_key_client(
    monkeypatch: pytest.MonkeyPatch,
    gateway_credentials: GatewayCredentials,
) -> AsyncIterator[httpx.AsyncClient]:
    monkeypatch.setattr(settings, "openai_api_key", None)
    async for client in _gateway_client(gateway_credentials):
        yield client


# --- 로컬 성공은 폴백을 건드리지 않는다 ---


@respx.mock
async def test_local_success_does_not_call_openai(
    fallback_client: httpx.AsyncClient,
) -> None:
    local = respx.post(LOCAL_URL).respond(200, json={"id": "local", "choices": []})
    openai = respx.post(OPENAI_URL).respond(200, json={"id": "openai", "choices": []})

    response = await fallback_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert response.status_code == 200
    assert response.json()["id"] == "local"
    assert local.called
    assert not openai.called


@respx.mock
async def test_local_redirect_status_switches_to_openai_without_location_leak(
    fallback_client: httpx.AsyncClient,
) -> None:
    # 1xx·3xx는 유효한 추론 응답도 전달 대상 오류도 아니다 — 폴백 대상 무효 응답이다.
    local = respx.post(LOCAL_URL).respond(
        302, headers={"Location": "http://127.0.0.1:11434/internal"}
    )
    openai = respx.post(OPENAI_URL).respond(200, json={"id": "openai", "choices": []})

    response = await fallback_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert response.status_code == 200
    assert response.json()["id"] == "openai"
    assert local.called
    assert openai.called
    assert "location" not in response.headers


@respx.mock
async def test_openai_redirect_status_is_secretless_502(
    fallback_client: httpx.AsyncClient,
) -> None:
    respx.post(LOCAL_URL).respond(503, json={"error": "down"})
    respx.post(OPENAI_URL).respond(
        302, headers={"Location": "https://api.openai.com/private"}
    )

    response = await fallback_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert response.status_code == 502
    assert "location" not in response.headers
    assert TEST_KEY not in response.text


# --- 폴백 대상 장애는 즉시 OpenAI로 우회한다 ---


@pytest.mark.parametrize(
    "local_failure",
    [
        respx.MockResponse(429, json={"error": "rate limited"}),
        respx.MockResponse(404, json={"error": "model not found"}),
        respx.MockResponse(500, json={"error": "boom"}),
        respx.MockResponse(503, json={"error": "unavailable"}),
    ],
    ids=["429", "404-model-missing", "500", "503"],
)
@respx.mock
async def test_fallback_status_switches_to_openai(
    fallback_client: httpx.AsyncClient, local_failure: respx.MockResponse
) -> None:
    local = respx.post(LOCAL_URL).mock(return_value=local_failure)
    openai = respx.post(OPENAI_URL).respond(200, json={"id": "openai", "choices": []})

    response = await fallback_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert response.status_code == 200
    assert response.json()["id"] == "openai"
    assert local.called
    assert openai.called
    forwarded = json.loads(openai.calls.last.request.content)
    assert forwarded["model"] == FALLBACK_MODEL


@pytest.mark.parametrize(
    "transport_error",
    [
        httpx.ConnectError("refused"),
        httpx.ConnectTimeout("connect timed out"),
        httpx.ReadTimeout("read timed out"),
    ],
    ids=["connect-error", "connect-timeout", "read-timeout"],
)
@respx.mock
async def test_transport_failure_switches_to_openai(
    fallback_client: httpx.AsyncClient, transport_error: httpx.RequestError
) -> None:
    local = respx.post(LOCAL_URL).mock(side_effect=transport_error)
    openai = respx.post(OPENAI_URL).respond(200, json={"id": "openai", "choices": []})

    response = await fallback_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert response.status_code == 200
    assert response.json()["id"] == "openai"
    assert local.called
    assert openai.called


# --- 성공 상태인데 유효한 Chat Completions 응답이 아닌 로컬 본문도 폴백한다 ---


@pytest.mark.parametrize(
    "local_body",
    [
        b"",
        b"<html>error</html>",
        b'{"id": ',
        b"{}",
        b'{"id":"x"}',
        b'{"choices":"not-a-list"}',
        b'{"choices":[],"extra":1e400}',
    ],
    ids=[
        "empty",
        "html",
        "truncated-json",
        "empty-object",
        "missing-choices",
        "non-list-choices",
        "overflow-number",
    ],
)
@respx.mock
async def test_empty_or_invalid_local_body_switches_to_openai(
    fallback_client: httpx.AsyncClient, local_body: bytes
) -> None:
    local = respx.post(LOCAL_URL).respond(200, content=local_body)
    openai = respx.post(OPENAI_URL).respond(200, json={"id": "openai", "choices": []})

    response = await fallback_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert response.status_code == 200
    assert response.json()["id"] == "openai"
    assert local.called
    assert openai.called


@respx.mock
async def test_valid_local_body_with_unknown_fields_is_served_unchanged(
    fallback_client: httpx.AsyncClient,
) -> None:
    # choices 리스트만 있으면 유효 — 미지의 필드는 보존하고 바이트를 재직렬화하지 않는다.
    local_body = b'{"id":"local","unknown":{"x":1},"choices":[{"index":0}]}'
    local = respx.post(LOCAL_URL).respond(
        200, content=local_body, headers={"content-type": "application/json"}
    )
    openai = respx.post(OPENAI_URL)

    response = await fallback_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert response.status_code == 200
    assert response.content == local_body
    assert local.called
    assert not openai.called


# --- 클라이언트 오류와 폴백 대상 외 로컬 4xx는 우회하지 않는다 ---


@pytest.mark.parametrize("status_code", [400, 401, 403, 422])
@respx.mock
async def test_non_fallback_4xx_does_not_call_openai(
    fallback_client: httpx.AsyncClient, status_code: int
) -> None:
    error_body = b'{"error":{"message":"client mistake"}}'
    local = respx.post(LOCAL_URL).respond(status_code, content=error_body)
    openai = respx.post(OPENAI_URL).respond(200, json={"id": "openai", "choices": []})

    response = await fallback_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert response.status_code == status_code
    assert response.content == error_body
    assert local.called
    assert not openai.called


@pytest.mark.parametrize("status_code", [400, 401, 403, 422])
@respx.mock
async def test_empty_non_fallback_4xx_does_not_call_openai(
    fallback_client: httpx.AsyncClient, status_code: int
) -> None:
    # 폴백 대상 아닌 4xx는 본문이 비어 있어도 로컬 판단을 그대로 전달한다 — 폴백하지 않는다.
    local = respx.post(LOCAL_URL).respond(status_code, content=b"")
    openai = respx.post(OPENAI_URL).respond(200, json={"id": "openai", "choices": []})

    response = await fallback_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert response.status_code == status_code
    assert response.content == b""
    assert local.called
    assert not openai.called


@respx.mock
async def test_unreadable_streaming_4xx_preserves_status_without_fallback(
    fallback_client: httpx.AsyncClient,
) -> None:
    local_stream = TrackingStream([], error=httpx.ReadError("body read failed"))
    respx.post(LOCAL_URL).mock(return_value=httpx.Response(400, stream=local_stream))
    openai = respx.post(OPENAI_URL)

    response = await fallback_client.post(
        "/v1/chat/completions", json={**CHAT_REQUEST, "stream": True}
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "upstream_response_unavailable"
    assert not openai.called
    assert local_stream.closed


# 상태 코드를 받은 순간 폴백 대상 아닌 4xx는 확정된다. 그 뒤 본문이 로컬 기한을 넘겨 도착하더라도
# 로컬이 판단한 클라이언트 오류이므로 상태를 보존해 돌려주고 OpenAI로는 넘기지 않는다.
@pytest.mark.parametrize("streaming", [False, True], ids=["buffered", "streaming"])
@respx.mock
async def test_slow_4xx_body_keeps_local_decision_without_fallback(
    fallback_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
    streaming: bool,
) -> None:
    monkeypatch.setattr(settings, "local_response_start_timeout_seconds", 0.05)
    local_stream = SlowStream(b'{"error":{"message":"client mistake"}}', 0.3)
    respx.post(LOCAL_URL).mock(return_value=httpx.Response(400, stream=local_stream))
    openai = respx.post(OPENAI_URL).respond(200, json={"id": "openai", "choices": []})

    response = await fallback_client.post(
        "/v1/chat/completions", json={**CHAT_REQUEST, "stream": streaming}
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "upstream_response_unavailable"
    assert not openai.called
    assert local_stream.closed


@pytest.mark.parametrize(
    "bad_request",
    [
        {
            "content": b'{"model": broken',
            "headers": {"content-type": "application/json"},
        },
        {
            "content": b'{"model":"chat","messages":[],"value":NaN}',
            "headers": {"content-type": "application/json"},
        },
        {
            "content": b'{"model":"chat","messages":[],"value":Infinity}',
            "headers": {"content-type": "application/json"},
        },
        {
            "content": b'{"model":"chat","messages":[],"value":-Infinity}',
            "headers": {"content-type": "application/json"},
        },
        {
            "content": b'{"model":"chat","messages":[],"value":1e400}',
            "headers": {"content-type": "application/json"},
        },
        {"json": {**CHAT_REQUEST, "reasoning_effort": ""}},
        {"json": {**CHAT_REQUEST, "model": "no-such-alias"}},
    ],
    ids=[
        "malformed-json",
        "nan-constant",
        "infinity-constant",
        "negative-infinity-constant",
        "overflow-number",
        "invalid-reasoning-effort",
        "unknown-alias",
    ],
)
@respx.mock
async def test_gateway_client_error_does_not_call_openai(
    fallback_client: httpx.AsyncClient, bad_request: dict
) -> None:
    local = respx.post(LOCAL_URL)
    openai = respx.post(OPENAI_URL)

    response = await fallback_client.post("/v1/chat/completions", **bad_request)

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert not local.called
    assert not openai.called


@pytest.mark.parametrize(
    "stream",
    [None, 0, 1, "false", "true", [], {}],
    ids=["null", "zero", "one", "false-string", "true-string", "list", "object"],
)
@respx.mock
async def test_non_boolean_stream_is_rejected_before_any_upstream_call(
    fallback_client: httpx.AsyncClient, stream: object
) -> None:
    local = respx.post(LOCAL_URL)
    openai = respx.post(OPENAI_URL)

    response = await fallback_client.post(
        "/v1/chat/completions", json={**CHAT_REQUEST, "stream": stream}
    )

    assert response.status_code == 400
    assert response.json()["error"]["message"] == "'stream' must be a boolean"
    assert not local.called
    assert not openai.called


# --- reasoning_effort provider 변환 ---


@respx.mock
async def test_none_effort_becomes_minimal_for_openai(
    fallback_client: httpx.AsyncClient,
) -> None:
    respx.post(LOCAL_URL).mock(side_effect=httpx.ConnectError("down"))
    openai = respx.post(OPENAI_URL).respond(200, json={"id": "openai", "choices": []})

    await fallback_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    forwarded = json.loads(openai.calls.last.request.content)
    assert forwarded["model"] == FALLBACK_MODEL
    assert forwarded["reasoning_effort"] == "minimal"


@respx.mock
async def test_explicit_effort_passed_through_to_openai(
    fallback_client: httpx.AsyncClient,
) -> None:
    respx.post(LOCAL_URL).mock(side_effect=httpx.ConnectError("down"))
    openai = respx.post(OPENAI_URL).respond(200, json={"id": "openai", "choices": []})

    await fallback_client.post(
        "/v1/chat/completions", json={**CHAT_REQUEST, "reasoning_effort": "high"}
    )

    forwarded = json.loads(openai.calls.last.request.content)
    assert forwarded["reasoning_effort"] == "high"


@respx.mock
async def test_vision_alias_falls_back_to_fallback_model(
    fallback_client: httpx.AsyncClient,
) -> None:
    respx.post(LOCAL_URL).mock(side_effect=httpx.ConnectError("down"))
    openai = respx.post(OPENAI_URL).respond(200, json={"id": "openai", "choices": []})

    response = await fallback_client.post(
        "/v1/chat/completions", json={**CHAT_REQUEST, "model": VISION_ALIAS}
    )

    assert response.json()["id"] == "openai"
    forwarded = json.loads(openai.calls.last.request.content)
    assert forwarded["model"] == FALLBACK_MODEL


# --- 별칭별 회로 차단기 격리와 open 시 로컬 생략 ---


@respx.mock
async def test_open_circuit_skips_local_and_is_isolated_per_alias(
    fallback_client: httpx.AsyncClient,
) -> None:
    threshold = settings.circuit_breaker_failure_threshold
    local = respx.post(LOCAL_URL).respond(503, json={"error": "down"})
    respx.post(OPENAI_URL).respond(200, json={"id": "openai", "choices": []})

    # chat 별칭을 임계값만큼 실패시켜 회로를 연다.
    for _ in range(threshold):
        await fallback_client.post("/v1/chat/completions", json=CHAT_REQUEST)
    local_calls_after_open = local.call_count
    assert local_calls_after_open == threshold

    # 회로가 열린 chat 요청은 로컬을 건너뛰고 바로 OpenAI로 간다.
    await fallback_client.post("/v1/chat/completions", json=CHAT_REQUEST)
    assert local.call_count == local_calls_after_open

    # vision 별칭은 독립된 회로라 여전히 로컬을 시도한다.
    await fallback_client.post(
        "/v1/chat/completions", json={**CHAT_REQUEST, "model": VISION_ALIAS}
    )
    assert local.call_count == local_calls_after_open + 1


# --- 스트리밍 전환 경계 ---

OPENAI_SSE = [
    b'data: {"choices":[{"delta":{"content":"open"}}]}\n\n',
    b"data: [DONE]\n\n",
]
LOCAL_SSE = [
    b'data: {"choices":[{"delta":{"content":"loc"}}]}\n\n',
    b'data: {"choices":[{"delta":{"content":"al"}}]}\n\n',
    b"data: [DONE]\n\n",
]


async def _read_stream(response: httpx.Response) -> bytes:
    received = b""
    async for chunk in response.aiter_raw():
        received += chunk
    return received


@pytest.mark.parametrize(
    "local_failure",
    [
        {"side_effect": httpx.ConnectError("down")},
        {"return_value": httpx.Response(503, json={"error": "down"})},
        {"return_value": httpx.Response(200, stream=TrackingStream([]))},
    ],
    ids=["connect-error", "503", "empty-invalid-stream"],
)
@respx.mock
async def test_stream_switches_to_openai_before_first_byte(
    fallback_client: httpx.AsyncClient, local_failure: dict
) -> None:
    respx.post(LOCAL_URL).mock(**local_failure)
    openai = respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(
            200,
            stream=TrackingStream(OPENAI_SSE),
            headers={"content-type": "text/event-stream"},
        )
    )

    async with fallback_client.stream(
        "POST", "/v1/chat/completions", json={**CHAT_REQUEST, "stream": True}
    ) as response:
        received = await _read_stream(response)

    assert response.status_code == 200
    assert received == b"".join(OPENAI_SSE)
    assert openai.called
    forwarded = json.loads(openai.calls.last.request.content)
    assert forwarded["model"] == FALLBACK_MODEL
    assert forwarded["stream"] is True


@respx.mock
async def test_compressed_openai_stream_is_decoded_and_relayed(
    fallback_client: httpx.AsyncClient,
) -> None:
    respx.post(LOCAL_URL).mock(side_effect=httpx.ConnectError("local down"))
    encoded = gzip.compress(b"".join(OPENAI_SSE))
    openai = respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(
            200,
            stream=TrackingStream([encoded]),
            headers={
                "content-type": "text/event-stream",
                "content-encoding": "gzip",
            },
        )
    )

    async with fallback_client.stream(
        "POST", "/v1/chat/completions", json={**CHAT_REQUEST, "stream": True}
    ) as response:
        received = await _read_stream(response)

    assert response.status_code == 200
    assert received == b"".join(OPENAI_SSE)
    assert "content-encoding" not in response.headers
    assert openai.calls.last.request.headers["accept-encoding"] == "identity"


@respx.mock
async def test_stream_mid_failure_does_not_mix_providers(
    fallback_client: httpx.AsyncClient,
) -> None:
    local_stream = TrackingStream(
        [LOCAL_SSE[0]], error=httpx.ReadError("mid-stream drop")
    )
    respx.post(LOCAL_URL).mock(
        return_value=httpx.Response(
            200, stream=local_stream, headers={"content-type": "text/event-stream"}
        )
    )
    openai = respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(200, stream=TrackingStream(OPENAI_SSE))
    )

    with pytest.raises(httpx.ReadError, match="mid-stream drop"):
        async with fallback_client.stream(
            "POST", "/v1/chat/completions", json={**CHAT_REQUEST, "stream": True}
        ) as response:
            await _read_stream(response)

    # 첫 로컬 바이트를 보낸 뒤 장애가 나면 provider를 이어 붙이거나 정상 EOF로 위장하지 않는다.
    assert not openai.called
    assert local_stream.closed


@respx.mock
async def test_stream_success_closes_local_resource(
    fallback_client: httpx.AsyncClient,
) -> None:
    local_stream = TrackingStream(LOCAL_SSE)
    respx.post(LOCAL_URL).mock(
        return_value=httpx.Response(
            200, stream=local_stream, headers={"content-type": "text/event-stream"}
        )
    )
    openai = respx.post(OPENAI_URL)

    async with fallback_client.stream(
        "POST", "/v1/chat/completions", json={**CHAT_REQUEST, "stream": True}
    ) as response:
        received = await _read_stream(response)

    assert received == b"".join(LOCAL_SSE)
    assert not openai.called
    assert local_stream.closed


# --- 스트리밍 커밋 전 검증: 첫 유효 이벤트가 없으면 폴백하고 로컬 자원을 닫는다 ---


@pytest.mark.parametrize(
    "local_chunks",
    [
        [b"<html>bad gateway</html>"],
        [b"data: [DONE]\n\n"],
        [b'data: {"choices"'],
        [b'data: {"choices":[]}\n'],
        [b'data: {"id":"x"}\n\n'],
    ],
    ids=[
        "html",
        "done-only",
        "incomplete-sse",
        "data-line-without-event-terminator",
        "data-without-choices",
    ],
)
@respx.mock
async def test_precommit_invalid_local_stream_falls_back_and_closes(
    fallback_client: httpx.AsyncClient, local_chunks: list[bytes]
) -> None:
    local_stream = TrackingStream(local_chunks)
    respx.post(LOCAL_URL).mock(
        return_value=httpx.Response(
            200, stream=local_stream, headers={"content-type": "text/event-stream"}
        )
    )
    openai = respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(
            200,
            stream=TrackingStream(OPENAI_SSE),
            headers={"content-type": "text/event-stream"},
        )
    )

    async with fallback_client.stream(
        "POST", "/v1/chat/completions", json={**CHAT_REQUEST, "stream": True}
    ) as response:
        received = await _read_stream(response)

    # 커밋 전 무효 스트림은 로컬 바이트를 전혀 흘리지 않고 OpenAI로 전환하며 로컬 자원을 닫는다.
    assert received == b"".join(OPENAI_SSE)
    assert openai.called
    assert local_stream.closed


@pytest.mark.parametrize(
    "local_chunks",
    [
        [
            b'data: {"choi',
            b'ces":[{"delta":{"content":"hi"}}]}\n\n',
            b"data: [DONE]\n\n",
        ],
        [
            b": keep-alive\n\n",
            b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
            b"data: [DONE]\n\n",
        ],
        [
            b"event: message\r\n",
            b'data: {"choices":[{"delta":{"content":"hi"}}]}\r\n\r\n',
            b"data: [DONE]\r\n\r\n",
        ],
    ],
    ids=["split-across-chunks", "comment-keepalive-prefixed", "auxiliary-field-crlf"],
)
@respx.mock
async def test_valid_local_stream_commits_and_relays_every_byte(
    fallback_client: httpx.AsyncClient, local_chunks: list[bytes]
) -> None:
    local_stream = TrackingStream(local_chunks)
    respx.post(LOCAL_URL).mock(
        return_value=httpx.Response(
            200, stream=local_stream, headers={"content-type": "text/event-stream"}
        )
    )
    openai = respx.post(OPENAI_URL)

    async with fallback_client.stream(
        "POST", "/v1/chat/completions", json={**CHAT_REQUEST, "stream": True}
    ) as response:
        received = await _read_stream(response)

    # 확정 후에는 버퍼링한 바이트와 남은 원문을 손실·중복 없이 원문 순서 그대로 흘린다.
    assert received == b"".join(local_chunks)
    assert not openai.called
    assert local_stream.closed


@respx.mock
async def test_compressed_local_stream_does_not_trigger_openai_fallback(
    fallback_client: httpx.AsyncClient,
) -> None:
    encoded = gzip.compress(b"".join(LOCAL_SSE))
    local_stream = TrackingStream([encoded])
    local = respx.post(LOCAL_URL).mock(
        return_value=httpx.Response(
            200,
            stream=local_stream,
            headers={
                "content-type": "text/event-stream",
                "content-encoding": "gzip",
            },
        )
    )
    openai = respx.post(OPENAI_URL)

    async with fallback_client.stream(
        "POST", "/v1/chat/completions", json={**CHAT_REQUEST, "stream": True}
    ) as response:
        received = await _read_stream(response)

    assert response.status_code == 200
    assert received == b"".join(LOCAL_SSE)
    assert "content-encoding" not in response.headers
    assert local.calls.last.request.headers["accept-encoding"] == "identity"
    assert not openai.called
    assert local_stream.closed


# --- 비밀 비노출과 OpenAI 자체 실패 ---


@respx.mock
async def test_missing_key_returns_secret_free_502(
    no_key_client: httpx.AsyncClient,
) -> None:
    local = respx.post(LOCAL_URL).mock(side_effect=httpx.ConnectError("down"))
    openai = respx.post(OPENAI_URL)

    response = await no_key_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert response.status_code == 502
    error = response.json()["error"]
    assert error["code"] == "fallback_unavailable"
    assert local.called
    assert not openai.called
    assert TEST_KEY not in response.text


@respx.mock
async def test_openai_connection_failure_returns_secret_free_502(
    fallback_client: httpx.AsyncClient,
) -> None:
    respx.post(LOCAL_URL).mock(side_effect=httpx.ConnectError("local down"))
    respx.post(OPENAI_URL).mock(side_effect=httpx.ConnectError("openai down"))

    response = await fallback_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "fallback_unavailable"
    assert TEST_KEY not in response.text


@respx.mock
async def test_openai_error_status_is_relayed(
    fallback_client: httpx.AsyncClient,
) -> None:
    respx.post(LOCAL_URL).mock(side_effect=httpx.ConnectError("down"))
    error_body = b'{"error":{"message":"openai rate limit","type":"rate_limit_error"}}'
    respx.post(OPENAI_URL).respond(429, content=error_body)

    response = await fallback_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert response.status_code == 429
    assert response.content == error_body


@pytest.mark.parametrize(
    "openai_body",
    [b"<html>bad gateway</html>", b'{"choices":[],"extra":1e400}'],
    ids=["non-json", "overflow-number"],
)
@respx.mock
async def test_openai_invalid_buffered_response_returns_secret_free_502(
    fallback_client: httpx.AsyncClient, openai_body: bytes
) -> None:
    respx.post(LOCAL_URL).mock(side_effect=httpx.ConnectError("local down"))
    respx.post(OPENAI_URL).respond(200, content=openai_body)

    response = await fallback_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "fallback_unavailable"
    assert TEST_KEY not in response.text


@respx.mock
async def test_unreadable_openai_error_stream_returns_secret_free_502(
    fallback_client: httpx.AsyncClient,
) -> None:
    respx.post(LOCAL_URL).mock(side_effect=httpx.ConnectError("local down"))
    openai_stream = TrackingStream([], error=httpx.ReadError("body read failed"))
    respx.post(OPENAI_URL).mock(return_value=httpx.Response(429, stream=openai_stream))

    response = await fallback_client.post(
        "/v1/chat/completions", json={**CHAT_REQUEST, "stream": True}
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "fallback_unavailable"
    assert TEST_KEY not in response.text
    assert openai_stream.closed


@pytest.mark.parametrize(
    "openai_stream",
    [
        {"return_value": httpx.Response(200, stream=TrackingStream([]))},
        {
            "return_value": httpx.Response(
                200,
                stream=TrackingStream([], error=httpx.ReadError("first read drop")),
            )
        },
    ],
    ids=["empty-stream", "first-read-failure"],
)
@respx.mock
async def test_openai_stream_without_first_byte_returns_secret_free_502(
    fallback_client: httpx.AsyncClient, openai_stream: dict
) -> None:
    respx.post(LOCAL_URL).mock(side_effect=httpx.ConnectError("local down"))
    respx.post(OPENAI_URL).mock(**openai_stream)

    # 첫 바이트를 확보하지 못한 폴백 스트림은 스트림을 시작하지 않고 비밀 없는 502로 합성한다.
    response = await fallback_client.post(
        "/v1/chat/completions", json={**CHAT_REQUEST, "stream": True}
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "fallback_unavailable"
    assert TEST_KEY not in response.text


@pytest.mark.parametrize(
    "openai_chunks",
    [[b"<html>bad</html>"], [b"data: [DONE]\n\n"], [b'data: {"id":"x"}\n\n']],
    ids=["html", "done-only", "data-without-choices"],
)
@respx.mock
async def test_openai_invalid_2xx_stream_returns_secret_free_502(
    fallback_client: httpx.AsyncClient, openai_chunks: list[bytes]
) -> None:
    respx.post(LOCAL_URL).mock(side_effect=httpx.ConnectError("local down"))
    openai_stream = TrackingStream(openai_chunks)
    respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(
            200, stream=openai_stream, headers={"content-type": "text/event-stream"}
        )
    )

    # 2xx이지만 첫 유효 이벤트가 없는 폴백 스트림은 시작하지 않고 비밀 없는 502로 합성한다.
    response = await fallback_client.post(
        "/v1/chat/completions", json={**CHAT_REQUEST, "stream": True}
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "fallback_unavailable"
    assert TEST_KEY not in response.text
    assert openai_stream.closed


# --- 폴백 비대상 chat 별칭은 로컬로만 라우팅하고 OpenAI를 호출하지 않는다 ---

_LOCAL_ONLY_ROUTING = """\
routes:
  - alias: chat
    endpoint: chat
    model: gemma4:12b-it-qat
  - alias: vision
    endpoint: chat
    model: gemma4:12b-it-qat
  - alias: summarize
    endpoint: chat
    model: gemma4:12b-it-qat
  - alias: embed
    endpoint: embeddings
    model: snowflake-arctic-embed2
"""


@respx.mock
async def test_extra_chat_alias_routes_local_only_and_never_calls_openai(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    gateway_credentials: GatewayCredentials,
) -> None:
    routing_path = tmp_path / "routing.yaml"
    routing_path.write_text(_LOCAL_ONLY_ROUTING, encoding="utf-8")
    monkeypatch.setattr(settings, "routing_config_path", routing_path)
    monkeypatch.setattr(settings, "openai_api_key", SecretStr(TEST_KEY))

    local = respx.post(LOCAL_URL).mock(side_effect=httpx.ConnectError("down"))
    openai = respx.post(OPENAI_URL)

    async for client in _gateway_client(gateway_credentials):
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "summarize", "messages": [{"role": "user", "content": "x"}]},
        )

    # 로컬로 라우팅됐지만(실제 모델로 치환) 장애 시 OpenAI로 우회하지 않고 로컬 업스트림 오류를 낸다.
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_unavailable"
    assert local.called
    assert not openai.called
    forwarded = json.loads(local.calls.last.request.content)
    assert forwarded["model"] == CHAT_MODEL


# --- 임베딩은 폴백에서 완전히 제외된다 ---


@respx.mock
async def test_embed_connection_failure_never_calls_openai(
    fallback_client: httpx.AsyncClient,
) -> None:
    embed = respx.post(EMBED_URL).mock(side_effect=httpx.ConnectError("embed down"))
    openai = respx.post(OPENAI_URL)

    response = await fallback_client.post(
        "/v1/embeddings", json={"model": EMBED_ALIAS, "input": "x"}
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_unavailable"
    assert embed.called
    assert not openai.called


@respx.mock
async def test_embed_uses_dedicated_base_url(
    fallback_client: httpx.AsyncClient,
) -> None:
    embed = respx.post(EMBED_URL).respond(
        200,
        json={
            "model": "snowflake-arctic-embed2",
            "embeddings": [[0.1] * NATIVE_EMBEDDING_VECTOR_DIMENSIONS],
        },
    )
    other = respx.post(LOCAL_URL)

    response = await fallback_client.post(
        "/v1/embeddings", json={"model": EMBED_ALIAS, "input": "x"}
    )

    assert response.status_code == 200
    assert embed.called
    assert not other.called
