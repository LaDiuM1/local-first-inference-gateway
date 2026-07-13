"""chat completions 별칭 라우팅 중계 계약 테스트 — 업스트림(Ollama)은 respx로 목킹한다."""

import json
from collections.abc import AsyncIterator

import httpx
import pytest
import respx

from gateway.config import settings

pytestmark = pytest.mark.anyio

UPSTREAM_URL = f"{settings.ollama_base_url}/v1/chat/completions"
CHAT_ALIAS = "chat"
CHAT_MODEL = "gemma4:12b-it-qat"
CHAT_REQUEST = {
    "model": CHAT_ALIAS,
    "messages": [{"role": "user", "content": "안녕"}],
}


@respx.mock
async def test_relay_returns_upstream_body_unchanged(
    gateway_client: httpx.AsyncClient,
) -> None:
    upstream_body = b'{"id":"chatcmpl-1","unknown_field":{"nested":1},"choices":[]}'
    respx.post(UPSTREAM_URL).respond(
        200, content=upstream_body, headers={"content-type": "application/json"}
    )

    response = await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert response.status_code == 200
    assert response.content == upstream_body
    assert response.headers["content-type"] == "application/json"


@respx.mock
async def test_relay_substitutes_alias_and_preserves_other_fields(
    gateway_client: httpx.AsyncClient,
) -> None:
    route = respx.post(UPSTREAM_URL).respond(200, json={})

    await gateway_client.post(
        "/v1/chat/completions",
        json={
            "model": CHAT_ALIAS,
            "messages": [{"role": "user", "content": "안녕"}],
            "unknown_option": 1,
        },
    )

    forwarded = json.loads(route.calls.last.request.content)
    assert forwarded["model"] == CHAT_MODEL
    assert forwarded["messages"] == [{"role": "user", "content": "안녕"}]
    assert forwarded["unknown_option"] == 1


@respx.mock
async def test_reasoning_effort_omitted_defaults_to_none(
    gateway_client: httpx.AsyncClient,
) -> None:
    route = respx.post(UPSTREAM_URL).respond(200, json={})

    await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    forwarded = json.loads(route.calls.last.request.content)
    assert forwarded["reasoning_effort"] == "none"


@pytest.mark.parametrize("effort", ["high", "medium", "low", "max", "none"])
@respx.mock
async def test_reasoning_effort_string_passed_through(
    gateway_client: httpx.AsyncClient, effort: str
) -> None:
    route = respx.post(UPSTREAM_URL).respond(200, json={})

    await gateway_client.post(
        "/v1/chat/completions", json={**CHAT_REQUEST, "reasoning_effort": effort}
    )

    forwarded = json.loads(route.calls.last.request.content)
    assert forwarded["reasoning_effort"] == effort


@pytest.mark.parametrize(
    "bad_effort",
    ["", 1, 1.5, True, [], {}, None],
    ids=["empty", "int", "float", "bool", "list", "dict", "null"],
)
@respx.mock
async def test_invalid_reasoning_effort_gets_400_without_upstream_call(
    gateway_client: httpx.AsyncClient, bad_effort: object
) -> None:
    route = respx.post(UPSTREAM_URL)

    response = await gateway_client.post(
        "/v1/chat/completions", json={**CHAT_REQUEST, "reasoning_effort": bad_effort}
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert not route.called


@respx.mock
async def test_vision_alias_maps_to_configured_model(
    gateway_client: httpx.AsyncClient,
) -> None:
    route = respx.post(UPSTREAM_URL).respond(200, json={})

    await gateway_client.post(
        "/v1/chat/completions", json={**CHAT_REQUEST, "model": "vision"}
    )

    forwarded = json.loads(route.calls.last.request.content)
    assert forwarded["model"] == CHAT_MODEL


@pytest.mark.parametrize("status_code", [400, 404, 500])
@respx.mock
async def test_relay_passes_upstream_error_through(
    gateway_client: httpx.AsyncClient, status_code: int
) -> None:
    error_body = b'{"error":{"message":"upstream says no"}}'
    respx.post(UPSTREAM_URL).respond(
        status_code, content=error_body, headers={"content-type": "application/json"}
    )

    response = await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert response.status_code == status_code
    assert response.content == error_body


@respx.mock
async def test_relay_synthesizes_502_when_upstream_unreachable(
    gateway_client: httpx.AsyncClient,
) -> None:
    respx.post(UPSTREAM_URL).mock(side_effect=httpx.ConnectError("connection refused"))

    response = await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert response.status_code == 502
    error = response.json()["error"]
    assert error["code"] == "upstream_unavailable"


@respx.mock
async def test_relay_drops_hop_by_hop_headers(
    gateway_client: httpx.AsyncClient,
) -> None:
    respx.post(UPSTREAM_URL).respond(
        200, json={}, headers={"transfer-encoding": "chunked", "x-request-id": "abc"}
    )

    response = await gateway_client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert "transfer-encoding" not in response.headers
    assert response.headers["x-request-id"] == "abc"


SSE_CHUNKS = [
    b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n',
    b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
    b"data: [DONE]\n\n",
]


class ChunkedStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


@respx.mock
async def test_stream_relay_substitutes_alias_and_passes_sse_bytes_through(
    gateway_client: httpx.AsyncClient,
) -> None:
    route = respx.post(UPSTREAM_URL).mock(
        return_value=httpx.Response(
            200,
            stream=ChunkedStream(SSE_CHUNKS),
            headers={"content-type": "text/event-stream"},
        )
    )

    received = b""
    async with gateway_client.stream(
        "POST",
        "/v1/chat/completions",
        json={**CHAT_REQUEST, "stream": True},
    ) as response:
        async for chunk in response.aiter_raw():
            received += chunk

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    assert received == b"".join(SSE_CHUNKS)
    forwarded = json.loads(route.calls.last.request.content)
    assert forwarded["model"] == CHAT_MODEL
    assert forwarded["stream"] is True
    # 스트리밍도 논스트리밍과 동일하게 reasoning_effort 생략 시 기본 none을 넣는다.
    assert forwarded["reasoning_effort"] == "none"


@respx.mock
async def test_stream_relay_passes_reasoning_effort_through(
    gateway_client: httpx.AsyncClient,
) -> None:
    route = respx.post(UPSTREAM_URL).mock(
        return_value=httpx.Response(
            200,
            stream=ChunkedStream(SSE_CHUNKS),
            headers={"content-type": "text/event-stream"},
        )
    )

    async with gateway_client.stream(
        "POST",
        "/v1/chat/completions",
        json={**CHAT_REQUEST, "stream": True, "reasoning_effort": "high"},
    ) as response:
        async for _ in response.aiter_raw():
            pass

    forwarded = json.loads(route.calls.last.request.content)
    assert forwarded["reasoning_effort"] == "high"


@respx.mock
async def test_stream_relay_rejects_invalid_reasoning_effort_without_upstream_call(
    gateway_client: httpx.AsyncClient,
) -> None:
    route = respx.post(UPSTREAM_URL)

    response = await gateway_client.post(
        "/v1/chat/completions",
        json={**CHAT_REQUEST, "stream": True, "reasoning_effort": ""},
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert not route.called


@respx.mock
async def test_stream_relay_synthesizes_502_when_upstream_unreachable(
    gateway_client: httpx.AsyncClient,
) -> None:
    respx.post(UPSTREAM_URL).mock(side_effect=httpx.ConnectError("connection refused"))

    response = await gateway_client.post(
        "/v1/chat/completions", json={**CHAT_REQUEST, "stream": True}
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_unavailable"


@pytest.mark.parametrize(
    "broken_body",
    [b'{"model": broken', b"[1, 2]", b'"just a string"', b"\xff\xfe"],
    ids=["malformed-json", "json-array", "json-string", "invalid-utf8"],
)
@respx.mock
async def test_invalid_json_gets_400_without_upstream_call(
    gateway_client: httpx.AsyncClient, broken_body: bytes
) -> None:
    route = respx.post(UPSTREAM_URL)

    response = await gateway_client.post(
        "/v1/chat/completions",
        content=broken_body,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert not route.called


@pytest.mark.parametrize(
    "rejected_model",
    [CHAT_MODEL, "embed", "chatt", "no-such-model"],
    ids=["real-model-name", "wrong-endpoint-alias", "typo-alias", "unknown-alias"],
)
@respx.mock
async def test_non_alias_model_gets_400_without_upstream_call(
    gateway_client: httpx.AsyncClient, rejected_model: str
) -> None:
    route = respx.post(UPSTREAM_URL)

    response = await gateway_client.post(
        "/v1/chat/completions", json={**CHAT_REQUEST, "model": rejected_model}
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert not route.called


@pytest.mark.parametrize("bad_model", [None, 123, ""], ids=["missing", "int", "empty"])
@respx.mock
async def test_missing_or_non_string_model_gets_400(
    gateway_client: httpx.AsyncClient, bad_model: object
) -> None:
    route = respx.post(UPSTREAM_URL)
    body = {"messages": [{"role": "user", "content": "안녕"}]}
    if bad_model is not None:
        body["model"] = bad_model

    response = await gateway_client.post("/v1/chat/completions", json=body)

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert not route.called
