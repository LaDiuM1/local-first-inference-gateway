"""chat completions 투명 중계 계약 테스트 — 업스트림(Ollama)은 respx로 목킹한다."""

from collections.abc import AsyncIterator

import httpx
import pytest
import respx

from gateway.config import settings

pytestmark = pytest.mark.anyio

UPSTREAM_URL = f"{settings.ollama_base_url}/v1/chat/completions"
CHAT_REQUEST = {
    "model": "gemma4:12b-it-qat",
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
async def test_relay_sends_request_body_unchanged(
    gateway_client: httpx.AsyncClient,
) -> None:
    route = respx.post(UPSTREAM_URL).respond(200, json={})
    raw_request = b'{"model":"gemma4:12b-it-qat","stream":false,"unknown_option":1}'

    await gateway_client.post(
        "/v1/chat/completions",
        content=raw_request,
        headers={"content-type": "application/json"},
    )

    assert route.calls.last.request.content == raw_request


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
async def test_stream_relay_passes_sse_bytes_through(
    gateway_client: httpx.AsyncClient,
) -> None:
    route = respx.post(UPSTREAM_URL).mock(
        return_value=httpx.Response(
            200,
            stream=ChunkedStream(SSE_CHUNKS),
            headers={"content-type": "text/event-stream"},
        )
    )
    raw_request = b'{"model":"gemma4:12b-it-qat","stream":true}'

    received = b""
    async with gateway_client.stream(
        "POST",
        "/v1/chat/completions",
        content=raw_request,
        headers={"content-type": "application/json"},
    ) as response:
        async for chunk in response.aiter_raw():
            received += chunk

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/event-stream"
    assert received == b"".join(SSE_CHUNKS)
    assert route.calls.last.request.content == raw_request


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
    "broken_body", [b'{"model": broken', b"[1, 2]", b'"just a string"']
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
