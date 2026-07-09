"""chat completions 투명 중계 계약 테스트 — 업스트림(Ollama)은 respx로 목킹한다."""

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
