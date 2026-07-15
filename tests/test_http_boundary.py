"""인증, 요청 본문 제한, 캐시와 공개 문서 HTTP 경계 테스트."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol

import httpx
import pytest
import respx
from fastapi import FastAPI

from gateway.api_keys import ApiKeyStore
from gateway.config import Settings, settings
from gateway.middleware import MAX_REQUEST_BODY_BYTES
from gateway.paths import PUBLIC_DOCS_PATH

pytestmark = pytest.mark.anyio

UPSTREAM_URL = f"{settings.ollama_base_url}/v1/chat/completions"
CHAT_REQUEST = {"model": "chat", "messages": [{"role": "user", "content": "x"}]}


class GatewayCredentials(Protocol):
    store_path: Path
    api_key: str

    @property
    def authorization_headers(self) -> dict[str, str]: ...


class CountingStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.iterations = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            self.iterations += 1
            yield chunk


async def _client(
    app: FastAPI, headers: dict[str, str] | None = None
) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://gateway.test",
        headers=headers,
    )


@pytest.mark.parametrize(
    "authorization",
    [None, "", "Basic abc", "Bearer", "Bearer one two", "Token abc"],
    ids=["missing", "empty", "basic", "no-token", "extra-part", "wrong-scheme"],
)
@respx.mock
async def test_missing_or_malformed_auth_is_401_before_upstream(
    running_gateway: FastAPI, authorization: str | None
) -> None:
    upstream = respx.post(UPSTREAM_URL)
    headers = {}
    if authorization is not None:
        headers["Authorization"] = authorization
    async with await _client(running_gateway, headers) as client:
        response = await client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["www-authenticate"] == "Bearer"
    assert not upstream.called


@respx.mock
async def test_revoked_key_is_401_before_upstream(
    running_gateway: FastAPI, gateway_credentials: GatewayCredentials
) -> None:
    store = ApiKeyStore(gateway_credentials.store_path)
    key_id = store.list_keys()[0].key_id
    store.revoke(key_id)
    upstream = respx.post(UPSTREAM_URL)

    async with await _client(
        running_gateway, gateway_credentials.authorization_headers
    ) as client:
        response = await client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert response.status_code == 401
    assert not upstream.called


@respx.mock
async def test_authentication_rejects_without_receiving_body(
    running_gateway: FastAPI,
) -> None:
    stream = CountingStream([b"must-not-be-read"])
    upstream = respx.post(UPSTREAM_URL)

    async with await _client(running_gateway) as client:
        response = await client.post(
            "/v1/chat/completions",
            content=stream,
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 401
    assert stream.iterations == 0
    assert not upstream.called


@respx.mock
async def test_duplicate_authorization_headers_are_rejected(
    running_gateway: FastAPI, gateway_credentials: GatewayCredentials
) -> None:
    upstream = respx.post(UPSTREAM_URL)
    headers = [
        ("Authorization", f"Bearer {gateway_credentials.api_key}"),
        ("Authorization", f"Bearer {gateway_credentials.api_key}"),
    ]

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=running_gateway),
        base_url="http://gateway.test",
        headers=headers,
    ) as client:
        response = await client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert response.status_code == 401
    assert not upstream.called


@respx.mock
async def test_corrupt_authentication_store_fails_closed_without_internal_details(
    running_gateway: FastAPI, gateway_credentials: GatewayCredentials
) -> None:
    gateway_credentials.store_path.write_text("{broken", encoding="utf-8")
    upstream = respx.post(UPSTREAM_URL)

    async with await _client(
        running_gateway, gateway_credentials.authorization_headers
    ) as client:
        response = await client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "authentication_unavailable"
    assert str(gateway_credentials.store_path) not in response.text
    assert gateway_credentials.api_key not in response.text
    assert not upstream.called


@respx.mock
async def test_oversized_content_length_is_rejected_before_body_receive(
    running_gateway: FastAPI, gateway_credentials: GatewayCredentials
) -> None:
    stream = CountingStream([b"must-not-be-read"])
    upstream = respx.post(UPSTREAM_URL)

    async with await _client(
        running_gateway, gateway_credentials.authorization_headers
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            content=stream,
            headers={"Content-Length": str(MAX_REQUEST_BODY_BYTES + 1)},
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"
    assert response.headers["cache-control"] == "no-store"
    assert stream.iterations == 0
    assert not upstream.called


@respx.mock
async def test_exact_20_mib_body_reaches_json_parser_and_upstream(
    gateway_client: httpx.AsyncClient,
) -> None:
    upstream = respx.post(UPSTREAM_URL).respond(200, json={"choices": []})
    document = b'{"model":"chat","messages":[]}'
    body = document + b" " * (MAX_REQUEST_BODY_BYTES - len(document))

    response = await gateway_client.post(
        "/v1/chat/completions",
        content=body,
        headers={"Content-Type": "application/json"},
    )

    assert len(body) == MAX_REQUEST_BODY_BYTES
    assert response.status_code == 200
    assert upstream.called


@respx.mock
async def test_chunked_body_is_stopped_when_actual_bytes_exceed_limit(
    running_gateway: FastAPI, gateway_credentials: GatewayCredentials
) -> None:
    stream = CountingStream(
        [b"{" + b" " * (10 * 1024 * 1024 - 1), b" " * (10 * 1024 * 1024), b"x", b"tail"]
    )
    upstream = respx.post(UPSTREAM_URL)

    async with await _client(
        running_gateway, gateway_credentials.authorization_headers
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            content=stream,
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 413
    assert stream.iterations == 3
    assert not upstream.called


# 과도하게 중첩된 JSON은 표준 JSON 파서의 재귀 한도를 넘긴다. 인증된 요청이 처리되지 않은 500이
# 아니라 계약 위반 400으로 끝나고 캐시 정책도 그대로 적용되는지 확인한다.
@pytest.mark.parametrize(
    "endpoint", ["/v1/chat/completions", "/v1/embeddings"], ids=["chat", "embeddings"]
)
@respx.mock
async def test_deeply_nested_json_is_400_without_upstream_call(
    gateway_client: httpx.AsyncClient, endpoint: str
) -> None:
    upstream = respx.post(UPSTREAM_URL)
    embedding_upstream = respx.post(f"{settings.embedding_ollama_base_url}/api/embed")
    depth = 200_000
    body = b'{"model":"chat","messages":' + b"[" * depth + b"]" * depth + b"}"

    response = await gateway_client.post(
        endpoint, content=body, headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert response.headers["cache-control"] == "no-store"
    assert not upstream.called
    assert not embedding_upstream.called


@respx.mock
async def test_deeply_nested_but_valid_json_still_reaches_upstream(
    gateway_client: httpx.AsyncClient,
) -> None:
    # 재귀 한도 안의 평범한 중첩은 계속 정상 요청이다 — 400 경계가 유효 JSON을 삼키지 않는다.
    upstream = respx.post(UPSTREAM_URL).respond(200, json={"choices": []})
    nested = {"model": "chat", "messages": [], "metadata": {}}
    cursor = nested["metadata"]
    for _ in range(50):
        cursor["child"] = {}
        cursor = cursor["child"]

    response = await gateway_client.post("/v1/chat/completions", json=nested)

    assert response.status_code == 200
    assert upstream.called


async def test_health_requires_auth_and_never_caches(
    running_gateway: FastAPI, gateway_credentials: GatewayCredentials
) -> None:
    async with await _client(running_gateway) as client:
        unauthorized = await client.get("/health")
    async with await _client(
        running_gateway, gateway_credentials.authorization_headers
    ) as client:
        healthy = await client.get("/health")

    assert unauthorized.status_code == 401
    assert unauthorized.headers["cache-control"] == "no-store"
    assert healthy.status_code == 200
    assert healthy.headers["cache-control"] == "no-store"


async def test_docs_are_public_cached_and_automatic_schema_routes_are_disabled(
    running_gateway: FastAPI,
) -> None:
    async with await _client(running_gateway) as client:
        docs = await client.get("/docs")
        openapi = await client.get("/openapi.json")
        redoc = await client.get("/redoc")

    assert docs.status_code == 200
    assert docs.headers["cache-control"] == "public, max-age=300"
    assert "https://&lt;public-host&gt;/v1" in docs.text
    assert "1024" in docs.text
    assert openapi.status_code == 404
    assert redoc.status_code == 404


def test_public_docs_markdown_is_single_secret_free_contract_source() -> None:
    source_path = Path("docs/API.md")
    source = source_path.read_text(encoding="utf-8")

    for required in [
        "https://<public-host>/v1",
        "Authorization: Bearer <API_KEY>",
        "/v1/chat/completions",
        "/v1/embeddings",
        "`chat`",
        "`vision`",
        "`embed`",
        "1024차원",
        "SSE",
        "image_url",
        '"base64"',
        "20MiB",
        "90초",
        "115초",
        "OpenAI 형식",
        "stream: true",
    ]:
        assert required in source
    for forbidden in ["127.0.0.1", "11434", "11435", "ProgramData", "C:\\"]:
        assert forbidden not in source
    assert not source_path.with_suffix(".html").exists()


def test_public_docs_source_cannot_be_redirected_by_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GATEWAY_PUBLIC_DOCS_PATH", str(tmp_path / "private.txt"))

    configuration = Settings(_env_file=None)

    assert not hasattr(configuration, "public_docs_path")
    expected_path = Path(__file__).resolve().parents[1] / "docs" / "API.md"
    assert expected_path == PUBLIC_DOCS_PATH
