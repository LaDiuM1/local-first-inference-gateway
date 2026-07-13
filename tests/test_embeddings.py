"""임베딩 엔드포인트 계약 테스트 — native /api/embed(CPU 강제)를 respx로 목킹한다."""

import base64
import json
import struct

import httpx
import pytest
import respx

from gateway.config import settings

pytestmark = pytest.mark.anyio

UPSTREAM_URL = f"{settings.ollama_base_url}/api/embed"
EMBED_ALIAS = "embed"
EMBED_MODEL = "snowflake-arctic-embed2"


def _native_response(vectors: list[list[float]], prompt_tokens: int = 3) -> dict:
    return {
        "model": EMBED_MODEL,
        "embeddings": vectors,
        "prompt_eval_count": prompt_tokens,
    }


def _decode_base64(encoded: str, dimensions: int) -> list[float]:
    # OpenAI SDK와 동일한 방식으로 base64를 float32 리틀엔디언 벡터로 되돌린다.
    return list(struct.unpack(f"<{dimensions}f", base64.b64decode(encoded)))


@respx.mock
async def test_embed_single_string_converts_to_openai_format(
    gateway_client: httpx.AsyncClient,
) -> None:
    route = respx.post(UPSTREAM_URL).respond(200, json=_native_response([[0.1, 0.2]]))

    response = await gateway_client.post(
        "/v1/embeddings", json={"model": EMBED_ALIAS, "input": "상품 설명"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["model"] == EMBED_MODEL
    assert body["data"] == [
        {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}
    ]
    assert body["usage"] == {"prompt_tokens": 3, "total_tokens": 3}

    native_request = json.loads(route.calls.last.request.content)
    assert native_request["model"] == EMBED_MODEL
    assert native_request["input"] == "상품 설명"
    assert native_request["options"] == {"num_gpu": 0}


@respx.mock
async def test_embed_array_preserves_order_and_index(
    gateway_client: httpx.AsyncClient,
) -> None:
    vectors = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    route = respx.post(UPSTREAM_URL).respond(200, json=_native_response(vectors))

    response = await gateway_client.post(
        "/v1/embeddings", json={"model": EMBED_ALIAS, "input": ["첫째", "둘째", "셋째"]}
    )

    body = response.json()
    assert [item["index"] for item in body["data"]] == [0, 1, 2]
    assert [item["embedding"] for item in body["data"]] == vectors
    assert json.loads(route.calls.last.request.content)["input"] == [
        "첫째",
        "둘째",
        "셋째",
    ]


@respx.mock
async def test_embed_forwards_cpu_only_option(
    gateway_client: httpx.AsyncClient,
) -> None:
    route = respx.post(UPSTREAM_URL).respond(200, json=_native_response([[0.1]]))

    await gateway_client.post(
        "/v1/embeddings", json={"model": EMBED_ALIAS, "input": "x"}
    )

    native_request = json.loads(route.calls.last.request.content)
    assert native_request["options"]["num_gpu"] == 0


@respx.mock
async def test_embed_explicit_float_returns_float_list(
    gateway_client: httpx.AsyncClient,
) -> None:
    route = respx.post(UPSTREAM_URL).respond(200, json=_native_response([[0.1, 0.2]]))

    response = await gateway_client.post(
        "/v1/embeddings",
        json={"model": EMBED_ALIAS, "input": "x", "encoding_format": "float"},
    )

    assert response.json()["data"][0]["embedding"] == [0.1, 0.2]
    native_request = json.loads(route.calls.last.request.content)
    assert "encoding_format" not in native_request
    assert native_request["options"] == {"num_gpu": 0}


@respx.mock
async def test_embed_base64_returns_decodable_float32(
    gateway_client: httpx.AsyncClient,
) -> None:
    vector = [0.1, 0.2, 0.3]
    respx.post(UPSTREAM_URL).respond(200, json=_native_response([vector]))

    response = await gateway_client.post(
        "/v1/embeddings",
        json={"model": EMBED_ALIAS, "input": "x", "encoding_format": "base64"},
    )

    encoded = response.json()["data"][0]["embedding"]
    assert isinstance(encoded, str)
    assert _decode_base64(encoded, len(vector)) == pytest.approx(vector)


@respx.mock
async def test_embed_base64_array_preserves_order_and_cpu_option(
    gateway_client: httpx.AsyncClient,
) -> None:
    vectors = [[0.1, 0.2], [0.3, 0.4]]
    route = respx.post(UPSTREAM_URL).respond(200, json=_native_response(vectors))

    response = await gateway_client.post(
        "/v1/embeddings",
        json={"model": EMBED_ALIAS, "input": ["a", "b"], "encoding_format": "base64"},
    )

    data = response.json()["data"]
    assert [item["index"] for item in data] == [0, 1]
    for item, vector in zip(data, vectors, strict=True):
        assert _decode_base64(item["embedding"], len(vector)) == pytest.approx(vector)
    native_request = json.loads(route.calls.last.request.content)
    assert native_request["options"] == {"num_gpu": 0}
    assert "encoding_format" not in native_request


@pytest.mark.parametrize(
    "encoding_format",
    ["utf-8", "int8", "", 123, []],
    ids=["utf-8", "int8", "empty", "int", "list"],
)
@respx.mock
async def test_embed_unsupported_encoding_format_gets_400_without_upstream_call(
    gateway_client: httpx.AsyncClient, encoding_format: object
) -> None:
    route = respx.post(UPSTREAM_URL)

    response = await gateway_client.post(
        "/v1/embeddings",
        json={"model": EMBED_ALIAS, "input": "x", "encoding_format": encoding_format},
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert not route.called


@respx.mock
async def test_embed_synthesizes_502_when_upstream_unreachable(
    gateway_client: httpx.AsyncClient,
) -> None:
    respx.post(UPSTREAM_URL).mock(side_effect=httpx.ConnectError("connection refused"))

    response = await gateway_client.post(
        "/v1/embeddings", json={"model": EMBED_ALIAS, "input": "x"}
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_unavailable"


@respx.mock
async def test_embed_wraps_upstream_error_as_openai_format(
    gateway_client: httpx.AsyncClient,
) -> None:
    respx.post(UPSTREAM_URL).respond(404, json={"error": "model not found"})

    response = await gateway_client.post(
        "/v1/embeddings", json={"model": EMBED_ALIAS, "input": "x"}
    )

    assert response.status_code == 404
    error = response.json()["error"]
    assert error["type"] == "upstream_error"
    assert error["message"] == "model not found"


@pytest.mark.parametrize(
    "rejected_model",
    [EMBED_MODEL, "chat", "vision", "embeddd"],
    ids=["real-model-name", "chat-alias", "vision-alias", "typo-alias"],
)
@respx.mock
async def test_embed_rejects_non_embed_alias_without_upstream_call(
    gateway_client: httpx.AsyncClient, rejected_model: str
) -> None:
    route = respx.post(UPSTREAM_URL)

    response = await gateway_client.post(
        "/v1/embeddings", json={"model": rejected_model, "input": "x"}
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert not route.called


@pytest.mark.parametrize(
    "body",
    [
        {"model": EMBED_ALIAS},
        {"model": EMBED_ALIAS, "input": 123},
        {"model": EMBED_ALIAS, "input": []},
        {"model": EMBED_ALIAS, "input": ["ok", 5]},
        {"input": "x"},
    ],
    ids=[
        "missing-input",
        "non-string-input",
        "empty-list",
        "non-string-item",
        "missing-model",
    ],
)
@respx.mock
async def test_embed_invalid_input_gets_400_without_upstream_call(
    gateway_client: httpx.AsyncClient, body: dict
) -> None:
    route = respx.post(UPSTREAM_URL)

    response = await gateway_client.post("/v1/embeddings", json=body)

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert not route.called


@pytest.mark.parametrize(
    "broken_body",
    [b'{"model": broken', b"[1, 2]", b'"just a string"', b"\xff\xfe"],
    ids=["malformed-json", "json-array", "json-string", "invalid-utf8"],
)
@respx.mock
async def test_embed_invalid_json_gets_400_without_upstream_call(
    gateway_client: httpx.AsyncClient, broken_body: bytes
) -> None:
    route = respx.post(UPSTREAM_URL)

    response = await gateway_client.post(
        "/v1/embeddings",
        content=broken_body,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert not route.called
