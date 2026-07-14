"""임베딩 엔드포인트 계약 테스트 — native /api/embed(CPU 강제)를 respx로 목킹한다."""

import base64
import json
import struct

import httpx
import pytest
import respx

from gateway.config import settings

pytestmark = pytest.mark.anyio

# 임베딩은 chat·vision과 분리된 임베딩 전용 인스턴스(별도 base URL)만 호출한다.
UPSTREAM_URL = f"{settings.embedding_ollama_base_url}/api/embed"
OPENAI_URL = f"{settings.openai_base_url}/chat/completions"
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


@pytest.mark.parametrize("status_code", [201, 202, 204, 206, 299])
@respx.mock
async def test_embed_non_200_success_status_becomes_502_without_openai(
    gateway_client: httpx.AsyncClient, status_code: int
) -> None:
    respx.post(UPSTREAM_URL).respond(
        status_code,
        content=b'{"embeddings":[[0.1]],"prompt_eval_count":1}',
        headers={"content-type": "application/json"},
    )
    openai = respx.post(OPENAI_URL)

    response = await gateway_client.post(
        "/v1/embeddings", json={"model": EMBED_ALIAS, "input": "x"}
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_invalid_response"
    assert not openai.called


# 200이지만 유효한 임베딩 응답이 아닌 경우 — 빈 성공 목록이나 파서 예외의 500이 아니라
# 비밀 없는 OpenAI 규격 502로 반환한다. 임베딩은 어떤 경우에도 다른 모델·OpenAI로 우회하지 않는다.
@pytest.mark.parametrize(
    "native_body_bytes",
    [
        b"not json at all",
        b'{"model":"snowflake-arctic-embed2"}',
        b'{"embeddings":[[0.1]]}',
        b'{"model":"different-model","embeddings":[[0.1]]}',
        b'{"model":"snowflake-arctic-embed2","embeddings":"nope"}',
        b'{"model":"snowflake-arctic-embed2","embeddings":[]}',
        b'{"model":"snowflake-arctic-embed2","embeddings":[[]]}',
        b'{"model":"snowflake-arctic-embed2","embeddings":[["a","b"]]}',
        b'{"model":"snowflake-arctic-embed2","embeddings":[[true,false]]}',
        b'{"model":"snowflake-arctic-embed2","embeddings":[[NaN]]}',
        b'{"model":"snowflake-arctic-embed2","embeddings":[[Infinity]]}',
        b'{"model":"snowflake-arctic-embed2","embeddings":[[1e400]]}',
    ],
    ids=[
        "invalid-json",
        "missing-embeddings",
        "missing-model",
        "wrong-model",
        "embeddings-not-list",
        "wrong-count",
        "empty-vector",
        "non-numeric-vector",
        "boolean-vector",
        "nan-vector",
        "infinity-vector",
        "overflow-number",
    ],
)
@respx.mock
async def test_embed_invalid_upstream_200_returns_openai_shaped_502(
    gateway_client: httpx.AsyncClient, native_body_bytes: bytes
) -> None:
    respx.post(UPSTREAM_URL).respond(
        200,
        content=native_body_bytes,
        headers={"content-type": "application/json"},
    )
    openai = respx.post(OPENAI_URL)

    response = await gateway_client.post(
        "/v1/embeddings", json={"model": EMBED_ALIAS, "input": "x"}
    )

    assert response.status_code == 502
    error = response.json()["error"]
    assert error["type"] == "upstream_error"
    assert error["code"] == "upstream_invalid_response"
    assert not openai.called


@pytest.mark.parametrize(
    "prompt_eval_count",
    [
        b"true",
        b"-1",
        b"1.5",
        b'"1"',
        b"{}",
        b"[]",
        b"NaN",
        b"Infinity",
        b"-Infinity",
    ],
    ids=[
        "boolean",
        "negative",
        "float",
        "string",
        "object",
        "array",
        "nan",
        "infinity",
        "negative-infinity",
    ],
)
@respx.mock
async def test_embed_invalid_prompt_eval_count_returns_502_without_openai(
    gateway_client: httpx.AsyncClient, prompt_eval_count: bytes
) -> None:
    body = (
        b'{"model":"snowflake-arctic-embed2","embeddings":[[0.1]],'
        b'"prompt_eval_count":' + prompt_eval_count + b"}"
    )
    respx.post(UPSTREAM_URL).respond(
        200, content=body, headers={"content-type": "application/json"}
    )
    openai = respx.post(OPENAI_URL)

    response = await gateway_client.post(
        "/v1/embeddings", json={"model": EMBED_ALIAS, "input": "x"}
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_invalid_response"
    assert not openai.called


@respx.mock
async def test_embed_vector_count_mismatch_returns_openai_shaped_502(
    gateway_client: httpx.AsyncClient,
) -> None:
    # 입력 2개인데 벡터 1개 — 개수 불일치는 검색 정합성을 깨는 무효 응답이다.
    respx.post(UPSTREAM_URL).respond(200, json=_native_response([[0.1, 0.2]]))

    response = await gateway_client.post(
        "/v1/embeddings", json={"model": EMBED_ALIAS, "input": ["a", "b"]}
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_invalid_response"


@respx.mock
async def test_embed_mixed_vector_dimensions_return_502_without_openai(
    gateway_client: httpx.AsyncClient,
) -> None:
    respx.post(UPSTREAM_URL).respond(200, json=_native_response([[0.1, 0.2], [0.3]]))
    openai = respx.post(OPENAI_URL)

    response = await gateway_client.post(
        "/v1/embeddings", json={"model": EMBED_ALIAS, "input": ["a", "b"]}
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_invalid_response"
    assert not openai.called


# float32로 인코딩할 수 없는 값 — float64로는 유한한 큰 수(1e39)나 float 변환이 넘치는 큰 정수는
# 처리되지 않은 500이 아니라 비밀 없는 OpenAI 규격 502로 반환해야 한다. float·base64 요청 모두 동일하다.
_HUGE_INT_BODY = (
    b'{"model":"snowflake-arctic-embed2","embeddings": [[' + b"1" + b"0" * 400 + b"]]}"
)


@pytest.mark.parametrize(
    ("encoding_format", "native_body_bytes"),
    [
        (
            "float",
            b'{"model":"snowflake-arctic-embed2","embeddings":[[1e39]]}',
        ),
        (
            "base64",
            b'{"model":"snowflake-arctic-embed2","embeddings":[[1e39]]}',
        ),
        ("float", _HUGE_INT_BODY),
        ("base64", _HUGE_INT_BODY),
    ],
    ids=[
        "float32-overflow-float",
        "float32-overflow-base64",
        "huge-int-float",
        "huge-int-base64",
    ],
)
@respx.mock
async def test_embed_out_of_float32_range_returns_openai_shaped_502(
    gateway_client: httpx.AsyncClient,
    encoding_format: str,
    native_body_bytes: bytes,
) -> None:
    respx.post(UPSTREAM_URL).respond(
        200,
        content=native_body_bytes,
        headers={"content-type": "application/json"},
    )

    response = await gateway_client.post(
        "/v1/embeddings",
        json={"model": EMBED_ALIAS, "input": "x", "encoding_format": encoding_format},
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_invalid_response"


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
    [
        b'{"model": broken',
        b"[1, 2]",
        b'"just a string"',
        b"\xff\xfe",
        b'{"model":"embed","input":"x","value":NaN}',
        b'{"model":"embed","input":"x","value":Infinity}',
        b'{"model":"embed","input":"x","value":-Infinity}',
        b'{"model":"embed","input":"x","value":1e400}',
    ],
    ids=[
        "malformed-json",
        "json-array",
        "json-string",
        "invalid-utf8",
        "nan-constant",
        "infinity-constant",
        "negative-infinity-constant",
        "overflow-number",
    ],
)
@respx.mock
async def test_embed_invalid_json_gets_400_without_upstream_call(
    gateway_client: httpx.AsyncClient, broken_body: bytes
) -> None:
    route = respx.post(UPSTREAM_URL)
    openai = respx.post(OPENAI_URL)

    response = await gateway_client.post(
        "/v1/embeddings",
        content=broken_body,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert not route.called
    assert not openai.called
