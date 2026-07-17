"""임베딩 엔드포인트 계약 테스트 — native /api/embed(CPU 강제)를 respx로 목킹한다."""

import base64
import json
import struct

import httpx
import pytest
import respx

from gateway.config import settings
from gateway.embeddings import (
    ALIAS_VECTOR_DIMENSIONS,
    NATIVE_EMBEDDING_VECTOR_DIMENSIONS,
    OPENAI_COMPAT_VECTOR_DIMENSIONS,
)

pytestmark = pytest.mark.anyio

# 임베딩은 chat·vision과 분리된 임베딩 전용 인스턴스(별도 base URL)만 호출한다.
UPSTREAM_URL = f"{settings.embedding_ollama_base_url}/api/embed"
OPENAI_URL = f"{settings.openai_base_url}/chat/completions"
EMBED_ALIAS = "embed"
OPENAI_EMBED_MODEL = "text-embedding-3-small"
EMBED_MODEL = "snowflake-arctic-embed2"


def _vector(start: float = 0.0) -> list[float]:
    """로컬 모델의 1024차원 벡터 — 성분마다 값이 달라 순서 보존도 함께 확인한다."""
    return [
        round(start + index / NATIVE_EMBEDDING_VECTOR_DIMENSIONS, 6)
        for index in range(NATIVE_EMBEDDING_VECTOR_DIMENSIONS)
    ]


def _padded_vector(vector: list[float]) -> list[float]:
    """OpenAI 호환 별칭의 공개 벡터 — 1024차원 뒤에 0을 붙여 1536차원으로 만든다."""
    padding = OPENAI_COMPAT_VECTOR_DIMENSIONS - NATIVE_EMBEDDING_VECTOR_DIMENSIONS
    return [*vector, *([0.0] * padding)]


def _native_response(vectors: list[list[float]], prompt_tokens: int = 3) -> dict:
    return {
        "model": EMBED_MODEL,
        "embeddings": vectors,
        "prompt_eval_count": prompt_tokens,
    }


def _native_body_with_value(raw_value: bytes) -> bytes:
    """마지막 성분만 raw_value로 바꾼 계약 차원 벡터 하나짜리 native 응답 바이트.

    차원은 계약대로 맞춰 값 검증만 남긴다 — 짧은 벡터로 만들면 차원 검사에서 먼저 걸려 값 검증이
    실행되지 않는다.
    """
    components = [b"0.1"] * (NATIVE_EMBEDDING_VECTOR_DIMENSIONS - 1) + [raw_value]
    return (
        b'{"model":"snowflake-arctic-embed2","embeddings":[['
        + b",".join(components)
        + b"]]}"
    )


def _decode_base64(encoded: str, dimensions: int) -> list[float]:
    # OpenAI SDK와 동일한 방식으로 base64를 float32 리틀엔디언 벡터로 되돌린다.
    return list(struct.unpack(f"<{dimensions}f", base64.b64decode(encoded)))


def test_alias_dimension_contract() -> None:
    """별칭별 공개 차원 계약 — embed는 native 1024, OpenAI 호환 별칭은 1536이다."""
    assert NATIVE_EMBEDDING_VECTOR_DIMENSIONS == 1024
    assert OPENAI_COMPAT_VECTOR_DIMENSIONS == 1536
    assert ALIAS_VECTOR_DIMENSIONS == {
        EMBED_ALIAS: 1024,
        OPENAI_EMBED_MODEL: 1536,
    }


@respx.mock
async def test_embed_single_string_converts_to_openai_format(
    gateway_client: httpx.AsyncClient,
) -> None:
    vector = _vector()
    route = respx.post(UPSTREAM_URL).respond(200, json=_native_response([vector]))

    response = await gateway_client.post(
        "/v1/embeddings", json={"model": EMBED_ALIAS, "input": "상품 설명"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["model"] == EMBED_ALIAS
    assert body["data"] == [{"object": "embedding", "index": 0, "embedding": vector}]
    assert body["usage"] == {"prompt_tokens": 3, "total_tokens": 3}

    native_request = json.loads(route.calls.last.request.content)
    assert native_request["model"] == EMBED_MODEL
    assert native_request["input"] == "상품 설명"
    assert native_request["options"] == {"num_gpu": 0}


@respx.mock
async def test_openai_model_name_is_a_1536_dimension_compatibility_alias(
    gateway_client: httpx.AsyncClient,
) -> None:
    native_vector = _vector()
    route = respx.post(UPSTREAM_URL).respond(
        200, json=_native_response([native_vector])
    )

    response = await gateway_client.post(
        "/v1/embeddings",
        json={"model": OPENAI_EMBED_MODEL, "input": "상품 설명"},
    )

    assert response.status_code == 200
    body = response.json()
    embedding = body["data"][0]["embedding"]
    assert body["model"] == OPENAI_EMBED_MODEL
    assert embedding[:NATIVE_EMBEDDING_VECTOR_DIMENSIONS] == native_vector
    assert embedding[NATIVE_EMBEDDING_VECTOR_DIMENSIONS:] == [0.0] * 512
    assert len(embedding) == OPENAI_COMPAT_VECTOR_DIMENSIONS
    # 호환 별칭도 내부 embed 라우트로 정규화돼 같은 모델을 호출한다 — 벡터 공간은 하나다.
    assert json.loads(route.calls.last.request.content)["model"] == EMBED_MODEL


@respx.mock
async def test_openai_alias_batch_preserves_order_with_padding(
    gateway_client: httpx.AsyncClient,
) -> None:
    vectors = [_vector(0.1), _vector(0.3)]
    respx.post(UPSTREAM_URL).respond(200, json=_native_response(vectors))

    response = await gateway_client.post(
        "/v1/embeddings", json={"model": OPENAI_EMBED_MODEL, "input": ["a", "b"]}
    )

    data = response.json()["data"]
    assert [item["index"] for item in data] == [0, 1]
    assert [item["embedding"] for item in data] == [
        _padded_vector(vector) for vector in vectors
    ]


@respx.mock
async def test_openai_alias_base64_returns_decodable_1536_float32(
    gateway_client: httpx.AsyncClient,
) -> None:
    vector = _vector()
    respx.post(UPSTREAM_URL).respond(200, json=_native_response([vector]))

    response = await gateway_client.post(
        "/v1/embeddings",
        json={"model": OPENAI_EMBED_MODEL, "input": "x", "encoding_format": "base64"},
    )

    encoded = response.json()["data"][0]["embedding"]
    assert isinstance(encoded, str)
    assert _decode_base64(encoded, OPENAI_COMPAT_VECTOR_DIMENSIONS) == pytest.approx(
        _padded_vector(vector)
    )


@respx.mock
async def test_explicit_1536_dimensions_is_accepted(
    gateway_client: httpx.AsyncClient,
) -> None:
    route = respx.post(UPSTREAM_URL).respond(200, json=_native_response([_vector()]))

    response = await gateway_client.post(
        "/v1/embeddings",
        json={"model": OPENAI_EMBED_MODEL, "input": "x", "dimensions": 1536},
    )

    assert response.status_code == 200
    assert len(response.json()["data"][0]["embedding"]) == 1536
    assert "dimensions" not in json.loads(route.calls.last.request.content)


@respx.mock
async def test_embed_explicit_1024_dimensions_is_accepted(
    gateway_client: httpx.AsyncClient,
) -> None:
    respx.post(UPSTREAM_URL).respond(200, json=_native_response([_vector()]))

    response = await gateway_client.post(
        "/v1/embeddings",
        json={"model": EMBED_ALIAS, "input": "x", "dimensions": 1024},
    )

    assert response.status_code == 200
    assert len(response.json()["data"][0]["embedding"]) == 1024


@respx.mock
async def test_embed_array_preserves_order_and_index(
    gateway_client: httpx.AsyncClient,
) -> None:
    vectors = [_vector(0.1), _vector(0.3), _vector(0.5)]
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
    route = respx.post(UPSTREAM_URL).respond(200, json=_native_response([_vector()]))

    await gateway_client.post(
        "/v1/embeddings", json={"model": EMBED_ALIAS, "input": "x"}
    )

    native_request = json.loads(route.calls.last.request.content)
    assert native_request["options"]["num_gpu"] == 0


@respx.mock
async def test_embed_explicit_float_returns_float_list(
    gateway_client: httpx.AsyncClient,
) -> None:
    vector = _vector()
    route = respx.post(UPSTREAM_URL).respond(200, json=_native_response([vector]))

    response = await gateway_client.post(
        "/v1/embeddings",
        json={"model": EMBED_ALIAS, "input": "x", "encoding_format": "float"},
    )

    assert response.json()["data"][0]["embedding"] == vector
    native_request = json.loads(route.calls.last.request.content)
    assert "encoding_format" not in native_request
    assert native_request["options"] == {"num_gpu": 0}


@respx.mock
async def test_embed_base64_returns_decodable_float32(
    gateway_client: httpx.AsyncClient,
) -> None:
    vector = _vector()
    respx.post(UPSTREAM_URL).respond(200, json=_native_response([vector]))

    response = await gateway_client.post(
        "/v1/embeddings",
        json={"model": EMBED_ALIAS, "input": "x", "encoding_format": "base64"},
    )

    encoded = response.json()["data"][0]["embedding"]
    assert isinstance(encoded, str)
    assert _decode_base64(encoded, NATIVE_EMBEDDING_VECTOR_DIMENSIONS) == pytest.approx(
        vector
    )


@respx.mock
async def test_embed_base64_array_preserves_order_and_cpu_option(
    gateway_client: httpx.AsyncClient,
) -> None:
    vectors = [_vector(0.1), _vector(0.3)]
    route = respx.post(UPSTREAM_URL).respond(200, json=_native_response(vectors))

    response = await gateway_client.post(
        "/v1/embeddings",
        json={"model": EMBED_ALIAS, "input": ["a", "b"], "encoding_format": "base64"},
    )

    data = response.json()["data"]
    assert [item["index"] for item in data] == [0, 1]
    for item, vector in zip(data, vectors, strict=True):
        assert _decode_base64(
            item["embedding"], NATIVE_EMBEDDING_VECTOR_DIMENSIONS
        ) == pytest.approx(vector)
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


# dimensions는 별칭의 고정 차원만 허용한다 — 호환 별칭에 1024를, embed에 1536을 요청해도 거절된다.
@pytest.mark.parametrize(
    ("model", "dimensions"),
    [
        (OPENAI_EMBED_MODEL, 1024),
        (OPENAI_EMBED_MODEL, 1535),
        (OPENAI_EMBED_MODEL, 1537),
        (OPENAI_EMBED_MODEL, True),
        (OPENAI_EMBED_MODEL, "1536"),
        (EMBED_ALIAS, 1536),
        (EMBED_ALIAS, 512),
        (EMBED_ALIAS, "1024"),
    ],
)
@respx.mock
async def test_embed_unsupported_dimensions_gets_400_without_upstream_call(
    gateway_client: httpx.AsyncClient, model: str, dimensions: object
) -> None:
    route = respx.post(UPSTREAM_URL)

    response = await gateway_client.post(
        "/v1/embeddings",
        json={"model": model, "input": "x", "dimensions": dimensions},
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert not route.called


@pytest.mark.parametrize(
    "model",
    ["chat", "vision", "snowflake-arctic-embed2", "text-embedding-3-large"],
    ids=["chat-alias", "vision-alias", "real-model-name", "unregistered-openai-name"],
)
@respx.mock
async def test_embeddings_rejects_non_embedding_aliases_without_upstream_call(
    gateway_client: httpx.AsyncClient, model: str
) -> None:
    route = respx.post(UPSTREAM_URL)

    response = await gateway_client.post(
        "/v1/embeddings", json={"model": model, "input": "x"}
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


@pytest.mark.parametrize("status_code", [201, 202, 204, 206, 299, 302, 601])
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
        b'{"model":"snowflake-arctic-embed2","embeddings":[0.1]}',
    ],
    ids=[
        "invalid-json",
        "missing-embeddings",
        "missing-model",
        "wrong-model",
        "embeddings-not-list",
        "wrong-count",
        "empty-vector",
        "vector-not-list",
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
    vector = json.dumps(_vector()).encode()
    body = (
        b'{"model":"snowflake-arctic-embed2","embeddings":[' + vector + b"],"
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
    respx.post(UPSTREAM_URL).respond(200, json=_native_response([_vector()]))

    response = await gateway_client.post(
        "/v1/embeddings", json={"model": EMBED_ALIAS, "input": ["a", "b"]}
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_invalid_response"


# 벡터 차원은 `embed` 외부 계약이다. 기존 색인과 벡터 공간을 공유하지 못하는 응답은 벡터끼리
# 차원이 같더라도 변환하지 않는다 — float·base64 어느 형식으로도 직렬화하기 전에 502로 끊는다.
@pytest.mark.parametrize("encoding_format", ["float", "base64"])
@pytest.mark.parametrize(
    "dimensions",
    [
        NATIVE_EMBEDDING_VECTOR_DIMENSIONS - 1,
        NATIVE_EMBEDDING_VECTOR_DIMENSIONS + 1,
    ],
    ids=["one-short", "one-long"],
)
@respx.mock
async def test_embed_wrong_vector_dimensions_return_502_without_openai(
    gateway_client: httpx.AsyncClient, dimensions: int, encoding_format: str
) -> None:
    respx.post(UPSTREAM_URL).respond(200, json=_native_response([[0.1] * dimensions]))
    openai = respx.post(OPENAI_URL)

    response = await gateway_client.post(
        "/v1/embeddings",
        json={"model": EMBED_ALIAS, "input": "x", "encoding_format": encoding_format},
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_invalid_response"
    assert not openai.called


@respx.mock
async def test_embed_batch_with_consistently_wrong_dimensions_returns_502(
    gateway_client: httpx.AsyncClient,
) -> None:
    # 벡터끼리 차원이 같아도 계약 차원이 아니면 통과시키지 않는다.
    wrong = [0.1] * (NATIVE_EMBEDDING_VECTOR_DIMENSIONS // 2)
    respx.post(UPSTREAM_URL).respond(200, json=_native_response([wrong, wrong]))
    openai = respx.post(OPENAI_URL)

    response = await gateway_client.post(
        "/v1/embeddings", json={"model": EMBED_ALIAS, "input": ["a", "b"]}
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_invalid_response"
    assert not openai.called


@respx.mock
async def test_embed_mixed_vector_dimensions_return_502_without_openai(
    gateway_client: httpx.AsyncClient,
) -> None:
    respx.post(UPSTREAM_URL).respond(
        200,
        json=_native_response(
            [_vector(), [0.3] * (NATIVE_EMBEDDING_VECTOR_DIMENSIONS - 1)]
        ),
    )
    openai = respx.post(OPENAI_URL)

    response = await gateway_client.post(
        "/v1/embeddings", json={"model": EMBED_ALIAS, "input": ["a", "b"]}
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_invalid_response"
    assert not openai.called


# 계약 차원을 채웠어도 성분 하나가 유효한 임베딩 값이 아니면 무효다 — 비수치·부울·비유한 수는 물론
# float32로 인코딩할 수 없는 큰 수(1e39)나 float 변환이 넘치는 큰 정수도 처리되지 않은 500이 아니라
# 비밀 없는 OpenAI 규격 502로 반환해야 한다. float·base64 요청 모두 직렬화 전에 같은 검증을 거친다.
@pytest.mark.parametrize("encoding_format", ["float", "base64"])
@pytest.mark.parametrize(
    "raw_value",
    [
        b'"a"',
        b"true",
        b"NaN",
        b"Infinity",
        b"-Infinity",
        b"1e400",
        b"1e39",
        b"1" + b"0" * 400,
    ],
    ids=[
        "string",
        "boolean",
        "nan",
        "infinity",
        "negative-infinity",
        "overflow-number",
        "float32-overflow",
        "huge-int",
    ],
)
@respx.mock
async def test_embed_invalid_vector_value_returns_502_before_encoding(
    gateway_client: httpx.AsyncClient, raw_value: bytes, encoding_format: str
) -> None:
    respx.post(UPSTREAM_URL).respond(
        200,
        content=_native_body_with_value(raw_value),
        headers={"content-type": "application/json"},
    )
    openai = respx.post(OPENAI_URL)

    response = await gateway_client.post(
        "/v1/embeddings",
        json={"model": EMBED_ALIAS, "input": "x", "encoding_format": encoding_format},
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_invalid_response"
    assert not openai.called


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
