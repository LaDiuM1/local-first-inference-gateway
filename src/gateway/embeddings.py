"""임베딩 엔드포인트 — `embed` 별칭을 실제 모델로 치환해 CPU 전용 Ollama에서 실행한다.

Ollama의 OpenAI 호환 `/v1/embeddings`는 실행 옵션을 받지 못해 CPU 상주를 보장할 수 없다.
따라서 native `/api/embed`에 `options.num_gpu: 0`을 실어 CPU 실행을 강제하고, 그 응답을
OpenAI embeddings 형식으로 변환해 반환한다. 응답 벡터는 클라이언트가 요청한 `encoding_format`
(기본 float, base64는 float32 리틀엔디언 바이트의 base64)에 맞춰 인코딩한다.

임베딩은 chat·vision과 분리된 임베딩 전용 Ollama 인스턴스만 호출한다. 모델이 바뀌면 벡터 공간이
달라져 검색 정합성이 깨지므로 다른 모델로 폴백하지 않는다 — 연결 실패·비정상 응답은 OpenAI 규격
오류로 반환하고 OpenAI fallback client는 호출하지 않는다.
"""

import base64
import math
import struct
from enum import StrEnum

import httpx
from fastapi import Response
from fastapi.responses import JSONResponse

from gateway.errors import (
    InvalidRequestError,
    invalid_request_response,
    upstream_invalid_response,
    upstream_unavailable_response,
)
from gateway.routing import EndpointKind, RoutingTable
from gateway.validation import load_json_object, load_standard_json, require_string

NATIVE_EMBED_PATH = "/api/embed"
CPU_ONLY_OPTIONS = {"num_gpu": 0}


class EncodingFormat(StrEnum):
    """OpenAI 임베딩 응답 벡터의 표현 형식. 지원하지 않는 값은 400으로 거절한다."""

    float = "float"
    base64 = "base64"


async def create_embeddings(
    embedding_client: httpx.AsyncClient, routing: RoutingTable, body: bytes
) -> Response:
    """입력을 검증하고 별칭을 실제 모델로 치환한 뒤 native /api/embed로 CPU 임베딩을 요청한다."""
    try:
        payload = load_json_object(body)
        alias = require_string(payload, "model")
        model = routing.resolve(EndpointKind.embeddings, alias)
        embedding_input = _require_embedding_input(payload)
        encoding_format = _require_encoding_format(payload)
    except InvalidRequestError as error:
        return invalid_request_response(error.message)

    native_request = {
        "model": model,
        "input": embedding_input,
        "options": CPU_ONLY_OPTIONS,
    }
    try:
        upstream_response = await embedding_client.post(
            NATIVE_EMBED_PATH, json=native_request
        )
    except httpx.RequestError as error:
        return upstream_unavailable_response(error)

    if 200 < upstream_response.status_code < 300:
        return upstream_invalid_response()
    if upstream_response.status_code != 200:
        return _upstream_error_response(upstream_response)

    # 200이어도 본문이 유효한 임베딩 응답임을 확인한 뒤에 변환한다 — 파싱 실패나 벡터 개수·형식
    # 불일치는 빈 성공 목록이나 500이 아니라 비밀 없는 OpenAI 규격 502로 돌려준다.
    expected_count = _expected_vector_count(embedding_input)
    try:
        native_body = load_standard_json(upstream_response.content)
    except ValueError:
        return upstream_invalid_response()
    if not _is_valid_embedding_body(native_body, expected_count, model):
        return upstream_invalid_response()
    return _openai_embeddings_response(native_body, model, encoding_format)


def _expected_vector_count(embedding_input: str | list[str]) -> int:
    if isinstance(embedding_input, str):
        return 1
    return len(embedding_input)


def _is_valid_embedding_body(
    native_body: object, expected_count: int, expected_model: str
) -> bool:
    if not isinstance(native_body, dict):
        return False
    if native_body.get("model") != expected_model:
        return False
    vectors = native_body.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != expected_count:
        return False
    if "prompt_eval_count" in native_body and not _is_valid_prompt_eval_count(
        native_body["prompt_eval_count"]
    ):
        return False
    dimensions: int | None = None
    for vector in vectors:
        if not _is_valid_vector(vector):
            return False
        if dimensions is None:
            dimensions = len(vector)
        elif len(vector) != dimensions:
            return False
    return True


def _is_valid_prompt_eval_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_valid_vector(vector: object) -> bool:
    if not isinstance(vector, list) or not vector:
        return False
    return all(_is_valid_embedding_value(value) for value in vector)


def _is_valid_embedding_value(value: object) -> bool:
    # bool은 int의 하위형이라 명시적으로 배제한다. 임베딩 벡터는 float32로 인코딩되므로, NaN·무한대는
    # 물론 float 변환이 넘치는 큰 정수나 float32 범위를 벗어나는 값도 유효한 임베딩 값이 아니다 —
    # 이들을 통과시키면 base64 인코딩(float32 pack)에서 처리되지 않은 500으로 새어 나간다.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        as_float = float(value)
    except OverflowError:
        return False
    if not math.isfinite(as_float):
        return False
    return _fits_float32(as_float)


def _fits_float32(value: float) -> bool:
    try:
        struct.pack("<f", value)
    except OverflowError:
        return False
    return True


def _require_embedding_input(payload: dict) -> str | list[str]:
    value = payload.get("input")
    if isinstance(value, str) and value:
        return value
    if (
        isinstance(value, list)
        and value
        and all(isinstance(item, str) for item in value)
    ):
        return value
    raise InvalidRequestError("'input' must be a non-empty string or array of strings")


def _require_encoding_format(payload: dict) -> EncodingFormat:
    # 생략은 OpenAI 기본값 float으로 처리하고, 지정된 값은 지원 형식만 통과시킨다.
    if "encoding_format" not in payload:
        return EncodingFormat.float
    try:
        return EncodingFormat(payload["encoding_format"])
    except ValueError:
        raise InvalidRequestError(
            "'encoding_format' must be 'float' or 'base64'"
        ) from None


def _openai_embeddings_response(
    native_body: dict, model: str, encoding_format: EncodingFormat
) -> JSONResponse:
    vectors = native_body.get("embeddings", [])
    data = [
        {
            "object": "embedding",
            "index": index,
            "embedding": _encode_embedding(vector, encoding_format),
        }
        for index, vector in enumerate(vectors)
    ]
    prompt_tokens = native_body.get("prompt_eval_count", 0)
    return JSONResponse(
        content={
            "object": "list",
            "data": data,
            "model": model,
            "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
        }
    )


def _encode_embedding(
    vector: list[float], encoding_format: EncodingFormat
) -> str | list[float]:
    # base64는 OpenAI와 동일하게 float32 리틀엔디언 바이트를 인코딩한다 — SDK가 그대로 역디코딩한다.
    if encoding_format is EncodingFormat.base64:
        packed = struct.pack(f"<{len(vector)}f", *vector)
        return base64.b64encode(packed).decode("ascii")
    return vector


def _upstream_error_response(upstream_response: httpx.Response) -> JSONResponse:
    message = "upstream embedding request failed"
    try:
        native_body = load_standard_json(upstream_response.content)
    except ValueError:
        native_body = None
    if isinstance(native_body, dict) and isinstance(native_body.get("error"), str):
        message = native_body["error"]
    return JSONResponse(
        status_code=upstream_response.status_code,
        content={
            "error": {
                "message": message,
                "type": "upstream_error",
                "param": None,
                "code": None,
            }
        },
    )
