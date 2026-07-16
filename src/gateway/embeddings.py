"""임베딩 엔드포인트 — 공개 별칭을 실제 모델로 치환해 CPU 전용 Ollama에서 실행한다.

Ollama의 OpenAI 호환 `/v1/embeddings`는 실행 옵션을 받지 못해 CPU 상주를 보장할 수 없다.
따라서 native `/api/embed`에 `options.num_gpu: 0`을 실어 CPU 실행을 강제하고, 그 응답을
OpenAI embeddings 형식으로 변환해 반환한다. 응답 벡터는 클라이언트가 요청한 `encoding_format`
(기본 float, base64는 float32 리틀엔디언 바이트의 base64)에 맞춰 인코딩한다.

공개 별칭은 둘뿐이고 모두 내부 `embed` 라우트의 같은 모델로 정규화된다 — 벡터 공간은 하나다.
`embed`는 로컬 모델의 1024차원을 그대로 반환하고, 검색 모듈 호환 별칭 `text-embedding-3-small`은
뒤에 0을 붙여 OpenAI와 같은 1536차원으로 반환한다(zero-padding은 코사인 유사도를 보존한다).

임베딩은 chat·vision과 분리된 임베딩 전용 Ollama 인스턴스만 호출한다. 모델이 바뀌면 벡터 공간이
달라져 검색 정합성이 깨지므로 다른 모델로 폴백하지 않는다 — 연결 실패·비정상 응답은 OpenAI 규격
오류로 반환하고 OpenAI fallback client는 호출하지 않는다.
"""

import base64
import math
import struct
from enum import StrEnum

import anyio
import httpx
from fastapi import Response
from fastapi.responses import JSONResponse

from gateway.deadline import ResponseStartDeadline
from gateway.errors import (
    InvalidRequestError,
    invalid_request_response,
    response_start_timeout_response,
    upstream_invalid_response,
    upstream_unavailable_response,
)
from gateway.observability import (
    observe_alias,
    observe_local_failure,
    observe_provider,
    observe_response_start,
    observe_stream,
    observe_upstream_start,
)
from gateway.routing import EndpointKind, RoutingTable
from gateway.validation import load_json_object, load_standard_json, require_string

NATIVE_EMBED_PATH = "/api/embed"
CPU_ONLY_OPTIONS = {"num_gpu": 0}
# snowflake-arctic-embed2의 원본 차원과 OpenAI text-embedding-3-small의 공개 차원.
NATIVE_EMBEDDING_VECTOR_DIMENSIONS = 1024
OPENAI_COMPAT_VECTOR_DIMENSIONS = 1536

# 공개 별칭 → 응답 차원. 두 별칭 모두 내부 embed 라우트로 정규화되므로 라우팅 필수 검증(embed)이
# 곧 두 별칭의 기동 시점 보장이고, 여기 없는 별칭은 업스트림 호출 없이 400으로 거절된다.
INTERNAL_EMBED_ALIAS = "embed"
ALIAS_VECTOR_DIMENSIONS: dict[str, int] = {
    INTERNAL_EMBED_ALIAS: NATIVE_EMBEDDING_VECTOR_DIMENSIONS,
    "text-embedding-3-small": OPENAI_COMPAT_VECTOR_DIMENSIONS,
}


class EncodingFormat(StrEnum):
    """OpenAI 임베딩 응답 벡터의 표현 형식. 지원하지 않는 값은 400으로 거절한다."""

    float = "float"
    base64 = "base64"


async def create_embeddings(
    embedding_client: httpx.AsyncClient,
    routing: RoutingTable,
    body: bytes,
    deadline: ResponseStartDeadline,
) -> Response:
    """입력을 검증하고 별칭을 실제 모델로 치환한 뒤 native /api/embed로 CPU 임베딩을 요청한다."""
    try:
        payload = load_json_object(body)
        requested_model = require_string(payload, "model")
        target_dimensions = _require_known_alias(requested_model)
        model = routing.resolve(EndpointKind.embeddings, INTERNAL_EMBED_ALIAS)
        embedding_input = _require_embedding_input(payload)
        encoding_format = _require_encoding_format(payload)
        _require_dimensions(payload, target_dimensions)
    except InvalidRequestError as error:
        return invalid_request_response(error.message)

    observe_alias(requested_model)
    observe_stream(streaming=False)
    native_request = {
        "model": model,
        "input": embedding_input,
        "options": CPU_ONLY_OPTIONS,
    }
    timeout_seconds = deadline.local_remaining_seconds()
    if timeout_seconds <= 0:
        observe_local_failure("local_deadline")
        return response_start_timeout_response()
    observe_upstream_start()
    try:
        with anyio.fail_after(timeout_seconds):
            upstream_response = await embedding_client.post(
                NATIVE_EMBED_PATH, json=native_request
            )
    except TimeoutError:
        observe_local_failure("local_deadline")
        return response_start_timeout_response()
    except httpx.RequestError as error:
        observe_local_failure("local_unreachable")
        return upstream_unavailable_response(error)

    if 200 < upstream_response.status_code < 300:
        observe_local_failure("local_invalid_body")
        return upstream_invalid_response("embedding")
    if upstream_response.status_code != 200:
        observe_provider("local")
        observe_response_start()
        return _upstream_error_response(upstream_response)

    # 200이어도 본문이 유효한 임베딩 응답임을 확인한 뒤에 변환한다 — 파싱 실패나 벡터 개수·차원·
    # 형식 불일치는 빈 성공 목록이나 500이 아니라 비밀 없는 OpenAI 규격 502로 돌려준다.
    expected_count = _expected_vector_count(embedding_input)
    try:
        native_body = load_standard_json(upstream_response.content)
    except ValueError:
        observe_local_failure("local_invalid_body")
        return upstream_invalid_response("embedding")
    if not _is_valid_embedding_body(native_body, expected_count, model):
        observe_local_failure("local_invalid_body")
        return upstream_invalid_response("embedding")
    observe_provider("local")
    observe_response_start()
    return _openai_embeddings_response(
        native_body, requested_model, encoding_format, target_dimensions
    )


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
    return all(_is_valid_vector(vector) for vector in vectors)


def _is_valid_prompt_eval_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_valid_vector(vector: object) -> bool:
    if (
        not isinstance(vector, list)
        or len(vector) != NATIVE_EMBEDDING_VECTOR_DIMENSIONS
    ):
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


def _require_known_alias(requested_model: str) -> int:
    target_dimensions = ALIAS_VECTOR_DIMENSIONS.get(requested_model)
    if target_dimensions is None:
        raise InvalidRequestError(f"unknown model alias '{requested_model}'")
    return target_dimensions


def _require_dimensions(payload: dict, target_dimensions: int) -> None:
    if "dimensions" not in payload:
        return
    dimensions = payload["dimensions"]
    if (
        isinstance(dimensions, bool)
        or not isinstance(dimensions, int)
        or dimensions != target_dimensions
    ):
        raise InvalidRequestError(
            f"'dimensions' must be {target_dimensions} when provided"
        )


def _openai_embeddings_response(
    native_body: dict,
    requested_model: str,
    encoding_format: EncodingFormat,
    target_dimensions: int,
) -> JSONResponse:
    vectors = native_body.get("embeddings", [])
    data = [
        {
            "object": "embedding",
            "index": index,
            "embedding": _encode_embedding(
                _pad_to_dimensions(vector, target_dimensions), encoding_format
            ),
        }
        for index, vector in enumerate(vectors)
    ]
    prompt_tokens = native_body.get("prompt_eval_count", 0)
    return JSONResponse(
        content={
            "object": "list",
            "data": data,
            "model": requested_model,
            "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
        }
    )


def _pad_to_dimensions(vector: list[float], target_dimensions: int) -> list[float]:
    padding = target_dimensions - NATIVE_EMBEDDING_VECTOR_DIMENSIONS
    return [*vector, *([0.0] * padding)]


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
