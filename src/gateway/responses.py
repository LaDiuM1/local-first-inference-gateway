"""검색 이미지 분석에 필요한 OpenAI Responses API 호환 경계.

`/v1/responses`는 검색 모듈이 쓰는 OpenAI 모델명 그대로의 호환 별칭(`gpt-5.4-nano`)만 받고,
요청을 Chat Completions 형식으로 변환해 내부 `vision` 별칭으로 중계한다 — 별도 라우트·회로
차단기를 만들지 않고 vision의 라우팅·회로 차단기·OpenAI 폴백 정책을 그대로 공유한다.

변환 경계라 투명 중계가 성립하지 않으므로, 지원 목록에 없는 최상위 필드는 조용히 버리는 대신
업스트림 호출 전에 400으로 거절한다 — 적용되지 않은 생성 파라미터를 적용됐다고 믿는 사고를 막는다.

로컬이 2xx를 주더라도 변환 가능한 출력(비어 있지 않은 텍스트 + 지원하는 종료 사유)이 없으면
로컬 실패로 판정해 회로 차단기에 기록하고 폴백하며, 폴백 응답까지 무효면 502로 합성한다.
`finish_reason`이 stop이면 completed로, length·content_filter면 Responses 규격의 incomplete로
변환하고, 지원하지 않는 종료 사유를 임의로 completed로 만들지 않는다.
"""

import json
from dataclasses import dataclass
from time import time
from uuid import uuid4

import httpx
from fastapi import Response
from fastapi.responses import JSONResponse

from gateway.circuit_breaker import CircuitBreaker
from gateway.deadline import ResponseStartDeadline
from gateway.errors import (
    InvalidRequestError,
    invalid_request_response,
    upstream_invalid_response,
)
from gateway.openai_fallback import OpenAIFallback
from gateway.relay import relay_chat_completions
from gateway.relay_common import is_success_status
from gateway.routing import RoutingTable
from gateway.validation import load_json_object, load_standard_json, require_boolean

# 검색 모듈 호환 별칭 — 이 엔드포인트가 받는 유일한 model 값이고, 내부에서는 vision으로 정규화된다.
RESPONSES_MODEL_ALIAS = "gpt-5.4-nano"
VISION_ALIAS = "vision"

# 요청 최상위 계약 — 여기 없는 필드는 변환에서 조용히 탈락하므로 받지 않고 400으로 거절한다.
SUPPORTED_REQUEST_FIELDS = frozenset({"model", "instructions", "input", "stream"})

# Chat finish_reason → Responses incomplete_details.reason. stop만 completed가 되고,
# 이 표에 없는 종료 사유는 변환 대상이 아니다.
_INCOMPLETE_REASONS = {
    "length": "max_output_tokens",
    "content_filter": "content_filter",
}


async def create_response(
    local_client: httpx.AsyncClient,
    fallback: OpenAIFallback,
    breakers: dict[str, CircuitBreaker],
    routing: RoutingTable,
    body: bytes,
    deadline: ResponseStartDeadline,
) -> Response:
    try:
        payload = load_json_object(body)
        requested_model = _required_string(payload, "model")
        if requested_model != RESPONSES_MODEL_ALIAS:
            raise InvalidRequestError(f"unknown model alias '{requested_model}'")
        unsupported_fields = sorted(payload.keys() - SUPPORTED_REQUEST_FIELDS)
        if unsupported_fields:
            raise InvalidRequestError(
                f"unsupported fields: {', '.join(unsupported_fields)}"
            )
        if require_boolean(payload, "stream", default=False):
            raise InvalidRequestError("streaming Responses are not supported")
        messages = _chat_messages(payload)
    except InvalidRequestError as error:
        return invalid_request_response(error.message)

    chat_body = json.dumps(
        {"model": VISION_ALIAS, "messages": messages, "stream": False},
        allow_nan=False,
    ).encode("utf-8")
    chat_response = await relay_chat_completions(
        local_client,
        fallback,
        breakers,
        routing,
        chat_body,
        deadline,
        body_validator=_is_convertible_chat_body,
    )
    if not is_success_status(chat_response.status_code):
        return chat_response
    return _responses_success(
        chat_response, requested_model, payload.get("instructions")
    )


def _chat_messages(payload: dict) -> list[dict]:
    messages: list[dict] = []
    instructions = payload.get("instructions")
    if instructions is not None:
        messages.append(
            {"role": "system", "content": _required_string(payload, "instructions")}
        )

    response_input = payload.get("input")
    if isinstance(response_input, str) and response_input:
        messages.append({"role": "user", "content": response_input})
        return messages
    if not isinstance(response_input, list) or not response_input:
        raise InvalidRequestError("'input' must be a non-empty string or array")
    messages.extend(_chat_message(item) for item in response_input)
    return messages


def _chat_message(item: object) -> dict:
    if not isinstance(item, dict):
        raise InvalidRequestError("each 'input' item must be an object")
    role = _required_string(item, "role")
    if role == "developer":
        role = "system"
    if role not in {"system", "user", "assistant"}:
        raise InvalidRequestError("input message role is not supported")

    content = item.get("content")
    if isinstance(content, str) and content:
        return {"role": role, "content": content}
    if not isinstance(content, list) or not content:
        raise InvalidRequestError("input message content must be non-empty")
    return {"role": role, "content": [_chat_content(part) for part in content]}


def _chat_content(part: object) -> dict:
    if not isinstance(part, dict):
        raise InvalidRequestError("each input content item must be an object")
    content_type = _required_string(part, "type")
    if content_type == "input_text":
        return {"type": "text", "text": _required_string(part, "text")}
    if content_type != "input_image":
        raise InvalidRequestError(f"unsupported input content type '{content_type}'")

    image_url: dict[str, str] = {"url": _required_string(part, "image_url")}
    if "detail" in part:
        detail = _required_string(part, "detail")
        # Chat Completions의 detail 집합과 같아야 로컬·OpenAI 폴백 어느 쪽으로도 그대로 전달된다.
        if detail not in {"auto", "low", "high"}:
            raise InvalidRequestError("input image detail is not supported")
        image_url["detail"] = detail
    return {"type": "image_url", "image_url": image_url}


def _required_string(payload: dict, field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise InvalidRequestError(f"'{field}' must be a non-empty string")
    return value


@dataclass(frozen=True)
class _ChatOutput:
    """Chat 응답에서 뽑은 변환 재료 — incomplete_reason이 None이면 정상 완료다."""

    text: str
    incomplete_reason: str | None


def _is_convertible_chat_body(body: bytes) -> bool:
    """2xx 본문이 Responses로 변환 가능한 Chat 응답인지 본다 — 아니면 폴백/502 대상이다."""
    try:
        chat_body = load_standard_json(body)
    except ValueError:
        return False
    return _chat_output(chat_body) is not None


def _chat_output(chat_body: object) -> _ChatOutput | None:
    if not isinstance(chat_body, dict):
        return None
    choices = chat_body.get("choices")
    if not isinstance(choices, list):
        return None
    for choice in choices:
        output = _choice_output(choice)
        if output is not None:
            return output
    return None


def _choice_output(choice: object) -> _ChatOutput | None:
    if not isinstance(choice, dict):
        return None
    message = choice.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if not isinstance(content, str) or not content:
        return None

    finish_reason = choice.get("finish_reason")
    if finish_reason == "stop":
        return _ChatOutput(text=content, incomplete_reason=None)
    if isinstance(finish_reason, str) and finish_reason in _INCOMPLETE_REASONS:
        return _ChatOutput(
            text=content, incomplete_reason=_INCOMPLETE_REASONS[finish_reason]
        )
    return None


def _responses_success(
    chat_response: Response, requested_model: str, instructions: object
) -> Response:
    try:
        chat_body = load_standard_json(chat_response.body)
    except ValueError:
        return upstream_invalid_response("inference")
    output = _chat_output(chat_body)
    if output is None:
        return upstream_invalid_response("inference")

    created_at = int(time())
    if isinstance(chat_body, dict):
        candidate = chat_body.get("created")
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            created_at = candidate

    status = "completed"
    completed_at: int | None = created_at
    incomplete_details: dict[str, str] | None = None
    if output.incomplete_reason is not None:
        status = "incomplete"
        completed_at = None
        incomplete_details = {"reason": output.incomplete_reason}

    response_id = f"resp_{uuid4().hex}"
    message_id = f"msg_{uuid4().hex}"
    return JSONResponse(
        content={
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "status": status,
            "completed_at": completed_at,
            "error": None,
            "incomplete_details": incomplete_details,
            "instructions": instructions,
            "model": requested_model,
            "output": [
                {
                    "id": message_id,
                    "type": "message",
                    "status": status,
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": output.text, "annotations": []}
                    ],
                }
            ],
            "usage": _responses_usage(chat_body),
        }
    )


def _responses_usage(chat_body: object) -> dict[str, object]:
    usage = chat_body.get("usage") if isinstance(chat_body, dict) else None
    if not isinstance(usage, dict):
        usage = {}
    input_tokens = _token_count(usage.get("prompt_tokens"))
    output_tokens = _token_count(usage.get("completion_tokens"))
    total_tokens = _token_count(usage.get("total_tokens"))
    if total_tokens == 0:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {
            "cached_tokens": _detail_token_count(
                usage.get("prompt_tokens_details"), "cached_tokens"
            )
        },
        "output_tokens": output_tokens,
        "output_tokens_details": {
            "reasoning_tokens": _detail_token_count(
                usage.get("completion_tokens_details"), "reasoning_tokens"
            )
        },
        "total_tokens": total_tokens,
    }


def _detail_token_count(details: object, field: str) -> int:
    if not isinstance(details, dict):
        return 0
    return _token_count(details.get(field))


def _token_count(value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0
