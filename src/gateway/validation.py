"""요청 본문을 OpenAI 계약에 맞게 검증한다 — 두 엔드포인트가 공유하는 최소 규칙."""

import json

from gateway.errors import InvalidRequestError


def load_json_object(body: bytes) -> dict:
    # 잘못된 UTF-8 바이트는 JSONDecodeError가 아니라 UnicodeDecodeError로 떠오르므로 함께 잡는다.
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise InvalidRequestError("request body is not a valid JSON object") from error
    if not isinstance(payload, dict):
        raise InvalidRequestError("request body is not a JSON object")
    return payload


def require_string(payload: dict, field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise InvalidRequestError(f"'{field}' must be a non-empty string")
    return value
