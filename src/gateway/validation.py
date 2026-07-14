"""요청·업스트림 응답에 같은 표준 JSON 정의를 적용하는 최소 검증 헬퍼."""

import json
import math

from gateway.errors import InvalidRequestError


def load_standard_json(body: bytes) -> object:
    """비표준 상수와 유한 범위를 넘는 수를 허용하지 않고 JSON 한 값을 읽는다."""
    parsed = json.loads(body, parse_constant=_reject_nonstandard_constant)
    _reject_non_finite_numbers(parsed)
    return parsed


def load_json_object(body: bytes) -> dict:
    # 잘못된 UTF-8과 비표준 상수도 요청 계약 위반으로 같은 400 응답에 매핑한다.
    try:
        payload = load_standard_json(body)
    except ValueError as error:
        raise InvalidRequestError("request body is not a valid JSON object") from error
    if not isinstance(payload, dict):
        raise InvalidRequestError("request body is not a JSON object")
    return payload


def require_string(payload: dict, field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise InvalidRequestError(f"'{field}' must be a non-empty string")
    return value


def require_boolean(payload: dict, field: str, *, default: bool) -> bool:
    if field not in payload:
        return default
    value = payload[field]
    if not isinstance(value, bool):
        raise InvalidRequestError(f"'{field}' must be a boolean")
    return value


def _reject_nonstandard_constant(constant: str) -> object:
    raise ValueError(f"non-standard JSON constant: {constant}")


def _reject_non_finite_numbers(parsed: object) -> None:
    """`1e400`처럼 파싱 결과가 무한대가 되는 표준 표기까지 반복 없이 검사한다."""
    pending = [parsed]
    while pending:
        value = pending.pop()
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("JSON number is outside the finite range")
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
