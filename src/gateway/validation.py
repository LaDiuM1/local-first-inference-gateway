"""요청·업스트림 응답에 같은 표준 JSON 정의를 적용하는 최소 검증 헬퍼."""

import json
import math

from gateway.errors import InvalidRequestError


def load_standard_json(body: bytes) -> object:
    """비표준 상수, 유한 범위를 넘는 수, UTF-8 인코딩 불가 문자열 없이 JSON 한 값을 읽는다.

    표준 JSON으로 읽을 수 없는 입력은 종류와 무관하게 ValueError 하나로 드러낸다. 과도한 중첩은
    파서 재귀 한도를 넘겨 RecursionError가 되므로 여기서 같은 계약 위반으로 통일한다.
    """
    try:
        parsed = json.loads(body, parse_constant=_reject_nonstandard_constant)
    except RecursionError as error:
        raise ValueError("JSON nesting is too deep") from error
    _reject_unrepresentable_values(parsed)
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


def _reject_unrepresentable_values(parsed: object) -> None:
    """`1e400`처럼 무한대가 되는 수와 UTF-8로 인코딩할 수 없는 문자열을 반복 없이 검사한다.

    JSON escape(`"\\ud800"`)로 들어온 고립 surrogate는 파싱은 통과하지만 응답·중계 직렬화의
    UTF-8 인코딩에서 터진다 — 키를 포함한 모든 문자열을 경계에서 거절한다.
    """
    pending = [parsed]
    while pending:
        value = pending.pop()
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("JSON number is outside the finite range")
        if isinstance(value, str):
            _require_utf8_encodable(value)
        elif isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)


def _require_utf8_encodable(value: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("JSON string is not encodable as UTF-8") from error
