"""요청 검증 실패와 업스트림 오류를 OpenAI 규격 에러 응답으로 표현한다.

게이트웨이가 직접 응답하는 경우 — 계약에 맞지 않는 요청의 400 거절, 로컬 업스트림 연결 실패의
502 합성, 임베딩 업스트림의 비정상 응답을 OpenAI 형식으로 감싼 전달, 폴백 비대상 별칭의 로컬
장애 502, 임베딩 업스트림이 유효한 응답을 못 준 502, 폴백 provider를 쓸 수 없을 때의 502.
어느 경우에도 인증 헤더·키 같은 비밀은 담지 않는다.
"""

import httpx
from fastapi.responses import JSONResponse


class InvalidRequestError(Exception):
    """클라이언트 요청이 OpenAI 계약에 맞지 않는다 — 업스트림 호출 없이 400으로 응답한다."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def invalid_request_response(message: str) -> JSONResponse:
    return _error_response(
        status_code=400,
        message=message,
        error_type="invalid_request_error",
        code=None,
    )


def upstream_unavailable_response(error: httpx.RequestError) -> JSONResponse:
    return _error_response(
        status_code=502,
        message=f"upstream connection failed: {type(error).__name__}",
        error_type="upstream_error",
        code="upstream_unavailable",
    )


def local_inference_unavailable_response() -> JSONResponse:
    """폴백 비대상 별칭인데 로컬 추론이 유효한 응답을 못 줬다 — 외부로 우회하지 않고 502를 합성한다."""
    return _error_response(
        status_code=502,
        message="local inference upstream unavailable",
        error_type="upstream_error",
        code="upstream_unavailable",
    )


def upstream_invalid_response() -> JSONResponse:
    """임베딩 업스트림이 200이지만 유효한 임베딩 응답을 못 줬다 — 비밀 없는 일반 오류로 합성한다."""
    return _error_response(
        status_code=502,
        message="upstream returned an invalid embedding response",
        error_type="upstream_error",
        code="upstream_invalid_response",
    )


def upstream_body_unavailable_response(status_code: int) -> JSONResponse:
    """로컬이 확정한 오류 상태의 본문을 읽지 못했다 — 상태는 보존하고 비밀 없는 오류를 합성한다."""
    return _error_response(
        status_code=status_code,
        message="upstream response body could not be read",
        error_type="upstream_error",
        code="upstream_response_unavailable",
    )


def fallback_unavailable_response(detail: str) -> JSONResponse:
    """폴백 provider를 쓸 수 없다 — 키 미설정이나 provider 연결 실패. 비밀은 담지 않는다."""
    return _error_response(
        status_code=502,
        message=f"fallback provider unavailable: {detail}",
        error_type="upstream_error",
        code="fallback_unavailable",
    )


def _error_response(
    *, status_code: int, message: str, error_type: str, code: str | None
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": code,
            }
        },
    )
