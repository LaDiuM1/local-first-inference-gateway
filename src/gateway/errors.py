"""요청 검증 실패와 업스트림 오류를 OpenAI 규격 에러 응답으로 표현한다.

게이트웨이가 직접 응답하는 경우는 세 가지다 — 계약에 맞지 않는 요청의 400 거절,
업스트림 연결 실패의 502 합성, 임베딩 업스트림의 비정상 응답을 OpenAI 형식으로 감싼 전달.
"""

import httpx
from fastapi.responses import JSONResponse


class InvalidRequestError(Exception):
    """클라이언트 요청이 OpenAI 계약에 맞지 않는다 — 업스트림 호출 없이 400으로 응답한다."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def invalid_request_response(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        },
    )


def upstream_unavailable_response(error: httpx.RequestError) -> JSONResponse:
    return JSONResponse(
        status_code=502,
        content={
            "error": {
                "message": f"upstream connection failed: {type(error).__name__}",
                "type": "upstream_error",
                "param": None,
                "code": "upstream_unavailable",
            }
        },
    )
