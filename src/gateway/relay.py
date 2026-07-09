"""Ollama의 OpenAI 호환 경로로의 투명 중계.

요청·응답 본문에 손을 대지 않는다 — 파싱·재직렬화가 없을수록 OpenAI 호환
계약이 깨질 여지가 없다. 업스트림이 응답 자체를 주지 못한 경우(연결 실패)에만
OpenAI 에러 포맷으로 502를 합성한다.
"""

import httpx
from fastapi import Response
from fastapi.responses import JSONResponse

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

# hop-by-hop 헤더(RFC 9110 §7.6.1)는 구간 전용이라 중계하지 않고,
# 본문 길이·인코딩은 httpx가 디코딩한 본문 기준으로 재계산에 맡긴다.
EXCLUDED_UPSTREAM_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "content-encoding",
}


async def relay_chat_completions(upstream: httpx.AsyncClient, body: bytes) -> Response:
    """요청 본문을 업스트림에 그대로 넘기고, 응답 상태·헤더·본문을 원문 그대로 돌려준다."""
    try:
        upstream_response = await upstream.post(
            CHAT_COMPLETIONS_PATH,
            content=body,
            headers={"content-type": "application/json"},
        )
    except httpx.RequestError as error:
        return _upstream_unavailable_response(error)

    relayed_headers = {
        name: value
        for name, value in upstream_response.headers.items()
        if name.lower() not in EXCLUDED_UPSTREAM_HEADERS
    }
    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers=relayed_headers,
    )


def _upstream_unavailable_response(error: httpx.RequestError) -> JSONResponse:
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
