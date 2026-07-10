"""Ollama의 OpenAI 호환 경로로의 중계.

요청·응답 본문은 원문 바이트 그대로 전달한다. 게이트웨이가 직접 응답하는
경우는 둘뿐이다 — 형식이 깨진 요청의 400 거절, 업스트림 연결 실패의 502 합성.
"""

import json

import httpx
from fastapi import Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

# hop-by-hop 헤더(RFC 9110 §7.6.1)는 구간 전용이라 중계하지 않는다.
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
# 버퍼링 경로는 디코딩된 본문을 반환하므로 길이·인코딩을 재계산에 맡기고,
# 스트리밍 경로는 원문 바이트를 그대로 흘리므로 인코딩 헤더가 바이트를 따라간다.
EXCLUDED_BUFFERED_HEADERS = HOP_BY_HOP_HEADERS | {"content-length", "content-encoding"}
EXCLUDED_STREAMING_HEADERS = HOP_BY_HOP_HEADERS | {"content-length"}


async def relay_chat_completions(upstream: httpx.AsyncClient, body: bytes) -> Response:
    """stream 여부에 따라 중계 경로를 고르고, 요청 원문과 응답을 그대로 전달한다."""
    try:
        wants_stream = _parse_stream_flag(body)
    except ValueError:
        return _invalid_request_response()

    if wants_stream:
        return await _relay_streaming(upstream, body)
    return await _relay_buffered(upstream, body)


def _parse_stream_flag(body: bytes) -> bool:
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError("request body is not a JSON object")
    return bool(payload.get("stream"))


async def _relay_buffered(upstream: httpx.AsyncClient, body: bytes) -> Response:
    try:
        upstream_response = await upstream.post(
            CHAT_COMPLETIONS_PATH,
            content=body,
            headers={"content-type": "application/json"},
        )
    except httpx.RequestError as error:
        return _upstream_unavailable_response(error)

    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers=_relayed_headers(upstream_response, EXCLUDED_BUFFERED_HEADERS),
    )


async def _relay_streaming(upstream: httpx.AsyncClient, body: bytes) -> Response:
    stream_request = upstream.build_request(
        "POST",
        CHAT_COMPLETIONS_PATH,
        content=body,
        headers={"content-type": "application/json"},
    )
    try:
        upstream_response = await upstream.send(stream_request, stream=True)
    except httpx.RequestError as error:
        return _upstream_unavailable_response(error)

    return StreamingResponse(
        upstream_response.aiter_raw(),
        status_code=upstream_response.status_code,
        headers=_relayed_headers(upstream_response, EXCLUDED_STREAMING_HEADERS),
        background=BackgroundTask(upstream_response.aclose),
    )


def _relayed_headers(
    upstream_response: httpx.Response, excluded: set[str]
) -> dict[str, str]:
    return {
        name: value
        for name, value in upstream_response.headers.items()
        if name.lower() not in excluded
    }


def _invalid_request_response() -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": "request body is not a valid JSON object",
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        },
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
