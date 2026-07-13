"""chat completions 별칭 라우팅 중계.

요청에 `reasoning_effort`가 없으면 기본 일반 모드(`none`)를 넣고, 있으면 비어 있지 않은
문자열만 그대로 업스트림에 전달한다. 이어서 `model` 별칭을 실제 모델명으로 치환하며, 나머지
클라이언트 필드는 그대로 업스트림에 전달한다. stream 여부와 무관하게 동일한 규칙을 적용한다.
게이트웨이가 직접 응답하는 경우는 계약 위반 요청의 400, 업스트림 연결 실패의 502뿐이다.
"""

import json

import httpx
from fastapi import Response
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from gateway.errors import (
    InvalidRequestError,
    invalid_request_response,
    upstream_unavailable_response,
)
from gateway.routing import EndpointKind, RoutingTable
from gateway.validation import load_json_object, require_string

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

REASONING_EFFORT_FIELD = "reasoning_effort"
NORMAL_MODE_EFFORT = "none"


async def relay_chat_completions(
    upstream: httpx.AsyncClient, routing: RoutingTable, body: bytes
) -> Response:
    """사고 수준을 확정하고 별칭을 치환한 뒤 stream 여부에 따라 중계 경로를 고른다."""
    try:
        payload = load_json_object(body)
        _apply_reasoning_effort(payload)
        alias = require_string(payload, "model")
        payload["model"] = routing.resolve(EndpointKind.chat, alias)
    except InvalidRequestError as error:
        return invalid_request_response(error.message)

    routed_body = json.dumps(payload).encode("utf-8")
    if payload.get("stream"):
        return await _relay_streaming(upstream, routed_body)
    return await _relay_buffered(upstream, routed_body)


def _apply_reasoning_effort(payload: dict) -> None:
    """업스트림에 전달할 `reasoning_effort`를 확정한다.

    필드가 없으면 기본 일반 모드(`none`)를 넣고, 있으면 비어 있지 않은 문자열만 클라이언트가
    보낸 값 그대로 업스트림에 전달한다. 비문자열·빈 문자열은 OpenAI 규격 400으로 거절한다.
    """
    if REASONING_EFFORT_FIELD not in payload:
        payload[REASONING_EFFORT_FIELD] = NORMAL_MODE_EFFORT
        return
    payload[REASONING_EFFORT_FIELD] = require_string(payload, REASONING_EFFORT_FIELD)


async def _relay_buffered(upstream: httpx.AsyncClient, body: bytes) -> Response:
    try:
        upstream_response = await upstream.post(
            CHAT_COMPLETIONS_PATH,
            content=body,
            headers={"content-type": "application/json"},
        )
    except httpx.RequestError as error:
        return upstream_unavailable_response(error)

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
        return upstream_unavailable_response(error)

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
