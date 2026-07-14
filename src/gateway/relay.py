"""chat completions 별칭 라우팅 중계와 로컬 장애 시 OpenAI 폴백.

요청에 `reasoning_effort`가 없으면 기본 일반 모드(`none`)를 넣고, 있으면 비어 있지 않은 문자열만
그대로 전달한다. 이어서 `model` 별칭을 실제 모델명으로 치환하며 나머지 필드는 그대로 둔다.
계약 위반 요청은 업스트림 호출 없이 400으로 거절한다.

외부 우회(데이터 반출) 대상은 `FALLBACK_ELIGIBLE_ALIASES` 한 곳에만 적었다. 이 별칭(chat·vision)만
로컬이 유효한 응답을 못 줄 때(연결 실패·시간 초과·429·5xx·모델 부재 404·응답 시작 전 빈/유효하지
않은 응답) 같은 요청을 OpenAI로 우회한다. 그 밖의 chat 엔드포인트 별칭은 로컬로 라우팅하되 장애 시
OpenAI를 호출하지 않고 로컬 업스트림 오류를 돌려준다. 별칭별 회로 차단기가 반복 장애를 끊고,
스트리밍은 첫 바이트를 클라이언트에 보내기 전까지만 폴백하며 이후에는 provider를 섞지 않는다.
"""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
from fastapi import Response

from gateway.circuit_breaker import CircuitBreaker, LocalAttempt, LocalDecision
from gateway.errors import (
    InvalidRequestError,
    invalid_request_response,
    local_inference_unavailable_response,
    upstream_body_unavailable_response,
)
from gateway.openai_fallback import OpenAIFallback
from gateway.relay_common import (
    CHAT_COMPLETIONS_PATH,
    EXCLUDED_BUFFERED_HEADERS,
    EXCLUDED_STREAMING_HEADERS,
    NORMAL_MODE_EFFORT,
    REASONING_EFFORT_FIELD,
    ManagedStreamingResponse,
    StreamCleanup,
    StreamPrefix,
    is_fallback_status,
    is_success_status,
    is_valid_chat_completion_body,
    iter_committed_stream,
    read_and_close_response,
    relayed_headers,
    secure_success_stream,
)
from gateway.routing import EndpointKind, RoutingTable
from gateway.validation import load_json_object, require_boolean, require_string

JSON_HEADERS = {"content-type": "application/json"}
STREAMING_JSON_HEADERS = {
    "content-type": "application/json",
    "accept-encoding": "identity",
}

# 데이터 반출(외부 OpenAI 우회) 정책의 유일한 출처 — 이 두 별칭만 로컬 장애 시 OpenAI로 폴백한다.
# main은 이 집합으로만 회로 차단기를 만들고, 여기에 없는 chat 별칭은 폴백 없이 로컬로만 라우팅한다.
FALLBACK_ELIGIBLE_ALIASES: frozenset[str] = frozenset({"chat", "vision"})


@dataclass(frozen=True)
class _LocalServed:
    """로컬이 클라이언트에 그대로 전달할 최종 버퍼 응답을 냈다 — 성공 2xx 또는 폴백 대상 아닌 4xx."""

    response: Response


@dataclass(frozen=True)
class _LocalCommittedStream:
    """로컬 스트림이 첫 유효 이벤트까지 확정됐다 — 이제 provider를 섞지 않고 끝까지 로컬로 흘린다."""

    prefix: StreamPrefix


@dataclass(frozen=True)
class _LocalFailed:
    """로컬이 유효한 추론 응답을 못 줬다 — 폴백 대상(연결 실패·429·5xx·404·빈/무효 응답)."""


_LocalOutcome = _LocalServed | _LocalCommittedStream | _LocalFailed


async def relay_chat_completions(
    local_client: httpx.AsyncClient,
    fallback: OpenAIFallback,
    breakers: dict[str, CircuitBreaker],
    routing: RoutingTable,
    body: bytes,
) -> Response:
    """사고 수준을 확정하고 별칭을 치환한 뒤, 폴백 자격에 따라 로컬 또는 OpenAI로 중계한다."""
    try:
        payload = load_json_object(body)
        _apply_reasoning_effort(payload)
        alias = require_string(payload, "model")
        payload["model"] = routing.resolve(EndpointKind.chat, alias)
        streaming = require_boolean(payload, "stream", default=False)
    except InvalidRequestError as error:
        return invalid_request_response(error.message)

    routed_body = json.dumps(payload, allow_nan=False).encode("utf-8")
    breaker = breakers.get(alias)
    if breaker is None:
        # 폴백 비대상 chat 별칭 — 로컬로만 라우팅하고 장애 시 OpenAI를 호출하지 않는다.
        return await _relay_local_only(local_client, streaming, routed_body)
    return await _relay_eligible(
        local_client, fallback, breaker, streaming, payload, routed_body
    )


def _apply_reasoning_effort(payload: dict) -> None:
    """업스트림에 전달할 `reasoning_effort`를 확정한다.

    필드가 없으면 기본 일반 모드(`none`)를 넣고, 있으면 비어 있지 않은 문자열만 클라이언트가
    보낸 값 그대로 전달한다. 비문자열·빈 문자열은 OpenAI 규격 400으로 거절한다.
    """
    if REASONING_EFFORT_FIELD not in payload:
        payload[REASONING_EFFORT_FIELD] = NORMAL_MODE_EFFORT
        return
    payload[REASONING_EFFORT_FIELD] = require_string(payload, REASONING_EFFORT_FIELD)


async def _relay_eligible(
    local_client: httpx.AsyncClient,
    fallback: OpenAIFallback,
    breaker: CircuitBreaker,
    streaming: bool,
    payload: dict,
    routed_body: bytes,
) -> Response:
    """폴백 대상 별칭 — 회로 차단기 판단에 따라 로컬을 시도하고, 폴백 대상 장애면 OpenAI로 우회한다."""
    attempt = breaker.begin()
    if attempt.decision is LocalDecision.skip:
        return await _fallback(fallback, payload, streaming)
    try:
        outcome = await _attempt_local(local_client, streaming, routed_body)
    except BaseException:
        breaker.release(attempt)
        raise

    if isinstance(outcome, _LocalServed):
        breaker.record_success(attempt)
        return outcome.response
    if isinstance(outcome, _LocalCommittedStream):
        cleanup = StreamCleanup(
            outcome.prefix.response, lambda: breaker.release(attempt)
        )
        return ManagedStreamingResponse(
            _stream_committed(outcome.prefix, breaker, attempt, cleanup),
            cleanup,
            status_code=outcome.prefix.response.status_code,
            headers=relayed_headers(
                outcome.prefix.response, EXCLUDED_STREAMING_HEADERS
            ),
        )
    breaker.record_failure(attempt)
    return await _fallback(fallback, payload, streaming)


async def _relay_local_only(
    local_client: httpx.AsyncClient, streaming: bool, routed_body: bytes
) -> Response:
    """폴백 비대상 별칭 — 로컬로만 라우팅하고 장애 시 로컬 업스트림 오류를 돌려준다(OpenAI 미호출)."""
    outcome = await _attempt_local(local_client, streaming, routed_body)
    if isinstance(outcome, _LocalServed):
        return outcome.response
    if isinstance(outcome, _LocalCommittedStream):
        cleanup = StreamCleanup(outcome.prefix.response)
        return ManagedStreamingResponse(
            iter_committed_stream(outcome.prefix, cleanup),
            cleanup,
            status_code=outcome.prefix.response.status_code,
            headers=relayed_headers(
                outcome.prefix.response, EXCLUDED_STREAMING_HEADERS
            ),
        )
    return local_inference_unavailable_response()


async def _fallback(
    fallback: OpenAIFallback, payload: dict, streaming: bool
) -> Response:
    if streaming:
        return await fallback.relay_stream(payload)
    return await fallback.relay_buffered(payload)


async def _attempt_local(
    local_client: httpx.AsyncClient, streaming: bool, routed_body: bytes
) -> _LocalOutcome:
    if streaming:
        return await _attempt_local_stream(local_client, routed_body)
    return await _attempt_local_buffered(local_client, routed_body)


async def _attempt_local_buffered(
    local_client: httpx.AsyncClient, routed_body: bytes
) -> _LocalServed | _LocalFailed:
    try:
        response = await local_client.post(
            CHAT_COMPLETIONS_PATH, content=routed_body, headers=JSON_HEADERS
        )
    except httpx.RequestError:
        return _LocalFailed()

    if is_fallback_status(response.status_code):
        return _LocalFailed()
    # 성공 상태인데 유효한 Chat Completions 응답이 아니면 로컬이 추론 응답을 만들지 못한 것으로 본다.
    # 폴백 대상이 아닌 4xx는 로컬이 판단한 클라이언트 오류이므로 본문을 검사하지 않고 그대로 전달한다.
    if is_success_status(response.status_code) and not is_valid_chat_completion_body(
        response.content
    ):
        return _LocalFailed()
    return _LocalServed(
        Response(
            content=response.content,
            status_code=response.status_code,
            headers=relayed_headers(response, EXCLUDED_BUFFERED_HEADERS),
        )
    )


async def _attempt_local_stream(
    local_client: httpx.AsyncClient, routed_body: bytes
) -> _LocalOutcome:
    request = local_client.build_request(
        "POST",
        CHAT_COMPLETIONS_PATH,
        content=routed_body,
        headers=STREAMING_JSON_HEADERS,
    )
    try:
        response = await local_client.send(request, stream=True)
    except httpx.RequestError:
        return _LocalFailed()

    if is_fallback_status(response.status_code):
        await response.aclose()
        return _LocalFailed()
    if not is_success_status(response.status_code):
        # 폴백 대상 아닌 4xx — 본문을 읽어 그대로 전달한다(빈 본문이어도 폴백하지 않는다).
        try:
            body = await read_and_close_response(response)
        except httpx.RequestError:
            return _LocalServed(
                upstream_body_unavailable_response(response.status_code)
            )
        return _LocalServed(
            Response(
                content=body,
                status_code=response.status_code,
                headers=relayed_headers(response, EXCLUDED_BUFFERED_HEADERS),
            )
        )

    prefix = await secure_success_stream(response)
    if prefix is None:
        return _LocalFailed()
    return _LocalCommittedStream(prefix)


async def _stream_committed(
    prefix: StreamPrefix,
    breaker: CircuitBreaker,
    attempt: LocalAttempt,
    cleanup: StreamCleanup,
) -> AsyncIterator[bytes]:
    """확정된 로컬 스트림을 흘린다. 중간 장애는 스트림을 끊고 실패로 기록하되 provider를 섞지 않는다."""
    try:
        for chunk in prefix.initial_chunks:
            yield chunk
        async for chunk in prefix.remaining:
            yield chunk
        breaker.record_success(attempt)
        cleanup.resolve()
    except httpx.RequestError:
        breaker.record_failure(attempt)
        cleanup.resolve()
        raise
    finally:
        await cleanup.close()
