"""로컬 추론 장애 시 chat·vision 요청을 OpenAI Chat Completions로 우회한다.

라우팅된 로컬 payload를 받아 model을 폴백 모델로 바꾸고, 로컬 일반 모드(`none`)만 저지연 기본값
`minimal`로 변환한 뒤 그대로 중계한다. 명시된 사고 수준(`high` 등)은 그대로 전달해 provider가
지원 범위를 판단하게 한다. 인증 헤더는 요청에만 실리고 응답·오류·로그에는 노출하지 않는다.
API 키가 없거나 OpenAI 연결이 실패하거나 2xx인데 유효한 Chat Completions 응답을 못 주면
비밀을 담지 않은 OpenAI 규격 502로 합성한다. 오류 상태(비2xx)는 본문 그대로 전달한다.
"""

import json

import anyio
import httpx
from fastapi import Response
from pydantic import SecretStr

from gateway.errors import (
    fallback_unavailable_response,
    response_start_timeout_response,
)
from gateway.relay_common import (
    EXCLUDED_BUFFERED_HEADERS,
    EXCLUDED_STREAMING_HEADERS,
    NORMAL_MODE_EFFORT,
    OPENAI_MINIMAL_EFFORT,
    REASONING_EFFORT_FIELD,
    ManagedStreamingResponse,
    StreamCleanup,
    is_success_status,
    is_valid_chat_completion_body,
    iter_committed_stream,
    read_and_close_response,
    relayed_headers,
    secure_success_stream,
)

# OpenAI base URL은 이미 /v1을 포함하므로 여기서는 엔드포인트 경로만 붙인다.
OPENAI_CHAT_COMPLETIONS_PATH = "/chat/completions"

# 폴백 대상 모델 — DECISIONS 계약상 gpt-5-mini로 고정한다. 배포 설정으로 다른 모델을 향하게 열지 않는다.
FALLBACK_MODEL = "gpt-5-mini"

_INVALID_RESPONSE_DETAIL = "provider returned an invalid response"


class OpenAIFallback:
    """OpenAI Chat Completions로의 폴백 중계. 키·수명은 lifespan이 소유하는 client가 관리한다."""

    def __init__(self, client: httpx.AsyncClient, api_key: SecretStr | None) -> None:
        self._client = client
        self._api_key = api_key

    async def relay_buffered(
        self, routed_payload: dict, response_start_timeout_seconds: float
    ) -> Response:
        headers = self._auth_headers()
        if headers is None:
            return fallback_unavailable_response("API key is not configured")
        try:
            with anyio.fail_after(response_start_timeout_seconds):
                response = await self._client.post(
                    OPENAI_CHAT_COMPLETIONS_PATH,
                    content=self._openai_body(routed_payload),
                    headers=headers,
                )
        except TimeoutError:
            return response_start_timeout_response()
        except httpx.RequestError as error:
            return fallback_unavailable_response(
                f"connection failed: {type(error).__name__}"
            )
        # 2xx인데 유효한 Chat Completions 응답이 아니면 비밀 없는 502로 합성한다.
        # 오류 상태(비2xx)는 provider의 오류 본문을 그대로 클라이언트에 전달한다.
        if is_success_status(
            response.status_code
        ) and not is_valid_chat_completion_body(response.content):
            return fallback_unavailable_response(_INVALID_RESPONSE_DETAIL)
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=relayed_headers(response, EXCLUDED_BUFFERED_HEADERS),
        )

    async def relay_stream(
        self, routed_payload: dict, response_start_timeout_seconds: float
    ) -> Response:
        headers = self._auth_headers()
        if headers is None:
            return fallback_unavailable_response("API key is not configured")
        request = self._client.build_request(
            "POST",
            OPENAI_CHAT_COMPLETIONS_PATH,
            content=self._openai_body(routed_payload),
            headers=headers,
        )
        try:
            with anyio.fail_after(response_start_timeout_seconds):
                response = await self._client.send(request, stream=True)
                return await self._prepare_stream_response(response)
        except TimeoutError:
            return response_start_timeout_response()
        except httpx.RequestError as error:
            return fallback_unavailable_response(
                f"connection failed: {type(error).__name__}"
            )

    async def _prepare_stream_response(self, response: httpx.Response) -> Response:
        if not is_success_status(response.status_code):
            # 오류 상태 — 본문을 읽어 그대로 전달한다(버퍼 경로와 같은 규칙).
            try:
                body = await read_and_close_response(response)
            except httpx.RequestError as error:
                return fallback_unavailable_response(
                    f"response body read failed: {type(error).__name__}"
                )
            return Response(
                content=body,
                status_code=response.status_code,
                headers=relayed_headers(response, EXCLUDED_BUFFERED_HEADERS),
            )

        prefix = await secure_success_stream(response)
        if prefix is None:
            # 첫 유효 이벤트를 못 확보한 2xx 스트림 — 시작하지 않고 비밀 없는 502로 합성한다.
            return fallback_unavailable_response(_INVALID_RESPONSE_DETAIL)
        cleanup = StreamCleanup(prefix.response)
        return ManagedStreamingResponse(
            iter_committed_stream(prefix, cleanup),
            cleanup,
            status_code=prefix.response.status_code,
            headers=relayed_headers(prefix.response, EXCLUDED_STREAMING_HEADERS),
        )

    def _auth_headers(self) -> dict[str, str] | None:
        if self._api_key is None:
            return None
        secret = self._api_key.get_secret_value()
        if not secret:
            return None
        return {
            "content-type": "application/json",
            "authorization": f"Bearer {secret}",
            # 스트림은 HTTPX 디코딩 계층을 거쳐 중계하지만, provider에 압축을 요청하지 않는 것이
            # 가장 단순한 정상 경로다. provider가 압축을 보내더라도 content-encoding은 제거된다.
            "accept-encoding": "identity",
        }

    def _openai_body(self, routed_payload: dict) -> bytes:
        payload = dict(routed_payload)
        payload["model"] = FALLBACK_MODEL
        if payload.get(REASONING_EFFORT_FIELD) == NORMAL_MODE_EFFORT:
            payload[REASONING_EFFORT_FIELD] = OPENAI_MINIMAL_EFFORT
        return json.dumps(payload, allow_nan=False).encode("utf-8")
