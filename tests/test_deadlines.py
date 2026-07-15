"""로컬 90초와 전체 115초 응답 시작 기한 및 스트리밍 커밋 경계 테스트."""

import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Protocol

import anyio
import httpx
import pytest
import respx
from asgi_lifespan import LifespanManager
from pydantic import SecretStr

from gateway.config import Settings, settings
from gateway.deadline import ResponseStartDeadline
from gateway.main import create_app

pytestmark = pytest.mark.anyio

LOCAL_URL = f"{settings.ollama_base_url}/v1/chat/completions"
OPENAI_URL = f"{settings.openai_base_url}/chat/completions"
EMBED_URL = f"{settings.embedding_ollama_base_url}/api/embed"
CHAT_REQUEST = {"model": "chat", "messages": [{"role": "user", "content": "x"}]}
LOCAL_SSE = b'data: {"choices":[{"delta":{"content":"local"}}]}\n\n'
OPENAI_SSE = b'data: {"choices":[{"delta":{"content":"openai"}}]}\n\n'
DONE = b"data: [DONE]\n\n"


class GatewayCredentials(Protocol):
    api_key: str


class DelayedStream(httpx.AsyncByteStream):
    def __init__(
        self, first_delay: float, remaining_delay: float, marker: bytes
    ) -> None:
        self._first_delay = first_delay
        self._remaining_delay = remaining_delay
        self._marker = marker
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        await anyio.sleep(self._first_delay)
        yield self._marker
        await anyio.sleep(self._remaining_delay)
        yield DONE

    async def aclose(self) -> None:
        self.closed = True


class CheckpointClosingStream(httpx.AsyncByteStream):
    """aclose()에 체크포인트가 있는 스트림 — 실제 네트워크 응답과 같다.

    취소된 스코프 안의 await는 곧바로 다시 취소되므로, 커밋 전 검사가 기한 초과로 취소된 뒤
    shield 없이 닫으면 이 체크포인트에서 닫기가 끝나지 못하고 연결이 남는다.
    """

    def __init__(self, first_delay: float, marker: bytes) -> None:
        self._first_delay = first_delay
        self._marker = marker
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        await anyio.sleep(self._first_delay)
        yield self._marker
        yield DONE

    async def aclose(self) -> None:
        await anyio.sleep(0)
        self.closed = True


class SlowRequestBody(httpx.AsyncByteStream):
    """마지막 조각이 늦게 도착하는 요청 본문 — 수신에 쓴 시간이 기한에 포함되는지 확인한다."""

    def __init__(self, body: bytes, delay: float) -> None:
        self._body = body
        self._delay = delay

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._body[:1]
        await anyio.sleep(self._delay)
        yield self._body[1:]


@asynccontextmanager
async def _gateway_client(
    credentials: GatewayCredentials,
) -> AsyncIterator[httpx.AsyncClient]:
    # 임시 키 저장소는 앱 팩터리 인자로만 준다 — 운영 앱의 저장소 자리는 설정으로 열리지 않는다.
    app = create_app(api_key_store_path=credentials.store_path)
    async with (
        LifespanManager(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://gateway.test",
            headers={"Authorization": f"Bearer {credentials.api_key}"},
        ) as client,
    ):
        yield client


def test_timeout_defaults_are_90_local_and_115_total() -> None:
    defaults = Settings(_env_file=None)

    assert defaults.local_response_start_timeout_seconds == 90.0
    assert defaults.total_response_start_timeout_seconds == 115.0


def test_remaining_time_counts_from_request_arrival() -> None:
    deadline = ResponseStartDeadline(
        started_at=0.0,
        local_limit_seconds=90.0,
        total_limit_seconds=115.0,
        clock=lambda: 20.0,
    )

    assert deadline.local_remaining_seconds() == 70.0
    assert deadline.total_remaining_seconds() == 95.0


def test_local_remaining_is_capped_by_total_remaining() -> None:
    deadline = ResponseStartDeadline(
        started_at=0.0,
        local_limit_seconds=90.0,
        total_limit_seconds=30.0,
        clock=lambda: 10.0,
    )

    assert deadline.local_remaining_seconds() == 20.0


def test_local_budget_is_exhausted_while_total_still_remains() -> None:
    deadline = ResponseStartDeadline(
        started_at=10.0,
        local_limit_seconds=90.0,
        total_limit_seconds=115.0,
        clock=lambda: 100.0,
    )

    assert deadline.local_remaining_seconds() == 0.0
    assert deadline.total_remaining_seconds() == 25.0


def test_backwards_clock_never_grants_more_than_the_configured_limits() -> None:
    deadline = ResponseStartDeadline(
        started_at=50.0,
        local_limit_seconds=90.0,
        total_limit_seconds=115.0,
        clock=lambda: 10.0,
    )

    assert deadline.local_remaining_seconds() == 90.0
    assert deadline.total_remaining_seconds() == 115.0


@respx.mock
async def test_local_buffered_timeout_falls_back_within_total_deadline(
    gateway_credentials: GatewayCredentials, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "local_response_start_timeout_seconds", 0.02)
    monkeypatch.setattr(settings, "total_response_start_timeout_seconds", 0.5)
    monkeypatch.setattr(settings, "openai_api_key", SecretStr("fake-openai-key"))

    async def slow_local(request: httpx.Request) -> httpx.Response:
        await anyio.sleep(0.1)
        return httpx.Response(200, json={"id": "local", "choices": []})

    respx.post(LOCAL_URL).mock(side_effect=slow_local)
    openai = respx.post(OPENAI_URL).respond(200, json={"id": "openai", "choices": []})

    async with _gateway_client(gateway_credentials) as client:
        response = await client.post("/v1/chat/completions", json=CHAT_REQUEST)

    assert response.status_code == 200
    assert response.json()["id"] == "openai"
    assert openai.called


@respx.mock
async def test_fallback_receives_only_total_deadline_remainder(
    gateway_credentials: GatewayCredentials, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "local_response_start_timeout_seconds", 0.2)
    monkeypatch.setattr(settings, "total_response_start_timeout_seconds", 0.18)
    monkeypatch.setattr(settings, "openai_api_key", SecretStr("fake-openai-key"))

    async def delayed_local_failure(request: httpx.Request) -> httpx.Response:
        await anyio.sleep(0.1)
        return httpx.Response(503, json={"error": "down"})

    async def slow_fallback(request: httpx.Request) -> httpx.Response:
        await anyio.sleep(0.2)
        return httpx.Response(200, json={"id": "too-late", "choices": []})

    respx.post(LOCAL_URL).mock(side_effect=delayed_local_failure)
    respx.post(OPENAI_URL).mock(side_effect=slow_fallback)

    async with _gateway_client(gateway_credentials) as client:
        started = time.monotonic()
        response = await client.post("/v1/chat/completions", json=CHAT_REQUEST)
        elapsed = time.monotonic() - started

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "response_start_timeout"
    assert elapsed < 0.27


@respx.mock
async def test_streaming_timeout_before_first_event_switches_provider(
    gateway_credentials: GatewayCredentials, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "local_response_start_timeout_seconds", 0.02)
    monkeypatch.setattr(settings, "total_response_start_timeout_seconds", 0.5)
    monkeypatch.setattr(settings, "openai_api_key", SecretStr("fake-openai-key"))
    local_stream = DelayedStream(0.1, 0.0, LOCAL_SSE)
    openai_stream = DelayedStream(0.0, 0.0, OPENAI_SSE)
    respx.post(LOCAL_URL).mock(return_value=httpx.Response(200, stream=local_stream))
    openai = respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(200, stream=openai_stream)
    )

    async with _gateway_client(gateway_credentials) as client:
        response = await client.post(
            "/v1/chat/completions", json={**CHAT_REQUEST, "stream": True}
        )

    assert response.status_code == 200
    assert b"openai" in response.content
    assert b"local" not in response.content
    assert local_stream.closed
    assert openai.called


@respx.mock
async def test_deadline_stops_after_first_valid_sse_event_without_provider_mixing(
    gateway_credentials: GatewayCredentials, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "local_response_start_timeout_seconds", 0.03)
    monkeypatch.setattr(settings, "total_response_start_timeout_seconds", 0.03)
    monkeypatch.setattr(settings, "openai_api_key", SecretStr("fake-openai-key"))
    local_stream = DelayedStream(0.0, 0.08, LOCAL_SSE)
    respx.post(LOCAL_URL).mock(return_value=httpx.Response(200, stream=local_stream))
    openai = respx.post(OPENAI_URL)

    async with _gateway_client(gateway_credentials) as client:
        response = await client.post(
            "/v1/chat/completions", json={**CHAT_REQUEST, "stream": True}
        )

    assert response.status_code == 200
    assert response.content == LOCAL_SSE + DONE
    assert not openai.called
    assert local_stream.closed


@respx.mock
async def test_local_stream_timeout_closes_the_upstream_response_it_abandons(
    gateway_credentials: GatewayCredentials, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "local_response_start_timeout_seconds", 0.02)
    monkeypatch.setattr(settings, "total_response_start_timeout_seconds", 0.5)
    monkeypatch.setattr(settings, "openai_api_key", SecretStr("fake-openai-key"))
    local_stream = CheckpointClosingStream(0.1, LOCAL_SSE)
    respx.post(LOCAL_URL).mock(return_value=httpx.Response(200, stream=local_stream))
    respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(200, stream=DelayedStream(0.0, 0.0, OPENAI_SSE))
    )

    async with _gateway_client(gateway_credentials) as client:
        response = await client.post(
            "/v1/chat/completions", json={**CHAT_REQUEST, "stream": True}
        )

    assert response.status_code == 200
    assert b"openai" in response.content
    assert local_stream.closed


@respx.mock
async def test_fallback_stream_timeout_closes_the_openai_response_it_abandons(
    gateway_credentials: GatewayCredentials, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "local_response_start_timeout_seconds", 0.5)
    monkeypatch.setattr(settings, "total_response_start_timeout_seconds", 0.15)
    monkeypatch.setattr(settings, "openai_api_key", SecretStr("fake-openai-key"))
    openai_stream = CheckpointClosingStream(0.5, OPENAI_SSE)
    respx.post(LOCAL_URL).respond(503, json={"error": "down"})
    respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, stream=openai_stream))

    async with _gateway_client(gateway_credentials) as client:
        response = await client.post(
            "/v1/chat/completions", json={**CHAT_REQUEST, "stream": True}
        )

    assert response.status_code == 504
    assert openai_stream.closed


@respx.mock
async def test_embedding_local_timeout_returns_504_without_openai_fallback(
    gateway_credentials: GatewayCredentials, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "local_response_start_timeout_seconds", 0.02)
    monkeypatch.setattr(settings, "total_response_start_timeout_seconds", 0.5)
    monkeypatch.setattr(settings, "openai_api_key", SecretStr("fake-openai-key"))

    async def slow_embedding(request: httpx.Request) -> httpx.Response:
        await anyio.sleep(0.1)
        return httpx.Response(200, json={"model": "x", "embeddings": [[0.1]]})

    respx.post(EMBED_URL).mock(side_effect=slow_embedding)
    openai = respx.post(OPENAI_URL)

    async with _gateway_client(gateway_credentials) as client:
        response = await client.post(
            "/v1/embeddings", json={"model": "embed", "input": "x"}
        )

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "response_start_timeout"
    assert not openai.called


@respx.mock
async def test_local_budget_spent_on_request_body_goes_straight_to_fallback(
    gateway_credentials: GatewayCredentials, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 본문 수신에 로컬 기한을 다 쓰면 로컬 시도는 의미가 없다 — 전체 기한의 잔여로 바로 폴백한다.
    monkeypatch.setattr(settings, "local_response_start_timeout_seconds", 0.05)
    monkeypatch.setattr(settings, "total_response_start_timeout_seconds", 5.0)
    monkeypatch.setattr(settings, "openai_api_key", SecretStr("fake-openai-key"))
    local = respx.post(LOCAL_URL)
    respx.post(OPENAI_URL).respond(200, json={"id": "openai", "choices": []})

    async with _gateway_client(gateway_credentials) as client:
        response = await client.post(
            "/v1/chat/completions",
            content=SlowRequestBody(json.dumps(CHAT_REQUEST).encode(), 0.1),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 200
    assert response.json()["id"] == "openai"
    assert not local.called


@respx.mock
async def test_slow_request_body_leaves_only_the_remainder_for_fallback(
    gateway_credentials: GatewayCredentials, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "local_response_start_timeout_seconds", 5.0)
    monkeypatch.setattr(settings, "total_response_start_timeout_seconds", 0.15)
    monkeypatch.setattr(settings, "openai_api_key", SecretStr("fake-openai-key"))

    async def slow_fallback(request: httpx.Request) -> httpx.Response:
        await anyio.sleep(0.5)
        return httpx.Response(200, json={"id": "too-late", "choices": []})

    respx.post(LOCAL_URL).respond(503, json={"error": "down"})
    respx.post(OPENAI_URL).mock(side_effect=slow_fallback)

    async with _gateway_client(gateway_credentials) as client:
        started = time.monotonic()
        response = await client.post(
            "/v1/chat/completions",
            content=SlowRequestBody(json.dumps(CHAT_REQUEST).encode(), 0.1),
            headers={"Content-Type": "application/json"},
        )
        elapsed = time.monotonic() - started

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "response_start_timeout"
    assert elapsed < 0.4


@respx.mock
async def test_total_deadline_includes_request_body_reception(
    gateway_credentials: GatewayCredentials, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "local_response_start_timeout_seconds", 0.2)
    monkeypatch.setattr(settings, "total_response_start_timeout_seconds", 0.02)
    local = respx.post(LOCAL_URL)

    async with _gateway_client(gateway_credentials) as client:
        response = await client.post(
            "/v1/chat/completions",
            content=SlowRequestBody(b"{}", 0.1),
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "response_start_timeout"
    assert not local.called
