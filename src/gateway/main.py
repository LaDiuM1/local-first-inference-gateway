"""OpenAI 호환 추론 게이트웨이 진입점."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from time import monotonic

import anyio
import httpx
from fastapi import APIRouter, FastAPI, Request, Response
from fastapi.responses import HTMLResponse

from gateway.api_keys import ApiKeyStore
from gateway.circuit_breaker import CircuitBreaker
from gateway.config import settings
from gateway.deadline import ResponseStartDeadline
from gateway.embeddings import create_embeddings
from gateway.errors import response_start_timeout_response
from gateway.middleware import (
    AuthenticationMiddleware,
    CacheControlMiddleware,
    RequestBodyLimitMiddleware,
)
from gateway.openai_fallback import OpenAIFallback
from gateway.paths import API_KEY_STORE_PATH, PUBLIC_DOCS_PATH
from gateway.public_docs import render_public_docs
from gateway.relay import FALLBACK_ELIGIBLE_ALIASES, relay_chat_completions
from gateway.responses import create_response
from gateway.routing import load_routing_table

router = APIRouter()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # 라우팅 설정은 기동 시 로드한다 — 구성 오류는 여기서 명확히 실패시키고 런타임으로 미루지 않는다.
    routing = load_routing_table(settings.routing_config_path)
    api_key_store = ApiKeyStore(app.state.api_key_store_path)
    api_key_store.validate()
    public_docs_html = render_public_docs(PUBLIC_DOCS_PATH)
    timeout = httpx.Timeout(
        settings.upstream_read_timeout_seconds,
        connect=settings.upstream_connect_timeout_seconds,
    )
    # chat·vision 로컬, 임베딩 전용 로컬, OpenAI 폴백 — 셋의 수명을 lifespan이 분리해 소유한다.
    async with (
        httpx.AsyncClient(
            base_url=settings.ollama_base_url,
            timeout=timeout,
            trust_env=False,
        ) as chat_client,
        httpx.AsyncClient(
            base_url=settings.embedding_ollama_base_url,
            timeout=timeout,
            trust_env=False,
        ) as embedding_client,
        httpx.AsyncClient(
            base_url=settings.openai_base_url, timeout=timeout
        ) as openai_client,
    ):
        app.state.chat_client = chat_client
        app.state.embedding_client = embedding_client
        app.state.routing = routing
        app.state.api_key_store = api_key_store
        app.state.public_docs_html = public_docs_html
        app.state.fallback = OpenAIFallback(openai_client, settings.openai_api_key)
        # 폴백 대상 별칭(chat, vision)마다 독립 회로 차단기를 둔다 — 폴백 자격의 유일한 출처.
        # 여기에 없는 chat 별칭은 회로 차단기 없이 로컬로만 라우팅된다.
        app.state.breakers = {
            alias: CircuitBreaker(
                settings.circuit_breaker_failure_threshold,
                settings.circuit_breaker_open_seconds,
                monotonic,
            )
            for alias in FALLBACK_ELIGIBLE_ALIASES
        }
        yield


@router.get("/docs", response_class=HTMLResponse)
async def public_docs(request: Request) -> HTMLResponse:
    return HTMLResponse(
        request.app.state.public_docs_html,
        headers={
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    deadline = _request_deadline(request)
    body = await _read_request_body(request, deadline)
    if isinstance(body, Response):
        return body
    state = request.app.state
    return await relay_chat_completions(
        state.chat_client,
        state.fallback,
        state.breakers,
        state.routing,
        body,
        deadline,
    )


@router.post("/v1/embeddings")
async def embeddings(request: Request) -> Response:
    deadline = _request_deadline(request)
    body = await _read_request_body(request, deadline)
    if isinstance(body, Response):
        return body
    state = request.app.state
    return await create_embeddings(
        state.embedding_client, state.routing, body, deadline
    )


@router.post("/v1/responses")
async def responses(request: Request) -> Response:
    deadline = _request_deadline(request)
    body = await _read_request_body(request, deadline)
    if isinstance(body, Response):
        return body
    state = request.app.state
    return await create_response(
        state.chat_client,
        state.fallback,
        state.breakers,
        state.routing,
        body,
        deadline,
    )


def _request_deadline(request: Request) -> ResponseStartDeadline:
    return ResponseStartDeadline(
        started_at=request.state.request_started_at,
        local_limit_seconds=settings.local_response_start_timeout_seconds,
        total_limit_seconds=settings.total_response_start_timeout_seconds,
    )


async def _read_request_body(
    request: Request, deadline: ResponseStartDeadline
) -> bytes | Response:
    remaining = deadline.total_remaining_seconds()
    if remaining <= 0:
        return response_start_timeout_response()
    try:
        with anyio.fail_after(remaining):
            return await request.body()
    except TimeoutError:
        return response_start_timeout_response()


def create_app(api_key_store_path: Path = API_KEY_STORE_PATH) -> FastAPI:
    """게이트웨이 앱을 만든다. 키 저장소 인자는 테스트에서만 바꾼다."""
    app = FastAPI(
        title="local-first-inference-gateway",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.api_key_store_path = api_key_store_path
    app.include_router(router)
    # 등록 역순으로 감싸지므로 최종 순서는 Cache-Control -> 인증 -> 본문 제한 -> 라우트다.
    # 이에 인증 실패는 request body receive를 한 번도 호출하지 않고 끝난다.
    app.add_middleware(RequestBodyLimitMiddleware)
    app.add_middleware(
        AuthenticationMiddleware, store_provider=lambda: app.state.api_key_store
    )
    app.add_middleware(CacheControlMiddleware)
    return app


app = create_app()
