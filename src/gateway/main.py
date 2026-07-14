"""OpenAI 호환 추론 게이트웨이 진입점."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import monotonic

import httpx
from fastapi import FastAPI, Request, Response

from gateway.circuit_breaker import CircuitBreaker
from gateway.config import settings
from gateway.embeddings import create_embeddings
from gateway.openai_fallback import OpenAIFallback
from gateway.relay import FALLBACK_ELIGIBLE_ALIASES, relay_chat_completions
from gateway.routing import load_routing_table


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # 라우팅 설정은 기동 시 로드한다 — 구성 오류는 여기서 명확히 실패시키고 런타임으로 미루지 않는다.
    routing = load_routing_table(settings.routing_config_path)
    timeout = httpx.Timeout(
        settings.upstream_read_timeout_seconds,
        connect=settings.upstream_connect_timeout_seconds,
    )
    # chat·vision 로컬, 임베딩 전용 로컬, OpenAI 폴백 — 셋의 수명을 lifespan이 분리해 소유한다.
    async with (
        httpx.AsyncClient(
            base_url=settings.ollama_base_url, timeout=timeout
        ) as chat_client,
        httpx.AsyncClient(
            base_url=settings.embedding_ollama_base_url, timeout=timeout
        ) as embedding_client,
        httpx.AsyncClient(
            base_url=settings.openai_base_url, timeout=timeout
        ) as openai_client,
    ):
        app.state.chat_client = chat_client
        app.state.embedding_client = embedding_client
        app.state.routing = routing
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


app = FastAPI(title="local-first-inference-gateway", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    body = await request.body()
    state = request.app.state
    return await relay_chat_completions(
        state.chat_client, state.fallback, state.breakers, state.routing, body
    )


@app.post("/v1/embeddings")
async def embeddings(request: Request) -> Response:
    body = await request.body()
    state = request.app.state
    return await create_embeddings(state.embedding_client, state.routing, body)
