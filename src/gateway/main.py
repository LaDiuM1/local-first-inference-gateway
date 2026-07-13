"""OpenAI 호환 추론 게이트웨이 진입점."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response

from gateway.config import settings
from gateway.embeddings import create_embeddings
from gateway.relay import relay_chat_completions
from gateway.routing import load_routing_table


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # 라우팅 설정은 기동 시 로드한다 — 구성 오류는 여기서 명확히 실패시키고 런타임으로 미루지 않는다.
    routing = load_routing_table(settings.routing_config_path)
    timeout = httpx.Timeout(
        settings.upstream_read_timeout_seconds,
        connect=settings.upstream_connect_timeout_seconds,
    )
    async with httpx.AsyncClient(
        base_url=settings.ollama_base_url, timeout=timeout
    ) as upstream:
        app.state.upstream = upstream
        app.state.routing = routing
        yield


app = FastAPI(title="local-first-inference-gateway", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    body = await request.body()
    return await relay_chat_completions(
        request.app.state.upstream, request.app.state.routing, body
    )


@app.post("/v1/embeddings")
async def embeddings(request: Request) -> Response:
    body = await request.body()
    return await create_embeddings(
        request.app.state.upstream, request.app.state.routing, body
    )
