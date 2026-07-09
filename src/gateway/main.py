"""OpenAI 호환 추론 게이트웨이 진입점."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response

from gateway.config import settings
from gateway.relay import relay_chat_completions


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    timeout = httpx.Timeout(
        settings.upstream_read_timeout_seconds,
        connect=settings.upstream_connect_timeout_seconds,
    )
    async with httpx.AsyncClient(
        base_url=settings.ollama_base_url, timeout=timeout
    ) as upstream:
        app.state.upstream = upstream
        yield


app = FastAPI(title="local-first-inference-gateway", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    body = await request.body()
    return await relay_chat_completions(request.app.state.upstream, body)
