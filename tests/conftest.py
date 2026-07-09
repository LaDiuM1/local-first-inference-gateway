"""테스트 공용 픽스처 — lifespan을 살린 인프로세스 게이트웨이 클라이언트."""

from collections.abc import AsyncIterator

import httpx
import pytest
from asgi_lifespan import LifespanManager

from gateway.main import app


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def gateway_client() -> AsyncIterator[httpx.AsyncClient]:
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://gateway.test"
        ) as client:
            yield client
