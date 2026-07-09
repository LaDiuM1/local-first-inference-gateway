"""/health 엔드포인트 계약 테스트."""

import httpx
import pytest

pytestmark = pytest.mark.anyio


async def test_health_returns_ok(gateway_client: httpx.AsyncClient) -> None:
    response = await gateway_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
