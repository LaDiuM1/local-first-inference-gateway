"""테스트 공용 픽스처 — lifespan을 살린 인프로세스 게이트웨이 클라이언트."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI

from gateway.api_keys import ApiKeyStore
from gateway.config import settings
from gateway.main import create_app


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def isolated_request_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """요청 관측 로그를 테스트 임시 경로로 격리한다 — 운영 로그 디렉터리를 건드리지 않는다."""
    monkeypatch.setattr(settings, "request_log_directory", tmp_path / "request-logs")


@dataclass(frozen=True)
class GatewayCredentials:
    store_path: Path
    api_key: str

    @property
    def authorization_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}


@pytest.fixture
def gateway_credentials(tmp_path: Path) -> GatewayCredentials:
    store_path = tmp_path / "api-keys.json"
    issued = ApiKeyStore(store_path).issue("pytest-client")
    return GatewayCredentials(store_path, issued.api_key)


@pytest.fixture
async def running_gateway(
    gateway_credentials: GatewayCredentials,
) -> AsyncIterator[FastAPI]:
    # 운영 앱이 읽는 키 저장소 자리는 설정으로 열 수 없다 — 임시 저장소는 앱 팩터리 인자로만 준다.
    app = create_app(api_key_store_path=gateway_credentials.store_path)
    async with LifespanManager(app):
        yield app


@pytest.fixture
async def gateway_client(
    running_gateway: FastAPI, gateway_credentials: GatewayCredentials
) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=running_gateway)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://gateway.test",
        headers=gateway_credentials.authorization_headers,
    ) as client:
        yield client
