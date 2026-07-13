"""게이트웨이 설정 — `GATEWAY_` 접두사 환경변수와 저장소 `.env` 파일에서 읽는다."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # .env에는 다른 용도의 키(OPENAI_API_KEY 등)도 있으므로 GATEWAY_ 필드 외는 무시한다.
    model_config = SettingsConfigDict(
        env_prefix="GATEWAY_", env_file=".env", extra="ignore"
    )

    ollama_base_url: str = "http://127.0.0.1:11434"
    upstream_connect_timeout_seconds: float = 5.0
    upstream_read_timeout_seconds: float = 120.0
    routing_config_path: Path = Path("routing.yaml")


settings = Settings()
