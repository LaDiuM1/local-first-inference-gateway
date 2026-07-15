"""게이트웨이 설정 — `GATEWAY_` 접두사 환경변수와 저장소 `.env` 파일에서 읽는다.

OpenAI 폴백 키만 예외로 표준 `OPENAI_API_KEY`에서 읽고 `SecretStr`로 다뤄 노출을 막는다.
키가 없어도 로컬 정상 요청과 앱 기동은 되며, 실제 폴백이 필요할 때만 비밀 없는 502로 응답한다.

인증 저장소와 공개 문서처럼 배포 경계에 속하는 경로는 `gateway.paths`에 고정한다.
"""

from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # .env에는 다른 용도의 키(OPENAI_API_KEY 등)도 있으므로 GATEWAY_ 필드 외는 무시한다.
    model_config = SettingsConfigDict(
        env_prefix="GATEWAY_", env_file=".env", extra="ignore"
    )

    # chat·vision을 서비스하는 로컬 Ollama(GPU).
    ollama_base_url: str = "http://127.0.0.1:11434"
    # embed 전용 로컬 Ollama(CPU) — chat·vision과 분리된 별도 인스턴스·주소다.
    embedding_ollama_base_url: str = "http://127.0.0.1:11435"
    upstream_connect_timeout_seconds: float = 5.0
    upstream_read_timeout_seconds: float = 120.0
    local_response_start_timeout_seconds: float = Field(default=90.0, gt=0)
    total_response_start_timeout_seconds: float = Field(default=115.0, gt=0)
    routing_config_path: Path = Path("routing.yaml")

    # 별칭별 회로 차단기 — 연속 실패가 임계값에 닿으면 open_seconds 동안 로컬을 건너뛴다.
    circuit_breaker_failure_threshold: int = 3
    circuit_breaker_open_seconds: float = 30.0

    # OpenAI 폴백 — chat·vision 별칭만 로컬 장애 시 우회한다. 폴백 모델은 계약상 고정이라 설정으로
    # 열지 않고 openai_fallback 모듈 상수로 둔다.
    openai_base_url: str = "https://api.openai.com/v1"
    # 표준 OPENAI_API_KEY에서만 읽는다 — GATEWAY_ 접두사를 붙이지 않고 SecretStr로 감싼다.
    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_API_KEY")
    )


settings = Settings()
