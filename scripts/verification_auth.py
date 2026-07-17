"""검증 게이트웨이에만 주입하는 출력 없는 임시 API 키 저장소.

운영 진입점 `gateway.main:app`이 읽는 키 저장소 자리는 설치기가 잠근 고정 경로이고 설정으로 열지
않는다. 그래서 검증 하네스는 운영 진입점 대신 임시 저장소를 인자로 받는 이 팩터리로 게이트웨이를
기동한다 — SYSTEM 작업은 이 모듈을 실행하지 않는다.
"""

import os
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI

from gateway.api_keys import ApiKeyStore
from gateway.main import create_app

_STORE_PATH_VARIABLE = "OPENAT_VERIFICATION_KEY_STORE_PATH"
UVICORN_APPLICATION_ARGUMENTS = [
    "--factory",
    "verification_auth:create_verification_app",
]


def create_verification_app() -> FastAPI:
    return create_app(api_key_store_path=Path(os.environ[_STORE_PATH_VARIABLE]))


@dataclass
class VerificationAuth:
    store_path: Path
    api_key: str = field(repr=False)
    _directory: TemporaryDirectory = field(repr=False)

    @classmethod
    def create(cls, client: str) -> "VerificationAuth":
        directory = TemporaryDirectory(prefix="openat-verify-")
        store_path = Path(directory.name) / "api-keys.json"
        issued = ApiKeyStore(store_path).issue(client)
        return cls(store_path, issued.api_key, directory)

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def apply_to(self, environment: dict[str, str]) -> None:
        environment[_STORE_PATH_VARIABLE] = str(self.store_path)
        # 검증 게이트웨이의 요청 관측 로그도 임시 디렉터리로 격리한다 — 운영 로그를 오염시키지 않는다.
        environment["GATEWAY_REQUEST_LOG_DIRECTORY"] = str(
            Path(self._directory.name) / "request-logs"
        )
        # 검증 게이트웨이는 별도 프로세스다 — 이 모듈을 팩터리로 import할 수 있어야 한다.
        scripts_directory = str(Path(__file__).resolve().parent)
        existing = environment.get("PYTHONPATH")
        if existing:
            scripts_directory = f"{scripts_directory}{os.pathsep}{existing}"
        environment["PYTHONPATH"] = scripts_directory

    def close(self) -> None:
        """임시 저장소를 정리한다 — 검증 결과에는 영향이 없다.

        Windows는 게이트웨이 프로세스 종료 직후 관측 로그 파일 핸들을 늦게 놓아
        삭제가 WinError 32로 경합할 수 있다. 짧게 재시도하고, 끝내 남으면 임시
        디렉터리를 그대로 두고 넘어간다 — 정리 실패로 검증·측정을 깨뜨리지 않는다.
        """
        for _ in range(10):
            try:
                self._directory.cleanup()
                return
            except PermissionError:
                time.sleep(0.3)
        with suppress(PermissionError):
            self._directory.cleanup()
