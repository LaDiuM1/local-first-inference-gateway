"""로컬 상태와 Windows 관리 상태를 확인해 구성 요소를 개별 복구한다."""

import asyncio
import logging
import subprocess
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from time import monotonic
from typing import Protocol

import httpx
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from gateway.paths import (
    DEFAULT_WATCHDOG_LOG_PATH,
    TASK_CONTROL_SCRIPT,
    WATCHDOG_KEY_PATH,
)
from gateway.secret_store import ProtectedSecretError, load_machine_secret

TASK_PATH = "\\LocalFirstInferenceGateway\\"
GATEWAY_TASK_NAME = "Gateway"
CHAT_TASK_NAME = "Chat Ollama"
EMBEDDING_TASK_NAME = "Embedding Ollama"
CLOUDFLARED_SERVICE_NAME = "cloudflared"
GATEWAY_HEALTH_URL = "http://127.0.0.1:8000/health"
CHAT_HEALTH_URL = "http://127.0.0.1:11434/api/version"
EMBEDDING_HEALTH_URL = "http://127.0.0.1:11435/api/version"


class ControlKind(StrEnum):
    task = "Task"
    service = "Service"


class ManagedState(StrEnum):
    running = "Running"
    ready = "Ready"
    disabled = "Disabled"
    missing = "Missing"
    unknown = "Unknown"


class ProbeResult(StrEnum):
    healthy = "healthy"
    unauthorized = "unauthorized"
    failed = "failed"


class WatchdogAction(StrEnum):
    healthy = "healthy"
    failure = "failure"
    restarted = "restarted"
    cooldown = "cooldown"
    configuration_error = "configuration_error"
    restart_failed = "restart_failed"


@dataclass(frozen=True)
class ManagedTarget:
    name: str
    control_kind: ControlKind
    control_name: str
    probe_url: str | None = None
    authenticated_probe: bool = False


@dataclass(frozen=True)
class WatchdogEvent:
    target: str
    action: WatchdogAction
    failures: int


@dataclass
class _TargetState:
    failures: int = 0
    next_restart_at: float = 0.0
    cooldown_seconds: float = 0.0


class WatchdogSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WATCHDOG_", env_file=".env", extra="ignore"
    )

    probe_timeout_seconds: float = Field(default=5.0, gt=0)
    poll_seconds: float = Field(default=15.0, gt=0)
    failure_threshold: int = Field(default=3, ge=1)
    restart_cooldown_seconds: float = Field(default=60.0, gt=0)
    maximum_restart_cooldown_seconds: float = Field(default=600.0, gt=0)


class TaskControlError(Exception):
    """Windows 관리 상태를 조회하거나 복구할 수 없다."""


class ManagedController(Protocol):
    def state(self, target: ManagedTarget) -> ManagedState: ...

    def restart(self, target: ManagedTarget) -> None: ...


class PowerShellController:
    def __init__(self, script_path: Path, task_path: str) -> None:
        self._script_path = script_path
        self._task_path = task_path
        self._powershell = Path(
            r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
        )

    def state(self, target: ManagedTarget) -> ManagedState:
        result = self._run("State", target)
        try:
            return ManagedState(result.stdout.strip())
        except ValueError:
            return ManagedState.unknown

    def restart(self, target: ManagedTarget) -> None:
        self._run("Restart", target)

    def _run(
        self, action: str, target: ManagedTarget
    ) -> subprocess.CompletedProcess[str]:
        if not self._powershell.exists() or not self._script_path.exists():
            raise TaskControlError("Windows task control is unavailable")
        command = [
            str(self._powershell),
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(self._script_path),
            "-Action",
            action,
            "-Kind",
            target.control_kind.value,
            "-Name",
            target.control_name,
        ]
        if target.control_kind is ControlKind.task:
            command.extend(["-TaskPath", self._task_path])
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise TaskControlError("Windows task control command failed") from error
        if result.returncode != 0:
            raise TaskControlError("Windows task control command failed")
        return result


class HttpProbe:
    def __init__(self, client: httpx.AsyncClient, api_key: str) -> None:
        self._client = client
        self._api_key = api_key

    async def __call__(self, target: ManagedTarget) -> ProbeResult:
        if target.probe_url is None:
            return ProbeResult.healthy
        headers = None
        if target.authenticated_probe:
            headers = {"Authorization": f"Bearer {self._api_key}"}
        try:
            response = await self._client.get(target.probe_url, headers=headers)
        except httpx.RequestError:
            return ProbeResult.failed
        if response.status_code == 200:
            return ProbeResult.healthy
        if target.authenticated_probe and response.status_code == 401:
            return ProbeResult.unauthorized
        return ProbeResult.failed


class Watchdog:
    def __init__(
        self,
        targets: list[ManagedTarget],
        probe: Callable[[ManagedTarget], Awaitable[ProbeResult]],
        controller: ManagedController,
        failure_threshold: int,
        restart_cooldown_seconds: float,
        maximum_restart_cooldown_seconds: float,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self._targets = targets
        self._probe = probe
        self._controller = controller
        self._failure_threshold = failure_threshold
        self._initial_cooldown = restart_cooldown_seconds
        self._maximum_cooldown = maximum_restart_cooldown_seconds
        self._clock = clock
        self._states = {
            target.name: _TargetState(cooldown_seconds=restart_cooldown_seconds)
            for target in targets
        }

    async def run_once(self) -> list[WatchdogEvent]:
        events: list[WatchdogEvent] = []
        for target in self._targets:
            events.append(await self._check(target))
        return events

    async def _check(self, target: ManagedTarget) -> WatchdogEvent:
        state = self._states[target.name]
        try:
            managed_state = self._controller.state(target)
        except TaskControlError:
            managed_state = ManagedState.unknown
        probe_result = await self._probe(target)

        if probe_result is ProbeResult.unauthorized:
            state.failures = 0
            return WatchdogEvent(target.name, WatchdogAction.configuration_error, 0)
        if (
            managed_state is ManagedState.running
            and probe_result is ProbeResult.healthy
        ):
            state.failures = 0
            state.next_restart_at = 0.0
            state.cooldown_seconds = self._initial_cooldown
            return WatchdogEvent(target.name, WatchdogAction.healthy, 0)

        state.failures += 1
        if state.failures < self._failure_threshold:
            return WatchdogEvent(target.name, WatchdogAction.failure, state.failures)
        if self._clock() < state.next_restart_at:
            return WatchdogEvent(target.name, WatchdogAction.cooldown, state.failures)

        try:
            self._controller.restart(target)
        except TaskControlError:
            action = WatchdogAction.restart_failed
        else:
            action = WatchdogAction.restarted
        state.failures = 0
        state.next_restart_at = self._clock() + state.cooldown_seconds
        state.cooldown_seconds = min(state.cooldown_seconds * 2, self._maximum_cooldown)
        return WatchdogEvent(target.name, action, 0)


def _targets() -> list[ManagedTarget]:
    return [
        ManagedTarget(
            "gateway",
            ControlKind.task,
            GATEWAY_TASK_NAME,
            GATEWAY_HEALTH_URL,
            authenticated_probe=True,
        ),
        ManagedTarget(
            "chat-ollama",
            ControlKind.task,
            CHAT_TASK_NAME,
            CHAT_HEALTH_URL,
        ),
        ManagedTarget(
            "embedding-ollama",
            ControlKind.task,
            EMBEDDING_TASK_NAME,
            EMBEDDING_HEALTH_URL,
        ),
        ManagedTarget(
            "cloudflared",
            ControlKind.service,
            CLOUDFLARED_SERVICE_NAME,
        ),
    ]


async def run(configuration: WatchdogSettings) -> None:
    api_key = load_machine_secret(WATCHDOG_KEY_PATH)
    logger = _watchdog_logger(DEFAULT_WATCHDOG_LOG_PATH)
    timeout = httpx.Timeout(configuration.probe_timeout_seconds)
    controller = PowerShellController(TASK_CONTROL_SCRIPT, TASK_PATH)
    # 로컬 상태 검사는 시스템 프록시나 인터넷 연결 상태의 영향을 받지 않는다.
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        watchdog = Watchdog(
            _targets(),
            HttpProbe(client, api_key),
            controller,
            configuration.failure_threshold,
            configuration.restart_cooldown_seconds,
            configuration.maximum_restart_cooldown_seconds,
        )
        while True:
            for event in await watchdog.run_once():
                if event.action is not WatchdogAction.healthy:
                    logger.warning(
                        f"watchdog target={event.target} action={event.action.value} "
                        f"failures={event.failures}"
                    )
            await asyncio.sleep(configuration.poll_seconds)


def _watchdog_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("gateway.watchdog")
    logger.setLevel(logging.WARNING)
    logger.handlers.clear()
    handler = RotatingFileHandler(
        path, maxBytes=1024 * 1024, backupCount=2, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def main() -> int:
    try:
        asyncio.run(run(WatchdogSettings()))
    except ProtectedSecretError:
        print("watchdog configuration error: operational API key is unavailable")
        return 1
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
