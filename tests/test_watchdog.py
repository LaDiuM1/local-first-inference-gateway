"""Watchdog 연속 실패, 대상 격리, cooldown/backoff와 401 처리, 운영 경로 고정 테스트."""

from pathlib import Path

import pytest

import gateway
from gateway.paths import STATE_DIRECTORY, TASK_CONTROL_SCRIPT, WATCHDOG_KEY_PATH
from gateway.watchdog import (
    CHAT_HEALTH_URL,
    CHAT_TASK_NAME,
    CLOUDFLARED_SERVICE_NAME,
    EMBEDDING_HEALTH_URL,
    EMBEDDING_TASK_NAME,
    GATEWAY_HEALTH_URL,
    GATEWAY_TASK_NAME,
    TASK_PATH,
    ControlKind,
    ManagedState,
    ManagedTarget,
    ProbeResult,
    TaskControlError,
    Watchdog,
    WatchdogAction,
    WatchdogSettings,
)
from gateway.watchdog import (
    _targets as _managed_targets,
)

pytestmark = pytest.mark.anyio


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeController:
    def __init__(self, targets: list[ManagedTarget]) -> None:
        self.states = {target.name: ManagedState.running for target in targets}
        self.restarted: list[str] = []
        self.fail_restart = False

    def state(self, target: ManagedTarget) -> ManagedState:
        return self.states[target.name]

    def restart(self, target: ManagedTarget) -> None:
        if self.fail_restart:
            raise TaskControlError("failed")
        self.restarted.append(target.name)


class FakeProbe:
    def __init__(self, targets: list[ManagedTarget]) -> None:
        self.results = {target.name: ProbeResult.healthy for target in targets}

    async def __call__(self, target: ManagedTarget) -> ProbeResult:
        return self.results[target.name]


def _targets() -> list[ManagedTarget]:
    return [
        ManagedTarget(
            "gateway", ControlKind.task, "Gateway", "http://local/health", True
        ),
        ManagedTarget("chat", ControlKind.task, "Chat", "http://local/chat"),
        ManagedTarget("embed", ControlKind.task, "Embed", "http://local/embed"),
    ]


def _watchdog(
    targets: list[ManagedTarget],
    probe: FakeProbe,
    controller: FakeController,
    clock: FakeClock,
    threshold: int = 3,
) -> Watchdog:
    return Watchdog(targets, probe, controller, threshold, 60.0, 600.0, clock)


async def test_restarts_only_after_consecutive_failure_threshold() -> None:
    targets = _targets()
    probe = FakeProbe(targets)
    controller = FakeController(targets)
    clock = FakeClock()
    watchdog = _watchdog(targets, probe, controller, clock)
    probe.results["gateway"] = ProbeResult.failed

    first = await watchdog.run_once()
    second = await watchdog.run_once()
    third = await watchdog.run_once()

    assert first[0].action is WatchdogAction.failure
    assert second[0].failures == 2
    assert third[0].action is WatchdogAction.restarted
    assert controller.restarted == ["gateway"]


async def test_restarts_only_the_unresponsive_target() -> None:
    targets = _targets()
    probe = FakeProbe(targets)
    controller = FakeController(targets)
    clock = FakeClock()
    watchdog = _watchdog(targets, probe, controller, clock, threshold=1)
    probe.results["embed"] = ProbeResult.failed

    events = await watchdog.run_once()

    assert controller.restarted == ["embed"]
    assert [event.action for event in events] == [
        WatchdogAction.healthy,
        WatchdogAction.healthy,
        WatchdogAction.restarted,
    ]


async def test_restart_cooldown_doubles_and_healthy_probe_resets_it() -> None:
    targets = _targets()[:1]
    probe = FakeProbe(targets)
    controller = FakeController(targets)
    clock = FakeClock()
    watchdog = _watchdog(targets, probe, controller, clock, threshold=1)
    probe.results["gateway"] = ProbeResult.failed

    assert (await watchdog.run_once())[0].action is WatchdogAction.restarted
    assert (await watchdog.run_once())[0].action is WatchdogAction.cooldown
    clock.advance(60.0)
    assert (await watchdog.run_once())[0].action is WatchdogAction.restarted
    clock.advance(60.0)
    assert (await watchdog.run_once())[0].action is WatchdogAction.cooldown

    probe.results["gateway"] = ProbeResult.healthy
    assert (await watchdog.run_once())[0].action is WatchdogAction.healthy
    probe.results["gateway"] = ProbeResult.failed
    assert (await watchdog.run_once())[0].action is WatchdogAction.restarted


async def test_gateway_401_is_configuration_error_and_never_restarts() -> None:
    targets = _targets()[:1]
    probe = FakeProbe(targets)
    controller = FakeController(targets)
    watchdog = _watchdog(targets, probe, controller, FakeClock(), threshold=1)
    probe.results["gateway"] = ProbeResult.unauthorized

    for _ in range(5):
        event = (await watchdog.run_once())[0]
        assert event.action is WatchdogAction.configuration_error

    assert controller.restarted == []


async def test_non_running_task_counts_as_failure_even_if_endpoint_responds() -> None:
    targets = _targets()[:1]
    probe = FakeProbe(targets)
    controller = FakeController(targets)
    controller.states["gateway"] = ManagedState.ready
    watchdog = _watchdog(targets, probe, controller, FakeClock(), threshold=1)

    event = (await watchdog.run_once())[0]

    assert event.action is WatchdogAction.restarted
    assert controller.restarted == ["gateway"]


def test_task_control_script_is_fixed_to_the_running_deployment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 이 경로는 SYSTEM PowerShell의 -File 인자가 된다. 설정으로 열면 배포된 .env 한 줄이나 환경변수
    # 하나가 사용자 쓰기 가능한 스크립트를 SYSTEM으로 실행시킨다.
    attacker_script = tmp_path / "attacker.ps1"
    attacker_script.write_text("Write-Output 'runs as SYSTEM'", encoding="utf-8")
    monkeypatch.setenv("WATCHDOG_TASK_CONTROL_SCRIPT", str(attacker_script))

    configuration = WatchdogSettings(_env_file=None)

    assert not hasattr(configuration, "task_control_script")
    deployment_root = Path(gateway.__file__).resolve().parents[2]
    assert deployment_root / "scripts" / "task_control.ps1" == TASK_CONTROL_SCRIPT
    assert TASK_CONTROL_SCRIPT.exists()


def test_managed_targets_cannot_be_redirected_by_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    redirected = {
        "WATCHDOG_TASK_PATH": "\\Foreign\\",
        "WATCHDOG_GATEWAY_TASK_NAME": "Foreign Gateway",
        "WATCHDOG_CHAT_TASK_NAME": "Foreign Chat",
        "WATCHDOG_EMBEDDING_TASK_NAME": "Foreign Embed",
        "WATCHDOG_CLOUDFLARED_SERVICE_NAME": "foreign-service",
        "WATCHDOG_GATEWAY_HEALTH_URL": "https://example.invalid/steal-key",
        "WATCHDOG_CHAT_HEALTH_URL": "https://example.invalid/chat",
        "WATCHDOG_EMBEDDING_HEALTH_URL": "https://example.invalid/embed",
        "WATCHDOG_LOG_PATH": str(tmp_path / "system-write.log"),
    }
    for name, value in redirected.items():
        monkeypatch.setenv(name, value)

    configuration = WatchdogSettings(_env_file=None)
    targets = _managed_targets()

    for attribute in [
        "task_path",
        "gateway_task_name",
        "chat_task_name",
        "embedding_task_name",
        "cloudflared_service_name",
        "gateway_health_url",
        "chat_health_url",
        "embedding_health_url",
        "log_path",
    ]:
        assert not hasattr(configuration, attribute)
    assert TASK_PATH == "\\LocalFirstInferenceGateway\\"
    assert [(target.control_name, target.probe_url) for target in targets] == [
        (GATEWAY_TASK_NAME, GATEWAY_HEALTH_URL),
        (CHAT_TASK_NAME, CHAT_HEALTH_URL),
        (EMBEDDING_TASK_NAME, EMBEDDING_HEALTH_URL),
        (CLOUDFLARED_SERVICE_NAME, None),
    ]


def test_operational_key_path_cannot_be_moved_out_of_the_protected_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 설치기는 %ProgramData%의 보호 키만 잠근다. 이 경로를 설정으로 열면 배포된 .env 한 줄이 Watchdog을
    # 사용자가 쓸 수 있는 파일의 키로 게이트웨이에 붙게 만든다.
    planted_key = tmp_path / "planted-key.dpapi"
    planted_key.write_bytes(b"not-a-real-protected-key")
    monkeypatch.setenv("WATCHDOG_API_KEY_PATH", str(planted_key))

    configuration = WatchdogSettings(_env_file=None)

    assert not hasattr(configuration, "api_key_path")
    assert WATCHDOG_KEY_PATH == STATE_DIRECTORY / "watchdog-key.dpapi"


def test_env_file_cannot_redirect_the_operational_key_path(tmp_path: Path) -> None:
    deployed_env = tmp_path / ".env"
    deployed_env.write_text(
        f"WATCHDOG_API_KEY_PATH={tmp_path / 'planted-key.dpapi'}\n", encoding="utf-8"
    )

    configuration = WatchdogSettings(_env_file=deployed_env)

    assert not hasattr(configuration, "api_key_path")


async def test_restart_failure_is_reported_without_affecting_other_targets() -> None:
    targets = _targets()
    probe = FakeProbe(targets)
    controller = FakeController(targets)
    controller.fail_restart = True
    watchdog = _watchdog(targets, probe, controller, FakeClock(), threshold=1)
    probe.results["chat"] = ProbeResult.failed

    events = await watchdog.run_once()

    assert events[0].action is WatchdogAction.healthy
    assert events[1].action is WatchdogAction.restart_failed
    assert events[2].action is WatchdogAction.healthy
