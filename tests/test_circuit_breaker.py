"""회로 차단기 상태 전이 단위 테스트 — 가짜 clock으로 실제 sleep 없이 열림·복구를 검증한다."""

from gateway.circuit_breaker import CircuitBreaker, LocalDecision

FAILURE_THRESHOLD = 3
OPEN_SECONDS = 30.0


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _new_breaker(clock: FakeClock) -> CircuitBreaker:
    return CircuitBreaker(FAILURE_THRESHOLD, OPEN_SECONDS, clock)


def _drive_to_open(breaker: CircuitBreaker) -> None:
    for _ in range(FAILURE_THRESHOLD):
        attempt = breaker.begin()
        assert attempt.decision is LocalDecision.attempt
        breaker.record_failure(attempt)


def test_starts_closed_and_attempts_local() -> None:
    breaker = _new_breaker(FakeClock())

    assert breaker.begin().decision is LocalDecision.attempt


def test_stays_closed_below_threshold() -> None:
    breaker = _new_breaker(FakeClock())

    for _ in range(FAILURE_THRESHOLD - 1):
        attempt = breaker.begin()
        assert attempt.decision is LocalDecision.attempt
        breaker.record_failure(attempt)

    assert breaker.begin().decision is LocalDecision.attempt


def test_opens_after_consecutive_threshold_failures() -> None:
    breaker = _new_breaker(FakeClock())

    _drive_to_open(breaker)

    assert breaker.begin().decision is LocalDecision.skip


def test_success_resets_failure_count() -> None:
    breaker = _new_breaker(FakeClock())

    for _ in range(FAILURE_THRESHOLD - 1):
        breaker.record_failure(breaker.begin())
    breaker.record_success(breaker.begin())

    # 초기화됐으므로 다시 임계값 직전까지 실패해도 열리지 않는다.
    for _ in range(FAILURE_THRESHOLD - 1):
        attempt = breaker.begin()
        assert attempt.decision is LocalDecision.attempt
        breaker.record_failure(attempt)
    assert breaker.begin().decision is LocalDecision.attempt


def test_stays_open_until_open_seconds_elapse() -> None:
    clock = FakeClock()
    breaker = _new_breaker(clock)
    _drive_to_open(breaker)

    clock.advance(OPEN_SECONDS - 0.1)
    assert breaker.begin().decision is LocalDecision.skip


def test_half_open_allows_single_probe_after_open_seconds() -> None:
    clock = FakeClock()
    breaker = _new_breaker(clock)
    _drive_to_open(breaker)

    clock.advance(OPEN_SECONDS)
    assert breaker.begin().decision is LocalDecision.probe
    # 같은 시점의 나머지 요청은 로컬을 건너뛴다.
    assert breaker.begin().decision is LocalDecision.skip
    assert breaker.begin().decision is LocalDecision.skip


def test_probe_success_closes_circuit() -> None:
    clock = FakeClock()
    breaker = _new_breaker(clock)
    _drive_to_open(breaker)
    clock.advance(OPEN_SECONDS)

    probe = breaker.begin()
    assert probe.decision is LocalDecision.probe
    breaker.record_success(probe)

    assert breaker.begin().decision is LocalDecision.attempt


def test_probe_failure_reopens_circuit() -> None:
    clock = FakeClock()
    breaker = _new_breaker(clock)
    _drive_to_open(breaker)
    clock.advance(OPEN_SECONDS)

    probe = breaker.begin()
    assert probe.decision is LocalDecision.probe
    breaker.record_failure(probe)

    # 다시 열렸으므로 open_seconds가 다시 지나야 probe를 허용한다.
    assert breaker.begin().decision is LocalDecision.skip
    clock.advance(OPEN_SECONDS)
    assert breaker.begin().decision is LocalDecision.probe


def test_release_returns_probe_slot() -> None:
    clock = FakeClock()
    breaker = _new_breaker(clock)
    _drive_to_open(breaker)
    clock.advance(OPEN_SECONDS)

    probe = breaker.begin()
    assert probe.decision is LocalDecision.probe
    breaker.release(probe)

    # 확정 없이 슬롯만 반납했으므로 다음 요청이 새 probe로 나갈 수 있다.
    assert breaker.begin().decision is LocalDecision.probe


def test_breakers_are_independent_per_instance() -> None:
    clock = FakeClock()
    chat = _new_breaker(clock)
    vision = _new_breaker(clock)

    _drive_to_open(chat)

    assert chat.begin().decision is LocalDecision.skip
    assert vision.begin().decision is LocalDecision.attempt


# --- 교차 완료 동시성: 열림 이전에 시작돼 half-open 도중 뒤늦게 끝난 옛 요청은 무시된다 ---


def test_stale_attempt_success_does_not_disturb_half_open_probe() -> None:
    clock = FakeClock()
    breaker = _new_breaker(clock)

    # closed 상태에서 시작해 아직 끝나지 않은 느린 로컬 시도.
    stale = breaker.begin()
    assert stale.decision is LocalDecision.attempt

    # 다른 요청들의 연속 실패로 회로가 열리고, open_seconds 뒤 half-open probe가 나간다.
    _drive_to_open(breaker)
    clock.advance(OPEN_SECONDS)
    probe = breaker.begin()
    assert probe.decision is LocalDecision.probe

    # 세대가 지난 옛 시도의 성공은 회로를 닫지도, probe 슬롯을 풀지도 못한다.
    breaker.record_success(stale)
    assert breaker.begin().decision is LocalDecision.skip

    # 오직 현 세대 probe만 회로를 닫는다.
    breaker.record_success(probe)
    assert breaker.begin().decision is LocalDecision.attempt


def test_stale_attempt_failure_does_not_release_probe_slot() -> None:
    clock = FakeClock()
    breaker = _new_breaker(clock)

    stale = breaker.begin()
    _drive_to_open(breaker)
    clock.advance(OPEN_SECONDS)
    probe = breaker.begin()
    assert probe.decision is LocalDecision.probe

    # 옛 시도의 실패도 무시된다 — probe 슬롯을 풀어 두 번째 probe를 만들지 않는다.
    breaker.record_failure(stale)
    assert breaker.begin().decision is LocalDecision.skip

    # 현 세대 probe는 여전히 정상적으로 회로를 닫는다.
    breaker.record_success(probe)
    assert breaker.begin().decision is LocalDecision.attempt


def test_concurrent_closed_attempts_share_generation() -> None:
    breaker = _new_breaker(FakeClock())

    # 같은 closed 세대의 두 시도 — 하나가 실패해도 다른 하나의 성공이 정상 반영된다.
    first = breaker.begin()
    second = breaker.begin()
    breaker.record_failure(first)
    breaker.record_success(second)

    for _ in range(FAILURE_THRESHOLD - 1):
        attempt = breaker.begin()
        assert attempt.decision is LocalDecision.attempt
        breaker.record_failure(attempt)
    assert breaker.begin().decision is LocalDecision.attempt


def test_transition_listener_receives_full_recovery_sequence() -> None:
    clock = FakeClock()
    transitions: list[str] = []
    breaker = CircuitBreaker(
        FAILURE_THRESHOLD, OPEN_SECONDS, clock, on_transition=transitions.append
    )

    _drive_to_open(breaker)
    clock.advance(OPEN_SECONDS)
    probe = breaker.begin()
    assert probe.decision is LocalDecision.probe
    breaker.record_success(probe)

    assert transitions == ["open", "half_open", "closed"]
