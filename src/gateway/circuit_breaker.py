"""별칭별 회로 차단기 — 로컬 추론 장애가 반복되면 로컬 시도를 건너뛰고, 일정 시간 뒤 probe로 복구한다.

단일 게이트웨이 프로세스(단일 이벤트 루프) 기준 인메모리 상태다. `begin()`은 이번 요청의 판단과 함께
그 시점의 상태 세대를 담은 `LocalAttempt`를 돌려주고, 결과 기록 시 그 토큰을 되돌려 받는다. 상태가
바뀔 때마다 세대가 오르므로, 열림 직전에 시작돼 half-open 도중 뒤늦게 끝난 요청의 결과는 세대가
어긋나 무시된다 — 이 덕분에 옛 시도가 회로를 임의로 닫거나 probe 슬롯을 풀어 두 번째 probe를 만드는
교차 완료 문제가 생기지 않는다. 멀티 워커 공유 상태는 현재 범위가 아니다.

시간 판단은 주입된 monotonic clock으로만 하므로, 테스트는 실제 sleep 없이 열림·half-open·복구를
결정적으로 검증할 수 있다.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum


class LocalDecision(StrEnum):
    """이번 요청이 로컬을 어떻게 다룰지의 판단 — `begin()`이 돌려주고 결과 기록 시 되돌려 받는다."""

    attempt = "attempt"  # closed 상태 — 로컬을 시도하고, 폴백 대상 장애면 실패로 기록한 뒤 폴백한다.
    probe = "probe"  # half-open 단일 복구 시도 — 성공하면 닫고 실패하면 다시 연다.
    skip = "skip"  # open 또는 probe 진행 중 — 로컬을 건너뛰고 바로 폴백한다.


@dataclass(frozen=True)
class LocalAttempt:
    """`begin()`이 돌려주는 이번 요청의 판단과 상태 세대. 결과를 기록할 때 그대로 되돌려 준다."""

    decision: LocalDecision
    generation: int


class _State(StrEnum):
    closed = "closed"
    open = "open"
    half_open = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int,
        open_seconds: float,
        clock: Callable[[], float],
        on_transition: Callable[[str], None] | None = None,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._open_seconds = open_seconds
        self._clock = clock
        # 상태 전환 관측 훅 — 새 상태 이름(closed·open·half_open)만 받는다.
        self._on_transition = on_transition
        self._state = _State.closed
        self._consecutive_failures = 0
        self._opened_at = 0.0
        self._probe_in_flight = False
        self._generation = 0

    def begin(self) -> LocalAttempt:
        """이번 요청이 로컬을 시도할지, 단일 probe로 나갈지, 건너뛸지를 정한다."""
        if (
            self._state is _State.open
            and self._clock() - self._opened_at >= self._open_seconds
        ):
            self._transition(_State.half_open)
        if self._state is _State.closed:
            return LocalAttempt(LocalDecision.attempt, self._generation)
        if self._state is _State.half_open and not self._probe_in_flight:
            self._probe_in_flight = True
            return LocalAttempt(LocalDecision.probe, self._generation)
        return LocalAttempt(LocalDecision.skip, self._generation)

    def record_success(self, attempt: LocalAttempt) -> None:
        """로컬이 응답했다 — 성공이든 폴백 대상이 아닌 4xx든 서버가 살아 있으므로 회로를 닫는다."""
        if attempt.generation != self._generation:
            return
        if self._state is _State.half_open:
            self._transition(_State.closed)
            return
        self._consecutive_failures = 0

    def record_failure(self, attempt: LocalAttempt) -> None:
        """로컬이 폴백 대상 장애를 냈다 — probe였으면 즉시 다시 열고, 아니면 임계값에서 연다."""
        if attempt.generation != self._generation:
            return
        if attempt.decision is LocalDecision.probe:
            self._transition(_State.open)
            return
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._failure_threshold:
            self._transition(_State.open)

    def release(self, attempt: LocalAttempt) -> None:
        """성공·실패로 확정하지 못하고 끝난 경로(클라이언트 취소 등) — 현 세대 probe 슬롯만 되돌린다."""
        if attempt.generation != self._generation:
            return
        if attempt.decision is LocalDecision.probe:
            self._probe_in_flight = False

    def _transition(self, state: _State) -> None:
        """상태를 바꾸고 세대를 올린다 — 이전 세대에서 시작된 요청의 결과는 이후 무시된다."""
        self._state = state
        self._generation += 1
        self._probe_in_flight = False
        if state is _State.closed:
            self._consecutive_failures = 0
        elif state is _State.open:
            self._opened_at = self._clock()
        if self._on_transition is not None:
            self._on_transition(state.value)
