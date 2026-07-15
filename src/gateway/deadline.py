"""Cloudflare 앞에서 응답 시작을 보장하기 위한 요청 단위 기한 계산.

두 기한 모두 요청 도착 시각(`started_at`)부터 흐른다 — 인증과 본문 수신에 쓴 시간도 이미 지나간
시간이므로 남은 예산에서 빠진다. 로컬 잔여는 로컬 한도와 전체 한도 중 먼저 닿는 쪽을 따른다.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from time import monotonic


@dataclass(frozen=True)
class ResponseStartDeadline:
    started_at: float
    local_limit_seconds: float
    total_limit_seconds: float
    clock: Callable[[], float] = field(default=monotonic, repr=False, compare=False)

    def local_remaining_seconds(self) -> float:
        return min(
            self._remaining_seconds(self.local_limit_seconds),
            self.total_remaining_seconds(),
        )

    def total_remaining_seconds(self) -> float:
        return self._remaining_seconds(self.total_limit_seconds)

    def _remaining_seconds(self, limit_seconds: float) -> float:
        # 뒤로 가는 clock에서도 잔여가 한도를 넘지 않도록 경과 시간을 0 아래로 두지 않는다.
        elapsed = max(0.0, self.clock() - self.started_at)
        return max(0.0, limit_seconds - elapsed)
