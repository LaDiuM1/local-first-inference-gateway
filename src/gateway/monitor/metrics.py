"""요청 관측 로그(JSON Lines) 후처리 — 운영 지표 집계의 순수 로직.

게이트웨이가 요청 단위로 남기는 기록을 창(window) 단위 운영 지표로 요약한다.
파일을 읽는 로더를 제외한 모든 함수는 입력만으로 동작해 pytest로 단독 검증한다.
"""

import json
import math
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

LOG_FILE_NAME = "requests.jsonl"


@dataclass(frozen=True)
class LogSnapshot:
    """관측 로그 읽기 결과 — 접근 불가와 무트래픽을 구분해 보고한다."""

    records: list[dict]
    invalid_lines: int
    readable: bool
    files_skipped: int


def _reject_nonstandard_constant(constant: str) -> object:
    raise ValueError(f"non-standard JSON constant: {constant}")


def _is_serializable_record(record: dict) -> bool:
    """상태 JSON에 안전하게 실을 수 있는 기록인지 본다 — 무한대 수와 고립 surrogate 차단."""
    pending: list[object] = [record]
    while pending:
        value = pending.pop()
        if isinstance(value, float) and not math.isfinite(value):
            return False
        if isinstance(value, str):
            try:
                value.encode("utf-8")
            except UnicodeEncodeError:
                return False
        elif isinstance(value, dict):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return True


def parse_log_lines(lines: list[str]) -> tuple[list[dict], int]:
    """JSON Lines를 기록 목록으로 파싱한다 — 회전 경계에서 잘린 줄은 세기만 한다.

    `NaN`·`1e400` 같은 비유한 수나 고립 surrogate가 담긴 줄은 손상 줄로 격리한다 —
    지표에 흘러들면 상태 JSON 직렬화와 집계를 오염시킨다.
    """
    records: list[dict] = []
    invalid_count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped, parse_constant=_reject_nonstandard_constant)
        except (json.JSONDecodeError, ValueError):
            invalid_count += 1
            continue
        if isinstance(parsed, dict) and _is_serializable_record(parsed):
            records.append(parsed)
        else:
            invalid_count += 1
    return records, invalid_count


def _parse_log_bytes(data: bytes) -> tuple[list[dict], int]:
    """바이트 단위 로그 조각을 줄 단위로 디코딩·파싱한다 — 깨진 인코딩도 줄 격리."""
    lines: list[str] = []
    invalid_count = 0
    for raw_line in data.splitlines():
        try:
            lines.append(raw_line.decode("utf-8"))
        except UnicodeDecodeError:
            invalid_count += 1
    records, parse_invalid = parse_log_lines(lines)
    return records, invalid_count + parse_invalid


def rotation_age(path: Path) -> int:
    """회전 파일명에서 세대를 꺼낸다 — 0이 활성 파일, 클수록 오래된 회전분이다."""
    suffix = path.name.removeprefix(LOG_FILE_NAME).lstrip(".")
    if not suffix:
        return 0
    if suffix.isdigit():
        return int(suffix)
    return -1


def read_log_snapshot(log_directory: Path) -> LogSnapshot:
    """회전 파일을 포함한 전체 관측 로그를 오래된 것부터 읽는다.

    운영 로그는 관리자만 읽으므로 접근 거부·경로 부재는 빈 결과가 아니라
    readable=False로 보고해 실제 무트래픽과 구분한다.
    """
    try:
        if not log_directory.is_dir():
            return LogSnapshot([], 0, readable=False, files_skipped=0)
        candidates = [
            path
            for path in log_directory.glob(f"{LOG_FILE_NAME}*")
            if rotation_age(path) >= 0
        ]
    except OSError:
        return LogSnapshot([], 0, readable=False, files_skipped=0)
    records: list[dict] = []
    invalid_lines = 0
    files_skipped = 0
    for path in sorted(candidates, key=rotation_age, reverse=True):
        try:
            parsed, invalid = _parse_log_bytes(path.read_bytes())
        except OSError:
            # 읽는 도중 회전으로 파일이 사라질 수 있다 — 스킵을 세고 남은 파일로 집계한다.
            files_skipped += 1
            continue
        records.extend(parsed)
        invalid_lines += invalid
    if candidates and files_skipped == len(candidates):
        # 파일이 있는데 하나도 못 읽었다 — 접근 불가로 본다(비관리자 ACL 거부).
        return LogSnapshot([], 0, readable=False, files_skipped=files_skipped)
    return LogSnapshot(
        records, invalid_lines, readable=True, files_skipped=files_skipped
    )


@dataclass
class _RotatedCache:
    size: int
    mtime_ns: int
    records: list[dict]
    invalid_lines: int


class LogReader:
    """관측 로그 리더 — 회전 파일은 파일별로, 활성 파일은 증분으로 파싱해 재사용한다.

    게이트웨이는 모니터의 /health 프로브까지 기록하므로 활성 파일은 폴링마다 자란다 —
    디렉터리 전체 서명 캐시는 매번 무효가 되어 회전 상한(약 80MiB)을 통째로 재파싱하게
    된다. 회전 파일(불변)은 (크기, mtime)이 같으면 파싱 결과를 재사용하고, 활성 파일은
    파일 식별자가 같고 크기가 줄지 않은 동안 마지막 오프셋 이후만 읽어 이어 붙인다.
    잘린 마지막 줄은 다음 읽기로 이월하고, 동시 조회는 잠금으로 직렬화한다.
    """

    def __init__(self, log_directory: Path) -> None:
        self._log_directory = log_directory
        self._lock = threading.Lock()
        self._rotated: dict[str, _RotatedCache] = {}
        self._active_identity: int | None = None
        self._active_offset = 0
        self._active_tail = b""
        self._active_records: list[dict] = []
        self._active_invalid = 0

    def read(self) -> LogSnapshot:
        with self._lock:
            return self._read_locked()

    def _read_locked(self) -> LogSnapshot:
        try:
            if not self._log_directory.is_dir():
                return LogSnapshot([], 0, readable=False, files_skipped=0)
            candidates = {
                rotation_age(path): path
                for path in self._log_directory.glob(f"{LOG_FILE_NAME}*")
                if rotation_age(path) >= 0
            }
        except OSError:
            return LogSnapshot([], 0, readable=False, files_skipped=0)
        records: list[dict] = []
        invalid_lines = 0
        files_skipped = 0
        for age in sorted((age for age in candidates if age > 0), reverse=True):
            cached = self._read_rotated(candidates[age])
            if cached is None:
                files_skipped += 1
                continue
            records.extend(cached.records)
            invalid_lines += cached.invalid_lines
        # 사라진 회전 파일의 캐시는 버린다 — 삭제된 로그를 계속 보고하지 않는다.
        current_names = {path.name for path in candidates.values()}
        self._rotated = {
            name: cached
            for name, cached in self._rotated.items()
            if name in current_names
        }
        active_path = candidates.get(0)
        if active_path is None:
            # 활성 파일이 없다(삭제 또는 회전 직후 공백) — 누적분은 회전 캐시가 대신한다.
            self._reset_active(None)
        elif not self._read_active_increment(active_path):
            files_skipped += 1
        records.extend(self._active_records)
        invalid_lines += self._active_invalid
        if candidates and files_skipped == len(candidates):
            return LogSnapshot([], 0, readable=False, files_skipped=files_skipped)
        return LogSnapshot(
            records, invalid_lines, readable=True, files_skipped=files_skipped
        )

    def _read_rotated(self, path: Path) -> _RotatedCache | None:
        try:
            stat = path.stat()
            cached = self._rotated.get(path.name)
            if (
                cached is not None
                and cached.size == stat.st_size
                and cached.mtime_ns == stat.st_mtime_ns
            ):
                return cached
            parsed, invalid = _parse_log_bytes(path.read_bytes())
        except OSError:
            return None
        cached = _RotatedCache(stat.st_size, stat.st_mtime_ns, parsed, invalid)
        self._rotated[path.name] = cached
        return cached

    def _read_active_increment(self, path: Path) -> bool:
        # 활성 로그는 게이트웨이가 append 전용으로 소유한다 — 같은 크기의 제자리
        # 재작성은 위협 모델 밖이라 식별자·크기 축소 검사만으로 회전·절단을 잡는다.
        try:
            with path.open("rb") as stream:
                # 경로가 아니라 열린 핸들을 stat한다 — 읽는 도중 회전해도 같은 파일 기준이다.
                stat = os.fstat(stream.fileno())
                identity = stat.st_ino
                if (
                    identity != self._active_identity
                    or stat.st_size < self._active_offset
                ):
                    # 회전·절단으로 다른 파일이 됐다 — 옛 누적은 회전 캐시가 이어받으므로 버린다.
                    self._reset_active(identity)
                stream.seek(self._active_offset)
                data = self._active_tail + stream.read()
        except OSError:
            return False
        self._active_offset += len(data) - len(self._active_tail)
        complete, separator, remainder = data.rpartition(b"\n")
        if not separator:
            self._active_tail = data
            return True
        self._active_tail = remainder
        parsed, invalid = _parse_log_bytes(complete)
        self._active_records.extend(parsed)
        self._active_invalid += invalid
        return True

    def _reset_active(self, identity: int | None) -> None:
        self._active_identity = identity
        self._active_offset = 0
        self._active_tail = b""
        self._active_records = []
        self._active_invalid = 0


def percentile(values: list[float], ratio: float) -> float:
    """최근접 순위(nearest-rank) 백분위."""
    if not values:
        raise ValueError("백분위를 계산할 값이 없다")
    ordered = sorted(values)
    rank = max(1, math.ceil(ratio * len(ordered)))
    return ordered[rank - 1]


def _is_operational_error(record: dict) -> bool:
    """운영 오류 판정 — 클라이언트 계약 오류만 제외하고 서버·업스트림 실패는 전부 센다.

    5xx·상태 미기록·미완결에 더해, 429(용량)와 로컬 실패 사유가 남은 4xx(폴백 경로
    실패 후 전달), 별칭이 해석된 404(업스트림 모델 부재)는 provider성 실패다 —
    클라이언트 오류로 분류하면 운영 오류율이 정상처럼 보인다.
    """
    status = record.get("status")
    if not isinstance(status, int) or status >= 500:
        return True
    if record.get("completed") is not True:
        return True
    if status == 429:
        return True
    if 400 <= status < 500 and record.get("local_failure_reason") is not None:
        return True
    return status == 404 and record.get("alias") is not None


def _metric_ms(record: dict, field: str) -> float | None:
    """지연 지표 값 — 음수로 기록된 손상 값은 백분위를 왜곡하므로 버린다."""
    value = record.get(field)
    if not isinstance(value, int | float) or value < 0:
        return None
    return float(value)


def _is_client_error(record: dict) -> bool:
    """클라이언트 계약 오류 — 운영 오류로 분류된 4xx는 이중 집계하지 않는다."""
    status = record.get("status")
    if not isinstance(status, int) or not 400 <= status < 500:
        return False
    return not _is_operational_error(record)


def max_concurrent_requests(requests: list[dict]) -> int:
    """시작 시각과 소요 시간만으로 동시에 처리 중이던 요청 수의 최댓값을 재구성한다."""
    boundaries: list[tuple[float, int]] = []
    for record in requests:
        started = record.get("started_at")
        duration_ms = record.get("duration_ms")
        if not isinstance(started, int | float) or not isinstance(
            duration_ms, int | float
        ):
            continue
        boundaries.append((float(started), 1))
        boundaries.append((float(started) + float(duration_ms) / 1000, -1))
    active = peak = 0
    for _, delta in sorted(boundaries):
        active += delta
        peak = max(peak, active)
    return peak


def _circuit_events_in_window(records: list[dict], window_start: float) -> list[dict]:
    """창 안의 회로 전환 이벤트 — 시각을 파싱할 수 없는 기록은 그 자리에서 거른다."""
    events: list[dict] = []
    for record in records:
        if record.get("event") != "circuit":
            continue
        raw_time = record.get("time")
        if not isinstance(raw_time, str):
            continue
        try:
            epoch = datetime.fromisoformat(raw_time).timestamp()
        except (ValueError, OverflowError, OSError):
            # Windows는 epoch 밖 시각의 timestamp()가 OSError다 — 손상 시각은 거른다.
            continue
        if epoch >= window_start:
            events.append(
                {
                    "time": raw_time,
                    "alias": record.get("alias"),
                    "state": record.get("state"),
                }
            )
    return events


def _latency_summary(values: list[float]) -> dict[str, float]:
    """지연 요약 — 표본이 없으면 None 신호 대신 빈 요약({})으로 부재를 나타낸다."""
    if not values:
        return {}
    return {
        "p50": round(percentile(values, 0.50), 2),
        "p95": round(percentile(values, 0.95), 2),
    }


def summarize_requests(
    records: list[dict], *, now_epoch: float, window_seconds: int
) -> dict:
    """창 안의 요청 기록을 운영 지표로 요약한다 — 빈 창은 0과 None으로 채운다."""
    window_start = now_epoch - window_seconds
    # 미래 시각으로 기록된 손상 행은 어느 창에도 넣지 않는다.
    window_records = [
        record
        for record in records
        if record.get("event") == "request"
        and isinstance(record.get("started_at"), int | float)
        and window_start <= record["started_at"] <= now_epoch
    ]
    # 모니터·watchdog의 /health 폴링과 /docs 조회가 추론 지표를 오염시키지 않도록
    # 추론 경로(/v1/*)만 집계한다 — 제외 건수는 non_inference_count로 드러낸다.
    requests = [
        record
        for record in window_records
        if isinstance(record.get("path"), str) and record["path"].startswith("/v1/")
    ]
    non_inference_count = len(window_records) - len(requests)
    # 최대 동시 처리는 창 시작 전에 시작해 창 안까지 실행된 요청도 겹침에 넣는다 —
    # 시작 시각 필터만 쓰면 경계에 걸친 동시 실행이 빠진다(종료가 창 시작과 같은
    # 반개방 경계와 음수 소요 시간은 제외).
    overlapping_requests = [
        record
        for record in records
        if record.get("event") == "request"
        and isinstance(record.get("path"), str)
        and record["path"].startswith("/v1/")
        and isinstance(record.get("started_at"), int | float)
        and record["started_at"] <= now_epoch
        and isinstance(record.get("duration_ms"), int | float)
        and record["duration_ms"] >= 0
        and record["started_at"] + record["duration_ms"] / 1000 > window_start
    ]
    circuit_events = _circuit_events_in_window(records, window_start)

    operational_errors = [r for r in requests if _is_operational_error(r)]
    providers: dict[str, int] = {}
    aliases: dict[str, dict] = {}
    response_start_values: list[float] = []
    duration_values: list[float] = []
    for record in requests:
        provider = record.get("provider")
        provider_key = provider if isinstance(provider, str) else "none"
        providers[provider_key] = providers.get(provider_key, 0) + 1
        alias = record.get("alias")
        if isinstance(alias, str):
            entry = aliases.setdefault(
                alias, {"count": 0, "operational_errors": 0, "durations_ms": []}
            )
            entry["count"] += 1
            if _is_operational_error(record):
                entry["operational_errors"] += 1
            if _metric_ms(record, "duration_ms") is not None:
                entry["durations_ms"].append(_metric_ms(record, "duration_ms"))
        if not _is_operational_error(record):
            response_start = _metric_ms(record, "response_start_ms")
            duration = _metric_ms(record, "duration_ms")
            if response_start is not None:
                response_start_values.append(response_start)
            if duration is not None:
                duration_values.append(duration)

    alias_summary = {}
    for alias, entry in sorted(aliases.items()):
        summary_entry: dict = {
            "count": entry["count"],
            "operational_errors": entry["operational_errors"],
        }
        if entry["durations_ms"]:
            summary_entry["total_p95_ms"] = round(
                percentile(entry["durations_ms"], 0.95), 2
            )
        alias_summary[alias] = summary_entry
    count = len(requests)
    return {
        "window_seconds": window_seconds,
        "count": count,
        "non_inference_count": non_inference_count,
        "operational_errors": len(operational_errors),
        "client_errors": sum(1 for r in requests if _is_client_error(r)),
        "error_rate": round(len(operational_errors) / count, 4) if count else 0.0,
        "streaming": sum(1 for r in requests if r.get("stream") is True),
        "providers": providers,
        "latency_ms": {
            "response_start": _latency_summary(response_start_values),
            "total": _latency_summary(duration_values),
        },
        "aliases": alias_summary,
        "max_concurrency": max_concurrent_requests(overlapping_requests),
        "circuit_events": circuit_events,
    }
