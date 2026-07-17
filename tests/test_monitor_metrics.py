"""운영 모니터 지표 집계 검증 — 로그 파싱·창 필터·오류 분류·백분위가 계약대로 동작한다."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from gateway.monitor.metrics import (
    LogReader,
    max_concurrent_requests,
    parse_log_lines,
    percentile,
    read_log_snapshot,
    summarize_requests,
)

NOW_EPOCH = 1_000_000.0


def request_record(**overrides: object) -> dict:
    record = {
        "event": "request",
        "started_at": NOW_EPOCH - 100,
        "path": "/v1/chat/completions",
        "alias": "chat",
        "stream": False,
        "status": 200,
        "provider": "local",
        "response_start_ms": 2500.0,
        "duration_ms": 4000.0,
        "completed": True,
    }
    record.update(overrides)
    return record


def circuit_record(epoch: float, state: str) -> dict:
    return {
        "event": "circuit",
        "time": datetime.fromtimestamp(epoch, tz=UTC).isoformat(),
        "alias": "chat",
        "state": state,
    }


def test_parse_log_lines_counts_broken_lines_without_failing() -> None:
    lines = [
        json.dumps({"event": "request"}),
        "",
        '{"event": "request"',
        '"문자열"',
    ]
    records, invalid_count = parse_log_lines(lines)
    assert len(records) == 1
    assert invalid_count == 2


def test_parse_log_lines_isolates_nonstandard_number_constants() -> None:
    # NaN·Infinity는 json.loads 기본값으로는 통과한다 — 지표·직렬화를 오염시키는
    # 손상 줄로 격리해야 한다.
    lines = [
        '{"event": "request", "duration_ms": NaN}',
        '{"event": "request", "response_start_ms": Infinity}',
        json.dumps({"event": "request"}),
    ]
    records, invalid_count = parse_log_lines(lines)
    assert len(records) == 1
    assert invalid_count == 2


def test_read_log_snapshot_isolates_undecodable_bytes(tmp_path: Path) -> None:
    log_path = tmp_path / "requests.jsonl"
    log_path.write_bytes(
        json.dumps({"order": 1}).encode() + b"\n" + b"\xff\xfe broken\n"
    )
    snapshot = read_log_snapshot(tmp_path)
    assert snapshot.readable is True
    assert [record["order"] for record in snapshot.records] == [1]
    assert snapshot.invalid_lines == 1


def test_load_log_records_reads_rotated_files_oldest_first(tmp_path: Path) -> None:
    (tmp_path / "requests.jsonl.2").write_text(
        json.dumps({"order": 1}) + "\n", encoding="utf-8"
    )
    (tmp_path / "requests.jsonl.1").write_text(
        json.dumps({"order": 2}) + "\n", encoding="utf-8"
    )
    (tmp_path / "requests.jsonl").write_text(
        json.dumps({"order": 3}) + "\n", encoding="utf-8"
    )
    (tmp_path / "unrelated.log").write_text("무시 대상", encoding="utf-8")
    snapshot = read_log_snapshot(tmp_path)
    assert [record["order"] for record in snapshot.records] == [1, 2, 3]
    assert snapshot.invalid_lines == 0
    assert snapshot.readable is True


def test_read_log_snapshot_marks_missing_directory_unreadable(
    tmp_path: Path,
) -> None:
    """접근 불가·경로 부재는 무트래픽(0건)이 아니라 읽기 불가로 구분해야 한다."""
    snapshot = read_log_snapshot(tmp_path / "없는-디렉터리")
    assert snapshot.records == []
    assert snapshot.readable is False


def test_read_log_snapshot_reports_empty_but_existing_directory_readable(
    tmp_path: Path,
) -> None:
    snapshot = read_log_snapshot(tmp_path)
    assert snapshot.records == []
    assert snapshot.readable is True


def test_percentile_uses_nearest_rank() -> None:
    values = [50.0, 3000.0, 4000.0, 5000.0]
    assert percentile(values, 0.50) == 3000.0
    assert percentile(values, 0.95) == 5000.0
    with pytest.raises(ValueError):
        percentile([], 0.5)


def test_max_concurrent_requests_reconstructs_overlap() -> None:
    requests = [
        request_record(started_at=100.0, duration_ms=4000.0),
        request_record(started_at=102.0, duration_ms=3000.0),
        request_record(started_at=200.0, duration_ms=1000.0),
    ]
    assert max_concurrent_requests(requests) == 2


def test_summarize_requests_filters_window_and_classifies_errors() -> None:
    records = [
        request_record(
            started_at=NOW_EPOCH - 100,
            stream=True,
            response_start_ms=2500.0,
            duration_ms=4000.0,
        ),
        request_record(
            started_at=NOW_EPOCH - 98, response_start_ms=3000.0, duration_ms=3000.0
        ),
        request_record(
            started_at=NOW_EPOCH - 300,
            alias="vision",
            provider="openai",
            local_failure_reason="local_error_status",
            response_start_ms=1000.0,
            duration_ms=5000.0,
        ),
        request_record(
            started_at=NOW_EPOCH - 200,
            status=502,
            provider=None,
            response_start_ms=None,
            duration_ms=100.0,
        ),
        request_record(
            started_at=NOW_EPOCH - 150,
            status=400,
            provider=None,
            response_start_ms=None,
            duration_ms=50.0,
        ),
        request_record(started_at=NOW_EPOCH - 90000, duration_ms=1000.0),
        circuit_record(NOW_EPOCH - 50, "open"),
        circuit_record(NOW_EPOCH - 90000, "closed"),
        {"event": "circuit", "time": "시각 아님", "alias": "chat", "state": "open"},
        {"event": "circuit", "time": None, "alias": "chat", "state": "open"},
    ]
    summary = summarize_requests(records, now_epoch=NOW_EPOCH, window_seconds=3600)
    assert summary["count"] == 5
    assert summary["operational_errors"] == 1
    assert summary["client_errors"] == 1
    assert summary["error_rate"] == pytest.approx(0.2)
    assert summary["streaming"] == 1
    assert summary["providers"] == {"local": 2, "openai": 1, "none": 2}
    assert summary["latency_ms"]["response_start"] == {"p50": 2500.0, "p95": 3000.0}
    assert summary["latency_ms"]["total"] == {"p50": 3000.0, "p95": 5000.0}
    assert summary["aliases"]["chat"]["count"] == 4
    assert summary["aliases"]["chat"]["operational_errors"] == 1
    assert summary["aliases"]["chat"]["total_p95_ms"] == 4000.0
    assert summary["aliases"]["vision"] == {
        "count": 1,
        "operational_errors": 0,
        "total_p95_ms": 5000.0,
    }
    assert summary["max_concurrency"] == 2
    assert [event["state"] for event in summary["circuit_events"]] == ["open"]

    day_summary = summarize_requests(
        records, now_epoch=NOW_EPOCH, window_seconds=86400 * 2
    )
    assert day_summary["count"] == 6
    assert len(day_summary["circuit_events"]) == 2


def test_summarize_requests_excludes_non_inference_paths() -> None:
    """모니터·watchdog의 /health 폴링이 추론 지표를 오염시키지 않아야 한다."""
    records = [
        request_record(started_at=NOW_EPOCH - 10),
        request_record(
            started_at=NOW_EPOCH - 20,
            path="/health",
            alias=None,
            status=401,
            provider=None,
            response_start_ms=0.5,
            duration_ms=1.0,
        ),
        request_record(
            started_at=NOW_EPOCH - 30,
            path="/health",
            alias=None,
            status=401,
            provider=None,
            response_start_ms=0.5,
            duration_ms=1.0,
        ),
        request_record(started_at=NOW_EPOCH - 40, path="/docs", status=200),
        request_record(started_at=NOW_EPOCH - 50, path=None),
    ]
    summary = summarize_requests(records, now_epoch=NOW_EPOCH, window_seconds=3600)
    assert summary["count"] == 1
    assert summary["non_inference_count"] == 4
    assert summary["client_errors"] == 0
    assert summary["providers"] == {"local": 1}
    assert summary["latency_ms"]["total"] == {"p50": 4000.0, "p95": 4000.0}
    assert summary["max_concurrency"] == 1


def test_log_reader_reads_appended_lines_incrementally(tmp_path: Path) -> None:
    log_path = tmp_path / "requests.jsonl"
    log_path.write_text(json.dumps({"order": 1}) + "\n", encoding="utf-8")
    reader = LogReader(tmp_path)
    assert [record["order"] for record in reader.read().records] == [1]
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"order": 2}) + "\n")
    assert [record["order"] for record in reader.read().records] == [1, 2]


def test_log_reader_carries_partial_line_to_next_read(tmp_path: Path) -> None:
    log_path = tmp_path / "requests.jsonl"
    log_path.write_text(json.dumps({"order": 1}) + "\n" + '{"order"', encoding="utf-8")
    reader = LogReader(tmp_path)
    first = reader.read()
    # 쓰기 도중 잘린 마지막 줄은 손상이 아니라 미완성이다 — 다음 읽기로 이월한다.
    assert [record["order"] for record in first.records] == [1]
    assert first.invalid_lines == 0
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(": 2}\n")
    assert [record["order"] for record in reader.read().records] == [1, 2]


def test_log_reader_survives_rotation_without_duplicating_records(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "requests.jsonl"
    log_path.write_text(json.dumps({"order": 1}) + "\n", encoding="utf-8")
    reader = LogReader(tmp_path)
    assert len(reader.read().records) == 1
    # 회전: 활성 파일이 .1로 밀려나고 새 활성 파일이 시작된다.
    log_path.rename(tmp_path / "requests.jsonl.1")
    log_path.write_text(json.dumps({"order": 2}) + "\n", encoding="utf-8")
    refreshed = reader.read()
    assert [record["order"] for record in refreshed.records] == [1, 2]


def test_summarize_requests_counts_provider_failures_as_operational() -> None:
    records = [
        # 429는 용량 실패, 별칭이 해석된 404는 업스트림 모델 부재, 로컬 실패 사유가
        # 남은 4xx는 폴백 경로 실패 후 전달 — 전부 운영 오류다.
        request_record(started_at=NOW_EPOCH - 10, status=429, duration_ms=50.0),
        request_record(started_at=NOW_EPOCH - 20, status=404, duration_ms=50.0),
        request_record(
            started_at=NOW_EPOCH - 30,
            status=400,
            local_failure_reason="local_unreachable",
            duration_ms=50.0,
        ),
        # 클라이언트 계약 오류(400)와 라우트 미매칭 404(alias 없음)는 제외한다.
        request_record(started_at=NOW_EPOCH - 40, status=400, duration_ms=50.0),
        request_record(
            started_at=NOW_EPOCH - 50, status=404, alias=None, duration_ms=50.0
        ),
    ]
    summary = summarize_requests(records, now_epoch=NOW_EPOCH, window_seconds=3600)
    assert summary["operational_errors"] == 3
    # 4xx 지표는 클라이언트 계약 오류만 남는다 — 운영 오류로 센 4xx는 이중 집계하지 않는다.
    assert summary["client_errors"] == 2


def test_summarize_requests_counts_overlap_from_before_window_start() -> None:
    # 창 시작 30초 전에 시작해 창 안까지 실행 중이던 요청도 동시 처리에 넣는다.
    records = [
        request_record(started_at=NOW_EPOCH - 3630, duration_ms=60_000.0),
        request_record(started_at=NOW_EPOCH - 3595, duration_ms=60_000.0),
    ]
    summary = summarize_requests(records, now_epoch=NOW_EPOCH, window_seconds=3600)
    assert summary["count"] == 1
    assert summary["max_concurrency"] == 2


def test_summarize_requests_handles_empty_window() -> None:
    summary = summarize_requests([], now_epoch=NOW_EPOCH, window_seconds=3600)
    assert summary["count"] == 0
    assert summary["error_rate"] == 0.0
    assert summary["latency_ms"] == {"response_start": {}, "total": {}}
    assert summary["aliases"] == {}
    assert summary["max_concurrency"] == 0
