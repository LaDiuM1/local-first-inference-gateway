"""9단계 벤치마크 순수 로직 검증 — 채점·통계·검색 지표·비용 계산이 계약대로 동작한다."""

import json
import sys
from pathlib import Path

import httpx
import pytest

SCRIPTS_DIRECTORY = str(Path(__file__).resolve().parent.parent / "scripts")
if SCRIPTS_DIRECTORY not in sys.path:
    sys.path.insert(0, SCRIPTS_DIRECTORY)

from bench_stage9_core import (  # noqa: E402
    DatasetError,
    JudgeParseError,
    api_cost_usd,
    build_blind_samples,
    build_judge_messages,
    chunk_delta_content,
    chunk_usage,
    cosine_similarity,
    electricity_krw,
    keyword_match,
    load_dataset,
    marginal_energy_wh,
    parse_judge_score,
    percentile,
    rank_documents,
    render_png,
    retrieval_metrics,
    sse_data_payload,
)
from bench_stage9_report import (  # noqa: E402
    completion_token_count,
    cost_verdict,
    followup_suggestions,
    measurement_completeness,
    quality_verdict,
    single_shot_performance,
    stability_verdict,
    ttft_verdict,
    usage_rows_cost_per_query_krw,
)
from benchmark_stage9 import timed_stream_request  # noqa: E402

REAL_DATASET_PATH = Path(__file__).resolve().parent.parent / (
    "scripts/bench_stage9_dataset.json"
)


def test_percentile_uses_nearest_rank() -> None:
    values = [float(value) for value in range(1, 11)]
    assert percentile(values, 0.50) == 5.0
    assert percentile(values, 0.95) == 10.0
    assert percentile([3.0], 0.95) == 3.0


def test_percentile_rejects_empty_values() -> None:
    with pytest.raises(ValueError):
        percentile([], 0.5)


def test_keyword_match_checks_any_and_all_case_insensitively() -> None:
    expected = {"any": ["빨간", "RED"], "all": ["색"]}
    assert keyword_match("이 이미지는 red 계열의 색입니다", expected)
    assert not keyword_match("이 이미지는 파란 색입니다", expected)
    assert not keyword_match("빨간 계열입니다", expected)


def test_parse_judge_score_accepts_json_and_embedded_json() -> None:
    assert parse_judge_score('{"score": 8, "reason": "좋다"}') == 8
    assert parse_judge_score('채점 결과: {"score": 3, "reason": "부족"} 입니다') == 3


@pytest.mark.parametrize(
    "reply",
    ["점수는 8점", '{"score": 11, "reason": "x"}', '{"score": true}', '{"score": "8"}'],
)
def test_parse_judge_score_rejects_invalid_replies(reply: str) -> None:
    with pytest.raises(JudgeParseError):
        parse_judge_score(reply)


def test_retrieval_metrics_from_known_rankings() -> None:
    rankings = {
        "q1": ["d1", "d2", "d3"],
        "q2": ["d9", "d2", "d3"],
        "q3": ["d9", "d8", "d7"],
    }
    relevant = {"q1": ("d1",), "q2": ("d2",), "q3": ("d3",)}
    metrics = retrieval_metrics(rankings, relevant)
    assert metrics["recall_at_1"] == pytest.approx(1 / 3)
    assert metrics["recall_at_3"] == pytest.approx(2 / 3)
    assert metrics["mrr"] == pytest.approx((1 + 0.5 + 0) / 3)


def test_rank_documents_orders_by_cosine_similarity() -> None:
    document_vectors = {
        "same": [1.0, 0.0],
        "opposite": [-1.0, 0.0],
        "orthogonal": [0.0, 1.0],
    }
    assert rank_documents([1.0, 0.0], document_vectors) == [
        "same",
        "orthogonal",
        "opposite",
    ]


def test_cosine_similarity_rejects_zero_vector() -> None:
    with pytest.raises(ValueError):
        cosine_similarity([0.0, 0.0], [1.0, 0.0])


def test_render_png_is_deterministic_and_kind_sensitive() -> None:
    solid = {"kind": "solid", "colors": [[255, 0, 0]], "size": 32}
    split = {
        "kind": "top_bottom",
        "colors": [[255, 0, 0], [0, 0, 255]],
        "size": 32,
    }
    first, second = render_png(solid), render_png(solid)
    assert first == second
    assert first.startswith(b"\x89PNG\r\n\x1a\n")
    assert render_png(split) != first


def test_load_dataset_accepts_repository_dataset() -> None:
    dataset = load_dataset(REAL_DATASET_PATH)
    identifiers = [item.id for item in dataset.generation]
    assert len(identifiers) == len(set(identifiers))
    assert len(dataset.items_for_task("rag")) >= 10
    assert all(item.context for item in dataset.items_for_task("rag"))
    assert all(
        item.expected and item.expected.get("any")
        for item in dataset.generation
        if item.task in ("direct", "classify", "vision")
    )


def test_load_dataset_reports_broken_items(tmp_path: Path) -> None:
    broken = {
        "generation": [
            {"id": "x-01", "task": "direct", "prompt": "질문"},
            {"id": "v-01", "task": "vision", "prompt": "색?", "image": {"kind": "?"}},
        ],
        "load_prompts": [],
        "embedding": {
            "documents": [{"id": "d1", "text": "문서"}],
            "queries": [{"id": "q1", "text": "질의", "relevant": ["없는문서"]}],
        },
    }
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(broken, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(DatasetError) as error:
        load_dataset(path)
    message = str(error.value)
    assert "expected.any" in message
    assert "image kind" in message
    assert "load_prompts" in message
    assert "존재하지 않는 문서" in message


def test_sse_helpers_extract_payload_content_and_usage() -> None:
    assert sse_data_payload('data: {"a":1}') == '{"a":1}'
    assert sse_data_payload(": keep-alive") is None
    assert chunk_delta_content({"choices": [{"delta": {"content": "안"}}]}) == "안"
    assert chunk_delta_content({"choices": []}) == ""
    usage = {"prompt_tokens": 1, "completion_tokens": 2}
    assert chunk_usage({"usage": usage}) == usage
    assert chunk_usage({"usage": {"prompt_tokens": 1}}) is None


def test_judge_messages_include_context_only_when_present() -> None:
    with_context = build_judge_messages("질문", "근거 문단", "응답")
    without_context = build_judge_messages("질문", None, "응답")
    assert "제공된 근거" in with_context[1]["content"]
    assert "제공된 근거" not in without_context[1]["content"]
    assert with_context[0]["role"] == "system"


def test_cost_helpers_compute_expected_values() -> None:
    pricing = {"usd_per_million_input": 0.25, "usd_per_million_output": 2.0}
    assert api_cost_usd(1_000_000, 500_000, pricing) == pytest.approx(1.25)
    assert marginal_energy_wh(160.0, 60.0, 3600.0) == pytest.approx(100.0)
    assert marginal_energy_wh(50.0, 60.0, 3600.0) == 0.0
    assert electricity_krw(1000.0, 210.0) == pytest.approx(210.0)


def test_completion_token_count_prefers_usage_over_delta_count() -> None:
    with_usage = {"delta_count": 3, "usage": {"completion_tokens": 41}}
    without_usage = {"delta_count": 3, "usage": None}
    assert completion_token_count(with_usage) == (41, True)
    assert completion_token_count(without_usage) == (3, False)


def test_generation_speed_pools_tokens_after_first_token() -> None:
    """생성 속도는 (토큰-1)·구간 합산의 표본 가중이다 — 짧은 표본의 행별 평균 부풀림 방지."""

    def streamed_row(tokens: int, ttft: float, total: float) -> dict:
        return {
            "task": "rag",
            "ok": True,
            "ttft_seconds": ttft,
            "total_seconds": total,
            "delta_count": tokens,
            "usage": {"completion_tokens": tokens},
        }

    local_results = {
        "items": [
            streamed_row(tokens=101, ttft=1.0, total=11.0),
            streamed_row(tokens=3, ttft=1.0, total=1.1),
        ],
        "reasoning_high_probe": None,
    }
    single = single_shot_performance(local_results)
    # (100 + 2) / (10.0 + 0.1) ≈ 10.1 — 행별 비율 평균이면 (10 + 20)/2 = 15로
    # 3토큰 표본이 전체 속도를 부풀린다.
    assert single["tokens_per_second_mean"] == pytest.approx(102 / 10.1)


def load_stage_row(
    concurrency: int,
    request_count: int = 10,
    error_count: int = 0,
    gpu_spillover: bool | None = False,
) -> dict:
    return {
        "concurrency": concurrency,
        "request_count": request_count,
        "error_count": error_count,
        "gpu_spillover": gpu_spillover,
    }


def test_stability_verdict_requires_four_concurrency_and_real_measurements() -> None:
    assert stability_verdict([], log_max_overlap=8) is None
    assert (
        stability_verdict([load_stage_row(2, error_count=1)], log_max_overlap=8) is None
    )
    assert (
        stability_verdict([load_stage_row(1), load_stage_row(4)], log_max_overlap=8)
        is True
    )
    assert (
        stability_verdict(
            [load_stage_row(2, error_count=1), load_stage_row(4)], log_max_overlap=8
        )
        is False
    )
    assert (
        stability_verdict([load_stage_row(4, gpu_spillover=True)], log_max_overlap=8)
        is False
    )
    # 요청이 하나도 없는 단계는 '무오류'가 아니라 측정 누락이다.
    assert (
        stability_verdict([load_stage_row(4, request_count=0)], log_max_overlap=8)
        is None
    )
    # 적재 스냅숏이 빠지면 스필오버 없음을 단정할 수 없다.
    assert (
        stability_verdict([load_stage_row(4, gpu_spillover=None)], log_max_overlap=8)
        is None
    )
    # 관측 로그로 재구성한 실제 동시 처리가 요구 수준에 못 미치면 판정 불가다.
    assert stability_verdict([load_stage_row(4)], log_max_overlap=3) is None
    assert stability_verdict([load_stage_row(4)], log_max_overlap=None) is None
    # 한 번도 요청을 못 보낸 워커가 있거나 워커 수가 표시 동시성과 다르면 판정 불가다.
    idle_worker = {**load_stage_row(4), "worker_request_counts": [3, 3, 3, 0]}
    assert stability_verdict([idle_worker], log_max_overlap=8) is None
    short_workers = {**load_stage_row(4), "worker_request_counts": [3, 3]}
    assert stability_verdict([short_workers], log_max_overlap=8) is None
    # 양의 정수 목록이 아닌 워커 기록은 측정 형식 위반이다 — 비목록 값도 예외가 아니라
    # 판정 불가로 끝난다.
    for broken_counts in (
        [-1, 1, 1, 1],
        ["1", "1", "1", "1"],
        [True, True, True, True],
        4,
    ):
        broken = {**load_stage_row(4), "worker_request_counts": broken_counts}
        assert stability_verdict([broken], log_max_overlap=8) is None
    busy_workers = {**load_stage_row(4), "worker_request_counts": [3, 3, 3, 3]}
    assert stability_verdict([busy_workers], log_max_overlap=8) is True


def deep_quality_report(
    local: float,
    mini: float,
    nano: float,
    failures: dict[str, list[str]] | None = None,
) -> dict:
    failures = failures or {}
    scores = {"local-none": local, "mini-minimal": mini, "nano-none": nano}
    return {
        "scored": {key: {"failures": failures.get(key, [])} for key in scores},
        "judge": {key: {"judged_overall": value} for key, value in scores.items()},
    }


def test_quality_verdict_uses_deep_evaluation_judge_scores() -> None:
    comparisons = ("gpt-5-mini", "gpt-5.4-nano")
    passed, measured = quality_verdict(
        deep_quality_report(8.89, 7.67, 7.11), comparisons
    )
    assert passed is True
    assert measured == {"local": 8.89, "gpt-5-mini": 7.67, "gpt-5.4-nano": 7.11}

    assert (
        quality_verdict(deep_quality_report(6.0, 7.67, 7.11), comparisons)[0] is False
    )

    # 심층 평가 결과가 없으면 통과가 아니라 판정 불가다.
    undetermined, measured = quality_verdict(None, comparisons)
    assert undetermined is None
    assert measured["local"] is None

    # 판정 대상 구성의 수집 실패는 점수가 있어도 판정 불가다.
    with_failures = deep_quality_report(
        8.89, 7.67, 7.11, failures={"mini-minimal": ["calc-01"]}
    )
    assert quality_verdict(with_failures, comparisons)[0] is None


def test_cost_verdict_requires_every_comparison_model_and_full_measurement() -> None:
    comparisons = ("gpt-5-mini", "gpt-5.4-nano")
    full_costs = {"gpt-5-mini": 0.25, "gpt-5.4-nano": 0.15}
    no_missing = {"gpt-5-mini": 0, "gpt-5.4-nano": 0}
    assert cost_verdict(0.03, full_costs, no_missing, comparisons)[0] is True
    assert cost_verdict(0.2, full_costs, no_missing, comparisons)[0] is False
    assert cost_verdict(0.03, {"gpt-5-mini": 0.25}, no_missing, comparisons)[0] is None
    assert cost_verdict(None, full_costs, no_missing, comparisons)[0] is None
    partially_missing = {"gpt-5-mini": 0, "gpt-5.4-nano": 3}
    assert cost_verdict(0.03, full_costs, partially_missing, comparisons)[0] is None


def test_usage_rows_cost_counts_failures_and_missing_usage() -> None:
    rows = [
        {"ok": True, "usage": {"prompt_tokens": 100, "completion_tokens": 100}},
        {"ok": False, "error": "HTTP 400"},
        {"ok": True, "usage": None},
    ]
    cost, missing = usage_rows_cost_per_query_krw(rows, "gpt-5-mini")
    assert cost is not None and cost > 0
    assert missing == 2
    none_cost, all_missing = usage_rows_cost_per_query_krw(
        [{"ok": False, "error": "x"}], "gpt-5-mini"
    )
    assert none_cost is None and all_missing == 1
    unknown_cost, unknown_missing = usage_rows_cost_per_query_krw(rows, "unknown")
    assert unknown_cost is None and unknown_missing == 3


def test_ttft_verdict_is_undetermined_on_missing_samples() -> None:
    passing = {"ttft_p95_seconds": 3.2, "ttft_missing_count": 0}
    failing = {"ttft_p95_seconds": 7.0, "ttft_missing_count": 0}
    partial = {"ttft_p95_seconds": 3.2, "ttft_missing_count": 2}
    empty = {"ttft_p95_seconds": None, "ttft_missing_count": 5}
    assert ttft_verdict(passing) is True
    assert ttft_verdict(failing) is False
    assert ttft_verdict(partial) is None
    assert ttft_verdict(empty) is None


def test_followups_surface_undetermined_criteria_instead_of_all_pass() -> None:
    summary = {
        "completeness": {},
        "load_stages": [],
        "single_shot": {"ttft_p95_seconds": 3.0},
    }
    criteria = {
        "stability": {"passed": None},
        "ttft": {"passed": True},
        "quality": {"passed": True},
        "cost": {"passed": None},
    }
    suggestions = followup_suggestions(summary, criteria)
    assert any("판정 불가 항목" in line for line in suggestions)
    assert all("전 항목 충족" not in line for line in suggestions)
    all_pass = {name: {"passed": True} for name in criteria}
    passed_suggestions = followup_suggestions(summary, all_pass)
    assert any("전 항목 충족" in line for line in passed_suggestions)


def test_measurement_completeness_counts_failures_and_missing_scores() -> None:
    attempted = {
        "local(gemma4:12b-it-qat)": [
            {"id": "rag-01", "task": "rag", "ok": True},
            {"id": "sum-01", "task": "summary", "ok": False},
            {"id": "dir-01", "task": "direct", "ok": False},
        ],
        "gpt-5-mini": [
            {"id": "rag-01", "task": "rag", "ok": True},
            {"id": "dir-01", "task": "direct", "ok": True},
        ],
    }
    judge_results = {
        "passes": 2,
        "scores": [
            {"id": "rag-01", "model": "local(gemma4:12b-it-qat)", "score": 8},
            {"id": "rag-01", "model": "local(gemma4:12b-it-qat)", "score": None},
            {"id": "rag-01", "model": "gpt-5-mini", "score": 9},
            {"id": "rag-01", "model": "gpt-5-mini", "score": 9},
        ],
    }
    completeness = measurement_completeness(4, attempted, judge_results)
    local = completeness["local(gemma4:12b-it-qat)"]
    assert local["unattempted"] == 1
    assert local["generation_failures"] == 2
    assert local["auto_missing"] == 1
    assert local["judge_expected_scores"] == 2
    assert local["judge_missing_scores"] == 1
    mini = completeness["gpt-5-mini"]
    assert mini["unattempted"] == 2
    assert mini["generation_failures"] == 0
    assert mini["judge_missing_scores"] == 0


def _stream_client(body: bytes) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=body
        )

    return httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://bench.test"
    )


def test_timed_stream_request_rejects_incomplete_streams() -> None:
    payload = {"model": "chat", "messages": [{"role": "user", "content": "x"}]}
    delta_event = 'data: {"choices":[{"delta":{"content":"토큰"}}]}\n\n'.encode()
    with _stream_client(delta_event) as client:
        result = timed_stream_request(client, payload)
    assert result["ok"] is False and "[DONE]" in result["error"]

    without_content = b"data: [DONE]\n\n"
    with _stream_client(without_content) as client:
        result = timed_stream_request(client, payload)
    assert result["ok"] is False and "content" in result["error"]

    complete = (
        delta_event
        + b'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n'
        + b"data: [DONE]\n\n"
    )
    with _stream_client(complete) as client:
        result = timed_stream_request(client, payload)
    assert result["ok"] is True and result["usage"]["completion_tokens"] == 1


def test_build_blind_samples_hides_models_and_keeps_key_mapping() -> None:
    samples = [
        ("rag-01", "질문 1", {"local": "응답 A", "gpt-5-mini": "응답 B"}),
    ]
    markdown, key = build_blind_samples(samples, seed=7)
    repeat_markdown, repeat_key = build_blind_samples(samples, seed=7)
    assert markdown == repeat_markdown and key == repeat_key
    assert "local" not in markdown and "gpt-5-mini" not in markdown
    mapping = key["rag-01"]
    assert sorted(mapping.values()) == ["gpt-5-mini", "local"]
    assert set(mapping) == {"A", "B"}
