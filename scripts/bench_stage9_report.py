"""9단계 벤치마크 집계·보고 — phase 결과 JSON을 효용 비교 결과표로 만든다.

결과표(report.md·report.json)와 블라인드 확인 자료(blind_samples.md·blind_key.json)를
결과 디렉터리에 쓴다. 생성 속도는 SSE 청크 수가 아니라 스트리밍 usage의 실제 토큰
수를 우선 사용하고, 측정 누락(생성 실패·심판 채점 누락·usage 부재)을 세어 결과표에
노출한다. 생성 품질 합격선은 심층 평가(bench_stage9_quality_deep.py) 결과의 블라인드
심판 점수로 판정하며, 어떤 항목이든 측정이 비거나 누락되면 통과가 아니라 판정 불가로
처리한다. 비용 비교는 로컬 부하 측정과 같은 프롬프트·출력 상한으로 수집한 API usage를
기준으로 한다. 비용 단가·환율·전기요금은 가정임을 표에 그대로 노출한다.
"""

import json
import math
import time
from pathlib import Path

from bench_stage9_core import (
    Dataset,
    api_cost_usd,
    build_blind_samples,
    electricity_krw,
    keyword_match,
    marginal_energy_wh,
    percentile,
    rank_documents,
    retrieval_metrics,
)

LOCAL_MODEL_LABEL = "local(gemma4:12b-it-qat)"
AUTO_TASKS = ("direct", "classify", "vision")
JUDGED_TASKS = ("rag", "summary")
BLIND_SAMPLE_IDS = ("rag-01", "rag-05", "sum-02", "sum-05", "dir-04", "cls-02")
BLIND_SEED = 20260717

# 생성 품질 합격선의 근거는 심층 평가(bench_stage9_quality_deep.py)의 블라인드 심판
# 점수다 — 서비스 질의 셋은 성능·부하·비용 측정에 쓰고, 그 자동·심판 채점은 참고로만
# 남긴다(전 모델이 고득점하는 천장 효과로 품질 변별 근거가 되지 못한다).
DEEP_QUALITY_LOCAL_KEY = "local-none"
DEEP_QUALITY_COMPARISON_KEYS = {
    "gpt-5-mini": "mini-minimal",
    "gpt-5.4-nano": "nano-none",
}

# 비용 가정 — 백만 토큰당 USD. 출처와 기준일을 보고서에 그대로 노출한다.
PRICING_USD_PER_MILLION = {
    "gpt-5-mini": {"usd_per_million_input": 0.25, "usd_per_million_output": 2.00},
    "gpt-5.4-nano": {"usd_per_million_input": 0.20, "usd_per_million_output": 1.25},
    "text-embedding-3-small": {
        "usd_per_million_input": 0.02,
        "usd_per_million_output": 0.0,
    },
}
PRICING_SOURCE = "developers.openai.com/api/docs/pricing (2026-07-17 확인)"
USD_TO_KRW = 1500.0
KRW_PER_KWH = 210.0

PASS_TTFT_P95_SECONDS = 5.0
PASS_QUALITY_RATIO = 0.9
PASS_STABILITY_MAX_CONCURRENCY = 4


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("평균을 낼 값이 없다")
    return sum(values) / len(values)


def completion_token_count(row: dict) -> tuple[int, bool]:
    """스트리밍 행의 생성 토큰 수 — usage가 있으면 실제 토큰, 없으면 델타 수 대체."""
    usage = row.get("usage")
    if usage and usage.get("completion_tokens") is not None:
        return usage["completion_tokens"], True
    return row.get("delta_count", 0), False


def aggregate_auto_scores(
    dataset: Dataset, texts_by_model: dict[str, dict[str, str]]
) -> dict[str, dict[str, float | None]]:
    """자동 채점 과제(direct·classify·vision)의 모델·과제별 정답률."""
    expected = {
        item.id: item.expected for item in dataset.generation if item.task in AUTO_TASKS
    }
    tasks_by_id = {item.id: item.task for item in dataset.generation}
    accuracy: dict[str, dict[str, float | None]] = {}
    for model, texts in texts_by_model.items():
        per_task: dict[str, list[bool]] = {task: [] for task in AUTO_TASKS}
        for item_id, answer in texts.items():
            if item_id in expected:
                per_task[tasks_by_id[item_id]].append(
                    keyword_match(answer, expected[item_id])
                )
        accuracy[model] = {
            task: (sum(results) / len(results) if results else None)
            for task, results in per_task.items()
        }
        graded = [value for results in per_task.values() for value in results]
        accuracy[model]["auto_overall"] = sum(graded) / len(graded) if graded else None
    return accuracy


def aggregate_judge_scores(
    judge_results: dict, tasks_by_id: dict[str, str]
) -> dict[str, dict[str, float | None]]:
    """심판 채점(1~10)의 모델·과제별 평균 — 실패 채점은 제외하고 누락으로 센다."""
    collected: dict[str, dict[str, list[int]]] = {}
    for row in judge_results["scores"]:
        if row["score"] is None:
            continue
        task = tasks_by_id[row["id"]]
        collected.setdefault(row["model"], {}).setdefault(task, []).append(row["score"])
    aggregated: dict[str, dict[str, float | None]] = {}
    for model, per_task in collected.items():
        aggregated[model] = {
            task: mean([float(v) for v in scores]) if scores else None
            for task, scores in per_task.items()
        }
        merged = [float(v) for scores in per_task.values() for v in scores]
        aggregated[model]["judged_overall"] = mean(merged) if merged else None
    return aggregated


def measurement_completeness(
    expected_item_count: int,
    attempted_by_model: dict[str, list[dict]],
    judge_results: dict,
) -> dict[str, dict[str, int]]:
    """모델별 측정 완전성 — 미시도·생성 실패·자동/심판 채점 누락을 센다.

    성공한 응답만 집계하면 실패가 많을수록 점수가 후해지고, 표본을 줄인 실행이
    완전한 측정처럼 보인다. 질의 셋 전체 대비 미시도까지 세어 결과표에 노출하고
    합격선 판정의 유효성 조건으로 사용한다.
    """
    judge_passes = judge_results.get("passes", 1)
    valid_scores: dict[str, int] = {}
    for row in judge_results["scores"]:
        if row["score"] is not None:
            valid_scores[row["model"]] = valid_scores.get(row["model"], 0) + 1
    completeness: dict[str, dict[str, int]] = {}
    for model, rows in attempted_by_model.items():
        failures = [row for row in rows if not row["ok"]]
        judged_ok = [row for row in rows if row["ok"] and row["task"] in JUDGED_TASKS]
        auto_attempted = [row for row in rows if row["task"] in AUTO_TASKS]
        auto_scored = [row for row in auto_attempted if row["ok"]]
        completeness[model] = {
            "attempted": len(rows),
            "unattempted": expected_item_count - len(rows),
            "generation_failures": len(failures),
            "auto_missing": len(auto_attempted) - len(auto_scored),
            "judge_expected_scores": len(judged_ok) * judge_passes,
            "judge_missing_scores": len(judged_ok) * judge_passes
            - valid_scores.get(model, 0),
        }
    return completeness


def single_shot_performance(local_results: dict) -> dict:
    """단건 스트리밍 성능 요약 — 측정 누락 수를 함께 산출해 판정 유효성에 쓴다."""
    streaming_attempted = [
        row for row in local_results["items"] if row["task"] != "vision"
    ]
    streamed = [
        row
        for row in streaming_attempted
        if row["ok"] and row.get("ttft_seconds") is not None
    ]
    ttft_missing = len(streaming_attempted) - len(streamed)
    ttfts = [row["ttft_seconds"] for row in streamed]
    totals = [row["total_seconds"] for row in streamed]
    # 생성 속도는 표본 가중(토큰·시간 합산)으로 낸다 — 분모가 첫 토큰 이후
    # 시간이므로 분자도 첫 토큰을 제외하며, 짧은 응답의 행별 비율을 평균하면
    # 몇 토큰짜리 표본이 전체 속도를 부풀린다.
    generation_tokens = 0
    generation_seconds = 0.0
    usage_missing = 0
    for row in streamed:
        tokens, from_usage = completion_token_count(row)
        if not from_usage:
            usage_missing += 1
        if row["total_seconds"] > row["ttft_seconds"] and tokens > 1:
            generation_tokens += tokens - 1
            generation_seconds += row["total_seconds"] - row["ttft_seconds"]
    vision_totals = [
        row["total_seconds"]
        for row in local_results["items"]
        if row["ok"] and row["task"] == "vision"
    ]
    reasoning = local_results.get("reasoning_high_probe")
    return {
        "sample_count": len(streamed),
        "ttft_missing_count": ttft_missing,
        "ttft_p50_seconds": percentile(ttfts, 0.50) if ttfts else None,
        "ttft_p95_seconds": percentile(ttfts, 0.95) if ttfts else None,
        "total_p50_seconds": percentile(totals, 0.50) if totals else None,
        "tokens_per_second_mean": (
            generation_tokens / generation_seconds if generation_seconds > 0 else None
        ),
        "usage_missing_count": usage_missing,
        "vision_total_p50_seconds": (
            percentile(vision_totals, 0.50) if vision_totals else None
        ),
        "reasoning_high_ttft_seconds": (
            reasoning.get("ttft_seconds") if reasoning and reasoning["ok"] else None
        ),
    }


def load_stage_summary(stage: dict, idle_watt: float | None) -> dict:
    ok_requests = [row for row in stage["requests"] if row["ok"]]
    error_count = len(stage["requests"]) - len(ok_requests)
    ttfts = [
        row["ttft_seconds"] for row in ok_requests if row["ttft_seconds"] is not None
    ]
    totals = [row["total_seconds"] for row in ok_requests]
    watts = [s["watt"] for s in stage["power_samples"] if s.get("watt") is not None]
    vrams = [
        s["vram_mib"] for s in stage["power_samples"] if s.get("vram_mib") is not None
    ]
    # 적재 스냅숏이 비거나 크기 필드가 빠지면 '스필오버 없음'이 아니라 측정 누락이다 —
    # 판정 불가로 둔다.
    spillover: bool | None = None

    def usable_size(value: object) -> bool:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return False
        return math.isfinite(value)

    snapshot_complete = bool(stage["ollama_ps"]) and all(
        usable_size(model.get("size_vram")) and usable_size(model.get("size"))
        for model in stage["ollama_ps"]
    )
    if snapshot_complete:
        spillover = any(
            model["size_vram"] < model["size"] for model in stage["ollama_ps"]
        )
    total_tokens = 0
    usage_missing = 0
    for row in ok_requests:
        tokens, from_usage = completion_token_count(row)
        total_tokens += tokens
        if not from_usage:
            usage_missing += 1
    summary = {
        "concurrency": stage["concurrency"],
        "stage_seconds": stage["stage_seconds"],
        "request_count": len(stage["requests"]),
        "worker_request_counts": stage.get("worker_request_counts"),
        "error_count": error_count,
        "ttft_p50_seconds": percentile(ttfts, 0.50) if ttfts else None,
        "ttft_p95_seconds": percentile(ttfts, 0.95) if ttfts else None,
        "total_p95_seconds": percentile(totals, 0.95) if totals else None,
        "throughput_tokens_per_second": (
            total_tokens / stage["stage_seconds"] if stage["stage_seconds"] else None
        ),
        "requests_per_minute": (
            len(ok_requests) / stage["stage_seconds"] * 60
            if stage["stage_seconds"]
            else None
        ),
        "usage_missing_count": usage_missing,
        "power_avg_watt": mean(watts) if watts else None,
        "vram_max_mib": max(vrams) if vrams else None,
        "gpu_spillover": spillover,
        "energy_marginal_wh": None,
        "electricity_per_request_krw": None,
    }
    if watts and idle_watt is not None and ok_requests:
        energy_wh = marginal_energy_wh(mean(watts), idle_watt, stage["stage_seconds"])
        summary["energy_marginal_wh"] = energy_wh
        summary["electricity_per_request_krw"] = electricity_krw(
            energy_wh, KRW_PER_KWH
        ) / len(ok_requests)
    return summary


def embedding_comparison(dataset: Dataset, local: dict, openai: dict) -> dict:
    relevant = {query.id: query.relevant for query in dataset.queries}
    document_ids = [doc_id for doc_id, _ in dataset.documents]
    query_ids = [query.id for query in dataset.queries]
    comparison: dict[str, dict] = {}
    for label, payload in (
        ("local", local["embedding"]),
        ("openai", openai["embedding"]),
    ):
        document_vectors = dict(zip(document_ids, payload["documents"], strict=True))
        rankings = {
            query_id: rank_documents(vector, document_vectors)
            for query_id, vector in zip(query_ids, payload["queries"], strict=True)
        }
        comparison[label] = {
            "model": payload["model"],
            **retrieval_metrics(rankings, relevant),
        }
    local_elapsed = local["embedding"].get("elapsed_seconds")
    if local_elapsed:
        text_count = len(document_ids) + len(query_ids)
        comparison["local"]["seconds_per_text"] = local_elapsed / text_count
    return comparison


def usage_rows_cost_per_query_krw(
    rows: list[dict], model: str
) -> tuple[float | None, int]:
    """응답 행들의 평균 비용(질의당 원화)과 누락(실패·usage 부재) 건수를 돌려준다."""
    pricing = PRICING_USD_PER_MILLION.get(model)
    if pricing is None:
        return None, len(rows)
    costs = [
        api_cost_usd(
            row["usage"]["prompt_tokens"], row["usage"]["completion_tokens"], pricing
        )
        for row in rows
        if row["ok"] and row.get("usage")
    ]
    missing = len(rows) - len(costs)
    if not costs:
        return None, missing
    return mean(costs) * USD_TO_KRW, missing


def stability_verdict(
    load_stages: list[dict], log_max_overlap: int | None
) -> bool | None:
    """동시 4건 이하 부하의 무오류·무스필오버 판정.

    4건 단계가 없거나, 요청이 하나도 없는 단계가 있거나, 적재 스냅숏이 빠진
    단계가 있거나, 관측 로그로 재구성한 실제 동시 처리가 요구 수준에 못 미치면
    통과가 아니라 판정 불가다 — 측정이 빈 단계는 무오류가 아니다.
    """
    relevant = [
        stage
        for stage in load_stages
        if stage["concurrency"] <= PASS_STABILITY_MAX_CONCURRENCY
    ]
    measured_required_level = any(
        stage["concurrency"] == PASS_STABILITY_MAX_CONCURRENCY for stage in relevant
    )
    if not measured_required_level:
        return None
    if log_max_overlap is None or log_max_overlap < PASS_STABILITY_MAX_CONCURRENCY:
        return None
    for stage in relevant:
        if stage["request_count"] == 0 or stage["gpu_spillover"] is None:
            return None
        # 워커별 시도 수가 있는 실행에서는 전 워커가 실제로 부하를 만들었는지도 본다 —
        # 양의 정수 목록이 아니면 측정 형식 위반이므로 판정 불가다.
        worker_counts = stage.get("worker_request_counts")
        if worker_counts is not None and (
            not isinstance(worker_counts, list)
            or len(worker_counts) != stage["concurrency"]
            or not all(
                isinstance(count, int) and not isinstance(count, bool) and count > 0
                for count in worker_counts
            )
        ):
            return None
    return all(
        stage["error_count"] == 0 and not stage["gpu_spillover"] for stage in relevant
    )


def load_deep_quality_report(path: Path) -> dict | None:
    """심층 평가 결과(report.json)를 읽는다 — 없으면 품질 판정 불가의 근거가 된다."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def quality_verdict(
    deep_report: dict | None, comparisons: tuple[str, ...]
) -> tuple[bool | None, dict]:
    """심층 평가 블라인드 심판 점수가 두 비교 모델 각각의 기준 비율 이상인지 판정한다.

    심층 평가 결과가 없거나, 판정에 쓰는 구성(로컬 일반·두 비교 일반)에 수집 실패나
    심판 점수 누락이 있으면 통과가 아니라 판정 불가다.
    """
    measured: dict[str, float | None] = {"local": None}
    for model in comparisons:
        measured[model] = None
    if not isinstance(deep_report, dict):
        return None, measured
    scored = deep_report.get("scored")
    judge = deep_report.get("judge")
    if not isinstance(scored, dict) or not isinstance(judge, dict):
        return None, measured

    def system_score(system_key: str) -> float | None:
        # fail-closed — 구성 자체가 없거나 형식이 다르면 점수가 있어도 판정 불가다.
        entry = scored.get(system_key)
        if not isinstance(entry, dict) or not isinstance(entry.get("failures"), list):
            return None
        if entry["failures"]:
            return None
        judged = judge.get(system_key)
        if not isinstance(judged, dict):
            return None
        score = judged.get("judged_overall")
        if isinstance(score, bool) or not isinstance(score, int | float):
            return None
        if not math.isfinite(score) or not 1 <= score <= 10:
            return None
        return float(score)

    local_score = system_score(DEEP_QUALITY_LOCAL_KEY)
    measured["local"] = local_score
    if local_score is None:
        return None, measured
    verdicts: list[bool] = []
    for model in comparisons:
        comparison_key = DEEP_QUALITY_COMPARISON_KEYS.get(model)
        if comparison_key is None:
            return None, measured
        comparison_score = system_score(comparison_key)
        measured[model] = comparison_score
        if comparison_score is None:
            return None, measured
        verdicts.append(local_score >= PASS_QUALITY_RATIO * comparison_score)
    return all(verdicts), measured


def cost_verdict(
    local_cost: float | None,
    api_costs: dict[str, float | None],
    api_missing: dict[str, int],
    comparisons: tuple[str, ...],
) -> tuple[bool | None, dict]:
    """질의당 로컬 전기비가 두 비교 모델의 동등 조건 API 환산액보다 낮은지 판정한다.

    비교 모델의 부하 프로파일 요청이 하나라도 실패했거나 usage가 비면 부분 평균으로
    통과시키지 않고 판정 불가로 처리한다.
    """
    measured: dict[str, float | None] = {"local_krw": local_cost}
    for model in comparisons:
        measured[f"{model}_krw"] = api_costs.get(model)
    if local_cost is None:
        return None, measured
    verdicts: list[bool] = []
    for model in comparisons:
        api_cost = api_costs.get(model)
        if api_cost is None or api_missing.get(model, 0) > 0:
            return None, measured
        verdicts.append(local_cost < api_cost)
    return all(verdicts), measured


def ttft_verdict(single_shot: dict) -> bool | None:
    """단건 첫 토큰 p95 합격선 — 스트리밍 표본 누락이 있으면 판정 불가."""
    ttft_p95 = single_shot["ttft_p95_seconds"]
    if ttft_p95 is None or single_shot["ttft_missing_count"] > 0:
        return None
    return ttft_p95 <= PASS_TTFT_P95_SECONDS


def evaluate_pass_criteria(
    summary: dict, deep_report: dict | None, comparisons: tuple[str, ...]
) -> dict:
    """설계된 합격선을 판정한다 — 측정이 비거나 누락되면 통과가 아니라 판정 불가다."""
    stability = stability_verdict(summary["load_stages"], summary["log_max_overlap"])
    quality, quality_measured = quality_verdict(deep_report, comparisons)
    cost, cost_measured = cost_verdict(
        summary["cost"]["local_electricity_per_request_krw"],
        summary["cost"]["api_cost_per_query_krw"],
        summary["cost"]["api_cost_missing"],
        comparisons,
    )
    return {
        "stability": {
            "passed": stability,
            "criteria": "동시 4건 이하 부하에서 오류·GPU 스필오버 0 (4건 단계 필수)",
        },
        "ttft": {
            "passed": ttft_verdict(summary["single_shot"]),
            "criteria": (
                f"단건 스트리밍 첫 토큰 p95 <= {PASS_TTFT_P95_SECONDS}초"
                " (측정 누락 시 판정 불가)"
            ),
            "measured": summary["single_shot"]["ttft_p95_seconds"],
        },
        "quality": {
            "passed": quality,
            "criteria": (
                "심층 평가 블라인드 심판 점수가 두 비교 모델 각각의"
                f" {PASS_QUALITY_RATIO:.0%} 이상 (심층 평가 결과 없음·수집 실패·"
                "채점 누락 시 판정 불가)"
            ),
            "measured": quality_measured,
        },
        "cost": {
            "passed": cost,
            "criteria": "질의당 로컬 한계 전기비 < 두 비교 모델의 동등 조건 API 환산액",
            "measured": cost_measured,
        },
    }


def followup_suggestions(summary: dict, criteria: dict) -> list[str]:
    suggestions: list[str] = []
    if criteria["stability"]["passed"] is False:
        suggestions.append(
            "동시 4건 이하에서 오류 또는 GPU 스필오버 발생 — 게이트웨이 동시성 상한"
            " 도입 또는 컨텍스트 축소 검토"
        )
    if criteria["ttft"]["passed"] is False:
        suggestions.append("단건 첫 토큰 p95 초과 — 프롬프트 길이·모델 상주 정책 점검")
    if criteria["quality"]["passed"] is False:
        suggestions.append(
            "품질 합격선 미달 — 미달 과제 유형의 외부 우선 라우팅(폴백 모델 상시 사용) 검토"
        )
    if (
        criteria["quality"]["passed"] is None
        and criteria["quality"]["measured"]["local"] is None
    ):
        suggestions.append(
            "생성 품질 심층 평가 결과 없음 — bench_stage9_quality_deep.py 실행 후"
            " --quality-report로 연결"
        )
    incomplete_models = [
        model
        for model, counts in summary["completeness"].items()
        if counts["unattempted"] > 0
        or counts["generation_failures"] > 0
        or counts["judge_missing_scores"] > 0
    ]
    if incomplete_models:
        suggestions.append(
            f"측정 누락 발생({', '.join(incomplete_models)}) — 재실행으로 완전한"
            " 측정 확보 후 판정"
        )
    undetermined = [
        name for name, verdict in criteria.items() if verdict["passed"] is None
    ]
    if undetermined:
        suggestions.append(
            f"판정 불가 항목({', '.join(undetermined)}) — 누락 측정을 보완해 재실행"
            " 후 판정"
        )
    high_load = [stage for stage in summary["load_stages"] if stage["concurrency"] >= 8]
    single = summary["single_shot"]
    for stage in high_load:
        if (
            stage["ttft_p95_seconds"] is not None
            and single["ttft_p95_seconds"] is not None
            and stage["ttft_p95_seconds"] > single["ttft_p95_seconds"] * 3
        ):
            suggestions.append(
                f"동시 {stage['concurrency']}건에서 첫 토큰 p95가 단건의 3배 초과 —"
                " 라우팅 정책상 동시성 한계선 후보"
            )
    all_passed = all(verdict["passed"] is True for verdict in criteria.values())
    if all_passed and not suggestions:
        suggestions.append("합격선 전 항목 충족 — 현행 라우팅·폴백 정책 유지 근거 확보")
    return suggestions


def _cost_summary(openai: dict, single_stage: dict | None) -> dict:
    """비용 요약 — 부하 프로파일·질의 셋 환산액과 함께 누락 건수를 산출한다."""
    profile_costs: dict[str, float | None] = {}
    profile_missing: dict[str, int] = {}
    for model, rows in openai.get("load_profile", {}).items():
        profile_costs[model], profile_missing[model] = usage_rows_cost_per_query_krw(
            rows, model
        )
    reference_costs: dict[str, float | None] = {}
    for model, rows in openai["chat"].items():
        reference_costs[model], _ = usage_rows_cost_per_query_krw(rows, model)
    return {
        "local_electricity_per_request_krw": (
            single_stage["electricity_per_request_krw"] if single_stage else None
        ),
        "api_cost_per_query_krw": profile_costs,
        "api_cost_missing": profile_missing,
        "api_cost_generation_reference_krw": reference_costs,
        "assumptions": (
            f"환율 {USD_TO_KRW:.0f}원/USD, 전기 {KRW_PER_KWH:.0f}원/kWh,"
            f" 단가 출처: {PRICING_SOURCE}"
        ),
    }


def format_number(value: float | None, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}{suffix}"


def _spillover_label(spillover: bool | None) -> str:
    if spillover is None:
        return "측정 없음"
    if spillover:
        return "있음"
    return "없음"


def render_report_markdown(summary: dict, condition_notes: list[str]) -> str:
    lines = [
        "# 9단계 성능 벤치마크 결과표",
        "",
        f"생성: {summary['generated_at']}",
        "",
        "## 측정 조건",
        "",
    ]
    lines.extend(f"- {note}" for note in condition_notes)
    single = summary["single_shot"]
    lines += [
        "",
        "## 성능 — 로컬 단건 (게이트웨이 경유, 스트리밍)",
        "",
        "| 지표 | 값 |",
        "|---|---|",
        f"| 표본 수(측정 누락) | {single['sample_count']}"
        f"({single['ttft_missing_count']}) |",
        f"| 첫 토큰 p50 / p95 | {format_number(single['ttft_p50_seconds'])}초 /"
        f" {format_number(single['ttft_p95_seconds'])}초 |",
        f"| 전체 응답 p50 | {format_number(single['total_p50_seconds'])}초 |",
        f"| 생성 속도 평균(usage 토큰 기준) |"
        f" {format_number(single['tokens_per_second_mean'], 1)} tok/s |",
        f"| vision 버퍼 p50 | {format_number(single['vision_total_p50_seconds'])}초 |",
        f"| reasoning high 첫 토큰(참고) |"
        f" {format_number(single['reasoning_high_ttft_seconds'])}초 |",
        "",
        "## 성능 — 동시 부하 (스트리밍, max_tokens 제한)",
        "",
        "| 동시 | 요청 | 오류 | 첫 토큰 p50/p95(초) | 전체 p95(초) |"
        " 처리량(tok/s) | 평균 전력(W) | 최대 VRAM(MiB) | 스필오버 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for stage in summary["load_stages"]:
        lines.append(
            f"| {stage['concurrency']} | {stage['request_count']} |"
            f" {stage['error_count']} |"
            f" {format_number(stage['ttft_p50_seconds'])}/"
            f"{format_number(stage['ttft_p95_seconds'])} |"
            f" {format_number(stage['total_p95_seconds'])} |"
            f" {format_number(stage['throughput_tokens_per_second'], 1)} |"
            f" {format_number(stage['power_avg_watt'], 1)} |"
            f" {format_number(stage['vram_max_mib'], 0)} |"
            f" {_spillover_label(stage['gpu_spillover'])} |"
        )
    lines += [
        "",
        f"- 관측 로그 재구성 최대 동시 처리: {summary['log_max_overlap']}건,"
        f" 유휴 전력: {format_number(summary['idle_watt'], 1)}W",
        "",
        "## 측정 완전성",
        "",
        "| 모델 | 시도(미시도) | 생성 실패 | 자동 채점 누락 | 심판 채점 누락(기대/누락) |",
        "|---|---|---|---|---|",
    ]
    for model, counts in summary["completeness"].items():
        lines.append(
            f"| {model} | {counts['attempted']}({counts['unattempted']}) |"
            f" {counts['generation_failures']} |"
            f" {counts['auto_missing']} |"
            f" {counts['judge_expected_scores']}/{counts['judge_missing_scores']} |"
        )
    lines += [
        "",
        f"- 로컬 스트리밍 usage 누락: 단건 {single['usage_missing_count']}건,"
        " 부하 단계별 "
        + ", ".join(
            f"동시{stage['concurrency']} {stage['usage_missing_count']}건"
            for stage in summary["load_stages"]
        ),
        "- 비용 부하 프로파일 누락: "
        + ", ".join(
            f"{model} {missing}건"
            for model, missing in summary["cost"]["api_cost_missing"].items()
        ),
        "",
        "## 품질 — 생성 품질 판정 근거 (심층 평가)",
        "",
        f"- 근거 경로: {summary['quality_deep_report_path']}",
        "- 판정 점수: "
        + ", ".join(
            f"{name} {format_number(value)}"
            for name, value in summary["pass_criteria"]["quality"]["measured"].items()
        ),
        "",
        "## 품질 — 서비스 셋 심판 채점 (1~10, 모델명 비공개 절대 채점 2회 평균, 참고)",
        "",
        "| 모델 | rag | summary | 종합 |",
        "|---|---|---|---|",
    ]
    for model, scores in summary["quality_judged"].items():
        lines.append(
            f"| {model} | {format_number(scores.get('rag'))} |"
            f" {format_number(scores.get('summary'))} |"
            f" {format_number(scores.get('judged_overall'))} |"
        )
    lines += [
        "",
        f"- 심판 모델: {summary['judge_model']}"
        + (
            " (비교 대상과 같은 계열 — 자기선호 편향 가능성 명시)"
            if summary["judge_self_preference_bias"]
            else ""
        ),
        "",
        "## 품질 — 서비스 셋 자동 채점 정답률 (참고)",
        "",
        "| 모델 | direct | classify | vision | 종합 |",
        "|---|---|---|---|---|",
    ]
    for model, scores in summary["quality_auto"].items():
        lines.append(
            f"| {model} | {format_number(scores.get('direct'))} |"
            f" {format_number(scores.get('classify'))} |"
            f" {format_number(scores.get('vision'))} |"
            f" {format_number(scores.get('auto_overall'))} |"
        )
    lines += [
        "",
        "## 임베딩 검색 품질 (동일 질의·문서 셋)",
        "",
        "| provider | 모델 | Recall@1 | Recall@3 | MRR |",
        "|---|---|---|---|---|",
    ]
    for label, metrics in summary["embedding"].items():
        lines.append(
            f"| {label} | {metrics['model']} | {format_number(metrics['recall_at_1'])} |"
            f" {format_number(metrics['recall_at_3'])} | {format_number(metrics['mrr'])} |"
        )
    cost = summary["cost"]
    lines += [
        "",
        "## 비용 (질의당, 가정 명시)",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        "| 로컬 한계 전기비(동시 1, 부하 프로파일) |"
        f" {format_number(cost['local_electricity_per_request_krw'], 3)}원 |",
    ]
    for model, value in cost["api_cost_per_query_krw"].items():
        lines.append(
            f"| {model} API 환산(동일 부하 프로파일) | {format_number(value, 3)}원 |"
        )
    for model, value in cost["api_cost_generation_reference_krw"].items():
        lines.append(
            f"| {model} API 환산(서비스 질의 셋, 참고) | {format_number(value, 3)}원 |"
        )
    lines += [
        f"| 가정 | {cost['assumptions']} |",
        "",
        "## 합격선 판정",
        "",
        "| 항목 | 기준 | 결과 |",
        "|---|---|---|",
    ]
    for name, verdict in summary["pass_criteria"].items():
        state = {True: "통과", False: "미달", None: "판정 불가"}[verdict["passed"]]
        lines.append(f"| {name} | {verdict['criteria']} | {state} |")
    lines += ["", "## 후속 조정 근거", ""]
    lines.extend(f"- {suggestion}" for suggestion in summary["followups"])
    lines.append("")
    return "\n".join(lines)


def build_report_files(config, dataset: Dataset, outputs: dict) -> dict:
    """phase 결과를 집계해 결과표와 블라인드 자료를 결과 디렉터리에 쓴다."""
    local, load = outputs["local"], outputs["load"]
    openai, judge = outputs["openai"], outputs["judge"]
    comparisons: tuple[str, ...] = tuple(config.comparison_models)
    tasks_by_id = {item.id: item.task for item in dataset.generation}
    attempted_by_model: dict[str, list[dict]] = {
        LOCAL_MODEL_LABEL: local["items"],
        **openai["chat"],
    }
    texts_by_model: dict[str, dict[str, str]] = {
        model: {row["id"]: row.get("text", "") for row in rows if row["ok"]}
        for model, rows in attempted_by_model.items()
    }
    idle_watts = [
        sample["watt"]
        for sample in load["idle_power_samples"]
        if sample.get("watt") is not None
    ]
    idle_watt = mean(idle_watts) if idle_watts else None
    load_stages = [load_stage_summary(stage, idle_watt) for stage in load["stages"]]
    single_stage = next(
        (stage for stage in load_stages if stage["concurrency"] == 1), None
    )
    deep_report = load_deep_quality_report(config.quality_report_path)
    summary: dict = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "single_shot": single_shot_performance(local),
        "load_stages": load_stages,
        "log_max_overlap": load["log_max_overlap"],
        "idle_watt": idle_watt,
        "completeness": measurement_completeness(
            len(dataset.generation), attempted_by_model, judge
        ),
        "quality_deep_report_path": str(config.quality_report_path),
        "quality_judged": aggregate_judge_scores(judge, tasks_by_id),
        "quality_auto": aggregate_auto_scores(dataset, texts_by_model),
        "embedding": embedding_comparison(dataset, local, openai),
        "judge_model": judge["judge_model"],
        "judge_self_preference_bias": judge["self_preference_bias"],
        "cost": _cost_summary(openai, single_stage),
    }
    summary["pass_criteria"] = evaluate_pass_criteria(summary, deep_report, comparisons)
    summary["followups"] = followup_suggestions(summary, summary["pass_criteria"])

    condition_notes = [
        "로컬 측정은 저장소 코드 게이트웨이(테스트 포트)를 실 Ollama 두 인스턴스에"
        " 연결하고 OpenAI 폴백을 비활성화한 상태 — 성공 응답 전건 provider=local 단언",
        "chat GPU Ollama(11434)는 운영 스택과 공유하는 인스턴스이며 심야 무트래픽"
        " 시간대에 측정",
        "비교 모델 응답·심판 채점은 OpenAI 직접 호출(게이트웨이 미경유), 저지연 일반"
        " 모드 — gpt-5-mini는 reasoning_effort minimal, gpt-5.4-nano는 minimal"
        " 미지원으로 none. 지연에는 인터넷 왕복이 포함되어 로컬 수치와 직접 비교"
        " 대상이 아님",
        "비용 비교의 기준은 로컬 부하 측정과 같은 프롬프트·출력 상한으로 비교 모델에"
        " 실제 요청해 얻은 usage — 서비스 질의 셋 환산은 참고 값",
        f"부하 단계는 max_tokens {getattr(config, 'load_max_tokens', '-')}로 응답"
        " 길이를 표준화",
        "질의 셋은 자체 제작 서비스 시나리오 셋 — scripts/bench_stage9_dataset.json",
    ]
    report_markdown = render_report_markdown(summary, condition_notes)
    results_directory: Path = config.results_directory
    (results_directory / "report.md").write_text(report_markdown, encoding="utf-8")
    (results_directory / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    blind_entries = []
    prompts_by_id = {item.id: item.prompt for item in dataset.generation}
    for item_id in BLIND_SAMPLE_IDS:
        responses = {
            model: texts[item_id]
            for model, texts in texts_by_model.items()
            if item_id in texts
        }
        if len(responses) >= 2:
            blind_entries.append((item_id, prompts_by_id[item_id], responses))
    blind_markdown, blind_key = build_blind_samples(blind_entries, BLIND_SEED)
    (results_directory / "blind_samples.md").write_text(
        blind_markdown, encoding="utf-8"
    )
    (results_directory / "blind_key.json").write_text(
        json.dumps(blind_key, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
