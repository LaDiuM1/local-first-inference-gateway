"""9단계 결정적 검증: 벤치마크 파이프라인이 효용 비교 결과표를 산출하는지 확인한다.

게이트웨이의 두 Ollama와 비교·심판용 OpenAI를 모두 로컬 fake로 돌려 실 GPU·유료 API
없이 재실행할 수 있다. 질의 셋 무결성, provider=local 단언, 동시 처리 재구성, 심판
선택·채점 집계, 임베딩 검색 지표, 블라인드 자료 생성과 생성 품질 합격선의 심층 평가
연동까지 전 구간을 검사한다.

실행: uv run python scripts/verify_stage9.py
"""

import asyncio
import json
import socket
import sys
import threading
import time
import zlib
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

import uvicorn
from bench_stage9_core import load_dataset
from bench_stage9_quality_deep import build_quality_report
from benchmark_stage9 import BenchConfig, run_benchmark
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")

HOST = "127.0.0.1"
STARTUP_TIMEOUT_SECONDS = 15
DATASET_PATH = Path("scripts/bench_stage9_dataset.json")
JUDGE_MODEL_ID = "gpt-5.2-2026-05-01"
FAKE_ANSWER = "근거를 확인했습니다. 요청하신 내용은 정상 처리 기준으로 안내드립니다."

failures: list[str] = []


def check(label: str, passed: bool, detail: str) -> None:
    mark = "PASS"
    if not passed:
        mark = "FAIL"
        failures.append(label)
    print(f"[{mark}] {label} - {detail}")


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind((HOST, 0))
        return probe.getsockname()[1]


def deterministic_vector(text: str, dimensions: int) -> list[float]:
    generator = Random(zlib.crc32(text.encode("utf-8")))
    return [generator.uniform(-1.0, 1.0) for _ in range(dimensions)]


fake = FastAPI()


async def local_sse() -> AsyncIterator[bytes]:
    # Windows의 time.monotonic() 해상도(약 15.6ms)보다 길게 지연해 TTFT가 0으로
    # 측정되지 않게 한다 — 실측에서는 첫 토큰이 초 단위라 문제가 없다.
    await asyncio.sleep(0.05)
    for token in ("근거를 ", "확인해 ", "안내드립니다."):
        chunk = {"choices": [{"index": 0, "delta": {"content": token}}]}
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()
        await asyncio.sleep(0.02)
    usage_chunk = {
        "choices": [],
        "usage": {"prompt_tokens": 80, "completion_tokens": 30, "total_tokens": 110},
    }
    yield f"data: {json.dumps(usage_chunk)}\n\n".encode()
    yield b"data: [DONE]\n\n"


@fake.post("/v1/chat/completions")
async def fake_local_chat(request: Request) -> object:
    body = json.loads(await request.body())
    if body.get("stream") is True:
        return StreamingResponse(local_sse(), media_type="text/event-stream")
    await asyncio.sleep(0.02)
    return JSONResponse(
        {
            "id": "fake-local",
            "object": "chat.completion",
            "model": body.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": FAKE_ANSWER},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 80,
                "completion_tokens": 30,
                "total_tokens": 110,
            },
        }
    )


@fake.post("/api/embed")
async def fake_native_embed(request: Request) -> object:
    body = json.loads(await request.body())
    inputs = body["input"]
    if isinstance(inputs, str):
        inputs = [inputs]
    return JSONResponse(
        {
            "model": body["model"],
            "embeddings": [deterministic_vector(text, 1024) for text in inputs],
            "prompt_eval_count": len(inputs),
        }
    )


@fake.get("/api/ps")
async def fake_ollama_ps() -> object:
    return JSONResponse(
        {"models": [{"name": "fake-chat", "size": 100, "size_vram": 100}]}
    )


@fake.get("/openai-v1/models")
async def fake_openai_models() -> object:
    return JSONResponse(
        {
            "data": [
                {"id": "gpt-5-mini"},
                {"id": "gpt-5.4-nano"},
                {"id": JUDGE_MODEL_ID},
                {"id": "gpt-5.2-mini"},
            ]
        }
    )


@fake.post("/openai-v1/chat/completions")
async def fake_openai_chat(request: Request) -> object:
    body = json.loads(await request.body())
    content = FAKE_ANSWER
    if body.get("model", "").startswith("gpt-5.2"):
        content = '{"score": 7, "reason": "근거에 충실하고 형식을 지켰다"}'
    await asyncio.sleep(0.01)
    return JSONResponse(
        {
            "id": "fake-openai",
            "object": "chat.completion",
            "model": body.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 45,
                "total_tokens": 165,
            },
        }
    )


@fake.post("/openai-v1/embeddings")
async def fake_openai_embeddings(request: Request) -> object:
    body = json.loads(await request.body())
    inputs = body["input"]
    if isinstance(inputs, str):
        inputs = [inputs]
    data = [
        {"index": index, "embedding": deterministic_vector(text, 1536)}
        for index, text in enumerate(inputs)
    ]
    return JSONResponse(
        {"object": "list", "data": data, "model": body.get("model", "")}
    )


def start_fake(port: int) -> uvicorn.Server:
    server = uvicorn.Server(
        uvicorn.Config(fake, host=HOST, port=port, log_level="critical")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if server.started:
            return server
        time.sleep(0.05)
    raise RuntimeError("fake 업스트림이 기동하지 않았다")


def verify_dataset() -> int:
    dataset = load_dataset(DATASET_PATH)
    task_counts = {
        task: len(dataset.items_for_task(task))
        for task in ("rag", "direct", "summary", "classify", "vision")
    }
    check(
        "질의 셋 구성 무결성",
        task_counts["rag"] >= 10
        and task_counts["direct"] >= 6
        and task_counts["summary"] >= 4
        and task_counts["classify"] >= 4
        and task_counts["vision"] >= 4
        and len(dataset.documents) >= 30
        and len(dataset.queries) >= 20
        and len(dataset.load_prompts) >= 4,
        f"generation={task_counts}, documents={len(dataset.documents)},"
        f" queries={len(dataset.queries)}",
    )
    return len(dataset.generation)


def run_verification() -> None:
    generation_item_count = verify_dataset()
    fake_port = free_port()
    start_fake(fake_port)
    fake_base = f"http://{HOST}:{fake_port}"
    with TemporaryDirectory(prefix="stage9-verify-") as results_directory:
        # 심층 평가 경로도 임시 디렉터리로 고정한다 — 저장소의 실제 측정 산출물을
        # 읽으면 검증이 실행 환경에 따라 달라진다.
        deep_report_path = Path(results_directory) / "quality-deep-report.json"
        config = BenchConfig(
            dataset_path=DATASET_PATH,
            results_directory=Path(results_directory),
            chat_ollama_url=fake_base,
            embedding_ollama_url=fake_base,
            openai_base_url=f"{fake_base}/openai-v1",
            openai_api_key="stage9-fake-key",
            concurrency_levels=(2,),
            load_seconds=2.0,
            measure_power=False,
            sample_limit=1,
            request_timeout_seconds=15.0,
            quality_report_path=deep_report_path,
        )
        outputs = run_benchmark(config, ["local", "load", "openai", "judge", "report"])
        summary = outputs["report"]
        local = outputs["local"]

        check(
            "로컬 수집 provider=local 단언",
            local["provider_assertion"]["non_local"] == 0
            and local["provider_assertion"]["checked_records"] > 0,
            f"검사 기록 {local['provider_assertion']['checked_records']}건",
        )
        single = summary["single_shot"]
        check(
            "단건 성능 지표 산출(usage 토큰 기준·누락 0)",
            single["sample_count"] >= 3
            and single["ttft_p50_seconds"] > 0
            and single["tokens_per_second_mean"] > 0
            and single["usage_missing_count"] == 0
            and single["ttft_missing_count"] == 0,
            f"표본 {single['sample_count']}건, ttft_p50="
            f"{single['ttft_p50_seconds']:.3f}s, usage 누락"
            f" {single['usage_missing_count']}건",
        )
        completeness = summary["completeness"]
        expected_unattempted = generation_item_count - 5
        check(
            "측정 완전성 집계(부분 표본의 미시도 노출)",
            len(completeness) == 3
            and all(
                counts["unattempted"] == expected_unattempted
                and counts["generation_failures"] == 0
                and counts["auto_missing"] == 0
                and counts["judge_missing_scores"] == 0
                and counts["judge_expected_scores"] > 0
                for counts in completeness.values()
            ),
            f"모델 {len(completeness)}종, 미시도 {expected_unattempted}건씩 기록",
        )
        check(
            "심층 평가 결과 없는 실행의 품질 합격선 '판정 불가' 처리",
            summary["pass_criteria"]["quality"]["passed"] is None
            and any("심층 평가" in suggestion for suggestion in summary["followups"]),
            "quality_report 부재는 통과가 아니라 판정 불가 + 후속 근거로 안내",
        )
        stage = summary["load_stages"][0]
        check(
            "동시 부하 실측·관측 로그 재구성",
            stage["error_count"] == 0
            and stage["request_count"] >= 2
            and summary["log_max_overlap"] >= 2,
            f"요청 {stage['request_count']}건, 최대 동시"
            f" {summary['log_max_overlap']}건",
        )
        judged = summary["quality_judged"]
        check(
            "심판 자동 선택·채점 집계",
            summary["judge_model"] == JUDGE_MODEL_ID
            and summary["judge_self_preference_bias"] is False
            and all(scores["judged_overall"] is not None for scores in judged.values())
            and len(judged) == 3,
            f"judge={summary['judge_model']}, 모델 {len(judged)}종 집계",
        )
        auto = summary["quality_auto"]
        check(
            "자동 채점 집계",
            len(auto) == 3
            and all("auto_overall" in scores for scores in auto.values()),
            f"모델 {len(auto)}종",
        )
        embedding = summary["embedding"]
        check(
            "임베딩 검색 지표 산출(양 provider)",
            set(embedding) == {"local", "openai"}
            and all(
                0.0 <= metrics[key] <= 1.0
                for metrics in embedding.values()
                for key in ("recall_at_1", "recall_at_3", "mrr")
            ),
            f"local mrr={embedding['local']['mrr']:.2f},"
            f" openai mrr={embedding['openai']['mrr']:.2f}",
        )
        cost = summary["cost"]
        check(
            "동등 조건 비용 환산(두 비교 모델·부하 프로파일·누락 0)·가정 명시",
            cost["api_cost_per_query_krw"]["gpt-5-mini"] is not None
            and cost["api_cost_per_query_krw"]["gpt-5.4-nano"] is not None
            and all(missing == 0 for missing in cost["api_cost_missing"].values())
            and cost["api_cost_generation_reference_krw"]["gpt-5-mini"] is not None
            and "원/kWh" in cost["assumptions"],
            f"mini={cost['api_cost_per_query_krw']['gpt-5-mini']:.3f}원/질의(부하"
            " 프로파일)",
        )
        check(
            "측정 없는 합격선의 '판정 불가' 처리",
            summary["pass_criteria"]["cost"]["passed"] is None
            and summary["pass_criteria"]["stability"]["passed"] is None,
            "전력 미측정·동시 4건 단계 없음 경로",
        )
        results_path = Path(results_directory)
        report_markdown = (results_path / "report.md").read_text(encoding="utf-8")
        blind_key = json.loads(
            (results_path / "blind_key.json").read_text(encoding="utf-8")
        )
        check(
            "결과표·블라인드 자료 산출",
            "합격선 판정" in report_markdown
            and "후속 조정 근거" in report_markdown
            and (results_path / "blind_samples.md").exists()
            and len(blind_key) >= 1
            and all(len(mapping) >= 2 for mapping in blind_key.values()),
            f"report.md {len(report_markdown.splitlines())}줄,"
            f" 블라인드 {len(blind_key)}문항",
        )
        check(
            "판정 불가 항목이 후속 근거에 드러남(전 항목 충족 문구 없음)",
            "판정 불가 항목" in report_markdown
            and "합격선 전 항목 충족" not in report_markdown,
            "stability·quality·cost 판정 불가 경로의 보고 문구",
        )
        reused = run_benchmark(config, ["report"])
        check(
            "동일 조건 캐시 재사용",
            reused["report"]["single_shot"]["sample_count"]
            == summary["single_shot"]["sample_count"],
            "같은 설정의 report 재실행이 캐시로 성립",
        )
        stale_config = replace(config, sample_limit=2)
        stale_blocked = False
        try:
            run_benchmark(stale_config, ["report"])
        except RuntimeError as error:
            stale_blocked = "캐시" in str(error)
        check(
            "실행 조건 불일치 캐시 재사용 차단",
            stale_blocked,
            "sample_limit 변경 시 명시적 거부",
        )

        # fake 심층 보고서도 생산자와 같은 스키마 함수로 만든다 — 스키마가 갈라지면
        # 이 검증이 실제 파이프라인과 무관해진다.
        deep_report_path.write_text(
            json.dumps(
                build_quality_report(
                    scored={
                        "local-none": {"failures": []},
                        "mini-minimal": {"failures": []},
                        "nano-none": {"failures": []},
                    },
                    judge_result={
                        "local-none": {"judged_overall": 8.9},
                        "mini-minimal": {"judged_overall": 7.7},
                        "nano-none": {"judged_overall": 7.1},
                    },
                ),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        deep_summary = run_benchmark(config, ["report"])["report"]
        quality = deep_summary["pass_criteria"]["quality"]
        check(
            "심층 평가 연동 품질 합격선 판정",
            quality["passed"] is True
            and quality["measured"]["local"] == 8.9
            and quality["measured"]["gpt-5-mini"] == 7.7,
            f"측정 점수 {quality['measured']}",
        )
        deep_report_path.write_text(
            json.dumps(
                build_quality_report(
                    scored={
                        "local-none": {"failures": []},
                        "mini-minimal": {"failures": ["calc-01"]},
                        "nano-none": {"failures": []},
                    },
                    judge_result={
                        "local-none": {"judged_overall": 8.9},
                        "mini-minimal": {"judged_overall": 7.7},
                        "nano-none": {"judged_overall": 7.1},
                    },
                ),
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        incomplete_summary = run_benchmark(config, ["report"])["report"]
        check(
            "심층 평가 수집 실패 시 품질 '판정 불가' 처리",
            incomplete_summary["pass_criteria"]["quality"]["passed"] is None,
            "비교 구성의 수집 실패는 점수가 있어도 통과로 판정하지 않음",
        )
        deep_report_path.write_text(
            json.dumps(build_quality_report(scored={}, judge_result=None)),
            encoding="utf-8",
        )
        empty_summary = run_benchmark(config, ["report"])["report"]
        check(
            "빈 심층 보고서의 품질 'fail-closed' 처리",
            empty_summary["pass_criteria"]["quality"]["passed"] is None,
            "구성 자체가 없는 보고서는 점수 형식과 무관하게 판정 불가",
        )

    if failures:
        print(f"\n9단계 검증 실패 {len(failures)}건: {', '.join(failures)}")
        raise SystemExit(1)
    print("\n9단계 자동 검증 통과 - 벤치마크 파이프라인이 효용 비교 결과표를 산출한다.")


if __name__ == "__main__":
    run_verification()
