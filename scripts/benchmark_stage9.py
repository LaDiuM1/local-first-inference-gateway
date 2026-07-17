"""9단계 벤치마크 실행기 — 로컬 게이트웨이와 비교 대상의 성능·품질·비용을 실측한다.

phase(local·load·openai·judge·report)별 결과를 JSON으로 캐시하며, 캐시가 있으면
재사용하므로 중단 후 같은 명령으로 이어서 실행할 수 있다(전체 재실행은 --force).
로컬 측정 게이트웨이는 브랜치 코드로 기동하고 OpenAI 폴백을 비활성화해, 모든 성공
응답이 provider=local임을 관측 기록으로 단언한다. 비교 응답·심판은 OpenAI 직접 호출이다.

실행: uv run python scripts/benchmark_stage9.py [--phases ...] [--sample N]
결정적 검증(fake 업스트림)은 verify_stage9.py가 run_benchmark를 호출해 수행한다.
"""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from bench_stage9_core import (
    Dataset,
    GenerationItem,
    build_judge_messages,
    chunk_delta_content,
    chunk_usage,
    image_data_url,
    load_dataset,
    parse_judge_score,
    sse_data_payload,
)
from verification_auth import UVICORN_APPLICATION_ARGUMENTS, VerificationAuth

HOST = "127.0.0.1"
STARTUP_TIMEOUT_SECONDS = 20
LOCAL_MODEL_LABEL = "local(gemma4:12b-it-qat)"
COMPARISON_MODELS = ("gpt-5-mini", "gpt-5.4-nano")
EMBEDDING_COMPARISON_MODEL = "text-embedding-3-small"
JUDGE_PREFERENCE = ("gpt-5.2", "gpt-5.1", "gpt-5-chat", "gpt-5-2", "gpt-5")
JUDGE_PASSES = 2
EMBEDDING_BATCH_SIZE = 16


@dataclass(frozen=True)
class BenchConfig:
    dataset_path: Path
    results_directory: Path
    chat_ollama_url: str = "http://127.0.0.1:11434"
    embedding_ollama_url: str = "http://127.0.0.1:11435"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    judge_model: str | None = None
    comparison_models: tuple[str, ...] = COMPARISON_MODELS
    concurrency_levels: tuple[int, ...] = (1, 2, 4, 8)
    load_seconds: float = 60.0
    load_max_tokens: int = 120
    measure_power: bool = True
    reasoning_probe: bool = True
    sample_limit: int | None = None
    request_timeout_seconds: float = 180.0
    force: bool = False
    # 생성 품질 합격선의 근거 — bench_stage9_quality_deep.py가 만든 report.json 경로.
    # 없으면 품질 판정은 통과가 아니라 판정 불가다.
    quality_report_path: Path = Path("docs/_local/stage9-bench-quality/report.json")


@dataclass
class GatewayUnderTest:
    process: subprocess.Popen
    base_url: str
    headers: dict[str, str]
    request_log_path: Path
    auth: VerificationAuth = field(repr=False)

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.auth.close()

    def request_records(self) -> list[dict]:
        try:
            text = self.request_log_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        records = [json.loads(line) for line in text.splitlines()]
        return [record for record in records if record.get("event") == "request"]


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind((HOST, 0))
        return probe.getsockname()[1]


def start_gateway(config: BenchConfig) -> GatewayUnderTest:
    """브랜치 코드 게이트웨이를 실 업스트림 대상·폴백 비활성으로 기동한다."""
    auth = VerificationAuth.create("stage9-benchmark")
    port = free_port()
    environment = {
        **os.environ,
        "GATEWAY_OLLAMA_BASE_URL": config.chat_ollama_url,
        "GATEWAY_EMBEDDING_OLLAMA_BASE_URL": config.embedding_ollama_url,
        # 빈 키는 폴백을 비활성화한다(verify_stage4와 동일) — 로컬 장애가 폴백 성공으로
        # 가려지지 않아야 성능·품질 수치가 전부 로컬 것임이 보장된다.
        "OPENAI_API_KEY": "",
    }
    auth.apply_to(environment)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            *UVICORN_APPLICATION_ARGUMENTS,
            "--host",
            HOST,
            "--port",
            str(port),
            "--log-level",
            "critical",
        ],
        env=environment,
    )
    gateway = GatewayUnderTest(
        process=process,
        base_url=f"http://{HOST}:{port}",
        headers=auth.headers,
        request_log_path=auth.store_path.parent / "request-logs" / "requests.jsonl",
        auth=auth,
    )
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            auth.close()
            raise RuntimeError("벤치마크 게이트웨이가 기동 중 종료됐다")
        try:
            health = httpx.get(
                f"{gateway.base_url}/health", headers=gateway.headers, timeout=1
            )
            if health.status_code == 200:
                return gateway
        except httpx.TransportError:
            time.sleep(0.1)
    gateway.close()
    raise RuntimeError("벤치마크 게이트웨이가 기동하지 않았다")


def chat_payload(item: GenerationItem, alias: str) -> dict:
    if item.task == "vision":
        content = [
            {"type": "text", "text": item.prompt},
            {"type": "image_url", "image_url": {"url": image_data_url(item.image)}},
        ]
        return {"model": alias, "messages": [{"role": "user", "content": content}]}
    return {
        "model": alias,
        "messages": [{"role": "user", "content": item.user_message_text()}],
    }


def timed_stream_request(client: httpx.Client, payload: dict) -> dict:
    """스트리밍 1건을 보내고 TTFT·전체 시간·델타 수·본문·usage를 수집한다."""
    started = time.monotonic()
    first_content_seconds: float | None = None
    delta_count = 0
    usage: dict | None = None
    done_received = False
    parts: list[str] = []
    try:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                **payload,
                "stream": True,
                # 생성 속도를 SSE 청크 수가 아니라 실제 토큰 수로 계산하기 위해
                # 스트리밍 마지막 청크의 usage를 요청한다(Ollama 0.31 지원 실측).
                "stream_options": {"include_usage": True},
            },
        ) as response:
            if response.status_code != 200:
                response.read()
                return {"ok": False, "error": f"HTTP {response.status_code}"}
            for line in response.iter_lines():
                data = sse_data_payload(line)
                if data is None:
                    continue
                if data == "[DONE]":
                    done_received = True
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    # 손상 SSE는 이 요청의 실패다 — 측정 루프나 부하 워커를 죽이지 않는다.
                    return {"ok": False, "error": "SSE JSON 파싱 실패"}
                usage = chunk_usage(chunk) or usage
                content = chunk_delta_content(chunk)
                if not content:
                    continue
                delta_count += 1
                parts.append(content)
                if first_content_seconds is None:
                    first_content_seconds = time.monotonic() - started
    except httpx.HTTPError as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}
    total_seconds = time.monotonic() - started
    # 빈 스트림이나 중간에 끊긴 스트림을 성공으로 세면 부하 오류율과 안정성 판정이
    # 왜곡된다 — 정상 종료([DONE])와 content 델타가 모두 있어야 성공이다.
    if not done_received:
        return {"ok": False, "error": "SSE가 [DONE] 없이 종료됨"}
    if first_content_seconds is None:
        return {"ok": False, "error": "content 델타 없이 종료됨"}
    return {
        "ok": True,
        "ttft_seconds": first_content_seconds,
        "total_seconds": total_seconds,
        "delta_count": delta_count,
        "usage": usage,
        "text": "".join(parts),
    }


def timed_buffered_request(client: httpx.Client, payload: dict) -> dict:
    started = time.monotonic()
    try:
        response = client.post("/v1/chat/completions", json=payload)
    except httpx.HTTPError as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}
    total_seconds = time.monotonic() - started
    if response.status_code != 200:
        return {"ok": False, "error": f"HTTP {response.status_code}"}
    body = response.json()
    message = body.get("choices", [{}])[0].get("message", {})
    return {
        "ok": True,
        "total_seconds": total_seconds,
        "usage": body.get("usage"),
        "text": message.get("content") or "",
        "response_model": body.get("model"),
    }


def collect_embeddings(
    client: httpx.Client, model: str, texts: list[str], path: str = "/v1/embeddings"
) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        response = client.post(
            path, json={"model": model, "input": batch, "encoding_format": "float"}
        )
        response.raise_for_status()
        data = sorted(response.json()["data"], key=lambda row: row["index"])
        vectors.extend(row["embedding"] for row in data)
    return vectors


def sampled_generation(dataset: Dataset, limit: int | None) -> list[GenerationItem]:
    if limit is None:
        return list(dataset.generation)
    items: list[GenerationItem] = []
    for task in ("rag", "direct", "summary", "classify", "vision"):
        items.extend(dataset.items_for_task(task)[:limit])
    return items


def phase_local(config: BenchConfig, dataset: Dataset) -> dict:
    """로컬 단건 수집 — 품질 응답과 스트리밍 성능을 함께 실측하고 provider를 단언한다."""
    gateway = start_gateway(config)
    items = sampled_generation(dataset, config.sample_limit)
    try:
        with httpx.Client(
            base_url=gateway.base_url,
            headers=gateway.headers,
            timeout=config.request_timeout_seconds,
        ) as client:
            warmup = timed_buffered_request(
                client,
                {
                    "model": "chat",
                    "messages": [{"role": "user", "content": "준비 확인"}],
                    "max_tokens": 8,
                },
            )
            if not warmup["ok"]:
                raise RuntimeError(f"로컬 웜업 실패: {warmup['error']}")
            results = []
            for item in items:
                alias = "vision" if item.task == "vision" else "chat"
                payload = chat_payload(item, alias)
                if item.task == "vision":
                    measured = timed_buffered_request(client, payload)
                else:
                    measured = timed_stream_request(client, payload)
                results.append({"id": item.id, "task": item.task, **measured})
            reasoning = None
            if config.reasoning_probe:
                probe_item = dataset.items_for_task("rag")[0]
                reasoning = timed_stream_request(
                    client,
                    {**chat_payload(probe_item, "chat"), "reasoning_effort": "high"},
                )
                reasoning.pop("text", None)
            document_texts = [text for _, text in dataset.documents]
            query_texts = [query.text for query in dataset.queries]
            embedding_started = time.monotonic()
            document_vectors = collect_embeddings(client, "embed", document_texts)
            query_vectors = collect_embeddings(client, "embed", query_texts)
            embedding_seconds = time.monotonic() - embedding_started
        records = gateway.request_records()
        chat_records = [
            record
            for record in records
            if record["path"] == "/v1/chat/completions" and record["status"] == 200
        ]
        non_local = [record for record in chat_records if record["provider"] != "local"]
    finally:
        gateway.close()
    if non_local:
        raise RuntimeError(f"로컬이 아닌 provider가 섞였다: {len(non_local)}건")
    return {
        "model": LOCAL_MODEL_LABEL,
        "items": results,
        "reasoning_high_probe": reasoning,
        "embedding": {
            "model": "snowflake-arctic-embed2",
            "documents": document_vectors,
            "queries": query_vectors,
            "elapsed_seconds": embedding_seconds,
        },
        "provider_assertion": {
            "checked_records": len(chat_records),
            "non_local": len(non_local),
        },
    }


def power_sampler(stop_event: threading.Event, samples: list[dict]) -> None:
    while not stop_event.is_set():
        try:
            output = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
            memory_text, power_text = output.split(",")
            samples.append({"vram_mib": float(memory_text), "watt": float(power_text)})
        except (subprocess.SubprocessError, ValueError, OSError):
            pass
        stop_event.wait(1.0)


def sample_power(seconds: float) -> list[dict]:
    stop_event = threading.Event()
    samples: list[dict] = []
    thread = threading.Thread(target=power_sampler, args=(stop_event, samples))
    thread.start()
    time.sleep(seconds)
    stop_event.set()
    thread.join()
    return samples


def ollama_ps_snapshot(chat_ollama_url: str) -> list[dict]:
    try:
        response = httpx.get(f"{chat_ollama_url}/api/ps", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
    except (httpx.HTTPError, ValueError):
        return []
    return [
        {
            "name": model.get("name"),
            "size": model.get("size"),
            "size_vram": model.get("size_vram"),
        }
        for model in models
    ]


def run_load_stage(
    gateway: GatewayUnderTest,
    config: BenchConfig,
    prompts: tuple[str, ...],
    concurrency: int,
) -> dict:
    requests: list[dict] = []
    worker_errors: list[Exception] = []
    # 워커별 시도 수 — 일부 워커가 한 번도 못 보낸 단계는 표시 동시성보다 낮게 측정된
    # 것이므로 집계에서 판정 불가로 처리할 수 있게 남긴다.
    worker_request_counts = [0] * concurrency
    lock = threading.Lock()
    power_samples: list[dict] = []
    stop_power = threading.Event()
    deadline = time.monotonic() + config.load_seconds

    def worker(worker_index: int) -> None:
        # 스레드 예외는 join()이 삼킨다 — 모아서 단계 실패로 승격하지 않으면
        # 워커가 죽은 단계가 '요청 없음 = 오류 0'으로 통과한다.
        try:
            prompt_index = worker_index
            with httpx.Client(
                base_url=gateway.base_url,
                headers=gateway.headers,
                timeout=config.request_timeout_seconds,
            ) as client:
                while time.monotonic() < deadline:
                    prompt = prompts[prompt_index % len(prompts)]
                    prompt_index += concurrency
                    measured = timed_stream_request(
                        client,
                        {
                            "model": "chat",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": config.load_max_tokens,
                        },
                    )
                    measured.pop("text", None)
                    with lock:
                        requests.append(measured)
                        worker_request_counts[worker_index] += 1
        except Exception as error:
            with lock:
                worker_errors.append(error)

    power_thread = None
    if config.measure_power:
        power_thread = threading.Thread(
            target=power_sampler, args=(stop_power, power_samples)
        )
        power_thread.start()
    stage_started = time.monotonic()
    threads = [
        threading.Thread(target=worker, args=(index,)) for index in range(concurrency)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    stage_seconds = time.monotonic() - stage_started
    if power_thread is not None:
        stop_power.set()
        power_thread.join()
    if worker_errors:
        raise RuntimeError(
            f"동시 {concurrency}건 부하 워커 {len(worker_errors)}개가 예외로 중단됨:"
            f" {worker_errors[0]!r}"
        )
    return {
        "concurrency": concurrency,
        "stage_seconds": stage_seconds,
        "requests": requests,
        "worker_request_counts": worker_request_counts,
        "power_samples": power_samples,
        "ollama_ps": ollama_ps_snapshot(config.chat_ollama_url),
    }


def max_overlap(intervals: list[tuple[float, float]]) -> int:
    boundaries: list[tuple[float, int]] = []
    for start, end in intervals:
        boundaries.append((start, 1))
        boundaries.append((end, -1))
    active = peak = 0
    for _, delta in sorted(boundaries):
        active += delta
        peak = max(peak, active)
    return peak


def phase_load(config: BenchConfig, dataset: Dataset) -> dict:
    """동시 부하 실측 — 단계별 지연 분포·오류율·VRAM·전력을 수집한다."""
    gateway = start_gateway(config)
    try:
        with httpx.Client(
            base_url=gateway.base_url,
            headers=gateway.headers,
            timeout=config.request_timeout_seconds,
        ) as client:
            warmup = timed_buffered_request(
                client,
                {
                    "model": "chat",
                    "messages": [{"role": "user", "content": "준비 확인"}],
                    "max_tokens": 8,
                },
            )
            if not warmup["ok"]:
                raise RuntimeError(f"부하 웜업 실패: {warmup['error']}")
        idle_power = sample_power(8.0) if config.measure_power else []
        stages = [
            run_load_stage(gateway, config, dataset.load_prompts, concurrency)
            for concurrency in config.concurrency_levels
        ]
        records = gateway.request_records()
        intervals = [
            (record["started_at"], record["started_at"] + record["duration_ms"] / 1000)
            for record in records
            if record["path"] == "/v1/chat/completions"
        ]
    finally:
        gateway.close()
    return {
        "idle_power_samples": idle_power,
        "stages": stages,
        "log_max_overlap": max_overlap(intervals),
    }


def openai_client(config: BenchConfig) -> httpx.Client:
    return httpx.Client(
        base_url=config.openai_base_url,
        headers={"Authorization": f"Bearer {config.openai_api_key}"},
        timeout=config.request_timeout_seconds,
    )


def openai_chat_request(client: httpx.Client, model: str, item: GenerationItem) -> dict:
    # 로컬 기본(none)이 폴백에서 minimal로 변환되는 계약과 같은 조건으로 비교한다.
    # minimal 미지원 모델(gpt-5.4-nano 실측)은 일반 모드 none으로 재시도한다 —
    # 필드 생략 시 기본 동작도 none과 동일함(reasoning 토큰 0)을 실측으로 확인함.
    payload = {**chat_payload(item, model), "reasoning_effort": "minimal"}
    started = time.monotonic()
    try:
        response = client.post("/chat/completions", json=payload)
        if response.status_code == 400:
            payload["reasoning_effort"] = "none"
            response = client.post("/chat/completions", json=payload)
    except httpx.HTTPError as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}
    total_seconds = time.monotonic() - started
    if response.status_code != 200:
        return {"ok": False, "error": f"HTTP {response.status_code}"}
    body = response.json()
    message = body.get("choices", [{}])[0].get("message", {})
    return {
        "ok": True,
        "total_seconds": total_seconds,
        "usage": body.get("usage"),
        "text": message.get("content") or "",
        "response_model": body.get("model"),
    }


def openai_load_profile_request(
    client: httpx.Client, model: str, prompt: str, max_tokens: int
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        # OpenAI 최신 모델은 max_tokens를 거부한다 — 로컬 부하의 max_tokens와 같은
        # 출력 상한을 max_completion_tokens로 지정해 동등 조건을 유지한다.
        "max_completion_tokens": max_tokens,
        "reasoning_effort": "minimal",
    }
    try:
        response = client.post("/chat/completions", json=payload)
        if response.status_code == 400:
            # minimal 미지원 모델은 일반 모드 none으로 재시도한다.
            payload["reasoning_effort"] = "none"
            response = client.post("/chat/completions", json=payload)
    except httpx.HTTPError as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}
    if response.status_code != 200:
        return {"ok": False, "error": f"HTTP {response.status_code}"}
    return {"ok": True, "usage": response.json().get("usage")}


def phase_openai(config: BenchConfig, dataset: Dataset) -> dict:
    """비교 모델 응답과 비교 임베딩을 OpenAI 직접 호출로 수집한다.

    비용 비교의 동등 조건을 위해 로컬 부하 측정과 같은 프롬프트·출력 상한
    (load_prompts·load_max_tokens)의 usage도 모델별로 함께 수집한다.
    """
    items = sampled_generation(dataset, config.sample_limit)
    with openai_client(config) as client:
        models_response = client.get("/models")
        models_response.raise_for_status()
        available_models = sorted(
            row["id"] for row in models_response.json().get("data", [])
        )
        results: dict[str, list[dict]] = {}
        load_profile: dict[str, list[dict]] = {}
        for model in config.comparison_models:
            with ThreadPoolExecutor(max_workers=6) as executor:
                measured = list(
                    executor.map(
                        lambda item, model=model: {
                            "id": item.id,
                            "task": item.task,
                            **openai_chat_request(client, model, item),
                        },
                        items,
                    )
                )
                profiled = list(
                    executor.map(
                        lambda prompt, model=model: openai_load_profile_request(
                            client, model, prompt, config.load_max_tokens
                        ),
                        dataset.load_prompts,
                    )
                )
            results[model] = measured
            load_profile[model] = profiled
        document_texts = [text for _, text in dataset.documents]
        query_texts = [query.text for query in dataset.queries]
        document_vectors = collect_embeddings(
            client, EMBEDDING_COMPARISON_MODEL, document_texts, path="/embeddings"
        )
        query_vectors = collect_embeddings(
            client, EMBEDDING_COMPARISON_MODEL, query_texts, path="/embeddings"
        )
    return {
        "available_models": available_models,
        "chat": results,
        "load_profile": load_profile,
        "embedding": {
            "model": EMBEDDING_COMPARISON_MODEL,
            "documents": document_vectors,
            "queries": query_vectors,
        },
    }


def select_judge_model(
    available: list[str], comparisons: tuple[str, ...]
) -> tuple[str, bool]:
    """가용 모델 중 비교 대상보다 상위인 심판을 고른다 — 없으면 편향 주의 플래그."""
    small_markers = ("mini", "nano", "embedding", "audio", "realtime", "image", "tts")
    for prefix in JUDGE_PREFERENCE:
        candidates = sorted(
            model
            for model in available
            if model.startswith(prefix)
            and not any(marker in model for marker in small_markers)
            and model not in comparisons
        )
        if candidates:
            return candidates[0], False
    for fallback in comparisons:
        if fallback in available:
            return fallback, True
    raise RuntimeError("심판으로 쓸 수 있는 모델이 없다")


def phase_judge(
    config: BenchConfig, dataset: Dataset, local_results: dict, openai_results: dict
) -> dict:
    """심판 채점 — 개방형(rag·summary) 응답을 모델명 없이 2회 절대 채점한다."""
    judge_model = config.judge_model
    biased = False
    if judge_model is None:
        judge_model, biased = select_judge_model(
            openai_results["available_models"], config.comparison_models
        )
    items_by_id = {item.id: item for item in dataset.generation}
    answers: list[tuple[str, str, str]] = []
    for row in local_results["items"]:
        if row["ok"] and items_by_id[row["id"]].judged:
            answers.append((row["id"], LOCAL_MODEL_LABEL, row.get("text", "")))
    for model, rows in openai_results["chat"].items():
        for row in rows:
            if row["ok"] and items_by_id[row["id"]].judged:
                answers.append((row["id"], model, row.get("text", "")))

    def score_once(client: httpx.Client, entry: tuple[str, str, str]) -> dict:
        item_id, answer_model, answer_text = entry
        item = items_by_id[item_id]
        payload = {
            "model": judge_model,
            "messages": build_judge_messages(item.prompt, item.context, answer_text),
            "reasoning_effort": "low",
        }
        try:
            response = client.post("/chat/completions", json=payload)
            if response.status_code == 400:
                payload.pop("reasoning_effort")
                response = client.post("/chat/completions", json=payload)
            response.raise_for_status()
            reply = response.json()["choices"][0]["message"]["content"] or ""
            score = parse_judge_score(reply)
        except (httpx.HTTPError, ValueError, KeyError) as error:
            return {
                "id": item_id,
                "model": answer_model,
                "score": None,
                "error": f"{type(error).__name__}: {error}",
            }
        return {"id": item_id, "model": answer_model, "score": score}

    tasks = [entry for entry in answers for _ in range(JUDGE_PASSES)]
    with openai_client(config) as client, ThreadPoolExecutor(max_workers=8) as executor:
        scores = list(executor.map(lambda entry: score_once(client, entry), tasks))
    failed = [row for row in scores if row["score"] is None]
    if len(failed) > len(scores) // 2:
        raise RuntimeError(f"심판 채점 실패가 과반이다: {len(failed)}/{len(scores)}")
    return {
        "judge_model": judge_model,
        "self_preference_bias": biased,
        "passes": JUDGE_PASSES,
        "scores": scores,
    }


def _sha256_of_file(path: Path) -> str:
    # 줄바꿈을 정규화해 해시한다 — git autocrlf 체크아웃으로 CRLF/LF만 바뀐 동일
    # 데이터셋이 '실행 조건 변경'으로 오판되지 않게 한다.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _answers_digest(rows: list[dict]) -> str:
    """성공 응답의 (id, text) 다이제스트 — 심판 캐시가 채점한 응답과 결합을 보장한다."""
    canonical = json.dumps(
        sorted((row["id"], row.get("text", "")) for row in rows if row["ok"]),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run_benchmark(config: BenchConfig, phases: list[str]) -> dict[str, dict]:
    """요청한 phase를 실행한다 — 캐시는 실행 조건(meta)이 일치할 때만 재사용한다.

    데이터셋 해시·표본 조건·측정 대상과, judge는 채점 대상 응답의 다이제스트까지
    비교해 서로 다른 실행의 결과가 한 보고서에 섞이지 않게 한다.
    """
    dataset = load_dataset(config.dataset_path)
    config.results_directory.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict] = {}
    base_meta = {
        "dataset_sha256": _sha256_of_file(config.dataset_path),
        "sample_limit": config.sample_limit,
    }

    def cached(name: str, producer, meta: dict) -> dict:
        cache_path = config.results_directory / f"{name}_results.json"
        # tuple/list 차이로 오탐하지 않도록 JSON 왕복 후 비교한다.
        normalized_meta = json.loads(json.dumps(meta, ensure_ascii=False))
        if cache_path.exists() and not config.force:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if payload.get("meta") == normalized_meta:
                return payload["result"]
            raise RuntimeError(
                f"{name} 캐시가 현재 실행 조건과 다르다 — --force로 다시 측정하라"
            )
        if name not in phases:
            raise RuntimeError(
                f"{name} 결과 캐시가 없다 — --phases에 {name}을 포함하라"
            )
        result = producer()
        cache_path.write_text(
            json.dumps({"meta": normalized_meta, "result": result}, ensure_ascii=False),
            encoding="utf-8",
        )
        return result

    needs_local = {"local", "judge", "report"} & set(phases)
    needs_openai = {"openai", "judge", "report"} & set(phases)
    if needs_local:
        local_meta = {
            **base_meta,
            "chat_url": config.chat_ollama_url,
            "embed_url": config.embedding_ollama_url,
            "reasoning_probe": config.reasoning_probe,
        }
        outputs["local"] = cached(
            "local", lambda: phase_local(config, dataset), local_meta
        )
    if {"load", "report"} & set(phases):
        load_meta = {
            **base_meta,
            "chat_url": config.chat_ollama_url,
            "concurrency": list(config.concurrency_levels),
            "load_seconds": config.load_seconds,
            "max_tokens": config.load_max_tokens,
            "measure_power": config.measure_power,
        }
        outputs["load"] = cached("load", lambda: phase_load(config, dataset), load_meta)
    if needs_openai:
        openai_meta = {
            **base_meta,
            "base_url": config.openai_base_url,
            "models": list(config.comparison_models),
            "max_tokens": config.load_max_tokens,
        }
        outputs["openai"] = cached(
            "openai", lambda: phase_openai(config, dataset), openai_meta
        )
    if {"judge", "report"} & set(phases):
        judge_meta = {
            **base_meta,
            "judge_model_override": config.judge_model,
            "local_answers": _answers_digest(outputs["local"]["items"]),
            "comparison_answers": {
                model: _answers_digest(rows)
                for model, rows in outputs["openai"]["chat"].items()
            },
        }
        outputs["judge"] = cached(
            "judge",
            lambda: phase_judge(config, dataset, outputs["local"], outputs["openai"]),
            judge_meta,
        )
    if "report" in phases:
        from bench_stage9_report import build_report_files

        outputs["report"] = build_report_files(config, dataset, outputs)
    return outputs


def read_openai_key(env_path: Path) -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


def parse_arguments() -> tuple[BenchConfig, list[str]]:
    parser = argparse.ArgumentParser(description="9단계 벤치마크 실행기")
    parser.add_argument("--dataset", default="scripts/bench_stage9_dataset.json")
    parser.add_argument("--results-dir", default="docs/_local/stage9-bench")
    parser.add_argument("--phases", default="local,load,openai,judge,report")
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--load-seconds", type=float, default=60.0)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--skip-power", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--quality-report",
        default="docs/_local/stage9-bench-quality/report.json",
        help="생성 품질 합격선의 근거 — bench_stage9_quality_deep.py 결과 경로",
    )
    arguments = parser.parse_args()
    config = BenchConfig(
        dataset_path=Path(arguments.dataset),
        results_directory=Path(arguments.results_dir),
        openai_api_key=read_openai_key(Path(".env")),
        judge_model=arguments.judge_model,
        concurrency_levels=tuple(
            int(level) for level in arguments.concurrency.split(",")
        ),
        load_seconds=arguments.load_seconds,
        measure_power=not arguments.skip_power,
        sample_limit=arguments.sample,
        force=arguments.force,
        quality_report_path=Path(arguments.quality_report),
    )
    phases = [phase.strip() for phase in arguments.phases.split(",") if phase.strip()]
    return config, phases


if __name__ == "__main__":
    configuration, requested_phases = parse_arguments()
    run_benchmark(configuration, requested_phases)
    print(f"완료 - 결과: {configuration.results_directory}")
