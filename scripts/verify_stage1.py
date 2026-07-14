"""1단계 검증: 확정 모델 구성이 이 하드웨어에서 성립하는지 실측으로 확인한다.

확정 구성(docs/DECISIONS.md 26.07.08):
- 추론(챗봇·이미지 분석): gemma4:12b-it-qat — 8k 컨텍스트 기준 100% GPU 적재
- 검색 임베딩: snowflake-arctic-embed2 — CPU 실행 (VRAM은 추론 모델에 올인)

실행 전제: 채팅 Ollama(기본 11434)와 CPU 임베딩 Ollama(기본 11435)가 각각 기동된 상태,
채팅 프로세스의 OLLAMA_CONTEXT_LENGTH=8192 환경변수.
표준 라이브러리만 사용한다 — 프로젝트 구성(2단계) 전에도 실행 가능해야 하기 때문이다.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request

CHAT_OLLAMA_BASE_URL = os.environ.get(
    "GATEWAY_OLLAMA_BASE_URL", "http://127.0.0.1:11434"
).rstrip("/")
EMBEDDING_OLLAMA_BASE_URL = os.environ.get(
    "GATEWAY_EMBEDDING_OLLAMA_BASE_URL", "http://127.0.0.1:11435"
).rstrip("/")
CHAT_MODEL = "gemma4:12b-it-qat"
EMBED_MODEL = "snowflake-arctic-embed2"
SERVING_CONTEXT = 8192
EMBED_DIMENSIONS = 1024
PASCAL_LAST_DRIVER_BRANCH = "582."
REQUEST_TIMEOUT_SECONDS = 300

failures: list[str] = []


def check(label: str, passed: bool, detail: str) -> None:
    mark = "FAIL"
    if passed:
        mark = "PASS"
    print(f"[{mark}] {label} — {detail}")
    if not passed:
        failures.append(label)


def get_ollama(base_url: str, path: str) -> dict:
    with urllib.request.urlopen(
        base_url + path, timeout=REQUEST_TIMEOUT_SECONDS
    ) as response:
        return json.load(response)


def post_ollama(base_url: str, path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        base_url + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return json.load(response)


def read_gpu_stats(*fields: str) -> list[str]:
    query = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return [value.strip() for value in query.stdout.strip().split(",")]


# 1. 드라이버 — Pascal 지원 마지막 브랜치(582.x) 확인. 이전 브랜치는 신형 CUDA 커널을 컴파일하지 못한다.
driver_version = read_gpu_stats("driver_version")[0]
check(
    "드라이버 582.x",
    driver_version.startswith(PASCAL_LAST_DRIVER_BRANCH),
    f"현재 {driver_version}",
)

# 2. 추론 모델 — 서빙 기본 설정 그대로 적재하고 생성 속도를 잰다.
generation = post_ollama(
    CHAT_OLLAMA_BASE_URL,
    "/api/generate",
    {
        "model": CHAT_MODEL,
        "prompt": "추론 서버 상태 점검용 요청이다. 한 문장으로 답하라.",
        "stream": False,
    },
)
generation_speed = (
    generation["eval_count"] / generation["eval_duration"] * 1_000_000_000
)
check("추론 모델 생성", generation["eval_count"] > 0, f"{generation_speed:.1f} tok/s")

# 3. 임베딩 모델 — CPU로 적재(num_gpu 0)하고 차원과 단건 지연을 확인한다.
embed_started = time.perf_counter()
embedding = post_ollama(
    EMBEDDING_OLLAMA_BASE_URL,
    "/api/embed",
    {
        "model": EMBED_MODEL,
        "input": "상품 검색 임베딩 검증용 문장이다.",
        "options": {"num_gpu": 0},
    },
)
embed_latency_ms = (time.perf_counter() - embed_started) * 1000
dimensions = len(embedding["embeddings"][0])
check(
    "임베딩 차원",
    dimensions == EMBED_DIMENSIONS,
    f"{dimensions}차원, 단건 {embed_latency_ms:.0f}ms",
)

# 4. 동시 상주 — 분리된 두 프로세스에서 각 모델의 적재 위치와 서빙 컨텍스트를 확인한다.
chat_resident = {
    model["name"].removesuffix(":latest"): model
    for model in get_ollama(CHAT_OLLAMA_BASE_URL, "/api/ps")["models"]
}
embedding_resident = {
    model["name"].removesuffix(":latest"): model
    for model in get_ollama(EMBEDDING_OLLAMA_BASE_URL, "/api/ps")["models"]
}

if CHAT_MODEL in chat_resident:
    chat_state = chat_resident[CHAT_MODEL]
    chat_context = chat_state.get("context_length", 0)
    fully_on_gpu = chat_state["size_vram"] == chat_state["size"]
    check(
        "추론 모델 100% GPU",
        fully_on_gpu,
        f"적재 {chat_state['size'] / 2**30:.1f}GiB 중 VRAM {chat_state['size_vram'] / 2**30:.1f}GiB",
    )
    check("서빙 컨텍스트 8k", chat_context == SERVING_CONTEXT, f"현재 {chat_context}")
else:
    check("추론 모델 100% GPU", False, "적재되어 있지 않음")

if EMBED_MODEL in embedding_resident:
    embed_state = embedding_resident[EMBED_MODEL]
    check(
        "임베딩 모델 100% CPU",
        embed_state["size_vram"] == 0,
        f"VRAM 점유 {embed_state['size_vram'] / 2**20:.0f}MiB",
    )
else:
    check("임베딩 모델 100% CPU", False, "적재되어 있지 않음")

# 5. VRAM 여유 — 동시 상주 상태의 실측 기록 (판정이 아니라 기록 목적)
memory_used, memory_total = read_gpu_stats("memory.used", "memory.total")
print(
    f"[INFO] VRAM {memory_used}/{memory_total} MiB 사용 — 여유 {int(memory_total) - int(memory_used)} MiB"
)

if failures:
    print(f"\n1단계 검증 실패: {', '.join(failures)}")
    sys.exit(1)
print("\n1단계 검증 통과 — 확정 모델 구성이 이 하드웨어에서 성립한다.")
