"""3단계 검증: 게이트웨이의 SSE 중계가 전체 버퍼링 없이 토큰 단위 순차 수신인지 확인한다.

검증 기준(docs/ROADMAP.md 3단계):
- 클라이언트에서 응답을 토큰 단위로 순차 수신하는지 확인
  (기본이 일반 모드라 사고 델타 없이 content 델타가 시간축에 분산되는지로 검증)

실행 전제: Ollama 기동 상태(gemma4:12b-it-qat 사용 가능). 게이트웨이는 스크립트가 직접 띄우며
실 OpenAI 폴백은 비활성화한다.
실행: uv run python scripts/verify_stage3.py
"""

import os
import socket
import subprocess
import sys
import time

import httpx
import openai
from verification_auth import UVICORN_APPLICATION_ARGUMENTS, VerificationAuth

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")

GATEWAY_HOST = "127.0.0.1"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((GATEWAY_HOST, 0))
        return probe.getsockname()[1]


GATEWAY_PORT = free_port()
GATEWAY_BASE_URL = f"http://{GATEWAY_HOST}:{GATEWAY_PORT}"
CHAT_ALIAS = "chat"
CHAT_MODEL = "gemma4:12b-it-qat"
STARTUP_TIMEOUT_SECONDS = 15
REQUEST_TIMEOUT_SECONDS = 180

failures: list[str] = []
AUTH = VerificationAuth.create("stage3-verifier")


def check(label: str, passed: bool, detail: str) -> None:
    mark = "FAIL"
    if passed:
        mark = "PASS"
    print(f"[{mark}] {label} — {detail}")
    if not passed:
        failures.append(label)


def start_gateway() -> subprocess.Popen:
    environment = os.environ.copy()
    # 라이브 검증은 로컬 Ollama만 대상으로 한다. 저장소 .env에 키가 있어도 외부로 우회하지 않는다.
    environment["OPENAI_API_KEY"] = ""
    AUTH.apply_to(environment)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            *UVICORN_APPLICATION_ARGUMENTS,
            "--host",
            GATEWAY_HOST,
            "--port",
            str(GATEWAY_PORT),
            "--log-level",
            "warning",
        ],
        env=environment,
    )
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            health = httpx.get(f"{GATEWAY_BASE_URL}/health", headers=AUTH.headers)
            if health.json() == {"status": "ok"}:
                return process
        except httpx.TransportError:
            time.sleep(0.3)
    process.terminate()
    raise RuntimeError("게이트웨이가 기동하지 않는다")


gateway = start_gateway()
try:
    # 1. SSE 순차 수신 — 델타 수신 시각이 시간축에 분산되는지 측정한다.
    client = openai.OpenAI(
        base_url=f"{GATEWAY_BASE_URL}/v1",
        api_key=AUTH.api_key,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    started = time.perf_counter()
    stream = client.chat.completions.create(
        model=CHAT_ALIAS,
        messages=[
            {
                "role": "user",
                "content": "추론 게이트웨이의 역할을 세 문장으로 설명하라.",
            }
        ],
        stream=True,
    )

    delta_seconds: list[float] = []
    content = ""
    chunk_models: set[str] = set()
    for chunk in stream:
        elapsed = time.perf_counter() - started
        chunk_models.add(chunk.model)
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        # 기본이 일반 모드라 사고 델타 없이 content 델타가 흐른다
        if delta.content:
            delta_seconds.append(elapsed)
            content += delta.content

    check(
        "스트림 델타 수신",
        len(delta_seconds) > 1 and bool(content.strip()),
        f"content 델타 {len(delta_seconds)}개, content {len(content)}자",
    )

    if delta_seconds:
        first, last = delta_seconds[0], delta_seconds[-1]
        spread = last - first
        # 전체 버퍼링이라면 모든 델타가 응답 종료 시점에 몰려 수신 구간이 0에 가깝다
        check(
            "토큰 단위 순차 수신",
            spread > last * 0.5,
            f"첫 델타 {first:.2f}s, 마지막 {last:.2f}s — 수신 구간 {spread:.2f}s",
        )

    check(
        "스트림 model 로컬 모델 일치",
        chunk_models == {CHAT_MODEL},
        f"수신 model {sorted(chunk_models)}",
    )

    # 2. 형식이 깨진 요청은 게이트웨이가 OpenAI 규격 400으로 직접 거절한다(3단계 의결).
    broken = httpx.post(
        f"{GATEWAY_BASE_URL}/v1/chat/completions",
        content=b'{"model": broken',
        headers={**AUTH.headers, "content-type": "application/json"},
    )
    check(
        "깨진 JSON 게이트웨이 400 거절",
        broken.status_code == 400
        and broken.json()["error"]["type"] == "invalid_request_error",
        f"HTTP {broken.status_code}",
    )
finally:
    gateway.terminate()
    gateway.wait(timeout=10)
    AUTH.close()

if failures:
    print(f"\n3단계 검증 실패: {', '.join(failures)}")
    sys.exit(1)
print("\n3단계 검증 통과 — 게이트웨이가 SSE를 버퍼링 없이 토큰 단위로 중계한다.")
