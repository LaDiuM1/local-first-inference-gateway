"""3단계 검증: 게이트웨이의 SSE 중계가 전체 버퍼링 없이 토큰 단위 순차 수신인지 확인한다.

검증 기준(docs/ROADMAP.md 3단계):
- 클라이언트에서 응답을 토큰 단위로 순차 수신하는지 확인
  (gemma4는 thinking이 기본이라 content 앞에 reasoning 델타가 먼저 흐른다 — 전체 델타 기준으로 검증)

실행 전제: Ollama 기동 상태(gemma4:12b-it-qat 사용 가능). 게이트웨이는 스크립트가 직접 띄운다.
실행: uv run python scripts/verify_stage3.py
"""

import subprocess
import sys
import time

import httpx
import openai

GATEWAY_HOST = "127.0.0.1"
GATEWAY_PORT = 8000
GATEWAY_BASE_URL = f"http://{GATEWAY_HOST}:{GATEWAY_PORT}"
CHAT_MODEL = "gemma4:12b-it-qat"
STARTUP_TIMEOUT_SECONDS = 15
REQUEST_TIMEOUT_SECONDS = 180

failures: list[str] = []


def check(label: str, passed: bool, detail: str) -> None:
    mark = "FAIL"
    if passed:
        mark = "PASS"
    print(f"[{mark}] {label} — {detail}")
    if not passed:
        failures.append(label)


def start_gateway() -> subprocess.Popen:
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "gateway.main:app",
            "--host",
            GATEWAY_HOST,
            "--port",
            str(GATEWAY_PORT),
            "--log-level",
            "warning",
        ]
    )
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            health = httpx.get(f"{GATEWAY_BASE_URL}/health")
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
        api_key="unused",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    started = time.perf_counter()
    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "user",
                "content": "추론 게이트웨이의 역할을 세 문장으로 설명하라.",
            }
        ],
        stream=True,
    )

    delta_seconds: list[float] = []
    first_content_second: float | None = None
    content = ""
    for chunk in stream:
        elapsed = time.perf_counter() - started
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        # gemma4는 thinking이 기본이라 content 앞에 reasoning 델타가 먼저 흐른다
        if delta.content or getattr(delta, "reasoning", None):
            delta_seconds.append(elapsed)
        if delta.content:
            content += delta.content
            if first_content_second is None:
                first_content_second = elapsed

    check(
        "스트림 델타 수신",
        len(delta_seconds) > 1 and bool(content.strip()),
        f"델타 {len(delta_seconds)}개, content {len(content)}자",
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
    if first_content_second is not None:
        print(
            f"(참고) 첫 content 델타 {first_content_second:.2f}s — thinking 구간 이후 시작"
        )

    # 2. 형식이 깨진 요청은 게이트웨이가 OpenAI 규격 400으로 직접 거절한다(3단계 의결).
    broken = httpx.post(
        f"{GATEWAY_BASE_URL}/v1/chat/completions",
        content=b'{"model": broken',
        headers={"content-type": "application/json"},
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

if failures:
    print(f"\n3단계 검증 실패: {', '.join(failures)}")
    sys.exit(1)
print("\n3단계 검증 통과 — 게이트웨이가 SSE를 버퍼링 없이 토큰 단위로 중계한다.")
