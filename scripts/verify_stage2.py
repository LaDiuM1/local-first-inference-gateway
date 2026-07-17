"""2단계 검증: OpenAI SDK 클라이언트가 Base URL만 바꿔 게이트웨이 호출에 성공하는지 확인한다.

검증 기준(docs/ROADMAP.md 2단계):
- OpenAI SDK를 사용하는 클라이언트에서 Base URL만 변경하여 요청 성공 확인

실행 전제: Ollama 기동 상태(gemma4:12b-it-qat 사용 가능). 게이트웨이는 스크립트가 직접 띄우며
실 OpenAI 폴백은 비활성화한다.
실행: uv run python scripts/verify_stage2.py
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
REQUEST_TIMEOUT_SECONDS = 120

failures: list[str] = []
AUTH = VerificationAuth.create("stage2-verifier")


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
    # 1. Base URL만 게이트웨이로 바꾼 OpenAI SDK 클라이언트 — 2단계 검증 기준의 전부다.
    client = openai.OpenAI(
        base_url=f"{GATEWAY_BASE_URL}/v1",
        api_key=AUTH.api_key,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    # 기본이 일반 모드(사고 없음)이므로 별도 옵션 없이 빠른 응답을 받는다.
    completion = client.chat.completions.create(
        model=CHAT_ALIAS,
        messages=[
            {
                "role": "user",
                "content": "게이트웨이 점검용 요청이다. 한 문장으로 답하라.",
            }
        ],
    )
    answer = completion.choices[0].message.content or ""
    check(
        "SDK 챗 응답 수신",
        bool(answer.strip()),
        f"model={completion.model}, {len(answer)}자 응답",
    )
    check(
        "응답 model 로컬 모델 일치",
        completion.model == CHAT_MODEL,
        f"model={completion.model}",
    )

    usage_total = 0
    if completion.usage is not None:
        usage_total = completion.usage.total_tokens
    check("사용량 집계 수신", usage_total > 0, f"총 {usage_total} 토큰")

    # 2. 미등록 별칭 거절 — 게이트웨이 400이 OpenAI 포맷 에러로 SDK에 해석돼야 한다.
    try:
        client.chat.completions.create(
            model="no-such-model", messages=[{"role": "user", "content": "x"}]
        )
        check("미등록 별칭 400 거절", False, "에러 없이 성공해버림")
    except openai.APIStatusError as error:
        check(
            "미등록 별칭 400 거절",
            error.status_code == 400,
            f"HTTP {error.status_code}, SDK가 OpenAI 에러로 해석",
        )
finally:
    gateway.terminate()
    gateway.wait(timeout=10)
    AUTH.close()

if failures:
    print(f"\n2단계 검증 실패: {', '.join(failures)}")
    sys.exit(1)
print("\n2단계 검증 통과 — OpenAI SDK가 Base URL 변경만으로 게이트웨이 응답을 받는다.")
