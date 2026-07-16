"""4단계 검증: 별칭 라우팅과 CPU 임베딩이 실제 Ollama 기준으로 성립하는지 확인한다.

검증 기준(docs/ROADMAP.md 4단계):
- 서버 설정(routing.yaml) 변경만으로 요청 대상 모델이 전환되는지 확인
- 임베딩 API 호출 및 결과 반환 성공

확인 항목:
- chat 별칭 → 실제 모델 치환으로 챗 응답 수신
- 사고 모드 — reasoning_effort 생략(기본 일반 모드)은 사고 출력이 없고, reasoning_effort:high면 사고 출력이 생긴다
- 제어 계약 — 비문자열·빈 문자열 reasoning_effort는 업스트림 호출 없이 400
- 엄격한 별칭 계약 — 실제 모델명·미등록 별칭·엔드포인트 불일치 별칭은 업스트림 호출 없이 400
- embed 별칭 임베딩 — 단건/배열 응답, 1024 차원, 입력 순서·인덱스 보존, 임베딩 모델 CPU 상주
- encoding_format — 기본 float 리스트, base64는 SDK가 되돌릴 수 있는 float32 리틀엔디언 바이트

실행 전제: 채팅 Ollama(기본 11434, gemma4:12b-it-qat)와 CPU 임베딩 Ollama
(기본 11435, snowflake-arctic-embed2)가 각각 기동된 상태. 게이트웨이는 스크립트가 직접 띄우며
실 OpenAI 폴백은 비활성화한다.
실행: uv run python scripts/verify_stage4.py
"""

import base64
import os
import socket
import struct
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
CHAT_OLLAMA_BASE_URL = os.environ.get(
    "GATEWAY_OLLAMA_BASE_URL", "http://127.0.0.1:11434"
).rstrip("/")
EMBEDDING_OLLAMA_BASE_URL = os.environ.get(
    "GATEWAY_EMBEDDING_OLLAMA_BASE_URL", "http://127.0.0.1:11435"
).rstrip("/")
CHAT_ALIAS = "chat"
EMBED_ALIAS = "embed"
CHAT_MODEL = "gemma4:12b-it-qat"
EMBED_MODEL = "snowflake-arctic-embed2"
EMBED_DIMENSIONS = 1024
STARTUP_TIMEOUT_SECONDS = 15
REQUEST_TIMEOUT_SECONDS = 180

failures: list[str] = []
AUTH = VerificationAuth.create("stage4-verifier")


def check(label: str, passed: bool, detail: str) -> None:
    mark = "FAIL"
    if passed:
        mark = "PASS"
    print(f"[{mark}] {label} — {detail}")
    if not passed:
        failures.append(label)


def stop_gateway(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def start_gateway() -> subprocess.Popen:
    environment = os.environ.copy()
    # 라이브 검증은 로컬 Ollama만 대상으로 한다. 저장소 .env에 키가 있어도 외부로 우회하지 않는다.
    environment["OPENAI_API_KEY"] = ""
    environment["GATEWAY_OLLAMA_BASE_URL"] = CHAT_OLLAMA_BASE_URL
    environment["GATEWAY_EMBEDDING_OLLAMA_BASE_URL"] = EMBEDDING_OLLAMA_BASE_URL
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
    stop_gateway(process)
    raise RuntimeError(
        "게이트웨이가 기동하지 않는다 "
        f"(chat={CHAT_OLLAMA_BASE_URL}, embedding={EMBEDDING_OLLAMA_BASE_URL})"
    )


def stream_reasoning_and_content(
    client: openai.OpenAI, reasoning_effort: str | None
) -> tuple[str, int]:
    """스트림에서 content와 reasoning 델타 개수를 모은다 — 사고 모드 동작을 관찰한다."""
    extra: dict[str, object] = {}
    if reasoning_effort is not None:
        extra["reasoning_effort"] = reasoning_effort

    stream = client.chat.completions.create(
        model=CHAT_ALIAS,
        messages=[
            {
                "role": "user",
                "content": "추론 게이트웨이가 무엇인지 한 문장으로 답하라.",
            }
        ],
        stream=True,
        **extra,
    )
    content = ""
    reasoning_deltas = 0
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            content += delta.content
        if getattr(delta, "reasoning", None):
            reasoning_deltas += 1
    return content, reasoning_deltas


def chat_body(model: str, **extra: object) -> dict:
    return {"model": model, "messages": [{"role": "user", "content": "x"}], **extra}


def expect_chat_400(body: dict) -> bool:
    response = httpx.post(
        f"{GATEWAY_BASE_URL}/v1/chat/completions",
        json=body,
        headers=AUTH.headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code != 400:
        return False
    return response.json()["error"]["type"] == "invalid_request_error"


gateway = start_gateway()
try:
    client = openai.OpenAI(
        base_url=f"{GATEWAY_BASE_URL}/v1",
        api_key=AUTH.api_key,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    # 1. 기본 일반 모드 — reasoning_effort 생략 시 게이트웨이가 none을 넣어 사고 출력이 없다.
    content, default_reasoning_deltas = stream_reasoning_and_content(client, None)
    check(
        "chat 별칭 챗 응답",
        bool(content.strip()),
        f"content {len(content)}자",
    )
    check(
        "기본 일반 모드(사고 없음)",
        default_reasoning_deltas == 0 and bool(content.strip()),
        f"사고 델타 {default_reasoning_deltas}개 (0이어야 함)",
    )

    # 2. reasoning_effort:high — SDK 정식 인자를 그대로 업스트림에 전달해 사고 출력이 생긴다.
    _, thinking_reasoning_deltas = stream_reasoning_and_content(client, "high")
    check(
        "reasoning_effort:high 사고 모드",
        thinking_reasoning_deltas > 0,
        f"reasoning_effort:high 시 사고 델타 {thinking_reasoning_deltas}개",
    )

    # 3. 제어 계약 — 비문자열·빈 문자열 reasoning_effort는 업스트림 없이 400.
    check(
        "비문자열 reasoning_effort 400",
        expect_chat_400(chat_body(CHAT_ALIAS, reasoning_effort=1)),
        "정수 reasoning_effort 거절",
    )
    check(
        "빈 문자열 reasoning_effort 400",
        expect_chat_400(chat_body(CHAT_ALIAS, reasoning_effort="")),
        "빈 문자열 reasoning_effort 거절",
    )

    # 4. 엄격한 별칭 계약 — 실제 모델명·미등록·엔드포인트 불일치는 업스트림 없이 400.
    check(
        "실제 모델명 직접 지정 400",
        expect_chat_400(chat_body(CHAT_MODEL)),
        f"'{CHAT_MODEL}' 거절",
    )
    check(
        "미등록 별칭 400",
        expect_chat_400(chat_body("chatt")),
        "오타 별칭 'chatt' 거절",
    )
    check(
        "엔드포인트 불일치 별칭 400",
        expect_chat_400(chat_body(EMBED_ALIAS)),
        "chat 엔드포인트의 embed 별칭 거절",
    )

    # 5. embed 별칭 임베딩 — 단건 응답, 1024 차원. encoding_format 생략 시 SDK가 base64를
    #    보내므로, 값이 float 리스트로 돌아오면 게이트웨이 base64를 SDK가 정상 역디코딩한 것이다.
    single = client.embeddings.create(
        model=EMBED_ALIAS, input="상품 검색 임베딩 검증용 문장이다."
    )
    single_dimensions = len(single.data[0].embedding)
    check(
        "임베딩 단건 응답(SDK base64 왕복)",
        single_dimensions == EMBED_DIMENSIONS
        and single.data[0].index == 0
        and all(isinstance(value, float) for value in single.data[0].embedding),
        f"{single_dimensions}차원, index {single.data[0].index}",
    )

    # 6. 배열 입력 — 입력 순서와 index가 보존된다.
    batch = client.embeddings.create(
        model=EMBED_ALIAS, input=["첫째 문장", "둘째 문장"]
    )
    indexes = [item.index for item in batch.data]
    check(
        "임베딩 배열 순서·인덱스 보존",
        len(batch.data) == 2 and indexes == [0, 1],
        f"{len(batch.data)}건, index {indexes}",
    )

    # 7. base64 명시 요청 — float32 리틀엔디언 바이트로 되돌려지는지 직접 확인한다.
    encoded = client.embeddings.create(
        model=EMBED_ALIAS,
        input="base64 인코딩 검증용 문장이다.",
        encoding_format="base64",
    )
    encoded_vector = encoded.data[0].embedding
    decoded_dimensions = 0
    if isinstance(encoded_vector, str):
        decoded_dimensions = len(base64.b64decode(encoded_vector)) // struct.calcsize(
            "f"
        )
    check(
        "base64 인코딩 응답",
        isinstance(encoded_vector, str) and decoded_dimensions == EMBED_DIMENSIONS,
        f"base64 문자열 → {decoded_dimensions}차원",
    )

    # 8. 임베딩 모델 CPU 상주 — native /api/embed의 num_gpu:0이 실제로 반영됐는지 확인한다.
    resident = {}
    for model in httpx.get(f"{EMBEDDING_OLLAMA_BASE_URL}/api/ps").json()["models"]:
        resident[model["name"].removesuffix(":latest")] = model
    if EMBED_MODEL in resident:
        embed_state = resident[EMBED_MODEL]
        check(
            "임베딩 모델 100% CPU",
            embed_state["size_vram"] == 0,
            f"VRAM 점유 {embed_state['size_vram'] / 2**20:.0f}MiB",
        )
    else:
        check("임베딩 모델 100% CPU", False, "적재되어 있지 않음")
finally:
    stop_gateway(gateway)
    AUTH.close()

if failures:
    print(f"\n4단계 검증 실패: {', '.join(failures)}")
    sys.exit(1)
print("\n4단계 검증 통과 — 별칭 라우팅과 CPU 임베딩이 설정 기준으로 성립한다.")
