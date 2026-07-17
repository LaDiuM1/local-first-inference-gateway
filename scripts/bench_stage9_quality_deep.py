"""9단계 품질 심층 평가 — 사고 모드를 포함한 6구성의 품질만 전문적으로 비교한다.

1차 벤치마크의 서비스 질의 셋은 전 모델이 9점대에 몰리는 천장 효과가 있었다. 이 평가는
변별력을 위해 (1) 공개 객관 셋 KMMLU 서브셋(4지선다, 자동 채점)과 (2) 자체 고난도 셋
(다단계 계산·함정 근거·형식 제약·추론, 정답 검증 + 심판)을 사용한다.

- 평가 대상 6구성: gemma4 일반(none)·사고(high) — 게이트웨이 경유,
  gpt-5-mini·gpt-5.4-nano 각 minimal·high — OpenAI 직접 호출.
- 심판: Codex CLI의 gpt-5.6-sol(reasoning xhigh)이 모델명을 가린 응답을 배치 채점.
- KMMLU 문항은 HuggingFace datasets-server에서 받아 결과 디렉터리에만 캐시한다
  (재배포 금지 라이선스 — 저장소에 커밋하지 않는다).

실행: uv run python scripts/bench_stage9_quality_deep.py [--skip-judge] [--force]
구성별 응답과 심판 결과는 JSON으로 캐시되어 중단 후 같은 명령으로 이어 간다.
"""

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
from benchmark_stage9 import (
    BenchConfig,
    read_openai_key,
    start_gateway,
    timed_stream_request,
)

RESULTS_DIRECTORY = Path("docs/_local/stage9-bench-quality")
HARD_DATASET_PATH = Path("scripts/bench_stage9_hard_dataset.json")
DATASETS_SERVER = "https://datasets-server.huggingface.co"
KMMLU_DATASET = "HAERAE-HUB/KMMLU"
KMMLU_CATEGORIES = (
    "Computer-Science",
    "Management",
    "Economics",
    "Law",
    "Math",
    "Marketing",
)
KMMLU_PER_CATEGORY = 8
# 사고 모드는 복잡한 질의에서 답변에 몇 분이 걸릴 수 있다 — 시간 제한이 측정을
# 자르지 않도록 여유 있게 둔다(스트리밍이라 read 타임아웃은 델타 간격 기준).
LOCAL_TIMEOUT_SECONDS = 1800.0
OPENAI_TIMEOUT_SECONDS = 240.0
JUDGE_MODEL = "gpt-5.6-sol"
JUDGE_REASONING_EFFORT = "xhigh"
JUDGE_TIMEOUT_SECONDS = 2400
BLIND_SEED = 20260717

SYSTEMS = (
    {"key": "local-none", "kind": "local", "reasoning": None},
    {"key": "local-high", "kind": "local", "reasoning": "high"},
    {
        "key": "mini-minimal",
        "kind": "openai",
        "model": "gpt-5-mini",
        "reasoning": "minimal",
    },
    {"key": "mini-high", "kind": "openai", "model": "gpt-5-mini", "reasoning": "high"},
    # gpt-5.4-nano는 minimal을 지원하지 않는다(400 실측) — 일반 모드는 none이다.
    {
        "key": "nano-none",
        "kind": "openai",
        "model": "gpt-5.4-nano",
        "reasoning": "none",
    },
    {
        "key": "nano-high",
        "kind": "openai",
        "model": "gpt-5.4-nano",
        "reasoning": "high",
    },
)

MCQ_INSTRUCTION = (
    "다음 4지선다 문제를 풀고, 마지막 줄에 반드시 '정답: X' 형태로 답하세요"
    "(X는 A, B, C, D 중 하나)."
)


def load_hard_items() -> list[dict]:
    return json.loads(HARD_DATASET_PATH.read_text(encoding="utf-8"))["items"]


def fetch_kmmlu_questions() -> list[dict]:
    """KMMLU 서브셋을 내려받아 캐시한다 — 분야당 test 스플릿 앞쪽 고정 구간."""
    cache_path = RESULTS_DIRECTORY / "kmmlu_questions.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    questions: list[dict] = []
    with httpx.Client(timeout=60) as client:
        for category in KMMLU_CATEGORIES:
            response = client.get(
                f"{DATASETS_SERVER}/rows",
                params={
                    "dataset": KMMLU_DATASET,
                    "config": category,
                    "split": "test",
                    "offset": 0,
                    "length": KMMLU_PER_CATEGORY,
                },
            )
            response.raise_for_status()
            for index, row in enumerate(response.json()["rows"]):
                record = row["row"]
                questions.append(
                    {
                        "id": f"kmmlu-{category}-{index:02d}",
                        "category": category,
                        "question": record["question"],
                        "choices": {
                            "A": record["A"],
                            "B": record["B"],
                            "C": record["C"],
                            "D": record["D"],
                        },
                        "answer": "ABCD"[record["answer"] - 1],
                    }
                )
    cache_path.write_text(
        json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return questions


def mcq_prompt(question: dict) -> str:
    choices = question["choices"]
    return (
        f"{MCQ_INSTRUCTION}\n\n문제: {question['question']}\n"
        f"A. {choices['A']}\nB. {choices['B']}\nC. {choices['C']}\nD. {choices['D']}"
    )


def hard_prompt(item: dict) -> str:
    if item.get("context"):
        return (
            "다음 근거만 참고해서 답하세요.\n\n[근거]\n"
            f"{item['context']}\n\n[질문]\n{item['prompt']}"
        )
    return item["prompt"]


def build_prompts() -> list[dict]:
    """수집 대상 전체 — KMMLU 객관식과 고난도 셋을 한 목록으로 합친다."""
    prompts = [
        {"id": question["id"], "prompt": mcq_prompt(question), "source": "kmmlu"}
        for question in fetch_kmmlu_questions()
    ]
    prompts.extend(
        {"id": item["id"], "prompt": hard_prompt(item), "source": "hard"}
        for item in load_hard_items()
    )
    return prompts


def chat_once(
    client: httpx.Client, url_path: str, model: str, prompt: str, reasoning: str | None
) -> dict:
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if reasoning is not None:
        payload["reasoning_effort"] = reasoning
    started = time.monotonic()
    try:
        response = client.post(url_path, json=payload)
    except httpx.HTTPError as error:
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}
    seconds = time.monotonic() - started
    if response.status_code != 200:
        return {"ok": False, "error": f"HTTP {response.status_code}"}
    body = response.json()
    content = body.get("choices", [{}])[0].get("message", {}).get("content") or ""
    if not content.strip():
        return {"ok": False, "error": "빈 응답"}
    return {
        "ok": True,
        "text": content,
        "seconds": seconds,
        "usage": body.get("usage"),
    }


def collect_system(system: dict, prompts: list[dict], openai_key: str) -> list[dict]:
    """한 구성의 전체 응답을 수집한다 — 로컬은 게이트웨이 경유, GPT는 직접 호출."""
    if system["kind"] == "local":
        bench_config = BenchConfig(
            dataset_path=HARD_DATASET_PATH, results_directory=RESULTS_DIRECTORY
        )
        gateway = start_gateway(bench_config)
        try:
            with httpx.Client(
                base_url=gateway.base_url,
                headers=gateway.headers,
                timeout=LOCAL_TIMEOUT_SECONDS,
            ) as client:
                rows = []
                for entry in prompts:
                    # 사고 모드의 버퍼 응답은 게이트웨이 응답 시작 기한(로컬 90초)을
                    # 넘길 수 있다 — 스트리밍은 첫 사고 델타가 곧바로 도착해 기한과
                    # 무관하고, content 델타만 모아 최종 답변 텍스트를 얻는다.
                    payload: dict = {
                        "model": "chat",
                        "messages": [{"role": "user", "content": entry["prompt"]}],
                    }
                    if system["reasoning"] is not None:
                        payload["reasoning_effort"] = system["reasoning"]
                    measured = timed_stream_request(client, payload)
                    row = {"id": entry["id"], "ok": measured["ok"]}
                    if measured["ok"]:
                        row["text"] = measured["text"]
                        row["seconds"] = measured["total_seconds"]
                        row["usage"] = measured.get("usage")
                    else:
                        row["error"] = measured["error"]
                    rows.append(row)
            records = gateway.request_records()
            non_local = [
                record
                for record in records
                if record["path"] == "/v1/chat/completions"
                and record["status"] == 200
                and record["provider"] != "local"
            ]
        finally:
            gateway.close()
        if non_local:
            raise RuntimeError(f"로컬이 아닌 provider가 섞였다: {len(non_local)}건")
        return rows
    with httpx.Client(
        base_url="https://api.openai.com/v1",
        headers={"Authorization": f"Bearer {openai_key}"},
        timeout=OPENAI_TIMEOUT_SECONDS,
    ) as client:

        def call(entry: dict) -> dict:
            result = chat_once(
                client,
                "/chat/completions",
                system["model"],
                entry["prompt"],
                system["reasoning"],
            )
            return {"id": entry["id"], **result}

        with ThreadPoolExecutor(max_workers=6) as executor:
            return list(executor.map(call, prompts))


def parse_choice(text: str) -> str | None:
    labeled = re.findall(r"정답[^ABCD]{0,10}([ABCD])", text)
    if labeled:
        return labeled[-1]
    standalone = re.findall(r"\b([ABCD])\b", text)
    if standalone:
        return standalone[-1]
    return None


NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
# '18이 아닙니다'처럼 값을 부정하는 줄 — 최종 답으로 읽지 않는다.
NEGATION_MARKERS = ("아니", "아닙")


def final_number(text: str) -> float | None:
    """마지막 답 줄의 마지막 숫자를 꺼낸다 — 최종 답만 판정한다.

    응답 본문 어딘가의 숫자를 전부 훑으면 중간 계산에 정답 값이 스치기만 해도
    오답이 통과한다. 마지막 내용 줄이 값을 부정하는 정정문('~이 아니라')이면
    이전 답으로 되돌아가지 않고 판정 불가(None)로 끝낸다 — 정정 이전의 답을
    되살리면 올바른 정정이 오답 처리되는 방향으로 뒤집힌다.
    """
    lines = [line for line in text.replace(",", "").splitlines() if line.strip()]
    for index, line in enumerate(reversed(lines)):
        if any(marker in line for marker in NEGATION_MARKERS):
            if index == 0:
                return None
            continue
        numbers = NUMBER_PATTERN.findall(line)
        if numbers:
            return float(numbers[-1])
    return None


def numeric_correct(text: str, value: float) -> bool:
    number = final_number(text)
    if number is None:
        return False
    return abs(number - value) < 1e-6


def keyword_correct(text: str, expected: dict) -> bool:
    """정답 키워드 판정 — 부정·비교 서술이 정답 키워드로 오탐되는 것을 막는다.

    `reject_any`는 결론을 뒤집는 구문('반품 대상이 되지 않', '판매자 부담이 아니')
    으로, 하나라도 있으면 오답이다. `negations`는 정답 키워드를 부분 문자열로
    포함하는 표현('가능' ⊂ '불가능')을 제거한 뒤 매칭해, '변심은 불가, 하자는
    가능' 같은 정답 서술은 살리고 '불가능합니다' 단독 오답은 걸러낸다. `all`은
    정답에 반드시 함께 있어야 하는 사실(예: 배송비 부담 주체)이다. 답 하나만
    요구하는 문항(`final_line_only`)은 마지막 내용 줄만 보고, 그 줄이 값을
    부정하는 정정문이면 이전 줄로 되돌아가지 않고 오답으로 끝낸다.
    """
    haystack = text.casefold()
    if expected.get("final_line_only"):
        lines = [line for line in haystack.splitlines() if line.strip()]
        if not lines or any(marker in lines[-1] for marker in NEGATION_MARKERS):
            return False
        haystack = lines[-1]
    if any(phrase.casefold() in haystack for phrase in expected.get("reject_any", [])):
        return False
    for negation in expected.get("negations", []):
        haystack = haystack.replace(negation.casefold(), "")
    if not all(keyword.casefold() in haystack for keyword in expected.get("all", [])):
        return False
    return any(keyword.casefold() in haystack for keyword in expected["any"])


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z]*\s*", "", stripped)
        stripped = re.sub(r"```\s*$", "", stripped)
    return stripped.strip()


def format_correct(check: dict, text: str) -> bool:
    """형식 준수 판정 — 문항이 지시한 형식 계약을 그대로 적용한다.

    'JSON으로만 출력'에는 코드펜스도 다른 텍스트이고, '글머리 기호(-)'에는
    다른 글머리 문자도 위반이다 — 채점기가 지시보다 관대해지면 형식 과제의
    변별력이 사라진다.
    """
    body = text.strip()
    if check["type"] == "json":
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return False
        if not isinstance(parsed, dict):
            return False
        # 지시한 형식은 세 키뿐이다 — 추가 키는 '지정된 형식으로만'의 위반이다.
        if set(parsed) != {"category", "urgency", "summary"}:
            return False
        urgency = parsed.get("urgency")
        summary = parsed.get("summary")
        return (
            parsed.get("category") == check["category"]
            and isinstance(urgency, int)
            and not isinstance(urgency, bool)
            and check["urgency_min"] <= urgency <= 5
            and isinstance(summary, str)
            and bool(summary.strip())
        )
    if check["type"] == "bullets":
        # 이모지 금지는 문항의 명시 지시다 — 그림 블록·국기·키캡 결합까지 검사한다.
        if any(
            0x1F1E6 <= ord(char) <= 0x1FAFF
            or 0x2600 <= ord(char) <= 0x27BF
            or ord(char) in (0xFE0F, 0x20E3)
            for char in body
        ):
            return False
        lines = [line.strip() for line in body.splitlines() if line.strip()]
        bullets = [line for line in lines if line.startswith("-")]
        if len(lines) != check["count"] or len(bullets) != check["count"]:
            return False
        return all(
            0 < len(bullet[1:].strip()) <= check["max_chars"] for bullet in bullets
        )
    if check["type"] == "forbidden":
        return all(word not in body for word in check["words"])
    raise ValueError(f"알 수 없는 형식 검사: {check['type']}")


def hard_auto_correct(item: dict, text: str) -> bool | None:
    """고난도 항목의 객관 판정 — 객관 기준이 없는 항목은 None(심판 전담)."""
    if "format_check" in item:
        return format_correct(item["format_check"], text)
    if "answer_value" in item:
        return numeric_correct(text, item["answer_value"])
    if "expected" in item:
        return keyword_correct(text, item["expected"])
    return None


def resolve_codex_command() -> str:
    candidates = sorted(
        Path(os.environ["LOCALAPPDATA"], "OpenAI", "Codex", "bin").glob("*/codex.exe"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return str(candidates[0])
    raise RuntimeError("Codex CLI를 찾지 못했다")


def build_judge_input(
    judged_items: list[dict],
    responses: dict[str, dict[str, str]],
    systems: tuple[dict, ...],
) -> tuple[str, dict[str, dict[str, str]]]:
    """모델명을 가린 심판 입력 문서와 라벨→구성 매핑을 만든다."""
    rng = random.Random(BLIND_SEED)
    lines = [
        "# 품질 심층 평가 — 심판 채점 대상",
        "",
        f"각 문항의 채점 기준을 적용해 응답 R1~R{len(systems)}을 1~10 정수로 채점한다.",
        "",
    ]
    mapping: dict[str, dict[str, str]] = {}
    for item in judged_items:
        keys = [system["key"] for system in systems]
        rng.shuffle(keys)
        labels = [f"R{index + 1}" for index in range(len(keys))]
        mapping[item["id"]] = dict(zip(labels, keys, strict=True))
        lines.append(f"## {item['id']}")
        if item.get("context"):
            lines.append(f"[근거]\n{item['context']}")
        lines.append(f"[과제]\n{item['prompt']}")
        lines.append(f"[채점 기준]\n{item['judge_note']}")
        for label, key in zip(labels, keys, strict=True):
            answer = responses[key].get(item["id"], "(응답 실패)")
            lines.append(f"[응답 {label}]\n{answer}")
        lines.append("")
    return "\n".join(lines), mapping


JUDGE_PROMPT_TEMPLATE = (
    "너는 엄격한 AI 응답 품질 심사관이다. 파일 {input_path} 를 읽어라.\n"
    "각 문항의 [채점 기준]을 적용해 [응답 R1]~[응답 R{last}]을 1~10 정수로 채점하라.\n"
    "기준: 채점 기준의 정답 포인트를 맞히지 못하거나 형식 지시를 어기면 4점 이하,"
    " 정답이면서 근거·간결성까지 좋으면 8점 이상을 준다. 응답 간 상대 비교로"
    " 점수 차이를 분명히 하라.\n"
    "최종 메시지는 다음 형태의 JSON 하나만 출력한다 — 설명·코드펜스 금지:\n"
    '{{"문항id": {{{score_example}}}, ...}}'
)


def run_codex_judge(judge_input: str, output_path: Path, label_count: int) -> dict:
    """블라인드 심판 실행 — 심판 입력만 있는 임시 작업 디렉터리에서 돌린다.

    저장소를 작업 디렉터리로 주면 read-only 샌드박스라도 라벨 매핑·응답 캐시를
    읽어 블라인드가 깨질 수 있다 — 심판이 볼 수 있는 파일을 입력 하나로 제한한다.
    """
    codex = resolve_codex_command()
    score_example = ", ".join(f'"R{index + 1}": 점수' for index in range(label_count))
    with TemporaryDirectory(prefix="stage9-judge-") as workspace:
        input_path = Path(workspace) / "judge_input.md"
        input_path.write_text(judge_input, encoding="utf-8")
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            input_path=input_path, last=label_count, score_example=score_example
        )
        completed = subprocess.run(
            [
                codex,
                "-C",
                workspace,
                "-s",
                "read-only",
                "-a",
                "never",
                "-m",
                JUDGE_MODEL,
                "-c",
                f"model_reasoning_effort={JUDGE_REASONING_EFFORT}",
                "exec",
                # 격리 작업 디렉터리는 git 저장소가 아니다 — 신뢰 검사를 명시적으로 끈다.
                "--skip-git-repo-check",
                "--ephemeral",
                "-o",
                str(output_path.resolve()),
                prompt,
            ],
            input="",
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=JUDGE_TIMEOUT_SECONDS,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Codex 심판 실행 실패(exit {completed.returncode})")
    reply = output_path.read_text(encoding="utf-8")
    body = strip_code_fence(reply)
    start, end = body.find("{"), body.rfind("}")
    if start == -1 or end <= start:
        raise RuntimeError(f"심판 응답에서 JSON을 찾지 못했다: {body[:200]!r}")
    return json.loads(body[start : end + 1])


def validate_judge_scores(
    judge_scores: dict, judged_items: list[dict], label_count: int
) -> None:
    """심판 산출 검증 — 문항·라벨의 정확한 집합과 1~10 정수 점수만 인정한다."""
    labels = {f"R{index + 1}" for index in range(label_count)}
    expected_ids = {item["id"] for item in judged_items}
    unexpected = sorted(set(judge_scores) - expected_ids)
    if unexpected:
        raise RuntimeError(f"심판 대상 밖 문항 채점: {unexpected}")
    for item in judged_items:
        row = judge_scores.get(item["id"])
        if not isinstance(row, dict) or set(row) != labels:
            raise RuntimeError(f"심판 채점 라벨 누락: {item['id']}")
        for label, score in row.items():
            valid_integer = isinstance(score, int) and not isinstance(score, bool)
            if not valid_integer or not 1 <= score <= 10:
                raise RuntimeError(
                    f"심판 점수 범위 위반: {item['id']}.{label}={score!r}"
                )


def local_chat_model() -> str:
    """로컬 구성이 실제로 라우팅되는 chat 모델 — 지문이 라우팅 변경을 감지하게 한다."""
    from gateway.routing import EndpointKind, load_routing_table

    table = load_routing_table(Path("routing.yaml"))
    return table.resolve(EndpointKind.chat, "chat")


def response_fingerprint(prompts: list[dict], system: dict, routed_model: str) -> str:
    """수집 캐시와 (질의 셋, 측정 구성, 실제 모델)의 결합 지문.

    채점 규칙은 넣지 않는다 — 채점기 개선은 재수집이 아니라 재채점 대상이다.
    """
    canonical = json.dumps(
        {
            "system": [
                system["key"],
                system["kind"],
                routed_model,
                system.get("reasoning"),
            ],
            "prompts": [(entry["id"], entry["prompt"]) for entry in prompts],
        },
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def judge_fingerprint(judge_input: str, mapping: dict[str, dict[str, str]]) -> str:
    """심판 캐시와 심판 실행 조건의 결합 지문.

    심판이 실제로 본 입력 문서 전체(문항·근거·채점 기준·가려진 응답)와 채점 지시
    템플릿, 라벨 매핑, 심판 모델·수준·블라인드 seed를 묶는다 — 어느 하나가 바뀌면
    캐시를 버리고 재채점한다.
    """
    canonical = json.dumps(
        {
            "judge": [
                JUDGE_MODEL,
                JUDGE_REASONING_EFFORT,
                BLIND_SEED,
                JUDGE_PROMPT_TEMPLATE,
            ],
            "input": judge_input,
            "mapping": mapping,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def accuracy(rows: list[tuple[str, bool]]) -> float | None:
    if not rows:
        return None
    return sum(1 for _, correct in rows if correct) / len(rows)


def build_quality_report(scored: dict, judge_result: dict[str, dict] | None) -> dict:
    """공식 품질 보고서(report.json)의 단일 스키마 — 합격선 집계기와 검증이 공유한다."""
    return {"scored": scored, "judge": judge_result}


def score_all(prompts: list[dict], responses_by_system: dict[str, list[dict]]) -> dict:
    """구성별 KMMLU·고난도 객관 채점과 실패 집계."""
    kmmlu_by_id = {q["id"]: q for q in fetch_kmmlu_questions()}
    hard_by_id = {item["id"]: item for item in load_hard_items()}
    expected_ids = {entry["id"] for entry in prompts}
    scored: dict[str, dict] = {}
    for system_key, rows in responses_by_system.items():
        kmmlu_rows: list[tuple[str, bool]] = []
        kmmlu_parse_failures = 0
        hard_kind_rows: dict[str, list[tuple[str, bool]]] = {}
        # 문항 집합은 정확히 일치해야 한다 — 중복 행은 정답 부풀리기, 누락은 수집 실패다.
        row_ids = [row["id"] for row in rows]
        duplicated = sorted({row_id for row_id in row_ids if row_ids.count(row_id) > 1})
        if duplicated:
            raise RuntimeError(f"[{system_key}] 중복 응답 행: {duplicated}")
        unexpected = sorted(set(row_ids) - expected_ids)
        if unexpected:
            raise RuntimeError(f"[{system_key}] 질의 셋 밖 응답 행: {unexpected}")
        missing_ids = sorted(expected_ids - set(row_ids))
        failures = [row["id"] for row in rows if not row["ok"]] + missing_ids
        seconds = [row["seconds"] for row in rows if row["ok"]]
        for row in rows:
            if not row["ok"]:
                continue
            if row["id"] in kmmlu_by_id:
                choice = parse_choice(row["text"])
                if choice is None:
                    kmmlu_parse_failures += 1
                correct = choice == kmmlu_by_id[row["id"]]["answer"]
                kmmlu_rows.append((row["id"], correct))
            else:
                item = hard_by_id[row["id"]]
                verdict = hard_auto_correct(item, row["text"])
                if verdict is not None:
                    hard_kind_rows.setdefault(item["kind"], []).append(
                        (row["id"], verdict)
                    )
        per_category: dict[str, list[tuple[str, bool]]] = {}
        for item_id, correct in kmmlu_rows:
            category = kmmlu_by_id[item_id]["category"]
            per_category.setdefault(category, []).append((item_id, correct))
        scored[system_key] = {
            "failures": failures,
            "mean_seconds": sum(seconds) / len(seconds) if seconds else None,
            "kmmlu_total": accuracy(kmmlu_rows),
            "kmmlu_count": len(kmmlu_rows),
            "kmmlu_parse_failures": kmmlu_parse_failures,
            "kmmlu_by_category": {
                category: accuracy(rows) for category, rows in per_category.items()
            },
            "hard_auto_by_kind": {
                kind: accuracy(rows) for kind, rows in hard_kind_rows.items()
            },
        }
    return scored


def judge_means(
    judge_scores: dict, mapping: dict[str, dict[str, str]], hard_by_id: dict
) -> dict[str, dict[str, float | None]]:
    collected: dict[str, dict[str, list[int]]] = {}
    for item_id, labels in judge_scores.items():
        for label, score in labels.items():
            system_key = mapping[item_id][label]
            kind = hard_by_id[item_id]["kind"]
            collected.setdefault(system_key, {}).setdefault(kind, []).append(int(score))
    means: dict[str, dict[str, float | None]] = {}
    for system_key, per_kind in collected.items():
        means[system_key] = {
            kind: sum(scores) / len(scores) for kind, scores in per_kind.items()
        }
        merged = [score for scores in per_kind.values() for score in scores]
        means[system_key]["judged_overall"] = sum(merged) / len(merged)
    return means


def format_ratio(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def render_report(
    scored: dict,
    judge: dict[str, dict] | None,
    note: str,
    systems: tuple[dict, ...],
) -> str:
    categories = list(KMMLU_CATEGORIES)
    lines = [
        "# 9단계 품질 심층 평가 결과",
        "",
        f"생성: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"- {note}",
        "- KMMLU 서브셋: "
        + ", ".join(categories)
        + f" 각 {KMMLU_PER_CATEGORY}문항(test 스플릿 고정 구간), 4지선다 자동 채점",
        "- 고난도 셋: scripts/bench_stage9_hard_dataset.json — 계산·추론·함정은 정답"
        " 검증, 함정·형식은 심판 채점 병행",
        f"- 심판: Codex CLI {JUDGE_MODEL} (reasoning {JUDGE_REASONING_EFFORT}),"
        " 모델명 비공개 배치 채점",
        "",
        "## KMMLU 정확도 (객관식)",
        "",
        "| 구성 | " + " | ".join(categories) + " | 전체 | 형식 미준수 |",
        "|---|" + "---|" * (len(categories) + 2),
    ]
    for system in systems:
        row = scored[system["key"]]
        cells = [
            format_ratio(row["kmmlu_by_category"].get(category))
            for category in categories
        ]
        lines.append(
            f"| {system['key']} | "
            + " | ".join(cells)
            + f" | **{format_ratio(row['kmmlu_total'])}** |"
            f" {row['kmmlu_parse_failures']} |"
        )
    lines += [
        "",
        "## 고난도 셋 — 객관 판정 정답률",
        "",
        "| 구성 | calc | logic | trap | format | 평균 응답 시간(s) | 수집 실패 |",
        "|---|---|---|---|---|---|---|",
    ]
    for system in systems:
        row = scored[system["key"]]
        kinds = row["hard_auto_by_kind"]
        lines.append(
            f"| {system['key']} | {format_ratio(kinds.get('calc'))} |"
            f" {format_ratio(kinds.get('logic'))} | {format_ratio(kinds.get('trap'))} |"
            f" {format_ratio(kinds.get('format'))} |"
            f" {format_ratio(row['mean_seconds'])} | {len(row['failures'])} |"
        )
    if judge:
        lines += [
            "",
            f"## 심판 채점 ({JUDGE_MODEL}, 1~10)",
            "",
            "| 구성 | trap | format | 종합 |",
            "|---|---|---|---|",
        ]
        for system in systems:
            row = judge.get(system["key"], {})
            lines.append(
                f"| {system['key']} | {format_ratio(row.get('trap'))} |"
                f" {format_ratio(row.get('format'))} |"
                f" **{format_ratio(row.get('judged_overall'))}** |"
            )
    lines.append("")
    return "\n".join(lines)


def retry_failed_rows(
    system: dict, rows: list[dict], prompts_by_id: dict[str, str], openai_key: str
) -> list[dict]:
    """실패한 항목만 다시 수집해 병합한다 — 성공 항목의 측정값은 그대로 둔다."""
    failed_ids = [row["id"] for row in rows if not row["ok"]]
    if not failed_ids:
        return rows
    entries = [
        {"id": item_id, "prompt": prompts_by_id[item_id]} for item_id in failed_ids
    ]
    retried = {row["id"]: row for row in collect_system(system, entries, openai_key)}
    return [
        retried[row["id"]] if row["id"] in retried and not row["ok"] else row
        for row in rows
    ]


def main() -> None:
    global RESULTS_DIRECTORY
    parser = argparse.ArgumentParser(description="9단계 품질 심층 평가")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--results-dir", default=str(RESULTS_DIRECTORY))
    parser.add_argument(
        "--systems", default=",".join(system["key"] for system in SYSTEMS)
    )
    parser.add_argument("--retry-failed", action="store_true")
    arguments = parser.parse_args()
    RESULTS_DIRECTORY = Path(arguments.results_dir)
    selected_keys = [key.strip() for key in arguments.systems.split(",") if key.strip()]
    unknown = set(selected_keys) - {system["key"] for system in SYSTEMS}
    if unknown:
        raise SystemExit(f"알 수 없는 구성: {sorted(unknown)}")
    systems = tuple(system for system in SYSTEMS if system["key"] in selected_keys)
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    prompts = build_prompts()
    prompts_by_id = {entry["id"]: entry["prompt"] for entry in prompts}
    print(f"평가 문항 {len(prompts)}개, 구성 {len(systems)}종")
    openai_key = read_openai_key(Path(".env"))

    routed_local_model = local_chat_model()
    responses_by_system: dict[str, list[dict]] = {}
    for system in systems:
        cache_path = RESULTS_DIRECTORY / f"responses_{system['key']}.json"
        if system["kind"] == "local":
            routed_model = routed_local_model
        else:
            routed_model = system["model"]
        system_fingerprint = response_fingerprint(prompts, system, routed_model)
        rows: list[dict] | None = None
        if cache_path.exists() and not arguments.force:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            # 캐시는 (질의 셋, 구성) 지문이 일치할 때만 채점 대상이다 — 조건이 바뀐 채
            # 옛 응답을 채점하면 결과가 측정과 결합되지 않는다.
            if (
                isinstance(payload, dict)
                and payload.get("fingerprint") == system_fingerprint
            ):
                rows = payload["rows"]
                print(f"[{system['key']}] 캐시 재사용")
            else:
                raise SystemExit(
                    f"[{system['key']}] 응답 캐시가 현재 질의 셋·구성과 다르다 —"
                    " --force로 다시 수집하라"
                )
        if rows is None:
            started = time.monotonic()
            rows = collect_system(system, prompts, openai_key)
            ok_count = sum(1 for row in rows if row["ok"])
            print(
                f"[{system['key']}] 수집 완료 {ok_count}/{len(rows)}건,"
                f" {time.monotonic() - started:.0f}초"
            )
        if arguments.retry_failed:
            for _ in range(3):
                if all(row["ok"] for row in rows):
                    break
                before = sum(1 for row in rows if not row["ok"])
                rows = retry_failed_rows(system, rows, prompts_by_id, openai_key)
                after = sum(1 for row in rows if not row["ok"])
                print(f"[{system['key']}] 실패 재시도: {before}건 → 잔여 {after}건")
        cache_path.write_text(
            json.dumps(
                {"fingerprint": system_fingerprint, "rows": rows},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        responses_by_system[system["key"]] = rows

    scored = score_all(prompts, responses_by_system)

    judge_result: dict[str, dict] | None = None
    if not arguments.skip_judge:
        hard_by_id = {item["id"]: item for item in load_hard_items()}
        judged_items = [item for item in load_hard_items() if item.get("judge")]
        texts = {
            system["key"]: {
                row["id"]: row["text"]
                for row in responses_by_system[system["key"]]
                if row["ok"]
            }
            for system in systems
        }
        judge_cache = RESULTS_DIRECTORY / "judge_scores.json"
        mapping_path = RESULTS_DIRECTORY / "judge_mapping.json"
        # 심판 입력·매핑은 seed로 결정적이다 — 먼저 만들어 실행 조건 지문을 캐시와 비교한다.
        judge_input, mapping = build_judge_input(judged_items, texts, systems)
        expected_fingerprint = judge_fingerprint(judge_input, mapping)
        judge_scores: dict | None = None
        if judge_cache.exists() and not arguments.force:
            payload = json.loads(judge_cache.read_text(encoding="utf-8"))
            if (
                isinstance(payload, dict)
                and payload.get("fingerprint") == expected_fingerprint
            ):
                judge_scores = payload["scores"]
            else:
                print("심판 캐시가 현재 실행 조건과 다르다 — 다시 채점한다")
        if judge_scores is None:
            print(f"심판 실행 — {JUDGE_MODEL} ({JUDGE_REASONING_EFFORT})")
            judge_scores = run_codex_judge(
                judge_input, RESULTS_DIRECTORY / "judge_raw.md", len(systems)
            )
            # 라벨→구성 매핑과 심판 입력 기록은 심판이 끝난 뒤에만 남긴다 —
            # 실행 중인 심판이 읽을 수 있는 자리에 매핑을 두지 않는다.
            (RESULTS_DIRECTORY / "judge_input.md").write_text(
                judge_input, encoding="utf-8"
            )
            mapping_path.write_text(
                json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            judge_cache.write_text(
                json.dumps(
                    {"fingerprint": expected_fingerprint, "scores": judge_scores},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        validate_judge_scores(judge_scores, judged_items, len(systems))
        judge_result = judge_means(judge_scores, mapping, hard_by_id)

    note = (
        "로컬 구성은 저장소 코드 게이트웨이(폴백 비활성, provider=local 단언),"
        " GPT 구성은 OpenAI 직접 호출"
    )
    report = render_report(scored, judge_result, note, systems)
    (RESULTS_DIRECTORY / "report.md").write_text(report, encoding="utf-8")
    (RESULTS_DIRECTORY / "report.json").write_text(
        json.dumps(
            build_quality_report(scored, judge_result), ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    print(f"완료 - 결과: {RESULTS_DIRECTORY}")


if __name__ == "__main__":
    main()
