"""9단계 벤치마크의 순수 로직 — 질의 셋 검증, 이미지 생성, 채점, 통계, 비용 계산.

네트워크와 파일 시스템 상태에 의존하지 않아 pytest로 단독 검증한다.
측정 절차는 benchmark_stage9.py, 결정적 검증은 verify_stage9.py가 담당한다.
"""

import base64
import json
import math
import random
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

GENERATION_TASKS = ("rag", "direct", "summary", "classify", "vision")
JUDGED_TASKS = ("rag", "summary")
IMAGE_KINDS = ("solid", "center_rect", "top_bottom")

JUDGE_RUBRIC = (
    "너는 커머스 서비스 AI 응답의 품질 심사관이다. 아래 과제와 응답을 보고"
    " 1~10점 정수로 채점하라.\n"
    "채점 기준: (1) 정확성 — 사실과 제공된 근거에 어긋나지 않는가."
    " (2) 과제 충실 — 질문이 요구한 내용과 형식을 지켰는가."
    " (3) 실용성 — 서비스 응답으로 바로 쓸 수 있게 간결하고 자연스러운가.\n"
    "근거가 제공된 과제에서 근거에 없는 내용을 지어냈다면 4점 이하를 준다.\n"
    '반드시 {"score": <1~10 정수>, "reason": "<한 문장>"} 형식의 JSON만 출력하라.'
)


class DatasetError(ValueError):
    """질의 셋 파일이 벤치마크 계약을 어겼을 때 문제 목록을 담아 던진다."""


class JudgeParseError(ValueError):
    """심판 응답에서 유효한 채점 JSON을 찾지 못했을 때 던진다."""


@dataclass(frozen=True)
class GenerationItem:
    id: str
    task: str
    prompt: str
    context: str | None
    expected: dict[str, list[str]] | None
    image: dict | None

    @property
    def judged(self) -> bool:
        return self.task in JUDGED_TASKS

    def user_message_text(self) -> str:
        if self.context is None:
            return self.prompt
        return f"다음 근거만 참고해서 답하세요.\n\n[근거]\n{self.context}\n\n[질문]\n{self.prompt}"


@dataclass(frozen=True)
class EmbeddingQuery:
    id: str
    text: str
    relevant: tuple[str, ...]


@dataclass(frozen=True)
class Dataset:
    generation: tuple[GenerationItem, ...]
    load_prompts: tuple[str, ...]
    documents: tuple[tuple[str, str], ...]
    queries: tuple[EmbeddingQuery, ...]

    def items_for_task(self, task: str) -> tuple[GenerationItem, ...]:
        return tuple(item for item in self.generation if item.task == task)


def load_dataset(path: Path) -> Dataset:
    """질의 셋 파일을 읽고 스키마·참조 무결성을 검증한 뒤 불변 구조로 돌려준다."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    problems: list[str] = []
    generation = _parse_generation(raw.get("generation", []), problems)
    load_prompts = tuple(raw.get("load_prompts", []))
    if not load_prompts:
        problems.append("load_prompts가 비어 있다")
    documents, queries = _parse_embedding(raw.get("embedding", {}), problems)
    if problems:
        raise DatasetError("; ".join(problems))
    return Dataset(generation, load_prompts, documents, queries)


def _parse_generation(
    raw_items: list[dict], problems: list[str]
) -> tuple[GenerationItem, ...]:
    items: list[GenerationItem] = []
    seen_ids: set[str] = set()
    for raw in raw_items:
        item_id = raw.get("id", "")
        if not item_id or item_id in seen_ids:
            problems.append(f"generation id 누락 또는 중복: {item_id!r}")
            continue
        seen_ids.add(item_id)
        task = raw.get("task", "")
        if task not in GENERATION_TASKS:
            problems.append(f"{item_id}: 알 수 없는 task {task!r}")
            continue
        expected = raw.get("expected")
        if task in JUDGED_TASKS:
            if not raw.get("context"):
                problems.append(f"{item_id}: {task} 과제에 context가 없다")
        elif not (expected and expected.get("any")):
            problems.append(f"{item_id}: 자동 채점 과제에 expected.any가 없다")
        image = raw.get("image")
        if task == "vision":
            image_problem = _image_problem(image)
            if image_problem:
                problems.append(f"{item_id}: {image_problem}")
        if not raw.get("prompt"):
            problems.append(f"{item_id}: prompt가 비어 있다")
        items.append(
            GenerationItem(
                id=item_id,
                task=task,
                prompt=raw.get("prompt", ""),
                context=raw.get("context"),
                expected=expected,
                image=image,
            )
        )
    if not items:
        problems.append("generation 항목이 없다")
    return tuple(items)


def _image_problem(image: dict | None) -> str | None:
    if not image:
        return "vision 과제에 image가 없다"
    kind = image.get("kind")
    if kind not in IMAGE_KINDS:
        return f"알 수 없는 image kind {kind!r}"
    colors = image.get("colors", [])
    required = 1 if kind == "solid" else 2
    if len(colors) < required:
        return f"image kind {kind}에는 색이 {required}개 필요하다"
    for color in colors:
        if len(color) != 3 or any(not 0 <= value <= 255 for value in color):
            return f"잘못된 RGB 값: {color!r}"
    return None


def _parse_embedding(
    raw: dict, problems: list[str]
) -> tuple[tuple[tuple[str, str], ...], tuple[EmbeddingQuery, ...]]:
    documents: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    for raw_doc in raw.get("documents", []):
        doc_id = raw_doc.get("id", "")
        if not doc_id or doc_id in seen_ids or not raw_doc.get("text"):
            problems.append(f"embedding document 불량: {doc_id!r}")
            continue
        seen_ids.add(doc_id)
        documents.append((doc_id, raw_doc["text"]))
    queries: list[EmbeddingQuery] = []
    for raw_query in raw.get("queries", []):
        relevant = tuple(raw_query.get("relevant", []))
        missing = [doc_id for doc_id in relevant if doc_id not in seen_ids]
        if not raw_query.get("id") or not raw_query.get("text") or not relevant:
            problems.append(f"embedding query 불량: {raw_query.get('id')!r}")
            continue
        if missing:
            problems.append(f"{raw_query['id']}: 존재하지 않는 문서 참조 {missing}")
            continue
        queries.append(EmbeddingQuery(raw_query["id"], raw_query["text"], relevant))
    if not documents or not queries:
        problems.append("embedding 문서 또는 질의가 없다")
    return tuple(documents), tuple(queries)


def render_png(image: dict) -> bytes:
    """질의 셋 이미지 명세로 결정적인 RGB PNG 바이트를 만든다 — 외부 라이브러리 불필요."""
    size = image.get("size", 96)
    colors = [tuple(color) for color in image["colors"]]
    kind = image["kind"]
    rows: list[bytes] = []
    quarter, half, three_quarter = size // 4, size // 2, size * 3 // 4
    for y in range(size):
        row = bytearray()
        for x in range(size):
            pixel = colors[0]
            if (
                kind == "center_rect"
                and quarter <= x < three_quarter
                and quarter <= y < three_quarter
            ):
                pixel = colors[1]
            if kind == "top_bottom" and y >= half:
                pixel = colors[1]
            row.extend(pixel)
        rows.append(bytes(row))
    return _encode_png(size, size, rows)


def _encode_png(width: int, height: int, rows: list[bytes]) -> bytes:
    def chunk(tag: bytes, payload: bytes) -> bytes:
        checksum = zlib.crc32(tag + payload)
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", checksum)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    scanlines = b"".join(b"\x00" + row for row in rows)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )


def image_data_url(image: dict) -> str:
    encoded = base64.b64encode(render_png(image)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def sse_data_payload(line: str) -> str | None:
    """SSE 한 줄에서 data 페이로드만 꺼낸다 — data 라인이 아니면 None."""
    if not line.startswith("data:"):
        return None
    return line[len("data:") :].strip()


def chunk_delta_content(chunk: dict) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    return choices[0].get("delta", {}).get("content") or ""


def chunk_usage(chunk: dict) -> dict | None:
    usage = chunk.get("usage")
    if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
        return usage
    return None


def keyword_match(text: str, expected: dict[str, list[str]]) -> bool:
    """자동 채점 — any 목록 중 하나 이상, all 목록 전부가 응답에 있어야 정답이다."""
    haystack = text.casefold()
    any_keywords = expected.get("any", [])
    all_keywords = expected.get("all", [])
    if any_keywords and not any(k.casefold() in haystack for k in any_keywords):
        return False
    return all(k.casefold() in haystack for k in all_keywords)


def percentile(values: list[float], ratio: float) -> float:
    """최근접 순위(nearest-rank) 백분위 — 부하 지연 분포 요약에 사용한다."""
    if not values:
        raise ValueError("백분위를 계산할 값이 없다")
    ordered = sorted(values)
    rank = max(1, math.ceil(ratio * len(ordered)))
    return ordered[rank - 1]


def build_judge_messages(
    prompt: str, context: str | None, answer: str
) -> list[dict[str, str]]:
    parts = [f"[과제]\n{prompt}"]
    if context:
        parts.append(f"[제공된 근거]\n{context}")
    parts.append(f"[채점할 응답]\n{answer}")
    return [
        {"role": "system", "content": JUDGE_RUBRIC},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def parse_judge_score(reply: str) -> int:
    """심판 응답에서 1~10 정수 score를 꺼낸다 — 형식 위반은 JudgeParseError."""
    candidates = [reply]
    start, end = reply.find("{"), reply.rfind("}")
    if start != -1 and end > start:
        candidates.append(reply[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        score = parsed.get("score") if isinstance(parsed, dict) else None
        if isinstance(score, int) and not isinstance(score, bool) and 1 <= score <= 10:
            return score
    raise JudgeParseError(f"채점 JSON을 찾지 못했다: {reply[:120]!r}")


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    if norm == 0:
        raise ValueError("영벡터와는 코사인 유사도를 계산할 수 없다")
    return dot / norm


def rank_documents(
    query_vector: list[float], document_vectors: dict[str, list[float]]
) -> list[str]:
    scored = [
        (cosine_similarity(query_vector, vector), doc_id)
        for doc_id, vector in document_vectors.items()
    ]
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return [doc_id for _, doc_id in scored]


def retrieval_metrics(
    rankings: dict[str, list[str]], relevant: dict[str, tuple[str, ...]]
) -> dict[str, float]:
    """질의별 순위 목록으로 Recall@1·Recall@3·MRR을 계산한다."""
    if not rankings:
        raise ValueError("순위 결과가 없다")
    recall_1 = recall_3 = reciprocal_sum = 0.0
    for query_id, ranked in rankings.items():
        relevant_ids = set(relevant[query_id])
        if set(ranked[:1]) & relevant_ids:
            recall_1 += 1
        if set(ranked[:3]) & relevant_ids:
            recall_3 += 1
        for position, doc_id in enumerate(ranked, start=1):
            if doc_id in relevant_ids:
                reciprocal_sum += 1 / position
                break
    count = len(rankings)
    return {
        "recall_at_1": recall_1 / count,
        "recall_at_3": recall_3 / count,
        "mrr": reciprocal_sum / count,
    }


def api_cost_usd(
    prompt_tokens: int, completion_tokens: int, pricing: dict[str, float]
) -> float:
    """토큰 사용량을 백만 토큰당 달러 단가로 환산한다."""
    input_cost = prompt_tokens / 1_000_000 * pricing["usd_per_million_input"]
    output_cost = completion_tokens / 1_000_000 * pricing["usd_per_million_output"]
    return input_cost + output_cost


def marginal_energy_wh(
    load_average_watt: float, idle_watt: float, duration_seconds: float
) -> float:
    """부하 구간에서 유휴 대비 추가로 쓴 전력량(Wh) — 음수면 0으로 본다."""
    extra_watt = max(0.0, load_average_watt - idle_watt)
    return extra_watt * duration_seconds / 3600


def electricity_krw(energy_wh: float, krw_per_kwh: float) -> float:
    return energy_wh / 1000 * krw_per_kwh


def build_blind_samples(
    samples: list[tuple[str, str, dict[str, str]]], seed: int
) -> tuple[str, dict[str, dict[str, str]]]:
    """모델명을 가린 블라인드 비교 자료와 별도 정답 매핑을 만든다."""
    rng = random.Random(seed)
    lines = [
        "# 9단계 블라인드 품질 확인 자료",
        "",
        "모델명을 가린 응답이다. 정답 매핑은 blind_key.json에 있다 — 비교 후에 열어볼 것.",
        "",
    ]
    key: dict[str, dict[str, str]] = {}
    for item_id, prompt, responses in samples:
        model_names = sorted(responses)
        rng.shuffle(model_names)
        labels = [chr(ord("A") + index) for index in range(len(model_names))]
        key[item_id] = dict(zip(labels, model_names, strict=True))
        lines.append(f"## {item_id}")
        lines.append(f"**과제:** {prompt}")
        lines.append("")
        for label, model in zip(labels, model_names, strict=True):
            lines.append(f"### 응답 {label}")
            lines.append(responses[model].strip() or "(빈 응답)")
            lines.append("")
    return "\n".join(lines), key
