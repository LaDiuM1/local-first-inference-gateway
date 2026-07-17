"""9단계 품질 심층 평가 순수 로직 검증 — 객관식 파싱·정답 판정·형식 검사."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS_DIRECTORY = str(Path(__file__).resolve().parent.parent / "scripts")
if SCRIPTS_DIRECTORY not in sys.path:
    sys.path.insert(0, SCRIPTS_DIRECTORY)

import bench_stage9_quality_deep as quality_deep  # noqa: E402
from bench_stage9_quality_deep import (  # noqa: E402
    format_correct,
    hard_auto_correct,
    judge_fingerprint,
    keyword_correct,
    load_hard_items,
    numeric_correct,
    parse_choice,
    response_fingerprint,
    strip_code_fence,
    validate_judge_scores,
)


def test_parse_choice_prefers_labeled_answer_and_falls_back_to_last_letter() -> None:
    assert parse_choice("풀이 과정...\n정답: B") == "B"
    assert parse_choice("정답은 C입니다") == "C"
    assert parse_choice("보기 중 최종 선택은 D 입니다") == "D"
    # 한글 조사에 붙은 글자는 선택지 표기로 단정하지 않는다 — 보수적으로 비판정.
    assert parse_choice("A와 B를 비교하면 결국 D가 맞다") is None
    assert parse_choice("정답을 고르기 어렵다") is None


def test_numeric_correct_reads_final_answer_only() -> None:
    assert numeric_correct("최종 결제 금액: 61,300원", 61300)
    assert numeric_correct("답은 18개입니다", 18)
    assert not numeric_correct("18.75이므로 반올림", 18)
    assert not numeric_correct("약 61,000원", 61300)
    # 자릿수가 다른 값은 부분 문자열로도 정답 처리하지 않는다.
    assert not numeric_correct("적립 포인트: 1000", 100)
    # 중간 계산에 정답 값이 있어도 마지막 줄의 최종 답이 틀리면 오답이다.
    assert not numeric_correct("할인액은 3,000원.\n적립 포인트: 3600", 3000)
    assert numeric_correct("계산: 100,000 × 3% = 3,000\n적립 포인트: 3,000", 3000)
    # 값을 부정하는 줄은 최종 답이 아니다.
    assert not numeric_correct("18이 아닙니다", 18)
    assert numeric_correct("18이 아닙니다.\n정답: 20", 20)
    # 마지막 줄이 정정문이면 이전 답으로 되돌아가지 않는다 — 올바른 정정을
    # 오답 처리하는 방향으로 뒤집히기 때문이다.
    assert not numeric_correct("정답: 18\n정정: 18이 아니라 17입니다", 18)


def test_keyword_correct_ignores_negated_answer_keywords() -> None:
    expected = {"any": ["가능"], "negations": ["불가능", "불가"]}
    assert keyword_correct("하자 반품이라 30일 이내 가능합니다.", expected)
    # '불가능'은 '가능'을 부분 문자열로 포함하지만 정답이 아니다.
    assert not keyword_correct("단순 변심 기준이라 반품이 불가능합니다.", expected)
    # 부정 표현과 정답 서술이 함께 있으면 정답 서술로 판정한다.
    assert keyword_correct("변심이면 불가하지만 하자라서 가능합니다.", expected)


def test_keyword_correct_final_line_only_skips_negated_mentions() -> None:
    expected = {"any": ["화요일"], "final_line_only": True}
    assert keyword_correct("계산해 보면...\n화요일", expected)
    # 본문 언급은 답이 아니다 — 마지막 답 줄만 본다.
    assert not keyword_correct("화요일 도착이 목표였지만\n수요일", expected)
    # 값을 부정하는 서술('화요일이 아니라 수요일')은 정답 처리하지 않는다.
    assert not keyword_correct("화요일이 아니라 수요일입니다", expected)
    # 마지막 줄이 정정문이면 이전 줄의 답으로 되돌아가지 않는다.
    assert not keyword_correct("화요일\n화요일이 아니라 수요일입니다", expected)


def test_keyword_correct_requires_all_facts_together() -> None:
    expected = {
        "any": ["가능", "할 수 있", "반품 대상"],
        "all": ["판매자"],
        "negations": ["불가능", "불가", "할 수 없", "안 됩", "안 되"],
        # '아닙니다'는 음절 조합상 '아니'를 포함하지 않는다 — 두 활용형을 모두 적는다.
        "reject_any": [
            "가능하지 않",
            "되지 않",
            "판매자가 아니",
            "판매자가 아닙",
            "부담이 아니",
            "부담이 아닙",
        ],
    }
    assert keyword_correct("반품 가능합니다. 배송비는 판매자가 부담합니다.", expected)
    assert keyword_correct(
        "제조 불량은 30일 이내 반품 대상이며 배송비는 판매자 부담입니다.", expected
    )
    # 필수 사실(배송비 부담 주체)이 틀리면 반품 가능 서술만으로는 정답이 아니다.
    assert not keyword_correct("반품 가능합니다. 배송비는 구매자 부담입니다.", expected)
    # '안 됩니다'는 '됩니다'류 표현으로 오탐되지 않는다.
    assert not keyword_correct(
        "반품은 안 됩니다. 배송비는 판매자가 부담합니다.", expected
    )
    # 반대 결론('가능하지 않다')과 관계 부정('판매자가 아니라')도 정답 처리하지 않는다.
    assert not keyword_correct(
        "반품은 가능하지 않습니다. 배송비는 판매자가 부담합니다.", expected
    )
    assert not keyword_correct(
        "반품 가능합니다. 배송비는 판매자가 아니라 구매자 부담입니다.", expected
    )
    # 거부 구문은 제거가 아니라 오답 확정이다 — 정답 키워드가 구문에 남아도 통과 불가.
    assert not keyword_correct(
        "반품 대상이 되지 않습니다. 배송비는 판매자가 부담합니다.", expected
    )
    assert not keyword_correct(
        "반품 가능합니다. 배송비는 판매자 부담이 아닙니다.", expected
    )


def test_fingerprints_change_with_model_and_judge_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts = [{"id": "calc-01", "prompt": "질문"}]
    system = {"key": "local-none", "kind": "local", "reasoning": None}
    base = response_fingerprint(prompts, system, "gemma4:12b-it-qat")
    # 라우팅되는 실제 모델이 바뀌면 응답 캐시 지문도 바뀌어야 한다.
    assert base != response_fingerprint(prompts, system, "other-model")

    judge_base = judge_fingerprint("입력 문서", {"trap-01": {"R1": "local-none"}})
    monkeypatch.setattr(
        quality_deep,
        "JUDGE_PROMPT_TEMPLATE",
        "다른 채점 지시 {input_path} {last} {score_example}",
    )
    # 채점 지시 템플릿이 바뀌면 심판 캐시 지문도 바뀌어야 한다.
    assert judge_base != judge_fingerprint(
        "입력 문서", {"trap-01": {"R1": "local-none"}}
    )


def test_run_codex_judge_builds_isolated_skip_git_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(args: list[str], **kwargs: object) -> SimpleNamespace:
        captured["args"] = args
        Path(args[args.index("-o") + 1]).write_text(
            '{"trap-01": {"R1": 9}}', encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(quality_deep.subprocess, "run", fake_run)
    monkeypatch.setattr(quality_deep, "resolve_codex_command", lambda: "codex.exe")

    scores = quality_deep.run_codex_judge("입력", tmp_path / "raw.md", label_count=1)

    args = captured["args"]
    # 격리 임시 디렉터리는 git 저장소가 아니다 — 신뢰 검사를 끄지 않으면 실행이 죽는다.
    assert args[args.index("exec") + 1] == "--skip-git-repo-check"
    workspace = Path(args[args.index("-C") + 1])
    assert workspace != Path.cwd()
    assert scores == {"trap-01": {"R1": 9}}


def test_strip_code_fence_removes_fences_only() -> None:
    fenced = '```json\n{"a": 1}\n```'
    assert strip_code_fence(fenced) == '{"a": 1}'
    assert strip_code_fence('{"a": 1}') == '{"a": 1}'


def test_format_correct_enforces_exact_output_contract() -> None:
    json_check = {"type": "json", "category": "배송", "urgency_min": 4}
    valid = '{"category": "배송", "urgency": 5, "summary": "냉장고 미도착 문의"}'
    assert format_correct(json_check, valid)
    # 'JSON으로만 출력' 과제에서 코드펜스는 다른 텍스트다 — 형식 위반.
    assert not format_correct(json_check, f"```json\n{valid}\n```")
    assert not format_correct(json_check, '{"category": "환불", "urgency": 5}')
    assert not format_correct(json_check, '{"category": "배송", "urgency": 2}')
    # JSON이지만 객체가 아니거나 summary가 문자열이 아니면 위반이다 — 예외가 아니라 오답.
    assert not format_correct(json_check, "[1, 2]")
    assert not format_correct(
        json_check, '{"category": "배송", "urgency": 5, "summary": null}'
    )

    bullet_check = {"type": "bullets", "count": 3, "max_chars": 20}
    bullets = "- 무료 배송 기준 5만 원\n- 도서산간 3천 원 통일\n- 새벽 배송 확대"
    assert format_correct(bullet_check, bullets)
    assert not format_correct(bullet_check, bullets + "\n추가 설명 문장")
    assert not format_correct(bullet_check, "- " + "긴 항목 " * 10 + "\n- 둘\n- 셋")
    # '글머리 기호(-)' 과제에서 다른 글머리 문자는 형식 위반이다.
    assert not format_correct(
        bullet_check, "* 무료 배송 기준 상향\n* 배송비 통일\n* 새벽 배송 확대"
    )
    # 빈 불릿과 이모지도 위반이다.
    assert not format_correct(bullet_check, "-\n-\n-")
    assert not format_correct(
        bullet_check, "- 무료 배송 기준 상향 🚚\n- 배송비 통일\n- 새벽 배송 확대"
    )

    forbidden_check = {"type": "forbidden", "words": ["교환"]}
    assert format_correct(forbidden_check, "다른 사이즈로 교체를 신청해 주세요.")
    assert not format_correct(forbidden_check, "교환 절차를 안내드립니다.")


def test_validate_judge_scores_rejects_missing_labels_and_bad_scores() -> None:
    judged_items = [{"id": "trap-01"}, {"id": "fmt-01"}]
    complete = {"trap-01": {"R1": 9, "R2": 4}, "fmt-01": {"R1": 1, "R2": 10}}
    validate_judge_scores(complete, judged_items, label_count=2)
    with pytest.raises(RuntimeError, match="라벨 누락"):
        validate_judge_scores(
            {"trap-01": {"R1": 9}, "fmt-01": {"R1": 1, "R2": 10}},
            judged_items,
            label_count=2,
        )
    with pytest.raises(RuntimeError, match="범위 위반"):
        validate_judge_scores(
            {"trap-01": {"R1": 99, "R2": 4}, "fmt-01": {"R1": 1, "R2": 10}},
            judged_items,
            label_count=2,
        )
    with pytest.raises(RuntimeError, match="범위 위반"):
        validate_judge_scores(
            {"trap-01": {"R1": True, "R2": 4}, "fmt-01": {"R1": 1, "R2": 10}},
            judged_items,
            label_count=2,
        )


def test_hard_auto_correct_dispatches_by_item_fields() -> None:
    format_item = {"format_check": {"type": "forbidden", "words": ["교환"]}}
    value_item = {"answer_value": 260}
    keyword_item = {"expected": {"any": ["화요일"]}}
    judge_only_item = {"judge": True}
    assert hard_auto_correct(format_item, "사이즈 교체 안내") is True
    assert hard_auto_correct(value_item, "최종 재고는 260개") is True
    assert hard_auto_correct(keyword_item, "화요일에 도착합니다") is True
    assert hard_auto_correct(judge_only_item, "아무 답변") is None


def test_hard_dataset_items_have_scoring_contract() -> None:
    items = load_hard_items()
    identifiers = [item["id"] for item in items]
    assert len(identifiers) == len(set(identifiers))
    for item in items:
        scorable = (
            "answer_value" in item or "expected" in item or "format_check" in item
        )
        assert scorable or item.get("judge"), item["id"]
        if item.get("judge"):
            assert item.get("judge_note") or "format_check" in item, item["id"]
