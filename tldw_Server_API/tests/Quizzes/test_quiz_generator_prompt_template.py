import os

import pytest

os.environ.setdefault("TEST_MODE", "1")
pytestmark = pytest.mark.unit

from tldw_Server_API.app.core.exceptions import BadRequestError
from tldw_Server_API.app.services import quiz_generator
from tldw_Server_API.app.services.quiz_generator import (
    QUIZ_GENERATION_PROMPT,
    _build_generation_profile_instruction,
    _coerce_question_types,
    get_quiz_generation_profiles,
)


@pytest.mark.parametrize("profile", ["unknown-profile", "osce_scenario"])
def test_profile_normalization_uses_domain_error_for_invalid_requests(profile: str):
    with pytest.raises(BadRequestError):
        quiz_generator._normalize_generation_profile(profile)


def test_quiz_generation_prompt_template_formats_with_literal_citation_object():
    rendered_prompt = QUIZ_GENERATION_PROMPT.format(
        num_questions=3,
        content="Sample content",
        difficulty="mixed",
        question_types="multiple_choice, true_false",
        focus_instruction="- Focus on these topics: testing",
        source_contract="- Allowed sources for source_citations.source_type/source_id: note:note-1",
    )

    assert '"label": "Optional citation label"' in rendered_prompt
    assert '"source_type": "media" | "note" | "flashcard_deck" | "flashcard_card"' in rendered_prompt
    assert '"group_id": "Optional EMQ group identifier"' in rendered_prompt
    assert '"group_prompt": "Optional shared EMQ group prompt"' in rendered_prompt
    assert '"correct_answer": 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9' in rendered_prompt
    assert "For EMQ" in rendered_prompt
    assert "at least two stems" in rendered_prompt
    assert "Allowed sources for source_citations.source_type/source_id: note:note-1" in rendered_prompt
    assert "{num_questions}" not in rendered_prompt
    assert "{content}" not in rendered_prompt


def test_quiz_generation_prompt_includes_all_planned_question_shapes():
    rendered_prompt = quiz_generator._format_quiz_generation_prompt(
        num_questions=5,
        content="Sample content",
        difficulty="mixed",
        question_types=None,
        question_plan=[
            {"question_type": "multiple_choice", "count": 1, "option_count": 5},
            {"question_type": "multi_select", "count": 1, "option_count": 5},
            {"question_type": "matching", "count": 1, "pair_count": 2},
            {"question_type": "true_false", "count": 1},
            {"question_type": "fill_blank", "count": 1},
        ],
        focus_instruction="",
        source_contract="- Allowed sources for source_citations.source_type/source_id: note:note-1",
    )

    assert "multiple_choice: 1 question(s), exactly 5 options" in rendered_prompt
    assert "multi_select: 1 question(s), exactly 5 options" in rendered_prompt
    assert "matching: 1 question(s), exactly 2 pairs" in rendered_prompt
    assert "true_false: 1 question(s)" in rendered_prompt
    assert "fill_blank: 1 question(s)" in rendered_prompt
    assert '"question_type": "multi_select"' in rendered_prompt
    assert '"question_type": "matching"' in rendered_prompt
    assert "options must be array of 4 strings" not in rendered_prompt


def test_quiz_generation_prompt_preserves_source_content_when_removing_legacy_hints():
    content = 'Evidence excerpt: keep literal "options": ["A", "B", "C", "D"] from the source.'

    rendered_prompt = quiz_generator._format_quiz_generation_prompt(
        num_questions=1,
        content=content,
        difficulty="mixed",
        question_types=None,
        question_plan=[{"question_type": "multiple_choice", "count": 1, "option_count": 5}],
        focus_instruction="",
        source_contract="- Allowed sources for source_citations.source_type/source_id: note:note-1",
    )

    assert content in rendered_prompt
    assert "Planned question requirements" in rendered_prompt


def test_best_of_five_profile_exposes_prompt_contract_and_question_defaults():
    profiles = get_quiz_generation_profiles()
    best_of_five = next(profile for profile in profiles if profile["id"] == "best_of_five")

    assert best_of_five["status"] == "available"
    assert best_of_five["default_question_types"] == ["multiple_choice"]
    assert _coerce_question_types(None, generation_profile="best_of_five") == ["multiple_choice"]

    instruction = _build_generation_profile_instruction("best_of_five")
    assert "Best of Five" in instruction
    assert "exactly five answer options" in instruction


def test_emq_profile_exposes_shared_bank_multiple_choice_contract():
    profiles = get_quiz_generation_profiles()
    emq = next(profile for profile in profiles if profile["id"] == "emq")

    assert emq["status"] == "available"
    assert emq["default_question_types"] == ["multiple_choice"]
    assert _coerce_question_types(None, generation_profile="emq") == ["multiple_choice"]
    assert _coerce_question_types(
        ["true_false", "multiple_choice", "fill_blank"],
        generation_profile="emq",
    ) == ["multiple_choice"]

    instruction = _build_generation_profile_instruction("emq")
    assert "shared option bank" in instruction.lower()
    assert "at least two stems" in instruction.lower()


def test_assertion_reasoning_profile_exposes_mcq_prompt_contract():
    profiles = get_quiz_generation_profiles()
    assertion_reasoning = next(profile for profile in profiles if profile["id"] == "assertion_reasoning")

    assert assertion_reasoning["status"] == "available"
    assert assertion_reasoning["default_question_types"] == ["multiple_choice"]
    assert _coerce_question_types(None, generation_profile="assertion_reasoning") == ["multiple_choice"]
    assert _coerce_question_types(
        ["true_false", "fill_blank"],
        generation_profile="assertion_reasoning",
    ) == ["multiple_choice"]

    instruction = _build_generation_profile_instruction("assertion_reasoning")
    assert "separate assertion and reason fields" in instruction
    assert "A. Both the assertion and reason are true, and the reason correctly explains the assertion." in instruction
    assert "B. Both the assertion and reason are true, but the reason does not explain the assertion." in instruction
    assert "C. The assertion is true, but the reason is false." in instruction
    assert "D. The assertion is false, but the reason is true." in instruction
    assert "E. Both the assertion and reason are false." in instruction
    assert "concise evidence-backed rationale" in instruction
    assert "Do not provide hidden chain-of-thought" in instruction


def test_common_prompt_supports_assertion_reasoning_fields_and_rules():
    assert '"assertion": "Optional assertion for assertion_reasoning"' in QUIZ_GENERATION_PROMPT
    assert '"reason": "Optional reason for assertion_reasoning"' in QUIZ_GENERATION_PROMPT
    assert "For Assertion / Reasoning" in QUIZ_GENERATION_PROMPT
    assert "Do not provide hidden chain-of-thought" in QUIZ_GENERATION_PROMPT
