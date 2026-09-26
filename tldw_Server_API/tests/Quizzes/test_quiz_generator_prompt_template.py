import os
from itertools import combinations
from typing import Any

import pytest

os.environ.setdefault("TEST_MODE", "1")
pytestmark = pytest.mark.unit

from tldw_Server_API.app.core.exceptions import BadRequestError
from tldw_Server_API.app.services import quiz_generator
from tldw_Server_API.app.services.quiz_generator import (
    _build_generation_profile_instruction,
    _coerce_question_types,
    get_quiz_generation_profiles,
)


def test_profile_normalization_uses_domain_error_for_invalid_requests():
    with pytest.raises(BadRequestError):
        quiz_generator._normalize_generation_profile("unknown-profile")


def test_profile_normalization_accepts_available_osce_profile():
    assert quiz_generator._normalize_generation_profile("osce_scenario") == "osce_scenario"
    assert quiz_generator._normalize_generation_profile("osce") == "osce_scenario"


def test_quiz_generation_prompt_formats_with_literal_citation_object():
    rendered_prompt = quiz_generator._format_quiz_generation_prompt(
        num_questions=3,
        content="Sample content",
        difficulty="mixed",
        question_types=["multiple_choice", "true_false"],
        focus_instruction="- Focus on these topics: testing",
        source_contract="- Allowed sources for source_citations.source_type/source_id: note:note-1",
    )

    assert '"label": "Optional citation label"' in rendered_prompt
    assert '"source_type": "media" | "note" | "flashcard_deck" | "flashcard_card"' in rendered_prompt
    assert '"question_type": "multiple_choice"' in rendered_prompt
    assert '"question_type": "true_false"' in rendered_prompt
    assert "For EMQ" not in rendered_prompt
    assert "Allowed sources for source_citations.source_type/source_id: note:note-1" in rendered_prompt
    assert "{num_questions}" not in rendered_prompt
    assert "{content}" not in rendered_prompt


def test_quiz_prompt_gives_canonical_citation_field_pairs():
    source_contract = quiz_generator._build_source_contract(
        [
            {"source_type": "media", "source_id": "59"},
            {"source_type": "media", "source_id": "60"},
        ]
    )
    rendered_prompt = quiz_generator._format_quiz_generation_prompt(
        num_questions=2,
        content="Source: media:59\nSample content",
        difficulty="mixed",
        question_types=["multiple_choice"],
        focus_instruction="",
        source_contract=source_contract,
    )

    assert '{"source_type": "media", "source_id": "59"}' in rendered_prompt
    assert '{"source_type": "media", "source_id": "60"}' in rendered_prompt
    assert '"source_id": "media:59"' not in rendered_prompt


def test_source_contract_preserves_existing_prefix_in_canonical_id():
    contract = quiz_generator._build_source_contract([{"source_type": "note", "source_id": "note:n1"}])

    assert '"source_id": "note:n1"' in contract
    assert "preserve source_id exactly" in contract
    assert "never include source_type in source_id" not in contract


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


@pytest.mark.parametrize(
    ("plan_item", "expected_shape", "forbidden_shape"),
    [
        (
            {"question_type": "multiple_choice", "count": 1, "option_count": 5},
            '"options": ["A", "B", "C", "D", "E"]',
            '"options": ["A", "B", "C", "D"],',
        ),
        (
            {"question_type": "multi_select", "count": 1, "option_count": 2},
            '"correct_answer": [0, 1]',
            '"correct_answer": [0, 2]',
        ),
        (
            {"question_type": "matching", "count": 1, "pair_count": 2},
            '"options": ["A", "B"]',
            '"options": ["CPU", "RAM", "Disk", "GPU"]',
        ),
    ],
)
def test_planned_prompt_examples_obey_selected_option_and_pair_counts(plan_item, expected_shape, forbidden_shape):
    rendered_prompt = quiz_generator._format_quiz_generation_prompt(
        num_questions=1,
        content="Sample content",
        difficulty="mixed",
        question_types=None,
        question_plan=[plan_item],
        focus_instruction="",
        source_contract="Allowed source: note:1",
    )

    assert expected_shape in rendered_prompt
    assert forbidden_shape not in rendered_prompt


@pytest.mark.parametrize(
    "selected_types",
    [
        ["multiple_choice"],
        ["true_false"],
        ["fill_blank"],
        ["multi_select"],
        ["matching"],
        ["multiple_choice", "true_false"],
    ],
)
def test_quiz_prompt_advertises_only_selected_question_shapes(selected_types):
    rendered_prompt = quiz_generator._format_quiz_generation_prompt(
        num_questions=1,
        content="The trial lasted 14 days.",
        difficulty="mixed",
        question_types=selected_types,
        focus_instruction="",
        source_contract="Allowed source: note:1",
    )

    for q_type in selected_types:
        assert f'"question_type": "{q_type}"' in rendered_prompt
    for q_type in set(quiz_generator.SUPPORTED_GENERATED_QUESTION_TYPES) - set(selected_types):
        assert f'"question_type": "{q_type}"' not in rendered_prompt
    assert "The trial lasted 14 days." in rendered_prompt


@pytest.mark.parametrize(
    "selected_types",
    [
        list(subset)
        for count in range(1, 6)
        for subset in combinations(
            ["multiple_choice", "true_false", "fill_blank", "multi_select", "matching"],
            count,
        )
    ],
)
def test_prompt_shape_choices_equal_selected_types_for_any_subset(selected_types):
    rendered_prompt = quiz_generator._format_quiz_generation_prompt(
        num_questions=1,
        content="Source evidence",
        difficulty="mixed",
        question_types=selected_types,
        focus_instruction="",
        source_contract="Allowed source: note:1",
    )

    advertised = {
        q_type
        for q_type in quiz_generator.SUPPORTED_GENERATED_QUESTION_TYPES
        if f'"question_type": "{q_type}"' in rendered_prompt
    }
    assert advertised == set(selected_types)


@pytest.mark.parametrize("profile", ["best_of_five", "emq", "assertion_reasoning"])
def test_quiz_prompt_preserves_locked_profile_instructions(profile):
    rendered_prompt = quiz_generator._format_quiz_generation_prompt(
        num_questions=2,
        content="The trial lasted 14 days.",
        difficulty="mixed",
        question_types=["multiple_choice"],
        focus_instruction=_build_generation_profile_instruction(profile),
        source_contract="Allowed source: note:1",
        generation_profile=profile,
    )

    assert '"question_type": "multiple_choice"' in rendered_prompt
    if profile == "best_of_five":
        assert "exactly five answer options" in rendered_prompt
        assert '"options": ["A", "B", "C", "D", "E"]' in rendered_prompt
    elif profile == "emq":
        assert "shared option bank" in rendered_prompt
        assert '"group_id": "group-1"' in rendered_prompt
        assert '"group_prompt": "Shared question prompt"' in rendered_prompt
    else:
        assert "separate assertion and reason fields" in rendered_prompt
        assert '"assertion": "An evidence-backed assertion"' in rendered_prompt
        assert '"reason": "An evidence-backed reason"' in rendered_prompt
        assert ('"options": ["' + quiz_generator.ASSERTION_REASONING_OPTIONS[0]) in rendered_prompt


@pytest.mark.parametrize("q_type", ["multi_select", "matching"])
def test_legacy_advanced_question_type_is_normalized(q_type):
    raw = {
        "question_type": q_type,
        "question_text": "Which terms match the source?",
        "options": ["A", "B", "C", "D"],
        "correct_answer": [0, 2] if q_type == "multi_select" else {"A": "one", "B": "two", "C": "three", "D": "four"},
    }

    questions = quiz_generator._normalize_questions([raw], "note", "1")

    assert len(questions) == 1
    assert questions[0]["question_type"] == q_type


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


def test_quiz_generation_prompt_threads_active_profile_to_plan_coercion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    original = quiz_generator._coerce_generation_plan

    def capture_profile(**kwargs: Any) -> list[dict[str, Any]]:
        captured["generation_profile"] = kwargs.get("generation_profile")
        return original(**kwargs)

    monkeypatch.setattr(quiz_generator, "_coerce_generation_plan", capture_profile)

    quiz_generator._format_quiz_generation_prompt(
        num_questions=1,
        content="Sample content",
        difficulty="mixed",
        question_types=["multiple_choice"],
        focus_instruction="",
        source_contract="- Allowed sources: note:note-1",
        generation_profile="best_of_five",
    )

    assert captured["generation_profile"] == "best_of_five"


@pytest.mark.parametrize("profile", ["best_of_five", "emq", "assertion_reasoning"])
def test_locked_generation_profiles_reject_question_plan(profile: str) -> None:
    with pytest.raises(ValueError, match="question_plan is only supported"):
        quiz_generator._coerce_generation_plan(
            num_questions=1,
            question_plan=[{"question_type": "multiple_choice", "count": 1, "option_count": 5}],
            generation_profile=profile,
        )


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


def test_assertion_reasoning_prompt_supports_required_fields_and_rules():
    rendered_prompt = quiz_generator._format_quiz_generation_prompt(
        num_questions=1,
        content="Sample content",
        difficulty="mixed",
        question_types=["multiple_choice"],
        focus_instruction=_build_generation_profile_instruction("assertion_reasoning"),
        source_contract="Allowed source: note:1",
        generation_profile="assertion_reasoning",
    )
    assert "separate assertion and reason fields" in rendered_prompt
    assert "For Assertion / Reasoning" in rendered_prompt
    assert "Do not provide hidden chain-of-thought" in rendered_prompt
