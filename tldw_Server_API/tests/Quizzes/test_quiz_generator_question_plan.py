import os

import pytest

os.environ.setdefault("TEST_MODE", "1")
pytestmark = pytest.mark.unit

from tldw_Server_API.app.services import quiz_generator


def test_coerce_options_keeps_exact_planned_option_count():
    options = ["A", "B", "C", "D", "E"]

    assert quiz_generator._coerce_options(options, expected_count=5) == options


def test_normalize_mc_answer_supports_five_options():
    options = ["A", "B", "C", "D", "E"]

    assert quiz_generator._normalize_mc_answer("E", options) == 4


@pytest.mark.parametrize(
    ("plan", "message"),
    [
        (
            [
                {"question_type": "multiple_choice", "count": 1},
                {"question_type": "multiple_choice", "count": 1},
            ],
            "duplicate question_type",
        ),
        ([{"question_type": "multiple_choice", "count": 1}], "counts must sum to num_questions"),
        ([{"question_type": "multiple_choice", "count": 2, "option_count": 7}], "option_count must be between 2 and 6"),
        ([{"question_type": "multi_select", "count": 2, "pair_count": 2}], "pair_count is only valid for matching"),
        ([{"question_type": "matching", "count": 2, "option_count": 4}], "option_count is not valid for matching"),
        ([{"question_type": "matching", "count": 2, "pair_count": 7}], "pair_count must be between 2 and 6"),
        ([{"question_type": "fill_blank", "count": 2, "option_count": 4}], "option_count and pair_count are not valid"),
        ([{"question_type": "true_false", "count": 2, "pair_count": 2}], "option_count and pair_count are not valid"),
    ],
)
def test_coerce_generation_plan_rejects_invalid_service_plan_invariants(plan, message):
    with pytest.raises(ValueError, match=message):
        quiz_generator._coerce_generation_plan(num_questions=2, question_plan=plan)


def test_coerce_generation_plan_defaults_planned_shape_counts():
    plan = quiz_generator._coerce_generation_plan(
        num_questions=2,
        question_plan=[
            {"question_type": "multi_select", "count": 1},
            {"question_type": "matching", "count": 1},
        ],
    )

    assert plan == [
        {"question_type": "multi_select", "count": 1, "option_count": 4},
        {"question_type": "matching", "count": 1, "pair_count": 4},
    ]


@pytest.mark.parametrize(
    ("raw_question", "plan_item", "message"),
    [
        (
            {
                "question_type": "multiple_choice",
                "question_text": "What does CPU mean?",
                "options": ["Processor", "Memory", "Storage", "Display"],
                "correct_answer": 0,
            },
            {"question_type": "multiple_choice", "count": 1, "option_count": 5},
            "Question 1 multiple_choice invalid: Expected 5 options, got 4",
        ),
        (
            {
                "question_type": "fill_blank",
                "question_text": "The CPU executes instructions.",
                "correct_answer": "CPU",
            },
            {"question_type": "fill_blank", "count": 1},
            "Question 1 fill_blank invalid: fill_blank question_text must contain ___",
        ),
    ],
)
def test_normalize_planned_questions_reports_original_normalization_error(raw_question, plan_item, message):
    with pytest.raises(ValueError, match=message):
        quiz_generator._normalize_planned_questions(
            [raw_question],
            [plan_item],
            default_source_type="note",
            default_source_id="note-1",
        )


def test_normalize_planned_questions_reports_invalid_extra_question_detail():
    with pytest.raises(ValueError, match="Question 2 multiple_choice invalid: Expected 5 options, got 4"):
        quiz_generator._normalize_planned_questions(
            [
                {
                    "question_type": "multiple_choice",
                    "question_text": "What does CPU mean?",
                    "options": ["Processor", "Memory", "Storage", "Display", "Network"],
                    "correct_answer": 0,
                },
                {
                    "question_type": "multiple_choice",
                    "question_text": "What does RAM mean?",
                    "options": ["Memory", "Storage", "Display", "Network"],
                    "correct_answer": 0,
                },
            ],
            [{"question_type": "multiple_choice", "count": 1, "option_count": 5}],
            default_source_type="note",
            default_source_id="note-1",
        )


@pytest.mark.parametrize("answer", [[], [1, 1], [0, 4]])
def test_planned_multi_select_rejects_invalid_indices(answer):
    with pytest.raises(ValueError):
        quiz_generator._normalize_planned_question(
            {
                "question_type": "multi_select",
                "question_text": "Which are hardware?",
                "options": ["CPU", "RAM", "Python", "HTML"],
                "correct_answer": answer,
            },
            {"question_type": "multi_select", "option_count": 4},
        )


def test_planned_multi_select_sorts_indices():
    question = quiz_generator._normalize_planned_question(
        {
            "question_type": "multi_select",
            "question_text": "Which are hardware?",
            "options": ["CPU", "RAM", "Python", "HTML"],
            "correct_answer": [1, 0],
        },
        {"question_type": "multi_select", "option_count": 4},
    )

    assert question["correct_answer"] == [0, 1]


def test_planned_multi_select_accepts_answer_letters():
    question = quiz_generator._normalize_planned_question(
        {
            "question_type": "multi_select",
            "question_text": "Which are hardware?",
            "options": ["CPU", "RAM", "Python", "HTML"],
            "correct_answer": ["B", "A"],
        },
        {"question_type": "multi_select", "option_count": 4},
    )

    assert question["correct_answer"] == [0, 1]


def test_planned_matching_accepts_left_to_right_mapping():
    question = quiz_generator._normalize_planned_question(
        {
            "question_type": "matching",
            "question_text": "Match each component.",
            "options": ["CPU", "RAM"],
            "correct_answer": {"CPU": "Processor", "RAM": "Memory"},
        },
        {"question_type": "matching", "pair_count": 2},
    )

    assert question["options"] == ["CPU", "RAM"]
    assert question["correct_answer"] == {"CPU": "Processor", "RAM": "Memory"}


def test_planned_matching_accepts_case_insensitive_left_side_keys():
    question = quiz_generator._normalize_planned_question(
        {
            "question_type": "matching",
            "question_text": "Match each component.",
            "options": ["CPU", "RAM"],
            "correct_answer": {"cpu": "Processor", "ram": "Memory"},
        },
        {"question_type": "matching", "pair_count": 2},
    )

    assert question["correct_answer"] == {"CPU": "Processor", "RAM": "Memory"}


@pytest.mark.parametrize("answer", ["True", "yes", True, " false "])
def test_planned_true_false_requires_exact_string(answer):
    with pytest.raises(ValueError):
        quiz_generator._normalize_planned_question(
            {
                "question_type": "true_false",
                "question_text": "True or false: CPUs execute instructions.",
                "correct_answer": answer,
            },
            {"question_type": "true_false"},
        )


@pytest.mark.parametrize(
    ("question_text", "answer"),
    [
        ("CPUs execute instructions.", "CPU"),
        ("The ___ executes instructions.", ""),
    ],
)
def test_planned_fill_blank_requires_blank_marker_and_answer(question_text, answer):
    with pytest.raises(ValueError):
        quiz_generator._normalize_planned_question(
            {
                "question_type": "fill_blank",
                "question_text": question_text,
                "correct_answer": answer,
            },
            {"question_type": "fill_blank"},
        )
