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
