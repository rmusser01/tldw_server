"""Persistence tests for additive EMQ question group metadata."""

from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.unit


@pytest.fixture
def quiz_db(tmp_path: Path):
    db = CharactersRAGDB(tmp_path / "emq-groups.db", client_id="emq-groups-test")
    try:
        yield db
    finally:
        db.close_all_connections()


def test_emq_group_metadata_round_trips_through_get_and_list(quiz_db: CharactersRAGDB) -> None:
    quiz_id = quiz_db.create_quiz(name="EMQ group")
    options = ["Aortic stenosis", "Mitral regurgitation", "Pericarditis"]
    group_id = "cardiac-auscultation"
    group_prompt = "For each presentation, select the most likely diagnosis."

    question_ids = [
        quiz_db.create_question(
            quiz_id=quiz_id,
            question_type="multiple_choice",
            question_text=question_text,
            options=options,
            correct_answer=correct_answer,
            group_id=group_id,
            group_prompt=group_prompt,
            order_index=order_index,
        )
        for order_index, (question_text, correct_answer) in enumerate(
            (("Exertional syncope with an ejection murmur", 0), ("Pleuritic pain relieved by leaning forward", 2))
        )
    ]

    fetched = [quiz_db.get_question(question_id) for question_id in question_ids]
    assert all(question is not None for question in fetched)
    assert [question["group_id"] for question in fetched if question is not None] == [group_id, group_id]
    assert [question["group_prompt"] for question in fetched if question is not None] == [
        group_prompt,
        group_prompt,
    ]

    listed = quiz_db.list_questions(quiz_id, include_answers=True, limit=10, offset=0)["items"]
    assert [question["group_id"] for question in listed] == [group_id, group_id]
    assert [question["group_prompt"] for question in listed] == [group_prompt, group_prompt]
    assert [question["options"] for question in listed] == [options, options]


def test_update_question_persists_emq_group_metadata(quiz_db: CharactersRAGDB) -> None:
    quiz_id = quiz_db.create_quiz(name="Updated EMQ group")
    question_id = quiz_db.create_question(
        quiz_id=quiz_id,
        question_type="multiple_choice",
        question_text="Initial stem",
        options=["A", "B"],
        correct_answer=0,
    )
    question = quiz_db.get_question(question_id)
    assert question is not None

    updated = quiz_db.update_question(
        question_id,
        {
            "group_id": "updated-group",
            "group_prompt": "Choose the best answer for each stem.",
            "expected_version": question["version"],
        },
    )

    assert updated is True
    persisted = quiz_db.get_question(question_id)
    assert persisted is not None
    assert persisted["group_id"] == "updated-group"
    assert persisted["group_prompt"] == "Choose the best answer for each stem."


def test_create_question_preserves_legacy_positional_argument_order(quiz_db: CharactersRAGDB) -> None:
    quiz_id = quiz_db.create_quiz(name="Legacy positional arguments")
    citations = [{"source_type": "note", "source_id": "note-1", "label": "Legacy source"}]

    question_id = quiz_db.create_question(
        quiz_id,
        "multiple_choice",
        "Legacy positional stem",
        1,
        ["A", "B"],
        "Legacy explanation",
        "Legacy hint",
        2,
        citations,
        3,
        4,
        ["legacy-tag"],
        "legacy-client",
    )

    question = quiz_db.get_question(question_id)
    assert question is not None
    assert question["question_type"] == "multiple_choice"
    assert question["question_text"] == "Legacy positional stem"
    assert question["correct_answer"] == 1
    assert question["options"] == ["A", "B"]
    assert question["explanation"] == "Legacy explanation"
    assert question["hint"] == "Legacy hint"
    assert question["hint_penalty_points"] == 2
    assert question["source_citations"] == citations
    assert question["points"] == 3
    assert question["order_index"] == 4
    assert question["tags"] == ["legacy-tag"]
    assert question["client_id"] == "legacy-client"
    assert question["group_id"] is None
    assert question["group_prompt"] is None
