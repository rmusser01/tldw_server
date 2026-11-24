import uuid

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError


@pytest.fixture(scope="function")
def quizzes_db(tmp_path):
    db_path = tmp_path / "remediation-conversions.db"
    db = CharactersRAGDB(str(db_path), client_id=f"test-{uuid.uuid4().hex[:6]}")
    yield db
    db.close_connection()


def _create_completed_attempt_with_one_miss(quizzes_db: CharactersRAGDB) -> tuple[int, int, int]:
    quiz_id = quizzes_db.create_quiz(name="Remediation Conversion Quiz")
    missed_question_id = quizzes_db.create_question(
        quiz_id=quiz_id,
        question_type="multiple_choice",
        question_text="Which structure filters blood?",
        correct_answer=1,
        options=["Loop of Henle", "Glomerulus", "Collecting duct"],
        explanation="The glomerulus performs the initial blood filtration step.",
        points=2,
        order_index=0,
    )
    correct_question_id = quizzes_db.create_question(
        quiz_id=quiz_id,
        question_type="multiple_choice",
        question_text="Which segment concentrates urine?",
        correct_answer=0,
        options=["Loop of Henle", "Glomerulus", "Proximal tubule"],
        explanation="The loop of Henle establishes the osmotic gradient.",
        points=1,
        order_index=1,
    )
    attempt = quizzes_db.start_attempt(quiz_id)
    quizzes_db.submit_attempt(
        int(attempt["id"]),
        answers=[
            {"question_id": missed_question_id, "user_answer": 2, "time_spent_ms": 1200},
            {"question_id": correct_question_id, "user_answer": 0, "time_spent_ms": 800},
        ],
    )
    return int(attempt["id"]), quiz_id, missed_question_id


def test_create_active_remediation_conversion(quizzes_db: CharactersRAGDB):
    attempt_id, quiz_id, question_id = _create_completed_attempt_with_one_miss(quizzes_db)
    deck_id = quizzes_db.flashcard_store.add_deck("Renal Deck")

    row = quizzes_db.create_quiz_remediation_conversion(
        attempt_id=attempt_id,
        quiz_id=quiz_id,
        question_id=question_id,
        target_deck_id=deck_id,
        target_deck_name_snapshot="Renal Deck",
        flashcard_uuids=["card-a", "card-b"],
        source_ref_id=f"quiz-attempt:{attempt_id}:question:{question_id}",
    )

    assert row["attempt_id"] == attempt_id
    assert row["quiz_id"] == quiz_id
    assert row["question_id"] == question_id
    assert row["status"] == "active"
    assert row["target_deck_id"] == deck_id
    assert row["target_deck_name_snapshot"] == "Renal Deck"
    assert row["flashcard_count"] == 2
    assert row["flashcard_uuids_json"] == ["card-a", "card-b"]


def test_create_active_remediation_conversion_rejects_second_active_row(quizzes_db: CharactersRAGDB):
    attempt_id, quiz_id, question_id = _create_completed_attempt_with_one_miss(quizzes_db)
    deck_id = quizzes_db.flashcard_store.add_deck("Renal Deck")

    quizzes_db.create_quiz_remediation_conversion(
        attempt_id=attempt_id,
        quiz_id=quiz_id,
        question_id=question_id,
        target_deck_id=deck_id,
        target_deck_name_snapshot="Renal Deck",
        flashcard_uuids=["card-a"],
        source_ref_id=f"quiz-attempt:{attempt_id}:question:{question_id}",
    )

    with pytest.raises(ConflictError):
        quizzes_db.create_quiz_remediation_conversion(
            attempt_id=attempt_id,
            quiz_id=quiz_id,
            question_id=question_id,
            target_deck_id=deck_id,
            target_deck_name_snapshot="Renal Deck",
            flashcard_uuids=["card-b"],
            source_ref_id=f"quiz-attempt:{attempt_id}:question:{question_id}",
        )


def test_supersede_quiz_remediation_conversion_marks_previous_row(quizzes_db: CharactersRAGDB):
    attempt_id, quiz_id, question_id = _create_completed_attempt_with_one_miss(quizzes_db)
    deck_id = quizzes_db.flashcard_store.add_deck("Renal Deck")
    original = quizzes_db.create_quiz_remediation_conversion(
        attempt_id=attempt_id,
        quiz_id=quiz_id,
        question_id=question_id,
        target_deck_id=deck_id,
        target_deck_name_snapshot="Renal Deck",
        flashcard_uuids=["card-a"],
        source_ref_id=f"quiz-attempt:{attempt_id}:question:{question_id}",
    )

    superseded = quizzes_db.supersede_quiz_remediation_conversion(
        attempt_id=attempt_id,
        question_id=question_id,
        superseded_by_id=None,
    )

    assert superseded is not None
    assert superseded["id"] == original["id"]
    assert superseded["status"] == "superseded"
    assert superseded["superseded_by_id"] is None


def test_list_attempt_remediation_conversions_returns_active_rows_and_superseded_count(
    quizzes_db: CharactersRAGDB,
):
    attempt_id, quiz_id, question_id = _create_completed_attempt_with_one_miss(quizzes_db)
    deck_id = quizzes_db.flashcard_store.add_deck("Renal Deck")
    original = quizzes_db.create_quiz_remediation_conversion(
        attempt_id=attempt_id,
        quiz_id=quiz_id,
        question_id=question_id,
        target_deck_id=deck_id,
        target_deck_name_snapshot="Renal Deck",
        flashcard_uuids=["card-a"],
        source_ref_id=f"quiz-attempt:{attempt_id}:question:{question_id}",
    )
    quizzes_db.supersede_quiz_remediation_conversion(
        attempt_id=attempt_id,
        question_id=question_id,
        superseded_by_id=None,
    )
    latest = quizzes_db.create_quiz_remediation_conversion(
        attempt_id=attempt_id,
        quiz_id=quiz_id,
        question_id=question_id,
        target_deck_id=deck_id,
        target_deck_name_snapshot="Renal Deck v2",
        flashcard_uuids=["card-b"],
        source_ref_id=f"quiz-attempt:{attempt_id}:question:{question_id}",
    )

    payload = quizzes_db.list_attempt_remediation_conversions(attempt_id)

    assert payload["attempt_id"] == attempt_id
    assert payload["items"]
    assert payload["count"] == 1
    assert payload["superseded_count"] == 1
    assert payload["items"][0]["id"] == latest["id"]
    assert payload["items"][0]["status"] == "active"
    assert payload["items"][0]["superseded_count"] == 1
    assert payload["items"][0]["flashcard_uuids_json"] == ["card-b"]
