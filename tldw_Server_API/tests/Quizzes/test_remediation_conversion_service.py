import json
import sqlite3
import uuid
from contextlib import contextmanager

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)


@pytest.fixture(scope="function")
def quizzes_db(tmp_path):
    db_path = tmp_path / "remediation-conversion-service.db"
    db = CharactersRAGDB(str(db_path), client_id=f"test-{uuid.uuid4().hex[:6]}")
    yield db
    db.close_connection()


def _create_attempt_with_one_miss_and_one_correct(quizzes_db: CharactersRAGDB) -> tuple[int, int, int, int]:
    quiz_id = quizzes_db.create_quiz(name="Remediation Service Quiz")
    missed_question_id = quizzes_db.create_question(
        quiz_id=quiz_id,
        question_type="multiple_choice",
        question_text="Which structure filters blood?",
        correct_answer=1,
        options=["Loop of Henle", "Glomerulus", "Collecting duct"],
        explanation="The glomerulus performs the initial filtration step.",
        points=2,
        order_index=0,
    )
    correct_question_id = quizzes_db.create_question(
        quiz_id=quiz_id,
        question_type="multiple_choice",
        question_text="Which structure concentrates urine?",
        correct_answer=0,
        options=["Loop of Henle", "Glomerulus", "Collecting duct"],
        explanation="The loop of Henle creates the osmotic gradient.",
        points=1,
        order_index=1,
    )
    attempt = quizzes_db.start_attempt(quiz_id)
    quizzes_db.submit_attempt(
        int(attempt["id"]),
        answers=[
            {"question_id": missed_question_id, "user_answer": 2, "time_spent_ms": 1200},
            {"question_id": correct_question_id, "user_answer": 0, "time_spent_ms": 900},
        ],
    )
    return int(attempt["id"]), quiz_id, missed_question_id, correct_question_id


def _create_incomplete_attempt(quizzes_db: CharactersRAGDB) -> tuple[int, int]:
    quiz_id = quizzes_db.create_quiz(name="Incomplete Quiz")
    question_id = quizzes_db.create_question(
        quiz_id=quiz_id,
        question_type="multiple_choice",
        question_text="Which structure filters blood?",
        correct_answer=1,
        options=["Loop of Henle", "Glomerulus", "Collecting duct"],
        order_index=0,
    )
    attempt = quizzes_db.start_attempt(quiz_id)
    return int(attempt["id"]), question_id


def _create_duplicate_missed_attempt(quizzes_db: CharactersRAGDB) -> tuple[int, int, list[int]]:
    quiz_id = quizzes_db.create_quiz(name="Duplicate Misses Quiz")
    question_ids = [
        quizzes_db.create_question(
            quiz_id=quiz_id,
            question_type="multiple_choice",
            question_text="Which structure filters blood?",
            correct_answer=1,
            options=["Loop of Henle", "Glomerulus", "Collecting duct"],
            explanation="The glomerulus performs the initial filtration step.",
            order_index=0,
        ),
        quizzes_db.create_question(
            quiz_id=quiz_id,
            question_type="multiple_choice",
            question_text="Which structure filters blood?",
            correct_answer=1,
            options=["Loop of Henle", "Glomerulus", "Collecting duct"],
            explanation="The glomerulus performs the initial filtration step.",
            order_index=1,
        ),
    ]
    attempt = quizzes_db.start_attempt(quiz_id)
    quizzes_db.submit_attempt(
        int(attempt["id"]),
        answers=[
            {"question_id": question_ids[0], "user_answer": 0, "time_spent_ms": 800},
            {"question_id": question_ids[1], "user_answer": 2, "time_spent_ms": 850},
        ],
    )
    return int(attempt["id"]), quiz_id, question_ids


def test_convert_quiz_remediation_questions_rejects_incomplete_attempt(quizzes_db: CharactersRAGDB):
    attempt_id, question_id = _create_incomplete_attempt(quizzes_db)

    with pytest.raises(InputError):
        quizzes_db.convert_quiz_remediation_questions(
            attempt_id=attempt_id,
            question_ids=[question_id],
            create_deck_name="Remediation Deck",
            replace_active=False,
        )


def test_convert_quiz_remediation_questions_rejects_question_ids_not_in_attempt(quizzes_db: CharactersRAGDB):
    attempt_id, _quiz_id, missed_question_id, _correct_question_id = _create_attempt_with_one_miss_and_one_correct(quizzes_db)
    deck_id = quizzes_db.add_deck("Renal Deck")

    with pytest.raises(InputError):
        quizzes_db.convert_quiz_remediation_questions(
            attempt_id=attempt_id,
            question_ids=[missed_question_id, 999999],
            target_deck_id=deck_id,
            replace_active=False,
        )


def test_convert_quiz_remediation_questions_rejects_correct_answers(quizzes_db: CharactersRAGDB):
    attempt_id, _quiz_id, _missed_question_id, correct_question_id = _create_attempt_with_one_miss_and_one_correct(quizzes_db)
    deck_id = quizzes_db.add_deck("Renal Deck")

    with pytest.raises(InputError):
        quizzes_db.convert_quiz_remediation_questions(
            attempt_id=attempt_id,
            question_ids=[correct_question_id],
            target_deck_id=deck_id,
            replace_active=False,
        )


def test_convert_quiz_remediation_questions_creates_new_deck_when_requested(quizzes_db: CharactersRAGDB):
    attempt_id, quiz_id, missed_question_id, _correct_question_id = _create_attempt_with_one_miss_and_one_correct(quizzes_db)

    payload = quizzes_db.convert_quiz_remediation_questions(
        attempt_id=attempt_id,
        question_ids=[missed_question_id],
        create_deck_name="Quiz Remediation Deck",
        replace_active=False,
    )

    assert payload["attempt_id"] == attempt_id
    assert payload["quiz_id"] == quiz_id
    assert payload["target_deck"]["name"] == "Quiz Remediation Deck"
    assert payload["results"][0]["status"] == "created"
    assert payload["results"][0]["conversion"]["status"] == "active"
    assert payload["created_flashcard_uuids"]
    assert quizzes_db.count_flashcards(deck_id=payload["target_deck"]["id"]) == 1


def test_convert_quiz_remediation_questions_creates_fsrs_deck_when_requested(quizzes_db: CharactersRAGDB):
    attempt_id, _quiz_id, missed_question_id, _correct_question_id = _create_attempt_with_one_miss_and_one_correct(quizzes_db)

    payload = quizzes_db.convert_quiz_remediation_questions(
        attempt_id=attempt_id,
        question_ids=[missed_question_id],
        create_deck_name="FSRS Remediation Deck",
        create_deck_scheduler_type="fsrs",
        create_deck_scheduler_settings={
            "sm2_plus": {
                "new_steps_minutes": [1, 5],
                "relearn_steps_minutes": [10],
                "graduating_interval_days": 1,
                "easy_interval_days": 3,
                "easy_bonus": 1.15,
                "interval_modifier": 0.9,
                "max_interval_days": 3650,
                "leech_threshold": 10,
                "enable_fuzz": False,
            },
            "fsrs": {
                "target_retention": 0.95,
                "maximum_interval_days": 1825,
                "enable_fuzz": True,
            },
        },
        replace_active=False,
    )

    created_deck = quizzes_db.get_deck(payload["target_deck"]["id"])
    assert created_deck is not None
    assert created_deck["scheduler_type"] == "fsrs"
    scheduler_settings = json.loads(created_deck["scheduler_settings_json"])
    assert scheduler_settings["fsrs"]["target_retention"] == pytest.approx(0.95)
    assert scheduler_settings["fsrs"]["maximum_interval_days"] == 1825


def test_convert_quiz_remediation_questions_preserves_dedupe_across_duplicate_misses(
    quizzes_db: CharactersRAGDB,
):
    attempt_id, _quiz_id, question_ids = _create_duplicate_missed_attempt(quizzes_db)
    deck_id = quizzes_db.add_deck("Renal Deck")

    payload = quizzes_db.convert_quiz_remediation_questions(
        attempt_id=attempt_id,
        question_ids=question_ids,
        target_deck_id=deck_id,
        replace_active=False,
    )

    assert len(payload["results"]) == 2
    assert payload["results"][0]["status"] == "created"
    assert payload["results"][1]["status"] == "created"
    assert payload["results"][0]["flashcard_uuids"] == payload["results"][1]["flashcard_uuids"]
    assert len(payload["created_flashcard_uuids"]) == 1
    assert quizzes_db.count_flashcards(deck_id=deck_id) == 1


def test_convert_quiz_remediation_questions_replace_active_supersedes_old_row(
    quizzes_db: CharactersRAGDB,
):
    attempt_id, _quiz_id, missed_question_id, _correct_question_id = _create_attempt_with_one_miss_and_one_correct(quizzes_db)
    deck_id = quizzes_db.add_deck("Renal Deck")

    initial = quizzes_db.convert_quiz_remediation_questions(
        attempt_id=attempt_id,
        question_ids=[missed_question_id],
        target_deck_id=deck_id,
        replace_active=False,
    )
    follow_up = quizzes_db.convert_quiz_remediation_questions(
        attempt_id=attempt_id,
        question_ids=[missed_question_id],
        target_deck_id=deck_id,
        replace_active=True,
    )

    assert initial["results"][0]["status"] == "created"
    assert follow_up["results"][0]["status"] == "superseded_and_created"
    assert follow_up["results"][0]["conversion"]["superseded_count"] == 1
    conversions = quizzes_db.list_attempt_remediation_conversions(attempt_id)
    assert conversions["count"] == 1
    assert conversions["superseded_count"] == 1


def test_convert_quiz_remediation_questions_translates_unique_violation_to_conflict(
    quizzes_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    class _FakeCursor:
        def __init__(self, rows=None):
            self._rows = rows or []
            self.lastrowid = None

        def fetchall(self):
            return list(self._rows)

        def fetchone(self):
            return self._rows[0] if self._rows else None

    class _FakeConnection:
        def execute(self, sql, params=()):
            if (
                "FROM quiz_remediation_conversions qrc" in sql
                and "WHERE qrc.attempt_id = ?" in sql
                and "AND qrc.status = ?" in sql
            ):
                return _FakeCursor([])
            if sql.startswith("INSERT INTO quiz_remediation_conversions"):
                raise sqlite3.IntegrityError(
                    "UNIQUE constraint failed: quiz_remediation_conversions.attempt_id, quiz_remediation_conversions.question_id"
                )
            raise AssertionError(f"Unexpected SQL in test double: {sql} :: {params}")

    class _FakeTransaction:
        def __enter__(self):
            return _FakeConnection()

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        quizzes_db,
        "_build_quiz_remediation_entries",
        lambda *, attempt_id, question_ids: (
            {"id": attempt_id, "quiz_id": 9},
            {"id": 9, "name": "Race Quiz"},
            [
                {
                    "question_id": question_ids[0],
                    "dedupe_key": "dedupe-key",
                    "source_ref_id": f"quiz-attempt:{attempt_id}:question:{question_ids[0]}",
                    "flashcard_payload": {
                        "front": "Question",
                        "back": "Answer",
                        "notes": "Notes",
                        "tags_json": "[]",
                        "source_ref_type": "manual",
                        "source_ref_id": f"quiz-attempt:{attempt_id}:question:{question_ids[0]}",
                    },
                }
            ],
        ),
    )
    monkeypatch.setattr(quizzes_db, "get_deck", lambda deck_id: {"id": deck_id, "name": "Target Deck", "deleted": False})
    monkeypatch.setattr(quizzes_db, "add_flashcards_bulk", lambda payloads: ["fc-race-1"])
    monkeypatch.setattr(quizzes_db, "transaction", lambda: _FakeTransaction())

    with pytest.raises(ConflictError):
        quizzes_db.convert_quiz_remediation_questions(
            attempt_id=101,
            question_ids=[12],
            target_deck_id=7,
            replace_active=False,
        )


def test_replace_remediation_missing_generated_id_restores_cards_and_prior_active(
    quizzes_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt_id, _quiz_id, question_id, _correct_id = _create_attempt_with_one_miss_and_one_correct(quizzes_db)
    deck_id = quizzes_db.add_deck("Rollback target")
    quizzes_db.convert_quiz_remediation_questions(
        attempt_id=attempt_id,
        question_ids=[question_id],
        target_deck_id=deck_id,
    )
    before_conversions = [
        dict(row)
        for row in quizzes_db.execute_query("SELECT * FROM quiz_remediation_conversions ORDER BY id").fetchall()
    ]
    before_cards = [dict(row) for row in quizzes_db.execute_query("SELECT * FROM flashcards ORDER BY id").fetchall()]
    original_transaction = quizzes_db.transaction
    inserted_ids = []

    class MissingIdCursor:
        lastrowid = None

        def __init__(self, cursor):
            self._cursor = cursor

        def __getattr__(self, name):
            return getattr(self._cursor, name)

    class MissingIdConnection:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, sql, params=()):
            cursor = self._conn.execute(sql, params)
            if " ".join(sql.split()).startswith("INSERT INTO quiz_remediation_conversions("):
                inserted_ids.append(cursor.lastrowid)
                return MissingIdCursor(cursor)
            return cursor

        def __getattr__(self, name):
            return getattr(self._conn, name)

    @contextmanager
    def missing_id_transaction():
        with original_transaction() as conn:
            yield MissingIdConnection(conn)

    monkeypatch.setattr(quizzes_db, "transaction", missing_id_transaction)

    with pytest.raises(CharactersRAGDBError) as error:
        quizzes_db.convert_quiz_remediation_questions(
            attempt_id=attempt_id,
            question_ids=[question_id],
            target_deck_id=deck_id,
            replace_active=True,
        )
    assert str(error.value) == "Failed to determine remediation conversion ID after insert"
    assert len(inserted_ids) == 1 and isinstance(inserted_ids[0], int)
    after_conversions = [
        dict(row)
        for row in quizzes_db.execute_query("SELECT * FROM quiz_remediation_conversions ORDER BY id").fetchall()
    ]
    assert after_conversions == before_conversions
    after_cards = [dict(row) for row in quizzes_db.execute_query("SELECT * FROM flashcards ORDER BY id").fetchall()]
    assert after_cards == before_cards
