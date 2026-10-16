from contextlib import contextmanager

import pytest

from tldw_Server_API.app.api.v1.schemas.flashcards import (
    StudyAssistantHistoryResponse,
    StudyAssistantMessage,
    StudyAssistantThreadSummary,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
)


@pytest.fixture
def chacha_db(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "study-assistant.db"), client_id="study-assistant-tests")
    try:
        yield db
    finally:
        db.close_connection()


def _create_flashcard(chacha_db: CharactersRAGDB) -> str:
    deck_id = chacha_db.add_deck("Study Assistant Deck")
    return chacha_db.add_flashcard(
        {
            "deck_id": deck_id,
            "front": "What is the nephron?",
            "back": "The functional unit of the kidney.",
            "notes": "Renal physiology",
            "extra": "",
        }
    )


def _create_attempt_question_context(chacha_db: CharactersRAGDB) -> tuple[int, int]:
    quiz_id = chacha_db.create_quiz(name="Renal Quiz")
    question_id = chacha_db.create_question(
        quiz_id=quiz_id,
        question_type="multiple_choice",
        question_text="Which structure filters blood?",
        correct_answer=1,
        options=["Loop of Henle", "Glomerulus", "Collecting duct"],
        explanation="The glomerulus performs initial blood filtration.",
    )
    attempt = chacha_db.start_attempt(quiz_id)
    return int(attempt["id"]), question_id


def test_study_assistant_schema_columns_exist_on_fresh_db(chacha_db: CharactersRAGDB):
    thread_columns = {
        row["name"] for row in chacha_db.execute_query("PRAGMA table_info('study_assistant_threads')").fetchall()
    }
    message_columns = {
        row["name"] for row in chacha_db.execute_query("PRAGMA table_info('study_assistant_messages')").fetchall()
    }

    assert "context_type" in thread_columns
    assert "flashcard_uuid" in thread_columns
    assert "quiz_attempt_id" in thread_columns
    assert "question_id" in thread_columns
    assert "last_message_at" in thread_columns
    assert "message_count" in thread_columns
    assert "role" in message_columns
    assert "action_type" in message_columns
    assert "input_modality" in message_columns
    assert "structured_payload_json" in message_columns
    assert "context_snapshot_json" in message_columns
    assert "provider" in message_columns
    assert "model" in message_columns


def test_get_or_create_study_assistant_thread_reuses_flashcard_context(chacha_db: CharactersRAGDB):
    flashcard_uuid = _create_flashcard(chacha_db)

    first = chacha_db.get_or_create_study_assistant_thread(
        context_type="flashcard",
        flashcard_uuid=flashcard_uuid,
    )
    second = chacha_db.get_or_create_study_assistant_thread(
        context_type="flashcard",
        flashcard_uuid=flashcard_uuid,
    )

    assert first["id"] == second["id"]
    assert first["context_type"] == "flashcard"
    assert first["flashcard_uuid"] == flashcard_uuid
    assert first["message_count"] == 0


def test_get_or_create_study_assistant_thread_reuses_quiz_attempt_question_context(chacha_db: CharactersRAGDB):
    attempt_id, question_id = _create_attempt_question_context(chacha_db)

    first = chacha_db.get_or_create_study_assistant_thread(
        context_type="quiz_attempt_question",
        quiz_attempt_id=attempt_id,
        question_id=question_id,
    )
    second = chacha_db.get_or_create_study_assistant_thread(
        context_type="quiz_attempt_question",
        quiz_attempt_id=attempt_id,
        question_id=question_id,
    )

    assert first["id"] == second["id"]
    assert first["context_type"] == "quiz_attempt_question"
    assert first["quiz_attempt_id"] == attempt_id
    assert first["question_id"] == question_id


def test_append_study_assistant_message_updates_thread_counts_and_persists_json(chacha_db: CharactersRAGDB):
    flashcard_uuid = _create_flashcard(chacha_db)
    thread = chacha_db.get_or_create_study_assistant_thread(
        context_type="flashcard",
        flashcard_uuid=flashcard_uuid,
    )

    first_message = chacha_db.append_study_assistant_message(
        thread_id=thread["id"],
        role="user",
        action_type="fact_check",
        input_modality="voice_transcript",
        content="I think the nephron filters blood in the collecting duct.",
        structured_payload={
            "transcript_confirmed": True,
        },
        context_snapshot={
            "flashcard_uuid": flashcard_uuid,
            "version": 1,
        },
        provider="openai",
        model="gpt-test",
    )
    second_message = chacha_db.append_study_assistant_message(
        thread_id=thread["id"],
        role="assistant",
        action_type="fact_check",
        input_modality="text",
        content="Partially correct. Filtration happens in the glomerulus.",
        structured_payload={
            "verdict": "partially_correct",
            "missing_points": ["Filtration occurs in the glomerulus."],
        },
        context_snapshot={
            "flashcard_uuid": flashcard_uuid,
            "version": 1,
        },
        provider="openai",
        model="gpt-test",
    )

    messages = chacha_db.list_study_assistant_messages(thread["id"])
    reloaded_thread = chacha_db.get_study_assistant_thread(thread["id"])

    assert first_message["role"] == "user"
    assert second_message["role"] == "assistant"
    assert [message["content"] for message in messages] == [
        "I think the nephron filters blood in the collecting duct.",
        "Partially correct. Filtration happens in the glomerulus.",
    ]
    assert messages[0]["structured_payload"]["transcript_confirmed"] is True
    assert messages[1]["structured_payload"]["verdict"] == "partially_correct"
    assert messages[1]["context_snapshot"]["flashcard_uuid"] == flashcard_uuid
    assert reloaded_thread["message_count"] == 2
    assert reloaded_thread["last_message_at"] == second_message["created_at"]


def test_append_study_assistant_message_rejects_stale_expected_thread_version(chacha_db: CharactersRAGDB):
    flashcard_uuid = _create_flashcard(chacha_db)
    thread = chacha_db.get_or_create_study_assistant_thread(
        context_type="flashcard",
        flashcard_uuid=flashcard_uuid,
    )

    first_message = chacha_db.append_study_assistant_message(
        thread_id=thread["id"],
        role="user",
        action_type="explain",
        input_modality="text",
        content="First message",
        expected_thread_version=int(thread["version"]),
    )

    assert first_message["content"] == "First message"

    with pytest.raises(ConflictError, match="Version mismatch updating study assistant thread"):
        chacha_db.append_study_assistant_message(
            thread_id=thread["id"],
            role="assistant",
            action_type="explain",
            input_modality="text",
            content="Stale concurrent reply",
            expected_thread_version=int(thread["version"]),
        )

    reloaded_thread = chacha_db.get_study_assistant_thread(thread["id"])
    messages = chacha_db.list_study_assistant_messages(thread["id"])

    assert reloaded_thread is not None
    assert reloaded_thread["message_count"] == 1
    assert [message["content"] for message in messages] == ["First message"]


def test_study_assistant_schema_models_accept_thread_and_messages():
    thread = StudyAssistantThreadSummary(
        id=7,
        context_type="flashcard",
        flashcard_uuid="card-123",
        quiz_attempt_id=None,
        question_id=None,
        last_message_at="2026-03-13T12:00:00Z",
        message_count=2,
        deleted=False,
        client_id="tests",
        version=3,
        created_at="2026-03-13T11:00:00Z",
        last_modified="2026-03-13T12:00:00Z",
    )
    message = StudyAssistantMessage(
        id=11,
        thread_id=7,
        role="assistant",
        action_type="explain",
        input_modality="text",
        content="The nephron filters blood and adjusts solute balance.",
        structured_payload={"summary": "renal physiology"},
        context_snapshot={"flashcard_uuid": "card-123"},
        provider="openai",
        model="gpt-test",
        created_at="2026-03-13T12:00:00Z",
        client_id="tests",
    )

    response = StudyAssistantHistoryResponse(thread=thread, messages=[message])

    assert response.thread.id == 7
    assert response.messages[0].structured_payload["summary"] == "renal physiology"


@pytest.mark.parametrize("target", ["thread", "message"])
def test_study_assistant_missing_generated_id_rolls_back_native_insert(
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    flashcard_uuid = _create_flashcard(chacha_db)
    if target == "message":
        thread = chacha_db.get_or_create_study_assistant_thread(context_type="flashcard", flashcard_uuid=flashcard_uuid)
        chacha_db.append_study_assistant_message(
            thread_id=thread["id"],
            role="user",
            action_type="explain",
            input_modality="text",
            content="Keep message",
        )
    before_threads = [
        dict(row) for row in chacha_db.execute_query("SELECT * FROM study_assistant_threads ORDER BY id").fetchall()
    ]
    before_messages = [
        dict(row) for row in chacha_db.execute_query("SELECT * FROM study_assistant_messages ORDER BY id").fetchall()
    ]
    target_table = "study_assistant_threads" if target == "thread" else "study_assistant_messages"
    original_transaction = chacha_db.transaction
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
            if " ".join(sql.split()).startswith(f"INSERT INTO {target_table}("):
                inserted_ids.append(cursor.lastrowid)
                return MissingIdCursor(cursor)
            return cursor

        def __getattr__(self, name):
            return getattr(self._conn, name)

    @contextmanager
    def missing_id_transaction():
        with original_transaction() as conn:
            yield MissingIdConnection(conn)

    monkeypatch.setattr(chacha_db, "transaction", missing_id_transaction)

    with pytest.raises(CharactersRAGDBError) as error:
        if target == "thread":
            chacha_db.get_or_create_study_assistant_thread(context_type="flashcard", flashcard_uuid=flashcard_uuid)
        else:
            chacha_db.append_study_assistant_message(
                thread_id=thread["id"],
                role="assistant",
                action_type="explain",
                input_modality="text",
                content="Absent ID",
            )
    assert str(error.value) == f"Failed to determine study assistant {target} ID after insert"
    assert len(inserted_ids) == 1 and isinstance(inserted_ids[0], int)
    after_threads = [
        dict(row) for row in chacha_db.execute_query("SELECT * FROM study_assistant_threads ORDER BY id").fetchall()
    ]
    after_messages = [
        dict(row) for row in chacha_db.execute_query("SELECT * FROM study_assistant_messages ORDER BY id").fetchall()
    ]
    assert after_threads == before_threads
    assert after_messages == before_messages
