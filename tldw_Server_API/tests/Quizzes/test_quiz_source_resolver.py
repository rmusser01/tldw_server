import uuid

import pytest

from tldw_Server_API.app.api.v1.schemas.quizzes import QuizGenerateSource, QuizSourceType
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.services import quiz_source_resolver as resolver_mod
from tldw_Server_API.app.services.quiz_source_resolver import resolve_quiz_sources


@pytest.fixture(scope="function")
def quizzes_db(tmp_path):
    db_path = tmp_path / "quiz_sources.db"
    db = CharactersRAGDB(str(db_path), client_id=f"test-{uuid.uuid4().hex[:6]}")
    yield db
    db.close_connection()


@pytest.fixture(scope="function")
def media_db(tmp_path):
    db_path = tmp_path / "media_sources.db"
    db = MediaDatabase(str(db_path), client_id=f"test-{uuid.uuid4().hex[:6]}")
    yield db
    db.close_connection()


def _create_attempt_with_missed_question(quizzes_db: CharactersRAGDB) -> tuple[int, int]:
    quiz_id = quizzes_db.create_quiz(name="Renal Quiz")
    question_id = quizzes_db.create_question(
        quiz_id=quiz_id,
        question_type="multiple_choice",
        question_text="Which structure filters blood?",
        correct_answer=1,
        options=["Loop of Henle", "Glomerulus", "Collecting duct"],
        explanation="The glomerulus performs initial blood filtration.",
        source_citations=[{"source_type": "note", "source_id": "renal-note", "quote": "Glomeruli filter blood."}],
        points=2,
    )
    quizzes_db.create_question(
        quiz_id=quiz_id,
        question_type="multiple_choice",
        question_text="Which structure concentrates urine?",
        correct_answer=0,
        options=["Loop of Henle", "Glomerulus", "Collecting duct"],
        explanation="The Loop of Henle builds the medullary gradient.",
        points=1,
    )
    attempt = quizzes_db.start_attempt(quiz_id)
    quizzes_db.submit_attempt(
        int(attempt["id"]),
        answers=[
            {"question_id": question_id, "user_answer": 2, "time_spent_ms": 1400},
        ],
    )
    return int(attempt["id"]), question_id


def test_resolves_note_source_to_evidence_chunks(quizzes_db: CharactersRAGDB, media_db: MediaDatabase):
    note_id = quizzes_db.add_note(title="Cell Theory", content="All living things are made of cells.")

    evidence = resolve_quiz_sources(
        [{"source_type": "note", "source_id": note_id}],
        db=quizzes_db,
        media_db=media_db,
    )

    assert evidence[0]["source_type"] == "note"
    assert evidence[0]["source_id"] == note_id
    assert evidence[0]["text"]


def test_resolves_flashcard_deck_to_card_evidence_chunks(
    quizzes_db: CharactersRAGDB,
    media_db: MediaDatabase,
):
    deck_id = quizzes_db.flashcard_store.add_deck(name="Biology")
    quizzes_db.flashcard_store.add_flashcard({"deck_id": deck_id, "front": "Mitochondria", "back": "Powerhouse of the cell"})
    quizzes_db.flashcard_store.add_flashcard({"deck_id": deck_id, "front": "Ribosome", "back": "Synthesizes proteins"})

    evidence = resolve_quiz_sources(
        [{"source_type": "flashcard_deck", "source_id": str(deck_id)}],
        db=quizzes_db,
        media_db=media_db,
    )

    assert len(evidence) >= 2


def test_resolves_flashcard_card_source(quizzes_db: CharactersRAGDB, media_db: MediaDatabase):
    card_uuid = quizzes_db.flashcard_store.add_flashcard({"front": "ATP", "back": "Cell energy currency"})

    evidence = resolve_quiz_sources(
        [{"source_type": "flashcard_card", "source_id": card_uuid}],
        db=quizzes_db,
        media_db=media_db,
    )

    assert evidence[0]["source_id"] == card_uuid


def test_deduplicates_duplicate_source_entries(quizzes_db: CharactersRAGDB, media_db: MediaDatabase):
    note_id = quizzes_db.add_note(title="DNA", content="DNA stores genetic information.")

    evidence = resolve_quiz_sources(
        [
            {"source_type": "note", "source_id": note_id},
            {"source_type": "note", "source_id": note_id},
        ],
        db=quizzes_db,
        media_db=media_db,
    )

    assert len(evidence) == 1
    assert evidence[0]["source_type"] == "note"
    assert evidence[0]["source_id"] == note_id


def test_quiz_generate_source_accepts_quiz_attempt_source_types():
    attempt_source = QuizGenerateSource(source_type=QuizSourceType.QUIZ_ATTEMPT, source_id="12")
    question_source = QuizGenerateSource(source_type=QuizSourceType.QUIZ_ATTEMPT_QUESTION, source_id="12:4")

    assert attempt_source.source_type == QuizSourceType.QUIZ_ATTEMPT
    assert question_source.source_type == QuizSourceType.QUIZ_ATTEMPT_QUESTION


def test_resolve_quiz_attempt_source_includes_missed_question_evidence(
    quizzes_db: CharactersRAGDB,
    media_db: MediaDatabase,
):
    attempt_id, _question_id = _create_attempt_with_missed_question(quizzes_db)

    evidence = resolve_quiz_sources(
        [{"source_type": "quiz_attempt", "source_id": str(attempt_id)}],
        db=quizzes_db,
        media_db=media_db,
    )

    assert evidence[0]["source_type"] == "quiz_attempt"
    assert evidence[0]["source_id"] == str(attempt_id)
    assert "Question:" in evidence[0]["text"]
    assert "User answer:" in evidence[0]["text"]
    assert "Correct answer:" in evidence[0]["text"]
    assert "Explanation:" in evidence[0]["text"]


def test_resolve_quiz_attempt_question_source_includes_user_answer_and_citations(
    quizzes_db: CharactersRAGDB,
    media_db: MediaDatabase,
):
    attempt_id, question_id = _create_attempt_with_missed_question(quizzes_db)

    evidence = resolve_quiz_sources(
        [{"source_type": "quiz_attempt_question", "source_id": f"{attempt_id}:{question_id}"}],
        db=quizzes_db,
        media_db=media_db,
    )

    assert len(evidence) == 1
    assert evidence[0]["source_type"] == "quiz_attempt_question"
    assert evidence[0]["source_id"] == f"{attempt_id}:{question_id}"
    assert "Which structure filters blood?" in evidence[0]["text"]
    assert "User answer: 2" in evidence[0]["text"]
    assert "Correct answer: 1" in evidence[0]["text"]
    assert "Source citations:" in evidence[0]["text"]


def test_resolves_media_source_from_latest_transcription_fallback(
    monkeypatch,
    quizzes_db: CharactersRAGDB,
    media_db: MediaDatabase,
):
    monkeypatch.setattr(
        media_db,
        "get_media_by_id",
        lambda media_id, include_deleted=False, include_trash=False: {
            "id": media_id,
            "title": "Renal Lecture",
            "content": "",
        },
    )
    monkeypatch.setattr(
        resolver_mod,
        "get_latest_transcription",
        lambda db, media_id: "Filtered transcript fallback.",
    )

    evidence = resolve_quiz_sources(
        [{"source_type": "media", "source_id": "7"}],
        db=quizzes_db,
        media_db=media_db,
    )

    assert evidence[0]["source_type"] == "media"
    assert evidence[0]["source_id"] == "7"
    assert evidence[0]["label"] == "Renal Lecture"
    assert evidence[0]["text"] == "Filtered transcript fallback."
