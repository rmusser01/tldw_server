import json

import pytest

from tldw_Server_API.app.api.v1.schemas.flashcards import (
    StudyAssistantFactCheckPayload,
    StudyAssistantRespondRequest,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from tldw_Server_API.app.core.Flashcards.study_assistant import (
    build_flashcard_assistant_context,
    build_study_assistant_prompt_package,
    build_quiz_attempt_question_context,
    normalize_fact_check_payload,
)


@pytest.fixture
def chacha_db(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "study-assistant-service.db"), client_id="study-assistant-service-tests")
    try:
        yield db
    finally:
        db.close_connection()


def _create_flashcard(chacha_db: CharactersRAGDB, *, notes: str | None = "Renal physiology") -> str:
    deck_id = chacha_db.add_deck("Study Assistant Context Deck")
    return chacha_db.add_flashcard(
        {
            "deck_id": deck_id,
            "front": "What is the nephron?",
            "back": "The functional unit of the kidney.",
            "notes": notes,
            "extra": "Associated with urine formation.",
            "tags": ["renal", "physiology"],
        }
    )


def _create_quiz_attempt_question(chacha_db: CharactersRAGDB) -> tuple[int, int]:
    quiz_id = chacha_db.create_quiz(name="Renal Quiz")
    first_question_id = chacha_db.create_question(
        quiz_id=quiz_id,
        question_type="multiple_choice",
        question_text="Which structure filters blood?",
        correct_answer=1,
        options=["Loop of Henle", "Glomerulus", "Collecting duct"],
        explanation="The glomerulus performs initial blood filtration.",
        source_citations=[{"source": "renal-notes", "quote": "Glomeruli filter plasma"}],
    )
    chacha_db.create_question(
        quiz_id=quiz_id,
        question_type="multiple_choice",
        question_text="Which structure concentrates urine?",
        correct_answer=0,
        options=["Loop of Henle", "Glomerulus", "Bowman's capsule"],
        explanation="The Loop of Henle creates the medullary concentration gradient.",
    )
    attempt = chacha_db.start_attempt(quiz_id)
    chacha_db.submit_attempt(
        int(attempt["id"]),
        answers=[
            {"question_id": first_question_id, "user_answer": 2, "time_spent_ms": 1200},
        ],
    )
    return int(attempt["id"]), first_question_id


def test_build_flashcard_assistant_context_uses_only_active_card_and_recent_history(chacha_db: CharactersRAGDB):
    flashcard_uuid = _create_flashcard(chacha_db)
    thread = chacha_db.get_or_create_study_assistant_thread(
        context_type="flashcard",
        flashcard_uuid=flashcard_uuid,
    )
    chacha_db.append_study_assistant_message(
        thread_id=thread["id"],
        role="user",
        action_type="follow_up",
        input_modality="text",
        content="What part of the nephron handles reabsorption?",
        structured_payload={"kind": "question"},
        context_snapshot={"flashcard_uuid": flashcard_uuid},
    )
    chacha_db.append_study_assistant_message(
        thread_id=thread["id"],
        role="assistant",
        action_type="follow_up",
        input_modality="text",
        content="The proximal tubule handles most bulk reabsorption.",
        structured_payload={"kind": "answer"},
        context_snapshot={"flashcard_uuid": flashcard_uuid},
    )

    context = build_flashcard_assistant_context(chacha_db, flashcard_uuid)

    assert context["flashcard"]["uuid"] == flashcard_uuid
    assert context["flashcard"]["front"] == "What is the nephron?"
    assert context["thread"]["id"] == thread["id"]
    assert [message["role"] for message in context["history"]] == ["user", "assistant"]
    assert "deck_cards" not in context


def test_build_flashcard_assistant_context_rejects_missing_card(chacha_db: CharactersRAGDB):
    with pytest.raises(ConflictError):
        build_flashcard_assistant_context(chacha_db, "missing-card-uuid")


def test_build_quiz_attempt_question_context_scopes_to_one_question(chacha_db: CharactersRAGDB):
    attempt_id, question_id = _create_quiz_attempt_question(chacha_db)

    context = build_quiz_attempt_question_context(chacha_db, attempt_id, question_id)

    assert context["question"]["id"] == question_id
    assert context["question"]["question_text"] == "Which structure filters blood?"
    assert context["question"]["user_answer"] == 2
    assert context["question"]["is_correct"] is False
    assert context["question"]["correct_answer"] == 1
    assert context["question"]["source_citations"][0]["source"] == "renal-notes"


def test_normalize_fact_check_payload_fills_required_keys():
    request = StudyAssistantRespondRequest(
        action="fact_check",
        message="I think the collecting duct filters blood.",
        input_modality="voice_transcript",
    )

    payload = normalize_fact_check_payload(
        {
            "verdict": "incorrect",
            "corrections": ["Filtration occurs in the glomerulus."],
        },
        assistant_text="Filtration occurs in the glomerulus.",
    )
    schema_payload = StudyAssistantFactCheckPayload.model_validate(payload)

    assert request.action == "fact_check"
    assert request.input_modality == "voice_transcript"
    assert schema_payload.verdict == "incorrect"
    assert schema_payload.corrections == ["Filtration occurs in the glomerulus."]
    assert schema_payload.missing_points == []
    assert schema_payload.next_prompt


def test_build_flashcard_assistant_context_limits_recent_history_and_field_length(chacha_db: CharactersRAGDB):
    flashcard_uuid = _create_flashcard(chacha_db, notes="N" * 240)
    thread = chacha_db.get_or_create_study_assistant_thread(
        context_type="flashcard",
        flashcard_uuid=flashcard_uuid,
    )
    for index in range(5):
        chacha_db.append_study_assistant_message(
            thread_id=thread["id"],
            role="user" if index % 2 == 0 else "assistant",
            action_type="freeform",
            input_modality="text",
            content=f"message-{index}-" + ("x" * 40),
            structured_payload={"index": index},
            context_snapshot={"flashcard_uuid": flashcard_uuid},
        )

    context = build_flashcard_assistant_context(
        chacha_db,
        flashcard_uuid,
        max_history_messages=2,
        max_field_chars=50,
    )

    assert [message["structured_payload"]["index"] for message in context["history"]] == [3, 4]
    assert len(context["flashcard"]["notes"]) == 50
    assert context["flashcard"]["notes"].endswith("...")


@pytest.mark.unit
def test_study_assistant_prompt_includes_bounded_flashcard_context(chacha_db: CharactersRAGDB) -> None:
    flashcard_uuid = _create_flashcard(chacha_db)
    thread = chacha_db.get_or_create_study_assistant_thread(
        context_type="flashcard",
        flashcard_uuid=flashcard_uuid,
    )
    chacha_db.append_study_assistant_message(
        thread_id=thread["id"],
        role="assistant",
        action_type="explain",
        input_modality="text",
        content="The nephron filters and processes fluid.",
        structured_payload={},
        context_snapshot={"flashcard_uuid": flashcard_uuid},
    )

    context = build_flashcard_assistant_context(chacha_db, flashcard_uuid)
    prompt = build_study_assistant_prompt_package(
        action="fact_check",
        context=context,
        user_message="The nephron is a kidney unit.",
    )

    assert "The functional unit of the kidney." in prompt["user_prompt"]
    assert "Renal physiology" in prompt["user_prompt"]
    assert "Associated with urine formation." in prompt["user_prompt"]
    assert "The nephron filters and processes fluid." in prompt["user_prompt"]


@pytest.mark.unit
def test_study_assistant_prompt_context_remains_valid_json_when_bounded() -> None:
    prompt = build_study_assistant_prompt_package(
        action="explain",
        context={
            "context_type": "flashcard",
            "flashcard": {
                "front": "What is the nephron?",
                "back": "A" * 10_000,
            },
            "history": [{"role": "user", "content": "B" * 10_000}],
        },
    )
    context_json = (
        prompt["user_prompt"]
        .split("Study context JSON:\n", 1)[1]
        .split("\nLearner message:", 1)[0]
    )

    parsed = json.loads(context_json)

    assert parsed["context_type"] == "flashcard"
