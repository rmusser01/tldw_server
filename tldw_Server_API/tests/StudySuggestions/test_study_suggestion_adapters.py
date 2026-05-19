from __future__ import annotations

import json
from pathlib import Path

import pytest

from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.StudySuggestions import snapshot_service
from tldw_Server_API.app.core.StudySuggestions.flashcard_adapter import (
    build_flashcard_suggestion_context,
    extract_flashcard_suggestion_evidence,
    is_source_grounded_session,
)
from tldw_Server_API.app.core.StudySuggestions.quiz_adapter import build_quiz_suggestion_context


def _admin_principal() -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        username="admin",
        roles=["admin"],
        permissions=["*"],
        is_admin=True,
    )


def _load_flashcard_audit_cases() -> list[dict[str, object]]:
    fixture_path = Path(__file__).with_name("fixtures") / "flashcard_grounding_audit_cases.json"
    return json.loads(fixture_path.read_text(encoding="utf-8"))


@pytest.fixture
def db(tmp_path):
    chacha = CharactersRAGDB(str(tmp_path / "study-suggestion-adapters.db"), client_id="study-suggestion-adapter-tests")
    try:
        yield chacha
    finally:
        chacha.close_connection()


def test_quiz_attempts_emit_stable_suggestion_context():
    context = build_quiz_suggestion_context(
        quiz_attempt={
            "id": 42,
            "quiz_id": 9,
            "workspace_id": "ws-7",
            "score": 3,
            "correct_answers": 3,
            "total_questions": 5,
            "question_results": [
                {"question_id": 1, "correct": False, "topic_label": "Kidney Function"},
                {"question_id": 2, "correct": True, "topic_label": "Electrolyte Balance"},
            ],
        },
        source_bundle=[
            {"source_type": "note", "source_id": "note-11", "label": "Kidney Function"},
        ],
    )

    assert context.service == "quiz"  # nosec B101
    assert context.activity_type == "quiz_attempt"  # nosec B101
    assert context.anchor_type == "quiz_attempt"  # nosec B101
    assert context.anchor_id == 42  # nosec B101
    assert context.workspace_id == "ws-7"  # nosec B101
    assert context.summary_metrics == {  # nosec B101
        "score": 3,
        "correct_answers": 3,
        "total_questions": 5,
    }
    assert context.performance_signals == {  # nosec B101
        "incorrect_count": 1,
        "accuracy": 0.6,
    }
    assert context.source_bundle == [  # nosec B101
        {"source_type": "note", "source_id": "note-11", "label": "Kidney Function"},
    ]


def test_flashcard_sessions_expose_grounded_flag_for_provenance_backed_session():
    session = {
        "id": 12,
        "deck_id": 5,
        "workspace_id": "ws-9",
        "review_mode": "due",
        "cards_reviewed": 8,
        "correct_count": 6,
        "tag_labels": ["renal basics"],
        "study_pack_id": 44,
        "source_bundle": [
            {"source_type": "note", "source_id": "note-8", "citation_ordinal": 0},
        ],
    }

    context = build_flashcard_suggestion_context(session)

    assert is_source_grounded_session(session) is True  # nosec B101
    assert context.service == "flashcards"  # nosec B101
    assert context.activity_type == "flashcard_review_session"  # nosec B101
    assert context.anchor_type == "flashcard_review_session"  # nosec B101
    assert context.anchor_id == 12  # nosec B101
    assert context.workspace_id == "ws-9"  # nosec B101
    assert context.performance_signals["is_source_grounded_session"] is True  # nosec B101


def test_flashcard_sessions_without_lineage_remain_exploratory_only():
    session = {
        "id": 15,
        "deck_id": None,
        "workspace_id": None,
        "review_mode": "manual",
        "cards_reviewed": 4,
        "correct_count": 2,
        "tag_labels": ["renal basics"],
        "source_bundle": [],
    }

    context = build_flashcard_suggestion_context(session)

    assert is_source_grounded_session(session) is False  # nosec B101
    assert context.summary_metrics["deck_id"] is None  # nosec B101
    assert context.performance_signals["is_source_grounded_session"] is False  # nosec B101
    assert context.performance_signals["supports_source_aware_adjacency"] is False  # nosec B101


def test_flashcard_evidence_extraction_uses_study_pack_and_reviewed_card_provenance():
    session = {
        "id": 18,
        "deck_id": 5,
        "workspace_id": "ws-3",
        "review_mode": "due",
        "cards_reviewed": 2,
        "correct_count": 1,
        "tag_filter": None,
        "source_bundle": [],
        "study_pack_id": 44,
    }

    evidence = extract_flashcard_suggestion_evidence(
        session,
        provenance={
            "deck_name": "Renal Review Deck",
            "study_pack": {
                "id": 44,
                "title": "Renal Pack",
                "source_bundle_json": {
                    "items": [
                        {"source_type": "note", "source_id": "note-7", "label": "Renal basics"},
                    ]
                },
            },
            "reviewed_cards": [
                {
                    "uuid": "card-1",
                    "tags_json": '["Electrolyte balance"]',
                    "source_ref_type": "note",
                    "source_ref_id": "note-7",
                },
            ],
        },
    )

    assert evidence["source_labels"] == ["Renal basics"]  # nosec B101
    assert evidence["tag_labels"] == ["Electrolyte balance"]  # nosec B101
    assert "Renal Pack" in evidence["derived_labels"]  # nosec B101
    assert "Renal Review Deck" in evidence["derived_labels"]  # nosec B101
    assert evidence["source_bundle"][0] == {  # nosec B101
        "source_type": "note",
        "source_id": "note-7",
        "label": "Renal basics",
    }
    assert any(  # nosec B101
        item["source_type"] == "note" and item["source_id"] == "note-7"
        for item in evidence["source_bundle"]
    )


def test_flashcard_evidence_extraction_accepts_json_encoded_string_tags():
    evidence = extract_flashcard_suggestion_evidence(
        {
            "id": 19,
            "deck_id": 5,
            "review_mode": "due",
            "cards_reviewed": 1,
            "correct_count": 1,
            "source_bundle": [],
        },
        provenance={
            "reviewed_cards": [
                {
                    "uuid": "card-1",
                    "tags_json": '"Renal focus"',
                    "source_ref_type": "note",
                    "source_ref_id": "note-9",
                }
            ],
        },
    )

    assert evidence["tag_labels"] == ["Renal focus"]  # nosec B101
    assert "Renal focus" in evidence["adjacent_labels"]  # nosec B101


def _create_flashcard_session_for_audit_case(
    db: CharactersRAGDB,
    case: dict[str, object],
) -> tuple[int, int, int]:
    deck_name = str(case["deck_name"])
    deck_id = db.add_deck(deck_name, "desc")
    session = db.get_or_create_flashcard_review_session(
        deck_id=deck_id,
        review_mode=str(case["review_mode"]),
        tag_filter=case.get("tag_filter") if isinstance(case.get("tag_filter"), str) else None,
        scope_key=f"{case['scope_prefix']}:{deck_id}",
    )
    db.execute_query(
        """
        UPDATE flashcard_review_sessions
           SET source_bundle_json = ?,
               study_pack_id = ?
         WHERE id = ?
        """,
        (
            json.dumps(case.get("source_bundle_json") or []),
            case.get("study_pack_id"),
            session["id"],
        ),
        commit=True,
    )
    review_specs = case.get("reviews") or []
    for index, review_spec in enumerate(review_specs):
        card_uuid = db.add_flashcard(
            {
                "deck_id": deck_id,
                "front": f"Front {index}",
                "back": f"Back {index}",
            }
        )
        db.review_flashcard(
            card_uuid,
            rating=int(review_spec["rating"]),
            answer_time_ms=review_spec.get("answer_time_ms"),
            review_session_id=int(session["id"]),
        )
    return deck_id, int(session["id"]), len(review_specs)


@pytest.mark.parametrize("case", _load_flashcard_audit_cases())
def test_flashcard_snapshot_audit_cases_classify_grounding_from_rollups(
    db: CharactersRAGDB,
    case: dict[str, object],
):
    _, session_id, review_count = _create_flashcard_session_for_audit_case(db, case)

    snapshot_id = snapshot_service.refresh_snapshot_for_anchor(
        note_db=db,
        anchor_type="flashcard_review_session",
        anchor_id=session_id,
        principal=_admin_principal(),
    )
    row = db.get_suggestion_snapshot(snapshot_id)
    payload = row["payload_json"]
    topics = payload["topics"][:3]
    grounded_types = {topic["type"] for topic in topics if isinstance(topic, dict)}
    expected_source_bundle = case.get("source_bundle_json") or []
    correct_count = sum(1 for review_spec in (case.get("reviews") or []) if int(review_spec["rating"]) >= 3)

    assert payload["summary"]["correct_count"] == correct_count  # nosec B101
    assert payload["summary"]["total_count"] == review_count  # nosec B101
    assert all("source_count" in topic for topic in topics)  # nosec B101

    if expected_source_bundle:
        assert grounded_types & {"grounded", "weakly_grounded"}  # nosec B101
    else:
        assert grounded_types.isdisjoint({"grounded", "weakly_grounded"})  # nosec B101


def test_flashcard_snapshot_assigns_source_refs_only_to_matching_topics(db: CharactersRAGDB):
    deck_id = db.add_deck("Grounded Audit Deck", "desc")
    session = db.get_or_create_flashcard_review_session(
        deck_id=deck_id,
        review_mode="due",
        tag_filter=None,
        scope_key=f"due:deck:{deck_id}",
    )
    db.execute_query(
        """
        UPDATE flashcard_review_sessions
           SET source_bundle_json = ?
         WHERE id = ?
        """,
        (
            '[{"source_type":"note","source_id":"note-7","label":"Renal basics"}]',
            session["id"],
        ),
        commit=True,
    )
    card_uuid = db.add_flashcard(
        {
            "deck_id": deck_id,
            "front": "Front",
            "back": "Back",
        }
    )
    db.review_flashcard(
        card_uuid,
        rating=4,
        answer_time_ms=700,
        review_session_id=int(session["id"]),
    )

    snapshot_id = snapshot_service.refresh_snapshot_for_anchor(
        note_db=db,
        anchor_type="flashcard_review_session",
        anchor_id=int(session["id"]),
        principal=_admin_principal(),
    )
    payload = db.get_suggestion_snapshot(snapshot_id)["payload_json"]
    topics_by_label = {
        topic["display_label"]: topic
        for topic in payload["topics"]
        if isinstance(topic, dict) and isinstance(topic.get("display_label"), str)
    }

    assert topics_by_label["Renal Basics"]["source_id"] == "note-7"  # nosec B101
    assert "source_id" not in topics_by_label["Grounded Audit Deck"]  # nosec B101
