import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.flashcards import (
    StudyAssistantContextResponse,
    StudyAssistantThreadSummary,
)
from tldw_Server_API.app.api.v1.schemas.study_packs import (
    FlashcardCitationResponse,
    FlashcardDeepDiveTarget,
    StudyPackCreateJobRequest,
    StudyPackSourceSelection,
    StudyPackSummaryResponse,
)


def _thread_summary() -> StudyAssistantThreadSummary:
    return StudyAssistantThreadSummary(
        id=7,
        context_type="flashcard",
        flashcard_uuid="card-123",
        quiz_attempt_id=None,
        question_id=None,
        last_message_at="2026-04-01T12:00:00Z",
        message_count=0,
        deleted=False,
        client_id="tests",
        version=1,
        created_at="2026-04-01T12:00:00Z",
        last_modified="2026-04-01T12:00:00Z",
    )


def test_create_job_request_defaults_to_new_deck_mode():
    payload = StudyPackCreateJobRequest(
        title="Operating Systems",
        source_items=[StudyPackSourceSelection(source_type="note", source_id="note-123")],
    )

    assert payload.deck_mode == "new"  # nosec B101
    assert payload.source_items[0].source_type == "note"  # nosec B101


def test_create_job_request_rejects_non_new_deck_mode():
    with pytest.raises(ValidationError):
        StudyPackCreateJobRequest(
            title="Operating Systems",
            deck_mode="existing",
            source_items=[StudyPackSourceSelection(source_type="note", source_id="note-123")],
        )


def test_create_job_request_requires_source_items_field():
    with pytest.raises(ValidationError):
        StudyPackCreateJobRequest(title="Operating Systems")


def test_study_pack_source_selection_rejects_whitespace_only_source_id():
    with pytest.raises(ValidationError):
        StudyPackSourceSelection(source_type="note", source_id="   ")


def test_study_pack_source_selection_accepts_locator_excerpt_and_source_title_alias():
    selection = StudyPackSourceSelection(
        source_type="media",
        source_id="42",
        source_title="Lecture 42",
        locator={"timestamp_seconds": 61.5},
        excerpt_text="The lecture explains additive increase.",
    )

    assert selection.label == "Lecture 42"  # nosec B101
    assert selection.locator["timestamp_seconds"] == 61.5  # nosec B101
    assert selection.excerpt_text == "The lecture explains additive increase."  # nosec B101


def test_study_pack_summary_status_only_allows_active_or_superseded():
    active = StudyPackSummaryResponse(
        id=11,
        workspace_id="ws-1",
        title="Networking",
        deck_id=22,
        source_bundle_json={"items": [{"source_type": "note", "source_id": "n1"}]},
        generation_options_json={"deck_mode": "new"},
        status="active",
        superseded_by_pack_id=None,
        created_at="2026-04-01T12:00:00Z",
        last_modified="2026-04-01T12:00:00Z",
        deleted=False,
        client_id="tests",
        version=1,
    )

    assert active.status == "active"  # nosec B101

    superseded = StudyPackSummaryResponse(
        id=12,
        workspace_id="ws-1",
        title="Networking (Updated)",
        deck_id=23,
        source_bundle_json={"items": [{"source_type": "note", "source_id": "n1"}]},
        generation_options_json={"deck_mode": "new"},
        status="superseded",
        superseded_by_pack_id=99,
        created_at="2026-04-01T12:00:00Z",
        last_modified="2026-04-01T12:05:00Z",
        deleted=False,
        client_id="tests",
        version=2,
    )

    assert superseded.status == "superseded"  # nosec B101
    assert superseded.superseded_by_pack_id == 99  # nosec B101

    with pytest.raises(ValidationError):
        StudyPackSummaryResponse(
            id=13,
            workspace_id="ws-1",
            title="Networking",
            deck_id=22,
            source_bundle_json={"items": [{"source_type": "note", "source_id": "n1"}]},
            generation_options_json={"deck_mode": "new"},
            status="archived",
            superseded_by_pack_id=None,
            created_at="2026-04-01T12:00:00Z",
            last_modified="2026-04-01T12:00:00Z",
            deleted=False,
            client_id="tests",
            version=1,
        )


def test_study_assistant_context_response_accepts_study_pack_provenance_fields():
    citation = FlashcardCitationResponse(
        id=3,
        flashcard_uuid="card-123",
        source_type="note",
        source_id="note-123",
        citation_text="Virtual memory uses disk as backing store.",
        locator="section-2",
        ordinal=0,
        created_at="2026-04-01T12:00:00Z",
        last_modified="2026-04-01T12:00:00Z",
        deleted=False,
        client_id="tests",
        version=1,
    )
    study_pack = StudyPackSummaryResponse(
        id=11,
        workspace_id="ws-1",
        title="Operating Systems",
        deck_id=22,
        source_bundle_json={"items": [{"source_type": "note", "source_id": "note-123"}]},
        generation_options_json={"deck_mode": "new"},
        status="active",
        superseded_by_pack_id=None,
        created_at="2026-04-01T12:00:00Z",
        last_modified="2026-04-01T12:00:00Z",
        deleted=False,
        client_id="tests",
        version=1,
    )
    response = StudyAssistantContextResponse(
        thread=_thread_summary(),
        messages=[],
        context_snapshot={"flashcard_uuid": "card-123"},
        available_actions=["explain"],
        citations=[citation],
        primary_citation=citation,
        deep_dive_target=FlashcardDeepDiveTarget(
            source_type="note",
            source_id="note-123",
            citation_ordinal=0,
        ),
        study_pack=study_pack,
    )

    assert response.citations[0].ordinal == 0  # nosec B101
    assert response.primary_citation.source_id == "note-123"  # nosec B101
    assert response.deep_dive_target.citation_ordinal == 0  # nosec B101
    assert response.study_pack.workspace_id == "ws-1"  # nosec B101
