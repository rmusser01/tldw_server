from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.chacha.note_graph_suggestion_models import (
    NoteGraphSuggestion,
    NoteGraphSuggestionEvidence,
    NoteGraphSuggestionEvidenceField,
    NoteGraphSuggestionEvidenceRead,
    NoteGraphSuggestionEvidenceSide,
    NoteGraphSuggestionKind,
    NoteGraphSuggestionState,
    SuggestionEvidenceNote,
)
from tldw_Server_API.app.core.Notes_Graph.suggestion_api import (
    NotesGraphSuggestionsAPI,
    OpaqueSuggestionCursorCodec,
    SuggestionAPIError,
)
from tldw_Server_API.app.core.Notes_Graph.suggestion_capabilities import (
    SuggestionCapabilities,
    SuggestionCapabilityLimits,
)
from tldw_Server_API.app.core.Notes_Graph.suggestion_content import content_fingerprint

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 27, 19, 0, tzinfo=timezone.utc)
SOURCE_ID = "00000000-0000-4000-8000-000000000001"
TARGET_ID = "00000000-0000-4000-8000-000000000002"
SOURCE_FINGERPRINT = content_fingerprint("Source", "source body")
TARGET_FINGERPRINT = content_fingerprint("Target", "target body")


def _capabilities(*, available: bool = True) -> SuggestionCapabilities:
    return SuggestionCapabilities(
        provider="openai",
        model="model-a",
        endpoint_origin_revision=f"sha256:{'3' * 64}",
        data_boundary="remote",
        disclosure_external=True,
        outbound_data_categories=("selected_note_excerpts",),
        generation_available=available,
        unavailable_reason=None if available else "notes_graph_provider_unavailable",
        limits=SuggestionCapabilityLimits(),
        allowed_actions=("generate", "cancel", "accept", "reject", "reset_rejections"),
        revision=f"sha256:{'4' * 64}",
    )


class Store:
    def __init__(self) -> None:
        self.after = None
        self.suggestion = NoteGraphSuggestion(
            id="suggestion-1",
            run_id="run-1",
            owner_user_id="owner-1",
            dataset_id="dataset-1",
            kind=NoteGraphSuggestionKind.RELATED_NOTE,
            source_note_id=SOURCE_ID,
            source_fingerprint=SOURCE_FINGERPRINT,
            target_note_id=TARGET_ID,
            target_fingerprint=TARGET_FINGERPRINT,
            state=NoteGraphSuggestionState.PENDING,
            revision=1,
            created_at=NOW.isoformat(),
            updated_at=NOW.isoformat(),
            match_strength="strong",
            rationale="Grounded rationale",
        )

    @staticmethod
    def load_source_note(*, dataset_id: str, note_id: str):
        assert dataset_id == "dataset-1"
        if note_id == SOURCE_ID:
            return SimpleNamespace(note_id=note_id, title="Source", content="source body")
        if note_id == TARGET_ID:
            return SimpleNamespace(note_id=note_id, title="Target", content="target body")
        raise ValueError("Notes graph source is unavailable")

    @staticmethod
    def ensure_fts_ready(*, dataset_id: str) -> None:
        assert dataset_id == "dataset-1"

    def list_suggestions(self, **kwargs):
        self.after = kwargs["after"]
        return SimpleNamespace(
            items=(self.suggestion,),
            next_position=(NOW.isoformat(), "suggestion-1") if self.after is None else None,
        )

    @staticmethod
    def list_suggestion_evidence(**_kwargs):
        return (
            NoteGraphSuggestionEvidenceRead(
                evidence=NoteGraphSuggestionEvidence(
                    suggestion_id="suggestion-1",
                    owner_user_id="owner-1",
                    dataset_id="dataset-1",
                    side=NoteGraphSuggestionEvidenceSide.SOURCE,
                    ordinal=0,
                    note_id=SOURCE_ID,
                    field=NoteGraphSuggestionEvidenceField.CONTENT,
                    content_fingerprint=SOURCE_FINGERPRINT,
                    start_offset=0,
                    end_offset=6,
                ),
                excerpt_note=SuggestionEvidenceNote(
                    note_id=SOURCE_ID,
                    title="Source",
                    content="source body",
                ),
            ),
        )

    @staticmethod
    def get_rejection_set(**_kwargs):
        return SimpleNamespace(revision=7, rejection_count=2)

    def get_suggestion(self, **_kwargs):
        return self.suggestion


def _api(
    *,
    store: Store | None = None,
    capabilities: SuggestionCapabilities | None = None,
    worker_ready=lambda: True,
    admission_service=None,
    cancellation_coordinator=None,
) -> NotesGraphSuggestionsAPI:
    resolved = SimpleNamespace(
        capabilities=capabilities or _capabilities(),
        provider=SimpleNamespace(adapter="openai", model="model-a"),
    )
    class DefaultAdmission:
        @staticmethod
        def admit(**kwargs):
            run = SimpleNamespace(
                provider="openai",
                model="model-a",
                capability_revision=(capabilities or _capabilities()).revision,
            )
            kwargs["validate_before_enqueue"](run)
            return SimpleNamespace(disposition="created", run=run, job=None)

    return NotesGraphSuggestionsAPI(
        store=store or Store(),
        owner_user_id="owner-1",
        dataset_id="dataset-1",
        admission_service=admission_service or DefaultAdmission(),
        cancellation_coordinator=cancellation_coordinator or SimpleNamespace(),
        decision_service=SimpleNamespace(),
        resolve_capability=lambda **_kwargs: resolved,
        worker_ready=worker_ready,
        feature_ready=lambda: True,
        cursor_codec=OpaqueSuggestionCursorCodec(b"cursor-secret-at-least-32-bytes"),
        clock=lambda: NOW,
    )


def test_capability_preflight_sanitizes_expected_worker_unavailability_without_revision_drift() -> None:
    available = _api().get_capabilities(
        note_id=SOURCE_ID,
        provider="openai",
        model="model-a",
    )
    unavailable = _api(worker_ready=lambda: False).get_capabilities(
        note_id=SOURCE_ID,
        provider="openai",
        model="model-a",
    )

    assert available.generation_available is True
    assert unavailable.generation_available is False
    assert unavailable.unavailable_reason == "notes_graph_suggestions_worker_unavailable"
    assert unavailable.revision == available.revision
    assert not hasattr(unavailable, "endpoint_url")
    assert not hasattr(unavailable, "api_key")


def test_admission_requires_exact_preflight_revision_and_uses_that_same_revision() -> None:
    calls: list[dict[str, object]] = []

    class Admission:
        @staticmethod
        def admit(**kwargs):
            run = SimpleNamespace(
                id="run-1",
                provider="openai",
                model="model-a",
                capability_revision=_capabilities().revision,
                state=SimpleNamespace(value="queued"),
            )
            kwargs["validate_before_enqueue"](run)
            calls.append(kwargs)
            return SimpleNamespace(
                disposition="created",
                run=run,
                job={"uuid": "job-1"},
            )

    api = _api(admission_service=Admission())
    result = api.admit_run(
        note_id=SOURCE_ID,
        provider="openai",
        model="model-a",
        capability_revision=_capabilities().revision,
        idempotency_key="admit-key",
    )

    assert result.run.id == "run-1"
    assert calls[0]["capability_revision"] == _capabilities().revision
    assert calls[0]["source_fingerprint"] == SOURCE_FINGERPRINT

    with pytest.raises(SuggestionAPIError) as exc_info:
        api.admit_run(
            note_id=SOURCE_ID,
            provider="openai",
            model="model-a",
            capability_revision=f"sha256:{'9' * 64}",
            idempotency_key="different-key",
        )
    assert (exc_info.value.status_code, exc_info.value.code) == (
        412,
        "notes_graph_capabilities_changed",
    )


def test_admission_maps_current_provider_unavailability_to_stable_503() -> None:
    api = _api(capabilities=replace(_capabilities(), generation_available=False, unavailable_reason="notes_graph_provider_unavailable"))

    with pytest.raises(SuggestionAPIError) as exc_info:
        api.admit_run(
            note_id=SOURCE_ID,
            provider="openai",
            model="model-a",
            capability_revision=_capabilities().revision,
            idempotency_key="admit-key",
        )

    assert (exc_info.value.status_code, exc_info.value.code) == (
        503,
        "notes_graph_provider_unavailable",
    )


def test_suggestion_page_cursor_is_encoded_outside_store_and_evidence_is_reconstructed() -> None:
    store = Store()
    api = _api(store=store)

    first = api.list_suggestions(note_id=SOURCE_ID, states=None, limit=1, cursor=None)
    assert store.after is None
    assert first.next_cursor is not None
    assert NOW.isoformat() not in first.next_cursor
    assert first.current_source_fingerprint == SOURCE_FINGERPRINT
    assert first.rejection_set_revision == 7
    assert first.items[0].evidence[0].text == "source"

    second = api.list_suggestions(
        note_id=SOURCE_ID,
        states=("pending", "accepting"),
        limit=1,
        cursor=first.next_cursor,
    )
    assert store.after == (NOW.isoformat(), "suggestion-1")
    assert second.next_cursor is None

    with pytest.raises(SuggestionAPIError) as exc_info:
        api.list_suggestions(
            note_id=SOURCE_ID,
            states=("pending",),
            limit=1,
            cursor=f"{first.next_cursor}x",
        )
    assert (exc_info.value.status_code, exc_info.value.code) == (
        422,
        "notes_graph_cursor_invalid",
    )


def test_run_page_rejects_an_invalid_opaque_cursor_with_stable_422() -> None:
    with pytest.raises(SuggestionAPIError) as exc_info:
        _api().list_runs(
            note_id=SOURCE_ID,
            states=None,
            limit=20,
            cursor="not-a-signed-cursor",
        )

    assert (exc_info.value.status_code, exc_info.value.code) == (
        422,
        "notes_graph_cursor_invalid",
    )


def test_cancellation_replays_owner_scoped_receipt_after_run_cleanup() -> None:
    class MissingRunStore(Store):
        @staticmethod
        def get_run(**_kwargs):
            raise RuntimeError("notes_graph_run_not_found")

    calls: list[dict[str, object]] = []

    class Cancellation:
        @staticmethod
        def cancel(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                cancellation=SimpleNamespace(
                    source_note_id=SOURCE_ID,
                    replay_envelope={"run_id": "run-1", "state": "cancelling", "revision": 4},
                ),
                accepted=True,
            )

    result = _api(
        store=MissingRunStore(),
        cancellation_coordinator=Cancellation(),
    ).cancel_run(
        note_id=SOURCE_ID,
        run_id="run-1",
        expected_revision=3,
        idempotency_key="cancel-key",
    )

    assert result.cancellation.replay_envelope["run_id"] == "run-1"
    assert calls[0]["expected_state"] == "running"


def test_cancellation_missing_run_and_receipt_remains_non_enumerating() -> None:
    class MissingRunStore(Store):
        @staticmethod
        def get_run(**_kwargs):
            raise RuntimeError("notes_graph_run_not_found")

    calls: list[dict[str, object]] = []

    class Cancellation:
        @staticmethod
        def cancel(**kwargs):
            calls.append(kwargs)
            raise RuntimeError("notes_graph_run_cancel_resource_missing")

    with pytest.raises(SuggestionAPIError) as exc_info:
        _api(
            store=MissingRunStore(),
            cancellation_coordinator=Cancellation(),
        ).cancel_run(
            note_id=SOURCE_ID,
            run_id="missing-run",
            expected_revision=3,
            idempotency_key="missing-cancel-key",
        )

    assert (exc_info.value.status_code, exc_info.value.code) == (
        404,
        "notes_graph_suggestion_not_found",
    )
    assert len(calls) == 1


def test_accept_permission_requirements_depend_on_persisted_suggestion_kind() -> None:
    store = Store()
    api = _api(store=store)
    assert api.accept_permission_requirements(
        note_id=SOURCE_ID,
        suggestion_id="suggestion-1",
    ) == ("notes.graph.write",)

    store.suggestion = replace(
        store.suggestion,
        kind=NoteGraphSuggestionKind.TAG,
        target_note_id=None,
        target_fingerprint=None,
        normalized_tag="research",
        display_tag="Research",
        keyword_sync_id=None,
    )
    assert api.accept_permission_requirements(
        note_id=SOURCE_ID,
        suggestion_id="suggestion-1",
    ) == ("notes.link_keyword", "keywords.create")
