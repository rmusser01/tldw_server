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
            NoteGraphSuggestionEvidenceRead(
                evidence=NoteGraphSuggestionEvidence(
                    suggestion_id="suggestion-1",
                    owner_user_id="owner-1",
                    dataset_id="dataset-1",
                    side=NoteGraphSuggestionEvidenceSide.TARGET,
                    ordinal=1,
                    note_id=TARGET_ID,
                    field=NoteGraphSuggestionEvidenceField.CONTENT,
                    content_fingerprint=TARGET_FINGERPRINT,
                    start_offset=0,
                    end_offset=6,
                ),
                excerpt_note=SuggestionEvidenceNote(
                    note_id=TARGET_ID,
                    title="Target",
                    content="target body",
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
    decision_service=None,
    resolve_capability=None,
) -> NotesGraphSuggestionsAPI:
    resolved = SimpleNamespace(
        capabilities=capabilities or _capabilities(),
        provider=SimpleNamespace(adapter="openai", model="model-a"),
    )
    class DefaultAdmission:
        @staticmethod
        def replay(**_kwargs):
            return None

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
        decision_service=decision_service or SimpleNamespace(),
        resolve_capability=resolve_capability or (lambda **_kwargs: resolved),
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


def test_terminal_admission_replay_precedes_source_and_capability_resolution() -> None:
    envelope = {
        "run_id": "run-replay",
        "provider": "openai",
        "model": "model-a",
        "state": "queued",
        "revision": 2,
        "created_at": NOW.isoformat(),
        "started_at": None,
        "completed_at": None,
        "suggestion_count": 0,
        "related_note_count": 0,
        "tag_count": 0,
        "invalid_item_count": 0,
        "cancellation_available": True,
        "error_code": None,
        "guidance_key": None,
    }

    class NoCurrentSourceStore(Store):
        @staticmethod
        def load_source_note(**_kwargs):
            raise AssertionError("terminal replay must not load the source note")

    class Admission:
        @staticmethod
        def replay(**kwargs):
            assert kwargs["source_note_id"] == SOURCE_ID
            return SimpleNamespace(
                disposition="terminal_replay",
                run=None,
                replay_envelope=envelope,
            )

        @staticmethod
        def admit(**_kwargs):
            raise AssertionError("terminal replay must not continue admission")

    def unavailable_resolver(**_kwargs):
        raise AssertionError("terminal replay must not resolve current capability")

    result = _api(
        store=NoCurrentSourceStore(),
        admission_service=Admission(),
        resolve_capability=unavailable_resolver,
    ).admit_run(
        note_id=SOURCE_ID,
        provider="openai",
        model="model-a",
        capability_revision=f"sha256:{'9' * 64}",
        idempotency_key="terminal-replay-key",
    )

    assert result.replay_envelope == envelope


def test_capability_preflight_sanitizes_missing_provider_while_admission_stays_422() -> None:
    def missing_provider(**_kwargs):
        raise ValueError("notes_graph_provider_model_disallowed")

    api = _api(resolve_capability=missing_provider)
    capabilities = api.get_capabilities(note_id=SOURCE_ID, provider=None, model=None)

    assert capabilities.generation_available is False
    assert capabilities.unavailable_reason == "notes_graph_provider_disallowed"
    assert capabilities.revision.startswith("sha256:")
    assert not hasattr(capabilities, "endpoint_url")

    with pytest.raises(SuggestionAPIError) as exc_info:
        api.admit_run(
            note_id=SOURCE_ID,
            provider=None,
            model=None,
            capability_revision=capabilities.revision,
            idempotency_key="missing-provider-key",
        )
    assert (exc_info.value.status_code, exc_info.value.code) == (
        422,
        "notes_graph_provider_disallowed",
    )


def test_capability_preflight_preserves_partially_resolved_safe_defaults() -> None:
    error = ValueError("notes_graph_provider_model_disallowed")
    error.provider = "openai"
    error.model = None

    def missing_model(**_kwargs):
        raise error

    capabilities = _api(resolve_capability=missing_model).get_capabilities(
        note_id=SOURCE_ID,
        provider=None,
        model=None,
    )

    assert capabilities.provider == "openai"
    assert capabilities.model == "unconfigured"
    assert capabilities.data_boundary == "unknown"
    assert capabilities.disclosure_external is True


def test_terminal_decision_replays_precede_shorter_lived_resources_and_sync_readiness() -> None:
    accept = SimpleNamespace(
        envelope={
            "suggestion_id": "suggestion-1",
            "state": "accepted",
            "revision": 2,
            "accepted_resource_identity": "edge-1",
            "authorization_scope": "relationship",
        }
    )
    reject = SimpleNamespace(
        envelope={"suggestion_id": "suggestion-1", "state": "rejected", "revision": 2}
    )
    reset = SimpleNamespace(
        envelope={
            "source_note_id": SOURCE_ID,
            "cleared_count": 2,
            "rejection_set_revision": 8,
        }
    )

    class ReceiptFirstStore(Store):
        @staticmethod
        def load_source_note(**_kwargs):
            raise AssertionError("terminal decision replay must not load a note")

        @staticmethod
        def get_suggestion(**_kwargs):
            raise AssertionError("terminal decision replay must not load a suggestion")

        @staticmethod
        def get_acceptance_authorization_scope(**kwargs):
            assert kwargs["idempotency_key"] == "accept-key"
            return "relationship"

        @staticmethod
        def get_terminal_acceptance_replay(**_kwargs):
            return accept

        @staticmethod
        def get_terminal_rejection_replay(**_kwargs):
            return reject

        @staticmethod
        def get_terminal_rejection_reset_replay(**_kwargs):
            return reset

    class Decisions:
        @staticmethod
        def accept(**_kwargs):
            return accept

        @staticmethod
        def reject(**_kwargs):
            return reject

        @staticmethod
        def reset_rejections(**_kwargs):
            return reset

    api = _api(store=ReceiptFirstStore(), decision_service=Decisions())
    request = {
        "expected_revision": 1,
        "expected_source_fingerprint": SOURCE_FINGERPRINT,
        "expected_target_fingerprint": TARGET_FINGERPRINT,
        "idempotency_key": "accept-key",
    }

    assert api.accept_permission_requirements(
        note_id=SOURCE_ID,
        suggestion_id="suggestion-1",
        **request,
    ) == ("notes.graph.write",)
    assert api.accept_suggestion(
        note_id=SOURCE_ID,
        suggestion_id="suggestion-1",
        **request,
    ) is accept
    assert api.reject_suggestion(
        note_id=SOURCE_ID,
        suggestion_id="suggestion-1",
        expected_revision=1,
        expected_source_fingerprint=SOURCE_FINGERPRINT,
        expected_target_fingerprint=TARGET_FINGERPRINT,
        idempotency_key="reject-key",
    ) is reject
    assert api.reset_rejections(
        note_id=SOURCE_ID,
        source_fingerprint=SOURCE_FINGERPRINT,
        expected_revision=7,
        idempotency_key="reset-key",
    ) is reset


def test_terminal_cancellation_replay_precedes_current_run_and_source_state() -> None:
    terminal = SimpleNamespace(
        cancellation=SimpleNamespace(
            source_note_id=SOURCE_ID,
            replay_envelope={"run_id": "run-1", "state": "cancelling", "revision": 4},
        ),
        accepted=True,
    )

    class NoCurrentRunStore(Store):
        @staticmethod
        def load_source_note(**_kwargs):
            raise AssertionError("terminal cancellation replay must not load the source")

        @staticmethod
        def get_run(**_kwargs):
            raise AssertionError("terminal cancellation replay must not load the run")

    class Cancellation:
        @staticmethod
        def cancel(**kwargs):
            assert kwargs["expected_state"] is None
            assert kwargs["expected_source_note_id"] == SOURCE_ID
            return terminal

    result = _api(
        store=NoCurrentRunStore(),
        cancellation_coordinator=Cancellation(),
    ).cancel_run(
        note_id=SOURCE_ID,
        run_id="run-1",
        expected_revision=3,
        idempotency_key="cancel-key",
    )

    assert result is terminal


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
    assert first.items[0].target_title == "Target"

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


def test_suggestion_target_title_requires_current_matching_target_evidence() -> None:
    class StaleTargetStore(Store):
        @staticmethod
        def list_suggestion_evidence(**_kwargs):
            rows = Store.list_suggestion_evidence()
            stale_target = replace(
                rows[1],
                excerpt_note=replace(rows[1].excerpt_note, title="Changed target"),
            )
            return (rows[0], stale_target)

    item = _api(store=StaleTargetStore()).list_suggestions(
        note_id=SOURCE_ID,
        states=None,
        limit=1,
        cursor=None,
    ).items[0]

    assert item.target_title is None


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
    assert calls[0]["expected_state"] is None
    assert calls[0]["expected_source_note_id"] == SOURCE_ID


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
