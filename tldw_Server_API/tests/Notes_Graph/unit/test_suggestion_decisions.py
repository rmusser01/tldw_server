"""Decision orchestration for reviewable Notes graph suggestions."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.chacha.note_graph_suggestion_models import (
    NoteGraphSuggestion,
    NoteGraphSuggestionKind,
    NoteGraphSuggestionState,
)
from tldw_Server_API.app.core.DB_Management.chacha.note_graph_suggestion_store import MutationResult
from tldw_Server_API.app.core.DB_Management.chacha.organization_sync_store import (
    NotesOrganizationSyncStore,
)
from tldw_Server_API.app.core.Notes_Graph import suggestion_decisions as decisions_module
from tldw_Server_API.app.core.Notes_Graph.suggestion_decisions import SuggestionDecisionService
from tldw_Server_API.app.core.Notes_Graph.suggestion_maintenance import (
    MaintenanceScope,
    SuggestionMaintenance,
)
from tldw_Server_API.app.core.Sync.v2.notes_organization import organization_link_id

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 27, 16, 0, tzinfo=timezone.utc)
SOURCE_ID = "11111111-1111-4111-8111-111111111111"
TARGET_ID = "22222222-2222-4222-8222-222222222222"
KEYWORD_ID = "33333333-3333-4333-8333-333333333333"
EDGE_ID = "44444444-4444-4444-8444-444444444444"


def _suggestion(kind: NoteGraphSuggestionKind) -> NoteGraphSuggestion:
    return NoteGraphSuggestion(
        id=f"suggestion-{kind.value}",
        run_id="run-1",
        owner_user_id="owner-1",
        dataset_id="dataset-1",
        kind=kind,
        source_note_id=SOURCE_ID,
        source_fingerprint="sha256:source",
        target_note_id=TARGET_ID if kind == NoteGraphSuggestionKind.RELATED_NOTE else None,
        target_fingerprint="sha256:target" if kind == NoteGraphSuggestionKind.RELATED_NOTE else None,
        normalized_tag="research" if kind == NoteGraphSuggestionKind.TAG else None,
        display_tag="Research" if kind == NoteGraphSuggestionKind.TAG else None,
        keyword_sync_id=None,
        state=NoteGraphSuggestionState.ACCEPTING,
        revision=2,
        acceptance_lease_token="lease-1",
        acceptance_lease_expires_at="2026-08-27T16:05:00+00:00",
        decision_receipt_id="receipt-1",
        created_at=NOW.isoformat(),
        updated_at=NOW.isoformat(),
    )


class FakeStore:
    def __init__(self, suggestion: NoteGraphSuggestion) -> None:
        self.suggestion = suggestion
        self.renewed: list[NoteGraphSuggestion] = []
        self.finalized: list[str] = []
        self.resolve_result: MutationResult | None = None

    def claim_acceptance(self, **_kwargs: Any) -> MutationResult:
        return MutationResult("completed", {}, suggestion=self.suggestion)

    def renew_acceptance(self, *, fence: NoteGraphSuggestion, **_kwargs: Any) -> NoteGraphSuggestion:
        self.renewed.append(fence)
        return fence

    def guard_acceptance_in_transaction(self, *, fence: NoteGraphSuggestion, **_kwargs: Any) -> None:
        assert fence == self.suggestion

    def finalize_existing_acceptance(self, **_kwargs: Any) -> MutationResult:
        return MutationResult("in_progress", {}, suggestion=self.suggestion)

    def finalize_acceptance_in_transaction(
        self, *, accepted_resource_identity: str, **_kwargs: Any
    ) -> MutationResult:
        self.finalized.append(accepted_resource_identity)
        accepted = replace(
            self.suggestion,
            state=NoteGraphSuggestionState.ACCEPTED,
            revision=3,
            accepted_resource_identity=accepted_resource_identity,
        )
        return MutationResult(
            "completed",
            {
                "suggestion_id": accepted.id,
                "state": "accepted",
                "revision": 3,
                "accepted_resource_identity": accepted_resource_identity,
            },
            suggestion=accepted,
        )

    def claim_expired_acceptances(self, **_kwargs: Any) -> tuple[NoteGraphSuggestion, ...]:
        return (self.suggestion,)

    def resolve_expired_acceptance(self, **_kwargs: Any) -> MutationResult:
        if self.resolve_result is not None:
            return self.resolve_result
        pending = replace(self.suggestion, state=NoteGraphSuggestionState.PENDING, revision=3)
        return MutationResult(
            "completed",
            {"suggestion_id": pending.id, "state": "pending", "revision": 3},
            suggestion=pending,
        )


class FakeLinkCoordinator:
    dataset_id = "dataset-1"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        guard = kwargs["guarded_mutation"]
        assert guard.expected_domain == "notes.link"
        guard.before(object())
        guard.after(object(), EDGE_ID)
        return SimpleNamespace(edge_id=EDGE_ID)


class FakeOrganizationCoordinator:
    def __init__(self) -> None:
        self.captures: list[dict[str, Any]] = []
        self.note_db = SimpleNamespace(get_keyword_by_text=lambda _value: None)
        self.service = SimpleNamespace(store=SimpleNamespace(get_current_head=lambda *_args: None))

    def active_dataset(self) -> SimpleNamespace:
        return SimpleNamespace(dataset_id="dataset-1")

    def plan_keyword_create(self, _display: str, **_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(steps=(SimpleNamespace(domain="notes.keyword", object_id=KEYWORD_ID),))

    def plan_relationship(
        self, _domain: str, members: dict[str, str], *_args: Any, **_kwargs: Any
    ) -> SimpleNamespace:
        object_id = organization_link_id(
            "notes.keyword_link",
            [members["subject_type"], members["subject_id"], members["keyword_sync_id"]],
        )
        return SimpleNamespace(
            steps=(SimpleNamespace(domain="notes.keyword_link", object_id=object_id),)
        )

    def capture(self, **kwargs: Any) -> None:
        self.captures.append(kwargs)
        for guard in kwargs["guarded_mutations"]:
            guard.before(object())
            if guard.after is not None:
                guard.after(object(), guard.expected_object_id)


def test_relationship_acceptance_uses_exact_guarded_manual_link_contract() -> None:
    suggestion = _suggestion(NoteGraphSuggestionKind.RELATED_NOTE)
    store = FakeStore(suggestion)
    links = FakeLinkCoordinator()
    service = SuggestionDecisionService(
        store=store,
        link_coordinator=links,
        organization_coordinator=FakeOrganizationCoordinator(),
        clock=lambda: NOW,
    )

    result = service.accept(
        dataset_id="dataset-1",
        suggestion_id=suggestion.id,
        expected_revision=1,
        expected_source_fingerprint=suggestion.source_fingerprint,
        expected_target_fingerprint=suggestion.target_fingerprint,
        idempotency_key="accept-request",
    )

    assert result.envelope["state"] == "accepted"
    assert store.renewed == [suggestion]
    assert store.finalized == [EDGE_ID]
    assert links.calls[0] | {"guarded_mutation": None} == {
        "source_note_id": SOURCE_ID,
        "target_note_id": TARGET_ID,
        "directed": False,
        "weight": 1.0,
        "label": None,
        "properties": {},
        "idempotency_key": f"notes-graph:{suggestion.id}:link",
        "guarded_mutation": None,
    }


def test_new_tag_renews_each_step_and_only_relationship_guard_finalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suggestion = _suggestion(NoteGraphSuggestionKind.TAG)
    store = FakeStore(suggestion)
    organization = FakeOrganizationCoordinator()
    keyword_lookups = iter((None, SimpleNamespace(sync_id=KEYWORD_ID)))
    monkeypatch.setattr(
        NotesOrganizationSyncStore,
        "find_keyword_by_normalized_identity",
        lambda *_args, **_kwargs: next(keyword_lookups),
    )
    service = SuggestionDecisionService(
        store=store,
        link_coordinator=FakeLinkCoordinator(),
        organization_coordinator=organization,
        clock=lambda: NOW,
    )

    result = service.accept(
        dataset_id="dataset-1",
        suggestion_id=suggestion.id,
        expected_revision=1,
        expected_source_fingerprint=suggestion.source_fingerprint,
        expected_target_fingerprint=None,
        idempotency_key="accept-tag-request",
    )

    assert result.envelope["state"] == "accepted"
    assert store.renewed == [suggestion, suggestion]
    assert [capture["idempotency_key"] for capture in organization.captures] == [
        f"notes-graph:{suggestion.id}:keyword",
        f"notes-graph:{suggestion.id}:tag-membership",
    ]
    guards = [capture["guarded_mutations"][0] for capture in organization.captures]
    assert [(guard.expected_domain, guard.after is None) for guard in guards] == [
        ("notes.keyword", True),
        ("notes.keyword_link", False),
    ]


def test_reconciliation_only_calls_store_resolution_and_never_mutates_product() -> None:
    suggestion = _suggestion(NoteGraphSuggestionKind.RELATED_NOTE)
    store = FakeStore(suggestion)
    links = FakeLinkCoordinator()
    organization = FakeOrganizationCoordinator()
    service = SuggestionDecisionService(
        store=store,
        link_coordinator=links,
        organization_coordinator=organization,
        clock=lambda: NOW,
    )

    results = service.reconcile_expired(dataset_id="dataset-1", limit=100)

    assert [result.envelope["state"] for result in results] == ["pending"]
    assert links.calls == []
    assert organization.captures == []


def test_maintenance_runs_acceptance_reconciliation_without_generation_authority() -> None:
    class MaintenanceStore:
        def claim_runs_for_maintenance(self, **_kwargs: Any) -> tuple[()]:
            return ()

        def cleanup_retention(self, **_kwargs: Any) -> dict[str, int]:
            return {"suggestions": 0, "receipts": 0, "runs": 0, "rejection_sets": 0}

    calls: list[tuple[str, int]] = []
    decisions = SimpleNamespace(
        reconcile_expired=(
            lambda *, dataset_id, limit, now: calls.append((dataset_id, limit)) or ()
        )
    )
    maintenance = SuggestionMaintenance(
        jobs=object(),
        scopes=(MaintenanceScope(MaintenanceStore(), "dataset-1", decisions),),
    )

    result = maintenance.run_pass(now=NOW, limit=100)

    assert calls == [("dataset-1", 100)]
    assert result.claimed == result.reconciled == 0


@pytest.mark.parametrize(
    "terminal_state",
    (NoteGraphSuggestionState.ACCEPTED, NoteGraphSuggestionState.STALE),
)
def test_reconciliation_emits_terminal_decision_before_reconciled_observability(
    monkeypatch: pytest.MonkeyPatch,
    terminal_state: NoteGraphSuggestionState,
) -> None:
    suggestion = _suggestion(NoteGraphSuggestionKind.RELATED_NOTE)
    terminal = replace(
        suggestion,
        state=terminal_state,
        revision=3,
        accepted_resource_identity=(
            EDGE_ID if terminal_state == NoteGraphSuggestionState.ACCEPTED else None
        ),
    )
    store = FakeStore(suggestion)
    store.resolve_result = MutationResult(
        "completed",
        {
            "suggestion_id": terminal.id,
            "state": terminal_state.value,
            "revision": terminal.revision,
        },
        suggestion=terminal,
    )
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        decisions_module,
        "record_decision_outcome",
        lambda outcome: calls.append(("decision_metric", outcome.value)),
    )
    monkeypatch.setattr(
        decisions_module,
        "record_acceptance_reconciliation",
        lambda outcome: calls.append(("reconciliation_metric", outcome.value)),
    )
    monkeypatch.setattr(
        decisions_module,
        "record_event",
        lambda event, **_kwargs: calls.append(("event", event.value)),
    )
    service = SuggestionDecisionService(
        store=store,
        link_coordinator=FakeLinkCoordinator(),
        organization_coordinator=FakeOrganizationCoordinator(),
        clock=lambda: NOW,
    )

    service.reconcile_expired(dataset_id="dataset-1", limit=1)

    assert calls == [
        ("decision_metric", terminal_state.value),
        ("event", terminal_state.value),
        ("reconciliation_metric", "completed"),
        ("event", "reconciled"),
    ]


def test_reconciliation_exception_emits_no_observability_before_durable_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suggestion = _suggestion(NoteGraphSuggestionKind.RELATED_NOTE)
    store = FakeStore(suggestion)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        decisions_module,
        "record_acceptance_reconciliation",
        lambda outcome: calls.append(("reconciliation_metric", outcome.value)),
    )
    service = SuggestionDecisionService(
        store=store,
        link_coordinator=FakeLinkCoordinator(),
        organization_coordinator=FakeOrganizationCoordinator(),
        clock=lambda: NOW,
    )
    monkeypatch.setattr(service, "_resolve", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        service.reconcile_expired(dataset_id="dataset-1", limit=1)

    assert calls == []


def test_maintenance_shares_one_acceptance_reconciliation_budget_across_scopes() -> None:
    class MaintenanceStore:
        def claim_runs_for_maintenance(self, **_kwargs: Any) -> tuple[()]:
            return ()

        def cleanup_retention(self, **_kwargs: Any) -> dict[str, int]:
            return {"suggestions": 0, "receipts": 0, "runs": 0, "rejection_sets": 0}

    calls: list[tuple[str, int]] = []

    def decisions_for(count: int) -> SimpleNamespace:
        def reconcile(*, dataset_id: str, limit: int, now: datetime) -> tuple[object, ...]:
            del now
            calls.append((dataset_id, limit))
            return tuple(object() for _ in range(min(count, limit)))

        return SimpleNamespace(reconcile_expired=reconcile)

    maintenance = SuggestionMaintenance(
        jobs=object(),
        scopes=(
            MaintenanceScope(MaintenanceStore(), "dataset-1", decisions_for(60)),
            MaintenanceScope(MaintenanceStore(), "dataset-2", decisions_for(60)),
            MaintenanceScope(MaintenanceStore(), "dataset-3", decisions_for(60)),
        ),
    )

    result = maintenance.run_pass(now=NOW, limit=100)

    assert calls == [("dataset-1", 100), ("dataset-2", 40)]
    assert result.claimed == result.reconciled == 100
