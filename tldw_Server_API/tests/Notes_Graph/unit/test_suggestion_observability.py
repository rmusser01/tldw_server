from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Notes_Graph import suggestion_observability as observability


class _Registry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float, dict[str, str]]] = []

    def increment(self, name, value=1, labels=None) -> None:
        self.calls.append(("increment", name, value, dict(labels or {})))

    def observe(self, name, value, labels=None) -> None:
        self.calls.append(("observe", name, value, dict(labels or {})))


def test_event_api_is_closed_and_rejects_unsafe_values(monkeypatch) -> None:
    recorded: list[dict[str, object]] = []
    monkeypatch.setattr(observability, "_write_event", recorded.append)

    for event in observability.SuggestionEventName:
        observability.record_event(event, run_id="run-1", count=1)

    assert {item["event"] for item in recorded} == {
        "run_admitted",
        "shortlist_completed",
        "provider_started",
        "provider_completed",
        "validation_rejected",
        "staged",
        "published",
        "cancelled",
        "failed",
        "accepted",
        "rejected",
        "stale",
        "reconciled",
    }
    with pytest.raises(ValueError):
        observability.record_event("provider_prompt", run_id="run-1")
    with pytest.raises(ValueError):
        observability.record_event(
            observability.SuggestionEventName.FAILED,
            run_id="unsafe/note text",
        )
    with pytest.raises(ValueError):
        observability.record_event(
            observability.SuggestionEventName.FAILED,
            run_id="run-1",
            count=-1,
        )


def test_metric_helpers_record_non_vacuous_safe_local_metrics(monkeypatch) -> None:
    registry = _Registry()
    monkeypatch.setattr(observability, "get_metrics_registry", lambda: registry)

    observability.record_queue_latency(0.25)
    observability.record_run_duration(1.5)
    observability.record_candidate_counts(candidates=12, evidence=24)
    observability.record_provider_usage(input_tokens=100, output_tokens=30)
    observability.record_validation_counts(validated=4, dropped=2)
    observability.record_run_error("notes_graph_provider_call_failed")
    observability.record_decision_outcome(observability.DecisionOutcome.ACCEPTED)
    observability.record_acceptance_reconciliation(observability.ReconciliationOutcome.RELEASED)

    names = {call[1] for call in registry.calls}
    assert names == {
        "notes_graph_suggestion_queue_latency_seconds",
        "notes_graph_suggestion_run_duration_seconds",
        "notes_graph_suggestion_candidate_count",
        "notes_graph_suggestion_evidence_count",
        "notes_graph_suggestion_provider_input_tokens",
        "notes_graph_suggestion_provider_output_tokens",
        "notes_graph_suggestion_validated_count",
        "notes_graph_suggestion_dropped_count",
        "notes_graph_suggestion_run_errors_total",
        "notes_graph_suggestion_decisions_total",
        "notes_graph_suggestion_acceptance_reconciliation_total",
    }
    assert all(set(labels) <= {"error_code", "outcome"} for _method, _name, _value, labels in registry.calls)
    with pytest.raises(ValueError):
        observability.record_run_error("note title")
    with pytest.raises(ValueError):
        observability.record_provider_usage(input_tokens=10**9, output_tokens=1)
