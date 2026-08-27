from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Notes_Graph.suggestion_jobs import (
    SuggestionCancellationCoordinator,
)

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 27, 18, 0, tzinfo=timezone.utc)


def _run(*, state: str = "cancelling", revision: int = 4):
    return SimpleNamespace(
        id="run-1",
        state=SimpleNamespace(value=state),
        revision=revision,
        job_id="job-1",
        owner_user_id="owner-1",
    )


def test_cancellation_admits_before_jobs_and_completes_after_acceptance() -> None:
    events: list[str] = []
    cancelling = _run()

    class Store:
        def admit_run_cancellation(self, **_kwargs):
            events.append("store.admit")
            return SimpleNamespace(
                disposition="created",
                operation_id="operation-1",
                run=cancelling,
                replay_envelope=None,
            )

        def get_run_cancellation_continuation(self, **_kwargs):
            events.append("store.continue")
            return SimpleNamespace(
                disposition="in_progress",
                operation_id="operation-1",
                run=cancelling,
                replay_envelope=None,
            )

        def complete_run_cancellation_receipt(self, **_kwargs):
            events.append("store.complete")
            return SimpleNamespace(
                disposition="completed",
                operation_id="operation-1",
                run=cancelling,
                replay_envelope={"run_id": "run-1", "state": "cancelling", "revision": 4},
            )

    class Jobs:
        def get_job_or_archived_by_uuid(self, *_args, **_kwargs):
            events.append("jobs.lookup")
            return {
                "id": 7,
                "uuid": "job-1",
                "owner_user_id": "owner-1",
                "domain": "notes",
                "queue": "graph-suggestions",
                "job_type": "note_graph_suggestions",
                "status": "processing",
            }

        def cancel_job(self, *_args, **_kwargs):
            events.append("jobs.cancel")
            return True

    result = SuggestionCancellationCoordinator(
        store=Store(),
        jobs=Jobs(),
        owner_user_id="owner-1",
    ).cancel(
        dataset_id="dataset-1",
        run_id="run-1",
        expected_state="running",
        expected_revision=3,
        idempotency_key="cancel-key",
        now=NOW,
    )

    assert events == [
        "store.admit",
        "store.continue",
        "jobs.lookup",
        "jobs.cancel",
        "store.complete",
    ]
    assert result.accepted is True
    assert result.cancellation.replay_envelope["run_id"] == "run-1"


def test_terminal_cancellation_replay_never_calls_jobs() -> None:
    envelope = {"run_id": "run-1", "state": "cancelled", "revision": 5}

    class Store:
        @staticmethod
        def admit_run_cancellation(**_kwargs):
            return SimpleNamespace(
                disposition="terminal_replay",
                operation_id="operation-1",
                run=None,
                replay_envelope=envelope,
            )

    class Jobs:
        @staticmethod
        def get_job_or_archived_by_uuid(*_args, **_kwargs):
            raise AssertionError("terminal replay must not consult Jobs")

    result = SuggestionCancellationCoordinator(
        store=Store(),
        jobs=Jobs(),
        owner_user_id="owner-1",
    ).cancel(
        dataset_id="dataset-1",
        run_id="run-1",
        expected_state="running",
        expected_revision=3,
        idempotency_key="cancel-key",
        now=NOW,
    )

    assert result.accepted is True
    assert result.cancellation.replay_envelope == envelope
