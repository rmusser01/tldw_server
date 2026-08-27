from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Notes_Graph.suggestion_maintenance import (
    MaintenanceScope,
    SuggestionMaintenance,
    classify_job_observation,
    run_maintenance_loop,
)
from tldw_Server_API.app.services import notes_graph_suggestions_worker
from tldw_Server_API.app.services.lifecycle_worker_specs import WorkerLifecycleContext
from tldw_Server_API.app.services.startup_study_privilege_jobs_pollers import (
    provide_study_privilege_jobs_worker_specs,
)

NOW = datetime(2026, 8, 27, 16, 0, tzinfo=timezone.utc)


def _context(*, sidecar_mode: bool) -> WorkerLifecycleContext:
    return WorkerLifecycleContext(
        app=object(),
        settings={},
        test_mode=True,
        route_enabled=lambda *_args, **_kwargs: True,
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
        sidecar_mode=sidecar_mode,
    )


def test_worker_configuration_binds_completion_token_and_exact_scope() -> None:
    config = notes_graph_suggestions_worker.build_worker_config(worker_id="worker-1")

    assert config.domain == "notes"
    assert config.queue == "graph-suggestions"
    assert config.bind_completion_token is True
    assert config.retry_on_exception is False


@pytest.mark.asyncio
async def test_completion_callback_loads_current_run_before_publication(monkeypatch) -> None:
    calls: list[object] = []
    run = SimpleNamespace(
        id="run-1",
        revision=19,
        job_id="job-1",
        owner_user_id="owner-1",
        expected_completion_token="lease-1",
        result_digest=f"sha256:{'a' * 64}",
    )

    class Store:
        def get_run(self, **kwargs):
            calls.append(("load", kwargs))
            return run

    db = SimpleNamespace(note_graph_suggestion_store=Store())

    async def open_database(owner_user_id):
        calls.append(("open", owner_user_id))
        return db

    class Publisher:
        def __init__(self, **kwargs):
            calls.append(("publisher", kwargs))

        def publish(self, **kwargs):
            calls.append(("publish", kwargs))

    monkeypatch.setattr(notes_graph_suggestions_worker, "_open_owner_database", open_database)
    monkeypatch.setattr(notes_graph_suggestions_worker, "_close_database", lambda value: calls.append(("close", value)))
    monkeypatch.setattr(notes_graph_suggestions_worker, "SuggestionPublisher", Publisher)

    await notes_graph_suggestions_worker._publish_completed(
        {
            "uuid": "job-1",
            "owner_user_id": "owner-1",
            "lease_id": "lease-1",
            "payload": {"dataset_id": "dataset-1"},
        },
        {"run_id": "run-1", "result_digest": f"sha256:{'a' * 64}"},
        jobs=object(),
    )

    assert calls[1] == (
        "load",
        {"dataset_id": "dataset-1", "run_id": "run-1"},
    )
    assert calls[3][1]["run"] is run
    assert calls[-1] == ("close", db)


def test_app_sidecar_ownership_prevents_duplicate_consumers(monkeypatch) -> None:
    monkeypatch.setenv("NOTES_GRAPH_SUGGESTIONS_WORKER_ENABLED", "true")
    monkeypatch.setenv("NOTES_GRAPH_SUGGESTIONS_MAINTENANCE_ENABLED", "true")
    specs = {spec.name: spec for spec in provide_study_privilege_jobs_worker_specs()}

    assert specs["notes_graph_suggestions_jobs_task"].enabled(_context(sidecar_mode=False)) is True
    assert specs["notes_graph_suggestions_maintenance_task"].enabled(_context(sidecar_mode=False)) is True
    assert specs["notes_graph_suggestions_jobs_task"].enabled(_context(sidecar_mode=True)) is False
    assert specs["notes_graph_suggestions_maintenance_task"].enabled(_context(sidecar_mode=True)) is False


class _ClaimStore:
    def __init__(self, count: int) -> None:
        self.count = count
        self.claim_limits: list[int] = []
        self.released: list[str] = []
        self.cleaned: list[int] = []

    def claim_runs_for_maintenance(self, *, limit, **_kwargs):
        self.claim_limits.append(limit)
        return tuple(
            SimpleNamespace(
                id=f"run-{index}",
                state=SimpleNamespace(value="queued"),
                revision=2,
                job_id=f"job-{index}",
                owner_user_id="owner-1",
                maintenance_lease_token=f"lease-{index}",
                created_at=NOW,
            )
            for index in range(min(limit, self.count))
        )

    def release_run_maintenance_lease(self, *, run_id, **_kwargs):
        self.released.append(run_id)

    def cleanup_retention(self, *, limit, **_kwargs):
        self.cleaned.append(limit)
        return {"suggestions": 0, "receipts": 0, "runs": 0, "rejection_sets": 0}


class _UnavailableJobs:
    def get_job_or_archived_by_uuid(self, *_args, **_kwargs):
        raise ConnectionError("temporary jobs outage")


def test_maintenance_is_provider_independent_and_claims_at_most_100_total() -> None:
    first = _ClaimStore(80)
    second = _ClaimStore(80)
    maintenance = SuggestionMaintenance(
        jobs=_UnavailableJobs(),
        scopes=(
            MaintenanceScope(first, "dataset-1"),
            MaintenanceScope(second, "dataset-2"),
        ),
    )

    result = maintenance.run_pass(now=NOW)

    assert result.claimed == 100
    assert first.claim_limits == [100]
    assert second.claim_limits == [20]
    assert len(first.released) + len(second.released) == 100
    assert sum(first.cleaned + second.cleaned) <= 100


@pytest.mark.asyncio
async def test_maintenance_loop_runs_at_most_once_per_minute() -> None:
    stop = asyncio.Event()
    sleeps: list[float] = []

    class Maintenance:
        calls = 0

        def run_pass(self, *, now):
            del now
            self.calls += 1
            if self.calls == 2:
                stop.set()

    async def sleep(seconds):
        sleeps.append(seconds)
        await asyncio.sleep(0)

    maintenance = Maintenance()
    await run_maintenance_loop(maintenance, stop, sleep=sleep, now=lambda: NOW)

    assert maintenance.calls == 2
    assert sleeps == [60.0]


@pytest.mark.parametrize(
    ("state", "age", "job", "expected"),
    [
        ("admitting", timedelta(minutes=9), None, None),
        ("admitting", timedelta(minutes=10), None, "definitively_missing"),
        ("publishing", timedelta(days=29), None, None),
        ("publishing", timedelta(days=30), None, None),
        ("publishing", timedelta(days=31), None, "publication_receipt_missing"),
        ("running", timedelta(minutes=20), {"status": "failed"}, "terminal_failed"),
        ("cancelling", timedelta(minutes=20), {"status": "cancelled"}, "terminal_cancelled"),
    ],
)
def test_maintenance_classifies_grace_horizons_and_terminal_jobs(state, age, job, expected) -> None:
    run = SimpleNamespace(
        state=SimpleNamespace(value=state),
        created_at=NOW - age,
    )

    assert classify_job_observation(run=run, job=job, now=NOW) == expected


def test_publishing_receipt_mismatch_fails_closed_immediately() -> None:
    run = SimpleNamespace(
        id="run-1",
        job_id="job-1",
        owner_user_id="owner-1",
        expected_completion_token="lease-1",
        result_digest=f"sha256:{'1' * 64}",
        state=SimpleNamespace(value="publishing"),
        created_at=NOW - timedelta(minutes=1),
    )
    job = {
        "uuid": "job-1",
        "owner_user_id": "owner-1",
        "domain": "notes",
        "queue": "graph-suggestions",
        "job_type": "note_graph_suggestions",
        "status": "completed",
        "completion_token": "wrong-token",
        "result": {},
    }

    assert classify_job_observation(run=run, job=job, now=NOW) == ("publication_receipt_mismatch")
