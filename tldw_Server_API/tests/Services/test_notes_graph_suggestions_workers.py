from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from threading import Event
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError
from tldw_Server_API.app.core.Notes_Graph.suggestion_maintenance import (
    MaintenanceScope,
    SuggestionMaintenance,
    classify_job_observation,
    missing_job_reference_at,
    run_maintenance_loop,
)
from tldw_Server_API.app.services import (
    notes_graph_suggestions_maintenance,
    notes_graph_suggestions_worker,
)
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


@pytest.mark.asyncio
async def test_completion_callback_yields_the_event_loop_during_sync_publication(
    monkeypatch,
) -> None:
    started = Event()
    release = Event()
    run = SimpleNamespace(id="run-1")

    class Store:
        @staticmethod
        def get_run(**_kwargs):
            return run

    class Publisher:
        def __init__(self, **_kwargs):
            pass

        @staticmethod
        def publish(**_kwargs):
            started.set()
            if not release.wait(timeout=1):
                raise AssertionError("event loop could not release synchronous publication")

    async def open_database(_owner_user_id):
        return SimpleNamespace(note_graph_suggestion_store=Store())

    monkeypatch.setattr(notes_graph_suggestions_worker, "_open_owner_database", open_database)
    monkeypatch.setattr(notes_graph_suggestions_worker, "_close_database", lambda _db: None)
    monkeypatch.setattr(notes_graph_suggestions_worker, "SuggestionPublisher", Publisher)
    task = asyncio.create_task(
        notes_graph_suggestions_worker._publish_completed(
            {
                "uuid": "job-1",
                "owner_user_id": "owner-1",
                "lease_id": "lease-1",
                "payload": {"dataset_id": "dataset-1"},
            },
            {"run_id": "run-1", "result_digest": f"sha256:{'a' * 64}"},
            jobs=object(),
        )
    )

    assert await asyncio.to_thread(started.wait, 0.5) is True
    release.set()
    await task


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

    def get_run_cancellation_maintenance_context(self, **_kwargs):
        return None

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


def test_maintenance_shares_run_and_acceptance_claim_budget_at_exact_cap() -> None:
    store = _ClaimStore(70)
    acceptance_limits: list[int] = []

    def reconcile_expired(*, limit, **_kwargs):
        acceptance_limits.append(limit)
        return tuple(object() for _ in range(min(limit, 40)))

    maintenance = SuggestionMaintenance(
        jobs=_UnavailableJobs(),
        scopes=(
            MaintenanceScope(
                store,
                "dataset-1",
                SimpleNamespace(reconcile_expired=reconcile_expired),
            ),
        ),
    )

    result = maintenance.run_pass(now=NOW)

    assert result.claimed == 100
    assert result.reconciled == 30
    assert result.released == 70
    assert store.claim_limits == [100]
    assert acceptance_limits == [30]
    assert store.cleaned == []


def test_maintenance_reports_acceptance_claims_before_reconciliation_failure() -> None:
    class Store(_ClaimStore):
        def __init__(self) -> None:
            super().__init__(0)

    class Decisions:
        @staticmethod
        def reconcile_expired(*, on_claimed, **_kwargs):
            on_claimed(30)
            raise RuntimeError("acceptance reconciliation failed after claim")

    claimed: list[int] = []
    maintenance = SuggestionMaintenance(
        jobs=_UnavailableJobs(),
        scopes=(MaintenanceScope(Store(), "dataset-1", Decisions()),),
    )

    with pytest.raises(RuntimeError, match="failed after claim"):
        maintenance.run_pass(now=NOW, on_claimed=claimed.append)

    assert claimed == [30]


@pytest.mark.asyncio
async def test_maintenance_loop_runs_at_most_once_per_minute() -> None:
    stop = asyncio.Event()
    sleeps: list[float] = []

    class Maintenance:
        calls = 0

        async def run_pass(self, *, now):
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


@pytest.mark.asyncio
async def test_maintenance_yields_the_event_loop_during_sync_store_work(
    monkeypatch,
) -> None:
    started = Event()
    release = Event()

    class UsersRepo:
        @staticmethod
        async def list_users(*, offset, limit):
            assert (offset, limit) == (0, 200)
            return ([{"id": 1}], 1)

    class Store:
        @staticmethod
        def list_maintenance_dataset_ids(*, limit):
            assert limit == 100
            started.set()
            if not release.wait(timeout=1):
                raise AssertionError("event loop could not release synchronous maintenance")
            return ()

    async def open_database(_owner_user_id):
        return SimpleNamespace(note_graph_suggestion_store=Store())

    monkeypatch.setattr(notes_graph_suggestions_maintenance, "_open_owner_database", open_database)
    monkeypatch.setattr(notes_graph_suggestions_maintenance, "_close_database", lambda _db: None)
    runner = notes_graph_suggestions_maintenance._MaintenanceRunner(
        jobs=_UnavailableJobs(),
        users_repo=UsersRepo(),
    )
    task = asyncio.create_task(runner.run_pass(now=NOW))

    assert await asyncio.to_thread(started.wait, 0.5) is True
    release.set()
    result = await task

    assert result.claimed == 0


@pytest.mark.parametrize(
    ("state", "age", "job", "missing_since", "expected"),
    [
        ("admitting", timedelta(minutes=9), None, None, None),
        ("admitting", timedelta(minutes=10), None, None, "definitively_missing"),
        ("queued", timedelta(hours=1), None, NOW - timedelta(minutes=9), None),
        (
            "queued",
            timedelta(hours=1),
            None,
            NOW - timedelta(minutes=10),
            "definitively_missing",
        ),
        ("running", timedelta(hours=1), None, NOW - timedelta(minutes=9), None),
        (
            "running",
            timedelta(hours=1),
            None,
            NOW - timedelta(minutes=10),
            "definitively_missing",
        ),
        ("cancelling", timedelta(hours=1), None, NOW - timedelta(minutes=9), None),
        (
            "cancelling",
            timedelta(hours=1),
            None,
            NOW - timedelta(minutes=10),
            "definitively_missing",
        ),
        ("publishing", timedelta(days=29), None, None, None),
        ("publishing", timedelta(days=30), None, None, None),
        ("publishing", timedelta(days=31), None, None, "publication_receipt_missing"),
        ("running", timedelta(minutes=20), {"status": "failed"}, None, "terminal_failed"),
        (
            "cancelling",
            timedelta(minutes=20),
            {"status": "cancelled"},
            None,
            "terminal_cancelled",
        ),
    ],
)
def test_maintenance_classifies_grace_horizons_and_terminal_jobs(
    state,
    age,
    job,
    missing_since,
    expected,
) -> None:
    run = SimpleNamespace(
        state=SimpleNamespace(value=state),
        created_at=NOW - age,
    )

    assert (
        classify_job_observation(
            run=run,
            job=job,
            now=NOW,
            missing_since=missing_since,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("state", "started_at", "cancellation_created_at", "expected"),
    [
        ("admitting", None, None, NOW - timedelta(hours=1)),
        ("queued", None, None, NOW - timedelta(hours=1)),
        ("running", NOW - timedelta(minutes=3), None, NOW - timedelta(minutes=3)),
        (
            "cancelling",
            NOW - timedelta(minutes=3),
            NOW - timedelta(minutes=1),
            NOW - timedelta(minutes=1),
        ),
        ("publishing", NOW - timedelta(minutes=3), None, NOW - timedelta(hours=1)),
    ],
)
def test_missing_job_reference_uses_best_authoritative_persisted_timestamp(
    state,
    started_at,
    cancellation_created_at,
    expected,
) -> None:
    run = SimpleNamespace(
        state=SimpleNamespace(value=state),
        created_at=(NOW - timedelta(hours=1)).isoformat(),
        started_at=started_at.isoformat() if started_at else None,
    )

    assert (
        missing_job_reference_at(
            run,
            cancellation_created_at=(cancellation_created_at.isoformat() if cancellation_created_at else None),
        )
        == expected
    )


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


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_point", ("open", "list", "maintenance", "close"))
async def test_maintenance_isolates_owner_failures_and_closes_each_database_promptly(
    monkeypatch,
    failure_point,
) -> None:
    events: list[tuple[str, str, int | None]] = []

    class UsersRepo:
        async def list_users(self, *, offset, limit):
            assert limit == 200
            events.append(("page", str(offset), None))
            return ([{"id": 1}, {"id": 2}, {"id": 3}], 3)

    class Store(_ClaimStore):
        def __init__(self, owner):
            super().__init__(40 if owner in {"1", "3"} else 0)
            self.owner = owner

        def list_maintenance_dataset_ids(self, *, limit):
            events.append(("list", self.owner, limit))
            if self.owner == "2" and failure_point == "list":
                raise RuntimeError("broken owner list")
            return (f"dataset-{self.owner}",)

        def claim_runs_for_maintenance(self, *, limit, **kwargs):
            events.append(("claim", self.owner, limit))
            if self.owner == "2" and failure_point == "maintenance":
                raise CharactersRAGDBError("broken owner maintenance")
            return super().claim_runs_for_maintenance(limit=limit, **kwargs)

    class Database:
        def __init__(self, owner):
            self.owner = owner
            self.note_graph_suggestion_store = Store(owner)

        def release_context_connection(self):
            events.append(("close", self.owner, None))
            if self.owner == "2" and failure_point == "close":
                raise RuntimeError("broken owner close")

    async def get_database(user_id, *, client_id):
        owner = str(user_id)
        assert client_id == owner
        events.append(("open", owner, None))
        if owner == "2" and failure_point == "open":
            raise RuntimeError("broken owner open")
        return Database(owner)

    monkeypatch.setattr(
        notes_graph_suggestions_maintenance,
        "get_chacha_db_for_user_id",
        get_database,
    )
    runner = notes_graph_suggestions_maintenance._MaintenanceRunner(
        jobs=_UnavailableJobs(),
        users_repo=UsersRepo(),
    )

    result = await runner.run_pass(now=NOW)

    assert (result.claimed, result.released, result.cleaned) == (80, 80, 0)
    assert ("claim", "1", 100) in events
    assert ("claim", "3", 60) in events
    assert events.index(("close", "1", None)) < events.index(("open", "2", None))
    if failure_point != "open":
        assert events.index(("close", "2", None)) < events.index(("open", "3", None))
    assert events[-1] == ("close", "3", None)


@pytest.mark.asyncio
async def test_maintenance_debits_claims_before_owner_reconciliation_failure(
    monkeypatch,
) -> None:
    stores: dict[str, _ClaimStore] = {}

    class UsersRepo:
        async def list_users(self, *, offset, limit):
            assert (offset, limit) == (0, 200)
            return ([{"id": 1}, {"id": 2}], 2)

    class Store(_ClaimStore):
        def __init__(self, owner: str) -> None:
            super().__init__(80)
            self.owner = owner

        def list_maintenance_dataset_ids(self, *, limit):
            assert limit == 100
            return (f"dataset-{self.owner}",)

        def claim_runs_for_maintenance(self, *, limit, **kwargs):
            runs = super().claim_runs_for_maintenance(limit=limit, **kwargs)
            for run in runs:
                run.owner_user_id = f"owner-{self.owner}"
            return runs

        def reconcile_run_after_job_lookup(self, **_kwargs):
            if self.owner == "1":
                raise CharactersRAGDBError("owner reconciliation failed after claim")
            raise AssertionError("healthy owner jobs should remain nonterminal")

    class Database:
        def __init__(self, owner: str) -> None:
            self.store = Store(owner)
            self.note_graph_suggestion_store = self.store
            stores[owner] = self.store

        def release_context_connection(self):
            return None

    async def get_database(user_id, *, client_id):
        assert client_id == str(user_id)
        return Database(str(user_id))

    class Jobs:
        @staticmethod
        def get_job_or_archived_by_uuid(job_id, *, owner_user_id, **_kwargs):
            if owner_user_id != "owner-1":
                return None
            return {
                "uuid": job_id,
                "owner_user_id": owner_user_id,
                "domain": "notes",
                "queue": "graph-suggestions",
                "job_type": "note_graph_suggestions",
                "status": "completed",
            }

    monkeypatch.setattr(
        notes_graph_suggestions_maintenance,
        "get_chacha_db_for_user_id",
        get_database,
    )
    result = await notes_graph_suggestions_maintenance._MaintenanceRunner(
        jobs=Jobs(),
        users_repo=UsersRepo(),
    ).run_pass(now=NOW)

    assert stores["1"].claim_limits == [100]
    assert stores["2"].claim_limits == [20]
    assert result.claimed == 100
    assert result.released == 20


def test_maintenance_resumes_cancellation_receipt_with_guarded_jobs_command(
    monkeypatch,
) -> None:
    run = SimpleNamespace(
        id="run-cancelling",
        state=SimpleNamespace(value="cancelling"),
        revision=9,
        job_id="job-cancelling",
        owner_user_id="owner-1",
        maintenance_lease_token="maintenance-lease",
        created_at=NOW - timedelta(hours=1),
        started_at=(NOW - timedelta(minutes=5)).isoformat(),
        error_code="user_cancelled",
    )
    calls: list[tuple[str, object]] = []

    class Store:
        def claim_runs_for_maintenance(self, **_kwargs):
            return (run,)

        def get_run_cancellation_maintenance_context(self, **kwargs):
            calls.append(("context", kwargs))
            return SimpleNamespace(
                operation_id="cancel-operation",
                state="in_progress",
                created_at=(NOW - timedelta(minutes=1)).isoformat(),
            )

        def get_run_cancellation_continuation(self, **kwargs):
            calls.append(("continuation", kwargs))
            return SimpleNamespace(disposition="in_progress", run=run)

        def complete_run_cancellation_receipt(self, **kwargs):
            calls.append(("complete", kwargs))
            return SimpleNamespace(disposition="completed", run=run)

        def reconcile_run_after_job_lookup(self, **kwargs):
            calls.append(("reconcile", kwargs))
            return SimpleNamespace(
                id=run.id,
                state=SimpleNamespace(value="cancelled"),
                error_code=None,
            )

        def cleanup_retention(self, **_kwargs):
            return {"suggestions": 0, "receipts": 0, "runs": 0, "rejection_sets": 0}

    class Jobs:
        lookups = 0

        def get_job_or_archived_by_uuid(self, *_args, **_kwargs):
            self.lookups += 1
            return {
                "id": 17,
                "uuid": run.job_id,
                "owner_user_id": run.owner_user_id,
                "domain": "notes",
                "queue": "graph-suggestions",
                "job_type": "note_graph_suggestions",
                "status": "processing" if self.lookups == 1 else "cancelled",
            }

        def cancel_job(self, job_id, **kwargs):
            calls.append(("cancel_job", (job_id, kwargs)))
            return True

    events: list[object] = []
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Notes_Graph.suggestion_maintenance.record_event",
        lambda event, **kwargs: events.append((event, kwargs)),
    )
    result = SuggestionMaintenance(
        jobs=Jobs(),
        scopes=(MaintenanceScope(Store(), "dataset-1"),),
    ).run_pass(now=NOW)

    assert (result.claimed, result.reconciled, result.released) == (1, 1, 0)
    assert [name for name, _detail in calls] == [
        "context",
        "continuation",
        "cancel_job",
        "complete",
        "reconcile",
    ]
    cancel_job_id, cancel_kwargs = calls[2][1]
    assert cancel_job_id == 17
    assert cancel_kwargs == {
        "reason": "requested",
        "expected_uuid": "job-cancelling",
        "expected_domain": "notes",
        "expected_job_type": "note_graph_suggestions",
        "cascade_dependents": False,
    }
    assert calls[3][1]["operation_id"] == "cancel-operation"
    assert any(event.value == "cancelled" for event, _kwargs in events)
    assert any(event.value == "reconciled" for event, _kwargs in events)


def test_maintenance_keeps_cancellation_receipt_in_progress_when_command_not_accepted() -> None:
    run = SimpleNamespace(
        id="run-cancelling",
        state=SimpleNamespace(value="cancelling"),
        revision=9,
        job_id="job-cancelling",
        owner_user_id="owner-1",
        maintenance_lease_token="maintenance-lease",
        created_at=NOW - timedelta(hours=1),
        started_at=(NOW - timedelta(minutes=5)).isoformat(),
        error_code="user_cancelled",
    )
    released: list[str] = []

    class Store:
        def claim_runs_for_maintenance(self, **_kwargs):
            return (run,)

        def get_run_cancellation_maintenance_context(self, **_kwargs):
            return SimpleNamespace(
                operation_id="cancel-operation",
                state="in_progress",
                created_at=(NOW - timedelta(minutes=1)).isoformat(),
            )

        def get_run_cancellation_continuation(self, **_kwargs):
            return SimpleNamespace(disposition="in_progress", run=run)

        def complete_run_cancellation_receipt(self, **_kwargs):
            raise AssertionError("unaccepted cancellation must not complete its receipt")

        def release_run_maintenance_lease(self, *, run_id, **_kwargs):
            released.append(run_id)

        def cleanup_retention(self, **_kwargs):
            return {"suggestions": 0, "receipts": 0, "runs": 0, "rejection_sets": 0}

    class Jobs:
        def get_job_or_archived_by_uuid(self, *_args, **_kwargs):
            return {
                "id": 17,
                "uuid": run.job_id,
                "owner_user_id": run.owner_user_id,
                "domain": "notes",
                "queue": "graph-suggestions",
                "job_type": "note_graph_suggestions",
                "status": "processing",
            }

        def cancel_job(self, *_args, **_kwargs):
            return False

    result = SuggestionMaintenance(
        jobs=Jobs(),
        scopes=(MaintenanceScope(Store(), "dataset-1"),),
    ).run_pass(now=NOW)

    assert (result.reconciled, result.released) == (0, 1)
    assert released == [run.id]


@pytest.mark.asyncio
async def test_production_maintenance_handler_reuses_shared_cadence_loop(monkeypatch) -> None:
    stop = asyncio.Event()
    captured: list[object] = []

    class UsersRepo:
        @classmethod
        async def from_pool(cls):
            return cls()

    class Jobs:
        pass

    async def shared_loop(maintenance, stop_event):
        captured.extend((maintenance, stop_event))
        stop_event.set()

    monkeypatch.setattr(notes_graph_suggestions_maintenance, "AuthnzUsersRepo", UsersRepo)
    monkeypatch.setattr(notes_graph_suggestions_maintenance, "JobManager", Jobs)
    monkeypatch.setattr(notes_graph_suggestions_maintenance, "run_maintenance_loop", shared_loop)

    await notes_graph_suggestions_maintenance.run_notes_graph_suggestions_maintenance(stop)

    assert captured[1] is stop
    assert hasattr(captured[0], "run_pass")
