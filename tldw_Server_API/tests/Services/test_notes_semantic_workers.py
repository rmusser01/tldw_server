"""Worker, recovery cadence, kill-switch, and shutdown contracts."""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.chacha.note_semantic_models import (
    SemanticDimensionState,
    SemanticGenerationState,
    SemanticIndexingError,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Notes_Graph.semantic_jobs import (
    JOB_DOMAIN,
    JOB_QUEUE,
    JOB_TYPE,
    SemanticJobCancelled,
    SemanticJobHandler,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_observability import (
    aggregate_semantic_health_snapshots,
    serialize_semantic_health_snapshots,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_publication import (
    revalidate_execution_fence,
)
from tldw_Server_API.app.core.Notes_Graph.semantic_settings import SemanticIndexSettings
from tldw_Server_API.app.core.Notes_Graph.semantic_vectors import NotesSemanticVectorStore
from tldw_Server_API.app.services import (
    notes_semantic_index_worker,
    notes_semantic_maintenance,
)
from tldw_Server_API.app.services.lifecycle_worker_specs import WorkerLifecycleContext
from tldw_Server_API.app.services.startup_study_privilege_jobs_pollers import (
    provide_study_privilege_jobs_worker_specs,
)

NOW = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)


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


def test_worker_config_is_exact_and_disables_unbounded_sdk_retries() -> None:
    config = notes_semantic_index_worker.build_worker_config(worker_id="semantic-1")

    assert config.domain == JOB_DOMAIN == "notes"
    assert config.queue == JOB_QUEUE
    assert config.retry_on_exception is False
    assert config.bind_completion_token is True


@pytest.mark.asyncio
async def test_store_cancellation_signal_preserves_worker_cancelled_semantics() -> None:
    class CancelledRuntime:
        async def recover(self, **_kwargs):
            return None

        async def execute(self, **_kwargs):
            raise SemanticIndexingError("notes_semantic_run_cancelled")

    handler = SemanticJobHandler(
        runtime_factory=lambda **_kwargs: CancelledRuntime(),
    )
    job = {
        "domain": JOB_DOMAIN,
        "queue": JOB_QUEUE,
        "job_type": JOB_TYPE,
        "owner_user_id": "owner-a",
        "uuid": "00000000-0000-4000-8000-000000000001",
        "payload": {
            "schema_version": 1,
            "dataset_id": "dataset-a",
            "configuration_revision": 1,
            "generation_id": None,
            "mode": "build",
        },
    }

    with pytest.raises(SemanticJobCancelled):
        await handler.handle(job, cancellation_requested=lambda: False)


@pytest.mark.asyncio
async def test_app_managed_and_standalone_entrypoints_share_one_handler(monkeypatch) -> None:
    seen: list[object] = []

    async def fake_runner(*, stop_event, handler):
        seen.append((stop_event, handler))

    monkeypatch.setattr(notes_semantic_index_worker, "_run_worker", fake_runner)
    first = asyncio.Event()
    second = asyncio.Event()

    await notes_semantic_index_worker.run_notes_semantic_index_worker(first)
    await notes_semantic_index_worker.run_standalone_notes_semantic_index_worker(second)

    assert seen == [
        (first, notes_semantic_index_worker.handle_notes_semantic_index_job),
        (second, notes_semantic_index_worker.handle_notes_semantic_index_job),
    ]


def test_startup_flags_cannot_create_duplicate_app_and_sidecar_ownership(monkeypatch) -> None:
    monkeypatch.setenv("NOTES_SEMANTIC_INDEX_WORKER_ENABLED", "true")
    monkeypatch.setenv("NOTES_SEMANTIC_MAINTENANCE_ENABLED", "true")
    specs = {spec.name: spec for spec in provide_study_privilege_jobs_worker_specs()}

    for name in (
        "notes_semantic_index_jobs_task",
        "notes_semantic_maintenance_task",
    ):
        assert specs[name].enabled(_context(sidecar_mode=False)) is True
        assert specs[name].enabled(_context(sidecar_mode=True)) is False


class _Scope:
    def __init__(self, name: str, *, dirty: int, failed: int, cleanup: int) -> None:
        self.name = name
        self.dirty = dirty
        self.failed = failed
        self.cleanup = cleanup
        self.calls: list[tuple[str, int]] = []
        self.admitted: list[tuple[str, object]] = []

    def reclaim_expired(self, *, limit: int, now):
        self.calls.append(("reclaim", limit))
        return min(limit, 2)

    def claim_dirty(self, *, limit: int, now):
        self.calls.append(("dirty", limit))
        return tuple(
            SimpleNamespace(
                owner_user_id="owner-a",
                dataset_id=self.name,
                generation_id="generation-a",
                dirty_generation=index,
            )
            for index in range(min(limit, self.dirty))
        )

    def claim_failed(self, *, limit: int, now):
        self.calls.append(("failed", limit))
        return tuple(f"failed-{index}" for index in range(min(limit, self.failed)))

    def claim_cleanup(self, *, limit: int, now):
        self.calls.append(("cleanup", limit))
        return tuple(f"cleanup-{index}" for index in range(min(limit, self.cleanup)))

    def admit(self, *, mode: str, claim):
        self.admitted.append((mode, claim))
        return True

    async def cleanup_claim(self, claim):
        self.admitted.append(("cleanup", claim))
        return True


@pytest.mark.asyncio
async def test_maintenance_shares_one_bounded_claim_budget_and_coalesces_dirty_work() -> None:
    first = _Scope("dataset-a", dirty=80, failed=20, cleanup=10)
    second = _Scope("dataset-b", dirty=80, failed=20, cleanup=10)
    coordinator = notes_semantic_maintenance.SemanticMaintenanceCoordinator(
        scopes=(first, second),
        indexing_enabled=True,
    )

    result = await coordinator.run_pass(now=NOW, limit=100)

    assert result.claimed <= 100
    assert result.dirty_admitted <= 100
    dirty_keys = {
        (
            claim.owner_user_id,
            claim.dataset_id,
            claim.generation_id,
            claim.dirty_generation,
        )
        for scope in (first, second)
        for mode, claim in scope.admitted
        if mode == "maintain"
    }
    assert len(dirty_keys) == result.dirty_admitted
    assert first.calls[0] == ("reclaim", 100)


@pytest.mark.asyncio
async def test_dataset_maintenance_does_not_publish_global_cleanup_gauges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations: list[dict[str, object]] = []
    monkeypatch.setattr(
        notes_semantic_maintenance,
        "record_semantic_cleanup_metrics",
        lambda **kwargs: observations.append(dict(kwargs)),
        raising=False,
    )
    scope = _Scope("dataset-a", dirty=0, failed=0, cleanup=2)
    coordinator = notes_semantic_maintenance.SemanticMaintenanceCoordinator(
        scopes=(scope,),
        indexing_enabled=False,
    )

    result = await coordinator.run_pass(now=NOW, limit=10)

    assert result.cleanup_confirmed == 2
    assert observations == []


@pytest.mark.asyncio
async def test_generation_cleanup_retry_counter_records_only_committed_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[dict[str, str]] = []

    class Store:
        committed = True

        def get_configuration(self, _dataset_id: str):
            return SimpleNamespace(vector_backend="chromadb")

        def retry_work(self, **_kwargs: object):
            return SimpleNamespace() if self.committed else None

    class Runtime:
        async def cleanup_claim(self, _claim: object) -> bool:
            return False

    store = Store()
    scope = notes_semantic_maintenance._ProductionScope(
        db=SimpleNamespace(note_semantic_store=store),
        jobs=SimpleNamespace(),
        owner_user_id="owner-a",
        dataset_id="dataset-a",
        settings=SemanticIndexSettings(),
    )

    async def runtime(_generation_id: str):
        return Runtime()

    monkeypatch.setattr(scope, "_runtime", runtime)
    monkeypatch.setattr(
        notes_semantic_maintenance,
        "record_semantic_cleanup_retry",
        lambda **kwargs: events.append(dict(kwargs)),
        raising=False,
    )
    claim = SimpleNamespace(
        id="cleanup-a",
        generation_id="generation-a",
        claim_token="claim-a",
        attempt_count=0,
    )

    assert not await scope.cleanup_claim(claim)
    store.committed = False
    assert not await scope.cleanup_claim(claim)

    assert events == [{"status": "failed", "backend": "chromadb"}]


@pytest.mark.parametrize(
    ("total_transitions", "cleanup_transitions", "expected_events"),
    [
        (3, 2, [{"status": "failed", "backend": "pgvector", "count": 2}]),
        (1, 0, []),
        (2, 2, [{"status": "failed", "backend": "pgvector", "count": 2}]),
    ],
    ids=("mixed", "index-only", "cleanup-only"),
)
def test_expired_maintenance_reclaims_emit_only_cleanup_retry_transitions(
    monkeypatch: pytest.MonkeyPatch,
    total_transitions: int,
    cleanup_transitions: int,
    expected_events: list[dict[str, object]],
) -> None:
    events: list[dict[str, object]] = []

    class Store:
        def reclaim_expired_dataset_work(self, **_kwargs: object):
            return SimpleNamespace(
                total_transitions=total_transitions,
                cleanup_transitions=cleanup_transitions,
            )

        def get_configuration(self, _dataset_id: str):
            return SimpleNamespace(vector_backend="pgvector")

    scope = notes_semantic_maintenance._ProductionScope(
        db=SimpleNamespace(note_semantic_store=Store()),
        jobs=SimpleNamespace(),
        owner_user_id="owner-a",
        dataset_id="dataset-a",
        settings=SemanticIndexSettings(),
    )
    monkeypatch.setattr(
        notes_semantic_maintenance,
        "record_semantic_cleanup_retry",
        lambda **kwargs: events.append(dict(kwargs)),
    )

    assert scope.reclaim_expired(limit=10, now=NOW) == total_transitions
    assert events == expected_events


@pytest.mark.asyncio
@pytest.mark.parametrize("churn", ("delete", "insert"))
async def test_health_sweep_owner_keyset_survives_churn_without_partial_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    churn: str,
) -> None:
    def owner(owner_id: int, created_at: datetime | None) -> dict[str, object]:
        return {"id": owner_id, "created_at": created_at}

    records = [
        owner(3, NOW),
        owner(2, NOW - timedelta(minutes=1)),
        owner(1, NOW - timedelta(minutes=2)),
    ]

    class Users:
        async def list_users(self, *, offset: int, limit: int):
            ordered = sorted(records, key=lambda row: row["id"], reverse=True)
            return ordered[offset : offset + limit], len(ordered)

        async def list_users_for_semantic_health_sweep(
            self,
            *,
            after_id: int | None,
            limit: int,
        ):
            ordered = sorted(records, key=lambda row: row["id"], reverse=True)
            return [row for row in ordered if after_id is None or row["id"] < after_id][:limit]

        async def get_user_by_id(self, user_id: int):
            return next((row for row in records if row["id"] == user_id), None)

    class Store:
        def __init__(self, owner_id: str) -> None:
            self.owner_id = owner_id

        def list_observability_dataset_ids(self, *, limit: int, after_dataset_id=None):
            del after_dataset_id
            return (f"dataset-{self.owner_id}",)[:limit]

        def get_observability_snapshot(self, _dataset_id: str, *, current_capability_revision: str):
            assert current_capability_revision == "capability-current"
            return SimpleNamespace(
                backend="chromadb",
                indexed_notes=int(self.owner_id in {"1", "3"}),
                excluded_notes=0,
                failed_notes=int(self.owner_id == "2"),
                dirty_notes=0,
                pending_notes=0,
                stale_generations=0,
                cleanup_backlog=0,
                cleanup_retries=0,
                oldest_cleanup_created_at=None,
            )

    async def open_database(owner_id: str):
        return SimpleNamespace(
            note_semantic_store=Store(owner_id),
            release_context_connection=lambda: None,
        )

    observations: list[tuple[object, ...]] = []
    monkeypatch.setattr(notes_semantic_maintenance, "_open_owner_database", open_database)
    monkeypatch.setattr(
        notes_semantic_maintenance,
        "resolve_semantic_capabilities",
        lambda _db, *, settings: SimpleNamespace(capability_revision="capability-current"),
    )
    monkeypatch.setattr(
        notes_semantic_maintenance,
        "record_semantic_aggregate_metrics",
        lambda *, snapshots, now: observations.append(tuple(snapshots)),
    )
    jobs = JobManager(tmp_path / f"semantic-health-{churn}.db")

    await notes_semantic_maintenance._MaintenanceRunner(
        jobs=jobs,
        users_repo=Users(),
        settings=SemanticIndexSettings(),
    )._run_health_sweep_page(now=NOW, limit=2)
    assert observations == []
    if churn == "delete":
        records[:] = [row for row in records if row["id"] != 3]
    else:
        records.append(owner(4, NOW + timedelta(minutes=1)))
    for record in records:
        record["created_at"] = None

    for _ in range(2):
        await notes_semantic_maintenance._MaintenanceRunner(
            jobs=JobManager(jobs.db_path),
            users_repo=Users(),
            settings=SemanticIndexSettings(),
        )._run_health_sweep_page(now=NOW, limit=2)
        assert observations == []
    await notes_semantic_maintenance._MaintenanceRunner(
        jobs=JobManager(jobs.db_path),
        users_repo=Users(),
        settings=SemanticIndexSettings(),
    )._run_health_sweep_page(now=NOW, limit=2)

    assert len(observations) == 1
    chromadb = next(value for value in observations[0] if value.backend == "chromadb")
    assert (chromadb.indexed_notes, chromadb.failed_notes) == (2, 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "corruption",
    (
        "totals",
        "totals-without-owner",
        "owner-without-totals",
        "owner-id",
        "dataset-id",
    ),
)
async def test_corrupt_health_checkpoint_resets_without_partial_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    corruption: str,
) -> None:
    observations: list[tuple[object, ...]] = []

    class Users:
        async def list_users_for_semantic_health_sweep(self, **_kwargs: object):
            return []

    jobs = JobManager(tmp_path / "semantic-health-corrupt.db")
    valid_totals = serialize_semantic_health_snapshots(aggregate_semantic_health_snapshots(()))
    with sqlite3.connect(jobs.db_path) as conn:
        conn.execute("PRAGMA ignore_check_constraints=ON")
        if corruption == "totals":
            conn.execute(
                "UPDATE notes_semantic_health_sweep SET totals_json=? WHERE singleton_id=1",
                ('{"backend":"chromadb"}',),
            )
        elif corruption == "totals-without-owner":
            conn.execute(
                "UPDATE notes_semantic_health_sweep SET totals_json=? WHERE singleton_id=1",
                (valid_totals,),
            )
        elif corruption == "owner-without-totals":
            conn.execute("UPDATE notes_semantic_health_sweep SET after_owner_id=1 WHERE singleton_id=1")
        elif corruption == "owner-id":
            conn.execute(
                "UPDATE notes_semantic_health_sweep SET after_owner_id=?,totals_json=? WHERE singleton_id=1",
                ("not-an-id", valid_totals),
            )
        else:
            conn.execute(
                "UPDATE notes_semantic_health_sweep SET after_owner_id=1,"
                "after_dataset_id='',totals_json=? WHERE singleton_id=1",
                (valid_totals,),
            )
    monkeypatch.setattr(
        notes_semantic_maintenance,
        "record_semantic_aggregate_metrics",
        lambda *, snapshots, now: observations.append(tuple(snapshots)),
    )

    await notes_semantic_maintenance._MaintenanceRunner(
        jobs=jobs,
        users_repo=Users(),
        settings=SemanticIndexSettings(),
    )._run_health_sweep_page(now=NOW, limit=2)

    assert observations == []
    reset = JobManager(jobs.db_path).get_notes_semantic_health_sweep()
    assert (reset["revision"], reset["totals_json"]) == (1, "[]")
    await notes_semantic_maintenance._MaintenanceRunner(
        jobs=JobManager(jobs.db_path),
        users_repo=Users(),
        settings=SemanticIndexSettings(),
    )._run_health_sweep_page(now=NOW, limit=2)
    assert len(observations) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("after_owner_id", "after_dataset_id", "storage_class"),
    (
        (2.5, None, "real"),
        ("2_0", None, "text"),
        ("true", None, "text"),
        (sqlite3.Binary(b"2"), None, "blob"),
        (0, None, "integer"),
        (-1, None, "integer"),
        (1, "   ", "integer"),
        (1, " dataset-a", "integer"),
        (1, "dataset-a ", "integer"),
    ),
    ids=(
        "owner-real",
        "owner-numeric-text",
        "owner-boolean-like",
        "owner-blob",
        "owner-zero",
        "owner-negative",
        "dataset-whitespace",
        "dataset-leading-space",
        "dataset-trailing-space",
    ),
)
async def test_raw_corrupt_health_cursor_resets_before_owner_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    after_owner_id: object,
    after_dataset_id: str | None,
    storage_class: str,
) -> None:
    observations: list[tuple[object, ...]] = []
    owner_accesses: list[str] = []

    class Users:
        async def list_users_for_semantic_health_sweep(self, **_kwargs: object):
            owner_accesses.append("list")
            return []

        async def get_user_by_id(self, _user_id: int):
            owner_accesses.append("get")
            return {"id": 1}

    class Store:
        def list_observability_dataset_ids(self, **_kwargs: object):
            owner_accesses.append("dataset")
            return ()

    async def open_database(_owner: str):
        owner_accesses.append("open")
        return SimpleNamespace(
            note_semantic_store=Store(),
            release_context_connection=lambda: None,
        )

    jobs = JobManager(tmp_path / "semantic-health-raw-corrupt.db")
    valid_totals = serialize_semantic_health_snapshots(aggregate_semantic_health_snapshots(()))
    with sqlite3.connect(jobs.db_path) as conn:
        conn.execute("PRAGMA ignore_check_constraints=ON")
        conn.execute(
            "UPDATE notes_semantic_health_sweep SET after_owner_id=?,"
            "after_dataset_id=?,totals_json=? WHERE singleton_id=1",
            (after_owner_id, after_dataset_id, valid_totals),
        )
        assert conn.execute(
            "SELECT typeof(after_owner_id) FROM notes_semantic_health_sweep WHERE singleton_id=1"
        ).fetchone() == (storage_class,)

    monkeypatch.setattr(notes_semantic_maintenance, "_open_owner_database", open_database)
    monkeypatch.setattr(
        notes_semantic_maintenance,
        "record_semantic_aggregate_metrics",
        lambda *, snapshots, now: observations.append(tuple(snapshots)),
    )

    await notes_semantic_maintenance._MaintenanceRunner(
        jobs=jobs,
        users_repo=Users(),
        settings=SemanticIndexSettings(),
    )._run_health_sweep_page(now=NOW, limit=2)

    assert owner_accesses == []
    assert observations == []
    reset = JobManager(jobs.db_path).get_notes_semantic_health_sweep()
    assert (reset["revision"], reset["after_owner_id"], reset["after_dataset_id"], reset["totals_json"]) == (
        1,
        None,
        None,
        "[]",
    )

    await notes_semantic_maintenance._MaintenanceRunner(
        jobs=JobManager(jobs.db_path),
        users_repo=Users(),
        settings=SemanticIndexSettings(),
    )._run_health_sweep_page(now=NOW, limit=2)
    assert owner_accesses == ["list"]
    assert len(observations) == 1


@pytest.mark.asyncio
async def test_stale_health_sweep_cas_loser_cannot_publish(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    observations: list[tuple[object, ...]] = []

    class Users:
        async def list_users_for_semantic_health_sweep(self, **_kwargs: object):
            return []

    authority = JobManager(tmp_path / "semantic-health-cas.db")
    stale_state = authority.get_notes_semantic_health_sweep()

    class Jobs:
        def get_notes_semantic_health_sweep(self):
            return dict(stale_state)

        def checkpoint_notes_semantic_health_sweep(self, **kwargs: object):
            return authority.checkpoint_notes_semantic_health_sweep(**kwargs)

    monkeypatch.setattr(
        notes_semantic_maintenance,
        "record_semantic_aggregate_metrics",
        lambda *, snapshots, now: observations.append(tuple(snapshots)),
    )
    runners = [
        notes_semantic_maintenance._MaintenanceRunner(
            jobs=Jobs(),
            users_repo=Users(),
            settings=SemanticIndexSettings(),
        )
        for _ in range(2)
    ]

    await asyncio.gather(*(runner._run_health_sweep_page(now=NOW, limit=2) for runner in runners))

    assert len(observations) == 1


@pytest.mark.asyncio
async def test_worker_records_terminal_metrics_and_publication_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []

    class Handler:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def handle(self, _job: dict[str, object], **_kwargs: object):
            return {
                "state": "completed",
                "indexed_notes": 4,
                "excluded_notes": 1,
                "failed_notes": 1,
                "published_chunks": 8,
                "cleanup_complete": True,
                "error_code": None,
            }

    db = SimpleNamespace(
        note_semantic_store=SimpleNamespace(
            get_configuration=lambda _dataset_id: SimpleNamespace(vector_backend="chromadb")
        )
    )
    monkeypatch.setattr(notes_semantic_index_worker, "SemanticJobHandler", Handler)

    async def open_database(_owner: str):
        return db

    monkeypatch.setattr(
        notes_semantic_index_worker,
        "_open_owner_database",
        open_database,
    )
    monkeypatch.setattr(
        notes_semantic_index_worker,
        "_close_database",
        lambda _db: None,
    )
    monkeypatch.setattr(
        notes_semantic_index_worker,
        "record_semantic_build_metrics",
        lambda **kwargs: metrics.append(dict(kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        notes_semantic_index_worker,
        "record_semantic_failure",
        lambda **kwargs: failures.append(dict(kwargs)),
        raising=False,
    )

    async def capture_audit(**kwargs: object) -> None:
        audits.append(dict(kwargs))

    monkeypatch.setattr(
        notes_semantic_index_worker,
        "emit_semantic_audit_event",
        capture_audit,
        raising=False,
    )
    job = {
        "owner_user_id": "owner-a",
        "uuid": "run-a",
        "payload": {
            "dataset_id": "dataset-a",
            "mode": "rebuild",
        },
    }

    result = await notes_semantic_index_worker.handle_notes_semantic_index_job(
        job,
        jobs=SimpleNamespace(),
        worker_id="worker-a",
    )

    assert result["state"] == "completed"
    assert metrics[0]["operation"] == "rebuild"
    assert metrics[0]["status"] == "degraded"
    assert metrics[0]["counts"]["indexed"] == 4
    assert failures == [
        {
            "component": "provider",
            "category": "execution",
            "backend": "chromadb",
        }
    ]
    assert audits[0]["event"] == "generation_publication"
    assert audits[0]["run_id"] == "run-a"


@pytest.mark.asyncio
async def test_worker_records_failed_build_terminal_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics: list[dict[str, object]] = []

    class Handler:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def handle(self, _job: dict[str, object], **_kwargs: object):
            raise SemanticIndexingError("notes_semantic_provider_unavailable")

    db = SimpleNamespace(
        note_semantic_store=SimpleNamespace(
            get_configuration=lambda _dataset_id: SimpleNamespace(vector_backend="chromadb")
        )
    )

    async def open_database(_owner: str):
        return db

    monkeypatch.setattr(notes_semantic_index_worker, "SemanticJobHandler", Handler)
    monkeypatch.setattr(notes_semantic_index_worker, "_open_owner_database", open_database)
    monkeypatch.setattr(notes_semantic_index_worker, "_close_database", lambda _db: None)
    monkeypatch.setattr(
        notes_semantic_index_worker,
        "record_semantic_build_metrics",
        lambda **kwargs: metrics.append(dict(kwargs)),
    )

    with pytest.raises(SemanticIndexingError):
        await notes_semantic_index_worker.handle_notes_semantic_index_job(
            {
                "owner_user_id": "owner-a",
                "uuid": "run-a",
                "payload": {"dataset_id": "dataset-a", "mode": "build"},
            },
            jobs=SimpleNamespace(),
            worker_id="worker-a",
        )

    assert len(metrics) == 1
    assert metrics[0]["operation"] == "build"
    assert metrics[0]["status"] == "failed"
    assert metrics[0]["backend"] == "chromadb"


@pytest.mark.asyncio
async def test_worker_does_not_audit_incomplete_cleanup_as_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_metrics: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []

    class Handler:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def handle(self, _job: dict[str, object], **_kwargs: object):
            return {
                "state": "completed",
                "indexed_notes": 0,
                "excluded_notes": 0,
                "failed_notes": 0,
                "published_chunks": 0,
                "cleanup_complete": False,
                "error_code": "notes_semantic_cleanup_unconfirmed",
            }

    db = SimpleNamespace(
        note_semantic_store=SimpleNamespace(
            get_configuration=lambda _dataset_id: SimpleNamespace(vector_backend="chromadb")
        )
    )

    async def open_database(_owner: str):
        return db

    async def capture_audit(**kwargs: object) -> None:
        audits.append(dict(kwargs))

    monkeypatch.setattr(notes_semantic_index_worker, "SemanticJobHandler", Handler)
    monkeypatch.setattr(notes_semantic_index_worker, "_open_owner_database", open_database)
    monkeypatch.setattr(notes_semantic_index_worker, "_close_database", lambda _db: None)
    monkeypatch.setattr(notes_semantic_index_worker, "record_semantic_build_metrics", lambda **_kwargs: None)
    monkeypatch.setattr(
        notes_semantic_index_worker,
        "record_semantic_cleanup_metrics",
        lambda **kwargs: cleanup_metrics.append(dict(kwargs)),
        raising=False,
    )
    monkeypatch.setattr(notes_semantic_index_worker, "emit_semantic_audit_event", capture_audit)

    result = await notes_semantic_index_worker.handle_notes_semantic_index_job(
        {
            "owner_user_id": "owner-a",
            "uuid": "run-a",
            "payload": {"dataset_id": "dataset-a", "mode": "delete"},
        },
        jobs=SimpleNamespace(),
        worker_id="worker-a",
    )

    assert result["cleanup_complete"] is False
    assert cleanup_metrics == []
    assert audits == []


@pytest.mark.asyncio
async def test_runtime_records_kill_switch_denial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    denials: list[str] = []
    monkeypatch.setattr(
        notes_semantic_index_worker,
        "record_semantic_denial",
        denials.append,
        raising=False,
    )
    runtime = notes_semantic_index_worker.ProductionSemanticRuntime(
        db=SimpleNamespace(note_semantic_store=SimpleNamespace()),
        owner_user_id="owner-a",
        dataset_id="dataset-a",
        configuration_revision=1,
        generation_id="generation-a",
        root_job_id="run-a",
        settings=SemanticIndexSettings(indexing_enabled=False),
    )

    with pytest.raises(SemanticIndexingError) as exc_info:
        await runtime.execute(
            mode="build",
            cancellation_requested=lambda: False,
        )

    assert exc_info.value.failure_code == "notes_semantic_indexing_disabled"
    assert denials == ["kill_switch"]


@pytest.mark.asyncio
async def test_failed_notes_retry_separately_and_cleanup_requires_confirmation() -> None:
    scope = _Scope("dataset-a", dirty=0, failed=2, cleanup=2)
    coordinator = notes_semantic_maintenance.SemanticMaintenanceCoordinator(
        scopes=(scope,),
        indexing_enabled=True,
    )

    result = await coordinator.run_pass(now=NOW, limit=10)

    assert result.failed_retries == 2
    assert result.cleanup_confirmed == 2
    assert [mode for mode, _claim in scope.admitted].count("retry_failed") == 2
    assert [mode for mode, _claim in scope.admitted].count("cleanup") == 2


@pytest.mark.asyncio
async def test_kill_switch_blocks_index_admission_but_keeps_cleanup_available() -> None:
    scope = _Scope("dataset-a", dirty=5, failed=3, cleanup=2)
    coordinator = notes_semantic_maintenance.SemanticMaintenanceCoordinator(
        scopes=(scope,),
        indexing_enabled=False,
    )

    result = await coordinator.run_pass(now=NOW, limit=20)

    assert result.dirty_admitted == 0
    assert result.failed_retries == 0
    assert result.cleanup_confirmed == 2
    assert all(mode == "cleanup" for mode, _claim in scope.admitted)
    assert not any(name in {"dirty", "failed"} for name, _limit in scope.calls)


@pytest.mark.asyncio
async def test_unresolved_generation_cleanup_confirms_no_storage_without_backend_io() -> None:
    class Authority:
        owner_user_id = "owner-a"

        def get_generation(self, dataset_id, generation_id):
            return SimpleNamespace(
                owner_user_id="owner-a",
                dataset_id=dataset_id,
                id=generation_id,
                dimension_state="pending",
                dimensions=None,
            )

    class Backend:
        def supports_dimensions(self, _dimensions):
            raise AssertionError("unresolved cleanup must not inspect backend dimensions")

        async def delete_generation(self, _binding):
            raise AssertionError("unresolved generation never created physical storage")

    vectors = NotesSemanticVectorStore(authority=Authority(), backend=Backend())

    result = await vectors.delete_generation("dataset-a", "generation-a")

    assert result.confirmed_absent is True


@pytest.mark.asyncio
async def test_maintenance_loop_drains_cleanly_on_shutdown() -> None:
    stop = asyncio.Event()
    calls: list[str] = []

    class Runner:
        async def run_pass(self, *, now, limit):
            del now, limit
            calls.append("pass")
            stop.set()

    await notes_semantic_maintenance.run_maintenance_loop(
        Runner(),
        stop,
        interval_seconds=60,
        now=lambda: NOW,
    )

    assert calls == ["pass"]


@pytest.mark.asyncio
async def test_worker_stop_event_stops_sdk_and_awaits_watcher(monkeypatch) -> None:
    calls: list[str] = []

    class SDK:
        def stop(self):
            calls.append("stop")

        async def run(self, **kwargs):
            assert kwargs["job_type"] == JOB_TYPE
            stop.set()
            await asyncio.sleep(0)

    stop = asyncio.Event()
    monkeypatch.setattr(notes_semantic_index_worker, "_build_sdk", lambda **_kwargs: SDK())
    await notes_semantic_index_worker._run_worker(
        stop_event=stop,
        handler=notes_semantic_index_worker.handle_notes_semantic_index_job,
    )

    assert calls == ["stop"]


def _enabled_configuration(db: CharactersRAGDB):
    created = db.note_semantic_store.create_configuration(
        dataset_id="dataset-a",
        capability_revision="capability-v1",
        disclosure_hash="disclosure-v1",
        provider="provider-a",
        model="model-a",
        model_revision=None,
        endpoint_origin_revision="origin-v1",
        endpoint_origin_display="https://api.example.test",
        data_boundary="external",
        vector_backend="chromadb",
        storage_boundary="local",
        storage_label="local-vectors",
        normalization_version="normalization-v1",
        chunker_version="chunker-v1",
        now=NOW,
    )
    enabled = db.note_semantic_store.enable_configuration(
        dataset_id="dataset-a",
        expected_configuration_revision=created.configuration_revision,
        capability_revision="capability-v1",
        now=NOW,
    )
    assert enabled is not None
    return enabled


@pytest.mark.asyncio
async def test_root_job_recovers_exact_dimension_transition_and_rejects_drift(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = CharactersRAGDB(str(tmp_path / "dimension-recovery.sqlite"), client_id="owner-a")
    root_job_id = "6ec1dfbe-f86f-4d2b-93af-f88f64cd9701"
    try:
        enabled = _enabled_configuration(db)
        generation = db.note_semantic_store.create_generation(
            dataset_id="dataset-a",
            configuration_revision=enabled.configuration_revision,
            compatibility_hash=None,
            dimension_state=SemanticDimensionState.PENDING,
            dimensions=None,
            root_job_id=root_job_id,
            model_revision=None,
            now=NOW,
        )
        resolved = db.note_semantic_store.resolve_generation_dimensions(
            dataset_id="dataset-a",
            generation_id=generation.id,
            expected_configuration_revision=enabled.configuration_revision,
            dimensions=1536,
            compatibility_hash="compatibility-v1",
            model_revision=None,
            now=NOW,
        )
        assert resolved is not None

        async def vectors(**_kwargs):
            return object()

        monkeypatch.setattr(notes_semantic_index_worker, "_build_vector_store", vectors)
        runtime = await notes_semantic_index_worker.build_production_runtime(
            db=db,
            settings=SemanticIndexSettings(),
            owner_user_id="owner-a",
            dataset_id="dataset-a",
            configuration_revision=enabled.configuration_revision,
            generation_id=None,
            root_job_id=root_job_id,
            mode="build",
        )

        assert runtime._fence().configuration_revision == resolved.configuration_revision

        disabled = db.note_semantic_store.disable_configuration(
            dataset_id="dataset-a",
            expected_configuration_revision=resolved.configuration_revision,
            now=NOW + timedelta(seconds=1),
        )
        assert disabled is not None
        with pytest.raises(SemanticIndexingError) as drift:
            await notes_semantic_index_worker.build_production_runtime(
                db=db,
                settings=SemanticIndexSettings(),
                owner_user_id="owner-a",
                dataset_id="dataset-a",
                configuration_revision=enabled.configuration_revision,
                generation_id=None,
                root_job_id=root_job_id,
                mode="build",
            )
        assert drift.value.code == "notes_semantic_configuration_drift"
    finally:
        db.close_all_connections()


@pytest.mark.asyncio
async def test_discovered_model_revision_survives_production_revalidation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = CharactersRAGDB(
        str(tmp_path / "model-revision-revalidation.sqlite"),
        client_id="1",
    )
    root_job_id = "6ec1dfbe-f86f-4d2b-93af-f88f64cd9701"
    try:
        enabled = _enabled_configuration(db)
        generation = db.note_semantic_store.create_generation(
            dataset_id="dataset-a",
            configuration_revision=enabled.configuration_revision,
            compatibility_hash=None,
            dimension_state=SemanticDimensionState.PENDING,
            dimensions=None,
            root_job_id=root_job_id,
            model_revision=None,
            now=NOW,
        )
        resolved = db.note_semantic_store.resolve_generation_dimensions(
            dataset_id="dataset-a",
            generation_id=generation.id,
            expected_configuration_revision=enabled.configuration_revision,
            dimensions=1536,
            compatibility_hash="compatibility-v1",
            model_revision="model-digest-a",
            now=NOW,
        )
        assert resolved is not None

        async def vectors(**_kwargs):
            return object()

        class ActiveUsers:
            @classmethod
            async def from_pool(cls):
                return cls()

            async def get_user_by_id(self, _user_id):
                return {"id": 1, "is_active": True}

        current = SimpleNamespace(
            capability_revision="capability-v1",
            disclosure_hash="disclosure-v1",
            provider_label="provider-a",
            model="model-a",
            model_revision=None,
            endpoint_display="https://api.example.test",
            endpoint_origin_revision="origin-v1",
            indexing_available=True,
            vector_backend="chromadb",
        )
        monkeypatch.setattr(notes_semantic_index_worker, "_build_vector_store", vectors)
        monkeypatch.setattr(notes_semantic_index_worker, "AuthnzUsersRepo", ActiveUsers)
        monkeypatch.setattr(
            notes_semantic_index_worker,
            "resolve_semantic_capabilities",
            lambda *_args, **_kwargs: current,
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.AuthNZ.rbac.user_has_permission",
            lambda *_args, **_kwargs: True,
        )
        runtime = await notes_semantic_index_worker.build_production_runtime(
            db=db,
            settings=SemanticIndexSettings(),
            owner_user_id="1",
            dataset_id="dataset-a",
            configuration_revision=enabled.configuration_revision,
            generation_id=None,
            root_job_id=root_job_id,
            mode="build",
        )

        first = await runtime._revalidate(runtime._fence())
        later = await runtime._revalidate(runtime._fence())

        assert first.model_revision == "model-digest-a"
        assert later.model_revision == "model-digest-a"
        assert db.note_semantic_store.get_configuration("dataset-a").model_revision == ("model-digest-a")
    finally:
        db.close_all_connections()


@pytest.mark.asyncio
async def test_missing_root_generation_rejects_unrelated_revision_before_creation() -> None:
    create_calls = 0
    config = SimpleNamespace(
        configuration_revision=9,
        desired_state=SimpleNamespace(value="enabled"),
        compatibility_hash="compatibility-v1",
        model_revision="model-digest-a",
        dimensions=1536,
        dimension_state=SemanticDimensionState.RESOLVED,
        vector_backend="chromadb",
    )

    class Store:
        def get_configuration(self, _dataset_id):
            return config

        def get_generation_by_root_job_id(self, _dataset_id, _root_job_id):
            return None

        def create_generation(self, **_kwargs):
            nonlocal create_calls
            create_calls += 1
            return SimpleNamespace(
                id="generation-a",
                configuration_revision=9,
                root_job_id="6ec1dfbe-f86f-4d2b-93af-f88f64cd9701",
                dimension_state=SemanticDimensionState.RESOLVED,
                dimensions=1536,
                compatibility_hash="compatibility-v1",
                model_revision="model-digest-a",
            )

    db = SimpleNamespace(note_semantic_store=Store())
    with pytest.raises(SemanticIndexingError) as drift:
        await notes_semantic_index_worker.build_production_runtime(
            db=db,
            settings=SemanticIndexSettings(),
            owner_user_id="owner-a",
            dataset_id="dataset-a",
            configuration_revision=8,
            generation_id=None,
            root_job_id="6ec1dfbe-f86f-4d2b-93af-f88f64cd9701",
            mode="build",
        )

    assert drift.value.code == "notes_semantic_configuration_drift"
    assert create_calls == 0


@pytest.mark.asyncio
async def test_committed_activation_recovers_before_unavailable_vector_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_job_id = "6ec1dfbe-f86f-4d2b-93af-f88f64cd9701"
    config = SimpleNamespace(
        configuration_revision=8,
        desired_state=SimpleNamespace(value="enabled"),
        capability_revision="capability-v1",
        disclosure_hash="disclosure-v1",
        provider="provider-a",
        model="model-a",
        model_revision="model-digest-a",
        endpoint_origin_display="https://api.example.test",
        endpoint_origin_revision="origin-v1",
        compatibility_hash="compatibility-v1",
        dimensions=1536,
        dimension_state=SemanticDimensionState.RESOLVED,
        vector_backend="chromadb",
    )
    generation = SimpleNamespace(
        id="generation-a",
        owner_user_id="owner-a",
        dataset_id="dataset-a",
        configuration_revision=8,
        state=SemanticGenerationState.ACTIVE,
        root_job_id=root_job_id,
        model_revision="model-digest-a",
        dimension_state=SemanticDimensionState.RESOLVED,
        dimensions=1536,
        compatibility_hash="compatibility-v1",
    )
    integrity = SimpleNamespace(
        indexed_note_count=3,
        excluded_note_count=0,
        failed_note_count=0,
        published_chunk_count=6,
        terminal_error_code=None,
    )

    class Store:
        owner_user_id = "owner-a"

        def get_configuration(self, _dataset_id):
            return config

        def get_generation_by_root_job_id(self, _dataset_id, _root_job_id):
            return generation

        def get_generation(self, _dataset_id, _generation_id):
            return generation

        def get_generation_integrity(self, _dataset_id, _generation_id):
            return integrity

        def has_pending_cleanup(self, _dataset_id):
            return False

        def has_pending_index_work(self, _dataset_id, _generation_id):
            return False

    db = SimpleNamespace(note_semantic_store=Store(), note_store=object())

    async def unavailable_vectors(**_kwargs):
        raise RuntimeError("vector DSN postgresql://secret@host/db?token=private")

    monkeypatch.setattr(
        notes_semantic_index_worker,
        "_build_vector_store",
        unavailable_vectors,
    )
    runtime = await notes_semantic_index_worker.build_production_runtime(
        db=db,
        settings=SemanticIndexSettings(),
        owner_user_id="owner-a",
        dataset_id="dataset-a",
        configuration_revision=7,
        generation_id=None,
        root_job_id=root_job_id,
        mode="build",
    )

    result = await runtime.recover(mode="build")

    assert result == {
        "state": "completed",
        "indexed_notes": 3,
        "excluded_notes": 0,
        "failed_notes": 0,
        "published_chunks": 6,
        "cleanup_complete": True,
        "error_code": None,
    }


@pytest.mark.asyncio
async def test_active_cancellation_callback_is_forwarded_into_task6_orchestration() -> None:
    checks = 0
    provider_called = False
    root_job_id = "6ec1dfbe-f86f-4d2b-93af-f88f64cd9701"
    config = SimpleNamespace(
        configuration_revision=8,
        desired_state=SimpleNamespace(value="enabled"),
        capability_revision="capability-v1",
        disclosure_hash="disclosure-v1",
        provider="provider-a",
        model="model-a",
        model_revision="model-digest-a",
        endpoint_origin_display="https://api.example.test",
        endpoint_origin_revision="origin-v1",
        compatibility_hash="compatibility-v1",
        dimensions=1536,
        vector_backend="chromadb",
    )
    generation = SimpleNamespace(
        id="generation-a",
        owner_user_id="owner-a",
        configuration_revision=8,
        state=SemanticGenerationState.STAGING,
        root_job_id=root_job_id,
        model_revision="model-digest-a",
    )

    class Store:
        owner_user_id = "owner-a"

        def get_configuration(self, _dataset_id):
            return config

        def get_generation(self, _dataset_id, _generation_id):
            return generation

        def get_generation_by_root_job_id(self, _dataset_id, _root_job_id):
            return generation

    class Builder:
        async def build_initial_generation(self, _request, *, before_side_effect):
            nonlocal provider_called
            await before_side_effect()
            provider_called = True
            raise AssertionError("provider must be fenced by active cancellation")

    runtime = notes_semantic_index_worker.ProductionSemanticRuntime(
        db=SimpleNamespace(note_semantic_store=Store(), note_store=object()),
        owner_user_id="owner-a",
        dataset_id="dataset-a",
        configuration_revision=8,
        generation_id="generation-a",
        root_job_id=root_job_id,
        vectors=object(),
        settings=SemanticIndexSettings(),
    )
    runtime._builder = Builder()

    async def cancellation_requested() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 2

    with pytest.raises(SemanticJobCancelled):
        await runtime.execute(
            mode="build",
            cancellation_requested=cancellation_requested,
        )

    assert checks == 2
    assert provider_called is False


@pytest.mark.asyncio
async def test_revalidation_requires_authoritative_active_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_job_id = "6ec1dfbe-f86f-4d2b-93af-f88f64cd9701"
    config = SimpleNamespace(
        configuration_revision=8,
        desired_state=SimpleNamespace(value="enabled"),
        capability_revision="capability-v1",
        disclosure_hash="disclosure-v1",
        provider="provider-a",
        model="model-a",
        model_revision="model-digest-a",
        endpoint_origin_display="https://api.example.test",
        endpoint_origin_revision="origin-v1",
        compatibility_hash="compatibility-v1",
        dimensions=1536,
        vector_backend="chromadb",
    )
    generation = SimpleNamespace(
        id="generation-a",
        owner_user_id="1",
        configuration_revision=8,
        state=SemanticGenerationState.STAGING,
        root_job_id=root_job_id,
        model_revision="model-digest-a",
    )

    class Store:
        owner_user_id = "1"

        def get_configuration(self, _dataset_id):
            return config

        def get_generation(self, _dataset_id, _generation_id):
            return generation

        def get_generation_by_root_job_id(self, _dataset_id, _root_job_id):
            return generation

    class MissingUsers:
        @classmethod
        async def from_pool(cls):
            return cls()

        async def get_user_by_id(self, _user_id):
            return None

    current = SimpleNamespace(
        capability_revision="capability-v1",
        disclosure_hash="disclosure-v1",
        provider_label="provider-a",
        model="model-a",
        model_revision="model-digest-a",
        endpoint_display="https://api.example.test",
        endpoint_origin_revision="origin-v1",
        indexing_available=True,
        vector_backend="chromadb",
    )
    capability_calls = 0

    def resolve_capabilities(*_args, **_kwargs):
        nonlocal capability_calls
        capability_calls += 1
        return current

    monkeypatch.setattr(
        notes_semantic_index_worker,
        "AuthnzUsersRepo",
        MissingUsers,
        raising=False,
    )
    monkeypatch.setattr(
        notes_semantic_index_worker,
        "resolve_semantic_capabilities",
        resolve_capabilities,
    )
    runtime = notes_semantic_index_worker.ProductionSemanticRuntime(
        db=SimpleNamespace(note_semantic_store=Store(), note_store=object()),
        owner_user_id="1",
        dataset_id="dataset-a",
        configuration_revision=8,
        generation_id="generation-a",
        root_job_id=root_job_id,
        vectors=object(),
        settings=SemanticIndexSettings(),
    )

    authority = await runtime._revalidate(runtime._fence())

    assert authority.user_exists is False
    assert capability_calls == 0


@pytest.mark.asyncio
async def test_revalidation_projects_live_model_revision_and_rejects_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_job_id = "6ec1dfbe-f86f-4d2b-93af-f88f64cd9701"
    config = SimpleNamespace(
        configuration_revision=8,
        desired_state=SimpleNamespace(value="enabled"),
        capability_revision="capability-v1",
        disclosure_hash="disclosure-v1",
        provider="provider-a",
        model="model-a",
        model_revision="model-digest-a",
        endpoint_origin_display="https://api.example.test",
        endpoint_origin_revision="origin-v1",
        compatibility_hash="compatibility-v1",
        dimensions=1536,
        vector_backend="chromadb",
    )
    generation = SimpleNamespace(  # nosec B106
        id="generation-a",
        owner_user_id="1",
        configuration_revision=8,
        state=SemanticGenerationState.STAGING,
        root_job_id=root_job_id,
        model_revision="model-digest-a",
        fencing_token="generation-fence-a",
    )

    class Store:
        owner_user_id = "1"

        def get_configuration(self, _dataset_id):
            return config

        def get_generation(self, _dataset_id, _generation_id):
            return generation

    class ActiveUsers:
        @classmethod
        async def from_pool(cls):
            return cls()

        async def get_user_by_id(self, _user_id):
            return {"id": 1, "is_active": True}

    current = SimpleNamespace(
        capability_revision="capability-v1",
        disclosure_hash="disclosure-v1",
        provider_label="provider-a",
        model="model-a",
        model_revision="model-digest-b",
        endpoint_display="https://api.example.test",
        endpoint_origin_revision="origin-v1",
        indexing_available=True,
        vector_backend="chromadb",
    )
    monkeypatch.setattr(notes_semantic_index_worker, "AuthnzUsersRepo", ActiveUsers)
    monkeypatch.setattr(
        notes_semantic_index_worker,
        "resolve_semantic_capabilities",
        lambda *_args, **_kwargs: current,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.rbac.user_has_permission",
        lambda *_args, **_kwargs: True,
    )
    runtime = notes_semantic_index_worker.ProductionSemanticRuntime(
        db=SimpleNamespace(note_semantic_store=Store(), note_store=object()),
        owner_user_id="1",
        dataset_id="dataset-a",
        configuration_revision=8,
        generation_id="generation-a",
        root_job_id=root_job_id,
        vectors=object(),
        settings=SemanticIndexSettings(),
    )

    authority = await runtime._revalidate(runtime._fence())

    assert authority.model_revision == "model-digest-b"
    with pytest.raises(SemanticIndexingError) as exc_info:
        await revalidate_execution_fence(runtime._revalidate, runtime._fence())
    assert exc_info.value.failure_code == "notes_semantic_model_revision_drift"


@pytest.mark.asyncio
async def test_revalidation_projects_pinned_durable_model_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_job_id = "6ec1dfbe-f86f-4d2b-93af-f88f64cd9701"
    config = SimpleNamespace(
        configuration_revision=8,
        desired_state=SimpleNamespace(value="enabled"),
        capability_revision="capability-v1",
        disclosure_hash="disclosure-v1",
        provider="provider-a",
        model="model-a",
        model_revision="model-digest-a",
        endpoint_origin_display="https://api.example.test",
        endpoint_origin_revision="origin-v1",
        compatibility_hash="compatibility-v1",
        dimensions=1536,
        vector_backend="chromadb",
    )
    generation = SimpleNamespace(  # nosec B106
        id="generation-a",
        owner_user_id="1",
        configuration_revision=8,
        state=SemanticGenerationState.STAGING,
        root_job_id=root_job_id,
        model_revision="model-digest-a",
        fencing_token="generation-fence-a",
    )

    class Store:
        owner_user_id = "1"

        def get_configuration(self, _dataset_id):
            return config

        def get_generation(self, _dataset_id, _generation_id):
            return generation

    class ActiveUsers:
        @classmethod
        async def from_pool(cls):
            return cls()

        async def get_user_by_id(self, _user_id):
            return {"id": 1, "is_active": True}

    current = SimpleNamespace(
        capability_revision="capability-v1",
        disclosure_hash="disclosure-v1",
        provider_label="provider-a",
        model="model-a",
        model_revision=None,
        endpoint_display="https://api.example.test",
        endpoint_origin_revision="origin-v1",
        indexing_available=True,
        vector_backend="chromadb",
    )
    monkeypatch.setattr(notes_semantic_index_worker, "AuthnzUsersRepo", ActiveUsers)
    monkeypatch.setattr(
        notes_semantic_index_worker,
        "resolve_semantic_capabilities",
        lambda *_args, **_kwargs: current,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.rbac.user_has_permission",
        lambda *_args, **_kwargs: True,
    )
    runtime = notes_semantic_index_worker.ProductionSemanticRuntime(
        db=SimpleNamespace(note_semantic_store=Store(), note_store=object()),
        owner_user_id="1",
        dataset_id="dataset-a",
        configuration_revision=8,
        generation_id="generation-a",
        root_job_id=root_job_id,
        vectors=object(),
        settings=SemanticIndexSettings(),
    )

    authority = await runtime._revalidate(runtime._fence())

    assert authority.model_revision == "model-digest-a"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage",
    ["database", "runtime", "release"],
)
async def test_worker_sanitizes_secret_bearing_dependency_failures(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    secret = "postgresql://user:password@private/db?token=super-secret"  # nosec B105
    closed: list[bool] = []

    def release_database():
        closed.append(True)
        if stage == "release":
            raise RuntimeError(secret)

    db = SimpleNamespace(release_context_connection=release_database)

    async def open_database(_owner):
        if stage == "database":
            raise RuntimeError(secret)
        return db

    async def build_runtime(**_kwargs):
        if stage == "runtime":
            raise RuntimeError(secret)
        return _RecoveredRuntime()

    class _RecoveredRuntime:
        async def recover(self, **_kwargs):
            return {
                "state": "completed",
                "indexed_notes": 0,
                "excluded_notes": 0,
                "failed_notes": 0,
                "published_chunks": 0,
                "cleanup_complete": True,
                "error_code": None,
            }

    monkeypatch.setattr(notes_semantic_index_worker, "_open_owner_database", open_database)
    monkeypatch.setattr(notes_semantic_index_worker, "build_production_runtime", build_runtime)
    job = {
        "uuid": "6ec1dfbe-f86f-4d2b-93af-f88f64cd9701",
        "owner_user_id": "1",
        "domain": JOB_DOMAIN,
        "queue": JOB_QUEUE,
        "job_type": JOB_TYPE,
        "payload": {
            "schema_version": 1,
            "dataset_id": "dataset-a",
            "configuration_revision": 7,
            "generation_id": None,
            "mode": "build",
        },
    }

    with pytest.raises(SemanticIndexingError) as exc_info:
        await notes_semantic_index_worker.handle_notes_semantic_index_job(
            job,
            jobs=SimpleNamespace(),
            worker_id="worker-a",
        )

    assert exc_info.value.failure_code == "notes_semantic_worker_runtime_failed"
    assert secret not in str(exc_info.value)
    assert secret not in repr(exc_info.value)
    assert closed == ([] if stage == "database" else [True])


@pytest.mark.asyncio
async def test_cleanup_backend_failure_retries_exact_claim_with_sanitized_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retries: list[dict[str, object]] = []

    class Store:
        def retry_work(self, **kwargs):
            retries.append(kwargs)
            return object()

    scope = notes_semantic_maintenance._ProductionScope(
        db=SimpleNamespace(note_semantic_store=Store()),
        jobs=SimpleNamespace(),
        owner_user_id="owner-a",
        dataset_id="dataset-a",
        settings=SemanticIndexSettings(),
    )

    class Runtime:
        async def cleanup_claim(self, _claim):
            raise RuntimeError("/private/vector?token=super-secret")

    async def runtime(_generation_id):
        return Runtime()

    monkeypatch.setattr(scope, "_runtime", runtime)
    claim = SimpleNamespace(  # nosec B106
        id="work-a",
        generation_id="generation-a",
        claim_token="claim-a",
        attempt_count=2,
    )

    assert await scope.cleanup_claim(claim) is False
    assert retries[0]["work_id"] == "work-a"
    assert retries[0]["expected_claim_token"] == "claim-a"
    assert retries[0]["error_code"] == "notes_semantic_cleanup_failed"
    assert "super-secret" not in repr(retries)


@pytest.mark.asyncio
async def test_production_maintenance_audits_confirmed_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audits: list[dict[str, object]] = []
    scope = notes_semantic_maintenance._ProductionScope(
        db=SimpleNamespace(note_semantic_store=SimpleNamespace()),
        jobs=SimpleNamespace(),
        owner_user_id="owner-a",
        dataset_id="dataset-a",
        settings=SemanticIndexSettings(),
    )

    class Runtime:
        async def cleanup_claim(self, _claim):
            return True

    async def runtime(_generation_id: str):
        return Runtime()

    async def capture_audit(**kwargs: object) -> None:
        audits.append(dict(kwargs))

    monkeypatch.setattr(scope, "_runtime", runtime)
    monkeypatch.setattr(
        notes_semantic_maintenance,
        "emit_semantic_audit_event",
        capture_audit,
        raising=False,
    )

    assert await scope.cleanup_claim(SimpleNamespace(generation_id="generation-a")) is True
    assert audits == [
        {
            "owner_user_id": "owner-a",
            "dataset_id": "dataset-a",
            "event": "cleanup_completion",
            "status": "success",
            "generation_id": "generation-a",
        }
    ]


@pytest.mark.asyncio
async def test_no_work_owner_discovery_is_bounded_and_continues_fairly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class Users:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        async def list_users(self, *, offset: int, limit: int):
            self.calls.append((offset, limit))
            end = min(offset + limit, 50)
            return ([{"id": value + 1} for value in range(offset, end)], 50)

        async def list_users_for_semantic_health_sweep(
            self,
            *,
            after_id: int | None,
            limit: int,
        ) -> list[dict[str, object]]:
            return []

    class Store:
        def list_maintenance_dataset_ids(self, *, limit: int):
            assert limit <= 4
            return ()

    users = Users()
    opened: list[str] = []

    async def open_database(owner):
        opened.append(owner)
        return SimpleNamespace(
            note_semantic_store=Store(),
            release_context_connection=lambda: None,
        )

    monkeypatch.setattr(notes_semantic_maintenance, "_open_owner_database", open_database)
    runner = notes_semantic_maintenance._MaintenanceRunner(
        jobs=JobManager(tmp_path / "fair-health-jobs.db"),
        users_repo=users,
        settings=SemanticIndexSettings(),
    )

    await runner.run_pass(now=NOW, limit=4)
    first_pass_calls = len(users.calls)
    first_pass_opened = len(opened)
    await runner.run_pass(now=NOW + timedelta(minutes=1), limit=4)

    assert first_pass_opened <= 4
    assert len(opened) - first_pass_opened <= 4
    assert first_pass_calls == 1
    assert users.calls[first_pass_calls][0] > 0


@pytest.mark.asyncio
async def test_complete_health_sweep_aggregates_all_owners_and_rebuilds_after_restart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    def snapshot(
        backend: str,
        *,
        indexed: int,
        failed: int = 0,
        pending: int = 0,
        stale: int = 0,
        cleanup: int = 0,
        retries: int = 0,
        oldest: datetime | None = None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            backend=backend,
            indexed_notes=indexed,
            excluded_notes=0,
            failed_notes=failed,
            dirty_notes=pending,
            pending_notes=pending,
            stale_generations=stale,
            cleanup_backlog=cleanup,
            cleanup_retries=retries,
            oldest_cleanup_created_at=oldest,
        )

    owner_snapshots = {
        "1": {
            "dataset-a": snapshot("chromadb", indexed=10),
            "dataset-b": snapshot(
                "chromadb",
                indexed=1,
                failed=2,
                pending=1,
                stale=1,
                cleanup=2,
                retries=4,
                oldest=NOW - timedelta(minutes=5),
            ),
        },
        "2": {
            "dataset-c": snapshot(
                "pgvector",
                indexed=3,
                cleanup=1,
                retries=1,
                oldest=NOW - timedelta(minutes=2),
            ),
        },
    }

    class Users:
        users = [{"id": 1}, {"id": 2}]

        async def list_users_for_semantic_health_sweep(
            self,
            *,
            after_id: int | None,
            limit: int,
        ):
            users = sorted(self.users, key=lambda user: user["id"], reverse=True)
            return [user for user in users if after_id is None or user["id"] < after_id][:limit]

        async def get_user_by_id(self, user_id: int):
            return next((user for user in self.users if user["id"] == user_id), None)

    class Store:
        def __init__(self, values: dict[str, SimpleNamespace]) -> None:
            self.values = values

        def list_observability_dataset_ids(
            self,
            *,
            limit: int,
            after_dataset_id: str | None = None,
        ) -> tuple[str, ...]:
            dataset_ids = sorted(self.values)
            if after_dataset_id is not None:
                dataset_ids = [value for value in dataset_ids if value > after_dataset_id]
            return tuple(dataset_ids[:limit])

        def get_observability_snapshot(
            self,
            dataset_id: str,
            *,
            current_capability_revision: str | None,
        ) -> SimpleNamespace:
            assert current_capability_revision == "capability-current"
            return self.values[dataset_id]

    async def open_database(owner: str):
        return SimpleNamespace(
            owner=owner,
            note_semantic_store=Store(owner_snapshots[owner]),
            release_context_connection=lambda: None,
        )

    class Coordinator:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def run_pass(self, **_kwargs: object):
            return notes_semantic_maintenance.SemanticMaintenanceResult(0, 0, 0, 0)

    observations: list[tuple[SimpleNamespace, ...]] = []
    monkeypatch.setattr(notes_semantic_maintenance, "_open_owner_database", open_database)
    monkeypatch.setattr(notes_semantic_maintenance, "SemanticMaintenanceCoordinator", Coordinator)
    monkeypatch.setattr(
        notes_semantic_maintenance,
        "resolve_semantic_capabilities",
        lambda _db, *, settings: SimpleNamespace(capability_revision="capability-current"),
        raising=False,
    )
    monkeypatch.setattr(
        notes_semantic_maintenance,
        "record_semantic_aggregate_metrics",
        lambda *, snapshots, now: observations.append(tuple(snapshots)),
        raising=False,
    )

    jobs = JobManager(tmp_path / "semantic-health-jobs.db")

    async def complete_one_sweep() -> tuple[SimpleNamespace, ...]:
        for _ in range(6):
            runner = notes_semantic_maintenance._MaintenanceRunner(
                jobs=JobManager(jobs.db_path),
                users_repo=Users(),
                settings=SemanticIndexSettings(),
            )
            await runner._run_health_sweep_page(now=NOW, limit=2)
            if observations:
                return observations[-1]
        raise AssertionError("bounded health sweep did not converge")

    first_runner = notes_semantic_maintenance._MaintenanceRunner(
        jobs=jobs,
        users_repo=Users(),
        settings=SemanticIndexSettings(),
    )
    await first_runner._run_health_sweep_page(now=NOW, limit=2)
    assert observations == []
    persisted = JobManager(jobs.db_path).get_notes_semantic_health_sweep()
    assert persisted["after_owner_id"] == 2
    assert persisted["after_dataset_id"] is None
    first = await complete_one_sweep()

    chromadb = next(value for value in first if value.backend == "chromadb")
    pgvector = next(value for value in first if value.backend == "pgvector")
    assert (chromadb.indexed_notes, chromadb.failed_notes, chromadb.pending_notes) == (11, 2, 1)
    assert (chromadb.stale_generations, chromadb.cleanup_backlog, chromadb.cleanup_retries) == (1, 2, 4)
    assert chromadb.oldest_cleanup_created_at == NOW - timedelta(minutes=5)
    assert (pgvector.cleanup_backlog, pgvector.cleanup_retries) == (1, 1)

    observations.clear()
    restarted = notes_semantic_maintenance._MaintenanceRunner(
        jobs=JobManager(jobs.db_path),
        users_repo=Users(),
        settings=SemanticIndexSettings(),
    )
    await restarted._run_health_sweep_page(now=NOW, limit=2)
    assert observations == []
    rebuilt = await complete_one_sweep()

    assert rebuilt == first


@pytest.mark.asyncio
async def test_health_sweep_bounds_empty_owner_database_scans(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    opened: list[str] = []
    observations: list[tuple[object, ...]] = []

    class Users:
        users = [{"id": 1}, {"id": 2}, {"id": 3}]

        async def list_users_for_semantic_health_sweep(
            self,
            *,
            after_id: int | None,
            limit: int,
        ):
            users = sorted(self.users, key=lambda user: user["id"], reverse=True)
            return [user for user in users if after_id is None or user["id"] < after_id][:limit]

        async def get_user_by_id(self, user_id: int):
            return next((user for user in self.users if user["id"] == user_id), None)

    class Store:
        def list_observability_dataset_ids(
            self,
            *,
            limit: int,
            after_dataset_id: str | None = None,
        ) -> tuple[str, ...]:
            del limit, after_dataset_id
            return ()

    async def open_database(owner: str):
        opened.append(owner)
        return SimpleNamespace(
            note_semantic_store=Store(),
            release_context_connection=lambda: None,
        )

    monkeypatch.setattr(notes_semantic_maintenance, "_open_owner_database", open_database)
    monkeypatch.setattr(
        notes_semantic_maintenance,
        "record_semantic_aggregate_metrics",
        lambda *, snapshots, now: observations.append(tuple(snapshots)),
    )
    runner = notes_semantic_maintenance._MaintenanceRunner(
        jobs=JobManager(tmp_path / "empty-health-jobs.db"),
        users_repo=Users(),
        settings=SemanticIndexSettings(),
    )

    await runner._run_health_sweep_page(now=NOW, limit=1)

    assert opened == ["3"]
    assert observations == []


@pytest.mark.asyncio
async def test_health_sweep_uses_unfiltered_authority_cursor_across_empty_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    observations: list[tuple[object, ...]] = []

    class Users:
        users = [{"id": 1}]

        async def list_users_for_semantic_health_sweep(
            self,
            *,
            after_id: int | None,
            limit: int,
        ):
            return [user for user in self.users if after_id is None or user["id"] < after_id][:limit]

        async def get_user_by_id(self, user_id: int):
            return next((user for user in self.users if user["id"] == user_id), None)

    class Store:
        def list_observability_dataset_ids(
            self,
            *,
            limit: int,
            after_dataset_id: str | None = None,
        ) -> tuple[str, ...]:
            dataset_ids = ("dataset-empty", "dataset-unhealthy")
            if after_dataset_id is not None:
                dataset_ids = tuple(value for value in dataset_ids if value > after_dataset_id)
            return dataset_ids[:limit]

        def get_observability_snapshot(
            self,
            dataset_id: str,
            *,
            current_capability_revision: str | None,
        ) -> SimpleNamespace:
            assert current_capability_revision == "capability-current"
            return SimpleNamespace(
                backend="chromadb" if dataset_id == "dataset-unhealthy" else "unavailable",
                indexed_notes=0,
                excluded_notes=0,
                failed_notes=int(dataset_id == "dataset-unhealthy"),
                dirty_notes=0,
                pending_notes=0,
                stale_generations=0,
                cleanup_backlog=0,
                cleanup_retries=0,
                oldest_cleanup_created_at=None,
            )

    async def open_database(_owner: str):
        return SimpleNamespace(
            note_semantic_store=Store(),
            release_context_connection=lambda: None,
        )

    monkeypatch.setattr(notes_semantic_maintenance, "_open_owner_database", open_database)
    monkeypatch.setattr(
        notes_semantic_maintenance,
        "resolve_semantic_capabilities",
        lambda _db, *, settings: SimpleNamespace(capability_revision="capability-current"),
    )
    monkeypatch.setattr(
        notes_semantic_maintenance,
        "record_semantic_aggregate_metrics",
        lambda *, snapshots, now: observations.append(tuple(snapshots)),
    )
    runner = notes_semantic_maintenance._MaintenanceRunner(
        jobs=JobManager(tmp_path / "cursor-health-jobs.db"),
        users_repo=Users(),
        settings=SemanticIndexSettings(),
    )

    for _ in range(4):
        await runner._run_health_sweep_page(now=NOW, limit=1)

    assert len(observations) == 1
    unhealthy = next(value for value in observations[0] if value.backend == "chromadb")
    assert unhealthy.failed_notes == 1


@pytest.mark.asyncio
async def test_dataset_discovery_reserves_budget_for_later_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator_limits: list[int] = []

    class Users:
        async def list_users(self, *, offset: int, limit: int):
            return ([{"id": 1}] if offset == 0 else [], 1)

    class Store:
        def list_maintenance_dataset_ids(self, *, limit: int):
            return ("dataset-a", "dataset-b", "dataset-c")[:limit]

    async def open_database(_owner):
        return SimpleNamespace(
            note_semantic_store=Store(),
            release_context_connection=lambda: None,
        )

    class Coordinator:
        def __init__(self, *, scopes, indexing_enabled):
            assert scopes
            assert indexing_enabled is True

        async def run_pass(self, *, now, limit):
            coordinator_limits.append(limit)
            return notes_semantic_maintenance.SemanticMaintenanceResult(1, 1, 0, 0)

    monkeypatch.setattr(notes_semantic_maintenance, "_open_owner_database", open_database)
    monkeypatch.setattr(
        notes_semantic_maintenance,
        "SemanticMaintenanceCoordinator",
        Coordinator,
    )
    runner = notes_semantic_maintenance._MaintenanceRunner(
        jobs=SimpleNamespace(),
        users_repo=Users(),
        settings=SemanticIndexSettings(),
    )

    result = await runner.run_pass(now=NOW, limit=4)

    assert result.claimed == 1
    assert coordinator_limits and coordinator_limits[0] >= 1


@pytest.mark.asyncio
async def test_hot_early_owner_advances_fairly_to_later_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visited: list[str] = []

    class Users:
        async def list_users(self, *, offset: int, limit: int):
            values = [{"id": owner} for owner in range(1, 5)]
            return (values[offset : offset + limit], len(values))

    class Store:
        def list_maintenance_dataset_ids(self, *, limit: int):
            return ("dataset-a",)[:limit]

    async def open_database(_owner):
        return SimpleNamespace(
            note_semantic_store=Store(),
            release_context_connection=lambda: None,
        )

    class Coordinator:
        def __init__(self, *, scopes, indexing_enabled):
            assert indexing_enabled is True
            self.owner = scopes[0]._owner_user_id

        async def run_pass(self, *, now, limit):
            del now
            visited.append(self.owner)
            return notes_semantic_maintenance.SemanticMaintenanceResult(
                limit,
                1,
                0,
                0,
            )

    monkeypatch.setattr(notes_semantic_maintenance, "_open_owner_database", open_database)
    monkeypatch.setattr(
        notes_semantic_maintenance,
        "SemanticMaintenanceCoordinator",
        Coordinator,
    )
    runner = notes_semantic_maintenance._MaintenanceRunner(
        jobs=SimpleNamespace(),
        users_repo=Users(),
        settings=SemanticIndexSettings(),
    )

    await runner.run_pass(now=NOW, limit=4)
    await runner.run_pass(now=NOW + timedelta(minutes=1), limit=4)

    assert visited == ["1", "2"]


@pytest.mark.parametrize(
    ("mode", "terminal"),
    [
        ("maintain", "cancelled"),
        ("maintain", "failed"),
        ("retry_failed", "failed"),
    ],
)
def test_terminal_maintenance_predecessor_admits_one_new_active_retry(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    terminal: str,
) -> None:
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_NOTES", JOB_QUEUE)
    jobs = JobManager(tmp_path / "maintenance-retry-jobs.sqlite")
    config = SimpleNamespace(
        desired_state=SimpleNamespace(value="enabled"),
        configuration_revision=8,
    )

    class Store:
        def get_configuration(self, _dataset_id):
            return config

    scope = notes_semantic_maintenance._ProductionScope(
        db=SimpleNamespace(note_semantic_store=Store()),
        jobs=jobs,
        owner_user_id="owner-a",
        dataset_id="dataset-a",
        settings=SemanticIndexSettings(),
    )
    if mode == "maintain":
        claim = notes_semantic_maintenance._DirtyClaim(
            owner_user_id="owner-a",
            dataset_id="dataset-a",
            generation_id="generation-a",
            dirty_generation=4,
        )
    else:
        claim = notes_semantic_maintenance._FailedClaim(
            owner_user_id="owner-a",
            dataset_id="dataset-a",
            generation_id="generation-a",
        )

    assert scope.admit(mode=mode, claim=claim) is True
    first = jobs.list_jobs(domain=JOB_DOMAIN, owner_user_id="owner-a", limit=10)[0]
    if terminal == "cancelled":
        assert jobs.cancel_job(first["id"], expected_uuid=first["uuid"]) is True
    else:
        acquired = jobs.acquire_next_job(
            domain=JOB_DOMAIN,
            queue=JOB_QUEUE,
            lease_seconds=30,
            worker_id="maintenance-test-worker",
        )
        assert acquired is not None
        assert (
            jobs.fail_job(
                acquired["id"],
                error="notes_semantic_worker_runtime_failed",
                retryable=False,
                worker_id=acquired["worker_id"],
                lease_id=acquired["lease_id"],
                error_code="notes_semantic_worker_runtime_failed",
            )
            is True
        )
    monkeypatch.setattr(jobs, "list_jobs", lambda **_kwargs: ())

    assert scope.admit(mode=mode, claim=claim) is True
    second_pass = JobManager.list_jobs(
        jobs,
        domain=JOB_DOMAIN,
        owner_user_id="owner-a",
        limit=10,
    )
    assert len(second_pass) == 2
    assert second_pass[0]["uuid"] != first["uuid"]
    assert second_pass[0]["status"] == "queued"

    scope.admit(mode=mode, claim=claim)
    assert (
        len(
            JobManager.list_jobs(
                jobs,
                domain=JOB_DOMAIN,
                owner_user_id="owner-a",
                limit=10,
            )
        )
        == 2
    )
