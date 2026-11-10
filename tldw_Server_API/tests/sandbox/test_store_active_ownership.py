"""Admitted work retains capacity until an explicit terminal transition."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunStatus, RuntimeType
from tldw_Server_API.app.core.Sandbox.store import InMemoryStore, PostgresStore, SandboxStore, SQLiteStore

pytest_plugins = ["tldw_Server_API.tests._plugins.postgres"]


@pytest.fixture(params=["memory", "sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def ownership_store(request: pytest.FixtureRequest, tmp_path: Path) -> SandboxStore:
    if request.param == "memory":
        return InMemoryStore()
    if request.param == "sqlite":
        return SQLiteStore(str(tmp_path / "ownership.db"))
    return PostgresStore(dsn=request.getfixturevalue("pg_temp_db")["dsn"])


@pytest.mark.parametrize(
    "operation, expected_phase",
    [
        ("put", RunPhase.queued),
        ("get", RunPhase.queued),
        ("update", RunPhase.queued),
        ("claim", RunPhase.queued),
        ("admit", RunPhase.starting),
    ],
)
def test_in_memory_status_mutation_requires_explicit_update(operation: str, expected_phase: RunPhase) -> None:
    store = InMemoryStore()
    status = RunStatus(id="owned", phase=RunPhase.queued, runtime=RuntimeType.docker)
    store.put_run("user", status)
    if operation == "get":
        status = store.get_run(status.id)
    elif operation == "update":
        store.update_run(status)
    elif operation in {"claim", "admit"}:
        status = store.try_claim_run(status.id, worker_id="worker")
        if operation == "admit":
            status = store.try_admit_run_start(status.id, worker_id="worker", max_active_runs=1)

    status.phase = RunPhase.completed

    assert store.get_run(status.id).phase == expected_phase


def test_service_held_status_cannot_overwrite_confirmed_cancellation(ownership_store: SandboxStore) -> None:
    store = ownership_store
    status = RunStatus(id="late", phase=RunPhase.running, runtime=RuntimeType.docker)
    store.put_run("user", status)
    cancellation = store.get_run(status.id)
    cancellation.phase = RunPhase.killed
    store.update_run(cancellation)

    status.phase = RunPhase.completed
    store.update_run(status)

    assert store.get_run(status.id).phase == RunPhase.killed


@pytest.mark.parametrize("lose_claim", [False, True])
def test_admitted_start_retains_capacity_before_backend_identity(
    ownership_store: SandboxStore,
    lose_claim: bool,
) -> None:
    store = ownership_store
    for run_id in ("stalled-start", "next"):
        store.put_run("user", RunStatus(id=run_id, phase=RunPhase.queued, runtime=RuntimeType.docker))
        store.try_claim_run(run_id, worker_id=run_id)
    admitted = store.try_admit_run_start("stalled-start", worker_id="stalled-start", max_active_runs=1)
    admitted.claim_expires_at = datetime.now(timezone.utc) - timedelta(seconds=60)
    if lose_claim:
        admitted.claim_owner = None
        admitted.claim_expires_at = None
    store.update_run(admitted)

    assert store.try_admit_run_start("next", worker_id="next", max_active_runs=1) is None

    admitted.phase = RunPhase.killed
    store.update_run(admitted)
    assert store.try_admit_run_start("next", worker_id="next", max_active_runs=1) is not None
