import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    CreateJobCommand,
    IdempotentOperationCommand,
    IdempotentOperationDisposition,
    IdempotentOperationUnavailableError,
)


def _set_env(monkeypatch):


    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    # Do not set SINGLE_USER_API_KEY so tests use deterministic key from settings
    monkeypatch.delenv("SINGLE_USER_API_KEY", raising=False)
    # Ensure API endpoints use the same SQLite DB as this test (per tmp CWD)
    import os as _os
    monkeypatch.setenv("JOBS_DB_PATH", _os.path.join(_os.getcwd(), "Databases", "jobs.db"))


def _backdate_sqlite(job_id: int, days: int = 2):
    jm = JobManager()
    conn = jm._connect()
    try:
        cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        with conn:
            conn.execute(
                "UPDATE jobs SET completed_at = ?, updated_at = ? WHERE id = ?",
                (cutoff, cutoff, int(job_id)),
            )
    finally:
        try:
            conn.close()
        except Exception:
            _ = None


def _receipt_command(
    *,
    key_digest: str = "a" * 64,
    operation_scope: str = "share:share-1",
) -> IdempotentOperationCommand:
    return IdempotentOperationCommand(
        job=CreateJobCommand(
            domain="sharing",
            queue="workspace-clone",
            job_type="workspace_clone",
            payload={"schema_version": 1},
            owner_user_id="recipient-1",
            batch_group=operation_scope,
            priority=5,
            max_retries=0,
        ),
        key_digest=key_digest,
        request_fingerprint="b" * 64,
        operation_scope=operation_scope,
        receipt_expires_at=datetime.now(timezone.utc) + timedelta(days=31),
    )


def _terminalize_old(manager: JobManager, *job_ids: int) -> None:
    old = (datetime.now(timezone.utc) - timedelta(days=40)).strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )
    conn = sqlite3.connect(manager.db_path)
    try:
        with conn:
            placeholders = ",".join("?" for _ in job_ids)
            conn.execute(
                f"UPDATE jobs SET status='completed', completed_at=? "  # nosec B608
                f"WHERE id IN ({placeholders})",  # nosec B608
                (old, *job_ids),
            )
    finally:
        conn.close()


def _receipt_count(manager: JobManager) -> int:
    conn = sqlite3.connect(manager.db_path)
    try:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM job_idempotency_receipts"
            ).fetchone()[0]
        )
    finally:
        conn.close()


def test_notes_graph_prune_retains_29_and_30_days_and_forces_31_day_archive(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_NOTES", "graph-suggestions")
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_OTHER", "default")
    monkeypatch.delenv("JOBS_ARCHIVE_BEFORE_DELETE", raising=False)
    manager = JobManager(tmp_path / "notes-retention.db")
    notes_jobs = [
        manager.create_job(
            domain="notes",
            queue="graph-suggestions",
            job_type="note_graph_suggestions",
            payload={"schema_version": 1},
            owner_user_id="owner-1",
            max_retries=0,
        )
        for _ in range(3)
    ]
    unrelated = manager.create_job(
        domain="other",
        queue="default",
        job_type="ordinary",
        payload={},
        owner_user_id="owner-1",
    )
    conn = sqlite3.connect(manager.db_path)
    try:
        with conn:
            for job, age in zip(notes_jobs, (29, 30, 31), strict=True):
                completed = (datetime.now(timezone.utc) - timedelta(days=age)).strftime(
                    "%Y-%m-%d %H:%M:%S.%f"
                )
                conn.execute(
                    "UPDATE jobs SET status='completed',completed_at=? WHERE id=?",
                    (completed, int(job["id"])),
                )
            old = (datetime.now(timezone.utc) - timedelta(days=2)).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )
            conn.execute(
                "UPDATE jobs SET status='completed',completed_at=? WHERE id=?",
                (old, int(unrelated["id"])),
            )
    finally:
        conn.close()

    assert manager.prune_jobs(statuses=["completed"], older_than_days=1) == 2
    assert manager.get_job_by_uuid(notes_jobs[0]["uuid"]) is not None
    assert manager.get_job_by_uuid(notes_jobs[1]["uuid"]) is not None
    archived = manager.get_job_or_archived_by_uuid(notes_jobs[2]["uuid"])
    assert archived is not None and archived["archived"] is True
    assert manager.get_job_or_archived_by_uuid(unrelated["uuid"]) is None

    assert manager.prune_jobs(
        statuses=["completed"],
        older_than_days=1,
        domain="notes",
    ) == 0


def test_prune_archives_receipt_job_when_global_archive_is_disabled(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_SHARING", "workspace-clone")
    monkeypatch.delenv("JOBS_ARCHIVE_BEFORE_DELETE", raising=False)
    manager = JobManager(tmp_path / "jobs.db")
    receipt_job = manager.admit_idempotent_operation(_receipt_command()).job
    ordinary_job = manager.create_job(
        domain="sharing",
        queue="workspace-clone",
        job_type="ordinary",
        payload={"kind": "ordinary"},
        owner_user_id="recipient-1",
    )
    _terminalize_old(manager, int(receipt_job["id"]), int(ordinary_job["id"]))

    deleted = manager.prune_jobs(statuses=["completed"], older_than_days=30)

    archived = manager.get_job_or_archived_by_uuid(receipt_job["uuid"])
    assert deleted == 2
    assert archived is not None
    assert archived["archived"] is True
    assert manager.get_job_or_archived_by_uuid(ordinary_job["uuid"]) is None
    replay = manager.admit_idempotent_operation(_receipt_command())
    assert replay.disposition is IdempotentOperationDisposition.REPLAYED
    assert replay.job["archived"] is True


def test_receipt_replay_remains_available_during_archive_move(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_SHARING", "workspace-clone")
    monkeypatch.delenv("JOBS_ARCHIVE_BEFORE_DELETE", raising=False)
    db_path = tmp_path / "jobs.db"
    prune_manager = JobManager(db_path)
    replay_manager = JobManager(db_path)
    receipt_job = prune_manager.admit_idempotent_operation(_receipt_command()).job
    _terminalize_old(prune_manager, int(receipt_job["id"]))

    archive_verified = threading.Event()
    release_prune = threading.Event()
    original_exact = prune_manager._exact_receipt_archive_uuids

    def _block_after_archive_copy(*args, **kwargs):
        archived_uuids = original_exact(*args, **kwargs)
        if archived_uuids and not archive_verified.is_set():
            archive_verified.set()
            assert release_prune.wait(timeout=10)
        return archived_uuids

    monkeypatch.setattr(
        prune_manager,
        "_exact_receipt_archive_uuids",
        _block_after_archive_copy,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        prune_future = executor.submit(
            prune_manager.prune_jobs,
            statuses=["completed"],
            older_than_days=30,
        )
        assert archive_verified.wait(timeout=10)
        try:
            replay = executor.submit(
                replay_manager.admit_idempotent_operation,
                _receipt_command(),
            ).result(timeout=10)
        finally:
            release_prune.set()
        assert prune_future.result(timeout=10) == 1

    assert replay.disposition is IdempotentOperationDisposition.REPLAYED
    assert replay.job["uuid"] == receipt_job["uuid"]
    assert replay.job["archived"] is False
    archived_replay = replay_manager.admit_idempotent_operation(_receipt_command())
    assert archived_replay.disposition is IdempotentOperationDisposition.REPLAYED
    assert archived_replay.job["archived"] is True


def test_prune_rolls_back_when_receipt_correlation_is_ambiguous(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_SHARING", "workspace-clone")
    manager = JobManager(tmp_path / "jobs.db")
    receipt_job = manager.admit_idempotent_operation(_receipt_command()).job
    ordinary_job = manager.create_job(
        domain="sharing",
        queue="workspace-clone",
        job_type="ordinary",
        payload={},
        owner_user_id="recipient-1",
    )
    _terminalize_old(manager, int(receipt_job["id"]), int(ordinary_job["id"]))
    conn = sqlite3.connect(manager.db_path)
    try:
        with conn:
            conn.execute(
                "UPDATE job_idempotency_receipts SET job_id=?",
                (int(ordinary_job["id"]),),
            )
    finally:
        conn.close()

    with pytest.raises(IdempotentOperationUnavailableError):
        manager.prune_jobs(statuses=["completed"], older_than_days=30)

    assert manager.get_job_by_uuid(receipt_job["uuid"]) is not None
    assert manager.get_job_by_uuid(ordinary_job["uuid"]) is not None
    conn = sqlite3.connect(manager.db_path)
    try:
        assert int(conn.execute("SELECT COUNT(*) FROM jobs_archive").fetchone()[0]) == 0
    finally:
        conn.close()


def test_receipt_pruning_requires_expired_unique_terminal_archive(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_SHARING", "workspace-clone")
    monkeypatch.delenv("JOBS_ARCHIVE_BEFORE_DELETE", raising=False)
    manager = JobManager(tmp_path / "jobs.db")
    archived_job = manager.admit_idempotent_operation(_receipt_command()).job
    alias = manager.admit_idempotent_operation(
        _receipt_command(key_digest="e" * 64)
    )
    assert alias.disposition is IdempotentOperationDisposition.CONVERGED
    active_terminal = manager.admit_idempotent_operation(
        _receipt_command(
            key_digest="c" * 64,
            operation_scope="share:share-2",
        )
    ).job
    active_nonterminal = manager.admit_idempotent_operation(
        _receipt_command(
            key_digest="d" * 64,
            operation_scope="share:share-3",
        )
    ).job
    _terminalize_old(manager, int(archived_job["id"]))
    conn = sqlite3.connect(manager.db_path)
    try:
        with conn:
            conn.execute(
                "UPDATE jobs SET status='completed', completed_at=? WHERE id=?",
                (
                    datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f"),
                    int(active_terminal["id"]),
                ),
            )
    finally:
        conn.close()
    assert manager.prune_jobs(
        statuses=["completed"],
        older_than_days=30,
        job_type="workspace_clone",
    ) == 1

    future = datetime.now(timezone.utc) + timedelta(days=32)
    assert manager.prune_idempotency_receipts(now=future, limit=1) == 1
    assert _receipt_count(manager) == 3
    assert manager.prune_idempotency_receipts(now=future, limit=1) == 1
    assert _receipt_count(manager) == 2
    assert manager.prune_idempotency_receipts(now=future, limit=1) == 0
    assert manager.get_job_by_uuid(active_terminal["uuid"]) is not None
    assert manager.get_job_by_uuid(active_nonterminal["uuid"]) is not None


def test_receipt_pruning_is_idempotent_after_archival(monkeypatch, tmp_path):
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_SHARING", "workspace-clone")
    manager = JobManager(tmp_path / "jobs.db")
    job = manager.admit_idempotent_operation(_receipt_command()).job
    _terminalize_old(manager, int(job["id"]))
    assert manager.prune_jobs(statuses=["completed"], older_than_days=30) == 1
    future = datetime.now(timezone.utc) + timedelta(days=32)

    assert manager.prune_idempotency_receipts(now=future) == 1
    assert manager.prune_idempotency_receipts(now=future) == 0
    archived = manager.get_job_or_archived_by_uuid(job["uuid"])
    assert archived is not None
    assert archived["archived"] is True


@pytest.mark.parametrize("limit", (0, -1, 10_001, True))
def test_receipt_pruning_requires_bounded_positive_limit(tmp_path, limit):
    manager = JobManager(tmp_path / "jobs.db")

    with pytest.raises(ValueError, match="limit"):
        manager.prune_idempotency_receipts(limit=limit)


def test_jobs_prune_dry_run_and_filters_sqlite(monkeypatch, tmp_path):


     # Isolate DB in a temp CWD so Databases/jobs.db is per-test
    monkeypatch.chdir(tmp_path)
    _set_env(monkeypatch)

    # Import app after env is set
    # Reset settings before importing app to pick up TEST_MODE
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
    reset_settings()
    from tldw_Server_API.app.main import app
    try:
        app.dependency_overrides.clear()
    except Exception:
        _ = None

    jm = JobManager()
    # Seed: 2 completed + 1 failed (old), 1 failed (recent)
    j1 = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    jm.complete_job(int(j1["id"]))
    _backdate_sqlite(int(j1["id"]))

    j2 = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    jm.complete_job(int(j2["id"]))
    _backdate_sqlite(int(j2["id"]))

    j3 = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    jm.fail_job(int(j3["id"]), error="x", retryable=False)
    _backdate_sqlite(int(j3["id"]))

    j4 = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    jm.fail_job(int(j4["id"]), error="x", retryable=False)  # recent (should not be pruned with older_than_days=1)

    # Use the deterministic single-user key from settings
    headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
    with TestClient(app, headers=headers) as client:
        body = {
            "statuses": ["completed", "failed"],
            "older_than_days": 1,
            "domain": "chatbooks",
            "queue": "default",
            "job_type": "export",
            "dry_run": True,
        }
        r = client.post("/api/v1/jobs/prune", json=body)
        assert r.status_code == 200, r.text
        assert r.json()["deleted"] == 3  # two completed + one failed (old)

        # Execute prune
        body["dry_run"] = False
        r2 = client.post("/api/v1/jobs/prune", json=body)
        assert r2.status_code == 200
        assert r2.json()["deleted"] == 3

        # Subsequent dry-run should report 0
        body["dry_run"] = True
        r3 = client.post("/api/v1/jobs/prune", json=body)
        assert r3.status_code == 200
        assert r3.json()["deleted"] == 0


def test_jobs_prune_filters_scope_sqlite(monkeypatch, tmp_path):


     # New temp CWD for isolation
    monkeypatch.chdir(tmp_path)
    _set_env(monkeypatch)

    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
    reset_settings()
    from tldw_Server_API.app.main import app
    try:
        app.dependency_overrides.clear()
    except Exception:
        _ = None

    jm = JobManager()
    # Seed one job in a different domain/queue
    jx = jm.create_job(domain="other", queue="low", job_type="export", payload={}, owner_user_id="1")
    jm.complete_job(int(jx["id"]))
    _backdate_sqlite(int(jx["id"]))

    headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
    with TestClient(app, headers=headers) as client:
        # Scoped to chatbooks/default/export - should not match the seeded job
        body = {
            "statuses": ["completed"],
            "older_than_days": 1,
            "domain": "chatbooks",
            "queue": "default",
            "job_type": "export",
            "dry_run": True,
        }
        r = client.post("/api/v1/jobs/prune", json=body)
        assert r.status_code == 200
        assert r.json()["deleted"] == 0


def test_jobs_prune_sanitizes_generic_failure(monkeypatch, tmp_path):


    monkeypatch.chdir(tmp_path)
    _set_env(monkeypatch)

    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
    reset_settings()
    from tldw_Server_API.app.main import app

    def boom(self, **_kwargs):
        raise RuntimeError("jobs prune backend exploded")

    monkeypatch.setattr(JobManager, "prune_jobs", boom)

    headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
    with TestClient(app, headers=headers) as client:
        r = client.post(
            "/api/v1/jobs/prune",
            json={
                "statuses": ["completed"],
                "older_than_days": 1,
                "domain": "chatbooks",
                "queue": "default",
                "job_type": "export",
                "dry_run": True,
            },
        )
        assert r.status_code == 500
        assert r.json()["detail"] == "Prune failed"
