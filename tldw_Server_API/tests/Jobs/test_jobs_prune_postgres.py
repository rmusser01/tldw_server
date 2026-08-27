import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest

psycopg = pytest.importorskip("psycopg")
pytestmark = pytest.mark.pg_jobs

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    CreateJobCommand,
    IdempotentOperationCommand,
    IdempotentOperationDisposition,
    IdempotentOperationUnavailableError,
)
from tldw_Server_API.app.core.Jobs.pg_migrations import ensure_jobs_tables_pg


def _backdate_pg(dsn: str, job_id: int, days: int = 2):
    conn = psycopg.connect(dsn)
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE jobs SET completed_at = NOW() - (%s || ' days')::interval WHERE id = %s",
                (int(days), int(job_id)),
            )
    finally:
        conn.close()


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


def _set_terminal_pg(
    dsn: str,
    job_id: int,
    *,
    days_old: int,
) -> None:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE jobs SET status='completed', "
            "completed_at=NOW() - (%s || ' days')::interval WHERE id=%s",
            (days_old, job_id),
        )


def _receipt_count_pg(dsn: str) -> int:
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM job_idempotency_receipts")
        return int(cur.fetchone()[0])


def test_pg_notes_graph_prune_retains_29_and_30_days_and_forces_31_day_archive(
    monkeypatch,
    jobs_pg_dsn,
) -> None:
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_NOTES", "graph-suggestions")
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_OTHER", "default")
    monkeypatch.delenv("JOBS_ARCHIVE_BEFORE_DELETE", raising=False)
    manager = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
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
    for job, age in zip(notes_jobs, (29, 30, 31), strict=True):
        _set_terminal_pg(jobs_pg_dsn, int(job["id"]), days_old=age)
    _set_terminal_pg(jobs_pg_dsn, int(unrelated["id"]), days_old=2)

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


def test_pg_prune_archives_receipt_job_without_global_archive(
    monkeypatch,
    jobs_pg_dsn,
):
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_SHARING", "workspace-clone")
    monkeypatch.delenv("JOBS_ARCHIVE_BEFORE_DELETE", raising=False)
    manager = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    receipt_job = manager.admit_idempotent_operation(_receipt_command()).job
    ordinary_job = manager.create_job(
        domain="sharing",
        queue="workspace-clone",
        job_type="ordinary",
        payload={"kind": "ordinary"},
        owner_user_id="recipient-1",
    )
    _set_terminal_pg(jobs_pg_dsn, int(receipt_job["id"]), days_old=40)
    _set_terminal_pg(jobs_pg_dsn, int(ordinary_job["id"]), days_old=40)

    deleted = manager.prune_jobs(statuses=["completed"], older_than_days=30)

    archived = manager.get_job_or_archived_by_uuid(receipt_job["uuid"])
    assert deleted == 2
    assert archived is not None
    assert archived["archived"] is True
    assert manager.get_job_or_archived_by_uuid(ordinary_job["uuid"]) is None
    replay = manager.admit_idempotent_operation(_receipt_command())
    assert replay.disposition is IdempotentOperationDisposition.REPLAYED
    assert replay.job["archived"] is True


def test_pg_receipt_replay_remains_available_during_archive_move(
    monkeypatch,
    jobs_pg_dsn,
):
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_SHARING", "workspace-clone")
    monkeypatch.delenv("JOBS_ARCHIVE_BEFORE_DELETE", raising=False)
    prune_manager = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    replay_manager = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    receipt_job = prune_manager.admit_idempotent_operation(_receipt_command()).job
    _set_terminal_pg(jobs_pg_dsn, int(receipt_job["id"]), days_old=40)

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


def test_pg_prune_rolls_back_on_ambiguous_receipt(
    monkeypatch,
    jobs_pg_dsn,
):
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_SHARING", "workspace-clone")
    manager = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    receipt_job = manager.admit_idempotent_operation(_receipt_command()).job
    ordinary_job = manager.create_job(
        domain="sharing",
        queue="workspace-clone",
        job_type="ordinary",
        payload={},
        owner_user_id="recipient-1",
    )
    _set_terminal_pg(jobs_pg_dsn, int(receipt_job["id"]), days_old=40)
    _set_terminal_pg(jobs_pg_dsn, int(ordinary_job["id"]), days_old=40)
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE job_idempotency_receipts SET job_id=%s",
            (int(ordinary_job["id"]),),
        )

    with pytest.raises(IdempotentOperationUnavailableError):
        manager.prune_jobs(statuses=["completed"], older_than_days=30)

    assert manager.get_job_by_uuid(receipt_job["uuid"]) is not None
    assert manager.get_job_by_uuid(ordinary_job["uuid"]) is not None
    with psycopg.connect(jobs_pg_dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM jobs_archive")
        assert int(cur.fetchone()[0]) == 0


def test_pg_receipt_pruning_requires_expired_unique_terminal_archive(
    monkeypatch,
    jobs_pg_dsn,
):
    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_SHARING", "workspace-clone")
    monkeypatch.delenv("JOBS_ARCHIVE_BEFORE_DELETE", raising=False)
    manager = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
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
    _set_terminal_pg(jobs_pg_dsn, int(archived_job["id"]), days_old=40)
    _set_terminal_pg(jobs_pg_dsn, int(active_terminal["id"]), days_old=0)
    assert manager.prune_jobs(
        statuses=["completed"],
        older_than_days=30,
        job_type="workspace_clone",
    ) == 1

    future = datetime.now(timezone.utc) + timedelta(days=32)
    assert manager.prune_idempotency_receipts(now=future, limit=1) == 1
    assert _receipt_count_pg(jobs_pg_dsn) == 3
    assert manager.prune_idempotency_receipts(now=future, limit=1) == 1
    assert _receipt_count_pg(jobs_pg_dsn) == 2
    assert manager.prune_idempotency_receipts(now=future, limit=1) == 0
    assert manager.get_job_by_uuid(active_terminal["uuid"]) is not None
    assert manager.get_job_by_uuid(active_nonterminal["uuid"]) is not None


def test_jobs_prune_dry_run_and_filters_postgres(monkeypatch, jobs_pg_dsn):


     # Set env so endpoint manager uses PG
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.delenv("SINGLE_USER_API_KEY", raising=False)
    monkeypatch.setenv("JOBS_DB_URL", jobs_pg_dsn)
    monkeypatch.setenv("JOBS_ADMIN_COMPLETE_QUEUED_ALLOW_DOMAINS", "chatbooks")

    ensure_jobs_tables_pg(jobs_pg_dsn)
    jm = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    # Seed: 1 completed (old), 1 failed (old), 1 failed (recent)
    jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    acq1 = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=10, worker_id="w1")
    assert acq1 is not None
    assert jm.complete_job(int(acq1["id"]), worker_id="w1", lease_id=str(acq1.get("lease_id")), enforce=True)
    _backdate_pg(jobs_pg_dsn, int(acq1["id"]))

    jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    acq2 = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=10, worker_id="w2")
    assert acq2 is not None
    assert jm.fail_job(
        int(acq2["id"]),
        error="x",
        retryable=False,
        worker_id="w2",
        lease_id=str(acq2.get("lease_id")),
        enforce=True,
    )
    _backdate_pg(jobs_pg_dsn, int(acq2["id"]))

    jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    acq3 = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=10, worker_id="w3")
    assert acq3 is not None
    assert jm.fail_job(
        int(acq3["id"]),
        error="x",
        retryable=False,
        worker_id="w3",
        lease_id=str(acq3.get("lease_id")),
        enforce=True,
    )

    from fastapi.testclient import TestClient

    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
    reset_settings()
    from tldw_Server_API.app.main import app
    try:
        app.dependency_overrides.clear()
    except Exception:
        _ = None

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
        assert r.json()["deleted"] == 2

        body["dry_run"] = False
        r2 = client.post("/api/v1/jobs/prune", json=body)
        assert r2.status_code == 200
        assert r2.json()["deleted"] == 2


def test_jobs_prune_filters_scope_postgres(monkeypatch, jobs_pg_dsn):


     # Configure PG and single-user test mode
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.delenv("SINGLE_USER_API_KEY", raising=False)
    monkeypatch.setenv("JOBS_DB_URL", jobs_pg_dsn)
    monkeypatch.setenv("JOBS_ADMIN_COMPLETE_QUEUED_ALLOW_DOMAINS", "chatbooks")

    ensure_jobs_tables_pg(jobs_pg_dsn)
    jm = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    # Seed a job in a different domain/queue
    jm.create_job(domain="other", queue="low", job_type="export", payload={}, owner_user_id="1")
    acq = jm.acquire_next_job(domain="other", queue="low", lease_seconds=10, worker_id="w4")
    assert acq is not None
    assert jm.complete_job(int(acq["id"]), worker_id="w4", lease_id=str(acq.get("lease_id")), enforce=True)
    _backdate_pg(jobs_pg_dsn, int(acq["id"]))

    from fastapi.testclient import TestClient

    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
    reset_settings()
    from tldw_Server_API.app.main import app
    try:
        app.dependency_overrides.clear()
    except Exception:
        _ = None

    headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
    with TestClient(app, headers=headers) as client:
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


def test_pg_prune_waits_for_inflight_payload_replacement_before_archiving(
    monkeypatch,
    jobs_pg_dsn,
):
    monkeypatch.setenv("JOBS_DB_URL", jobs_pg_dsn)
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "true")
    jm = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    job = jm.create_job(
        domain="prompt_studio",
        queue="default",
        job_type="optimization",
        payload={"version": "original"},
        owner_user_id="1",
    )
    acquired = jm.acquire_next_job(
        domain="prompt_studio",
        queue="default",
        lease_seconds=30,
        worker_id="worker-a",
    )
    assert acquired is not None
    assert jm.complete_job(
        int(acquired["id"]),
        worker_id="worker-a",
        lease_id=str(acquired["lease_id"]),
        enforce=True,
    )
    _backdate_pg(jobs_pg_dsn, int(job["id"]), days=2)

    replacement_manager = JobManager(
        None,
        backend="postgres",
        db_url=jobs_pg_dsn,
    )
    prune_manager = JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    entered_serialization = threading.Event()
    release_serialization = threading.Event()
    prune_connection_ready = threading.Event()
    prune_backend_pid: list[int] = []
    original_encrypt = replacement_manager._maybe_encrypt_json
    original_replacement_connect = replacement_manager._connect
    original_prune_connect = prune_manager._connect

    def _replacement_connect_without_idle_timeout():
        conn = original_replacement_connect()
        with conn.cursor() as cur:
            cur.execute("SET idle_in_transaction_session_timeout = 0")
        conn.commit()
        return conn

    def _tracked_prune_connect():
        conn = original_prune_connect()
        with conn.cursor() as cur:
            cur.execute("SET lock_timeout = '10s'")
            cur.execute("SELECT pg_backend_pid()")
            prune_backend_pid.append(int(cur.fetchone()[0]))
        conn.commit()
        prune_connection_ready.set()
        return conn

    def _blocking_encrypt(payload, domain):
        entered_serialization.set()
        assert release_serialization.wait(timeout=5)
        return original_encrypt(payload, domain)

    monkeypatch.setattr(
        replacement_manager,
        "_maybe_encrypt_json",
        _blocking_encrypt,
    )
    monkeypatch.setattr(
        replacement_manager,
        "_connect",
        _replacement_connect_without_idle_timeout,
    )
    monkeypatch.setattr(
        prune_manager,
        "_connect",
        _tracked_prune_connect,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        replace_future = executor.submit(
            replacement_manager.replace_job_payload,
            int(job["id"]),
            payload={"version": "replacement"},
            expected_uuid=str(job["uuid"]),
            expected_domain="prompt_studio",
        )
        assert entered_serialization.wait(timeout=5)
        prune_future = executor.submit(
            prune_manager.prune_jobs,
            statuses=["completed"],
            older_than_days=0,
            domain="prompt_studio",
        )
        assert prune_connection_ready.wait(timeout=5)

        observer = psycopg.connect(jobs_pg_dsn, autocommit=True)
        try:
            deadline = time.monotonic() + 5
            prune_is_waiting = False
            while time.monotonic() < deadline:
                with observer.cursor() as cur:
                    cur.execute(
                        """
                        SELECT 1
                        FROM pg_stat_activity
                        WHERE pid = %s
                          AND wait_event_type = 'Lock'
                        LIMIT 1
                        """,
                        (prune_backend_pid[0],),
                    )
                    prune_is_waiting = cur.fetchone() is not None
                if prune_is_waiting:
                    break
                time.sleep(0.02)
            assert prune_is_waiting
        finally:
            observer.close()
            release_serialization.set()

        assert replace_future.result(timeout=5) is True
        assert prune_future.result(timeout=5) == 1

    archived = jm.get_job_or_archived(
        int(job["id"]),
        domain="prompt_studio",
    )
    assert archived is not None
    assert archived["archived"] is True
    assert archived["payload"] == {"version": "replacement"}
