import json
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta

import pytest

from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables
from tldw_Server_API.app.core.Jobs.manager import JobManager


@pytest.fixture()
def jobs_db(tmp_path):
    db_path = tmp_path / "jobs.db"
    ensure_jobs_tables(db_path)
    yield db_path


def test_create_and_acquire_and_complete(jobs_db):


    jm = JobManager(jobs_db)
    job = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={"action": "export", "chatbooks_job_id": "abc"},
        owner_user_id="1",
    )
    assert job["status"] == "queued"

    nextj = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=30, worker_id="w1")
    assert nextj is not None
    assert nextj["status"] == "processing"
    ok = jm.renew_job_lease(int(nextj["id"]), seconds=30)
    assert ok
    ok2 = jm.complete_job(int(nextj["id"]))
    assert ok2
    got = jm.get_job(int(nextj["id"]))
    assert got["status"] == "completed"


def test_update_job_result_merges(jobs_db):
    jm = JobManager(jobs_db)
    job = jm.create_job(
        domain="embeddings",
        queue="default",
        job_type="embeddings_pipeline",
        payload={"media_id": 1},
        owner_user_id="1",
    )
    ok1 = jm.update_job_result(int(job["id"]), result={"total_chunks": 12})
    assert ok1
    ok2 = jm.update_job_result(int(job["id"]), result={"embedding_count": 12})
    assert ok2
    updated = jm.get_job(int(job["id"]))
    assert updated["result"]["total_chunks"] == 12
    assert updated["result"]["embedding_count"] == 12


def test_acquire_decrypts_payload(jobs_db, monkeypatch):


    from tldw_Server_API.app.core.Security.crypto import encrypt_json_blob

    monkeypatch.setenv("JOBS_ENCRYPT_SECURE", "true")
    key = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo0NTY3ODkwMTIzNDU2Nzg5MDEy"[:44]
    monkeypatch.setenv("WORKFLOWS_ARTIFACT_ENC_KEY", key)
    if encrypt_json_blob({"probe": True}) is None:
        pytest.skip("Crypto backend unavailable; skipping encryption test")

    jm = JobManager(jobs_db)
    payload = {"secret": "value", "count": 1}
    job = jm.create_job(domain="secure", queue="default", job_type="t", payload=payload, owner_user_id="1")

    conn = sqlite3.connect(jobs_db)
    try:
        raw = conn.execute("SELECT payload FROM jobs WHERE id = ?", (int(job["id"]),)).fetchone()[0]
    finally:
        conn.close()
    raw_obj = json.loads(raw) if raw else {}
    assert isinstance(raw_obj, dict) and ("_encrypted" in raw_obj or raw_obj.get("_enc") == "aesgcm:v1")

    acq = jm.acquire_next_job(domain="secure", queue="default", lease_seconds=5, worker_id="w1")
    assert acq is not None
    assert acq["payload"] == payload


def test_rotate_encryption_keys_respects_filters_sqlite(jobs_db, monkeypatch):


    from tldw_Server_API.app.core.Security.crypto import encrypt_json_blob

    monkeypatch.setenv("JOBS_ENCRYPT", "true")
    old_key = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo0NTY3ODkwMTIzNDU2Nzg5MDEy"[:44]
    new_key = "MDEyMzQ1Njc4OTAxMjM0NTY3ODkwQUJDREVGR0hJSktMTU5PUFFSU1RVVldY"[:44]
    monkeypatch.setenv("WORKFLOWS_ARTIFACT_ENC_KEY", old_key)
    if encrypt_json_blob({"probe": True}) is None:
        pytest.skip("Crypto backend unavailable; skipping encryption test")

    jm = JobManager(jobs_db)
    jm.create_job(domain="d1", queue="default", job_type="t", payload={"x": 1}, owner_user_id="1")
    jm.create_job(domain="d2", queue="default", job_type="t", payload={"x": 2}, owner_user_id="1")

    count = jm.rotate_encryption_keys(
        domain="d1",
        old_key_b64=old_key,
        new_key_b64=new_key,
        fields=["payload"],
        dry_run=True,
    )
    assert count == 1


def test_retryable_fail_and_backoff(jobs_db):


    jm = JobManager(jobs_db)
    job = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="import",
        payload={"action": "import", "chatbooks_job_id": "xyz"},
        owner_user_id="1",
        max_retries=2,
    )
    j = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=5, worker_id="w2")
    assert j is not None
    # Retryable fail schedules back to queued
    ok = jm.fail_job(int(j["id"]), error="boom", retryable=True, backoff_seconds=1)
    assert ok
    j2 = jm.get_job(int(j["id"]))
    assert j2["status"] in ("queued", "failed")
    if j2["status"] == "queued":
        assert j2["retry_count"] >= 1


def test_cancel_paths(jobs_db):


    jm = JobManager(jobs_db)
    j1 = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    # cancel queued
    ok = jm.cancel_job(int(j1["id"]))
    assert ok
    j1r = jm.get_job(int(j1["id"]))
    assert j1r["status"] == "cancelled"

    # cancel request on processing
    j2 = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    acq = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=5, worker_id="w3")
    assert acq is not None
    ok2 = jm.cancel_job(int(acq["id"]))
    assert ok2
    j2r = jm.get_job(int(acq["id"]))
    # either processing with cancel_requested_at set, or cancelled if race
    assert j2r["status"] in ("processing", "cancelled")
    if j2r["status"] == "processing":
        assert j2r.get("cancel_requested_at") is not None


def test_idempotency_key_returns_existing(jobs_db):


    jm = JobManager(jobs_db)
    idem_key = "cb-export-uniq-key"
    j1 = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={"action": "export"},
        owner_user_id="1",
        idempotency_key=idem_key,
    )
    assert j1["status"] == "queued"
    # Second create with same idempotency key should return the same row
    j2 = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={"action": "export"},
        owner_user_id="1",
        idempotency_key=idem_key,
    )
    assert int(j2["id"]) == int(j1["id"])  # idempotent
    assert j2["status"] == "queued"


def test_available_at_scheduling_delays_acquire(jobs_db):


    jm = JobManager(jobs_db)
    future = datetime.utcnow() + timedelta(seconds=1)
    jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={"action": "export"},
        owner_user_id="1",
        available_at=future,
    )
    # Should not acquire before available_at
    j = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=5, worker_id="w4")
    assert j is None
    # Wait for availability window
    import time as _t
    _t.sleep(1.2)
    j2 = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=5, worker_id="w4")
    assert j2 is not None
    assert j2["status"] == "processing"


def test_create_job_backfills_missing_batch_group(tmp_path, monkeypatch):


    db_path = tmp_path / "jobs_legacy.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE jobs (
              id INTEGER PRIMARY KEY,
              uuid TEXT UNIQUE,
              domain TEXT NOT NULL,
              queue TEXT NOT NULL,
              job_type TEXT NOT NULL,
              owner_user_id TEXT,
              project_id INTEGER,
              idempotency_key TEXT,
              payload TEXT,
              result TEXT,
              status TEXT NOT NULL,
              priority INTEGER DEFAULT 5,
              max_retries INTEGER DEFAULT 3,
              retry_count INTEGER DEFAULT 0,
              available_at TEXT,
              created_at TEXT,
              updated_at TEXT,
              request_id TEXT,
              trace_id TEXT
            );
            CREATE TABLE job_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              job_id INTEGER,
              domain TEXT,
              queue TEXT,
              job_type TEXT,
              event_type TEXT NOT NULL,
              attrs_json TEXT,
              owner_user_id TEXT,
              request_id TEXT,
              trace_id TEXT,
              created_at TEXT NOT NULL DEFAULT (DATETIME('now'))
            );
            """
        )
        conn.commit()
    finally:
        conn.close()

    import tldw_Server_API.app.core.Jobs.manager as jobs_manager

    def _no_migrate(path=None):
        return path if path is not None else db_path

    monkeypatch.setattr(jobs_manager, "ensure_jobs_tables", _no_migrate, raising=True)

    jm = jobs_manager.JobManager(db_path)
    job = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={"action": "export"},
        owner_user_id="1",
    )
    assert job["status"] == "queued"

    conn = sqlite3.connect(db_path)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()}
        assert "batch_group" in cols
    finally:
        conn.close()


def test_count_jobs_backfills_missing_batch_group(tmp_path, monkeypatch):
    db_path = tmp_path / "jobs_legacy_count.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE jobs (
              id INTEGER PRIMARY KEY,
              uuid TEXT UNIQUE,
              domain TEXT NOT NULL,
              queue TEXT NOT NULL,
              job_type TEXT NOT NULL,
              owner_user_id TEXT,
              project_id INTEGER,
              idempotency_key TEXT,
              payload TEXT,
              result TEXT,
              status TEXT NOT NULL,
              priority INTEGER DEFAULT 5,
              max_retries INTEGER DEFAULT 3,
              retry_count INTEGER DEFAULT 0,
              available_at TEXT,
              created_at TEXT,
              updated_at TEXT,
              request_id TEXT,
              trace_id TEXT
            );
            INSERT INTO jobs (
              id, uuid, domain, queue, job_type, owner_user_id, payload, status,
              priority, max_retries, retry_count, created_at, updated_at
            ) VALUES (
              1, 'job-1', 'chatbooks', 'default', 'export', '1', '{}', 'queued',
              5, 3, 0, '2026-04-30 00:00:00', '2026-04-30 00:00:00'
            );
            """
        )
        conn.commit()
    finally:
        conn.close()

    import tldw_Server_API.app.core.Jobs.manager as jobs_manager

    def _no_migrate(path=None):
        return path if path is not None else db_path

    monkeypatch.setattr(jobs_manager, "ensure_jobs_tables", _no_migrate, raising=True)

    jm = jobs_manager.JobManager(db_path)
    assert jm.count_jobs(domain="chatbooks", batch_group="batch-1") == 0

    conn = sqlite3.connect(db_path)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()}
        assert "batch_group" in cols
    finally:
        conn.close()


def test_dependencies_gate_acquire_and_unblock(jobs_db):


    jm = JobManager(jobs_db)
    root = jm.create_job(
        domain="embeddings",
        queue="default",
        job_type="stage_root",
        payload={"step": "root"},
        owner_user_id="1",
        priority=5,
    )
    child = jm.create_job(
        domain="embeddings",
        queue="default",
        job_type="stage_child",
        payload={"step": "child"},
        owner_user_id="1",
        priority=1,
    )
    assert jm.add_job_dependency(child["uuid"], root["uuid"])

    first = jm.acquire_next_job(domain="embeddings", queue="default", lease_seconds=5, worker_id="w1")
    assert first is not None
    assert first["uuid"] == root["uuid"]
    assert jm.complete_job(int(first["id"]))

    second = jm.acquire_next_job(domain="embeddings", queue="default", lease_seconds=5, worker_id="w1")
    assert second is not None
    assert second["uuid"] == child["uuid"]


def test_dependency_failure_cancels_children(jobs_db):


    jm = JobManager(jobs_db)
    root = jm.create_job(
        domain="embeddings",
        queue="default",
        job_type="stage_root",
        payload={"step": "root"},
        owner_user_id="1",
    )
    child = jm.create_job(
        domain="embeddings",
        queue="default",
        job_type="stage_child",
        payload={"step": "child"},
        owner_user_id="1",
    )
    assert jm.add_job_dependency(child["uuid"], root["uuid"])

    first = jm.acquire_next_job(domain="embeddings", queue="default", lease_seconds=5, worker_id="w2")
    assert first is not None
    assert jm.fail_job(int(first["id"]), error="boom", retryable=False)

    child_row = jm.get_job(int(child["id"]))
    assert child_row["status"] == "cancelled"


def test_dependency_cancel_cascades(jobs_db):


    jm = JobManager(jobs_db)
    root = jm.create_job(
        domain="embeddings",
        queue="default",
        job_type="stage_root",
        payload={"step": "root"},
        owner_user_id="1",
    )
    child = jm.create_job(
        domain="embeddings",
        queue="default",
        job_type="stage_child",
        payload={"step": "child"},
        owner_user_id="1",
    )
    assert jm.add_job_dependency(child["uuid"], root["uuid"])

    assert jm.cancel_job(int(root["id"]))
    child_row = jm.get_job(int(child["id"]))
    assert child_row["status"] == "cancelled"


def test_dependency_cycle_rejected(jobs_db):


    jm = JobManager(jobs_db)
    a = jm.create_job(
        domain="embeddings",
        queue="default",
        job_type="stage_a",
        payload={"step": "a"},
        owner_user_id="1",
    )
    b = jm.create_job(
        domain="embeddings",
        queue="default",
        job_type="stage_b",
        payload={"step": "b"},
        owner_user_id="1",
    )
    assert jm.add_job_dependency(b["uuid"], a["uuid"])
    with pytest.raises(ValueError):
        jm.add_job_dependency(a["uuid"], b["uuid"])
