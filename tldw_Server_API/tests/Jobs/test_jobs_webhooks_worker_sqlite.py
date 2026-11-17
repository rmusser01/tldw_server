import os
import json
import asyncio
import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager


def _env(monkeypatch, tmp_path):


    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("JOBS_DB_PATH", os.path.join(os.getcwd(), "Databases", "jobs.db"))
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setenv("JOBS_WEBHOOKS_ENABLED", "true")
    monkeypatch.setenv("JOBS_WEBHOOKS_URL", "http://example.test/hook")
    # The worker enforces the egress policy in every mode; allow the stub host explicitly.
    monkeypatch.setenv("WORKFLOWS_EGRESS_ALLOWLIST", "example.test")
    monkeypatch.setenv("JOBS_WEBHOOKS_SECRET_KEYS", "devsecret")
    monkeypatch.setenv("JOBS_WEBHOOKS_INTERVAL_SEC", "0.01")


@pytest.mark.asyncio
async def test_jobs_webhooks_worker_emits_signed_event_sqlite(monkeypatch, tmp_path):
    _env(monkeypatch, tmp_path)
    from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables
    ensure_jobs_tables(tmp_path / "jobs.db")

    # Seed a completed job to generate job.completed outbox row
    jm = JobManager()
    j = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="u")
    acq = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=1, worker_id="w")
    assert acq
    # Complete with lease enforcement parameters to ensure outbox event emission
    jm.complete_job(int(acq["id"]), worker_id=acq["worker_id"], lease_id=acq["lease_id"])  # type: ignore[index]

    # Assert exactly one job.completed outbox row for this job
    import sqlite3
    db_path = tmp_path / "Databases" / "jobs.db"
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "SELECT COUNT(*) FROM job_events WHERE job_id = ? AND event_type = 'job.completed'",
            (int(j["id"]),),
        )
        cnt = int(cur.fetchone()[0])
        assert cnt == 1, f"expected 1 job.completed event, found {cnt}"

    # Capture outgoing webhook requests
    sent = {"count": 0, "headers": None, "body": None}

    class _Resp:
        status_code = 200
        text = "ok"
        async def aclose(self):
            return None

    # Fix timestamp used for signature
    import time as _time
    monkeypatch.setattr(_time, "time", lambda: 1700000000)

    # Monkeypatch async fetch to return our stub response
    import tldw_Server_API.app.services.jobs_webhooks_service as svc
    async def _fake_afetch(*, method, url, headers=None, data=None, **kwargs):
        sent["count"] += 1
        sent["headers"] = headers
        sent["body"] = data
        return _Resp()
    monkeypatch.setattr(svc, "afetch", _fake_afetch)

    stop = asyncio.Event()
    # Run worker briefly
    task = asyncio.create_task(svc.run_jobs_webhooks_worker(stop))
    # Allow a few iterations
    await asyncio.sleep(0.05)
    stop.set()
    await asyncio.wait_for(task, timeout=1.0)

    # Validate a request was sent with expected headers and signature
    assert sent["count"] >= 1
    assert sent["headers"]["X-Jobs-Event"] == "job.completed"
    ts = sent["headers"].get("X-Jobs-Timestamp")
    sig = sent["headers"].get("X-Jobs-Signature")
    assert ts and sig and sig.startswith("v1=")

    # Verify signature round-trip via helper implementation
    from Helper_Scripts.tldw_jobs import _verify_sig
    # Build args object
    class _Args:
        def __init__(self, timestamp, signature, secrets, body):
            self.timestamp = timestamp
            self.signature = signature
            self.secrets = secrets
            self.body = body
    # Write body to temp file for the CLI helper
    body_path = tmp_path / "body.json"
    body_path.write_bytes(sent["body"])  # type: ignore
    args = _Args(ts, sig, "devsecret", str(body_path))
    # Should not raise SystemExit
    _verify_sig(args)


@pytest.mark.asyncio
async def test_webhooks_cursor_persist_and_resume_sqlite(monkeypatch, tmp_path):
    # Enable outbox + webhooks
    _env(monkeypatch, tmp_path)
    monkeypatch.setenv("JOBS_WEBHOOKS_URL", "http://example.test/hook")
    monkeypatch.setenv("JOBS_WEBHOOKS_SECRET_KEYS", "devsecret")
    monkeypatch.setenv("JOBS_WEBHOOKS_INTERVAL_SEC", "0.01")
    # Persist cursor to a tmp file
    cursor_file = tmp_path / "cursor.txt"
    monkeypatch.setenv("JOBS_WEBHOOKS_CURSOR_PATH", str(cursor_file))

    from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables
    ensure_jobs_tables(tmp_path / "jobs.db")
    jm = JobManager()

    # Seed an event
    j = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="u")
    acq = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=1, worker_id="w")
    assert acq
    jm.complete_job(int(acq["id"]), worker_id=acq["worker_id"], lease_id=acq["lease_id"])  # type: ignore[index]

    sent = {"ids": []}

    class _Resp:
        status_code = 200
        text = "ok"
        async def aclose(self):
            return None

    import tldw_Server_API.app.services.jobs_webhooks_service as svc
    async def _fake_afetch(*, method, url, headers=None, data=None, **kwargs):
        try:
            sent["ids"].append(int(headers.get("X-Jobs-Event-Id")))
        except Exception:
            _ = None
        return _Resp()
    monkeypatch.setattr(svc, "afetch", _fake_afetch)

    stop = asyncio.Event()
    task = asyncio.create_task(svc.run_jobs_webhooks_worker(stop))
    await asyncio.sleep(0.05)
    stop.set()
    await asyncio.wait_for(task, timeout=1.0)

    # Cursor file should exist with the last processed id
    assert cursor_file.exists()
    fid = int((cursor_file.read_text() or "0").strip() or 0)
    assert any(i >= fid for i in sent["ids"]) or fid in sent["ids"]
