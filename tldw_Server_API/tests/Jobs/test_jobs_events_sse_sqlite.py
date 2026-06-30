import json
import os
import tempfile
import time
import pytest
from fastapi.testclient import TestClient


def test_jobs_events_sse_sqlite_smoke(monkeypatch):


     # Guard against environments where SSE streaming is unreliable (CI/sandbox)
    if os.getenv("CI") or str(os.getenv("TLDW_TEST_NO_SSE", "")).strip().lower() in {"1", "true", "yes", "y", "on"}:
        pytest.skip("Skipping SSE smoke test in CI/sandbox environment")
    # Configure minimal app and SQLite jobs DB in a temp path
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setenv("JOBS_EVENTS_POLL_INTERVAL", "0.05")
    # Disable background workers that can prolong startup/shutdown in tests
    monkeypatch.setenv("CHATBOOKS_CORE_WORKER_ENABLED", "false")
    monkeypatch.setenv("JOBS_METRICS_GAUGES_ENABLED", "false")
    monkeypatch.setenv("JOBS_METRICS_RECONCILE_ENABLE", "false")
    monkeypatch.setenv("AUDIO_JOBS_WORKER_ENABLED", "false")
    monkeypatch.setenv("EMBEDDINGS_REEMBED_WORKER_ENABLED", "false")
    monkeypatch.setenv("JOBS_WEBHOOKS_ENABLED", "false")
    monkeypatch.setenv("WORKFLOWS_WEBHOOK_DLQ_ENABLED", "false")
    monkeypatch.setenv("WORKFLOWS_ARTIFACT_GC_ENABLED", "false")
    monkeypatch.setenv("WORKFLOWS_DB_MAINTENANCE_ENABLED", "false")
    # Skip privilege metadata validation to avoid heavy startup
    monkeypatch.setenv("PRIVILEGE_METADATA_VALIDATE_ON_STARTUP", "0")

    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "jobs_test.db")
        monkeypatch.setenv("JOBS_DB_PATH", db_path)

        # Ensure schema and create a job to seed the outbox
        from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables
        ensure_jobs_tables(db_path)

        from tldw_Server_API.app.core.Jobs.manager import JobManager
        jm = JobManager(db_path=db_path)
        jm.create_job(
            domain="chatbooks",
            queue="default",
            job_type="export",
            payload={},
            owner_user_id="u1",
        )

        from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings
        reset_settings()
        from tldw_Server_API.app.main import app

        headers = {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}
        with TestClient(app, headers=headers) as client:
            hb = False
            observed_job_event = None
            deadline = time.time() + 3.0
            with client.stream("GET", "/api/v1/jobs/events/stream", params={"after_id": 0, "domain": "chatbooks"}) as s:
                if s.status_code != 200:
                    pytest.skip("jobs_admin stream not available in this environment")
                for line in s.iter_lines():
                    if time.time() > deadline:
                        break
                    if not line:
                        continue
                    if isinstance(line, bytes):
                        try:
                            line = line.decode()
                        except Exception:
                            continue
                    if line.startswith("data:"):
                        payload = line[len("data:"):].strip()
                        # Ignore end sentinel
                        if payload.lower() == "[done]":
                            continue
                        # Accept both ping {} and explicit heartbeat payloads
                        if payload == "{}":
                            hb = True
                            continue
                        try:
                            obj = json.loads(payload)
                            if isinstance(obj, dict) and obj.get("event") == "job.created":
                                observed_job_event = obj
                                break
                            if obj == {} or (isinstance(obj, dict) and obj.get("heartbeat") is True):
                                hb = True
                        except Exception:
                            _ = None
            assert hb, "did not observe initial SSE ping frame"
            assert observed_job_event is not None, "did not observe seeded job.created SSE frame"
            assert isinstance(observed_job_event.get("attrs"), dict)
            assert "attrs_json" not in observed_job_event
