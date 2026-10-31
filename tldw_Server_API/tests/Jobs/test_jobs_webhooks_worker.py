import os
import json
import asyncio
import hmac
import hashlib
from pathlib import Path
import contextlib

import pytest


pytestmark = pytest.mark.jobs


def _set_base_env(monkeypatch, tmp_path: Path):
    # Core test-mode and single-user defaults
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.delenv("SINGLE_USER_API_KEY", raising=False)
    # Jobs DB under tmpdir
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "Databases" / "jobs.db"))
    # Webhooks worker configuration
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    monkeypatch.setenv("JOBS_WEBHOOKS_ENABLED", "true")
    monkeypatch.setenv("JOBS_WEBHOOKS_URL", "http://127.0.0.1/webhook")  # loopback OK in TEST_MODE
    monkeypatch.setenv("JOBS_WEBHOOKS_SECRET_KEYS", "testsecret,oldsecret")
    monkeypatch.setenv("JOBS_WEBHOOKS_INTERVAL_SEC", "0.01")
    monkeypatch.setenv("JOBS_WEBHOOKS_TIMEOUT_SEC", "1.0")
    # Persist cursor to a test-specific path to allow resume in TEST_MODE
    monkeypatch.setenv("JOBS_WEBHOOKS_CURSOR_PATH", str(tmp_path / "Databases" / "jobs_webhooks_cursor.txt"))


@pytest.mark.asyncio
async def test_webhooks_signed_and_cursor_resume(monkeypatch, tmp_path):
    _set_base_env(monkeypatch, tmp_path)

    # Prepare a mock transport that validates the signature and captures deliveries
    delivered = []

    class _Resp:
        status_code = 200
        text = "ok"
        async def aclose(self):
            return None

    async def _fake_afetch(*, method, url, headers=None, data=None, **kwargs):
        assert method == "POST"
        # Validate headers
        ts = headers.get("X-Jobs-Timestamp")
        sig = headers.get("X-Jobs-Signature")
        et = headers.get("X-Jobs-Event")
        assert et in {"job.completed", "job.failed"}
        assert sig and sig.startswith("v1=")
        body = data or b""
        # Verify HMAC: HMAC(secret, f"{ts}.{body}")
        secret = "testsecret".encode("utf-8")
        expected = hmac.new(secret, (ts.encode("utf-8") + b"." + body), hashlib.sha256).hexdigest()
        assert sig == f"v1={expected}"
        delivered.append({
            "ts": ts,
            "sig": sig,
            "event": et,
            "body": json.loads(body.decode("utf-8")),
        })
        return _Resp()

    from tldw_Server_API.app.services import jobs_webhooks_service as svc
    monkeypatch.setattr(svc, "afetch", _fake_afetch)

    # Create events: one completed now, one to be created after first run
    from tldw_Server_API.app.core.Jobs.manager import JobManager
    jm = JobManager()

    j1 = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    acq1 = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=60, worker_id="w1")
    assert acq1 is not None
    ok1 = jm.complete_job(int(acq1["id"]), result={"ok": True}, enforce=False)
    assert ok1

    # Run worker for a short period to pick the first event
    stop_event = asyncio.Event()
    task = asyncio.create_task(svc.run_jobs_webhooks_worker(stop_event=stop_event))
    try:
        # Wait until at least one delivery is observed or timeout
        for _ in range(200):
            if delivered:
                break
            await asyncio.sleep(0.01)
        assert delivered, "expected at least one delivered webhook"
        # Stop worker
        stop_event.set()
        await asyncio.wait_for(task, timeout=2.0)
    finally:
        if not task.done():
            stop_event.set()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(task, timeout=2.0)

    # Cursor file should be persisted with last outbox id
    cursor_path = Path(os.getenv("JOBS_WEBHOOKS_CURSOR_PATH"))
    assert cursor_path.exists()
    first_after = int(cursor_path.read_text().strip() or "0")
    assert first_after > 0

    # Add another event after stopping the worker
    j2 = jm.create_job(domain="chatbooks", queue="default", job_type="export", payload={}, owner_user_id="1")
    acq2 = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=60, worker_id="w2")
    assert acq2 is not None
    ok2 = jm.complete_job(int(acq2["id"]), result={"ok": True}, enforce=False)
    assert ok2

    # Clear deliveries and run worker again; it should resume from cursor and send only the new event
    delivered.clear()
    stop_event2 = asyncio.Event()
    task2 = asyncio.create_task(svc.run_jobs_webhooks_worker(stop_event=stop_event2))
    try:
        for _ in range(200):
            if delivered:
                break
            await asyncio.sleep(0.01)
        assert delivered, "expected resumed worker to deliver second webhook"
        stop_event2.set()
        await asyncio.wait_for(task2, timeout=2.0)
    finally:
        if not task2.done():
            stop_event2.set()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(task2, timeout=2.0)

    # Cursor should advance
    second_after = int(cursor_path.read_text().strip() or "0")
    assert second_after > first_after


@pytest.mark.asyncio
async def test_webhooks_enabled_without_outbox_refuses_before_database_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_base_env(monkeypatch, tmp_path)
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "false")

    from tldw_Server_API.app.services import jobs_webhooks_service as svc

    manager_calls: list[str] = []

    class _UnexpectedJobManager:
        @staticmethod
        def set_rls_context(**_kwargs) -> None:
            manager_calls.append("set_rls_context")

        def __init__(self) -> None:
            manager_calls.append("init")

    monkeypatch.setattr(svc, "JobManager", _UnexpectedJobManager)
    stop_event = asyncio.Event()
    stop_event.set()

    await svc.run_jobs_webhooks_worker(stop_event=stop_event)

    assert manager_calls == []


@pytest.mark.asyncio
async def test_webhooks_tldw_test_mode_y_skips_egress_policy_check(monkeypatch, tmp_path):
    _set_base_env(monkeypatch, tmp_path)
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.setenv("TLDW_TEST_MODE", "y")

    egress_calls = {"count": 0}

    from tldw_Server_API.app.core.Security import egress as egress_mod

    def _fake_eval_policy(url):  # noqa: ANN001
        egress_calls["count"] += 1

        class _Denied:
            allowed = False
            reason = "blocked-by-test"

        return _Denied()

    monkeypatch.setattr(egress_mod, "evaluate_url_policy", _fake_eval_policy)

    from tldw_Server_API.app.services import jobs_webhooks_service as svc

    stop_event = asyncio.Event()
    stop_event.set()
    await svc.run_jobs_webhooks_worker(stop_event=stop_event)

    assert egress_calls["count"] == 0
