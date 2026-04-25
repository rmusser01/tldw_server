from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
import pytest

from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.services import workflows_webhook_dlq_service as dlq_mod


pytestmark = pytest.mark.integration


def test_host_allow_deny_logic(monkeypatch):


     # Global allow/deny lists
    monkeypatch.setenv("WORKFLOWS_WEBHOOK_ALLOWLIST", "*.ok.test,allowed.example")
    monkeypatch.setenv("WORKFLOWS_WEBHOOK_DENYLIST", "deny.test,*.blocked.tld")
    assert dlq_mod._host_allowed("https://foo.ok.test/h", "default") is True
    assert dlq_mod._host_allowed("https://allowed.example/h", "default") is True
    assert dlq_mod._host_allowed("https://deny.test/h", "default") is False
    assert dlq_mod._host_allowed("https://x.blocked.tld/h", "default") is False


def test_host_allow_deny_logic_uses_test_mode_single_letter_y_fallback(monkeypatch):
    monkeypatch.setenv("WORKFLOWS_WEBHOOK_ALLOWLIST", "*.ok.test")
    monkeypatch.setenv("WORKFLOWS_WEBHOOK_DENYLIST", "")
    monkeypatch.setenv("TEST_MODE", "y")

    monkeypatch.setattr(dlq_mod, "is_explicit_pytest_runtime", lambda: False, raising=True)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Security.egress.evaluate_url_policy",
        lambda *_args, **_kwargs: SimpleNamespace(allowed=False, reason="blocked"),
        raising=True,
    )

    assert dlq_mod._host_allowed("https://foo.ok.test/h", "default") is True


@pytest.mark.asyncio
async def test_dlq_worker_backoff_and_delivery(monkeypatch, tmp_path):
    # Use SQLite DB for worker loop; behavior is the same for DLQ table mechanics
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    # Enqueue a row due now
    db.enqueue_webhook_dlq(
        tenant_id="default",
        run_id="runX",
        url="https://post.test/hook",
        body={"ok": True},
        last_error="init",
    )

    # Force allow host
    monkeypatch.setenv("WORKFLOWS_WEBHOOK_ALLOWLIST", "post.test")
    # Minimize loop interval
    monkeypatch.setenv("WORKFLOWS_WEBHOOK_DLQ_INTERVAL_SEC", "1")
    monkeypatch.setenv("WORKFLOWS_WEBHOOK_DLQ_BATCH", "10")
    monkeypatch.setenv("WORKFLOWS_WEBHOOK_DLQ_TIMEOUT_SEC", "1")

    # Monkeypatch list_webhook_dlq_due to pull from our db
    monkeypatch.setattr(dlq_mod, "create_workflows_database", lambda backend=None: db)
    monkeypatch.setattr(dlq_mod, "get_content_backend_instance", lambda: None)

    # Stub afetch behavior: first call fails, second succeeds
    class DummyResp:
        def __init__(self, status_code=200):
            self.status_code = status_code
            self.text = "ok"
        async def aclose(self):
            return None

    async def _fake_afetch(*, method, url, json=None, timeout=None, **kwargs):
        _fake_afetch.calls += 1
        if _fake_afetch.calls == 1:
            return DummyResp(500)
        return DummyResp(200)
    _fake_afetch.calls = 0
    monkeypatch.setattr(dlq_mod, "afetch", _fake_afetch)
    # Fix backoff to 1 second to make assertion deterministic
    monkeypatch.setattr(dlq_mod, "_compute_next_backoff", lambda attempts: 1)

    # Run worker for a couple of cycles
    stop = asyncio.Event()
    task = asyncio.create_task(dlq_mod.run_workflows_webhook_dlq_worker(stop))
    # Allow first cycle
    await asyncio.sleep(0.2)
    # Fetch row and assert attempts incremented and next_attempt_at set
    rows = db.list_webhook_dlq_due(limit=10)
    if rows:
        r = rows[0]
        assert int(r.get("attempts", 0)) >= 1
    # Allow second cycle to deliver and delete
    await asyncio.sleep(1.2)
    stop.set()
    try:
        await asyncio.wait_for(task, timeout=2)
    except asyncio.TimeoutError:
        task.cancel()
    # Ensure DLQ is drained after successful retry
    rows2 = db.list_webhook_dlq_due(limit=10)
    assert not rows2, f"Expected DLQ to be empty, found: {rows2}"


@pytest.mark.asyncio
async def test_dlq_worker_stops_retrying_after_max_attempts(monkeypatch, tmp_path):
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    db.enqueue_webhook_dlq(
        tenant_id="default",
        run_id="run-max-attempts",
        url="https://post.test/hook",
        body={"ok": True},
        last_error="init",
    )
    rows = db.list_webhook_dlq_all(limit=10)
    assert rows, "expected DLQ seed row"
    dlq_id = rows[0]["id"]
    db.update_webhook_dlq_failure(
        dlq_id=dlq_id,
        last_error="retryable",
        next_attempt_at_iso=None,
        attempts=1,
    )

    monkeypatch.setenv("WORKFLOWS_WEBHOOK_ALLOWLIST", "post.test")
    monkeypatch.setenv("WORKFLOWS_WEBHOOK_DLQ_INTERVAL_SEC", "1")
    monkeypatch.setenv("WORKFLOWS_WEBHOOK_DLQ_BATCH", "10")
    monkeypatch.setenv("WORKFLOWS_WEBHOOK_DLQ_TIMEOUT_SEC", "1")
    monkeypatch.setenv("WORKFLOWS_WEBHOOK_DLQ_MAX_ATTEMPTS", "1")

    monkeypatch.setattr(dlq_mod, "create_workflows_database", lambda backend=None: db)
    monkeypatch.setattr(dlq_mod, "get_content_backend_instance", lambda: None)

    calls = {"count": 0}

    async def _fake_afetch(*args, **kwargs):  # noqa: ANN003
        calls["count"] += 1
        raise AssertionError("delivery should not be attempted after max attempts")

    monkeypatch.setattr(dlq_mod, "afetch", _fake_afetch)

    stop = asyncio.Event()
    task = asyncio.create_task(dlq_mod.run_workflows_webhook_dlq_worker(stop))
    await asyncio.sleep(0.2)
    stop.set()
    try:
        await asyncio.wait_for(task, timeout=2)
    except asyncio.TimeoutError:
        task.cancel()

    assert calls["count"] == 0
    due_rows = db.list_webhook_dlq_due(limit=10)
    assert not due_rows
    all_rows = db.list_webhook_dlq_all(limit=10)
    assert all_rows
    assert "max_attempts" in str(all_rows[0].get("last_error", "")).lower()
