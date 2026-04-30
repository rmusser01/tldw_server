from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.Meetings_DB import MeetingsDatabase


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_meetings_webhook_dlq_worker_retries_then_delivers(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.Meetings import integration_service as integration_mod
    from tldw_Server_API.app.services import meetings_webhook_dlq_service as worker_mod

    db_path = tmp_path / "Media_DB_v2.db"
    db = MeetingsDatabase(db_path=db_path, client_id="meetings-worker-test", user_id="1")
    try:
        session_id = db.create_session(title="DLQ", meeting_type="standup")
        db.create_artifact(
            session_id=session_id,
            kind="summary",
            format="json",
            payload_json={"text": "Artifact payload"},
            version=1,
        )

        monkeypatch.setattr(
            integration_mod,
            "evaluate_url_policy",
            lambda *_args, **_kwargs: SimpleNamespace(allowed=True, reason=None),
            raising=True,
        )
        service = integration_mod.MeetingIntegrationService(db=db)
        dispatch = service.queue_dispatch(
            session_id=session_id,
            integration_type="webhook",
            webhook_url="https://hooks.example.test/meeting",
            artifact_ids=[],
        )

        monkeypatch.setenv("MEETINGS_WEBHOOK_DLQ_ENABLED", "true")
        monkeypatch.setenv("MEETINGS_WEBHOOK_DLQ_INTERVAL_SEC", "1")
        monkeypatch.setenv("MEETINGS_WEBHOOK_DLQ_BATCH", "20")
        monkeypatch.setenv("MEETINGS_WEBHOOK_DLQ_TIMEOUT_SEC", "1")
        monkeypatch.setenv("MEETINGS_WEBHOOK_DLQ_MAX_ATTEMPTS", "5")
        monkeypatch.setenv("MEETINGS_WEBHOOK_DLQ_BASE_SEC", "1")
        monkeypatch.setenv("MEETINGS_WEBHOOK_DLQ_MAX_BACKOFF_SEC", "2")

        monkeypatch.setattr(
            worker_mod,
            "discover_meetings_db_targets",
            lambda: [(db_path, "1")],
            raising=True,
        )
        monkeypatch.setattr(
            worker_mod,
            "evaluate_url_policy",
            lambda *_args, **_kwargs: SimpleNamespace(allowed=True, reason=None),
            raising=True,
        )
        monkeypatch.setattr(worker_mod, "_compute_next_backoff", lambda _attempts: 1, raising=True)

        class _Resp:
            def __init__(self, status_code: int):
                self.status_code = status_code
                self.text = "ok"

            async def aclose(self):
                return None

        async def _fake_afetch(*, method, url, json=None, timeout=None, retry=None, **_kwargs):
            _fake_afetch.calls += 1
            if _fake_afetch.calls == 1:
                return _Resp(status_code=500)
            return _Resp(status_code=200)

        _fake_afetch.calls = 0
        monkeypatch.setattr(worker_mod, "afetch", _fake_afetch, raising=True)

        stop_event = asyncio.Event()
        task = asyncio.create_task(worker_mod.run_meetings_webhook_dlq_worker(stop_event))
        await asyncio.sleep(0.25)

        after_first_attempt = db.get_integration_dispatch(dispatch_id=int(dispatch["id"]))
        assert after_first_attempt is not None
        assert int(after_first_attempt.get("attempts") or 0) >= 1

        await asyncio.sleep(1.25)
        stop_event.set()
        await asyncio.wait_for(task, timeout=2.0)

        delivered = db.get_integration_dispatch(dispatch_id=int(dispatch["id"]))
        assert delivered is not None
        assert delivered.get("status") == "delivered"
        assert int(delivered.get("attempts") or 0) >= 2
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_meetings_attempt_dispatch_sanitizes_network_errors(monkeypatch):
    from tldw_Server_API.app.services import meetings_webhook_dlq_service as worker_mod

    monkeypatch.setattr(
        worker_mod,
        "evaluate_url_policy",
        lambda *_args, **_kwargs: SimpleNamespace(allowed=True, reason=None),
        raising=True,
    )

    async def _raise_network_error(*_args, **_kwargs):
        raise RuntimeError("meeting webhook token at /private/meeting-webhook.key")

    monkeypatch.setattr(worker_mod, "afetch", _raise_network_error, raising=True)

    ok, error, response = await worker_mod._attempt_dispatch(
        dispatch_row={
            "payload_json": {
                "destination": {"url": "https://hooks.example.test/meeting"},
                "request_body": {"ok": True},
            }
        },
        timeout_sec=1.0,
    )

    assert ok is False
    assert response is None
    assert error == "Meeting webhook delivery failed"
    assert "meeting webhook token" not in error
    assert "/private/meeting-webhook.key" not in error


@pytest.mark.asyncio
async def test_meetings_attempt_dispatch_preserves_timeout_classification(monkeypatch):
    from tldw_Server_API.app.services import meetings_webhook_dlq_service as worker_mod

    monkeypatch.setattr(
        worker_mod,
        "evaluate_url_policy",
        lambda *_args, **_kwargs: SimpleNamespace(allowed=True, reason=None),
        raising=True,
    )

    async def _raise_timeout(*_args, **_kwargs):
        raise TimeoutError("timed out at /private/meeting-webhook-timeout.key")

    monkeypatch.setattr(worker_mod, "afetch", _raise_timeout, raising=True)

    ok, error, response = await worker_mod._attempt_dispatch(
        dispatch_row={
            "payload_json": {
                "destination": {"url": "https://hooks.example.test/meeting"},
                "request_body": {"ok": True},
            }
        },
        timeout_sec=1.0,
    )

    assert ok is False
    assert response is None
    assert error == "Meeting webhook delivery timed out"
    assert "timed out at" not in error
    assert "/private/meeting-webhook-timeout.key" not in error
