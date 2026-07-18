import json
import threading
from contextlib import contextmanager

import pytest

from tldw_Server_API.app.core.Claims_Extraction import claims_job_handlers
from tldw_Server_API.app.core.Claims_Extraction.claims_job_contracts import (
    CLAIMS_DELIVER_ALERT_JOB_TYPE,
    CLAIMS_DELIVER_REVIEW_NOTIFICATION_JOB_TYPE,
    CLAIMS_REBUILD_MEDIA_JOB_TYPE,
    ClaimsJobError,
)

pytestmark = pytest.mark.unit


async def test_rebuild_handler_uses_owner_db_path_and_returns_result(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        claims_job_handlers,
        "get_user_media_db_path",
        lambda owner: f"/tmp/user-{owner}/Media_DB_v2.db",
    )
    monkeypatch.setattr(
        claims_job_handlers,
        "rebuild_claims_for_media",
        lambda **kwargs: calls.append(kwargs) or {"outcome": "ok", "media_id": 42, "deleted": 1, "inserted": 2},
    )

    result = await claims_job_handlers.process_claims_job(
        {
            "id": 1,
            "job_type": CLAIMS_REBUILD_MEDIA_JOB_TYPE,
            "owner_user_id": "7",
            "payload": {"version": 1, "owner_user_id": "7", "media_id": 42},
        }
    )

    assert result["outcome"] == "ok"
    assert calls == [{"db_path": "/tmp/user-7/Media_DB_v2.db", "media_id": 42}]


async def test_rebuild_handler_runs_sync_work_off_event_loop_thread(monkeypatch) -> None:
    event_loop_thread = threading.get_ident()
    handler_threads: list[int] = []

    monkeypatch.setattr(
        claims_job_handlers,
        "get_user_media_db_path",
        lambda owner: f"/tmp/user-{owner}/Media_DB_v2.db",
    )
    monkeypatch.setattr(
        claims_job_handlers,
        "rebuild_claims_for_media",
        lambda **_kwargs: handler_threads.append(threading.get_ident()) or {"outcome": "ok", "media_id": 42},
    )

    result = await claims_job_handlers.process_claims_job(
        {
            "id": 1,
            "job_type": CLAIMS_REBUILD_MEDIA_JOB_TYPE,
            "owner_user_id": "7",
            "payload": {"version": 1, "owner_user_id": "7", "media_id": 42},
        }
    )

    assert result["outcome"] == "ok"
    assert handler_threads and handler_threads[0] != event_loop_thread


async def test_handler_rejects_owner_mismatch() -> None:
    with pytest.raises(ClaimsJobError) as excinfo:
        await claims_job_handlers.process_claims_job(
            {
                "id": 1,
                "job_type": CLAIMS_REBUILD_MEDIA_JOB_TYPE,
                "owner_user_id": "8",
                "payload": {"version": 1, "owner_user_id": "7", "media_id": 42},
            }
        )

    assert excinfo.value.retryable is False
    assert excinfo.value.failure_code == "claims_owner_scope_violation"


async def test_handler_rejects_missing_row_owner() -> None:
    with pytest.raises(ClaimsJobError) as excinfo:
        await claims_job_handlers.process_claims_job(
            {
                "id": 1,
                "job_type": CLAIMS_REBUILD_MEDIA_JOB_TYPE,
                "owner_user_id": "",
                "payload": {"version": 1, "owner_user_id": "7", "media_id": 42},
            }
        )

    assert excinfo.value.retryable is False
    assert excinfo.value.failure_code == "claims_owner_scope_violation"


async def test_handler_rejects_noncanonical_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        claims_job_handlers,
        "rebuild_claims_for_media",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError(f"rebuild should not run: {kwargs}")),
    )

    with pytest.raises(ClaimsJobError) as excinfo:
        await claims_job_handlers.process_claims_job(
            {
                "id": 1,
                "job_type": CLAIMS_REBUILD_MEDIA_JOB_TYPE,
                "owner_user_id": "007",
                "payload": {"version": 1, "owner_user_id": "7", "media_id": 42},
            }
        )

    assert excinfo.value.retryable is False
    assert excinfo.value.failure_code == "claims_owner_scope_violation"


async def test_review_notification_delivery_failure_is_retryable(monkeypatch) -> None:
    monkeypatch.setattr(
        claims_job_handlers,
        "get_user_media_db_path",
        lambda owner: f"/tmp/user-{owner}/Media_DB_v2.db",
    )
    monkeypatch.setattr(
        claims_job_handlers,
        "deliver_claim_review_notifications_now",
        lambda **_kwargs: {"outcome": "failed", "reason": "delivery_failed"},
    )

    with pytest.raises(ClaimsJobError) as excinfo:
        await claims_job_handlers.process_claims_job(
            {
                "id": 1,
                "job_type": CLAIMS_DELIVER_REVIEW_NOTIFICATION_JOB_TYPE,
                "owner_user_id": "7",
                "payload": {"version": 1, "owner_user_id": "7", "notification_ids": [5]},
            }
        )

    assert excinfo.value.retryable is True
    assert excinfo.value.failure_code == "claims_review_notification_delivery_failed"


async def test_review_notification_handler_runs_sync_work_off_event_loop_thread(monkeypatch) -> None:
    event_loop_thread = threading.get_ident()
    handler_threads: list[int] = []

    monkeypatch.setattr(
        claims_job_handlers,
        "get_user_media_db_path",
        lambda owner: f"/tmp/user-{owner}/Media_DB_v2.db",
    )
    monkeypatch.setattr(
        claims_job_handlers,
        "deliver_claim_review_notifications_now",
        lambda **_kwargs: handler_threads.append(threading.get_ident()) or {"outcome": "ok", "notification_ids": [5]},
    )

    result = await claims_job_handlers.process_claims_job(
        {
            "id": 1,
            "job_type": CLAIMS_DELIVER_REVIEW_NOTIFICATION_JOB_TYPE,
            "owner_user_id": "7",
            "payload": {"version": 1, "owner_user_id": "7", "notification_ids": [5]},
        }
    )

    assert result["outcome"] == "ok"
    assert handler_threads and handler_threads[0] != event_loop_thread


async def test_alert_delivery_uses_existing_db_and_preserves_slack_payload(monkeypatch) -> None:
    open_kwargs: dict[str, object] = {}
    delivered: list[dict[str, object]] = []

    class _Db:
        def get_claims_monitoring_event(self, event_id: int) -> dict[str, object]:
            assert event_id == 9
            return {
                "id": 9,
                "user_id": "7",
                "payload_json": json.dumps({"window_ratio": 0.42, "threshold": 0.25, "baseline_ratio": 0.10}),
            }

        def get_claims_monitoring_alert(self, alert_id: int) -> dict[str, object]:
            assert alert_id == 3
            return {
                "id": 3,
                "user_id": "7",
                "enabled": True,
                "channels_json": json.dumps({"slack": True}),
                "slack_webhook_url": "https://example.test/slack",
            }

        def has_successful_claims_monitoring_event_delivery(self, **_kwargs) -> bool:
            return False

    @contextmanager
    def _fake_managed_media_database(*_args, **kwargs):
        open_kwargs.update(kwargs)
        yield _Db()

    monkeypatch.setattr(
        claims_job_handlers, "get_user_media_db_path", lambda owner: f"/tmp/user-{owner}/Media_DB_v2.db"
    )
    monkeypatch.setattr(claims_job_handlers, "managed_media_database", _fake_managed_media_database)
    monkeypatch.setattr(
        claims_job_handlers,
        "deliver_claims_alert_webhook",
        lambda **kwargs: delivered.append(kwargs) or True,
    )

    result = await claims_job_handlers.process_claims_job(
        {
            "id": 1,
            "job_type": CLAIMS_DELIVER_ALERT_JOB_TYPE,
            "owner_user_id": "7",
            "payload": {"version": 1, "owner_user_id": "7", "event_id": 9, "alert_id": 3, "channel": "slack"},
        }
    )

    assert result["outcome"] == "ok"
    assert open_kwargs["initialize"] is False
    assert (
        delivered[0]["payload"]["text"] == "Claims alert: unsupported ratio 42.00% (threshold 25.00%, baseline 10.00%)"
    )


async def test_alert_delivery_handler_runs_sync_work_off_event_loop_thread(monkeypatch) -> None:
    event_loop_thread = threading.get_ident()
    handler_threads: list[int] = []

    monkeypatch.setattr(
        claims_job_handlers,
        "_deliver_alert",
        lambda _payload: handler_threads.append(threading.get_ident()) or {"outcome": "ok", "alert_id": 3},
    )

    result = await claims_job_handlers.process_claims_job(
        {
            "id": 1,
            "job_type": CLAIMS_DELIVER_ALERT_JOB_TYPE,
            "owner_user_id": "7",
            "payload": {"version": 1, "owner_user_id": "7", "event_id": 9, "alert_id": 3, "channel": "slack"},
        }
    )

    assert result["outcome"] == "ok"
    assert handler_threads and handler_threads[0] != event_loop_thread


async def test_alert_delivery_skips_event_owner_mismatch(monkeypatch) -> None:
    class _Db:
        def get_claims_monitoring_event(self, event_id: int) -> dict[str, object]:
            assert event_id == 9
            return {"id": 9, "user_id": "8", "payload_json": "{}"}

        def get_claims_monitoring_alert(self, alert_id: int) -> dict[str, object]:
            raise AssertionError(f"alert should not be loaded after event mismatch: {alert_id}")

    @contextmanager
    def _fake_managed_media_database(*_args, **_kwargs):
        yield _Db()

    monkeypatch.setattr(claims_job_handlers, "get_user_media_db_path", lambda owner: f"/tmp/user-{owner}/Media_DB_v2.db")
    monkeypatch.setattr(claims_job_handlers, "managed_media_database", _fake_managed_media_database)

    result = await claims_job_handlers.process_claims_job(
        {
            "id": 1,
            "job_type": CLAIMS_DELIVER_ALERT_JOB_TYPE,
            "owner_user_id": "7",
            "payload": {"version": 1, "owner_user_id": "7", "event_id": 9, "alert_id": 3, "channel": "slack"},
        }
    )

    assert result == {"outcome": "skipped", "reason": "event_missing", "event_id": 9}


async def test_alert_delivery_skips_alert_owner_mismatch(monkeypatch) -> None:
    class _Db:
        def get_claims_monitoring_event(self, event_id: int) -> dict[str, object]:
            assert event_id == 9
            return {"id": 9, "user_id": "7", "payload_json": "{}"}

        def get_claims_monitoring_alert(self, alert_id: int) -> dict[str, object]:
            assert alert_id == 3
            return {"id": 3, "user_id": "8", "enabled": True, "channels_json": json.dumps({"slack": True})}

    @contextmanager
    def _fake_managed_media_database(*_args, **_kwargs):
        yield _Db()

    monkeypatch.setattr(claims_job_handlers, "get_user_media_db_path", lambda owner: f"/tmp/user-{owner}/Media_DB_v2.db")
    monkeypatch.setattr(claims_job_handlers, "managed_media_database", _fake_managed_media_database)

    result = await claims_job_handlers.process_claims_job(
        {
            "id": 1,
            "job_type": CLAIMS_DELIVER_ALERT_JOB_TYPE,
            "owner_user_id": "7",
            "payload": {"version": 1, "owner_user_id": "7", "event_id": 9, "alert_id": 3, "channel": "slack"},
        }
    )

    assert result == {"outcome": "skipped", "reason": "alert_missing", "alert_id": 3}


async def test_alert_delivery_skips_already_delivered(monkeypatch) -> None:
    class _Db:
        def get_claims_monitoring_event(self, event_id: int) -> dict[str, object]:
            assert event_id == 9
            return {"id": 9, "user_id": "7", "payload_json": "{}"}

        def get_claims_monitoring_alert(self, alert_id: int) -> dict[str, object]:
            assert alert_id == 3
            return {
                "id": 3,
                "user_id": "7",
                "enabled": True,
                "channels_json": json.dumps({"webhook": True}),
                "webhook_url": "https://example.test/webhook",
            }

        def has_successful_claims_monitoring_event_delivery(self, **kwargs) -> bool:
            assert kwargs == {"user_id": "7", "event_id": 9, "alert_id": 3, "channel": "webhook"}
            return True

    @contextmanager
    def _fake_managed_media_database(*_args, **_kwargs):
        yield _Db()

    monkeypatch.setattr(claims_job_handlers, "get_user_media_db_path", lambda owner: f"/tmp/user-{owner}/Media_DB_v2.db")
    monkeypatch.setattr(claims_job_handlers, "managed_media_database", _fake_managed_media_database)
    monkeypatch.setattr(
        claims_job_handlers,
        "deliver_claims_alert_webhook",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError(f"webhook should not run: {kwargs}")),
    )

    result = await claims_job_handlers.process_claims_job(
        {
            "id": 1,
            "job_type": CLAIMS_DELIVER_ALERT_JOB_TYPE,
            "owner_user_id": "7",
            "payload": {"version": 1, "owner_user_id": "7", "event_id": 9, "alert_id": 3, "channel": "webhook"},
        }
    )

    assert result == {"outcome": "skipped", "reason": "already_delivered", "alert_id": 3}


async def test_alert_delivery_failure_is_retryable(monkeypatch) -> None:
    class _Db:
        def get_claims_monitoring_event(self, event_id: int) -> dict[str, object]:
            assert event_id == 9
            return {"id": 9, "user_id": "7", "payload_json": "{}"}

        def get_claims_monitoring_alert(self, alert_id: int) -> dict[str, object]:
            assert alert_id == 3
            return {
                "id": 3,
                "user_id": "7",
                "enabled": True,
                "channels_json": json.dumps({"webhook": True}),
                "webhook_url": "https://example.test/webhook",
            }

        def has_successful_claims_monitoring_event_delivery(self, **_kwargs) -> bool:
            return False

    @contextmanager
    def _fake_managed_media_database(*_args, **_kwargs):
        yield _Db()

    monkeypatch.setattr(claims_job_handlers, "get_user_media_db_path", lambda owner: f"/tmp/user-{owner}/Media_DB_v2.db")
    monkeypatch.setattr(claims_job_handlers, "managed_media_database", _fake_managed_media_database)
    monkeypatch.setattr(claims_job_handlers, "deliver_claims_alert_webhook", lambda **_kwargs: False)

    with pytest.raises(ClaimsJobError) as excinfo:
        await claims_job_handlers.process_claims_job(
            {
                "id": 1,
                "job_type": CLAIMS_DELIVER_ALERT_JOB_TYPE,
                "owner_user_id": "7",
                "payload": {"version": 1, "owner_user_id": "7", "event_id": 9, "alert_id": 3, "channel": "webhook"},
            }
        )

    assert excinfo.value.retryable is True
    assert excinfo.value.failure_code == "claims_alert_delivery_failed"
