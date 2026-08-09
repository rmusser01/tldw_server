import json
import sqlite3
import threading
from contextlib import contextmanager

import pytest

from tldw_Server_API.app.core.Claims_Extraction import claims_job_handlers
from tldw_Server_API.app.core.Claims_Extraction.claims_analytics_exports import (
    ClaimsAnalyticsExportError,
)
from tldw_Server_API.app.core.Claims_Extraction.claims_job_contracts import (
    CLAIMS_DELIVER_ALERT_JOB_TYPE,
    CLAIMS_DELIVER_REVIEW_NOTIFICATION_JOB_TYPE,
    CLAIMS_GENERATE_ANALYTICS_EXPORT_JOB_TYPE,
    CLAIMS_REBUILD_MEDIA_JOB_TYPE,
    ClaimsJobError,
)
from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.backends.base import (
    NotSupportedError,
)
from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    ConflictError,
    InputError,
    SchemaError,
)
from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    DatabaseError as MediaDatabaseError,
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

    monkeypatch.setattr(
        claims_job_handlers, "get_user_media_db_path", lambda owner: f"/tmp/user-{owner}/Media_DB_v2.db"
    )
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

    monkeypatch.setattr(
        claims_job_handlers, "get_user_media_db_path", lambda owner: f"/tmp/user-{owner}/Media_DB_v2.db"
    )
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

    monkeypatch.setattr(
        claims_job_handlers, "get_user_media_db_path", lambda owner: f"/tmp/user-{owner}/Media_DB_v2.db"
    )
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

    monkeypatch.setattr(
        claims_job_handlers, "get_user_media_db_path", lambda owner: f"/tmp/user-{owner}/Media_DB_v2.db"
    )
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


_DEFAULT_ANALYTICS_PAYLOAD = object()


def _analytics_export_job(
    *,
    job_id: object = 81,
    row_owner: object = "7",
    payload: object = _DEFAULT_ANALYTICS_PAYLOAD,
) -> dict[str, object]:
    return {
        "id": job_id,
        "job_type": CLAIMS_GENERATE_ANALYTICS_EXPORT_JOB_TYPE,
        "owner_user_id": row_owner,
        "payload": {
            "version": 1,
            "owner_user_id": "7",
            "export_id": "a" * 32,
        }
        if payload is _DEFAULT_ANALYTICS_PAYLOAD
        else payload,
    }


async def test_analytics_export_handler_uses_owner_database_factory_and_processing_call(monkeypatch) -> None:
    db = object()
    open_calls: list[dict[str, object]] = []
    process_calls: list[dict[str, object]] = []

    @contextmanager
    def _fake_managed_media_database(**kwargs):
        open_calls.append(kwargs)
        yield db

    monkeypatch.setattr(
        claims_job_handlers,
        "get_user_media_db_path",
        lambda owner: f"/owner-databases/{owner}/Media_DB_v2.db",
    )
    monkeypatch.setattr(claims_job_handlers, "managed_media_database", _fake_managed_media_database)
    monkeypatch.setattr(
        claims_job_handlers,
        "process_export_artifact",
        lambda actual_db, **kwargs: (
            process_calls.append({"db": actual_db, **kwargs})
            or {
                "outcome": "ok",
                "export_id": "a" * 32,
                "format": "json",
                "event_count": 2,
                "size_bytes": 128,
            }
        ),
    )

    result = await claims_job_handlers.process_claims_job(_analytics_export_job())

    assert result == {
        "outcome": "ok",
        "export_id": "a" * 32,
        "format": "json",
        "event_count": 2,
        "size_bytes": 128,
    }
    assert open_calls == [
        {
            "client_id": "claims_jobs_worker",
            "db_path": "/owner-databases/7/Media_DB_v2.db",
            "initialize": False,
            "suppress_init_exceptions": claims_job_handlers._CLAIMS_HANDLER_NONCRITICAL_EXCEPTIONS,
            "suppress_close_exceptions": claims_job_handlers._CLAIMS_HANDLER_NONCRITICAL_EXCEPTIONS,
        }
    ]
    assert process_calls == [
        {
            "db": db,
            "owner_user_id": "7",
            "export_id": "a" * 32,
            "job_id": 81,
        }
    ]


async def test_analytics_export_handler_runs_processing_off_event_loop_thread(monkeypatch) -> None:
    event_loop_thread = threading.get_ident()
    handler_threads: list[int] = []

    monkeypatch.setattr(
        claims_job_handlers,
        "_process_analytics_export",
        lambda **_kwargs: (
            handler_threads.append(threading.get_ident())
            or {"outcome": "skipped", "reason": "already_ready", "export_id": "a" * 32}
        ),
    )

    result = await claims_job_handlers.process_claims_job(_analytics_export_job())

    assert result == {"outcome": "skipped", "reason": "already_ready", "export_id": "a" * 32}
    assert handler_threads and handler_threads[0] != event_loop_thread


async def test_analytics_export_handler_returns_already_ready_result(monkeypatch) -> None:
    monkeypatch.setattr(
        claims_job_handlers,
        "_process_analytics_export",
        lambda **_kwargs: {"outcome": "skipped", "reason": "already_ready", "export_id": "a" * 32},
    )

    result = await claims_job_handlers.process_claims_job(_analytics_export_job())

    assert result == {"outcome": "skipped", "reason": "already_ready", "export_id": "a" * 32}


@pytest.mark.parametrize("job_id", [None, 0, -1, True, False, "81", 1.5])
async def test_analytics_export_handler_rejects_non_positive_or_non_integer_job_id(job_id: object) -> None:
    with pytest.raises(ClaimsJobError) as excinfo:
        await claims_job_handlers.process_claims_job(_analytics_export_job(job_id=job_id))

    assert excinfo.value.retryable is False
    assert excinfo.value.failure_code == "claims_invalid_payload"


async def test_analytics_export_handler_rejects_row_owner_mismatch() -> None:
    with pytest.raises(ClaimsJobError) as excinfo:
        await claims_job_handlers.process_claims_job(_analytics_export_job(row_owner="8"))

    assert excinfo.value.retryable is False
    assert excinfo.value.failure_code == "claims_owner_scope_violation"


@pytest.mark.parametrize(
    ("payload", "failure_code"),
    [
        (None, "claims_invalid_payload"),
        ('{"version":', "claims_invalid_payload"),
        ({"version": 1, "owner_user_id": "7"}, "claims_export_invalid_payload"),
        (
            {"version": 1, "owner_user_id": "07", "export_id": "a" * 32},
            "claims_missing_owner",
        ),
        (
            {
                "version": 1,
                "owner_user_id": "7",
                "export_id": "a" * 32,
                "filters": {"severity": "high"},
            },
            "claims_export_invalid_payload",
        ),
    ],
)
async def test_analytics_export_handler_rejects_noncanonical_payload(
    payload: object,
    failure_code: str,
) -> None:
    with pytest.raises(ClaimsJobError) as excinfo:
        await claims_job_handlers.process_claims_job(_analytics_export_job(payload=payload))

    assert excinfo.value.retryable is False
    assert excinfo.value.failure_code == failure_code


@pytest.mark.parametrize(
    ("code", "message"),
    [
        ("claims_export_missing", "Claims analytics export was not found."),
        ("claims_owner_scope_violation", "Invalid Claims analytics export owner."),
        ("claims_export_invalid_artifact", "Claims analytics export artifact is invalid."),
        ("claims_export_too_large", "Claims analytics export exceeds the configured size limit."),
        ("claims_export_serialization_failed", "Claims analytics export could not be serialized."),
    ],
)
async def test_analytics_export_handler_preserves_safe_domain_failure(
    monkeypatch,
    code: str,
    message: str,
) -> None:
    @contextmanager
    def _fake_managed_media_database(**_kwargs):
        yield object()

    def _raise_domain_error(_db, **_kwargs):
        raise ClaimsAnalyticsExportError(
            message,
            code=code,
            retryable=False,
            http_status=413 if code == "claims_export_too_large" else 400,
        )

    monkeypatch.setattr(claims_job_handlers, "managed_media_database", _fake_managed_media_database)
    monkeypatch.setattr(claims_job_handlers, "process_export_artifact", _raise_domain_error)

    with pytest.raises(ClaimsJobError) as excinfo:
        await claims_job_handlers.process_claims_job(_analytics_export_job())

    assert str(excinfo.value) == message
    assert excinfo.value.retryable is False
    assert excinfo.value.failure_code == code


@pytest.mark.parametrize(
    "error_type",
    [
        sqlite3.OperationalError,
        BackendDatabaseError,
        MediaDatabaseError,
        OSError,
        TimeoutError,
    ],
)
async def test_analytics_export_handler_redacts_transient_storage_failure(monkeypatch, error_type) -> None:
    secret = "/private/customer.db?token=secret-value"

    @contextmanager
    def _locked_database(**_kwargs):
        raise error_type(f"database is locked: {secret}")
        yield

    monkeypatch.setattr(claims_job_handlers, "managed_media_database", _locked_database)
    monkeypatch.setattr(claims_job_handlers, "get_user_media_db_path", lambda _owner: secret)

    with pytest.raises(ClaimsJobError) as excinfo:
        await claims_job_handlers.process_claims_job(_analytics_export_job())

    assert str(excinfo.value) == "Claims analytics export storage is temporarily unavailable."
    assert secret not in str(excinfo.value)
    assert "secret-value" not in str(excinfo.value)
    assert excinfo.value.retryable is True
    assert excinfo.value.failure_code == "claims_export_storage_unavailable"


@pytest.mark.parametrize(
    "error",
    [
        sqlite3.DatabaseError("non-operational database failure"),
        sqlite3.IntegrityError("constraint failed"),
        ValueError("invalid value"),
        TypeError("bad type"),
        KeyError("missing"),
        AttributeError("bad attribute"),
        json.JSONDecodeError("invalid JSON", "{", 1),
        InputError("invalid input"),
        ConflictError("conflict"),
        SchemaError("schema mismatch"),
        NotSupportedError("unsupported backend operation"),
        RuntimeError("programmer failure"),
    ],
)
async def test_analytics_export_handler_redacts_and_does_not_retry_nontransient_exceptions(
    monkeypatch,
    error: Exception,
) -> None:
    @contextmanager
    def _failing_database(**_kwargs):
        raise error
        yield

    monkeypatch.setattr(claims_job_handlers, "managed_media_database", _failing_database)

    with pytest.raises(ClaimsJobError) as excinfo:
        await claims_job_handlers.process_claims_job(_analytics_export_job())

    assert str(excinfo.value) == "Claims analytics export failed."
    assert str(error) not in str(excinfo.value)
    assert excinfo.value.retryable is False
    assert excinfo.value.failure_code == "claims_export_failed"
    assert excinfo.value.__cause__ is error


async def test_unsupported_claims_job_type_remains_terminal() -> None:
    with pytest.raises(ClaimsJobError) as excinfo:
        await claims_job_handlers.process_claims_job(
            {
                "id": 81,
                "job_type": "claims_unknown_type",
                "owner_user_id": "7",
                "payload": {},
            }
        )

    assert excinfo.value.retryable is False
    assert excinfo.value.failure_code == "claims_unsupported_job_type"
