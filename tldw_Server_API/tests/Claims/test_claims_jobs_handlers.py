import errno
import json
import sqlite3
import threading
from configparser import ConfigParser
from contextlib import contextmanager
from datetime import date
from types import SimpleNamespace

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
    BackendType,
    NotSupportedError,
)
from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.media_db import api as media_db_api
from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    ConflictError,
    InputError,
    SchemaError,
)
from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    DatabaseError as MediaDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.media_db.runtime import defaults as media_db_runtime_defaults
from tldw_Server_API.app.core.DB_Management.media_db.runtime.factory import MediaDbRuntimeConfig
from tldw_Server_API.app.core.DB_Management.scope_context import get_scope, scoped_context

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


def test_handler_owner_validation_enforces_routable_integer_range() -> None:
    maximum = "9223372036854775807"

    assert claims_job_handlers._canonical_owner_user_id(maximum) == maximum
    assert claims_job_handlers._canonical_owner_user_id(int(maximum)) == maximum

    with pytest.raises(ClaimsJobError) as excinfo:
        claims_job_handlers._canonical_owner_user_id("9223372036854775808")

    assert excinfo.value.failure_code == "claims_owner_scope_violation"


def test_handler_owner_validation_rejects_huge_integer_with_stable_error() -> None:
    with pytest.raises(ClaimsJobError) as excinfo:
        claims_job_handlers._canonical_owner_user_id(10**5000)

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


class _PostgresFailure(Exception):
    def __init__(self, sqlstate: str, message: str) -> None:
        super().__init__(message)
        self.sqlstate = sqlstate


def _caused_by(outer: Exception, cause: Exception) -> Exception:
    outer.__cause__ = cause
    return outer


def _sqlite_operational_error(message: str, *, code: int | None = None) -> sqlite3.OperationalError:
    error = sqlite3.OperationalError(message)
    if code is not None:
        error.sqlite_errorcode = code
    return error


@pytest.mark.parametrize(
    "error",
    [
        sqlite3.OperationalError("database is locked"),
        _caused_by(
            BackendDatabaseError("backend query failed"),
            sqlite3.OperationalError("database is busy"),
        ),
        _caused_by(
            MediaDatabaseError("media query failed"),
            _PostgresFailure("40001", "serialization failure"),
        ),
        _caused_by(
            MediaDatabaseError("media query failed"),
            _PostgresFailure("08006", "connection failure"),
        ),
        _caused_by(
            MediaDatabaseError("media query failed"),
            _PostgresFailure("40P01", "deadlock detected"),
        ),
        OSError(errno.EBUSY, "resource busy"),
        TimeoutError("connection timed out"),
    ],
)
async def test_analytics_export_handler_redacts_transient_storage_failure(monkeypatch, error: Exception) -> None:
    secret = "/private/customer.db?token=secret-value"

    @contextmanager
    def _locked_database(**_kwargs):
        raise error
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
        sqlite3.OperationalError("no such table: claims_analytics_exports"),
        _sqlite_operational_error(
            "no such table: database is locked",
            code=sqlite3.SQLITE_ERROR,
        ),
        BackendDatabaseError("invalid backend configuration"),
        MediaDatabaseError("missing required schema"),
        _caused_by(
            MediaDatabaseError("constraint failure"),
            _PostgresFailure("23505", "token=secret-value"),
        ),
        PermissionError(errno.EACCES, "permission denied"),
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


async def test_analytics_export_handler_logs_sanitized_unexpected_failure_context(monkeypatch) -> None:
    secret = "token=secret-value"
    bound: list[dict[str, object]] = []
    logged: list[str] = []

    @contextmanager
    def _failing_database(**_kwargs):
        raise RuntimeError(secret)
        yield

    class _Logger:
        def bind(self, **context: object) -> "_Logger":
            bound.append(context)
            return self

        def warning(self, message: str) -> None:
            logged.append(message)

    monkeypatch.setattr(claims_job_handlers, "managed_media_database", _failing_database)
    monkeypatch.setattr(claims_job_handlers, "logger", _Logger())

    with pytest.raises(ClaimsJobError):
        await claims_job_handlers.process_claims_job(_analytics_export_job())

    assert bound == [
        {
            "operation": "process_analytics_export",
            "export_id": "a" * 32,
            "job_id": 81,
            "error_code": "claims_export_failed",
            "error_type": "RuntimeError",
        }
    ]
    assert logged == ["Claims analytics export worker failed"]
    assert secret not in repr((bound, logged))


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


def _review_metrics_job(**overrides):
    return {
        "id": 92,
        "job_type": "claims_aggregate_review_metrics",
        "owner_user_id": "42",
        "payload": {
            "version": 1, "owner_user_id": "42", "scheduled_for": "2026-09-07T00:00:00Z",
            "start_date": "2026-09-06", "end_date": "2026-09-07",
        },
        **overrides,
    }


@pytest.mark.parametrize("row_owner", [True, False, "043", "43", 42, None])
async def test_review_metrics_handler_rejects_row_owner_before_db_routing(monkeypatch, row_owner) -> None:
    def _unexpected_open(**_kwargs):
        pytest.fail("database must not be resolved on an owner mismatch")

    monkeypatch.setattr(claims_job_handlers, "managed_media_database", _unexpected_open)
    with pytest.raises(ClaimsJobError) as excinfo:
        await claims_job_handlers.process_claims_job(_review_metrics_job(owner_user_id=row_owner))
    assert excinfo.value.failure_code == "claims_owner_scope_violation"
    assert excinfo.value.retryable is False


@pytest.mark.parametrize("postgres", [False, True])
@pytest.mark.parametrize("written,expected", [(3, {"outcome": "ok", "groups_written": 3}), (0, {"outcome": "skipped", "reason": "no_activity", "groups_written": 0})])
async def test_review_metrics_handler_uses_captured_window_and_scoped_thread_session(monkeypatch, postgres, written, expected) -> None:
    events = []
    parent_thread = threading.get_ident()
    db = SimpleNamespace(backend_type=BackendType.POSTGRESQL if postgres else BackendType.SQLITE)

    @contextmanager
    def _session(**kwargs):
        events.append(("open", get_scope(), threading.get_ident(), kwargs))
        try:
            yield db
        finally:
            events.append(("close", get_scope(), threading.get_ident(), {}))

    def _aggregate(**kwargs):
        events.append(("aggregate", get_scope(), threading.get_ident(), kwargs))
        return written

    monkeypatch.setattr(media_db_runtime_defaults, "postgres_content_mode", postgres)
    monkeypatch.setattr(claims_job_handlers, "managed_media_database", _session)
    monkeypatch.setattr(claims_job_handlers, "aggregate_claims_review_metrics_window", _aggregate, raising=False)
    monkeypatch.setattr(claims_job_handlers, "get_user_media_db_path", lambda _owner: pytest.fail("creating path resolver must not run"))
    job = _review_metrics_job()
    job["payload"] = json.dumps(job["payload"])
    with scoped_context(user_id=99, org_ids=[5], team_ids=[6], active_org_id=5, active_team_id=6):
        prior = get_scope()
        result = await claims_job_handlers.process_claims_job(job)
        assert get_scope() is prior

    assert result == {**expected, "start_date": "2026-09-06", "end_date": "2026-09-07"}
    assert [event[0] for event in events] == ["open", "aggregate", "close"]
    for _, scope, thread, _ in events:
        assert thread != parent_thread
        assert scope.user_id == 42 and scope.is_admin is True
        assert scope.org_ids == [] and scope.team_ids == []
        assert scope.active_org_id is None and scope.active_team_id is None
        assert scope.session_role is None
    open_kwargs = events[0][3]
    assert open_kwargs["initialize"] is False
    assert open_kwargs["existing_only"] is True
    assert "suppress_init_exceptions" not in open_kwargs
    assert "suppress_close_exceptions" not in open_kwargs
    if postgres:
        assert "db_path" not in open_kwargs
    else:
        assert open_kwargs["db_path"].endswith("/42/Media_DB_v2.db")
    assert events[1][3] == {"db": db, "owner_user_id": "42", "start_date": date(2026, 9, 6), "end_date": date(2026, 9, 7)}


@pytest.mark.parametrize("postgres", [False, True])
@pytest.mark.parametrize("error", [FileNotFoundError(errno.ENOENT, "private-path"), OSError(errno.ENOENT, "private-path")])
async def test_review_metrics_missing_database_skips_only_sqlite_open(monkeypatch, postgres, error) -> None:
    @contextmanager
    def _session(**_kwargs):
        raise error
        yield

    monkeypatch.setattr(media_db_runtime_defaults, "postgres_content_mode", postgres)
    monkeypatch.setattr(claims_job_handlers, "managed_media_database", _session)
    if postgres:
        with pytest.raises(ClaimsJobError) as excinfo:
            await claims_job_handlers.process_claims_job(_review_metrics_job())
        assert excinfo.value.retryable is False
    else:
        assert await claims_job_handlers.process_claims_job(_review_metrics_job()) == {
            "outcome": "skipped", "reason": "owner_database_missing", "groups_written": 0,
            "start_date": "2026-09-06", "end_date": "2026-09-07",
        }


async def test_review_metrics_path_resolution_never_creates_missing_owner_directory(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(media_db_runtime_defaults, "postgres_content_mode", False)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))

    @contextmanager
    def _session(**kwargs):
        assert not (tmp_path / "42").exists()
        assert kwargs["db_path"] == str(tmp_path / "42" / "Media_DB_v2.db")
        raise FileNotFoundError(kwargs["db_path"])
        yield

    monkeypatch.setattr(claims_job_handlers, "managed_media_database", _session)
    result = await claims_job_handlers.process_claims_job(_review_metrics_job())
    assert result["reason"] == "owner_database_missing"
    assert not (tmp_path / "42").exists()


@pytest.mark.parametrize("error", [PermissionError(errno.EACCES, "token=secret"), sqlite3.OperationalError("unable to open database file"), sqlite3.DatabaseError("file is not a database"), RuntimeError("token=secret"), sqlite3.OperationalError("database is locked")])
async def test_review_metrics_storage_errors_are_sanitized_and_scope_restored(monkeypatch, error) -> None:
    events = []
    logs = []

    @contextmanager
    def _session(**_kwargs):
        try:
            yield object()
        finally:
            events.append(get_scope())

    def _aggregate(**_kwargs):
        raise error

    class _Logger:
        def bind(self, **kwargs):
            logs.append(kwargs)
            return self

        def warning(self, message):
            logs.append(message)

    monkeypatch.setattr(media_db_runtime_defaults, "postgres_content_mode", False)
    monkeypatch.setattr(claims_job_handlers, "managed_media_database", _session)
    monkeypatch.setattr(claims_job_handlers, "aggregate_claims_review_metrics_window", _aggregate, raising=False)
    monkeypatch.setattr(claims_job_handlers, "logger", _Logger())
    with scoped_context(user_id=99, org_ids=[5], team_ids=[6]):
        prior = get_scope()
        with pytest.raises(ClaimsJobError) as excinfo:
            await claims_job_handlers.process_claims_job(_review_metrics_job())
        assert get_scope() is prior
    assert events[0].user_id == 42 and events[0].is_admin is True
    transient = isinstance(error, sqlite3.OperationalError) and str(error) == "database is locked"
    assert excinfo.value.retryable is transient
    assert excinfo.value.failure_code == ("claims_review_metrics_storage_unavailable" if transient else "claims_review_metrics_failed")
    assert str(error) not in str(excinfo.value)
    assert "token=secret" not in repr(logs)
    assert logs[0]["operation"] == "aggregate_review_metrics"
    assert logs[0]["owner_user_id"] == "42" and logs[0]["job_id"] == 92


async def test_review_metrics_aggregation_file_not_found_is_not_missing_owner_skip(monkeypatch) -> None:
    @contextmanager
    def _session(**_kwargs):
        yield object()

    def _aggregate(**_kwargs):
        raise FileNotFoundError("aggregation artifact missing")

    monkeypatch.setattr(media_db_runtime_defaults, "postgres_content_mode", False)
    monkeypatch.setattr(claims_job_handlers, "managed_media_database", _session)
    monkeypatch.setattr(claims_job_handlers, "aggregate_claims_review_metrics_window", _aggregate, raising=False)
    with pytest.raises(ClaimsJobError) as excinfo:
        await claims_job_handlers.process_claims_job(_review_metrics_job())
    assert excinfo.value.failure_code == "claims_review_metrics_failed"


def _legacy_storage_matrix():
    cases = []
    for code in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
        cases.append((_sqlite_operational_error("private", code=code), True))
        cases.append((_sqlite_operational_error("private", code=code | 256), True))
    for message in ("database is busy", "database is locked", "database schema is locked", "database table is locked"):
        cases.append((sqlite3.OperationalError(message.upper() + " "), True))
        cases.append((_sqlite_operational_error(message, code=sqlite3.SQLITE_ERROR), False))
    for state in ("40001", "40P01", "53300", "55P03", "57P01", "57P02", "57P03", "08000", "08006", "23505", "42P01", "XX000"):
        cases.append((_caused_by(MediaDatabaseError("private"), _PostgresFailure(state, "private")), state not in {"23505", "42P01", "XX000"}))
        cases.append((_PostgresFailure(state, "private"), False))
    for name in ("EAGAIN", "EBUSY", "ECONNABORTED", "ECONNRESET", "EHOSTUNREACH", "EINTR", "ENETDOWN", "ENETUNREACH", "ETIMEDOUT", "EWOULDBLOCK", "ENOENT", "EACCES", "ENOSPC"):
        cases.append((OSError(getattr(errno, name), "private"), name not in {"ENOENT", "EACCES", "ENOSPC"}))
    for error in (ConnectionError("private"), TimeoutError("private")):
        cases.append((error, True))
        cases.append((_caused_by(BackendDatabaseError("private"), error), True))
        cases.append((_caused_by(RuntimeError("private"), error), False))
    for error in (ValueError("private"), TypeError("private"), KeyError("private"), AttributeError("private"), RuntimeError("private"), sqlite3.DatabaseError("private"), sqlite3.IntegrityError("private"), InputError("private"), ConflictError("private"), SchemaError("private"), NotSupportedError("private")):
        cases.append((error, False))
    context_error = BackendDatabaseError("private")
    context_error.__context__ = TimeoutError("private")
    cases.append((context_error, True))
    suppressed = BackendDatabaseError("private")
    suppressed.__context__ = TimeoutError("private")
    suppressed.__suppress_context__ = True
    cases.append((suppressed, False))
    cyclic = BackendDatabaseError("private")
    cyclic.__cause__ = cyclic
    cases.append((cyclic, False))
    return cases


@pytest.mark.parametrize("error,expected", _legacy_storage_matrix())
def test_export_storage_classifier_legacy_matrix(error, expected) -> None:
    assert claims_job_handlers._is_transient_export_storage_error(error) is expected


@pytest.mark.parametrize("error,expected", _legacy_storage_matrix())
def test_shared_claims_storage_classifier_preserves_export_matrix(error, expected) -> None:
    assert claims_job_handlers._is_transient_claims_storage_error(error) is expected


@pytest.mark.parametrize("error_name,transient", [("SerializationFailure", True), ("DeadlockDetected", True), ("ConnectionFailure", True), ("LockNotAvailable", True), ("UndefinedTable", False), ("UniqueViolation", False)])
async def test_review_metrics_handler_classifies_native_psycopg_errors_without_changing_export_gate(monkeypatch, error_name, transient) -> None:
    psycopg = pytest.importorskip("psycopg")
    error = getattr(psycopg.errors, error_name)("token=secret")

    @contextmanager
    def _session(**_kwargs):
        raise error
        yield

    monkeypatch.setattr(media_db_runtime_defaults, "postgres_content_mode", True)
    monkeypatch.setattr(claims_job_handlers, "managed_media_database", _session)
    assert claims_job_handlers._is_transient_export_storage_error(error) is False
    with pytest.raises(ClaimsJobError) as excinfo:
        await claims_job_handlers.process_claims_job(_review_metrics_job())
    assert excinfo.value.retryable is transient
    assert "token=secret" not in str(excinfo.value)


@pytest.mark.parametrize("wrapped", [False, True])
async def test_review_metrics_native_connection_timeouts_are_retryable_without_export_change(monkeypatch, wrapped):
    error = pytest.importorskip("psycopg").errors.ConnectionTimeout("token=secret")
    if wrapped:
        error = _caused_by(BackendDatabaseError("private storage detail"), error)

    @contextmanager
    def session(**kwargs):
        raise error
        yield

    monkeypatch.setattr(media_db_runtime_defaults, "postgres_content_mode", True)
    monkeypatch.setattr(claims_job_handlers, "managed_media_database", session)
    assert claims_job_handlers._is_transient_export_storage_error(error) is False
    with pytest.raises(ClaimsJobError) as excinfo:
        await claims_job_handlers.process_claims_job(_review_metrics_job())
    assert excinfo.value.retryable is True
    assert excinfo.value.failure_code == "claims_review_metrics_storage_unavailable"
    assert "token=secret" not in str(excinfo.value)


@pytest.mark.parametrize("failure", [False, True])
def test_review_metrics_sync_scope_restores_inherited_scope_after_close(monkeypatch, failure) -> None:
    scopes = []

    @contextmanager
    def _session(**_kwargs):
        try:
            yield object()
        finally:
            scopes.append(get_scope())

    def _aggregate(**_kwargs):
        if failure:
            raise RuntimeError("private")
        return 1

    monkeypatch.setattr(media_db_runtime_defaults, "postgres_content_mode", True)
    monkeypatch.setattr(claims_job_handlers, "managed_media_database", _session)
    monkeypatch.setattr(claims_job_handlers, "aggregate_claims_review_metrics_window", _aggregate)
    with scoped_context(user_id=99, org_ids=[5], team_ids=[6], session_role="caller-role"):
        prior = get_scope()
        kwargs = {"owner_user_id": "42", "start_date": "2026-09-06", "end_date": "2026-09-07", "job_id": 92}
        if failure:
            with pytest.raises(ClaimsJobError):
                claims_job_handlers._aggregate_review_metrics(**kwargs)
        else:
            claims_job_handlers._aggregate_review_metrics(**kwargs)
        assert get_scope() is prior
    assert scopes[0].user_id == 42 and scopes[0].is_admin is True
    assert scopes[0].org_ids == [] and scopes[0].team_ids == []
    assert scopes[0].session_role is None


async def test_review_metrics_handler_real_existing_only_open_does_not_create_missing_owner(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(media_db_runtime_defaults, "postgres_content_mode", False)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    result = await claims_job_handlers.process_claims_job(_review_metrics_job())
    assert result["reason"] == "owner_database_missing"
    assert not (tmp_path / "42").exists()


@pytest.mark.parametrize("ddl_allowed", [True, False])
async def test_review_metrics_postgres_real_constructor_never_bootstraps_schema(
    monkeypatch, tmp_path, ddl_allowed,
) -> None:
    events = []
    backend = SimpleNamespace(backend_type=BackendType.POSTGRESQL)
    unused_path = tmp_path / "absent" / "unused.db"
    runtime = MediaDbRuntimeConfig(
        default_db_path=str(unused_path),
        default_config=ConfigParser(),
        postgres_content_mode=True,
        backend_loader=lambda: backend,
    )
    close_connection = MediaDatabase.close_connection

    def _bootstrap(_db):
        events.append("schema_bootstrap")
        if not ddl_allowed:
            raise PermissionError("schema bootstrap denied to runtime role")

    def _aggregate(**kwargs):
        assert isinstance(kwargs["db"], MediaDatabase)
        assert kwargs["db"].backend is backend
        assert kwargs["owner_user_id"] == "42"
        events.append("aggregate")
        return 2

    def _close(db):
        events.append("close")
        assert get_scope().user_id == 42 and get_scope().is_admin is True
        close_connection(db)

    monkeypatch.setattr(media_db_runtime_defaults, "postgres_content_mode", True)
    monkeypatch.setattr(media_db_api, "build_media_runtime_config", lambda: runtime)
    monkeypatch.setattr(MediaDatabase, "_initialize_schema", _bootstrap)
    monkeypatch.setattr(MediaDatabase, "close_connection", _close)
    monkeypatch.setattr(claims_job_handlers, "aggregate_claims_review_metrics_window", _aggregate)

    result = await claims_job_handlers.process_claims_job(_review_metrics_job())

    assert events == ["aggregate", "close"]
    assert result == {
        "outcome": "ok", "start_date": "2026-09-06", "end_date": "2026-09-07",
        "groups_written": 2,
    }
    assert not unused_path.parent.exists()


@pytest.mark.integration
@pytest.mark.postgres
async def test_review_metrics_postgres_restricted_role_job_never_bootstraps_schema(
    pg_restricted_backend, monkeypatch, tmp_path,
) -> None:
    with scoped_context(user_id=42, is_admin=True):
        setup_db = MediaDatabase(
            db_path=":memory:", client_id="42", backend=pg_restricted_backend,
        )
        setup_db.close_connection()

    runtime = MediaDbRuntimeConfig(
        default_db_path=str(tmp_path / "absent" / "unused.db"),
        default_config=ConfigParser(),
        postgres_content_mode=True,
        backend_loader=lambda: pg_restricted_backend,
    )

    def _forbid_bootstrap(_db):
        pytest.fail("Metrics workers must not bootstrap PostgreSQL schemas")

    monkeypatch.setattr(media_db_runtime_defaults, "postgres_content_mode", True)
    monkeypatch.setattr(media_db_api, "build_media_runtime_config", lambda: runtime)
    monkeypatch.setattr(MediaDatabase, "_initialize_schema", _forbid_bootstrap)

    with scoped_context(user_id=99, org_ids=[5], team_ids=[6], session_role="caller-role"):
        prior = get_scope()
        result = await claims_job_handlers.process_claims_job(_review_metrics_job())
        assert get_scope() is prior

    assert result == {
        "outcome": "skipped", "reason": "no_activity", "groups_written": 0,
        "start_date": "2026-09-06", "end_date": "2026-09-07",
    }
    assert not (tmp_path / "absent").exists()
