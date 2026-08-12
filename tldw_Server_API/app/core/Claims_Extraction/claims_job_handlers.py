"""WorkerSDK handlers for Claims jobs."""

from __future__ import annotations

import asyncio
import errno
import json
import sqlite3
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import get_user_media_db_path
from tldw_Server_API.app.core.DB_Management.media_db.api import managed_media_database
from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    DatabaseError as MediaDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import MEDIA_NONCRITICAL_EXCEPTIONS

from .claims_alert_delivery import (
    build_claims_alert_delivery_payload,
    deliver_claims_alert_webhook,
    normalize_claims_alert_channels,
)
from .claims_analytics_exports import (
    ClaimsAnalyticsExportError,
    process_export_artifact,
)
from .claims_job_contracts import (
    CLAIMS_DELIVER_ALERT_JOB_TYPE,
    CLAIMS_DELIVER_REVIEW_NOTIFICATION_JOB_TYPE,
    CLAIMS_GENERATE_ANALYTICS_EXPORT_JOB_TYPE,
    CLAIMS_REBUILD_MEDIA_JOB_TYPE,
    ClaimsJobError,
    is_routable_claims_owner_id_text,
    validate_alert_delivery_payload,
    validate_analytics_export_payload,
    validate_rebuild_media_payload,
    validate_review_notification_payload,
)
from .claims_notifications import deliver_claim_review_notifications_now
from .claims_rebuild_service import rebuild_claims_for_media

_CLAIMS_HANDLER_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = MEDIA_NONCRITICAL_EXCEPTIONS
_CLAIMS_EXPORT_STORAGE_ERROR_TYPES: tuple[type[BaseException], ...] = (
    sqlite3.OperationalError,
    BackendDatabaseError,
    MediaDatabaseError,
    OSError,
)
_TRANSIENT_SQLITE_CODES = frozenset({sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED})
_TRANSIENT_SQLITE_MESSAGES = frozenset(
    {
        "database is busy",
        "database is locked",
        "database schema is locked",
        "database table is locked",
    }
)
_TRANSIENT_POSTGRES_SQLSTATES = frozenset(
    {
        "40001",  # serialization_failure
        "40P01",  # deadlock_detected
        "53300",  # too_many_connections
        "55P03",  # lock_not_available
        "57P01",  # admin_shutdown
        "57P02",  # crash_shutdown
        "57P03",  # cannot_connect_now
    }
)
_TRANSIENT_OS_ERRNOS = frozenset(
    code
    for name in (
        "EAGAIN",
        "EBUSY",
        "ECONNABORTED",
        "ECONNRESET",
        "EHOSTUNREACH",
        "EINTR",
        "ENETDOWN",
        "ENETUNREACH",
        "ETIMEDOUT",
        "EWOULDBLOCK",
    )
    if (code := getattr(errno, name, None)) is not None
)


def _payload(job: dict[str, Any]) -> dict[str, Any]:
    """Extract and normalize a Jobs payload object from a worker job row."""
    value = job.get("payload")
    if value is None:
        value = {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ClaimsJobError(
                "claims job payload is not valid JSON",
                retryable=False,
                failure_code="claims_invalid_payload",
            ) from exc
        if isinstance(parsed, dict):
            return dict(parsed)
    raise ClaimsJobError(
        "claims job payload must be an object",
        retryable=False,
        failure_code="claims_invalid_payload",
    )


def _owner_scope_error(message: str = "claims job owner mismatch") -> ClaimsJobError:
    """Build a non-retryable owner-scope error for invalid job routing."""
    return ClaimsJobError(
        message,
        retryable=False,
        failure_code="claims_owner_scope_violation",
    )


def _canonical_owner_user_id(value: Any) -> str:
    """Normalize and validate the owner id used for user database routing."""
    if isinstance(value, bool):
        raise _owner_scope_error("claims job owner must be a canonical positive integer")
    if isinstance(value, int):
        text = str(value)
    elif isinstance(value, str):
        text = value
    else:
        raise _owner_scope_error("claims job owner must be a canonical positive integer")
    if not is_routable_claims_owner_id_text(text):
        raise _owner_scope_error("claims job owner must be a canonical positive integer")
    return text


def _assert_owner(job: dict[str, Any], owner_user_id: Any) -> str:
    """Ensure the job row owner matches the owner embedded in the payload."""
    row_owner = _canonical_owner_user_id(job.get("owner_user_id"))
    payload_owner = _canonical_owner_user_id(owner_user_id)
    if row_owner != payload_owner:
        raise _owner_scope_error()
    return payload_owner


def _positive_job_id(value: Any) -> int:
    """Require the acquired Jobs row to expose a positive integer ID."""
    if type(value) is not int or value <= 0:
        raise ClaimsJobError(
            "claims job id must be a positive integer",
            retryable=False,
            failure_code="claims_invalid_payload",
        )
    return value


def _exception_chain(exc: BaseException) -> list[BaseException]:
    """Return a cycle-safe explicit/context exception chain."""
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__
        if current is None and not chain[-1].__suppress_context__:
            current = chain[-1].__context__
    return chain


def _is_transient_export_storage_error(exc: BaseException) -> bool:
    """Classify only explicit temporary database and storage failures."""
    if not isinstance(exc, _CLAIMS_EXPORT_STORAGE_ERROR_TYPES):
        return False

    for current in _exception_chain(exc):
        if isinstance(current, (ConnectionError, TimeoutError)):
            return True
        if isinstance(current, sqlite3.OperationalError):
            code = getattr(current, "sqlite_errorcode", None)
            if isinstance(code, int) and (code & 0xFF) in _TRANSIENT_SQLITE_CODES:
                return True
            if code is None and str(current).strip().lower() in _TRANSIENT_SQLITE_MESSAGES:
                return True

        sqlstate = getattr(current, "sqlstate", None) or getattr(current, "pgcode", None)
        if isinstance(sqlstate, str) and (sqlstate.startswith("08") or sqlstate in _TRANSIENT_POSTGRES_SQLSTATES):
            return True

        if isinstance(current, OSError) and current.errno in _TRANSIENT_OS_ERRNOS:
            return True
    return False


def _db_path(owner_user_id: Any) -> str:
    """Resolve the media database path for a canonical Claims owner id."""
    canonical_owner = _canonical_owner_user_id(owner_user_id)
    return str(get_user_media_db_path(int(canonical_owner)))


def _payload_dict(row: dict[str, Any]) -> dict[str, Any]:
    """Decode a monitoring-event payload row into a dictionary."""
    raw = row.get("payload_json") or "{}"
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _enabled(value: Any) -> bool:
    """Interpret alert enabled values while preserving the default-on behavior."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _already_delivered(db: Any, *, owner_user_id: str, event_id: int, alert_id: int, channel: str) -> bool:
    """Return whether a matching successful alert delivery is already recorded."""
    return bool(
        db.has_successful_claims_monitoring_event_delivery(
            user_id=str(owner_user_id),
            event_id=int(event_id),
            alert_id=int(alert_id),
            channel=str(channel),
        )
    )


def _deliver_alert(payload: dict[str, Any]) -> dict[str, Any]:
    """Deliver one alert channel for a validated Claims alert job payload."""
    owner_user_id = _canonical_owner_user_id(payload["owner_user_id"])
    db_path = _db_path(owner_user_id)
    try:
        with managed_media_database(
            client_id="claims_jobs_worker",
            db_path=db_path,
            initialize=False,
            suppress_init_exceptions=_CLAIMS_HANDLER_NONCRITICAL_EXCEPTIONS,
            suppress_close_exceptions=_CLAIMS_HANDLER_NONCRITICAL_EXCEPTIONS,
        ) as db:
            event = db.get_claims_monitoring_event(int(payload["event_id"]))
            if not event or str(event.get("user_id")) != str(owner_user_id):
                return {"outcome": "skipped", "reason": "event_missing", "event_id": payload["event_id"]}
            alert = db.get_claims_monitoring_alert(int(payload["alert_id"]))
            if not alert or str(alert.get("user_id")) != str(owner_user_id):
                return {"outcome": "skipped", "reason": "alert_missing", "alert_id": payload["alert_id"]}
            if not _enabled(alert.get("enabled", True)):
                return {"outcome": "skipped", "reason": "alert_disabled", "alert_id": payload["alert_id"]}
            if _already_delivered(
                db,
                owner_user_id=owner_user_id,
                event_id=payload["event_id"],
                alert_id=payload["alert_id"],
                channel=payload["channel"],
            ):
                return {"outcome": "skipped", "reason": "already_delivered", "alert_id": payload["alert_id"]}
            channels = normalize_claims_alert_channels(alert.get("channels_json") or alert.get("channels"))
            if not channels.get(payload["channel"]):
                return {"outcome": "skipped", "reason": "channel_disabled", "channel": payload["channel"]}
            event_payload = _payload_dict(event)
            if payload["channel"] == "slack":
                url = alert.get("slack_webhook_url")
            else:
                url = alert.get("webhook_url")
            if not url:
                return {"outcome": "skipped", "reason": "channel_missing_url", "channel": payload["channel"]}
            body = build_claims_alert_delivery_payload(channel=payload["channel"], event_payload=event_payload)
            delivered = deliver_claims_alert_webhook(
                url=str(url),
                payload=body,
                channel=payload["channel"],
                db_path=db_path,
                user_id=owner_user_id,
                alert_id=payload["alert_id"],
                event_id=payload["event_id"],
            )
            if not delivered:
                raise ClaimsJobError(
                    "claims alert delivery failed",
                    retryable=True,
                    failure_code="claims_alert_delivery_failed",
                )
            return {
                "outcome": "ok",
                "event_id": payload["event_id"],
                "alert_id": payload["alert_id"],
                "channel": payload["channel"],
            }
    except ClaimsJobError:
        raise
    except _CLAIMS_HANDLER_NONCRITICAL_EXCEPTIONS as exc:
        raise ClaimsJobError(
            "claims alert delivery failed",
            retryable=True,
            failure_code="claims_alert_delivery_failed",
        ) from exc


def _process_analytics_export(
    *,
    owner_user_id: str,
    export_id: str,
    job_id: int,
) -> dict[str, Any]:
    """Process one analytics export through the owner-scoped Media DB."""
    try:
        with managed_media_database(
            client_id="claims_jobs_worker",
            db_path=_db_path(owner_user_id),
            initialize=False,
            suppress_init_exceptions=_CLAIMS_HANDLER_NONCRITICAL_EXCEPTIONS,
            suppress_close_exceptions=_CLAIMS_HANDLER_NONCRITICAL_EXCEPTIONS,
        ) as db:
            return process_export_artifact(
                db,
                owner_user_id=owner_user_id,
                export_id=export_id,
                job_id=job_id,
            )
    except ClaimsAnalyticsExportError as exc:
        raise ClaimsJobError(
            exc.public_message,
            retryable=exc.retryable,
            failure_code=exc.code,
        ) from exc
    except Exception as exc:  # noqa: BLE001 - translate worker failures without leaking raw text.
        retryable = _is_transient_export_storage_error(exc)
        failure_code = "claims_export_storage_unavailable" if retryable else "claims_export_failed"
        public_message = (
            "Claims analytics export storage is temporarily unavailable."
            if retryable
            else "Claims analytics export failed."
        )
        logger.warning(
            "Claims analytics export worker failed: operation={} export_id={} job_id={} error_code={} error_type={}",
            "process_analytics_export",
            export_id,
            job_id,
            failure_code,
            type(exc).__name__,
        )
        raise ClaimsJobError(
            public_message,
            retryable=retryable,
            failure_code=failure_code,
        ) from exc


async def process_claims_job(job: dict[str, Any]) -> dict[str, Any]:
    """Validate and dispatch one Claims job through the Jobs worker runtime.

    Args:
        job: WorkerSDK job dictionary containing a Claims job_type, owner_user_id,
            and payload object or JSON object string.

    Returns:
        Structured handler outcome data for the processed job.

    Raises:
        ClaimsJobError: If the job type, owner scope, payload, or downstream
            Claims operation cannot be processed by this worker.
    """
    job_type = str(job.get("job_type") or "").strip()
    if job_type == CLAIMS_REBUILD_MEDIA_JOB_TYPE:
        payload = validate_rebuild_media_payload(_payload(job))
        owner_user_id = _assert_owner(job, payload["owner_user_id"])
        return await asyncio.to_thread(
            rebuild_claims_for_media,
            db_path=_db_path(owner_user_id),
            media_id=payload["media_id"],
        )
    if job_type == CLAIMS_DELIVER_REVIEW_NOTIFICATION_JOB_TYPE:
        payload = validate_review_notification_payload(_payload(job))
        owner_user_id = _assert_owner(job, payload["owner_user_id"])
        result = await asyncio.to_thread(
            deliver_claim_review_notifications_now,
            db_path=_db_path(owner_user_id),
            owner_user_id=owner_user_id,
            notification_ids=payload["notification_ids"],
        )
        if result.get("outcome") == "failed":
            raise ClaimsJobError(
                str(result.get("reason") or "claims review notification delivery failed"),
                retryable=True,
                failure_code="claims_review_notification_delivery_failed",
            )
        return result
    if job_type == CLAIMS_DELIVER_ALERT_JOB_TYPE:
        payload = validate_alert_delivery_payload(_payload(job))
        _assert_owner(job, payload["owner_user_id"])
        return await asyncio.to_thread(_deliver_alert, payload)
    if job_type == CLAIMS_GENERATE_ANALYTICS_EXPORT_JOB_TYPE:
        payload = validate_analytics_export_payload(_payload(job))
        owner_user_id = _assert_owner(job, payload["owner_user_id"])
        job_id = _positive_job_id(job.get("id"))
        return await asyncio.to_thread(
            _process_analytics_export,
            owner_user_id=owner_user_id,
            export_id=payload["export_id"],
            job_id=job_id,
        )
    raise ClaimsJobError(
        "unsupported claims job type",
        retryable=False,
        failure_code="claims_unsupported_job_type",
    )
