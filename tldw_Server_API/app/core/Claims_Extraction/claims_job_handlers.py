from __future__ import annotations

import asyncio
import json
from typing import Any

from tldw_Server_API.app.core.DB_Management.db_path_utils import get_user_media_db_path
from tldw_Server_API.app.core.DB_Management.media_db.api import managed_media_database
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import MEDIA_NONCRITICAL_EXCEPTIONS

from .claims_alert_delivery import (
    build_claims_alert_delivery_payload,
    deliver_claims_alert_webhook,
    normalize_claims_alert_channels,
)
from .claims_job_contracts import (
    CLAIMS_DELIVER_ALERT_JOB_TYPE,
    CLAIMS_DELIVER_REVIEW_NOTIFICATION_JOB_TYPE,
    CLAIMS_REBUILD_MEDIA_JOB_TYPE,
    ClaimsJobError,
    validate_alert_delivery_payload,
    validate_rebuild_media_payload,
    validate_review_notification_payload,
)
from .claims_notifications import deliver_claim_review_notifications_now
from .claims_rebuild_service import rebuild_claims_for_media

_CLAIMS_HANDLER_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = MEDIA_NONCRITICAL_EXCEPTIONS


def _payload(job: dict[str, Any]) -> dict[str, Any]:
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
    return ClaimsJobError(
        message,
        retryable=False,
        failure_code="claims_owner_scope_violation",
    )


def _canonical_owner_user_id(value: Any) -> str:
    if isinstance(value, bool):
        raise _owner_scope_error("claims job owner must be a canonical positive integer")
    if isinstance(value, int):
        if value <= 0:
            raise _owner_scope_error("claims job owner must be a canonical positive integer")
        return str(value)
    if not isinstance(value, str):
        raise _owner_scope_error("claims job owner must be a canonical positive integer")

    text = value.strip()
    if text != value or not text or not text.isascii() or not text.isdigit():
        raise _owner_scope_error("claims job owner must be a canonical positive integer")
    if text == "0" or (len(text) > 1 and text.startswith("0")):
        raise _owner_scope_error("claims job owner must be a canonical positive integer")
    return text


def _assert_owner(job: dict[str, Any], owner_user_id: Any) -> str:
    row_owner = _canonical_owner_user_id(job.get("owner_user_id"))
    payload_owner = _canonical_owner_user_id(owner_user_id)
    if row_owner != payload_owner:
        raise _owner_scope_error()
    return payload_owner


def _db_path(owner_user_id: Any) -> str:
    canonical_owner = _canonical_owner_user_id(owner_user_id)
    return str(get_user_media_db_path(int(canonical_owner)))


def _payload_dict(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("payload_json") or "{}"
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _positive_int_or_zero(value: Any) -> int:
    try:
        if isinstance(value, bool):
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _enabled(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "no", "off"}
    return bool(value)


def _already_delivered(db: Any, *, owner_user_id: str, event_id: int, alert_id: int, channel: str) -> bool:
    rows = db.list_claims_monitoring_events(
        user_id=str(owner_user_id),
        event_type="webhook_delivery",
    )
    for row in rows:
        payload = _payload_dict(dict(row))
        if (
            str(payload.get("status")) == "success"
            and _positive_int_or_zero(payload.get("event_id")) == int(event_id)
            and _positive_int_or_zero(payload.get("alert_id")) == int(alert_id)
            and str(payload.get("channel") or "") == str(channel)
        ):
            return True
    return False


def _deliver_alert(payload: dict[str, Any]) -> dict[str, Any]:
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


async def process_claims_job(job: dict[str, Any]) -> dict[str, Any]:
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
    raise ClaimsJobError(
        "unsupported claims job type",
        retryable=False,
        failure_code="claims_unsupported_job_type",
    )
