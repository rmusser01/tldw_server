from __future__ import annotations

import asyncio
import contextlib
import html
import json
import random
import socket
import ssl
import threading
import time
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Claims_Extraction.monitoring import (
    record_claims_review_email_delivery,
    record_claims_review_webhook_delivery,
)
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.media_db.api import managed_media_database
from tldw_Server_API.app.core.exceptions import EgressPolicyError, RetryExhaustedError

_CLAIMS_NOTIFICATION_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
_CLAIMS_NOTIFICATION_PARSE_EXCEPTIONS = (TypeError, ValueError, json.JSONDecodeError)
_CLAIMS_NOTIFICATION_MAX_PENDING = 32
_notification_slots = threading.BoundedSemaphore(_CLAIMS_NOTIFICATION_MAX_PENDING)

try:
    import httpx as _claims_httpx
except ImportError:
    _CLAIMS_HTTPX_EXCEPTIONS: tuple[type[BaseException], ...] = ()
else:
    _CLAIMS_HTTPX_EXCEPTIONS = (_claims_httpx.HTTPError,)

_CLAIMS_WEBHOOK_EXCEPTIONS = (
    ConnectionError,
    EgressPolicyError,
    RetryExhaustedError,
    ssl.SSLError,
    socket.gaierror,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
) + _CLAIMS_HTTPX_EXCEPTIONS


def submit_claims_notification_delivery(fn, *args, **kwargs) -> bool:
    if not _notification_slots.acquire(blocking=False):
        logger.warning("Claims notification dispatch queue is full")
        return False

    def _run() -> None:
        try:
            fn(*args, **kwargs)
        finally:
            _notification_slots.release()

    try:
        threading.Thread(target=_run, daemon=True).start()
    except _CLAIMS_NOTIFICATION_NONCRITICAL_EXCEPTIONS as exc:
        _notification_slots.release()
        logger.debug("Claims notification dispatch thread failed to start: {}", exc)
        return False
    return True


def record_review_assignment_notifications(
    *,
    db: Any,
    owner_user_id: str,
    assignments: list[dict[str, Any]],
) -> list[int]:
    if not assignments:
        return []
    uuids = [str(item.get("uuid")) for item in assignments if item.get("uuid")]
    if not uuids:
        return []
    rows = db.get_claims_by_uuid(uuids)
    row_by_uuid = {str(row.get("uuid")): row for row in rows if row.get("uuid")}
    inserted_ids: list[int] = []
    for item in assignments:
        claim_uuid = str(item.get("uuid") or "")
        row = row_by_uuid.get(claim_uuid)
        if not row:
            continue
        payload = {
            "claim_id": int(row.get("id") or 0),
            "claim_uuid": claim_uuid,
            "media_id": int(row.get("media_id") or 0),
            "chunk_index": int(row.get("chunk_index") or 0),
            "claim_text": str(row.get("claim_text") or ""),
            "reviewer_id": item.get("reviewer_id"),
            "review_group": item.get("review_group"),
        }
        try:
            created = db.insert_claim_notification(
                user_id=str(owner_user_id),
                kind="review_assignment",
                target_user_id=str(item.get("reviewer_id")) if item.get("reviewer_id") is not None else None,
                target_review_group=str(item.get("review_group")) if item.get("review_group") else None,
                resource_type="claim",
                resource_id=str(row.get("id") or ""),
                payload_json=json.dumps(payload),
            )
            notif_id = created.get("id")
            if notif_id is not None:
                inserted_ids.append(int(notif_id))
        except _CLAIMS_NOTIFICATION_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"Failed to insert review assignment notification: {exc}")
    return inserted_ids


def record_watchlist_cluster_notifications(
    *,
    db: Any,
    owner_user_id: str,
    clusters: dict[int, dict[str, Any]],
    member_counts: dict[int, int],
    subscriptions: dict[int, list[int]],
) -> int:
    if not subscriptions:
        return 0
    inserted = 0
    for cluster_id, job_ids in subscriptions.items():
        cluster = clusters.get(int(cluster_id))
        if not cluster:
            continue
        member_count = int(member_counts.get(int(cluster_id), 0))
        if member_count <= 0:
            continue
        latest = db.get_latest_claim_notification(
            user_id=str(owner_user_id),
            kind="watchlist_cluster_update",
            resource_type="cluster",
            resource_id=str(cluster_id),
        )
        if latest:
            try:
                payload = json.loads(latest.get("payload_json") or "{}")
            except _CLAIMS_NOTIFICATION_PARSE_EXCEPTIONS:
                payload = {}
            try:
                if int(payload.get("member_count") or 0) == member_count:
                    continue
            except (TypeError, ValueError):
                pass
        payload = {
            "cluster_id": int(cluster_id),
            "canonical_claim_text": str(cluster.get("canonical_claim_text") or ""),
            "member_count": member_count,
            "watchlist_job_ids": [int(j) for j in job_ids if j is not None],
        }
        try:
            db.insert_claim_notification(
                user_id=str(owner_user_id),
                kind="watchlist_cluster_update",
                target_user_id=str(owner_user_id),
                resource_type="cluster",
                resource_id=str(cluster_id),
                payload_json=json.dumps(payload),
            )
            inserted += 1
        except _CLAIMS_NOTIFICATION_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"Failed to insert watchlist cluster notification: {exc}")
    return inserted


def _parse_email_recipients(raw_value: str | None) -> list[str]:
    if raw_value is None:
        return []
    text = str(raw_value).strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
        if isinstance(payload, list):
            return [str(v).strip() for v in payload if str(v).strip()]
    except _CLAIMS_NOTIFICATION_PARSE_EXCEPTIONS:
        pass
    return [item.strip() for item in text.split(",") if item.strip()]


def _normalize_review_channels(config_row: dict[str, Any]) -> dict[str, bool]:
    slack_url = config_row.get("slack_webhook_url")
    webhook_url = config_row.get("webhook_url")
    email_recipients = _parse_email_recipients(config_row.get("email_recipients"))
    return {
        "slack": bool(slack_url),
        "webhook": bool(webhook_url),
        "email": bool(email_recipients),
    }


def _normalize_notification_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    raw = normalized.get("payload_json")
    try:
        normalized["payload"] = json.loads(raw) if raw else {}
    except _CLAIMS_NOTIFICATION_PARSE_EXCEPTIONS:
        normalized["payload"] = {}
    normalized.pop("payload_json", None)
    return normalized


def _classify_httpx_exception(exc: Exception, msg: str) -> str | None:
    module = getattr(exc.__class__, "__module__", "")
    if not module.startswith("httpx"):
        return None
    name = exc.__class__.__name__
    if "Timeout" in name:
        return "timeout"
    if "Connect" in name:
        if isinstance(getattr(exc, "__cause__", None), ssl.SSLError):
            return "tls"
        if isinstance(getattr(exc, "__cause__", None), socket.gaierror):
            return "dns"
        if "name or service not known" in msg or "dns" in msg:
            return "dns"
    return None


def _classify_webhook_exception(exc: Exception) -> str:
    if isinstance(exc, EgressPolicyError):
        return "invalid_url"
    if isinstance(exc, RetryExhaustedError):
        return "timeout"
    msg = str(exc).lower()
    if "timeout" in msg:
        return "timeout"
    if isinstance(exc, ssl.SSLError) or "ssl" in msg or "tls" in msg:
        return "tls"
    if isinstance(exc, socket.gaierror) or "name or service not known" in msg:
        return "dns"
    httpx_class = _classify_httpx_exception(exc, msg)
    if httpx_class:
        return httpx_class
    return "other"


def _deliver_review_webhook(
    *,
    url: str,
    payload: dict[str, Any],
    channel: str,
) -> bool:
    try:
        from tldw_Server_API.app.core.http_client import RetryPolicy, create_client, fetch
    except ImportError:
        return False
    backoff_schedule = [5, 15, 45, 120, 300]
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            base_delay = backoff_schedule[min(attempt - 2, len(backoff_schedule) - 1)]
            jitter = random.uniform(0.8, 1.2)  # nosec B311
            time.sleep(max(0.0, base_delay * jitter))
        start_ts = time.time()
        try:
            with create_client(timeout=5.0) as client:
                response = fetch(
                    method="POST",
                    url=url,
                    client=client,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=5.0,
                    retry=RetryPolicy(attempts=1, retry_on_unsafe=False),
                )
            status_code = int(getattr(response, "status_code", 0) or 0)
            duration = time.time() - start_ts
            if 200 <= status_code < 300:
                record_claims_review_webhook_delivery(status="success", latency_s=duration)
                return True
            if 400 <= status_code < 500:
                reason = "http_4xx"
            elif 500 <= status_code < 600:
                reason = "http_5xx"
            else:
                reason = "other"
            record_claims_review_webhook_delivery(status="failure", reason=reason, latency_s=duration)
        except _CLAIMS_WEBHOOK_EXCEPTIONS as exc:
            reason = _classify_webhook_exception(exc)
            duration = time.time() - start_ts
            record_claims_review_webhook_delivery(status="failure", reason=reason, latency_s=duration)
        if attempt >= max_attempts:
            return False
    return False


async def _deliver_review_email(
    *,
    recipients: list[str],
    subject: str,
    html_body: str,
    text_body: str,
) -> bool:
    if not recipients:
        return False
    try:
        from tldw_Server_API.app.core.AuthNZ.email_service import get_email_service
    except ImportError:
        return False
    service = get_email_service()
    deliveries: list[bool] = []
    for addr in recipients:
        try:
            ok = await service.send_email(
                to_email=addr,
                subject=subject,
                html_body=html_body,
                text_body=text_body,
            )
            deliveries.append(bool(ok))
        except _CLAIMS_NOTIFICATION_NONCRITICAL_EXCEPTIONS:
            deliveries.append(False)
    return any(deliveries)


def _deliver_review_email_sync(
    *,
    recipients: list[str],
    subject: str,
    html_body: str,
    text_body: str,
) -> bool:
    start_ts = time.time()
    ok = False
    try:
        ok = asyncio.run(
            _deliver_review_email(
                recipients=recipients,
                subject=subject,
                html_body=html_body,
                text_body=text_body,
            )
        )
        return ok
    except _CLAIMS_NOTIFICATION_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(f"Claims review email delivery failed: {exc}")
        return False
    finally:
        duration = time.time() - start_ts
        status = "success" if ok else "failure"
        record_claims_review_email_delivery(status=status, latency_s=duration)


def _build_review_digest_payload(
    *,
    user_id: str,
    notifications: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "event": "claims_review_notifications",
        "user_id": str(user_id),
        "count": len(notifications),
        "notifications": notifications,
    }


def _build_review_email_bodies(notifications: list[dict[str, Any]]) -> tuple[str, str]:
    lines: list[str] = []
    html_lines: list[str] = []
    for item in notifications:
        kind = str(item.get("kind") or "notification")
        payload = item.get("payload") or {}
        claim_text = payload.get("claim_text")
        status = payload.get("new_status") or payload.get("status")
        created_at = item.get("created_at") or "unknown"
        summary = f"{kind} | status={status or 'n/a'} | {created_at}"
        if claim_text:
            summary = f"{summary} | {claim_text}"
        lines.append(f"- {summary}")
        html_lines.append(f"<li>{html.escape(summary)}</li>")
    text_body = "Claims review notifications:\n" + "\n".join(lines)
    html_body = "<h2>Claims review notifications</h2><ul>" + "".join(html_lines) + "</ul>"
    return html_body, text_body


def _normalize_notification_ids(notification_ids: list[int]) -> list[int]:
    normalized: set[int] = set()
    for raw_id in notification_ids:
        if isinstance(raw_id, bool):
            continue
        try:
            notif_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if notif_id > 0:
            normalized.add(notif_id)
    return sorted(normalized)


def deliver_claim_review_notifications_now(
    *,
    db_path: str,
    owner_user_id: str,
    notification_ids: list[int],
    initialize: bool = False,
) -> dict[str, Any]:
    normalized_ids = _normalize_notification_ids(notification_ids)
    if not normalized_ids:
        return {"outcome": "skipped", "reason": "no_notification_ids", "notification_ids": []}

    with managed_media_database(
        client_id=str(settings.get("SERVER_CLIENT_ID", "SERVER_API_V1")),
        db_path=str(db_path),
        initialize=initialize,
        suppress_init_exceptions=_CLAIMS_NOTIFICATION_NONCRITICAL_EXCEPTIONS,
        suppress_close_exceptions=_CLAIMS_NOTIFICATION_NONCRITICAL_EXCEPTIONS,
    ) as db:
        config_row = db.get_claims_monitoring_settings(str(owner_user_id)) or {}
        if config_row and not bool(config_row.get("enabled", True)):
            return {"outcome": "skipped", "reason": "settings_disabled", "notification_ids": normalized_ids}

        channels = _normalize_review_channels(config_row)
        if not any(channels.values()):
            return {"outcome": "skipped", "reason": "no_channels", "notification_ids": normalized_ids}

        rows = db.get_claim_notifications_by_ids(normalized_ids)
        if not rows:
            return {"outcome": "skipped", "reason": "notifications_missing", "notification_ids": normalized_ids}

        pending_id_set: set[int] = set()
        notifications: list[dict[str, Any]] = []
        for row in rows:
            if row.get("delivered_at"):
                continue
            notifications.append(_normalize_notification_row(row))
            raw_row_id = row.get("id")
            if isinstance(raw_row_id, bool):
                continue
            try:
                pending_id = int(raw_row_id)
            except (TypeError, ValueError):
                continue
            if pending_id in normalized_ids:
                pending_id_set.add(pending_id)
        if not notifications:
            return {"outcome": "skipped", "reason": "already_delivered", "notification_ids": normalized_ids}
        pending_ids = [notification_id for notification_id in normalized_ids if notification_id in pending_id_set]

        payload = _build_review_digest_payload(user_id=str(owner_user_id), notifications=notifications)
        delivered = False

        slack_url = config_row.get("slack_webhook_url")
        webhook_url = config_row.get("webhook_url")
        recipients = _parse_email_recipients(config_row.get("email_recipients"))

        if channels.get("slack") and slack_url:
            slack_text = f"Claims review notifications: {len(notifications)} items"
            delivered = (
                _deliver_review_webhook(
                    url=str(slack_url),
                    payload={"text": slack_text},
                    channel="slack",
                )
                or delivered
            )

        if channels.get("webhook") and webhook_url:
            delivered = (
                _deliver_review_webhook(
                    url=str(webhook_url),
                    payload=payload,
                    channel="webhook",
                )
                or delivered
            )

        if channels.get("email") and recipients:
            html_body, text_body = _build_review_email_bodies(notifications)
            delivered = (
                _deliver_review_email_sync(
                    recipients=recipients,
                    subject=f"Claims review notifications ({len(notifications)})",
                    html_body=html_body,
                    text_body=text_body,
                )
                or delivered
            )

        if not delivered:
            return {"outcome": "failed", "reason": "delivery_failed", "notification_ids": normalized_ids}

        marked = db.mark_claim_notifications_delivered(pending_ids)
        return {"outcome": "ok", "notification_ids": normalized_ids, "delivered": int(marked)}


def dispatch_claim_review_notifications(
    *,
    db_path: str,
    owner_user_id: str,
    notification_ids: list[int],
) -> None:
    if not notification_ids:
        return

    def _deliver() -> None:
        try:
            deliver_claim_review_notifications_now(
                db_path=db_path,
                owner_user_id=owner_user_id,
                notification_ids=notification_ids,
                initialize=True,
            )
        except _CLAIMS_NOTIFICATION_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"Claims review notification delivery failed: {exc}")

    submit_claims_notification_delivery(_deliver)
