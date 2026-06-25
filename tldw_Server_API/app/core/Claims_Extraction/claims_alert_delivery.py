from __future__ import annotations

import json
import random
import socket
import ssl
import time
from typing import Any

from fastapi import HTTPException
from loguru import logger

from tldw_Server_API.app.core.Claims_Extraction.monitoring import record_claims_webhook_delivery
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.media_db.api import managed_media_database
from tldw_Server_API.app.core.exceptions import EgressPolicyError, RetryExhaustedError

_CLAIMS_ALERT_DELIVERY_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    json.JSONDecodeError,
    socket.timeout,
    ssl.SSLError,
    HTTPException,
    EgressPolicyError,
    RetryExhaustedError,
)


def normalize_claims_alert_channels(raw_value: Any | None) -> dict[str, bool]:
    if isinstance(raw_value, dict):
        data = raw_value
    else:
        data: dict[str, Any] = {}
        if raw_value:
            try:
                parsed = json.loads(str(raw_value))
                data = parsed if isinstance(parsed, dict) else {}
            except _CLAIMS_ALERT_DELIVERY_NONCRITICAL_EXCEPTIONS:
                data = {}
    return {
        "slack": bool(data.get("slack")),
        "webhook": bool(data.get("webhook")),
        "email": bool(data.get("email")),
    }


def format_claims_alert_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.2f}%"
    except _CLAIMS_ALERT_DELIVERY_NONCRITICAL_EXCEPTIONS:
        return "n/a"


def build_claims_alert_delivery_payload(*, channel: str, event_payload: dict[str, Any]) -> dict[str, Any]:
    normalized_channel = str(channel or "").strip().lower()
    if normalized_channel == "slack":
        return {
            "text": (
                "Claims alert: unsupported ratio "
                f"{format_claims_alert_ratio(event_payload.get('window_ratio'))} "
                f"(threshold {format_claims_alert_ratio(event_payload.get('threshold'))}, "
                f"baseline {format_claims_alert_ratio(event_payload.get('baseline_ratio'))})"
            )
        }
    return dict(event_payload)


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


def _record_webhook_event(
    *,
    db_path: str,
    user_id: str,
    channel: str,
    status: str,
    attempt: int,
    reason: str | None = None,
    status_code: int | None = None,
    alert_id: int | None = None,
    event_id: int | None = None,
) -> None:
    try:
        with managed_media_database(
            client_id=str(settings.get("SERVER_CLIENT_ID", "SERVER_API_V1")),
            db_path=db_path,
            suppress_init_exceptions=_CLAIMS_ALERT_DELIVERY_NONCRITICAL_EXCEPTIONS,
            suppress_close_exceptions=_CLAIMS_ALERT_DELIVERY_NONCRITICAL_EXCEPTIONS,
        ) as db:
            payload: dict[str, Any] = {
                "channel": channel,
                "status": status,
                "attempt": int(attempt),
            }
            if reason:
                payload["reason"] = reason
            if status_code is not None:
                payload["status_code"] = int(status_code)
            if alert_id is not None:
                payload["alert_id"] = int(alert_id)
            if event_id is not None:
                payload["event_id"] = int(event_id)
            db.insert_claims_monitoring_event(
                user_id=str(user_id),
                event_type="webhook_delivery",
                severity="info" if status == "success" else "warning",
                payload_json=json.dumps(payload),
            )
    except _CLAIMS_ALERT_DELIVERY_NONCRITICAL_EXCEPTIONS:
        pass


def deliver_claims_alert_webhook(
    *,
    url: str,
    payload: dict[str, Any],
    channel: str,
    db_path: str,
    user_id: str,
    alert_id: int | None = None,
    event_id: int | None = None,
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
                logger.info(
                    "Claims webhook delivered channel={} attempt={} status={}",
                    channel,
                    attempt,
                    status_code,
                )
                record_claims_webhook_delivery(status="success", latency_s=duration)
                _record_webhook_event(
                    db_path=db_path,
                    user_id=user_id,
                    channel=channel,
                    status="success",
                    attempt=attempt,
                    status_code=status_code,
                    alert_id=alert_id,
                    event_id=event_id,
                )
                return True
            if 400 <= status_code < 500:
                reason = "http_4xx"
            elif 500 <= status_code < 600:
                reason = "http_5xx"
            else:
                reason = "other"
            logger.warning(
                "Claims webhook failed channel={} attempt={} status={} reason={}",
                channel,
                attempt,
                status_code,
                reason,
            )
            record_claims_webhook_delivery(status="failure", reason=reason, latency_s=duration)
            _record_webhook_event(
                db_path=db_path,
                user_id=user_id,
                channel=channel,
                status="failure",
                attempt=attempt,
                reason=reason,
                status_code=status_code,
                alert_id=alert_id,
                event_id=event_id,
            )
        except _CLAIMS_ALERT_DELIVERY_NONCRITICAL_EXCEPTIONS as exc:
            reason = _classify_webhook_exception(exc)
            duration = time.time() - start_ts
            logger.warning(
                "Claims webhook failed channel={} attempt={} reason={}",
                channel,
                attempt,
                reason,
            )
            record_claims_webhook_delivery(status="failure", reason=reason, latency_s=duration)
            _record_webhook_event(
                db_path=db_path,
                user_id=user_id,
                channel=channel,
                status="failure",
                attempt=attempt,
                reason=reason,
                alert_id=alert_id,
                event_id=event_id,
            )
        if attempt >= max_attempts:
            return False
    return False
