"""Admin webhooks service — CRUD, HMAC-signed delivery, and delivery logging.

Reuses HMAC signing patterns from ``jobs_webhooks_service.py``.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import httpx
from loguru import logger

from tldw_Server_API.app.core.Admin_Webhooks.domain import redact_target
from tldw_Server_API.app.core.AuthNZ.admin_webhook_secrets import (
    decrypt_admin_webhook_secret,
    encrypt_admin_webhook_secret,
)
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.exceptions import EgressPolicyError, WebhookError
from tldw_Server_API.app.core.http_client import RetryPolicy, afetch, create_async_client

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WebhookRecord:
    id: int
    url: str
    secret: str
    event_types: list[str]
    description: str
    active: bool
    retry_count: int
    timeout_seconds: int
    created_by: int | None
    created_at: str | None
    updated_at: str | None


@dataclass
class DeliveryLogEntry:
    id: int
    webhook_id: int
    event_type: str
    status_code: int | None
    latency_ms: int | None
    retry_attempt: int
    error_message: str | None
    delivered_at: str | None
    created_at: str | None


# ---------------------------------------------------------------------------
# HMAC helpers (mirrors jobs_webhooks_service.py)
# ---------------------------------------------------------------------------

def generate_signature(secret: str, timestamp: str, body: str) -> str:
    """Produce ``v1=<hex>`` HMAC-SHA256 signature over *timestamp.body*."""
    payload = f"{timestamp}.{body}"
    sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"v1={sig}"


def _url_for_logs(url: str) -> str:
    """Origin only: Slack/Discord-style webhook URLs carry their secret in the path."""
    try:
        return redact_target(url)
    except WebhookError:
        return "<invalid webhook url>"


def _default_async_client_factory(*, timeout: int) -> httpx.AsyncClient:
    """Create the outbound webhook client used for delivery attempts.

    Every attempt goes through ``afetch``, which checks the egress policy and pins the
    resolved IP per request, so a stored URL is re-checked at delivery time.
    """
    return create_async_client(timeout=timeout)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

@dataclass
class AdminWebhooksService:
    """Manages admin webhook registrations, delivery, and logs."""

    db_pool: DatabasePool = field(default=None)  # type: ignore[assignment]
    http_client_factory: Callable[..., Any] = field(default=_default_async_client_factory)

    async def _get_pool(self) -> DatabasePool:
        if self.db_pool is not None:
            return self.db_pool
        return await get_db_pool()

    # ---- CRUD ----

    async def create_webhook(
        self,
        *,
        url: str,
        event_types: list[str],
        description: str = "",
        secret: str | None = None,
        active: bool = True,
        retry_count: int = 3,
        timeout_seconds: int = 10,
        created_by: int | None = None,
    ) -> WebhookRecord:
        pool = await self._get_pool()
        if secret is None:
            secret = secrets.token_hex(32)
        encrypted_secret = encrypt_admin_webhook_secret(secret)
        event_types_json = json.dumps(event_types)
        now = datetime.now(timezone.utc).isoformat()

        row = await pool.fetchone(
            """
            INSERT INTO admin_webhooks
                (url, secret_encrypted, secret_key_id, event_types, description, active,
                 retry_count, timeout_seconds, created_by, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id, url, secret_encrypted, secret_key_id, event_types, description, active,
                      retry_count, timeout_seconds, created_by, created_at, updated_at
            """,
            (
                url,
                encrypted_secret.encrypted_blob,
                encrypted_secret.key_id,
                event_types_json,
                description,
                int(active),
                retry_count, timeout_seconds, created_by, now, now,
            ),
        )
        return self._row_to_record(row)

    async def get_webhook(self, webhook_id: int) -> WebhookRecord | None:
        pool = await self._get_pool()
        row = await pool.fetchone(
            "SELECT * FROM admin_webhooks WHERE id = ?",
            (webhook_id,),
        )
        if row is None:
            return None
        return self._row_to_record(row)

    async def list_webhooks(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        active_only: bool = False,
    ) -> tuple[list[WebhookRecord], int]:
        pool = await self._get_pool()
        if active_only:
            count_row = await pool.fetchone(
                "SELECT COUNT(*) as cnt FROM admin_webhooks WHERE active = ?",
                (1,),
            )
            rows = await pool.fetchall(
                "SELECT * FROM admin_webhooks WHERE active = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (1, limit, offset),
            )
        else:
            count_row = await pool.fetchone(
                "SELECT COUNT(*) as cnt FROM admin_webhooks",
            )
            rows = await pool.fetchall(
                "SELECT * FROM admin_webhooks ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        total = count_row["cnt"] if count_row else 0
        return [self._row_to_record(r) for r in rows], total

    async def update_webhook(
        self, webhook_id: int, **fields: Any,
    ) -> WebhookRecord | None:
        pool = await self._get_pool()
        serialized_event_types = (
            json.dumps(fields["event_types"]) if fields.get("event_types") is not None else None
        )
        active_value = int(fields["active"]) if fields.get("active") is not None else None
        serialized_secret = (
            encrypt_admin_webhook_secret(fields["secret"]) if fields.get("secret") is not None else None
        )

        has_changes = any(
            fields.get(key) is not None
            for key in (
                "url",
                "event_types",
                "description",
                "secret",
                "active",
                "retry_count",
                "timeout_seconds",
            )
        )
        if not has_changes:
            return await self.get_webhook(webhook_id)

        await pool.execute(
            """
            UPDATE admin_webhooks
            SET
                url = COALESCE(?, url),
                event_types = COALESCE(?, event_types),
                description = COALESCE(?, description),
                secret_encrypted = COALESCE(?, secret_encrypted),
                secret_key_id = COALESCE(?, secret_key_id),
                active = COALESCE(?, active),
                retry_count = COALESCE(?, retry_count),
                timeout_seconds = COALESCE(?, timeout_seconds),
                updated_at = ?
            WHERE id = ?
            """,
            (
                fields.get("url"),
                serialized_event_types,
                fields.get("description"),
                serialized_secret.encrypted_blob if serialized_secret is not None else None,
                serialized_secret.key_id if serialized_secret is not None else None,
                active_value,
                fields.get("retry_count"),
                fields.get("timeout_seconds"),
                datetime.now(timezone.utc).isoformat(),
                webhook_id,
            ),
        )
        return await self.get_webhook(webhook_id)

    async def delete_webhook(self, webhook_id: int) -> bool:
        pool = await self._get_pool()
        await pool.execute(
            "DELETE FROM admin_webhooks WHERE id = ?", (webhook_id,),
        )
        return True

    # ---- Delivery ----

    async def deliver(
        self,
        webhook: WebhookRecord,
        event_type: str,
        payload: dict[str, Any],
    ) -> DeliveryLogEntry:
        """Send a webhook delivery with retries and log the result."""
        body = json.dumps(payload, default=str)
        status_code: int | None = None
        response_body: str | None = None
        error_message: str | None = None
        latency_ms: int | None = None
        delivered_at: str | None = None
        last_attempt = 0
        signature = ""

        max_attempts = max(1, webhook.retry_count + 1)
        async with self.http_client_factory(timeout=webhook.timeout_seconds) as client:
            for attempt in range(max_attempts):
                last_attempt = attempt
                if attempt > 0:
                    await asyncio.sleep(min(2 ** attempt, 30))
                start = time.monotonic()
                try:
                    timestamp = str(int(time.time()))
                    signature = generate_signature(webhook.secret, timestamp, body)
                    headers = {
                        "Content-Type": "application/json",
                        "X-Admin-Webhook-Event": event_type,
                        "X-Admin-Webhook-Timestamp": timestamp,
                        "X-Admin-Webhook-Signature": signature,
                    }
                    # One attempt per loop pass; redirects are returned, not followed. The
                    # timeout must be explicit: afetch passes timeout=None to httpx as no timeout.
                    # sensitive_observability keeps the URL (Slack/Discord put the secret in
                    # the path) out of http_client's logs, spans and the stdlib httpx log.
                    resp = await afetch(
                        method="POST",
                        url=webhook.url,
                        client=client,
                        headers=headers,
                        content=body,
                        timeout=webhook.timeout_seconds,
                        allow_redirects=False,
                        retry=RetryPolicy(attempts=1),
                        sensitive_observability=True,
                    )
                    latency_ms = int((time.monotonic() - start) * 1000)
                    status_code = resp.status_code
                    response_body = resp.text[:4096]
                    if 200 <= resp.status_code < 300:
                        delivered_at = datetime.now(timezone.utc).isoformat()
                        break
                    error_message = f"HTTP {resp.status_code}"
                except EgressPolicyError as exc:
                    # A refused target is a configuration problem, not a transient failure.
                    latency_ms = int((time.monotonic() - start) * 1000)
                    # Nothing was sent: drop any earlier attempt's status and body.
                    status_code = None
                    response_body = None
                    # The sensitive-mode message is generic; the reason code says why.
                    error_message = f"egress policy refused: {exc.reason_code or 'denied'}"
                    logger.warning("Webhook delivery to {} {}", _url_for_logs(webhook.url), error_message)
                    break
                except Exception as exc:
                    latency_ms = int((time.monotonic() - start) * 1000)
                    error_message = str(exc)[:1024]
                    logger.warning(
                        "Webhook delivery attempt {}/{} to {} failed: {}",
                        attempt + 1, max_attempts, _url_for_logs(webhook.url), error_message,
                    )

        # Log the delivery
        entry = await self._log_delivery(
            webhook_id=webhook.id,
            event_type=event_type,
            payload_json=body,
            signature=signature,
            status_code=status_code,
            response_body=response_body,
            latency_ms=latency_ms,
            retry_attempt=last_attempt,
            error_message=error_message,
            delivered_at=delivered_at,
        )
        return entry

    async def test_webhook(self, webhook_id: int) -> dict[str, Any]:
        """Send a test ping event to a webhook."""
        wh = await self.get_webhook(webhook_id)
        if wh is None:
            return {"success": False, "error": "Webhook not found"}
        payload = {"event": "webhook.test", "webhook_id": wh.id, "timestamp": time.time()}
        entry = await self.deliver(wh, "webhook.test", payload)
        return {
            "success": entry.status_code is not None and 200 <= entry.status_code < 300,
            "status_code": entry.status_code,
            "latency_ms": entry.latency_ms,
            "error": entry.error_message,
        }

    async def dispatch_event(self, event_type: str, payload: dict[str, Any]) -> int:
        """Deliver an event to all active webhooks subscribed to *event_type*."""
        webhooks, _ = await self.list_webhooks(active_only=True, limit=1000)
        matching = [
            wh for wh in webhooks
            if "*" in wh.event_types or event_type in wh.event_types
        ]
        if not matching:
            return 0

        async def _deliver_one(wh: WebhookRecord) -> bool:
            try:
                entry = await self.deliver(wh, event_type, payload)
                return entry.status_code is not None and 200 <= entry.status_code < 300
            except Exception as exc:
                logger.error("Failed to deliver {} to webhook {}: {}", event_type, wh.id, exc)
                return False

        results = await asyncio.gather(*[_deliver_one(wh) for wh in matching])
        return sum(1 for ok in results if ok)

    # ---- Delivery Log ----

    async def list_delivery_log(
        self,
        webhook_id: int,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[DeliveryLogEntry], int]:
        pool = await self._get_pool()
        count_row = await pool.fetchone(
            "SELECT COUNT(*) as cnt FROM admin_webhooks_delivery_log WHERE webhook_id = ?",
            (webhook_id,),
        )
        total = count_row["cnt"] if count_row else 0

        rows = await pool.fetchall(
            """SELECT id, webhook_id, event_type, status_code, latency_ms,
                      retry_attempt, error_message, delivered_at, created_at
               FROM admin_webhooks_delivery_log
               WHERE webhook_id = ?
               ORDER BY created_at DESC
               LIMIT ? OFFSET ?""",
            (webhook_id, limit, offset),
        )
        return [self._delivery_row_to_entry(r) for r in rows], total

    # ---- Internal ----

    async def _log_delivery(
        self,
        *,
        webhook_id: int,
        event_type: str,
        payload_json: str,
        signature: str,
        status_code: int | None,
        response_body: str | None,
        latency_ms: int | None,
        retry_attempt: int,
        error_message: str | None,
        delivered_at: str | None,
    ) -> DeliveryLogEntry:
        pool = await self._get_pool()
        now = datetime.now(timezone.utc).isoformat()
        row = await pool.fetchone(
            """
            INSERT INTO admin_webhooks_delivery_log
                (webhook_id, event_type, payload_json, signature,
                 status_code, response_body, latency_ms, retry_attempt,
                 error_message, delivered_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id, webhook_id, event_type, status_code, latency_ms,
                      retry_attempt, error_message, delivered_at, created_at
            """,
            (
                webhook_id, event_type, payload_json, signature,
                status_code, response_body, latency_ms, retry_attempt,
                error_message, delivered_at, now,
            ),
        )
        return self._delivery_row_to_entry(row)

    @staticmethod
    def _row_to_record(row: Any) -> WebhookRecord:
        if isinstance(row, dict):
            event_types_raw = row["event_types"]
            encrypted_secret = row.get("secret_encrypted")
            plaintext_secret = row.get("secret")
        else:
            is_hardened_schema = len(row) >= 12
            event_types_raw = row[4] if is_hardened_schema else row[3]
            encrypted_secret = row[2] if is_hardened_schema else None
            plaintext_secret = None if is_hardened_schema else row[2]
        try:
            event_types = json.loads(event_types_raw) if isinstance(event_types_raw, str) else event_types_raw
        except (json.JSONDecodeError, TypeError):
            event_types = []

        if isinstance(encrypted_secret, str) and encrypted_secret:
            secret_value = decrypt_admin_webhook_secret(encrypted_secret)
        elif isinstance(plaintext_secret, str) and plaintext_secret:
            secret_value = plaintext_secret
        else:
            raise ValueError("Webhook row is missing a usable secret")

        if isinstance(row, dict):
            return WebhookRecord(
                id=row["id"],
                url=row["url"],
                secret=secret_value,
                event_types=event_types,
                description=row.get("description", ""),
                active=bool(row.get("active", True)),
                retry_count=row.get("retry_count", 3),
                timeout_seconds=row.get("timeout_seconds", 10),
                created_by=row.get("created_by"),
                created_at=row.get("created_at"),
                updated_at=row.get("updated_at"),
            )
        return WebhookRecord(
            id=row[0],
            url=row[1],
            secret=secret_value,
            event_types=event_types,
            description=row[5] if len(row) >= 12 else row[4],
            active=bool(row[6] if len(row) >= 12 else row[5]),
            retry_count=row[7] if len(row) >= 12 else row[6],
            timeout_seconds=row[8] if len(row) >= 12 else row[7],
            created_by=row[9] if len(row) >= 12 else row[8],
            created_at=row[10] if len(row) >= 12 else row[9],
            updated_at=row[11] if len(row) >= 12 else row[10],
        )

    @staticmethod
    def _delivery_row_to_entry(row: Any) -> DeliveryLogEntry:
        if isinstance(row, dict):
            return DeliveryLogEntry(
                id=row["id"],
                webhook_id=row["webhook_id"],
                event_type=row["event_type"],
                status_code=row.get("status_code"),
                latency_ms=row.get("latency_ms"),
                retry_attempt=row.get("retry_attempt", 0),
                error_message=row.get("error_message"),
                delivered_at=row.get("delivered_at"),
                created_at=row.get("created_at"),
            )
        return DeliveryLogEntry(
            id=row[0], webhook_id=row[1], event_type=row[2],
            status_code=row[3], latency_ms=row[4], retry_attempt=row[5],
            error_message=row[6], delivered_at=row[7], created_at=row[8],
        )


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------

_service: AdminWebhooksService | None = None


def get_admin_webhooks_service() -> AdminWebhooksService:
    global _service
    if _service is None:
        _service = AdminWebhooksService()
    return _service
