from __future__ import annotations

"""
Notification scaffolding for Topic Monitoring.

Goals (Phase 1):
- Provide a minimal, local-first notification hook for high-severity alerts.
- Avoid external dependencies; log to JSONL file and simulate webhook/email via configuration flags.

Configuration (env or config dict under 'monitoring.notifications'):
- MONITORING_NOTIFY_ENABLED: 'true'|'false' (default: false)
- MONITORING_NOTIFY_MIN_SEVERITY: 'info'|'warning'|'critical' (default: 'critical')
- MONITORING_NOTIFY_FILE: path for JSONL sink (default: 'Databases/monitoring_notifications.log')
- MONITORING_NOTIFY_WEBHOOK_URL: URL (future use; not sent in offline mode)
- MONITORING_NOTIFY_EMAIL_TO: comma-separated emails (future use; not sent in Phase 1)
- MONITORING_NOTIFY_DIGEST_MODE: 'immediate'|'hourly'|'daily' (default: 'immediate').
  'hourly'/'daily' buffer generic/guardian notifications until callers invoke
  flush_digest(), which emits one compiled monitoring_digest payload per recipient.

This scaffolding records intent locally so operators can forward to their systems.
"""

import contextlib
import json
import os
import smtplib
import threading
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from loguru import logger
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

from tldw_Server_API.app.core.config import load_and_log_configs
from tldw_Server_API.app.core.DB_Management.TopicMonitoring_DB import TopicAlert
from tldw_Server_API.app.core.testing import is_truthy

_SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}


def _safe_exception_label(exc: BaseException) -> str:
    return exc.__class__.__name__


def _find_project_root(start: Path) -> Path | None:
    """Best-effort search for the repository root starting from a file/dir path."""
    start_dir = start if start.is_dir() else start.parent
    for candidate in (start_dir, *start_dir.parents):
        if (candidate / ".git").exists():
            return candidate
        if (candidate / "pyproject.toml").is_file() and (candidate / "tldw_Server_API").is_dir():
            return candidate
        if (candidate / "AGENTS.md").is_file() and (candidate / "tldw_Server_API").is_dir():
            return candidate
        if candidate.name != "tldw_Server_API" and (candidate / "tldw_Server_API").is_dir():
            return candidate
    return None


class NotificationService:
    def __init__(self) -> None:
        cfg = load_and_log_configs() or {}
        ncfg = (cfg.get("monitoring") or {}).get("notifications") if isinstance(cfg, dict) else None
        self.enabled = is_truthy(os.getenv("MONITORING_NOTIFY_ENABLED", str((ncfg or {}).get("enabled", False))))
        self.min_severity = str(os.getenv("MONITORING_NOTIFY_MIN_SEVERITY", (ncfg or {}).get("min_severity", "critical"))).strip().lower()
        raw_file = os.getenv("MONITORING_NOTIFY_FILE", (ncfg or {}).get("file", "Databases/monitoring_notifications.log"))
        self.file_path = self._resolve_file_path(raw_file)
        self.webhook_url = os.getenv("MONITORING_NOTIFY_WEBHOOK_URL", (ncfg or {}).get("webhook_url", ""))
        self.email_to = os.getenv("MONITORING_NOTIFY_EMAIL_TO", (ncfg or {}).get("email_to", ""))
        # SMTP configuration (optional)
        self.smtp_host = os.getenv("MONITORING_NOTIFY_SMTP_HOST", (ncfg or {}).get("smtp_host", ""))
        raw_smtp_port = os.getenv("MONITORING_NOTIFY_SMTP_PORT", (ncfg or {}).get("smtp_port", "587"))
        self.smtp_port = self._coerce_int(raw_smtp_port, 587)
        self.smtp_starttls = is_truthy(os.getenv("MONITORING_NOTIFY_SMTP_STARTTLS", (ncfg or {}).get("smtp_starttls", "true")))
        self.smtp_user = os.getenv("MONITORING_NOTIFY_SMTP_USER", (ncfg or {}).get("smtp_user", ""))
        self.smtp_password = os.getenv("MONITORING_NOTIFY_SMTP_PASSWORD", (ncfg or {}).get("smtp_password", ""))
        self.email_from = os.getenv("MONITORING_NOTIFY_EMAIL_FROM", (ncfg or {}).get("email_from", self.smtp_user or ""))
        self.digest_mode = os.getenv(
            "MONITORING_NOTIFY_DIGEST_MODE",
            (ncfg or {}).get("digest_mode", "immediate"),
        ).strip().lower()
        self._lock = threading.RLock()
        self._pending_digests: dict[str, list[dict[str, Any]]] = {}
        with contextlib.suppress(OSError):
            Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _coerce_int(value: Any, default: int) -> int:
        if value is None or value == "":
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            logger.warning(f"Invalid MONITORING_NOTIFY_SMTP_PORT={value!r}; using {default}")
            return default

    @staticmethod
    def _resolve_file_path(raw_file: str) -> str:
        try:
            fp = Path(str(raw_file))
            if not fp.is_absolute():
                try:
                    from tldw_Server_API.app.core.Utils.Utils import get_project_root as _gpr
                    root = Path(_gpr()).resolve()
                    fp = root / fp
                except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
                    root = _find_project_root(Path(__file__).resolve())
                    if root is None:
                        root = Path(__file__).resolve().parent
                    fp = root / fp
            return str(fp)
        except (OSError, RuntimeError, TypeError, ValueError):
            fallback_root = _find_project_root(Path(__file__).resolve()) or Path(__file__).resolve().parent
            return str(fallback_root / str(raw_file))

    @staticmethod
    def _parse_email_recipients(raw: str | None) -> list[str]:
        if not raw:
            return []
        if isinstance(raw, (list, tuple, set)):
            return [str(addr).strip() for addr in raw if addr and str(addr).strip()]
        parts = str(raw).replace(";", ",").split(",")
        return [part.strip() for part in parts if part.strip()]

    def get_notification_file_path(self) -> str | None:
        """Return the path to the notification JSONL file."""
        return self.file_path or None

    def get_settings(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "min_severity": self.min_severity,
            "file": self.file_path,
            "webhook_url": self.webhook_url,
            "email_to": self.email_to,
            "smtp_host": self.smtp_host,
            "smtp_port": self.smtp_port,
            "smtp_starttls": self.smtp_starttls,
            "smtp_user": self.smtp_user,
            "email_from": self.email_from,
        }

    def update_settings(
        self,
        *,
        enabled: bool | None = None,
        min_severity: str | None = None,
        file: str | None = None,
        webhook_url: str | None = None,
        email_to: str | None = None,
        smtp_host: str | None = None,
        smtp_port: int | None = None,
        smtp_starttls: bool | None = None,
        smtp_user: str | None = None,
        smtp_password: str | None = None,
        email_from: str | None = None,
    ) -> dict[str, Any]:
        # Update runtime settings (non-persistent). Best-effort.
        if enabled is not None:
            self.enabled = bool(enabled)
        if min_severity is not None:
            self.min_severity = str(min_severity).lower()
        if file is not None:
            try:
                resolved = self._resolve_file_path(file)
                Path(resolved).parent.mkdir(parents=True, exist_ok=True)
                self.file_path = resolved
            except (OSError, RuntimeError, TypeError, ValueError) as e:
                logger.warning("Failed to update MONITORING_NOTIFY_FILE ({})", _safe_exception_label(e))
        if webhook_url is not None:
            self.webhook_url = webhook_url
        if email_to is not None:
            self.email_to = email_to
        if smtp_host is not None:
            self.smtp_host = smtp_host
        if smtp_port is not None:
            self.smtp_port = self._coerce_int(smtp_port, self.smtp_port)
        if smtp_starttls is not None:
            self.smtp_starttls = bool(smtp_starttls)
        if smtp_user is not None:
            self.smtp_user = smtp_user
        if smtp_password is not None:
            self.smtp_password = smtp_password
        if email_from is not None:
            self.email_from = email_from
        return self.get_settings()

    def _meets_threshold(self, severity: str | None) -> bool:
        if not self.enabled:
            return False
        sev = (severity or "info").lower()
        try:
            return _SEVERITY_ORDER.get(sev, 0) >= _SEVERITY_ORDER.get(self.min_severity, 2)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return False

    def notify(self, alert: TopicAlert) -> str:
        """Record a notification intent for an alert. Phase 1: JSONL file sink only.

        Future: send to webhook/email if configured and networking permitted.
        """
        if not self._meets_threshold(alert.rule_severity):
            return "skipped"
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": "topic_alert",
            "user_id": alert.user_id,
            "scope_type": alert.scope_type,
            "scope_id": alert.scope_id,
            "source": alert.source,
            "watchlist_id": alert.watchlist_id,
            "rule_id": alert.rule_id,
            "rule_category": alert.rule_category,
            "rule_severity": alert.rule_severity,
            "pattern": alert.pattern,
            "source_id": alert.source_id,
            "chunk_id": alert.chunk_id,
            "chunk_seq": alert.chunk_seq,
            "snippet": alert.text_snippet,
            "metadata": alert.metadata or {},
            "route_tags": {"scope_type": alert.scope_type, "scope_id": alert.scope_id},
        }
        # Always append to JSONL file (local-first scaffold)
        file_written = True
        try:
            with self._lock, open(self.file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except (OSError, RuntimeError, TypeError, ValueError) as e:
            file_written = False
            logger.warning("Notification file sink failed ({})", _safe_exception_label(e))
        # Best-effort asynchronous sends (non-blocking)
        try:
            if self.webhook_url:
                threading.Thread(target=self._send_webhook_safe, args=(payload,), daemon=True).start()
        except (OSError, RuntimeError) as e:
            logger.debug("Webhook thread start failed ({})", _safe_exception_label(e))
        try:
            # Email optional and only if SMTP configured and recipients provided
            recipients = self._parse_email_recipients(self.email_to)
            if recipients and self.smtp_host and self.email_from:
                threading.Thread(target=self._send_email_safe, args=(alert,), daemon=True).start()
        except (OSError, RuntimeError) as e:
            logger.debug("Email thread start failed ({})", _safe_exception_label(e))
        return "logged" if file_written else "failed"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=False)
    def _send_webhook(self, payload: dict[str, Any]) -> None:
        from tldw_Server_API.app.core.http_client import create_client, fetch
        # 3s connect, 5s read/write aligns with defaults but explicit here
        with create_client(timeout=5.0) as client:
            headers = {"Content-Type": "application/json"}
            fetch(method="POST", url=self.webhook_url, client=client, headers=headers, json=payload, timeout=5.0)

    def _send_webhook_safe(self, payload: dict[str, Any]) -> None:
        try:
            self._send_webhook(payload)
        except RetryError as e:
            logger.info("Webhook notify failed ({})", _safe_exception_label(e))
        except (OSError, RuntimeError, TypeError, ValueError) as e:
            logger.info("Webhook notify failed ({})", _safe_exception_label(e))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=False)
    def _send_email(self, alert: TopicAlert) -> None:
        recipients = self._parse_email_recipients(self.email_to)
        if not (self.smtp_host and self.email_from and recipients):
            return
        subject = f"Topic Alert: {alert.rule_category or 'topic'} ({alert.rule_severity or 'info'})"
        body = (
            f"Source: {alert.source}\n"
            f"User: {alert.user_id}\n"
            f"Watchlist: {alert.watchlist_id}\n"
            f"Category: {alert.rule_category}\n"
            f"Severity: {alert.rule_severity}\n"
            f"Pattern: {alert.pattern}\n\n"
            f"Snippet:\n{alert.text_snippet}\n"
        )
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = self.email_from
        msg["To"] = ", ".join(recipients)

        with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as server:
            if self.smtp_starttls:
                with contextlib.suppress(OSError, RuntimeError, smtplib.SMTPException):
                    server.starttls()
            if self.smtp_user:
                server.login(self.smtp_user, self.smtp_password or "")
            server.sendmail(self.email_from, recipients, msg.as_string())

    def notify_generic(self, payload: dict[str, Any]) -> str:
        """Record a generic notification payload (not tied to TopicAlert).

        Applies severity threshold filtering and writes to JSONL sink.
        Adds ``ts`` field if not present.
        """
        severity = payload.get("severity") or payload.get("rule_severity")
        if not self._meets_threshold(severity):
            return "skipped"
        if "ts" not in payload:
            payload["ts"] = datetime.now(timezone.utc).isoformat()
        file_written = True
        try:
            with self._lock, open(self.file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except (OSError, RuntimeError, TypeError, ValueError) as e:
            file_written = False
            logger.warning("Notification file sink failed ({})", _safe_exception_label(e))
        try:
            if self.webhook_url:
                threading.Thread(target=self._send_webhook_safe, args=(payload,), daemon=True).start()
        except (OSError, RuntimeError) as e:
            logger.debug("Webhook thread start failed ({})", _safe_exception_label(e))
        return "logged" if file_written else "failed"

    @staticmethod
    def _digest_recipient_key(recipient: Any) -> str:
        """Normalize digest recipient keys without collapsing falsy identifiers."""
        if recipient is None:
            return "_default"
        return str(recipient)

    def notify_or_batch(self, payload: dict[str, Any]) -> str:
        """Route to immediate send or batching depending on digest_mode."""
        severity = payload.get("severity") or payload.get("rule_severity")
        if not self._meets_threshold(severity):
            return "skipped"
        if self.digest_mode in ("hourly", "daily"):
            recipient = self._digest_recipient_key(payload.get("user_id", "_default"))
            with self._lock:
                self._pending_digests.setdefault(recipient, []).append(dict(payload))
            return "batched"
        return self.notify_generic(payload)

    @staticmethod
    def _digest_severity(items: list[dict[str, Any]]) -> str:
        """Return the highest severity present in a digest batch."""
        highest = "info"
        for item in items:
            severity = str(
                item.get("severity") or item.get("rule_severity") or "info"
            ).strip().lower()
            if _SEVERITY_ORDER.get(severity, 0) > _SEVERITY_ORDER.get(highest, 0):
                highest = severity
        return highest

    def _build_digest_payload(self, recipient: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        item_copies = [dict(item) for item in items]
        return {
            "type": "monitoring_digest",
            "severity": self._digest_severity(item_copies),
            "recipient": recipient,
            "digest_mode": self.digest_mode,
            "item_count": len(item_copies),
            "items": item_copies,
        }

    def flush_digest(self, recipient: str | None = None) -> int:
        """Deliver pending digest alerts and return count of processed items.

        Delivery emits one compiled digest payload per recipient through the
        generic notification path. Failed recipients are requeued for normal
        delivery exceptions so callers can retry a later flush without losing
        buffered items. Threshold-skipped digests are considered processed.
        """
        with self._lock:
            if recipient is not None:
                recipient_key = self._digest_recipient_key(recipient)
                pending = {recipient_key: self._pending_digests.pop(recipient_key, [])}
            else:
                pending = dict(self._pending_digests)
                self._pending_digests.clear()

        processed_count = 0
        failed: dict[str, list[dict[str, Any]]] = {}
        for recipient_key, items in pending.items():
            if not items:
                continue
            payload = self._build_digest_payload(recipient_key, items)
            try:
                result = self.notify_generic(payload)
            except Exception as exc:
                logger.warning(
                    "Monitoring digest delivery failed for {} ({})",
                    recipient_key,
                    _safe_exception_label(exc),
                )
                failed[recipient_key] = items
                continue
            if result in ("logged", "skipped"):
                processed_count += len(items)
            else:
                logger.warning(
                    "Monitoring digest delivery for {} returned {}; requeueing {} item(s)",
                    recipient_key,
                    result,
                    len(items),
                )
                failed[recipient_key] = items

        if failed:
            with self._lock:
                for recipient_key, items in failed.items():
                    if items:
                        self._pending_digests[recipient_key] = (
                            items + self._pending_digests.get(recipient_key, [])
                        )

        return processed_count

    def get_pending_digest_count(self, recipient: str | None = None) -> int:
        """Return count of pending digest items, optionally for a specific recipient."""
        with self._lock:
            if recipient is not None:
                recipient_key = self._digest_recipient_key(recipient)
                return len(self._pending_digests.get(recipient_key, []))
            return sum(len(v) for v in self._pending_digests.values())

    def _send_email_safe(self, alert: TopicAlert) -> None:
        try:
            self._send_email(alert)
        except RetryError as e:
            logger.info("Email notify failed ({})", _safe_exception_label(e))
        except (OSError, RuntimeError, TypeError, ValueError, smtplib.SMTPException) as e:
            logger.info("Email notify failed ({})", _safe_exception_label(e))


_notify_singleton: NotificationService | None = None


def get_notification_service() -> NotificationService:
    global _notify_singleton
    if _notify_singleton is None:
        _notify_singleton = NotificationService()
    return _notify_singleton
