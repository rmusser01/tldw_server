from __future__ import annotations

"""
Notification delivery for Topic Monitoring.

Goals:
- Provide a local-first notification hook for high-severity alerts.
- Log to a bounded JSONL sink and optionally deliver webhook/email notifications.

Configuration (env or config dict under 'monitoring.notifications'):
- MONITORING_NOTIFY_ENABLED: 'true'|'false' (default: false)
- MONITORING_NOTIFY_MIN_SEVERITY: 'info'|'warning'|'critical' (default: 'critical')
- MONITORING_NOTIFY_FILE: path for JSONL sink (default: 'Databases/monitoring_notifications.log')
- MONITORING_NOTIFY_ALLOWED_DIRS: additional absolute directories allowed for file sinks
- MONITORING_NOTIFY_WEBHOOK_URL: optional webhook URL for async delivery
- MONITORING_NOTIFY_EMAIL_TO: comma-separated emails for async SMTP delivery
- MONITORING_NOTIFY_MAX_QUEUE: maximum pending webhook/email deliveries (default: 1000)
- MONITORING_NOTIFY_MAX_DIGEST_ITEMS_PER_RECIPIENT: per-recipient digest cap (default: 1000)
- MONITORING_NOTIFY_DIGEST_MODE: 'immediate'|'hourly'|'daily' (default: 'immediate').
  'hourly'/'daily' buffer generic/guardian notifications until callers invoke
  flush_digest(), which emits one compiled monitoring_digest payload per recipient.

The JSONL file is restricted to trusted notification directories so the recent
notifications endpoint cannot be pointed at arbitrary host files.
"""

import contextlib
import json
import os
import queue
import smtplib
import tempfile
import threading
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from loguru import logger
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

from tldw_Server_API.app.core.config import load_and_log_configs
from tldw_Server_API.app.core.DB_Management.TopicMonitoring_DB import TopicAlert
from tldw_Server_API.app.core.testing import is_explicit_pytest_runtime, is_test_mode, is_truthy

_SEVERITY_ORDER = {"info": 0, "warning": 1, "critical": 2}
_REDACTED_VALUE = "[REDACTED]"
_SENSITIVE_NOTIFICATION_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "access_token",
    "refresh_token",
    "password",
    "passwd",
    "secret",
    "token",
    "credential",
    "private_key",
    "webhook_url",
)
_SENSITIVE_NOTIFICATION_TEXT_MARKERS = (
    "api_key=",
    "apikey=",
    "x-api-key=",
    "access_token=",
    "refresh_token=",
    "authorization=",
    "password=",
    "passwd=",
    "secret=",
    "token=",
    "api_key:",
    "apikey:",
    "x-api-key:",
    "authorization:",
    "password:",
    "passwd:",
    "secret:",
    "token:",
    "bearer ",
)
_SENSITIVE_NOTIFICATION_TEXT_END = set("\r\n&;,\"'<>")


def _sensitive_notification_secret_end(value: str, secret_start: int) -> int:
    """Find the end of one inline secret while keeping surrounding prose intact."""
    secret_end = secret_start
    while secret_end < len(value):
        char = value[secret_end]
        if char in _SENSITIVE_NOTIFICATION_TEXT_END:
            break
        if char in " \t":
            if value[secret_start:secret_end].casefold() == "bearer":
                secret_end += 1
                while secret_end < len(value) and value[secret_end] in " \t":
                    secret_end += 1
                continue
            break
        secret_end += 1
    return secret_end


def _safe_exception_label(exc: BaseException) -> str:
    """Return only the exception class name for public notification logs."""
    return exc.__class__.__name__


def _is_sensitive_notification_key(key: Any) -> bool:
    """Return True when a payload key name commonly carries credentials."""
    normalized = str(key).strip().replace("-", "_").lower()
    return any(part in normalized for part in _SENSITIVE_NOTIFICATION_KEY_PARTS)


def _redact_notification_text(value: str) -> str:
    """Redact inline credential-looking text while preserving surrounding content."""
    lower = value.lower()
    cursor = 0
    pieces: list[str] = []

    while cursor < len(value):
        next_marker_start = -1
        next_marker = ""
        for marker in _SENSITIVE_NOTIFICATION_TEXT_MARKERS:
            marker_start = lower.find(marker, cursor)
            if marker_start == -1:
                continue
            if next_marker_start == -1 or marker_start < next_marker_start:
                next_marker_start = marker_start
                next_marker = marker
        if next_marker_start == -1:
            pieces.append(value[cursor:])
            break

        marker_end = next_marker_start + len(next_marker)
        secret_start = marker_end
        while secret_start < len(value) and value[secret_start] in " \t":
            secret_start += 1
        secret_end = _sensitive_notification_secret_end(value, secret_start)

        pieces.append(value[cursor:secret_start])
        pieces.append(_REDACTED_VALUE)
        cursor = secret_end

    return "".join(pieces)


def _sanitize_notification_payload(value: Any) -> Any:
    """Recursively sanitize notification payloads before storage or delivery."""
    if isinstance(value, dict):
        sanitized: dict[Any, Any] = {}
        for key, item in value.items():
            if _is_sensitive_notification_key(key):
                sanitized[key] = _REDACTED_VALUE
            else:
                sanitized[key] = _sanitize_notification_payload(item)
        return sanitized
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_notification_payload(item) for item in value]
    if isinstance(value, str):
        return _redact_notification_text(value)
    return value


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
        try:
            self.file_path = self._resolve_file_path(raw_file)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning("Invalid MONITORING_NOTIFY_FILE; using default ({})", _safe_exception_label(exc))
            self.file_path = str(self._default_file_path())
        self.webhook_url = os.getenv("MONITORING_NOTIFY_WEBHOOK_URL", (ncfg or {}).get("webhook_url", ""))
        self.email_to = os.getenv("MONITORING_NOTIFY_EMAIL_TO", (ncfg or {}).get("email_to", ""))
        # SMTP configuration (optional)
        self.smtp_host = os.getenv("MONITORING_NOTIFY_SMTP_HOST", (ncfg or {}).get("smtp_host", ""))
        raw_smtp_port = os.getenv("MONITORING_NOTIFY_SMTP_PORT", (ncfg or {}).get("smtp_port", "587"))
        self.smtp_port = self._coerce_int(raw_smtp_port, 587, "MONITORING_NOTIFY_SMTP_PORT")
        self.smtp_starttls = is_truthy(os.getenv("MONITORING_NOTIFY_SMTP_STARTTLS", (ncfg or {}).get("smtp_starttls", "true")))
        self.smtp_user = os.getenv("MONITORING_NOTIFY_SMTP_USER", (ncfg or {}).get("smtp_user", ""))
        self.smtp_password = os.getenv("MONITORING_NOTIFY_SMTP_PASSWORD", (ncfg or {}).get("smtp_password", ""))
        self.email_from = os.getenv("MONITORING_NOTIFY_EMAIL_FROM", (ncfg or {}).get("email_from", self.smtp_user or ""))
        self.max_delivery_queue_size = max(
            1,
            self._coerce_int(
                os.getenv(
                    "MONITORING_NOTIFY_MAX_QUEUE",
                    (ncfg or {}).get("max_queue_size", 1000),
                ),
                1000,
                "MONITORING_NOTIFY_MAX_QUEUE",
            ),
        )
        self.max_digest_items_per_recipient = max(
            1,
            self._coerce_int(
                os.getenv(
                    "MONITORING_NOTIFY_MAX_DIGEST_ITEMS_PER_RECIPIENT",
                    (ncfg or {}).get("max_digest_items_per_recipient", 1000),
                ),
                1000,
                "MONITORING_NOTIFY_MAX_DIGEST_ITEMS_PER_RECIPIENT",
            ),
        )
        self.digest_mode = os.getenv(
            "MONITORING_NOTIFY_DIGEST_MODE",
            (ncfg or {}).get("digest_mode", "immediate"),
        ).strip().lower()
        self._lock = threading.RLock()
        self._pending_digests: dict[str, list[dict[str, Any]]] = {}
        self._delivery_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=self.max_delivery_queue_size)
        self._delivery_worker_started = False
        self._delivery_worker_lock = threading.RLock()
        with contextlib.suppress(OSError):
            Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _coerce_int(value: Any, default: int, label: str = "integer") -> int:
        if value is None or value == "":
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            logger.warning("Invalid {}={!r}; using {}", label, value, default)
            return default

    @staticmethod
    def _project_root() -> Path:
        try:
            from tldw_Server_API.app.core.Utils.Utils import get_project_root as _gpr
            return Path(_gpr()).resolve()
        except (AttributeError, ImportError, OSError, RuntimeError, TypeError, ValueError):
            root = _find_project_root(Path(__file__).resolve())
            if root is None:
                root = Path(__file__).resolve().parent
            return root.resolve()

    @classmethod
    def _default_file_path(cls) -> Path:
        return (cls._project_root() / "Databases" / "monitoring_notifications.log").resolve(strict=False)

    @staticmethod
    def _parse_allowed_dirs(raw_dirs: str | None) -> list[Path]:
        if not raw_dirs:
            return []
        parts: list[str] = []
        for chunk in str(raw_dirs).split(os.pathsep):
            parts.extend(piece.strip() for piece in chunk.split(","))
        allowed: list[Path] = []
        for part in parts:
            if not part:
                continue
            with contextlib.suppress(OSError, RuntimeError, TypeError, ValueError):
                candidate = Path(part).expanduser().resolve(strict=False)
                if candidate.is_absolute():
                    allowed.append(candidate)
        return allowed

    @classmethod
    def _allowed_file_roots(cls) -> list[Path]:
        root = cls._project_root()
        allowed = [
            (root / "Databases").resolve(strict=False),
            (root / "logs").resolve(strict=False),
        ]
        allowed.extend(cls._parse_allowed_dirs(os.getenv("MONITORING_NOTIFY_ALLOWED_DIRS")))
        if is_test_mode() or is_explicit_pytest_runtime():
            with contextlib.suppress(OSError, RuntimeError, TypeError, ValueError):
                allowed.append(Path(tempfile.gettempdir()).resolve(strict=False))
        return allowed

    @staticmethod
    def _path_is_within(path: Path, root: Path) -> bool:
        with contextlib.suppress(ValueError):
            path.relative_to(root)
            return True
        return False

    @classmethod
    def _resolve_file_path(cls, raw_file: str) -> str:
        if raw_file is None or not str(raw_file).strip():
            raise ValueError("Notification file path must be non-empty")
        try:
            fp = Path(str(raw_file))
            if not fp.is_absolute():
                fp = cls._project_root() / fp
            resolved = fp.expanduser().resolve(strict=False)
        except (OSError, RuntimeError, TypeError, ValueError):
            raise
        if not any(cls._path_is_within(resolved, root) for root in cls._allowed_file_roots()):
            raise ValueError("Notification file path is outside allowed directories")
        return str(resolved)

    def is_file_path_allowed(self, raw_file: str) -> bool:
        try:
            self._resolve_file_path(raw_file)
            return True
        except (OSError, RuntimeError, TypeError, ValueError):
            return False

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
        safe_payload = _sanitize_notification_payload(payload)
        # Always append to JSONL file (local-first scaffold)
        file_written = True
        try:
            with self._lock, open(self.file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(safe_payload, ensure_ascii=False) + "\n")
        except (OSError, RuntimeError, TypeError, ValueError) as e:
            file_written = False
            logger.warning("Notification file sink failed ({})", _safe_exception_label(e))
        # Best-effort asynchronous sends through a bounded worker queue.
        if self.webhook_url:
            self._enqueue_delivery("webhook", safe_payload)
        try:
            # Email optional and only if SMTP configured and recipients provided
            recipients = self._parse_email_recipients(self.email_to)
            if recipients and self.smtp_host and self.email_from:
                self._enqueue_delivery("email", alert)
        except (OSError, RuntimeError) as e:
            logger.debug("Email delivery enqueue failed ({})", _safe_exception_label(e))
        return "logged" if file_written else "failed"

    def _start_delivery_worker(self) -> None:
        with self._delivery_worker_lock:
            if self._delivery_worker_started:
                return
            threading.Thread(target=self._delivery_worker, daemon=True).start()
            self._delivery_worker_started = True

    def _enqueue_delivery(self, kind: str, item: Any) -> bool:
        try:
            self._start_delivery_worker()
            self._delivery_queue.put_nowait((kind, item))
            return True
        except queue.Full:
            logger.warning("Notification delivery queue full; dropped {} delivery", kind)
        except (OSError, RuntimeError, TypeError, ValueError) as e:
            logger.debug("{} delivery enqueue failed ({})", kind.capitalize(), _safe_exception_label(e))
        return False

    def _delivery_worker(self) -> None:
        while True:
            kind, item = self._delivery_queue.get()
            try:
                if kind == "webhook":
                    self._send_webhook_safe(item)
                elif kind == "email":
                    self._send_email_safe(item)
                else:
                    logger.debug("Unknown notification delivery kind {}", kind)
            except Exception as exc:  # defensive: safe senders should not raise
                logger.info("Notification delivery failed ({})", _safe_exception_label(exc))
            finally:
                self._delivery_queue.task_done()

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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
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
                try:
                    server.starttls()
                except (OSError, RuntimeError, smtplib.SMTPException) as exc:
                    raise RuntimeError("SMTP STARTTLS failed") from exc
            if self.smtp_user:
                server.login(self.smtp_user, self.smtp_password or "")
            server.sendmail(self.email_from, recipients, msg.as_string())

    def notify_generic(self, payload: dict[str, Any]) -> str:
        """Record a generic notification payload (not tied to TopicAlert).

        Applies severity threshold filtering and writes to JSONL sink.
        Adds ``ts`` to the recorded and delivered copy if not present.
        """
        severity = payload.get("severity") or payload.get("rule_severity")
        if not self._meets_threshold(severity):
            return "skipped"
        payload_to_record = dict(payload)
        payload_to_record.setdefault("ts", datetime.now(timezone.utc).isoformat())
        # lgtm[py/clear-text-storage-sensitive-data]: payload is redacted before notification persistence.
        safe_payload = _sanitize_notification_payload(payload_to_record)
        file_written = True
        try:
            with self._lock, open(self.file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(safe_payload, ensure_ascii=False) + "\n")
        except (OSError, RuntimeError, TypeError, ValueError) as e:
            file_written = False
            logger.warning("Notification file sink failed ({})", _safe_exception_label(e))
        if self.webhook_url:
            self._enqueue_delivery("webhook", safe_payload)
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
                self._store_digest_items_locked(recipient, [dict(payload)])
            return "batched"
        return self.notify_generic(payload)

    def _store_digest_items_locked(
        self,
        recipient_key: str,
        items: list[dict[str, Any]],
        *,
        prepend: bool = False,
    ) -> None:
        existing = self._pending_digests.get(recipient_key, [])
        combined = list(items) + existing if prepend else existing + list(items)
        if len(combined) > self.max_digest_items_per_recipient:
            combined = combined[-self.max_digest_items_per_recipient :]
        self._pending_digests[recipient_key] = combined

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
                        self._store_digest_items_locked(recipient_key, items, prepend=True)

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
