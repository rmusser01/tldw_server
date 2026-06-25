"""
Email Notification Delivery

Formats notification emails and delegates delivery to the AuthNZ email service.
The legacy SMTP config helper still recognizes:
- SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD
- EMAIL_FROM (sender email)
- SMTP_USE_TLS (default: true), SMTP_TIMEOUT (default: 10)

Usage:
    from tldw_Server_API.app.core.Notifications.email_delivery import send_notification_email

    await send_notification_email(
        to="user@example.com",
        subject="Watchlist Alert",
        body_text="Your run completed with 0 items.",
        body_html="<p>Your run completed with <b>0 items</b>.</p>",
    )
"""

from __future__ import annotations

import html
import os
from urllib.parse import urlparse, urlunparse

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.email_service import get_email_service


def _read_env_text(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            stripped = value.strip()
            if stripped:
                return stripped
    return default


def _get_smtp_config() -> dict[str, str | int | bool | float] | None:
    """Read SMTP config from environment. Returns None if not configured."""
    host = os.environ.get("SMTP_HOST", "").strip()
    if not host:
        return None

    raw_port = os.environ.get("SMTP_PORT", "587").strip()
    try:
        port = int(raw_port)
    except ValueError:
        logger.warning("SMTP_PORT is invalid; email delivery disabled")
        return None
    if port < 1 or port > 65535:
        logger.warning(
            "SMTP_PORT is outside the valid TCP port range; email delivery disabled"
        )
        return None

    raw_timeout = os.environ.get("SMTP_TIMEOUT", "10").strip()
    try:
        timeout = float(raw_timeout)
    except ValueError:
        logger.warning("SMTP_TIMEOUT is invalid; email delivery disabled")
        return None
    if timeout <= 0:
        logger.warning("SMTP_TIMEOUT must be positive; email delivery disabled")
        return None

    return {
        "host": host,
        "port": port,
        "user": _read_env_text("SMTP_USERNAME", "SMTP_USER"),
        "password": os.environ.get("SMTP_PASSWORD", "").strip(),
        "from_address": _read_env_text(
            "EMAIL_FROM",
            "SMTP_FROM_ADDRESS",
            default="noreply@tldw.local",
        ),
        "use_tls": os.environ.get("SMTP_USE_TLS", "true").lower()
        in ("true", "1", "yes"),
        "timeout": timeout,
    }


def is_email_delivery_configured() -> bool:
    """Check if legacy SMTP environment settings are configured."""
    return _get_smtp_config() is not None


def _mask_email_address(address: str) -> str:
    local, separator, domain = address.partition("@")
    if not separator or not domain:
        return "[invalid-recipient]"
    prefix = local[:1] if local else "*"
    return f"{prefix}***@{domain}"


def _plain_text_to_html(body_text: str) -> str:
    escaped = html.escape(body_text)
    return f"<pre>{escaped}</pre>"


def _redacted_exception_for_log(exc: BaseException) -> RuntimeError:
    return RuntimeError(f"{type(exc).__name__}: redacted").with_traceback(exc.__traceback__)


async def send_notification_email(
    to: str,
    subject: str,
    body_text: str,
    body_html: str | None = None,
) -> bool:
    """Send a notification email through the AuthNZ email service.

    Returns True on success, False on failure (logged, not raised).
    """
    try:
        return await get_email_service().send_email(
            to_email=to,
            subject=subject,
            html_body=body_html or _plain_text_to_html(body_text),
            text_body=body_text,
        )
    except Exception as exc:
        logger.bind(
            operation="notifications.send_notification_email",
            recipient=_mask_email_address(to),
            exception_type=type(exc).__name__,
        ).opt(exception=_redacted_exception_for_log(exc)).error(
            "Failed to send notification email"
        )
        return False


# ---------------------------------------------------------------------------
# Notification-to-email formatting
# ---------------------------------------------------------------------------

_SAFE_LINK_SCHEMES = {"http", "https", "mailto"}


def _normalize_safe_link_url(link_url: str | None) -> str | None:
    if not link_url:
        return None
    candidate = link_url.strip()
    if not candidate:
        return None
    try:
        parsed = urlparse(candidate)
        if parsed.port is not None and (parsed.port < 1 or parsed.port > 65535):
            return None
    except ValueError:
        return None
    scheme = parsed.scheme.lower()
    if scheme not in _SAFE_LINK_SCHEMES:
        return None
    if scheme in {"http", "https"} and not parsed.netloc:
        return None
    if scheme == "mailto" and not parsed.path:
        return None
    return urlunparse(parsed._replace(scheme=scheme))


def format_notification_email(
    kind: str,
    title: str,
    message: str,
    severity: str,
    link_url: str | None = None,
) -> tuple[str, str, str]:
    """Format a notification into email subject, text body, and HTML body.

    Returns: (subject, body_text, body_html)
    """
    safe_kind = kind.strip() or "notification"
    severity_emoji = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "critical": "🚨"}.get(
        severity, "📬"
    )

    subject = f"{severity_emoji} {title}"

    safe_link_url = _normalize_safe_link_url(link_url)
    body_text = f"{title}\n\n{message}\n\nNotification type: {safe_kind}"
    if safe_link_url:
        body_text += f"\n\nView details: {safe_link_url}"

    severity_color = {
        "info": "#3b82f6",
        "warning": "#f59e0b",
        "error": "#ef4444",
        "critical": "#dc2626",
    }.get(severity, "#6b7280")

    escaped_kind = html.escape(safe_kind)
    escaped_title = html.escape(title)
    escaped_message = html.escape(message)
    link_html = ""
    if safe_link_url:
        escaped_link_url = html.escape(safe_link_url, quote=True)
        link_html = (
            f'<p><a href="{escaped_link_url}" '
            f'style="color: {severity_color}; text-decoration: none; font-weight: 500;">'
            "View details &rarr;</a></p>"
        )

    body_html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 600px; margin: 0 auto;">
        <div style="border-left: 4px solid {severity_color}; padding: 16px; margin: 16px 0; background: #f9fafb; border-radius: 0 8px 8px 0;">
            <h2 style="margin: 0 0 8px 0; font-size: 16px; color: #111827;">{escaped_title}</h2>
            <p style="margin: 0; color: #6b7280; font-size: 14px;">{escaped_message}</p>
        </div>
        {link_html}
        <p style="color: #9ca3af; font-size: 12px; margin-top: 16px;">Notification type: {escaped_kind}</p>
        <p style="color: #9ca3af; font-size: 12px; margin-top: 24px;">Sent by tldw server</p>
    </div>
    """

    return subject, body_text, body_html
