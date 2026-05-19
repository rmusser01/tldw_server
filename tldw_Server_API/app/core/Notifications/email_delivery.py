"""
Email Notification Delivery

Sends notification emails via SMTP. Configured through environment variables:
- SMTP_HOST, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD
- EMAIL_FROM (sender email)
- SMTP_USE_TLS (default: true)

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

import asyncio
import html
import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from urllib.parse import urlparse, urlunparse
from uuid import uuid4

from loguru import logger


def _read_env_text(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            stripped = value.strip()
            if stripped:
                return stripped
    return default


def _get_smtp_config() -> dict[str, str | int | bool] | None:
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
    }


def is_email_delivery_configured() -> bool:
    """Check if SMTP is configured for email delivery."""
    return _get_smtp_config() is not None


def _mask_email_address(address: str) -> str:
    local, separator, domain = address.partition("@")
    if not separator or not domain:
        return "[invalid-recipient]"
    prefix = local[:1] if local else "*"
    return f"{prefix}***@{domain}"


def _redact_log_text(value: object, *sensitive_values: str) -> str:
    redacted = str(value)
    for sensitive_value in sensitive_values:
        if sensitive_value:
            redacted = redacted.replace(sensitive_value, "[redacted]")
    return redacted


async def send_notification_email(
    to: str,
    subject: str,
    body_text: str,
    body_html: str | None = None,
) -> bool:
    """Send a notification email via SMTP.

    Returns True on success, False on failure (logged, not raised).
    """
    return await asyncio.to_thread(
        _send_notification_email_sync,
        to,
        subject,
        body_text,
        body_html,
    )


def _send_notification_email_sync(
    to: str,
    subject: str,
    body_text: str,
    body_html: str | None = None,
) -> bool:
    config = _get_smtp_config()
    if not config:
        logger.debug("Email delivery not configured (SMTP_HOST not set)")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = str(config["from_address"])
    msg["To"] = to

    msg.attach(MIMEText(body_text, "plain"))
    if body_html:
        msg.attach(MIMEText(body_html, "html"))

    notification_id = uuid4().hex[:12]
    masked_to = _mask_email_address(to)
    try:
        with smtplib.SMTP(str(config["host"]), int(config["port"])) as server:
            if config["use_tls"]:
                context = ssl.create_default_context()
                server.starttls(context=context)
            if config["user"] and config["password"]:
                server.login(str(config["user"]), str(config["password"]))
            server.sendmail(str(config["from_address"]), to, msg.as_string())

        logger.info(
            "Notification email sent (id={}, recipient={})",
            notification_id,
            masked_to,
        )
        return True
    except Exception as exc:
        logger.error(
            "Failed to send notification email (id={}, recipient={}): {}",
            notification_id,
            masked_to,
            _redact_log_text(exc, to, subject),
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
