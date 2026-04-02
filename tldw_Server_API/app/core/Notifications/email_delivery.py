"""
Email Notification Delivery

Sends notification emails via SMTP. Configured through environment variables:
- SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD
- SMTP_FROM_ADDRESS (sender email)
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

import os
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from loguru import logger


def _get_smtp_config() -> dict[str, str | int | bool] | None:
    """Read SMTP config from environment. Returns None if not configured."""
    host = os.environ.get("SMTP_HOST", "").strip()
    if not host:
        return None
    return {
        "host": host,
        "port": int(os.environ.get("SMTP_PORT", "587")),
        "user": os.environ.get("SMTP_USER", "").strip(),
        "password": os.environ.get("SMTP_PASSWORD", "").strip(),
        "from_address": os.environ.get("SMTP_FROM_ADDRESS", "noreply@tldw.local").strip(),
        "use_tls": os.environ.get("SMTP_USE_TLS", "true").lower() in ("true", "1", "yes"),
    }


def is_email_delivery_configured() -> bool:
    """Check if SMTP is configured for email delivery."""
    return _get_smtp_config() is not None


def send_notification_email(
    to: str,
    subject: str,
    body_text: str,
    body_html: str | None = None,
) -> bool:
    """Send a notification email via SMTP.

    Returns True on success, False on failure (logged, not raised).
    """
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

    try:
        if config["use_tls"]:
            context = ssl.create_default_context()
            with smtplib.SMTP(str(config["host"]), int(config["port"])) as server:
                server.starttls(context=context)
                if config["user"] and config["password"]:
                    server.login(str(config["user"]), str(config["password"]))
                server.sendmail(str(config["from_address"]), to, msg.as_string())
        else:
            with smtplib.SMTP(str(config["host"]), int(config["port"])) as server:
                if config["user"] and config["password"]:
                    server.login(str(config["user"]), str(config["password"]))
                server.sendmail(str(config["from_address"]), to, msg.as_string())

        logger.info(f"Notification email sent to {to}: {subject}")
        return True
    except Exception as exc:
        logger.error(f"Failed to send notification email to {to}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Notification-to-email formatting
# ---------------------------------------------------------------------------

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
    severity_emoji = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "critical": "🚨"}.get(
        severity, "📬"
    )

    subject = f"{severity_emoji} {title}"

    body_text = f"{title}\n\n{message}"
    if link_url:
        body_text += f"\n\nView details: {link_url}"

    severity_color = {
        "info": "#3b82f6",
        "warning": "#f59e0b",
        "error": "#ef4444",
        "critical": "#dc2626",
    }.get(severity, "#6b7280")

    body_html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; max-width: 600px; margin: 0 auto;">
        <div style="border-left: 4px solid {severity_color}; padding: 16px; margin: 16px 0; background: #f9fafb; border-radius: 0 8px 8px 0;">
            <h2 style="margin: 0 0 8px 0; font-size: 16px; color: #111827;">{title}</h2>
            <p style="margin: 0; color: #6b7280; font-size: 14px;">{message}</p>
        </div>
        {"<p><a href='" + link_url + "' style='color: " + severity_color + "; text-decoration: none; font-weight: 500;'>View details →</a></p>" if link_url else ""}
        <p style="color: #9ca3af; font-size: 12px; margin-top: 24px;">Sent by tldw server</p>
    </div>
    """

    return subject, body_text, body_html
