"""Email integration adapters.

This module includes adapters for email operations:
- email_send: Send email via SMTP
"""

from __future__ import annotations

import os
import re
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.core.Workflows.adapters._registry import registry
from tldw_Server_API.app.core.Workflows.adapters.integration._config import EmailSendConfig


@registry.register(
    "email_send",
    category="integration",
    description="Send email",
    parallelizable=False,
    tags=["integration", "email"],
    config_model=EmailSendConfig,
)
async def run_email_send_adapter(config: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Send email via SMTP.

    Config:
      - to: str or list[str] - Recipient(s)
      - subject: str - Email subject
      - body: str - Email body (plain text or HTML)
      - html: bool - Is body HTML (default: False)
      - from_addr: str - From address (default: from env)
      - smtp_host: str - SMTP host (default: from env)
      - smtp_port: int - SMTP port (default: 587)
      - smtp_user: str - SMTP username (default: from env)
      - smtp_pass: str - SMTP password (default: from env)
      - timeout: int - Connection timeout in seconds (default: 30)
    Output:
      - sent: bool
      - message_id: str
    """
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    from tldw_Server_API.app.core.Chat.prompt_template_manager import apply_template_to_string as _tmpl

    # Email validation pattern
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

    if callable(context.get("is_cancelled")) and context["is_cancelled"]():
        return {"__status__": "cancelled"}

    to = config.get("to")
    if not to:
        return {"sent": False, "error": "missing_recipient"}

    if isinstance(to, str):
        to = [t.strip() for t in to.split(",")]

    # Validate email addresses
    for addr in to:
        if not EMAIL_PATTERN.match(addr):
            return {"sent": False, "error": f"invalid_email: {addr}"}

    subject = config.get("subject", "")
    if isinstance(subject, str):
        subject = _tmpl(subject, context) or subject

    # Sanitize subject to prevent header injection (remove newlines and control chars)
    subject = subject.replace("\n", " ").replace("\r", " ")
    # Remove other control characters
    subject = re.sub(r'[\x00-\x1f\x7f]', '', subject)

    body = config.get("body", "")
    if isinstance(body, str):
        body = _tmpl(body, context) or body

    if not body:
        prev = context.get("prev") or context.get("last") or {}
        if isinstance(prev, dict):
            body = prev.get("text") or prev.get("content") or ""

    is_html = bool(config.get("html", False))
    from_addr = config.get("from_addr") or os.getenv("SMTP_FROM", "noreply@localhost")
    smtp_host = config.get("smtp_host") or os.getenv("SMTP_HOST", "localhost")
    smtp_port = int(config.get("smtp_port") or os.getenv("SMTP_PORT", "587"))
    smtp_user = config.get("smtp_user") or os.getenv("SMTP_USER")
    smtp_pass = config.get("smtp_pass") or os.getenv("SMTP_PASS")
    timeout = int(config.get("timeout", 30))

    # TEST_MODE: return simulated result without actually sending
    if is_test_mode():
        return {
            "sent": True,
            "recipients": to,
            "subject": subject,
            "simulated": True,
        }

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = ", ".join(to)

        if is_html:
            msg.attach(MIMEText(body, "html"))
        else:
            msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(smtp_host, smtp_port, timeout=timeout) as server:
            # Try STARTTLS, but allow plaintext on port 25 as fallback
            try:
                server.starttls()
            except smtplib.SMTPNotSupportedError:
                if smtp_port != 25:
                    raise  # Only allow plaintext on port 25
            if smtp_user and smtp_pass:
                server.login(smtp_user, smtp_pass)
            server.sendmail(from_addr, to, msg.as_string())

        return {"sent": True, "recipients": to, "subject": subject}

    except Exception:
        logger.exception("Email send failed")
        return {"sent": False, "error": "Email send failed"}
