# email_service.py
# Description: Email service with mock provider for development and testing
#
# Imports
import asyncio
import json
import os
import smtplib
import re
from datetime import datetime, timezone
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Optional

#
# 3rd-party imports
from jinja2 import Template
from loguru import logger

#
# Local imports
from tldw_Server_API.app.core.AuthNZ.settings import Settings, get_settings

#######################################################################################################################
#
# Email Templates
#

EMAIL_TEMPLATES = {
    "password_reset": {
        "subject": "Password Reset Request - {{ app_name }}",
        "html": """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background-color: #007bff; color: white; padding: 20px; text-align: center; }
        .content { background-color: #f8f9fa; padding: 30px; margin-top: 20px; }
        .button { display: inline-block; padding: 12px 30px; background-color: #007bff;
                  color: white; text-decoration: none; border-radius: 5px; margin: 20px 0; }
        .footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6;
                  font-size: 0.9em; color: #6c757d; }
        .warning { background-color: #fff3cd; border: 1px solid #ffc107; padding: 10px;
                   margin: 20px 0; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ app_name }}</h1>
        </div>
        <div class="content">
            <h2>Password Reset Request</h2>
            <p>Hello {{ username }},</p>
            <p>We received a request to reset your password. Click the button below to create a new password:</p>
            <center>
                <a href="{{ reset_link }}" class="button">Reset Password</a>
            </center>
            <p>Or copy and paste this link into your browser:</p>
            <p style="word-break: break-all; background: #fff; padding: 10px; border: 1px solid #dee2e6;">
                {{ reset_link }}
            </p>
            <div class="warning">
                <strong>⚠️ Security Notice:</strong><br>
                This link will expire in {{ expiry_hours }} hour(s).<br>
                If you didn't request this, please ignore this email or contact support if you're concerned.
            </div>
        </div>
        <div class="footer">
            <p>This is an automated message from {{ app_name }}.</p>
            <p>Request made from IP: {{ ip_address }}<br>
            Time: {{ timestamp }}</p>
        </div>
    </div>
</body>
</html>
""",
        "text": """
Password Reset Request - {{ app_name }}

Hello {{ username }},

We received a request to reset your password.

To reset your password, visit this link:
{{ reset_link }}

This link will expire in {{ expiry_hours }} hour(s).

If you didn't request this, please ignore this email.

Security Information:
- Request from IP: {{ ip_address }}
- Time: {{ timestamp }}

This is an automated message from {{ app_name }}.
"""
    },
    "email_verification": {
        "subject": "Verify Your Email - {{ app_name }}",
        "html": """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background-color: #28a745; color: white; padding: 20px; text-align: center; }
        .content { background-color: #f8f9fa; padding: 30px; margin-top: 20px; }
        .button { display: inline-block; padding: 12px 30px; background-color: #28a745;
                  color: white; text-decoration: none; border-radius: 5px; margin: 20px 0; }
        .footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6;
                  font-size: 0.9em; color: #6c757d; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Welcome to {{ app_name }}!</h1>
        </div>
        <div class="content">
            <h2>Verify Your Email Address</h2>
            <p>Hello {{ username }},</p>
            <p>Thank you for registering! Please verify your email address to activate your account:</p>
            <center>
                <a href="{{ verification_link }}" class="button">Verify Email</a>
            </center>
            <p>Or copy and paste this link:</p>
            <p style="word-break: break-all; background: #fff; padding: 10px; border: 1px solid #dee2e6;">
                {{ verification_link }}
            </p>
            <p><strong>This link expires in {{ expiry_hours }} hours.</strong></p>
        </div>
        <div class="footer">
            <p>If you didn't create an account, please ignore this email.</p>
        </div>
    </div>
</body>
</html>
""",
        "text": """
Welcome to {{ app_name }}!

Hello {{ username }},

Thank you for registering! Please verify your email address by visiting:

{{ verification_link }}

This link expires in {{ expiry_hours }} hours.

If you didn't create an account, please ignore this email.
"""
    },
    "magic_link": {
        "subject": "Your sign-in link - {{ app_name }}",
        "html": """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background-color: #0f172a; color: white; padding: 20px; text-align: center; }
        .content { background-color: #f8f9fa; padding: 30px; margin-top: 20px; }
        .button { display: inline-block; padding: 12px 30px; background-color: #2563eb;
                  color: white; text-decoration: none; border-radius: 5px; margin: 20px 0; }
        .code-box { background: #fff; border: 1px solid #e5e7eb; padding: 12px;
                    margin: 16px 0; border-radius: 6px; font-family: monospace; word-break: break-all; }
        .footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6;
                  font-size: 0.9em; color: #6c757d; }
        .note { background-color: #fff3cd; border: 1px solid #ffc107; padding: 10px;
                margin: 16px 0; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ app_name }}</h1>
        </div>
        <div class="content">
            <h2>Sign in to your account</h2>
            <p>Hello{{ user_label }},</p>
            <p>Use the link below to sign in. This link expires in {{ expiry_minutes }} minute(s).</p>
            <center>
                <a href="{{ magic_link }}" class="button">Sign in</a>
            </center>
            <p>If you prefer, copy the token below and paste it in the extension:</p>
            <div class="code-box">{{ magic_token }}</div>
            <div class="note">
                <strong>Security notice:</strong> If you did not request this, you can ignore this email.
            </div>
        </div>
        <div class="footer">
            <p>This is an automated message from {{ app_name }}.</p>
        </div>
    </div>
</body>
</html>
""",
        "text": """
Sign in to {{ app_name }}

Hello{{ user_label }},

Use this link to sign in (expires in {{ expiry_minutes }} minute(s)):
{{ magic_link }}

Or copy this token into the extension:
{{ magic_token }}

If you did not request this, you can ignore this email.
"""
    },
    "admin_reauth": {
        "subject": "Confirm admin action - {{ app_name }}",
        "html": """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background-color: #7c2d12; color: white; padding: 20px; text-align: center; }
        .content { background-color: #f8f9fa; padding: 30px; margin-top: 20px; }
        .button { display: inline-block; padding: 12px 30px; background-color: #b45309;
                  color: white; text-decoration: none; border-radius: 5px; margin: 20px 0; }
        .code-box { background: #fff; border: 1px solid #e5e7eb; padding: 12px;
                    margin: 16px 0; border-radius: 6px; font-family: monospace; word-break: break-all; }
        .footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6;
                  font-size: 0.9em; color: #6c757d; }
        .note { background-color: #fff7ed; border: 1px solid #fdba74; padding: 10px;
                margin: 16px 0; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ app_name }}</h1>
        </div>
        <div class="content">
            <h2>Confirm this admin action</h2>
            <p>Hello{{ user_label }},</p>
            <p>Use the link below to complete your admin reauthentication. This link expires in {{ expiry_minutes }} minute(s).</p>
            <center>
                <a href="{{ reauth_link }}" class="button">Confirm admin action</a>
            </center>
            <div class="note">
                <strong>Security notice:</strong> This step-up token only authorizes a high-risk admin action. If you did not request this, ignore the email.
            </div>
        </div>
        <div class="footer">
            <p>This is an automated message from {{ app_name }}.</p>
        </div>
    </div>
</body>
</html>
""",
        "text": """
Confirm admin action in {{ app_name }}

Hello{{ user_label }},

Use this link to complete your admin reauthentication (expires in {{ expiry_minutes }} minute(s)):
{{ reauth_link }}

If you did not request this, ignore this email.
"""
    },
    "user_invitation": {
        "subject": "You've been invited to {{ app_name }}",
        "html": """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background-color: #2563eb; color: white; padding: 20px; text-align: center; }
        .content { background-color: #f8f9fa; padding: 30px; margin-top: 20px; }
        .button { display: inline-block; padding: 12px 30px; background-color: #2563eb;
                  color: white; text-decoration: none; border-radius: 5px; margin: 20px 0; }
        .footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6;
                  font-size: 0.9em; color: #6c757d; }
        .note { background-color: #eff6ff; border: 1px solid #bfdbfe; padding: 10px;
                margin: 16px 0; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ app_name }}</h1>
        </div>
        <div class="content">
            <h2>You're Invited!</h2>
            <p>Hello,</p>
            <p>You have been invited to join <strong>{{ app_name }}</strong> as a <strong>{{ role }}</strong>.</p>
            <p>Click the button below to create your account:</p>
            <center>
                <a href="{{ invite_url }}" class="button">Accept Invitation</a>
            </center>
            <p>Or copy and paste this link into your browser:</p>
            <p style="word-break: break-all; background: #fff; padding: 10px; border: 1px solid #dee2e6;">
                {{ invite_url }}
            </p>
            <div class="note">
                <strong>Note:</strong> This invitation expires in {{ expiry_days }} day(s).
            </div>
        </div>
        <div class="footer">
            <p>If you did not expect this invitation, you can safely ignore this email.</p>
            <p>This is an automated message from {{ app_name }}.</p>
        </div>
    </div>
</body>
</html>
""",
        "text": """
You're Invited to {{ app_name }}!

Hello,

You have been invited to join {{ app_name }} as a {{ role }}.

To accept your invitation and create your account, visit:
{{ invite_url }}

This invitation expires in {{ expiry_days }} day(s).

If you did not expect this invitation, you can safely ignore this email.
"""
    },
    "mfa_enabled": {
        "subject": "Two-Factor Authentication Enabled - {{ app_name }}",
        "html": """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .header { background-color: #17a2b8; color: white; padding: 20px; text-align: center; }
        .content { background-color: #f8f9fa; padding: 30px; margin-top: 20px; }
        .code-box { background: #fff; border: 2px solid #17a2b8; padding: 15px;
                    margin: 20px 0; border-radius: 5px; font-family: monospace; }
        .footer { margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6;
                  font-size: 0.9em; color: #6c757d; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔐 Two-Factor Authentication Enabled</h1>
        </div>
        <div class="content">
            <h2>Your Account is Now More Secure!</h2>
            <p>Hello {{ username }},</p>
            <p>Two-factor authentication has been successfully enabled on your account.</p>

            <h3>Your Backup Codes</h3>
            <p>Save these backup codes in a safe place. Each code can be used once if you lose access to your authenticator app:</p>
            <div class="code-box">
                {% for code in backup_codes %}
                {{ code }}<br>
                {% endfor %}
            </div>

            <p><strong>⚠️ Important:</strong> Store these codes securely. You won't be able to see them again.</p>
        </div>
        <div class="footer">
            <p>Enabled from IP: {{ ip_address }}<br>
            Time: {{ timestamp }}</p>
        </div>
    </div>
</body>
</html>
""",
        "text": """
Two-Factor Authentication Enabled - {{ app_name }}

Hello {{ username }},

Two-factor authentication has been successfully enabled on your account.

Your Backup Codes (save these securely):
{% for code in backup_codes %}
{{ code }}
{% endfor %}

These codes can be used once each if you lose access to your authenticator app.

Enabled from IP: {{ ip_address }}
Time: {{ timestamp }}
"""
    }
}

#######################################################################################################################
#
# Email Service Class
#

class EmailService:
    """
    Email service with support for multiple providers including mock for development
    """

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize email service"""
        self.settings = settings or get_settings()

        # Email configuration
        self.provider = os.getenv("EMAIL_PROVIDER", "mock")  # mock, smtp, sendgrid, etc.
        self.mock_output = os.getenv("EMAIL_MOCK_OUTPUT", "console")  # console, file, both
        # Anchor mock file path to project root when relative
        raw_mock = Path(os.getenv("EMAIL_MOCK_FILE_PATH", "./mock_emails"))
        try:
            if not raw_mock.is_absolute():
                from tldw_Server_API.app.core.Utils.Utils import get_project_root
                raw_mock = Path(get_project_root()) / raw_mock
        except Exception:
            # Fallback: anchor to package root to avoid CWD effects
            if not raw_mock.is_absolute():
                raw_mock = Path(__file__).resolve().parents[4] / raw_mock
        self.mock_file_path = raw_mock

        # SMTP configuration (if using SMTP)
        self.smtp_host = os.getenv("SMTP_HOST", "localhost")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_username = os.getenv("SMTP_USERNAME", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.smtp_use_tls = os.getenv("SMTP_USE_TLS", "true").lower() == "true"

        # Default sender
        self.default_sender = os.getenv("EMAIL_FROM", "noreply@example.com")
        self.app_name = os.getenv("APP_NAME", "TLDW Server")

        # Create mock email directory if needed
        if self.provider == "mock" and self.mock_output in ["file", "both"]:
            self.mock_file_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"EmailService initialized with provider: {self.provider}")

    def _read_setting_text(self, env_name: str, default: Optional[str] = None) -> Optional[str]:
        """Read a string setting from env first, then from the settings object."""
        env_value = os.getenv(env_name)
        if env_value is not None:
            stripped = env_value.strip()
            return stripped or default

        settings_value = getattr(self.settings, env_name, None)
        if isinstance(settings_value, str):
            stripped = settings_value.strip()
            return stripped or default

        return default

    @staticmethod
    def _normalize_public_path(path: str) -> str:
        normalized = str(path or "").strip() or "/"
        if not normalized.startswith("/"):
            normalized = f"/{normalized}"
        return normalized

    def _resolve_public_web_base_url(
        self,
        *,
        base_url: Optional[str],
        hosted_default_path: str,
        legacy_default_path: str,
        configured_public_path: Optional[str],
        token: str,
    ) -> str:
        if base_url is not None:
            resolved_base = base_url.rstrip("/")
            resolved_path = self._normalize_public_path(configured_public_path or legacy_default_path)
            return f"{resolved_base}{resolved_path}?token={token}"

        public_web_base_url = self._read_setting_text("PUBLIC_WEB_BASE_URL")
        if public_web_base_url:
            resolved_path = self._normalize_public_path(
                configured_public_path or hosted_default_path
            )
            return f"{public_web_base_url.rstrip('/')}{resolved_path}?token={token}"

        fallback_base_url = self._read_setting_text("BASE_URL", "http://localhost:8000")
        resolved_path = self._normalize_public_path(legacy_default_path)
        return f"{str(fallback_base_url).rstrip('/')}{resolved_path}?token={token}"

    @staticmethod
    def _record_delivery(
        *,
        recipient: str,
        subject: str,
        template: Optional[str] = None,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        """Record an email delivery attempt to the system-ops store (fire-and-forget)."""
        try:
            from tldw_Server_API.app.services.admin_system_ops_service import (
                record_email_delivery,
            )

            record_email_delivery(
                recipient=recipient,
                subject=subject,
                template=template,
                status=status,
                error=error,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to record email delivery: {}", exc)

    async def send_email(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: Optional[str] = None,
        from_email: Optional[str] = None,
        attachments: Optional[list[dict[str, Any]]] = None,
        *,
        redact_mock_tokens: bool = False,
        _template: Optional[str] = None,
    ) -> bool:
        """
        Send an email

        Args:
            to_email: Recipient email address
            subject: Email subject
            html_body: HTML version of email
            text_body: Plain text version (optional)
            from_email: Sender email (uses default if not provided)
            attachments: List of attachments
            redact_mock_tokens: Whether mock transport should redact token-bearing content
            _template: Internal hint identifying the email template (for delivery tracking)

        Returns:
            True if email was sent successfully
        """
        from_email = from_email or self.default_sender

        if self.provider == "mock":
            result = await self._send_mock_email(
                to_email,
                subject,
                html_body,
                text_body,
                from_email,
                attachments,
                redact_mock_tokens=redact_mock_tokens,
            )
            self._record_delivery(
                recipient=to_email,
                subject=subject,
                template=_template,
                status="sent" if result else "failed",
            )
            return result
        elif self.provider == "smtp":
            try:
                result = await self._send_smtp_email(
                    to_email, subject, html_body, text_body, from_email, attachments
                )
            except Exception as exc:
                self._record_delivery(
                    recipient=to_email,
                    subject=subject,
                    template=_template,
                    status="failed",
                    error=str(exc),
                )
                raise
            self._record_delivery(
                recipient=to_email,
                subject=subject,
                template=_template,
                status="sent" if result else "failed",
                error=None if result else "SMTP send returned False",
            )
            return result
        else:
            logger.error(f"Unsupported email provider: {self.provider}")
            self._record_delivery(
                recipient=to_email,
                subject=subject,
                template=_template,
                status="skipped",
                error=f"Unsupported email provider: {self.provider}",
            )
            return False

    async def _send_mock_email(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: Optional[str],
        from_email: str,
        attachments: Optional[list[dict[str, Any]]],
        *,
        redact_mock_tokens: bool = False,
    ) -> bool:
        """Send mock email for development/testing"""

        timestamp = datetime.now(timezone.utc).isoformat()
        email_id = f"{timestamp}_{to_email.replace('@', '_at_')}"
        stored_html_body = self._redact_mock_email_body(html_body) if redact_mock_tokens else html_body
        stored_text_body = (
            self._redact_mock_email_body(text_body or "")
            if redact_mock_tokens
            else (text_body or "")
        )

        # Create email data structure
        email_data = {
            "id": email_id,
            "timestamp": timestamp,
            "from": from_email,
            "to": to_email,
            "subject": subject,
            "html_body": stored_html_body,
            "text_body": stored_text_body,
            "attachments": len(attachments) if attachments else 0,
            "provider": "mock"
        }

        # Output to console
        if self.mock_output in ["console", "both"]:
            logger.info("=" * 80)
            logger.info("📧 MOCK EMAIL SENT")
            logger.info("=" * 80)
            logger.info(f"From: {from_email}")
            logger.info(f"To: {to_email}")
            logger.info(f"Subject: {subject}")
            logger.info("-" * 80)
            if stored_text_body:
                logger.info("Text Body:")
                logger.info(
                    stored_text_body[:500]
                    + ("..." if len(stored_text_body) > 500 else "")
                )
            logger.info("-" * 80)
            logger.info(f"HTML Body Length: {len(stored_html_body)} characters")
            if attachments:
                logger.info(f"Attachments: {len(attachments)}")
            logger.info("=" * 80)

        # Save to file
        if self.mock_output in ["file", "both"]:
            file_path = self.mock_file_path / f"{email_id}.json"
            with open(file_path, "w") as f:
                json.dump(email_data, f, indent=2)

            # Persist a stub HTML artifact instead of the rendered body so local
            # mock mailboxes do not store sensitive message content at rest.
            html_path = self.mock_file_path / f"{email_id}.html"
            with open(html_path, "w") as f:
                f.write(
                    "<!DOCTYPE html><html><body>"
                    "<p>Mock email body omitted from persisted mock output for security.</p>"
                    "</body></html>"
                )

            logger.debug(f"Mock email saved to: {file_path}")

        # Simulate small delay
        await asyncio.sleep(0.1)

        return True

    @staticmethod
    def _redact_mock_email_body(body: str) -> str:
        """Redact token-bearing content before mock transport persists it."""
        redacted = re.sub(
            r"([?&]token=)([^&\"'\\s<]+)",
            r"\1[REDACTED]",
            body,
        )
        redacted = re.sub(
            r"(?im)(manual token:\s*)(\S+)",
            r"\1[REDACTED]",
            redacted,
        )
        return redacted

    async def _send_smtp_email(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: Optional[str],
        from_email: str,
        attachments: Optional[list[dict[str, Any]]]
    ) -> bool:
        """Send email via SMTP (offloaded to a background thread)."""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = from_email
            msg['To'] = to_email

            if text_body:
                text_part = MIMEText(text_body, 'plain')
                msg.attach(text_part)

            html_part = MIMEText(html_body, 'html')
            msg.attach(html_part)

            if attachments:
                for attachment in attachments:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment['content'])
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= {attachment["filename"]}'
                    )
                    msg.attach(part)

            await asyncio.to_thread(self._deliver_smtp_message, msg)
            logger.info(f"Email sent successfully to {to_email}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email via SMTP: {e}")
            return False

    def _deliver_smtp_message(self, message: MIMEMultipart) -> None:
        """Blocking SMTP delivery executed in a worker thread."""
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            if self.smtp_use_tls:
                server.starttls()

            if self.smtp_username and self.smtp_password:
                server.login(self.smtp_username, self.smtp_password)

            server.send_message(message)

    async def send_password_reset_email(
        self,
        to_email: str,
        username: str,
        reset_token: str,
        ip_address: str = "Unknown",
        base_url: Optional[str] = None
    ) -> bool:
        """
        Send password reset email

        Args:
            to_email: Recipient email
            username: User's username
            reset_token: Password reset token
            ip_address: IP address of request
            base_url: Base URL for reset link

        Returns:
            True if sent successfully
        """
        reset_link = self._resolve_public_web_base_url(
            base_url=base_url,
            hosted_default_path="/auth/reset-password",
            legacy_default_path="/auth/reset-password",
            configured_public_path=self._read_setting_text("PUBLIC_PASSWORD_RESET_PATH"),
            token=reset_token,
        )

        template_data = {
            "app_name": self.app_name,
            "username": username,
            "reset_link": reset_link,
            "expiry_hours": 1,
            "ip_address": ip_address,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        }

        # Render templates
        html_template = Template(EMAIL_TEMPLATES["password_reset"]["html"])
        text_template = Template(EMAIL_TEMPLATES["password_reset"]["text"])

        html_body = html_template.render(**template_data)
        text_body = text_template.render(**template_data)
        subject = Template(EMAIL_TEMPLATES["password_reset"]["subject"]).render(**template_data)

        return await self.send_email(to_email, subject, html_body, text_body, _template="password_reset")

    async def send_verification_email(
        self,
        to_email: str,
        username: str,
        verification_token: str,
        base_url: Optional[str] = None
    ) -> bool:
        """Send email verification email"""
        verification_link = self._resolve_public_web_base_url(
            base_url=base_url,
            hosted_default_path="/auth/verify-email",
            legacy_default_path="/auth/verify-email",
            configured_public_path=self._read_setting_text("PUBLIC_EMAIL_VERIFICATION_PATH"),
            token=verification_token,
        )

        template_data = {
            "app_name": self.app_name,
            "username": username,
            "verification_link": verification_link,
            "expiry_hours": 24
        }

        # Render templates
        html_template = Template(EMAIL_TEMPLATES["email_verification"]["html"])
        text_template = Template(EMAIL_TEMPLATES["email_verification"]["text"])

        html_body = html_template.render(**template_data)
        text_body = text_template.render(**template_data)
        subject = Template(EMAIL_TEMPLATES["email_verification"]["subject"]).render(**template_data)

        return await self.send_email(to_email, subject, html_body, text_body, _template="email_verification")

    async def send_magic_link_email(
        self,
        *,
        to_email: str,
        magic_token: str,
        expires_in_minutes: int,
        username: Optional[str] = None,
        base_url: Optional[str] = None,
        link_path: Optional[str] = None,
    ) -> bool:
        """Send magic link sign-in email."""
        magic_link = self._resolve_public_web_base_url(
            base_url=base_url,
            hosted_default_path="/auth/magic-link",
            legacy_default_path="/magic-link",
            configured_public_path=link_path or self._read_setting_text("PUBLIC_MAGIC_LINK_PATH"),
            token=magic_token,
        )

        user_label = f" {username}" if username else ""
        template_data = {
            "app_name": self.app_name,
            "user_label": user_label,
            "magic_link": magic_link,
            "magic_token": magic_token,
            "expiry_minutes": expires_in_minutes,
        }

        html_template = Template(EMAIL_TEMPLATES["magic_link"]["html"])
        text_template = Template(EMAIL_TEMPLATES["magic_link"]["text"])

        html_body = html_template.render(**template_data)
        text_body = text_template.render(**template_data)
        subject = Template(EMAIL_TEMPLATES["magic_link"]["subject"]).render(**template_data)

        return await self.send_email(to_email, subject, html_body, text_body, _template="magic_link")

    async def send_admin_reauth_email(
        self,
        *,
        to_email: str,
        reauth_token: str,
        expires_in_minutes: int,
        username: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> bool:
        """Send a dedicated admin reauthentication email."""
        base_url = base_url or os.getenv("BASE_URL", "http://localhost:8000")
        reauth_link = f"{base_url.rstrip('/')}/admin/reauth?token={reauth_token}"

        user_label = f" {username}" if username else ""
        template_data = {
            "app_name": self.app_name,
            "user_label": user_label,
            "reauth_link": reauth_link,
            "reauth_token": reauth_token,
            "expiry_minutes": expires_in_minutes,
        }

        html_template = Template(EMAIL_TEMPLATES["admin_reauth"]["html"])
        text_template = Template(EMAIL_TEMPLATES["admin_reauth"]["text"])

        html_body = html_template.render(**template_data)
        text_body = text_template.render(**template_data)
        subject = Template(EMAIL_TEMPLATES["admin_reauth"]["subject"]).render(**template_data)

        return await self.send_email(
            to_email,
            subject,
            html_body,
            text_body,
            redact_mock_tokens=True,
            _template="admin_reauth",
        )

    async def send_user_invitation_email(
        self,
        *,
        to_email: str,
        invite_token: str,
        role: str = "user",
        expiry_days: int = 7,
        base_url: Optional[str] = None,
    ) -> bool:
        """Send a user invitation email with a registration link."""
        invite_url = self._resolve_public_web_base_url(
            base_url=base_url,
            hosted_default_path="/register",
            legacy_default_path="/register",
            configured_public_path=self._read_setting_text("PUBLIC_REGISTRATION_PATH"),
            token=invite_token,
        )

        template_data = {
            "app_name": self.app_name,
            "invite_url": invite_url,
            "role": role,
            "expiry_days": expiry_days,
        }

        html_template = Template(EMAIL_TEMPLATES["user_invitation"]["html"])
        text_template = Template(EMAIL_TEMPLATES["user_invitation"]["text"])

        html_body = html_template.render(**template_data)
        text_body = text_template.render(**template_data)
        subject = Template(EMAIL_TEMPLATES["user_invitation"]["subject"]).render(**template_data)

        return await self.send_email(to_email, subject, html_body, text_body, _template="user_invitation")

    async def send_mfa_enabled_email(
        self,
        to_email: str,
        username: str,
        backup_codes: list[str],
        ip_address: str = "Unknown"
    ) -> bool:
        """Send MFA enabled notification with backup codes"""

        template_data = {
            "app_name": self.app_name,
            "username": username,
            "backup_codes": backup_codes,
            "ip_address": ip_address,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        }

        # Render templates
        html_template = Template(EMAIL_TEMPLATES["mfa_enabled"]["html"])
        text_template = Template(EMAIL_TEMPLATES["mfa_enabled"]["text"])

        html_body = html_template.render(**template_data)
        text_body = text_template.render(**template_data)
        subject = Template(EMAIL_TEMPLATES["mfa_enabled"]["subject"]).render(**template_data)

        return await self.send_email(to_email, subject, html_body, text_body, _template="mfa_enabled")


#######################################################################################################################
#
# Module Functions
#

# Global instance
_email_service: Optional[EmailService] = None


def get_email_service() -> EmailService:
    """Get email service singleton instance"""
    global _email_service
    if not _email_service:
        _email_service = EmailService()
    return _email_service


#
# End of email_service.py
#######################################################################################################################
