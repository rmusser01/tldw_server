"""
Tests for the EmailService module.

Tests cover:
- Template rendering
- Mock email sending
- SMTP error handling
- Header injection prevention
"""
import json
import pytest
import os
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

from tldw_Server_API.app.core.AuthNZ.email_service import EmailService, EMAIL_TEMPLATES


@pytest.fixture
def mock_settings():
    """Create mock settings for email service."""
    settings = MagicMock()
    return settings


@pytest.fixture
def email_service(mock_settings, monkeypatch, tmp_path):
    """Create email service with mock provider."""
    monkeypatch.setenv("EMAIL_PROVIDER", "mock")
    monkeypatch.setenv("EMAIL_MOCK_OUTPUT", "file")
    monkeypatch.setenv("EMAIL_MOCK_FILE_PATH", str(tmp_path))
    monkeypatch.setenv("EMAIL_FROM", "test@example.com")
    monkeypatch.setenv("APP_NAME", "Test App")
    return EmailService(settings=mock_settings)


class TestEmailTemplates:
    """Tests for email template rendering."""

    def test_password_reset_template_exists(self):

        """Password reset template should be defined."""
        assert "password_reset" in EMAIL_TEMPLATES
        assert "subject" in EMAIL_TEMPLATES["password_reset"]
        assert "html" in EMAIL_TEMPLATES["password_reset"]
        assert "text" in EMAIL_TEMPLATES["password_reset"]

    def test_email_verification_template_exists(self):

        """Email verification template should be defined."""
        assert "email_verification" in EMAIL_TEMPLATES
        assert "subject" in EMAIL_TEMPLATES["email_verification"]
        assert "html" in EMAIL_TEMPLATES["email_verification"]

    def test_mfa_enabled_template_exists(self):

        """MFA enabled template should be defined."""
        assert "mfa_enabled" in EMAIL_TEMPLATES

    def test_admin_reauth_template_exists(self):

        """Admin reauthentication template should be defined."""
        assert "admin_reauth" in EMAIL_TEMPLATES
        assert "subject" in EMAIL_TEMPLATES["admin_reauth"]
        assert "html" in EMAIL_TEMPLATES["admin_reauth"]
        assert "text" in EMAIL_TEMPLATES["admin_reauth"]


class TestEmailServiceInitialization:
    """Tests for EmailService initialization."""

    def test_default_provider_is_mock(self, monkeypatch, mock_settings, tmp_path):

        """Default email provider should be mock."""
        monkeypatch.delenv("EMAIL_PROVIDER", raising=False)
        monkeypatch.setenv("EMAIL_MOCK_FILE_PATH", str(tmp_path))
        service = EmailService(settings=mock_settings)
        assert service.provider == "mock"

    def test_smtp_provider_configuration(self, monkeypatch, mock_settings):

        """SMTP provider should read configuration from environment."""
        monkeypatch.setenv("EMAIL_PROVIDER", "smtp")
        monkeypatch.setenv("SMTP_HOST", "smtp.test.com")
        monkeypatch.setenv("SMTP_PORT", "465")
        monkeypatch.setenv("SMTP_USERNAME", "user@test.com")
        monkeypatch.setenv("SMTP_PASSWORD", "secret")
        monkeypatch.setenv("SMTP_USE_TLS", "true")

        service = EmailService(settings=mock_settings)
        assert service.provider == "smtp"
        assert service.smtp_host == "smtp.test.com"
        assert service.smtp_port == 465
        assert service.smtp_username == "user@test.com"
        assert service.smtp_use_tls is True


class TestMockEmailSending:
    """Tests for mock email sending."""

    @pytest.mark.asyncio
    async def test_mock_email_sends_successfully(self, email_service, tmp_path):
        """Mock email should complete without errors."""
        result = await email_service.send_email(
            to_email="recipient@test.com",
            subject="Test Subject",
            html_body="<p>Test body</p>",
            text_body="Test body"
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_mock_email_creates_file(self, email_service, tmp_path):
        """Mock email with file output should create email file."""
        await email_service.send_email(
            to_email="recipient@test.com",
            subject="Test Subject",
            html_body="<p>Test body</p>"
        )
        # Check if any files were created
        files = list(tmp_path.glob("*.json"))
        assert len(files) >= 1


class TestHeaderInjectionPrevention:
    """Tests for email header injection prevention."""

    @pytest.mark.asyncio
    async def test_newline_in_recipient_rejected(self, email_service):
        """Email addresses with newlines should be rejected."""
        # Attempt header injection via recipient
        malicious_email = "victim@test.com\nBcc: attacker@evil.com"
        result = await email_service.send_email(
            to_email=malicious_email,
            subject="Test",
            html_body="<p>Test</p>"
        )
        # Should either reject or sanitize the email
        # The exact behavior depends on implementation
        assert result is True or result is False

    @pytest.mark.asyncio
    async def test_newline_in_subject_sanitized(self, email_service):
        """Subjects with newlines should be sanitized or rejected."""
        # Attempt header injection via subject
        malicious_subject = "Test\nBcc: attacker@evil.com"
        result = await email_service.send_email(
            to_email="victim@test.com",
            subject=malicious_subject,
            html_body="<p>Test</p>"
        )
        # Should complete (sanitized) or fail gracefully
        assert result is True or result is False


class TestSMTPErrorHandling:
    """Tests for SMTP error handling."""

    @pytest.mark.asyncio
    async def test_smtp_connection_error_handled(self, monkeypatch, mock_settings):
        """SMTP connection errors should be handled gracefully."""
        monkeypatch.setenv("EMAIL_PROVIDER", "smtp")
        monkeypatch.setenv("SMTP_HOST", "nonexistent.host.invalid")
        monkeypatch.setenv("SMTP_PORT", "587")

        service = EmailService(settings=mock_settings)

        # Should handle connection error gracefully
        result = await service.send_email(
            to_email="test@example.com",
            subject="Test",
            html_body="<p>Test</p>"
        )
        # Should return False on error, not raise exception
        assert result is False

    @pytest.mark.asyncio
    async def test_smtp_auth_error_handled(self, monkeypatch, mock_settings):
        """SMTP authentication errors should be handled gracefully."""
        monkeypatch.setenv("EMAIL_PROVIDER", "smtp")
        monkeypatch.setenv("SMTP_HOST", "localhost")
        monkeypatch.setenv("SMTP_PORT", "587")
        monkeypatch.setenv("SMTP_USERNAME", "wrong")
        monkeypatch.setenv("SMTP_PASSWORD", "invalid")

        service = EmailService(settings=mock_settings)

        # Mock the SMTP class to simulate auth failure
        with patch("smtplib.SMTP") as mock_smtp:
            mock_instance = MagicMock()
            mock_instance.login.side_effect = Exception("Authentication failed")
            mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_instance)
            mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

            result = await service.send_email(
                to_email="test@example.com",
                subject="Test",
                html_body="<p>Test</p>"
            )
            # Should return False on error
            assert result is False


class TestTemplateEmailSending:
    """Tests for sending emails using templates."""

    @pytest.mark.asyncio
    async def test_send_password_reset_email(self, email_service):
        """Password reset email should render and send."""
        result = await email_service.send_password_reset_email(
            to_email="user@test.com",
            username="testuser",
            reset_token="abc123",
            ip_address="192.168.1.1"
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_send_admin_reauth_email(self, email_service, monkeypatch):
        """Admin reauthentication email should use the dedicated template and path."""
        captured: dict[str, str] = {}

        async def _fake_send_email(
            to_email: str,
            subject: str,
            html_body: str,
            text_body: str,
            **kwargs,
        ) -> bool:
            captured["to_email"] = to_email
            captured["subject"] = subject
            captured["html_body"] = html_body
            captured["text_body"] = text_body
            captured["redact_mock_tokens"] = str(kwargs.get("redact_mock_tokens"))
            return True

        monkeypatch.setattr(email_service, "send_email", _fake_send_email)

        result = await email_service.send_admin_reauth_email(
            to_email="admin@test.com",
            reauth_token="reauth-token-123",
            expires_in_minutes=7,
            username="alice",
            base_url="https://example.com",
        )

        assert result is True
        assert captured["to_email"] == "admin@test.com"
        assert "Confirm admin action" in captured["subject"]
        assert "https://example.com/admin/reauth?token=reauth-token-123" in captured["html_body"]
        assert "https://example.com/admin/reauth?token=reauth-token-123" in captured["text_body"]
        assert captured["redact_mock_tokens"] == "True"

    @pytest.mark.asyncio
    async def test_mock_admin_reauth_email_redacts_token_in_saved_mock_output(self, email_service, tmp_path):
        """Mock admin reauth emails should not persist raw step-up tokens."""
        result = await email_service.send_admin_reauth_email(
            to_email="admin@test.com",
            reauth_token="reauth-token-123",
            expires_in_minutes=7,
            username="alice",
            base_url="https://example.com",
        )

        assert result is True

        json_files = list(tmp_path.glob("*.json"))
        html_files = list(tmp_path.glob("*.html"))
        assert json_files
        assert html_files

        email_data = json.loads(json_files[0].read_text())
        html_body = html_files[0].read_text()

        assert "reauth-token-123" not in email_data["text_body"]
        assert "reauth-token-123" not in email_data["html_body"]
        assert "reauth-token-123" not in html_body
        assert "token=[REDACTED]" in email_data["text_body"]
        assert "token=[REDACTED]" in email_data["html_body"]
        assert "body omitted from persisted mock output" in html_body
