from unittest.mock import MagicMock

import pytest

from tldw_Server_API.app.core.AuthNZ.email_service import EmailService


@pytest.fixture
def email_service(monkeypatch, tmp_path):
    monkeypatch.setenv("EMAIL_PROVIDER", "mock")
    monkeypatch.setenv("EMAIL_MOCK_OUTPUT", "file")
    monkeypatch.setenv("EMAIL_MOCK_FILE_PATH", str(tmp_path))
    monkeypatch.setenv("EMAIL_FROM", "test@example.com")
    monkeypatch.setenv("APP_NAME", "Test App")
    return EmailService(settings=MagicMock())


@pytest.mark.asyncio
async def test_password_reset_uses_public_web_base_url_and_path(
    email_service, monkeypatch
):
    monkeypatch.setenv("PUBLIC_WEB_BASE_URL", "https://app.example.com")
    monkeypatch.setenv("PUBLIC_PASSWORD_RESET_PATH", "/account/recover")
    email_service.app_name = "Test & App"

    captured: dict[str, str] = {}

    async def _fake_send_email(to_email, subject, html_body, text_body, **kwargs):
        captured["to_email"] = to_email
        captured["subject"] = subject
        captured["html_body"] = html_body
        captured["text_body"] = text_body
        return True

    monkeypatch.setattr(email_service, "send_email", _fake_send_email)

    result = await email_service.send_password_reset_email(
        to_email="user@example.com",
        username="test-user",
        reset_token="reset-token-123",
        ip_address="127.0.0.1",
    )

    assert result is True
    reset_url = "https://app.example.com/account/recover?token=reset-token-123"
    assert captured["subject"] == "Reset your Test & App password"
    assert "Test &amp; App" in captured["html_body"]
    assert captured["html_body"].count("<a ") == 1
    assert f'href="{reset_url}"' in captured["html_body"]
    assert "Choose a new password</a>" in captured["html_body"]
    assert f"Choose a new password:\n{reset_url}" in captured["text_body"]
    assert "127.0.0.1" in captured["html_body"]
    assert "127.0.0.1" in captured["text_body"]


@pytest.mark.asyncio
async def test_verification_uses_public_web_base_url_and_path(
    email_service, monkeypatch
):
    monkeypatch.setenv("PUBLIC_WEB_BASE_URL", "https://app.example.com")
    monkeypatch.setenv("PUBLIC_EMAIL_VERIFICATION_PATH", "/auth/verify-email")

    captured: dict[str, str] = {}

    async def _fake_send_email(to_email, subject, html_body, text_body, **kwargs):
        captured["to_email"] = to_email
        captured["subject"] = subject
        captured["html_body"] = html_body
        captured["text_body"] = text_body
        return True

    monkeypatch.setattr(email_service, "send_email", _fake_send_email)

    result = await email_service.send_verification_email(
        to_email="user@example.com",
        username="test-user",
        verification_token="verify-token-123",
    )

    assert result is True
    assert "https://app.example.com/auth/verify-email?token=verify-token-123" in captured["html_body"]
    assert "https://app.example.com/auth/verify-email?token=verify-token-123" in captured["text_body"]


@pytest.mark.asyncio
async def test_magic_link_uses_public_web_base_url_and_path(
    email_service, monkeypatch
):
    monkeypatch.setenv("PUBLIC_WEB_BASE_URL", "https://app.example.com")
    monkeypatch.setenv("PUBLIC_MAGIC_LINK_PATH", "/account/sign-in")

    captured: dict[str, str] = {}

    async def _fake_send_email(to_email, subject, html_body, text_body, **kwargs):
        captured["to_email"] = to_email
        captured["subject"] = subject
        captured["html_body"] = html_body
        captured["text_body"] = text_body
        return True

    monkeypatch.setattr(email_service, "send_email", _fake_send_email)

    result = await email_service.send_magic_link_email(
        to_email="user@example.com",
        magic_token="magic-token-123",
        expires_in_minutes=15,
        username="test-user",
    )

    assert result is True
    sign_in_url = "https://app.example.com/account/sign-in?token=magic-token-123"
    assert captured["subject"] == "Your sign-in link for Test App"
    assert captured["html_body"].count("<a ") == 1
    assert f'href="{sign_in_url}"' in captured["html_body"]
    assert "Sign in</a>" in captured["html_body"]
    assert ">magic-token-123</div>" in captured["html_body"]
    assert f"Sign in:\n{sign_in_url}" in captured["text_body"]
    assert "Copy this token into the extension:\nmagic-token-123" in captured["text_body"]


@pytest.mark.asyncio
async def test_auth_emails_fall_back_to_base_url_when_public_web_base_url_missing(
    email_service, monkeypatch
):
    monkeypatch.delenv("PUBLIC_WEB_BASE_URL", raising=False)
    monkeypatch.delenv("PUBLIC_PASSWORD_RESET_PATH", raising=False)
    monkeypatch.delenv("PUBLIC_EMAIL_VERIFICATION_PATH", raising=False)
    monkeypatch.delenv("PUBLIC_MAGIC_LINK_PATH", raising=False)
    monkeypatch.setenv("BASE_URL", "https://api.example.com")

    captured: list[tuple[str, str]] = []

    async def _fake_send_email(to_email, subject, html_body, text_body, **kwargs):
        captured.append((html_body, text_body))
        return True

    monkeypatch.setattr(email_service, "send_email", _fake_send_email)

    await email_service.send_password_reset_email(
        to_email="user@example.com",
        username="test-user",
        reset_token="reset-token-123",
    )
    await email_service.send_verification_email(
        to_email="user@example.com",
        username="test-user",
        verification_token="verify-token-123",
    )
    await email_service.send_magic_link_email(
        to_email="user@example.com",
        magic_token="magic-token-123",
        expires_in_minutes=15,
        username="test-user",
    )

    assert "https://api.example.com/auth/reset-password?token=reset-token-123" in captured[0][0]
    assert "https://api.example.com/auth/verify-email?token=verify-token-123" in captured[1][0]
    assert "https://api.example.com/magic-link?token=magic-token-123" in captured[2][0]
