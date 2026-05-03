import pytest
from loguru import logger

from tldw_Server_API.app.core.Notifications import email_delivery


def _clear_smtp_env(monkeypatch):
    for name in (
        "SMTP_HOST",
        "SMTP_PORT",
        "SMTP_USER",
        "SMTP_USERNAME",
        "SMTP_PASSWORD",
        "SMTP_FROM_ADDRESS",
        "EMAIL_FROM",
        "SMTP_USE_TLS",
    ):
        monkeypatch.delenv(name, raising=False)


def test_invalid_smtp_port_is_treated_as_unconfigured(monkeypatch):
    _clear_smtp_env(monkeypatch)
    monkeypatch.setenv("SMTP_HOST", "smtp.example.test")
    monkeypatch.setenv("SMTP_PORT", "not-a-number")

    assert email_delivery.is_email_delivery_configured() is False


def test_smtp_config_uses_canonical_email_service_env_names(monkeypatch):
    _clear_smtp_env(monkeypatch)
    monkeypatch.setenv("SMTP_HOST", "smtp.example.test")
    monkeypatch.setenv("SMTP_PORT", "2525")
    monkeypatch.setenv("SMTP_USERNAME", "smtp-user")
    monkeypatch.setenv("SMTP_PASSWORD", "smtp-password")
    monkeypatch.setenv("EMAIL_FROM", "alerts@example.test")

    config = email_delivery._get_smtp_config()

    assert config is not None
    assert config["user"] == "smtp-user"
    assert config["from_address"] == "alerts@example.test"


def test_format_notification_email_escapes_html_and_drops_unsafe_links():
    _subject, body_text, body_html = email_delivery.format_notification_email(
        kind="watchlist",
        title="<img src=x onerror=alert(1)>",
        message="Hello <script>alert(1)</script>",
        severity="warning",
        link_url="javascript:alert(1)",
    )

    assert "<img" not in body_html
    assert "<script" not in body_html
    assert "&lt;img" in body_html
    assert "&lt;script&gt;" in body_html
    assert "javascript:" not in body_html
    assert "View details" not in body_text


@pytest.mark.asyncio
async def test_send_notification_email_uses_safe_logs_and_canonical_sender(
    monkeypatch,
):
    _clear_smtp_env(monkeypatch)
    monkeypatch.setenv("SMTP_HOST", "smtp.example.test")
    monkeypatch.setenv("SMTP_PORT", "2525")
    monkeypatch.setenv("SMTP_USERNAME", "smtp-user")
    monkeypatch.setenv("SMTP_PASSWORD", "smtp-password")
    monkeypatch.setenv("EMAIL_FROM", "alerts@example.test")

    smtp_calls = []

    class FakeSMTP:
        def __init__(self, host, port):
            smtp_calls.append(("connect", host, port))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def starttls(self, context):
            smtp_calls.append(("starttls", context is not None))

        def login(self, user, password):
            smtp_calls.append(("login", user, password))

        def sendmail(self, from_address, to_address, payload):
            smtp_calls.append(("sendmail", from_address, to_address, payload))

    monkeypatch.setattr(email_delivery.smtplib, "SMTP", FakeSMTP)

    logs = []
    sink_id = logger.add(lambda message: logs.append(str(message)), format="{message}")
    try:
        result = await email_delivery.send_notification_email(
            to="person@example.test",
            subject="Sensitive subject",
            body_text="plain body",
            body_html="<p>html body</p>",
        )
    finally:
        logger.remove(sink_id)

    assert result is True
    assert ("login", "smtp-user", "smtp-password") in smtp_calls
    assert any(call[:3] == ("sendmail", "alerts@example.test", "person@example.test") for call in smtp_calls)
    rendered_logs = "\n".join(logs)
    assert "person@example.test" not in rendered_logs
    assert "Sensitive subject" not in rendered_logs
    assert "p***@example.test" in rendered_logs


@pytest.mark.asyncio
async def test_send_notification_email_redacts_failure_logs(monkeypatch):
    _clear_smtp_env(monkeypatch)
    monkeypatch.setenv("SMTP_HOST", "smtp.example.test")
    monkeypatch.setenv("SMTP_PORT", "2525")
    monkeypatch.setenv("EMAIL_FROM", "alerts@example.test")

    class FailingSMTP:
        def __init__(self, host, port):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def starttls(self, context):
            pass

        def sendmail(self, from_address, to_address, payload):
            raise RuntimeError(
                "failed person@example.test while sending Sensitive subject"
            )

    monkeypatch.setattr(email_delivery.smtplib, "SMTP", FailingSMTP)

    logs = []
    sink_id = logger.add(lambda message: logs.append(str(message)), format="{message}")
    try:
        result = await email_delivery.send_notification_email(
            to="person@example.test",
            subject="Sensitive subject",
            body_text="plain body",
        )
    finally:
        logger.remove(sink_id)

    assert result is False
    rendered_logs = "\n".join(logs)
    assert "person@example.test" not in rendered_logs
    assert "Sensitive subject" not in rendered_logs
    assert "p***@example.test" in rendered_logs
    assert "[redacted]" in rendered_logs
