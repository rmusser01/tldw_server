import pytest

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
        "SMTP_TIMEOUT",
    ):
        monkeypatch.delenv(name, raising=False)


class _FakeLogger:
    def __init__(self):
        self.bound_calls = []
        self.opt_calls = []
        self.error_calls = []

    def bind(self, **kwargs):
        self.bound_calls.append(kwargs)
        return self

    def opt(self, **kwargs):
        self.opt_calls.append(kwargs)
        return self

    def error(self, message, *args, **kwargs):
        self.error_calls.append((message, args, kwargs))


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


def test_smtp_config_includes_explicit_timeout(monkeypatch):
    _clear_smtp_env(monkeypatch)
    monkeypatch.setenv("SMTP_HOST", "smtp.example.test")
    monkeypatch.setenv("SMTP_PORT", "2525")
    monkeypatch.setenv("SMTP_TIMEOUT", "7.5")

    config = email_delivery._get_smtp_config()

    assert config is not None
    assert config["timeout"] == 7.5


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
async def test_send_notification_email_delegates_to_authnz_email_service(
    monkeypatch,
):
    _clear_smtp_env(monkeypatch)

    calls = []

    class FakeEmailService:
        async def send_email(self, **kwargs):
            calls.append(kwargs)
            return True

    monkeypatch.setattr(email_delivery, "get_email_service", lambda: FakeEmailService(), raising=False)

    result = await email_delivery.send_notification_email(
        to="person@example.test",
        subject="Sensitive subject",
        body_text="plain body",
        body_html="<p>html body</p>",
    )

    assert result is True
    assert calls == [
        {
            "to_email": "person@example.test",
            "subject": "Sensitive subject",
            "html_body": "<p>html body</p>",
            "text_body": "plain body",
        }
    ]


@pytest.mark.asyncio
async def test_send_notification_email_returns_false_when_authnz_delivery_fails(monkeypatch):
    _clear_smtp_env(monkeypatch)
    fake_logger = _FakeLogger()

    class FailingEmailService:
        async def send_email(self, **kwargs):  # noqa: ARG002
            raise RuntimeError("failed person@example.test while sending Sensitive subject")

    monkeypatch.setattr(email_delivery, "get_email_service", lambda: FailingEmailService(), raising=False)
    monkeypatch.setattr(email_delivery, "logger", fake_logger)

    result = await email_delivery.send_notification_email(
        to="person@example.test",
        subject="Sensitive subject",
        body_text="plain body",
    )

    assert result is False
    assert fake_logger.bound_calls[0] == {
        "operation": "notifications.send_notification_email",
        "recipient": "p***@example.test",
        "exception_type": "RuntimeError",
    }
    redacted_exception = fake_logger.opt_calls[0]["exception"]
    assert redacted_exception.__traceback__ is not None
    assert "person@example.test" not in str(redacted_exception)
    assert "Sensitive subject" not in str(redacted_exception)
