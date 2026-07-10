from array import array

import pytest

from tldw_Server_API.app.core.Notifications import service as notifications_service
from tldw_Server_API.app.core.Notifications.service import NotificationsService


class _FakeEmailService:
    def __init__(self):
        self.calls = []

    async def send_email(self, *, to_email, subject, html_body, text_body, attachments=None):
        self.calls.append(
            {
                "to_email": to_email,
                "subject": subject,
                "html_body": html_body,
                "text_body": text_body,
                "attachments": attachments,
            }
        )
        return True


class _FailingEmailService:
    async def send_email(self, *, to_email, subject, html_body, text_body, attachments=None):  # noqa: ARG002
        raise RuntimeError(f"failed to send to {to_email} with subject {subject}")


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


class _FakeDocService:
    def __init__(self):
        self.calls = []

    def create_manual_document(
        self,
        *,
        title,
        content,
        document_type,
        metadata,
        provider,
        model,
        conversation_id=None,
    ):

        self.calls.append(
            {
                "title": title,
                "content": content,
                "document_type": document_type,
                "metadata": metadata,
                "provider": provider,
                "model": model,
                "conversation_id": conversation_id,
            }
        )
        return 42


@pytest.mark.asyncio
async def test_notifications_email_delivery(monkeypatch):
    fake_email = _FakeEmailService()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Notifications.service.get_email_service",
        lambda: fake_email,
    )

    svc = NotificationsService(user_id=1, user_email="user@example.com")
    result = await svc.deliver_email(
        subject="Hello",
        html_body="<p>Hello</p>",
        text_body="Hello",
        recipients=None,
        attachments=[{"filename": "demo.txt", "content": "ZXhhbXBsZQ=="}],
    )

    assert result.channel == "email"
    assert result.status == "sent"
    assert fake_email.calls
    call = fake_email.calls[0]
    assert call["to_email"] == "user@example.com"
    assert call["subject"] == "Hello"


@pytest.mark.asyncio
async def test_notifications_email_skips_without_recipient(monkeypatch):
    fake_email = _FakeEmailService()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Notifications.service.get_email_service",
        lambda: fake_email,
    )

    svc = NotificationsService(user_id=1, user_email=None)
    result = await svc.deliver_email(
        subject="Nope",
        html_body="<p>ignored</p>",
        text_body=None,
        recipients=[],
        fallback_to_user_email=False,
    )

    assert result.status == "skipped"
    assert result.details["reason"] == "no_recipients"
    assert not fake_email.calls


@pytest.mark.asyncio
async def test_notifications_email_rejects_too_many_recipients(monkeypatch):
    fake_email = _FakeEmailService()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Notifications.service.get_email_service",
        lambda: fake_email,
    )
    monkeypatch.setattr(notifications_service, "MAX_EMAIL_RECIPIENTS", 2, raising=False)

    svc = NotificationsService(user_id=1, user_email=None)
    result = await svc.deliver_email(
        subject="Bulk",
        html_body="<p>Bulk</p>",
        text_body="Bulk",
        recipients=["a@example.com", "b@example.com", "c@example.com"],
    )

    assert result.status == "failed"
    assert result.details["reason"] == "too_many_recipients"
    assert result.details["recipient_count"] == 3
    assert fake_email.calls == []


@pytest.mark.asyncio
async def test_notifications_email_recipient_validation_rejects_malformed_addresses(monkeypatch):
    fake_email = _FakeEmailService()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Notifications.service.get_email_service",
        lambda: fake_email,
    )

    svc = NotificationsService(user_id=1, user_email=None)
    result = await svc.deliver_email(
        subject="Bad recipients",
        html_body="<p>Bad</p>",
        text_body="Bad",
        recipients=[
            "valid@example.com",
            "missing-domain@",
            "missing-dot@example",
            "extra@example.com@example.org",
            "white space@example.org",
        ],
    )

    assert result.status == "failed"
    assert result.details["reason"] == "invalid_recipients"
    assert result.details["invalid_recipient_count"] == 4
    assert fake_email.calls == []


@pytest.mark.asyncio
async def test_notifications_email_rejects_oversized_attachments(monkeypatch):
    fake_email = _FakeEmailService()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Notifications.service.get_email_service",
        lambda: fake_email,
    )
    monkeypatch.setattr(notifications_service, "MAX_EMAIL_ATTACHMENT_BYTES", 4, raising=False)

    svc = NotificationsService(user_id=1, user_email="user@example.com")
    result = await svc.deliver_email(
        subject="Large",
        html_body="<p>Large</p>",
        text_body="Large",
        recipients=None,
        attachments=[{"filename": "large.txt", "content": b"12345"}],
    )

    assert result.status == "failed"
    assert result.details["reason"] == "attachment_limit_exceeded"
    assert result.details["attachment_count"] == 1
    assert fake_email.calls == []


@pytest.mark.asyncio
async def test_notifications_email_sizes_memoryview_attachments_by_bytes(monkeypatch):
    fake_email = _FakeEmailService()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Notifications.service.get_email_service",
        lambda: fake_email,
    )
    monkeypatch.setattr(notifications_service, "MAX_EMAIL_ATTACHMENT_BYTES", 3, raising=False)

    svc = NotificationsService(user_id=1, user_email="user@example.com")
    result = await svc.deliver_email(
        subject="Wide memoryview",
        html_body="<p>Large</p>",
        text_body="Large",
        recipients=None,
        attachments=[
            {
                "filename": "wide.bin",
                "content": memoryview(array("H", [1, 2])),
            }
        ],
    )

    assert result.status == "failed"
    assert result.details["reason"] == "attachment_limit_exceeded"
    assert fake_email.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("filename", ["null-\x00.txt", "unit-\x1f.txt", "delete-\x7f.txt"])
async def test_notifications_email_rejects_control_char_attachment_filenames(
    monkeypatch,
    filename,
):
    fake_email = _FakeEmailService()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Notifications.service.get_email_service",
        lambda: fake_email,
    )

    svc = NotificationsService(user_id=1, user_email="user@example.com")
    result = await svc.deliver_email(
        subject="Bad filename",
        html_body="<p>Bad</p>",
        text_body="Bad",
        recipients=None,
        attachments=[{"filename": filename, "content": b"ok"}],
    )

    assert result.status == "failed"
    assert result.details["reason"] == "invalid_attachment_filename"
    assert fake_email.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "attachment",
    [
        {"filename": "missing-content.txt"},
        {"filename": "bad-content.txt", "content": object()},
    ],
)
async def test_notifications_email_rejects_invalid_attachment_content(monkeypatch, attachment):
    fake_email = _FakeEmailService()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Notifications.service.get_email_service",
        lambda: fake_email,
    )

    svc = NotificationsService(user_id=1, user_email="user@example.com")
    result = await svc.deliver_email(
        subject="Invalid attachment",
        html_body="<p>Invalid</p>",
        text_body="Invalid",
        recipients=None,
        attachments=[attachment],
    )

    assert result.status == "failed"
    assert result.details["reason"] == "invalid_attachment_content"
    assert result.details["attachment_count"] == 1
    assert fake_email.calls == []


@pytest.mark.asyncio
async def test_notifications_email_redacts_failure_details(monkeypatch):
    fake_logger = _FakeLogger()
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Notifications.service.get_email_service",
        lambda: _FailingEmailService(),
    )
    monkeypatch.setattr(notifications_service, "logger", fake_logger)

    svc = NotificationsService(user_id=1, user_email=None)
    result = await svc.deliver_email(
        subject="Sensitive Subject",
        html_body="<p>fail</p>",
        text_body="fail",
        recipients=["person@example.com"],
    )

    rendered_details = repr(result.details)
    assert result.status == "failed"
    assert "person@example.com" not in rendered_details
    assert "Sensitive Subject" not in rendered_details
    assert "RuntimeError" in rendered_details
    assert "p***@example.com" in rendered_details
    assert fake_logger.bound_calls[0] == {
        "operation": "notifications.deliver_email",
        "user_id": 1,
        "recipient": "p***@example.com",
        "exception_type": "RuntimeError",
    }
    redacted_exception = fake_logger.opt_calls[0]["exception"]
    assert redacted_exception.__traceback__ is not None
    assert "person@example.com" not in str(redacted_exception)
    assert "Sensitive Subject" not in str(redacted_exception)


def test_notifications_chatbook_delivery(monkeypatch):

    fake_doc = _FakeDocService()
    svc = NotificationsService(user_id=2, user_email=None)

    monkeypatch.setattr(
        NotificationsService,
        "_ensure_doc_service",
        lambda self: fake_doc,
    )

    result = svc.deliver_chatbook(
        title="Watchlist Brief",
        content="Summary content",
        description="Daily brief",
        metadata={"source": "watchlist"},
    )

    assert result.channel == "chatbook"
    assert result.status == "stored"
    assert fake_doc.calls
    call = fake_doc.calls[0]
    assert call["metadata"]["source"] == "watchlist"
    assert call["metadata"]["description"] == "Daily brief"


def test_notifications_chatbook_failure_redacts_logged_and_returned_error(monkeypatch):
    class FailingDocService:
        def create_manual_document(self, **_kwargs):
            raise RuntimeError("secret@example.com private briefing content")

    fake_logger = _FakeLogger()
    monkeypatch.setattr(notifications_service, "logger", fake_logger)
    monkeypatch.setattr(
        NotificationsService,
        "_ensure_doc_service",
        lambda self: FailingDocService(),
    )

    result = NotificationsService(user_id=2).deliver_chatbook(
        title="Secret title",
        content="Private body",
    )

    assert result.status == "failed"
    assert result.details == {"error_type": "RuntimeError"}
    assert fake_logger.bound_calls[0] == {
        "operation": "notifications.deliver_chatbook",
        "user_id": 2,
        "exception_type": "RuntimeError",
    }
    assert "secret@example.com" not in str(fake_logger.opt_calls[0]["exception"])
    assert "private briefing" not in str(fake_logger.opt_calls[0]["exception"])
