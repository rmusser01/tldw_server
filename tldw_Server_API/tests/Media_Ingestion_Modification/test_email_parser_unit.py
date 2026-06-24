import email
from email.message import EmailMessage
from typing import Any

from _pytest.monkeypatch import MonkeyPatch

from tldw_Server_API.app.core.Ingestion_Media_Processing.Email.Email_Processing_Lib import (
    parse_eml_bytes,
    process_email_task,
)


def build_simple_eml(subject: str = "Test Email") -> bytes:
    msg = EmailMessage()
    msg["From"] = "Alice <alice@example.com>"
    msg["To"] = "Bob <bob@example.com>"
    msg["Subject"] = subject
    msg.set_content("Hello Bob,\nThis is a test.")
    msg.add_alternative("<html><body><p>Hello Bob,</p><p>This is a <b>test</b>.</p></body></html>", subtype="html")
    return msg.as_bytes()


def build_nested_eml() -> bytes:


    # Inner message
    inner = EmailMessage()
    inner["From"] = "Inner <inner@example.com>"
    inner["To"] = "Bob <bob@example.com>"
    inner["Subject"] = "Inner"
    inner.set_content("Inner body.")

    # Outer with message/rfc822 attachment
    outer = EmailMessage()
    outer["From"] = "Alice <alice@example.com>"
    outer["To"] = "Bob <bob@example.com>"
    outer["Subject"] = "Outer"
    outer.set_content("Outer body.")
    outer.add_attachment(inner, subtype="rfc822", filename="nested.eml")
    return outer.as_bytes()


def test_email_ingest_module_does_not_patch_stdlib_add_attachment():
    assert EmailMessage.add_attachment.__module__ == "email.message"


def test_parse_eml_bytes_basic():


    data = build_simple_eml()
    content, meta, children = parse_eml_bytes(data, filename="simple.eml", return_children=True)
    assert "Hello Bob" in content
    assert meta["email"]["subject"] == "Test Email"
    assert isinstance(children, list)
    assert len(children) == 0


def test_parse_eml_bytes_does_not_decode_regular_attachment_for_metadata(
    monkeypatch: MonkeyPatch,
) -> None:
    """Verify regular attachment metadata preserves size without decode=True."""
    msg = EmailMessage()
    msg["From"] = "Alice <alice@example.com>"
    msg["To"] = "Bob <bob@example.com>"
    msg["Subject"] = "Attachment"
    msg.set_content("Hello Bob")
    msg.add_attachment(
        b"payload bytes",
        maintype="application",
        subtype="octet-stream",
        filename="payload.bin",
    )

    decoded_attachment_payloads = []
    original_get_payload = email.message.Message.get_payload

    def spy_get_payload(self: email.message.Message, *args: Any, **kwargs: Any) -> Any:
        """Record decode=True calls for the regular attachment payload."""
        decode_arg = kwargs.get("decode", False)
        if len(args) >= 2:
            decode_arg = args[1]
        if decode_arg is True and self.get_filename() == "payload.bin":
            decoded_attachment_payloads.append(self.get_filename())
        return original_get_payload(self, *args, **kwargs)

    monkeypatch.setattr(email.message.Message, "get_payload", spy_get_payload)

    content, meta, children = parse_eml_bytes(
        msg.as_bytes(),
        filename="attachment.eml",
        return_children=False,
    )

    assert "Hello Bob" in content
    assert meta["email"]["attachments"][0]["name"] == "payload.bin"
    assert meta["email"]["attachments"][0]["size"] == len(b"payload bytes")
    assert children == []
    assert decoded_attachment_payloads == []


def test_parse_eml_html_only_chunking_alignment():


    # Build HTML-only email to ensure HTML->text fallback and chunking compatibility
    msg = EmailMessage()
    msg["From"] = "Alice <alice@example.com>"
    msg["To"] = "Bob <bob@example.com>"
    msg["Subject"] = "HTML Only"
    msg.set_content("This is plain body placeholder.")
    # Overwrite payload to be HTML-only part
    msg = EmailMessage()
    msg["From"] = "Alice <alice@example.com>"
    msg["To"] = "Bob <bob@example.com>"
    msg["Subject"] = "HTML Only"
    msg.add_alternative("<html><body><h1>Headline</h1><p>Para one.</p><p>Para two.</p></body></html>", subtype="html")
    data = msg.as_bytes()
    res = process_email_task(
        file_bytes=data,
        filename="html_only.eml",
        perform_chunking=True,
        chunk_options={"method": "sentences", "max_size": 1000, "overlap": 200},
    )
    assert res["status"] in ("Success", "Warning")
    assert isinstance(res.get("chunks"), list)
    assert len(res["chunks"]) >= 1
    # Verify that chunk text has some expected content
    assert any("Para" in ch.get("text", "") or "Headline" in ch.get("text", "") for ch in res["chunks"])  # basic signal


def test_process_email_task_with_nested_children():


    data = build_nested_eml()
    res = process_email_task(
        file_bytes=data,
        filename="outer.eml",
        ingest_attachments=True,
        max_depth=2,
        perform_chunking=True,
    )
    assert res["status"] in ("Success", "Warning")
    # Ensure children parsed
    children = res.get("children")
    assert isinstance(children, list)
    assert len(children) == 1
    child_meta = children[0].get("metadata", {})
    assert child_meta.get("email", {}).get("subject") == "Inner"
