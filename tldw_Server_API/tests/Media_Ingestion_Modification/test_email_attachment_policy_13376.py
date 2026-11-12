"""Synthetic attachment policy behavior; no provider or model calls."""
import io
import quopri
import sys
import zipfile
from email.message import EmailMessage, Message
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.media_request_models import AddMediaForm, ProcessEmailsForm
from tldw_Server_API.app.core.Ingestion_Media_Processing.Email import Email_Processing_Lib as lib


def _message(*, mime="message/rfc822", name="nested.eml", grandchild=False):
    """Build a real MIME tree with a nested EML and regular binary descriptor."""
    inner = EmailMessage()
    inner["Subject"] = "Inner synthetic"
    inner.set_content("PRIVATE ATTACHMENT BODY")
    if grandchild:
        deepest = EmailMessage()
        deepest["Subject"] = "Deep synthetic"
        deepest.set_content("Deep body")
        inner.add_attachment(deepest, filename="deep.eml")
    outer = EmailMessage()
    outer["Subject"] = "Outer synthetic"
    outer.set_content("PUBLIC PARENT BODY")
    if mime == "message/rfc822":
        outer.add_attachment(inner, filename=name)
    else:
        maintype, subtype = mime.split("/")
        outer.add_attachment(inner.as_bytes(), maintype=maintype, subtype=subtype, filename=name)
    outer.add_attachment(b"binary metadata", maintype="application", subtype="pdf", filename="paper.pdf")
    return outer.as_bytes()


def _process(data=None, **options):
    return lib.process_email_task(file_bytes=data or _message(), filename="outer.eml", perform_chunking=False, **options)


def test_default_metadata_only_never_serializes_attachment_or_processes_child(monkeypatch):
    data = _message()
    original_as_bytes = Message.as_bytes
    original_process = lib.process_email_task

    def intercept_serialization(self, *args, **kwargs):
        if self.get("Subject") == "Inner synthetic":
            pytest.fail("Metadata-only import serialized an attachment")
        return original_as_bytes(self, *args, **kwargs)

    def intercept_child(**kwargs):
        pytest.fail("Metadata-only import invoked a child processor")

    monkeypatch.setattr(Message, "as_bytes", intercept_serialization)
    monkeypatch.setattr(lib, "process_email_task", intercept_child)
    result = original_process(file_bytes=data, filename="outer.eml", perform_chunking=False)
    assert result["status"] == "Success"
    assert result["metadata"]["email"]["attachments"][0]["extraction_reason"] == "disabled"


def test_explicit_metadata_only_overrides_legacy_extraction_and_does_not_decode(monkeypatch):
    data = _message(mime="application/octet-stream")
    original_payload = Message.get_payload

    def intercept_decode(self, *args, **kwargs):
        if self.get_filename() and kwargs.get("decode"):
            pytest.fail("Metadata-only import decoded an attachment")
        return original_payload(self, *args, **kwargs)

    monkeypatch.setattr(Message, "get_payload", intercept_decode)
    result = _process(data, ingest_attachments=True, extract_attachments=False, max_depth=3)
    assert "children" not in result
    assert len(result["metadata"]["email"]["attachments"]) == 2


@pytest.mark.parametrize("options", [
    {},
    {"ingest_attachments": True, "extract_attachments": False},
])
def test_metadata_only_never_decodes_quoted_printable_attachment(monkeypatch, options):
    """Attachment size inspection must not decode skipped payloads."""
    message = EmailMessage()
    message["Subject"] = "Quoted-printable synthetic"
    message.set_content("PUBLIC PARENT BODY")
    message.add_attachment(
        b"PRIVATE = ATTACHMENT", maintype="application", subtype="octet-stream",
        filename="payload.bin", cte="quoted-printable",
    )
    data = message.as_bytes()

    def fail_decode(*_args, **_kwargs):
        pytest.fail("Metadata-only import decoded a quoted-printable attachment")

    monkeypatch.setattr(quopri, "decodestring", fail_decode)
    result = _process(data, **options)
    assert result["status"] == "Success"
    assert "children" not in result
    assert result["content"] == "PUBLIC PARENT BODY"
    descriptor = result["metadata"]["email"]["attachments"][0]
    assert descriptor["name"] == "payload.bin"
    assert descriptor["size"] is None
    assert descriptor["extraction_reason"] == "disabled"


def test_nested_attachment_body_is_excluded_from_parent():
    result = _process(ingest_attachments=True, max_depth=2)
    assert result["content"] == "PUBLIC PARENT BODY"
    assert result["children"][0]["content"] == "PRIVATE ATTACHMENT BODY"


@pytest.mark.parametrize("options,expected", [
    ({"extract_attachments": True}, True),
    ({"ingest_attachments": True}, True),
    ({"extract_attachments": True, "attachment_mime_allowlist": []}, False),
    ({"extract_attachments": True, "attachment_mime_allowlist": ["application/pdf"]}, False),
    ({"extract_attachments": True, "attachment_mime_allowlist": ["message/*"]}, True),
    ({"extract_attachments": True, "attachment_mime_allowlist": ["*/*"], "attachment_mime_denylist": ["message/*"]}, False),
])
def test_mime_selection_uses_supported_processor_only(options, expected):
    result = _process(max_depth=2, **options)
    assert bool(result.get("children")) is expected
    assert result["metadata"]["email"]["attachments"][1]["extraction_status"] == "skipped"


@pytest.mark.parametrize("denied", ["application/octet-stream", "message/rfc822", "*/*"])
def test_denylist_wins_over_legacy_eml_filename_fallback(denied):
    result = _process(_message(mime="application/octet-stream"), ingest_attachments=True,
                      attachment_mime_denylist=[denied], max_depth=2)
    assert "children" not in result
    assert result["metadata"]["email"]["attachments"][0]["extraction_reason"] == "mime_denied"


def test_legacy_eml_filename_fallback_remains():
    result = _process(_message(mime="application/octet-stream"), ingest_attachments=True, max_depth=2)
    assert result["children"][0]["metadata"]["email"]["subject"] == "Inner synthetic"


def test_depth_limit_keeps_descriptor_without_capturing(monkeypatch):
    data = _message(grandchild=True)
    original_as_bytes = Message.as_bytes

    def intercept_deep(self, *args, **kwargs):
        if self.get("Subject") == "Deep synthetic":
            pytest.fail("Depth-limited attachment was serialized")
        return original_as_bytes(self, *args, **kwargs)

    monkeypatch.setattr(Message, "as_bytes", intercept_deep)
    result = _process(data, extract_attachments=True, max_depth=2)
    child = result["children"][0]
    assert "children" not in child
    assert child["metadata"]["email"]["attachments"][0]["extraction_reason"] == "depth_limit"


def test_nested_size_limit_keeps_metadata(monkeypatch):
    monkeypatch.setitem(lib.DEFAULT_MEDIA_TYPE_CONFIG["archive"], "max_member_uncompressed_size_mb", 0.000001)
    result = _process(extract_attachments=True, max_depth=2)
    assert "children" not in result
    assert result["metadata"]["email"]["attachments"][0]["extraction_reason"] == "size_limit"


def test_nested_count_limit_keeps_metadata(monkeypatch):
    monkeypatch.setitem(lib.DEFAULT_MEDIA_TYPE_CONFIG["archive"], "max_internal_files", 0)
    result = _process(extract_attachments=True, max_depth=2)
    assert "children" not in result
    assert result["metadata"]["email"]["attachments"][0]["extraction_reason"] == "count_limit"


@pytest.mark.parametrize("container", ["zip", "mbox"])
def test_container_expansion_propagates_selection(container):
    data = _message()
    if container == "zip":
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("synthetic.eml", data)
        results = lib.process_eml_archive_bytes(file_bytes=stream.getvalue(), archive_name="synthetic.zip",
            extract_attachments=True, attachment_mime_denylist=["message/rfc822"], max_depth=2, perform_chunking=False)
    else:
        results = lib.process_mbox_bytes(file_bytes=b"From synthetic@example.invalid Sat Jan 01 00:00:00 2022\n" + data + b"\n",
            mbox_name="synthetic.mbox", extract_attachments=True, attachment_mime_denylist=["message/rfc822"],
            max_depth=2, perform_chunking=False)
    assert results[0]["status"] == "Success"
    assert "children" not in results[0]
    assert results[0]["metadata"]["email"]["attachments"][0]["extraction_reason"] == "mime_denied"


def test_missing_pst_dependency_has_deterministic_degraded_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "pypff", None)
    result = lib.process_pst_bytes(file_bytes=b"synthetic", pst_name="synthetic.pst", extract_attachments=True)
    assert result[0]["status"] == "Error"
    assert result[0]["error"] == "PST/OST support not enabled. Install and configure 'pypff' (libpff) or integrate 'readpst' for parsing."


@pytest.mark.parametrize("form_class", [AddMediaForm, ProcessEmailsForm])
def test_form_mime_fields_normalize_and_preserve_empty_allowlist(form_class):
    form = form_class(media_type="email", extract_attachments=False,
        attachment_mime_allowlist=[], attachment_mime_denylist=[" MESSAGE/*, application/PDF "])
    assert form.extract_attachments is False
    assert form.attachment_mime_allowlist == []
    assert form.attachment_mime_denylist == ["message/*", "application/pdf"]


@pytest.mark.parametrize("invalid", ["not-mime", "message/rfc822; charset=utf-8", "*/rfc822", "message/", "message/rfc822\nsecret"])
def test_invalid_mime_selection_is_rejected(invalid):
    with pytest.raises(ValidationError):
        ProcessEmailsForm(attachment_mime_allowlist=[invalid])


@pytest.mark.parametrize("form_options,children_expected", [
    ({"extract_attachments": "true"}, True),
    ({"extract_attachments": "false", "ingest_attachments": "true"}, False),
    ({"extract_attachments": "true", "attachment_mime_allowlist": "message/*", "attachment_mime_denylist": "message/rfc822"}, False),
])
def test_processing_endpoint_honors_multipart_policy(client_user_only, form_options, children_expected):
    response = client_user_only.post("/api/v1/media/process-emails",
        files={"files": ("outer.eml", io.BytesIO(_message()), "message/rfc822")},
        data={"perform_analysis": "false", "perform_chunking": "false", **form_options})
    assert response.status_code == 200, response.text
    result = response.json()["results"][0]
    assert bool(result.get("children")) is children_expected
    assert result["metadata"]["email"]["attachments"][0]["name"] == "nested.eml"


def test_processing_endpoint_rejects_invalid_mime_rule(client_user_only):
    response = client_user_only.post("/api/v1/media/process-emails",
        files={"files": ("outer.eml", io.BytesIO(_message()), "message/rfc822")},
        data={"attachment_mime_allowlist": "broken-rule", "perform_analysis": "false"})
    assert response.status_code == 422


def test_pst_attachment_descriptors_remain_metadata_only(monkeypatch):
    class Attachment:
        name = "nested.eml"
        size = 300
        mime_type = "message/rfc822"

        def read_buffer(self, *_args):
            pytest.fail("PST attachment content was loaded")

    class PSTMessage:
        subject = "Synthetic PST"
        plain_text_body = "Parent PST body"
        number_of_attachments = 1

        def get_attachment(self, _index):
            return Attachment()

    class Folder:
        number_of_sub_messages = 1
        number_of_sub_folders = 0

        def get_sub_message(self, _index):
            return PSTMessage()

    class PST:
        def open(self, _path):
            pass

        def close(self):
            pass

        def get_root_folder(self):
            return Folder()

    monkeypatch.setitem(sys.modules, "pypff", SimpleNamespace(file=PST))
    results = lib.process_pst_bytes(file_bytes=b"synthetic PST bytes", pst_name="synthetic.pst",
        extract_attachments=True, perform_chunking=False, max_depth=2)
    assert results[0]["status"] == "Success"
    assert results[0]["metadata"]["email"]["attachments"][0] == {
        "name": "nested.eml", "content_type": "message/rfc822", "size": 300,
        "extraction_status": "skipped", "extraction_reason": "pst_metadata_only"}
    assert "children" not in results[0]


def test_parse_metrics_count_selected_children_once_without_container_success(monkeypatch):
    observations = []
    monkeypatch.setattr(lib, "record_email_parse", lambda **kwargs: observations.append(kwargs), raising=False)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("outer.eml", _message())
    results = lib.process_eml_archive_bytes(file_bytes=stream.getvalue(), archive_name="synthetic.zip",
        extract_attachments=True, max_depth=2, perform_chunking=False)
    assert results[0]["status"] == "Success"
    assert [(item["source_format"], item["outcome"]) for item in observations] == [("eml", "parsed"), ("eml", "parsed")]
    assert all(item["duration_seconds"] >= 0 for item in observations)


def test_parse_failure_records_one_error_observation(monkeypatch):
    observations = []
    monkeypatch.setattr(lib, "record_email_parse", lambda **kwargs: observations.append(kwargs), raising=False)

    def fail_parse(*_args, **_kwargs):
        raise ValueError("synthetic parser error")

    monkeypatch.setattr(lib, "parse_eml_bytes", fail_parse)
    assert _process()["status"] == "Error"
    assert [(item["source_format"], item["outcome"]) for item in observations] == [("eml", "error")]


@pytest.mark.parametrize("source_format", ["zip", "mbox", "pst", "ost"])
def test_container_guard_failure_records_its_format_once(monkeypatch, source_format):
    observations = []
    monkeypatch.setattr(lib, "record_email_parse", lambda **kwargs: observations.append(kwargs), raising=False)
    monkeypatch.setitem(lib.DEFAULT_MEDIA_TYPE_CONFIG["archive"], "max_internal_uncompressed_size_mb", 0)
    if source_format == "zip":
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("outer.eml", _message())
        result = lib.process_eml_archive_bytes(file_bytes=stream.getvalue(), archive_name="synthetic.zip")
    elif source_format == "mbox":
        result = lib.process_mbox_bytes(file_bytes=b"synthetic", mbox_name="synthetic.mbox")
    else:
        result = lib.process_pst_bytes(file_bytes=b"synthetic", pst_name="synthetic." + source_format)
    assert result[0]["status"] == "Error"
    assert [(item["source_format"], item["outcome"]) for item in observations] == [(source_format, "error")]
