import tarfile
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import (
    FileValidator,
)


@pytest.mark.unit
def test_validate_tar_archive_simple(tmp_path: Path):
    # Create a simple tar archive with one small file
    inner_dir = tmp_path / "inner"
    inner_dir.mkdir()
    (inner_dir / "a.txt").write_text("hello", encoding="utf-8")

    tar_path = tmp_path / "sample.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(inner_dir / "a.txt", arcname="a.txt")

    # Avoid MIME enforcement to make test robust across environments
    validator = FileValidator(custom_media_configs={
        "archive": {"allowed_mimetypes": None}
    })

    res = validator.validate_archive_contents(tar_path)
    assert res.is_valid, f"Expected valid tar archive, got issues: {res.issues}"


@pytest.mark.unit
def test_validate_tar_archive_limits(tmp_path: Path):
    # Create a tar archive with two files and set limit=1
    (tmp_path / "f1.txt").write_text("x", encoding="utf-8")
    (tmp_path / "f2.txt").write_text("y", encoding="utf-8")
    tar_path = tmp_path / "many.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(tmp_path / "f1.txt", arcname="f1.txt")
        tar.add(tmp_path / "f2.txt", arcname="f2.txt")

    validator = FileValidator(custom_media_configs={
        "archive": {
            "allowed_mimetypes": None,
            "max_internal_files": 1,
        }
    })

    res = validator.validate_archive_contents(tar_path)
    assert not res.is_valid
    joined = " ".join(res.issues).lower()
    assert "too many files" in joined or "exceeded max internal file limit" in joined


@pytest.mark.unit
def test_nested_archive_scanning_detects_inner_payload(tmp_path: Path):
    inner_zip = tmp_path / "inner.zip"
    with zipfile.ZipFile(inner_zip, "w") as zf:
        zf.writestr("payload.exe", b"MZ")

    outer_zip = tmp_path / "outer.zip"
    with zipfile.ZipFile(outer_zip, "w") as zf:
        zf.write(inner_zip, arcname="inner.zip")

    validator = FileValidator(custom_media_configs={
        "archive": {
            "allowed_mimetypes": None,
            "max_internal_files": 10,
            "max_depth": 2,
        }
    })

    res = validator.validate_archive_contents(outer_zip)
    assert not res.is_valid
    message = " ".join(res.issues).lower()
    assert "nested archive" in message or ".exe" in message


@pytest.mark.unit
def test_html_sanitization_removes_script():
    validator = FileValidator()
    html = "<html><head><script>alert('x')</script></head><body><p>OK</p></body></html>"
    cleaned = validator.sanitize_html_content(html, config={"strip": True})
    assert "script" not in cleaned.lower()
    assert "ok" in cleaned.lower()


@pytest.mark.unit
def test_xml_sanitization_strips_comments_and_pi():
    try:
        import defusedxml  # noqa: F401
    except Exception:
        pytest.skip("defusedxml not installed")

    validator = FileValidator()
    xml = """<?xml version='1.0'?>
    <?processing instruction?>
    <!-- a comment -->
    <root><child>ok</child></root>
    """
    cleaned = validator.sanitize_xml_content(xml, config={"strip_comments": True, "strip_processing_instructions": True})
    low = cleaned.lower()
    assert "processing" not in low and "comment" not in low
    assert "child" in low and "ok" in low


@pytest.mark.unit
def test_email_html_body_is_sanitized_before_text_conversion():
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Email import (
        Email_Processing_Lib as email_lib,
    )

    eml = (
        b"From: A <a@example.com>\r\n"
        b"To: B <b@example.com>\r\n"
        b"Subject: HTML Body\r\n"
        b"MIME-Version: 1.0\r\n"
        b"Content-Type: text/html; charset=utf-8\r\n\r\n"
        b"<html><body><script>alert('x')</script><p>Hello</p></body></html>"
    )

    result = email_lib.process_email_task(
        file_bytes=eml,
        filename="test.eml",
        perform_chunking=False,
        perform_analysis=False,
    )

    assert result.get("status") == "Success"
    content = str(result.get("content") or "").lower()
    assert "hello" in content
    assert "alert" not in content
    assert "script" not in content


@pytest.mark.unit
def test_email_task_sanitizes_parse_failure(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Email import (
        Email_Processing_Lib as email_lib,
    )

    def _fail_parse(*_args, **_kwargs):
        raise RuntimeError("email parser exploded at /private/email/source.eml")

    monkeypatch.setattr(email_lib, "parse_eml_bytes", _fail_parse)

    result = email_lib.process_email_task(
        file_bytes=b"not an email",
        filename="source.eml",
        perform_chunking=False,
        perform_analysis=False,
    )

    assert result["status"] == "Error"
    assert result["error"] == "Email processing failed"
    assert "email parser exploded" not in str(result)
    assert "/private/email/source.eml" not in str(result)


@pytest.mark.unit
def test_email_task_sanitizes_chunking_failure(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Email import (
        Email_Processing_Lib as email_lib,
    )

    monkeypatch.setattr(
        email_lib,
        "parse_eml_bytes",
        lambda *_args, **_kwargs: ("hello world", {"title": "Email"}, []),
    )

    def _fail_chunking(*_args, **_kwargs):
        raise RuntimeError("email chunker exploded at /private/email/chunks")

    monkeypatch.setattr(email_lib, "improved_chunking_process", _fail_chunking)

    result = email_lib.process_email_task(
        file_bytes=b"email",
        filename="source.eml",
        perform_chunking=True,
        perform_analysis=False,
    )

    assert result["status"] == "Success"
    assert result["warnings"] == ["Chunking failed"]
    assert result["chunks"][0]["metadata"]["error"] == "Email chunking failed"
    assert "email chunker exploded" not in str(result)
    assert "/private/email/chunks" not in str(result)


@pytest.mark.unit
def test_email_archive_rejects_oversized_member(tmp_path: Path):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Email import (
        Email_Processing_Lib as email_lib,
    )

    archive_cfg = email_lib.DEFAULT_MEDIA_TYPE_CONFIG.get("archive", {})
    original_member_cap = archive_cfg.get("max_member_uncompressed_size_mb", 100)
    try:
        archive_cfg["max_member_uncompressed_size_mb"] = 1
        large_body = b"x" * (2 * 1024 * 1024)
        eml_payload = (
            b"From: A <a@example.com>\r\n"
            b"To: B <b@example.com>\r\n"
            b"Subject: Big\r\n"
            b"MIME-Version: 1.0\r\n"
            b"Content-Type: text/plain; charset=utf-8\r\n\r\n"
            + large_body
        )
        bio = BytesIO()
        with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("big.eml", eml_payload)

        results = email_lib.process_eml_archive_bytes(
            file_bytes=bio.getvalue(),
            archive_name="emails.zip",
            perform_chunking=False,
            perform_analysis=False,
        )
        assert results and results[0].get("status") == "Error"
        assert "member exceeds uncompressed size limit" in str(
            results[0].get("error", "")
        ).lower()
    finally:
        archive_cfg["max_member_uncompressed_size_mb"] = original_member_cap


@pytest.mark.unit
def test_email_archive_rejects_unsafe_member_path():
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Email import (
        Email_Processing_Lib as email_lib,
    )

    bio = BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("../escape.eml", b"From: A <a@example.com>\r\n\r\nbody")

    results = email_lib.process_eml_archive_bytes(
        file_bytes=bio.getvalue(),
        archive_name="emails.zip",
        perform_chunking=False,
        perform_analysis=False,
    )
    assert results and results[0].get("status") == "Error"
    assert "unsafe member path" in str(results[0].get("error", "")).lower()
