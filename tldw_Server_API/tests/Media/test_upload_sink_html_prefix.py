"""Recognize real HTML after leading whitespace without admitting generic text."""

import io
from types import SimpleNamespace

import pytest
from starlette.datastructures import UploadFile

from tldw_Server_API.app.core.Ingestion_Media_Processing.input_sourcing import save_uploaded_files
from tldw_Server_API.app.core.Ingestion_Media_Processing.persistence import _validate_downloaded_url_file
from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import FileValidator

pytestmark = pytest.mark.unit
HTML = b"<!DOCTYPE html><html><head><title>Example</title></head><body><p>Article</p></body></html>"


@pytest.mark.parametrize("prefix", [b"", b"\n" * 6, b"\xef\xbb\xbf\n\t "])
def test_html_detection_preserves_original_bytes(tmp_path, prefix):
    path = tmp_path / "article.html"
    content = prefix + HTML
    path.write_bytes(content)
    result = FileValidator().validate_file(path, media_type_key="html")
    assert result, result.issues
    assert result.detected_mime_type == "text/html"
    assert path.read_bytes() == content


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["upload", "download"])
async def test_html_prefix_uses_shared_ingestion_validation_and_sanitizer(tmp_path, entry):
    content = b"\n" * 6 + HTML.replace(b"<p>", b"<script>untrusted()</script><p>")
    validator = FileValidator()
    if entry == "upload":
        saved, errors = await save_uploaded_files(
            [UploadFile(filename="article.html", file=io.BytesIO(content))],
            tmp_path,
            validator,
            expected_media_type_key="html",
            allowed_extensions=[".html"],
        )
        assert not errors, errors
        assert len(saved) == 1
        path = saved[0]["path"]
    else:
        path = tmp_path / "article.html"
        path.write_bytes(content)
        _validate_downloaded_url_file(
            downloaded_path=path,
            processing_filename=path.name,
            media_type="document",
            form_data=SimpleNamespace(),
            file_validator=validator,
            allowed_extensions={".html"},
        )
    sanitized = path.read_bytes()
    assert b"<p>Article</p>" in sanitized
    assert b"untrusted" not in sanitized
    assert b"<script>" not in sanitized


@pytest.mark.parametrize(
    "content",
    [
        b"\nPlain text without markup",
        b"\nThe string <html> occurs later in plain text",
        b"\n<!doc type html><html><body>Malformed prefix</body></html>",
        b"\xef\xbb\xbf\xef\xbb\xbf" + HTML,
        b"\n" * 4096 + HTML,
        b"\x89PNG\r\n\x1a\n" + HTML,
        b"MZ" + b"\x00" * 100 + HTML,
    ],
    ids=["plain", "embedded-tag", "malformed", "repeated-bom", "past-prefix-bound", "png", "executable"],
)
def test_html_prefix_does_not_admit_non_html_or_unbounded_prefix(tmp_path, content):
    path = tmp_path / "fake.html"
    path.write_bytes(content)
    result = FileValidator().validate_file(path, media_type_key="html")
    assert not result
    assert any("MIME" in issue for issue in result.issues), result.issues
    assert path.read_bytes() == content


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["upload", "download"])
async def test_html_ingestion_wrappers_still_reject_plaintext(tmp_path, entry):
    content = b"\nThis is plain text, despite the claimed HTML filename."
    validator = FileValidator()
    if entry == "upload":
        saved, errors = await save_uploaded_files(
            [UploadFile(filename="fake.html", file=io.BytesIO(content))],
            tmp_path,
            validator,
            expected_media_type_key="html",
            allowed_extensions=[".html"],
        )
        assert not saved
        assert len(errors) == 1
        assert "MIME" in errors[0]["error"]
    else:
        path = tmp_path / "fake.html"
        path.write_bytes(content)
        with pytest.raises(ValueError, match="Downloaded file failed validation.*MIME"):
            _validate_downloaded_url_file(
                downloaded_path=path,
                processing_filename=path.name,
                media_type="document",
                form_data=SimpleNamespace(),
                file_validator=validator,
                allowed_extensions={".html"},
            )


@pytest.mark.parametrize("guard", ["size", "extension", "yara", "non-html-target"])
def test_html_prefix_preserves_other_validation_guards(tmp_path, monkeypatch, guard):
    path = tmp_path / ("article.html.exe" if guard == "extension" else "article.html")
    path.write_bytes(b"\n" * 6 + HTML)
    validator = FileValidator()
    kwargs = {"media_type_key": "html"}
    if guard == "size":
        kwargs["max_size_mb_override"] = 0.00001
    elif guard == "yara":
        monkeypatch.setattr(validator, "_scan_file_with_yara", lambda _: (False, ["synthetic rule match"]))
    elif guard == "non-html-target":
        kwargs.update(media_type_key="xml", allowed_extensions_override={".html"})
    result = validator.validate_file(path, **kwargs)
    assert not result
    expected = {"size": "exceeds limit", "extension": "security reasons", "yara": "Yara", "non-html-target": "MIME"}
    assert any(expected[guard] in issue for issue in result.issues), result.issues


@pytest.mark.parametrize("failure", ["read", "magic"])
def test_failed_html_prefix_probe_retains_original_mime_rejection(tmp_path, monkeypatch, failure):
    from pathlib import Path

    from tldw_Server_API.app.core.Ingestion_Media_Processing import Upload_Sink as sink

    path = tmp_path / "article.html"
    path.write_bytes(b"\n" * 6 + HTML)
    validator = FileValidator()
    with monkeypatch.context() as patch:
        if failure == "read":
            original = Path.open

            def fail_read(self, *args, **kwargs):
                if self == path:
                    raise OSError("unreadable normalized prefix")
                return original(self, *args, **kwargs)

            patch.setattr(Path, "open", fail_read)
        else:

            def fail_magic(*args, **kwargs):
                raise sink.puremagic.PureError("unrecognized normalized prefix")

            patch.setattr(sink.puremagic, "from_string", fail_magic)
        result = validator.validate_file(path, media_type_key="html")
    assert not result
    assert result.detected_mime_type == "text/plain"
    assert any("MIME" in issue for issue in result.issues), result.issues
    assert path.read_bytes() == b"\n" * 6 + HTML
