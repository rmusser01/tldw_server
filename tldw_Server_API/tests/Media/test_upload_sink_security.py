import io
import stat
import tarfile
import zipfile
from types import SimpleNamespace
from typing import Any

import pytest
from starlette.datastructures import UploadFile

from tldw_Server_API.app.core.Ingestion_Media_Processing.input_sourcing import save_uploaded_files
from tldw_Server_API.app.core.Ingestion_Media_Processing.persistence import _validate_downloaded_url_file
from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import FileValidator

pytestmark = pytest.mark.unit


def test_validate_file_mime_mismatch_hard_fail(tmp_path, monkeypatch):


    """Reject when magic-detected MIME disagrees with allowed and do not accept fallback."""
    # Arrange: create a non-PDF file with .pdf extension
    p = tmp_path / "fake.pdf"
    p.write_bytes(b"not-a-pdf")

    # Monkeypatch puremagic in the module to simulate a strong MIME detection
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink as US

    class DummyMagic:
        @staticmethod
        def from_file(path, mime=True):
            return "application/x-msdownload"

    monkeypatch.setattr(US, "puremagic", DummyMagic)

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import FileValidator

    v = FileValidator()
    res = v.validate_file(p, original_filename="fake.pdf", media_type_key="pdf")
    assert not res, f"Expected failure; issues: {res.issues}"
    assert any("Detected MIME" in i for i in res.issues), res.issues


def test_validate_audio_file_accepts_mp4a_latm(tmp_path, monkeypatch):


    """Accept AAC-in-MP4 streams commonly detected as audio/mp4a-latm."""
    p = tmp_path / "sample.m4a"
    p.write_bytes(b"fake-audio")

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink as US

    class DummyMagic:
        @staticmethod
        def from_file(path, mime=True):
            return "audio/mp4a-latm"

    monkeypatch.setattr(US, "puremagic", DummyMagic)

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import FileValidator

    v = FileValidator()
    res = v.validate_file(p, original_filename="sample.m4a", media_type_key="audio")
    assert res, res.issues
    assert (res.detected_mime_type or "").lower() == "audio/mp4a-latm"


def test_validate_archive_rejects_encrypted_zip(tmp_path, monkeypatch):


    """Explicitly reject encrypted ZIP entries using flag_bits check."""
    zpath = tmp_path / "enc.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("ok.txt", "hello")

    # Provide a fake encrypted entry via infolist monkeypatch
    class DummyInfo:
        filename = "secret.txt"
        file_size = 10
        external_attr = 0
        flag_bits = 0x1  # Encrypted

        def is_dir(self):

            return False

    def fake_infolist(self):

        return [DummyInfo()]

    monkeypatch.setattr(zipfile.ZipFile, "infolist", fake_infolist, raising=False)

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import FileValidator

    v = FileValidator()
    res = v.validate_archive_contents(zpath)
    assert not res, "Encrypted archive should be rejected"
    assert any("encrypted member" in i.lower() for i in res.issues), res.issues


def test_validate_archive_flags_zip_symlink(tmp_path, monkeypatch):


    """Symlink entries in ZIP are flagged and skipped."""
    zpath = tmp_path / "sym.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("ok.txt", "hello")

    # Provide a symlink-like entry via external_attr high bits
    mode = stat.S_IFLNK | 0o777

    class DummyInfo:
        filename = "link"
        file_size = 0
        external_attr = (mode & 0xFFFF) << 16
        flag_bits = 0

        def is_dir(self):

            return False

    def fake_infolist(self):

        return [DummyInfo()]

    monkeypatch.setattr(zipfile.ZipFile, "infolist", fake_infolist, raising=False)

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import FileValidator

    v = FileValidator()
    res = v.validate_archive_contents(zpath)
    assert not res, "ZIP symlink entry should be flagged and cause validation failure"
    assert any("symbolic link" in i.lower() for i in res.issues), res.issues


def test_validate_archive_flags_tar_bad_types(tmp_path, monkeypatch):


    """Non-file TAR members are flagged and not extracted."""
    tpath = tmp_path / "bad.tar"
    with tarfile.open(tpath, "w") as tf:
        data = b"hello"
        ti = tarfile.TarInfo(name="ok.txt")
        ti.size = len(data)
        tf.addfile(ti, io.BytesIO(data))

    # Fake getmembers to return a symlink-like member
    class DummyTarInfo:
        name = "weird"
        size = 0
        type = b"?"  # unknown type marker for message

        def isdir(self):

            return False

        def issym(self):

            return True

        def islnk(self):

            return False

        def isfile(self):

            return False

    def fake_getmembers(self):

        return [DummyTarInfo()]

    monkeypatch.setattr(tarfile.TarFile, "getmembers", fake_getmembers, raising=False)

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import FileValidator

    v = FileValidator()
    res = v.validate_archive_contents(tpath)
    assert not res, "TAR with non-file member should be rejected"
    assert any("link entry" in i.lower() or "unsupported member type" in i.lower() for i in res.issues), res.issues


def test_svg_treated_as_xml_and_allowed(tmp_path):


    """SVG is handled under XML rules and accepted (sanitization path available)."""
    svg_path = tmp_path / "img.svg"
    svg_path.write_text("<svg xmlns='http://www.w3.org/2000/svg'><title>T</title></svg>")

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import FileValidator

    v = FileValidator()
    res = v.validate_file(svg_path, original_filename="img.svg", media_type_key="xml")
    assert res, res.issues
    # Expect MIME either detected as image/svg+xml or via fallback
    assert (res.detected_mime_type or "").lower() in ("image/svg+xml", "text/xml", "application/xml")


def test_missing_mime_libraries_warn_about_extension_fallback(monkeypatch):
    from loguru import logger

    from tldw_Server_API.app.core.Ingestion_Media_Processing import Upload_Sink as sink

    monkeypatch.setattr(sink, "puremagic", None)
    monkeypatch.setattr(sink, "_get_python_magic_module", lambda: None)
    output: list[Any] = []
    token = logger.add(
        output.append, level="WARNING", filter=lambda record: record["name"] == sink.__name__
    )
    try:
        sink.FileValidator()
    finally:
        logger.remove(token)
    messages = " ".join(message.record["message"] for message in output).lower()
    assert "mime" in messages and "extension" in messages
    assert "permit scanner errors" not in messages

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
    kwargs: dict[str, Any] = {"media_type_key": "html"}
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
