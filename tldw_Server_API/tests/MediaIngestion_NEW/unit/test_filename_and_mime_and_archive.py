"""
Unit tests for:
- Long filename truncation preserving extension and suffixing
- python-magic fallback path for MIME detection via monkeypatch
- Archive scanning unavailable: returns warning without hard failure
"""

import io
import os
import zipfile
import tarfile
from pathlib import Path

import pytest


def test_sanitize_filename_truncation_and_suffix():


    from tldw_Server_API.app.core.Utils.Utils import sanitize_filename

    base = "a" * 300
    ext = ".pdf"
    cap = 200

    # Truncate base to fit cap when extension is present
    truncated = sanitize_filename(base, max_total_length=cap, extension=ext)
    assert len(truncated + ext) <= cap
    assert (truncated + ext).endswith(ext)

    # Simulate uniqueness suffixing similar to endpoint logic
    def build_filename(b: str, e: str, cap_len: int, suffix: str | None = None) -> str:
        suffix_txt = f"_{suffix}" if suffix else ""
        reserved = len(suffix_txt) + len(e)
        available = cap_len - reserved
        trunc_b = b if len(b) <= available else b[: max(1, available)]
        return f"{trunc_b}{suffix_txt}{e}"

    # No suffix
    fname0 = build_filename(truncated, ext, cap, None)
    assert len(fname0) <= cap
    assert fname0.endswith(ext)

    # With numeric suffix
    fname1 = build_filename(truncated, ext, cap, "1")
    assert len(fname1) <= cap
    assert fname1.endswith(ext)

    # With longer suffix (ensure further truncation happens)
    fname_long = build_filename(truncated, ext, cap, "1234567890")
    assert len(fname_long) <= cap
    assert fname_long.endswith(ext)


@pytest.mark.unit
def test_validate_file_python_magic_fallback(monkeypatch, tmp_path: Path):
    # Import the sink module for monkeypatching internals
    from tldw_Server_API.app.core.Ingestion_Media_Processing import Upload_Sink as sink

    # Force puremagic absence and provide python-magic stub
    monkeypatch.setattr(sink, "puremagic", None, raising=False)

    class _DummyMagic:
        def __init__(self, mime: bool = True, magic_file: str | None = None):
            self._mime = mime
            self._magic_file = magic_file

        def from_file(self, path: str) -> str:
            # Return a controlled MIME type regardless of extension
            return "text/plain"

    class _DummyPythonMagicModule:
        @staticmethod
        def Magic(mime: bool = True, magic_file: str | None = None):
            return _DummyMagic(mime=mime, magic_file=magic_file)

    monkeypatch.setattr(sink, "_python_magic", _DummyPythonMagicModule, raising=False)

    # Create a .bin file; extension alone would not match text/plain
    file_path = tmp_path / "payload.bin"
    file_path.write_text("hello world", encoding="utf-8")

    # Configure validator to allow .bin with text/plain MIME only
    validator = sink.FileValidator(custom_media_configs={
        "document": {
            "allowed_extensions": {".bin"},
            "allowed_mimetypes": {"text/plain"},
            "max_size_mb": 5,
        }
    })

    res = validator.validate_file(file_path, media_type_key="document")
    assert res.is_valid is True
    # Ensure detected MIME reflects python-magic stub
    assert res.detected_mime_type == "text/plain"


@pytest.mark.unit
def test_file_validator_skips_python_magic_import_on_windows(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing import Upload_Sink as sink

    monkeypatch.setattr(sink, "puremagic", None, raising=False)
    monkeypatch.setattr(sink, "_python_magic", None, raising=False)
    monkeypatch.setattr(sink, "_python_magic_loaded", False, raising=False)
    monkeypatch.setattr(sink.os, "name", "nt", raising=False)

    def _unexpected_import(_name: str):
        raise AssertionError("python-magic should not be imported on Windows")

    monkeypatch.setattr(sink.importlib, "import_module", _unexpected_import, raising=True)

    validator = sink.FileValidator()

    assert validator.python_magic_available is False
    assert validator._python_magic_mime is None


@pytest.mark.unit
def test_validate_archive_scanning_unavailable_returns_warning(tmp_path: Path):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import FileValidator

    # Create a simple ZIP archive
    zip_path = tmp_path / "sample.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("doc.txt", "content")

    validator = FileValidator()
    # Simulate unavailable archive scanning facility
    validator.archive_scanning_available = False

    res = validator.validate_archive_contents(zip_path)
    assert res.is_valid is True  # no hard failure
    issues_text = "\n".join(res.issues or [])
    assert "Archive content scanning not available" in issues_text


@pytest.mark.unit
def test_tar_path_traversal_rejected(tmp_path: Path):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import FileValidator

    # Create payload files
    (tmp_path / "safe.txt").write_text("ok", encoding="utf-8")

    tar_path = tmp_path / "traversal.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(tmp_path / "safe.txt", arcname="safe.txt")
        # Add a traversal entry
        info = tarfile.TarInfo(name="../evil.txt")
        data = io.BytesIO(b"bad")
        info.size = len(data.getbuffer())
        tar.addfile(info, data)

    validator = FileValidator(custom_media_configs={"archive": {"allowed_mimetypes": None}})
    res = validator.validate_archive_contents(tar_path)
    assert not res.is_valid
    msg = " ".join(res.issues).lower()
    assert "path traversal" in msg or "malicious path" in msg


@pytest.mark.unit
def test_tar_absolute_path_rejected(tmp_path: Path):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import FileValidator

    tar_path = tmp_path / "abs.tar"
    with tarfile.open(tar_path, "w") as tar:
        info = tarfile.TarInfo(name="/abs/evil.txt")
        data = io.BytesIO(b"evil")
        info.size = len(data.getbuffer())
        tar.addfile(info, data)

    validator = FileValidator(custom_media_configs={"archive": {"allowed_mimetypes": None}})
    res = validator.validate_archive_contents(tar_path)
    assert not res.is_valid
    msg = " ".join(res.issues).lower()
    assert "malicious path" in msg or "path traversal" in msg


@pytest.mark.unit
def test_tar_symlink_and_hardlink_rejected(tmp_path: Path):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import FileValidator

    tar_path = tmp_path / "links.tar"
    with tarfile.open(tar_path, "w") as tar:
        # Add a symlink entry
        sym = tarfile.TarInfo(name="link_to_nowhere")
        sym.type = tarfile.SYMTYPE
        sym.linkname = "nowhere"
        tar.addfile(sym)

        # Add a hardlink entry
        hlnk = tarfile.TarInfo(name="hardlink")
        hlnk.type = tarfile.LNKTYPE
        hlnk.linkname = "target"
        tar.addfile(hlnk)

    validator = FileValidator(custom_media_configs={"archive": {"allowed_mimetypes": None}})
    res = validator.validate_archive_contents(tar_path)
    assert not res.is_valid
    msg = " ".join(res.issues).lower()
    assert "unsupported link" in msg or "unsupported member type" in msg


@pytest.mark.unit
def test_tar_nested_depth_exceeded(tmp_path: Path):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import FileValidator

    # Create inner-most tar (level 3)
    level3 = tmp_path / "l3.tar"
    with tarfile.open(level3, "w") as tar:
        info = tarfile.TarInfo(name="leaf.txt")
        data = io.BytesIO(b"x")
        info.size = len(data.getbuffer())
        tar.addfile(info, data)

    # Level 2 tar containing level 3
    level2 = tmp_path / "l2.tar"
    with tarfile.open(level2, "w") as tar:
        tar.add(level3, arcname="l3.tar")

    # Level 1 tar containing level 2
    level1 = tmp_path / "l1.tar"
    with tarfile.open(level1, "w") as tar:
        tar.add(level2, arcname="l2.tar")

    # Outer tar containing level 1
    outer = tmp_path / "outer.tar"
    with tarfile.open(outer, "w") as tar:
        tar.add(level1, arcname="l1.tar")

    validator = FileValidator(custom_media_configs={
        "archive": {"allowed_mimetypes": None, "max_depth": 2}
    })
    res = validator.validate_archive_contents(outer)
    assert not res.is_valid
    msg = " ".join(res.issues).lower()
    assert "exceeds maximum" in msg or "max depth" in msg


@pytest.mark.unit
def test_tar_weird_path_normalized(tmp_path: Path):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import FileValidator

    tar_path = tmp_path / "weird.tar"
    with tarfile.open(tar_path, "w") as tar:
        info = tarfile.TarInfo(name="dir//../evil.txt")
        data = io.BytesIO(b"x")
        info.size = len(data.getbuffer())
        tar.addfile(info, data)

    validator = FileValidator(custom_media_configs={"archive": {"allowed_mimetypes": None}})
    res = validator.validate_archive_contents(tar_path)
    assert not res.is_valid
    msg = " ".join(res.issues).lower()
    assert "path traversal" in msg or "malicious path" in msg


@pytest.mark.unit
def test_zip_path_traversal_rejected(tmp_path: Path):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import FileValidator

    zip_path = tmp_path / "trav.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("../evil.txt", b"bad")

    validator = FileValidator(custom_media_configs={"archive": {"allowed_mimetypes": None}})
    res = validator.validate_archive_contents(zip_path)
    assert not res.is_valid
    msg = " ".join(res.issues).lower()
    assert "path traversal" in msg or "malicious path" in msg


@pytest.mark.unit
def test_zip_absolute_path_rejected(tmp_path: Path):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import FileValidator

    zip_path = tmp_path / "abs.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("/abs/evil.txt", b"bad")

    validator = FileValidator(custom_media_configs={"archive": {"allowed_mimetypes": None}})
    res = validator.validate_archive_contents(zip_path)
    assert not res.is_valid
    msg = " ".join(res.issues).lower()
    assert "malicious path" in msg or "path traversal" in msg


@pytest.mark.unit
def test_zip_symlink_rejected(tmp_path: Path):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import FileValidator

    zip_path = tmp_path / "links.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        info = zipfile.ZipInfo("link")
        # Mark as Unix symlink: file type bits 0120000 in upper 16 bits
        info.create_system = 3  # Unix
        info.external_attr = (0o120777 << 16)
        # For symlink entries, data can be empty
        zf.writestr(info, b"")

    validator = FileValidator(custom_media_configs={"archive": {"allowed_mimetypes": None}})
    res = validator.validate_archive_contents(zip_path)
    assert not res.is_valid
    msg = " ".join(res.issues).lower()
    assert "symbolic link" in msg or "unsupported" in msg


@pytest.mark.unit
def test_zip_valid_simple_passes(tmp_path: Path):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import FileValidator

    zip_path = tmp_path / "ok.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("safe.txt", b"ok")

    validator = FileValidator(custom_media_configs={"archive": {"allowed_mimetypes": None}})
    res = validator.validate_archive_contents(zip_path)
    assert res.is_valid, f"Expected valid archive; got issues: {res.issues}"
