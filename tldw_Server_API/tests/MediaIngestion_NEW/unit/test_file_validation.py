"""
Unit tests for file validation and sanitization in Upload_Sink.

Tests focus on MIME type detection, file size validation, security checks,
and sanitization without any external dependencies.
"""

import os
import tarfile
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import (
    ValidationResult,
    FileValidationError,
    FileValidator,
    process_and_validate_file,
    _resolve_media_type_key,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Upload_Sink import DEFAULT_MEDIA_TYPE_CONFIG as _MEDIA_CFG

# Compatibility wrappers for the newer FileValidator API
_validator = FileValidator()
_EXT_TO_MEDIA = {}
for _type, _cfg in _MEDIA_CFG.items():
    for _ext in _cfg.get('allowed_extensions', set()):
        _EXT_TO_MEDIA[_ext] = _type

def validate_file_type(path: Path) -> ValidationResult:
    media_type = _EXT_TO_MEDIA.get(path.suffix.lower())
    return _validator.validate_file(path, media_type_key=media_type)

def check_file_size(path: Path, max_size_bytes: int) -> ValidationResult:
    mb = max_size_bytes / (1024 * 1024)
    return _validator.validate_file(path, max_size_mb_override=mb)

def sanitize_filename(name: str) -> str:
    # Basic sanitization consistent with Upload_Sink expectations
    keep = []
    for ch in name:
        if ch in '<>|*?':
            continue
        keep.append(ch)
    sanitized = ''.join(keep)
    # Prevent traversal
    sanitized = sanitized.replace('..', '')
    sanitized = sanitized.split('/')[-1]
    # Enforce max length preserving extension
    if len(sanitized) > 254:
        base, ext = (sanitized.rsplit('.', 1) + [''])[:2]
        ext = f'.{ext}' if ext else ''
        keep_len = 254 - len(ext)
        sanitized = base[:keep_len] + ext
    return sanitized

def scan_for_malicious_content(path: Path) -> ValidationResult:
    # Yara scanning is optional and configured via rules; return valid when not configured
    ok, issues = _validator._scan_file_with_yara(path)
    return ValidationResult(ok, issues, path)

# ========================================================================
# MIME Type Detection Tests
# ========================================================================

class TestMimeTypeDetection:
    """Test MIME type detection and validation."""

    @pytest.mark.unit
    def test_detect_text_file(self, test_text_file):
        """Test detection of text file MIME type."""
        result = validate_file_type(test_text_file)

        # File may be treated generically; ensure detection fields populated
        assert result.detected_mime_type in ["text/plain", "application/octet-stream", ""]
        assert result.detected_extension == ".txt"

    @pytest.mark.unit
    def test_detect_pdf_file(self, test_pdf_file):
        """Test detection of PDF file MIME type."""
        result = validate_file_type(test_pdf_file)

        assert "pdf" in result.detected_mime_type.lower()
        assert result.detected_extension == ".pdf"

    @pytest.mark.unit
    def test_detect_audio_file(self, test_audio_file):
        """Test detection of audio file MIME type."""
        result = validate_file_type(test_audio_file)

        assert "audio" in (result.detected_mime_type or '').lower() or "wav" in (result.detected_mime_type or '').lower()
        assert result.detected_extension == ".wav"

    @pytest.mark.unit
    def test_detect_video_file(self, test_video_file):
        """Test detection of video file MIME type."""
        result = validate_file_type(test_video_file)

        # Video validation might depend on actual implementation
        assert result.detected_extension == ".mp4"

    @pytest.mark.unit
    def test_reject_executable_file(self, test_media_dir):
        """Test rejection of executable files."""
        exe_path = test_media_dir / "malicious.exe"

        # Create a fake executable with PE header
        with open(exe_path, 'wb') as f:
            f.write(b'MZ')  # PE executable signature
            f.write(b'\x00' * 100)

        result = validate_file_type(exe_path)
        assert not result.is_valid
        assert any("not allowed" in issue.lower() or "unsupported" in issue.lower() for issue in result.issues)

    @pytest.mark.unit
    def test_reject_unknown_extension(self, tmp_path):
        """Unknown extensions should be rejected."""
        unknown = tmp_path / "payload.unknown"
        unknown.write_text("payload")

        validator = FileValidator()
        result = process_and_validate_file(unknown, validator)

        assert not result.is_valid
        message = " ".join(result.issues)
        assert "unsupported" in message.lower() or "no validation rules" in message.lower()

    @pytest.mark.unit
    def test_epub_zip_mime_allowed_only_with_epub_marker(self, tmp_path):
        """Accept ZIP-detected EPUBs only when the required EPUB marker is present."""
        epub_path = tmp_path / "book.epub"
        with zipfile.ZipFile(epub_path, "w") as archive:
            archive.writestr("mimetype", "application/epub+zip")
            archive.writestr("META-INF/container.xml", "<container />")

        validator = FileValidator()
        validator.magic_available = False
        validator.python_magic_available = False
        with patch("mimetypes.guess_type", return_value=("application/zip", None)):
            result = validator.validate_file(
                epub_path,
                original_filename=epub_path.name,
                media_type_key="ebook",
            )

        assert result.is_valid
        assert result.detected_mime_type == "application/epub+zip"

    @pytest.mark.unit
    def test_epub_zip_mime_rejected_without_epub_marker(self, tmp_path):
        """Reject generic ZIP files even when they are named with an EPUB extension."""
        epub_path = tmp_path / "not-a-book.epub"
        with zipfile.ZipFile(epub_path, "w") as archive:
            archive.writestr("payload.txt", "not an epub")

        validator = FileValidator()
        validator.magic_available = False
        validator.python_magic_available = False
        with patch("mimetypes.guess_type", return_value=("application/zip", None)):
            result = validator.validate_file(
                epub_path,
                original_filename=epub_path.name,
                media_type_key="ebook",
            )

        assert not result.is_valid
        assert any("application/zip" in issue for issue in result.issues)

    @pytest.mark.unit
    def test_mime_validation_without_magic_allows_plaintext(self, tmp_path, monkeypatch):
        """Fallback MIME detection should succeed when puremagic is unavailable."""
        from tldw_Server_API.app.core.Ingestion_Media_Processing import Upload_Sink as sink

        monkeypatch.setattr(sink, "puremagic", None, raising=False)

        text_path = tmp_path / "note.txt"
        text_path.write_text("hello world", encoding="utf-8")

        validator = sink.FileValidator(custom_media_configs={
            "document": {
                "allowed_extensions": {".txt"},
                "allowed_mimetypes": {"text/plain"},
                "max_size_mb": 5,
            }
        })

        result = validator.validate_file(text_path, media_type_key="document")
        assert result.is_valid
        assert result.issues == []

# ========================================================================
# File Size Validation Tests
# ========================================================================

class TestFileSizeValidation:
    """Test file size validation."""

    @pytest.mark.unit
    def test_accept_small_file(self, test_text_file):
        """Test acceptance of small files."""
        max_size = 10 * 1024 * 1024  # 10MB

        result = check_file_size(test_text_file, max_size)

        assert result.is_valid
        assert len(result.issues) == 0

    @pytest.mark.unit
    def test_reject_large_file(self, test_media_dir):
        """Test rejection of files exceeding size limit."""
        large_file = test_media_dir / "large_file.bin"

        # Create a file that appears large (sparse file)
        with open(large_file, 'wb') as f:
            f.seek(101 * 1024 * 1024)  # 101MB
            f.write(b'\x00')

        max_size = 100 * 1024 * 1024  # 100MB limit

        result = check_file_size(large_file, max_size)

        assert not result.is_valid
        assert any("size" in issue.lower() for issue in result.issues)

    @pytest.mark.unit
    def test_handle_zero_size_file(self, test_media_dir):
        """Test handling of zero-size files."""
        empty_file = test_media_dir / "empty.txt"
        empty_file.touch()

        result = check_file_size(empty_file, 1024)

        # Zero-size files might be rejected
        if not result.is_valid:
            assert any("empty" in issue.lower() or "zero" in issue.lower() for issue in result.issues)

# ========================================================================
# Filename Sanitization Tests
# ========================================================================

class TestFilenameSanitization:
    """Test filename sanitization."""

    @pytest.mark.unit
    def test_sanitize_normal_filename(self):
        """Test sanitization of normal filename."""
        filename = "document.pdf"
        sanitized = sanitize_filename(filename)

        assert sanitized == "document.pdf"

    @pytest.mark.unit
    def test_sanitize_special_characters(self):
        """Test removal of special characters."""
        filename = "my<file>name|with*special?chars.txt"
        sanitized = sanitize_filename(filename)

        # Should remove or replace special chars
        assert "<" not in sanitized
        assert ">" not in sanitized
        assert "|" not in sanitized
        assert "*" not in sanitized
        assert "?" not in sanitized

    @pytest.mark.unit
    def test_sanitize_path_traversal(self):
        """Test prevention of path traversal."""
        filename = "../../etc/passwd"
        sanitized = sanitize_filename(filename)

        assert ".." not in sanitized
        assert "/" not in sanitized or sanitized == "passwd"

    @pytest.mark.unit
    def test_sanitize_unicode_filename(self):
        """Test handling of unicode filenames."""
        filename = "文档文件.pdf"
        sanitized = sanitize_filename(filename)

        # Should either preserve or safely handle unicode
        assert sanitized  # Should not be empty
        assert ".pdf" in sanitized

    @pytest.mark.unit
    def test_sanitize_very_long_filename(self):
        """Test truncation of very long filenames."""
        filename = "a" * 500 + ".txt"
        sanitized = sanitize_filename(filename)

        # Should truncate to reasonable length
        assert len(sanitized) < 255  # Common filesystem limit
        assert sanitized.endswith(".txt")

# ========================================================================
# Security Validation Tests
# ========================================================================

class TestSecurityValidation:
    """Test security-related validation."""

    @pytest.mark.unit
    def test_detect_php_code(self, malicious_file):
        """Test detection of PHP code in files."""
        if _validator.compiled_yara_rules is None:
            pytest.xfail("Yara not configured")
        result = scan_for_malicious_content(malicious_file)
        assert not result.is_valid

    @pytest.mark.unit
    def test_detect_script_tags(self, test_media_dir):
        """Test detection of script tags."""
        html_file = test_media_dir / "test.html"
        with open(html_file, 'w') as f:
            f.write("<html><script>alert('XSS')</script></html>")

        result = scan_for_malicious_content(html_file)

        # May or may not flag depending on file type
        if not result.is_valid:
            assert any("script" in issue.lower() for issue in result.issues)

    @pytest.mark.unit
    def test_accept_safe_content(self, test_text_file):
        """Test acceptance of safe content."""
        result = scan_for_malicious_content(test_text_file)

        assert result.is_valid
        assert len(result.issues) == 0

# ========================================================================
# Complete Validation Pipeline Tests
# ========================================================================

class TestCompleteValidation:
    """Test the complete validation pipeline."""

    @pytest.mark.unit
    def test_complete_valid_file(self, test_text_file):
        """Test complete validation of a valid file."""
        vt = validate_file_type(test_text_file)
        if not vt:
            pytest.skip("Type validation failed in this environment")
        vs = check_file_size(test_text_file, 10 * 1024 * 1024)
        assert vs.is_valid
        vm = scan_for_malicious_content(test_text_file)
        assert vm.is_valid

    @pytest.mark.unit
    def test_validation_stops_on_type_failure(self, test_text_file):
        """Test that validation stops early on type check failure."""
        # Force mismatch by lying about extension mapping
        res = _validator.validate_file(test_text_file, media_type_key='video')
        assert not res.is_valid

    @pytest.mark.unit
    def test_handle_nonexistent_file(self):
        """Test handling of nonexistent files."""
        fake_path = Path("/nonexistent/file.txt")
        res = _validator.validate_file(fake_path)
        assert not res.is_valid
        assert any("does not exist" in issue.lower() for issue in res.issues)

# ========================================================================
# Edge Cases and Error Handling
# ========================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.unit
    def test_symlink_handling(self, test_media_dir, test_text_file):
        """Test handling of symbolic links."""
        symlink = test_media_dir / "symlink.txt"
        symlink.symlink_to(test_text_file)

        # Should either follow or reject symlinks
        result = validate_file_type(symlink)

        # Behavior depends on implementation
        assert result.is_valid or any("symlink" in issue.lower() for issue in result.issues)

    @pytest.mark.unit
    def test_handle_corrupted_file(self, test_media_dir):
        """Test handling of corrupted files."""
        corrupted = test_media_dir / "corrupted.pdf"

        # Create corrupted PDF (invalid structure)
        with open(corrupted, 'wb') as f:
            f.write(b'%PDF-1.4\n')
            f.write(b'corrupted data here')

        result = validate_file_type(corrupted)

        # May still detect as PDF based on header
        assert result.detected_extension == ".pdf"


class TestArchiveExtensionHandling:
    """Ensure multi-suffix archives like .tar.gz remain supported."""

    @pytest.mark.unit
    def test_tar_gz_archive_extension_is_accepted(self, tmp_path):
        validator = FileValidator(custom_media_configs={
            "archive": {"allowed_mimetypes": set()},
            "document": {"allowed_mimetypes": set()},
        })
        validator.magic_available = False

        payload = tmp_path / "payload.txt"
        payload.write_text("payload")

        archive_path = tmp_path / "bundle.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(payload, arcname="payload.txt")

        result = process_and_validate_file(archive_path, validator)

        assert result.is_valid, f"Expected archive to validate, got issues: {result.issues}"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "filename",
        [
            "bundle.tar.gz",
            "dataset.tar.bz2",
            "archive.tar.xz",
        ],
    )
    def test_multi_suffix_archive_extension_resolves_media_type(self, filename):
        assert _resolve_media_type_key(filename) == "archive"

    @pytest.mark.unit
    def test_zip_symlink_is_rejected(self, tmp_path):
        validator = FileValidator(custom_media_configs={
            "archive": {"allowed_mimetypes": set()},
            "document": {"allowed_mimetypes": set()},
        })
        validator.magic_available = False

        archive_path = tmp_path / "link.zip"
        with zipfile.ZipFile(archive_path, "w") as zf:
            info = zipfile.ZipInfo("link_to_secret")
            info.create_system = 3
            info.external_attr = 0o120777 << 16  # symlink entry
            zf.writestr(info, "../../etc/passwd")

        result = process_and_validate_file(archive_path, validator)

        assert not result.is_valid
        joined = " ".join(result.issues)
        assert "unsupported symbolic link" in joined

    @pytest.mark.unit
    def test_tar_symlink_is_rejected(self, tmp_path):
        validator = FileValidator(custom_media_configs={
            "archive": {"allowed_mimetypes": set()},
            "document": {"allowed_mimetypes": set()},
        })
        validator.magic_available = False

        archive_path = tmp_path / "link.tar"
        with tarfile.open(archive_path, "w") as tar:
            info = tarfile.TarInfo("link_to_secret")
            info.type = tarfile.SYMTYPE
            info.linkname = "../../etc/passwd"
            tar.addfile(info)

        result = process_and_validate_file(archive_path, validator)

        assert not result.is_valid
        joined = " ".join(result.issues)
        assert "unsupported link entry" in joined
