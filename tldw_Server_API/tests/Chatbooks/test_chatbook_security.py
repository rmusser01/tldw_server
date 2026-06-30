# test_chatbook_security.py
# Security tests for chatbook API path traversal and other vulnerabilities

import os
import io
import json
import zipfile
import tempfile
from pathlib import Path
import pytest
from unittest.mock import Mock, patch

from tldw_Server_API.app.core.Chatbooks.chatbook_validators import ChatbookValidator


def make_chatbook_manifest(**overrides):
    manifest = {
        "version": "1.0.0",
        "name": "Security Test",
        "description": "Archive validation test manifest",
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
        "content_items": [],
        "configuration": {},
        "statistics": {},
        "metadata": {},
        "user_info": {"user_id": "test"},
    }
    manifest.update(overrides)
    return manifest


def build_chatbook_archive_bytes(
    *,
    members: list[tuple[str | zipfile.ZipInfo, str | bytes]] | None = None,
    manifest: dict | None = None,
    compression: int = zipfile.ZIP_STORED,
) -> bytes:
    zip_buffer = io.BytesIO()
    archive_members = members or []
    manifest_payload = manifest or make_chatbook_manifest()

    with zipfile.ZipFile(zip_buffer, "w", compression=compression) as zf:
        zf.writestr("manifest.json", json.dumps(manifest_payload))
        for archive_member, content in archive_members:
            zf.writestr(archive_member, content)

    return zip_buffer.getvalue()


def build_traversal_archive_bytes(member_name: str = "../../../etc/passwd") -> bytes:
    return build_chatbook_archive_bytes(
        members=[(member_name, "malicious content")],
    )


def build_symlink_archive_bytes(link_name: str = "symlink_file") -> bytes:
    symlink_info = zipfile.ZipInfo(link_name)
    symlink_info.external_attr = 0xA1ED0000
    return build_chatbook_archive_bytes(
        members=[
            ("regular_file.txt", "content"),
            (symlink_info, "/etc/passwd"),
        ],
    )


def build_dangerous_file_archive_bytes(filename: str = "malware.exe") -> bytes:
    return build_chatbook_archive_bytes(
        members=[(filename, "potentially malicious content")],
    )


class TestPathTraversalSecurity:
    """Test suite for path traversal vulnerability prevention."""

    def test_validate_filename_with_path_traversal(self):
        """Test that filenames with path traversal attempts are sanitized."""
        test_cases = [
            "../../../etc/passwd.zip",
            "..\\..\\windows\\system32\\config.zip",
            "/etc/shadow.zip",
            "~/../../root/.ssh/id_rsa.zip",
            "chatbook/../../../etc/passwd.zip",
            "\\\\server\\share\\file.zip"
        ]

        for filename in test_cases:
            valid, error, safe_name = ChatbookValidator.validate_filename(filename)
            assert valid is True  # Should sanitize, not reject
            assert error is None
            assert ".." not in safe_name
            assert safe_name.endswith('.zip')
            # Verify dangerous path components are removed
            assert not safe_name.startswith('/')
            assert not safe_name.startswith('..')

        # Test invalid extension gets rejected
        invalid_cases = [
            "valid.zip/../../../etc/passwd",  # No valid extension
            "../../../etc/passwd.exe",  # Invalid extension
        ]

        for filename in invalid_cases:
            valid, error, safe_name = ChatbookValidator.validate_filename(filename)
            assert valid is False
            assert "Invalid file type" in error or "Allowed" in error

    def test_validate_filename_sanitization(self):
        """Test that dangerous characters are sanitized from filenames."""
        test_cases = [
            ("my chatbook.zip", "my_chatbook.zip"),
            ("chatbook@2024.zip", "chatbook_2024.zip"),
            ("chatbook#1.zip", "chatbook_1.zip"),
            ("chatbook$test.zip", "chatbook_test.zip"),
            ("chatbook%20file.zip", "chatbook_20file.zip"),
            ("chatbook&test.zip", "chatbook_test.zip"),
            ("chatbook*file.zip", "chatbook_file.zip"),
        ]

        for input_name, expected_output in test_cases:
            valid, error, safe_name = ChatbookValidator.validate_filename(input_name)
            assert valid is True
            assert safe_name == expected_output

    def test_validate_filename_double_extension(self):
        """Test that double extensions are handled properly.

        Files must END with a valid extension (.zip or .chatbook).
        Files that contain .zip but end with another extension should be REJECTED.
        """
        # Files ending with valid extension should be accepted
        valid_filenames = [
            "file.pdf.zip",
            "archive.tar.gz.zip",
            "my_backup.chatbook"
        ]

        for filename in valid_filenames:
            valid, error, safe_name = ChatbookValidator.validate_filename(filename)
            assert valid is True, f"Expected {filename} to be valid"
            # Should keep only last valid extension
            assert safe_name.count('.') == 1
            assert safe_name.endswith('.zip') or safe_name.endswith('.chatbook')

        # Files that contain .zip but end with dangerous extension should be REJECTED
        dangerous_filenames = [
            "chatbook.zip.exe",
            "malicious.chatbook.exe",
            "backup.zip.bat"
        ]

        for filename in dangerous_filenames:
            valid, error, safe_name = ChatbookValidator.validate_filename(filename)
            assert valid is False, f"Expected {filename} to be rejected (ends with dangerous extension)"
            assert "Invalid file type" in error

    def test_validate_sqlite_filename_accepts_safe_db_extensions(self):
        """OpenWebUI database imports accept safe SQLite filenames only."""
        test_cases = [
            ("webui.db", "webui.db"),
            ("webui.sqlite", "webui.sqlite"),
            ("my OpenWebUI backup.db", "my_OpenWebUI_backup.db"),
            ("archive.backup.sqlite", "archive_backup.sqlite"),
        ]

        for input_name, expected_output in test_cases:
            valid, error, safe_name = ChatbookValidator.validate_sqlite_filename(input_name)
            assert valid is True
            assert error is None
            assert safe_name == expected_output

    def test_validate_sqlite_filename_rejects_unsupported_and_dangerous_extensions(self):
        """OpenWebUI database filename validation rejects extension bypass attempts."""
        test_cases = [
            "webui.db.exe",
            "webui.sqlite.bat",
            "webui.json",
            "webui.chatbook",
            "webui.zip",
        ]

        for filename in test_cases:
            valid, error, safe_name = ChatbookValidator.validate_sqlite_filename(filename)
            assert valid is False, f"Expected {filename} to be rejected"
            assert "Invalid file type" in error

    def test_validate_zip_with_path_traversal(self):
        """Test ZIP validation catches path traversal in archive."""
        # Create a malicious ZIP in memory
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            # Add manifest (required)
            zf.writestr('manifest.json', '{"version": "1.0"}')
            # Add file with path traversal
            zf.writestr('../../../etc/passwd', 'malicious content')

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            tmp.write(zip_buffer.getvalue())
            tmp_path = tmp.name

        try:
            valid, error = ChatbookValidator.validate_zip_file(tmp_path)
            assert valid is False
            assert "unsafe" in error.lower()
            # Should not expose the actual path
            assert "../../../etc/passwd" not in error
        finally:
            os.unlink(tmp_path)

    def test_validate_zip_with_symlink(self):
        """Test ZIP validation rejects symbolic links."""
        # Create a ZIP with a symlink (mocked external_attr)
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            # Add manifest
            zf.writestr('manifest.json', '{"version": "1.0"}')

            # Add a regular file
            info = zipfile.ZipInfo('regular_file.txt')
            zf.writestr(info, 'content')

            # Add a symlink (set external_attr to indicate symlink)
            symlink_info = zipfile.ZipInfo('symlink_file')
            symlink_info.external_attr = 0xA1ED0000  # Unix symlink mode
            zf.writestr(symlink_info, '/etc/passwd')

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            tmp.write(zip_buffer.getvalue())
            tmp_path = tmp.name

        try:
            valid, error = ChatbookValidator.validate_zip_file(tmp_path)
            assert valid is False
            assert "symlink" in error.lower()
        finally:
            os.unlink(tmp_path)

    def test_validate_zip_bomb_detection(self):
        """Test ZIP bomb detection based on compression ratio."""
        # Create a ZIP with suspicious compression ratio
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
            # Add manifest
            zf.writestr('manifest.json', '{"version": "1.0"}')

            # Create highly compressible content (simulating zip bomb)
            # 10MB of zeros compresses to very small size
            huge_content = b'0' * (10 * 1024 * 1024)
            zf.writestr('compressed_file.txt', huge_content)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            tmp.write(zip_buffer.getvalue())
            tmp_path = tmp.name

        try:
            # Get file size for ratio check
            file_size = os.path.getsize(tmp_path)
            uncompressed_size = 10 * 1024 * 1024
            ratio = uncompressed_size / file_size if file_size > 0 else 0

            if ratio > 100:  # If compression ratio is suspicious
                valid, error = ChatbookValidator.validate_zip_file(tmp_path)
                assert valid is False
                assert "compression ratio" in error.lower() or "suspicious" in error.lower()
        finally:
            os.unlink(tmp_path)

    def test_validate_zip_skips_integrity_check_on_size_violation(self, monkeypatch):
        """Size violations should short-circuit before integrity checks run."""
        monkeypatch.setattr(ChatbookValidator, "MAX_UNCOMPRESSED_SIZE", 1024)
        monkeypatch.setattr(ChatbookValidator, "MAX_UPLOAD_SIZE", 10 * 1024 * 1024)

        called = {"testzip": False}

        def _boom(_self):
            called["testzip"] = True
            raise RuntimeError("testzip called")

        monkeypatch.setattr(zipfile.ZipFile, "testzip", _boom, raising=True)

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", '{"version": "1.0"}')
            zf.writestr("big.txt", b"x" * 2048)

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            tmp.write(zip_buffer.getvalue())
            tmp_path = tmp.name

        try:
            valid, error = ChatbookValidator.validate_zip_file(tmp_path)
            assert valid is False
            assert "maximum" in error.lower() or "too large" in error.lower()
            assert called["testzip"] is False
        finally:
            os.unlink(tmp_path)

    def test_validate_zip_with_null_bytes(self):
        """Test ZIP validation rejects filenames with null bytes."""
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            # Add manifest
            zf.writestr('manifest.json', '{"version": "1.0"}')
            # Add file with null byte in name
            zf.writestr('file\x00.txt', 'content')

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            tmp.write(zip_buffer.getvalue())
            tmp_path = tmp.name

        try:
            valid, error = ChatbookValidator.validate_zip_file(tmp_path)
            assert valid is False
            assert "invalid" in error.lower()
        finally:
            os.unlink(tmp_path)

    def test_validate_zip_with_dangerous_file_types(self):
        """Test ZIP validation rejects dangerous file types."""
        dangerous_files = [
            'malware.exe',
            'script.sh',
            'batch.bat',
            'powershell.ps1',
            'library.dll',
            'shared.so',
            'installer.msi'
        ]

        for dangerous_file in dangerous_files:
            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(zip_buffer, 'w') as zf:
                # Add manifest
                zf.writestr('manifest.json', '{"version": "1.0"}')
                # Add dangerous file
                zf.writestr(dangerous_file, 'potentially malicious content')

            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
                tmp.write(zip_buffer.getvalue())
                tmp_path = tmp.name

            try:
                valid, error = ChatbookValidator.validate_zip_file(tmp_path)
                assert valid is False
                assert "dangerous" in error.lower()
            finally:
                os.unlink(tmp_path)

    def test_path_traversal_detection(self):
        """Test the internal path traversal detection method."""
        dangerous_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "/etc/shadow",
            "~/../../root/.ssh/id_rsa",
            "./../../etc/hosts",
            "etc/passwd",  # Dangerous directory name
            "usr/bin/sh",
            "var/log/secure"
        ]

        for path in dangerous_paths:
            assert ChatbookValidator._is_path_traversal(path) is True

    def test_safe_paths(self):
        """Test that legitimate paths are not flagged as traversal."""
        safe_paths = [
            "content/file.json",
            "data/conversations/chat1.json",
            "media/images/avatar.png",
            "characters/character1.json",
            "notes/note123.md"
        ]

        for path in safe_paths:
            assert ChatbookValidator._is_path_traversal(path) is False

    def test_sanitize_path(self):
        """Test path sanitization function."""
        test_cases = [
            ("../../../etc/passwd", "passwd"),
            ("..\\..\\windows\\system32\\config", "config"),
            ("/etc/shadow", "shadow"),
            ("normal_file.txt", "normal_file.txt"),
            ("path/to/file.txt", "file.txt"),
            ("file/../other.txt", "other.txt"),
            ("file<>:\"|?.txt", "file______.txt")
        ]

        for input_path, expected_output in test_cases:
            sanitized = ChatbookValidator.sanitize_path(input_path)
            assert ".." not in sanitized
            assert "/" not in sanitized
            assert "\\" not in sanitized
            # Should only get the filename part
            assert sanitized == expected_output or sanitized == expected_output.replace('.', '_')


class TestErrorMessageSecurity:
    """Test that error messages don't expose sensitive information."""

    def test_error_messages_dont_expose_paths(self):
        """Ensure error messages don't reveal internal paths."""
        # Create various invalid ZIPs
        test_cases = [
            ("../../../etc/passwd", "unsafe"),
            ("C:\\Windows\\System32\\config.sys", "unsafe"),
            ("/home/user/.ssh/id_rsa", "unsafe")
        ]

        for malicious_path, expected_keyword in test_cases:
            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(zip_buffer, 'w') as zf:
                zf.writestr('manifest.json', '{"version": "1.0"}')
                zf.writestr(malicious_path, 'content')

            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
                tmp.write(zip_buffer.getvalue())
                tmp_path = tmp.name

            try:
                valid, error = ChatbookValidator.validate_zip_file(tmp_path)
                assert valid is False
                # Error should not contain the actual malicious path
                assert malicious_path not in error
                # Should contain generic error keyword
                assert expected_keyword in error.lower()
            finally:
                os.unlink(tmp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
