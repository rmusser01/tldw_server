"""
Security tests for MediaWiki path traversal vulnerabilities.
Tests ensure that all file operations are protected against path traversal attacks.
"""

import pytest
import tempfile
import os
from pathlib import Path

from tldw_Server_API.app.core.Ingestion_Media_Processing.MediaWiki.Media_Wiki import (
    validate_file_path,
    sanitize_wiki_name,
    get_safe_checkpoint_path,
    get_safe_log_path,
    import_mediawiki_dump,
)
from tldw_Server_API.app.core.exceptions import InvalidStoragePathError


class TestPathTraversalProtection:
    """Test path traversal attack prevention in MediaWiki module."""

    def test_null_byte_in_file_path(self):

        """Test that null bytes in file paths are rejected."""
        with pytest.raises(InvalidStoragePathError, match="Null byte"):
            validate_file_path("/etc/passwd\x00.txt")

    def test_path_traversal_dots(self):

        """Test that path traversal with .. is blocked."""
        with pytest.raises(InvalidStoragePathError, match="Path traversal"):
            validate_file_path("../../../etc/passwd")

    def test_path_traversal_absolute(self):

        """Test that absolute paths outside allowed dir are blocked."""
        # This should fail with access denied or file not exist
        with pytest.raises(InvalidStoragePathError):
            validate_file_path("/etc/passwd")

    def test_import_rejects_outside_allowed_base(self, tmp_path: Path):

        """Test that import rejects files outside the allowed base directory."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        outside_file = outside_dir / "dump.xml"
        outside_file.write_text("<mediawiki></mediawiki>")

        events = list(
            import_mediawiki_dump(
                file_path=str(outside_file),
                wiki_name="TestWiki",
                store_to_db=False,
                store_to_vector_db=False,
                allowed_dir=allowed_dir,
            )
        )

        error_events = [event for event in events if event.get("type") == "error"]
        assert error_events
        assert "outside allowed directory" in error_events[0].get("message", "").lower()

    def test_import_accepts_inside_allowed_base(self, tmp_path: Path):

        """Test that import accepts files within the allowed base directory."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        inside_file = allowed_dir / "dump.xml"
        inside_file.write_text(
            """
<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.10/" version="0.10" xml:lang="en">
  <siteinfo>
    <sitename>TestWiki</sitename>
    <dbname>testwiki</dbname>
    <base>http://example.org/wiki/Main_Page</base>
    <generator>MediaWiki 1.42</generator>
    <case>first-letter</case>
  </siteinfo>
  <page>
    <title>Allowed Page</title>
    <ns>0</ns>
    <id>1</id>
    <revision>
      <id>11</id>
      <timestamp>2024-10-08T12:34:56Z</timestamp>
      <contributor><username>Tester</username><id>100</id></contributor>
      <comment>init</comment>
      <model>wikitext</model>
      <format>text/x-wiki</format>
      <text xml:space="preserve">Allowed content.</text>
      <sha1>dummy</sha1>
    </revision>
  </page>
</mediawiki>
            """.strip()
        )

        events = list(
            import_mediawiki_dump(
                file_path=str(inside_file),
                wiki_name="TestWiki",
                store_to_db=False,
                store_to_vector_db=False,
                allowed_dir=allowed_dir,
            )
        )

        error_events = [event for event in events if event.get("type") == "error"]
        assert not error_events
        assert any(event.get("type") == "summary" for event in events)

    def test_wiki_name_null_byte(self):

        """Test that null bytes in wiki names are rejected."""
        with pytest.raises(ValueError, match="null byte"):
            sanitize_wiki_name("test\x00wiki")

    def test_wiki_name_path_traversal(self):

        """Test that path traversal in wiki names is blocked."""
        with pytest.raises(ValueError, match="Only alphanumeric"):
            sanitize_wiki_name("../wiki")

    def test_wiki_name_forward_slash(self):

        """Test that forward slashes in wiki names are blocked."""
        with pytest.raises(ValueError, match="Only alphanumeric"):
            sanitize_wiki_name("test/wiki")

    def test_wiki_name_backslash(self):

        """Test that backslashes in wiki names are blocked."""
        with pytest.raises(ValueError, match="Only alphanumeric"):
            sanitize_wiki_name("test\\wiki")

    def test_wiki_name_too_long(self):

        """Test that overly long wiki names are rejected."""
        long_name = "a" * 101
        with pytest.raises(ValueError, match="too long"):
            sanitize_wiki_name(long_name)

    def test_safe_wiki_name(self):

        """Test that valid wiki names are accepted."""
        valid_names = [
            "TestWiki",
            "test_wiki",
            "test-wiki",
            "Test Wiki 123",
            "Wiki_2024-test"
        ]
        for name in valid_names:
            result = sanitize_wiki_name(name)
            assert result.replace(" ", "_") == result  # Spaces replaced with underscores

    def test_checkpoint_path_traversal(self):

        """Test that checkpoint paths are protected."""
        # Test path traversal in wiki name
        with pytest.raises(InvalidStoragePathError):
            get_safe_checkpoint_path("../evil")

        # Test null byte in wiki name
        with pytest.raises(InvalidStoragePathError):
            get_safe_checkpoint_path("test\x00wiki")

    def test_checkpoint_path_valid(self):

        """Test that valid checkpoint paths work."""
        path = get_safe_checkpoint_path("TestWiki")
        assert path.name == "TestWiki_import_checkpoint.json"
        assert "checkpoints" in str(path)

    def test_log_path_traversal(self):

        """Test that log paths are protected against traversal."""
        # Test path traversal
        assert get_safe_log_path("../../../etc/passwd") is None

        # Test null byte
        assert get_safe_log_path("test\x00.log") is None

        # Test directory separator
        assert get_safe_log_path("test/test.log") is None
        assert get_safe_log_path("test\\test.log") is None

    def test_log_path_invalid_extension(self):

        """Test that only .log files are allowed."""
        assert get_safe_log_path("test.txt") is None
        assert get_safe_log_path("test") is None
        assert get_safe_log_path("test.log.txt") is None

    def test_log_path_valid(self):

        """Test that valid log filenames work."""
        path = get_safe_log_path("mediawiki_import.log")
        assert path is not None
        assert path.name == "mediawiki_import.log"
        assert "Logs" in str(path)

    def test_symlink_attack_protection(self):

        """Test that symlink attacks are detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file outside the allowed directory
            outside_file = Path(tmpdir) / "outside.txt"
            outside_file.write_text("secret")

            # Create an allowed directory
            allowed_dir = Path(tmpdir) / "allowed"
            allowed_dir.mkdir()

            # Create a symlink inside allowed dir pointing outside
            symlink = allowed_dir / "link.txt"
            symlink.symlink_to(outside_file)

            # This should fail because symlink points outside allowed dir
            with pytest.raises(InvalidStoragePathError):
                validate_file_path(str(symlink), allowed_dir)

    def test_file_size_limit(self):

        """Test that file validation works for normal sized files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file in the temp directory
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_text("test content")

            # This should succeed - file exists, is small, and we specify the allowed dir
            result = validate_file_path(str(test_file), allowed_dir=Path(tmpdir))
            assert result == test_file.resolve()

            # Note: Testing the 1GB limit would require creating a huge file
            # which is impractical for unit tests. The implementation has
            # the check in place at line ~175 of Media_Wiki.py

    def test_error_messages_no_path_disclosure(self):

        """Test that error messages don't disclose sensitive paths."""
        try:
            validate_file_path("/etc/shadow/../passwd")
        except InvalidStoragePathError as e:
            error_msg = str(e)
            # Check that the actual path is not in the error message
            assert "/etc/shadow" not in error_msg
            assert "[REDACTED]" in error_msg or "Path traversal" in error_msg


if __name__ == "__main__":
    # Run basic tests
    print("Running MediaWiki security tests...")
    test = TestPathTraversalProtection()

    # Run a few key tests
    try:
        test.test_null_byte_in_file_path()
        print("✓ Null byte protection working")
    except AssertionError:
        print("✗ Null byte protection FAILED")

    try:
        test.test_path_traversal_dots()
        print("✓ Path traversal protection working")
    except AssertionError:
        print("✗ Path traversal protection FAILED")

    try:
        test.test_wiki_name_null_byte()
        print("✓ Wiki name null byte protection working")
    except AssertionError:
        print("✗ Wiki name null byte protection FAILED")

    try:
        test.test_log_path_traversal()
        print("✓ Log path protection working")
    except AssertionError:
        print("✗ Log path protection FAILED")

    print("\nAll manual tests completed!")
