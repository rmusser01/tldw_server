"""Security tests for _validate_path in database_retrievers.

These tests verify that path validation correctly blocks:
1. URI scheme bypass attacks (file:// with malicious paths)
2. Non-file URI schemes (http://, ftp://, etc.)
3. Path traversal attacks (../)
4. URL-encoded path traversal
5. Access to sensitive system directories
"""

import pytest
from pathlib import Path

from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import (
    MediaDBRetriever,
    _extract_file_uri_path,
)


@pytest.fixture
def temp_db_path(tmp_path: Path) -> str:
    """Create a temporary database path for testing valid paths."""
    db_file = tmp_path / "test.db"
    db_file.touch()
    return str(db_file)


@pytest.fixture
def retriever_instance(temp_db_path: str) -> MediaDBRetriever:
    """Create a MediaDBRetriever instance for testing _validate_path."""
    return MediaDBRetriever(db_path=temp_db_path)


@pytest.mark.unit
class TestPathValidationURISchemes:
    """Tests for URI scheme handling in _validate_path."""

    def test_file_uri_with_etc_passwd_blocked(self, retriever_instance: MediaDBRetriever):
        """file:///etc/passwd should be rejected (restricted directory)."""
        with pytest.raises(ValueError, match="suspicious pattern|not allowed"):
            retriever_instance._validate_path("file:///etc/passwd")

    def test_file_uri_with_proc_blocked(self, retriever_instance: MediaDBRetriever):
        """file:///proc/self/environ should be rejected."""
        with pytest.raises(ValueError, match="suspicious pattern|not allowed"):
            retriever_instance._validate_path("file:///proc/self/environ")

    def test_file_uri_with_query_params_stripped(self, retriever_instance: MediaDBRetriever, tmp_path: Path):
        """file:// URIs should have query parameters stripped (SQLite mode options)."""
        db_file = tmp_path / "safe.db"
        db_file.touch()
        # The validated path should be the absolute path, not the URI
        result = retriever_instance._validate_path(f"file://{db_file}?mode=ro")
        assert result == str(db_file.resolve())
        assert "?" not in result
        assert "mode" not in result

    @pytest.mark.parametrize(
        ("uri", "expected"),
        [
            (
                "file://C:/Users/runneradmin/AppData/Local/Temp/safe.db?mode=ro",
                "C:/Users/runneradmin/AppData/Local/Temp/safe.db",
            ),
            (
                r"file://C:\Users\runneradmin\AppData\Local\Temp\safe.db?mode=ro",
                r"C:\Users\runneradmin\AppData\Local\Temp\safe.db",
            ),
            (
                "file:///C:/Users/runneradmin/AppData/Local/Temp/safe.db?mode=ro",
                "C:/Users/runneradmin/AppData/Local/Temp/safe.db",
            ),
        ],
    )
    def test_windows_file_uri_drive_paths_preserved_before_validation(self, uri: str, expected: str):
        """Windows drive-letter file URIs should not collapse to the current directory."""
        assert _extract_file_uri_path(uri) == expected

    def test_http_uri_rejected(self, retriever_instance: MediaDBRetriever):
        """http:// URIs should be rejected."""
        with pytest.raises(ValueError, match="Unsupported URI scheme"):
            retriever_instance._validate_path("http://example.com/db.sqlite")

    def test_https_uri_rejected(self, retriever_instance: MediaDBRetriever):
        """https:// URIs should be rejected."""
        with pytest.raises(ValueError, match="Unsupported URI scheme"):
            retriever_instance._validate_path("https://example.com/db.sqlite")

    def test_ftp_uri_rejected(self, retriever_instance: MediaDBRetriever):
        """ftp:// URIs should be rejected."""
        with pytest.raises(ValueError, match="Unsupported URI scheme"):
            retriever_instance._validate_path("ftp://example.com/db.sqlite")

    def test_data_uri_rejected(self, retriever_instance: MediaDBRetriever):
        """data:// URIs should be rejected."""
        with pytest.raises(ValueError, match="Unsupported URI scheme"):
            retriever_instance._validate_path("data://text/plain;base64,SGVsbG8=")


@pytest.mark.unit
class TestPathValidationTraversal:
    """Tests for path traversal attack prevention."""

    def test_simple_traversal_blocked(self, retriever_instance: MediaDBRetriever):
        """../../../etc/passwd should be rejected."""
        with pytest.raises(ValueError, match="Path traversal"):
            retriever_instance._validate_path("../../../etc/passwd")

    def test_traversal_in_middle_blocked(self, retriever_instance: MediaDBRetriever):
        """Paths with .. in the middle should be rejected."""
        with pytest.raises(ValueError, match="Path traversal"):
            retriever_instance._validate_path("/some/path/../../../etc/passwd")

    def test_windows_traversal_blocked(self, retriever_instance: MediaDBRetriever):
        """Windows-style traversal should be rejected."""
        with pytest.raises(ValueError, match="Path traversal"):
            retriever_instance._validate_path("..\\..\\Windows\\System32\\config")

    def test_url_encoded_traversal_in_file_uri_blocked(self, retriever_instance: MediaDBRetriever):
        """URL-encoded traversal in file:// URI should be rejected."""
        # %2e = '.', so %2e%2e = '..'
        with pytest.raises(ValueError, match="Path traversal|not allowed"):
            retriever_instance._validate_path("file://%2e%2e/%2e%2e/etc/passwd")

    def test_double_encoded_traversal_blocked(self, retriever_instance: MediaDBRetriever):
        """Double URL-encoded traversal should be rejected.

        Note: urllib.parse.unquote only decodes once, so %252e becomes %2e (not '.').
        This test verifies that the validation still catches the suspicious /etc/ pattern
        even when the path contains URL-encoded characters.
        """
        # %252e = '%2e' after first decode (urllib only decodes once)
        # The path becomes "/%2e%2e/etc/passwd" which still contains /etc/
        with pytest.raises(ValueError, match="Path traversal|suspicious pattern|not allowed"):
            retriever_instance._validate_path("file://%252e%252e/etc/passwd")


@pytest.mark.unit
class TestPathValidationRestrictedDirs:
    """Tests for restricted directory access prevention."""

    @pytest.mark.parametrize("restricted_path", [
        "/etc/passwd",
        "/etc/shadow",
        "/proc/self/environ",
        "/proc/1/cmdline",
        "/sys/kernel/config",
        "/dev/null",
        "/boot/vmlinuz",
        "/root/.ssh/id_rsa",
    ])
    def test_restricted_unix_paths_blocked(
        self,
        retriever_instance: MediaDBRetriever,
        restricted_path: str
    ):
        """Sensitive Unix paths should be rejected."""
        with pytest.raises(ValueError, match="not allowed|suspicious"):
            retriever_instance._validate_path(restricted_path)


@pytest.mark.unit
class TestPathValidationValidPaths:
    """Tests to ensure valid paths still work after security hardening."""

    def test_absolute_path_accepted(self, retriever_instance: MediaDBRetriever, tmp_path: Path):
        """Valid absolute paths should be accepted."""
        db_file = tmp_path / "valid.db"
        db_file.touch()
        result = retriever_instance._validate_path(str(db_file))
        assert result == str(db_file.resolve())

    def test_relative_path_resolved(self, retriever_instance: MediaDBRetriever, tmp_path: Path, monkeypatch):
        """Relative paths should be resolved to absolute paths."""
        db_file = tmp_path / "relative.db"
        db_file.touch()
        monkeypatch.chdir(tmp_path)
        result = retriever_instance._validate_path("relative.db")
        assert result == str(db_file.resolve())

    def test_none_returns_none(self, retriever_instance: MediaDBRetriever):
        """None input should return None."""
        result = retriever_instance._validate_path(None)
        assert result is None

    def test_pathlib_path_accepted(self, retriever_instance: MediaDBRetriever, tmp_path: Path):
        """pathlib.Path objects should be accepted."""
        db_file = tmp_path / "pathlib.db"
        db_file.touch()
        result = retriever_instance._validate_path(db_file)  # type: ignore[arg-type]
        assert result == str(db_file.resolve())

    def test_file_uri_to_valid_path(self, retriever_instance: MediaDBRetriever, tmp_path: Path):
        """file:// URI pointing to a valid path should be accepted."""
        db_file = tmp_path / "fileuri.db"
        db_file.touch()
        result = retriever_instance._validate_path(f"file://{db_file}")
        assert result == str(db_file.resolve())

    def test_memory_sentinel_preserved(self, retriever_instance: MediaDBRetriever):
        """':memory:' must remain in-memory and never resolve to a filesystem path."""
        result = retriever_instance._validate_path(":memory:")
        assert result == ":memory:"

    def test_memory_uri_preserved(self, retriever_instance: MediaDBRetriever):
        """file::memory URI must remain in-memory and keep URI semantics."""
        memory_uri = "file::memory:?cache=shared"
        result = retriever_instance._validate_path(memory_uri)
        assert result == memory_uri

    def test_file_uri_memory_normalized(self, retriever_instance: MediaDBRetriever):
        """file:///:memory: URIs should normalize to SQLite's memory URI form."""
        result = retriever_instance._validate_path("file:///:memory:?cache=shared")
        assert result == "file::memory:?cache=shared"
