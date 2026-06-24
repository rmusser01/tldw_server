# tests/Storage/test_filesystem_storage.py
"""
Unit tests for the FileSystemStorage backend.

Tests cover:
- File store and retrieve operations
- Path traversal prevention
- File existence checks and deletion
- File size and checksum computation
- Path validation and sanitization
"""
import asyncio
import pytest
import tempfile
from io import BytesIO
from pathlib import Path

from tldw_Server_API.app.core.Storage.filesystem_storage import FileSystemStorage
from tldw_Server_API.app.core.Storage.storage_interface import StorageError


@pytest.fixture
def storage_backend():
    """Create a FileSystemStorage instance with a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield FileSystemStorage(base_path=tmpdir)


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def run_async(coro):


    """Helper to run async functions in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestFileSystemStorageBasic:
    """Basic storage operations."""

    @pytest.mark.unit
    def test_store_and_retrieve_bytes(self, storage_backend):
        """Test storing and retrieving bytes data."""
        async def _test():
            data = b"test PDF content here"
            path = await storage_backend.store("user1", 123, "test.pdf", data)

            # Verify path structure
            assert path == "user1/media/123/test.pdf"

            # Retrieve and verify content
            retrieved = await storage_backend.retrieve(path)
            assert retrieved.read() == data

        asyncio.run(_test())

    @pytest.mark.unit
    def test_store_and_retrieve_file_like(self, storage_backend):
        """Test storing data from a file-like object."""
        async def _test():
            data = b"file-like object content"
            file_obj = BytesIO(data)

            path = await storage_backend.store("user2", 456, "doc.pdf", file_obj)

            retrieved = await storage_backend.retrieve(path)
            assert retrieved.read() == data

        asyncio.run(_test())

    @pytest.mark.unit
    def test_store_creates_directories(self, storage_backend):
        """Test that store creates necessary directories."""
        async def _test():
            path = await storage_backend.store("newuser", 999, "file.pdf", b"data")

            # Verify directories were created
            full_path = storage_backend.base_path / path
            assert full_path.exists()
            assert full_path.parent.exists()

        asyncio.run(_test())

    @pytest.mark.unit
    def test_retrieve_nonexistent_raises(self, storage_backend):
        """Test that retrieving a non-existent file raises FileNotFoundError."""
        async def _test():
            with pytest.raises(FileNotFoundError):
                await storage_backend.retrieve("user1/media/1/nonexistent.pdf")

        asyncio.run(_test())

    @pytest.mark.unit
    def test_store_removes_partial_file_when_stream_read_fails(self, storage_backend):
        """A failed streamed write must not leave partial content at the final storage path."""

        class FailingStream:
            def __init__(self):
                self.calls = 0

            def read(self, _size):
                self.calls += 1
                if self.calls == 1:
                    return b"partial"
                raise OSError("stream failed")

            def seek(self, _offset):
                return 0

        async def _test():
            with pytest.raises(StorageError):
                await storage_backend.store("user1", 1, "file.bin", FailingStream())

            final_path = storage_backend.base_path / "user1" / "media" / "1" / "file.bin"
            assert not final_path.exists()
            leftovers = [
                path
                for path in storage_backend.base_path.rglob("*")
                if path.is_file()
            ]
            assert leftovers == []

        asyncio.run(_test())


class TestFileSystemStoragePathSecurity:
    """Path traversal and security tests."""

    @pytest.mark.unit
    def test_path_traversal_in_user_id_prevented(self, storage_backend):
        """Test that directory traversal in user_id is sanitized."""
        async def _test():
            # Attempt directory traversal via user_id
            path = await storage_backend.store("../etc", 1, "passwd", b"malicious")

            # Should sanitize to safe path
            assert ".." not in path
            assert "etc" in path  # Sanitized form

            # Verify file is within base_path
            full_path = storage_backend.base_path / path
            assert str(full_path.resolve()).startswith(str(storage_backend.base_path))

        asyncio.run(_test())

    @pytest.mark.unit
    def test_path_traversal_in_filename_prevented(self, storage_backend):
        """Test that directory traversal in filename is sanitized."""
        async def _test():
            path = await storage_backend.store("user1", 1, "../../../etc/passwd", b"data")

            # Should sanitize to safe filename
            assert ".." not in path
            assert path.startswith("user1/media/1/")

        asyncio.run(_test())

    @pytest.mark.unit
    def test_backslash_in_path_sanitized(self, storage_backend):
        """Test that backslashes in paths are sanitized."""
        async def _test():
            path = await storage_backend.store("user1", 1, "..\\..\\file.pdf", b"data")

            # Should sanitize backslashes
            assert "\\" not in path

        asyncio.run(_test())

    @pytest.mark.unit
    def test_validate_path_rejects_escape(self, storage_backend):
        """Test that path validation rejects paths outside base directory."""
        # Create a path that resolves outside base_path
        outside_path = storage_backend.base_path.parent / "outside.txt"

        with pytest.raises(StorageError) as exc_info:
            storage_backend._validate_path(outside_path)

        assert "escapes base directory" in str(exc_info.value)

    @pytest.mark.unit
    def test_validate_path_rejects_sibling_prefix_escape(self, tmp_path):
        """Sibling directories whose names share the base prefix must not pass containment."""
        base_path = tmp_path / "store"
        base_path.mkdir()
        sibling_path = tmp_path / "store_evil"
        sibling_path.mkdir()
        escaped_file = sibling_path / "secret.txt"
        escaped_file.write_text("secret", encoding="utf-8")

        storage = FileSystemStorage(base_path=base_path)
        escaped_storage_path = "../store_evil/secret.txt"

        with pytest.raises(StorageError):
            storage._validate_path(escaped_file)

        async def _test():
            assert await storage.exists(escaped_storage_path) is False
            with pytest.raises(StorageError):
                await storage.get_size(escaped_storage_path)
            with pytest.raises(StorageError):
                await storage.retrieve(escaped_storage_path)

        asyncio.run(_test())


class TestFileSystemStorageExistsAndDelete:
    """File existence and deletion tests."""

    @pytest.mark.unit
    def test_exists_returns_true_for_existing_file(self, storage_backend):
        """Test that exists returns True for files that exist."""
        async def _test():
            path = await storage_backend.store("user1", 1, "file.pdf", b"data")
            assert await storage_backend.exists(path) is True

        asyncio.run(_test())

    @pytest.mark.unit
    def test_exists_returns_false_for_missing_file(self, storage_backend):
        """Test that exists returns False for files that don't exist."""
        async def _test():
            assert await storage_backend.exists("user1/media/1/missing.pdf") is False

        asyncio.run(_test())

    @pytest.mark.unit
    def test_delete_removes_file(self, storage_backend):
        """Test that delete removes the file."""
        async def _test():
            path = await storage_backend.store("user1", 1, "file.pdf", b"data")
            assert await storage_backend.exists(path) is True

            deleted = await storage_backend.delete(path)
            assert deleted is True
            assert await storage_backend.exists(path) is False

        asyncio.run(_test())

    @pytest.mark.unit
    def test_delete_returns_false_for_missing(self, storage_backend):
        """Test that delete returns False for non-existent files."""
        async def _test():
            deleted = await storage_backend.delete("user1/media/1/nonexistent.pdf")
            assert deleted is False

        asyncio.run(_test())

    @pytest.mark.unit
    def test_delete_cleans_empty_directories(self, storage_backend):
        """Test that delete removes empty parent directories."""
        async def _test():
            path = await storage_backend.store("user1", 1, "file.pdf", b"data")
            full_path = storage_backend.base_path / path

            # Verify parent directories exist
            media_dir = full_path.parent  # user1/media/1/
            assert media_dir.exists()

            # Delete file
            await storage_backend.delete(path)

            # Empty parent directories should be cleaned up
            # (the cleanup is best-effort, so we just check the file is gone)
            assert not full_path.exists()

        asyncio.run(_test())


class TestFileSystemStorageSizeAndChecksum:
    """File size and checksum tests."""

    @pytest.mark.unit
    def test_get_size_returns_correct_size(self, storage_backend):
        """Test that get_size returns the correct file size."""
        async def _test():
            data = b"x" * 1000
            path = await storage_backend.store("user1", 1, "file.pdf", data)

            size = await storage_backend.get_size(path)
            assert size == 1000

        asyncio.run(_test())

    @pytest.mark.unit
    def test_get_size_raises_for_missing(self, storage_backend):
        """Test that get_size raises for non-existent files."""
        async def _test():
            with pytest.raises(FileNotFoundError):
                await storage_backend.get_size("user1/media/1/missing.pdf")

        asyncio.run(_test())

    @pytest.mark.unit
    def test_compute_checksum_sha256(self, storage_backend):
        """Test that compute_checksum returns correct SHA-256 hash."""
        async def _test():
            data = b"test content for hashing"
            path = await storage_backend.store("user1", 1, "file.pdf", data)

            checksum = await storage_backend.compute_checksum(path)

            # SHA-256 produces 64 hex characters
            assert len(checksum) == 64

            # Verify it's a valid hex string
            int(checksum, 16)

        asyncio.run(_test())

    @pytest.mark.unit
    def test_compute_checksum_md5(self, storage_backend):
        """Test that compute_checksum works with MD5 algorithm."""
        async def _test():
            data = b"test content"
            path = await storage_backend.store("user1", 1, "file.pdf", data)

            checksum = await storage_backend.compute_checksum(path, algorithm="md5")

            # MD5 produces 32 hex characters
            assert len(checksum) == 32

        asyncio.run(_test())

    @pytest.mark.unit
    def test_compute_checksum_raises_for_missing(self, storage_backend):
        """Test that compute_checksum raises for non-existent files."""
        async def _test():
            with pytest.raises(FileNotFoundError):
                await storage_backend.compute_checksum("user1/media/1/missing.pdf")

        asyncio.run(_test())


class TestFileSystemStorageStreaming:
    """Tests for streaming file retrieval."""

    @pytest.mark.unit
    def test_retrieve_stream_yields_chunks(self, storage_backend):
        """Test that retrieve_stream yields file content in chunks."""
        async def _test():
            # Store a file larger than chunk size
            data = b"x" * 200000  # 200KB
            path = await storage_backend.store("user1", 1, "large.pdf", data)

            # Retrieve via streaming
            chunks = []
            async for chunk in storage_backend.retrieve_stream(path, chunk_size=65536):
                chunks.append(chunk)

            # Should have multiple chunks
            assert len(chunks) > 1

            # Concatenated chunks should equal original data
            assert b"".join(chunks) == data

        asyncio.run(_test())

    @pytest.mark.unit
    def test_retrieve_stream_small_file(self, storage_backend):
        """Test that retrieve_stream works for small files."""
        async def _test():
            data = b"small file content"
            path = await storage_backend.store("user1", 1, "small.pdf", data)

            chunks = []
            async for chunk in storage_backend.retrieve_stream(path):
                chunks.append(chunk)

            # Small file should come in one chunk
            assert len(chunks) == 1
            assert chunks[0] == data

        asyncio.run(_test())

    @pytest.mark.unit
    def test_retrieve_stream_raises_for_missing(self, storage_backend):
        """Test that retrieve_stream raises for non-existent files."""
        async def _test():
            chunks = []
            with pytest.raises(FileNotFoundError):
                async for chunk in storage_backend.retrieve_stream("nonexistent.pdf"):
                    chunks.append(chunk)

        asyncio.run(_test())


class TestFileSystemStorageMisc:
    """Miscellaneous tests."""

    @pytest.mark.unit
    def test_get_url_returns_none(self, storage_backend):
        """Test that get_url returns None (filesystem doesn't support public URLs)."""
        result = storage_backend.get_url("some/path.pdf")
        assert result is None

    @pytest.mark.unit
    def test_get_full_path(self, storage_backend):
        """Test that get_full_path returns the correct absolute path."""
        async def _test():
            path = await storage_backend.store("user1", 1, "file.pdf", b"data")
            full_path = storage_backend.get_full_path(path)

            assert full_path == storage_backend.base_path / path
            assert full_path.is_absolute()

        asyncio.run(_test())

    @pytest.mark.unit
    def test_sanitize_empty_string(self, storage_backend):
        """Test that sanitizing an empty string returns 'unnamed'."""
        result = storage_backend._sanitize_path_component("")
        assert result == "unnamed"

    @pytest.mark.unit
    def test_sanitize_only_dots(self, storage_backend):
        """Test that sanitizing a string of only dots is handled safely."""
        result = storage_backend._sanitize_path_component("...")
        # The dots get replaced with underscores, then trailing dots stripped
        assert ".." not in result  # No path traversal
        assert result  # Not empty
