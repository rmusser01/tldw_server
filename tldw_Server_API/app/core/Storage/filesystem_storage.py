# Filesystem Storage Backend
# Implementation of StorageBackend for local filesystem storage
#
# Stores files in a directory structure:
#   {base_path}/{user_id}/media/{media_id}/{filename}

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import uuid
from collections.abc import AsyncIterator
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import aiofiles
import aiofiles.os
from loguru import logger

from tldw_Server_API.app.core.Storage.storage_interface import (
    StorageBackend,
    StorageError,
)


class FileSystemStorage(StorageBackend):
    """
    Filesystem-based storage backend.

    Stores files in a hierarchical directory structure organized by
    user_id and media_id for efficient lookup and isolation.
    """

    def __init__(self, base_path: str | Path):
        """
        Initialize the filesystem storage backend.

        Args:
            base_path: The root directory for file storage
        """
        self.base_path = Path(base_path).resolve()
        logger.info(f"FileSystemStorage initialized with base_path: {self.base_path}")

    def _build_path(self, user_id: str, media_id: int, filename: str) -> Path:
        """
        Build the full filesystem path for a file.

        Args:
            user_id: User ID
            media_id: Media ID
            filename: File name

        Returns:
            Full path to the file
        """
        # Sanitize inputs to prevent directory traversal
        safe_user_id = self._sanitize_path_component(str(user_id))
        safe_filename = self._sanitize_path_component(filename)

        return self.base_path / safe_user_id / "media" / str(media_id) / safe_filename

    def _sanitize_path_component(self, component: str) -> str:
        """
        Sanitize a path component to prevent directory traversal attacks.

        Args:
            component: The path component to sanitize

        Returns:
            Sanitized path component
        """
        # Remove any path separators and parent directory references
        sanitized = component.replace("/", "_").replace("\\", "_")
        sanitized = sanitized.replace("..", "_")
        # Remove any leading/trailing whitespace and dots
        sanitized = sanitized.strip().strip(".")
        if not sanitized:
            sanitized = "unnamed"
        return sanitized

    def _validate_path(self, path: Path) -> bool:
        """
        Validate that a path is within the base directory.

        Args:
            path: Path to validate

        Returns:
            True if valid, raises StorageError otherwise
        """
        try:
            resolved = path.resolve()
            try:
                resolved.relative_to(self.base_path)
            except ValueError:
                raise StorageError(
                    f"Path escapes base directory: {path}",
                    path=str(path),
                )
            return True
        except Exception as e:
            if isinstance(e, StorageError):
                raise
            raise StorageError(f"Invalid path: {e}", path=str(path)) from e

    async def store(
        self,
        user_id: str,
        media_id: int,
        filename: str,
        data: BinaryIO | bytes,
        mime_type: str | None = None,
    ) -> str:
        """
        Store a file in the filesystem.

        Args:
            user_id: The ID of the user who owns this file
            media_id: The ID of the media item this file belongs to
            filename: The name to store the file as
            data: File content as bytes or file-like object
            mime_type: Optional MIME type (not used for filesystem storage)

        Returns:
            The storage path relative to base_path
        """
        file_path = self._build_path(user_id, media_id, filename)
        self._validate_path(file_path)
        temp_path = file_path.with_name(f".{file_path.name}.{uuid.uuid4().hex}.tmp")
        self._validate_path(temp_path)

        try:
            # Ensure parent directory exists
            await aiofiles.os.makedirs(file_path.parent, exist_ok=True)

            # Get bytes from data
            total_bytes = 0
            if isinstance(data, bytes):
                file_bytes = data
                async with aiofiles.open(temp_path, 'wb') as f:
                    await f.write(file_bytes)
                total_bytes = len(file_bytes)
            else:
                # Stream from file-like object to avoid loading into memory
                async with aiofiles.open(temp_path, 'wb') as f:
                    while True:
                        chunk = await asyncio.to_thread(data.read, 1024 * 1024)
                        if not chunk:
                            break
                        total_bytes += len(chunk)
                        await f.write(chunk)
                if hasattr(data, 'seek'):
                    data.seek(0)  # Reset position for potential re-use

            await aiofiles.os.replace(temp_path, file_path)

            # Return relative path for storage in database
            relative_path = str(file_path.relative_to(self.base_path))
            logger.info(
                f"Stored file: {relative_path} ({total_bytes} bytes)"
            )
            return relative_path

        except Exception as e:
            with contextlib.suppress(FileNotFoundError, OSError):
                await aiofiles.os.remove(temp_path)
            logger.error(f"Failed to store file {file_path}: {e}")
            raise StorageError(f"Failed to store file: {e}", path=str(file_path)) from e

    async def retrieve(self, path: str) -> BinaryIO:
        """
        Retrieve a file from the filesystem.

        Args:
            path: The storage path (relative to base_path)

        Returns:
            A BytesIO object containing the file data
        """
        full_path = self.base_path / path
        self._validate_path(full_path)

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        try:
            async with aiofiles.open(full_path, 'rb') as f:
                content = await f.read()
            return BytesIO(content)
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to retrieve file {path}: {e}")
            raise StorageError(f"Failed to retrieve file: {e}", path=path) from e

    async def retrieve_stream(
        self, path: str, chunk_size: int = 65536
    ) -> AsyncIterator[bytes]:
        """
        Stream file content in chunks without loading the entire file into memory.

        This is the preferred method for serving large files, as it uses
        async file I/O and yields chunks to avoid memory pressure.

        Args:
            path: The storage path (relative to base_path)
            chunk_size: Size of each chunk in bytes (default: 64KB)

        Yields:
            Chunks of file content as bytes
        """
        full_path = self.base_path / path
        self._validate_path(full_path)

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        try:
            async with aiofiles.open(full_path, 'rb') as f:
                while True:
                    chunk = await f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to stream file {path}: {e}")
            raise StorageError(f"Failed to stream file: {e}", path=path) from e

    async def delete(self, path: str) -> bool:
        """
        Delete a file from the filesystem.

        Args:
            path: The storage path (relative to base_path)

        Returns:
            True if the file was deleted, False if it didn't exist
        """
        full_path = self.base_path / path
        self._validate_path(full_path)

        if not full_path.exists():
            logger.debug(f"File already deleted or not found: {path}")
            return False

        try:
            await aiofiles.os.remove(full_path)
            logger.info(f"Deleted file: {path}")

            # Try to clean up empty parent directories
            await self._cleanup_empty_dirs(full_path.parent)

            return True
        except Exception as e:
            logger.error(f"Failed to delete file {path}: {e}")
            raise StorageError(f"Failed to delete file: {e}", path=path) from e

    async def _cleanup_empty_dirs(self, dir_path: Path) -> None:
        """
        Remove empty parent directories up to base_path.

        Args:
            dir_path: Starting directory to check
        """
        try:
            current = dir_path
            while current != self.base_path:
                try:
                    current.resolve().relative_to(self.base_path)
                except ValueError:
                    break
                if current.exists() and not any(current.iterdir()):
                    await aiofiles.os.rmdir(current)
                    logger.debug(f"Removed empty directory: {current}")
                else:
                    break
                current = current.parent
        except Exception as e:
            # Non-critical, just log
            logger.debug(f"Could not clean up directories: {e}")

    async def exists(self, path: str) -> bool:
        """
        Check if a file exists.

        Args:
            path: The storage path (relative to base_path)

        Returns:
            True if the file exists
        """
        full_path = self.base_path / path
        try:
            self._validate_path(full_path)
            return full_path.exists() and full_path.is_file()
        except StorageError:
            return False

    async def get_size(self, path: str) -> int:
        """
        Get the size of a file in bytes.

        Args:
            path: The storage path (relative to base_path)

        Returns:
            File size in bytes
        """
        full_path = self.base_path / path
        self._validate_path(full_path)

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        try:
            stat = await aiofiles.os.stat(full_path)
            return stat.st_size
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get file size for {path}: {e}")
            raise StorageError(f"Failed to get file size: {e}", path=path) from e

    def get_url(self, path: str) -> str | None:
        """
        Get a public URL for the file.

        Filesystem storage doesn't support public URLs, so this returns None.
        Files should be served through the API endpoint.

        Args:
            path: The storage path

        Returns:
            None (filesystem storage doesn't support public URLs)
        """
        return None

    async def compute_checksum(self, path: str, algorithm: str = "sha256") -> str:
        """
        Compute a checksum for a stored file.

        Args:
            path: The storage path (relative to base_path)
            algorithm: Hash algorithm to use (default: sha256)

        Returns:
            Hex-encoded checksum string
        """
        full_path = self.base_path / path
        self._validate_path(full_path)

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        hasher = hashlib.new(algorithm)
        chunk_size = 1024 * 1024  # 1MB chunks

        try:
            async with aiofiles.open(full_path, 'rb') as f:
                while True:
                    chunk = await f.read(chunk_size)
                    if not chunk:
                        break
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Failed to compute checksum for {path}: {e}")
            raise StorageError(f"Failed to compute checksum: {e}", path=path) from e

    def get_full_path(self, path: str) -> Path:
        """
        Get the full filesystem path for a storage path.

        This is useful for debugging and direct file access.

        Args:
            path: The storage path (relative to base_path)

        Returns:
            Full filesystem Path object
        """
        full_path = self.base_path / path
        self._validate_path(full_path)
        return full_path
