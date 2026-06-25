# Storage Interface
# Abstract base class defining the storage backend contract
#
# This interface allows for pluggable storage backends (filesystem, S3, etc.)
# while maintaining a consistent API for storing and retrieving media files.

import asyncio
import contextlib
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import BinaryIO, Optional, Union


class StorageBackend(ABC):
    """
    Abstract base class for storage backends.

    Implementations must provide methods for storing, retrieving, and managing
    files in the underlying storage system.
    """

    @abstractmethod
    async def store(
        self,
        user_id: str,
        media_id: int,
        filename: str,
        data: Union[BinaryIO, bytes],
        mime_type: Optional[str] = None,
    ) -> str:
        """
        Store a file in the storage backend.

        Args:
            user_id: The ID of the user who owns this file
            media_id: The ID of the media item this file belongs to
            filename: The name to store the file as (e.g., 'original.pdf')
            data: File content as bytes or file-like object
            mime_type: Optional MIME type of the file

        Returns:
            The storage path/key that can be used to retrieve the file later

        Raises:
            StorageError: If the file could not be stored
        """
        pass

    @abstractmethod
    async def retrieve(self, path: str) -> BinaryIO:
        """
        Retrieve a file from the storage backend.

        Note: This method loads the entire file into memory. For large files,
        consider using retrieve_stream() instead.

        Args:
            path: The storage path/key returned from store()

        Returns:
            A file-like object containing the file data

        Raises:
            FileNotFoundError: If the file does not exist
            StorageError: If the file could not be retrieved
        """
        pass

    async def retrieve_stream(
        self, path: str, chunk_size: int = 65536
    ) -> AsyncIterator[bytes]:
        """
        Stream file content in chunks without loading the entire file into memory.

        This is the preferred method for serving large files, as it avoids
        loading the entire file into memory at once.

        Args:
            path: The storage path/key returned from store()
            chunk_size: Size of each chunk in bytes (default: 64KB)

        Yields:
            Chunks of file content as bytes

        Raises:
            FileNotFoundError: If the file does not exist
            StorageError: If the file could not be retrieved

        Note:
            This is an optional method with a default fallback implementation
            that uses retrieve(). Subclasses should override this for better
            memory efficiency.
        """
        # Default implementation: fall back to retrieve() for backwards compatibility
        file_obj = await self.retrieve(path)
        try:
            while True:
                chunk = await asyncio.to_thread(file_obj.read, chunk_size)
                if not chunk:
                    break
                yield chunk
        finally:
            close = getattr(file_obj, "close", None)
            if callable(close):
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(close)

    @abstractmethod
    async def delete(self, path: str) -> bool:
        """
        Delete a file from the storage backend.

        Args:
            path: The storage path/key of the file to delete

        Returns:
            True if the file was deleted, False if it didn't exist

        Raises:
            StorageError: If the file could not be deleted
        """
        pass

    @abstractmethod
    async def exists(self, path: str) -> bool:
        """
        Check if a file exists in the storage backend.

        Args:
            path: The storage path/key to check

        Returns:
            True if the file exists, False otherwise
        """
        pass

    @abstractmethod
    async def get_size(self, path: str) -> int:
        """
        Get the size of a file in bytes.

        Args:
            path: The storage path/key of the file

        Returns:
            The file size in bytes

        Raises:
            FileNotFoundError: If the file does not exist
        """
        pass

    def get_url(self, path: str) -> Optional[str]:
        """
        Get a public URL for the file if applicable.

        This is primarily used for S3-like backends that can generate
        presigned URLs. Filesystem backends typically return None.

        Args:
            path: The storage path/key of the file

        Returns:
            A URL string if the backend supports it, None otherwise
        """
        return None


class StorageError(Exception):
    """Exception raised for storage-related errors."""

    def __init__(self, message: str, path: Optional[str] = None):
        self.message = message
        self.path = path
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.path:
            return f"{self.message} (path: {self.path})"
        return self.message
