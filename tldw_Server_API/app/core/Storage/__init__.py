# Storage Module
# Provides abstracted file storage backends for media files
#
# This module supports storing original uploaded files (PDFs, etc.) with
# a pluggable backend architecture for future S3/object storage support.

from tldw_Server_API.app.core.Storage.filesystem_storage import FileSystemStorage
from tldw_Server_API.app.core.Storage.storage_interface import StorageBackend

__all__ = ["StorageBackend", "FileSystemStorage", "get_storage_backend"]


_storage_backend_instance: FileSystemStorage | None = None


def get_storage_backend() -> StorageBackend:
    """Resolve the current profile's durable blob root, retaining explicit overrides."""
    import os
    from pathlib import Path

    from tldw_Server_API.app.core.config import settings
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

    global _storage_backend_instance

    storage_type = getattr(settings, "storage_backend", "filesystem")
    if storage_type != "filesystem":
        raise ValueError(f"Unknown storage backend: {storage_type}")

    configured = os.getenv("FILE_STORAGE_PATH", "").strip() or getattr(settings, "file_storage_path", None)
    root = Path(configured) if configured else DatabasePaths.get_user_db_base_dir() / "user_files"
    root = root.expanduser().resolve()
    backend = _storage_backend_instance
    if backend is None or backend.base_path != root:
        backend = FileSystemStorage(base_path=root)
        _storage_backend_instance = backend
    return backend


def reset_storage_backend() -> None:
    """Reset the storage backend instance (useful for testing)."""
    global _storage_backend_instance
    _storage_backend_instance = None
