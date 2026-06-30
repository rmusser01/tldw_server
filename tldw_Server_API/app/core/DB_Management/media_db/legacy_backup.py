"""Legacy backup helpers extracted from the media DB shim."""

from __future__ import annotations

from typing import Any
from loguru import logger

from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)


def create_automated_backup(db_path: Any, backup_dir: Any) -> str:
    """Create a full backup using the DB_Backups helper."""
    try:
        from tldw_Server_API.app.core.DB_Management.DB_Backups import (
            create_backup as _create_backup,
        )

        return _create_backup(db_path, backup_dir, "media")
    except MEDIA_NONCRITICAL_EXCEPTIONS as exc:
        logger.exception("create_automated_backup failed")
        return "Failed to create backup."


def create_incremental_backup(db_path: Any, backup_dir: Any) -> str:
    """Create an incremental backup using the DB_Backups helper."""
    try:
        from tldw_Server_API.app.core.DB_Management.DB_Backups import (
            create_incremental_backup as _create_incremental_backup,
        )

        return _create_incremental_backup(db_path, backup_dir, "media")
    except MEDIA_NONCRITICAL_EXCEPTIONS as exc:
        logger.exception("create_incremental_backup failed")
        return "Failed to create incremental backup."


def rotate_backups(backup_dir: Any, max_backups: int = 10) -> str:
    """Rotate backup files in a directory, keeping only the newest entries."""
    try:
        from pathlib import Path

        backup_path = Path(backup_dir)
        if not backup_path.exists():
            return "No rotation needed."

        files = [
            path for path in backup_path.iterdir()
            if path.is_file() and path.suffix in {".db", ".sqlib"}
        ]
        files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        if len(files) <= max_backups:
            return "No rotation needed."

        removed = 0
        for path in files[max_backups:]:
            try:
                path.unlink()
                removed += 1
            except MEDIA_NONCRITICAL_EXCEPTIONS:
                continue

        return f"Removed {removed} old backups."
    except MEDIA_NONCRITICAL_EXCEPTIONS as exc:
        logger.exception("rotate_backups failed")
        return "Failed to rotate backups."


__all__ = [
    "create_automated_backup",
    "create_incremental_backup",
    "rotate_backups",
]
