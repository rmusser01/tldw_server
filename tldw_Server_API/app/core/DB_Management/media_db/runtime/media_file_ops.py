"""Package-owned MediaFiles wrapper helpers."""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.DB_Management.media_db.repositories.media_files_repository import (
    MediaFilesRepository,
)


def insert_media_file(
    self,
    media_id: int,
    file_type: str,
    storage_path: str,
    *,
    original_filename: str | None = None,
    file_size: int | None = None,
    mime_type: str | None = None,
    checksum: str | None = None,
) -> str:
    return MediaFilesRepository.from_legacy_db(self).insert(
        media_id=media_id,
        file_type=file_type,
        storage_path=storage_path,
        original_filename=original_filename,
        file_size=file_size,
        mime_type=mime_type,
        checksum=checksum,
    )


def get_media_file(
    self,
    media_id: int,
    file_type: str = "original",
    *,
    include_deleted: bool = False,
) -> dict[str, Any] | None:
    return MediaFilesRepository.from_legacy_db(self).get_for_media(
        media_id=media_id,
        file_type=file_type,
        include_deleted=include_deleted,
    )


def get_media_files(
    self,
    media_id: int,
    *,
    include_deleted: bool = False,
) -> list[dict[str, Any]]:
    return MediaFilesRepository.from_legacy_db(self).list_for_media(
        media_id=media_id,
        include_deleted=include_deleted,
    )


def has_original_file(self, media_id: int) -> bool:
    return MediaFilesRepository.from_legacy_db(self).has_original_file(media_id)


def soft_delete_media_file(
    self,
    file_id: int,
    *,
    hard_delete: bool = False,
) -> None:
    """Delete only the selected file registration, optionally removing its row."""
    MediaFilesRepository.from_legacy_db(self).soft_delete(file_id, hard_delete=hard_delete)


def soft_delete_media_files_for_media(
    self,
    media_id: int,
    *,
    hard_delete: bool = False,
) -> None:
    MediaFilesRepository.from_legacy_db(self).soft_delete_for_media(
        media_id=media_id,
        hard_delete=hard_delete,
    )
