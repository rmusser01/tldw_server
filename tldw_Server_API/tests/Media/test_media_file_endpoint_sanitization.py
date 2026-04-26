from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.media import file as file_mod
from tldw_Server_API.app.core.Storage.storage_interface import StorageError


class _MediaFileDb:
    def get_media_by_id(self, media_id: int, *, include_deleted: bool, include_trash: bool):
        return {"id": media_id, "title": "Stored media"}

    def get_media_file(self, media_id: int, file_type: str):
        return {
            "storage_path": "/private/storage/original.pdf",
            "mime_type": "application/pdf",
            "original_filename": "original.pdf",
            "file_size": 12,
            "checksum": "checksum",
        }


class _EmptyStoragePathDb(_MediaFileDb):
    def get_media_file(self, media_id: int, file_type: str):
        record = super().get_media_file(media_id, file_type)
        record["storage_path"] = ""
        return record


class _ExistenceErrorStorage:
    async def exists(self, storage_path: str) -> bool:
        raise StorageError("storage backend leaked /private/storage/original.pdf")


@pytest.mark.asyncio
async def test_get_media_file_sanitizes_empty_storage_path_log(monkeypatch) -> None:
    fake_logger = MagicMock()
    monkeypatch.setattr(file_mod, "logger", fake_logger)

    with pytest.raises(HTTPException) as exc_info:
        await file_mod.get_media_file(
            media_id=7,
            file_type="original",
            range_header=None,
            if_none_match=None,
            db=_EmptyStoragePathDb(),
            current_user=SimpleNamespace(id=1),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "File record is corrupted"
    fake_logger.error.assert_called_once_with("File record has empty storage path")


@pytest.mark.asyncio
async def test_get_media_file_sanitizes_storage_existence_error_log(monkeypatch) -> None:
    fake_logger = MagicMock()
    monkeypatch.setattr(file_mod, "logger", fake_logger)
    monkeypatch.setattr(file_mod, "get_storage_backend", lambda: _ExistenceErrorStorage())

    with pytest.raises(HTTPException) as exc_info:
        await file_mod.get_media_file(
            media_id=7,
            file_type="original",
            range_header=None,
            if_none_match=None,
            db=_MediaFileDb(),
            current_user=SimpleNamespace(id=1),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Error accessing file storage"
    fake_logger.error.assert_called_once_with("Storage error checking file existence")
