from __future__ import annotations

import threading
from io import BytesIO
from typing import BinaryIO

import pytest

from tldw_Server_API.app.core.Storage.storage_interface import StorageBackend


class _RecordingFile(BytesIO):
    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.read_thread_ids: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.read_thread_ids.append(threading.get_ident())
        return super().read(size)


class _FallbackStorage(StorageBackend):
    def __init__(self, file_obj: BinaryIO) -> None:
        self.file_obj = file_obj

    async def store(
        self,
        user_id: str,
        media_id: int,
        filename: str,
        data: BinaryIO | bytes,
        mime_type: str | None = None,
    ) -> str:
        return filename

    async def retrieve(self, path: str) -> BinaryIO:
        return self.file_obj

    async def delete(self, path: str) -> bool:
        return True

    async def exists(self, path: str) -> bool:
        return True

    async def get_size(self, path: str) -> int:
        return 0


@pytest.mark.asyncio
async def test_default_retrieve_stream_offloads_reads_and_closes_file() -> None:
    main_thread_id = threading.get_ident()
    file_obj = _RecordingFile(b"abcdef")
    storage = _FallbackStorage(file_obj)

    chunks = [
        chunk
        async for chunk in storage.retrieve_stream("storage/path.bin", chunk_size=3)
    ]

    assert chunks == [b"abc", b"def"]
    assert file_obj.closed is True
    assert file_obj.read_thread_ids
    assert all(thread_id != main_thread_id for thread_id in file_obj.read_thread_ids)
