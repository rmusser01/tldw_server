"""Runtime-owned wrappers for unvectorized chunk read helpers."""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.DB_Management.media_db import api as media_db_api


def get_unvectorized_chunk_count(self, media_id: int) -> int | None:
    return media_db_api.get_unvectorized_chunk_count(self, media_id)


def get_unvectorized_max_chunk_index(self, media_id: int) -> int | None:
    return media_db_api.get_unvectorized_max_chunk_index(self, media_id)


def get_unvectorized_anchor_index_for_offset(self, media_id: int, approx_offset: int) -> int | None:
    return media_db_api.get_unvectorized_anchor_index_for_offset(self, media_id, approx_offset)


def get_unvectorized_chunk_index_by_uuid(self, media_id: int, chunk_uuid: str) -> int | None:
    return media_db_api.get_unvectorized_chunk_index_by_uuid(self, media_id, chunk_uuid)


def get_unvectorized_chunk_by_index(self, media_id: int, chunk_index: int) -> dict[str, Any] | None:
    return media_db_api.get_unvectorized_chunk_by_index(self, media_id, chunk_index)


def get_unvectorized_chunks_in_range(
    self,
    media_id: int,
    start_index: int,
    end_index: int,
) -> list[dict[str, Any]]:
    return media_db_api.get_unvectorized_chunks_in_range(
        self,
        media_id,
        start_index,
        end_index,
    )


__all__ = [
    "get_unvectorized_anchor_index_for_offset",
    "get_unvectorized_chunk_by_index",
    "get_unvectorized_chunk_count",
    "get_unvectorized_chunk_index_by_uuid",
    "get_unvectorized_chunks_in_range",
    "get_unvectorized_max_chunk_index",
]
