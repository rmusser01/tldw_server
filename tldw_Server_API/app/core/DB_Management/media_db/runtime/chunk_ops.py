"""Chunk helpers owned by the package-native Media DB runtime."""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.DB_Management.media_db.runtime import (
    chunk_batch_ops,
    chunk_template_ops,
)


def add_media_chunks_in_batches(
    self: Any,
    media_id: int,
    chunks_to_add: list[dict[str, Any]],
    batch_size: int = 100,
) -> int:
    return chunk_batch_ops.add_media_chunks_in_batches(
        self,
        media_id,
        chunks_to_add,
        batch_size=batch_size,
    )


def batch_insert_chunks(self: Any, media_id: int, chunks: list[dict[str, Any]]) -> int:
    return chunk_batch_ops.batch_insert_chunks(self, media_id, chunks)


def process_chunks(
    self: Any,
    media_id: int,
    chunks: list[dict[str, Any]],
    batch_size: int = 100,
):
    return chunk_batch_ops.process_chunks(self, media_id, chunks, batch_size=batch_size)


def clear_unvectorized_chunks(self: Any, media_id: int) -> int:
    return chunk_template_ops.clear_unvectorized_chunks(self, media_id)


def create_chunking_template(
    self: Any,
    name: str,
    template_json: str,
    description: str | None = None,
    is_builtin: bool = False,
    tags: list[str] | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    return chunk_template_ops.create_chunking_template(
        self,
        name,
        template_json,
        description=description,
        is_builtin=is_builtin,
        tags=tags,
        user_id=user_id,
    )


def delete_chunking_template(
    self: Any,
    template_id: int | None = None,
    name: str | None = None,
    uuid: str | None = None,
    hard_delete: bool = False,
    *,
    owner_user_id: str | None = None,
) -> bool:
    return chunk_template_ops.delete_chunking_template(
        self,
        template_id=template_id,
        name=name,
        uuid=uuid,
        hard_delete=hard_delete,
        owner_user_id=owner_user_id,
    )


def get_chunking_template(
    self: Any,
    template_id: int | None = None,
    name: str | None = None,
    uuid: str | None = None,
    include_deleted: bool = False,
    *,
    owner_user_id: str | None = None,
) -> dict[str, Any] | None:
    return chunk_template_ops.get_chunking_template(
        self,
        template_id=template_id,
        name=name,
        uuid=uuid,
        include_deleted=include_deleted,
        owner_user_id=owner_user_id,
    )


def process_unvectorized_chunks(
    self: Any,
    media_id: int,
    chunks: list[dict[str, Any]],
    batch_size: int = 100,
) -> None:
    return chunk_template_ops.process_unvectorized_chunks(
        self,
        media_id,
        chunks,
        batch_size=batch_size,
    )


def update_chunking_template(
    self: Any,
    template_id: int | None = None,
    name: str | None = None,
    uuid: str | None = None,
    template_json: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    *,
    owner_user_id: str | None = None,
) -> bool:
    return chunk_template_ops.update_chunking_template(
        self,
        template_id=template_id,
        name=name,
        uuid=uuid,
        template_json=template_json,
        description=description,
        tags=tags,
        owner_user_id=owner_user_id,
    )

__all__ = [
    "add_media_chunks_in_batches",
    "batch_insert_chunks",
    "clear_unvectorized_chunks",
    "create_chunking_template",
    "delete_chunking_template",
    "get_chunking_template",
    "process_chunks",
    "process_unvectorized_chunks",
    "update_chunking_template",
]
