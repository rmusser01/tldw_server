from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from tldw_Server_API.app.api.v1.schemas.pagination import PagePaginationMeta


class Item(BaseModel):
    id: int
    content_item_id: int | None = None
    media_id: int | None = None
    title: str
    url: str | None = None
    domain: str | None = None
    summary: str | None = None
    published_at: str | None = None
    status: str | None = "saved"
    favorite: bool = False
    tags: list[str] = []
    type: str | None = None


class ItemsListResponse(BaseModel):
    items: list[Item]
    total: int
    page: int
    size: int
    pagination: PagePaginationMeta


BulkAction = Literal[
    "set_status",
    "set_favorite",
    "add_tags",
    "remove_tags",
    "replace_tags",
    "delete",
]


class ItemsBulkRequest(BaseModel):
    item_ids: list[int]
    action: BulkAction
    status: str | None = None
    favorite: bool | None = None
    tags: list[str] | None = None
    hard: bool = False


class ItemsBulkResult(BaseModel):
    item_id: int
    success: bool
    error: str | None = None


class ItemsBulkResponse(BaseModel):
    total: int
    succeeded: int
    failed: int
    results: list[ItemsBulkResult]
