"""Runtime-owned query wrappers for canonical MediaDatabase lookup helpers."""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.DB_Management.media_db import api as media_db_api


def get_media_by_uuid(
    self,
    media_uuid: str,
    include_deleted: bool = False,
    include_trash: bool = False,
) -> dict[str, Any] | None:
    return media_db_api.get_media_by_uuid(
        self,
        media_uuid,
        include_deleted=include_deleted,
        include_trash=include_trash,
    )


def get_media_by_id(
    self,
    media_id: int,
    include_deleted: bool = False,
    include_trash: bool = False,
) -> dict[str, Any] | None:
    return media_db_api.get_media_by_id(
        self,
        media_id,
        include_deleted=include_deleted,
        include_trash=include_trash,
    )


def get_media_status_by_id(
    self,
    media_id: int,
    include_deleted: bool = False,
    include_trash: bool = False,
) -> dict[str, Any] | None:
    return media_db_api.get_media_status_by_id(
        self,
        media_id,
        include_deleted=include_deleted,
        include_trash=include_trash,
    )


def get_media_by_url(
    self,
    url: str,
    include_deleted: bool = False,
    include_trash: bool = False,
) -> dict[str, Any] | None:
    return media_db_api.get_media_by_url(
        self,
        url,
        include_deleted=include_deleted,
        include_trash=include_trash,
    )


def get_media_by_hash(
    self,
    content_hash: str,
    include_deleted: bool = False,
    include_trash: bool = False,
) -> dict[str, Any] | None:
    return media_db_api.get_media_by_hash(
        self,
        content_hash,
        include_deleted=include_deleted,
        include_trash=include_trash,
    )


def get_media_by_title(
    self,
    title: str,
    include_deleted: bool = False,
    include_trash: bool = False,
) -> dict[str, Any] | None:
    return media_db_api.get_media_by_title(
        self,
        title,
        include_deleted=include_deleted,
        include_trash=include_trash,
    )


def get_paginated_files(
    self,
    page: int = 1,
    results_per_page: int = 50,
) -> tuple[list[dict[str, Any]], int, int, int]:
    return media_db_api.get_paginated_files(
        self,
        page=page,
        results_per_page=results_per_page,
    )


def get_paginated_media_list(
    self,
    page: int = 1,
    results_per_page: int = 10,
) -> tuple[list[dict[str, Any]], int, int, int]:
    return media_db_api.get_paginated_files(
        self,
        page=page,
        results_per_page=results_per_page,
    )


def get_paginated_trash_list(
    self,
    page: int = 1,
    results_per_page: int = 10,
) -> tuple[list[dict[str, Any]], int, int, int]:
    return media_db_api.get_paginated_trash_files(
        self,
        page=page,
        results_per_page=results_per_page,
    )


def get_distinct_media_types(
    self,
    include_deleted: bool = False,
    include_trash: bool = False,
) -> list[str]:
    return media_db_api.get_distinct_media_types(
        self,
        include_deleted=include_deleted,
        include_trash=include_trash,
    )


def has_unvectorized_chunks(self, media_id: int) -> bool:
    return media_db_api.has_unvectorized_chunks(self, media_id)


def fetch_all_keywords(self) -> list[str]:
    return media_db_api.fetch_all_keywords(self)


__all__ = [
    "get_media_by_id",
    "get_media_by_uuid",
    "get_media_by_url",
    "get_media_by_hash",
    "get_media_by_title",
    "get_paginated_media_list",
    "get_paginated_files",
    "get_paginated_trash_list",
    "get_distinct_media_types",
    "has_unvectorized_chunks",
    "fetch_all_keywords",
]
