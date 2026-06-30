"""Structural validation helpers for Media DB-like objects."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MediaDbLike(Protocol):
    """Structural contract shared by Media DB implementations and wrappers."""

    client_id: str
    db_path_str: str

    def execute_query(self, query: str, params: Any = None) -> Any:
        """Execute a query and return the backend-specific result."""

    def transaction(self) -> AbstractContextManager[Any]:
        """Return a transaction context manager."""

    def _fetchall_with_connection(
        self,
        connection: Any,
        query: str,
        params: Any = None,
    ) -> Any:
        """Fetch all rows using an existing transaction-bound connection."""

    def _fetchone_with_connection(
        self,
        connection: Any,
        query: str,
        params: Any = None,
    ) -> Any:
        """Fetch one row using an existing transaction-bound connection."""

    def _execute_with_connection(
        self,
        connection: Any,
        query: str,
        params: Any = None,
    ) -> Any:
        """Execute a mutation using an existing transaction-bound connection."""

    def _get_current_utc_timestamp_str(self) -> str:
        """Return the current UTC timestamp in the DB's canonical string format."""

    def _log_sync_event(
        self,
        connection: Any,
        table_name: str,
        entity_uuid: str,
        action: str,
        version: int,
        payload: Any = None,
    ) -> None:
        """Record a sync event for replication-aware mutations."""

    def initialize_db(self) -> None:
        """Initialize schema and other database state."""

    def close_connection(self) -> None:
        """Release local database resources."""


@runtime_checkable
class MediaDbReadLike(MediaDbLike, Protocol):
    """Structural contract for the caller-facing Media DB read surface."""

    def get_media_by_id(
        self,
        media_id: int,
        include_deleted: bool = False,
        include_trash: bool = False,
    ) -> dict[str, Any] | None:
        """Return a media row by integer identifier."""

    def get_media_status_by_id(
        self,
        media_id: int,
        include_deleted: bool = False,
        include_trash: bool = False,
    ) -> dict[str, Any] | None:
        """Return lightweight media readiness fields by integer identifier."""

    def get_media_by_uuid(
        self,
        media_uuid: str,
        include_deleted: bool = False,
        include_trash: bool = False,
    ) -> dict[str, Any] | None:
        """Return a media row by UUID."""

    def get_media_by_url(
        self,
        url: str,
        include_deleted: bool = False,
        include_trash: bool = False,
    ) -> dict[str, Any] | None:
        """Return a media row by URL."""

    def get_media_by_hash(
        self,
        content_hash: str,
        include_deleted: bool = False,
        include_trash: bool = False,
    ) -> dict[str, Any] | None:
        """Return a media row by content hash."""

    def get_media_by_title(
        self,
        title: str,
        include_deleted: bool = False,
        include_trash: bool = False,
    ) -> dict[str, Any] | None:
        """Return a media row by title."""

    def get_distinct_media_types(
        self,
        include_deleted: bool = False,
        include_trash: bool = False,
    ) -> list[str]:
        """Return distinct media type values."""

    def get_paginated_files(
        self,
        page: int = 1,
        results_per_page: int = 50,
    ) -> tuple[list[dict[str, Any]], int, int, int]:
        """Return paginated active media rows."""

    def get_paginated_trash_files(
        self,
        page: int = 1,
        results_per_page: int = 50,
    ) -> tuple[list[dict[str, Any]], int, int, int]:
        """Return paginated trashed media rows."""

    def has_unvectorized_chunks(self, media_id: int) -> bool:
        """Return whether active unvectorized chunks exist for a media row."""

    def search_media_db(self, search_query: str | None, **kwargs: Any) -> tuple[list[dict[str, Any]], int]:
        """Search media rows and return the legacy `(rows, total)` tuple."""

    def get_all_document_versions(
        self,
        media_id: int,
        include_content: bool = False,
        include_deleted: bool = False,
        limit: int | None = None,
        offset: int | None = 0,
    ) -> list[dict[str, Any]]:
        """Return document versions for a media item."""

    def has_original_file(self, media_id: int) -> bool:
        """Return whether a media item has a stored original file."""


def is_media_database_like(candidate: Any) -> bool:
    required_callables = (
        "execute_query",
        "transaction",
        "_fetchall_with_connection",
        "_fetchone_with_connection",
        "_execute_with_connection",
        "_get_current_utc_timestamp_str",
        "_log_sync_event",
    )
    required_attrs = ("client_id", "db_path_str")
    return (
        candidate is not None
        and all(callable(getattr(candidate, name, None)) for name in required_callables)
        and all(getattr(candidate, name, None) is not None for name in required_attrs)
    )


def is_media_db_read_like(candidate: Any) -> bool:
    db_instance = unwrap_media_database_like(candidate)
    required_callables = (
        "get_media_by_id",
        "get_media_by_uuid",
        "get_media_by_url",
        "get_media_by_hash",
        "get_media_by_title",
        "get_distinct_media_types",
        "get_paginated_files",
        "get_paginated_trash_files",
        "has_unvectorized_chunks",
        "search_media_db",
        "get_all_document_versions",
        "has_original_file",
    )
    return is_media_database_like(db_instance) and all(
        callable(getattr(db_instance, name, None)) for name in required_callables
    )


def unwrap_media_database_like(candidate: Any) -> Any:
    if is_media_database_like(candidate):
        return candidate

    wrapped = getattr(candidate, "database", None)
    if is_media_database_like(wrapped):
        return wrapped

    return candidate


def require_media_db_read_like(
    candidate: Any,
    *,
    error_message: str,
) -> MediaDbReadLike:
    db_instance = unwrap_media_database_like(candidate)
    if not is_media_db_read_like(db_instance):
        raise TypeError(error_message)  # noqa: TRY003
    return db_instance


def require_media_database_like(
    candidate: Any,
    *,
    error_message: str,
) -> MediaDbLike:
    db_instance = unwrap_media_database_like(candidate)
    if not is_media_database_like(db_instance):
        raise TypeError(error_message)  # noqa: TRY003
    return db_instance


__all__ = [
    "MediaDbLike",
    "MediaDbReadLike",
    "is_media_db_read_like",
    "is_media_database_like",
    "require_media_db_read_like",
    "require_media_database_like",
    "unwrap_media_database_like",
]
