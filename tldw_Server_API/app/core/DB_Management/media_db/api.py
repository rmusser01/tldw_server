"""Public API for the incremental Media DB package."""

import contextlib
import json
from collections.abc import Iterator
from typing import Any, Protocol

from loguru import logger

from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.repositories import (
    DocumentVersionsRepository,
    KeywordsRepository,
    MediaRepository,
)
from tldw_Server_API.app.core.DB_Management.media_db.repositories.media_lookup_repository import (
    MediaLookupRepository,
)
from tldw_Server_API.app.core.DB_Management.media_db.repositories.media_search_repository import (
    MediaSearchRepository,
)
from tldw_Server_API.app.core.DB_Management.media_db import legacy_content_queries
from tldw_Server_API.app.core.DB_Management.media_db import legacy_maintenance
from tldw_Server_API.app.core.DB_Management.media_db import legacy_reads
from tldw_Server_API.app.core.DB_Management.media_db import legacy_state
from tldw_Server_API.app.core.DB_Management.media_db import legacy_wrappers
from tldw_Server_API.app.core.DB_Management.media_db.runtime.factory import (
    MediaDbRuntimeConfig,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.factory import (
    create_media_database as runtime_create_media_database,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.defaults import (
    build_media_runtime_config,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.session import (
    MediaDbFactory,
    MediaDbSession,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.validation import (
    MediaDbLike,
    MediaDbReadLike,
    is_media_database_like,
    unwrap_media_database_like,
)
from tldw_Server_API.app.core.DB_Management.media_db.services import (
    media_details_service,
)


class MediaWriterLike(Protocol):
    """Structural contract for repository-like media writers."""

    def add_media_with_keywords(self, **kwargs: Any) -> tuple[Any, Any, Any]:
        """Persist media content and return the legacy result tuple."""


def _require_read_method(
    db: Any,
    method_name: str,
    *,
    error_message: str,
) -> Any:
    db_instance = unwrap_media_database_like(db)
    if not callable(getattr(db_instance, method_name, None)):
        raise TypeError(error_message)  # noqa: TRY003
    return db_instance


def _deleted_false_value(db_instance: Any) -> bool | int:
    backend_name = getattr(getattr(db_instance, "backend_type", None), "name", None)
    return False if backend_name == "POSTGRESQL" else 0


def _raise_read_error(operation: str, exc: Exception) -> None:
    raise DatabaseError(f"{operation} failed") from exc  # noqa: TRY003


def _supports_keyword_delete_repository(db_instance: Any) -> bool:
    """Return whether a DB-like object can drive KeywordsRepository.soft_delete()."""
    required_attrs = (
        "client_id",
        "transaction",
        "_fetchone_with_connection",
        "_execute_with_connection",
        "_get_current_utc_timestamp_str",
        "_log_sync_event",
        "_delete_fts_keyword",
    )
    return all(hasattr(db_instance, attr) for attr in required_attrs)


def create_media_database(
    client_id: str,
    *,
    db_path: str | None = None,
    backend: Any = None,
    config: Any = None,
) -> MediaDbLike:
    """Create a MediaDatabase using the shared content runtime defaults."""
    return runtime_create_media_database(
        client_id,
        db_path=db_path,
        backend=backend,
        config=config,
        runtime=build_media_runtime_config(),
    )


@contextlib.contextmanager
def managed_media_database(
    client_id: str,
    *,
    db_path: str | None = None,
    backend=None,
    config=None,
    initialize: bool = True,
    suppress_init_exceptions: tuple[type[BaseException], ...] = (),
    suppress_close_exceptions: tuple[type[BaseException], ...] = (),
) -> Iterator[MediaDbLike]:
    """Create a MediaDatabase, optionally initialize it, and always close it on exit."""
    db = create_media_database(
        client_id,
        db_path=db_path,
        backend=backend,
        config=config,
    )
    try:
        if initialize:
            if suppress_init_exceptions:
                with contextlib.suppress(*suppress_init_exceptions):
                    db.initialize_db()
            else:
                db.initialize_db()
        yield db
    finally:
        if suppress_close_exceptions:
            with contextlib.suppress(*suppress_close_exceptions):
                db.close_connection()
        else:
            db.close_connection()


def get_media_repository(db: MediaDbLike | MediaWriterLike) -> MediaRepository | MediaWriterLike:
    """Return a repository-backed media ingest interface for DB sessions or writer doubles."""
    add_media = getattr(db, "add_media_with_keywords", None)
    transaction = getattr(db, "transaction", None)
    if callable(add_media) and not callable(transaction):
        return db
    return MediaRepository.from_legacy_db(db)


def get_media_by_id(
    db: MediaDbLike | MediaDbReadLike,
    media_id: int,
    *,
    include_deleted: bool = False,
    include_trash: bool = False,
) -> dict[str, Any] | None:
    """Return a media record by ID through the package-level read contract."""
    db_instance = unwrap_media_database_like(db)
    if is_media_database_like(db_instance):
        return MediaLookupRepository.from_legacy_db(db_instance).by_id(
            media_id,
            include_deleted=include_deleted,
            include_trash=include_trash,
        )
    reader = _require_read_method(
        db,
        "get_media_by_id",
        error_message="db must expose the Media DB read contract.",
    )
    return reader.get_media_by_id(
        media_id,
        include_deleted=include_deleted,
        include_trash=include_trash,
    )


def has_unvectorized_chunks(
    db: MediaDbLike | MediaDbReadLike,
    media_id: int,
) -> bool:
    """Return whether active unvectorized chunks exist for a media item."""
    db_instance = unwrap_media_database_like(db)
    if is_media_database_like(db_instance):
        try:
            cursor = db_instance.execute_query(
                "SELECT 1 FROM UnvectorizedMediaChunks WHERE media_id = ? AND deleted = 0 LIMIT 1",
                (media_id,),
            )
            return cursor.fetchone() is not None
        except Exception as exc:
            _raise_read_error("has_unvectorized_chunks", exc)

    reader = _require_read_method(
        db,
        "has_unvectorized_chunks",
        error_message="db must expose the Media DB read contract.",
    )
    return bool(reader.has_unvectorized_chunks(media_id))


def get_media_by_uuid(
    db: MediaDbLike | MediaDbReadLike,
    media_uuid: str,
    *,
    include_deleted: bool = False,
    include_trash: bool = False,
) -> dict[str, Any] | None:
    """Return a media record by UUID through the package-level read contract."""
    db_instance = unwrap_media_database_like(db)
    if is_media_database_like(db_instance):
        return MediaLookupRepository.from_legacy_db(db_instance).by_uuid(
            media_uuid,
            include_deleted=include_deleted,
            include_trash=include_trash,
        )
    reader = _require_read_method(
        db,
        "get_media_by_uuid",
        error_message="db must expose the Media DB read contract.",
    )
    return reader.get_media_by_uuid(
        media_uuid,
        include_deleted=include_deleted,
        include_trash=include_trash,
    )


def get_media_by_url(
    db: MediaDbLike | MediaDbReadLike,
    url: str,
    *,
    include_deleted: bool = False,
    include_trash: bool = False,
) -> dict[str, Any] | None:
    """Return a media record by URL through the package-level read contract."""
    db_instance = unwrap_media_database_like(db)
    if is_media_database_like(db_instance):
        return MediaLookupRepository.from_legacy_db(db_instance).by_url(
            url,
            include_deleted=include_deleted,
            include_trash=include_trash,
        )
    reader = _require_read_method(
        db,
        "get_media_by_url",
        error_message="db must expose the Media DB read contract.",
    )
    return reader.get_media_by_url(
        url,
        include_deleted=include_deleted,
        include_trash=include_trash,
    )


def get_media_by_hash(
    db: MediaDbLike | MediaDbReadLike,
    content_hash: str,
    *,
    include_deleted: bool = False,
    include_trash: bool = False,
) -> dict[str, Any] | None:
    """Return a media record by hash through the package-level read contract."""
    db_instance = unwrap_media_database_like(db)
    if is_media_database_like(db_instance):
        return MediaLookupRepository.from_legacy_db(db_instance).by_hash(
            content_hash,
            include_deleted=include_deleted,
            include_trash=include_trash,
        )
    reader = _require_read_method(
        db,
        "get_media_by_hash",
        error_message="db must expose the Media DB read contract.",
    )
    return reader.get_media_by_hash(
        content_hash,
        include_deleted=include_deleted,
        include_trash=include_trash,
    )


def get_media_by_title(
    db: MediaDbLike | MediaDbReadLike,
    title: str,
    *,
    include_deleted: bool = False,
    include_trash: bool = False,
) -> dict[str, Any] | None:
    """Return a media record by title through the package-level read contract."""
    db_instance = unwrap_media_database_like(db)
    if is_media_database_like(db_instance):
        return MediaLookupRepository.from_legacy_db(db_instance).by_title(
            title,
            include_deleted=include_deleted,
            include_trash=include_trash,
        )
    reader = _require_read_method(
        db,
        "get_media_by_title",
        error_message="db must expose the Media DB read contract.",
    )
    return reader.get_media_by_title(
        title,
        include_deleted=include_deleted,
        include_trash=include_trash,
    )


def search_media(
    db: MediaDbLike | MediaDbReadLike,
    search_query: str | None,
    **kwargs: Any,
) -> tuple[list[dict[str, Any]], int]:
    """Search media through the package-level read contract."""
    db_instance = unwrap_media_database_like(db)
    if is_media_database_like(db_instance):
        return MediaSearchRepository.from_legacy_db(db_instance).search(
            search_query=search_query,
            **kwargs,
        )

    reader = _require_read_method(
        db,
        "search_media_db",
        error_message="db must expose the Media DB read contract.",
    )
    return reader.search_media_db(search_query=search_query, **kwargs)


def list_document_versions(
    db: MediaDbLike | MediaDbReadLike,
    media_id: int,
    *,
    include_content: bool = False,
    include_deleted: bool = False,
    limit: int | None = None,
    offset: int | None = 0,
) -> list[dict[str, Any]]:
    """List document versions through the package-level read contract."""
    db_instance = unwrap_media_database_like(db)
    if is_media_database_like(db_instance):
        return DocumentVersionsRepository.from_legacy_db(db_instance).list(
            media_id=media_id,
            include_content=include_content,
            include_deleted=include_deleted,
            limit=limit,
            offset=offset,
        )
    reader = _require_read_method(
        db,
        "get_all_document_versions",
        error_message="db must expose the Media DB read contract.",
    )
    return reader.get_all_document_versions(
        media_id=media_id,
        include_content=include_content,
        include_deleted=include_deleted,
        limit=limit,
        offset=offset,
    )


def get_all_document_versions(
    db: MediaDbLike | MediaDbReadLike,
    media_id: int,
    *,
    include_content: bool = False,
    include_deleted: bool = False,
    limit: int | None = None,
    offset: int | None = 0,
) -> list[dict[str, Any]]:
    """Backward-compatible alias for listing document versions."""
    return list_document_versions(
        db,
        media_id,
        include_content=include_content,
        include_deleted=include_deleted,
        limit=limit,
        offset=offset,
    )


def create_document_version(
    db: MediaDbLike,
    media_id: int,
    content: str,
    *,
    prompt: str | None = None,
    analysis_content: str | None = None,
    safe_metadata: str | None = None,
) -> dict[str, Any]:
    """Create a document version through the package-level write helper."""
    db_instance = unwrap_media_database_like(db)
    if is_media_database_like(db_instance):
        return DocumentVersionsRepository.from_legacy_db(db_instance).create(
            media_id=media_id,
            content=content,
            prompt=prompt,
            analysis_content=analysis_content,
            safe_metadata=safe_metadata,
        )

    create_version = getattr(db_instance, "create_document_version", None)
    if not callable(create_version):
        raise TypeError("db must expose the Media DB document-version contract.")  # noqa: TRY003
    return create_version(
        media_id=media_id,
        content=content,
        prompt=prompt,
        analysis_content=analysis_content,
        safe_metadata=safe_metadata,
    )


def update_keywords_for_media(
    db: MediaDbLike,
    media_id: int,
    keywords: list[str],
    *,
    conn: Any | None = None,
):
    """Replace media keywords through the package-level write helper."""
    db_instance = unwrap_media_database_like(db)
    if is_media_database_like(db_instance):
        return KeywordsRepository.from_legacy_db(db_instance).replace_keywords(
            media_id=media_id,
            keywords=keywords,
            conn=conn,
        )

    update_keywords = getattr(db_instance, "update_keywords_for_media", None)
    if not callable(update_keywords):
        raise TypeError("db must expose the Media DB keyword-update contract.")  # noqa: TRY003
    return update_keywords(media_id=media_id, keywords=keywords, conn=conn)


def soft_delete_keyword(
    db: MediaDbLike,
    keyword: str,
) -> bool:
    """Soft-delete a keyword through the package-level write helper."""
    db_instance = unwrap_media_database_like(db)
    if is_media_database_like(db_instance) or _supports_keyword_delete_repository(db_instance):
        return KeywordsRepository.from_legacy_db(db_instance).soft_delete(keyword)

    soft_delete = getattr(db_instance, "soft_delete_keyword", None)
    if not callable(soft_delete):
        raise TypeError("db must expose the Media DB keyword-delete contract.")  # noqa: TRY003
    return bool(soft_delete(keyword))


def soft_delete_document_version(
    db: MediaDbLike,
    version_uuid: str,
) -> bool:
    """Soft-delete a document version through the package-level write helper."""
    db_instance = unwrap_media_database_like(db)
    if is_media_database_like(db_instance):
        return DocumentVersionsRepository.from_legacy_db(db_instance).soft_delete(version_uuid)

    soft_delete = getattr(db_instance, "soft_delete_document_version", None)
    if not callable(soft_delete):
        raise TypeError("db must expose the Media DB document-version delete contract.")  # noqa: TRY003
    return bool(soft_delete(version_uuid))


def get_paginated_files(
    db: MediaDbLike,
    *,
    page: int = 1,
    results_per_page: int = 50,
) -> tuple[list[dict[str, Any]], int, int, int]:
    """Return paginated active media rows using the database's native list method."""
    db_instance = unwrap_media_database_like(db)
    if is_media_database_like(db_instance):
        return MediaLookupRepository.from_legacy_db(db_instance).paginated_files(
            page=page,
            results_per_page=results_per_page,
        )
    paginated = getattr(db_instance, "get_paginated_files", None)
    if callable(paginated):
        return paginated(page=page, results_per_page=results_per_page)

    paginated = getattr(db_instance, "get_paginated_media_list", None)
    if callable(paginated):
        return paginated(page=page, results_per_page=results_per_page)

    reader = _require_read_method(
        db,
        "get_paginated_files",
        error_message="db must expose the Media DB list contract.",
    )
    return reader.get_paginated_files(page=page, results_per_page=results_per_page)


def get_paginated_trash_files(
    db: MediaDbLike,
    *,
    page: int = 1,
    results_per_page: int = 50,
) -> tuple[list[dict[str, Any]], int, int, int]:
    """Return paginated trashed media rows using the database's native list method."""
    db_instance = unwrap_media_database_like(db)
    if is_media_database_like(db_instance):
        return MediaLookupRepository.from_legacy_db(db_instance).paginated_trash(
            page=page,
            results_per_page=results_per_page,
        )
    paginated = getattr(db_instance, "get_paginated_trash_files", None)
    if callable(paginated):
        return paginated(page=page, results_per_page=results_per_page)

    paginated = getattr(db_instance, "get_paginated_trash_list", None)
    if callable(paginated):
        return paginated(page=page, results_per_page=results_per_page)

    reader = _require_read_method(
        db,
        "get_paginated_trash_files",
        error_message="db must expose the Media DB list contract.",
    )
    return reader.get_paginated_trash_files(page=page, results_per_page=results_per_page)


def get_distinct_media_types(
    db: MediaDbLike | MediaDbReadLike,
    *,
    include_deleted: bool = False,
    include_trash: bool = False,
) -> list[str]:
    """Return distinct media types through the package-level read contract."""
    db_instance = unwrap_media_database_like(db)
    if is_media_database_like(db_instance):
        return MediaLookupRepository.from_legacy_db(db_instance).distinct_media_types(
            include_deleted=include_deleted,
            include_trash=include_trash,
        )
    reader = _require_read_method(
        db,
        "get_distinct_media_types",
        error_message="db must expose the Media DB read contract.",
    )
    return reader.get_distinct_media_types(
        include_deleted=include_deleted,
        include_trash=include_trash,
    )


def get_unvectorized_chunk_count(
    db: MediaDbLike | MediaDbReadLike,
    media_id: int,
) -> int | None:
    """Return the active unvectorized chunk count for a media item."""
    db_instance = unwrap_media_database_like(db)
    if is_media_database_like(db_instance):
        try:
            media_id_int = int(media_id)
        except (TypeError, ValueError):
            return None

        try:
            cursor = db_instance.execute_query(
                "SELECT COUNT(*) AS chunk_count FROM UnvectorizedMediaChunks "
                "WHERE media_id = ? AND deleted = 0",
                (media_id_int,),
            )
            row = cursor.fetchone()
            if not row:
                return 0
            if isinstance(row, dict):
                return int(row.get("chunk_count", 0) or 0)
            with contextlib.suppress(Exception):
                return int(row["chunk_count"] or 0)
            with contextlib.suppress(Exception):
                return int(row[0] or 0)
            return 0
        except Exception as exc:
            _raise_read_error("get_unvectorized_chunk_count", exc)

    reader = _require_read_method(
        db,
        "get_unvectorized_chunk_count",
        error_message="db must expose the Media DB read contract.",
    )
    return reader.get_unvectorized_chunk_count(media_id)


def get_unvectorized_anchor_index_for_offset(
    db: MediaDbLike | MediaDbReadLike,
    media_id: int,
    approx_offset: int,
) -> int | None:
    """Map an approximate character offset to an unvectorized chunk index."""
    db_instance = unwrap_media_database_like(db)
    if is_media_database_like(db_instance):
        try:
            cursor = db_instance.execute_query(
                """
                SELECT chunk_index
                FROM UnvectorizedMediaChunks
                WHERE media_id = ? AND deleted = 0 AND start_char IS NOT NULL AND end_char IS NOT NULL
                  AND start_char <= ? AND end_char > ?
                ORDER BY chunk_index ASC
                LIMIT 1
                """,
                (media_id, approx_offset, approx_offset),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return int(row["chunk_index"]) if isinstance(row, dict) else int(row[0])
        except Exception as exc:
            _raise_read_error("get_unvectorized_anchor_index_for_offset", exc)

    reader = _require_read_method(
        db,
        "get_unvectorized_anchor_index_for_offset",
        error_message="db must expose the Media DB read contract.",
    )
    return reader.get_unvectorized_anchor_index_for_offset(media_id, approx_offset)


def get_unvectorized_chunk_index_by_uuid(
    db: MediaDbLike | MediaDbReadLike,
    media_id: int,
    chunk_uuid: str,
) -> int | None:
    """Return the chunk index for a given unvectorized chunk UUID."""
    db_instance = unwrap_media_database_like(db)
    if is_media_database_like(db_instance):
        try:
            cursor = db_instance.execute_query(
                "SELECT chunk_index FROM UnvectorizedMediaChunks "
                "WHERE media_id = ? AND uuid = ? AND deleted = 0",
                (media_id, chunk_uuid),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return int(row["chunk_index"]) if isinstance(row, dict) else int(row[0])
        except Exception as exc:
            _raise_read_error("get_unvectorized_chunk_index_by_uuid", exc)

    reader = _require_read_method(
        db,
        "get_unvectorized_chunk_index_by_uuid",
        error_message="db must expose the Media DB read contract.",
    )
    return reader.get_unvectorized_chunk_index_by_uuid(media_id, chunk_uuid)


def get_unvectorized_chunk_by_index(
    db: MediaDbLike | MediaDbReadLike,
    media_id: int,
    chunk_index: int,
) -> dict[str, Any] | None:
    """Return one unvectorized chunk row by media id and chunk index."""
    db_instance = unwrap_media_database_like(db)
    if is_media_database_like(db_instance):
        try:
            cursor = db_instance.execute_query(
                """
                SELECT chunk_index, chunk_text, start_char, end_char, chunk_type
                FROM UnvectorizedMediaChunks
                WHERE media_id = ? AND chunk_index = ? AND deleted = 0
                ORDER BY id DESC
                LIMIT 1
                """,
                (int(media_id), int(chunk_index)),
            )
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as exc:
            _raise_read_error("get_unvectorized_chunk_by_index", exc)

    reader = _require_read_method(
        db,
        "get_unvectorized_chunk_by_index",
        error_message="db must expose the Media DB read contract.",
    )
    return reader.get_unvectorized_chunk_by_index(media_id, chunk_index)


def get_unvectorized_chunks_in_range(
    db: MediaDbLike | MediaDbReadLike,
    media_id: int,
    start_index: int,
    end_index: int,
) -> list[dict[str, Any]]:
    """Return unvectorized chunk rows in an inclusive chunk-index range."""
    db_instance = unwrap_media_database_like(db)
    if is_media_database_like(db_instance):
        if end_index < start_index:
            start_index, end_index = end_index, start_index
        try:
            cursor = db_instance.execute_query(
                """
                SELECT chunk_index, uuid, chunk_text, start_char, end_char, chunk_type
                FROM UnvectorizedMediaChunks
                WHERE media_id = ? AND deleted = 0 AND chunk_index BETWEEN ? AND ?
                ORDER BY chunk_index ASC
                """,
                (media_id, start_index, end_index),
            )
            return [dict(row) for row in cursor.fetchall()]
        except Exception as exc:
            _raise_read_error("get_unvectorized_chunks_in_range", exc)

    reader = _require_read_method(
        db,
        "get_unvectorized_chunks_in_range",
        error_message="db must expose the Media DB read contract.",
    )
    return reader.get_unvectorized_chunks_in_range(media_id, start_index, end_index)


def list_chunking_templates(
    db: MediaDbLike,
    *,
    include_builtin: bool = True,
    include_custom: bool = True,
    tags: list[str] | None = None,
    user_id: str | None = None,
    include_deleted: bool = False,
) -> list[dict[str, Any]]:
    """Return chunking templates with the same filtering shape as the legacy DB API."""
    db_instance = unwrap_media_database_like(db)
    if is_media_database_like(db_instance):
        if not include_builtin and not include_custom:
            return []

        conditions: list[str] = []
        params: list[Any] = []

        if not include_deleted:
            conditions.append("deleted = ?")
            params.append(False)

        if include_builtin != include_custom:
            conditions.append("is_builtin = ?")
            params.append(include_builtin)

        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)

        query = "SELECT * FROM ChunkingTemplates"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY is_builtin DESC, name ASC"

        cursor = db_instance.execute_query(query, tuple(params))
        templates: list[dict[str, Any]] = []
        for row in cursor.fetchall():
            template = {
                "id": row["id"],
                "uuid": row["uuid"],
                "name": row["name"],
                "description": row["description"],
                "template_json": row["template_json"],
                "is_builtin": bool(row["is_builtin"]),
                "tags": json.loads(row["tags"]) if row["tags"] else [],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "version": row["version"],
                "user_id": row["user_id"],
            }
            if tags and not any(tag in template["tags"] for tag in tags):
                continue
            templates.append(template)

        return templates

    reader = _require_read_method(
        db,
        "list_chunking_templates",
        error_message="db must expose the Media DB chunking-template contract.",
    )
    return reader.list_chunking_templates(
        include_builtin=include_builtin,
        include_custom=include_custom,
        tags=tags,
        user_id=user_id,
        include_deleted=include_deleted,
    )


def seed_builtin_templates(
    db: MediaDbLike,
    templates: list[dict[str, Any]],
) -> int:
    """Seed built-in chunking templates while preserving legacy restore semantics."""
    db_instance = unwrap_media_database_like(db)
    if not is_media_database_like(db_instance):
        writer = _require_read_method(
            db,
            "seed_builtin_templates",
            error_message="db must expose the Media DB chunking-template contract.",
        )
        return writer.seed_builtin_templates(templates)

    count = 0
    for template in templates:
        existing = db_instance.get_chunking_template(
            name=template["name"],
            include_deleted=True,
        )

        if not existing:
            try:
                db_instance.create_chunking_template(
                    name=template["name"],
                    template_json=json.dumps(template.get("template", template)),
                    description=template.get("description", ""),
                    is_builtin=True,
                    tags=template.get("tags", []),
                )
                count += 1
                logger.info(f"Seeded built-in template: {template['name']}")
            except Exception:
                logger.exception(f"Failed to seed template {template['name']}")
            continue

        if not existing["deleted"]:
            continue

        current_time = db_instance._get_current_utc_timestamp_str()
        with db_instance.transaction() as conn:
            db_instance._execute_with_connection(
                conn,
                """
                UPDATE ChunkingTemplates
                SET deleted = ?,
                    template_json = ?,
                    description = ?,
                    tags = ?,
                    updated_at = ?,
                    last_modified = ?,
                    version = version + 1,
                    client_id = ?
                WHERE id = ?
                """,
                (
                    False,
                    json.dumps(template.get("template", template)),
                    template.get("description", ""),
                    json.dumps(template.get("tags", [])),
                    current_time,
                    current_time,
                    db_instance.client_id,
                    existing["id"],
                ),
            )
        count += 1
        logger.info(f"Restored built-in template: {template['name']}")

    return count


def lookup_section_for_offset(
    db: MediaDbLike,
    media_id: int,
    char_offset: int,
) -> dict[str, Any] | None:
    """Return the most specific document section covering the supplied character offset."""
    db_instance = unwrap_media_database_like(db)
    if is_media_database_like(db_instance):
        if media_id is None or char_offset is None:
            return None
        try:
            with db_instance.transaction() as conn:
                cur = db_instance._execute_with_connection(
                    conn,
                    "SELECT id, title, level, start_char, end_char FROM DocumentStructureIndex "
                    "WHERE media_id = ? AND deleted = ? AND kind IN ('section','header') "
                    "AND start_char <= ? AND end_char > ? "
                    "ORDER BY COALESCE(level, 0) DESC, start_char DESC LIMIT 1",
                    (media_id, _deleted_false_value(db_instance), char_offset, char_offset),
                )
                row = cur.fetchone()
        except Exception as exc:
            _raise_read_error("lookup_section_for_offset", exc)

        if not row:
            return None
        if isinstance(row, dict):
            return row
        return {
            "id": row["id"],
            "title": row["title"],
            "level": row["level"],
            "start_char": row["start_char"],
            "end_char": row["end_char"],
        }

    reader = _require_read_method(
        db,
        "lookup_section_for_offset",
        error_message="db must expose the Media DB section lookup contract.",
    )
    return reader.lookup_section_for_offset(media_id, char_offset)


def lookup_section_by_heading(
    db: MediaDbLike,
    media_id: int,
    heading: str,
) -> tuple[int, int, str] | None:
    """Best-effort lookup of a section by a case-insensitive heading match."""
    db_instance = unwrap_media_database_like(db)
    if is_media_database_like(db_instance):
        if not media_id or not heading:
            return None
        try:
            with db_instance.transaction() as conn:
                cur = db_instance._execute_with_connection(
                    conn,
                    "SELECT start_char, end_char, title FROM DocumentStructureIndex "
                    "WHERE media_id = ? AND deleted = ? AND kind IN ('section','header') "
                    "AND LOWER(title) LIKE LOWER(?) "
                    "ORDER BY COALESCE(level,0) DESC, (end_char - start_char) DESC LIMIT 1",
                    (media_id, _deleted_false_value(db_instance), f"%{heading.strip()}%"),
                )
                row = cur.fetchone()
        except Exception as exc:
            _raise_read_error("lookup_section_by_heading", exc)

        if not row:
            return None
        return (int(row["start_char"]), int(row["end_char"]), str(row["title"]))

    reader = _require_read_method(
        db,
        "lookup_section_by_heading",
        error_message="db must expose the Media DB section lookup contract.",
    )
    return reader.lookup_section_by_heading(media_id, heading)


def fetch_keywords_for_media(db: MediaDbLike, media_id: int) -> list[str]:
    """Return keywords for a single media row through the package-level helper."""
    return legacy_content_queries.fetch_keywords_for_media(media_id=media_id, db_instance=db)


def fetch_keywords_for_media_batch(
    db: MediaDbLike,
    media_ids: list[int],
) -> dict[int, list[str]]:
    """Return keywords for a batch of media rows through the package-level helper."""
    return legacy_content_queries.fetch_keywords_for_media_batch(
        media_ids=media_ids,
        db_instance=db,
    )


def get_full_media_details(
    db: MediaDbLike | MediaDbReadLike,
    media_id: int,
    *,
    include_content: bool = True,
) -> dict[str, Any] | None:
    """Return full media details through the package-level read contract."""
    return media_details_service.get_full_media_details(
        db_instance=db,
        media_id=media_id,
        include_content=include_content,
    )


def get_full_media_details_rich(
    db: MediaDbLike | MediaDbReadLike,
    media_id: int,
    *,
    include_content: bool = True,
    include_versions: bool = True,
    include_version_content: bool = False,
) -> dict[str, Any] | None:
    """Return rich media details through the package-level read contract."""
    return media_details_service.get_full_media_details_rich(
        db_instance=db,
        media_id=media_id,
        include_content=include_content,
        include_versions=include_versions,
        include_version_content=include_version_content,
    )


def get_document_version(
    db: MediaDbLike | MediaDbReadLike,
    media_id: int,
    version_number: int | None = None,
    include_content: bool = True,
) -> dict[str, Any] | None:
    """Return one document version through the package-level read contract."""
    db_instance = unwrap_media_database_like(db)
    if is_media_database_like(db_instance):
        return legacy_wrappers.get_document_version(
            db_instance=db_instance,
            media_id=media_id,
            version_number=version_number,
            include_content=include_content,
        )

    get_version = getattr(db_instance, "get_document_version", None)
    if not callable(get_version):
        raise TypeError("db must expose the Media DB read contract.")
    return get_version(
        media_id=media_id,
        version_number=version_number,
        include_content=include_content,
    )


def check_media_exists(
    db: MediaDbLike,
    media_id: int | None = None,
    url: str | None = None,
    content_hash: str | None = None,
) -> int | None:
    """Return the active media id for the given lookup fields."""
    return legacy_state.check_media_exists(
        db_instance=db,
        media_id=media_id,
        url=url,
        content_hash=content_hash,
    )


def permanently_delete_item(db: MediaDbLike, media_id: int) -> bool:
    """Permanently delete a media item through the maintenance helper facade."""
    return legacy_maintenance.permanently_delete_item(
        db_instance=db,
        media_id=media_id,
    )


def get_latest_transcription(db: MediaDbLike, media_id: int) -> str | None:
    """Return the most recent transcript text for a media item."""
    return legacy_reads.get_latest_transcription(db_instance=db, media_id=media_id)


def fetch_all_keywords(db: MediaDbLike | MediaDbReadLike) -> list[str]:
    """Return all active keyword strings through the package-level read contract."""
    db_instance = unwrap_media_database_like(db)
    if is_media_database_like(db_instance):
        order_expr = db_instance._keyword_order_expression("keyword")
        query = f"SELECT keyword FROM Keywords WHERE deleted = ? ORDER BY {order_expr}"  # nosec B608
        cursor = db_instance.execute_query(query, (False,))
        return [row["keyword"] for row in cursor.fetchall()]

    reader = _require_read_method(
        db,
        "fetch_all_keywords",
        error_message="db must expose the Media DB read contract.",
    )
    return reader.fetch_all_keywords()


def get_media_prompts(db: MediaDbLike, media_id: int) -> list[dict[str, Any]]:
    """Return prompt rows for a media item."""
    return legacy_reads.get_media_prompts(db_instance=db, media_id=media_id)


def get_media_transcripts(db: MediaDbLike | MediaDbReadLike, media_id: int) -> list[dict[str, Any]]:
    """Return transcript rows for a media item."""
    db_instance = unwrap_media_database_like(db)
    return legacy_reads.get_media_transcripts(db_instance=db_instance, media_id=media_id)


__all__ = [
    "MediaDbFactory",
    "MediaDbRuntimeConfig",
    "MediaDbSession",
    "MediaDbReadLike",
    "MediaWriterLike",
    "MediaRepository",
    "create_media_database",
    "create_document_version",
    "get_document_version",
    "get_all_document_versions",
    "get_full_media_details",
    "get_full_media_details_rich",
    "get_latest_transcription",
    "get_distinct_media_types",
    "get_media_by_hash",
    "get_media_prompts",
    "get_media_by_title",
    "get_media_transcripts",
    "get_media_by_url",
    "has_unvectorized_chunks",
    "get_unvectorized_anchor_index_for_offset",
    "get_unvectorized_chunk_by_index",
    "get_unvectorized_chunk_count",
    "get_unvectorized_chunk_index_by_uuid",
    "get_unvectorized_chunks_in_range",
    "get_paginated_files",
    "get_paginated_trash_files",
    "list_chunking_templates",
    "seed_builtin_templates",
    "lookup_section_for_offset",
    "lookup_section_by_heading",
    "fetch_keywords_for_media",
    "fetch_keywords_for_media_batch",
    "fetch_all_keywords",
    "check_media_exists",
    "permanently_delete_item",
    "get_media_by_id",
    "get_media_by_uuid",
    "managed_media_database",
    "get_media_repository",
    "list_document_versions",
    "search_media",
    "soft_delete_document_version",
    "soft_delete_keyword",
    "update_keywords_for_media",
]
