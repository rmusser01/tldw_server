from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Sequence
from datetime import datetime
from math import isfinite
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.backends.fts_translator import (
    FTSQueryTranslator,
)
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.runtime.validation import (
    MediaDbLike,
    require_media_database_like,
)
from tldw_Server_API.app.core.DB_Management.scope_context import get_scope


def _append_case_insensitive_like(
    backend_type: BackendType,
    clauses: list[str],
    params: list[Any],
    column: str,
    pattern: str,
) -> None:
    if backend_type == BackendType.POSTGRESQL:
        clauses.append(f"{column} ILIKE ?")
    else:
        clauses.append(f"{column} LIKE ? COLLATE NOCASE")
    params.append(pattern)


def _parse_safe_metadata(raw_value: Any) -> dict[str, Any] | None:
    """Normalize latest-version safe_metadata JSON for media search results."""
    if isinstance(raw_value, dict):
        return raw_value
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None
    try:
        parsed = json.loads(raw_value)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


LATEST_SOURCE_METADATA_JOIN = """
LEFT JOIN (
    SELECT media_id, safe_metadata
    FROM (
        SELECT
            dv.media_id,
            dv.safe_metadata,
            ROW_NUMBER() OVER (
                PARTITION BY dv.media_id
                ORDER BY dv.version_number DESC, dv.id DESC
            ) AS row_number
        FROM DocumentVersions dv
        WHERE dv.deleted = 0
    ) latest_document_versions
    WHERE row_number = 1
) latest_source_metadata ON latest_source_metadata.media_id = m.id
"""


class MediaSearchRepository:
    """Repository for canonical media search reads."""

    def __init__(self, session: MediaDbLike):
        self.session = session

    @classmethod
    def from_legacy_db(cls, db: MediaDbLike) -> "MediaSearchRepository":
        return cls(
            session=require_media_database_like(
                db,
                error_message="db_instance must expose the Media DB core contract.",
            )
        )

    def search(
        self,
        search_query: str | None,
        search_fields: list[str] | None = None,
        media_types: list[str] | None = None,
        date_range: dict[str, datetime] | None = None,
        must_have_keywords: list[str] | None = None,
        must_not_have_keywords: list[str] | None = None,
        sort_by: str | None = "last_modified_desc",
        boost_fields: dict[str, float] | None = None,
        media_ids_filter: list[int | str] | None = None,
        page: int = 1,
        results_per_page: int = 20,
        include_trash: bool = False,
        include_deleted: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        db = self.session
        backend_type = getattr(db, "backend_type", None)
        if backend_type not in (BackendType.SQLITE, BackendType.POSTGRESQL):
            raise TypeError("db must expose a supported backend_type for search operations.")  # noqa: TRY003

        if page < 1:
            raise ValueError("Page number must be 1 or greater")  # noqa: TRY003
        if results_per_page < 1:
            raise ValueError("Results per page must be 1 or greater")  # noqa: TRY003

        if search_query and not search_fields:
            search_fields = ["title", "content"]
        elif not search_fields:
            search_fields = []

        valid_text_search_fields = {"title", "content", "author", "type"}
        sanitized_text_search_fields = [field for field in search_fields if field in valid_text_search_fields]
        supplied_boost_fields = boost_fields if isinstance(boost_fields, dict) else None
        boost_fields_supplied = bool(supplied_boost_fields)

        def _sanitize_field_boost(field_name: str, default_value: float = 1.0) -> float:
            if not supplied_boost_fields:
                return default_value
            raw_value = supplied_boost_fields.get(field_name, default_value)
            try:
                parsed_value = float(raw_value)
            except (TypeError, ValueError):
                logger.debug(
                    "Invalid boost_fields value for '{}': {}. Using default {}.",
                    field_name,
                    raw_value,
                    default_value,
                )
                return default_value
            if not isfinite(parsed_value):
                logger.debug(
                    "Non-finite boost_fields value for '{}': {}. Using default {}.",
                    field_name,
                    raw_value,
                    default_value,
                )
                return default_value
            return max(0.05, min(50.0, parsed_value))

        title_boost = _sanitize_field_boost("title")
        content_boost = _sanitize_field_boost("content")

        if isinstance(search_query, str):
            try:
                max_chars = int((os.getenv("FTS_QUERY_MAX_CHARS") or "1000").strip() or 1000)
            except (TypeError, ValueError):
                max_chars = 1000
            if len(search_query) > max_chars:
                logger.warning(
                    "Clamping search_query from {} to {} chars for FTS hardening",
                    len(search_query),
                    max_chars,
                )
                search_query = search_query[:max_chars]

        offset = (page - 1) * results_per_page
        base_select_parts = [
            "m.id",
            "m.uuid",
            "m.url",
            "m.title",
            "m.content",
            "m.type",
            "m.author",
            "m.ingestion_date",
            "m.transcription_model",
            "m.is_trash",
            "m.trash_date",
            "m.chunking_status",
            "m.vector_processing",
            "m.content_hash",
            "m.last_modified",
            "m.version",
            "m.client_id",
            "m.deleted",
            "latest_source_metadata.safe_metadata AS safe_metadata",
        ]
        count_select = "COUNT(DISTINCT m.id)"
        base_from = "FROM Media m"
        joins: list[str] = []
        conditions: list[str] = []
        params: list[Any] = []
        fts_condition_index: int | None = None
        fts_param_index: int | None = None
        fts_relevance_added = False

        fts_select_params: list[Any] = []
        fts_condition_params: list[Any] = []
        postgres_tsquery: str | None = None

        def _is_sqlite_fts_query_error(err: Exception) -> bool:
            if backend_type != BackendType.SQLITE or fts_condition_index is None:
                return False
            msg = str(err).lower()
            return (
                "unable to use function match" in msg
                or "no such column" in msg
                or "no such table: media_fts" in msg
                or "fts5: syntax error" in msg
                or ("malformed" in msg and "match" in msg)
            )

        if not include_deleted:
            conditions.append("m.deleted = 0")
        if not include_trash:
            conditions.append("m.is_trash = 0")

        if backend_type == BackendType.SQLITE:
            try:
                scope = get_scope()
            except Exception as scope_err:  # pragma: no cover - defensive
                logger.debug(
                    "Failed to resolve scope for SQLite visibility filter; falling back to no scope: {}",
                    scope_err,
                )
                scope = None

            if scope and not scope.is_admin:
                visibility_parts: list[str] = []
                user_id_str = str(scope.user_id) if scope.user_id is not None else ""
                if user_id_str:
                    visibility_parts.append(
                        "(COALESCE(m.visibility, 'personal') = 'personal' "
                        "AND (COALESCE(CAST(m.owner_user_id AS TEXT), m.client_id) = ?))"
                    )
                    params.append(user_id_str)
                if scope.team_ids:
                    team_placeholders = ",".join("?" * len(scope.team_ids))
                    visibility_parts.append(
                        f"(m.visibility = 'team' AND m.team_id IN ({team_placeholders}))"
                    )
                    params.extend(scope.team_ids)
                if scope.org_ids:
                    org_placeholders = ",".join("?" * len(scope.org_ids))
                    visibility_parts.append(
                        f"(m.visibility = 'org' AND m.org_id IN ({org_placeholders}))"
                    )
                    params.extend(scope.org_ids)

                if visibility_parts:
                    conditions.append(f"({' OR '.join(visibility_parts)})")
                else:
                    conditions.append("(0 = 1)")

        if media_ids_filter:
            if not all(isinstance(media_id, (int, str)) for media_id in media_ids_filter):
                raise ValueError("media_ids_filter must be a list of ints or strings.")  # noqa: TRY003
            int_ids = [media_id for media_id in media_ids_filter if isinstance(media_id, int)]
            uuid_ids = [media_id for media_id in media_ids_filter if isinstance(media_id, str) and media_id]
            if int_ids:
                id_placeholders = ",".join("?" * len(int_ids))
                conditions.append(f"m.id IN ({id_placeholders})")
                params.extend(int_ids)
            if uuid_ids:
                uuid_placeholders = ",".join("?" * len(uuid_ids))
                conditions.append(f"m.uuid IN ({uuid_placeholders})")
                params.extend(uuid_ids)

        if media_types:
            if not all(isinstance(media_type, str) for media_type in media_types):
                raise ValueError("media_types must be a list of strings.")  # noqa: TRY003
            type_placeholders = ",".join("?" * len(media_types))
            conditions.append(f"m.type IN ({type_placeholders})")
            params.extend(media_types)

        if date_range:
            start_date = date_range.get("start_date")
            end_date = date_range.get("end_date")
            if start_date:
                if not isinstance(start_date, datetime):
                    raise ValueError("date_range['start_date'] must be a datetime object.")  # noqa: TRY003
                conditions.append("m.ingestion_date >= ?")
                params.append(start_date.isoformat())
            if end_date:
                if not isinstance(end_date, datetime):
                    raise ValueError("date_range['end_date'] must be a datetime object.")  # noqa: TRY003
                conditions.append("m.ingestion_date <= ?")
                params.append(end_date.isoformat())

        cleaned_must_have = [
            keyword.strip().lower()
            for keyword in (must_have_keywords or [])
            if keyword and keyword.strip()
        ]
        if cleaned_must_have:
            kw_mh_placeholders = ",".join("?" * len(cleaned_must_have))
            must_have_condition = (
                "(SELECT COUNT(DISTINCT k_mh.id) "
                "FROM MediaKeywords mk_mh "
                "JOIN Keywords k_mh ON mk_mh.keyword_id = k_mh.id "
                "WHERE mk_mh.media_id = m.id AND k_mh.deleted = 0 "
                f"AND LOWER(k_mh.keyword) IN ({kw_mh_placeholders})) = ?"  # nosec B608
            )
            conditions.append(must_have_condition)
            params.extend(cleaned_must_have)
            params.append(len(cleaned_must_have))

        cleaned_must_not_have = [
            keyword.strip().lower()
            for keyword in (must_not_have_keywords or [])
            if keyword and keyword.strip()
        ]
        if cleaned_must_not_have:
            kw_mnh_placeholders = ",".join("?" * len(cleaned_must_not_have))
            must_not_have_condition = (
                "NOT EXISTS ("
                "SELECT 1 "
                "FROM MediaKeywords mk_mnh "
                "JOIN Keywords k_mnh ON mk_mnh.keyword_id = k_mnh.id "
                "WHERE mk_mnh.media_id = m.id AND k_mnh.deleted = 0 "
                f"AND LOWER(k_mnh.keyword) IN ({kw_mnh_placeholders}))"  # nosec B608
            )
            conditions.append(must_not_have_condition)
            params.extend(cleaned_must_not_have)

        fts_search_active = False
        if search_query:
            like_conditions: list[str] = []
            like_params: list[Any] = []
            like_search_query = (
                search_query.strip('"')
                if search_query.startswith('"') and search_query.endswith('"')
                else search_query
            )

            if any(field in sanitized_text_search_fields for field in ["title", "content"]):
                fts_search_active = True
                fts_query_parts: list[str] = []
                if len(search_query) <= 2 and not (search_query.startswith('"') and search_query.endswith('"')):
                    fts_query_parts.append(f"{search_query}*")
                    if search_query.lower() != search_query:
                        fts_query_parts.append(f"{search_query.lower()}*")
                else:
                    fts_query_parts.append(search_query)
                    if not (search_query.startswith('"') and search_query.endswith('"')) and search_query.lower() != search_query:
                        fts_query_parts.append(search_query.lower())

                combined_fts_query = " OR ".join(fts_query_parts)
                logger.debug(f"Combined FTS query: '{combined_fts_query}'")
                logger.info(f"Search using FTS with query parts: {fts_query_parts}")

                if backend_type == BackendType.SQLITE:
                    if not any("media_fts fts" in join_item for join_item in joins):
                        joins.append("JOIN media_fts fts ON fts.rowid = m.id")
                    conditions.append("media_fts MATCH ?")
                    params.append(combined_fts_query)
                    fts_condition_index = len(conditions) - 1
                    fts_param_index = len(params) - 1
                elif backend_type == BackendType.POSTGRESQL:
                    postgres_tsquery = FTSQueryTranslator.normalize_query(
                        combined_fts_query,
                        "postgresql",
                    )
                    if postgres_tsquery:
                        conditions.append("m.media_fts_tsv @@ to_tsquery('english', ?)")
                        fts_condition_params.append(postgres_tsquery)
                        fts_condition_index = len(conditions) - 1
                    else:
                        logger.debug(
                            "PostgreSQL tsquery normalization produced empty output; falling back to LIKE-only search."
                        )
                        fts_search_active = False
                else:  # pragma: no cover - defensive
                    logger.warning("FTS requested for unsupported backend {}", backend_type)
                    fts_search_active = False

                if not fts_search_active:
                    title_content_like_parts: list[str] = []
                    for field in ["title", "content"]:
                        if field in sanitized_text_search_fields:
                            column = f"m.{field}"
                            _append_case_insensitive_like(
                                backend_type,
                                title_content_like_parts,
                                like_params,
                                column,
                                f"%{like_search_query}%",
                            )
                            if len(like_search_query) <= 2 and not (search_query.startswith('"') and search_query.endswith('"')):
                                _append_case_insensitive_like(
                                    backend_type,
                                    title_content_like_parts,
                                    like_params,
                                    column,
                                    f"%{like_search_query}",
                                )
                    if title_content_like_parts:
                        like_conditions.append(f"({' OR '.join(title_content_like_parts)})")

            like_fields_to_search = [field for field in sanitized_text_search_fields if field in ["author", "type"]]
            if like_fields_to_search:
                like_parts: list[str] = []
                for field in like_fields_to_search:
                    if field == "type" and media_types:
                        logger.debug("LIKE search on 'type' skipped due to active 'media_types' filter.")
                        continue
                    _append_case_insensitive_like(
                        backend_type,
                        like_parts,
                        like_params,
                        f"m.{field}",
                        f"%{like_search_query}%",
                    )
                    if len(like_search_query) <= 2 and not (search_query.startswith('"') and search_query.endswith('"')):
                        _append_case_insensitive_like(
                            backend_type,
                            like_parts,
                            like_params,
                            f"m.{field}",
                            f"%{like_search_query}",
                        )
                if like_parts:
                    like_conditions.append(f"({' OR '.join(like_parts)})")

            if like_conditions:
                logger.info(f"Search using LIKE with patterns: {like_params}")
                conditions.append(f"({' OR '.join(like_conditions)})")
                params.extend(like_params)
        elif sanitized_text_search_fields:
            conditions.append("1=1")

        order_by_clause_str = ""
        default_order_by = "ORDER BY m.last_modified DESC, m.id DESC"
        if fts_search_active and (sort_by == "relevance" or not sort_by):
            if backend_type == BackendType.SQLITE:
                if not any("AS relevance_score" in part for part in base_select_parts):
                    if boost_fields_supplied:
                        base_select_parts.append(
                            f"bm25(media_fts, {title_boost:.6f}, {content_boost:.6f}) AS relevance_score"
                        )
                    else:
                        base_select_parts.append("bm25(media_fts) AS relevance_score")
                    fts_relevance_added = True
                order_by_clause_str = "ORDER BY relevance_score ASC, m.last_modified DESC, m.id DESC"
            elif backend_type == BackendType.POSTGRESQL and postgres_tsquery:
                if not any("relevance_score" in part for part in base_select_parts):
                    if boost_fields_supplied:
                        postgres_weights_literal = (
                            f"{content_boost:.6f},1.000000,{content_boost:.6f},{title_boost:.6f}"
                        )
                        base_select_parts.append(
                            "ts_rank("
                            f"ARRAY[{postgres_weights_literal}]::float4[], "
                            "m.media_fts_tsv, to_tsquery('english', ?)"
                            ") AS relevance_score"
                        )
                    else:
                        base_select_parts.append(
                            "ts_rank(m.media_fts_tsv, to_tsquery('english', ?)) AS relevance_score"
                        )
                    fts_select_params.append(postgres_tsquery)
                    fts_relevance_added = True
                order_by_clause_str = "ORDER BY relevance_score DESC, m.last_modified DESC, m.id DESC"
        else:
            if sort_by == "date_desc":
                order_by_clause_str = "ORDER BY m.ingestion_date DESC, m.last_modified DESC, m.id DESC"
            elif sort_by == "date_asc":
                order_by_clause_str = "ORDER BY m.ingestion_date ASC, m.last_modified ASC, m.id ASC"
            elif sort_by == "title_asc":
                if backend_type == BackendType.POSTGRESQL:
                    order_by_clause_str = "ORDER BY LOWER(m.title) ASC, m.title ASC, m.id ASC"
                else:
                    order_by_clause_str = "ORDER BY m.title ASC COLLATE NOCASE, m.id ASC"
            elif sort_by == "title_desc":
                if backend_type == BackendType.POSTGRESQL:
                    order_by_clause_str = "ORDER BY LOWER(m.title) DESC, m.title DESC, m.id DESC"
                else:
                    order_by_clause_str = "ORDER BY m.title DESC COLLATE NOCASE, m.id DESC"
            elif sort_by == "last_modified_asc":
                order_by_clause_str = "ORDER BY m.last_modified ASC, m.id ASC"
            else:
                order_by_clause_str = default_order_by

        final_select_stmt = f"SELECT DISTINCT {', '.join(base_select_parts)}"
        join_clause = " ".join(list(dict.fromkeys(joins)))
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

        def _extract_total(row: Any) -> int:
            if not row:
                return 0
            if isinstance(row, dict):
                value = (
                    row.get("count")
                    or row.get("total")
                    or next(iter(row.values()))
                    or 0
                )
                return int(value)
            return int(row[0])

        try:
            count_sql = f"SELECT {count_select} {base_from} {join_clause} {where_clause}"
            logger.debug(f"Search Count SQL ({db.db_path_str}): {count_sql}")
            count_params_seq: Sequence[Any]
            if backend_type == BackendType.POSTGRESQL:
                count_params_seq = list(fts_condition_params) + list(params)
            else:
                count_params_seq = list(params)
            logger.debug(f"Search Count Params: {count_params_seq}")

            try:
                count_cursor = db.execute_query(count_sql, tuple(count_params_seq))
                total_matches = _extract_total(count_cursor.fetchone())
                logger.info(f"Search query '{search_query}' found {total_matches} total matches")
            except (sqlite3.OperationalError, DatabaseError) as exc:
                if not _is_sqlite_fts_query_error(exc):
                    raise

                logger.warning(f"FTS MATCH error, falling back to LIKE-only search: {exc}")
                fallback_conditions = [
                    condition
                    for idx, condition in enumerate(conditions)
                    if idx != fts_condition_index
                ]
                fallback_params = list(params)
                if fts_param_index is not None and 0 <= fts_param_index < len(fallback_params):
                    fallback_params.pop(fts_param_index)
                fallback_joins = [join for join in joins if "media_fts fts" not in join]

                if not fallback_conditions and not fallback_params and not fallback_joins and not search_query:
                    logger.warning("No valid search conditions after removing FTS MATCH, returning empty results")
                    return [], 0

                if fts_relevance_added:
                    base_select_parts[:] = [
                        part for part in base_select_parts if "relevance_score" not in part
                    ]
                    order_by_clause_str = default_order_by
                    fts_relevance_added = False
                final_select_stmt = f"SELECT DISTINCT {', '.join(base_select_parts)}"

                conditions[:] = fallback_conditions
                params[:] = fallback_params
                joins[:] = fallback_joins
                fts_condition_index = None
                fts_param_index = None
                join_clause = " ".join(list(dict.fromkeys(joins)))
                where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
                count_sql = f"SELECT {count_select} {base_from} {join_clause} {where_clause}"
                logger.debug(f"Fallback Count SQL ({db.db_path_str}): {count_sql}")
                logger.debug(f"Fallback Count Params: {params}")
                count_cursor = db.execute_query(count_sql, tuple(params))
                total_matches = _extract_total(count_cursor.fetchone())
                logger.info(f"Fallback search query '{search_query}' found {total_matches} total matches")

            results_list: list[dict[str, Any]] = []
            if total_matches > 0 and offset < total_matches:
                results_join_clause = " ".join(
                    part for part in (join_clause, LATEST_SOURCE_METADATA_JOIN.strip()) if part
                )
                results_sql = (
                    f"{final_select_stmt} {base_from} {results_join_clause} {where_clause} "
                    f"{order_by_clause_str} LIMIT ? OFFSET ?"
                )
                if backend_type == BackendType.POSTGRESQL:
                    paginated_params = tuple(
                        list(fts_select_params)
                        + list(fts_condition_params)
                        + list(params)
                        + [results_per_page, offset]
                    )
                else:
                    paginated_params = tuple(list(params) + [results_per_page, offset])
                logger.debug(f"Search Results SQL ({db.db_path_str}): {results_sql}")
                logger.debug(f"Search Results Params: {paginated_params}")

                try:
                    results_cursor = db.execute_query(results_sql, paginated_params)
                    results_list = []
                    for row in results_cursor.fetchall():
                        item = dict(row)
                        item["safe_metadata"] = _parse_safe_metadata(item.get("safe_metadata"))
                        results_list.append(item)
                except (sqlite3.OperationalError, DatabaseError) as exc:
                    if not _is_sqlite_fts_query_error(exc):
                        raise

                    logger.warning(f"FTS MATCH error in results query, falling back to LIKE-only search: {exc}")
                    fallback_conditions = [
                        condition
                        for idx, condition in enumerate(conditions)
                        if idx != fts_condition_index
                    ]
                    fallback_params = list(params)
                    if fts_param_index is not None and 0 <= fts_param_index < len(fallback_params):
                        fallback_params.pop(fts_param_index)
                    fallback_joins = [join for join in joins if "media_fts fts" not in join]

                    if fts_relevance_added:
                        base_select_parts[:] = [
                            part for part in base_select_parts if "relevance_score" not in part
                        ]
                        order_by_clause_str = default_order_by
                        fts_relevance_added = False

                    conditions[:] = fallback_conditions
                    params[:] = fallback_params
                    joins[:] = fallback_joins
                    fts_condition_index = None
                    fts_param_index = None

                    join_clause = " ".join(list(dict.fromkeys(joins)))
                    results_join_clause = " ".join(
                        part for part in (join_clause, LATEST_SOURCE_METADATA_JOIN.strip()) if part
                    )
                    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
                    final_select_stmt = f"SELECT DISTINCT {', '.join(base_select_parts)}"
                    results_sql = (
                        f"{final_select_stmt} {base_from} {results_join_clause} {where_clause} "
                        f"{order_by_clause_str} LIMIT ? OFFSET ?"
                    )
                    paginated_params = tuple(list(params) + [results_per_page, offset])
                    logger.debug(f"Fallback Results SQL ({db.db_path_str}): {results_sql}")
                    logger.debug(f"Fallback Results Params: {paginated_params}")
                    results_cursor = db.execute_query(results_sql, paginated_params)
                    results_list = []
                    for row in results_cursor.fetchall():
                        item = dict(row)
                        item["safe_metadata"] = _parse_safe_metadata(item.get("safe_metadata"))
                        results_list.append(item)

                titles = [row.get("title", "Untitled") for row in results_list]
                logger.info(f"Search results for '{search_query}' (page {page}): {titles}")

            return results_list, total_matches
        except sqlite3.Error as exc:
            if "no such table: media_fts" in str(exc).lower():
                logger.exception(
                    f"FTS table 'media_fts' missing in database '{db.db_path_str}'. Search will fail."
                )
                raise DatabaseError(f"FTS table 'media_fts' not found in {db.db_path_str}.") from exc  # noqa: TRY003
            logger.error(
                f"Database error during media search in '{db.db_path_str}': {exc}",
                exc_info=True,
            )
            raise DatabaseError(f"Failed to search media in {db.db_path_str}: {exc}") from exc  # noqa: TRY003
        except Exception as exc:
            logger.error(
                f"Unexpected error during media search in '{db.db_path_str}': {exc}",
                exc_info=True,
            )
            raise DatabaseError(f"An unexpected error occurred during media search: {exc}") from exc  # noqa: TRY003


__all__ = ["MediaSearchRepository"]
