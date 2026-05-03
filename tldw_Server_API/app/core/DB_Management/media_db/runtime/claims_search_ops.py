"""Package-owned claims search helper."""

from __future__ import annotations

import sqlite3
from contextlib import suppress
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.backends.fts_translator import (
    FTSQueryTranslator,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)
from tldw_Server_API.app.core.DB_Management.scope_context import get_scope


_MEDIA_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = MEDIA_NONCRITICAL_EXCEPTIONS


def _append_scope_filters(
    conditions: list[str],
    params: list[Any],
    *,
    scope,
) -> None:
    if not scope or scope.is_admin:
        return

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
        conditions.append("(" + " OR ".join(visibility_parts) + ")")


def _count_from_row(row: Any) -> int:
    if row is None:
        return 0
    try:
        value = row["total"]
    except (KeyError, TypeError, IndexError):
        value = row[0]
    return int(value if value is not None else 0)


def search_claims(
    self,
    query: str,
    *,
    limit: int = 20,
    offset: int = 0,
    fallback_to_like: bool = True,
    owner_user_id: int | None = None,
    include_total: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], int]:
    """Search claims using the configured backend."""
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        return ([], 0) if include_total else []

    try:
        limit = max(1, int(limit))
    except (TypeError, ValueError):
        limit = 20
    try:
        offset = max(0, int(offset))
    except (TypeError, ValueError):
        offset = 0

    results: list[dict[str, Any]] = []
    total_matches = 0
    try:
        scope = get_scope()
    except _MEDIA_NONCRITICAL_EXCEPTIONS as scope_err:
        logger.debug("Failed to resolve scope for claims search: {}", scope_err)
        scope = None

    try:
        with self.transaction() as conn:
            if self.backend_type == BackendType.SQLITE:
                with suppress(sqlite3.Error):
                    conn.execute("INSERT INTO claims_fts(claims_fts) VALUES('rebuild')")

                conditions: list[str] = ["c.deleted = 0"]
                params: list[Any] = []
                if owner_user_id is not None:
                    conditions.append("COALESCE(CAST(m.owner_user_id AS TEXT), m.client_id) = ?")
                    params.append(str(owner_user_id))
                _append_scope_filters(conditions, params, scope=scope)

                sql = (
                    "SELECT c.id, c.media_id, c.chunk_index, c.claim_text, c.claim_cluster_id, "  # nosec B608
                    "       bm25(claims_fts) AS relevance_score "
                    "FROM claims_fts JOIN Claims c ON claims_fts.rowid = c.id "
                    "JOIN Media m ON c.media_id = m.id "
                    f"WHERE claims_fts MATCH ? AND {' AND '.join(conditions)} "
                    "ORDER BY relevance_score ASC LIMIT ? OFFSET ?"
                )
                count_sql = (
                    "SELECT COUNT(*) AS total "  # nosec B608
                    "FROM claims_fts JOIN Claims c ON claims_fts.rowid = c.id "
                    "JOIN Media m ON c.media_id = m.id "
                    f"WHERE claims_fts MATCH ? AND {' AND '.join(conditions)}"
                )
                if include_total:
                    total_matches = _count_from_row(
                        self._fetchone_with_connection(conn, count_sql, (cleaned_query, *params))
                    )
                rows = self._fetchall_with_connection(conn, sql, (cleaned_query, *params, limit, offset))
                results.extend(dict(row) for row in rows)
            elif self.backend_type == BackendType.POSTGRESQL:
                tsquery = FTSQueryTranslator.normalize_query(cleaned_query, "postgresql")
                if tsquery:
                    conditions = ["c.deleted IS FALSE"]
                    params = []
                    if owner_user_id is not None:
                        conditions.append("COALESCE(CAST(m.owner_user_id AS TEXT), m.client_id) = ?")
                        params.append(str(owner_user_id))
                    _append_scope_filters(conditions, params, scope=scope)

                    sql = (
                        "SELECT c.id, c.media_id, c.chunk_index, c.claim_text, c.claim_cluster_id, "  # nosec B608
                        "       ts_rank(c.claims_fts_tsv, to_tsquery('english', ?)) AS relevance_score "
                        "FROM claims c JOIN media m ON c.media_id = m.id "
                        f"WHERE c.claims_fts_tsv @@ to_tsquery('english', ?) AND {' AND '.join(conditions)} "
                        "ORDER BY relevance_score DESC LIMIT ? OFFSET ?"
                    )
                    count_sql = (
                        "SELECT COUNT(*) AS total "  # nosec B608
                        "FROM claims c JOIN media m ON c.media_id = m.id "
                        f"WHERE c.claims_fts_tsv @@ to_tsquery('english', ?) AND {' AND '.join(conditions)}"
                    )
                    if include_total:
                        total_matches = _count_from_row(
                            self._fetchone_with_connection(conn, count_sql, (tsquery, *params))
                        )
                    rows = self._fetchall_with_connection(conn, sql, (tsquery, tsquery, *params, limit, offset))
                    results.extend(dict(row) for row in rows)
            else:
                raise NotImplementedError(
                    f"Claims search not implemented for backend {self.backend_type}"
                )

            has_fts_matches = total_matches > 0 if include_total else bool(results)
            if fallback_to_like and not has_fts_matches:
                like_conditions: list[str] = []
                like_params: list[Any] = []
                if owner_user_id is not None:
                    like_conditions.append("COALESCE(CAST(m.owner_user_id AS TEXT), m.client_id) = ?")
                    like_params.append(str(owner_user_id))
                _append_scope_filters(like_conditions, like_params, scope=scope)

                like_clause = " AND " + " AND ".join(like_conditions) if like_conditions else ""
                like_pattern = f"%{cleaned_query}%"
                if self.backend_type == BackendType.POSTGRESQL:
                    like_sql = (
                        "SELECT c.id, c.media_id, c.chunk_index, c.claim_text, c.claim_cluster_id "  # nosec B608
                        "FROM claims c JOIN media m ON c.media_id = m.id "
                        "WHERE c.deleted IS FALSE AND c.claim_text ILIKE ?"
                        + like_clause
                        + " LIMIT ? OFFSET ?"
                    )
                    like_count_sql = (
                        "SELECT COUNT(*) AS total "  # nosec B608
                        "FROM claims c JOIN media m ON c.media_id = m.id "
                        "WHERE c.deleted IS FALSE AND c.claim_text ILIKE ?"
                        + like_clause
                    )
                else:
                    like_sql = (
                        "SELECT c.id, c.media_id, c.chunk_index, c.claim_text, c.claim_cluster_id "  # nosec B608
                        "FROM Claims c JOIN Media m ON c.media_id = m.id "
                        "WHERE c.deleted = 0 AND c.claim_text LIKE ?"
                        + like_clause
                        + " LIMIT ? OFFSET ?"
                    )
                    like_count_sql = (
                        "SELECT COUNT(*) AS total "  # nosec B608
                        "FROM Claims c JOIN Media m ON c.media_id = m.id "
                        "WHERE c.deleted = 0 AND c.claim_text LIKE ?"
                        + like_clause
                    )
                if include_total:
                    total_matches = _count_from_row(
                        self._fetchone_with_connection(conn, like_count_sql, (like_pattern, *like_params))
                    )
                fallback_rows = self._fetchall_with_connection(
                    conn,
                    like_sql,
                    (like_pattern, *like_params, limit, offset),
                )
                for row in fallback_rows:
                    row_dict = dict(row)
                    row_dict.setdefault("relevance_score", 0.0)
                    results.append(row_dict)
    except _MEDIA_NONCRITICAL_EXCEPTIONS as exc:
        logger.exception("Failed to search claims: {}", exc)
        return ([], 0) if include_total else []

    return (results, total_matches) if include_total else results
