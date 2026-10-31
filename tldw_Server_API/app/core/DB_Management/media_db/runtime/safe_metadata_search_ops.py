"""Package-owned helper for safe-metadata search queries."""

from __future__ import annotations

from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)

_MEDIA_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = MEDIA_NONCRITICAL_EXCEPTIONS


def search_by_safe_metadata(
    self: Any,
    filters: list[dict[str, Any]] | None = None,
    match_all: bool = True,
    page: int = 1,
    per_page: int = 20,
    group_by_media: bool = True,
    text_query: str | None = None,
    media_types: list[str] | None = None,
    must_have_keywords: list[str] | None = None,
    must_not_have_keywords: list[str] | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
    sort_by: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Search by fields inside safe metadata and identifier projections."""
    try:
        offset = (max(1, page) - 1) * per_page
        clauses: list[str] = [
            "dv.deleted = 0",
            "m.deleted = 0",
            "m.system_operation_id IS NULL",
        ]
        params: list[Any] = []
        join_ident = False

        id_fields = {"doi", "pmid", "pmcid", "arxiv_id", "s2_paper_id"}
        ops_sql = {
            "eq": lambda col: (f"{col} = ?", lambda v: v),
            "contains": lambda col: (f"{col} LIKE ?", lambda v: f"%{v}%"),
            "icontains": lambda col: (
                f"LOWER({col}) LIKE ?",
                lambda v: f"%{str(v).lower()}%",
            ),
            "startswith": lambda col: (f"{col} LIKE ?", lambda v: f"{v}%"),
            "endswith": lambda col: (f"{col} LIKE ?", lambda v: f"%{v}"),
        }

        filter_exprs: list[str] = []
        if filters:
            for flt in filters:
                field = (flt.get("field") or "").strip()
                op = (flt.get("op") or "icontains").lower()
                val = flt.get("value")
                if not field or val is None:
                    continue
                if field in id_fields:
                    join_ident = True
                    col = f"dvi.{field}"
                    sql_tpl, xform = ops_sql.get(op, ops_sql["icontains"])(col)
                    filter_exprs.append(sql_tpl)
                    params.append(xform(val))
                else:
                    if op == "eq":
                        filter_exprs.append("(dv.safe_metadata LIKE ? OR dv.safe_metadata LIKE ?)")
                        params.extend(
                            [
                                f'%"{field}":"{val}"%',
                                f'%"{field}": "{val}"%',
                            ]
                        )
                    elif op == "icontains":
                        filter_exprs.append("LOWER(dv.safe_metadata) LIKE ?")
                        params.append(f"%{str(val).lower()}%")
                    else:
                        filter_exprs.append("dv.safe_metadata LIKE ?")
                        like_val = val
                        if op == "contains":
                            like_val = f"%{val}%"
                        elif op == "startswith":
                            like_val = f"{val}%"
                        elif op == "endswith":
                            like_val = f"%{val}"
                        else:
                            like_val = f"%{val}%"
                        params.append(like_val)

        if filter_exprs:
            join_op = " AND " if match_all else " OR "
            clauses.append("(" + join_op.join(filter_exprs) + ")")

        normalized_text_query = (text_query or "").strip()
        if normalized_text_query:
            clauses.append(
                "(LOWER(COALESCE(m.title, '')) LIKE ? OR LOWER(COALESCE(dv.safe_metadata, '')) LIKE ?)"
            )
            text_query_like = f"%{normalized_text_query.lower()}%"
            params.extend([text_query_like, text_query_like])

        normalized_media_types = [
            str(media_type).strip().lower()
            for media_type in (media_types or [])
            if str(media_type).strip()
        ]
        if normalized_media_types:
            media_type_placeholders = ",".join("?" * len(normalized_media_types))
            clauses.append(f"LOWER(m.type) IN ({media_type_placeholders})")
            params.extend(normalized_media_types)

        cleaned_must_have = [
            str(keyword).strip().lower()
            for keyword in (must_have_keywords or [])
            if str(keyword).strip()
        ]
        if cleaned_must_have:
            kw_mh_placeholders = ",".join("?" * len(cleaned_must_have))
            clauses.append(
                """
            (SELECT COUNT(DISTINCT k_mh.id)
             FROM MediaKeywords mk_mh
             JOIN Keywords k_mh ON mk_mh.keyword_id = k_mh.id
             WHERE mk_mh.media_id = m.id AND k_mh.deleted = 0 AND LOWER(k_mh.keyword) IN ({kw_mh_placeholders})
            ) = ?
        """.format_map(locals())  # nosec B608
            )
            params.extend(cleaned_must_have)
            params.append(len(cleaned_must_have))

        cleaned_must_not_have = [
            str(keyword).strip().lower()
            for keyword in (must_not_have_keywords or [])
            if str(keyword).strip()
        ]
        if cleaned_must_not_have:
            kw_mnh_placeholders = ",".join("?" * len(cleaned_must_not_have))
            clauses.append(
                """
            NOT EXISTS (
                SELECT 1
                FROM MediaKeywords mk_mnh
                JOIN Keywords k_mnh ON mk_mnh.keyword_id = k_mnh.id
                WHERE mk_mnh.media_id = m.id AND k_mnh.deleted = 0 AND LOWER(k_mnh.keyword) IN ({kw_mnh_placeholders})
            )
        """.format_map(locals())  # nosec B608
            )
            params.extend(cleaned_must_not_have)

        normalized_date_start = (date_start or "").strip()
        if normalized_date_start:
            clauses.append("dv.created_at >= ?")
            params.append(normalized_date_start)

        normalized_date_end = (date_end or "").strip()
        if normalized_date_end:
            clauses.append("dv.created_at <= ?")
            params.append(normalized_date_end)

        base_from = "FROM DocumentVersions dv JOIN Media m ON dv.media_id = m.id"
        if join_ident:
            base_from += " LEFT JOIN DocumentVersionIdentifiers dvi ON dvi.dv_id = dv.id"

        if group_by_media:
            count_sql = (
                f"SELECT COUNT(DISTINCT m.id) AS total_count {base_from} "
                f"WHERE {' AND '.join(clauses)}"
            )
        else:
            count_sql = (
                f"SELECT COUNT(*) AS total_count {base_from} "
                f"WHERE {' AND '.join(clauses)}"
            )
        count_cursor = self.execute_query(count_sql, tuple(params))
        count_row = count_cursor.fetchone()
        total = count_row["total_count"] if count_row else 0

        if total == 0:
            return [], 0

        select_cols = (
            "m.id AS media_id, m.title, m.type, dv.version_number, dv.created_at, dv.safe_metadata"
        )
        normalized_sort_by = (sort_by or "").strip().lower()
        if normalized_sort_by == "date_desc":
            order_clause = "ORDER BY dv.created_at DESC, m.id DESC"
        elif normalized_sort_by == "date_asc":
            order_clause = "ORDER BY dv.created_at ASC, m.id ASC"
        elif normalized_sort_by == "title_asc":
            if self.backend_type == BackendType.POSTGRESQL:
                order_clause = "ORDER BY LOWER(m.title) ASC, m.title ASC, m.id ASC"
            else:
                order_clause = "ORDER BY m.title COLLATE NOCASE ASC, m.id ASC"
        elif normalized_sort_by == "title_desc":
            if self.backend_type == BackendType.POSTGRESQL:
                order_clause = "ORDER BY LOWER(m.title) DESC, m.title DESC, m.id DESC"
            else:
                order_clause = "ORDER BY m.title COLLATE NOCASE DESC, m.id DESC"
        else:
            order_clause = "ORDER BY m.last_modified DESC, m.id DESC"

        if group_by_media:
            results_sql = f"""
                SELECT {select_cols}
                {base_from}
                WHERE {' AND '.join(clauses)}
                GROUP BY m.id
                {order_clause}
                LIMIT ? OFFSET ?
            """
            res_params = tuple(params + [per_page, offset])
        else:
            results_sql = f"""
                SELECT {select_cols}
                {base_from}
                WHERE {' AND '.join(clauses)}
                {order_clause}
                LIMIT ? OFFSET ?
            """
            res_params = tuple(params + [per_page, offset])

        cur = self.execute_query(results_sql, res_params)
        rows = [dict(r) for r in cur.fetchall()]
    except _MEDIA_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(f"Metadata search failed: {exc}", exc_info=True)
        raise DatabaseError(f"Failed metadata search: {exc}") from exc  # noqa: TRY003
    else:
        return rows, total


__all__ = ["search_by_safe_metadata"]
