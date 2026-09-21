"""Package-owned email read/query helpers."""

from __future__ import annotations

import json
import re
import shlex
import time
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.errors import InputError
from tldw_Server_API.app.core.DB_Management.media_db.runtime.email_search_cursor import (
    decode_email_cursor,
    email_cursor_scope,
    encode_email_cursor,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter, log_histogram

_MEDIA_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = MEDIA_NONCRITICAL_EXCEPTIONS
_EMAIL_WINDOW_RE = re.compile(r"(\d+)([smhdwy])")


def _coerce_email_metric_label(value: Any, *, default: str = "unknown") -> str:
    raw = str(value or "").strip().lower()
    return raw or default


def _emit_email_metric_counter(
    metric_name: str,
    *,
    labels: dict[str, Any] | None = None,
) -> None:
    with suppress(_MEDIA_NONCRITICAL_EXCEPTIONS):
        safe_labels = (
            {str(k): _coerce_email_metric_label(v, default="none") for k, v in labels.items()}
            if labels
            else None
        )
        log_counter(metric_name, labels=safe_labels)


def _emit_email_metric_histogram(
    metric_name: str,
    value: float,
    *,
    labels: dict[str, Any] | None = None,
) -> None:
    with suppress(_MEDIA_NONCRITICAL_EXCEPTIONS):
        safe_labels = (
            {str(k): _coerce_email_metric_label(v, default="none") for k, v in labels.items()}
            if labels
            else None
        )
        safe_value = max(0.0, float(value))
        log_histogram(metric_name, safe_value, labels=safe_labels)


def _parse_email_relative_window(value: str) -> timedelta | None:
    text = str(value or "").strip().lower()
    match = _EMAIL_WINDOW_RE.fullmatch(text)
    if not match:
        return None
    magnitude = int(match.group(1))
    unit = match.group(2)
    if magnitude <= 0:
        return None
    if unit == "s":
        return timedelta(seconds=magnitude)
    if unit == "m":
        return timedelta(minutes=magnitude)
    if unit == "h":
        return timedelta(hours=magnitude)
    if unit == "d":
        return timedelta(days=magnitude)
    if unit == "w":
        return timedelta(days=magnitude * 7)
    if unit == "y":
        return timedelta(days=magnitude * 365)
    return None


def _sqlite_fts_literal_term(value: str) -> str | None:
    """Return a safely quoted SQLite FTS5 literal term."""

    text = str(value or "").strip()
    if not text:
        return None
    escaped = text.replace('"', '""')
    return f'"{escaped}"'


def _parse_email_operator_query(
    self,
    query: str | None,
    *,
    as_of: datetime | None = None,
) -> list[list[dict[str, Any]]]:
    cleaned = str(query or "").strip()
    if not cleaned:
        return [[]]
    if "(" in cleaned or ")" in cleaned:
        raise InputError("Parentheses are not supported in email query v1.")  # noqa: TRY003
    try:
        tokens = shlex.split(cleaned)
    except ValueError as exc:
        raise InputError(f"Invalid email query syntax: {exc}") from exc  # noqa: TRY003
    if not tokens:
        return [[]]

    groups: list[list[dict[str, Any]]] = [[]]
    for token in tokens:
        raw_token = str(token or "").strip()
        if not raw_token:
            continue
        if raw_token.upper() == "OR":
            if not groups[-1]:
                raise InputError("Invalid email query: OR requires terms on both sides.")  # noqa: TRY003
            groups.append([])
            continue

        negated = raw_token.startswith("-")
        core = raw_token[1:] if negated else raw_token
        if not core:
            raise InputError("Invalid email query: empty negated token.")  # noqa: TRY003

        term: dict[str, Any] = {"kind": "text", "value": core, "negated": negated}
        field_name = ""
        field_value = ""
        if ":" in core:
            field_name, field_value = core.split(":", 1)
            field_name = field_name.strip().lower()
            field_value = field_value.strip()

        if field_name in {"from", "to", "cc", "bcc"} and field_value:
            term = {
                "kind": "participant",
                "role": field_name,
                "value": field_value,
                "negated": negated,
            }
        elif field_name == "subject" and field_value:
            term = {"kind": "subject", "value": field_value, "negated": negated}
        elif field_name == "label" and field_value:
            term = {"kind": "label", "value": field_value, "negated": negated}
        elif field_name == "has" and field_value.lower() == "attachment":
            term = {"kind": "has_attachment", "value": True, "negated": negated}
        elif field_name in {"before", "after"} and field_value:
            try:
                parsed_date = datetime.strptime(field_value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError as exc:
                raise InputError(
                    f"Invalid {field_name}: expected YYYY-MM-DD."
                ) from exc  # noqa: TRY003
            term = {
                "kind": field_name,
                "value": parsed_date.isoformat(),
                "negated": negated,
            }
        elif field_name in {"older_than", "newer_than"} and field_value:
            try:
                delta = _parse_email_relative_window(field_value)
                threshold = (
                    ((as_of or datetime.now(timezone.utc)) - delta).isoformat()
                    if delta is not None else None
                )
            except (ValueError, OverflowError) as exc:
                raise InputError("Email relative date is outside the supported range.") from exc
            if delta is None:
                raise InputError(
                    f"Invalid {field_name}: expected patterns like 7d, 12h, 30m."
                )  # noqa: TRY003
            term = {
                "kind": field_name,
                "value": threshold,
                "negated": negated,
            }
        elif field_name and not field_value:
            raise InputError(f"Invalid email query token '{raw_token}'.")  # noqa: TRY003

        groups[-1].append(term)

    if not groups or not groups[-1]:
        raise InputError("Invalid email query: dangling OR without trailing term.")  # noqa: TRY003
    return groups


def _email_like_clause(self, column_sql: str) -> str:
    return (
        f"{column_sql} ILIKE ?"
        if self.backend_type == BackendType.POSTGRESQL
        else f"{column_sql} LIKE ? COLLATE NOCASE"
    )


def search_email_messages(
    self,
    *,
    query: str | None = None,
    tenant_id: str | None = None,
    include_deleted: bool = False,
    limit: int = 50,
    offset: int = 0,
    cursor: str | None = None,
) -> tuple[list[dict[str, Any]], int] | tuple[list[dict[str, Any]], int, str | None]:
    """Search emails; omitted cursor returns (rows, total), otherwise adds next_cursor.

    An empty cursor starts keyset traversal. Cursors use descending date/ID order
    with missing dates last, and cannot be combined with a nonzero offset.
    """

    query_present = "true" if isinstance(query, str) and query.strip() else "false"
    include_deleted_label = "true" if include_deleted else "false"
    started_at = time.perf_counter()
    _emit_email_metric_counter(
        "email_native_search_requests_total",
        labels={
            "phase": "attempt",
            "query_present": query_present,
            "include_deleted": include_deleted_label,
        },
    )

    try:
        if cursor is not None and offset != 0:
            raise InputError("Email cursor pagination requires offset=0.")
        try:
            limit_int = max(1, min(500, int(limit)))
        except _MEDIA_NONCRITICAL_EXCEPTIONS:
            limit_int = 50
        try:
            offset_int = max(0, int(offset))
        except _MEDIA_NONCRITICAL_EXCEPTIONS:
            offset_int = 0

        resolved_tenant = self._resolve_email_tenant_id(tenant_id)
        scope = email_cursor_scope(resolved_tenant, query, include_deleted)
        as_of = datetime.now(timezone.utc)
        position = None
        if cursor is not None and cursor != "":
            as_of, date, message_id = decode_email_cursor(cursor, scope)
            position = (date, message_id)
        parsed_groups = _parse_email_operator_query(self, query, as_of=as_of)

        where_clauses = ["em.tenant_id = ?"]
        if not include_deleted:
            if self.backend_type == BackendType.POSTGRESQL:
                where_clauses.extend(
                    [
                        "COALESCE(m.deleted, FALSE) = FALSE",
                        "COALESCE(m.is_trash, FALSE) = FALSE",
                    ]
                )
            else:
                where_clauses.extend(
                    [
                        "COALESCE(m.deleted, 0) = 0",
                        "COALESCE(m.is_trash, 0) = 0",
                    ]
                )
        where_params: list[Any] = [resolved_tenant]

        group_sql_clauses: list[str] = []
        for group in parsed_groups:
            group_parts: list[str] = []
            for term in group:
                kind = str(term.get("kind") or "").strip().lower()
                value = term.get("value")
                negated = bool(term.get("negated"))
                part_sql = ""
                part_params: list[Any] = []

                if kind == "participant":
                    role = str(term.get("role") or "").strip().lower()
                    like_value = f"%{str(value or '').strip()}%"
                    participant_display_expr = "COALESCE(ep.display_name, '')"
                    participant_text_clause = (
                        f"{_email_like_clause(self, 'ep.email_normalized')} "
                        f"OR {_email_like_clause(self, participant_display_expr)}"
                    )
                    part_sql = (
                        "EXISTS ("  # nosec B608
                        "SELECT 1 FROM email_message_participants emp "
                        "JOIN email_participants ep ON ep.id = emp.participant_id "
                        "WHERE emp.email_message_id = em.id "
                        "AND emp.role = ? "
                        "AND ep.tenant_id = em.tenant_id "
                        f"AND ({participant_text_clause})"
                        ")"
                    )
                    part_params.extend([role, like_value, like_value])
                elif kind == "subject":
                    like_value = f"%{str(value or '').strip()}%"
                    part_sql = _email_like_clause(self, "COALESCE(em.subject, '')")
                    part_params.append(like_value)
                elif kind == "label":
                    like_value = f"%{str(value or '').strip()}%"
                    label_text_clause = (
                        f"{_email_like_clause(self, 'el.label_name')} "
                        f"OR {_email_like_clause(self, 'el.label_key')}"
                    )
                    part_sql = (
                        "EXISTS ("  # nosec B608
                        "SELECT 1 FROM email_message_labels eml "
                        "JOIN email_labels el ON el.id = eml.label_id "
                        "WHERE eml.email_message_id = em.id "
                        "AND el.tenant_id = em.tenant_id "
                        f"AND ({label_text_clause})"
                        ")"
                    )
                    part_params.extend([like_value, like_value])
                elif kind == "has_attachment":
                    bool_true = True if self.backend_type == BackendType.POSTGRESQL else 1
                    part_sql = (
                        "(em.has_attachments = ? OR EXISTS ("
                        "SELECT 1 FROM email_attachments ea WHERE ea.email_message_id = em.id"
                        "))"
                    )
                    part_params.append(bool_true)
                elif kind == "before":
                    part_sql = "em.internal_date < ?"
                    part_params.append(str(value))
                elif kind == "after":
                    part_sql = "em.internal_date >= ?"
                    part_params.append(str(value))
                elif kind == "older_than":
                    part_sql = "em.internal_date < ?"
                    part_params.append(str(value))
                elif kind == "newer_than":
                    part_sql = "em.internal_date >= ?"
                    part_params.append(str(value))
                else:
                    text_value = str(value or "").strip()
                    like_value = f"%{text_value}%"
                    like_clauses = [
                        _email_like_clause(self, "COALESCE(em.subject, '')"),
                        _email_like_clause(self, "COALESCE(em.body_text, '')"),
                        _email_like_clause(self, "COALESCE(em.from_text, '')"),
                        _email_like_clause(self, "COALESCE(em.to_text, '')"),
                        _email_like_clause(self, "COALESCE(em.cc_text, '')"),
                        _email_like_clause(self, "COALESCE(em.bcc_text, '')"),
                        _email_like_clause(self, "COALESCE(em.label_text, '')"),
                    ]
                    part_sql = "(" + " OR ".join(like_clauses) + ")"
                    part_params.extend([like_value] * len(like_clauses))
                    if self.backend_type == BackendType.SQLITE and text_value:
                        fts_term = _sqlite_fts_literal_term(text_value)
                        if fts_term:
                            part_sql = (
                                "("  # nosec B608
                                "em.id IN (SELECT rowid FROM email_fts WHERE email_fts MATCH ?) "
                                f"OR {part_sql}"
                                ")"
                            )
                            part_params = [fts_term, *part_params]

                if not part_sql:
                    continue
                if negated:
                    part_sql = f"NOT ({part_sql})"
                group_parts.append(part_sql)
                where_params.extend(part_params)

            if group_parts:
                group_sql_clauses.append("(" + " AND ".join(group_parts) + ")")

        if group_sql_clauses:
            where_clauses.append("(" + " OR ".join(group_sql_clauses) + ")")

        where_sql = " AND ".join(where_clauses)
        base_from = (
            " FROM email_messages em "
            "JOIN Media m ON m.id = em.media_id "
            "WHERE " + where_sql
        )
        page_from = base_from
        page_params = list(where_params)
        if position is not None:
            date, message_id = position
            if date is None:
                page_from += " AND em.internal_date IS NULL AND em.id < ?"
                page_params.append(message_id)
            else:
                page_from += (
                    " AND (em.internal_date < ? OR (em.internal_date = ? AND em.id < ?) OR em.internal_date IS NULL)"
                )
                page_params.extend([date, date, message_id])
        ordering = " ORDER BY em.internal_date DESC, em.id DESC "
        if cursor is not None:
            ordering = " ORDER BY em.internal_date DESC NULLS LAST, em.id DESC "

        with self.transaction() as conn:
            count_row = self._fetchone_with_connection(
                conn,
                "SELECT COUNT(*) AS total" + base_from,
                tuple(where_params),
            )
            total = int((count_row or {}).get("total", 0) or 0)

            rows = self._fetchall_with_connection(
                conn,
                (
                    "SELECT "  # nosec B608
                    "em.id AS email_message_id, "
                    "em.media_id AS media_id, "
                    "m.uuid AS media_uuid, "
                    "m.url AS media_url, "
                    "m.title AS media_title, "
                    "em.source_id AS source_id, "
                    "em.source_message_id AS source_message_id, "
                    "em.message_id AS message_id, "
                    "em.subject AS subject, "
                    "em.internal_date AS internal_date, "
                    "em.from_text AS from_text, "
                    "em.to_text AS to_text, "
                    "em.cc_text AS cc_text, "
                    "em.bcc_text AS bcc_text, "
                    "em.label_text AS label_text, "
                    "em.has_attachments AS has_attachments, "
                    "(SELECT COUNT(*) FROM email_attachments ea WHERE ea.email_message_id = em.id) "
                    "AS attachment_count" + page_from + ordering + "LIMIT ? OFFSET ?"
                ),
                (*page_params, limit_int + (cursor is not None), offset_int),
            )

        _emit_email_metric_counter(
            "email_native_search_requests_total",
            labels={
                "phase": "success",
                "query_present": query_present,
                "include_deleted": include_deleted_label,
            },
        )
        _emit_email_metric_histogram(
            "email_native_search_results_total",
            float(total),
            labels={
                "query_present": query_present,
                "include_deleted": include_deleted_label,
            },
        )
        if cursor is not None:
            has_more = len(rows) > limit_int
            rows = rows[:limit_int]
            next_cursor = encode_email_cursor(scope, as_of, rows[-1]) if has_more else None
            return rows, total, next_cursor
        return rows, total
    except InputError:
        _emit_email_metric_counter(
            "email_native_search_requests_total",
            labels={
                "phase": "parse_error",
                "query_present": query_present,
                "include_deleted": include_deleted_label,
            },
        )
        _emit_email_metric_counter(
            "email_native_search_parse_failures_total",
            labels={
                "query_present": query_present,
                "include_deleted": include_deleted_label,
            },
        )
        raise
    except _MEDIA_NONCRITICAL_EXCEPTIONS as exc:
        _emit_email_metric_counter(
            "email_native_search_requests_total",
            labels={
                "phase": "error",
                "query_present": query_present,
                "include_deleted": include_deleted_label,
                "error_type": type(exc).__name__,
            },
        )
        raise
    finally:
        _emit_email_metric_histogram(
            "email_native_search_duration_seconds",
            time.perf_counter() - started_at,
            labels={
                "query_present": query_present,
                "include_deleted": include_deleted_label,
            },
        )


def get_email_message_detail(
    self,
    *,
    email_message_id: int,
    tenant_id: str | None = None,
    include_deleted: bool = False,
) -> dict[str, Any] | None:
    """Fetch a normalized email message graph for detail API responses."""

    try:
        message_id_int = int(email_message_id)
    except _MEDIA_NONCRITICAL_EXCEPTIONS as exc:
        raise InputError("email_message_id must be an integer.") from exc  # noqa: TRY003

    if message_id_int <= 0:
        raise InputError("email_message_id must be greater than zero.")  # noqa: TRY003

    resolved_tenant = self._resolve_email_tenant_id(tenant_id)

    deleted_clause = "" if include_deleted else "AND m.deleted = 0 "
    with self.transaction() as conn:
        message_row = self._fetchone_with_connection(
            conn,
            (
                "SELECT "  # nosec B608
                "em.id AS email_message_id, "
                "em.media_id AS media_id, "
                "m.uuid AS media_uuid, "
                "m.url AS media_url, "
                "m.title AS media_title, "
                "em.source_id AS source_id, "
                "es.provider AS source_provider, "
                "es.source_key AS source_key, "
                "es.display_name AS source_display_name, "
                "em.source_message_id AS source_message_id, "
                "em.message_id AS message_id, "
                "em.subject AS subject, "
                "em.body_text AS body_text, "
                "em.internal_date AS internal_date, "
                "em.from_text AS from_text, "
                "em.to_text AS to_text, "
                "em.cc_text AS cc_text, "
                "em.bcc_text AS bcc_text, "
                "em.label_text AS label_text, "
                "em.has_attachments AS has_attachments, "
                "em.raw_metadata_json AS raw_metadata_json "
                "FROM email_messages em "
                "JOIN Media m ON m.id = em.media_id "
                "JOIN email_sources es ON es.id = em.source_id "
                "WHERE em.id = ? AND em.tenant_id = ? "
                + deleted_clause +
                "LIMIT 1"
            ),
            (message_id_int, resolved_tenant),
        )
        if message_row is None:
            return None

        participant_rows = self._fetchall_with_connection(
            conn,
            (
                "SELECT emp.role AS role, ep.email_normalized AS email, ep.display_name AS display_name "
                "FROM email_message_participants emp "
                "JOIN email_participants ep ON ep.id = emp.participant_id "
                "WHERE emp.email_message_id = ? AND ep.tenant_id = ? "
                "ORDER BY "
                "CASE emp.role "
                "WHEN 'from' THEN 0 "
                "WHEN 'to' THEN 1 "
                "WHEN 'cc' THEN 2 "
                "WHEN 'bcc' THEN 3 "
                "ELSE 9 END, "
                "ep.email_normalized ASC"
            ),
            (message_id_int, resolved_tenant),
        )

        label_rows = self._fetchall_with_connection(
            conn,
            (
                "SELECT el.label_key AS label_key, el.label_name AS label_name "
                "FROM email_message_labels eml "
                "JOIN email_labels el ON el.id = eml.label_id "
                "WHERE eml.email_message_id = ? AND el.tenant_id = ? "
                "ORDER BY el.label_name ASC"
            ),
            (message_id_int, resolved_tenant),
        )

        attachment_rows = self._fetchall_with_connection(
            conn,
            (
                "SELECT id, filename, content_type, size_bytes, content_id, disposition, "
                "extracted_text_available "
                "FROM email_attachments "
                "WHERE email_message_id = ? "
                "ORDER BY id ASC"
            ),
            (message_id_int,),
        )

    participants: dict[str, list[dict[str, str | None]]] = {
        "from": [],
        "to": [],
        "cc": [],
        "bcc": [],
    }
    for row in participant_rows:
        role = str(row.get("role") or "").strip().lower()
        if role not in participants:
            continue
        participants[role].append(
            {
                "email": row.get("email"),
                "display_name": row.get("display_name"),
            }
        )

    labels = [
        {
            "label_key": row.get("label_key"),
            "label_name": row.get("label_name"),
        }
        for row in label_rows
    ]

    attachments = [
        {
            "id": row.get("id"),
            "filename": row.get("filename"),
            "content_type": row.get("content_type"),
            "size_bytes": row.get("size_bytes"),
            "content_id": row.get("content_id"),
            "disposition": row.get("disposition"),
            "extracted_text_available": bool(row.get("extracted_text_available")),
        }
        for row in attachment_rows
    ]

    raw_metadata = None
    raw_metadata_json = message_row.get("raw_metadata_json")
    if isinstance(raw_metadata_json, str) and raw_metadata_json.strip():
        try:
            raw_metadata = json.loads(raw_metadata_json)
        except json.JSONDecodeError:
            raw_metadata = None

    return {
        "email_message_id": message_row.get("email_message_id"),
        "message_id": message_row.get("message_id"),
        "source_message_id": message_row.get("source_message_id"),
        "subject": message_row.get("subject"),
        "internal_date": message_row.get("internal_date"),
        "body_text": message_row.get("body_text"),
        "has_attachments": bool(message_row.get("has_attachments")),
        "search_text": {
            "from": message_row.get("from_text"),
            "to": message_row.get("to_text"),
            "cc": message_row.get("cc_text"),
            "bcc": message_row.get("bcc_text"),
            "labels": message_row.get("label_text"),
        },
        "media": {
            "id": message_row.get("media_id"),
            "uuid": message_row.get("media_uuid"),
            "url": message_row.get("media_url"),
            "title": message_row.get("media_title"),
        },
        "source": {
            "id": message_row.get("source_id"),
            "provider": message_row.get("source_provider"),
            "source_key": message_row.get("source_key"),
            "display_name": message_row.get("source_display_name"),
        },
        "participants": participants,
        "labels": labels,
        "attachments": attachments,
        "raw_metadata": raw_metadata,
    }
