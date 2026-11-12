"""Package-owned claims write helpers."""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError


def upsert_claims(self, claims: list[dict[str, Any]]) -> int:
    """Insert claims in bulk."""
    if not claims:
        return 0
    now = self._get_current_utc_timestamp_str()
    rows: list[tuple] = []
    for c in claims:
        media_id = int(c["media_id"])
        extractor = str(c.get("extractor", "heuristic"))
        reviewer_id = c.get("reviewer_id")
        review_group = c.get("review_group")
        rows.append(
            (
                media_id,
                int(c.get("chunk_index", 0)),
                c.get("span_start"),
                c.get("span_end"),
                str(c["claim_text"]),
                float(c.get("confidence")) if c.get("confidence") is not None else None,
                extractor,
                str(c.get("extractor_version", "v1")),
                str(c["chunk_hash"]),
                str(c.get("uuid", self._generate_uuid())),
                str(c.get("last_modified", now)),
                int(c.get("version", 1)),
                str(c.get("client_id", self.client_id)),
                c.get("prev_version"),
                c.get("merge_parent_uuid"),
                int(reviewer_id) if reviewer_id is not None else None,
                str(review_group) if review_group else None,
            )
        )
    with self.transaction() as conn:
        self.execute_many(
            """
            INSERT INTO Claims (
                media_id, chunk_index, span_start, span_end, claim_text, confidence,
                extractor, extractor_version, chunk_hash, uuid, last_modified,
                version, client_id, prev_version, merge_parent_uuid, reviewer_id, review_group
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
            commit=False,
            connection=conn,
        )
    return len(rows)


def update_claim(
    self,
    claim_id: int,
    *,
    claim_text: str | None = None,
    span_start: int | None = None,
    span_end: int | None = None,
    confidence: float | None = None,
    extractor: str | None = None,
    extractor_version: str | None = None,
    deleted: bool | None = None,
) -> dict[str, Any] | None:
    """Update a claim row and return the updated record."""
    update_parts: list[str] = []
    params: list[Any] = []

    if claim_text is not None:
        update_parts.append("claim_text = ?")
        params.append(str(claim_text))
    if span_start is not None:
        update_parts.append("span_start = ?")
        params.append(int(span_start))
    if span_end is not None:
        update_parts.append("span_end = ?")
        params.append(int(span_end))
    if confidence is not None:
        update_parts.append("confidence = ?")
        params.append(float(confidence))
    if extractor is not None:
        update_parts.append("extractor = ?")
        params.append(str(extractor))
    if extractor_version is not None:
        update_parts.append("extractor_version = ?")
        params.append(str(extractor_version))
    if deleted is not None:
        update_parts.append("deleted = ?")
        params.append(1 if deleted else 0)

    if not update_parts:
        return self.get_claim_with_media(int(claim_id), include_deleted=True)

    now = self._get_current_utc_timestamp_str()
    update_parts.append("last_modified = ?")
    params.append(now)
    update_parts.append("version = version + 1")
    update_parts.append("client_id = ?")
    params.append(str(self.client_id))

    params.append(int(claim_id))

    sql = "UPDATE Claims SET " + ", ".join(update_parts) + " WHERE id = ?"  # nosec B608
    self.execute_query(sql, tuple(params), commit=True)

    if self.backend_type == BackendType.POSTGRESQL and claim_text is not None:
        self.execute_query(
            "UPDATE Claims "
            "SET claims_fts_tsv = CASE "
            "WHEN deleted = 0 THEN to_tsvector('english', coalesce(claim_text, '')) "
            "ELSE NULL END "
            "WHERE id = ?",
            (int(claim_id),),
            commit=True,
        )

    return self.get_claim_with_media(int(claim_id), include_deleted=True)


def update_claim_review(
    self,
    claim_id: int,
    *,
    review_status: str | None = None,
    reviewer_id: int | None = None,
    review_group: str | None = None,
    review_notes: str | None = None,
    review_reason_code: str | None = None,
    corrected_text: str | None = None,
    span_start: int | None = None,
    span_end: int | None = None,
    expected_version: int | None = None,
    action_ip: str | None = None,
    action_user_agent: str | None = None,
    actor_user_id: int | None = None,
) -> dict[str, Any] | None:
    """Update review fields on a claim with optional optimistic locking."""
    with self.transaction() as conn:
        row = self._fetchone_with_connection(
            conn,
            "SELECT * FROM Claims WHERE id = ?",
            (int(claim_id),),
        )
        if not row:
            return None

        current_review_version = int(row.get("review_version") or 1)
        if expected_version is not None and current_review_version != int(expected_version):
            return {"conflict": True, "current": dict(row)}

        update_parts: list[str] = []
        params: list[Any] = []
        now = self._get_current_utc_timestamp_str()

        if review_status is not None:
            update_parts.append("review_status = ?")
            params.append(str(review_status))
        if reviewer_id is not None:
            update_parts.append("reviewer_id = ?")
            params.append(int(reviewer_id))
        if review_group is not None:
            update_parts.append("review_group = ?")
            params.append(str(review_group))
        if review_notes is not None:
            update_parts.append("review_notes = ?")
            params.append(str(review_notes))
        if review_reason_code is not None:
            update_parts.append("review_reason_code = ?")
            params.append(str(review_reason_code))
        if span_start is not None:
            update_parts.append("span_start = ?")
            params.append(int(span_start))
        if span_end is not None:
            update_parts.append("span_end = ?")
            params.append(int(span_end))

        if corrected_text is not None:
            update_parts.append("claim_text = ?")
            params.append(str(corrected_text))
            update_parts.append("last_modified = ?")
            params.append(now)
            update_parts.append("version = version + 1")
            update_parts.append("client_id = ?")
            params.append(str(self.client_id))

        if update_parts:
            update_parts.append("reviewed_at = ?")
            params.append(now)
            update_parts.append("review_version = review_version + 1")

        if not update_parts:
            return dict(row)

        where_clause = "id = ?"
        params.append(int(claim_id))
        if expected_version is not None:
            where_clause += " AND review_version = ?"
            params.append(int(expected_version))

        sql = "UPDATE Claims SET " + ", ".join(update_parts) + " WHERE " + where_clause  # nosec B608
        cur = self._execute_with_connection(conn, sql, tuple(params))
        if cur.rowcount == 0:
            return {"conflict": True, "current": dict(row)}

        if self.backend_type == BackendType.POSTGRESQL and corrected_text is not None:
            self._execute_with_connection(
                conn,
                "UPDATE Claims "
                "SET claims_fts_tsv = CASE "
                "WHEN deleted = 0 THEN to_tsvector('english', coalesce(claim_text, '')) "
                "ELSE NULL END "
                "WHERE id = ?",
                (int(claim_id),),
            )

        old_status = row.get("review_status")
        old_text = row.get("claim_text")
        new_status = review_status if review_status is not None else old_status
        new_text = corrected_text if corrected_text is not None else old_text
        self._execute_with_connection(
            conn,
            (
                "INSERT INTO claims_review_log "
                "(claim_id, old_status, new_status, old_text, new_text, reviewer_id, notes, reason_code, "
                "action_ip, action_user_agent, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ),
            (
                int(claim_id),
                old_status,
                new_status,
                old_text,
                new_text,
                int(actor_user_id) if actor_user_id is not None else (
                    int(reviewer_id) if reviewer_id is not None else row.get("reviewer_id")
                ),
                review_notes,
                review_reason_code,
                action_ip,
                action_user_agent,
                now,
            ),
        )

    return self.get_claim_with_media(int(claim_id), include_deleted=True)


def soft_delete_claims_for_media(self, media_id: int) -> int:
    """Soft-delete all claims for a given media_id."""
    try:
        with self.transaction() as conn:
            current_time = self._get_current_utc_timestamp_str()
            update_sql = (
                "UPDATE Claims "
                "SET deleted = 1, version = version + 1, last_modified = ?, client_id = ? "
                "WHERE media_id = ? AND deleted = 0"
            )
            cur = self._execute_with_connection(
                conn,
                update_sql,
                (current_time, self.client_id, int(media_id)),
            )
            affected = cur.rowcount or 0

            if self.backend_type == BackendType.SQLITE:
                try:
                    self._execute_with_connection(
                        conn,
                        "INSERT INTO claims_fts(claims_fts, rowid, claim_text) "
                        "SELECT 'delete', id, claim_text FROM Claims WHERE media_id = ?",
                        (int(media_id),),
                    )
                except sqlite3.Error as exc:
                    logger.warning(
                        "Failed to update SQLite claims_fts delete markers for media_id={}: {}",
                        media_id,
                        exc,
                    )

            return affected
    except sqlite3.Error as e:
        logging.error(f"Failed to soft-delete claims for media_id={media_id}: {e}", exc_info=True)
        raise DatabaseError(f"Failed to soft-delete claims: {e}") from e  # noqa: TRY003
