"""Package-owned claims cluster CRUD/link/member helpers."""

from __future__ import annotations

import logging
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.claims_cluster_rebuild_types import ClusterRebuildStorageError
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_cluster_rebuild_snapshot import lock_claim_rows
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)
from tldw_Server_API.app.core.DB_Management.scope_context import get_scope

_MEDIA_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = MEDIA_NONCRITICAL_EXCEPTIONS


def list_claim_clusters(
    self,
    user_id: str,
    *,
    keyword: str | None = None,
    updated_since: str | None = None,
    watchlisted: bool | None = None,
    min_size: int | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    try:
        limit = int(limit)
        offset = int(offset)
    except (TypeError, ValueError):
        limit, offset = 100, 0
    limit = max(1, min(1000, limit))
    offset = max(0, offset)

    conditions: list[str] = ["c.user_id = ?"]
    params: list[Any] = [str(user_id)]
    if updated_since:
        conditions.append("c.updated_at >= ?")
        params.append(str(updated_since))
    if keyword:
        conditions.append("(c.canonical_claim_text LIKE ? OR c.summary LIKE ?)")
        like = f"%{keyword}%"
        params.extend([like, like])
    if watchlisted is not None:
        if watchlisted:
            conditions.append("c.watchlist_count > 0")
        else:
            conditions.append("c.watchlist_count = 0")
    if min_size is not None:
        conditions.append("COALESCE(m.member_count, 0) >= ?")
        params.append(int(min_size))

    sql = (
        "SELECT c.id, c.user_id, c.canonical_claim_text, c.representative_claim_id, c.summary, "  # nosec B608
        "c.cluster_version, c.watchlist_count, c.created_at, c.updated_at, "
        "COALESCE(m.member_count, 0) AS member_count "
        "FROM claim_clusters c "
        "LEFT JOIN (SELECT cluster_id, COUNT(*) AS member_count "
        "FROM claim_cluster_membership GROUP BY cluster_id) m "
        "ON m.cluster_id = c.id "
        f"WHERE {' AND '.join(conditions)} "
        "ORDER BY c.updated_at DESC LIMIT ? OFFSET ?"
    )
    params.extend([limit, offset])
    rows = self.execute_query(sql, tuple(params)).fetchall()
    return [dict(row) for row in rows]


def get_claim_cluster(self, cluster_id: int) -> dict[str, Any]:
    row = self.execute_query(
        "SELECT id, user_id, canonical_claim_text, representative_claim_id, summary, "
        "cluster_version, watchlist_count, created_at, updated_at "
        "FROM claim_clusters WHERE id = ?",
        (int(cluster_id),),
    ).fetchone()
    return dict(row) if row else {}


def get_claim_cluster_link(
    self,
    *,
    parent_cluster_id: int,
    child_cluster_id: int,
) -> dict[str, Any]:
    row = self.execute_query(
        (
            "SELECT parent_cluster_id, child_cluster_id, relation_type, created_at "
            "FROM claim_cluster_links WHERE parent_cluster_id = ? AND child_cluster_id = ?"
        ),
        (int(parent_cluster_id), int(child_cluster_id)),
    ).fetchone()
    return dict(row) if row else {}


def list_claim_cluster_links(
    self,
    *,
    cluster_id: int,
    direction: str = "both",
) -> list[dict[str, Any]]:
    direction_norm = str(direction or "both").lower()
    conditions: list[str] = []
    params: list[Any] = []
    if direction_norm in {"outbound", "parent"}:
        conditions.append("parent_cluster_id = ?")
        params.append(int(cluster_id))
    elif direction_norm in {"inbound", "child"}:
        conditions.append("child_cluster_id = ?")
        params.append(int(cluster_id))
    else:
        conditions.append("(parent_cluster_id = ? OR child_cluster_id = ?)")
        params.extend([int(cluster_id), int(cluster_id)])
    rows = self.execute_query(
        (
            "SELECT parent_cluster_id, child_cluster_id, relation_type, created_at "  # nosec B608
            "FROM claim_cluster_links WHERE "
            + " AND ".join(conditions)
            + " ORDER BY created_at DESC"
        ),
        tuple(params),
    ).fetchall()
    return [dict(row) for row in rows]


def create_claim_cluster_link(
    self,
    *,
    parent_cluster_id: int,
    child_cluster_id: int,
    relation_type: str | None = None,
) -> dict[str, Any]:
    now = self._get_current_utc_timestamp_str()
    self.execute_query(
        (
            "INSERT INTO claim_cluster_links "
            "(parent_cluster_id, child_cluster_id, relation_type, created_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING"
        ),
        (
            int(parent_cluster_id),
            int(child_cluster_id),
            relation_type,
            now,
        ),
        commit=True,
    )
    return self.get_claim_cluster_link(
        parent_cluster_id=parent_cluster_id,
        child_cluster_id=child_cluster_id,
    )


def delete_claim_cluster_link(
    self,
    *,
    parent_cluster_id: int,
    child_cluster_id: int,
) -> int:
    cur = self.execute_query(
        (
            "DELETE FROM claim_cluster_links "
            "WHERE parent_cluster_id = ? AND child_cluster_id = ?"
        ),
        (int(parent_cluster_id), int(child_cluster_id)),
        commit=True,
    )
    try:
        deleted = int(cur.rowcount or 0)
    except _MEDIA_NONCRITICAL_EXCEPTIONS:
        deleted = 0
    return max(deleted, 0)


def list_claim_cluster_members(
    self,
    cluster_id: int,
    *,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    try:
        limit = int(limit)
        offset = int(offset)
    except (TypeError, ValueError):
        limit, offset = 100, 0
    limit = max(1, min(1000, limit))
    offset = max(0, offset)
    conditions: list[str] = ["cm.cluster_id = ?", "c.media_id = m.id"]
    params: list[Any] = [int(cluster_id)]

    try:
        scope = get_scope()
    except _MEDIA_NONCRITICAL_EXCEPTIONS as scope_err:
        logging.debug(
            "Failed to resolve scope for cluster membership visibility filter: %s",
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

    sql = (
        "SELECT c.id, c.media_id, c.chunk_index, c.span_start, c.span_end, c.claim_text, "  # nosec B608
        "c.confidence, c.extractor, c.extractor_version, c.chunk_hash, c.created_at, c.uuid, "
        "c.last_modified, c.version, c.client_id, c.deleted, "
        "c.review_status, c.reviewer_id, c.review_group, c.reviewed_at, "
        "c.review_notes, c.review_version, c.review_reason_code, c.claim_cluster_id, "
        "m.title AS media_title, m.visibility AS media_visibility, "
        "m.owner_user_id AS media_owner_user_id, m.org_id AS media_org_id, "
        "m.team_id AS media_team_id, m.client_id AS media_client_id, "
        "cm.similarity_score, cm.cluster_joined_at "
        "FROM claim_cluster_membership cm "
        "JOIN Claims c ON c.id = cm.claim_id "
        "JOIN Media m ON c.media_id = m.id "
        f"WHERE {' AND '.join(conditions)} "
        "ORDER BY cm.cluster_joined_at DESC "
        "LIMIT ? OFFSET ?"
    )
    params.extend([limit, offset])
    rows = self.execute_query(sql, tuple(params)).fetchall()
    return [dict(row) for row in rows]


def create_claim_cluster(
    self,
    *,
    user_id: str,
    canonical_claim_text: str | None = None,
    representative_claim_id: int | None = None,
    summary: str | None = None,
) -> dict[str, Any]:
    now = self._get_current_utc_timestamp_str()
    insert_sql = (
        "INSERT INTO claim_clusters "
        "(user_id, canonical_claim_text, representative_claim_id, summary, "
        "cluster_version, watchlist_count, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    )
    if self.backend_type == BackendType.POSTGRESQL:
        insert_sql += " RETURNING id"
    cursor = self.execute_query(
        insert_sql,
        (
            str(user_id),
            canonical_claim_text,
            int(representative_claim_id) if representative_claim_id is not None else None,
            summary,
            1,
            0,
            now,
            now,
        ),
        commit=True,
    )
    if self.backend_type == BackendType.POSTGRESQL:
        row = cursor.fetchone()
        cluster_id = int(row["id"]) if row else None
    else:
        cluster_id = cursor.lastrowid
    return self.get_claim_cluster(cluster_id) if cluster_id else {}


def add_claim_to_cluster(
    self,
    *,
    cluster_id: int,
    claim_id: int,
    similarity_score: float | None = None,
) -> None:
    now = self._get_current_utc_timestamp_str()
    with self.transaction() as conn:
        query = "SELECT id FROM claim_clusters WHERE id = ?"
        if self.backend_type == BackendType.POSTGRESQL:
            query = "SELECT id FROM claim_clusters WHERE id = ? FOR UPDATE"
        destination = self._fetchone_with_connection(conn, query, (int(cluster_id),))
        if not destination:
            raise ClusterRebuildStorageError("claims_cluster_source_changed", retryable=True)
        lock_claim_rows(self, conn, (int(claim_id),), owner_user_id=None)
        self._execute_with_connection(
            conn,
            (
                "INSERT INTO claim_cluster_membership "
                "(cluster_id, claim_id, similarity_score, cluster_joined_at) "
                "VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING"
            ),
            (int(cluster_id), int(claim_id), similarity_score, now),
        )
        self._execute_with_connection(
            conn,
            "UPDATE Claims SET claim_cluster_id = ? WHERE id = ?",
            (int(cluster_id), int(claim_id)),
        )
        self._execute_with_connection(
            conn,
            "UPDATE claim_clusters SET cluster_version = cluster_version + 1, updated_at = ? WHERE id = ?",
            (now, int(cluster_id)),
        )
