"""Package-owned claims cluster aggregate helpers."""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)


_MEDIA_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = MEDIA_NONCRITICAL_EXCEPTIONS


def get_claim_clusters_by_ids(
    self,
    cluster_ids: list[int],
    *,
    owner_user_id: str | None = None,
) -> list[dict[str, Any]]:
    """Load cluster text, optionally restricted to the explicit owner."""
    if not cluster_ids:
        return []
    placeholders = ",".join("?" * len(cluster_ids))
    sql = (
        "SELECT id, canonical_claim_text, updated_at "  # nosec B608
        f"FROM claim_clusters WHERE id IN ({placeholders})"
    )
    params: list[Any] = [int(cid) for cid in cluster_ids]
    if owner_user_id is not None:
        sql += " AND claim_clusters.user_id = ?"
        params.append(owner_user_id)
    rows = self.execute_query(sql, tuple(params)).fetchall()
    return [dict(row) for row in rows]


def get_claim_cluster_member_counts(
    self,
    cluster_ids: list[int],
    *,
    owner_user_id: str | None = None,
) -> dict[int, int]:
    """Count members only through owned clusters when an owner is supplied."""
    if not cluster_ids:
        return {}
    placeholders = ",".join("?" * len(cluster_ids))
    params: list[Any] = [int(cid) for cid in cluster_ids]
    if owner_user_id is None:
        sql = (
            "SELECT cluster_id, COUNT(*) AS member_count "  # nosec B608
            f"FROM claim_cluster_membership WHERE cluster_id IN ({placeholders}) "
            "GROUP BY cluster_id"
        )
    else:
        owned_members_sql = (
            "SELECT cm.cluster_id, COUNT(*) AS member_count "
            "FROM claim_cluster_membership cm "
            "JOIN claim_clusters ON claim_clusters.id = cm.cluster_id "
            "WHERE cm.cluster_id IN ({ids}) AND claim_clusters.user_id = ? "
            "GROUP BY cm.cluster_id"
        )
        sql = owned_members_sql.replace("{ids}", placeholders)
        params.append(owner_user_id)
    rows = self.execute_query(sql, tuple(params)).fetchall()
    counts: dict[int, int] = {}
    for row in rows:
        try:
            cluster_id = int(row["cluster_id"])
            member_count = int(row["member_count"])
        except _MEDIA_NONCRITICAL_EXCEPTIONS:
            try:
                cluster_id = int(row[0])
                member_count = int(row[1])
            except _MEDIA_NONCRITICAL_EXCEPTIONS:
                continue
        counts[cluster_id] = member_count
    return counts


def update_claim_clusters_watchlist_counts(
    self,
    counts: dict[int, int],
    *,
    owner_user_id: str | None = None,
) -> int:
    """Update counts with a write-time owner guard; legacy calls count inputs."""
    if not counts:
        return 0
    if owner_user_id is not None:
        affected = 0
        with self.transaction() as conn:
            for cluster_id, count in counts.items():
                cursor = self._execute_with_connection(
                    conn,
                    "UPDATE claim_clusters SET watchlist_count = ? WHERE id = ? AND claim_clusters.user_id = ?",
                    (int(count), int(cluster_id), owner_user_id),
                )
                affected += int(cursor.rowcount)
        return affected
    params = [(int(count), int(cluster_id)) for cluster_id, count in counts.items()]
    self.execute_many(
        "UPDATE claim_clusters SET watchlist_count = ? WHERE id = ?",
        params,
    )
    return len(params)
