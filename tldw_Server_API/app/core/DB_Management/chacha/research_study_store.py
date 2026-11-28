"""Owner-qualified persistence and live native projections for Study receipts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
    from tldw_Server_API.app.core.Research_Workspace.study_output_models import ResearchStudySource


_VISIBLE_RECEIPTS = """
SELECT r.owner_user_id, r.workspace_id, r.operation_id, r.output_kind,
       r.owner_id, r.generation_json, r.created_at,
       COALESCE(d.name, q.name) AS name, COALESCE(d.version, q.version) AS version
FROM research_study_outputs r
JOIN workspaces w ON w.id = r.workspace_id AND w.deleted = FALSE AND w.system_operation_state IS NULL
LEFT JOIN decks d ON r.output_kind = 'flashcards' AND d.id = r.owner_id
    AND d.workspace_id = r.workspace_id AND d.deleted = FALSE AND (? OR d.client_id = r.owner_user_id)
LEFT JOIN quizzes q ON r.output_kind = 'quiz' AND q.id = r.owner_id
    AND q.workspace_id = r.workspace_id AND q.deleted = FALSE AND (? OR q.client_id = r.owner_user_id)
WHERE r.owner_user_id = ? AND r.workspace_id = ? AND COALESCE(d.id, q.id) IS NOT NULL
"""


# Compose two fixed projections. All request values are bound where executed.
_PAGED_RECEIPTS = (
    f"WITH visible AS ({_VISIBLE_RECEIPTS})"  # nosec B608
    """
    SELECT page.*, totals.total
    FROM (SELECT COUNT(*) AS total FROM visible) AS totals
    LEFT JOIN (
        SELECT * FROM visible ORDER BY created_at DESC, operation_id LIMIT ? OFFSET ?
    ) AS page ON TRUE
    ORDER BY page.created_at DESC, page.operation_id
    """
)


class ResearchStudyStore:
    """Keep the receipt separate from the native deck, cards, and quiz rows."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self.db = db

    def _scope(self, workspace_id: str) -> tuple:
        sqlite = self.db.backend_type == BackendType.SQLITE
        return (sqlite, sqlite, self.db.owner_user_id, workspace_id)

    def sources_current(self, conn: Any, workspace_id: str, sources: tuple[ResearchStudySource, ...]) -> bool:
        """Check and lock the captured selected associations through publication."""
        # The number of placeholders varies with the bounded source snapshot.
        marks = ", ".join("?" for _ in sources)
        query = (
            "SELECT id, media_id, source_type, selected, version FROM workspace_sources "  # nosec B608
            f"WHERE workspace_id = ? AND id IN ({marks}) ORDER BY id"
        )
        if self.db.backend_type == BackendType.POSTGRESQL:
            query += " FOR SHARE"
        rows = {
            row["id"]: row
            for row in conn.execute(query, (workspace_id, *(source.source_id for source in sources))).fetchall()
        }
        return all(
            (row := rows.get(source.source_id)) is not None
            and bool(row["selected"])
            and str(row["media_id"]) == source.catalog_item_id
            and (row["source_type"] or "media") == source.source_type
            and row["version"] == source.association_version
            for source in sources
        )

    def find_operation(self, conn: Any, workspace_id: str, operation_id: str) -> dict[str, Any] | None:
        """Read the operation even if the native record is no longer visible."""
        row = conn.execute(
            "SELECT request_fingerprint, output_kind, owner_id FROM research_study_outputs "
            "WHERE owner_user_id = ? AND workspace_id = ? AND operation_id = ?",
            (self.db.owner_user_id, workspace_id, operation_id),
        ).fetchone()
        return dict(row) if row else None

    def add_receipt(
        self,
        conn: Any,
        workspace_id: str,
        *,
        operation_id: str,
        fingerprint: str,
        kind: str,
        owner_id: int,
        generation_json: str,
    ) -> None:
        """Publish the provenance in the native owner's creation transaction."""
        conn.execute(
            "INSERT INTO research_study_outputs "
            "(owner_user_id, workspace_id, operation_id, request_fingerprint, output_kind, owner_id, "
            "generation_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self.db.owner_user_id,
                workspace_id,
                operation_id,
                fingerprint,
                kind,
                owner_id,
                generation_json,
                self.db.ctx.utc_now_iso(),
            ),
        )

    def get(self, workspace_id: str, kind: str, owner_id: int, *, conn: Any = None) -> dict[str, Any] | None:
        """Read the current exact owner if it still belongs to this workspace."""
        query = _VISIBLE_RECEIPTS + " AND r.output_kind = ? AND r.owner_id = ?"
        params = (*self._scope(workspace_id), kind, owner_id)
        cursor = self.db.execute_query(query, params, read_only=True) if conn is None else conn.execute(query, params)
        row = cursor.fetchone()
        return dict(row) if row else None

    def page(self, workspace_id: str, *, limit: int, offset: int) -> tuple[list[dict[str, Any]], int]:
        """List bounded output rows and a total from the same statement snapshot."""
        rows = self.db.execute_query(
            _PAGED_RECEIPTS, (*self._scope(workspace_id), limit, offset), read_only=True
        ).fetchall()
        return [dict(row) for row in rows if row["owner_id"] is not None], int(rows[0]["total"])
