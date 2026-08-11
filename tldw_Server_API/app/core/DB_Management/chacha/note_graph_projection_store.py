"""Persistence for bounded, local-only Notes graph projections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.Notes.wikilinks import WikilinkProjection

from ..ChaChaNotes_DB import BackendConnectionWrapper, BackendType

if TYPE_CHECKING:
    import sqlite3

    from ..ChaChaNotes_DB import CharactersRAGDB


@dataclass(frozen=True, slots=True)
class DirtyProjection:
    note_id: str
    generation: int


@dataclass(frozen=True, slots=True)
class NoteProjectionState:
    note_id: str
    source_version: int
    parser_version: int
    truncated: bool


@dataclass(frozen=True, slots=True)
class ProjectionStatus:
    parser_version: int
    rebuild_state: str
    rebuild_cursor: str | None


@dataclass(frozen=True, slots=True)
class WikilinkProjectionEdge:
    source_note_id: str
    target_note_id: str


class NoteGraphProjectionStore:
    """Owner-bound projection, dirty-generation, and graph-revision store."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db

    @property
    def _postgres(self) -> bool:
        return self._db.backend_type == BackendType.POSTGRESQL

    def claim_dirty(
        self,
        *,
        limit: int,
        conn: sqlite3.Connection | BackendConnectionWrapper | None = None,
    ) -> tuple[DirtyProjection, ...]:
        if not 1 <= limit <= 1_000:
            raise ValueError("limit must be between 1 and 1000")
        query = "SELECT note_id, generation FROM note_graph_dirty"
        params: tuple[object, ...]
        if self._postgres:
            query += " WHERE owner_user_id = ? ORDER BY note_id LIMIT ? FOR UPDATE SKIP LOCKED"
            params = (self._db.client_id, limit)
        else:
            query += " ORDER BY note_id LIMIT ?"
            params = (limit,)

        def execute(inner_conn: sqlite3.Connection | BackendConnectionWrapper):
            return tuple(
                DirtyProjection(str(row["note_id"]), int(row["generation"]))
                for row in inner_conn.execute(query, params).fetchall()
            )

        if conn is not None:
            return execute(conn)
        with self._db.transaction() as transaction_conn:
            return execute(transaction_conn)

    def replace_projection(
        self,
        *,
        note_id: str,
        source_version: int,
        projection: WikilinkProjection,
        claimed_generation: int | None = None,
        parser_version: int | None = None,
        bump_revision: bool = False,
        conn: sqlite3.Connection | BackendConnectionWrapper | None = None,
    ) -> bool:
        effective_parser_version = parser_version or projection.parser_version
        if source_version < 1 or effective_parser_version < 1:
            raise ValueError("source and parser versions must be positive")

        def execute(inner_conn: sqlite3.Connection | BackendConnectionWrapper) -> bool:
            owner_clause = " AND owner_user_id = ?" if self._postgres else ""
            delete_params: tuple[object, ...] = (note_id, self._db.client_id) if self._postgres else (note_id,)
            inner_conn.execute(
                f"DELETE FROM note_wikilink_edges WHERE source_note_id = ?{owner_clause}",  # nosec B608
                delete_params,
            )
            for target_note_id in projection.target_note_ids:
                if self._postgres:
                    inner_conn.execute(
                        "INSERT INTO note_wikilink_edges (owner_user_id, source_note_id, "
                        "target_note_id, source_version, parser_version) VALUES (?, ?, ?, ?, ?)",
                        (
                            self._db.client_id,
                            note_id,
                            target_note_id,
                            source_version,
                            effective_parser_version,
                        ),
                    )
                else:
                    inner_conn.execute(
                        "INSERT INTO note_wikilink_edges (source_note_id, target_note_id, "
                        "source_version, parser_version) VALUES (?, ?, ?, ?)",
                        (note_id, target_note_id, source_version, effective_parser_version),
                    )
            if self._postgres:
                inner_conn.execute(
                    "INSERT INTO note_graph_note_state (owner_user_id, note_id, source_version, "
                    "parser_version, truncated, updated_at) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP) "
                    "ON CONFLICT(owner_user_id, note_id) DO UPDATE SET "
                    "source_version = excluded.source_version, parser_version = excluded.parser_version, "
                    "truncated = excluded.truncated, updated_at = CURRENT_TIMESTAMP",
                    (
                        self._db.client_id,
                        note_id,
                        source_version,
                        effective_parser_version,
                        projection.truncated,
                    ),
                )
            else:
                inner_conn.execute(
                    "INSERT INTO note_graph_note_state (note_id, source_version, parser_version, "
                    "truncated, updated_at) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP) "
                    "ON CONFLICT(note_id) DO UPDATE SET source_version = excluded.source_version, "
                    "parser_version = excluded.parser_version, truncated = excluded.truncated, "
                    "updated_at = CURRENT_TIMESTAMP",
                    (
                        note_id,
                        source_version,
                        effective_parser_version,
                        int(projection.truncated),
                    ),
                )
            cleared = self._clear_dirty(
                inner_conn,
                note_id=note_id,
                claimed_generation=claimed_generation,
            )
            if bump_revision:
                self._bump_revision(inner_conn)
            return cleared

        if conn is not None:
            return execute(conn)
        with self._db.transaction() as transaction_conn:
            return execute(transaction_conn)

    def mark_lifecycle(
        self,
        *,
        note_id: str,
        source_version: int,
        conn: sqlite3.Connection | BackendConnectionWrapper,
    ) -> None:
        if self._postgres:
            cursor = conn.execute(
                "UPDATE note_graph_note_state SET source_version = ?, updated_at = CURRENT_TIMESTAMP "
                "WHERE owner_user_id = ? AND note_id = ?",
                (source_version, self._db.client_id, note_id),
            )
        else:
            cursor = conn.execute(
                "UPDATE note_graph_note_state SET source_version = ?, updated_at = CURRENT_TIMESTAMP WHERE note_id = ?",
                (source_version, note_id),
            )
        if cursor.rowcount > 0:
            self._clear_dirty(conn, note_id=note_id, claimed_generation=None)

    def list_outgoing(self, note_id: str) -> tuple[str, ...]:
        query = "SELECT target_note_id FROM note_wikilink_edges WHERE source_note_id = ?"
        params: tuple[object, ...] = (note_id,)
        if self._postgres:
            query += " AND owner_user_id = ?"
            params += (self._db.client_id,)
        query += " ORDER BY target_note_id"
        return tuple(str(row[0]) for row in self._db.execute_query(query, params).fetchall())

    def list_live_outgoing(self, note_id: str) -> tuple[str, ...]:
        query = (
            "SELECT edge.target_note_id FROM note_wikilink_edges edge "
            "JOIN notes source ON source.id = edge.source_note_id "
            "JOIN notes target ON target.id = edge.target_note_id "
            "WHERE edge.source_note_id = ? AND source.deleted = ? AND target.deleted = ?"
        )
        params: tuple[object, ...] = (
            note_id,
            False if self._postgres else 0,
            False if self._postgres else 0,
        )
        if self._postgres:
            query += " AND edge.owner_user_id = ? AND source.client_id = ? AND target.client_id = ?"
            params += (self._db.client_id,) * 3
        query += " ORDER BY edge.target_note_id"
        return tuple(str(row[0]) for row in self._db.execute_query(query, params).fetchall())

    def list_live_edges_for_notes(
        self,
        note_ids: tuple[str, ...] | list[str],
    ) -> tuple[WikilinkProjectionEdge, ...]:
        """Return live projected edges touching a bounded set of live notes."""

        normalized = tuple(dict.fromkeys(str(note_id) for note_id in note_ids))
        if not normalized:
            return ()
        if len(normalized) > 1_000:
            raise ValueError("note graph projection query is limited to 1000 note IDs")
        results: set[tuple[str, str]] = set()
        for offset in range(0, len(normalized), 400):
            batch = normalized[offset : offset + 400]
            placeholders = ",".join("?" for _ in batch)
            query = (
                "SELECT edge.source_note_id, edge.target_note_id "
                "FROM note_wikilink_edges edge "
                "JOIN notes source ON source.id = edge.source_note_id "
                "JOIN notes target ON target.id = edge.target_note_id "
                f"WHERE (edge.source_note_id IN ({placeholders}) OR "  # nosec B608
                f"edge.target_note_id IN ({placeholders})) "  # nosec B608
                "AND source.deleted = ? AND target.deleted = ?"
            )
            params: list[object] = [*batch, *batch]
            params.extend((False if self._postgres else 0,) * 2)
            if self._postgres:
                query += " AND edge.owner_user_id = ? AND source.client_id = ? AND target.client_id = ?"
                params.extend((self._db.client_id,) * 3)
            for row in self._db.execute_query(query, tuple(params)).fetchall():
                results.add((str(row[0]), str(row[1])))
        return tuple(WikilinkProjectionEdge(*edge) for edge in sorted(results))

    def list_orphan_note_ids(
        self,
        *,
        after_note_id: str | None,
        limit: int,
    ) -> tuple[str, ...]:
        """List live notes without live manual or projected note relationships."""

        if not 1 <= limit <= 201:
            raise ValueError("orphan limit must be between 1 and 201")
        live = False if self._postgres else 0
        note_owner_clause = " AND note.client_id = ?" if self._postgres else ""
        query = (
            "SELECT note.id FROM notes note "
            f"WHERE note.deleted = ? AND note.id > ?{note_owner_clause} "  # nosec B608
            "AND NOT EXISTS ("
            "SELECT 1 FROM note_edges manual "
            "JOIN notes other ON other.id = manual.to_note_id "
            "WHERE manual.deleted = ? AND other.deleted = ? "
            "AND manual.from_note_id = note.id AND manual.user_id = ?"
        )
        params: list[object] = [live, after_note_id or ""]
        if self._postgres:
            params.append(self._db.client_id)
        params.extend((live, live, self._db.client_id))
        if self._postgres:
            query += " AND other.client_id = ?"
            params.append(self._db.client_id)
        query += (
            ") AND NOT EXISTS ("
            "SELECT 1 FROM note_edges manual "
            "JOIN notes other ON other.id = manual.from_note_id "
            "WHERE manual.deleted = ? AND other.deleted = ? "
            "AND manual.to_note_id = note.id AND manual.user_id = ?"
        )
        params.extend((live, live, self._db.client_id))
        if self._postgres:
            query += " AND other.client_id = ?"
            params.append(self._db.client_id)
        query += (
            ") AND NOT EXISTS ("
            "SELECT 1 FROM note_wikilink_edges derived "
            "JOIN notes other ON other.id = derived.target_note_id "
            "WHERE other.deleted = ? AND derived.source_note_id = note.id"
        )
        params.append(live)
        if self._postgres:
            query += " AND derived.owner_user_id = ? AND other.client_id = ?"
            params.extend((self._db.client_id,) * 2)
        query += (
            ") AND NOT EXISTS ("
            "SELECT 1 FROM note_wikilink_edges derived "
            "JOIN notes other ON other.id = derived.source_note_id "
            "WHERE other.deleted = ? AND derived.target_note_id = note.id"
        )
        params.append(live)
        if self._postgres:
            query += " AND derived.owner_user_id = ? AND other.client_id = ?"
            params.extend((self._db.client_id,) * 2)
        query += ") ORDER BY note.id LIMIT ?"
        params.append(limit)
        return tuple(str(row[0]) for row in self._db.execute_query(query, tuple(params)).fetchall())

    def get_note_state(self, note_id: str) -> NoteProjectionState | None:
        query = "SELECT note_id, source_version, parser_version, truncated FROM note_graph_note_state WHERE note_id = ?"
        params: tuple[object, ...] = (note_id,)
        if self._postgres:
            query += " AND owner_user_id = ?"
            params += (self._db.client_id,)
        row = self._db.execute_query(query, params).fetchone()
        if row is None:
            return None
        return NoteProjectionState(
            note_id=str(row["note_id"]),
            source_version=int(row["source_version"]),
            parser_version=int(row["parser_version"]),
            truncated=bool(row["truncated"]),
        )

    def count_dirty(self, *, conn: Any | None = None) -> int:
        query = "SELECT COUNT(*) FROM note_graph_dirty"
        params: tuple[object, ...] = ()
        if self._postgres:
            query += " WHERE owner_user_id = ?"
            params = (self._db.client_id,)
        cursor = conn.execute(query, params) if conn is not None else self._db.execute_query(query, params)
        return int(cursor.fetchone()[0])

    def get_revision(self) -> int:
        if self._postgres:
            row = self._db.execute_query(
                "SELECT revision FROM note_graph_revisions WHERE owner_user_id = ?",
                (self._db.client_id,),
            ).fetchone()
        else:
            row = self._db.execute_query("SELECT revision FROM note_graph_revisions WHERE singleton_id = 1").fetchone()
        return int(row[0]) if row else 0

    def get_projection_status(self) -> ProjectionStatus:
        if self._postgres:
            row = self._db.execute_query(
                "SELECT parser_version, rebuild_state, rebuild_cursor "
                "FROM note_graph_projection_state WHERE owner_user_id = ?",
                (self._db.client_id,),
            ).fetchone()
        else:
            row = self._db.execute_query(
                "SELECT parser_version, rebuild_state, rebuild_cursor "
                "FROM note_graph_projection_state WHERE singleton_id = 1"
            ).fetchone()
        if row is None:
            return ProjectionStatus(1, "ready", None)
        return ProjectionStatus(int(row[0]), str(row[1]), row[2])

    def prepare_rebuild(
        self,
        *,
        parser_version: int,
        conn: sqlite3.Connection | BackendConnectionWrapper,
    ) -> bool:
        status = self._projection_status(conn)
        if status.parser_version == parser_version and status.rebuild_state == "ready":
            return False
        if self._postgres:
            conn.execute(
                "INSERT INTO note_graph_projection_state (owner_user_id, parser_version, rebuild_state, "
                "rebuild_cursor, updated_at) VALUES (?, ?, 'pending', NULL, CURRENT_TIMESTAMP) "
                "ON CONFLICT(owner_user_id) DO UPDATE SET parser_version = excluded.parser_version, "
                "rebuild_state = 'pending', rebuild_cursor = NULL, updated_at = CURRENT_TIMESTAMP",
                (self._db.client_id, parser_version),
            )
        else:
            conn.execute(
                "UPDATE note_graph_projection_state SET parser_version = ?, rebuild_state = 'pending', "
                "rebuild_cursor = NULL, updated_at = CURRENT_TIMESTAMP WHERE singleton_id = 1",
                (parser_version,),
            )
        return True

    def queue_rebuild_page(
        self,
        *,
        limit: int,
        conn: sqlite3.Connection | BackendConnectionWrapper,
    ) -> int:
        if not 1 <= limit <= 1_000:
            raise ValueError("limit must be between 1 and 1000")
        status = self._projection_status(conn)
        query = "SELECT id FROM notes WHERE id > ?"
        params: list[object] = [status.rebuild_cursor or ""]
        if self._postgres:
            query += " AND client_id = ?"
            params.append(self._db.client_id)
        query += " ORDER BY id LIMIT ?"
        params.append(limit)
        note_ids = [str(row[0]) for row in conn.execute(query, tuple(params)).fetchall()]
        for note_id in note_ids:
            self._enqueue_dirty(conn, note_id)
        cursor = note_ids[-1] if note_ids else status.rebuild_cursor
        if self._postgres:
            conn.execute(
                "UPDATE note_graph_projection_state SET rebuild_state = 'running', rebuild_cursor = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE owner_user_id = ?",
                (cursor, self._db.client_id),
            )
        else:
            conn.execute(
                "UPDATE note_graph_projection_state SET rebuild_state = 'running', rebuild_cursor = ?, "
                "updated_at = CURRENT_TIMESTAMP WHERE singleton_id = 1",
                (cursor,),
            )
        return len(note_ids)

    def finish_rebuild_if_idle(
        self,
        *,
        conn: sqlite3.Connection | BackendConnectionWrapper,
    ) -> bool:
        status = self._projection_status(conn)
        if status.rebuild_state not in {"pending", "running"} or self.count_dirty(conn=conn):
            return False
        query = "SELECT 1 FROM notes WHERE id > ?"
        params: tuple[object, ...] = (status.rebuild_cursor or "",)
        if self._postgres:
            query += " AND client_id = ?"
            params += (self._db.client_id,)
        query += " LIMIT 1"
        if conn.execute(query, params).fetchone() is not None:
            return False
        if self._postgres:
            conn.execute(
                "UPDATE note_graph_projection_state SET rebuild_state = 'ready', rebuild_cursor = NULL, "
                "updated_at = CURRENT_TIMESTAMP WHERE owner_user_id = ?",
                (self._db.client_id,),
            )
        else:
            conn.execute(
                "UPDATE note_graph_projection_state SET rebuild_state = 'ready', rebuild_cursor = NULL, "
                "updated_at = CURRENT_TIMESTAMP WHERE singleton_id = 1"
            )
        self._bump_revision(conn)
        return True

    def _projection_status(self, conn: Any) -> ProjectionStatus:
        if self._postgres:
            row = conn.execute(
                "SELECT parser_version, rebuild_state, rebuild_cursor "
                "FROM note_graph_projection_state WHERE owner_user_id = ?",
                (self._db.client_id,),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO note_graph_projection_state (owner_user_id) VALUES (?)",
                    (self._db.client_id,),
                )
                return ProjectionStatus(1, "ready", None)
        else:
            row = conn.execute(
                "SELECT parser_version, rebuild_state, rebuild_cursor "
                "FROM note_graph_projection_state WHERE singleton_id = 1"
            ).fetchone()
        return ProjectionStatus(int(row[0]), str(row[1]), row[2])

    def _enqueue_dirty(self, conn: Any, note_id: str) -> None:
        if self._postgres:
            conn.execute(
                "INSERT INTO note_graph_dirty (owner_user_id, note_id, generation, last_modified) "
                "VALUES (?, ?, 1, CURRENT_TIMESTAMP) ON CONFLICT(owner_user_id, note_id) "
                "DO UPDATE SET generation = note_graph_dirty.generation + 1, "
                "last_modified = CURRENT_TIMESTAMP",
                (self._db.client_id, note_id),
            )
        else:
            conn.execute(
                "INSERT INTO note_graph_dirty (note_id, generation, last_modified) "
                "VALUES (?, 1, CURRENT_TIMESTAMP) ON CONFLICT(note_id) DO UPDATE SET "
                "generation = note_graph_dirty.generation + 1, last_modified = CURRENT_TIMESTAMP",
                (note_id,),
            )

    def _clear_dirty(
        self,
        conn: Any,
        *,
        note_id: str,
        claimed_generation: int | None,
    ) -> bool:
        query = "DELETE FROM note_graph_dirty WHERE note_id = ?"
        params: list[object] = [note_id]
        if self._postgres:
            query += " AND owner_user_id = ?"
            params.append(self._db.client_id)
        if claimed_generation is not None:
            query += " AND generation = ?"
            params.append(claimed_generation)
        cursor = conn.execute(query, tuple(params))
        return cursor.rowcount > 0

    def _bump_revision(self, conn: Any) -> None:
        if self._postgres:
            conn.execute("SELECT notes_graph_bump_revision(?)", (self._db.client_id,))
        else:
            conn.execute(
                "UPDATE note_graph_revisions SET revision = revision + 1, "
                "updated_at = CURRENT_TIMESTAMP WHERE singleton_id = 1"
            )


__all__ = [
    "DirtyProjection",
    "NoteGraphProjectionStore",
    "NoteProjectionState",
    "ProjectionStatus",
    "WikilinkProjectionEdge",
]
