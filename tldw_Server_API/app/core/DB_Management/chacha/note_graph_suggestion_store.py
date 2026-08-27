"""Owner-bound SQL seam for Notes graph suggestion persistence."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from ..ChaChaNotes_DB import BackendConnectionWrapper, BackendType

if TYPE_CHECKING:
    from ..ChaChaNotes_DB import CharactersRAGDB


SuggestionConnection = sqlite3.Connection | BackendConnectionWrapper
SuggestionReadT = TypeVar("SuggestionReadT")


class NotesGraphSourceTooLargeError(ValueError):
    """Raised when a source note exceeds the non-truncating analysis limit."""


class NotesGraphFTSNotReadyError(RuntimeError):
    """Raised when Notes FTS structures are unavailable or structurally incomplete."""


@dataclass(frozen=True, slots=True)
class SuggestionNoteRecord:
    """A bounded owner-scoped note payload returned after a SQL byte predicate."""

    note_id: str
    title: str
    content: str
    version: int
    last_modified: str


class NoteGraphSuggestionStore:
    """Own the database boundary for future suggestion reads and transitions."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db

    @property
    def owner_user_id(self) -> str:
        """Return the ChaChaNotes owner bound to this store instance."""
        return str(self._db.client_id)

    @property
    def is_postgres(self) -> bool:
        """Return whether this store is backed by PostgreSQL."""
        return self._db.backend_type == BackendType.POSTGRESQL

    def _scope(self, dataset_id: str) -> str:
        dataset = str(dataset_id).strip()
        if not dataset:
            raise ValueError("dataset_id cannot be empty")
        return dataset

    def _deleted_value(self) -> bool | int:
        return False if self.is_postgres else 0

    def _with_dataset_scope(
        self,
        dataset_id: str,
        fn: Callable[[SuggestionConnection], SuggestionReadT],
    ) -> SuggestionReadT:
        if not self.is_postgres:
            with self._db.transaction() as conn:
                return fn(conn)
        with self._db.transaction() as conn:
            conn.execute("SELECT set_config('app.current_dataset_id', ?, true)", (dataset_id,))
            return fn(conn)

    def _source_byte_expression(self) -> str:
        if self.is_postgres:
            return "octet_length(COALESCE(n.title, '')) + octet_length(COALESCE(n.content, ''))"
        return "length(CAST(COALESCE(n.title, '') AS BLOB)) + length(CAST(COALESCE(n.content, '') AS BLOB))"

    def _ensure_fts_ready(self, conn: SuggestionConnection) -> None:
        if self.is_postgres:
            row = conn.execute(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND table_name = 'notes' "
                "AND column_name = 'notes_fts_tsv'"
            ).fetchone()
            if row is None:
                raise NotesGraphFTSNotReadyError("notes_graph_fts_not_ready")
            return
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'trigger') "
            "AND name IN ('notes_fts', 'notes_ai', 'notes_au', 'notes_ad')"
        ).fetchall()
        if {str(row["name"]) for row in rows} != {"notes_fts", "notes_ai", "notes_au", "notes_ad"}:
            raise NotesGraphFTSNotReadyError("notes_graph_fts_not_ready")

    def load_source_note(self, *, dataset_id: str, note_id: str) -> SuggestionNoteRecord:
        """Load one active owned source only after its SQL byte-limit predicate passes."""

        dataset = self._scope(dataset_id)
        max_bytes = 1_000_000

        def read(conn: SuggestionConnection) -> SuggestionNoteRecord:
            byte_expression = self._source_byte_expression()
            row = conn.execute(
                "SELECT n.id, n.title, n.content, n.version, n.last_modified FROM notes n "
                f"WHERE n.id = ? AND n.client_id = ? AND n.deleted = ? AND ({byte_expression}) <= ?",  # nosec B608
                (note_id, self.owner_user_id, self._deleted_value(), max_bytes),
            ).fetchone()
            if row is not None:
                return SuggestionNoteRecord(
                    note_id=str(row["id"]),
                    title=str(row["title"] or ""),
                    content=str(row["content"] or ""),
                    version=int(row["version"]),
                    last_modified=str(row["last_modified"]),
                )
            size = conn.execute(
                f"SELECT ({byte_expression}) AS byte_count FROM notes n "  # nosec B608
                "WHERE n.id = ? AND n.client_id = ? AND n.deleted = ?",
                (note_id, self.owner_user_id, self._deleted_value()),
            ).fetchone()
            if size is not None and int(size["byte_count"]) > max_bytes:
                raise NotesGraphSourceTooLargeError("notes_graph_source_too_large")
            raise ValueError("Notes graph source is unavailable")

        return self._with_dataset_scope(dataset, read)

    def fetch_ranked_candidates(
        self,
        *,
        dataset_id: str,
        source_note_id: str,
        terms: tuple[str, ...],
        source_fingerprint: str,
        limit: int,
    ) -> tuple[tuple[SuggestionNoteRecord, ...], int]:
        """Return at most 60 FTS-ranked byte-safe candidates and an oversized aggregate."""

        del source_fingerprint
        dataset = self._scope(dataset_id)
        if not terms or not 1 <= len(terms) <= 24 or not 1 <= limit <= 60:
            return (), 0

        def read(conn: SuggestionConnection) -> tuple[tuple[SuggestionNoteRecord, ...], int]:
            self._ensure_fts_ready(conn)
            byte_expression = self._source_byte_expression()
            direct_exclusion = (
                "NOT EXISTS (SELECT 1 FROM note_edges edge WHERE edge.user_id = ? "
                "AND edge.deleted = ? AND ((edge.from_note_id = ? AND edge.to_note_id = n.id) "
                "OR (edge.to_note_id = ? AND edge.from_note_id = n.id))) "
                "AND NOT EXISTS (SELECT 1 FROM note_wikilink_edges edge WHERE "
                "((edge.source_note_id = ? AND edge.target_note_id = n.id) "
                "OR (edge.target_note_id = ? AND edge.source_note_id = n.id))"
            )
            direct_params: tuple[object, ...] = (
                self.owner_user_id,
                self._deleted_value(),
                source_note_id,
                source_note_id,
                source_note_id,
                source_note_id,
            )
            if self.is_postgres:
                direct_exclusion += " AND edge.owner_user_id = ?"
                direct_params += (self.owner_user_id,)
            direct_exclusion += ")"
            if self.is_postgres:
                tsquery = " | ".join(terms)
                select_sql = (
                    "SELECT n.id, n.title, n.content, n.version, n.last_modified "
                    "FROM notes n WHERE n.client_id = ? AND n.deleted = ? AND n.id <> ? "
                    "AND n.notes_fts_tsv @@ to_tsquery('english', ?) "
                    f"AND ({byte_expression}) <= ? AND {direct_exclusion} "  # nosec B608
                    "ORDER BY ts_rank(n.notes_fts_tsv, to_tsquery('english', ?)) DESC, n.id ASC LIMIT ?"
                )
                count_sql = (
                    "SELECT COUNT(*) AS count FROM notes n WHERE n.client_id = ? AND n.deleted = ? "
                    "AND n.id <> ? AND n.notes_fts_tsv @@ to_tsquery('english', ?) "
                    f"AND ({byte_expression}) > ? AND {direct_exclusion}"  # nosec B608
                )
                prefix = (self.owner_user_id, self._deleted_value(), source_note_id, tsquery)
                rows = conn.execute(select_sql, (*prefix, 250_000, *direct_params, tsquery, limit)).fetchall()
                count_row = conn.execute(count_sql, (*prefix, 250_000, *direct_params)).fetchone()
            else:
                fts_query = " OR ".join(f'"{term}"' for term in terms)
                select_sql = (
                    "SELECT n.id, n.title, n.content, n.version, n.last_modified "
                    "FROM notes_fts JOIN notes n ON notes_fts.rowid = n.rowid "
                    "WHERE notes_fts MATCH ? AND n.client_id = ? AND n.deleted = ? AND n.id <> ? "
                    f"AND ({byte_expression}) <= ? AND {direct_exclusion} "  # nosec B608
                    "ORDER BY bm25(notes_fts) ASC, n.id ASC LIMIT ?"
                )
                count_sql = (
                    "SELECT COUNT(*) AS count FROM notes_fts JOIN notes n ON notes_fts.rowid = n.rowid "
                    "WHERE notes_fts MATCH ? AND n.client_id = ? AND n.deleted = ? AND n.id <> ? "
                    f"AND ({byte_expression}) > ? AND {direct_exclusion}"  # nosec B608
                )
                prefix = (fts_query, self.owner_user_id, self._deleted_value(), source_note_id)
                rows = conn.execute(select_sql, (*prefix, 250_000, *direct_params, limit)).fetchall()
                count_row = conn.execute(count_sql, (*prefix, 250_000, *direct_params)).fetchone()
            return (
                tuple(
                    SuggestionNoteRecord(
                        note_id=str(row["id"]),
                        title=str(row["title"] or ""),
                        content=str(row["content"] or ""),
                        version=int(row["version"]),
                        last_modified=str(row["last_modified"]),
                    )
                    for row in rows
                ),
                int(count_row["count"]),
            )

        return self._with_dataset_scope(dataset, read)

    def list_rejected_candidate_fingerprints(
        self,
        *,
        dataset_id: str,
        source_note_id: str,
        source_fingerprint: str,
    ) -> frozenset[tuple[str, str]]:
        """Return only exact current-fingerprint relationship rejections for one scope."""

        dataset = self._scope(dataset_id)

        def read(conn: SuggestionConnection) -> frozenset[tuple[str, str]]:
            rows = conn.execute(
                "SELECT target_note_id, target_fingerprint FROM note_graph_suggestions "
                "WHERE owner_user_id = ? AND dataset_id = ? AND source_note_id = ? "
                "AND source_fingerprint = ? AND kind = 'related_note' AND state = 'rejected'",
                (self.owner_user_id, dataset, source_note_id, source_fingerprint),
            ).fetchall()
            return frozenset((str(row["target_note_id"]), str(row["target_fingerprint"])) for row in rows)

        return self._with_dataset_scope(dataset, read)

    def list_tag_catalog(self, *, dataset_id: str, terms: tuple[str, ...], limit: int) -> tuple[str, ...]:
        """Return at most 100 active owner tags with parameterized overlap predicates."""

        dataset = self._scope(dataset_id)
        if not terms or not 1 <= limit <= 100:
            return ()
        predicates = " OR ".join("LOWER(k.keyword) LIKE ?" for _ in terms)
        keyword_table = self._db._map_table_for_backend("keywords")
        params: tuple[object, ...] = (
            self.owner_user_id,
            self._deleted_value(),
            *(f"%{term}%" for term in terms),
            limit,
        )

        def read(conn: SuggestionConnection) -> tuple[str, ...]:
            rows = conn.execute(
                f"SELECT k.keyword FROM {keyword_table} k WHERE k.client_id = ? AND k.deleted = ? "  # nosec B608
                f"AND ({predicates}) ORDER BY LOWER(k.keyword) ASC LIMIT ?",  # nosec B608
                params,
            ).fetchall()
            return tuple(str(row["keyword"]) for row in rows)

        return self._with_dataset_scope(dataset, read)

    def is_projection_fresh(self, *, dataset_id: str, note_id: str) -> bool:
        """Report whether the source wikilink projection matches its current note version."""

        dataset = self._scope(dataset_id)

        def read(conn: SuggestionConnection) -> bool:
            query = (
                "SELECT state.source_version, n.version FROM note_graph_note_state state "
                "JOIN notes n ON n.id = state.note_id WHERE state.note_id = ? AND n.client_id = ?"
            )
            params: tuple[object, ...] = (note_id, self.owner_user_id)
            if self.is_postgres:
                query += " AND state.owner_user_id = ?"
                params += (self.owner_user_id,)
            row = conn.execute(query, params).fetchone()
            return row is not None and int(row["source_version"]) == int(row["version"])

        return self._with_dataset_scope(dataset, read)
