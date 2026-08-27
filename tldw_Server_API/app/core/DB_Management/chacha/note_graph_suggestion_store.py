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


class NotesGraphDatasetScopeError(RuntimeError):
    """Raised when an owner is not authorized for the requested Notes dataset."""


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
            raise NotesGraphDatasetScopeError("notes_graph_dataset_scope_invalid")
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
                self._require_dataset_scope(conn, dataset_id)
                return fn(conn)
        with self._db.transaction() as conn:
            conn.execute("SELECT set_config('app.current_dataset_id', ?, true)", (dataset_id,))
            self._require_dataset_scope(conn, dataset_id)
            return fn(conn)

    def _require_dataset_scope(self, conn: SuggestionConnection, dataset_id: str) -> None:
        row = conn.execute(
            "SELECT 1 FROM note_task_scope_authority "
            "WHERE owner_user_id = ? AND dataset_id = ?",
            (self.owner_user_id, dataset_id),
        ).fetchone()
        if row is None:
            raise NotesGraphDatasetScopeError("notes_graph_dataset_scope_invalid")

    def _source_byte_expression(self) -> str:
        if self.is_postgres:
            return "octet_length(COALESCE(n.title, '')) + octet_length(COALESCE(n.content, ''))"
        return "length(CAST(COALESCE(n.title, '') AS BLOB)) + length(CAST(COALESCE(n.content, '') AS BLOB))"

    @staticmethod
    def _normalized_sql(value: str) -> str:
        return " ".join(value.lower().split()).rstrip(";")

    def _ensure_fts_ready(self, conn: SuggestionConnection) -> None:
        if self.is_postgres:
            row = conn.execute(
                "SELECT data_type, udt_name FROM information_schema.columns "
                "WHERE table_schema = current_schema() AND table_name = 'notes' "
                "AND column_name = 'notes_fts_tsv'"
            ).fetchone()
            if row is None or row["data_type"] != "tsvector" or row["udt_name"] != "tsvector":
                raise NotesGraphFTSNotReadyError("notes_graph_fts_not_ready")
            trigger = conn.execute(
                "SELECT trigger_row.tgtype, function_row.proname, function_row.prosrc "
                "FROM pg_trigger trigger_row "
                "JOIN pg_proc function_row ON function_row.oid = trigger_row.tgfoid "
                "WHERE trigger_row.tgrelid = 'notes'::regclass "
                "AND trigger_row.tgname = 'update_notes_fts_tsv_trigger' "
                "AND NOT trigger_row.tgisinternal"
            ).fetchone()
            expected_function = self._normalized_sql(
                """
                BEGIN
                    NEW."notes_fts_tsv" := to_tsvector('english', coalesce(NEW."title", '') || ' ' || coalesce(NEW."content", ''));
                    RETURN NEW;
                END;
                """
            )
            if (
                trigger is None
                or int(trigger["tgtype"]) != 23
                or str(trigger["proname"]) != "update_notes_fts_tsv_function"
                or self._normalized_sql(str(trigger["prosrc"])) != expected_function
            ):
                raise NotesGraphFTSNotReadyError("notes_graph_fts_not_ready")
            index = conn.execute(
                "SELECT access_method.amname, index_row.indisvalid, "
                "array_agg(attribute_row.attname ORDER BY key_column.ordinality) AS columns "
                "FROM pg_index index_row "
                "JOIN pg_class index_relation ON index_relation.oid = index_row.indexrelid "
                "JOIN pg_class table_relation ON table_relation.oid = index_row.indrelid "
                "JOIN pg_am access_method ON access_method.oid = index_relation.relam "
                "JOIN unnest(index_row.indkey) WITH ORDINALITY AS key_column(attnum, ordinality) ON true "
                "JOIN pg_attribute attribute_row ON attribute_row.attrelid = table_relation.oid "
                "AND attribute_row.attnum = key_column.attnum "
                "WHERE table_relation.oid = 'notes'::regclass "
                "AND index_relation.relname = 'idx_notes_notes_fts_tsv' "
                "GROUP BY access_method.amname, index_row.indisvalid"
            ).fetchone()
            if (
                index is None
                or str(index["amname"]) != "gin"
                or not bool(index["indisvalid"])
                or tuple(index["columns"]) != ("notes_fts_tsv",)
            ):
                raise NotesGraphFTSNotReadyError("notes_graph_fts_not_ready")
            return
        rows = conn.execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master WHERE type IN ('table', 'trigger') "
            "AND name IN ('notes_fts', 'notes_ai', 'notes_au', 'notes_ad')"
        ).fetchall()
        by_name = {str(row["name"]): row for row in rows}
        table_definition, trigger_definitions = self._db._notes_fts_sqlite_contract()
        expected_trigger_definitions = {
            definition.split()[2]: self._normalized_sql(definition)
            for definition in trigger_definitions
        }
        table = by_name.get("notes_fts")
        if (
            table is None
            or str(table["type"]) != "table"
            or self._normalized_sql(str(table["sql"] or ""))
            != self._normalized_sql(table_definition)
        ):
            raise NotesGraphFTSNotReadyError("notes_graph_fts_not_ready")
        if set(by_name) != {"notes_fts", *expected_trigger_definitions}:
            raise NotesGraphFTSNotReadyError("notes_graph_fts_not_ready")
        for trigger_name, expected_definition in expected_trigger_definitions.items():
            trigger = by_name.get(trigger_name)
            if (
                trigger is None
                or str(trigger["type"]) != "trigger"
                or str(trigger["tbl_name"]) != "notes"
                or self._normalized_sql(str(trigger["sql"] or "")) != expected_definition
            ):
                raise NotesGraphFTSNotReadyError("notes_graph_fts_not_ready")
        columns = conn.execute("PRAGMA table_info(notes_fts)").fetchall()
        if tuple(str(column["name"]) for column in columns) != ("title", "content"):
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
    ) -> tuple[tuple[SuggestionNoteRecord, ...], int, int]:
        """Return at most 60 FTS-ranked byte-safe candidates and an oversized aggregate."""

        del source_fingerprint
        dataset = self._scope(dataset_id)
        if not terms or not 1 <= len(terms) <= 24 or not 1 <= limit <= 60:
            return (), 0, 0

        def read(conn: SuggestionConnection) -> tuple[tuple[SuggestionNoteRecord, ...], int, int]:
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
                ranked_sql = (
                    "WITH ranked AS MATERIALIZED ("
                    "SELECT n.id AS note_id, ts_rank(n.notes_fts_tsv, to_tsquery('english', ?)) AS rank_value "
                    "FROM notes n WHERE n.client_id = ? AND n.deleted = ? AND n.id <> ? "
                    "AND n.notes_fts_tsv @@ to_tsquery('english', ?) "
                    "ORDER BY rank_value DESC, n.id ASC LIMIT ?"
                    ") SELECT ranked.note_id, "
                    f"({byte_expression}) AS byte_count, "  # nosec B608
                    f"CASE WHEN {direct_exclusion} THEN 0 ELSE 1 END AS direct_connected "  # nosec B608
                    "FROM ranked JOIN notes n ON n.id = ranked.note_id "
                    "ORDER BY ranked.rank_value DESC, ranked.note_id ASC"
                )
                ranked_rows = conn.execute(
                    ranked_sql,
                    (tsquery, self.owner_user_id, self._deleted_value(), source_note_id, tsquery, limit, *direct_params),
                ).fetchall()
            else:
                fts_query = " OR ".join(f'"{term}"' for term in terms)
                ranked_sql = (
                    "WITH ranked AS MATERIALIZED ("
                    "SELECT n.id AS note_id, bm25(notes_fts) AS rank_value "
                    "FROM notes_fts JOIN notes n ON notes_fts.rowid = n.rowid "
                    "WHERE notes_fts MATCH ? AND n.client_id = ? AND n.deleted = ? AND n.id <> ? "
                    "ORDER BY rank_value ASC, n.id ASC LIMIT ?"
                    ") SELECT ranked.note_id, "
                    f"({byte_expression}) AS byte_count, "  # nosec B608
                    f"CASE WHEN {direct_exclusion} THEN 0 ELSE 1 END AS direct_connected "  # nosec B608
                    "FROM ranked JOIN notes n ON n.id = ranked.note_id "
                    "ORDER BY ranked.rank_value ASC, ranked.note_id ASC"
                )
                ranked_rows = conn.execute(
                    ranked_sql,
                    (fts_query, self.owner_user_id, self._deleted_value(), source_note_id, limit, *direct_params),
                ).fetchall()

            oversized_count = sum(
                int(row["byte_count"]) > 250_000 and not bool(row["direct_connected"])
                for row in ranked_rows
            )
            eligible_ids = tuple(
                str(row["note_id"])
                for row in ranked_rows
                if int(row["byte_count"]) <= 250_000 and not bool(row["direct_connected"])
            )
            if not eligible_ids:
                return (), oversized_count, len(ranked_rows)
            placeholders = ", ".join("?" for _ in eligible_ids)
            payload_rows = conn.execute(
                "SELECT n.id, n.title, n.content, n.version, n.last_modified FROM notes n "
                f"WHERE n.id IN ({placeholders}) AND n.client_id = ? AND n.deleted = ? "  # nosec B608
                f"AND ({byte_expression}) <= ?",  # nosec B608
                (*eligible_ids, self.owner_user_id, self._deleted_value(), 250_000),
            ).fetchall()
            records_by_id = {
                str(row["id"]): SuggestionNoteRecord(
                    note_id=str(row["id"]),
                    title=str(row["title"] or ""),
                    content=str(row["content"] or ""),
                    version=int(row["version"]),
                    last_modified=str(row["last_modified"]),
                )
                for row in payload_rows
            }
            return (
                tuple(records_by_id[note_id] for note_id in eligible_ids if note_id in records_by_id),
                oversized_count,
                len(ranked_rows),
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
