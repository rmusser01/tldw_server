"""Bounded maintenance for local derived Notes graph projections."""

from __future__ import annotations

from dataclasses import dataclass

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    BackendType,
    CharactersRAGDB,
)
from tldw_Server_API.app.core.Notes.wikilinks import (
    WIKILINK_PARSER_VERSION,
    parse_wikilinks,
)


@dataclass(slots=True)
class NoteGraphProjectionService:
    """Process only one authenticated owner's projection queue."""

    db: CharactersRAGDB
    parser_version: int = WIKILINK_PARSER_VERSION

    def process_dirty(self, *, limit: int = 50) -> int:
        """Project one bounded claimed batch and preserve newer generations."""

        store = self.db.note_graph_projection_store
        processed = 0
        with self.db.transaction() as conn:
            for claim in store.claim_dirty(limit=limit, conn=conn):
                query = "SELECT id, content, version FROM notes WHERE id = ?"
                params: tuple[object, ...] = (claim.note_id,)
                if self.db.backend_type == BackendType.POSTGRESQL:
                    query += " AND client_id = ?"
                    params += (self.db.client_id,)
                row = conn.execute(query, params).fetchone()
                if row is None:
                    continue
                projection = parse_wikilinks(
                    str(row["content"] or ""),
                    source_note_id=str(row["id"]),
                )
                store.replace_projection(
                    note_id=str(row["id"]),
                    source_version=int(row["version"]),
                    projection=projection,
                    claimed_generation=claim.generation,
                    parser_version=self.parser_version,
                    bump_revision=True,
                    conn=conn,
                )
                processed += 1
            store.finish_rebuild_if_idle(conn=conn)
        return processed

    def prepare_rebuild(self) -> bool:
        """Start or reset a parser-version rebuild without global unscoped work."""

        with self.db.transaction() as conn:
            return self.db.note_graph_projection_store.prepare_rebuild(
                parser_version=self.parser_version,
                conn=conn,
            )

    def queue_rebuild_page(self, *, limit: int = 100) -> int:
        """Queue one immutable-ID page so interrupted rebuilds resume safely."""

        with self.db.transaction() as conn:
            count = self.db.note_graph_projection_store.queue_rebuild_page(
                limit=limit,
                conn=conn,
            )
            self.db.note_graph_projection_store.finish_rebuild_if_idle(conn=conn)
            return count


__all__ = ["NoteGraphProjectionService"]
