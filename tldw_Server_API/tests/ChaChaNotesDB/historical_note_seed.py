"""Seed the reviewed Notes row shape before source-language metadata existed."""

from __future__ import annotations

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Notes.wikilinks import parse_wikilinks


def seed_pre_language_note(
    db: CharactersRAGDB,
    title: str,
    content: str,
    *,
    note_id: str | None = None,
) -> str:
    """Insert the historical row and retain its real FTS/graph side effects."""
    identifier = note_id or db._generate_uuid()
    timestamp = db._get_current_utc_timestamp_iso()
    deleted = False if db.backend_type == BackendType.POSTGRESQL else 0
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO notes (id, title, content, last_modified, client_id, "
            "version, deleted, created_at, conversation_id, message_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (identifier, title, content, timestamp, db.client_id, 1, deleted, timestamp, None, None),
        )
        db.note_graph_projection_store.replace_projection(
            note_id=identifier,
            source_version=1,
            projection=parse_wikilinks(content, source_note_id=identifier),
            conn=conn,
        )
        db.note_graph_projection_store.refresh_title_referrers((title,), conn=conn)
    return identifier
