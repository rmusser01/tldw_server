"""``note_ids_with_note_links`` on PostgreSQL (NL-11 review item 3, #3101).

The owner clauses differ per backend, so the Postgres branch gets its own
run: manual links and [[links]] count either way, a trashed partner and a
tag do not, and another owner's notes never answer for this owner.
"""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration


def test_postgres_note_link_status(pg_database_config: DatabaseConfig) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(db_path=":memory:", client_id="940001", backend=backend)
    other = CharactersRAGDB(db_path=":memory:", client_id="940002", backend=backend)
    try:
        target = db.add_note("Target", "plain")
        source = db.add_note("Source", f"[[id:{target}]]")
        first = db.add_note("First", "one")
        second = db.add_note("Second", "two")
        db.create_manual_note_edge(
            user_id="940001",
            from_note_id=first,
            to_note_id=second,
            directed=False,
            weight=1.0,
            created_by="940001",
        )
        tagged = db.add_note("Tagged", "alone")
        db.link_note_to_keyword(tagged, db.add_keyword("loose"))
        lonely = db.add_note("Lonely", "plain")
        trashed = db.add_note("Trashed", "gone")
        db.update_note(lonely, {"content": f"[[id:{trashed}]]"}, expected_version=1)
        db.soft_delete_note(trashed, expected_version=1)
        foreign = other.add_note("Foreign", f"[[id:{tagged}]]")

        linked = db.note_graph_projection_store.note_ids_with_note_links(
            [target, source, first, second, tagged, lonely, foreign]
        )

        assert linked == {target, source, first, second}
    finally:
        db.close_all_connections()
        other.close_all_connections()
