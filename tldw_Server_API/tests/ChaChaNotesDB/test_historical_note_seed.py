"""Historical migration seeds retain real note and derived-index behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.tests.ChaChaNotesDB.historical_note_seed import seed_pre_language_note


@pytest.mark.parametrize(
    ("backend_name", "version"),
    [("sqlite", 58), ("sqlite", 59), ("sqlite", 60), ("postgres", 64)],
    ids=["sqlite58", "sqlite59", "sqlite60", "postgres64"],
)
def test_pre_language_seed_preserves_catalog_source_fts_graph_and_foreign_keys(
    backend_name: str,
    version: int,
    tmp_path: Path,
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = None
    if backend_name == "postgres":
        monkeypatch.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", version)
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    else:
        monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", version)
    db = CharactersRAGDB(str(tmp_path / "historical-note.db"), client_id="seed-owner", backend=backend)
    target_id = "11111111-1111-4111-8111-111111111111"
    source_id = "22222222-2222-4222-8222-222222222222"
    source = f"\ufeffHistoricNeedle [[id:{target_id}]]\r\n"
    try:
        columns = db.backend.get_table_info("notes")
        assert "content_language" not in {column["name"] for column in columns}
        seed_pre_language_note(db, "Target", "Target body", note_id=target_id)
        seed_pre_language_note(db, "Source", source, note_id=source_id)
        row = db.get_note_by_id(source_id)
        assert (row["title"], row["content"], row["client_id"], row["version"], bool(row["deleted"])) == (
            "Source",
            source,
            "seed-owner",
            1,
            False,
        )
        assert db.backend.get_table_info("notes") == columns
        with db.transaction() as conn:
            assert (
                conn.execute(
                    "SELECT version FROM db_schema_version WHERE schema_name = ?", (db._SCHEMA_NAME,)
                ).fetchone()["version"]
                == version
            )
            edge = conn.execute(
                "SELECT target_note_id, source_version FROM note_wikilink_edges WHERE source_note_id = ?",
                (source_id,),
            ).fetchone()
            assert (edge["target_note_id"], edge["source_version"]) == (target_id, 1)
            assert (
                conn.execute(
                    "SELECT source_version FROM note_graph_note_state WHERE note_id = ?", (source_id,)
                ).fetchone()["source_version"]
                == 1
            )
            if backend_name == "postgres":
                match = conn.execute(
                    "SELECT notes_fts_tsv @@ plainto_tsquery('english', ?) AS matched FROM notes WHERE id = ?",
                    ("HistoricNeedle", source_id),
                ).fetchone()
                assert match["matched"] is True
            else:
                match = conn.execute(
                    "SELECT notes.id FROM notes_fts JOIN notes ON notes.rowid = notes_fts.rowid "
                    "WHERE notes_fts MATCH ?",
                    ("HistoricNeedle",),
                ).fetchall()
                assert [item["id"] for item in match] == [source_id]
            conn.execute("DELETE FROM notes WHERE id = ?", (source_id,))
            assert (
                conn.execute(
                    "SELECT source_note_id FROM note_wikilink_edges WHERE source_note_id = ?", (source_id,)
                ).fetchall()
                == []
            )
            assert (
                conn.execute("SELECT note_id FROM note_graph_note_state WHERE note_id = ?", (source_id,)).fetchall()
                == []
            )
    finally:
        db.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()
