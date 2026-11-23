from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.chacha import sqlite_schema


pytestmark = pytest.mark.unit


def _build_legacy_flashcards_db_with_stale_fts(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE flashcards(
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              uuid TEXT UNIQUE NOT NULL,
              deck_id INTEGER,
              front TEXT NOT NULL,
              back TEXT NOT NULL,
              notes TEXT,
              is_cloze BOOLEAN NOT NULL DEFAULT 0,
              tags_json TEXT,
              source_ref_type TEXT DEFAULT 'manual',
              source_ref_id TEXT,
              ef REAL NOT NULL DEFAULT 2.5,
              interval_days INTEGER NOT NULL DEFAULT 0,
              repetitions INTEGER NOT NULL DEFAULT 0,
              lapses INTEGER NOT NULL DEFAULT 0,
              due_at DATETIME,
              last_reviewed_at DATETIME,
              created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              deleted BOOLEAN NOT NULL DEFAULT 0,
              client_id TEXT NOT NULL DEFAULT 'unknown',
              version INTEGER NOT NULL DEFAULT 1,
              queue_state TEXT NOT NULL DEFAULT 'new',
              step_index INTEGER,
              suspended_reason TEXT
            );

            CREATE VIRTUAL TABLE flashcards_fts
            USING fts5(front, back, notes, content='flashcards', content_rowid='id');

            CREATE TRIGGER flashcards_ai
            AFTER INSERT ON flashcards BEGIN
              INSERT INTO flashcards_fts(rowid,front,back,notes)
              SELECT new.id,new.front,new.back,new.notes
              WHERE new.deleted = 0;
            END;

            CREATE TRIGGER flashcards_au
            AFTER UPDATE ON flashcards BEGIN
              INSERT INTO flashcards_fts(flashcards_fts,rowid,front,back,notes)
              VALUES('delete',old.id,old.front,old.back,old.notes);

              INSERT INTO flashcards_fts(rowid,front,back,notes)
              SELECT new.id,new.front,new.back,new.notes
              WHERE new.deleted = 0;
            END;

            CREATE TRIGGER flashcards_ad
            AFTER DELETE ON flashcards BEGIN
              INSERT INTO flashcards_fts(flashcards_fts,rowid,front,back,notes)
              VALUES('delete',old.id,old.front,old.back,old.notes);
            END;
            """
        )
        conn.execute(
            """
            INSERT INTO flashcards(
              uuid, front, back, notes, queue_state, created_at, last_modified, client_id
            )
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
            """,
            ("card-1", "Front text", "Back text", "Note text", "new", "user-1"),
        )
        conn.commit()
    finally:
        conn.close()


def _make_db_for_helpers(db_path: Path) -> CharactersRAGDB:
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db.db_path = db_path
    db.db_path_str = str(db_path)
    return db


def test_flashcard_asset_schema_repairs_legacy_fts_before_scheduler_style_updates(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy-flashcards-fts.db"
    _build_legacy_flashcards_db_with_stale_fts(db_path)

    db = _make_db_for_helpers(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        sqlite_schema._ensure_flashcard_asset_schema_sqlite(db.ctx, conn)

        flashcard_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info('flashcards')")
        }
        assert {"front_search", "back_search", "notes_search"}.issubset(flashcard_columns)

        conn.execute("UPDATE flashcards SET queue_state = queue_state WHERE id = 1")

        trigger_sql = {
            row["name"]: row["sql"]
            for row in conn.execute(
                """
                SELECT name, sql
                  FROM sqlite_master
                 WHERE type = 'trigger'
                   AND name IN ('flashcards_ai', 'flashcards_au', 'flashcards_ad')
                """
            )
        }
        assert "front_search" in trigger_sql["flashcards_au"]
