from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


def _assert_canonical_uuid4(value: str) -> None:
    parsed = uuid.UUID(value)
    assert parsed.version == 4
    assert str(parsed) == value


def _build_v54_fixture(db_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, list[int]]:
    def initialize_historical(db: CharactersRAGDB) -> None:
        with db.transaction() as conn:
            db._apply_schema_v4(conn)
            for prior in range(4, 54):
                db._run_sqlite_linear_migration_step(conn, from_version=prior, target_version=54, initial_version=4)
            assert db._get_db_version(conn) == 54
            tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            assert {"note_folder_sync_suppressions", "note_attachments"}.isdisjoint(tables)
            # Folders were an optional runtime backfill before V55, outside the
            # numbered steps. Preserve the original fixture's existing tree.
            conn.execute("""
                CREATE TABLE note_folders(
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  path TEXT UNIQUE NOT NULL COLLATE NOCASE,
                  parent_id INTEGER REFERENCES note_folders(id) ON DELETE CASCADE ON UPDATE CASCADE,
                  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  deleted BOOLEAN NOT NULL DEFAULT 0,
                  client_id TEXT NOT NULL DEFAULT 'unknown',
                  version INTEGER NOT NULL DEFAULT 1
                )
            """)
            conn.execute("""
                CREATE TABLE note_folder_memberships(
                  note_id TEXT NOT NULL REFERENCES notes(id) ON DELETE CASCADE ON UPDATE CASCADE,
                  folder_id INTEGER NOT NULL REFERENCES note_folders(id) ON DELETE CASCADE ON UPDATE CASCADE,
                  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  PRIMARY KEY(note_id, folder_id)
                )
            """)

    # The maintained V4 template includes two later portable-ID declarations.
    # Remove only those known V55 additions before constructing the old schema.
    historical_v4 = CharactersRAGDB._FULL_SCHEMA_SQL_V4
    column = "  sync_id       TEXT    NOT NULL,\n"
    assert historical_v4.count(column) == 2
    historical_v4 = historical_v4.replace(column, "")
    for table in ("keywords", "keyword_collections"):
        index = f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table}_sync_id_unique ON {table}(sync_id);"
        assert historical_v4.count(index) == 1
        historical_v4 = historical_v4.replace(index, "")
    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_FULL_SCHEMA_SQL_V4", historical_v4)
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 54)
        patch.setattr(CharactersRAGDB, "_initialize_schema", initialize_historical)
        db = CharactersRAGDB(str(db_path), client_id="migration-v55-fixture")
    try:
        note_id = str(uuid.uuid4())
        with db.transaction() as conn:
            conn.execute("INSERT INTO notes(id,title,content,client_id) VALUES (?,?,?,?)",
                         (note_id, "migration note", "fixture", "migration-v55-fixture"))
            has_sync_id = {
                table: "sync_id"
                in {row["name"] for row in conn.execute(f"PRAGMA table_info('{table}')").fetchall()}
                for table in ("keywords", "keyword_collections", "note_folders")
            }
            assert not any(has_sync_id.values())
            keyword_rows = []
            for keyword, deleted in (("active-keyword", 0), ("deleted-keyword", 1)):
                if has_sync_id["keywords"]:
                    cursor = conn.execute(
                        "INSERT INTO keywords(sync_id, keyword, deleted, client_id, version) VALUES (?, ?, ?, ?, ?)",
                        (str(uuid.uuid4()), keyword, deleted, "fixture", 3),
                    )
                else:
                    cursor = conn.execute(
                        "INSERT INTO keywords(keyword, deleted, client_id, version) VALUES (?, ?, ?, ?)",
                        (keyword, deleted, "fixture", 3),
                    )
                keyword_rows.append(int(cursor.lastrowid))

            collection_rows = []
            for name, deleted in (("active collection", 0), ("deleted collection", 1)):
                if has_sync_id["keyword_collections"]:
                    cursor = conn.execute(
                        "INSERT INTO keyword_collections(sync_id, name, deleted, client_id, version) VALUES (?, ?, ?, ?, ?)",
                        (str(uuid.uuid4()), name, deleted, "fixture", 4),
                    )
                else:
                    cursor = conn.execute(
                        "INSERT INTO keyword_collections(name, deleted, client_id, version) VALUES (?, ?, ?, ?)",
                        (name, deleted, "fixture", 4),
                    )
                collection_rows.append(int(cursor.lastrowid))

            if has_sync_id["note_folders"]:
                parent = conn.execute(
                    "INSERT INTO note_folders(sync_id, name, path, deleted, client_id, version) VALUES (?, ?, ?, ?, ?, ?)",
                    (str(uuid.uuid4()), "Parent", "Parent", 0, "fixture", 5),
                )
            else:
                parent = conn.execute(
                    "INSERT INTO note_folders(name, path, deleted, client_id, version) VALUES (?, ?, ?, ?, ?)",
                    ("Parent", "Parent", 0, "fixture", 5),
                )
            parent_id = int(parent.lastrowid)
            if has_sync_id["note_folders"]:
                child = conn.execute(
                    "INSERT INTO note_folders(sync_id, name, path, parent_id, deleted, client_id, version) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (str(uuid.uuid4()), "Child", "Parent/Child", parent_id, 1, "fixture", 6),
                )
            else:
                child = conn.execute(
                    "INSERT INTO note_folders(name, path, parent_id, deleted, client_id, version) VALUES (?, ?, ?, ?, ?, ?)",
                    ("Child", "Parent/Child", parent_id, 1, "fixture", 6),
                )
            child_id = int(child.lastrowid)

            conn.execute(
                "INSERT INTO collection_keywords(collection_id, keyword_id) VALUES (?, ?)",
                (collection_rows[0], keyword_rows[0]),
            )
            conn.execute(
                "INSERT INTO note_folder_memberships(note_id, folder_id) VALUES (?, ?)",
                (note_id, child_id),
            )
    finally:
        db.close_connection()

    return {
        "keywords": keyword_rows,
        "keyword_collections": collection_rows,
        "note_folders": [parent_id, child_id],
    }


def test_v54_migration_adds_stable_unique_sync_ids_and_preserves_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "notes-organization-v54.sqlite"
    original_ids = _build_v54_fixture(db_path, monkeypatch)

    migrated = CharactersRAGDB(str(db_path), client_id="migration-v55")
    try:
        with migrated.transaction() as conn:
            version = conn.execute(
                "SELECT version FROM db_schema_version WHERE schema_name = ?",
                (CharactersRAGDB._SCHEMA_NAME,),
            ).fetchone()["version"]
            assert version == CharactersRAGDB._CURRENT_SCHEMA_VERSION

            first_sync_ids: dict[str, list[str]] = {}
            for table, expected_ids in original_ids.items():
                rows = conn.execute(
                    f"SELECT id, sync_id FROM {table} ORDER BY id"  # nosec B608
                ).fetchall()
                assert [int(row["id"]) for row in rows] == expected_ids
                sync_ids = [str(row["sync_id"]) for row in rows]
                assert len(set(sync_ids)) == len(sync_ids)
                for sync_id in sync_ids:
                    _assert_canonical_uuid4(sync_id)
                first_sync_ids[table] = sync_ids

            assert conn.execute("SELECT COUNT(*) FROM collection_keywords").fetchone()[0] == 1
            folder_link = conn.execute(
                "SELECT folder_id FROM note_folder_memberships"
            ).fetchone()
            assert int(folder_link["folder_id"]) == original_ids["note_folders"][1]
            child = conn.execute(
                "SELECT parent_id, deleted, version FROM note_folders WHERE id = ?",
                (original_ids["note_folders"][1],),
            ).fetchone()
            assert (int(child["parent_id"]), int(child["deleted"]), int(child["version"])) == (
                original_ids["note_folders"][0],
                1,
                6,
            )
            suppression_columns = {
                row["name"]
                for row in conn.execute(
                    "PRAGMA table_info('note_folder_sync_suppressions')"
                ).fetchall()
            }
            assert suppression_columns >= {"note_id", "folder_id", "created_at"}
            note_id = str(conn.execute("SELECT id FROM notes LIMIT 1").fetchone()["id"])
            conn.execute(
                "INSERT INTO note_folder_sync_suppressions(note_id, folder_id) VALUES (?, ?)",
                (note_id, original_ids["note_folders"][1]),
            )
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO note_folder_sync_suppressions(note_id, folder_id) VALUES (?, ?)",
                    (note_id, original_ids["note_folders"][1]),
                )
    finally:
        migrated.close_connection()

    reopened = CharactersRAGDB(str(db_path), client_id="migration-v55-rerun")
    try:
        with reopened.transaction() as conn:
            for table, expected_sync_ids in first_sync_ids.items():
                actual = [
                    str(row["sync_id"])
                    for row in conn.execute(f"SELECT sync_id FROM {table} ORDER BY id").fetchall()  # nosec B608
                ]
                assert actual == expected_sync_ids
    finally:
        reopened.close_connection()


def test_sqlite_migration_map_contains_v54_to_v55_step(tmp_path: Path) -> None:
    db = CharactersRAGDB(str(tmp_path / "migration-map.sqlite"), client_id="migration-map")
    try:
        steps = db._sqlite_linear_migration_steps()
        assert steps[54].__name__ == "_migrate_from_v54_to_v55"
    finally:
        db.close_connection()
