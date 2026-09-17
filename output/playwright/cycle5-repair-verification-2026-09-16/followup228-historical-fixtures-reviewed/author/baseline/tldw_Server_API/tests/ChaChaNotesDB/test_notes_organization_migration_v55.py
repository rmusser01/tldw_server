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


def _build_v54_fixture(db_path: Path) -> dict[str, list[int]]:
    db = CharactersRAGDB(str(db_path), client_id="migration-v55-fixture")
    try:
        note_id = db.add_note(title="migration note", content="fixture")
        with db.transaction() as conn:
            has_sync_id = {
                table: "sync_id"
                in {row["name"] for row in conn.execute(f"PRAGMA table_info('{table}')").fetchall()}
                for table in ("keywords", "keyword_collections", "note_folders")
            }
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

    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE IF EXISTS note_folder_sync_suppressions")
        for table, index_name in (
            ("keywords", "idx_keywords_sync_id_unique"),
            ("keyword_collections", "idx_keyword_collections_sync_id_unique"),
            ("note_folders", "idx_note_folders_sync_id_unique"),
        ):
            columns = {row[1] for row in conn.execute(f"PRAGMA table_info('{table}')")}
            if "sync_id" in columns:
                conn.execute(f"DROP INDEX IF EXISTS {index_name}")
                conn.execute(f"ALTER TABLE {table} DROP COLUMN sync_id")
        conn.execute(
            "UPDATE db_schema_version SET version = 54 WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        )

    return {
        "keywords": keyword_rows,
        "keyword_collections": collection_rows,
        "note_folders": [parent_id, child_id],
    }


def test_v54_migration_adds_stable_unique_sync_ids_and_preserves_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "notes-organization-v54.sqlite"
    original_ids = _build_v54_fixture(db_path)

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
