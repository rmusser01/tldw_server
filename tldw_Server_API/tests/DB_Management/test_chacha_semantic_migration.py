"""SQLite schema-v65 contracts for Notes semantic-index persistence."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


pytestmark = pytest.mark.unit


_TABLES = {
    "note_semantic_index_configs",
    "note_semantic_generations",
    "note_semantic_note_state",
    "note_semantic_chunks",
    "note_semantic_work",
}


def _version(conn: sqlite3.Connection) -> int:
    return int(
        conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()[0]
    )


def _initialize(path: Path) -> CharactersRAGDB:
    return CharactersRAGDB(str(path), client_id="owner-a")


def _prepare_v64_database(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 64)
    db = _initialize(path)
    db.close_all_connections()


def test_sqlite_v65_fresh_schema_has_semantic_tables_constraints_and_indexes(tmp_path: Path) -> None:
    db_path = tmp_path / "chacha-v65-fresh.sqlite"
    db = _initialize(db_path)
    db.close_all_connections()

    with sqlite3.connect(db_path) as conn:
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        config_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(note_semantic_index_configs)")
        }
        generation_indexes = {
            str(row[1]) for row in conn.execute("PRAGMA index_list(note_semantic_generations)")
        }
        work_indexes = {
            str(row[1]) for row in conn.execute("PRAGMA index_list(note_semantic_work)")
        }
        chunk_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(note_semantic_chunks)")
        }
        generation_fks = list(conn.execute("PRAGMA foreign_key_list(note_semantic_note_state)"))

        assert _version(conn) == 65
        assert tables >= _TABLES
        assert {
            "owner_user_id",
            "dataset_id",
            "desired_state",
            "configuration_revision",
            "semantic_index_revision",
            "endpoint_origin_display",
            "dimension_state",
            "dimensions",
        } <= config_columns
        assert {"chunk_id", "field", "start_offset", "end_offset", "chunk_fingerprint"} <= chunk_columns
        assert {
            "idx_note_semantic_generations_one_active",
            "idx_note_semantic_generations_one_staging",
        } <= generation_indexes
        assert {
            "idx_note_semantic_work_claimable",
            "idx_note_semantic_work_note_coalesce",
            "idx_note_semantic_work_generation_coalesce",
        } <= work_indexes
        assert any(str(row[2]) == "note_semantic_generations" for row in generation_fks)

        conn.execute(
            """
            INSERT INTO note_semantic_index_configs(
                owner_user_id, dataset_id, desired_state, configuration_revision,
                semantic_index_revision, dimension_state, metric,
                normalization_version, chunker_version, updated_at
            ) VALUES ('owner-a', 'dataset-a', 'disabled', 1, 0, 'pending', 'cosine', 'v1', 'v1', CURRENT_TIMESTAMP)
            """
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "UPDATE note_semantic_index_configs SET desired_state = 'unknown' "
                "WHERE owner_user_id = 'owner-a'"
            )


def test_sqlite_v64_to_v65_upgrade_creates_semantic_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "chacha-v64-upgrade.sqlite"
    _prepare_v64_database(db_path, monkeypatch)
    monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 65)
    db = _initialize(db_path)
    db.close_all_connections()

    with sqlite3.connect(db_path) as conn:
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert _version(conn) == 65
        assert tables >= _TABLES


def test_sqlite_v65_rolls_back_partial_ddl_and_preserves_v64(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "chacha-v65-rollback.sqlite"
    _prepare_v64_database(db_path, monkeypatch)
    injected_sql = (
        CharactersRAGDB._MIGRATION_SQL_V64_TO_V65
        + "\nTHIS IS AN INJECTED MIGRATION FAILURE;"
    )
    monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 65)
    monkeypatch.setattr(CharactersRAGDB, "_MIGRATION_SQL_V64_TO_V65", injected_sql)

    with pytest.raises(Exception):
        _initialize(db_path)

    with sqlite3.connect(db_path) as conn:
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert _version(conn) == 64
        assert not (tables & _TABLES)
