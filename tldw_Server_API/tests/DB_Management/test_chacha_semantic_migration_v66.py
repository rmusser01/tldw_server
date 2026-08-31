"""SQLite schema-v66 contracts for durable semantic vector cleanup authority."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management import ChaChaNotes_DB as chacha_db_module
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    SchemaError,
)

pytestmark = pytest.mark.unit

_TABLE = "note_semantic_obsolete_vectors"


def _version(conn: sqlite3.Connection) -> int:
    return int(
        conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()[0]
    )


def _initialize(path: Path) -> None:
    db = CharactersRAGDB(str(path), client_id="owner-a")
    db.close_all_connections()


def _assert_cleanup_schema(conn: sqlite3.Connection) -> None:
    columns = {
        str(row[1]) for row in conn.execute(f"PRAGMA table_info({_TABLE})")  # nosec B608
    }
    indexes = {
        str(row[1]) for row in conn.execute(f"PRAGMA index_list({_TABLE})")  # nosec B608
    }
    foreign_tables = {
        str(row[2]) for row in conn.execute(f"PRAGMA foreign_key_list({_TABLE})")  # nosec B608
    }

    assert {
        "id",
        "owner_user_id",
        "dataset_id",
        "generation_id",
        "vector_id",
        "note_id",
        "source_kind",
        "dirty_generation",
        "claim_state",
        "attempt_count",
        "next_eligible_at",
        "claim_token",
        "claimed_at",
        "error_code",
        "created_at",
        "updated_at",
    } <= columns
    assert {
        "idx_note_semantic_obsolete_vectors_claimable",
        "idx_note_semantic_obsolete_vectors_generation",
    } <= indexes
    assert "notes" not in foreign_tables
    assert "note_semantic_generations" not in foreign_tables


def test_sqlite_v66_fresh_schema_has_durable_cleanup_ledger(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "semantic-v66-fresh.sqlite"

    _initialize(db_path)

    with sqlite3.connect(db_path) as conn:
        assert _version(conn) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        _assert_cleanup_schema(conn)
        conn.execute(
            f"""
            INSERT INTO {_TABLE}(
                id,owner_user_id,dataset_id,generation_id,vector_id,note_id,
                source_kind,dirty_generation,claim_state,attempt_count,
                next_eligible_at,created_at,updated_at
            ) VALUES (
                'cleanup-a','owner-a','dataset-a','generation-a','vector-a','note-a',
                'manifest_replace',1,'pending',0,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP
            )
            """  # nosec B608 - fixed migration relation.
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                f"UPDATE {_TABLE} SET claim_state='claimed',claim_token=NULL "
                "WHERE id='cleanup-a'"  # nosec B608 - fixed migration relation.
            )


def test_sqlite_v65_to_v66_upgrade_preserves_semantic_rows(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "semantic-v65-upgrade.sqlite"
    monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 65)
    _initialize(db_path)
    with sqlite3.connect(db_path) as conn:
        assert _version(conn) == 65
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (_TABLE,),
        ).fetchone() is None
        conn.execute(
            """
            INSERT INTO note_semantic_index_configs(
                owner_user_id,dataset_id,desired_state,configuration_revision,
                semantic_index_revision,metric,dimension_state,
                normalization_version,chunker_version,updated_at
            ) VALUES ('owner-a','dataset-a','disabled',1,0,'cosine','pending','v1','v1',CURRENT_TIMESTAMP)
            """
        )
        conn.commit()

    monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 66)
    _initialize(db_path)

    with sqlite3.connect(db_path) as conn:
        assert _version(conn) == 66
        _assert_cleanup_schema(conn)
        assert conn.execute(
            "SELECT desired_state FROM note_semantic_index_configs "
            "WHERE owner_user_id='owner-a' AND dataset_id='dataset-a'"
        ).fetchone()[0] == "disabled"


def test_sqlite_target_v65_dispatch_does_not_apply_v66(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "semantic-target-v65.sqlite"
    monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 65)

    _initialize(db_path)

    with sqlite3.connect(db_path) as conn:
        assert _version(conn) == 65
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (_TABLE,),
        ).fetchone() is None


def test_sqlite_v66_migration_failure_rolls_back_table_and_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "semantic-v66-rollback.sqlite"
    monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 65)
    db = CharactersRAGDB(str(db_path), client_id="owner-a")
    original_split = chacha_db_module.split_sql_statements

    def broken_statements(script: str) -> list[str]:
        statements = original_split(script)
        return [statements[0], "INSERT INTO missing_v66_relation(value) VALUES (1)"]

    monkeypatch.setattr(chacha_db_module, "split_sql_statements", broken_statements)
    try:
        with db.transaction() as conn:
            with pytest.raises(SchemaError, match="Notes semantic v66 SQLite migration failed"):
                db._migrate_from_v65_to_v66_sqlite(conn)
            assert db._get_db_version(conn) == 65
            assert conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (_TABLE,),
            ).fetchone() is None
    finally:
        db.close_all_connections()


def test_sqlite_v66_cleanup_ledger_survives_note_and_generation_deletion(tmp_path: Path) -> None:
    db_path = tmp_path / "semantic-v66-no-cascade.sqlite"
    _initialize(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            f"""
            INSERT INTO {_TABLE}(
                id,owner_user_id,dataset_id,generation_id,vector_id,note_id,
                source_kind,dirty_generation,claim_state,attempt_count,
                next_eligible_at,created_at,updated_at
            ) VALUES (
                'cleanup-a','owner-a','dataset-a','generation-gone','vector-a','note-gone',
                'hard_delete',2,'pending',0,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP
            )
            """  # nosec B608 - fixed migration relation.
        )
        conn.commit()

        assert conn.execute(f"SELECT vector_id FROM {_TABLE}").fetchall() == [  # nosec B608
            ("vector-a",)
        ]
