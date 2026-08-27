"""SQLite schema-v64 contracts for Notes graph suggestion persistence."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, SchemaError

pytestmark = pytest.mark.unit


_TABLES = {
    "note_graph_suggestion_runs",
    "note_graph_suggestion_operation_receipts",
    "note_graph_suggestion_rejection_sets",
    "note_graph_suggestions",
    "note_graph_suggestion_evidence",
}


def _initialize(path: Path) -> CharactersRAGDB:
    return CharactersRAGDB(str(path), client_id="owner-a")


def _version(conn: sqlite3.Connection) -> int:
    return int(
        conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()[0]
    )


def _create_note(conn: sqlite3.Connection, note_id: str) -> None:
    conn.execute(
        "INSERT INTO notes(id, title, content, client_id) VALUES (?, ?, ?, ?)",
        (note_id, note_id, "content", "owner-a"),
    )


def _prepare_v63_database(path: Path) -> CharactersRAGDB:
    db = _initialize(path)
    conn = db.get_connection()
    conn.execute("DROP TABLE note_graph_suggestion_evidence")
    conn.execute("DROP TABLE note_graph_suggestions")
    conn.execute("DROP TABLE note_graph_suggestion_rejection_sets")
    conn.execute("DROP TABLE note_graph_suggestion_runs")
    conn.execute("DROP TABLE note_graph_suggestion_operation_receipts")
    conn.execute(
        "UPDATE db_schema_version SET version = 63 WHERE schema_name = ?",
        (CharactersRAGDB._SCHEMA_NAME,),
    )
    conn.commit()
    return db


def test_sqlite_v64_fresh_schema_has_graph_suggestion_tables_constraints_and_indexes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "chacha-v64-fresh.sqlite"
    monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 64)
    db = _initialize(db_path)
    db.close_all_connections()

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        run_indexes = {
            str(row[1]) for row in conn.execute("PRAGMA index_list(note_graph_suggestion_runs)")
        }
        suggestion_indexes = {
            str(row[1]) for row in conn.execute("PRAGMA index_list(note_graph_suggestions)")
        }
        receipt_indexes = {
            str(row[1])
            for row in conn.execute(
                "PRAGMA index_list(note_graph_suggestion_operation_receipts)"
            )
        }
        evidence_fks = conn.execute(
            "PRAGMA foreign_key_list(note_graph_suggestion_evidence)"
        ).fetchall()
        suggestion_columns = {
            str(row[1]) for row in conn.execute("PRAGMA table_info(note_graph_suggestions)")
        }

        _create_note(conn, "source-note")
        conn.execute(
            """
            INSERT INTO note_graph_suggestion_runs(
                id, owner_user_id, dataset_id, source_note_id, source_fingerprint,
                state, revision, created_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """,
            ("run-active", "owner-a", "dataset-a", "source-note", "source-fingerprint", "queued", 1),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO note_graph_suggestion_runs(
                    id, owner_user_id, dataset_id, source_note_id, source_fingerprint,
                    state, revision, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                ("run-conflict", "owner-a", "dataset-a", "source-note", "new-fingerprint", "running", 1),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO note_graph_suggestion_runs(
                    id, owner_user_id, dataset_id, source_note_id, source_fingerprint,
                    state, revision, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                ("run-invalid", "owner-a", "dataset-a", "source-note", "source-fingerprint", "invalid", 1),
            )

    assert _version(sqlite3.connect(db_path)) == 64
    assert tables >= _TABLES
    assert "keyword_sync_id" in suggestion_columns
    assert "keyword_id" not in suggestion_columns
    assert {
        "idx_note_graph_suggestion_runs_owner_dataset_note_state",
        "idx_note_graph_suggestion_runs_active_source",
        "idx_note_graph_suggestion_runs_retention",
    } <= run_indexes
    assert {
        "idx_note_graph_suggestions_owner_dataset_source_state",
        "idx_note_graph_suggestions_acceptance_lease",
        "idx_note_graph_suggestions_retention",
    } <= suggestion_indexes
    assert "idx_note_graph_suggestion_operation_receipts_retention" in receipt_indexes
    assert {
        (str(row[2]), str(row[3]), str(row[6]).upper()) for row in evidence_fks
    } >= {
        ("note_graph_suggestions", "suggestion_id", "CASCADE"),
        ("notes", "note_id", "CASCADE"),
    }


def test_sqlite_v63_to_v64_upgrade_creates_graph_suggestion_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "chacha-v63-upgrade.sqlite"
    base = _prepare_v63_database(db_path)
    base.close_all_connections()
    upgraded = _initialize(db_path)
    upgraded.close_all_connections()

    with sqlite3.connect(db_path) as conn:
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert _version(conn) == 64
        assert tables >= _TABLES


def test_sqlite_v64_failure_rolls_back_partial_ddl_and_preserves_v63(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "chacha-v64-interrupted.sqlite"
    db = _prepare_v63_database(db_path)
    original_sql = CharactersRAGDB._MIGRATION_SQL_V63_TO_V64
    injected_sql = original_sql.replace(
        "CREATE TABLE note_graph_suggestion_rejection_sets",
        "SELECT * FROM injected_v64_failure;\n"
        "CREATE TABLE note_graph_suggestion_rejection_sets",
        1,
    )

    try:
        monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 64)
        monkeypatch.setattr(CharactersRAGDB, "_MIGRATION_SQL_V63_TO_V64", injected_sql)

        with pytest.raises(SchemaError, match="Notes graph suggestion v64 SQLite migration failed"):
            db._initialize_schema_sqlite()

        conn = db.get_connection()
        tables_after_failure = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert _version(conn) == 63
        assert not _TABLES.intersection(tables_after_failure)
    finally:
        db.close_all_connections()
