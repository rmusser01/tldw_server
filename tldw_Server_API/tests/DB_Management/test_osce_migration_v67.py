"""SQLite schema-v67 contracts for OSCE quiz persistence."""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import reset_managed_sqlite_backends
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

EXPECTED_OSCE_INDEXES = {
    "idx_osce_stations_quiz_active_order",
    "idx_osce_attempts_station_state_modified",
    "idx_osce_attempts_quiz_state_modified",
}


def _close_db(db: CharactersRAGDB, db_path: Path) -> None:
    db.close_all_connections()
    with contextlib.suppress(Exception):
        reset_managed_sqlite_backends(
            sqlite_targets=[str(db_path), str(db_path.resolve())],
            mode="hard",
        )


@pytest.fixture
def legacy_v66_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "legacy-v66.db"
    with monkeypatch.context() as context:
        context.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 66)
        db = CharactersRAGDB(db_path, client_id="migration-seed")
        _close_db(db, db_path)

    with sqlite3.connect(db_path) as conn:
        version = conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()
        assert version == (66,)
        conn.execute(
            "INSERT INTO quizzes(id, name, total_questions, client_id) VALUES (?, ?, ?, ?)",
            (1, "Legacy questions", 3, "migration-seed"),
        )
        conn.commit()
    return db_path


def _sqlite_osce_schema(db: CharactersRAGDB) -> dict[str, object]:
    conn = db.get_connection()
    quiz_columns = {
        row["name"]: (row["type"], row["notnull"], row["dflt_value"])
        for row in conn.execute("PRAGMA table_info('quizzes')").fetchall()
        if row["name"] in {"activity_type", "generation_profile", "total_stations"}
    }
    tables = {
        row["name"]: " ".join(str(row["sql"]).split())
        for row in conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'table' AND name IN (?, ?)",
            ("osce_stations", "osce_practice_attempts"),
        ).fetchall()
    }
    indexes = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND name LIKE 'idx_osce_%'"
        ).fetchall()
    }
    return {"quiz_columns": quiz_columns, "tables": tables, "indexes": indexes}


def _assert_sqlite_osce_schema(db: CharactersRAGDB) -> None:
    schema = _sqlite_osce_schema(db)
    assert schema["quiz_columns"] == {
        "activity_type": ("TEXT", 1, "'questions'"),
        "generation_profile": ("TEXT", 0, None),
        "total_stations": ("INTEGER", 1, "0"),
    }
    assert set(schema["tables"]) == {"osce_stations", "osce_practice_attempts"}
    assert schema["indexes"] == EXPECTED_OSCE_INDEXES

    station_sql = schema["tables"]["osce_stations"]
    attempt_sql = schema["tables"]["osce_practice_attempts"]
    for clause in (
        "CHECK (schema_version = 'osce.station.v1')",
        "CHECK (order_index >= 0)",
        "CHECK (version >= 1)",
        "CHECK (origin IN ('generated', 'manual'))",
        "CHECK (deleted IN (0, 1))",
    ):
        assert clause in station_sql
    for clause in (
        "CHECK (state IN ('in_progress', 'self_assessment', 'completed'))",
        "CHECK (length(candidate_notes) <= 10000)",
        "CHECK (frozen_elapsed_seconds IS NULL OR frozen_elapsed_seconds >= 0)",
        "CHECK (version >= 1)",
        "UNIQUE (station_id, client_attempt_id)",
    ):
        assert clause in attempt_sql

    conn = db.get_connection()
    station_fks = {
        (row["from"], row["table"], row["to"], row["on_delete"])
        for row in conn.execute("PRAGMA foreign_key_list('osce_stations')").fetchall()
    }
    attempt_fks = {
        (row["from"], row["table"], row["to"], row["on_delete"])
        for row in conn.execute("PRAGMA foreign_key_list('osce_practice_attempts')").fetchall()
    }
    assert station_fks == {("quiz_id", "quizzes", "id", "CASCADE")}
    assert attempt_fks == {
        ("station_id", "osce_stations", "id", "CASCADE"),
        ("quiz_id", "quizzes", "id", "CASCADE"),
    }


def test_v67_upgrade_preserves_legacy_quiz_as_questions(legacy_v66_db: Path) -> None:
    db = CharactersRAGDB(legacy_v66_db, client_id="migration-test")
    try:
        quiz = db.get_quiz(1)
        assert quiz is not None
        assert quiz["activity_type"] == "questions"
        assert quiz["generation_profile"] is None
        assert quiz["total_questions"] == 3
        assert quiz["total_stations"] == 0
        _assert_sqlite_osce_schema(db)
    finally:
        _close_db(db, legacy_v66_db)


def test_v67_fresh_and_upgraded_schemas_are_equivalent(
    legacy_v66_db: Path,
    tmp_path: Path,
) -> None:
    upgraded = CharactersRAGDB(legacy_v66_db, client_id="migration-test")
    fresh_path = tmp_path / "fresh-v67.db"
    fresh = CharactersRAGDB(fresh_path, client_id="fresh-test")
    try:
        _assert_sqlite_osce_schema(upgraded)
        _assert_sqlite_osce_schema(fresh)
        assert _sqlite_osce_schema(upgraded) == _sqlite_osce_schema(fresh)
    finally:
        _close_db(upgraded, legacy_v66_db)
        _close_db(fresh, fresh_path)


def test_v67_osce_foreign_keys_cascade_on_quiz_purge(tmp_path: Path) -> None:
    db_path = tmp_path / "cascade-v67.db"
    db = CharactersRAGDB(db_path, client_id="cascade-test")
    try:
        quiz_id = db.create_quiz(name="OSCE", activity_type="osce")
        now = db._get_current_utc_timestamp_iso()
        with db.transaction() as conn:
            station_id = conn.execute(
                """
                INSERT INTO osce_stations(
                    quiz_id, schema_version, content_json, order_index, origin,
                    verification_state, created_at, updated_at
                ) VALUES (?, 'osce.station.v1', '{}', 0, 'manual', 'manually_authored', ?, ?)
                """,
                (quiz_id, now, now),
            ).lastrowid
            conn.execute(
                """
                INSERT INTO osce_practice_attempts(
                    station_id, quiz_id, client_attempt_id, station_snapshot_json,
                    state, started_at, last_modified_at
                ) VALUES (?, ?, 'client-1', '{}', 'in_progress', ?, ?)
                """,
                (station_id, quiz_id, now, now),
            )

        assert db.delete_quiz(quiz_id, hard_delete=True) is True
        conn = db.get_connection()
        assert conn.execute("SELECT COUNT(*) FROM osce_stations").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM osce_practice_attempts").fetchone()[0] == 0
    finally:
        _close_db(db, db_path)
