"""PostgreSQL schema-v67 contracts for OSCE quiz persistence."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


class _FakeTransaction:
    def __enter__(self) -> object:
        return object()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


class _FakeBackend:
    backend_type = BackendType.POSTGRESQL

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction()

    def table_exists(self, _name: str, connection: object = None) -> bool:
        return True


class _RecordingCursor:
    def fetchone(self) -> dict[str, object]:
        return {
            "id": 42,
            "version": 1,
            "activity_type": "questions",
            "total_questions": 0,
            "total_stations": 0,
            "time_limit_seconds": None,
            "passing_score": None,
        }


class _RecordingConnection:
    def __init__(self) -> None:
        self.queries: list[tuple[str, tuple[object, ...]]] = []

    def execute(self, query: str, params: tuple[object, ...]) -> _RecordingCursor:
        self.queries.append((query, params))
        return _RecordingCursor()


def test_postgres_quiz_content_mutations_lock_the_quiz_row() -> None:
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db._backend = _FakeBackend()
    db._uses_shared_content_backend = False
    db._local = SimpleNamespace()
    connection = _RecordingConnection()

    row = db._get_quiz_row_for_mutation(connection, 42)

    assert row["activity_type"] == "questions"
    assert connection.queries == [
        (
            "SELECT id, version, activity_type, total_questions, total_stations, "
            "time_limit_seconds, passing_score FROM quizzes WHERE id = ? AND deleted = FALSE FOR UPDATE",
            (42,),
        )
    ]
    assert "_get_quiz_row_for_mutation" in CharactersRAGDB.update_quiz.__code__.co_names
    assert "_get_quiz_row_for_mutation" in CharactersRAGDB.create_question.__code__.co_names


def test_postgres_initializer_routes_schema_v66_through_v67(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db._backend = _FakeBackend()
    db._uses_shared_content_backend = False
    db._backend_refresh_suspended = False
    db._local = SimpleNamespace()

    monkeypatch.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 67)
    monkeypatch.setattr(db, "_get_schema_version_postgres", lambda _conn, lock=False: 66)
    monkeypatch.setattr(db, "_verify_note_attachment_schema_postgres", lambda _conn: None)
    monkeypatch.setattr(db, "_verify_note_task_schema_postgres", lambda _conn: None)
    monkeypatch.setattr(db, "_verify_notes_moodboard_studio_schema_postgres", lambda _conn: None)
    monkeypatch.setattr(db, "_ensure_note_graph_suggestion_schema_postgres", lambda _conn: None)
    monkeypatch.setattr(
        db,
        "_configure_notes_moodboard_studio_v61_postgres_transaction",
        lambda _conn: None,
    )

    def _reached_v67(_conn: object) -> None:
        raise RuntimeError("reached-v67")

    monkeypatch.setattr(db, "_migrate_from_v66_to_v67_postgres", _reached_v67, raising=False)

    with pytest.raises(RuntimeError, match="^reached-v67$"):
        db._initialize_schema_postgres()


def test_postgres_v67_ddl_has_columns_tables_constraints_and_indexes() -> None:
    sql = " ".join(CharactersRAGDB._MIGRATION_SQL_V66_TO_V67_POSTGRES.split())

    for clause in (
        "ALTER TABLE quizzes ADD COLUMN IF NOT EXISTS activity_type TEXT NOT NULL DEFAULT 'questions'",
        "ALTER TABLE quizzes ADD COLUMN IF NOT EXISTS generation_profile TEXT",
        "ALTER TABLE quizzes ADD COLUMN IF NOT EXISTS total_stations INTEGER NOT NULL DEFAULT 0",
        "CHECK (activity_type IN ('questions', 'osce'))",
        "CHECK (total_stations >= 0)",
        "CREATE TABLE osce_stations",
        "schema_version TEXT NOT NULL CHECK (schema_version = 'osce.station.v1')",
        "origin TEXT NOT NULL CHECK (origin IN ('generated', 'manual'))",
        "CREATE TABLE osce_practice_attempts",
        "state TEXT NOT NULL CHECK (state IN ('in_progress', 'self_assessment', 'completed'))",
        "CHECK (char_length(candidate_notes) <= 10000)",
        "REFERENCES quizzes(id) ON DELETE CASCADE",
        "REFERENCES osce_stations(id) ON DELETE CASCADE",
        "CREATE INDEX idx_osce_stations_quiz_active_order ON osce_stations(quiz_id, deleted, order_index)",
        "CREATE INDEX idx_osce_attempts_station_state_modified ON osce_practice_attempts(station_id, state, last_modified_at DESC)",
        "CREATE INDEX idx_osce_attempts_quiz_state_modified ON osce_practice_attempts(quiz_id, state, last_modified_at DESC)",
        "UNIQUE (station_id, client_attempt_id)",
    ):
        assert clause in sql


def _assert_live_postgres_osce_schema(backend: object) -> None:
    columns = backend.execute(  # type: ignore[attr-defined]
        """
        SELECT column_name, data_type, is_nullable, column_default
          FROM information_schema.columns
         WHERE table_schema = current_schema()
           AND table_name = 'quizzes'
           AND column_name IN ('activity_type', 'generation_profile', 'total_stations')
        """
    ).rows
    by_name = {row["column_name"]: row for row in columns}
    assert set(by_name) == {"activity_type", "generation_profile", "total_stations"}
    assert by_name["activity_type"]["is_nullable"] == "NO"
    assert "questions" in str(by_name["activity_type"]["column_default"])
    assert by_name["generation_profile"]["is_nullable"] == "YES"
    assert by_name["total_stations"]["is_nullable"] == "NO"
    assert str(by_name["total_stations"]["column_default"]).startswith("0")

    tables = backend.execute(  # type: ignore[attr-defined]
        """
        SELECT table_name
          FROM information_schema.tables
         WHERE table_schema = current_schema()
           AND table_name IN ('osce_stations', 'osce_practice_attempts')
        """
    ).rows
    assert {row["table_name"] for row in tables} == {"osce_stations", "osce_practice_attempts"}

    indexes = backend.execute(  # type: ignore[attr-defined]
        """
        SELECT indexname
          FROM pg_indexes
         WHERE schemaname = current_schema()
           AND tablename IN ('osce_stations', 'osce_practice_attempts')
        """
    ).rows
    assert {
        "idx_osce_stations_quiz_active_order",
        "idx_osce_attempts_station_state_modified",
        "idx_osce_attempts_quiz_state_modified",
    }.issubset({row["indexname"] for row in indexes})
    assert "idx_osce_attempts_station_client" not in {row["indexname"] for row in indexes}

    constraints = backend.execute(  # type: ignore[attr-defined]
        """
        SELECT relation.relname AS table_name, pg_get_constraintdef(constraint_row.oid) AS definition
          FROM pg_constraint AS constraint_row
          JOIN pg_class AS relation ON relation.oid = constraint_row.conrelid
          JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
         WHERE namespace.nspname = current_schema()
           AND relation.relname IN ('quizzes', 'osce_stations', 'osce_practice_attempts')
        """
    ).rows
    definitions = " ".join(str(row["definition"]) for row in constraints)
    assert "activity_type" in definitions
    assert "questions" in definitions and "osce" in definitions
    assert "UNIQUE (station_id, client_attempt_id)" in definitions
    assert "FOREIGN KEY (quiz_id) REFERENCES quizzes(id) ON DELETE CASCADE" in definitions
    assert "FOREIGN KEY (station_id) REFERENCES osce_stations(id) ON DELETE CASCADE" in definitions


@pytest.mark.integration
@pytest.mark.timeout(90)
def test_postgres_v67_fresh_schema_matches_osce_contract(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="fresh-v67", backend=backend)
    try:
        version = backend.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = %s",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).scalar
        assert int(version) == 67
        _assert_live_postgres_osce_schema(backend)
    finally:
        db.close_all_connections()


@pytest.mark.integration
@pytest.mark.timeout(90)
def test_postgres_v67_upgrade_preserves_legacy_quiz_as_questions(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    with monkeypatch.context() as context:
        context.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 66)
        legacy = CharactersRAGDB(":memory:", client_id="legacy-v66", backend=backend)
        backend.execute(
            "INSERT INTO quizzes(id, name, total_questions, client_id) VALUES (%s, %s, %s, %s)",
            (1, "Legacy questions", 3, "legacy-v66"),
        )
        legacy.close_all_connections()

    upgraded_backend = DatabaseBackendFactory.create_backend(pg_database_config)
    upgraded = CharactersRAGDB(":memory:", client_id="upgrade-v67", backend=upgraded_backend)
    try:
        quiz = upgraded.get_quiz(1)
        assert quiz is not None
        assert quiz["activity_type"] == "questions"
        assert quiz["generation_profile"] is None
        assert quiz["total_questions"] == 3
        assert quiz["total_stations"] == 0
        _assert_live_postgres_osce_schema(upgraded_backend)
    finally:
        upgraded.close_all_connections()
