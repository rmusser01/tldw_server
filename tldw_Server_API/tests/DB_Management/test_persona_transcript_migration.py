"""Persona transcript installation must preserve old storage or roll back entirely."""

import sqlite3

import pytest

from tldw_Server_API.app.core.DB_Management.chacha import persona_transcript_store as transcript_storage
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError, SchemaError

pytestmark = pytest.mark.integration


def test_sqlite_upgrade_failure_rolls_back_payload_table_and_marker(tmp_path, monkeypatch):
    path = tmp_path / "old-persona.db"
    with monkeypatch.context() as legacy:
        legacy.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 84)
        old = CharactersRAGDB(path, client_id="1")
        persona = old.create_persona_profile({"user_id": "1", "name": "Preserved"})
        session = old.create_persona_session({"user_id": "1", "persona_id": persona})
        before = old.get_persona_session(session, user_id="1")
        old.close_all_connections()
    migrate = transcript_storage.migrate_sqlite_transcripts

    def interrupted(conn, **kwargs):
        migrate(conn, **kwargs)
        raise SchemaError("Injected after transcript DDL and marker update")

    with monkeypatch.context() as fault:
        fault.setattr(transcript_storage, "migrate_sqlite_transcripts", interrupted)
        with pytest.raises(CharactersRAGDBError, match="Injected") as failed:
            CharactersRAGDB(path, client_id="1")
    assert isinstance(failed.value.__cause__, SchemaError)  # nosec B101 -- Preserve the exact migration/schema refusal.
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT version FROM db_schema_version").fetchone()[0] == 84  # nosec B101
        assert (
            conn.execute("SELECT name FROM sqlite_master WHERE name='persona_session_transcripts'").fetchone() is None
        )  # nosec B101
    repaired = CharactersRAGDB(path, client_id="1")
    try:
        assert repaired.get_persona_session(session, user_id="1") == before  # nosec B101
        assert repaired.persona_transcript_store.get_persona_transcript(session, user_id="1") is None  # nosec B101 -- No fabricated legacy history.
    finally:
        repaired.close_all_connections()


def test_postgres_upgrade_failure_rolls_back_table_rls_and_marker(pg_restricted_backend, tmp_path, monkeypatch):
    backend = pg_restricted_backend
    with monkeypatch.context() as legacy:
        legacy.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 89)
        old = CharactersRAGDB(tmp_path / "old.db", client_id="1", owner_user_id="1", backend=backend)
        persona = old.create_persona_profile({"user_id": "1", "name": "Preserved"})
        session = old.create_persona_session({"user_id": "1", "persona_id": persona})
        before = old.get_persona_session(session, user_id="1")
        old.close_connection()
    migrate = CharactersRAGDB._migrate_from_v89_to_v90_postgres

    def interrupted(self, conn):
        migrate(self, conn)
        raise SchemaError("Injected after transcript DDL/RLS and marker update")

    with monkeypatch.context() as fault:
        fault.setattr(CharactersRAGDB, "_migrate_from_v89_to_v90_postgres", interrupted)
        with pytest.raises(CharactersRAGDBError, match="Injected") as failed:
            CharactersRAGDB(tmp_path / "next.db", client_id="1", owner_user_id="1", backend=backend)
    assert isinstance(failed.value.__cause__, SchemaError)  # nosec B101
    assert backend.execute("SELECT version FROM db_schema_version").scalar == 89  # nosec B101
    assert not backend.table_exists("persona_session_transcripts")  # nosec B101
    repaired = CharactersRAGDB(tmp_path / "repaired.db", client_id="1", owner_user_id="1", backend=backend)
    try:
        assert repaired.get_persona_session(session, user_id="1") == before  # nosec B101
        assert repaired.persona_transcript_store.get_persona_transcript(session, user_id="1") is None  # nosec B101
    finally:
        repaired.close_all_connections()


def test_postgres_policy_drift_is_refused_without_request_repair(pg_restricted_backend, tmp_path):
    backend = pg_restricted_backend
    db = CharactersRAGDB(tmp_path / "current.db", client_id="1", owner_user_id="1", backend=backend)
    db.close_connection()
    backend.execute("ALTER TABLE persona_session_transcripts NO FORCE ROW LEVEL SECURITY")
    with pytest.raises(CharactersRAGDBError, match="ownership authority") as failed:
        CharactersRAGDB(tmp_path / "drift.db", client_id="1", owner_user_id="1", backend=backend)
    assert isinstance(failed.value.__cause__, SchemaError)  # nosec B101
    relation = backend.execute(
        "SELECT relforcerowsecurity FROM pg_class WHERE oid='persona_session_transcripts'::regclass"
    ).rows[0]
    assert relation["relforcerowsecurity"] is False  # nosec B101 -- Runtime verification never silently repairs authority.


def test_sqlite_current_missing_transcript_table_is_not_silently_recreated(tmp_path):
    path = tmp_path / "drift.db"
    db = CharactersRAGDB(path, client_id="1")
    with db.transaction() as conn:
        conn.execute("DROP TABLE persona_session_transcripts")
    db.close_all_connections()
    with pytest.raises(CharactersRAGDBError, match="missing or unsafe") as failed:
        CharactersRAGDB(path, client_id="1")
    assert isinstance(failed.value.__cause__, SchemaError)  # nosec B101 -- Preserve the exact migration/schema refusal.
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT version FROM db_schema_version").fetchone()[0] == 85  # nosec B101


def test_postgres_rls_denies_forged_owner_even_with_real_parent(pg_restricted_backend, tmp_path):
    from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError

    db = CharactersRAGDB(tmp_path / "owner.db", client_id="1", owner_user_id="1", backend=pg_restricted_backend)
    try:
        persona = db.create_persona_profile({"user_id": "1", "name": "Actual owner"})
        session = db.create_persona_session({"user_id": "1", "persona_id": persona})
        with pytest.raises(DatabaseError, match="row-level security"):
            with db.transaction() as conn:
                query, params = db._prepare_backend_statement(
                    "INSERT INTO persona_session_transcripts(session_id,user_id,persona_id,revision,last_activity,turn_count,turns_json) "
                    "VALUES (?,?,?,1,'2026-11-02T00:00:00+00:00',0,'[]')",
                    (session, "2", persona),
                )
                conn.execute(query, params)
        assert db.persona_transcript_store.get_persona_transcript(session, user_id="1") is None  # nosec B101
    finally:
        db.close_all_connections()


@pytest.mark.parametrize(
    "ddl",
    [
        "ALTER TABLE persona_session_transcripts DROP CONSTRAINT persona_session_transcripts_session_id_fkey",
        "ALTER TABLE persona_session_transcripts DROP CONSTRAINT persona_session_transcripts_pkey",
        "ALTER TABLE persona_session_transcripts ALTER COLUMN user_id DROP NOT NULL",
        "ALTER TABLE persona_session_transcripts DROP CONSTRAINT persona_session_transcripts_revision_check",
        "ALTER TABLE persona_session_transcripts DROP CONSTRAINT persona_session_transcripts_turn_count_check",
    ],
    ids=["missing-fk", "missing-pk", "nullable-owner", "missing-revision-check", "missing-count-check"],
)
def test_postgres_schema_constraint_drift_is_refused(pg_restricted_backend, tmp_path, ddl):
    backend = pg_restricted_backend
    db = CharactersRAGDB(tmp_path / "current.db", client_id="1", owner_user_id="1", backend=backend)
    db.close_connection()
    backend.execute(ddl)
    with pytest.raises(CharactersRAGDBError, match="schema constraints") as failed:
        CharactersRAGDB(tmp_path / "drift.db", client_id="1", owner_user_id="1", backend=backend)
    assert isinstance(failed.value.__cause__, SchemaError)  # nosec B101


def test_postgres_append_clock_is_sampled_after_session_lock(pg_restricted_backend, tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime, timedelta, timezone
    from threading import Event

    from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
    from tldw_Server_API.app.core.DB_Management.chacha.persona_transcript_store import PersonaTranscriptStore
    from tldw_Server_API.app.core.Persona.session_manager import SessionManager

    db = CharactersRAGDB(tmp_path / "clock.db", client_id="1", owner_user_id="1", backend=pg_restricted_backend)
    persona = db.create_persona_profile({"user_id": "1", "name": "Serialized clock"})
    session = db.create_persona_session({"user_id": "1", "persona_id": persona})
    clock = [datetime.now(timezone.utc)]
    waiting = Event()
    query = PersonaTranscriptStore._query

    def observed_query(self, conn, sql, params):
        if sql.startswith("SELECT s.id"):
            waiting.set()
        return query(self, conn, sql, params)

    monkeypatch.setattr(PersonaTranscriptStore, "_query", observed_query)
    monkeypatch.setattr(PersonaTranscriptStore, "_now", staticmethod(lambda: clock[0]))
    turn = SessionManager().prepare_turn(role="user", content="Serialized activity")

    @chacha_operation(independent=True)
    def append(db):
        return db.persona_transcript_store.append_persona_transcript_turn(session_id=session, user_id="1", persona_id=persona, turn=turn)

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            with db.transaction() as conn:
                statement, params = db._prepare_backend_statement(
                    "SELECT id FROM persona_sessions WHERE id=? FOR UPDATE", (session,)
                )
                conn.execute(statement, params).fetchone()
                pending = pool.submit(append, db)
                assert waiting.wait(5)  # nosec B101 -- Actual independent checkout reached the held row lock.
                clock[0] += timedelta(hours=2)
            saved = pending.result(timeout=10)
        activity = db.execute_query(
            "SELECT last_activity FROM persona_session_transcripts WHERE session_id=?", (session,)
        ).fetchone()["last_activity"]
        assert activity == clock[0].isoformat()  # nosec B101 -- Never commit an older pre-lock activity timestamp.
        assert saved["turn_count"] == 1  # nosec B101
    finally:
        db.close_connection()
