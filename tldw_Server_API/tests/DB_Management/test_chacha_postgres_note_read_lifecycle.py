"""Notes reads must release their own PostgreSQL locks, not caller transactions."""

import asyncio
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.api.v1.endpoints import notes
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
)

pytestmark = pytest.mark.integration


def _seed_notes(db):
    note_id = db.add_note(title="Committed title", content="Read lifecycle fixture")
    keyword_id = db.add_keyword("Lifecycle")
    db.link_note_to_keyword(note_id, keyword_id)
    db.sync_note_folders(note_id, ["Research"])
    db.close_connection()
    return note_id


@pytest.fixture
def pg_notes(pg_database_config):
    """Use the official function-scoped DB_Management PostgreSQL database."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="1", backend=backend)
    try:
        yield db, _seed_notes(db)
    finally:
        db.close_connection()
        backend.get_pool().close_all()


def _folder_read(db, note_id, operation):
    if operation == "path":
        return [db.get_note_folder_by_path("Research")]
    if operation == "list":
        return db.list_note_folders()
    if operation == "single":
        return db.get_note_folders_for_note(note_id)
    return db.get_note_folders_for_notes([note_id])[note_id]


def _notes_endpoint_read(db, note_id, operation):
    limiter = SimpleNamespace(check_user_rate_limit=AsyncMock(return_value=(True, {})))
    user = SimpleNamespace(id=1)
    if operation == "get":
        return asyncio.run(
            notes.get_note(
                note_id=note_id,
                db=db,
                rate_limiter=limiter,
                current_user=user,
            )
        )
    result = asyncio.run(
        notes.list_notes(
            db=db,
            limit=100,
            offset=0,
            include_keywords=True,
            rate_limiter=limiter,
            current_user=user,
        )
    )
    assert result["total"] == 1
    return result["notes"][0]


def _assert_bootstrap_ddl_unblocked(db):
    # An independent connection executes the exact DDL blocked in native UAT.
    # The short timeout makes a retained AccessShareLock a deterministic failure.
    with db.backend.transaction() as observer:
        observer.execute("SET LOCAL lock_timeout = '100ms'")
        observer.execute("ALTER TABLE note_folders DROP CONSTRAINT IF EXISTS note_folders_path_key")
    raw = db._get_thread_connection()
    assert raw.info.transaction_status.name == "IDLE"
    assert (
        db.backend.execute(
            "SELECT COUNT(*) FROM pg_locks WHERE pid = %s AND relation = 'note_folders'::regclass",
            (raw.info.backend_pid,),
        ).scalar
        == 0
    )


@pytest.mark.parametrize("operation", ["path", "list", "single", "bulk"])
def test_standalone_folder_read_does_not_block_bootstrap(pg_notes, operation):
    db, note_id = pg_notes
    folders = _folder_read(db, note_id, operation)
    assert [row["path"] for row in folders] == ["Research"]
    _assert_bootstrap_ddl_unblocked(db)


@pytest.mark.parametrize("operation", ["get", "list"])
def test_actual_notes_read_chain_does_not_block_bootstrap(pg_notes, operation):
    db, note_id = pg_notes
    result = _notes_endpoint_read(db, note_id, operation)
    assert result["id"] == note_id
    assert [row["path"] for row in result["folders"]] == ["Research"]
    assert [row["keyword"] for row in result["keywords"]] == ["Lifecycle"]
    _assert_bootstrap_ddl_unblocked(db)


def _all_pure_reads(db, note_id):
    assert db.get_note_by_id(note_id)["title"] == "Pending title"
    assert db.list_notes()[0]["title"] == "Pending title"
    assert db.count_notes() == 1
    assert len(db.get_keywords_for_note(note_id)) == 1
    assert len(db.get_keywords_for_notes([note_id])[note_id]) == 1
    for operation in ("path", "list", "single", "bulk"):
        assert len(_folder_read(db, note_id, operation)) == 1


@pytest.mark.parametrize("owner", ["implicit-write", "write-cte", "raw-begin", "chacha-nested", "backend"])
@pytest.mark.parametrize("commit", [False, True], ids=["rollback", "commit"])
def test_pure_reads_preserve_caller_pending_writes(pg_notes, owner, commit):
    db, note_id = pg_notes
    raw = db._get_thread_connection()
    with ExitStack() as stack:
        if owner == "raw-begin":
            raw.execute("BEGIN")
        elif owner == "chacha-nested":
            stack.enter_context(db.transaction())
            stack.enter_context(db.transaction())
        elif owner == "backend":
            stack.enter_context(db.backend.transaction(connection=raw))

        if owner == "write-cte":
            db.execute_query(
                "WITH changed AS (UPDATE notes SET title = ? WHERE id = ? RETURNING title) SELECT title FROM changed",
                ("Pending title", note_id),
            )
        else:
            db.execute_query("UPDATE notes SET title = ? WHERE id = ?", ("Pending title", note_id))
        _all_pure_reads(db, note_id)
        assert raw.info.transaction_status.name == "INTRANS"
        assert db.backend.execute("SELECT title FROM notes WHERE id = %s", (note_id,)).scalar == "Committed title"

        # The caller alone chooses the outcome; no read may commit or erase it.
        if commit:
            raw.commit()
        else:
            raw.rollback()
    expected = "Pending title" if commit else "Committed title"
    assert db.backend.execute("SELECT title FROM notes WHERE id = %s", (note_id,)).scalar == expected


@pytest.mark.parametrize("owner", ["chacha", "chacha-nested", "backend"])
def test_first_read_inside_explicit_scope_keeps_its_lock(pg_notes, owner):
    db, note_id = pg_notes
    raw = db._get_thread_connection()
    with ExitStack() as stack:
        if owner == "backend":
            stack.enter_context(db.backend.transaction(connection=raw))
        else:
            stack.enter_context(db.transaction())
            if owner == "chacha-nested":
                stack.enter_context(db.transaction())
        assert raw.info.transaction_status.name == "IDLE"
        assert len(db.get_note_folders_for_note(note_id)) == 1
        assert raw.info.transaction_status.name == "INTRANS"
        assert (
            db.backend.execute(
                "SELECT COUNT(*) FROM pg_locks WHERE pid = %s AND relation = 'note_folders'::regclass",
                (raw.info.backend_pid,),
            ).scalar
            > 0
        )
    _assert_bootstrap_ddl_unblocked(db)


def test_generic_locking_select_keeps_caller_transaction(pg_notes):
    db, note_id = pg_notes
    raw = db._get_thread_connection()
    db.execute_query("SELECT id FROM notes WHERE id = ? FOR UPDATE", (note_id,))
    db.get_note_folders_for_note(note_id)
    assert raw.info.transaction_status.name == "INTRANS"
    assert (
        db.backend.execute(
            "SELECT COUNT(*) FROM pg_locks WHERE pid = %s AND relation = 'notes'::regclass AND mode = 'RowShareLock'",
            (raw.info.backend_pid,),
        ).scalar
        > 0
    )
    raw.rollback()


def test_generic_select_function_keeps_its_transaction_local_effect(pg_notes):
    db, note_id = pg_notes
    raw = db._get_thread_connection()
    db.execute_query("SELECT set_config('app.uat171_pending', 'pending', true)")
    db.get_note_folders_for_note(note_id)
    assert raw.info.transaction_status.name == "INTRANS"
    assert db.execute_query("SELECT current_setting('app.uat171_pending') AS value").fetchone()["value"] == "pending"
    raw.rollback()


def test_failed_standalone_read_releases_its_aborted_transaction(pg_notes):
    db, note_id = pg_notes
    with pytest.raises(CharactersRAGDBError):
        db.list_notes(limit="invalid-integer")
    assert db._get_thread_connection().info.transaction_status.name == "IDLE"
    assert len(db.get_note_folders_for_note(note_id)) == 1
    _assert_bootstrap_ddl_unblocked(db)


@pytest.mark.parametrize("operation", ["get", "list"])
def test_sqlite_notes_read_chain_keeps_existing_behavior(tmp_path, operation):
    db = CharactersRAGDB(tmp_path / "notes.sqlite", client_id="1")
    try:
        note_id = _seed_notes(db)
        result = _notes_endpoint_read(db, note_id, operation)
        assert result["id"] == note_id
        assert [row["path"] for row in result["folders"]] == ["Research"]
        assert not db.get_connection().in_transaction
        with pytest.raises(RuntimeError, match="rollback control"):
            with db.transaction():
                db.execute_query("UPDATE notes SET title = ? WHERE id = ?", ("Pending title", note_id))
                _all_pure_reads(db, note_id)
                raise RuntimeError("rollback control")
        assert db.get_note_by_id(note_id)["title"] == "Committed title"
    finally:
        db.close_connection()


def test_note_duplicate_and_projection_unique_errors_remain_distinct(pg_notes, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.backends.base import UniqueConstraintError
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError

    db, note_id = pg_notes
    with pytest.raises(ConflictError, match="already exists"):
        db.add_note("Replacement", "replacement body", note_id=note_id)
    assert db.get_note_by_id(note_id)["title"] == "Committed title"

    def failed_projection(**kwargs):
        raise UniqueConstraintError("PostgreSQL query execution failed")

    monkeypatch.setattr(db.note_graph_projection_store, "replace_projection", failed_projection)
    with pytest.raises(CharactersRAGDBError) as error:
        db.add_note("Projection failure", "body", note_id="projection-failure")
    assert not isinstance(error.value, ConflictError)
    assert db.get_note_by_id("projection-failure") is None


@pytest.mark.parametrize("delete_method", ["soft_delete_note", "delete_note"])
def test_owned_note_mutations_and_keyword_reads_use_postgres_booleans(pg_notes, delete_method):
    db, note_id = pg_notes
    keyword_id = db.get_keywords_for_note(note_id)[0]["id"]
    assert db.update_note(note_id, {"title": "Updated title"}, expected_version=1)
    assert db.get_note_by_id(note_id)["title"] == "Updated title"
    assert [row["id"] for row in db.get_keywords_for_notes([note_id])[note_id]] == [keyword_id]
    assert [row["id"] for row in db.get_notes_for_keyword(keyword_id)] == [note_id]

    assert getattr(db, delete_method)(note_id, expected_version=2)
    assert db.get_note_by_id(note_id) is None
    assert db.get_notes_for_keyword(keyword_id) == []
    assert db.restore_note(note_id, expected_version=3)
    assert db.get_note_by_id(note_id)["version"] == 4
    assert [row["id"] for row in db.get_notes_for_keyword(keyword_id)] == [note_id]
