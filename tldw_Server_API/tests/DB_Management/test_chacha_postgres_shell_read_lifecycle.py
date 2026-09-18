"""Shell read predecessors must not retain later Buddy/Notes bootstrap locks."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as deps
from tldw_Server_API.app.api.v1.endpoints import notes, persona
from tldw_Server_API.app.core.Buddy.service import BuddyService
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError

pytestmark = pytest.mark.integration


def _seed(db):
    character = db.add_character_card({"name": deps.DEFAULT_CHARACTER_NAME, "description": "Existing fixture"})
    profile = db.create_persona_profile({"user_id": "1", "name": "Existing fixture persona"})
    note = db.add_note(title="Committed title", content="Shell lifecycle fixture")
    db.sync_note_folders(note, ["Research"])
    db.close_connection()
    return SimpleNamespace(db=db, character=character, profile=profile, note=note)


@pytest.fixture
def pg_shell(pg_database_config, tmp_path, monkeypatch):
    """Use the official disposable PG database and real production read chains."""
    monkeypatch.setattr(persona, "is_persona_enabled", lambda: True)
    # Preserve actual initialization statements, only shorten their lock budget.
    monkeypatch.setattr(CharactersRAGDB, "_NOTES_MOODBOARD_STUDIO_V61_POSTGRES_LOCK_TIMEOUT", "100ms")
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(tmp_path / "shell.db", client_id="1", backend=backend)
    try:
        yield _seed(db), pg_database_config
    finally:
        db.close_connection()
        backend.get_pool().close_all()


def _persona_endpoint(f):
    result = asyncio.run(
        persona.list_persona_profiles(
            active_only=False,
            include_deleted=False,
            limit=100,
            offset=0,
            _current_user=SimpleNamespace(id=1),
            db=f.db,
        )
    )
    assert [str(row.id) for row in result] == [f.profile]


def _predecessor(f, kind):
    if kind == "default":
        assert deps._ensure_default_character(f.db) == f.character
    elif kind == "persona":
        _persona_endpoint(f)


def _downstream(f, kind):
    if kind == "buddy":
        service = BuddyService(f.db, "1")
        assert service.list_profiles(limit=10, offset=0) == {"buddies": []}
        assert service.attachment("default")["attachment"] is None
        return
    result = asyncio.run(
        notes.list_notes(
            db=f.db,
            limit=100,
            offset=0,
            include_keywords=True,
            rate_limiter=SimpleNamespace(check_user_rate_limit=AsyncMock(return_value=(True, {}))),
            current_user=SimpleNamespace(id=1),
        )
    )
    assert result["total"] == 1
    assert result["notes"][0]["id"] == f.note
    assert [row["path"] for row in result["notes"][0]["folders"]] == ["Research"]


def _state(db, raw):
    relations = db.backend.execute(
        "SELECT DISTINCT c.relname FROM pg_locks l JOIN pg_class c ON c.oid=l.relation "
        "JOIN pg_namespace n ON n.oid=c.relnamespace WHERE l.pid=%s AND n.nspname='public' "
        "AND c.relkind='r' ORDER BY c.relname",
        (raw.info.backend_pid,),
    ).rows
    return {"transaction": raw.info.transaction_status.name, "relations": [row["relname"] for row in relations]}


def _replacement(f, config, tmp_path):
    from psycopg.conninfo import make_conninfo

    replacement_config = replace(
        config,
        connection_string=make_conninfo(
            host=config.pg_host,
            port=str(config.pg_port),
            dbname=config.pg_database,
            user=config.pg_user,
            password=config.pg_password,
            options="-c lock_timeout=100ms",
        ),
    )
    backend = DatabaseBackendFactory.create_backend(replacement_config)
    replacement = None
    try:
        replacement = CharactersRAGDB(tmp_path / "replacement.db", client_id="1", backend=backend)
        assert replacement.get_note_by_id(f.note)["content"] == "Shell lifecycle fixture"
    finally:
        if replacement is not None:
            replacement.close_connection()
        backend.get_pool().close_all()


@pytest.mark.parametrize("read", ["readiness", "character-name", "persona-list", "persona-buddies"])
def test_each_shell_predecessor_releases_its_own_transaction(pg_shell, read):
    f, _ = pg_shell
    if read == "readiness":
        f.db.ensure_character_tables_ready()
    elif read == "character-name":
        assert f.db.get_character_card_by_name(deps.DEFAULT_CHARACTER_NAME)["id"] == f.character
    elif read == "persona-list":
        assert [row["id"] for row in f.db.list_persona_profiles(user_id="1")] == [f.profile]
    else:
        assert f.db.list_persona_buddies(user_id="1", persona_ids=[f.profile]) == {f.profile: None}
    raw = f.db._get_thread_connection()
    assert raw.info.transaction_status.name == "IDLE", _state(f.db, raw)


@pytest.mark.parametrize("predecessor", ["none", "default", "persona"])
@pytest.mark.parametrize("downstream", ["buddy", "notes"])
def test_actual_shell_chain_allows_replacement_initialization(pg_shell, tmp_path, predecessor, downstream):
    f, config = pg_shell
    raw = f.db._get_thread_connection()
    _predecessor(f, predecessor)
    _downstream(f, downstream)
    state = _state(f.db, raw)
    try:
        _replacement(f, config, tmp_path)
    except CharactersRAGDBError as exc:
        pytest.fail(f"Actual replacement failed with {type(exc).__name__}; predecessor state: {state}")
    assert raw.info.transaction_status.name == "IDLE", state


def test_actual_default_maintenance_executor_releases_existing_card_read(pg_shell, tmp_path, monkeypatch):
    f, config = pg_shell
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(deps, "_get_chacha_executor", lambda: executor)
        try:
            asyncio.run(deps._ensure_default_character_async(f.db, 1))
            raw = executor.submit(f.db._get_thread_connection).result()
            state = _state(f.db, raw)
            try:
                _replacement(f, config, tmp_path)
            except CharactersRAGDBError as exc:
                pytest.fail(f"Actual replacement failed with {type(exc).__name__}; executor state: {state}")
            assert raw.info.transaction_status.name == "IDLE", state
        finally:
            executor.submit(f.db.close_connection).result()


@pytest.mark.parametrize("predecessor", ["default", "persona"])
@pytest.mark.parametrize("owner", ["implicit", "nested", "backend"])
@pytest.mark.parametrize("commit", [False, True], ids=["rollback", "commit"])
def test_shell_chain_preserves_pending_caller_writes(pg_shell, predecessor, owner, commit):
    f, _ = pg_shell
    db = f.db
    raw = db._get_thread_connection()
    with ExitStack() as stack:
        if owner == "nested":
            stack.enter_context(db.transaction())
            stack.enter_context(db.transaction())
        elif owner == "backend":
            stack.enter_context(db.backend.transaction(connection=raw))
        db.execute_query("UPDATE notes SET title = ? WHERE id = ?", ("Pending title", f.note))
        _predecessor(f, predecessor)
        _downstream(f, "buddy")
        _downstream(f, "notes")
        assert raw.info.transaction_status.name == "INTRANS"
        assert db.get_note_by_id(f.note)["title"] == "Pending title"
        assert db.backend.execute("SELECT title FROM notes WHERE id=%s", (f.note,)).scalar == "Committed title"
        if commit:
            raw.commit()
        else:
            raw.rollback()
    expected = "Pending title" if commit else "Committed title"
    assert db.backend.execute("SELECT title FROM notes WHERE id=%s", (f.note,)).scalar == expected


@pytest.mark.parametrize("predecessor", ["default", "persona"])
@pytest.mark.parametrize("owner", ["chacha", "nested", "backend"])
def test_first_shell_read_inside_explicit_scope_preserves_scope(pg_shell, predecessor, owner):
    f, _ = pg_shell
    raw = f.db._get_thread_connection()
    with ExitStack() as stack:
        if owner == "backend":
            stack.enter_context(f.db.backend.transaction(connection=raw))
        else:
            stack.enter_context(f.db.transaction())
            if owner == "nested":
                stack.enter_context(f.db.transaction())
        assert raw.info.transaction_status.name == "IDLE"
        _predecessor(f, predecessor)
        assert raw.info.transaction_status.name == "INTRANS"
        assert _state(f.db, raw)["relations"]
    assert raw.info.transaction_status.name == "IDLE"


@pytest.mark.parametrize("predecessor", ["default", "persona"])
def test_sqlite_shell_chain_preserves_reads_and_caller_rollback(tmp_path, monkeypatch, predecessor):
    monkeypatch.setattr(persona, "is_persona_enabled", lambda: True)
    db = CharactersRAGDB(tmp_path / "shell.sqlite", client_id="1")
    try:
        f = _seed(db)
        _predecessor(f, predecessor)
        _downstream(f, "buddy")
        _downstream(f, "notes")
        assert not db.get_connection().in_transaction
        with pytest.raises(RuntimeError, match="caller rollback"):
            with db.transaction():
                db.execute_query("UPDATE notes SET title = ? WHERE id = ?", ("Pending title", f.note))
                _predecessor(f, predecessor)
                _downstream(f, "notes")
                raise RuntimeError("caller rollback")
        assert db.get_note_by_id(f.note)["title"] == "Committed title"
    finally:
        db.close_connection()
