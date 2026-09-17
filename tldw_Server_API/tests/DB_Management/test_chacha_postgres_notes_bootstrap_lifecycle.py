"""The complete Notes page bootstrap must release reads before replacement DDL."""

import asyncio
from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.api.v1.endpoints import notes
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError

pytestmark = pytest.mark.integration


def _seed(db, populated):
    note = db.add_note(title="Committed title", content="Notes bootstrap fixture")
    db.sync_note_folders(note, ["Research"])
    keyword = collection = None
    if populated:
        keyword = db.add_keyword("Citrine")
        collection = db.add_keyword_collection("Study")
        db.link_note_to_keyword(note, keyword)
        db.link_collection_to_keyword(collection, keyword)
    db.close_connection()
    return SimpleNamespace(db=db, note=note, keyword=keyword, collection=collection, populated=populated)


@pytest.fixture(params=[False, True], ids=["empty", "populated"])
def pg_bootstrap(request, pg_database_config, tmp_path, monkeypatch):
    """Official isolated PostgreSQL database; never a running application DB."""
    monkeypatch.setattr(CharactersRAGDB, "_NOTES_MOODBOARD_STUDIO_V61_POSTGRES_LOCK_TIMEOUT", "100ms")
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(tmp_path / "bootstrap.db", client_id="1", backend=backend)
    try:
        yield _seed(db, request.param), pg_database_config
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


async def _handler(f, name, *, title="Committed title"):
    shared = {
        "db": f.db,
        "rate_limiter": SimpleNamespace(check_user_rate_limit=AsyncMock(return_value=(True, {}))),
        "current_user": SimpleNamespace(id=1),
    }
    if name == "keywords":
        rows = await notes.list_keywords_endpoint(limit=1000, offset=0, include_note_counts=True, **shared)
        assert [row["keyword"] for row in rows] == (["Citrine"] if f.populated else [])
        if f.populated:
            assert rows[0]["id"] == f.keyword
            assert rows[0]["note_count"] == 1
    elif name == "collections":
        result = await notes.list_keyword_collections_endpoint(limit=200, offset=0, include_keywords=True, **shared)
        assert result["total"] == int(f.populated)
        assert [row["name"] for row in result["collections"]] == (["Study"] if f.populated else [])
        if f.populated:
            assert result["collections"][0]["id"] == f.collection
            assert [row["id"] for row in result["collections"][0]["keywords"]] == [f.keyword]
    else:
        # Ordinary page browse omits include_keywords; folders are still inline.
        result = await notes.list_notes(limit=100, offset=0, include_keywords=False, **shared)
        assert result["total"] == 1
        assert result["notes"][0]["id"] == f.note
        assert result["notes"][0]["title"] == title
        assert result["notes"][0]["content"] == "Notes bootstrap fixture"
        assert [row["path"] for row in result["notes"][0]["folders"]] == ["Research"]


def _state(db, raw):
    rows = db.backend.execute(
        "SELECT DISTINCT c.relname FROM pg_locks l JOIN pg_class c ON c.oid=l.relation "
        "JOIN pg_namespace n ON n.oid=c.relnamespace WHERE l.pid=%s AND n.nspname='public' "
        "AND c.relkind IN ('r','p') ORDER BY c.relname",
        (raw.info.backend_pid,),
    ).rows
    return {"transaction": raw.info.transaction_status.name, "relations": [row["relname"] for row in rows]}


async def _page_bootstrap(f, order, *, title="Committed title", observe=None):
    for name in (*order, "notes"):
        await _handler(f, name, title=title)
        if observe:
            observe(name)


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
        assert replacement.get_note_by_id(f.note)["content"] == "Notes bootstrap fixture"
    finally:
        if replacement is not None:
            replacement.close_all_connections()
        backend.get_pool().close_all()


@pytest.mark.parametrize("order", [("keywords", "collections"), ("collections", "keywords")])
def test_complete_notes_bootstrap_allows_replacement_while_first_connection_lives(pg_bootstrap, tmp_path, order):
    f, config = pg_bootstrap
    raw = f.db._get_thread_connection()
    assert raw.info.transaction_status.name == "IDLE"
    stages = []
    asyncio.run(_page_bootstrap(f, order, observe=lambda name: stages.append({"stage": name, **_state(f.db, raw)})))
    try:
        _replacement(f, config, tmp_path)
    except CharactersRAGDBError as exc:
        pytest.fail(f"Live replacement failed with {type(exc).__name__}; Notes bootstrap stages: {stages}")
    assert f.db._get_thread_connection() is raw
    assert not raw.closed
    assert _state(f.db, raw) == {"transaction": "IDLE", "relations": []}, stages


@pytest.mark.parametrize(
    "read", ["keywords", "collections", "collection-count", "collection-contents", "keyword-note-counts"]
)
def test_each_notes_bootstrap_read_finishes_its_own_transaction(pg_bootstrap, read):
    f, _ = pg_bootstrap
    db = f.db
    raw = db._get_thread_connection()
    assert raw.info.transaction_status.name == "IDLE"
    if read == "keywords":
        assert [row["keyword"] for row in db.list_keywords()] == (["Citrine"] if f.populated else [])
    elif read == "collections":
        assert [row["name"] for row in db.list_keyword_collections()] == (["Study"] if f.populated else [])
    elif read == "collection-count":
        assert db.count_keyword_collections() == int(f.populated)
    elif read == "collection-contents":
        assert [row["id"] for row in db.get_keywords_for_collection(f.collection or 999)] == (
            [f.keyword] if f.populated else []
        )
    else:
        assert db.get_note_counts_for_keywords([f.keyword] if f.populated else []) == (
            {f.keyword: 1} if f.populated else {}
        )
    assert _state(db, raw) == {"transaction": "IDLE", "relations": []}


def test_notes_without_bootstrap_predecessors_remains_a_control(pg_bootstrap, tmp_path):
    f, config = pg_bootstrap
    asyncio.run(_handler(f, "notes"))
    _replacement(f, config, tmp_path)
    assert _state(f.db, f.db._get_thread_connection()) == {"transaction": "IDLE", "relations": []}


@pytest.mark.parametrize("owner", ["implicit", "raw-begin", "chacha", "nested", "backend"])
@pytest.mark.parametrize("commit", [False, True], ids=["rollback", "commit"])
def test_notes_bootstrap_preserves_caller_pending_writes(pg_bootstrap, owner, commit):
    f, _ = pg_bootstrap
    db = f.db
    raw = db._get_thread_connection()
    with ExitStack() as stack:
        if owner == "raw-begin":
            db.get_connection().execute("BEGIN")
        elif owner == "backend":
            stack.enter_context(db.backend.transaction(connection=raw))
        elif owner in {"chacha", "nested"}:
            stack.enter_context(db.transaction())
            if owner == "nested":
                stack.enter_context(db.transaction())
        db.execute_query("UPDATE notes SET title = ? WHERE id = ?", ("Pending title", f.note))
        asyncio.run(_page_bootstrap(f, ("keywords", "collections"), title="Pending title"))
        assert raw.info.transaction_status.name == "INTRANS"
        assert db.backend.execute("SELECT title FROM notes WHERE id=%s", (f.note,)).scalar == "Committed title"
        if commit:
            raw.commit()
        else:
            raw.rollback()
    expected = "Pending title" if commit else "Committed title"
    assert db.backend.execute("SELECT title FROM notes WHERE id=%s", (f.note,)).scalar == expected


@pytest.mark.parametrize("owner", ["chacha", "nested", "backend"])
def test_first_bootstrap_read_preserves_explicit_scope(pg_bootstrap, owner):
    f, _ = pg_bootstrap
    raw = f.db._get_thread_connection()
    with ExitStack() as stack:
        if owner == "backend":
            stack.enter_context(f.db.backend.transaction(connection=raw))
        else:
            stack.enter_context(f.db.transaction())
            if owner == "nested":
                stack.enter_context(f.db.transaction())
        assert raw.info.transaction_status.name == "IDLE"
        asyncio.run(_page_bootstrap(f, ("keywords", "collections")))
        assert raw.info.transaction_status.name == "INTRANS"
        assert "notes" in _state(f.db, raw)["relations"]
    assert _state(f.db, raw) == {"transaction": "IDLE", "relations": []}


@pytest.mark.parametrize("populated", [False, True], ids=["empty", "populated"])
def test_sqlite_complete_notes_bootstrap_preserves_data_and_caller_rollback(tmp_path, populated):
    db = CharactersRAGDB(tmp_path / "bootstrap.sqlite", client_id="1")
    try:
        f = _seed(db, populated)
        asyncio.run(_page_bootstrap(f, ("keywords", "collections")))
        assert not db.get_connection().in_transaction
        with pytest.raises(RuntimeError, match="caller rollback"):
            with db.transaction():
                db.execute_query("UPDATE notes SET title = ? WHERE id = ?", ("Pending title", f.note))
                asyncio.run(_page_bootstrap(f, ("collections", "keywords"), title="Pending title"))
                raise RuntimeError("caller rollback")
        assert db.get_note_by_id(f.note)["title"] == "Committed title"
    finally:
        db.close_all_connections()
