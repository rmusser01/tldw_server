"""The complete Notes page bootstrap must release reads before replacement DDL."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tldw_Server_API.app.api.v1.endpoints import notes
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig, DatabaseError, FTSQuery
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


@pytest.mark.parametrize("reuse_backend", [True, False], ids=["warm-worker", "cold-worker"])
@pytest.mark.parametrize("pending_write", [False, True], ids=["read", "write"])
def test_second_user_bootstrap_preserves_an_active_first_user_read(
    pg_bootstrap: tuple[SimpleNamespace, DatabaseConfig], tmp_path: Path,
    reuse_backend: bool, pending_write: bool,
) -> None:
    """Any worker can open the current schema during another user's transaction."""
    f, config = pg_bootstrap
    backend = f.db.backend
    second_backend = backend if reuse_backend else DatabaseBackendFactory.create_backend(config)
    second_user = None
    try:
        with backend.transaction() as first_request:
            backend.execute("SELECT id FROM conversations LIMIT 1", connection=first_request)
            if pending_write:
                backend.execute("UPDATE notes SET title='Pending title' WHERE id=%s",
                                (f.note,), connection=first_request)
            second_user = CharactersRAGDB(tmp_path / "second-user.db", client_id="2", backend=second_backend)
            assert second_user.get_note_by_id(f.note) is None
            # Initialization must not commit or roll back the other caller's work.
            assert first_request.info.transaction_status.name == "INTRANS"
            assert f.db.get_note_by_id(f.note)["title"] == "Committed title"
        expected = "Pending title" if pending_write else "Committed title"
        assert f.db.get_note_by_id(f.note)["title"] == expected
    finally:
        if second_user is not None:
            second_user.close_connection()
        if not reuse_backend:
            second_backend.get_pool().close_all()


@pytest.mark.parametrize("damage", [
    "ALTER TABLE note_attachments DISABLE ROW LEVEL SECURITY",
    "DROP INDEX idx_note_attachments_owner_dataset_blob",
], ids=["rls-disabled", "index-missing"])
def test_later_user_rejects_schema_drift_after_completed_bootstrap(
    pg_bootstrap: tuple[SimpleNamespace, DatabaseConfig], tmp_path: Path, damage: str,
) -> None:
    """Successful setup must never cache away the existing fail-closed checks."""
    f, _ = pg_bootstrap
    with f.db.backend.transaction() as connection:
        f.db.backend.execute(damage, connection=connection)
    with pytest.raises(CharactersRAGDBError, match="catalog drifted"):
        CharactersRAGDB(tmp_path / "drift.db", client_id="2", backend=f.db.backend)


def test_recreated_schema_does_not_inherit_bootstrap_readiness(
    pg_bootstrap: tuple[SimpleNamespace, DatabaseConfig], tmp_path: Path,
) -> None:
    """Recreate only the official fixture's schema and require a usable new store."""
    f, _ = pg_bootstrap
    backend = f.db.backend
    f.db.close_connection()
    with backend.transaction() as connection:
        backend.execute("DROP SCHEMA public CASCADE", connection=connection)
        backend.execute("CREATE SCHEMA public", connection=connection)
    replacement = CharactersRAGDB(tmp_path / "recreated.db", client_id="2", backend=backend)
    try:
        assert replacement.get_note_by_id(f.note) is None
        created = replacement.add_note(title="Recreated", content="Fresh schema")
        assert replacement.get_note_by_id(created)["content"] == "Fresh schema"
    finally:
        replacement.close_connection()


def test_newer_schema_version_is_not_hidden_by_completed_bootstrap(
    pg_bootstrap: tuple[SimpleNamespace, DatabaseConfig], tmp_path: Path,
) -> None:
    """A later schema version must still reject an incompatible reader."""
    f, _ = pg_bootstrap
    backend = f.db.backend
    with backend.transaction() as connection:
        backend.execute("UPDATE db_schema_version SET version=version+1 WHERE schema_name=%s",
                        (f.db._SCHEMA_NAME,), connection=connection)
    with pytest.raises(CharactersRAGDBError, match="newer than supported"):
        CharactersRAGDB(tmp_path / "newer.db", client_id="2", backend=backend)


def test_failed_bootstrap_can_retry_on_the_same_backend(
    pg_database_config: DatabaseConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed first initialization cannot mark this backend as ready."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    replacement = None

    def fail_setup(*args: object, **kwargs: object) -> None:
        """Simulate a database outage at the public execution boundary."""
        raise DatabaseError("planned bootstrap outage")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(backend, "execute", fail_setup)
            with pytest.raises(CharactersRAGDBError, match="planned bootstrap outage"):
                CharactersRAGDB(tmp_path / "failed.db", client_id="1", backend=backend)
        replacement = CharactersRAGDB(tmp_path / "retry.db", client_id="2", backend=backend)
        note = replacement.add_note(title="Recovered", content="Retry works")
        assert replacement.get_note_by_id(note)["content"] == "Retry works"
    finally:
        if replacement is not None:
            replacement.close_connection()
        backend.get_pool().close_all()


@pytest.mark.parametrize("reuse_backend", [True, False], ids=["one-worker", "separate-workers"])
def test_simultaneous_first_users_share_successful_bootstrap(
    pg_database_config: DatabaseConfig, tmp_path: Path, reuse_backend: bool,
) -> None:
    """Concurrent tenant wrappers must all reach usable, isolated Notes stores."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    backends = [backend] * 3 if reuse_backend else [
        backend, *(DatabaseBackendFactory.create_backend(pg_database_config) for _ in range(2)),
    ]
    ready = threading.Barrier(3)

    def create_owner_note(owner: int) -> str:
        """Open one tenant after the barrier and persist a tenant-owned note."""
        ready.wait(timeout=10)
        db = CharactersRAGDB(tmp_path / f"owner-{owner}.db", client_id=str(owner), backend=backends[owner - 1])
        try:
            note = db.add_note(title=f"Owner {owner}", content="Isolated content")
            assert db.get_note_by_id(note)["title"] == f"Owner {owner}"
            return note
        finally:
            db.close_connection()

    try:
        with ThreadPoolExecutor(max_workers=3) as workers:
            notes_by_owner = list(workers.map(create_owner_note, (1, 2, 3)))
        for owner, note in enumerate(notes_by_owner, 1):
            db = CharactersRAGDB(tmp_path / f"verify-{owner}.db", client_id=str(owner), backend=backend)
            try:
                assert db.get_note_by_id(note) is not None
                for foreign in set(notes_by_owner) - {note}:
                    assert db.get_note_by_id(foreign) is None
            finally:
                db.close_connection()
    finally:
        for owned_backend in set(backends):
            owned_backend.get_pool().close_all()


def test_v70_upgrade_retries_after_late_failure_without_publishing_readiness(
    pg_database_config: DatabaseConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The readiness version and all reconciliation work share one transaction."""
    class LegacySchema(CharactersRAGDB):
        """Build the genuine schema preceding the versioned bootstrap."""
        _POSTGRES_SCHEMA_VERSION = 70

    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    legacy = replacement = None
    try:
        legacy = LegacySchema(tmp_path / "legacy.db", client_id="1", backend=backend)
        note = legacy.add_note(title="Survives", content="Existing v70 content")
        legacy.close_connection()
        original_execute = backend.execute

        def fail_last_ensure(query: str, *args: object, **kwargs: object):
            """Fail the final auxiliary table step at the public SQL boundary."""
            if "CREATE TABLE IF NOT EXISTS conversation_settings(" in query:
                raise DatabaseError("planned final bootstrap failure")
            return original_execute(query, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(backend, "execute", fail_last_ensure)
            with pytest.raises(CharactersRAGDBError, match="planned final bootstrap failure"):
                CharactersRAGDB(tmp_path / "failed-upgrade.db", client_id="1", backend=backend)
        assert backend.execute("SELECT version FROM db_schema_version WHERE schema_name=%s",
                               (legacy._SCHEMA_NAME,)).scalar == 70
        replacement = CharactersRAGDB(tmp_path / "upgraded.db", client_id="1", backend=backend)
        assert replacement.get_note_by_id(note)["content"] == "Existing v70 content"
        assert backend.execute("SELECT version FROM db_schema_version WHERE schema_name=%s",
                               (legacy._SCHEMA_NAME,)).scalar == 71
    finally:
        for db in (legacy, replacement):
            if db is not None:
                db.close_connection()
        backend.get_pool().close_all()


def test_current_schema_opens_and_searches_on_a_read_only_cold_backend(
    pg_bootstrap: tuple[SimpleNamespace, DatabaseConfig], tmp_path: Path,
) -> None:
    """Opening a current schema performs no DDL/DML and restores FTS aliases."""
    from psycopg.conninfo import make_conninfo

    f, config = pg_bootstrap
    read_only_config = replace(config, connection_string=make_conninfo(
        host=config.pg_host, port=str(config.pg_port), dbname=config.pg_database,
        user=config.pg_user, password=config.pg_password,
        options="-c default_transaction_read_only=on",
    ))
    backend = DatabaseBackendFactory.create_backend(read_only_config)
    replacement = None
    try:
        replacement = CharactersRAGDB(tmp_path / "read-only.db", client_id="1", backend=backend)
        assert replacement.get_note_by_id(f.note)["title"] == "Committed title"
        result = backend.fts_search(FTSQuery(query="bootstrap", table="notes_fts", filters={"client_id": "1"}))
        assert [row["id"] for row in result.rows] == [f.note]
    finally:
        if replacement is not None:
            replacement.close_connection()
        backend.get_pool().close_all()


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
