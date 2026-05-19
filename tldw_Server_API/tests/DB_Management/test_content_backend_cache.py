import configparser
import gc
import threading
import weakref
from contextlib import contextmanager

import pytest

from tldw_Server_API.app.core.DB_Management import ChaChaNotes_DB as chacha_db
from tldw_Server_API.app.core.DB_Management import content_backend
from tldw_Server_API.app.core.DB_Management import Collections_DB as collections_db
from tldw_Server_API.app.core.DB_Management import Evaluations_DB as evaluations_db
from tldw_Server_API.app.core.DB_Management import Watchlists_DB as watchlists_db
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.backends.base import QueryResult
from tldw_Server_API.app.core.DB_Management.media_db.runtime import (
    defaults as media_runtime_defaults,
)
from tldw_Server_API.app.core.RAG.rag_service import analytics_db


class _FakePool:
    def __init__(self, label: str = "backend") -> None:
        self.closed = 0
        self.label = label
        self.issued: list[_FakeConnection] = []
        self.returned: list[_FakeConnection] = []

    def get_connection(self):
        conn = _FakeConnection(self.label)
        self.issued.append(conn)
        return conn

    def return_connection(self, conn) -> None:
        self.returned.append(conn)

    def close_all(self) -> None:
        self.closed += 1


class _FakeConnection:
    def __init__(self, origin: str) -> None:
        self.origin = origin
        self.closed = False
        self.commit_calls = 0
        self.rollback_calls = 0

    def commit(self) -> None:
        self.commit_calls += 1

    def rollback(self) -> None:
        self.rollback_calls += 1

    def close(self) -> None:
        self.closed = True


class _FakeBackend:
    def __init__(
        self,
        *,
        backend_type: BackendType = BackendType.POSTGRESQL,
        label: str = "backend",
    ) -> None:
        self.pool = _FakePool(label)
        self.backend_type = backend_type
        self.label = label
        self.execute_calls: list[tuple[str, tuple | None, _FakeConnection | None]] = []
        self.execute_many_calls: list[tuple[str, list | None, _FakeConnection | None]] = []
        self.create_tables_calls: list[tuple[str, _FakeConnection | None]] = []
        self.config = type(
            "_FakeConfig",
            (),
            {
                "connection_string": f"postgresql:///{label}",
                "pg_host": "localhost",
                "pg_port": 5432,
                "pg_database": label,
                "pg_user": "tldw_user",
                "pg_sslmode": "prefer",
            },
        )()

    def get_pool(self):
        return self.pool

    def execute(self, query: str, params=None, connection=None):
        self.execute_calls.append((query, params, connection))
        if connection is not None:
            return QueryResult(rows=[{"backend": self.label}], rowcount=1)
        return self.label

    def execute_many(self, query: str, params_list=None, connection=None):
        self.execute_many_calls.append((query, params_list, connection))
        return QueryResult(rows=[{"backend": self.label}], rowcount=len(params_list or []))

    def create_tables(self, ddl: str, connection=None) -> None:
        self.create_tables_calls.append((ddl, connection))

    def get_table_info(self, _table: str):
        return []

    def vacuum(self) -> None:
        return None

    @contextmanager
    def transaction(self):
        connection = self.pool.get_connection()
        try:
            yield connection
        finally:
            self.pool.return_connection(connection)


def _set_fake_pg_target(
    backend: _FakeBackend,
    *,
    host: str,
    port: int,
    database: str,
    connection_string: str | None = None,
) -> _FakeBackend:
    backend.config.pg_host = host
    backend.config.pg_port = port
    backend.config.pg_database = database
    backend.config.connection_string = connection_string
    return backend


def _make_config(password: str, sslmode: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.add_section("Database")
    cfg.set("Database", "type", "postgresql")
    cfg.set("Database", "pg_host", "localhost")
    cfg.set("Database", "pg_port", "5432")
    cfg.set("Database", "pg_database", "tldw_content")
    cfg.set("Database", "pg_user", "tldw_user")
    cfg.set("Database", "pg_password", password)
    cfg.set("Database", "pg_sslmode", sslmode)
    return cfg


def _make_sqlite_config(sqlite_path: str) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.add_section("Database")
    cfg.set("Database", "type", "sqlite")
    cfg.set("Database", "sqlite_path", sqlite_path)
    return cfg


def test_get_content_backend_retires_superseded_cached_backend_until_references_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_backend = _FakeBackend()
    new_backend = _FakeBackend()
    old_pool = old_backend.pool
    old_ref = weakref.ref(old_backend)

    monkeypatch.setattr(content_backend, "_cached_backend", old_backend)
    monkeypatch.setattr(content_backend, "_cached_backend_signature", ("old",))
    monkeypatch.setattr(
        content_backend.DatabaseBackendFactory,
        "create_backend",
        staticmethod(lambda _cfg: new_backend),
    )

    cfg = _make_config("pw-new", "prefer")
    backend = content_backend.get_content_backend(cfg)

    if backend is not new_backend:
        pytest.fail("expected replacement backend to be returned")
    if old_pool.closed != 0:
        pytest.fail("expected superseded cached backend pool to remain open while references exist")

    del old_backend
    gc.collect()

    if old_ref() is not None:
        pytest.fail("expected superseded backend to be garbage collected after references release")
    if old_pool.closed != 1:
        pytest.fail("expected retired backend pool to close after the last reference is released")


def test_reset_media_runtime_defaults_retires_cached_backend_until_references_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cached_backend = _FakeBackend()
    cached_pool = cached_backend.pool
    cached_ref = weakref.ref(cached_backend)

    monkeypatch.setattr(content_backend, "_cached_backend", cached_backend)
    monkeypatch.setattr(content_backend, "_cached_backend_signature", ("cached",))

    media_runtime_defaults._clear_content_backend_cache()

    if cached_pool.closed != 0:
        pytest.fail("expected cache clear to retire the backend without closing active references")

    del cached_backend
    gc.collect()

    if cached_ref() is not None:
        pytest.fail("expected retired cached backend to be collectible after references release")
    if cached_pool.closed != 1:
        pytest.fail("expected retired cached backend pool to close after references release")


def test_reset_media_runtime_defaults_blocks_stale_backend_reads_during_clear(
    monkeypatch,
    tmp_path,
) -> None:
    stale_backend = _FakeBackend()
    sqlite_cfg = _make_sqlite_config(str(tmp_path / "media.db"))
    clear_started = threading.Event()
    allow_clear = threading.Event()
    load_finished = threading.Event()
    load_result = {}

    monkeypatch.setattr(media_runtime_defaults, "postgres_content_mode", True)
    monkeypatch.setattr(media_runtime_defaults, "content_db_backend", stale_backend)
    monkeypatch.setattr(media_runtime_defaults, "single_user_config", _make_config("pw1", "prefer"))

    def fake_clear_cached_backend() -> None:
        clear_started.set()
        allow_clear.wait(timeout=2)

    monkeypatch.setattr(content_backend, "clear_cached_backend", fake_clear_cached_backend)

    def run_reset() -> None:
        media_runtime_defaults.reset_media_runtime_defaults(config=sqlite_cfg, reload=False)

    def run_load() -> None:
        load_result["backend"] = media_runtime_defaults.ensure_content_backend_loaded()
        load_finished.set()

    reset_thread = threading.Thread(target=run_reset)
    reset_thread.start()

    if not clear_started.wait(timeout=2):
        pytest.fail("expected reset to begin clearing the cached backend")

    load_thread = threading.Thread(target=run_load)
    load_thread.start()

    if load_finished.wait(timeout=0.2):
        pytest.fail("expected runtime backend reads to block until reset finished")

    allow_clear.set()
    reset_thread.join(timeout=2)
    load_thread.join(timeout=2)

    if reset_thread.is_alive() or load_thread.is_alive():
        pytest.fail("expected reset/load threads to complete after cache clear was released")
    if load_result.get("backend") is not None:
        pytest.fail("expected reset to prevent returning the stale backend after cache clear")


def test_content_backend_cache_includes_password_and_sslmode(monkeypatch) -> None:
    created = []

    def fake_create(cfg):
        obj = object()
        created.append(obj)
        return obj

    monkeypatch.delenv("TLDW_CONTENT_DB_BACKEND", raising=False)
    monkeypatch.delenv("TLDW_CONTENT_PG_PASSWORD", raising=False)
    monkeypatch.delenv("TLDW_PG_PASSWORD", raising=False)
    monkeypatch.delenv("POSTGRES_TEST_PASSWORD", raising=False)
    monkeypatch.delenv("TLDW_CONTENT_PG_SSLMODE", raising=False)
    monkeypatch.delenv("TLDW_PG_SSLMODE", raising=False)

    monkeypatch.setattr(content_backend, "_cached_backend", None)
    monkeypatch.setattr(content_backend, "_cached_backend_signature", None)
    monkeypatch.setattr(
        content_backend.DatabaseBackendFactory,
        "create_backend",
        staticmethod(fake_create),
    )

    cfg = _make_config("pw1", "prefer")
    backend_a = content_backend.get_content_backend(cfg)
    backend_b = content_backend.get_content_backend(cfg)
    if backend_a is not backend_b:
        pytest.fail("expected identical config to reuse cached backend")
    if len(created) != 1:
        pytest.fail("expected only one backend instance for identical config")

    cfg.set("Database", "pg_password", "pw2")
    backend_c = content_backend.get_content_backend(cfg)
    if backend_c is backend_a:
        pytest.fail("expected password change to invalidate cached backend signature")

    cfg.set("Database", "pg_sslmode", "require")
    backend_d = content_backend.get_content_backend(cfg)
    if backend_d is backend_c:
        pytest.fail("expected sslmode change to invalidate cached backend signature")


def test_collections_database_refreshes_shared_backend_after_cache_rotation(monkeypatch) -> None:
    old_backend = _FakeBackend(label="old")
    new_backend = _FakeBackend(label="new")
    holder = {"backend": old_backend}

    monkeypatch.setenv("TLDW_CONTENT_DB_BACKEND", "postgresql")
    monkeypatch.setattr(collections_db, "load_comprehensive_config", lambda: _make_config("pw1", "prefer"))
    monkeypatch.setattr(collections_db, "get_content_backend", lambda _cfg: holder["backend"])
    monkeypatch.setattr(collections_db.CollectionsDatabase, "ensure_schema", lambda self: None)
    monkeypatch.setattr(
        collections_db.CollectionsDatabase,
        "_seed_watchlists_output_templates",
        lambda self: None,
    )
    monkeypatch.setattr(
        collections_db,
        "prepare_backend_statement",
        lambda *_args, **_kwargs: ("SELECT 1", ()),
    )

    db = collections_db.CollectionsDatabase(user_id="1")

    holder["backend"] = new_backend
    result = db._execute_insert("INSERT INTO collections VALUES (?)", ("value",))

    assert result == "new"
    assert [(query, params) for query, params, _conn in new_backend.execute_calls] == [("SELECT 1", ())]
    assert old_backend.execute_calls == []


def test_collections_database_keeps_existing_shared_backend_when_refresh_falls_back_to_sqlite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_backend = _FakeBackend(label="old")
    sqlite_fallback = _FakeBackend(backend_type=BackendType.SQLITE, label="sqlite-fallback")
    resolve_calls = {"count": 0}

    def fake_resolve(self):
        resolve_calls["count"] += 1
        if resolve_calls["count"] == 1:
            return old_backend, True
        return sqlite_fallback, False

    monkeypatch.setattr(collections_db.CollectionsDatabase, "_resolve_backend", fake_resolve)
    monkeypatch.setattr(collections_db.CollectionsDatabase, "_run_backend_bootstrap", lambda self: None)
    monkeypatch.setattr(
        collections_db,
        "prepare_backend_statement",
        lambda *_args, **_kwargs: ("SELECT 1", ()),
    )

    db = collections_db.CollectionsDatabase(user_id="1")

    result = db._execute_insert("INSERT INTO collections VALUES (?)", ("value",))

    assert result == "old"
    assert db._backend is old_backend
    assert db._uses_shared_content_backend is True
    assert sqlite_fallback.execute_calls == []
    assert [(query, params) for query, params, _conn in old_backend.execute_calls] == [("SELECT 1", ())]


def test_collections_database_keeps_previous_backend_when_rotated_bootstrap_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_backend = _FakeBackend(label="old")
    new_backend = _FakeBackend(label="new")
    holder = {"backend": old_backend}

    monkeypatch.setattr(collections_db.CollectionsDatabase, "_bootstrapped_backend_targets", set())
    monkeypatch.setattr(
        collections_db.CollectionsDatabase,
        "_resolve_backend",
        lambda self: (holder["backend"], True),
    )
    monkeypatch.setattr(collections_db.CollectionsDatabase, "_run_backend_bootstrap", lambda self: None)

    def fail_bootstrap(self, backend) -> None:
        if backend is new_backend:
            raise RuntimeError(f"bootstrap failed for {backend.label}")

    monkeypatch.setattr(
        collections_db.CollectionsDatabase,
        "_ensure_bootstrap_for_backend",
        fail_bootstrap,
    )

    db = collections_db.CollectionsDatabase(user_id="1")

    holder["backend"] = new_backend

    with pytest.raises(RuntimeError, match="bootstrap failed for new"):
        _ = db.backend

    assert db._backend is old_backend
    assert db._uses_shared_content_backend is True


def test_collections_database_pins_backend_for_multi_query_read_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_backend = _FakeBackend(label="old")
    new_backend = _FakeBackend(label="new")
    holder = {"backend": old_backend}

    row_template = {
        "id": 1,
        "user_id": "1",
        "name": "old",
        "type": "summary",
        "format": "markdown",
        "body": "template",
        "description": None,
        "is_default": True,
        "created_at": "2026-04-16T00:00:00+00:00",
        "updated_at": "2026-04-16T00:00:00+00:00",
        "metadata_json": None,
    }

    def make_execute(backend: _FakeBackend):
        def execute(query: str, params=None, connection=None) -> QueryResult:
            backend.execute_calls.append((query, params, connection))
            if "COUNT(*) AS cnt FROM output_templates" in query:
                if backend is old_backend:
                    holder["backend"] = new_backend
                return QueryResult(rows=[{"cnt": 1}], rowcount=1)
            if "FROM output_templates" in query:
                row = {**row_template, "name": backend.label}
                return QueryResult(rows=[row], rowcount=1)
            return QueryResult(rows=[], rowcount=0)

        return execute

    old_backend.execute = make_execute(old_backend)  # type: ignore[assignment]
    new_backend.execute = make_execute(new_backend)  # type: ignore[assignment]

    monkeypatch.setattr(
        collections_db.CollectionsDatabase,
        "_resolve_backend",
        lambda self: (holder["backend"], True),
    )
    monkeypatch.setattr(collections_db.CollectionsDatabase, "_run_backend_bootstrap", lambda self: None)

    db = collections_db.CollectionsDatabase(user_id="1")

    rows, total = db.list_output_templates(None, 10, 0)

    assert total == 1
    assert [row.name for row in rows] == ["old"]
    assert len(old_backend.execute_calls) == 2
    assert new_backend.execute_calls == []


def test_characters_rag_db_pins_backend_for_thread_local_connection_until_close(monkeypatch) -> None:
    old_backend = _FakeBackend(label="old")
    new_backend = _FakeBackend(label="new")
    holder = {"backend": old_backend}

    monkeypatch.setattr(
        chacha_db.CharactersRAGDB,
        "_resolve_backend",
        lambda self, *, backend, config: (holder["backend"], True),
    )
    monkeypatch.setattr(chacha_db.CharactersRAGDB, "_initialize_schema", lambda self: None)
    monkeypatch.setattr(
        chacha_db.CharactersRAGDB,
        "_apply_postgres_client_scope",
        lambda self, conn, pool: conn,
    )

    db = chacha_db.CharactersRAGDB(db_path=":memory:", client_id="tester")

    conn = db.get_connection()
    raw = db._local.conn

    holder["backend"] = new_backend
    cursor = conn.cursor()
    cursor.execute("SELECT 1")
    db.close_connection()

    assert [issued.origin for issued in old_backend.pool.issued] == ["old"]
    assert old_backend.pool.returned == [raw]
    assert new_backend.pool.returned == []
    assert [
        (query, params, connection.origin if connection else None)
        for query, params, connection in old_backend.execute_calls
    ] == [("SELECT 1", None, "old")]
    assert new_backend.execute_calls == []


def test_characters_rag_db_transaction_path_uses_pinned_backend_wrapper(monkeypatch) -> None:
    old_backend = _FakeBackend(label="old")
    new_backend = _FakeBackend(label="new")
    holder = {"backend": old_backend}

    monkeypatch.setattr(
        chacha_db.CharactersRAGDB,
        "_resolve_backend",
        lambda self, *, backend, config: (holder["backend"], True),
    )
    monkeypatch.setattr(chacha_db.CharactersRAGDB, "_initialize_schema", lambda self: None)
    monkeypatch.setattr(
        chacha_db.CharactersRAGDB,
        "_apply_postgres_client_scope",
        lambda self, conn, pool: conn,
    )

    db = chacha_db.CharactersRAGDB(db_path=":memory:", client_id="tester")

    with db.transaction() as conn:
        raw = db._local.conn
        holder["backend"] = new_backend
        conn.cursor().execute("SELECT 1")

    assert raw.commit_calls == 1
    assert old_backend.pool.returned == []
    assert [
        (query, params, connection.origin if connection else None)
        for query, params, connection in old_backend.execute_calls
    ] == [("SELECT 1", None, "old")]
    assert new_backend.execute_calls == []


def test_characters_rag_db_serializes_shared_backend_refresh_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_backend = _FakeBackend(label="old")
    new_backend = _FakeBackend(label="new")
    holder = {"backend": old_backend}
    first_refresh_started = threading.Event()
    release_refresh = threading.Event()
    bootstrap_keys: list[str] = []

    def fake_resolve(self, *, backend, config):
        if holder["backend"] is new_backend:
            if not first_refresh_started.is_set():
                first_refresh_started.set()
                release_refresh.wait(timeout=0.1)
            else:
                release_refresh.set()
        return holder["backend"], True

    def slow_initialize_schema(self) -> None:
        bootstrap_keys.append(self._backend_target_key(self._backend) or "<none>")
        threading.Event().wait(0.1)

    monkeypatch.setattr(chacha_db.CharactersRAGDB, "_resolve_backend", fake_resolve)
    monkeypatch.setattr(chacha_db.CharactersRAGDB, "_initialize_schema", slow_initialize_schema)
    monkeypatch.setattr(
        chacha_db.CharactersRAGDB,
        "_apply_postgres_client_scope",
        lambda self, conn, pool: conn,
    )

    db = chacha_db.CharactersRAGDB(db_path=":memory:", client_id="tester")
    bootstrap_keys.clear()
    holder["backend"] = new_backend

    errors: list[BaseException] = []

    def read_backend() -> None:
        try:
            _ = db.backend
        except BaseException as exc:  # pragma: no cover - surfaced by assert below
            errors.append(exc)

    first = threading.Thread(target=read_backend)
    second = threading.Thread(target=read_backend)

    first.start()
    second.start()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert bootstrap_keys == ["postgresql:///new"]


def test_evaluations_database_keeps_borrowed_backend_pinned_across_rotation(monkeypatch) -> None:
    old_backend = _FakeBackend(label="old")
    new_backend = _FakeBackend(label="new")
    holder = {"backend": old_backend}

    monkeypatch.setattr(
        evaluations_db.EvaluationsDatabase,
        "_resolve_backend",
        lambda self: (holder["backend"], True),
    )
    monkeypatch.setattr(evaluations_db.EvaluationsDatabase, "_initialize_database_postgres", lambda self: None)
    monkeypatch.setattr(evaluations_db.EvaluationsDatabase, "_init_abtest_store", lambda self: None)

    db = evaluations_db.EvaluationsDatabase(db_path=None)

    with db.get_connection() as conn:
        raw = conn._conn
        holder["backend"] = new_backend
        conn.cursor().execute("SELECT 1")

    assert [issued.origin for issued in old_backend.pool.issued] == ["old"]
    assert old_backend.pool.returned == [raw]
    assert new_backend.pool.returned == []
    assert [
        (query, params, connection.origin if connection else None)
        for query, params, connection in old_backend.execute_calls
    ] == [("SELECT 1", None, "old")]
    assert new_backend.execute_calls == []


def test_evaluations_database_reinitializes_abtest_store_when_shared_backend_rotates(monkeypatch) -> None:
    old_backend = _set_fake_pg_target(_FakeBackend(label="old"), host="pg-a", port=5432, database="shared", connection_string=None)
    new_backend = _set_fake_pg_target(_FakeBackend(label="new"), host="pg-b", port=6432, database="shared", connection_string=None)
    holder = {"backend": old_backend}
    stamps: list[str] = []

    monkeypatch.setattr(
        evaluations_db.EvaluationsDatabase,
        "_resolve_backend",
        lambda self: (holder["backend"], True),
    )
    monkeypatch.setattr(evaluations_db.EvaluationsDatabase, "_initialize_database_postgres", lambda self: None)

    def stamp_abtest_store(self) -> None:
        stamp = self._backend_target_key(self._backend) or "<none>"
        self._abtest_store = stamp
        stamps.append(stamp)

    monkeypatch.setattr(evaluations_db.EvaluationsDatabase, "_init_abtest_store", stamp_abtest_store)

    db = evaluations_db.EvaluationsDatabase(db_path=None)

    holder["backend"] = new_backend
    _ = db.backend  # Trigger backend refresh side effect.  # noqa: B018

    assert stamps == ["pg-a:5432/shared", "pg-b:6432/shared"]
    assert db._abtest_store == "pg-b:6432/shared"


def test_analytics_database_keeps_transaction_backend_pinned_across_rotation(monkeypatch) -> None:
    old_backend = _FakeBackend(label="old")
    new_backend = _FakeBackend(label="new")
    holder = {"backend": old_backend}

    monkeypatch.setattr(
        analytics_db.AnalyticsDatabase,
        "_resolve_backend",
        lambda self, *, db_path, backend, config: (holder["backend"], True),
    )
    monkeypatch.setattr(analytics_db.AnalyticsDatabase, "_initialize_database", lambda self: None)

    db = analytics_db.AnalyticsDatabase(db_path="analytics.db")

    with db.transaction() as conn:
        raw = conn
        holder["backend"] = new_backend
        db._execute(conn, "SELECT 1")

    assert [issued.origin for issued in old_backend.pool.issued] == ["old"]
    assert old_backend.pool.returned == [raw]
    assert new_backend.pool.returned == []
    assert [
        (query, params, connection.origin if connection else None)
        for query, params, connection in old_backend.execute_calls
    ] == [("SELECT 1", None, "old")]
    assert new_backend.execute_calls == []


def test_analytics_database_initialization_uses_pinned_backend_during_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_backend = _FakeBackend(label="old")
    new_backend = _FakeBackend(label="new")
    holder = {"backend": old_backend}

    @contextmanager
    def rotating_transaction():
        connection = old_backend.pool.get_connection()
        holder["backend"] = new_backend
        try:
            yield connection
        finally:
            old_backend.pool.returned.append(connection)

    old_backend.transaction = rotating_transaction  # type: ignore[assignment]

    monkeypatch.setattr(
        analytics_db.AnalyticsDatabase,
        "_resolve_backend",
        lambda self, *, db_path, backend, config: (holder["backend"], True),
    )
    monkeypatch.setattr(analytics_db.AnalyticsDatabase, "_bootstrapped_backend_targets", set())

    _ = analytics_db.AnalyticsDatabase(db_path="analytics.db")

    assert len(old_backend.create_tables_calls) == 1
    assert len(old_backend.execute_calls) == len(analytics_db.AnalyticsDatabase._INDEX_STATEMENTS)
    assert new_backend.create_tables_calls == []
    assert new_backend.execute_calls == []


def test_collections_database_reruns_bootstrap_when_shared_backend_target_changes(monkeypatch) -> None:
    old_backend = _FakeBackend(label="old")
    new_backend = _FakeBackend(label="new")
    holder = {"backend": old_backend}
    schema_calls: list[str] = []
    seed_calls: list[str] = []

    monkeypatch.setattr(collections_db.CollectionsDatabase, "_bootstrapped_backend_targets", set())
    monkeypatch.setenv("TLDW_CONTENT_DB_BACKEND", "postgresql")
    monkeypatch.setattr(collections_db, "load_comprehensive_config", lambda: _make_config("pw1", "prefer"))
    monkeypatch.setattr(collections_db, "get_content_backend", lambda _cfg: holder["backend"])
    monkeypatch.setattr(
        collections_db.CollectionsDatabase,
        "ensure_schema",
        lambda self: schema_calls.append(self._backend.label),
    )
    monkeypatch.setattr(
        collections_db.CollectionsDatabase,
        "_seed_watchlists_output_templates",
        lambda self: seed_calls.append(self._backend.label),
    )
    monkeypatch.setattr(
        collections_db,
        "prepare_backend_statement",
        lambda *_args, **_kwargs: ("SELECT 1", ()),
    )

    db = collections_db.CollectionsDatabase(user_id="1")

    holder["backend"] = new_backend
    db._execute_insert("INSERT INTO collections VALUES (?)", ("value",))

    assert schema_calls == ["old", "new"]
    assert seed_calls == ["old", "new"]


def test_collections_database_bootstrap_runs_once_per_shared_target_across_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_backend = _FakeBackend(label="shared")
    bootstrap_calls: list[str] = []

    monkeypatch.setattr(collections_db.CollectionsDatabase, "_bootstrapped_backend_targets", set())
    monkeypatch.setattr(
        collections_db.CollectionsDatabase,
        "_resolve_backend",
        lambda self: (shared_backend, True),
    )
    monkeypatch.setattr(
        collections_db.CollectionsDatabase,
        "_run_backend_bootstrap",
        lambda self: bootstrap_calls.append(self._backend_target_key(self._backend) or "<none>"),
    )

    _ = collections_db.CollectionsDatabase(user_id="1")
    _ = collections_db.CollectionsDatabase(user_id="1")

    assert bootstrap_calls == ["postgresql:///shared"]


def test_evaluations_database_bootstrap_runs_once_per_shared_target_across_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_backend = _FakeBackend(label="shared")
    bootstrap_calls: list[str] = []

    monkeypatch.setattr(evaluations_db.EvaluationsDatabase, "_bootstrapped_backend_targets", set())
    monkeypatch.setattr(
        evaluations_db.EvaluationsDatabase,
        "_resolve_backend",
        lambda self: (shared_backend, True),
    )
    monkeypatch.setattr(
        evaluations_db.EvaluationsDatabase,
        "_initialize_database_postgres",
        lambda self: bootstrap_calls.append(self._backend_target_key(self._backend) or "<none>"),
    )
    monkeypatch.setattr(evaluations_db.EvaluationsDatabase, "_init_abtest_store", lambda self: None)

    _ = evaluations_db.EvaluationsDatabase(db_path=None)
    _ = evaluations_db.EvaluationsDatabase(db_path=None)

    assert bootstrap_calls == ["postgresql:///shared"]


def test_analytics_database_bootstrap_runs_once_per_shared_target_across_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_backend = _FakeBackend(label="shared")
    bootstrap_calls: list[str] = []

    monkeypatch.setattr(analytics_db.AnalyticsDatabase, "_bootstrapped_backend_targets", set())
    monkeypatch.setattr(
        analytics_db.AnalyticsDatabase,
        "_resolve_backend",
        lambda self, *, db_path, backend, config: (shared_backend, True),
    )
    monkeypatch.setattr(
        analytics_db.AnalyticsDatabase,
        "_initialize_database",
        lambda self: bootstrap_calls.append(self._backend_target_key(self._backend) or "<none>"),
    )

    _ = analytics_db.AnalyticsDatabase(db_path="analytics.db")
    _ = analytics_db.AnalyticsDatabase(db_path="analytics.db")

    assert bootstrap_calls == ["postgresql:///shared"]


def test_watchlists_database_reruns_bootstrap_when_pg_host_changes_with_same_db_name(monkeypatch) -> None:
    old_backend = _set_fake_pg_target(_FakeBackend(label="old"), host="pg-a", port=5432, database="shared", connection_string=None)
    new_backend = _set_fake_pg_target(_FakeBackend(label="new"), host="pg-b", port=6432, database="shared", connection_string=None)
    holder = {"backend": old_backend}
    schema_calls: list[str] = []

    monkeypatch.setenv("TLDW_CONTENT_DB_BACKEND", "postgresql")
    monkeypatch.setattr(watchlists_db, "load_comprehensive_config", lambda: _make_config("pw1", "prefer"))
    monkeypatch.setattr(watchlists_db, "get_content_backend", lambda _cfg: holder["backend"])
    monkeypatch.setattr(
        watchlists_db.WatchlistsDatabase,
        "ensure_schema",
        lambda self: schema_calls.append(self._backend_target_key(self._backend) or "<none>"),
    )
    monkeypatch.setattr(watchlists_db.WatchlistsDatabase, "_schema_init_keys", set())

    db = watchlists_db.WatchlistsDatabase(user_id="1")

    holder["backend"] = new_backend
    _ = db.backend  # Trigger backend refresh side effect.  # noqa: B018

    assert schema_calls == ["pg-a:5432/shared", "pg-b:6432/shared"]


def test_watchlists_database_pins_backend_for_multi_query_read_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_backend = _FakeBackend(label="old")
    new_backend = _FakeBackend(label="new")
    holder = {"backend": old_backend}

    source_row = {
        "id": 1,
        "user_id": "1",
        "name": "Example source",
        "url": "https://example.invalid/feed",
        "source_type": "rss",
        "active": 1,
        "settings_json": None,
        "last_scraped_at": None,
        "etag": None,
        "last_modified": None,
        "defer_until": None,
        "status": None,
        "consec_not_modified": 0,
        "consec_errors": 0,
        "created_at": "2026-04-16T00:00:00+00:00",
        "updated_at": "2026-04-16T00:00:00+00:00",
    }

    def make_execute(backend: _FakeBackend):
        def execute(query: str, params=None, connection=None) -> QueryResult:
            backend.execute_calls.append((query, params, connection))
            if "FROM sources WHERE id" in query:
                if backend is old_backend:
                    holder["backend"] = new_backend
                return QueryResult(rows=[source_row], rowcount=1)
            if "FROM source_tags" in query:
                return QueryResult(rows=[{"name": backend.label}], rowcount=1)
            return QueryResult(rows=[], rowcount=0)

        return execute

    old_backend.execute = make_execute(old_backend)  # type: ignore[assignment]
    new_backend.execute = make_execute(new_backend)  # type: ignore[assignment]

    monkeypatch.setattr(
        watchlists_db.WatchlistsDatabase,
        "_resolve_backend",
        lambda self: (
            holder["backend"],
            f"postgresql:///{holder['backend'].label}",
            True,
        ),
    )
    monkeypatch.setattr(watchlists_db.WatchlistsDatabase, "_ensure_schema_for_key", lambda self, backend, db_key: None)

    db = watchlists_db.WatchlistsDatabase(user_id="1")

    source = db.get_source(1)

    assert source.tags == ["old"]
    assert len(old_backend.execute_calls) == 2
    assert new_backend.execute_calls == []


def test_watchlists_database_pins_backend_for_multi_query_write_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_backend = _FakeBackend(label="old")
    new_backend = _FakeBackend(label="new")
    holder = {"backend": old_backend}

    source_row = {
        "id": 101,
        "user_id": "1",
        "name": "Created source",
        "url": "https://example.invalid/created",
        "source_type": "rss",
        "active": 1,
        "settings_json": None,
        "last_scraped_at": None,
        "etag": None,
        "last_modified": None,
        "defer_until": None,
        "status": None,
        "consec_not_modified": 0,
        "consec_errors": 0,
        "created_at": "2026-04-16T00:00:00+00:00",
        "updated_at": "2026-04-16T00:00:00+00:00",
    }

    def make_execute(backend: _FakeBackend):
        def execute(query: str, params=None, connection=None) -> QueryResult:
            backend.execute_calls.append((query, params, connection))
            if query.startswith("INSERT INTO sources"):
                if backend is old_backend:
                    holder["backend"] = new_backend
                return QueryResult(rows=[{"id": 101}], rowcount=1, lastrowid=101)
            if "FROM sources WHERE id" in query:
                return QueryResult(rows=[source_row], rowcount=1)
            if "FROM source_tags" in query:
                return QueryResult(rows=[], rowcount=0)
            return QueryResult(rows=[], rowcount=0)

        return execute

    old_backend.execute = make_execute(old_backend)  # type: ignore[assignment]
    new_backend.execute = make_execute(new_backend)  # type: ignore[assignment]

    monkeypatch.setattr(
        watchlists_db.WatchlistsDatabase,
        "_resolve_backend",
        lambda self: (
            holder["backend"],
            f"postgresql:///{holder['backend'].label}",
            True,
        ),
    )
    monkeypatch.setattr(watchlists_db.WatchlistsDatabase, "_ensure_schema_for_key", lambda self, backend, db_key: None)
    monkeypatch.setattr(
        watchlists_db,
        "prepare_backend_statement",
        lambda *_args, **_kwargs: ("INSERT INTO sources", ()),
    )

    db = watchlists_db.WatchlistsDatabase(user_id="1")

    source = db.create_source(
        name="Created source",
        url="https://example.invalid/created",
        source_type="rss",
    )

    assert source.id == 101
    assert [query for query, _params, _connection in old_backend.execute_calls] == [
        "INSERT INTO sources",
        "SELECT id, user_id, name, url, source_type, active, settings_json, last_scraped_at, etag, last_modified, defer_until, status, consec_not_modified, consec_errors, created_at, updated_at FROM sources WHERE id = ? AND user_id = ?",
        "SELECT t.name FROM source_tags st JOIN tags t ON st.tag_id = t.id WHERE st.source_id = ?",
    ]
    assert new_backend.execute_calls == []
