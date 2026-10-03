"""Behavioral coverage for noncreating Media DB sessions."""

import sqlite3
from configparser import ConfigParser
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends import factory as backend_factory
from tldw_Server_API.app.core.DB_Management.backends import sqlite_backend
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import (
    DatabaseBackendFactory,
    is_factory_managed_backend,
)
from tldw_Server_API.app.core.DB_Management.backends.sqlite_backend import (
    SQLiteBackend,
    SQLiteConnectionPool,
)
from tldw_Server_API.app.core.DB_Management.media_db import api as media_db_api
from tldw_Server_API.app.core.DB_Management.media_db.api import (
    create_media_database,
    managed_media_database,
)
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.media_db.runtime import backend_resolution
from tldw_Server_API.app.core.DB_Management.media_db.runtime.factory import (
    MediaDbRuntimeConfig,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.factory import (
    create_media_database as runtime_create_media_database,
)


def test_existing_only_does_not_create_missing_database(tmp_path):
    path = tmp_path / "absent" / "media.db"
    with pytest.raises(FileNotFoundError):
        with managed_media_database("42", db_path=str(path), existing_only=True):
            pytest.fail("Missing database must not yield a session")
    assert not path.parent.exists()


@pytest.mark.parametrize("factory", [create_media_database, MediaDatabase])
def test_existing_only_constructor_eagerly_rejects_missing_database(tmp_path, factory):
    path = tmp_path / "absent" / "media.db"
    with pytest.raises(FileNotFoundError):
        factory(client_id="42", db_path=str(path), existing_only=True)
    assert not path.parent.exists()


def test_existing_only_preserves_existing_schema_and_rows(tmp_path, monkeypatch):
    path = tmp_path / "media.db"
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("CREATE TABLE existing_data (value TEXT)")
        conn.execute("INSERT INTO existing_data VALUES ('before')")

    def reject_bootstrap(_self):
        pytest.fail("Existing-only sessions must not bootstrap schemas")

    monkeypatch.setattr(MediaDatabase, "_initialize_schema", reject_bootstrap)
    monkeypatch.setattr(MediaDatabase, "initialize_db", reject_bootstrap)
    with managed_media_database("42", db_path=str(path), existing_only=True) as db:
        assert db.execute_query("SELECT value FROM existing_data").fetchone()["value"] == "before"
        with db.transaction():
            db.execute_query("UPDATE existing_data SET value = 'after'")

    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT value FROM existing_data").fetchone()[0] == "after"
        assert conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall() == [
            ("existing_data",)
        ]


def test_existing_only_ignores_creating_backend_and_keeps_registry_unchanged(tmp_path):
    path = tmp_path / "media.db"
    path.touch()
    config = DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(path))
    creating = DatabaseBackendFactory.create_backend(config)
    with managed_media_database(
        "42", db_path=str(path), backend=creating, existing_only=True
    ) as db:
        assert db.backend is not creating
        assert not is_factory_managed_backend(db.backend)
        assert db.backend.config.sqlite_path == path.as_uri() + "?mode=rw"
        assert DatabaseBackendFactory.create_backend(config) is creating
        private_pool = db.backend.get_pool()
    assert private_pool.get_stats()["closed"]
    assert not creating.get_pool().get_stats()["closed"]


@pytest.mark.parametrize("entrypoint", ["constructor", "api", "runtime"])
@pytest.mark.parametrize("file_exists", [False, True])
def test_existing_only_without_supplied_backend_leaves_registry_unchanged(
    tmp_path, monkeypatch, entrypoint, file_exists
):
    path = tmp_path / "fresh-media.db"
    if file_exists:
        path.touch()
    config = ConfigParser()
    monkeypatch.setenv("TLDW_CONTENT_DB_BACKEND", "sqlite")
    monkeypatch.delenv("CONTENT_DB_MODE", raising=False)
    loader_calls = []

    def creating_loader():
        loader_calls.append("called")
        return DatabaseBackendFactory.create_backend(DatabaseConfig(
            backend_type=BackendType.SQLITE, sqlite_path=str(path)
        ))

    runtime = MediaDbRuntimeConfig(
        default_db_path=str(path), default_config=config,
        postgres_content_mode=False, backend_loader=creating_loader,
    )
    monkeypatch.setattr(media_db_api, "build_media_runtime_config", lambda: runtime)
    before = dict(backend_factory._sqlite_backend_registry)

    def open_database():
        if entrypoint == "constructor":
            return MediaDatabase(str(path), "42", config=config, existing_only=True)
        if entrypoint == "api":
            return create_media_database("42", existing_only=True)
        return runtime_create_media_database("42", runtime=runtime, existing_only=True)

    if file_exists:
        db = open_database()
        try:
            assert dict(backend_factory._sqlite_backend_registry) == before
            assert not is_factory_managed_backend(db.backend)
        finally:
            db.close_connection()
    else:
        with pytest.raises(FileNotFoundError):
            open_database()
    assert dict(backend_factory._sqlite_backend_registry) == before
    assert loader_calls == []


def test_existing_only_closes_private_pool_on_body_failure(tmp_path):
    path = tmp_path / "media.db"
    path.touch()
    with pytest.raises(RuntimeError, match="body failure"):
        with managed_media_database("42", db_path=str(path), existing_only=True) as db:
            pool = db.backend.get_pool()
            raise RuntimeError("body failure")
    assert pool.get_stats()["closed"]


def test_existing_only_direct_constructor_closes_private_pool(tmp_path):
    path = tmp_path / "media.db"
    path.touch()
    db = MediaDatabase(db_path=str(path), client_id="42", existing_only=True)
    pool = db.backend.get_pool()
    db.close_connection()
    assert pool.get_stats()["closed"]


def test_existing_only_explicit_initialize_is_noncreating(tmp_path, monkeypatch):
    path = tmp_path / "media.db"
    path.touch()
    with managed_media_database("42", db_path=str(path), existing_only=True) as db:
        monkeypatch.setattr(db, "_initialize_schema", lambda: pytest.fail("Schema bootstrap"))
        assert db.initialize_db() is db


@pytest.mark.parametrize("remove_parent", [False, True])
def test_existing_only_deletion_between_precheck_and_open(tmp_path, monkeypatch, remove_parent):
    path = tmp_path / "owner" / "media.db"
    path.parent.mkdir()
    path.touch()
    pools = []
    original = SQLiteConnectionPool._create_connection

    def delete_then_connect(pool):
        pools.append(pool)
        path.unlink()
        if remove_parent:
            path.parent.rmdir()
        return original(pool)

    monkeypatch.setattr(SQLiteConnectionPool, "_create_connection", delete_then_connect)
    with pytest.raises(FileNotFoundError):
        with managed_media_database("42", db_path=str(path), existing_only=True):
            pytest.fail("Deleted database must not yield a session")
    assert not path.exists()
    assert path.parent.exists() is (not remove_parent)
    assert pools[0].get_stats()["closed"]


@pytest.mark.parametrize("error", [
    PermissionError("permission denied"),
    sqlite3.OperationalError("ambiguous open failure"),
])
def test_existing_only_open_errors_are_not_missing_skips(tmp_path, monkeypatch, error):
    path = tmp_path / "media.db"
    path.touch()
    pools = []

    def fail_connect(pool):
        pools.append(pool)
        path.unlink()
        raise error

    monkeypatch.setattr(SQLiteConnectionPool, "_create_connection", fail_connect)
    with pytest.raises(type(error)) as caught:
        create_media_database("42", db_path=str(path), existing_only=True)
    assert caught.value is error
    assert pools[0].get_stats()["closed"]


def test_existing_only_precheck_permission_error_is_not_absence(tmp_path, monkeypatch):
    path = tmp_path / "media.db"
    path.touch()
    error = PermissionError("stat denied")
    original_stat = Path.stat

    def denied_stat(candidate, *args, **kwargs):
        if candidate == path:
            raise error
        return original_stat(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", denied_stat)
    with pytest.raises(PermissionError) as caught:
        create_media_database("42", db_path=str(path), existing_only=True)
    assert caught.value is error


@pytest.mark.parametrize("postcheck_permission_error", [False, True])
def test_existing_only_cantopen_without_verified_absence_preserves_error(
    tmp_path, monkeypatch, postcheck_permission_error
):
    path = tmp_path / "media.db"
    path.touch()
    error = sqlite3.OperationalError("unable to open database file")
    error.sqlite_errorcode = sqlite3.SQLITE_CANTOPEN

    def fail_connect(_pool):
        if postcheck_permission_error:
            def denied_stat(_path):
                raise PermissionError("postcheck denied")
            monkeypatch.setattr(Path, "stat", denied_stat)
        raise error

    monkeypatch.setattr(SQLiteConnectionPool, "_create_connection", fail_connect)
    with pytest.raises(sqlite3.OperationalError) as caught:
        create_media_database("42", db_path=str(path), existing_only=True)
    assert caught.value is error


def test_existing_only_malformed_database_fails_eagerly_without_schema_repairs(tmp_path, monkeypatch):
    path = tmp_path / "media.db"
    content = b"not a sqlite database" * 100
    path.write_bytes(content)
    pools = []
    original = SQLiteBackend.get_pool

    def record_pool(backend):
        pool = original(backend)
        pools.append(pool)
        return pool

    monkeypatch.setattr(SQLiteBackend, "get_pool", record_pool)
    with pytest.raises(sqlite3.DatabaseError):
        create_media_database("42", db_path=str(path), existing_only=True)
    assert path.read_bytes() == content
    assert pools[-1].get_stats()["closed"]


@pytest.mark.parametrize("filename", ["spaces in name.db", "query?and#fragment.db"])
def test_existing_only_sqlite_uri_escapes_filename(tmp_path, filename):
    path = tmp_path / filename
    path.touch()
    with managed_media_database("42", db_path=str(path), existing_only=True) as db:
        assert db.execute_query("SELECT 1 AS value").fetchone()["value"] == 1
        assert db.backend.config.sqlite_path == path.as_uri() + "?mode=rw"


@pytest.mark.parametrize("method", ["execute_query", "execute_many"])
def test_existing_only_queries_cannot_recreate_a_deleted_database(tmp_path, method):
    path = tmp_path / "media.db"
    path.touch()
    with managed_media_database("42", db_path=str(path), existing_only=True) as db:
        path.unlink()
        try:
            if method == "execute_query":
                db.execute_query("SELECT 1")
            else:
                db.execute_many("CREATE TABLE forbidden (value TEXT)", [()])
        except (sqlite3.Error, media_db_api.DatabaseError):
            pass
        assert not path.exists()


@pytest.mark.parametrize("method", ["connect", "get_pool"])
def test_rw_backend_never_creates_parent_directories(tmp_path, method):
    path = tmp_path / "missing" / "media.db"
    backend = SQLiteBackend(DatabaseConfig(
        backend_type=BackendType.SQLITE, sqlite_path=path.as_uri() + "?mode=rw"
    ))
    with pytest.raises(sqlite3.OperationalError):
        if method == "get_pool":
            backend.get_pool().get_connection()
        else:
            backend.connect()
    assert not path.parent.exists()


def test_existing_only_postgres_preserves_routing_without_creating_local_path(tmp_path, monkeypatch):
    path = tmp_path / "absent" / "unused.db"
    backend = SimpleNamespace(backend_type=BackendType.POSTGRESQL)
    runtime = MediaDbRuntimeConfig(
        default_db_path=str(path), default_config=ConfigParser(),
        postgres_content_mode=True, backend_loader=lambda: backend,
    )
    monkeypatch.setattr(media_db_api, "build_media_runtime_config", lambda: runtime)
    monkeypatch.setattr(MediaDatabase, "_initialize_schema", lambda _self: pytest.fail("Schema bootstrap"))
    with managed_media_database("42", db_path=str(path), existing_only=True) as db:
        assert db.backend is backend
        assert db.backend_type == BackendType.POSTGRESQL
    assert not path.parent.exists()


@pytest.mark.parametrize("routing", ["config", "environment", "default_config"])
def test_existing_only_constructor_preserves_configured_postgres_resolution(tmp_path, monkeypatch, routing):
    path = tmp_path / "absent" / "unused.db"
    backend = SimpleNamespace(backend_type=BackendType.POSTGRESQL)
    config = ConfigParser()
    config.read_dict({"Database": {"type": "postgresql"}})
    monkeypatch.delenv("CONTENT_DB_MODE", raising=False)
    monkeypatch.delenv("TLDW_CONTENT_DB_BACKEND", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(backend_resolution, "is_test_mode", lambda: False)
    if routing == "environment":
        config.set("Database", "type", "sqlite")
        monkeypatch.setenv("TLDW_CONTENT_DB_BACKEND", "postgresql")
    elif routing == "default_config":
        monkeypatch.setattr(backend_resolution, "load_comprehensive_config", lambda: config)
    monkeypatch.setattr(backend_resolution, "get_content_backend", lambda _config: backend)
    monkeypatch.setattr(MediaDatabase, "_initialize_schema", lambda _self: pytest.fail("Schema bootstrap"))
    before = dict(backend_factory._sqlite_backend_registry)
    db = MediaDatabase(
        client_id="42", db_path=str(path), existing_only=True,
        config=None if routing == "default_config" else config,
    )
    try:
        assert db.backend is backend
    finally:
        db.close_connection()
    assert not path.parent.exists()
    assert dict(backend_factory._sqlite_backend_registry) == before


def test_existing_only_failed_postgres_resolution_does_not_fall_back_to_creating_sqlite(tmp_path, monkeypatch):
    path = tmp_path / "fresh-media.db"
    config = ConfigParser()
    config.read_dict({"Database": {"type": "postgresql"}})
    monkeypatch.delenv("TLDW_CONTENT_DB_BACKEND", raising=False)
    monkeypatch.delenv("CONTENT_DB_MODE", raising=False)
    monkeypatch.setattr(backend_resolution, "get_content_backend", lambda _config: None)
    before = dict(backend_factory._sqlite_backend_registry)
    with pytest.raises(media_db_api.DatabaseError, match="PostgreSQL content backend requested"):
        MediaDatabase(client_id="42", db_path=str(path), config=config, existing_only=True)
    assert dict(backend_factory._sqlite_backend_registry) == before
    assert not path.exists()


def test_runtime_existing_only_uses_default_sqlite_path(tmp_path):
    path = tmp_path / "media.db"
    path.touch()
    runtime = MediaDbRuntimeConfig(
        default_db_path=str(path), default_config=ConfigParser(),
        postgres_content_mode=False, backend_loader=lambda: None,
    )
    db = runtime_create_media_database("42", runtime=runtime, existing_only=True)
    try:
        assert db.db_path == path
        assert db.backend.config.sqlite_path == path.as_uri() + "?mode=rw"
    finally:
        db.close_connection()


def test_default_factory_still_creates_database_and_schema(tmp_path):
    path = tmp_path / "created" / "media.db"
    with managed_media_database("42", db_path=str(path)) as db:
        assert db.execute_query("SELECT name FROM sqlite_master WHERE name = 'Media'").fetchone()
    assert path.is_file()


@pytest.mark.parametrize("method", ["connect", "get_pool"])
def test_rw_backend_closes_connection_if_configuration_fails(tmp_path, monkeypatch, method):
    path = tmp_path / "media.db"
    path.touch()
    connections = []
    original_connect = sqlite3.connect
    error = sqlite3.DatabaseError("configuration failure")

    def record_connect(*args, **kwargs):
        conn = original_connect(*args, **kwargs)
        connections.append(conn)
        return conn

    def fail_configure(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(sqlite_backend.sqlite3, "connect", record_connect)
    monkeypatch.setattr(sqlite_backend, "configure_sqlite_connection", fail_configure)
    backend = SQLiteBackend(DatabaseConfig(
        backend_type=BackendType.SQLITE, sqlite_path=path.as_uri() + "?mode=rw"
    ))
    with pytest.raises(sqlite3.DatabaseError) as caught:
        if method == "get_pool":
            backend.get_pool().get_connection()
        else:
            backend.connect()
    assert caught.value is error
    try:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connections[0].execute("SELECT 1")
    finally:
        connections[0].close()
