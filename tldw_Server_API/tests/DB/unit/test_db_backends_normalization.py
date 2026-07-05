import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig, BackendType
from tldw_Server_API.app.core.DB_Management.backends.sqlite_backend import SQLiteBackend


def test_backend_alias_postgres(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("TLDW_DB_BACKEND", "postgres")
    cfg = DatabaseConfig.from_env()
    assert cfg.backend_type == BackendType.POSTGRESQL


def test_database_url_driver_suffix(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/testdb")
    cfg = DatabaseConfig.from_env()
    assert cfg.backend_type == BackendType.POSTGRESQL


def test_sqlite_memory_url_remains_memory(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    cfg = DatabaseConfig.from_env()
    assert cfg.backend_type == BackendType.SQLITE
    assert cfg.sqlite_path == ":memory:"


def test_sqlite_file_memory_uri_uses_memory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cfg = DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path="file::memory:?cache=shared")
    backend = SQLiteBackend(cfg)
    conn = backend.connect()
    conn.execute("CREATE TABLE items(id INTEGER PRIMARY KEY, name TEXT)")
    conn.execute("INSERT INTO items(name) VALUES ('alpha')")
    conn.close()
    assert not [path for path in tmp_path.rglob("*") if path.is_file()]


def test_from_env_accepts_y_for_sqlite_flags(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("TLDW_DB_BACKEND", "sqlite")
    monkeypatch.setenv("TLDW_DB_ECHO", "y")
    monkeypatch.setenv("TLDW_SQLITE_WAL_MODE", "y")
    monkeypatch.setenv("TLDW_SQLITE_FOREIGN_KEYS", "y")

    cfg = DatabaseConfig.from_env()
    assert cfg.backend_type == BackendType.SQLITE
    assert cfg.echo is True
    assert cfg.sqlite_wal_mode is True
    assert cfg.sqlite_foreign_keys is True
