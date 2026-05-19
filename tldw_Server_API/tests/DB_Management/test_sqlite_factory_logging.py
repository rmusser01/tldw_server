from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends import factory as factory_mod


class _RecordingLogger:
    def __init__(self) -> None:
        self.info_calls: list[str] = []
        self.debug_calls: list[str] = []

    def info(self, message, *args, **kwargs) -> None:
        self.info_calls.append(message.format(*args))

    def debug(self, message, *args, **kwargs) -> None:
        self.debug_calls.append(message.format(*args))


@pytest.fixture(autouse=True)
def _reset_backend_caches():
    factory_mod.close_all_backends()
    yield
    factory_mod.close_all_backends()


def test_create_backend_logs_sqlite_success_at_debug_not_info(tmp_path, monkeypatch):
    recorder = _RecordingLogger()
    monkeypatch.setattr(factory_mod, "logger", recorder, raising=True)

    cfg = DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(tmp_path / "db.sqlite"))

    backend = factory_mod.DatabaseBackendFactory.create_backend(cfg)

    assert backend.backend_type == BackendType.SQLITE
    assert recorder.info_calls == []
    assert any("sqlite" in msg.lower() for msg in recorder.debug_calls)


def test_create_backend_logs_postgres_success_at_info(tmp_path, monkeypatch):
    class FakePostgresBackend:
        def __init__(self, config: DatabaseConfig) -> None:
            self.config = config
            self.backend_type = config.backend_type

    recorder = _RecordingLogger()
    monkeypatch.setattr(factory_mod, "logger", recorder, raising=True)
    monkeypatch.setitem(factory_mod._BACKEND_REGISTRY, BackendType.POSTGRESQL, FakePostgresBackend)

    cfg = DatabaseConfig(backend_type=BackendType.POSTGRESQL, pg_database="db")

    backend = factory_mod.DatabaseBackendFactory.create_backend(cfg)

    assert backend.config == cfg
    assert recorder.debug_calls == []
    assert recorder.info_calls == [f"Creating {BackendType.POSTGRESQL.value} backend"]


def test_create_backend_logs_sqlite_only_on_actual_creation(tmp_path, monkeypatch):
    class _FakeSQLiteBackend:
        def __init__(self, config: DatabaseConfig) -> None:
            self.config = config
            self.backend_type = config.backend_type

    recorder = _RecordingLogger()
    monkeypatch.setattr(factory_mod, "logger", recorder, raising=True)
    monkeypatch.setitem(factory_mod._BACKEND_REGISTRY, BackendType.SQLITE, _FakeSQLiteBackend)

    cfg = DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(tmp_path / "reuse.sqlite"))

    first = factory_mod.DatabaseBackendFactory.create_backend(cfg)
    second = factory_mod.DatabaseBackendFactory.create_backend(cfg)

    assert first is second
    assert len(recorder.debug_calls) == 1
    assert "sqlite" in recorder.debug_calls[0].lower()
