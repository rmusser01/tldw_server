import configparser
from dataclasses import dataclass

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.RAG.rag_service import analytics_db


@dataclass(frozen=True)
class FakeBackendConfig:
    connection_string: str


class FakePostgresBackend:
    backend_type = BackendType.POSTGRESQL

    def __init__(self, name: str):
        self.name = name
        self.config = FakeBackendConfig(connection_string=f"postgresql://example/{name}")
        self.bootstrap_calls = 0

    def execute(self, *args, **kwargs):
        return []

    def fetch_all(self, *args, **kwargs):
        return []

    def fetch_one(self, *args, **kwargs):
        return None

    def transaction(self):
        class Transaction:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        return Transaction()


@pytest.fixture(autouse=True)
def isolate_bootstrapped_backend_targets():
    original_targets = set(analytics_db.AnalyticsDatabase._bootstrapped_backend_targets)
    analytics_db.AnalyticsDatabase._bootstrapped_backend_targets.clear()
    try:
        yield
    finally:
        analytics_db.AnalyticsDatabase._bootstrapped_backend_targets.clear()
        analytics_db.AnalyticsDatabase._bootstrapped_backend_targets.update(original_targets)


def test_analytics_database_refreshes_shared_content_backend_after_error(monkeypatch, tmp_path):
    first_backend = FakePostgresBackend("first")
    second_backend = FakePostgresBackend("second")
    backend_calls = iter([first_backend, second_backend])

    monkeypatch.setattr(analytics_db, "get_content_backend", lambda config: next(backend_calls))
    monkeypatch.setattr(analytics_db, "load_comprehensive_config", lambda: configparser.ConfigParser())
    monkeypatch.setattr(analytics_db.AnalyticsDatabase, "_initialize_database", lambda self: None)
    monkeypatch.setattr(
        analytics_db.AnalyticsDatabase,
        "_ensure_bootstrap_for_backend",
        lambda self, backend: setattr(backend, "bootstrap_calls", backend.bootstrap_calls + 1),
    )

    db = analytics_db.AnalyticsDatabase(db_path=str(tmp_path / "analytics.db"))

    def stale_execute(*args, **kwargs):  # noqa: ANN002, ANN003
        raise BackendDatabaseError("stale shared backend")

    first_backend.execute = stale_execute

    assert first_backend.bootstrap_calls == 1
    assert db.backend is first_backend
    with pytest.raises(BackendDatabaseError):
        db._execute_on_backend(first_backend, object(), "SELECT 1")
    assert db.backend is second_backend
    assert second_backend.bootstrap_calls == 1


def test_analytics_database_bootstraps_refreshed_backend_before_publish_after_error(monkeypatch, tmp_path):
    first_backend = FakePostgresBackend("first")
    second_backend = FakePostgresBackend("second")
    backend_calls = iter([first_backend, second_backend])
    bootstrap_targets: list[str | None] = []

    def fake_bootstrap(self, backend, target_identifier=None):  # noqa: ANN001
        bootstrap_targets.append(target_identifier)
        if backend is second_backend:
            assert self._backend is first_backend

    monkeypatch.setattr(analytics_db, "get_content_backend", lambda config: next(backend_calls))
    monkeypatch.setattr(analytics_db, "load_comprehensive_config", lambda: configparser.ConfigParser())
    monkeypatch.setattr(analytics_db.AnalyticsDatabase, "_bootstrap_backend_schema", fake_bootstrap)

    db = analytics_db.AnalyticsDatabase(db_path=str(tmp_path / "analytics.db"))

    def stale_execute(*args, **kwargs):  # noqa: ANN002, ANN003
        raise BackendDatabaseError("stale shared backend")

    first_backend.execute = stale_execute

    assert db.backend is first_backend
    with pytest.raises(BackendDatabaseError):
        db._execute_on_backend(first_backend, object(), "SELECT 1")
    assert db.backend is second_backend
    assert db._backend is second_backend
    assert bootstrap_targets[-1] == "postgresql://example/second"


def test_analytics_database_tracks_bootstrap_per_backend_target(monkeypatch, tmp_path):
    backend = FakePostgresBackend("stable")

    monkeypatch.setattr(analytics_db, "get_content_backend", lambda config: backend)
    monkeypatch.setattr(analytics_db, "load_comprehensive_config", lambda: configparser.ConfigParser())
    monkeypatch.setattr(
        analytics_db.AnalyticsDatabase,
        "_bootstrap_backend_schema",
        lambda self, backend, target_identifier=None: None,
    )

    db = analytics_db.AnalyticsDatabase(db_path=str(tmp_path / "analytics.db"))
    db._ensure_bootstrap_for_backend(backend)
    db._ensure_bootstrap_for_backend(backend)

    assert len(db._bootstrapped_backend_targets) == 1
