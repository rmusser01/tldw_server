"""Concurrent public AuthNZ database lookups share one completed initialization."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, wait
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.AuthNZ import db_config
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings


class _Database:
    """Record the lifetime of a constructed database without opening external resources."""

    def __init__(self, *, config, client_id: str) -> None:
        self.config = config
        self.client_id = client_id
        self.closed = False
        self.backend = SimpleNamespace(get_pool=lambda: self)

    def close_all(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _cold_auth_database_config(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users.db'}")
    reset_settings()
    monkeypatch.setattr(db_config.AuthDatabaseConfig, "_instance", None)
    monkeypatch.setattr(db_config, "UserDatabase", _Database)
    yield
    if db_config.AuthDatabaseConfig._instance is not None:
        db_config.AuthDatabaseConfig._instance.reset_lazy()
    reset_settings()


def test_concurrent_public_lookups_construct_one_database_and_reuse_it(monkeypatch) -> None:
    # Configuration may already exist while its expensive database cache is cold.
    db_config.AuthDatabaseConfig()
    constructor_entered = threading.Event()
    all_constructors_entered = threading.Event()
    release_constructor = threading.Event()
    created = []

    def construct(**kwargs):
        database = _Database(**kwargs)
        created.append(database)
        constructor_entered.set()
        if len(created) == 4:
            all_constructors_entered.set()
        assert release_constructor.wait(timeout=5), "Constructor watchdog expired"
        return database

    monkeypatch.setattr(db_config, "UserDatabase", construct)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(db_config.get_configured_user_database, f"caller-{index}") for index in range(4)]
        try:
            assert constructor_entered.wait(timeout=5)
            # The broken getter admits every caller; the fixed getter admits only
            # one constructor while the remaining callers wait for publication.
            all_constructors_entered.wait(timeout=0.2)
        finally:
            release_constructor.set()
        databases = [future.result(timeout=5) for future in futures]

    assert len(created) == 1
    assert all(database is databases[0] for database in databases)
    assert db_config.get_configured_user_database("warm-caller") is databases[0]
    assert len(created) == 1


def test_concurrent_public_lookups_wait_for_backend_detection(monkeypatch) -> None:
    detection_entered = threading.Event()
    release_detection = threading.Event()
    original_detect = db_config.AuthDatabaseConfig._detect_backend

    def detect_backend(config):
        detection_entered.set()
        assert release_detection.wait(timeout=5), "Configuration watchdog expired"
        original_detect(config)

    monkeypatch.setattr(db_config.AuthDatabaseConfig, "_detect_backend", detect_backend)
    with ThreadPoolExecutor(max_workers=4) as executor:
        first = executor.submit(db_config.get_configured_user_database, "first-caller")
        try:
            assert detection_entered.wait(timeout=5)
            others = [executor.submit(db_config.get_configured_user_database, f"caller-{index}") for index in range(3)]
            wait(others, timeout=0.2)
        finally:
            release_detection.set()
        databases = [future.result(timeout=5) for future in [first, *others]]

    assert all(database is databases[0] for database in databases)
    assert databases[0].config.sqlite_path.endswith("users.db")


def test_public_lookup_refreshes_config_after_interleaved_lazy_reset(monkeypatch, tmp_path) -> None:
    config_resolved = threading.Event()
    resume_lookup = threading.Event()
    original_get_config = db_config.get_auth_db_config
    config = original_get_config()
    replacement_path = str((tmp_path / "replacement.db").resolve())

    def get_config():
        resolved = original_get_config()
        config_resolved.set()
        assert resume_lookup.wait(timeout=5), "Public lookup watchdog expired"
        return resolved

    monkeypatch.setattr(db_config, "get_auth_db_config", get_config)
    with ThreadPoolExecutor(max_workers=1) as executor:
        getter = executor.submit(db_config.get_configured_user_database, "racing-caller")
        try:
            assert config_resolved.wait(timeout=5)
            monkeypatch.setenv("DATABASE_URL", f"sqlite:///{replacement_path}")
            reset_settings()
            config.reset_lazy()
        finally:
            resume_lookup.set()
        database = getter.result(timeout=5)

    later_database = db_config.get_configured_user_database("later-caller")
    assert database.config.sqlite_path == replacement_path
    assert later_database is database
    assert later_database.config.sqlite_path == replacement_path


@pytest.mark.parametrize("reset_method", ["reset", "reset_lazy"])
def test_reset_during_database_initialization_closes_and_discards_the_created_database(
    monkeypatch, reset_method
) -> None:
    config = db_config.AuthDatabaseConfig()
    constructor_entered = threading.Event()
    release_constructor = threading.Event()
    reset_entered = threading.Event()
    created = []

    def construct(**kwargs):
        database = _Database(**kwargs)
        created.append(database)
        constructor_entered.set()
        assert release_constructor.wait(timeout=5), "Constructor watchdog expired"
        return database

    def reset():
        reset_entered.set()
        getattr(config, reset_method)()

    monkeypatch.setattr(db_config, "UserDatabase", construct)
    with ThreadPoolExecutor(max_workers=2) as executor:
        getter = executor.submit(db_config.get_configured_user_database, "first-caller")
        try:
            assert constructor_entered.wait(timeout=5)
            resetter = executor.submit(reset)
            assert reset_entered.wait(timeout=5)
            wait([resetter], timeout=0.2)
        finally:
            release_constructor.set()
        first = getter.result(timeout=5)
        resetter.result(timeout=5)

    second = db_config.get_configured_user_database("after-reset")
    assert first.closed is True
    assert second is not first
    assert len(created) == 2
