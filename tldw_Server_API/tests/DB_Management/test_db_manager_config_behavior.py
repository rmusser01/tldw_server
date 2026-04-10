import os
from configparser import ConfigParser

import pytest

from tldw_Server_API.app.core.DB_Management import DB_Manager
from tldw_Server_API.app.core.DB_Management.UserDatabase_v2 import UserDatabase
from tldw_Server_API.app.core.DB_Management.backends import factory as backend_factory
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.media_db.runtime import session as media_db_session


def _cfg_with(database_type: str = "sqlite", backup_path: str | None = None) -> ConfigParser:
    cfg = ConfigParser()
    cfg.add_section("Database")
    cfg.set("Database", "type", database_type)
    if backup_path is not None:
        cfg.set("Database", "backup_path", backup_path)
    return cfg


@pytest.fixture(autouse=True)
def _reset_backend_registry():
    backend_factory.reset_backend_registry(mode="hard")
    yield
    backend_factory.reset_backend_registry(mode="hard")


def test_backup_dir_env_over_config(monkeypatch, tmp_path):


     # Ensure env var wins over config backup_path
    env_path = str(tmp_path / "env_backups")
    cfg_path = str(tmp_path / "cfg_backups")

    monkeypatch.setenv("TLDW_DB_BACKUP_PATH", env_path)
    monkeypatch.delenv("TLDW_CONTENT_DB_BACKEND", raising=False)

    cfg = _cfg_with(database_type="sqlite", backup_path=cfg_path)
    DB_Manager.reset_content_backend(config=cfg, reload=False)

    assert DB_Manager.single_user_backup_dir == env_path


def test_backup_dir_fallbacks(monkeypatch, tmp_path):


     # When env not set, use config backup_path; otherwise default dir
    monkeypatch.delenv("TLDW_DB_BACKUP_PATH", raising=False)
    monkeypatch.delenv("TLDW_CONTENT_DB_BACKEND", raising=False)

    cfg_path = str(tmp_path / "cfg_backups")
    cfg = _cfg_with(database_type="sqlite", backup_path=cfg_path)
    DB_Manager.reset_content_backend(config=cfg, reload=False)
    assert DB_Manager.single_user_backup_dir == cfg_path

    # No backup_path provided -> use module default
    cfg2 = _cfg_with(database_type="sqlite", backup_path=None)
    DB_Manager.reset_content_backend(config=cfg2, reload=False)
    assert DB_Manager.single_user_backup_dir == DB_Manager._DEFAULT_BACKUP_DIR


def test_db_type_derivation(monkeypatch):


     # Ensure environment does not override type
    monkeypatch.delenv("TLDW_CONTENT_DB_BACKEND", raising=False)

    # sqlite
    DB_Manager.reset_content_backend(config=_cfg_with("sqlite"), reload=False)
    assert DB_Manager.db_type == "sqlite"

    # postgres
    DB_Manager.reset_content_backend(config=_cfg_with("postgres"), reload=False)
    assert DB_Manager.db_type == "postgres"

    # elasticsearch -> explicitly reported as unsupported
    DB_Manager.reset_content_backend(config=_cfg_with("elasticsearch"), reload=False)
    assert DB_Manager.db_type == "elasticsearch"

    # unknown -> warn and fall back to sqlite
    DB_Manager.reset_content_backend(config=_cfg_with("mydb"), reload=False)
    assert DB_Manager.db_type == "sqlite"


def test_reset_content_backend_routes_runtime_reset_through_graceful_mode(monkeypatch):
    calls: list[tuple[ConfigParser, bool, str]] = []

    def _fake_reset_media_runtime_defaults(*, config, reload, reset_mode):
        calls.append((config, reload, reset_mode))
        return None

    monkeypatch.setattr(
        DB_Manager.media_db_runtime_defaults,
        "reset_media_runtime_defaults",
        _fake_reset_media_runtime_defaults,
        raising=True,
    )

    cfg = _cfg_with("sqlite")
    DB_Manager.reset_content_backend(config=cfg, reload=False)

    assert calls == [(cfg, False, "graceful")]


def test_reset_content_backend_evicts_existing_managed_sqlite_when_switching_to_postgres(
    monkeypatch,
    tmp_path,
):
    monkeypatch.delenv("TLDW_CONTENT_DB_BACKEND", raising=False)
    media_db_path = tmp_path / "managed-before-postgres.db"
    sqlite_cfg = _cfg_with("sqlite")
    sqlite_cfg.set("Database", "sqlite_path", str(media_db_path))
    DB_Manager.reset_content_backend(config=sqlite_cfg, reload=False)

    factory = media_db_session.MediaDbFactory.for_sqlite_path(
        str(media_db_path),
        client_id="cleanup-check",
    )
    backend = factory.backend
    assert backend is not None
    assert backend_factory.is_factory_managed_backend(backend) is True

    DB_Manager.reset_content_backend(config=_cfg_with("postgres"), reload=False)

    assert backend_factory.is_factory_managed_backend(backend) is False


def test_reset_content_backend_keeps_unrelated_userdatabase_shared_sqlite_backend_alive(
    monkeypatch,
    tmp_path,
):
    monkeypatch.delenv("TLDW_CONTENT_DB_BACKEND", raising=False)

    class _ImmediateThread:
        def __init__(self, *, target=None, args=(), kwargs=None, daemon=None):
            self._target = target
            self._args = args
            self._kwargs = kwargs or {}

        def start(self) -> None:
            if self._target is not None:
                self._target(*self._args, **self._kwargs)

    monkeypatch.setattr(backend_factory.time, "sleep", lambda _seconds: None, raising=True)
    monkeypatch.setattr(backend_factory, "Thread", _ImmediateThread, raising=True)

    media_db_path = tmp_path / "managed-media.db"
    users_db_path = tmp_path / "managed-users.db"

    sqlite_cfg = _cfg_with("sqlite")
    sqlite_cfg.set("Database", "sqlite_path", str(media_db_path))
    DB_Manager.reset_content_backend(config=sqlite_cfg, reload=False)

    media_factory = media_db_session.MediaDbFactory.for_sqlite_path(
        str(media_db_path),
        client_id="media-cleanup",
    )
    media_backend = media_factory.backend
    assert media_backend is not None

    users_db = UserDatabase(
        config=DatabaseConfig(
            backend_type=BackendType.SQLITE,
            sqlite_path=str(users_db_path),
            client_id="users-cleanup",
        ),
        client_id="users-cleanup",
    )
    users_backend = users_db.backend
    users_pool = users_backend.get_pool()
    users_pool.get_connection()

    assert backend_factory.is_factory_managed_backend(media_backend) is True
    assert backend_factory.is_factory_managed_backend(users_backend) is True

    DB_Manager.reset_content_backend(config=_cfg_with("postgres"), reload=False)

    assert backend_factory.is_factory_managed_backend(media_backend) is False
    assert backend_factory.is_factory_managed_backend(users_backend) is True
    assert users_pool.get_connection() is not None
