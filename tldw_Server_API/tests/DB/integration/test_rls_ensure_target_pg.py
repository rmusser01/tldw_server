"""The RLS auto-ensure targets the content backend, never AuthNZ's users DB (TASK-13950.21, row 4)."""

import configparser

import pytest

from tldw_Server_API.app.core import config
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.services import startup_infra_services as infra

pytestmark = pytest.mark.integration


@pytest.fixture
def content_backend_env(monkeypatch, pg_temp_db):
    """Point the content backend at the scratch database and put a SQLite users DB in config.txt."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    for name, key in (("HOST", "host"), ("PORT", "port"), ("DATABASE", "database"), ("USER", "user"), ("PASSWORD", "password")):
        monkeypatch.setenv(f"TLDW_PG_{name}", str(pg_temp_db[key]))
    monkeypatch.setenv("TLDW_DB_BACKEND", "postgresql")
    parser = configparser.ConfigParser(interpolation=None)
    parser.add_section("AuthNZ")
    parser.set("AuthNZ", "database_url", "sqlite:///./Databases/users.db")
    monkeypatch.setattr(config, "load_comprehensive_config", lambda: parser)


@pytest.mark.asyncio
async def test_rls_ensure_target_ignores_authnz_database_url(monkeypatch, pg_temp_db, content_backend_env):
    monkeypatch.setattr(infra, "_postgres_content_mode_active", lambda: True)

    seen: list = []
    await infra._maybe_ensure_pg_rls(lambda backend: seen.append(backend.config))

    assert seen[0].backend_type == BackendType.POSTGRESQL
    assert seen[0].pg_database == pg_temp_db["database"]
