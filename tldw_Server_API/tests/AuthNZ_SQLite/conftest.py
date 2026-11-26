"""Local conftest for AuthNZ_SQLite tests.

Pull in shared AuthNZ fixtures so this subtree works even if global
plugin registration is skipped by the CI runner.
"""

try:
    from tldw_Server_API.tests._plugins.authnz_fixtures import (
        authnz_schema_ready,       # noqa: F401
        authnz_schema_ready_sync,  # noqa: F401
        real_audit_service,        # noqa: F401
    )
except Exception:
    # If shared plugin import fails, tests that need these fixtures
    # will error explicitly; do not fail collection here.
    _ = None


import pytest  # noqa: E402

from tldw_Server_API.tests.AuthNZ_SQLite._sessions_http_fixtures import SessionsCtx  # noqa: E402

_SESSIONS_PASSWORD = "Kq7#vLm2$Zp9wRt"


@pytest.fixture
async def two_user_client(tmp_path, monkeypatch, fresh_services):
    """A fresh multi-user SQLite app with verified users alice and bob, both holding the ``user`` role."""
    from pathlib import Path

    from fastapi.testclient import TestClient

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
    from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.config import settings as app_settings
    from tldw_Server_API.tests.AuthNZ_SQLite._user_fixtures import create_authnz_test_user

    user_dbs = tmp_path / "user_dbs"
    user_dbs.mkdir()
    prev_base_dir = app_settings.get("USER_DB_BASE_DIR")
    app_settings.USER_DB_BASE_DIR = str(user_dbs)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(user_dbs))
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users.db'}")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-jwt-key-please-change-1234567890")
    monkeypatch.setenv("CSRF_ENABLED", "false")
    monkeypatch.setenv("MAX_SESSIONS_PER_USER", "4")
    monkeypatch.delenv("REDIS_URL", raising=False)
    reset_settings()
    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))
    ids = {}
    for name in ("alice", "bob"):
        ids[name] = await create_authnz_test_user(
            pool,
            username=name,
            email=f"{name}@example.com",
            password_hash=PasswordService().hash_password(_SESSIONS_PASSWORD),
            is_verified=True,
        )
        await AuthnzUsersRepo(pool).assign_role_if_missing(user_id=ids[name], role_name="user")

    from tldw_Server_API.app.main import app

    try:
        with TestClient(app) as client:
            yield SessionsCtx(client, ids["alice"], ids["bob"], _SESSIONS_PASSWORD)
    finally:
        reset_settings()
        if prev_base_dir is not None:
            app_settings.USER_DB_BASE_DIR = prev_base_dir
