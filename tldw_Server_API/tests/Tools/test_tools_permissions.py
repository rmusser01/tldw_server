from __future__ import annotations

import os
import importlib
from fastapi.testclient import TestClient
import pytest

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal

# Reuse Postgres AuthNZ fixtures (isolated_test_environment) as a plugin.
#
# Why register explicitly here?
# - Some CI/dev invocations run tests from subfolders or with
#   PYTEST_DISABLE_PLUGIN_AUTOLOAD=1, which can prevent project-level plugin
#   declarations in pyproject.toml from loading.
# - Explicitly listing the plugin at the module level ensures this test can be
#   run in isolation (e.g., `pytest -q this_file.py`) and still find the
#   required fixtures without reintroducing a root conftest.
pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_full_fixtures"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("sqlite", id="sqlite"),
        pytest.param("postgres", marks=pytest.mark.postgres, id="postgres"),
    ],
)
def test_tools_execute_forbidden_without_permission_multi_user(backend: str, request: pytest.FixtureRequest):
    # Fail fast on any DB pool acquisition issues during this test
    os.environ.setdefault("TLDW_DB_POOL_TIMEOUT", "2")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")
    os.environ["AUTH_MODE"] = "multi_user"
    # Ensure tools router is enabled regardless of suite defaults
    os.environ["ROUTES_ENABLE"] = ",".join(filter(None, {os.getenv("ROUTES_ENABLE", ""), "tools"}))
    # Provide a valid JWT secret for multi-user settings initialization
    os.environ["JWT_SECRET_KEY"] = os.getenv(
        "JWT_SECRET_KEY", "test-secret-jwt-key-please-change-1234567890-EXTRA"
    )

    if backend == "sqlite":
        # Pin AuthNZ to SQLite, reset singletons, and build a fresh app
        os.environ["TLDW_USER_DB_BACKEND"] = "sqlite"
        os.environ.setdefault("DATABASE_URL", "sqlite:///./Databases/users_tools_test.db")
        from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
        from tldw_Server_API.app.core.AuthNZ.db_config import get_auth_db_config

        reset_settings()
        get_auth_db_config().reset()

        if "tldw_Server_API.app.main" in importlib.sys.modules:
            importlib.reload(importlib.sys.modules["tldw_Server_API.app.main"])  # type: ignore[arg-type]
        main = importlib.import_module("tldw_Server_API.app.main")
        app = getattr(main, "app")

        # Override request user to a non-admin, no-permission user
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user

        async def _override_user() -> User:
            return User(id=123, username="nope", email=None, is_active=True, roles=[], permissions=[], is_admin=False)

        app.dependency_overrides[get_request_user] = _override_user

        async def _override_principal() -> AuthPrincipal:
            return AuthPrincipal(
                kind="user",
                user_id=123,
                subject="user:123",
                token_type="jwt",
                roles=[],
                permissions=[],
                is_admin=False,
                org_ids=[],
                team_ids=[],
            )

        app.dependency_overrides[auth_deps.get_auth_principal] = _override_principal

        try:
            with TestClient(app) as client:
                resp = client.post(
                    "/api/v1/tools/execute",
                    json={"tool_name": "any.tool", "arguments": {}, "dry_run": False},
                )
                assert resp.status_code == 403, resp.text
                body = resp.json()
                assert "Permission" in (body.get("detail") or "") or body.get("detail") == "Forbidden"
        finally:
            app.dependency_overrides.pop(get_request_user, None)
            app.dependency_overrides.pop(auth_deps.get_auth_principal, None)

    else:  # backend == "postgres"
        # Make sure the tools route is enabled before the fixture constructs the app
        os.environ["ROUTES_ENABLE"] = ",".join(filter(None, {os.getenv("ROUTES_ENABLE", ""), "tools"}))
        # Use the isolated Postgres environment fixture (per-test DB and client)
        client, _db_name = request.getfixturevalue("isolated_test_environment")  # type: ignore[assignment]

        # Override request user on the fixture-provided app
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user

        async def _override_user_pg() -> User:
            return User(id=456, username="nope", email=None, is_active=True, roles=[], permissions=[], is_admin=False)

        client.app.dependency_overrides[get_request_user] = _override_user_pg  # type: ignore[attr-defined]

        async def _override_principal_pg() -> AuthPrincipal:
            return AuthPrincipal(
                kind="user",
                user_id=456,
                subject="user:456",
                token_type="jwt",
                roles=[],
                permissions=[],
                is_admin=False,
                org_ids=[],
                team_ids=[],
            )

        client.app.dependency_overrides[auth_deps.get_auth_principal] = _override_principal_pg

        try:
            resp = client.post(
                "/api/v1/tools/execute",
                json={"tool_name": "any.tool", "arguments": {}, "dry_run": False},
            )
            assert resp.status_code == 403, resp.text
            body = resp.json()
            assert "Permission" in (body.get("detail") or "") or body.get("detail") == "Forbidden"
        finally:
            client.app.dependency_overrides.pop(get_request_user, None)  # type: ignore[attr-defined]
            client.app.dependency_overrides.pop(auth_deps.get_auth_principal, None)  # type: ignore[attr-defined]
