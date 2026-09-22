
import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.tests.helpers.authnz_seed import ensure_test_user


@pytest.mark.integration
@pytest.mark.asyncio
async def test_role_effective_permissions_postgres(test_db_pool):
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal

    # Disable CSRF for test client
    from tldw_Server_API.app.core.config import settings as app_settings
    from tldw_Server_API.app.main import app
    app_settings['CSRF_ENABLED'] = False

    pool = test_db_pool

    # Ensure RBAC tables exist
    await pool.execute(
        """
        CREATE TABLE IF NOT EXISTS roles (
            id SERIAL PRIMARY KEY,
            name VARCHAR(64) UNIQUE NOT NULL,
            description TEXT,
            is_system BOOLEAN DEFAULT FALSE
        )
        """
    )
    await pool.execute(
        """
        CREATE TABLE IF NOT EXISTS permissions (
            id SERIAL PRIMARY KEY,
            name VARCHAR(128) UNIQUE NOT NULL,
            description TEXT,
            category VARCHAR(64)
        )
        """
    )
    await pool.execute(
        """
        CREATE TABLE IF NOT EXISTS role_permissions (
            role_id INTEGER NOT NULL,
            permission_id INTEGER NOT NULL,
            PRIMARY KEY (role_id, permission_id)
        )
        """
    )

    # Insert admin user for override
    admin_user_id = await ensure_test_user(pool, "pgadmin2", "pgadmin2@example.com")

    # Create role and permissions
    role_name = "test_role_eff"
    await pool.execute("INSERT INTO roles (name, description, is_system) VALUES ($1, $2, FALSE) ON CONFLICT DO NOTHING", role_name, "test role")
    role_id = await pool.fetchval("SELECT id FROM roles WHERE name = $1", role_name)
    await pool.execute("INSERT INTO permissions (name, description, category) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING", "alpha.read", "Alpha Read", "alpha")
    await pool.execute("INSERT INTO permissions (name, description, category) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING", "tools.execute:foo", "Execute foo tool", "tools")
    pid_alpha = await pool.fetchval("SELECT id FROM permissions WHERE name = $1", "alpha.read")
    pid_tool = await pool.fetchval("SELECT id FROM permissions WHERE name = $1", "tools.execute:foo")
    await pool.execute("INSERT INTO role_permissions (role_id, permission_id) VALUES ($1, $2) ON CONFLICT DO NOTHING", role_id, pid_alpha)
    await pool.execute("INSERT INTO role_permissions (role_id, permission_id) VALUES ($1, $2) ON CONFLICT DO NOTHING", role_id, pid_tool)

    # Override auth principal with an admin user for claim-first RBAC
    async def _principal_override(request: Request) -> AuthPrincipal:  # type: ignore[override]
        principal = AuthPrincipal(
            kind="user",
            user_id=admin_user_id,
            api_key_id=None,
            subject="pgadmin2",
            token_type="access",
            jti=None,
            roles=["admin"],
            permissions=["system.configure"],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )
        try:
            request.state.auth = AuthContext(
                principal=principal,
                ip=None,
                user_agent=None,
                request_id=None,
            )
        except Exception:
            # Best-effort; not all code paths require request.state.auth
            _ = None
        return principal

    app.dependency_overrides[get_auth_principal] = _principal_override

    try:
        with TestClient(app) as client:
            r = client.get(f"/api/v1/admin/roles/{role_id}/permissions/effective")
            assert r.status_code == 200, r.text
            data = r.json()
            assert data["role_id"] == role_id
            assert data["role_name"] == role_name
            assert "alpha.read" in data["permissions"]
            assert "tools.execute:foo" in data["tool_permissions"]
            assert set(data["all_permissions"]) == {"alpha.read", "tools.execute:foo"}
    finally:
        app.dependency_overrides.pop(get_auth_principal, None)
