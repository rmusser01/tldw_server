import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.tests.helpers.audit_helpers import await_audit_action, flush_audit_events
from tldw_Server_API.tests.helpers.authnz_seed import ensure_test_user


@pytest.mark.real_audit
@pytest.mark.integration
@pytest.mark.asyncio
async def test_team_membership_audit_events_postgres(tmp_path, real_audit_service, test_db_pool, monkeypatch):
    # Use Postgres pool from fixture
    pool = test_db_pool

    # Disable CSRF for the TestClient
    from tldw_Server_API.app.core.config import settings as app_settings
    app_settings['CSRF_ENABLED'] = False

    # Ensure org/team/member tables exist (idempotent)
    await pool.execute(
        """
        CREATE TABLE IF NOT EXISTS organizations (
            id SERIAL PRIMARY KEY,
            uuid VARCHAR(64) UNIQUE,
            name VARCHAR(255) UNIQUE NOT NULL,
            slug VARCHAR(255) UNIQUE,
            owner_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
            is_active BOOLEAN DEFAULT TRUE,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    await pool.execute(
        """
        CREATE TABLE IF NOT EXISTS teams (
            id SERIAL PRIMARY KEY,
            org_id INTEGER NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
            name VARCHAR(255) NOT NULL,
            slug VARCHAR(255),
            description TEXT,
            is_active BOOLEAN DEFAULT TRUE,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (org_id, name)
        )
        """
    )
    await pool.execute(
        """
        CREATE TABLE IF NOT EXISTS team_members (
            team_id INTEGER NOT NULL REFERENCES teams(id) ON DELETE CASCADE,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            role VARCHAR(32) DEFAULT 'member',
            status VARCHAR(32) DEFAULT 'active',
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (team_id, user_id)
        )
        """
    )

    # Insert admin (role=admin) and target user
    admin_id = await ensure_test_user(
        pool, "pgadmin_audit", "pgadmin_audit@example.com", role="admin", is_verified=True
    )
    target_id = await ensure_test_user(
        pool, "pgvictim", "pgvictim@example.com"
    )

    # Override AuthPrincipal to treat this user as admin for claim-first gates
    from starlette.requests import Request

    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
    from tldw_Server_API.app.main import app

    async def _principal_override(request: Request) -> AuthPrincipal:  # type: ignore[override]
        principal = AuthPrincipal(
            kind="user",
            user_id=admin_id,
            api_key_id=None,
            subject="pgadmin_audit",
            token_type="access",
            jti=None,
            roles=["admin"],
            permissions=["system.configure"],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )
        if request is not None:
            try:
                request.state.auth = AuthContext(
                    principal=principal,
                    ip=None,
                    user_agent=None,
                    request_id=None,
                )
                request.state.user_id = admin_id
            except Exception:
                # Best-effort; do not fail tests if state attachment fails
                _ = None
        return principal

    app.dependency_overrides[get_auth_principal] = _principal_override

    with TestClient(app) as client:
        # Create org
        r = client.post("/api/v1/admin/orgs", json={"name": "Audit Org PG"})
        assert r.status_code == 200, r.text
        org = r.json()

        # Create team
        r = client.post(f"/api/v1/admin/orgs/{org['id']}/teams", json={"name": "QA-PG"})
        assert r.status_code == 200, r.text
        team = r.json()

        # Add team member -> should log audit event for actor (admin)
        r = client.post(
            f"/api/v1/admin/teams/{team['id']}/members",
            json={"user_id": int(target_id), "role": "member"},
        )
        assert r.status_code == 200, r.text

        flush_audit_events(client, int(admin_id))

    app.dependency_overrides.pop(get_auth_principal, None)

    # Ensure audit services flush events before inspection.
    from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import shutdown_all_audit_services
    await shutdown_all_audit_services()

    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    audit_db = DatabasePaths.get_audit_db_path(int(admin_id))
    assert audit_db.exists(), f"Audit DB not found: {audit_db}"

    cnt = await await_audit_action(audit_db, "team_member.add")
    assert cnt >= 1
