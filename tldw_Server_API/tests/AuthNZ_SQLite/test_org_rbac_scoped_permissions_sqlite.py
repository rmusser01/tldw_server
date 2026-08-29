from pathlib import Path

import pytest
from fastapi import Depends, FastAPI, Request
from httpx import ASGITransport, AsyncClient


async def _issue_access_token(
    user_row: dict,
    *,
    active_org_id: int | None = None,
    active_team_id: int | None = None,
) -> str:
    from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
    from tldw_Server_API.app.core.AuthNZ.orgs_teams import list_memberships_for_user

    user_id = int(user_row["id"])
    memberships = await list_memberships_for_user(user_id)
    team_ids = sorted({m.get("team_id") for m in memberships if m.get("team_id") is not None})
    org_ids = sorted({m.get("org_id") for m in memberships if m.get("org_id") is not None})

    claims = {"team_ids": team_ids, "org_ids": org_ids}
    if active_org_id is not None:
        claims["active_org_id"] = int(active_org_id)
    if active_team_id is not None:
        claims["active_team_id"] = int(active_team_id)

    jwt_service = get_jwt_service()
    return jwt_service.create_access_token(
        user_id=user_id,
        username=str(user_row.get("username") or user_id),
        role=str(user_row.get("role") or "user"),
        additional_claims=claims,
    )


@pytest.mark.asyncio
async def test_org_rbac_scoped_permissions_require_active_sqlite(tmp_path, monkeypatch):
    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("ORG_RBAC_PROPAGATION_ENABLED", "true")
    monkeypatch.setenv("ORG_RBAC_SCOPE_MODE", "require_active")

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.org_rbac import apply_scoped_permissions
    from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import (
        AuthnzOrgsTeamsRepo,
        DEFAULT_BASE_TEAM_NAME,
    )

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    async with pool.transaction() as conn:
        await conn.execute(
            """
            INSERT INTO users (username, email, password_hash, is_active)
            VALUES (?, ?, ?, 1)
            """,
            ("alice", "alice@example.com", "x"),
        )

    user_id = await pool.fetchval("SELECT id FROM users WHERE username = ?", ("alice",))

    repo = AuthnzOrgsTeamsRepo(pool)
    org = await repo.create_organization(name="Acme", owner_user_id=user_id)
    org_id = org["id"]
    await repo.add_org_member(org_id=org_id, user_id=user_id, role="owner")

    default_team_id = await pool.fetchval(
        "SELECT id FROM teams WHERE org_id = ? AND name = ?",
        (org_id, DEFAULT_BASE_TEAM_NAME),
    )

    async with pool.transaction() as conn:
        await conn.execute(
            """
            INSERT OR IGNORE INTO permissions (name, description, category)
            VALUES (?, ?, ?)
            """,
            ("tools.execute:demo", "Execute demo tool", "tools"),
        )
    perm_id = await pool.fetchval(
        "SELECT id FROM permissions WHERE name = ?", ("tools.execute:demo",)
    )
    async with pool.transaction() as conn:
        await conn.execute(
            """
            INSERT OR IGNORE INTO team_role_permissions (team_role, permission_id)
            VALUES (?, ?)
            """,
            ("member", perm_id),
        )

    result = await apply_scoped_permissions(
        user_id=user_id,
        base_permissions=["media.read"],
        org_ids=[org_id],
        team_ids=[default_team_id],
        active_org_id=None,
        active_team_id=None,
    )

    assert result.active_org_id == org_id
    assert "tools.execute:demo" in result.permissions
    assert "system.configure" not in result.permissions


@pytest.mark.asyncio
async def test_org_rbac_scoped_permissions_endpoint_allows_media_read(tmp_path, monkeypatch):
    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("ORG_RBAC_PROPAGATION_ENABLED", "true")
    monkeypatch.setenv("ORG_RBAC_SCOPE_MODE", "require_active")
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    from tldw_Server_API.app.core.AuthNZ.permissions import MEDIA_READ
    from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
    from tldw_Server_API.app.api.v1.API_Deps import auth_deps

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    async with pool.transaction() as conn:
        await conn.execute(
            """
            INSERT INTO users (username, email, password_hash, is_active, role)
            VALUES (?, ?, ?, 1, ?)
            """,
            ("scoped-user", "scoped@example.com", "x", "guest"),
        )

    user_id = await pool.fetchval(
        "SELECT id FROM users WHERE username = ?", ("scoped-user",)
    )

    repo = AuthnzOrgsTeamsRepo(pool)
    org = await repo.create_organization(name="Scoped Org", owner_user_id=user_id)
    await repo.add_org_member(org_id=org["id"], user_id=user_id, role="member")

    api_key_mgr = APIKeyManager()
    await api_key_mgr.initialize()
    key_info = await api_key_mgr.create_api_key(user_id=user_id, name="scoped")

    app = FastAPI()

    @app.get("/scoped-media", dependencies=[Depends(auth_deps.require_permissions(MEDIA_READ))])
    async def scoped_media_endpoint():
        return {"ok": True}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/scoped-media",
            headers={"X-API-KEY": key_info["key"]},
        )

    assert response.status_code == 200
    assert response.json().get("ok") is True


@pytest.mark.asyncio
async def test_org_rbac_scoped_permissions_denylist_blocks_admin(tmp_path, monkeypatch):
    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("ORG_RBAC_PROPAGATION_ENABLED", "true")
    monkeypatch.setenv("ORG_RBAC_SCOPE_MODE", "require_active")
    monkeypatch.setenv("ORG_RBAC_SCOPED_PERMISSION_DENYLIST", "system.configure")
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_CONFIGURE
    from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
    from tldw_Server_API.app.api.v1.API_Deps import auth_deps

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    async with pool.transaction() as conn:
        await conn.execute(
            """
            INSERT INTO users (username, email, password_hash, is_active, role)
            VALUES (?, ?, ?, 1, ?)
            """,
            ("deny-user", "deny@example.com", "x", "guest"),
        )

    user_id = await pool.fetchval(
        "SELECT id FROM users WHERE username = ?", ("deny-user",)
    )

    repo = AuthnzOrgsTeamsRepo(pool)
    org = await repo.create_organization(name="Deny Org", owner_user_id=user_id)
    await repo.add_org_member(org_id=org["id"], user_id=user_id, role="member")

    async with pool.transaction() as conn:
        await conn.execute(
            "INSERT OR IGNORE INTO permissions (name, description, category) VALUES (?, ?, ?)",
            (SYSTEM_CONFIGURE, "Configure system", "system"),
        )
    perm_id = await pool.fetchval(
        "SELECT id FROM permissions WHERE name = ?", (SYSTEM_CONFIGURE,)
    )
    async with pool.transaction() as conn:
        await conn.execute(
            """
            INSERT OR IGNORE INTO org_role_permissions (org_role, permission_id)
            VALUES (?, ?)
            """,
            ("member", perm_id),
        )

    api_key_mgr = APIKeyManager()
    await api_key_mgr.initialize()
    key_info = await api_key_mgr.create_api_key(user_id=user_id, name="deny-key")

    app = FastAPI()

    @app.get("/admin-guarded", dependencies=[Depends(auth_deps.require_permissions(SYSTEM_CONFIGURE))])
    async def admin_guarded_endpoint():
        return {"ok": True}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/admin-guarded",
            headers={"X-API-KEY": key_info["key"]},
        )

    assert response.status_code == 403
    assert "Permission denied" in response.json().get("detail", "")


@pytest.mark.asyncio
async def test_org_rbac_scoped_permissions_team_denylist_blocks_admin(tmp_path, monkeypatch):
    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("ORG_RBAC_PROPAGATION_ENABLED", "true")
    monkeypatch.setenv("ORG_RBAC_SCOPE_MODE", "require_active")
    monkeypatch.setenv("ORG_RBAC_SCOPED_PERMISSION_DENYLIST", "system.configure")
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_CONFIGURE
    from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import (
        AuthnzOrgsTeamsRepo,
        DEFAULT_BASE_TEAM_NAME,
    )
    from tldw_Server_API.app.api.v1.API_Deps import auth_deps

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    async with pool.transaction() as conn:
        await conn.execute(
            """
            INSERT INTO users (username, email, password_hash, is_active, role)
            VALUES (?, ?, ?, 1, ?)
            """,
            ("deny-team-user", "deny-team@example.com", "x", "guest"),
        )

    user_id = await pool.fetchval(
        "SELECT id FROM users WHERE username = ?", ("deny-team-user",)
    )

    repo = AuthnzOrgsTeamsRepo(pool)
    org = await repo.create_organization(name="Deny Team Org", owner_user_id=user_id)
    org_id = org["id"]
    await repo.add_org_member(org_id=org_id, user_id=user_id, role="member")

    default_team_id = await pool.fetchval(
        "SELECT id FROM teams WHERE org_id = ? AND name = ?",
        (org_id, DEFAULT_BASE_TEAM_NAME),
    )

    async with pool.transaction() as conn:
        await conn.execute(
            "INSERT OR IGNORE INTO permissions (name, description, category) VALUES (?, ?, ?)",
            (SYSTEM_CONFIGURE, "Configure system", "system"),
        )
    perm_id = await pool.fetchval(
        "SELECT id FROM permissions WHERE name = ?", (SYSTEM_CONFIGURE,)
    )
    async with pool.transaction() as conn:
        await conn.execute(
            """
            INSERT OR IGNORE INTO team_role_permissions (team_role, permission_id)
            VALUES (?, ?)
            """,
            ("member", perm_id),
        )

    api_key_mgr = APIKeyManager()
    await api_key_mgr.initialize()
    key_info = await api_key_mgr.create_api_key(user_id=user_id, name="deny-team-key")

    app = FastAPI()

    @app.get("/admin-guarded-team", dependencies=[Depends(auth_deps.require_permissions(SYSTEM_CONFIGURE))])
    async def admin_guarded_team_endpoint():
        return {"ok": True}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/admin-guarded-team",
            headers={"X-API-KEY": key_info["key"]},
        )

    assert response.status_code == 403
    assert "Permission denied" in response.json().get("detail", "")


@pytest.mark.asyncio
async def test_org_rbac_scoped_permissions_allows_tools_execute(tmp_path, monkeypatch):
    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("ORG_RBAC_PROPAGATION_ENABLED", "true")
    monkeypatch.setenv("ORG_RBAC_SCOPE_MODE", "require_active")
    monkeypatch.setenv("TEST_MODE", "1")

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
    from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import (
        AuthnzOrgsTeamsRepo,
        DEFAULT_BASE_TEAM_NAME,
    )
    from tldw_Server_API.app.api.v1.API_Deps import auth_deps

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    async with pool.transaction() as conn:
        await conn.execute(
            """
            INSERT INTO users (username, email, password_hash, is_active, role)
            VALUES (?, ?, ?, 1, ?)
            """,
            ("tool-user", "tool-user@example.com", "x", "guest"),
        )

    user_id = await pool.fetchval(
        "SELECT id FROM users WHERE username = ?", ("tool-user",)
    )

    repo = AuthnzOrgsTeamsRepo(pool)
    org = await repo.create_organization(name="Tool Org", owner_user_id=user_id)
    org_id = org["id"]
    await repo.add_org_member(org_id=org_id, user_id=user_id, role="member")

    default_team_id = await pool.fetchval(
        "SELECT id FROM teams WHERE org_id = ? AND name = ?",
        (org_id, DEFAULT_BASE_TEAM_NAME),
    )

    async with pool.transaction() as conn:
        await conn.execute(
            """
            INSERT OR IGNORE INTO permissions (name, description, category)
            VALUES (?, ?, ?)
            """,
            ("tools.execute:*", "Execute any MCP tool", "tools"),
        )
    perm_id = await pool.fetchval(
        "SELECT id FROM permissions WHERE name = ?", ("tools.execute:*",)
    )
    async with pool.transaction() as conn:
        await conn.execute(
            """
            INSERT OR IGNORE INTO team_role_permissions (team_role, permission_id)
            VALUES (?, ?)
            """,
            ("member", perm_id),
        )

    api_key_mgr = APIKeyManager()
    await api_key_mgr.initialize()
    key_info = await api_key_mgr.create_api_key(user_id=user_id, name="tool-key")

    app = FastAPI()

    @app.get("/tool-exec", dependencies=[Depends(auth_deps.require_permissions("tools.execute:*"))])
    async def tool_exec_endpoint():
        return {"ok": True}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/tool-exec",
            headers={"X-API-KEY": key_info["key"]},
        )

    assert response.status_code == 200
    assert response.json().get("ok") is True


@pytest.mark.asyncio
async def test_org_rbac_scoped_permissions_jwt_active_org_claims(tmp_path, monkeypatch):
    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("ORG_RBAC_PROPAGATION_ENABLED", "true")
    monkeypatch.setenv("ORG_RBAC_SCOPE_MODE", "require_active")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-jwt-key-please-change-1234567890")

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.jwt_service import reset_jwt_service
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB, reset_users_db
    from tldw_Server_API.app.api.v1.API_Deps import auth_deps

    reset_settings()
    reset_jwt_service()
    await reset_db_pool()
    await reset_users_db()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    users_db = UsersDB(pool)
    await users_db.initialize()
    user = await users_db.create_user(
        username="jwt-user",
        email="jwt-user@example.com",
        password_hash="hashed",
        role="user",
        is_active=True,
        is_verified=True,
    )

    repo = AuthnzOrgsTeamsRepo(pool)
    org_alpha = await repo.create_organization(name="Org Alpha", owner_user_id=int(user["id"]))
    org_beta = await repo.create_organization(name="Org Beta", owner_user_id=int(user["id"]))
    await repo.add_org_member(org_id=org_alpha["id"], user_id=int(user["id"]), role="member")
    await repo.add_org_member(org_id=org_beta["id"], user_id=int(user["id"]), role="lead")

    async with pool.transaction() as conn:
        await conn.execute(
            "INSERT OR IGNORE INTO permissions (name, description, category) VALUES (?, ?, ?)",
            ("custom.alpha", "Alpha scoped permission", "custom"),
        )
        await conn.execute(
            "INSERT OR IGNORE INTO permissions (name, description, category) VALUES (?, ?, ?)",
            ("custom.beta", "Beta scoped permission", "custom"),
        )
    alpha_id = await pool.fetchval("SELECT id FROM permissions WHERE name = ?", ("custom.alpha",))
    beta_id = await pool.fetchval("SELECT id FROM permissions WHERE name = ?", ("custom.beta",))
    async with pool.transaction() as conn:
        await conn.execute(
            "INSERT OR IGNORE INTO org_role_permissions (org_role, permission_id) VALUES (?, ?)",
            ("member", alpha_id),
        )
        await conn.execute(
            "INSERT OR IGNORE INTO org_role_permissions (org_role, permission_id) VALUES (?, ?)",
            ("lead", beta_id),
        )

    token_alpha = await _issue_access_token(user, active_org_id=int(org_alpha["id"]))
    token_beta = await _issue_access_token(user, active_org_id=int(org_beta["id"]))

    app = FastAPI()

    @app.get("/custom-alpha", dependencies=[Depends(auth_deps.require_permissions("custom.alpha"))])
    async def custom_alpha_endpoint():
        return {"ok": True}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp_alpha = await client.get(
            "/custom-alpha",
            headers={"Authorization": f"Bearer {token_alpha}"},
        )
        resp_beta = await client.get(
            "/custom-alpha",
            headers={"Authorization": f"Bearer {token_beta}"},
        )

    assert resp_alpha.status_code == 200
    assert resp_beta.status_code == 403


@pytest.mark.asyncio
async def test_org_rbac_scoped_permissions_active_org_includes_team_perms(tmp_path, monkeypatch):
    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("ORG_RBAC_PROPAGATION_ENABLED", "true")
    monkeypatch.setenv("ORG_RBAC_SCOPE_MODE", "require_active")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-jwt-key-please-change-1234567890")

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.jwt_service import reset_jwt_service
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import (
        AuthnzOrgsTeamsRepo,
        DEFAULT_BASE_TEAM_NAME,
    )
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB, reset_users_db
    from tldw_Server_API.app.api.v1.API_Deps import auth_deps

    reset_settings()
    reset_jwt_service()
    await reset_db_pool()
    await reset_users_db()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    users_db = UsersDB(pool)
    await users_db.initialize()
    user = await users_db.create_user(
        username="active-org-user",
        email="active-org-user@example.com",
        password_hash="hashed",
        role="user",
        is_active=True,
        is_verified=True,
    )

    repo = AuthnzOrgsTeamsRepo(pool)
    org_primary = await repo.create_organization(
        name="Org Primary", owner_user_id=int(user["id"])
    )
    org_secondary = await repo.create_organization(
        name="Org Secondary", owner_user_id=int(user["id"])
    )
    await repo.add_org_member(org_id=org_primary["id"], user_id=int(user["id"]), role="member")
    await repo.add_org_member(org_id=org_secondary["id"], user_id=int(user["id"]), role="member")

    primary_team_id = await pool.fetchval(
        "SELECT id FROM teams WHERE org_id = ? AND name = ?",
        (org_primary["id"], DEFAULT_BASE_TEAM_NAME),
    )
    secondary_team_id = await pool.fetchval(
        "SELECT id FROM teams WHERE org_id = ? AND name = ?",
        (org_secondary["id"], DEFAULT_BASE_TEAM_NAME),
    )

    await repo.update_team_member_role(
        team_id=int(secondary_team_id),
        user_id=int(user["id"]),
        role="lead",
    )

    async with pool.transaction() as conn:
        await conn.execute(
            "INSERT OR IGNORE INTO permissions (name, description, category) VALUES (?, ?, ?)",
            ("custom.team", "Team scoped permission", "custom"),
        )
    perm_id = await pool.fetchval(
        "SELECT id FROM permissions WHERE name = ?", ("custom.team",)
    )
    async with pool.transaction() as conn:
        await conn.execute(
            "INSERT OR IGNORE INTO team_role_permissions (team_role, permission_id) VALUES (?, ?)",
            ("member", perm_id),
        )

    token_primary = await _issue_access_token(user, active_org_id=int(org_primary["id"]))
    token_secondary = await _issue_access_token(user, active_org_id=int(org_secondary["id"]))

    app = FastAPI()

    @app.get("/custom-team", dependencies=[Depends(auth_deps.require_permissions("custom.team"))])
    async def custom_team_endpoint():
        return {"ok": True}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp_primary = await client.get(
            "/custom-team",
            headers={"Authorization": f"Bearer {token_primary}"},
        )
        resp_secondary = await client.get(
            "/custom-team",
            headers={"Authorization": f"Bearer {token_secondary}"},
        )

    assert resp_primary.status_code == 200
    assert resp_secondary.status_code == 403


@pytest.mark.asyncio
async def test_org_rbac_scoped_permissions_jwt_active_team_derives_org(tmp_path, monkeypatch):
    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("ORG_RBAC_PROPAGATION_ENABLED", "true")
    monkeypatch.setenv("ORG_RBAC_SCOPE_MODE", "require_active")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-jwt-key-please-change-1234567890")

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.jwt_service import reset_jwt_service
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB, reset_users_db
    from tldw_Server_API.app.api.v1.API_Deps import auth_deps

    reset_settings()
    reset_jwt_service()
    await reset_db_pool()
    await reset_users_db()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    users_db = UsersDB(pool)
    await users_db.initialize()
    user = await users_db.create_user(
        username="active-team-user",
        email="active-team-user@example.com",
        password_hash="hashed",
        role="user",
        is_active=True,
        is_verified=True,
    )

    repo = AuthnzOrgsTeamsRepo(pool)
    org = await repo.create_organization(
        name="Team Derive Org", owner_user_id=int(user["id"])
    )
    team = await repo.create_team(org_id=int(org["id"]), name="Team A")
    await repo.add_org_member(org_id=int(org["id"]), user_id=int(user["id"]), role="member")
    await repo.add_team_member(team_id=int(team["id"]), user_id=int(user["id"]), role="member")

    async with pool.transaction() as conn:
        await conn.execute(
            "INSERT OR IGNORE INTO permissions (name, description, category) VALUES (?, ?, ?)",
            ("custom.teamonly", "Team-only permission", "custom"),
        )
    perm_id = await pool.fetchval(
        "SELECT id FROM permissions WHERE name = ?", ("custom.teamonly",)
    )
    async with pool.transaction() as conn:
        await conn.execute(
            "INSERT OR IGNORE INTO team_role_permissions (team_role, permission_id) VALUES (?, ?)",
            ("member", perm_id),
        )

    token = await _issue_access_token(user, active_team_id=int(team["id"]))

    app = FastAPI()

    @app.get(
        "/team-derive",
        dependencies=[Depends(auth_deps.require_permissions("custom.teamonly"))],
    )
    async def team_derive_endpoint(request: Request):
        return {"active_org_id": getattr(request.state, "active_org_id", None)}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/team-derive",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200
    assert response.json().get("active_org_id") == int(org["id"])


@pytest.mark.asyncio
async def test_org_rbac_scoped_permissions_jwt_require_active_fallback(tmp_path, monkeypatch):
    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("ORG_RBAC_PROPAGATION_ENABLED", "true")
    monkeypatch.setenv("ORG_RBAC_SCOPE_MODE", "require_active")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-jwt-key-please-change-1234567890")

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.jwt_service import reset_jwt_service
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB, reset_users_db
    from tldw_Server_API.app.api.v1.API_Deps import auth_deps

    reset_settings()
    reset_jwt_service()
    await reset_db_pool()
    await reset_users_db()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    users_db = UsersDB(pool)
    await users_db.initialize()
    user = await users_db.create_user(
        username="fallback-user",
        email="fallback-user@example.com",
        password_hash="hashed",
        role="user",
        is_active=True,
        is_verified=True,
    )

    repo = AuthnzOrgsTeamsRepo(pool)
    org = await repo.create_organization(
        name="Fallback Org", owner_user_id=int(user["id"])
    )
    await repo.add_org_member(org_id=int(org["id"]), user_id=int(user["id"]), role="member")

    async with pool.transaction() as conn:
        await conn.execute(
            "INSERT OR IGNORE INTO permissions (name, description, category) VALUES (?, ?, ?)",
            ("custom.fallback", "Fallback permission", "custom"),
        )
    perm_id = await pool.fetchval(
        "SELECT id FROM permissions WHERE name = ?", ("custom.fallback",)
    )
    async with pool.transaction() as conn:
        await conn.execute(
            "INSERT OR IGNORE INTO org_role_permissions (org_role, permission_id) VALUES (?, ?)",
            ("member", perm_id),
        )

    token = await _issue_access_token(user)

    app = FastAPI()

    @app.get(
        "/fallback",
        dependencies=[Depends(auth_deps.require_permissions("custom.fallback"))],
    )
    async def fallback_endpoint(request: Request):
        return {"active_org_id": getattr(request.state, "active_org_id", None)}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/fallback",
            headers={"Authorization": f"Bearer {token}"},
        )

    assert response.status_code == 200
    assert response.json().get("active_org_id") == int(org["id"])
