from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

import tldw_Server_API.app.core.PrivilegeMaps.service as service_module
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
from tldw_Server_API.app.core.AuthNZ.privilege_catalog import PrivilegeCatalog
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.PrivilegeMaps.introspection import RouteMetadata
from tldw_Server_API.app.core.PrivilegeMaps.service import PrivilegeMapService


async def _fetch_id(pool, query: str, value: str) -> int:
    result = await pool.fetchval(query, (value,))
    assert result is not None, f"Expected ID for query {query} with value {value}"
    return int(result)


def _test_catalog(scope_ids: list[str], *, version: str = "test-privileges") -> PrivilegeCatalog:
    return PrivilegeCatalog.model_validate(
        {
            "version": version,
            "updated_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "rate_limit_classes": [
                {
                    "id": "standard",
                    "requests_per_min": 60,
                    "burst": 10,
                    "notes": None,
                }
            ],
            "feature_flags": [],
            "ownership_predicates": [],
            "scopes": [
                {
                    "id": scope_id,
                    "description": f"{scope_id} test scope",
                    "resource_tags": ["test"],
                    "sensitivity_tier": "low",
                    "rate_limit_class": "standard",
                    "default_roles": [],
                    "feature_flag_id": None,
                    "ownership_predicates": [],
                    "doc_url": None,
                }
                for scope_id in scope_ids
            ],
        }
    )


@pytest.mark.asyncio
async def test_privilege_service_honors_authnz_role_mappings(tmp_path, monkeypatch):
    db_path = tmp_path / "authnz.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-for-priv-service-123456")
    monkeypatch.setenv("TEST_MODE", "true")

    # Reset global singletons so we pick up the test configuration.
    reset_settings()
    await reset_db_pool()

    # Ensure migrations run so privilege_snapshots and RBAC tables exist ahead of service usage.
    ensure_authnz_tables(Path(db_path))

    pool = await get_db_pool()

    # Seed roles, permissions, users, and memberships.
    async with pool.transaction() as conn:
        for role_name, is_system in [
            ("admin", 1),
            ("media_manager", 0),
            ("analyst", 0),
            ("viewer", 0),
            ("researcher", 0),
        ]:
            await conn.execute(
                "INSERT OR IGNORE INTO roles (name, description, is_system) VALUES (?, ?, ?)",
                (role_name, f"{role_name} role", is_system),
            )

        for perm_name in [
            "rag.search",
            "media.catalog.view",
            "feature_flag:media_ingest_beta",
        ]:
            await conn.execute(
                "INSERT OR IGNORE INTO permissions (name, description, category) VALUES (?, ?, ?)",
                (perm_name, f"{perm_name} permission", "test"),
            )

        for username, email, primary_role in [
            ("admin-user", "admin@example.com", "admin"),
            ("media-manager", "media@example.com", "media_manager"),
            ("analyst-user", "analyst@example.com", "analyst"),
            ("researcher-user", "researcher@example.com", "researcher"),
        ]:
            await conn.execute(
                """
                INSERT INTO users (username, email, password_hash, is_active, role)
                VALUES (?, ?, ?, ?, ?)
                """,
                (username, email, "hashed", 1, primary_role),
            )

    admin_id = await _fetch_id(pool, "SELECT id FROM users WHERE username = ?", "admin-user")
    media_manager_id = await _fetch_id(pool, "SELECT id FROM users WHERE username = ?", "media-manager")
    analyst_id = await _fetch_id(pool, "SELECT id FROM users WHERE username = ?", "analyst-user")
    researcher_id = await _fetch_id(pool, "SELECT id FROM users WHERE username = ?", "researcher-user")

    role_ids = {}
    for role_name in ["admin", "media_manager", "analyst", "viewer", "researcher"]:
        role_ids[role_name] = await _fetch_id(pool, "SELECT id FROM roles WHERE name = ?", role_name)

    permission_ids = {}
    for perm_name in ["rag.search", "media.catalog.view", "feature_flag:media_ingest_beta"]:
        permission_ids[perm_name] = await _fetch_id(pool, "SELECT id FROM permissions WHERE name = ?", perm_name)

    async with pool.transaction() as conn:
        # Assign primary roles explicitly via mapping table.
        await conn.execute(
            "INSERT OR IGNORE INTO user_roles (user_id, role_id) VALUES (?, ?)",
            (admin_id, role_ids["admin"]),
        )
        await conn.execute(
            "INSERT OR IGNORE INTO user_roles (user_id, role_id) VALUES (?, ?)",
            (media_manager_id, role_ids["media_manager"]),
        )
        await conn.execute(
            "INSERT OR IGNORE INTO user_roles (user_id, role_id) VALUES (?, ?)",
            (analyst_id, role_ids["analyst"]),
        )
        await conn.execute(
            "INSERT OR IGNORE INTO user_roles (user_id, role_id) VALUES (?, ?)",
            (researcher_id, role_ids["researcher"]),
        )

        # researcher gains rag.search through RBAC role permissions
        await conn.execute(
            "INSERT OR IGNORE INTO role_permissions (role_id, permission_id) VALUES (?, ?)",
            (role_ids["researcher"], permission_ids["rag.search"]),
        )

        # Direct user override: researcher gets media ingest beta flag despite role not being allowed.
        await conn.execute(
            """
            INSERT OR REPLACE INTO user_permissions (user_id, permission_id, granted)
            VALUES (?, ?, 1)
            """,
            (researcher_id, permission_ids["feature_flag:media_ingest_beta"]),
        )

    # Create basic organization/team structure for membership lookups.
    async with pool.transaction() as conn:
        await conn.execute(
            "INSERT INTO organizations (name, slug, owner_user_id) VALUES (?, ?, ?)",
            ("Acme Corp", "acme-corp", admin_id),
        )

    org_id = await _fetch_id(pool, "SELECT id FROM organizations WHERE slug = ?", "acme-corp")

    async with pool.transaction() as conn:
        for user_id in [admin_id, media_manager_id, analyst_id, researcher_id]:
            await conn.execute(
                """
                INSERT OR IGNORE INTO org_members (org_id, user_id, role, status)
                VALUES (?, ?, ?, ?)
                """,
                (org_id, user_id, "member", "active"),
            )

        await conn.execute(
            """
            INSERT INTO teams (org_id, name, slug, is_active)
            VALUES (?, ?, ?, 1)
            """,
            (org_id, "Ingest Ops", "ingest-ops",),
        )

    team_id = await _fetch_id(pool, "SELECT id FROM teams WHERE slug = ?", "ingest-ops")

    async with pool.transaction() as conn:
        for user_id in [media_manager_id, researcher_id]:
            await conn.execute(
                """
                INSERT OR REPLACE INTO team_members (team_id, user_id, role, status)
                VALUES (?, ?, ?, ?)
                """,
                (team_id, user_id, "member", "active"),
            )

    service = PrivilegeMapService()

    users = await service._fetch_users()
    user_map = {user["username"]: user for user in users}

    all_scopes = {scope.id for scope in service.catalog.scopes}
    assert set(user_map["admin-user"]["allowed_scopes"]) == all_scopes
    assert "media_ingest_beta" in user_map["admin-user"]["feature_flags"]

    assert "media.ingest" in user_map["media-manager"]["allowed_scopes"]
    # media_manager gains feature flag through catalog allowed_roles
    assert "media_ingest_beta" in user_map["media-manager"]["feature_flags"]

    assert "rag.search" in user_map["researcher-user"]["allowed_scopes"]
    # Direct permission enables feature flag even though role is not whitelisted
    assert "media_ingest_beta" in user_map["researcher-user"]["feature_flags"]

    summary = await service.get_org_summary(group_by="role", include_trends=False, since=None)
    bucket_map = {bucket["key"]: bucket for bucket in summary["buckets"]}
    assert bucket_map["researcher"]["scopes"] >= 1
    assert bucket_map["media_manager"]["scopes"] >= 1

    team_detail = await service.get_team_detail(
        team_id=str(team_id),
        page=1,
        page_size=50,
        resource=None,
        dependency=None,
        role_filter=None,
    )
    ingest_rows = [
        row
        for row in team_detail["items"]
        if row["user_name"] == "media-manager" and row["privilege_scope_id"] == "media.ingest"
    ]
    assert ingest_rows and ingest_rows[0]["status"] == "allowed"

    await reset_db_pool()
    reset_settings()


@pytest.mark.asyncio
async def test_privilege_service_honors_expiry_and_explicit_permission_denies(tmp_path, monkeypatch):
    db_path = tmp_path / "authnz-effective-permissions.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-effective-permissions-123456")
    monkeypatch.setenv("TEST_MODE", "true")

    reset_settings()
    await reset_db_pool()
    ensure_authnz_tables(Path(db_path))

    pool = await get_db_pool()
    async with pool.transaction() as conn:
        await conn.execute(
            """
            INSERT INTO users (username, email, password_hash, is_active, role)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("effective-user", "effective@example.com", "hashed", 1, "viewer"),
        )
        for role_name in ["active-role", "expired-role"]:
            await conn.execute(
                "INSERT OR IGNORE INTO roles (name, description, is_system) VALUES (?, ?, 0)",
                (role_name, f"{role_name} role"),
            )
        for permission_name in [
            "scope.active",
            "scope.denied",
            "scope.expired_role",
            "scope.expired_allow",
        ]:
            await conn.execute(
                "INSERT OR IGNORE INTO permissions (name, description, category) VALUES (?, ?, ?)",
                (permission_name, f"{permission_name} permission", "test"),
            )

    user_id = await _fetch_id(pool, "SELECT id FROM users WHERE username = ?", "effective-user")
    active_role_id = await _fetch_id(pool, "SELECT id FROM roles WHERE name = ?", "active-role")
    expired_role_id = await _fetch_id(pool, "SELECT id FROM roles WHERE name = ?", "expired-role")
    permission_ids = {
        name: await _fetch_id(pool, "SELECT id FROM permissions WHERE name = ?", name)
        for name in ["scope.active", "scope.denied", "scope.expired_role", "scope.expired_allow"]
    }

    async with pool.transaction() as conn:
        await conn.execute(
            "INSERT OR IGNORE INTO user_roles (user_id, role_id) VALUES (?, ?)",
            (user_id, active_role_id),
        )
        await conn.execute(
            "INSERT OR REPLACE INTO user_roles (user_id, role_id, expires_at) VALUES (?, ?, ?)",
            (user_id, expired_role_id, "2000-01-01T00:00:00+00:00"),
        )
        await conn.execute(
            "INSERT OR IGNORE INTO role_permissions (role_id, permission_id) VALUES (?, ?)",
            (active_role_id, permission_ids["scope.denied"]),
        )
        await conn.execute(
            "INSERT OR IGNORE INTO role_permissions (role_id, permission_id) VALUES (?, ?)",
            (expired_role_id, permission_ids["scope.expired_role"]),
        )
        await conn.execute(
            "INSERT OR REPLACE INTO user_permissions (user_id, permission_id, granted) VALUES (?, ?, 1)",
            (user_id, permission_ids["scope.active"]),
        )
        await conn.execute(
            "INSERT OR REPLACE INTO user_permissions (user_id, permission_id, granted) VALUES (?, ?, 0)",
            (user_id, permission_ids["scope.denied"]),
        )
        await conn.execute(
            """
            INSERT OR REPLACE INTO user_permissions (user_id, permission_id, granted, expires_at)
            VALUES (?, ?, 1, ?)
            """,
            (user_id, permission_ids["scope.expired_allow"], "2000-01-01T00:00:00+00:00"),
        )

    service = PrivilegeMapService(
        route_registry={},
        catalog=_test_catalog(
            ["scope.active", "scope.denied", "scope.expired_role", "scope.expired_allow"],
            version="test-effective-permissions",
        ),
    )
    users = await service._fetch_users()
    effective_user = next(user for user in users if user["username"] == "effective-user")

    assert effective_user["roles"] == ["active-role"]
    assert set(effective_user["permissions"]) == {"scope.active"}
    assert effective_user["allowed_scopes"] == {"scope.active"}

    await reset_db_pool()
    reset_settings()


@pytest.mark.asyncio
async def test_privilege_service_multi_user_fetch_failure_fails_closed(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-fetch-failure-123456")
    reset_settings()

    async def broken_pool():
        raise RuntimeError("authnz database unavailable")

    monkeypatch.setattr(service_module, "get_db_pool", broken_pool)
    service = PrivilegeMapService(route_registry={}, catalog=_test_catalog(["scope.active"]))

    with pytest.raises(RuntimeError, match="authnz database unavailable"):
        await service._fetch_users()

    reset_settings()


@pytest.mark.asyncio
async def test_privilege_service_org_filter_uses_org_members(tmp_path, monkeypatch):
    db_path = tmp_path / "authnz-org.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-org-filter-123456")
    monkeypatch.setenv("TEST_MODE", "true")

    reset_settings()
    await reset_db_pool()
    ensure_authnz_tables(Path(db_path))

    pool = await get_db_pool()
    async with pool.transaction() as conn:
        for username, email, primary_role in [
            ("org-user-1", "org1@example.com", "viewer"),
            ("org-user-2", "org2@example.com", "viewer"),
            ("org-user-3", "org3@example.com", "viewer"),
        ]:
            await conn.execute(
                """
                INSERT INTO users (username, email, password_hash, is_active, role)
                VALUES (?, ?, ?, ?, ?)
                """,
                (username, email, "hashed", 1, primary_role),
            )
        await conn.execute(
            """
            INSERT INTO users (username, email, password_hash, is_active, role)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("org-user-4", "org4@example.com", "hashed", None, "viewer"),
        )

    user1_id = await _fetch_id(pool, "SELECT id FROM users WHERE username = ?", "org-user-1")
    user2_id = await _fetch_id(pool, "SELECT id FROM users WHERE username = ?", "org-user-2")
    user3_id = await _fetch_id(pool, "SELECT id FROM users WHERE username = ?", "org-user-3")
    user4_id = await _fetch_id(pool, "SELECT id FROM users WHERE username = ?", "org-user-4")

    async with pool.transaction() as conn:
        await conn.execute(
            "INSERT INTO organizations (name, slug, owner_user_id) VALUES (?, ?, ?)",
            ("Org One", "org-one", user1_id),
        )
        await conn.execute(
            "INSERT INTO organizations (name, slug, owner_user_id) VALUES (?, ?, ?)",
            ("Org Two", "org-two", user3_id),
        )

    org1_id = await _fetch_id(pool, "SELECT id FROM organizations WHERE slug = ?", "org-one")
    org2_id = await _fetch_id(pool, "SELECT id FROM organizations WHERE slug = ?", "org-two")

    async with pool.transaction() as conn:
        await conn.execute(
            """
            INSERT OR IGNORE INTO org_members (org_id, user_id, role, status)
            VALUES (?, ?, ?, ?)
            """,
            (org1_id, user1_id, "member", "active"),
        )
        await conn.execute(
            """
            INSERT OR IGNORE INTO org_members (org_id, user_id, role, status)
            VALUES (?, ?, ?, ?)
            """,
            (org1_id, user2_id, "member", "suspended"),
        )
        await conn.execute(
            """
            INSERT OR IGNORE INTO org_members (org_id, user_id, role, status)
            VALUES (?, ?, ?, ?)
            """,
            (org1_id, user4_id, "member", "active"),
        )
        await conn.execute(
            """
            INSERT OR IGNORE INTO org_members (org_id, user_id, role, status)
            VALUES (?, ?, ?, ?)
            """,
            (org2_id, user3_id, "member", "active"),
        )
        await conn.execute(
            "UPDATE organizations SET is_active = NULL WHERE id = ?",
            (org2_id,),
        )

    service = PrivilegeMapService()

    summary_org1, org1_users = await service.build_snapshot_summary(
        target_scope="org",
        org_id=str(org1_id),
        team_id=None,
        user_ids=None,
    )
    assert summary_org1["users"] == 1
    assert {user["username"] for user in org1_users} == {"org-user-1"}

    summary_org2, org2_users = await service.build_snapshot_summary(
        target_scope="org",
        org_id=str(org2_id),
        team_id=None,
        user_ids=None,
    )
    assert summary_org2["users"] == 0
    assert len(org2_users) == 0

    summary_org_none, org_none_users = await service.build_snapshot_summary(
        target_scope="org",
        org_id="9999",
        team_id=None,
        user_ids=None,
    )
    assert summary_org_none["users"] == 0
    assert len(org_none_users) == 0

    await reset_db_pool()
    reset_settings()


@pytest.mark.asyncio
async def test_privilege_service_team_filter_uses_active_memberships_and_teams(tmp_path, monkeypatch):
    db_path = tmp_path / "authnz-team-filter.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-key-team-filter-123456")
    monkeypatch.setenv("TEST_MODE", "true")

    reset_settings()
    await reset_db_pool()
    ensure_authnz_tables(Path(db_path))

    pool = await get_db_pool()
    async with pool.transaction() as conn:
        for username, email in [
            ("team-user-1", "team1@example.com"),
            ("team-user-2", "team2@example.com"),
        ]:
            await conn.execute(
                """
                INSERT INTO users (username, email, password_hash, is_active, role)
                VALUES (?, ?, ?, ?, ?)
                """,
                (username, email, "hashed", 1, "viewer"),
            )

    user1_id = await _fetch_id(pool, "SELECT id FROM users WHERE username = ?", "team-user-1")
    user2_id = await _fetch_id(pool, "SELECT id FROM users WHERE username = ?", "team-user-2")

    async with pool.transaction() as conn:
        await conn.execute(
            "INSERT INTO organizations (name, slug, owner_user_id) VALUES (?, ?, ?)",
            ("Team Org", "team-org", user1_id),
        )

    org_id = await _fetch_id(pool, "SELECT id FROM organizations WHERE slug = ?", "team-org")

    async with pool.transaction() as conn:
        await conn.execute(
            "INSERT INTO teams (org_id, name, slug, is_active) VALUES (?, ?, ?, 1)",
            (org_id, "Active Team", "active-team"),
        )
        await conn.execute(
            "INSERT INTO teams (org_id, name, slug, is_active) VALUES (?, ?, ?, ?)",
            (org_id, "Inactive Team", "inactive-team", None),
        )

    active_team_id = await _fetch_id(pool, "SELECT id FROM teams WHERE slug = ?", "active-team")
    inactive_team_id = await _fetch_id(pool, "SELECT id FROM teams WHERE slug = ?", "inactive-team")

    async with pool.transaction() as conn:
        await conn.execute(
            "INSERT OR IGNORE INTO team_members (team_id, user_id, role, status) VALUES (?, ?, ?, ?)",
            (active_team_id, user1_id, "member", "active"),
        )
        await conn.execute(
            "INSERT OR IGNORE INTO team_members (team_id, user_id, role, status) VALUES (?, ?, ?, ?)",
            (active_team_id, user2_id, "member", "suspended"),
        )
        await conn.execute(
            "INSERT OR IGNORE INTO team_members (team_id, user_id, role, status) VALUES (?, ?, ?, ?)",
            (inactive_team_id, user2_id, "member", "active"),
        )

    service = PrivilegeMapService()
    active_summary, active_users = await service.build_snapshot_summary(
        target_scope="team",
        org_id=None,
        team_id=str(active_team_id),
        user_ids=None,
    )
    assert active_summary["users"] == 1
    assert {user["username"] for user in active_users} == {"team-user-1"}

    inactive_summary, inactive_users = await service.build_snapshot_summary(
        target_scope="team",
        org_id=None,
        team_id=str(inactive_team_id),
        user_ids=None,
    )
    assert inactive_summary["users"] == 0
    assert inactive_users == []

    await reset_db_pool()
    reset_settings()


def test_privilege_detail_generation_stops_at_configured_cap(monkeypatch):
    monkeypatch.setattr(service_module, "MAX_DETAIL_ROWS", 2)
    catalog = _test_catalog(["scope.active"], version="test-detail-cap")
    service = PrivilegeMapService(
        catalog=catalog,
        route_registry={
            "scope.active": [
                RouteMetadata(
                    path="/unit/detail-cap",
                    methods=("GET", "POST", "DELETE"),
                    name="detail_cap",
                    tags=("test",),
                    endpoint="tests.detail_cap",
                )
            ]
        },
    )

    items = service.build_snapshot_detail(
        [
            {
                "id": "user-1",
                "username": "cap-user",
                "primary_role": "viewer",
                "roles": ["viewer"],
                "permissions": [],
                "feature_flags": set(),
                "allowed_scopes": {"scope.active"},
            }
        ]
    )

    assert len(items) == 2


@pytest.mark.asyncio
async def test_privilege_org_trends_are_scoped_to_org_id(monkeypatch):
    class RecordingTrendStore:
        def __init__(self) -> None:
            self.recorded_org_ids: list[str | None] = []
            self.computed_org_ids: list[str | None] = []

        async def record_snapshot(
            self,
            *,
            scope,
            group_by,
            catalog_version,
            generated_at,
            buckets,
            team_id=None,
            org_id=None,
        ):
            self.recorded_org_ids.append(org_id)

        async def compute_trends(
            self,
            *,
            scope,
            group_by,
            bucket_counts,
            window_start,
            window_end,
            team_id=None,
            org_id=None,
        ):
            self.computed_org_ids.append(org_id)
            return []

    trend_store = RecordingTrendStore()
    service = PrivilegeMapService(
        route_registry={},
        catalog=_test_catalog(["scope.active"], version="test-org-trends"),
        trend_store=trend_store,
    )

    async def fetch_users():
        return [
            {
                "id": "user-1",
                "username": "org-user",
                "primary_role": "viewer",
                "roles": ["viewer"],
                "permissions": [],
                "feature_flags": set(),
                "allowed_scopes": {"scope.active"},
            }
        ]

    async def fetch_org_memberships():
        return [{"org_id": "acme", "user_id": "user-1", "membership_role": "member"}]

    monkeypatch.setattr(service, "_fetch_users", fetch_users)
    monkeypatch.setattr(service, "_fetch_org_memberships", fetch_org_memberships)

    await service.get_org_summary(
        group_by="role",
        include_trends=True,
        since=None,
        org_id="acme",
    )

    assert trend_store.recorded_org_ids == ["acme"]
    assert trend_store.computed_org_ids == ["acme"]
