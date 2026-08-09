from __future__ import annotations

import asyncio
import uuid

from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.initialize import ensure_single_user_rbac_seed_if_needed
from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    TrustedMembershipReason,
    TrustedMembershipWriteContext,
)
from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
    add_org_member,
    add_team_member,
    create_organization,
    create_team,
)
from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import (
    _execute_membership_scope_sql,
    _mint_profile_user_sql,
    _profile_user_connection_identity,
    _revoke_profile_user_sql,
)
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.UserProfiles.overrides_repo import (
    OrgProfileOverridesRepo,
    TeamProfileOverridesRepo,
)
from tldw_Server_API.app.core.UserProfiles.service import UserProfileService
from tldw_Server_API.app.core.UserProfiles.version_gateway import ProfileVersionGateway
from tldw_Server_API.app.main import app

_BOOTSTRAP_MEMBERSHIP_CONTEXT = TrustedMembershipWriteContext(
    trusted_reason=TrustedMembershipReason.BOOTSTRAP,
)


def _run_async(coro):
    return asyncio.run(coro)


def _get_user_id(client: TestClient, auth_headers) -> int:
    resp = client.get("/api/v1/users/me/profile", headers=auth_headers)
    assert resp.status_code == 200
    return int(resp.json()["user"]["id"])


def test_select_lowest_id_overrides() -> None:
    rows = [
        {"org_id": 3, "key": "preferences.ui.theme", "value": "org-three"},
        {"org_id": 1, "key": "preferences.ui.theme", "value": "org-one"},
        {"org_id": 2, "key": "preferences.ui.density", "value": "org-two"},
        {"org_id": 1, "key": "preferences.ui.density", "value": "org-one-density"},
        {"org_id": 4, "key": "preferences.ui.theme", "value": None},
    ]

    selected = UserProfileService._select_lowest_id_overrides(rows, id_field="org_id")
    assert selected["preferences.ui.theme"]["value"] == "org-one"
    assert selected["preferences.ui.theme"]["id"] == 1
    assert selected["preferences.ui.density"]["value"] == "org-one-density"
    assert selected["preferences.ui.density"]["id"] == 1


def test_membership_ids_match_version_active_status_predicate() -> None:
    class MembershipRepo:
        async def list_org_memberships_for_user(self, user_id: int):
            assert user_id == 9
            return [
                {"org_id": 1, "status": "active"},
                {"org_id": 2, "status": None},
                {"org_id": 3, "status": "inactive"},
                {"org_id": 4, "status": ""},
            ]

        async def list_active_team_memberships_for_user(self, user_id: int):
            assert user_id == 9
            return [{"team_id": 7}]

        async def list_memberships_for_user(self, _user_id: int):
            raise AssertionError("unfiltered team memberships must not be used")

    service = UserProfileService.__new__(UserProfileService)
    service._orgs_repo = MembershipRepo()

    assert _run_async(service._get_membership_ids(9)) == ([1, 2], [7])


def test_effective_config_layering(auth_headers, tmp_path, monkeypatch) -> None:
    # Isolate AuthNZ SQLite to avoid cross-test/process locks on Databases/users.db.
    db_path = tmp_path / f"users_effective_layers_{uuid.uuid4().hex[:8]}.db"
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    # Ensure cached settings/pools are rebuilt against the isolated DB.
    reset_settings()
    _run_async(reset_db_pool())
    _run_async(ensure_single_user_rbac_seed_if_needed())

    try:
        with TestClient(app) as client:
            user_id = _get_user_id(client, auth_headers)
            suffix = uuid.uuid4().hex[:8]

            async def _setup_overrides():
                org = await create_organization(name=f"Config Org {suffix}", owner_user_id=None)
                await add_org_member(org_id=int(org["id"]), user_id=user_id, role="member", context=_BOOTSTRAP_MEMBERSHIP_CONTEXT)
                team = await create_team(org_id=int(org["id"]), name=f"Config Team {suffix}")
                await add_team_member(team_id=int(team["id"]), user_id=user_id, role="member", context=_BOOTSTRAP_MEMBERSHIP_CONTEXT)

                pool = await get_db_pool()
                org_repo = OrgProfileOverridesRepo(pool)
                team_repo = TeamProfileOverridesRepo(pool)
                await org_repo.ensure_tables()
                await team_repo.ensure_tables()

                await org_repo.upsert_override(
                    org_id=int(org["id"]),
                    key="preferences.ui.theme",
                    value="org-theme",
                    updated_by=user_id,
                )
                await team_repo.upsert_override(
                    team_id=int(team["id"]),
                    key="preferences.ui.theme",
                    value="team-theme",
                    updated_by=user_id,
                )
                return int(org["id"]), int(team["id"])

            org_id, team_id = _run_async(_setup_overrides())

            resp = client.patch(
                "/api/v1/users/me/profile",
                headers=auth_headers,
                json={
                    "updates": [{"key": "preferences.ui.theme", "value": "user-theme"}],
                },
            )
            assert resp.status_code == 200

            resp = client.get(
                "/api/v1/users/me/profile",
                headers=auth_headers,
                params={"sections": "effective_config", "include_sources": True},
            )
            assert resp.status_code == 200
            effective = resp.json().get("effective_config", {})
            assert effective["preferences.ui.theme"]["value"] == "user-theme"
            assert effective["preferences.ui.theme"]["source"] == "user"

            resp = client.patch(
                "/api/v1/users/me/profile",
                headers=auth_headers,
                json={
                    "updates": [{"key": "preferences.ui.theme", "value": None}],
                },
            )
            assert resp.status_code == 200

            resp = client.get(
                "/api/v1/users/me/profile",
                headers=auth_headers,
                params={"sections": "effective_config", "include_sources": True},
            )
            assert resp.status_code == 200
            effective = resp.json().get("effective_config", {})
            assert effective["preferences.ui.theme"]["value"] == "team-theme"
            assert effective["preferences.ui.theme"]["source"] == "team"

            async def _remove_team_override():
                pool = await get_db_pool()
                team_repo = TeamProfileOverridesRepo(pool)
                await team_repo.ensure_tables()
                await team_repo.delete_override(team_id=team_id, key="preferences.ui.theme")

            _run_async(_remove_team_override())

            resp = client.get(
                "/api/v1/users/me/profile",
                headers=auth_headers,
                params={"sections": "effective_config", "include_sources": True},
            )
            assert resp.status_code == 200
            effective = resp.json().get("effective_config", {})
            assert effective["preferences.ui.theme"]["value"] == "org-theme"
            assert effective["preferences.ui.theme"]["source"] == "org"

            async def _add_second_org_override():
                org = await create_organization(name=f"Config Org 2 {suffix}", owner_user_id=None)
                await add_org_member(org_id=int(org["id"]), user_id=user_id, role="member", context=_BOOTSTRAP_MEMBERSHIP_CONTEXT)
                pool = await get_db_pool()
                org_repo = OrgProfileOverridesRepo(pool)
                await org_repo.ensure_tables()
                await org_repo.upsert_override(
                    org_id=int(org["id"]),
                    key="preferences.ui.theme",
                    value="org-theme-2",
                    updated_by=user_id,
                )

            _run_async(_add_second_org_override())

            resp = client.get(
                "/api/v1/users/me/profile",
                headers=auth_headers,
                params={"sections": "effective_config", "include_sources": True},
            )
            assert resp.status_code == 200
            effective = resp.json().get("effective_config", {})
            assert effective["preferences.ui.theme"]["value"] == "org-theme"
            assert effective["preferences.ui.theme"]["source"] == "org"
    finally:
        # Best-effort cleanup so subsequent tests don't inherit this pool.
        _run_async(reset_db_pool())
        reset_settings()


def test_inactive_memberships_do_not_contribute_inherited_overrides(
    auth_headers,
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / f"users_active_layers_{uuid.uuid4().hex[:8]}.db"
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    reset_settings()
    _run_async(reset_db_pool())
    _run_async(ensure_single_user_rbac_seed_if_needed())

    try:
        with TestClient(app) as client:
            user_id = _get_user_id(client, auth_headers)
            suffix = uuid.uuid4().hex[:8]

            async def _setup_and_read():
                active_org = await create_organization(
                    name=f"Active Config Org {suffix}",
                    owner_user_id=None,
                )
                inactive_org = await create_organization(
                    name=f"Inactive Config Org {suffix}",
                    owner_user_id=None,
                )
                active_org_id = int(active_org["id"])
                inactive_org_id = int(inactive_org["id"])
                await add_org_member(
                    org_id=active_org_id,
                    user_id=user_id,
                    role="member",
                    context=_BOOTSTRAP_MEMBERSHIP_CONTEXT,
                )
                await add_org_member(
                    org_id=inactive_org_id,
                    user_id=user_id,
                    role="member",
                    context=_BOOTSTRAP_MEMBERSHIP_CONTEXT,
                )

                active_team = await create_team(
                    org_id=active_org_id,
                    name=f"Active Config Team {suffix}",
                )
                inactive_team = await create_team(
                    org_id=inactive_org_id,
                    name=f"Inactive Config Team {suffix}",
                )
                active_team_id = int(active_team["id"])
                inactive_team_id = int(inactive_team["id"])
                await add_team_member(
                    team_id=active_team_id,
                    user_id=user_id,
                    role="member",
                    context=_BOOTSTRAP_MEMBERSHIP_CONTEXT,
                )
                await add_team_member(
                    team_id=inactive_team_id,
                    user_id=user_id,
                    role="member",
                    context=_BOOTSTRAP_MEMBERSHIP_CONTEXT,
                )

                pool = await get_db_pool()
                org_repo = OrgProfileOverridesRepo(pool)
                team_repo = TeamProfileOverridesRepo(pool)
                await org_repo.ensure_tables()
                await team_repo.ensure_tables()
                await org_repo.upsert_override(
                    org_id=active_org_id,
                    key="preferences.ui.theme",
                    value="active-org",
                    updated_by=user_id,
                )
                await org_repo.upsert_override(
                    org_id=inactive_org_id,
                    key="preferences.ui.theme",
                    value="inactive-org",
                    updated_by=user_id,
                )
                await team_repo.upsert_override(
                    team_id=active_team_id,
                    key="preferences.ui.density",
                    value="active-team",
                    updated_by=user_id,
                )
                await team_repo.upsert_override(
                    team_id=inactive_team_id,
                    key="preferences.ui.density",
                    value="inactive-team",
                    updated_by=user_id,
                )

                async with pool.transaction() as conn:
                    await _execute_membership_scope_sql(
                        conn,
                        "UPDATE main.org_members SET status = 'inactive' "
                        "WHERE org_id = ? AND user_id = ?",
                        (inactive_org_id, user_id),
                        backend="sqlite",
                    )
                    await _execute_membership_scope_sql(
                        conn,
                        "UPDATE main.team_members SET status = 'inactive' "
                        "WHERE user_id = ? AND team_id IN "
                        "(SELECT id FROM main.teams WHERE org_id = ?)",
                        (user_id, inactive_org_id),
                        backend="sqlite",
                    )
                    profile_anchor_update = _mint_profile_user_sql(
                        "UPDATE main.users SET profile_version = ? WHERE id = ?",
                        backend="sqlite",
                        connection_identity=_profile_user_connection_identity(conn),
                        operation="update",
                        columns=("profile_version",),
                    )
                    try:
                        await conn.execute(
                            profile_anchor_update,
                            ("2026-01-01T00:00:00.000000Z", user_id),
                        )
                    finally:
                        _revoke_profile_user_sql(profile_anchor_update)
                    await conn.execute(
                        "UPDATE org_config_overrides SET updated_at = ? "
                        "WHERE org_id = ?",
                        ("2026-01-02T00:00:00.000000Z", active_org_id),
                    )
                    await conn.execute(
                        "UPDATE team_config_overrides SET updated_at = ? "
                        "WHERE team_id = ?",
                        ("2026-01-03T00:00:00.000000Z", active_team_id),
                    )
                    await conn.execute(
                        "UPDATE org_config_overrides SET updated_at = ? "
                        "WHERE org_id = ?",
                        ("2099-01-01T00:00:00.000000Z", inactive_org_id),
                    )
                    await conn.execute(
                        "UPDATE team_config_overrides SET updated_at = ? "
                        "WHERE team_id = ?",
                        ("2099-01-02T00:00:00.000000Z", inactive_team_id),
                    )

                service = UserProfileService(pool)
                org_ids, team_ids = await service._get_membership_ids(user_id)
                raw = await service._build_raw_overrides(
                    user_id,
                    mask_secrets=False,
                )
                version = await ProfileVersionGateway(pool).read(user_id)
                return {
                    "active_org_id": active_org_id,
                    "inactive_org_id": inactive_org_id,
                    "active_team_id": active_team_id,
                    "inactive_team_id": inactive_team_id,
                    "org_ids": org_ids,
                    "team_ids": team_ids,
                    "raw_org_ids": {entry["org_id"] for entry in raw["orgs"]},
                    "raw_team_ids": {entry["team_id"] for entry in raw["teams"]},
                    "version": version,
                }

            result = _run_async(_setup_and_read())

            assert result["active_org_id"] in result["org_ids"]
            assert result["inactive_org_id"] not in result["org_ids"]
            assert result["active_team_id"] in result["team_ids"]
            assert result["inactive_team_id"] not in result["team_ids"]
            assert result["raw_org_ids"] == {result["active_org_id"]}
            assert result["raw_team_ids"] == {result["active_team_id"]}
            assert result["version"].isoformat() == "2026-01-03T00:00:00+00:00"
    finally:
        _run_async(reset_db_pool())
        reset_settings()
