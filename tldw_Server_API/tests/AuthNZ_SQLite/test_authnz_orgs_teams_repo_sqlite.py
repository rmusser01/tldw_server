from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_authnz_orgs_teams_repo_membership_sqlite(tmp_path, monkeypatch):
    """AuthnzOrgsTeamsRepo membership helpers should work on SQLite."""
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.exceptions import DuplicateOrganizationError
    from tldw_Server_API.app.core.AuthNZ.membership_writer import (
        ActorMembershipWriteContext,
        MembershipAuthority,
        MembershipAuthorizationError,
        MembershipParentRequired,
        TrustedMembershipReason,
        TrustedMembershipWriteContext,
    )
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import (
        _execute_membership_scope_sql,
    )
    from tldw_Server_API.app.core.AuthNZ.profile_version import (
        VersionedUserWriteGateway,
    )
    from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import (
        DEFAULT_BASE_TEAM_NAME,
        AuthnzOrgsTeamsRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.UserProfiles.version_gateway import (
        ProfileVersionGateway,
    )

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))
    touch_calls: list[int] = []
    original_final_touch = VersionedUserWriteGateway.final_touch

    async def _record_final_touch(self, conn, *, user_id, version_floor):
        touch_calls.append(int(user_id))
        return await original_final_touch(
            self,
            conn,
            user_id=user_id,
            version_floor=version_floor,
        )

    monkeypatch.setattr(
        VersionedUserWriteGateway,
        "final_touch",
        _record_final_touch,
    )

    # Create two users for org/team membership
    async with pool.transaction() as conn:
        gateway = VersionedUserWriteGateway(
            "sqlite",
            clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        await gateway.insert_user(
            conn,
            values={
                "username": "owner",
                "email": "owner@example.com",
                "password_hash": "x",
                "is_active": True,
            },
        )
        await gateway.insert_user(
            conn,
            values={
                "username": "outsider",
                "email": "outsider@example.com",
                "password_hash": "x",
                "is_active": True,
            },
        )
        await gateway.insert_user(
            conn,
            values={
                "username": "member",
                "email": "member@example.com",
                "password_hash": "x",
                "is_active": True,
            },
        )

    owner_id = await pool.fetchval(
        "SELECT id FROM users WHERE username = ?", ("owner",)
    )
    member_id = await pool.fetchval(
        "SELECT id FROM users WHERE username = ?", ("member",)
    )
    outsider_id = await pool.fetchval(
        "SELECT id FROM users WHERE username = ?", ("outsider",)
    )

    repo = AuthnzOrgsTeamsRepo(pool)
    bootstrap_context = TrustedMembershipWriteContext(
        trusted_reason=TrustedMembershipReason.BOOTSTRAP,
    )
    owner_context = ActorMembershipWriteContext(
        actor_user_id=owner_id,
        required_authority=MembershipAuthority.SCOPED_MEMBERSHIP,
    )

    # Create organization and add members
    org = await repo.create_organization_with_owner_membership(
        name="Acme Corp",
        owner_user_id=owner_id,
        context=owner_context,
    )
    org_id = org["id"]

    updated_org = await repo.update_organization(
        org_id=org_id,
        name="Acme Corp Updated",
        slug="acme-corp-updated",
    )
    assert updated_org is not None
    assert updated_org["id"] == org_id
    assert updated_org["name"] == "Acme Corp Updated"
    assert updated_org["slug"] == "acme-corp-updated"
    assert updated_org.get("updated_at") is not None

    await repo.create_organization_with_owner_membership(
        name="Other Org",
        owner_user_id=owner_id,
        slug="other-org",
        context=bootstrap_context,
    )
    with pytest.raises(DuplicateOrganizationError):
        await repo.update_organization(org_id=org_id, slug="other-org")

    touch_calls.clear()
    owner_membership = await repo.add_org_member(
        org_id=org_id,
        user_id=owner_id,
        role="owner",
        context=owner_context,
    )
    assert touch_calls == []
    before_member_add = await ProfileVersionGateway(pool).read(member_id)
    touch_calls.clear()
    member_membership = await repo.add_org_member(
        org_id=org_id,
        user_id=member_id,
        role="member",
        context=owner_context,
    )
    after_member_add = await ProfileVersionGateway(pool).read(member_id)
    assert touch_calls == [member_id]

    assert set(owner_membership) == {"org_id", "user_id", "role"}
    assert owner_membership["org_id"] == org_id
    assert owner_membership["user_id"] == owner_id
    assert owner_membership["role"].lower() == "owner"

    assert set(member_membership) == {"org_id", "user_id", "role"}
    assert member_membership["org_id"] == org_id
    assert member_membership["user_id"] == member_id
    assert member_membership["role"].lower() == "member"
    assert after_member_add > before_member_add

    # List org members and memberships for user
    members = await repo.list_org_members(org_id=org_id)
    roles_by_user = {m["user_id"]: m["role"] for m in members}
    assert roles_by_user[owner_id].lower() == "owner"
    assert roles_by_user[member_id].lower() == "member"

    owner_memberships = await repo.list_org_memberships_for_user(owner_id)
    assert any(m["org_id"] == org_id and m["role"].lower() == "owner" for m in owner_memberships)

    # Default team should be created and both users enrolled
    default_team_id = await pool.fetchval(
        "SELECT id FROM teams WHERE org_id = ? AND name = ?",
        (org_id, DEFAULT_BASE_TEAM_NAME),
    )
    assert default_team_id is not None

    owner_team_count = await pool.fetchval(
        "SELECT COUNT(*) FROM team_members WHERE team_id = ? AND user_id = ?",
        (default_team_id, owner_id),
    )
    member_team_count = await pool.fetchval(
        "SELECT COUNT(*) FROM team_members WHERE team_id = ? AND user_id = ?",
        (default_team_id, member_id),
    )
    assert owner_team_count == 1
    assert member_team_count == 1

    touch_calls.clear()
    repeated_member_membership = await repo.add_org_member(
        org_id=org_id,
        user_id=member_id,
        role="owner",
        context=owner_context,
    )
    assert repeated_member_membership == {
        "org_id": org_id,
        "user_id": member_id,
        "role": "member",
    }
    assert touch_calls == []

    team = await repo.create_team(org_id=org_id, name="Review Team")
    team_id = int(team["id"])
    touch_calls.clear()
    with pytest.raises(MembershipParentRequired):
        await repo.add_team_member(
            team_id=team_id,
            user_id=outsider_id,
            role="member",
            context=owner_context,
        )
    assert await pool.fetchval(
        "SELECT COUNT(*) FROM team_members WHERE team_id = ? AND user_id = ?",
        (team_id, outsider_id),
    ) == 0
    assert touch_calls == []

    touch_calls.clear()
    parent_required = await repo.add_team_member(
        team_id=team_id,
        user_id=owner_id,
        role="member",
        context=owner_context,
    )
    assert parent_required["org_id"] == org_id
    assert touch_calls == [owner_id]
    await repo.remove_team_member(
        team_id=team_id,
        user_id=owner_id,
        context=owner_context,
    )
    added_team_member = await repo.add_team_member(
        team_id=team_id,
        user_id=member_id,
        role="member",
        context=owner_context,
    )
    assert set(added_team_member) == {"team_id", "user_id", "role", "org_id"}

    repeated_team_member = await repo.add_team_member(
        team_id=team_id,
        user_id=member_id,
        role="owner",
        context=owner_context,
    )
    assert repeated_team_member == {
        "team_id": team_id,
        "user_id": member_id,
        "role": "member",
        "org_id": org_id,
    }

    updated_team_member = await repo.update_team_member_role(
        team_id=team_id,
        user_id=member_id,
        role="admin",
        context=owner_context,
    )
    assert updated_team_member == {
        "team_id": team_id,
        "user_id": member_id,
        "role": "admin",
    }
    assert await repo.update_team_member_role(
        team_id=team_id,
        user_id=owner_id,
        role="admin",
        context=owner_context,
    ) is None

    removed_team_member = await repo.remove_team_member(
        team_id=team_id,
        user_id=member_id,
        context=owner_context,
    )
    assert removed_team_member == {
        "team_id": team_id,
        "user_id": member_id,
        "removed": True,
    }
    missing_team_member = await repo.remove_team_member(
        team_id=team_id,
        user_id=member_id,
        context=owner_context,
    )
    assert missing_team_member == {
        "team_id": team_id,
        "user_id": member_id,
        "removed": False,
    }
    await repo.add_team_member(
        team_id=team_id,
        user_id=member_id,
        role="member",
        context=owner_context,
    )

    # Update non-owner role
    updated_member = await repo.update_org_member_role(
        org_id=org_id,
        user_id=member_id,
        role="admin",
        context=owner_context,
    )
    assert updated_member is not None
    assert set(updated_member) == {"org_id", "user_id", "role"}
    assert updated_member["role"].lower() == "admin"

    # Cannot demote the last owner
    demote_owner = await repo.update_org_member_role(
        org_id=org_id,
        user_id=owner_id,
        role="member",
        context=owner_context,
    )
    assert demote_owner is not None
    assert set(demote_owner) == {"org_id", "user_id", "role", "error"}
    assert demote_owner["role"].lower() == "owner"
    assert demote_owner.get("error") == "owner_required"

    # Removing a non-owner should also remove them from the default team
    touch_calls.clear()
    remove_member = await repo.remove_org_member(
        org_id=org_id,
        user_id=member_id,
        context=owner_context,
    )
    assert set(remove_member) == {"org_id", "user_id", "removed"}
    assert remove_member["removed"] is True
    assert touch_calls == [member_id]

    assert await repo.update_org_member_role(
        org_id=org_id,
        user_id=member_id,
        role="member",
        context=owner_context,
    ) is None
    missing_org_member = await repo.remove_org_member(
        org_id=org_id,
        user_id=member_id,
        context=owner_context,
    )
    assert missing_org_member == {
        "org_id": org_id,
        "user_id": member_id,
        "removed": False,
    }

    remaining_members = await repo.list_org_members(org_id=org_id)
    assert all(m["user_id"] != member_id for m in remaining_members)

    member_team_count_after = await pool.fetchval(
        "SELECT COUNT(*) FROM team_members WHERE team_id = ? AND user_id = ?",
        (default_team_id, member_id),
    )
    owner_team_count_after = await pool.fetchval(
        "SELECT COUNT(*) FROM team_members WHERE team_id = ? AND user_id = ?",
        (default_team_id, owner_id),
    )
    assert member_team_count_after == 0
    assert owner_team_count_after == 1
    assert await pool.fetchval(
        "SELECT COUNT(*) FROM team_members WHERE team_id = ? AND user_id = ?",
        (team_id, member_id),
    ) == 0

    await repo.add_org_member(
        org_id=org_id,
        user_id=member_id,
        role="member",
        context=bootstrap_context,
    )
    await repo.add_team_member(
        team_id=team_id,
        user_id=member_id,
        role="member",
        context=bootstrap_context,
    )
    async with pool.transaction() as conn:
        await _execute_membership_scope_sql(
            conn,
            "DELETE FROM main.org_members WHERE org_id = ? AND user_id = ?",
            (org_id, member_id),
            backend="sqlite",
        )
    assert await repo.list_active_team_memberships_for_user(member_id) == []

    # Removing the last owner should be blocked with owner_required
    touch_calls.clear()
    remove_owner = await repo.remove_org_member(
        org_id=org_id,
        user_id=owner_id,
        context=owner_context,
    )
    assert set(remove_owner) == {"org_id", "user_id", "removed", "error"}
    assert remove_owner["removed"] is False
    assert remove_owner.get("error") == "owner_required"
    assert touch_calls == []

    for owner_status in ("inactive", None):
        status_label = owner_status or "null"
        inactive_owner_org = await repo.create_organization_with_owner_membership(
            name=f"Inactive Owner {status_label}",
            owner_user_id=outsider_id,
            context=bootstrap_context,
        )
        inactive_owner_org_id = int(inactive_owner_org["id"])
        await repo.add_org_member(
            org_id=inactive_owner_org_id,
            user_id=outsider_id,
            role="owner",
            context=bootstrap_context,
        )
        await repo.add_org_member(
            org_id=inactive_owner_org_id,
            user_id=owner_id,
            role="admin",
            context=bootstrap_context,
        )
        async with pool.transaction() as conn:
            await _execute_membership_scope_sql(
                conn,
                "UPDATE main.org_members SET status = ? "
                "WHERE org_id = ? AND user_id = ?",
                (owner_status, inactive_owner_org_id, outsider_id),
                backend="sqlite",
            )

        touch_calls.clear()
        removed_inactive_owner = await repo.remove_org_member(
            org_id=inactive_owner_org_id,
            user_id=outsider_id,
            context=bootstrap_context,
        )
        assert removed_inactive_owner == {
            "org_id": inactive_owner_org_id,
            "user_id": outsider_id,
            "removed": True,
        }
        assert touch_calls == [outsider_id]

    async with pool.transaction() as conn:
        mismatched_owner_org = await repo._create_organization_on_connection(  # noqa: SLF001
            conn,
            name="Mismatched Owner",
            owner_user_id=member_id,
        )
    with pytest.raises(MembershipAuthorizationError):
        await repo.add_org_member(
            org_id=int(mismatched_owner_org["id"]),
            user_id=owner_id,
            role="owner",
            context=owner_context,
        )
    assert await pool.fetchval(
        "SELECT COUNT(*) FROM teams WHERE org_id = ?",
        (int(mismatched_owner_org["id"]),),
    ) == 0

    # List organizations for user via new helper
    orgs_for_owner, total_for_owner = await repo.list_organizations_for_user(
        owner_id,
        limit=10,
        offset=0,
        with_total=True,
    )
    assert any(o["id"] == org_id for o in orgs_for_owner)
    assert total_for_owner >= 1
