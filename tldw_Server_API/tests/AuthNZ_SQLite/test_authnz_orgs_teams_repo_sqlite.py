from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_authnz_orgs_teams_repo_membership_sqlite(tmp_path, monkeypatch):
    """AuthnzOrgsTeamsRepo membership helpers should work on SQLite."""
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.exceptions import DuplicateOrganizationError
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import (
        AuthnzOrgsTeamsRepo,
        DEFAULT_BASE_TEAM_NAME,
    )

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    # Create two users for org/team membership
    async with pool.transaction() as conn:
        await conn.execute(
            """
            INSERT INTO users (username, email, password_hash, is_active)
            VALUES (?, ?, ?, 1)
            """,
            ("owner", "owner@example.com", "x"),
        )
        await conn.execute(
            """
            INSERT INTO users (username, email, password_hash, is_active)
            VALUES (?, ?, ?, 1)
            """,
            ("member", "member@example.com", "x"),
        )

    owner_id = await pool.fetchval(
        "SELECT id FROM users WHERE username = ?", ("owner",)
    )
    member_id = await pool.fetchval(
        "SELECT id FROM users WHERE username = ?", ("member",)
    )

    repo = AuthnzOrgsTeamsRepo(pool)

    # Create organization and add members
    org = await repo.create_organization(name="Acme Corp", owner_user_id=owner_id)
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

    await repo.create_organization(name="Other Org", owner_user_id=owner_id, slug="other-org")
    with pytest.raises(DuplicateOrganizationError):
        await repo.update_organization(org_id=org_id, slug="other-org")

    owner_membership = await repo.add_org_member(
        org_id=org_id, user_id=owner_id, role="owner"
    )
    member_membership = await repo.add_org_member(
        org_id=org_id, user_id=member_id, role="member"
    )

    assert owner_membership["org_id"] == org_id
    assert owner_membership["user_id"] == owner_id
    assert owner_membership["role"].lower() == "owner"

    assert member_membership["org_id"] == org_id
    assert member_membership["user_id"] == member_id
    assert member_membership["role"].lower() == "member"

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

    # Update non-owner role
    updated_member = await repo.update_org_member_role(
        org_id=org_id,
        user_id=member_id,
        role="admin",
    )
    assert updated_member is not None
    assert updated_member["role"].lower() == "admin"

    # Cannot demote the last owner
    demote_owner = await repo.update_org_member_role(
        org_id=org_id,
        user_id=owner_id,
        role="member",
    )
    assert demote_owner is not None
    assert demote_owner["role"].lower() == "owner"
    assert demote_owner.get("error") == "owner_required"

    # Removing a non-owner should also remove them from the default team
    remove_member = await repo.remove_org_member(org_id=org_id, user_id=member_id)
    assert remove_member["removed"] is True

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

    # Removing the last owner should be blocked with owner_required
    remove_owner = await repo.remove_org_member(org_id=org_id, user_id=owner_id)
    assert remove_owner["removed"] is False
    assert remove_owner.get("error") == "owner_required"

    # List organizations for user via new helper
    orgs_for_owner, total_for_owner = await repo.list_organizations_for_user(
        owner_id,
        limit=10,
        offset=0,
        with_total=True,
    )
    assert any(o["id"] == org_id for o in orgs_for_owner)
    assert total_for_owner >= 1
