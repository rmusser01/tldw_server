from __future__ import annotations

import uuid
from datetime import datetime

import pytest

from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    TrustedMembershipReason,
    TrustedMembershipWriteContext,
)
from tldw_Server_API.app.core.AuthNZ.profile_version import VersionedUserWriteGateway
from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import (
    DEFAULT_BASE_TEAM_NAME,
    AuthnzOrgsTeamsRepo,
)

pytestmark = pytest.mark.integration
_BOOTSTRAP_MEMBERSHIP_CONTEXT = TrustedMembershipWriteContext(
    trusted_reason=TrustedMembershipReason.BOOTSTRAP,
)


@pytest.mark.asyncio
async def test_authnz_orgs_teams_repo_membership_postgres(test_db_pool):
    """AuthnzOrgsTeamsRepo membership helpers should work on Postgres."""
    pool = test_db_pool

    # Create two users
    now = datetime.utcnow().replace(microsecond=0)
    async with pool.transaction() as conn:
        gateway = VersionedUserWriteGateway("postgres")
        for username in ("owner_pg", "member_pg"):
            await gateway.insert_user(
                conn,
                values={
                    "uuid": str(uuid.uuid4()),
                    "username": username,
                    "email": f"{username}@example.com",
                    "password_hash": "x",
                    "role": "user",
                    "is_active": True,
                    "is_verified": True,
                    "storage_quota_mb": 5120,
                    "storage_used_mb": 0.0,
                    "created_at": now,
                },
            )

    owner_id = await pool.fetchval(
        "SELECT id FROM users WHERE username = $1", "owner_pg"
    )
    member_id = await pool.fetchval(
        "SELECT id FROM users WHERE username = $1", "member_pg"
    )

    repo = AuthnzOrgsTeamsRepo(pool)

    # Create organization and add members (owner + member)
    org = await repo.create_organization(name="PG Acme Corp", owner_user_id=owner_id)
    org_id = org["id"]

    owner_membership = await repo.add_org_member(
        org_id=org_id,
        user_id=owner_id,
        role="owner",
        context=_BOOTSTRAP_MEMBERSHIP_CONTEXT,
    )
    member_membership = await repo.add_org_member(
        org_id=org_id,
        user_id=member_id,
        role="member",
        context=_BOOTSTRAP_MEMBERSHIP_CONTEXT,
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
    assert any(
        m["org_id"] == org_id and m["role"].lower() == "owner"
        for m in owner_memberships
    )

    # Default team should be created and both users enrolled
    default_team_id = await pool.fetchval(
        "SELECT id FROM teams WHERE org_id = $1 AND name = $2",
        org_id,
        DEFAULT_BASE_TEAM_NAME,
    )
    assert default_team_id is not None

    owner_team_count = await pool.fetchval(
        "SELECT COUNT(*) FROM team_members WHERE team_id = $1 AND user_id = $2",
        default_team_id,
        owner_id,
    )
    member_team_count = await pool.fetchval(
        "SELECT COUNT(*) FROM team_members WHERE team_id = $1 AND user_id = $2",
        default_team_id,
        member_id,
    )
    assert owner_team_count == 1
    assert member_team_count == 1

    # Update non-owner role
    updated_member = await repo.update_org_member_role(
        org_id=org_id,
        user_id=member_id,
        role="admin",
        context=_BOOTSTRAP_MEMBERSHIP_CONTEXT,
    )
    assert updated_member is not None
    assert updated_member["role"].lower() == "admin"

    # Cannot demote the last owner
    demote_owner = await repo.update_org_member_role(
        org_id=org_id,
        user_id=owner_id,
        role="member",
        context=_BOOTSTRAP_MEMBERSHIP_CONTEXT,
    )
    assert demote_owner is not None
    assert demote_owner["role"].lower() == "owner"
    assert demote_owner.get("error") == "owner_required"

    # Removing a non-owner should also remove them from the default team
    remove_member = await repo.remove_org_member(
        org_id=org_id,
        user_id=member_id,
        context=_BOOTSTRAP_MEMBERSHIP_CONTEXT,
    )
    assert remove_member["removed"] is True

    remaining_members = await repo.list_org_members(org_id=org_id)
    assert all(m["user_id"] != member_id for m in remaining_members)

    member_team_count_after = await pool.fetchval(
        "SELECT COUNT(*) FROM team_members WHERE team_id = $1 AND user_id = $2",
        default_team_id,
        member_id,
    )
    owner_team_count_after = await pool.fetchval(
        "SELECT COUNT(*) FROM team_members WHERE team_id = $1 AND user_id = $2",
        default_team_id,
        owner_id,
    )
    assert member_team_count_after == 0
    assert owner_team_count_after == 1

    # Removing the last owner should be blocked with owner_required
    remove_owner = await repo.remove_org_member(
        org_id=org_id,
        user_id=owner_id,
        context=_BOOTSTRAP_MEMBERSHIP_CONTEXT,
    )
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
    assert total_for_owner == 1


@pytest.mark.asyncio
async def test_authnz_orgs_teams_repo_list_organizations_for_user_variants_postgres(test_db_pool):
    """AuthnzOrgsTeamsRepo.list_organizations_for_user should support totals, pagination and empty memberships."""
    pool = test_db_pool
    now = datetime.utcnow().replace(microsecond=0)

    # Create 3 users: org owner, multi-org member, and a user with no memberships.
    async with pool.transaction() as conn:
        gateway = VersionedUserWriteGateway("postgres")
        for username, email in (
            ("owner_list_pg", "owner_list_pg@example.com"),
            ("member_list_pg", "member_list_pg@example.com"),
            ("no_orgs_pg", "no_orgs_pg@example.com"),
        ):
            await gateway.insert_user(
                conn,
                values={
                    "uuid": str(uuid.uuid4()),
                    "username": username,
                    "email": email,
                    "password_hash": "x",
                    "role": "user",
                    "is_active": True,
                    "is_verified": True,
                    "storage_quota_mb": 5120,
                    "storage_used_mb": 0.0,
                    "created_at": now,
                },
            )

    owner_id = await pool.fetchval(
        "SELECT id FROM users WHERE username = $1", "owner_list_pg"
    )
    member_id = await pool.fetchval(
        "SELECT id FROM users WHERE username = $1", "member_list_pg"
    )
    no_orgs_user_id = await pool.fetchval(
        "SELECT id FROM users WHERE username = $1", "no_orgs_pg"
    )

    repo = AuthnzOrgsTeamsRepo(pool)

    # Create 3 orgs and enroll the owner in all, member in 2.
    org_ids: list[int] = []
    for name in ("PG Org One", "PG Org Two", "PG Org Three"):
        org = await repo.create_organization(name=name, owner_user_id=owner_id)
        org_ids.append(org["id"])
        await repo.add_org_member(
            org_id=org["id"],
            user_id=owner_id,
            role="owner",
            context=_BOOTSTRAP_MEMBERSHIP_CONTEXT,
        )

    await repo.add_org_member(org_id=org_ids[0], user_id=member_id, role="member", context=_BOOTSTRAP_MEMBERSHIP_CONTEXT)
    await repo.add_org_member(org_id=org_ids[1], user_id=member_id, role="member", context=_BOOTSTRAP_MEMBERSHIP_CONTEXT)

    # Pagination: first page + second page should cover all orgs exactly once.
    page_1, total_1 = await repo.list_organizations_for_user(
        owner_id,
        limit=2,
        offset=0,
        with_total=True,
    )
    assert total_1 == 3
    assert len(page_1) == 2

    page_2, total_2 = await repo.list_organizations_for_user(
        owner_id,
        limit=2,
        offset=2,
        with_total=True,
    )
    assert total_2 == 3
    assert len(page_2) == 1

    page_1_ids = {o["id"] for o in page_1}
    page_2_ids = {o["id"] for o in page_2}
    assert page_1_ids.isdisjoint(page_2_ids)
    assert page_1_ids | page_2_ids == set(org_ids)

    # with_total=False path should return a zero total and still return rows.
    rows_no_total, total_no_total = await repo.list_organizations_for_user(
        owner_id,
        limit=10,
        offset=0,
        with_total=False,
    )
    assert total_no_total == 0
    assert {o["id"] for o in rows_no_total} == set(org_ids)

    # Users with multiple org memberships.
    member_orgs, member_total = await repo.list_organizations_for_user(
        member_id,
        limit=10,
        offset=0,
        with_total=True,
    )
    assert member_total == 2
    assert {o["id"] for o in member_orgs} == {org_ids[0], org_ids[1]}

    # Users with no org memberships.
    empty_orgs, empty_total = await repo.list_organizations_for_user(
        no_orgs_user_id,
        limit=10,
        offset=0,
        with_total=True,
    )
    assert empty_orgs == []
    assert empty_total == 0
