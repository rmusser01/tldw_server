"""Exercise profile-related array queries with the real PostgreSQL driver."""

from __future__ import annotations

import os

import asyncpg
import pytest
import pytest_asyncio

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import ensure_authnz_core_tables_pg
from tldw_Server_API.app.core.AuthNZ.repos.managed_secret_refs_repo import ManagedSecretRefsRepo
from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.core.UserProfiles.overrides_repo import (
    OrgProfileOverridesRepo,
    TeamProfileOverridesRepo,
)

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def profile_array_pool(isolated_test_environment):
    """Use the standard isolated database and production schema bootstrap."""
    pool = DatabasePool(Settings(DATABASE_URL=os.environ["TEST_DATABASE_URL"]))
    await pool.initialize()
    try:
        assert await ensure_authnz_core_tables_pg(pool)
        yield pool
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_profile_override_membership_arrays_postgres(profile_array_pool):
    pool = profile_array_pool
    org_repo = OrgProfileOverridesRepo(pool)
    team_repo = TeamProfileOverridesRepo(pool)
    org_ids = [
        await pool.fetchval("INSERT INTO organizations (name) VALUES ($1) RETURNING id", name)
        for name in ("array-org-one", "array-org-two", "array-org-unrelated")
    ]
    team_ids = [
        await pool.fetchval(
            "INSERT INTO teams (org_id, name) VALUES ($1, $2) RETURNING id",
            org_id,
            "array-team",
        )
        for org_id in org_ids
    ]
    for org_id, team_id in zip(org_ids, team_ids):
        await org_repo.upsert_override(org_id=org_id, key="theme", value="dark", updated_by=None)
        await team_repo.upsert_override(team_id=team_id, key="theme", value="light", updated_by=None)

    assert await org_repo.list_overrides_for_orgs([]) == []
    assert await team_repo.get_latest_update_for_teams([]) is None
    for size in (1, 2):
        organizations = await org_repo.list_overrides_for_orgs(org_ids[:size])
        teams = await team_repo.list_overrides_for_teams(team_ids[:size])
        assert [row["org_id"] for row in organizations] == org_ids[:size]
        assert [row["team_id"] for row in teams] == team_ids[:size]
        assert await org_repo.get_latest_update_for_orgs(org_ids[:size]) == max(
            row["updated_at"] for row in organizations
        )
        assert await team_repo.get_latest_update_for_teams(team_ids[:size]) == max(row["updated_at"] for row in teams)


@pytest.mark.asyncio
async def test_scoped_user_and_organization_counts_postgres(profile_array_pool):
    pool = profile_array_pool
    # Seed through an unmanaged fixture connection; production users writes have
    # a separate ownership guard and are not part of this read/binding regression.
    connection = await asyncpg.connect(os.environ["TEST_DATABASE_URL"])
    try:
        user_id = await connection.fetchval(
            "INSERT INTO users (username, email, password_hash) VALUES ($1, $2, $3) RETURNING id",
            "array-count-user",
            "array-count-user@example.test",
            "fixture-password-hash",
        )
    finally:
        await connection.close()
    org_id = await pool.fetchval("INSERT INTO organizations (name) VALUES ($1) RETURNING id", "scoped-org")
    await pool.execute("INSERT INTO org_members (org_id, user_id) VALUES ($1, $2)", org_id, user_id)

    organizations, org_total = await AuthnzOrgsTeamsRepo(pool).list_organizations(org_ids=[org_id], with_total=True)
    users, user_total = await AuthnzUsersRepo(pool).list_users(org_ids=[org_id], limit=10, offset=0)

    assert ([row["id"] for row in organizations], org_total) == ([org_id], 1)
    assert ([row["id"] for row in users], user_total) == ([user_id], 1)


@pytest.mark.asyncio
async def test_managed_secret_reference_arrays_postgres(profile_array_pool):
    repo = ManagedSecretRefsRepo(profile_array_pool)
    await repo.ensure_tables()
    await repo.ensure_backend_registration(name="array-test", display_name="Array test")
    first = await repo.upsert_ref(
        backend_name="array-test",
        owner_scope_type="user",
        owner_scope_id=71,
        provider_key="openai",
        backend_ref="test/first",
        metadata=None,
    )
    second = await repo.upsert_ref(
        backend_name="array-test",
        owner_scope_type="user",
        owner_scope_id=72,
        provider_key="openai",
        backend_ref="test/second",
        metadata=None,
    )

    assert await repo.list_refs_by_ids([]) == {}
    assert set(await repo.list_refs_by_ids([first["id"]])) == {first["id"]}
    assert set(await repo.list_refs_by_ids([first["id"], second["id"]])) == {first["id"], second["id"]}
