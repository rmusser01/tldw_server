"""PostgreSQL AuthNZ quota bootstrap regression tests."""

from __future__ import annotations

import os

import pytest

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
from tldw_Server_API.app.core.AuthNZ.repos.storage_quotas_repo import AuthnzStorageQuotasRepo
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.core.AuthNZ.settings import Settings

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_fresh_postgres_bootstrap_supports_org_quota_upsert(
    isolated_test_environment: object,
) -> None:
    """An organization quota can be inserted and updated after normal bootstrap."""
    del isolated_test_environment
    pool = DatabasePool(Settings(DATABASE_URL=os.environ["TEST_DATABASE_URL"]))
    await pool.initialize()
    try:
        user_id = await AuthnzUsersRepo(pool).create_user(
            username="quota_probe",
            email="quota_probe@example.test",
            password_hash="synthetic-hash",
            is_verified=True,
        )
        org = await AuthnzOrgsTeamsRepo(pool).create_organization(
            name="Synthetic Quota Organization",
            owner_user_id=user_id,
            slug="synthetic-quota-organization",
        )
        quotas = AuthnzStorageQuotasRepo(pool)

        await quotas.upsert_org_quota(org["id"], quota_mb=256)
        await quotas.upsert_org_quota(org["id"], quota_mb=512)

        stored = await quotas.get_org_quota(org["id"])
        assert stored is not None and stored["quota_mb"] == 512
    finally:
        await pool.close()
