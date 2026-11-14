"""Quota version and admin status regressions on the official isolated PostgreSQL fixture."""

from __future__ import annotations

import os

import pytest
import pytest_asyncio

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
from tldw_Server_API.app.core.AuthNZ.repos import rbac_admin_repo
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.core.Usage import quota_resolver
from tldw_Server_API.tests.helpers.authnz_seed import ensure_test_user
from tldw_Server_API.tests.UserProfile._storage_quota_helpers import (
    assert_admin_quota_flags,
    assert_quota_clear_preserves_version,
    new_quota_user,
)

pytestmark = [pytest.mark.integration, pytest.mark.postgres]


@pytest_asyncio.fixture
async def quota_accounts(isolated_test_environment, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    """Authenticate a managed admin and seed a target in the fixture-owned database."""
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    quota_resolver.invalidate_all()
    client, _db_name = isolated_test_environment
    password = "QuotaRegression@Test2026!"  # nosec B105 - isolated test credential
    pool = DatabasePool(Settings(DATABASE_URL=os.environ["TEST_DATABASE_URL"]))
    await pool.initialize()
    try:
        admin_id = await ensure_test_user(
            pool,
            "quota-regression-admin",
            password_hash=PasswordService().hash_password(password),
            role="admin",
            is_verified=True,
            is_superuser=True,
        )
        async with pool.transaction() as conn:
            _, roles = await rbac_admin_repo.roles_page(
                conn, is_postgres=True, role_search=None, role_names=["admin"], limit=1, offset=0
            )
            await rbac_admin_repo.add_user_role(
                conn, is_postgres=True, user_id=admin_id, role_id=int(roles[0]["id"])
            )
        used_mb = getattr(request, "param", 0.0)
        surviving_override = request.node.callspec.params.get("surviving_override", False)
        user_id = await new_quota_user(pool, used_mb=used_mb, surviving_override=surviving_override)
    finally:
        await pool.close()
    login = client.post("/api/v1/auth/login", data={"username": "quota-regression-admin", "password": password})
    assert login.status_code == 200  # nosec B101 - pytest regression assertion.
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    return client, headers, user_id


@pytest.mark.parametrize(
    "writer",
    [
        "/api/v1/storage/admin/quotas/user/{user_id}",
        "/api/v1/admin/storage-quotas/users/{user_id}",
        "/api/v1/admin/users/{user_id}",
    ],
    ids=["storage-service", "admin-storage-service", "admin-user"],
)
@pytest.mark.parametrize("surviving_override", [False, True], ids=["only-override", "latest-override"])
@pytest.mark.asyncio
async def test_quota_clear_rejects_old_profile_token_postgres(
    quota_accounts, writer: str, surviving_override: bool
) -> None:
    """Quota null deletion rejects pre-set tokens and accepts a fresh conditional retry."""
    client, headers, user_id = quota_accounts
    assert_quota_clear_preserves_version(client, headers, user_id, writer, surviving_override=surviving_override)


@pytest.mark.parametrize(
    "quota_mb,quota_accounts,soft,hard",
    [
        (100, 79.0, False, False),
        (100, 80.0, True, False),
        (100, 100.0, True, True),
        (100, 120.0, True, True),
        (0, 0.0, True, True),
        (0, 1.0, True, True),
        (None, 120.0, False, False),
    ],
    indirect=["quota_accounts"],
    ids=["below-soft", "soft", "full", "exceeded", "zero-empty", "zero-used", "unlimited"],
)
@pytest.mark.asyncio
async def test_admin_quota_get_put_threshold_flags_postgres(
    quota_accounts, quota_mb: int | None, soft: bool, hard: bool
) -> None:
    """Admin GET and PUT retain the same finite, zero and unlimited threshold policy."""
    client, headers, user_id = quota_accounts
    assert_admin_quota_flags(client, headers, user_id, quota_mb, soft=soft, hard=hard)
