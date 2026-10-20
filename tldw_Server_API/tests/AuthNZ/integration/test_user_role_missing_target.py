"""A deleted role cannot invalidate an earlier saved assignment or hide behind 500."""

import asyncio
import os

import pytest

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
from tldw_Server_API.app.core.AuthNZ.repos import rbac_admin_repo
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.tests.helpers.authnz_seed import ensure_test_user

pytestmark = [pytest.mark.integration, pytest.mark.postgres, pytest.mark.asyncio]


async def test_deleted_role_reports_not_found_and_retains_saved_authority(isolated_test_environment):
    """A stale role picker receives 404; earlier roles survive and a fresh pick works."""
    client, _ = isolated_test_environment
    password = "MissingRole@Test2026!"  # nosec B105 - isolated test credential.
    pool = DatabasePool(Settings(DATABASE_URL=os.environ["TEST_DATABASE_URL"]))
    await pool.initialize()
    try:
        admin_id = await ensure_test_user(
            pool,
            "missing-role-admin",
            role="admin",
            is_verified=True,
            is_superuser=True,
            password_hash=PasswordService().hash_password(password),
        )
        user_id = await ensure_test_user(pool, "missing-role-user", is_verified=True)
        async with pool.transaction() as conn:
            _, roles = await rbac_admin_repo.roles_page(
                conn, is_postgres=True, role_search=None, role_names=["admin", "user"], limit=2, offset=0
            )
            role_ids = {role["name"]: role["id"] for role in roles}
            await rbac_admin_repo.add_user_role(conn, is_postgres=True, user_id=admin_id, role_id=role_ids["admin"])
            await rbac_admin_repo.add_user_role(conn, is_postgres=True, user_id=user_id, role_id=role_ids["user"])
    finally:
        await pool.close()

    login = client.post("/api/v1/auth/login", data={"username": "missing-role-admin", "password": password})
    assert login.status_code == 200  # nosec B101 - pytest regression assertion.
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    roles_path = f"/api/v1/admin/users/{user_id}/roles"
    permissions_path = f"/api/v1/admin/users/{user_id}/effective-permissions"

    def create_role(name):
        response = client.post("/api/v1/admin/roles", headers=headers, json={"name": name})
        assert response.status_code == 200  # nosec B101 - pytest regression assertion.
        return response.json()["id"]

    saved_role = create_role("c08-saved-role")
    permission = next(
        p for p in client.get("/api/v1/admin/permissions", headers=headers).json() if p["name"] == "media.delete"
    )
    assert (
        client.post(f"/api/v1/admin/roles/{saved_role}/permissions/{permission['id']}", headers=headers).status_code
        == 200
    )  # nosec B101 - pytest regression assertion.
    assert client.post(f"{roles_path}/{saved_role}", headers=headers).status_code == 200  # nosec B101 - pytest regression assertion.
    saved_roles = client.get(roles_path, headers=headers).json()
    saved_permissions = client.get(permissions_path, headers=headers).json()
    assert "media.delete" in saved_permissions["permissions"]  # nosec B101 - pytest regression assertion.
    effective = client.get(f"/api/v1/admin/roles/{saved_role}/permissions/effective", headers=headers)
    assert effective.status_code == 200  # nosec B101 - pytest regression assertion.
    assert effective.json()["permissions"] == ["media.delete"]  # nosec B101 - pytest regression assertion.

    deleted_role = create_role("c08-deleted-role")
    assert client.delete(f"/api/v1/admin/roles/{deleted_role}", headers=headers).status_code == 200  # nosec B101 - pytest regression assertion.
    missing = client.post(f"{roles_path}/{deleted_role}", headers=headers)
    assert missing.status_code == 404  # nosec B101 - pytest regression assertion.
    assert missing.json()["detail"] == "Role not found"  # nosec B101 - pytest regression assertion.
    assert client.get(roles_path, headers=headers).json() == saved_roles  # nosec B101 - pytest regression assertion.
    assert client.get(permissions_path, headers=headers).json() == saved_permissions  # nosec B101 - pytest regression assertion.

    replacement_role = create_role("c08-replacement-role")
    assert client.post(f"{roles_path}/{replacement_role}", headers=headers).status_code == 200  # nosec B101 - pytest regression assertion.
    assert [role["id"] for role in client.get(roles_path, headers=headers).json()["roles"]].count(replacement_role) == 1  # nosec B101 - pytest regression assertion.
    assert client.post(f"{roles_path}/{replacement_role}", headers=headers).status_code == 200  # nosec B101 - pytest regression assertion.
    assert [role["id"] for role in client.get(roles_path, headers=headers).json()["roles"]].count(replacement_role) == 1  # nosec B101 - pytest regression assertion.
    assert client.delete(f"{roles_path}/{replacement_role}", headers=headers).status_code == 200  # nosec B101 - pytest regression assertion.
    assert client.get(roles_path, headers=headers).json() == saved_roles  # nosec B101 - pytest regression assertion.


async def test_role_lookup_holds_concurrent_deletion_until_assignment_commit(isolated_test_environment):
    """The checked role remains present through the same transaction's assignment."""
    from tldw_Server_API.app.services.admin_roles_permissions_service import create_role, delete_role

    pool = DatabasePool(Settings(DATABASE_URL=os.environ["TEST_DATABASE_URL"]))
    await pool.initialize()
    deletion = None
    try:
        user_id = await ensure_test_user(pool, "role-race-user", is_verified=True)
        async with pool.transaction() as conn:
            role = await create_role(conn, "c08-role-delete-race")
        deleting = asyncio.Event()

        async def remove_checked_role():
            async with pool.transaction() as other:
                deleting.set()
                await delete_role(other, role["id"])

        async with pool.transaction() as conn:
            checked = await rbac_admin_repo.get_role(conn, is_postgres=True, role_id=role["id"])
            assert checked is not None  # nosec B101 - pytest regression assertion.
            deletion = asyncio.create_task(remove_checked_role())
            await asyncio.wait_for(deleting.wait(), timeout=5)
            try:
                await asyncio.wait_for(asyncio.shield(deletion), timeout=1)
            except TimeoutError:
                pass
            else:
                pytest.fail("Role deletion completed before the checked assignment transaction committed")
            await rbac_admin_repo.add_user_role(conn, is_postgres=True, user_id=user_id, role_id=role["id"])
        await asyncio.wait_for(deletion, timeout=5)
        async with pool.transaction() as conn:
            assert await rbac_admin_repo.get_role(conn, is_postgres=True, role_id=role["id"]) is None  # nosec B101 - pytest regression assertion.
    finally:
        if deletion is not None and not deletion.done():
            deletion.cancel()
            await asyncio.gather(deletion, return_exceptions=True)
        await pool.close()
