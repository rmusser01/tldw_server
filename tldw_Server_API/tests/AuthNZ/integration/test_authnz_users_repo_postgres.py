import asyncio
from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_authnz_users_repo_fetch_by_id_postgres(isolated_test_environment):
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

    client, _db_name = isolated_test_environment
    assert client is not None  # sanity check fixture

    pool = await get_db_pool()

    users_db = UsersDB(pool)
    await users_db.initialize()
    created = await users_db.create_user(
        username="repo_pg_user",
        email="repo_pg_user@example.com",
        password_hash="hash",
        role="user",
        is_active=True,
        is_superuser=False,
        storage_quota_mb=10240,
    )
    user_id = int(created["id"])

    repo = AuthnzUsersRepo(db_pool=pool)
    row = await repo.get_user_by_id(int(user_id))
    assert row is not None
    assert row["username"] == "repo_pg_user"
    assert row["email"] == "repo_pg_user@example.com"
    assert bool(row.get("is_active", True)) is True


@pytest.mark.asyncio
async def test_concurrent_user_reads_do_not_run_schema_ddl_postgres(
    isolated_test_environment,
    monkeypatch,
):
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

    client, _db_name = isolated_test_environment
    assert client is not None

    pool = await get_db_pool()
    users_db = UsersDB(pool)
    await users_db.initialize()
    created = await users_db.create_user(
        username="repo_pg_concurrent_user",
        email="repo_pg_concurrent_user@example.com",
        password_hash="hash",
        role="user",
        is_active=True,
        is_superuser=False,
        storage_quota_mb=10240,
    )
    user_id = int(created["id"])
    repo = AuthnzUsersRepo(db_pool=pool)

    ddl_calls = 0

    async def count_create_tables(self: UsersDB) -> None:
        nonlocal ddl_calls
        ddl_calls += 1

    monkeypatch.setattr(UsersDB, "_create_tables", count_create_tables)

    rows = await asyncio.gather(
        *(repo.get_user_by_id(user_id) for _ in range(24))
    )

    assert all(row is not None and row["id"] == user_id for row in rows)
    assert ddl_calls == 0


@pytest.mark.asyncio
async def test_semantic_health_user_keyset_orders_postgres_created_at_ties(
    isolated_test_environment,
):
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.profile_version import VersionedUserWriteGateway
    from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

    client, _db_name = isolated_test_environment
    assert client is not None
    pool = await get_db_pool()
    users_db = UsersDB(pool)
    await users_db.initialize()
    users = [
        await users_db.create_user(
            username=f"semantic_health_pg_{index}",
            email=f"semantic_health_pg_{index}@example.com",
            password_hash="hash",
        )
        for index in range(3)
    ]
    tied = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)
    gateway = VersionedUserWriteGateway("postgres")
    async with pool.transaction() as conn:
        for user in users:
            await gateway.execute_update(
                conn,
                user_id=int(user["id"]),
                profile_visible_fields=("username",),
                statement="UPDATE users SET created_at=$1,username=username WHERE id=$2",
                parameters=(tied, int(user["id"])),
            )
    repo = AuthnzUsersRepo(db_pool=pool)

    first = await repo.list_users_for_semantic_health_sweep(
        after_created_at=None,
        after_id=None,
        limit=2,
    )
    second = await repo.list_users_for_semantic_health_sweep(
        after_created_at=first[-1]["created_at"],
        after_id=first[-1]["id"],
        limit=2,
    )

    expected = sorted((int(user["id"]) for user in users), reverse=True)
    assert [row["id"] for row in (*first, *second)] == expected
