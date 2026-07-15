from contextlib import asynccontextmanager

import pytest


@pytest.mark.asyncio
async def test_authnz_users_repo_fetch_by_id_sqlite(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

    db_path = tmp_path / "users_repo.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()

    # Use the existing UsersDB abstraction to create a user so the schema
    # details remain centralized.
    users_db = UsersDB(pool)
    await users_db.initialize()
    created = await users_db.create_user(
        username="repo_user",
        email="repo_user@example.com",
        password_hash="hash",
        role="user",
        is_active=True,
        is_superuser=False,
        storage_quota_mb=5120,
    )
    user_id = int(created["id"])

    repo = AuthnzUsersRepo(db_pool=pool)
    row = await repo.get_user_by_id(int(user_id))
    assert row is not None
    assert row["username"] == "repo_user"
    assert row["email"] == "repo_user@example.com"
    assert bool(row.get("is_active", True)) is True


@pytest.mark.asyncio
async def test_authnz_users_repo_create_does_not_inspect_or_mutate_schema(
    tmp_path,
    monkeypatch,
):
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    db_path = tmp_path / "users_repo_write.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()
    await reset_db_pool()
    pool = await get_db_pool()
    original_transaction = pool.transaction
    schema_queries: list[str] = []

    @asynccontextmanager
    async def track_schema_queries():
        async with original_transaction() as conn:
            class TrackingConnection:
                async def execute(self, query: str, *args):
                    statement = query.lstrip().upper()
                    if statement.startswith(("PRAGMA TABLE_INFO", "ALTER ", "CREATE ")):
                        schema_queries.append(statement)
                    return await conn.execute(query, *args)

                def __getattr__(self, name: str):
                    return getattr(conn, name)

            yield TrackingConnection()

    monkeypatch.setattr(pool, "transaction", track_schema_queries)
    repo = AuthnzUsersRepo(db_pool=pool)

    user_id = await repo.create_user(
        username="repo_write_user",
        email="repo_write_user@example.com",
        password_hash="hash",
    )

    assert user_id > 0
    assert schema_queries == []
