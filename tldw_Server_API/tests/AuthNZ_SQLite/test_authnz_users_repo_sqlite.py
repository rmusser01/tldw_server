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
    from tldw_Server_API.app.core.AuthNZ.database import (
        _GuardedSQLiteConnection,
        get_db_pool,
        reset_db_pool,
    )
    from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    db_path = tmp_path / "users_repo_write.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()
    await reset_db_pool()
    pool = await get_db_pool()
    original_execute = _GuardedSQLiteConnection.execute
    schema_queries: list[str] = []

    async def track_schema_queries(self, query: object, *args: object):
        if isinstance(query, str):
            statement = query.lstrip().upper()
            if statement.startswith(("PRAGMA TABLE_INFO", "ALTER ", "CREATE ")):
                schema_queries.append(statement)
        return await original_execute(self, query, *args)

    monkeypatch.setattr(_GuardedSQLiteConnection, "execute", track_schema_queries)
    repo = AuthnzUsersRepo(db_pool=pool)

    user_id = await repo.create_user(
        username="repo_write_user",
        email="repo_write_user@example.com",
        password_hash="hash",
    )

    assert user_id > 0
    assert schema_queries == []


@pytest.mark.asyncio
async def test_semantic_health_user_keyset_uses_id_to_break_created_at_ties(
    tmp_path,
    monkeypatch,
):
    import sqlite3
    from datetime import datetime, timezone

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

    db_path = tmp_path / "semantic_health_users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    reset_settings()
    await reset_db_pool()
    pool = await get_db_pool()
    users_db = UsersDB(pool)
    await users_db.initialize()
    users = [
        await users_db.create_user(
            username=f"semantic_health_{index}",
            email=f"semantic_health_{index}@example.com",
            password_hash="hash",
        )
        for index in range(3)
    ]
    tied = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE users SET created_at=? WHERE id IN (?,?,?)",
            (
                tied.replace(tzinfo=None).isoformat(sep=" "),
                *(int(user["id"]) for user in users),
            ),
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
    assert all(row["created_at"] == tied for row in (*first, *second))
