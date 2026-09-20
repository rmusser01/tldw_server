from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.core.DB_Management import Users_DB as users_db_module
from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB


class _ReadyPoolStub:
    pool = object()

    async def fetchone(self, query: str, *args: object) -> dict[str, object]:
        return {
            "id": 7,
            "uuid": "00000000-0000-0000-0000-000000000007",
            "username": "reader",
            "email": "reader@example.test",
            "password_hash": "hash",
            "is_active": True,
        }

    def transaction(self) -> None:
        raise AssertionError("repository lookup attempted schema DDL")


@pytest.mark.asyncio
async def test_user_lookup_does_not_run_schema_ddl() -> None:
    repo = AuthnzUsersRepo(db_pool=_ReadyPoolStub())  # type: ignore[arg-type]

    user = await repo.get_user_by_id(7)

    assert user is not None
    assert user["id"] == 7


@pytest.mark.asyncio
async def test_users_db_initialize_ensures_schema_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = UsersDB(db_pool=object())  # type: ignore[arg-type]
    calls = 0

    async def record_create_tables() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(db, "_create_tables", record_create_tables)

    await db.initialize()

    assert calls == 1


@pytest.mark.asyncio
async def test_schema_opt_out_still_acquires_global_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready_pool = object()
    db = UsersDB()

    async def get_ready_pool() -> Any:
        return ready_pool

    async def reject_create_tables() -> None:
        raise AssertionError("schema opt-out attempted DDL")

    monkeypatch.setattr(users_db_module, "get_db_pool", get_ready_pool)
    monkeypatch.setattr(db, "_create_tables", reject_create_tables)

    await db.initialize(ensure_schema=False)

    assert db.db_pool is ready_pool


@pytest.mark.asyncio
async def test_schema_opt_out_can_be_followed_by_schema_assurance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = UsersDB(db_pool=object())  # type: ignore[arg-type]
    calls = 0

    async def record_create_tables() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(db, "_create_tables", record_create_tables)

    await db.initialize(ensure_schema=False)
    await db.initialize()

    assert calls == 1


def test_users_schema_migration_harmonizes_legacy_write_columns(tmp_path: Path) -> None:
    from tldw_Server_API.app.core.AuthNZ.migrations import (
        migration_093_harmonize_users_write_columns,
    )

    db_path = tmp_path / "legacy-users.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user'
            )
            """
        )
        conn.execute(
            """
            INSERT INTO users (username, email, password_hash)
            VALUES (?, ?, ?)
            """,
            ("legacy-reader", "legacy@example.test", "hash"),
        )
        migration_093_harmonize_users_write_columns(conn)
        migration_093_harmonize_users_write_columns(conn)
        columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(users)").fetchall()
        }
        migrated_uuid = conn.execute(
            "SELECT uuid FROM users WHERE username = ?",
            ("legacy-reader",),
        ).fetchone()[0]
        indexes = {
            str(row[1])
            for row in conn.execute("PRAGMA index_list(users)").fetchall()
        }

    assert {
        "uuid",
        "is_active",
        "is_superuser",
        "email_verified",
        "is_verified",
        "storage_quota_mb",
        "storage_used_mb",
    } <= columns
    assert migrated_uuid
    assert "idx_users_uuid" in indexes
