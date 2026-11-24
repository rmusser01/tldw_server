"""TASK-15640: self-service email change needs the current password, keeps the previous address, notifies it."""

import sqlite3
from io import StringIO
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit, get_auth_principal
from tldw_Server_API.app.api.v1.endpoints import users
from tldw_Server_API.app.api.v2.endpoints import user_profiles as v2_profiles
from tldw_Server_API.app.core.AuthNZ import migrations
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.password_service import hash_password
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.DB_Management.user_profile_writes import update_user_email
from tldw_Server_API.app.core.UserProfiles.overrides_repo import UserProfileOverridesRepo
from tldw_Server_API.app.core.UserProfiles.update_service import UserProfileUpdateService
from tldw_Server_API.tests.AuthNZ_SQLite._user_fixtures import create_authnz_test_user

OWNER_EMAIL = "owner@example.test"
OWNER_PASSWORD = "Owner-Pass-2026!"


def test_migration_108_adds_email_change_columns_once(tmp_path) -> None:
    with sqlite3.connect(tmp_path / "users.db") as conn:
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT NOT NULL)")
        conn.execute("INSERT INTO users (id, email) VALUES (1, 'owner@example.test')")
        migrations.migration_108_add_users_email_change_columns(conn)
        migrations.migration_108_add_users_email_change_columns(conn)
        columns = [row[1] for row in conn.execute("PRAGMA table_info(users)")]
        row = conn.execute("SELECT email, previous_email, email_changed_at FROM users").fetchone()
    assert (columns.count("previous_email"), columns.count("email_changed_at")) == (1, 1)
    assert row == ("owner@example.test", None, None)


def test_migration_108_skips_without_users_table(tmp_path) -> None:
    """Synthetic databases migrate to the latest version without a users table (cf. migrations 100 and 107)."""
    with sqlite3.connect(tmp_path / "synthetic.db") as conn:
        migrations.migration_108_add_users_email_change_columns(conn)
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert "users" not in tables


@pytest.fixture
async def owner_db(tmp_path, monkeypatch, fresh_services):
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users.db'}")
    monkeypatch.setenv("JWT_SECRET_KEY", "synthetic-signing-key-for-regression-tests-only")
    monkeypatch.setenv("REDIS_URL", "")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_databases"))
    reset_settings()
    pool = await get_db_pool()
    user_id = await create_authnz_test_user(
        pool, username="owner", email=OWNER_EMAIL, password_hash=hash_password(OWNER_PASSWORD), is_verified=True
    )
    try:
        yield SimpleNamespace(pool=pool, user_id=user_id)
    finally:
        reset_settings()


async def _columns(db) -> tuple:
    row = dict(await db.pool.fetchone("SELECT email, previous_email, email_changed_at FROM users WHERE id = ?", db.user_id))
    return row["email"], row["previous_email"], row["email_changed_at"]


def _raw(db, statement: str) -> None:
    """Write outside the AuthNZ write guard, as an operator or the passage of time would."""
    with sqlite3.connect(db.pool.db_path) as conn:
        conn.execute(statement, (db.user_id,))


async def _change(db, email: str):
    async with db.pool.transaction() as conn:
        return await update_user_email(conn, backend="sqlite", user_id=db.user_id, email=email)


@pytest.mark.asyncio
async def test_email_change_keeps_the_previous_address_and_time(owner_db):
    await _change(owner_db, "second@example.test")
    email, previous, changed_at = await _columns(owner_db)
    assert (email, previous) == ("second@example.test", OWNER_EMAIL)
    assert changed_at is not None


@pytest.mark.asyncio
async def test_second_change_within_seven_days_keeps_the_original_previous_address(owner_db):
    await _change(owner_db, "typo@example.test")
    await _change(owner_db, "fixed@example.test")
    assert (await _columns(owner_db))[:2] == ("fixed@example.test", OWNER_EMAIL)


@pytest.mark.asyncio
async def test_change_after_seven_days_records_the_immediately_previous_address(owner_db):
    await _change(owner_db, "second@example.test")
    _raw(owner_db, "UPDATE users SET email_changed_at = datetime('now', '-8 days') WHERE id = ?")
    await _change(owner_db, "third@example.test")
    assert (await _columns(owner_db))[:2] == ("third@example.test", "second@example.test")


@pytest.mark.asyncio
async def test_same_address_in_any_case_changes_nothing(owner_db):
    result = await _change(owner_db, OWNER_EMAIL.upper())
    assert result.affected_user_ids == ()
    assert await _columns(owner_db) == (OWNER_EMAIL, None, None)


@pytest.mark.asyncio
async def test_own_profile_email_edit_records_the_previous_address(owner_db):
    async with owner_db.pool.transaction() as conn:
        result = await UserProfileUpdateService(owner_db.pool).apply_updates(
            user_id=owner_db.user_id, updates=(("identity.email", "Second@Example.com"),), roles={"user"},
            dry_run=False, db_conn=conn, updated_by=owner_db.user_id,
        )
    assert result.applied == ["identity.email"]
    assert (await _columns(owner_db))[:2] == ("second@example.com", OWNER_EMAIL)
