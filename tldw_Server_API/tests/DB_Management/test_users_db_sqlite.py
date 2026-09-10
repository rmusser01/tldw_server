import re
import sqlite3
from pathlib import Path

import pytest

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.exceptions import DatabaseError
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.core.DB_Management.Users_DB import DuplicateUserError, UsersDB

_CANONICAL_PROFILE_VERSION = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z"
)


@pytest.mark.asyncio
async def test_users_db_returns_boolean_flags_under_sqlite(tmp_path: Path):
    db_file = tmp_path / "users.db"
    if db_file.exists():
        db_file.unlink()

    with sqlite3.connect(db_file) as conn:
        conn.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT UNIQUE,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                metadata TEXT,
                is_active INTEGER DEFAULT 1,
                is_superuser INTEGER DEFAULT 0,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                email_verified INTEGER DEFAULT 0,
                is_verified INTEGER DEFAULT 0,
                storage_quota_mb INTEGER DEFAULT 5120,
                storage_used_mb INTEGER DEFAULT 0
            )
            """
        )

    settings = Settings(
        AUTH_MODE="single_user",
        DATABASE_URL=f"sqlite:///{db_file}",
    )

    pool = DatabasePool(settings)
    try:
        users_db = UsersDB(db_pool=pool)
        await users_db.initialize()

        async with pool.transaction() as conn:
            cur = await conn.execute("PRAGMA table_info(users)")
            column_rows = await cur.fetchall()
            columns = {row[1] for row in column_rows}
            assert "profile_version" in columns
            profile_column = {
                row[1]: row for row in column_rows
            }["profile_version"]
            assert profile_column[2].upper() == "TEXT"
            assert profile_column[3] == 1
            assert (
                profile_column[4]
                == "STRFTIME('%Y-%m-%dT%H:%M:%f000Z', 'now')"
            )
            cur = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
            tables = {row[0] for row in await cur.fetchall()}
            assert {
                "org_members",
                "team_members",
                "user_config_overrides",
                "org_config_overrides",
                "team_config_overrides",
            } <= tables

        created = await users_db.create_user(
            username="flagcheck",
            email="flag@example.com",
            password_hash="hashed-password",
            is_active=True,
            is_superuser=False,
        )

        assert created["uuid"]
        assert isinstance(created["uuid"], str)

        for field in ("is_active", "is_superuser", "email_verified"):
            assert isinstance(created[field], bool), f"{field} should be boolean on create"

        fetched = await users_db.get_user_by_username("flagcheck")
        assert fetched and fetched["uuid"]
        for field in ("is_active", "is_superuser", "email_verified"):
            assert isinstance(fetched[field], bool), f"{field} should be boolean via get_user_by_username"

        listed = await users_db.list_users()
        assert listed, "Expected at least one user from list_users"
        for field in ("is_active", "is_superuser", "email_verified"):
            assert isinstance(listed[0][field], bool), f"{field} should be boolean in list_users results"
    finally:
        await pool.close()

    with sqlite3.connect(db_file) as conn:
        conn.execute(
            """
            INSERT INTO users (username, email, password_hash)
            VALUES ('omitted', 'omitted@example.com', 'hash')
            """
        )
        omitted = conn.execute(
            "SELECT profile_version FROM users WHERE username = 'omitted'"
        ).fetchone()[0]
    assert _CANONICAL_PROFILE_VERSION.fullmatch(omitted)


@pytest.mark.asyncio
async def test_users_db_repairs_legacy_nullable_profile_version_before_serving(
    tmp_path: Path,
) -> None:
    db_file = tmp_path / "legacy-profile-version.db"
    with sqlite3.connect(db_file) as conn:
        conn.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT UNIQUE,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                metadata TEXT,
                is_active INTEGER DEFAULT 1,
                is_superuser INTEGER DEFAULT 0,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL,
                last_login TIMESTAMP,
                email_verified INTEGER DEFAULT 0,
                is_verified INTEGER DEFAULT 0,
                storage_quota_mb INTEGER DEFAULT 5120,
                storage_used_mb INTEGER DEFAULT 0,
                profile_version TEXT
            );
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TIMESTAMP NOT NULL
            );
            INSERT INTO schema_migrations VALUES (
                90, 'legacy current', CURRENT_TIMESTAMP
            );
            """
        )
        conn.execute(
            """
            INSERT INTO users (
                uuid, username, email, password_hash, updated_at, profile_version
            ) VALUES (?, ?, ?, ?, ?, NULL)
            """,
            (
                "legacy-uuid",
                "legacy",
                "legacy@example.com",
                "hash",
                "2026-01-02 03:04:05.123456",
            ),
        )

    settings = Settings(
        AUTH_MODE="single_user",
        DATABASE_URL=f"sqlite:///{db_file}",
    )
    pool = DatabasePool(settings)
    try:
        users_db = UsersDB(db_pool=pool)
        await users_db.initialize()
    finally:
        await pool.close()

    with sqlite3.connect(db_file) as conn:
        profile_column = {
            row[1]: row for row in conn.execute("PRAGMA table_info(users)")
        }["profile_version"]
        migrated = conn.execute(
            "SELECT profile_version FROM users WHERE username = 'legacy'"
        ).fetchone()[0]
        conn.execute(
            """
            INSERT INTO users (uuid, username, email, password_hash, updated_at)
            VALUES ('omitted-uuid', 'omitted-legacy', 'omitted-legacy@example.com',
                    'hash', CURRENT_TIMESTAMP)
            """
        )
        omitted = conn.execute(
            "SELECT profile_version FROM users WHERE username = 'omitted-legacy'"
        ).fetchone()[0]

    assert profile_column[2].upper() == "TEXT"
    assert profile_column[3] == 1
    assert profile_column[4] == "STRFTIME('%Y-%m-%dT%H:%M:%f000Z', 'now')"
    assert migrated == "2026-01-02T03:04:05.123456Z"
    assert _CANONICAL_PROFILE_VERSION.fullmatch(omitted)


@pytest.mark.asyncio
async def test_users_db_fails_closed_when_pool_leaves_profile_version_missing(
    tmp_path: Path,
) -> None:
    db_file = tmp_path / "accepted-drift.db"
    with sqlite3.connect(db_file) as conn:
        conn.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT UNIQUE,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                metadata TEXT,
                is_active INTEGER DEFAULT 1,
                is_superuser INTEGER DEFAULT 0,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                email_verified INTEGER DEFAULT 0,
                is_verified INTEGER DEFAULT 0,
                storage_quota_mb INTEGER DEFAULT 5120,
                storage_used_mb INTEGER DEFAULT 0
            );
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TIMESTAMP NOT NULL
            );
            INSERT INTO schema_migrations VALUES (91, 'already accepted', CURRENT_TIMESTAMP);
            """
        )

    settings = Settings(
        AUTH_MODE="single_user",
        DATABASE_URL=f"sqlite:///{db_file}",
    )
    pool = DatabasePool(settings)
    try:
        users_db = UsersDB(db_pool=pool)
        with pytest.raises(DatabaseError, match="profile_version"):
            await users_db.initialize()
    finally:
        await pool.close()

    with sqlite3.connect(db_file) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(users)")}
    assert "profile_version" not in columns


@pytest.mark.asyncio
async def test_users_db_migrates_uuid_column_without_unique_alter(tmp_path: Path):
    db_file = tmp_path / "users_missing_uuid.db"
    if db_file.exists():
        db_file.unlink()

    with sqlite3.connect(db_file) as conn:
        conn.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                metadata TEXT,
                is_active INTEGER DEFAULT 1,
                is_superuser INTEGER DEFAULT 0,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                email_verified INTEGER DEFAULT 0,
                is_verified INTEGER DEFAULT 0,
                storage_quota_mb INTEGER DEFAULT 5120,
                storage_used_mb INTEGER DEFAULT 0
            )
            """
        )
        conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            ("legacy", "legacy@example.com", "pw"),
        )
        conn.commit()

    settings = Settings(
        AUTH_MODE="single_user",
        DATABASE_URL=f"sqlite:///{db_file}",
    )

    pool = DatabasePool(settings)
    try:
        users_db = UsersDB(db_pool=pool)
        await users_db.initialize()

        async with pool.transaction() as conn:
            cur = await conn.execute("PRAGMA table_info(users)")
            columns = {row[1] for row in await cur.fetchall()}
            assert "uuid" in columns

            cur = await conn.execute("SELECT uuid FROM users WHERE username = ?", ("legacy",))
            row = await cur.fetchone()
            assert row is not None
            assert row[0]
    finally:
        await pool.close()


@pytest.mark.asyncio
async def test_create_user_duplicate_race_surfaces_duplicate_error(tmp_path: Path, monkeypatch):
    db_file = tmp_path / "users_race.db"
    settings = Settings(
        AUTH_MODE="single_user",
        DATABASE_URL=f"sqlite:///{db_file}",
    )

    pool = DatabasePool(settings)
    try:
        users_db = UsersDB(db_pool=pool)
        await users_db.initialize()

        async def _return_none(self, *args, **kwargs):  # noqa: ARG001
            return None

        monkeypatch.setattr(UsersDB, "get_user_by_username", _return_none)
        monkeypatch.setattr(UsersDB, "get_user_by_email", _return_none)

        await users_db.create_user(
            username="race",
            email="race@example.com",
            password_hash="pw",
        )

        with pytest.raises(DuplicateUserError):
            await users_db.create_user(
                username="race",
                email="race@example.com",
                password_hash="pw",
            )
    finally:
        await pool.close()
