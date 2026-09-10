from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import aiosqlite
import pytest

from tldw_Server_API.app.core.AuthNZ.database import _GuardedSQLiteConnection
from tldw_Server_API.app.core.AuthNZ.sqlite_profile_version_schema import (
    SQLITE_PROFILE_VERSION_COLUMN_SQL,
)
from tldw_Server_API.app.core.AuthNZ.exceptions import RegistrationError
from tldw_Server_API.app.services.registration_service import RegistrationService


class _PasswordServiceStub:
    def validate_password_strength(self, password: str, username: str | None = None) -> None:  # noqa: ARG002
        return None

    def hash_password(self, password: str) -> str:
        return f"hash-{password}"


class _SQLitePool:
    """Minimal stand-in for DatabasePool's SQLite transaction contract.

    The real pool yields a _GuardedSQLiteConnection, not a bare aiosqlite
    connection. Writes to profile-visible tables are handed to it as
    _ProfileUserSql capability objects that only the guard knows how to unwrap,
    so a stub that yields the raw connection fails with
    "execute() argument 1 must be str, not _ProfileUserSql".
    """

    pool = None
    backend = "sqlite"

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[_GuardedSQLiteConnection]:
        conn = await aiosqlite.connect(self.db_path)
        await conn.execute("PRAGMA foreign_keys = ON")
        await conn.execute("BEGIN IMMEDIATE")
        try:
            yield _GuardedSQLiteConnection(conn)
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise
        finally:
            await conn.close()


def _initialize_auth_db(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                uuid TEXT UNIQUE NOT NULL,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                is_active INTEGER NOT NULL,
                is_verified INTEGER NOT NULL,
                created_by INTEGER,
                storage_quota_mb INTEGER NOT NULL,
                {profile_version_column}
            );
            CREATE TABLE roles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL
            );
            CREATE TABLE user_roles (
                user_id INTEGER NOT NULL,
                role_id INTEGER NOT NULL,
                PRIMARY KEY (user_id, role_id),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (role_id) REFERENCES roles(id) ON DELETE CASCADE
            );
            CREATE TABLE password_history (
                user_id INTEGER NOT NULL,
                password_hash TEXT NOT NULL
            );
            CREATE TABLE audit_logs (
                user_id INTEGER,
                action TEXT,
                resource_type TEXT,
                resource_id INTEGER,
                status TEXT,
                details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE registration_codes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT UNIQUE NOT NULL,
                role_to_grant TEXT,
                times_used INTEGER NOT NULL DEFAULT 0,
                max_uses INTEGER NOT NULL,
                expires_at TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                description TEXT,
                allowed_email_domain TEXT,
                org_id INTEGER,
                org_role TEXT,
                team_id INTEGER,
                metadata TEXT
            );
            INSERT INTO roles (name) VALUES ('user'), ('reviewer');
            """.format(profile_version_column=SQLITE_PROFILE_VERSION_COLUMN_SQL)
        )


def _make_service(tmp_path: Path) -> tuple[RegistrationService, Path]:
    db_path = tmp_path / "users.db"
    _initialize_auth_db(db_path)
    settings = SimpleNamespace(
        ENABLE_REGISTRATION=True,
        REQUIRE_REGISTRATION_CODE=False,
        ENABLE_ORG_SCOPED_REGISTRATION_CODES=True,
        DEFAULT_USER_ROLE="user",
        DEFAULT_STORAGE_QUOTA_MB=1024,
        USER_DATA_BASE_PATH=str(tmp_path / "user_data"),
        CHROMADB_BASE_PATH=None,
    )
    service = RegistrationService(
        db_pool=_SQLitePool(db_path),
        password_service=_PasswordServiceStub(),
        settings=settings,
    )
    service._create_user_directories = lambda user_id: True  # noqa: ARG005
    return service, db_path


def _role_memberships(db_path: Path, username: str) -> list[str]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT r.name
            FROM roles r
            JOIN user_roles ur ON ur.role_id = r.id
            JOIN users u ON u.id = ur.user_id
            WHERE u.username = ?
            ORDER BY r.name
            """,
            (username,),
        ).fetchall()
    return [str(row[0]) for row in rows]


def _registration_audit(db_path: Path, username: str) -> tuple[str, str, str] | None:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT a.resource_type, u.username, a.status
            FROM audit_logs a
            JOIN users u ON u.id = a.resource_id
            WHERE a.action = 'user_registered'
              AND json_extract(a.details, '$.username') = ?
            """,
            (username,),
        ).fetchone()
    if row is None:
        return None
    return str(row[0]), str(row[1]), str(row[2])


@pytest.mark.asyncio
async def test_default_registration_persists_canonical_role_before_return(tmp_path: Path) -> None:
    service, db_path = _make_service(tmp_path)

    payload = await service.register_user(
        username="default-user",
        email="default-user@example.com",
        password="Strong!Pass9",
    )

    assert payload["role"] == "user"
    assert _role_memberships(db_path, "default-user") == ["user"]
    assert _registration_audit(db_path, "default-user") == (
        "user",
        "default-user",
        "success",
    )


@pytest.mark.asyncio
async def test_registration_code_role_is_persisted_as_canonical_membership(tmp_path: Path) -> None:
    service, db_path = _make_service(tmp_path)
    expires_at = (datetime.utcnow() + timedelta(days=1)).isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO registration_codes (
                code, role_to_grant, times_used, max_uses, expires_at, is_active
            ) VALUES (?, ?, 0, 1, ?, 1)
            """,
            ("reviewer-code", "reviewer", expires_at),
        )

    payload = await service.register_user(
        username="reviewer-user",
        email="reviewer-user@example.com",
        password="Strong!Pass9",
        registration_code="reviewer-code",
    )

    assert payload["role"] == "reviewer"
    assert _role_memberships(db_path, "reviewer-user") == ["reviewer"]


@pytest.mark.asyncio
async def test_unknown_registration_code_role_rolls_back_user_and_code_use(tmp_path: Path) -> None:
    service, db_path = _make_service(tmp_path)
    expires_at = (datetime.utcnow() + timedelta(days=1)).isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO registration_codes (
                code, role_to_grant, times_used, max_uses, expires_at, is_active
            ) VALUES (?, ?, 0, 1, ?, 1)
            """,
            ("unknown-role-code", "missing-role", expires_at),
        )

    with pytest.raises(RegistrationError, match="Registration role 'missing-role' does not exist"):
        await service.register_user(
            username="rolled-back-user",
            email="rolled-back-user@example.com",
            password="Strong!Pass9",
            registration_code="unknown-role-code",
        )

    with sqlite3.connect(db_path) as conn:
        user_count = conn.execute(
            "SELECT COUNT(*) FROM users WHERE username = ?",
            ("rolled-back-user",),
        ).fetchone()[0]
        times_used = conn.execute(
            "SELECT times_used FROM registration_codes WHERE code = ?",
            ("unknown-role-code",),
        ).fetchone()[0]
        membership_count = conn.execute("SELECT COUNT(*) FROM user_roles").fetchone()[0]

    assert user_count == 0
    assert times_used == 0
    assert membership_count == 0
