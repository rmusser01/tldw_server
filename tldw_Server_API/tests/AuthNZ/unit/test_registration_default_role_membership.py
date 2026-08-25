from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import aiosqlite
import pytest

from tldw_Server_API.app.core.AuthNZ.exceptions import RegistrationError
from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    AnchorOwnership,
    TrustedMembershipReason,
    TrustedMembershipWriteContext,
)
from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import (
    AuthnzOrgsTeamsRepo,
)
from tldw_Server_API.app.services.registration_service import RegistrationService


class _PasswordServiceStub:
    def validate_password_strength(self, password: str, username: str | None = None) -> None:  # noqa: ARG002
        return None

    def hash_password(self, password: str) -> str:
        return f"hash-{password}"


class _SQLitePool:
    pool = None
    backend = "sqlite"

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.transaction_connections: list[Any] = []

    @asynccontextmanager
    async def transaction(
        self,
        *,
        acquire_timeout_seconds: float | None = None,
    ) -> AsyncIterator[Any]:
        assert acquire_timeout_seconds is not None
        from tldw_Server_API.app.core.AuthNZ.database import (
            _GuardedSQLiteConnection,
        )

        conn = await aiosqlite.connect(self.db_path)
        await conn.execute("PRAGMA foreign_keys = ON")
        await conn.execute("BEGIN IMMEDIATE")
        try:
            guarded = _GuardedSQLiteConnection(conn)
            self.transaction_connections.append(guarded)
            yield guarded
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
                is_superuser INTEGER NOT NULL DEFAULT 0,
                created_by INTEGER,
                storage_quota_mb INTEGER NOT NULL,
                profile_version TEXT NOT NULL
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
            CREATE TABLE organizations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                owner_user_id INTEGER,
                is_active INTEGER NOT NULL DEFAULT 1,
                FOREIGN KEY (owner_user_id) REFERENCES users(id) ON DELETE SET NULL
            );
            CREATE TABLE org_members (
                org_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL DEFAULT 'member',
                status TEXT NOT NULL DEFAULT 'active',
                PRIMARY KEY (org_id, user_id),
                FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE TABLE teams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                org_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                slug TEXT,
                description TEXT,
                metadata TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                UNIQUE (org_id, name),
                FOREIGN KEY (org_id) REFERENCES organizations(id) ON DELETE CASCADE
            );
            CREATE TABLE team_members (
                team_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL DEFAULT 'member',
                status TEXT NOT NULL DEFAULT 'active',
                PRIMARY KEY (team_id, user_id),
                FOREIGN KEY (team_id) REFERENCES teams(id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE TABLE user_config_overrides (
                user_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, key)
            );
            CREATE TABLE org_config_overrides (
                org_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (org_id, key)
            );
            CREATE TABLE team_config_overrides (
                team_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (team_id, key)
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
            """
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


@pytest.mark.asyncio
async def test_org_scoped_registration_uses_registration_writer_on_caller_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, db_path = _make_service(tmp_path)
    expires_at = (datetime.utcnow() + timedelta(days=1)).isoformat()
    with sqlite3.connect(db_path) as conn:
        org_id = conn.execute(
            "INSERT INTO organizations (name) VALUES (?)",
            ("Registration org",),
        ).lastrowid
        team_id = conn.execute(
            "INSERT INTO teams (org_id, name) VALUES (?, ?)",
            (org_id, "Registration team"),
        ).lastrowid
        conn.execute(
            """
            INSERT INTO registration_codes (
                code, role_to_grant, times_used, max_uses, expires_at, is_active,
                org_id, org_role, team_id
            ) VALUES (?, ?, 0, 1, ?, 1, ?, ?, ?)
            """,
            (
                "org-registration-code",
                "user",
                expires_at,
                org_id,
                "admin",
                team_id,
            ),
        )

    observed: list[dict[str, Any]] = []
    original = AuthnzOrgsTeamsRepo.provision_org_membership_on_connection

    async def _record(self, **kwargs):
        observed.append(kwargs)
        return await original(
            self,
            **kwargs,
        )

    monkeypatch.setattr(
        AuthnzOrgsTeamsRepo,
        "provision_org_membership_on_connection",
        _record,
    )

    payload = await service.register_user(
        username="org-registration-user",
        email="org-registration-user@example.com",
        password="Strong!Pass9",
        registration_code="org-registration-code",
    )

    assert len(observed) == 1
    assert observed[0]["conn"] is service.db_pool.transaction_connections[0]
    assert observed[0]["context"] == TrustedMembershipWriteContext(
        trusted_reason=TrustedMembershipReason.REGISTRATION,
    )
    assert observed[0]["anchor_ownership"] is AnchorOwnership.WRITER_OWNS_ANCHOR
    assert observed[0]["org_role"] == "admin"
    assert observed[0]["team_id"] == team_id
    assert observed[0]["team_role"] == "member"
    assert observed[0]["team_failure_is_best_effort"] is False
    with sqlite3.connect(db_path) as conn:
        org_role = conn.execute(
            "SELECT role FROM org_members WHERE org_id = ? AND user_id = ?",
            (org_id, payload["user_id"]),
        ).fetchone()[0]
        team_role = conn.execute(
            "SELECT role FROM team_members WHERE team_id = ? AND user_id = ?",
            (team_id, payload["user_id"]),
        ).fetchone()[0]
    assert (org_role, team_role) == ("admin", "member")
