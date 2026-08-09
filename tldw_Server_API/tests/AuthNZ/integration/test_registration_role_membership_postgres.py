from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any

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


class _PasswordService:
    def validate_password_strength(self, _password: str, _username: str) -> None:
        return None

    def hash_password(self, _password: str) -> str:
        return "hashed-password"


class _ObservedPool:
    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self.pool = pool.pool
        self.transaction_connections: list[Any] = []

    @asynccontextmanager
    async def transaction(self):
        async with self._pool.transaction() as conn:
            self.transaction_connections.append(conn)
            yield conn

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pool, name)


@pytest.mark.asyncio
async def test_unknown_registration_code_role_rolls_back_every_postgres_write(
    test_db_pool,
    tmp_path,
    monkeypatch,
) -> None:
    async with test_db_pool.transaction() as conn:
        await conn.execute(
            """
            INSERT INTO registration_codes
                (code, max_uses, times_used, expires_at, role_to_grant, is_active)
            VALUES ($1, 1, 0, $2, $3, TRUE)
            """,
            "unknown-role-code",
            datetime.utcnow() + timedelta(hours=1),
            "missing-role",
        )

    settings = _registration_settings(tmp_path, org_scoped=False)
    service = RegistrationService(
        db_pool=_ObservedPool(test_db_pool),
        password_service=_PasswordService(),
        settings=settings,
    )
    monkeypatch.setattr(service, "_create_user_directories", lambda _user_id: True)

    with pytest.raises(RegistrationError, match="missing-role"):
        await service.register_user(
            username="postgres-rollback-user",
            email="postgres-rollback@example.com",
            password="StrongPass123!",
            registration_code="unknown-role-code",
        )

    user_count = await test_db_pool.fetchval(
        "SELECT COUNT(*) FROM users WHERE username = $1",
        "postgres-rollback-user",
    )
    membership_count = await test_db_pool.fetchval(
        """
        SELECT COUNT(*)
        FROM user_roles ur
        JOIN users u ON u.id = ur.user_id
        WHERE u.username = $1
        """,
        "postgres-rollback-user",
    )
    times_used = await test_db_pool.fetchval(
        "SELECT times_used FROM registration_codes WHERE code = $1",
        "unknown-role-code",
    )

    assert (user_count, membership_count, times_used) == (0, 0, 0)


def _registration_settings(tmp_path, *, org_scoped: bool) -> SimpleNamespace:
    return SimpleNamespace(
        ENABLE_REGISTRATION=True,
        REQUIRE_REGISTRATION_CODE=False,
        DEFAULT_USER_ROLE="user",
        DEFAULT_STORAGE_QUOTA_MB=5120,
        ENABLE_ORG_SCOPED_REGISTRATION_CODES=org_scoped,
        USER_DATA_BASE_PATH=str(tmp_path / "users"),
        CHROMADB_BASE_PATH=None,
    )


@pytest.mark.asyncio
async def test_org_scoped_registration_uses_registration_writer_on_same_pg_connection(
    test_db_pool,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = AuthnzOrgsTeamsRepo(test_db_pool)
    organization = await repo.create_organization(name="PG registration organization")
    org_id = int(organization["id"])
    team = await repo.create_team(org_id=org_id, name="PG registration team")
    team_id = int(team["id"])
    async with test_db_pool.transaction() as conn:
        await conn.execute(
            """
            INSERT INTO registration_codes
                (code, max_uses, times_used, expires_at, role_to_grant, is_active,
                 org_id, org_role, team_id)
            VALUES ($1, 1, 0, $2, $3, TRUE, $4, $5, $6)
            """,
            "pg-org-registration-code",
            datetime.utcnow() + timedelta(hours=1),
            "user",
            org_id,
            "admin",
            team_id,
        )

    observed_pool = _ObservedPool(test_db_pool)
    service = RegistrationService(
        db_pool=observed_pool,
        password_service=_PasswordService(),
        settings=_registration_settings(tmp_path, org_scoped=True),
    )
    monkeypatch.setattr(service, "_create_user_directories", lambda _user_id: True)
    observed: list[dict[str, Any]] = []
    original = AuthnzOrgsTeamsRepo.provision_org_membership_on_connection

    async def _record(self, **kwargs):
        observed.append(kwargs)
        return await original(self, **kwargs)

    monkeypatch.setattr(
        AuthnzOrgsTeamsRepo,
        "provision_org_membership_on_connection",
        _record,
    )

    payload = await service.register_user(
        username="pg-org-registration-user",
        email="pg-org-registration-user@example.com",
        password="StrongPass123!",
        registration_code="pg-org-registration-code",
    )

    assert len(observed) == 1
    assert observed[0]["conn"] is observed_pool.transaction_connections[0]
    assert observed[0]["context"] == TrustedMembershipWriteContext(
        trusted_reason=TrustedMembershipReason.REGISTRATION,
    )
    assert observed[0]["anchor_ownership"] is AnchorOwnership.WRITER_OWNS_ANCHOR
    assert observed[0]["team_id"] == team_id
    assert observed[0]["team_role"] == "member"
    assert observed[0]["team_failure_is_best_effort"] is False
    assert await test_db_pool.fetchval(
        "SELECT role FROM org_members WHERE org_id = $1 AND user_id = $2",
        org_id,
        payload["user_id"],
    ) == "admin"
    assert await test_db_pool.fetchval(
        "SELECT role FROM team_members WHERE team_id = $1 AND user_id = $2",
        team_id,
        payload["user_id"],
    ) == "member"
