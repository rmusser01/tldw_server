from __future__ import annotations

import os
from datetime import datetime, timedelta
from types import SimpleNamespace

import asyncpg
import pytest

from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.exceptions import RegistrationError
from tldw_Server_API.app.services.registration_service import RegistrationService


class _PasswordService:
    def validate_password_strength(self, _password: str, _username: str) -> None:
        return None

    def hash_password(self, _password: str) -> str:
        return "hashed-password"


@pytest.mark.asyncio
async def test_unknown_registration_code_role_rolls_back_every_postgres_write(
    isolated_test_environment,
    tmp_path,
    monkeypatch,
) -> None:
    _client, _db_name = isolated_test_environment
    pool = await asyncpg.create_pool(os.environ["DATABASE_URL"], min_size=1, max_size=1)
    try:
        async with pool.acquire() as conn:
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

        settings = SimpleNamespace(
            ENABLE_REGISTRATION=True,
            REQUIRE_REGISTRATION_CODE=False,
            DEFAULT_USER_ROLE="user",
            DEFAULT_STORAGE_QUOTA_MB=5120,
            ENABLE_ORG_SCOPED_REGISTRATION_CODES=False,
            USER_DATA_BASE_PATH=str(tmp_path / "users"),
            CHROMADB_BASE_PATH=None,
        )
        service = RegistrationService(
            # The managed pool: profile_user_write_guard only honours its connections.
            db_pool=await get_db_pool(),
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

        async with pool.acquire() as conn:
            user_count = await conn.fetchval(
                "SELECT COUNT(*) FROM users WHERE username = $1",
                "postgres-rollback-user",
            )
            membership_count = await conn.fetchval(
                """
                SELECT COUNT(*)
                FROM user_roles ur
                JOIN users u ON u.id = ur.user_id
                WHERE u.username = $1
                """,
                "postgres-rollback-user",
            )
            times_used = await conn.fetchval(
                "SELECT times_used FROM registration_codes WHERE code = $1",
                "unknown-role-code",
            )

        assert (user_count, membership_count, times_used) == (0, 0, 0)
    finally:
        await pool.close()
