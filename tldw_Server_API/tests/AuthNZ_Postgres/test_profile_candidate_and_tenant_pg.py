from __future__ import annotations

import uuid

import asyncpg
import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.admin import admin_tenant_provisioning
from tldw_Server_API.app.api.v1.endpoints.admin.admin_tenant_provisioning import (
    TenantProvisionRequest,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.profile_version import (
    VersionedUserWriteGateway,
)
from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

pytestmark = [pytest.mark.integration, pytest.mark.postgres]


def _principal(user_id: int = 1) -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=user_id,
        username="admin",
        roles=["admin"],
        is_admin=True,
    )


async def _execute_intentionally_invalid_candidate_ddl(conn, statement: str) -> None:
    """Bypass the users firewall only to construct invalid candidate metadata."""
    await asyncpg.Connection.execute(conn, statement)


@pytest.mark.asyncio
async def test_postgres_tenant_provisioning_uses_real_defaults_and_rolls_back(
    test_db_pool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suffix = uuid.uuid4().hex[:10]
    org_name = f"Tenant {suffix}"

    async def _get_pool():
        return test_db_pool

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.database.get_db_pool",
        _get_pool,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.password_service.get_password_service",
        lambda: type("PasswordService", (), {"hash_password": lambda _self, _value: "hash"})(),
    )

    async with test_db_pool.transaction() as conn:
        actor_result = await VersionedUserWriteGateway("postgres").insert_user(
            conn,
            values={
                "uuid": str(uuid.uuid4()),
                "username": f"admin_{suffix}",
                "email": f"admin_{suffix}@example.com",
                "password_hash": "hash",
                "role": "admin",
                "is_active": True,
                "is_verified": True,
            },
        )
        permission_id = await conn.fetchval(
            "INSERT INTO public.permissions (name, description, category) "
            "VALUES ('system.configure', 'Configure system', 'system') "
            "ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id"
        )
        actor_user_id = actor_result.affected_user_ids[0]
        await conn.execute(
            "INSERT INTO public.user_permissions (user_id, permission_id, granted) "
            "VALUES ($1, $2, TRUE) "
            "ON CONFLICT (user_id, permission_id) DO UPDATE SET granted = TRUE",
            actor_user_id,
            permission_id,
        )

    result = await admin_tenant_provisioning.provision_tenant(
        TenantProvisionRequest(
            username=f"tenant_{suffix}",
            email=f"tenant_{suffix}@example.com",
            password="securepass123",
            org_name=org_name,
        ),
        _principal(actor_user_id),
    )
    assert result.user_id > 0
    assert result.org_id > 0

    rolled_back_username = f"rollback_{suffix}"
    with pytest.raises(HTTPException) as raised:
        await admin_tenant_provisioning.provision_tenant(
            TenantProvisionRequest(
                username=rolled_back_username,
                email=f"{rolled_back_username}@example.com",
                password="securepass123",
                org_name=org_name,
            ),
            _principal(actor_user_id),
        )

    assert raised.value.status_code == 500
    assert await test_db_pool.fetchval(
        "SELECT COUNT(*) FROM public.users WHERE username = $1",
        rolled_back_username,
    ) == 0


@pytest.mark.asyncio
async def test_postgres_candidate_validation_rejects_shadow_fk_and_missing_id_default(
    test_db_pool,
) -> None:
    async with test_db_pool.pool.acquire() as conn:
        await UsersDB._validate_profile_candidate_tables(  # noqa: SLF001
            conn,
            is_postgres=True,
        )

        transaction = conn.transaction()
        await transaction.start()
        try:
            constraint_name = await conn.fetchval(
                """
                SELECT tc.constraint_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                 AND tc.constraint_schema = kcu.constraint_schema
                WHERE tc.table_schema = 'public'
                  AND tc.table_name = 'org_members'
                  AND tc.constraint_type = 'FOREIGN KEY'
                  AND kcu.column_name = 'user_id'
                """
            )
            assert constraint_name
            await conn.execute("CREATE SCHEMA profile_shadow")
            await _execute_intentionally_invalid_candidate_ddl(
                conn,
                "CREATE TABLE profile_shadow.users (id INTEGER PRIMARY KEY)"
            )
            await conn.execute(
                f'ALTER TABLE public.org_members DROP CONSTRAINT "{constraint_name}"'
            )
            await _execute_intentionally_invalid_candidate_ddl(
                conn,
                "ALTER TABLE public.org_members ADD CONSTRAINT "
                "org_members_shadow_user_fk FOREIGN KEY (user_id) "
                "REFERENCES profile_shadow.users(id) ON DELETE CASCADE"
            )

            with pytest.raises(Exception, match="candidate schema validation failed"):
                await UsersDB._validate_profile_candidate_tables(  # noqa: SLF001
                    conn,
                    is_postgres=True,
                )
        finally:
            await transaction.rollback()

        transaction = conn.transaction()
        await transaction.start()
        try:
            await conn.execute(
                "ALTER TABLE public.organizations ALTER COLUMN id DROP DEFAULT"
            )
            with pytest.raises(Exception, match="candidate schema validation failed"):
                await UsersDB._validate_profile_candidate_tables(  # noqa: SLF001
                    conn,
                    is_postgres=True,
                )
        finally:
            await transaction.rollback()


@pytest.mark.asyncio
async def test_postgres_core_startup_repairs_legacy_candidate_timestamps(
    test_db_pool,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
        ensure_authnz_core_tables_pg,
    )

    await test_db_pool.execute(
        "ALTER TABLE public.teams ALTER COLUMN updated_at "
        "TYPE TIMESTAMP WITHOUT TIME ZONE USING updated_at AT TIME ZONE 'UTC'"
    )
    assert await ensure_authnz_core_tables_pg(test_db_pool) is True
    data_type = await test_db_pool.fetchval(
        "SELECT data_type FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = 'teams' "
        "AND column_name = 'updated_at'"
    )
    assert data_type == "timestamp with time zone"
