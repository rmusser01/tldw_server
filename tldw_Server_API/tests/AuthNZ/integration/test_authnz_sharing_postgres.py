"""PostgreSQL integration coverage for canonical sharing storage."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import asyncpg
import pytest
from fastapi import HTTPException, Request

from tldw_Server_API.app.api.v1.schemas.sharing_schemas import (
    AuditEventResponse,
    ShareResponse,
    ShareWorkspaceRequest,
    TokenResponse,
)

pytestmark = pytest.mark.integration


async def _create_test_user(pool: Any, *, username: str, email: str) -> int:
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

    created = await UsersDB(pool).create_user(
        username=username,
        email=email,
        password_hash="test-hash",
    )
    return int(created["id"])


@asynccontextmanager
async def _unmanaged_connection(pool: Any) -> AsyncIterator[asyncpg.Connection]:
    connection = await asyncpg.connect(pool.settings.DATABASE_URL)
    try:
        yield connection
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_postgres_authoritative_share_queries_follow_current_membership_and_scope(
    isolated_test_environment,
) -> None:
    _client, _db_name = isolated_test_environment

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.initialize import setup_database
    from tldw_Server_API.app.core.AuthNZ.repos.shared_workspace_repo import (
        SharedWorkspaceRepo,
    )

    pool = await get_db_pool()
    assert await setup_database() is True
    owner_id = await _create_test_user(
        pool,
        username="sharing-authority-owner",
        email="sharing-authority-owner@example.test",
    )
    recipient_id = await _create_test_user(
        pool,
        username="sharing-authority-recipient",
        email="sharing-authority-recipient@example.test",
    )
    organization = await pool.fetchone(
        """
        INSERT INTO organizations (name, slug, owner_user_id, is_active)
        VALUES (?, ?, ?, TRUE)
        RETURNING id
        """,
        ("Sharing Authority Org", "sharing-authority-org", owner_id),
    )
    org_id = int(organization["id"])
    team = await pool.fetchone(
        """
        INSERT INTO teams (org_id, name, slug, is_active)
        VALUES (?, ?, ?, TRUE)
        RETURNING id
        """,
        (org_id, "Sharing Authority Team", "sharing-authority-team"),
    )
    team_id = int(team["id"])
    await pool.execute(
        "INSERT INTO org_members (org_id, user_id, status) VALUES (?, ?, 'active')",
        (org_id, recipient_id),
    )
    await pool.execute(
        "INSERT INTO team_members (team_id, user_id, status) VALUES (?, ?, 'active')",
        (team_id, recipient_id),
    )

    repo = SharedWorkspaceRepo(pool)
    team_share = await repo.create_share(
        workspace_id="pg-authoritative-team",
        owner_user_id=owner_id,
        share_scope_type="team",
        share_scope_id=team_id,
        created_by=owner_id,
    )
    org_share = await repo.create_share(
        workspace_id="pg-authoritative-org",
        owner_user_id=owner_id,
        share_scope_type="org",
        share_scope_id=org_id,
        created_by=owner_id,
    )
    owned_share = await repo.create_share(
        workspace_id="pg-authoritative-owned",
        owner_user_id=recipient_id,
        share_scope_type="team",
        share_scope_id=team_id,
        created_by=recipient_id,
    )

    assert (await repo.get_active_share_for_user(team_share["id"], recipient_id))["id"] == team_share["id"]
    assert (await repo.get_active_share_for_user(org_share["id"], owner_id))["id"] == org_share["id"]
    assert {row["id"] for row in await repo.list_active_shares_for_user(recipient_id)} == {
        team_share["id"],
        org_share["id"],
    }
    assert owned_share["id"] not in {
        row["id"] for row in await repo.list_active_shares_for_user(recipient_id)
    }

    await pool.execute(
        "UPDATE team_members SET status = 'suspended' WHERE team_id = ? AND user_id = ?",
        (team_id, recipient_id),
    )
    assert await repo.get_active_share_for_user(team_share["id"], recipient_id) is None
    assert [row["id"] for row in await repo.list_active_shares_for_user(recipient_id)] == [
        org_share["id"]
    ]

    await pool.execute(
        "DELETE FROM org_members WHERE org_id = ? AND user_id = ?",
        (org_id, recipient_id),
    )
    assert await repo.list_active_shares_for_user(recipient_id) == []

    await pool.execute("UPDATE teams SET is_active = FALSE WHERE id = ?", (team_id,))
    assert await repo.get_active_share_for_user(team_share["id"], owner_id) is None
    await pool.execute("UPDATE organizations SET is_active = FALSE WHERE id = ?", (org_id,))
    assert await repo.get_active_share_for_user(org_share["id"], owner_id) is None


@pytest.mark.asyncio
async def test_postgres_runtime_startup_bootstraps_sharing_tables(
    isolated_test_environment,
) -> None:
    _client, _db_name = isolated_test_environment

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.repos.shared_workspace_repo import (
        SharedWorkspaceRepo,
    )
    from tldw_Server_API.app.services.startup_auth import _ensure_pg_extras

    pool = await get_db_pool()
    async with _unmanaged_connection(pool) as connection:
        for table in (
            "sharing_config",
            "share_audit_log",
            "share_tokens",
            "shared_workspaces",
        ):
            await connection.execute(f"DROP TABLE IF EXISTS {table} CASCADE")

    await _ensure_pg_extras(pool)

    await SharedWorkspaceRepo(pool).ensure_tables()


@pytest.mark.asyncio
async def test_postgres_bootstrap_supports_sharing_repository_and_preserves_rows(
    isolated_test_environment,
) -> None:
    _client, _db_name = isolated_test_environment

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.initialize import setup_database
    from tldw_Server_API.app.core.AuthNZ.repos.shared_workspace_repo import (
        SharedWorkspaceRepo,
    )

    pool = await get_db_pool()
    async with _unmanaged_connection(pool) as connection:
        await connection.execute("DROP TABLE IF EXISTS sharing_config CASCADE")
        await connection.execute("DROP TABLE IF EXISTS share_audit_log CASCADE")
        await connection.execute("DROP TABLE IF EXISTS share_tokens CASCADE")
        await connection.execute("DROP TABLE IF EXISTS shared_workspaces CASCADE")

    assert await setup_database() is True

    repo = SharedWorkspaceRepo(pool)
    await repo.ensure_tables()
    owner_id = await _create_test_user(
        pool,
        username="sharing-pg-owner",
        email="sharing-pg-owner@example.test",
    )

    share = await repo.create_share(
        workspace_id="pg-workspace",
        owner_user_id=owner_id,
        share_scope_type="team",
        share_scope_id=101,
        created_by=owner_id,
    )
    share_id = int(share["id"])
    assert share["allow_clone"] is True
    assert isinstance(share["created_at"], str)
    ShareResponse.model_validate(share)
    assert len(await repo.list_shares_for_scope("team", 101)) == 1

    updated = await repo.update_share(
        share_id,
        access_level="full_edit",
        allow_clone=False,
    )
    assert updated is not None
    assert updated["access_level"] == "full_edit"
    assert updated["allow_clone"] is False

    token = await repo.create_token(
        token_hash="a" * 64,
        token_prefix="pg-token",
        resource_type="prototype_workspace",
        resource_id="prototype-1",
        owner_user_id=owner_id,
    )
    token_id = int(token["id"])
    assert token["resource_type"] == "prototype_workspace"
    assert isinstance(token["created_at"], str)
    TokenResponse.model_validate(token)
    assert await repo.claim_token_use(token_id) is True
    await repo.release_token_use(token_id)

    await repo.log_audit_event(
        event_type="share.created",
        resource_type="workspace",
        resource_id="pg-workspace",
        owner_user_id=owner_id,
        share_id=share_id,
    )
    audit_events = await repo.list_audit_events()
    assert len(audit_events) == 1
    assert isinstance(audit_events[0]["created_at"], str)
    AuditEventResponse.model_validate(audit_events[0])
    assert len(await repo.list_audit_events(owner_user_id=owner_id)) == 1
    assert await repo.count_audit_events(owner_user_id=owner_id) == 1
    await repo.set_config("sharing.enabled", "true")
    await repo.set_config("sharing.enabled", "false")
    assert await repo.get_config() == {"sharing.enabled": "false"}
    config_count = await pool.fetchone(
        """
        SELECT COUNT(*) AS count
        FROM sharing_config
        WHERE scope_type = ? AND scope_id IS NULL AND config_key = ?
        """,
        ("global", "sharing.enabled"),
    )
    assert config_count == {"count": 1}

    assert await setup_database() is True
    assert (await repo.get_share(share_id))["workspace_id"] == "pg-workspace"
    assert (await repo.get_token(token_id))["resource_id"] == "prototype-1"

    assert await repo.revoke_token(token_id) is True
    assert await repo.revoke_share(share_id) is True


@pytest.mark.asyncio
async def test_postgres_bootstrap_upgrades_legacy_token_constraint_without_data_loss(
    isolated_test_environment,
) -> None:
    _client, _db_name = isolated_test_environment

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.initialize import setup_database
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import ensure_sharing_tables_pg
    from tldw_Server_API.app.core.AuthNZ.repos.shared_workspace_repo import (
        SharedWorkspaceRepo,
    )

    pool = await get_db_pool()
    assert await ensure_sharing_tables_pg(pool) is True
    owner_id = await _create_test_user(
        pool,
        username="sharing-pg-legacy",
        email="sharing-pg-legacy@example.com",
    )
    repo = SharedWorkspaceRepo(pool)
    share = await repo.create_share(
        workspace_id="legacy-workspace",
        owner_user_id=owner_id,
        share_scope_type="team",
        share_scope_id=202,
        created_by=owner_id,
    )
    token = await repo.create_token(
        token_hash="b" * 64,
        token_prefix="legacy-token",
        resource_type="workspace",
        resource_id="legacy-workspace",
        owner_user_id=owner_id,
    )

    async with _unmanaged_connection(pool) as connection:
        await connection.execute(
            "ALTER TABLE share_tokens DROP CONSTRAINT ck_share_tokens_resource_type"
        )
        await connection.execute(
            """
            ALTER TABLE share_tokens
            ADD CONSTRAINT share_tokens_resource_type_check
            CHECK (resource_type IN ('chatbook', 'workspace'))
            """
        )
        await connection.execute("DROP INDEX uq_sharing_config_global_key")

    assert await setup_database() is True
    assert (await repo.get_share(int(share["id"])))["workspace_id"] == "legacy-workspace"
    assert (await repo.get_token(int(token["id"])))["resource_id"] == "legacy-workspace"
    constraint = await pool.fetchone(
        """
        SELECT pg_get_constraintdef(c.oid) AS definition
        FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        WHERE t.relname = 'share_tokens'
          AND c.conname = 'ck_share_tokens_resource_type'
        """
    )
    assert "prototype_workspace" in constraint["definition"]
    assert await pool.fetchone(
        """
        SELECT conname
        FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        WHERE t.relname = 'share_tokens'
          AND c.conname = 'share_tokens_resource_type_check'
        """
    ) is None
    upgraded_token = await repo.create_token(
        token_hash="c" * 64,
        token_prefix="upgraded-token",
        resource_type="prototype_workspace",
        resource_id="prototype-upgraded",
        owner_user_id=owner_id,
    )
    assert upgraded_token["resource_type"] == "prototype_workspace"
    assert await pool.fetchone(
        "SELECT indexname FROM pg_indexes WHERE indexname = ?",
        ("uq_sharing_config_global_key",),
    ) == {"indexname": "uq_sharing_config_global_key"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("custom_constraint_sql", "canonical_constraint_sql", "literal_fragment"),
    [
        (
            """
            ALTER TABLE share_tokens
            ADD CONSTRAINT custom_resource_type_guard
            CHECK (resource_type IN ('CHATBOOK', 'WORKSPACE'))
            """,
            """
            ALTER TABLE share_tokens
            ADD CONSTRAINT ck_share_tokens_resource_type
            CHECK (
                resource_type IN (
                    'CHATBOOK', 'WORKSPACE', 'PROTOTYPE_WORKSPACE'
                )
            )
            """,
            "CHATBOOK",
        ),
        (
            """
            ALTER TABLE share_tokens
            ADD CONSTRAINT custom_resource_type_guard
            CHECK (resource_type IN ('chat book', 'workspace'))
            """,
            """
            ALTER TABLE share_tokens
            ADD CONSTRAINT ck_share_tokens_resource_type
            CHECK (
                resource_type IN (
                    'chat book', 'workspace', 'prototype_workspace'
                )
            )
            """,
            "chat book",
        ),
    ],
    ids=["uppercase-literals", "literal-whitespace"],
)
async def test_postgres_schema_contract_preserves_and_rejects_literal_drift(
    isolated_test_environment,
    custom_constraint_sql: str,
    canonical_constraint_sql: str,
    literal_fragment: str,
) -> None:
    _client, _db_name = isolated_test_environment

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
        ensure_sharing_tables_pg,
        sharing_schema_issues_pg,
    )

    pool = await get_db_pool()
    assert await ensure_sharing_tables_pg(pool) is True
    async with _unmanaged_connection(pool) as connection:
        await connection.execute(
            "ALTER TABLE share_tokens DROP CONSTRAINT ck_share_tokens_resource_type"
        )
        await connection.execute(custom_constraint_sql)
        await connection.execute(canonical_constraint_sql)

    assert await ensure_sharing_tables_pg(pool) is False
    rows = await pool.fetchall(
        """
        SELECT c.conname AS constraint_name,
               pg_get_constraintdef(c.oid) AS definition
        FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        WHERE t.relname = 'share_tokens'
          AND c.conname IN (
              'custom_resource_type_guard',
              'ck_share_tokens_resource_type'
          )
        ORDER BY c.conname
        """,
        (),
    )
    assert {row["constraint_name"] for row in rows} == {
        "ck_share_tokens_resource_type",
        "custom_resource_type_guard",
    }
    assert all(literal_fragment in row["definition"] for row in rows)

    issues = await sharing_schema_issues_pg(pool)
    assert "invalid constraint share_tokens.ck_share_tokens_resource_type" in issues
    assert "invalid constraint share_tokens.custom_resource_type_guard" in issues


@pytest.mark.asyncio
async def test_postgres_schema_contract_rejects_incompatible_drift(
    isolated_test_environment,
) -> None:
    _client, _db_name = isolated_test_environment

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
        ensure_sharing_tables_pg,
        sharing_schema_issues_pg,
    )
    from tldw_Server_API.app.core.AuthNZ.repos.shared_workspace_repo import (
        SharedWorkspaceRepo,
    )

    pool = await get_db_pool()
    assert await ensure_sharing_tables_pg(pool) is True

    async with _unmanaged_connection(pool) as connection:
        await connection.execute(
            "ALTER TABLE share_tokens ALTER COLUMN allow_clone DROP DEFAULT"
        )
        await connection.execute(
            """
            ALTER TABLE share_tokens
            ALTER COLUMN allow_clone TYPE TEXT USING allow_clone::TEXT
            """
        )
        await connection.execute(
            "ALTER TABLE share_tokens ALTER COLUMN allow_clone SET DEFAULT 'true'"
        )
        await connection.execute(
            "ALTER TABLE share_audit_log ALTER COLUMN ip_address SET NOT NULL"
        )
        await connection.execute(
            "ALTER TABLE shared_workspaces ALTER COLUMN allow_clone DROP DEFAULT"
        )
        await connection.execute(
            "ALTER TABLE share_tokens DROP CONSTRAINT ck_share_tokens_resource_type"
        )
        await connection.execute(
            """
            ALTER TABLE share_tokens
            ADD CONSTRAINT ck_share_tokens_resource_type
            CHECK (
                resource_type IN ('chatbook', 'workspace', 'prototype_workspace')
                OR TRUE
            )
            """
        )
        await connection.execute(
            """
            ALTER TABLE share_tokens
            ADD CONSTRAINT share_tokens_resource_guard
            CHECK (
                resource_type IN ('chatbook', 'workspace')
                AND length(resource_id) > 0
            )
            """
        )
        await connection.execute(
            """
            ALTER TABLE shared_workspaces
            DROP CONSTRAINT ck_shared_workspaces_access_level
            """
        )
        await connection.execute(
            """
            ALTER TABLE shared_workspaces
            ADD CONSTRAINT ck_shared_workspaces_access_level
            CHECK (access_level IN ('view_chat', 'view_chat_add', 'full_edit'))
            NOT VALID
            """
        )
        await connection.execute("DROP INDEX uq_sharing_config_global_key")
        await connection.execute(
            """
            CREATE UNIQUE INDEX uq_sharing_config_global_key
            ON sharing_config(config_value)
            """
        )

    assert await ensure_sharing_tables_pg(pool) is False
    assert await pool.fetchone(
        """
        SELECT conname
        FROM pg_constraint c
        JOIN pg_class t ON t.oid = c.conrelid
        WHERE t.relname = 'share_tokens'
          AND c.conname = 'share_tokens_resource_guard'
        """
    ) == {"conname": "share_tokens_resource_guard"}

    issues = await sharing_schema_issues_pg(pool)
    assert "invalid column share_tokens.allow_clone" in issues
    assert "invalid column share_audit_log.ip_address" in issues
    assert "invalid default shared_workspaces.allow_clone" in issues
    assert (
        "invalid constraint shared_workspaces.ck_shared_workspaces_access_level"
        in issues
    )
    assert "invalid constraint share_tokens.ck_share_tokens_resource_type" in issues
    assert "invalid constraint share_tokens.share_tokens_resource_guard" in issues
    assert "missing or invalid index sharing_config.uq_sharing_config_global_key" in issues

    with pytest.raises(RuntimeError, match="schema contract mismatch"):
        await SharedWorkspaceRepo(pool).ensure_tables()


@pytest.mark.asyncio
async def test_postgres_share_endpoint_maps_duplicate_to_conflict(
    isolated_test_environment,
    monkeypatch,
) -> None:
    _client, _db_name = isolated_test_environment

    from tldw_Server_API.app.api.v1.endpoints import sharing
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.initialize import setup_database
    from tldw_Server_API.app.core.AuthNZ.repos.shared_workspace_repo import (
        SharedWorkspaceRepo,
    )

    pool = await get_db_pool()
    assert await setup_database() is True
    owner_id = await _create_test_user(
        pool,
        username="sharing-pg-endpoint",
        email="sharing-pg-endpoint@example.com",
    )
    repo = SharedWorkspaceRepo(pool)

    async def _allow(*args, **kwargs) -> None:
        return None

    class _Audit:
        async def log(self, *args, **kwargs) -> None:
            return None

    monkeypatch.setattr(sharing, "_get_repo", lambda: repo)
    monkeypatch.setattr(sharing, "_get_audit_service", lambda: _Audit())
    monkeypatch.setattr(sharing, "_verify_workspace_ownership", _allow)
    monkeypatch.setattr(sharing, "_validate_share_target_scope", _allow)

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/sharing/workspaces/endpoint-workspace/share",
            "headers": [],
            "client": ("127.0.0.1", 1234),
            "server": ("testserver", 80),
            "scheme": "http",
            "query_string": b"",
        }
    )
    body = ShareWorkspaceRequest(
        share_scope_type="team",
        share_scope_id=303,
    )
    user = type("EndpointUser", (), {"id": owner_id})()

    created = await sharing.share_workspace(
        "endpoint-workspace",
        body,
        request,
        user,
    )
    assert isinstance(created.created_at, str)

    with pytest.raises(HTTPException) as exc_info:
        await sharing.share_workspace(
            "endpoint-workspace",
            body,
            request,
            user,
        )
    assert exc_info.value.status_code == 409
