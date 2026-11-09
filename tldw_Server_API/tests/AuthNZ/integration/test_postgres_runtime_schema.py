"""Serving AuthNZ roles must read migrated PostgreSQL schemas without owning them."""

from dataclasses import replace

import pytest
from psycopg import sql

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.tests.AuthNZ.integration.test_database_runtime_selection import (
    _run_runtime,
    _runtime_env,
)
from tldw_Server_API.tests.helpers.authnz_seed import unmanaged_authnz_pg_connection

pytestmark = [pytest.mark.integration, pytest.mark.postgres]

_OWNER_SCRIPT = """
import asyncio
import json
from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
from tldw_Server_API.app.core.AuthNZ.initialize import setup_database
from tldw_Server_API.app.core.AuthNZ.db_config import get_configured_user_database
from tldw_Server_API.app.core.AuthNZ.password_service import get_password_service
from tldw_Server_API.app.core.DB_Management.Users_DB import get_users_db

async def main():
    try:
        assert await setup_database()
        users = await get_users_db()
        rbac = get_configured_user_database()
        result = {}
        for role in ('admin', 'user'):
            row = await users.create_user(
                username='runtime-' + role,
                email=role + '@runtime.example.com',
                password_hash=get_password_service().hash_password('X!a7-fixture-account'),
                role=role, is_verified=True, is_superuser=role == 'admin',
            )
            assert rbac.assign_role(row['id'], role)
            result[role] = row['id']
    finally:
        await reset_db_pool()
    print('RUNTIME_RESULT=' + json.dumps(result))

asyncio.run(main())
"""

_RUNTIME_SCRIPT = """
import asyncio
import json
from contextlib import AsyncExitStack
import traceback
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.core.AuthNZ.repos.rbac_repo import AuthnzRbacRepo

async def main():
    result = {'complete': False, 'surface': SURFACE}
    try:
        if SURFACE == 'candidate':
            import asyncpg
            from tldw_Server_API.app.core.AuthNZ.settings import get_settings
            from tldw_Server_API.app.core.AuthNZ.profile_candidate_schema import validate_postgres_profile_candidate_schema
            conn = await asyncpg.connect(get_settings().DATABASE_URL)
            try:
                await validate_postgres_profile_candidate_schema(conn)
            finally:
                await conn.close()
        elif SURFACE == 'provider-secrets':
            from datetime import datetime, timezone
            from tldw_Server_API.app.core.AuthNZ.repos.user_provider_secrets_repo import AuthnzUserProviderSecretsRepo
            repo = AuthnzUserProviderSecretsRepo(await get_db_pool())
            async with AsyncExitStack() as occupied:
                for _ in range(4):
                    await occupied.enter_async_context(repo.db_pool.pool.acquire())
                async with asyncio.timeout(15):
                    await repo.ensure_tables()
                    assert await repo.list_secrets_for_user(USER_ID) == []
            created = await repo.upsert_secret(user_id=USER_ID, provider='openai', encrypted_blob='synthetic-encrypted-fixture',
                key_hint='fixture', metadata={'scope':'runtime'}, updated_at=datetime.now(timezone.utc), created_by=USER_ID, updated_by=USER_ID)
            assert created['user_id'] == USER_ID and created['provider'] == 'openai'
            assert len(await repo.list_secrets_for_user(USER_ID)) == 1
            assert await repo.list_secrets_for_user(ADMIN_ID) == []
            assert await repo.delete_secret(USER_ID, 'openai', revoked_by=USER_ID)
            assert await repo.list_secrets_for_user(USER_ID) == []
        elif SURFACE in {'api-keys', 'api-key-write'}:
            from tldw_Server_API.app.core.AuthNZ.api_key_manager import APIKeyManager
            manager = APIKeyManager(await get_db_pool())
            async with AsyncExitStack() as occupied:
                for _ in range(4):
                    await occupied.enter_async_context(manager.db_pool.pool.acquire())
                async with asyncio.timeout(15):
                    await manager.initialize()
                    assert await manager.list_user_keys(user_id=ADMIN_ID, include_revoked=False) == []
                    assert await manager.list_user_keys(user_id=USER_ID, include_revoked=False) == []
            if SURFACE == 'api-key-write':
                created = await manager.create_virtual_key(user_id=USER_ID, name='runtime-key',
                    allowed_endpoints=['chat.completions'], allowed_providers=['openai'], allowed_models=['fixture-model'],
                    budget_day_usd=1.25, budget_month_usd=2.5)
                row = await manager.db_pool.fetchone('SELECT llm_allowed_endpoints,llm_allowed_providers,llm_allowed_models,llm_budget_day_usd,llm_budget_month_usd FROM api_keys WHERE id=$1', created['id'])
                for name, expected in (('llm_allowed_endpoints',['chat.completions']),('llm_allowed_providers',['openai']),('llm_allowed_models',['fixture-model'])):
                    assert json.loads(row[name]) == expected
                assert row['llm_budget_day_usd']==1.25 and row['llm_budget_month_usd']==2.5
        elif SURFACE == 'users':
            await get_db_pool()
            repo = await AuthnzUsersRepo.from_pool()
            admin = await repo.get_user_by_username('runtime-admin')
            user = await repo.get_user_by_username('runtime-user')
            assert admin['role'] == 'admin' and user['role'] == 'user'
            assert admin['id'] != user['id']
            assert admin['is_superuser'] is True and user['is_superuser'] is False
            result['identities'] = [admin['id'], user['id']]
        else:
            repo = AuthnzRbacRepo()
            assert repo.has_permission(ADMIN_ID, 'users.manage_roles')
            assert not repo.has_permission(USER_ID, 'users.manage_roles')
            assert 'users.manage_roles' in repo.get_effective_permissions(ADMIN_ID)
            assert 'users.manage_roles' not in repo.get_effective_permissions(USER_ID)
        result['complete'] = True
    except Exception as error:
        result['error'] = type(error).__name__
        if SURFACE in {'api-keys', 'api-key-write'}:
            result['initialized'] = manager._initialized
        result['functions'] = [frame.name for frame in traceback.extract_tb(error.__traceback__)[-4:]]
        cause = error.__cause__ or error.__context__
        result['causes'] = []
        seen = {id(error)}
        while cause is not None and id(cause) not in seen and len(seen) < 6:
            seen.add(id(cause))
            detail = {'type': type(cause).__name__, 'functions': [frame.name for frame in traceback.extract_tb(cause.__traceback__)[-3:]]}
            if type(cause).__name__ == 'SchemaReadinessError':
                detail['readiness'] = str(cause)
            result['causes'].append(detail)
            cause = cause.__cause__ or cause.__context__
    finally:
        await reset_db_pool()
    print('RUNTIME_RESULT=' + json.dumps(result))

asyncio.run(main())
"""


def _dsn(config):
    """Use the official fixture's hexadecimal credential without printing it."""
    return f"postgresql://{config.pg_user}:{config.pg_password}@{config.pg_host}:{config.pg_port}/{config.pg_database}"


def _snapshot(backend):
    """Retain nonsecret user state and schema metadata in memory only."""
    return {
        "users": backend.execute(
            "SELECT id, uuid, username, role, is_active, is_verified, is_superuser, updated_at, profile_version FROM public.users ORDER BY id"
        ).rows,
        "columns": backend.execute(
            "SELECT table_name, column_name, data_type, is_nullable, column_default "
            "FROM information_schema.columns WHERE table_schema = 'public' "
            "ORDER BY table_name, ordinal_position"
        ).rows,
        "constraints": backend.execute(
            "SELECT conrelid, conname, pg_get_constraintdef(oid) AS definition "
            "FROM pg_constraint WHERE connamespace = 'public'::regnamespace "
            "ORDER BY conrelid, conname"
        ).rows,
        "role_links": backend.execute(
            "SELECT role_id, permission_id FROM public.role_permissions ORDER BY role_id, permission_id"
        ).rows,
        "indexes": backend.execute(
            "SELECT tablename, indexname, indexdef FROM pg_indexes "
            "WHERE schemaname = 'public' ORDER BY tablename, indexname"
        ).rows,
    }


@pytest.fixture
def migrated_runtime(pg_database_config, pg_restricted_backend, tmp_path):
    """Separate real owner initialization from the official restricted serving role."""
    owner_dir = tmp_path / "owner"
    owner_dir.mkdir()
    owner_env = _runtime_env(owner_dir, _dsn(pg_database_config), backend="postgres", mode="multi_user")
    ids = _run_runtime(owner_dir, owner_env, _OWNER_SCRIPT)
    owner = DatabaseBackendFactory.create_backend(pg_database_config)
    restricted = pg_restricted_backend.config
    try:
        with owner.transaction() as conn:
            conn.execute(sql.SQL("ALTER SCHEMA public OWNER TO {}").format(sql.Identifier(pg_database_config.pg_user)))
            conn.execute(sql.SQL("REVOKE CREATE ON SCHEMA public FROM {}").format(sql.Identifier(restricted.pg_user)))
            conn.execute(
                sql.SQL("REVOKE CREATE ON DATABASE {} FROM {}").format(
                    sql.Identifier(pg_database_config.pg_database), sql.Identifier(restricted.pg_user)
                )
            )
            conn.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(sql.Identifier(restricted.pg_user)))
            conn.execute(
                sql.SQL("GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {}").format(
                    sql.Identifier(restricted.pg_user)
                )
            )
            conn.execute(
                sql.SQL("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {}").format(
                    sql.Identifier(restricted.pg_user)
                )
            )
        runtime_config = replace(restricted, connection_string=None)
        flags = pg_restricted_backend.execute(
            "SELECT rolsuper, rolbypassrls, rolcreatedb, rolcreaterole, "
            "has_schema_privilege(current_user, 'public', 'CREATE') AS can_create, "
            "EXISTS (SELECT 1 FROM pg_class WHERE relnamespace = 'public'::regnamespace "
            "AND pg_has_role(current_user, relowner, 'USAGE')) AS owns_objects "
            "FROM pg_roles WHERE rolname = current_user"
        ).first
        assert flags and not any(flags.values())  # nosec B101 -- real serving-authority boundary.
        runtime_dir = tmp_path / "runtime"
        runtime_dir.mkdir()
        runtime_env = _runtime_env(runtime_dir, _dsn(runtime_config), backend="postgres", mode="multi_user")
        runtime_env.update(TLDW_DB_POOL_SIZE="1", TLDW_DB_MAX_OVERFLOW="0", TLDW_DB_POOL_TIMEOUT="2")
        yield owner, runtime_dir, runtime_env, ids
    finally:
        pg_restricted_backend.get_pool().close_all()
        owner.get_pool().close_all()


def _read_runtime(runtime_dir, env, ids, surface):
    """Exercise the real repositories, with safe categorical failure diagnostics."""
    script = f"SURFACE={surface!r}\nADMIN_ID={ids['admin']}\nUSER_ID={ids['user']}\n" + _RUNTIME_SCRIPT
    return _run_runtime(runtime_dir, env, script)


@pytest.mark.parametrize("surface", ["users", "rbac", "candidate", "api-keys", "provider-secrets"])
def test_migrated_authnz_serves_without_owner_authority(migrated_runtime, surface):
    """Pool/user and configured permission reads preserve the owner-migrated store."""
    owner, runtime_dir, env, ids = migrated_runtime
    before = _snapshot(owner)
    result = _read_runtime(runtime_dir, env, ids, surface)
    assert result["complete"], result  # nosec B101 -- actual serving regression.
    assert _snapshot(owner) == before  # nosec B101 -- no runtime migration/data rewrite.


@pytest.mark.parametrize(
    ("surface", "drift"),
    [
        ("users", "ALTER TABLE public.users ALTER COLUMN profile_version DROP DEFAULT"),
        (
            "users",
            "ALTER TABLE public.users ALTER COLUMN is_superuser DROP DEFAULT; ALTER TABLE public.users ALTER COLUMN is_superuser TYPE TEXT USING is_superuser::text",
        ),
        ("users", "ALTER TABLE public.storage_quotas DROP COLUMN soft_limit_pct"),
        ("users", "DROP INDEX public.idx_audio_studio_media_tickets_expiry"),
        ("users", "ALTER TABLE public.team_members DROP CONSTRAINT team_members_pkey"),
        ("users", "ALTER TABLE public.storage_quotas DROP CONSTRAINT storage_quotas_check"),
        (
            "users",
            "DROP INDEX public.idx_storage_quotas_org_unique; CREATE UNIQUE INDEX idx_storage_quotas_org_unique ON public.storage_quotas(org_id) WHERE org_id IS NULL",
        ),
        ("rbac", "ALTER TABLE public.role_permissions DROP CONSTRAINT role_permissions_role_id_fkey"),
        ("rbac", "ALTER TABLE public.permissions DROP CONSTRAINT permissions_name_key"),
        ("rbac", "ALTER TABLE public.role_permissions DROP CONSTRAINT role_permissions_pkey"),
        (
            "rbac",
            "DELETE FROM public.role_permissions WHERE role_id = (SELECT id FROM public.roles WHERE name = 'admin') AND permission_id = (SELECT id FROM public.permissions WHERE name = 'media.read')",
        ),
        ("api-keys", "DROP TABLE public.api_key_audit_log"),
        ("api-keys", "ALTER TABLE public.api_keys DROP COLUMN key_prefix"),
        ("api-keys", "ALTER TABLE public.api_keys ALTER COLUMN is_virtual DROP DEFAULT; ALTER TABLE public.api_keys ALTER COLUMN is_virtual TYPE TEXT USING is_virtual::text"),
        ("api-keys", "ALTER TABLE public.api_key_audit_log DROP CONSTRAINT api_key_audit_log_pkey"),
        ("api-keys", "ALTER TABLE public.api_keys DROP CONSTRAINT api_keys_key_hash_key"),
        ("api-keys", "DROP INDEX public.idx_api_keys_key_id; CREATE UNIQUE INDEX idx_api_keys_key_id ON public.api_keys(key_id) WHERE key_id IS NULL"),
        ("api-keys", "ALTER TABLE public.api_keys DROP CONSTRAINT api_keys_parent_key_id_fkey"),
        ("api-keys", "ALTER TABLE public.api_key_audit_log DROP CONSTRAINT api_key_audit_log_api_key_id_fkey"),
        ("api-keys", "DROP INDEX public.idx_api_keys_user_id; CREATE INDEX idx_api_keys_user_id ON public.api_keys(status)"),
        ("api-keys", "ALTER TABLE public.api_key_audit_log ALTER COLUMN action DROP NOT NULL"),
        ("api-keys", "ALTER TABLE public.api_keys ALTER COLUMN usage_count DROP DEFAULT"),
        ("api-keys", "ALTER TABLE public.api_key_audit_log ALTER COLUMN id DROP DEFAULT"),
        ("api-keys", "ALTER TABLE public.api_keys ALTER COLUMN llm_budget_day_usd TYPE INTEGER USING llm_budget_day_usd::integer"),
        ("provider-secrets", 'DROP TABLE public.user_provider_secrets'),
        ("provider-secrets", 'ALTER TABLE public.user_provider_secrets DROP COLUMN encrypted_blob'),
        ("provider-secrets", 'ALTER TABLE public.user_provider_secrets ALTER COLUMN provider TYPE INTEGER USING 0'),
        ("provider-secrets", 'ALTER TABLE public.user_provider_secrets DROP CONSTRAINT user_provider_secrets_pkey'),
        ("provider-secrets", 'ALTER TABLE public.user_provider_secrets DROP CONSTRAINT user_provider_secrets_user_id_fkey'),
        ("provider-secrets", 'ALTER TABLE public.user_provider_secrets DROP CONSTRAINT uq_user_provider_secrets'),
        ("provider-secrets", 'ALTER TABLE public.user_provider_secrets DROP CONSTRAINT uq_user_provider_secrets; ALTER TABLE public.user_provider_secrets ADD CONSTRAINT uq_user_provider_secrets UNIQUE(user_id, provider) DEFERRABLE INITIALLY DEFERRED'),
        ("provider-secrets", 'DROP INDEX public.idx_user_provider_secrets_user_id; CREATE INDEX idx_user_provider_secrets_user_id ON public.user_provider_secrets(provider)'),
        ("provider-secrets", 'ALTER TABLE public.user_provider_secrets ALTER COLUMN encrypted_blob DROP NOT NULL'),
        ("provider-secrets", 'ALTER TABLE public.user_provider_secrets ALTER COLUMN created_at DROP DEFAULT'),
        ("provider-secrets", 'ALTER TABLE public.user_provider_secrets ALTER COLUMN id DROP DEFAULT'),
        ("provider-secrets", "ALTER TABLE public.user_provider_secrets ADD CONSTRAINT x01_deferred_credentials UNIQUE(user_id, provider) DEFERRABLE INITIALLY DEFERRED"),
        ("provider-secrets", "ALTER TABLE public.user_provider_secrets ADD CONSTRAINT x01_deferred_credentials UNIQUE(provider, user_id) DEFERRABLE INITIALLY DEFERRED"),
    ],
    ids=[
        "profile-default",
        "truthy-text-flag",
        "core-column",
        "audio-index",
        "candidate-primary-key",
        "quota-check",
        "quota-partial-uniqueness",
        "rbac-foreign-key",
        "rbac-unique-key",
        "rbac-primary-key",
        "rbac-required-link",
        "api-keys-table", "api-keys-column", "api-keys-type", "api-keys-primary", "api-keys-hash-unique",
        "api-keys-partial-id", "api-keys-parent", "api-keys-audit-link", "api-keys-index-columns",
        "api-keys-nullability", "api-keys-insert-default", "api-keys-generated-id", "api-keys-integer-usd",
        "provider-secrets-table",
        "provider-secrets-column",
        "provider-secrets-type",
        "provider-secrets-primary",
        "provider-secrets-user-link",
        "provider-secrets-unique",
        "provider-secrets-deferred-unique",
        "provider-secrets-index",
        "provider-secrets-nullability",
        "provider-secrets-default",
        "provider-secrets-generated-id",
        "provider-secrets-coexisting-deferred", "provider-secrets-reversed-deferred",
    ],
)
@pytest.mark.asyncio
async def test_nonowner_authnz_refuses_schema_drift_without_repair(migrated_runtime, surface, drift):
    """Serving admission must reject incomplete schemas without altering them."""
    owner, runtime_dir, env, ids = migrated_runtime
    async with unmanaged_authnz_pg_connection(owner.config.pg_database) as conn:
        await conn.execute(drift)
    before = _snapshot(owner)
    result = _read_runtime(runtime_dir, env, ids, surface)
    assert not result["complete"], result  # nosec B101 -- malformed-schema refusal.
    if surface == "api-keys":
        assert result["initialized"] is False  # nosec B101 -- failed initialization cannot be cached.
        assert any(cause["type"] == "SchemaReadinessError" for cause in result["causes"])  # nosec B101 -- reject drift before DDL.
    if surface == "provider-secrets":
        assert result["error"] == "SchemaReadinessError" or any(cause["type"] == "SchemaReadinessError" for cause in result["causes"]), result  # nosec B101 -- refuse drift before DDL.
    assert _snapshot(owner) == before  # nosec B101 -- failed readiness leaves all state intact.


@pytest.mark.asyncio
async def test_nonowner_api_keys_preserve_supported_jsonb_lists_and_fractional_budgets(migrated_runtime):
    """A supported JSONB representation retains real virtual-key write behavior."""
    owner, runtime_dir, env, ids = migrated_runtime
    async with unmanaged_authnz_pg_connection(owner.config.pg_database) as conn:
        await conn.execute(
            "ALTER TABLE public.api_keys ALTER COLUMN llm_allowed_endpoints TYPE JSONB USING llm_allowed_endpoints::jsonb; "
            "ALTER TABLE public.api_keys ALTER COLUMN llm_allowed_providers TYPE JSONB USING llm_allowed_providers::jsonb; "
            "ALTER TABLE public.api_keys ALTER COLUMN llm_allowed_models TYPE JSONB USING llm_allowed_models::jsonb"
        )
    before = _snapshot(owner)
    result = _read_runtime(runtime_dir, env, ids, "api-key-write")
    assert result["complete"], result  # nosec B101 -- real supported representation and fractional write.
    assert _snapshot(owner) == before  # nosec B101 -- no serving schema or identity mutation.


_STARTUP_SCRIPT = """
import asyncio
import json
from contextlib import AsyncExitStack
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.services.startup_auth import _ensure_pg_extras, AuthStartupError

async def main():
    result = {'complete': False}
    try:
        pool = await get_db_pool()
        from tldw_Server_API.app.core.AuthNZ.postgres_runtime_schema import POSTGRES_AUTH_MIGRATION_AUTHORITY_SQL
        async with pool.transaction() as conn:
            can_migrate = await conn.fetchval(POSTGRES_AUTH_MIGRATION_AUTHORITY_SQL)
        async with AsyncExitStack() as occupied:
            if not can_migrate:
                for _ in range(4):
                    await occupied.enter_async_context(pool.pool.acquire())
            async with asyncio.timeout(15):
                await _ensure_pg_extras(pool)
            result['complete'] = True
    except AuthStartupError as error:
        result['error'] = str(error)
    except Exception as error:
        result['unexpected_error'] = type(error).__name__
    finally:
        await reset_db_pool()
    print('RUNTIME_RESULT=' + json.dumps(result))

asyncio.run(main())
"""



def _grant_required_auth_runtime(owner, config):
    """Grant only required DML after canonical owner startup creates new tables."""
    role = sql.Identifier(config.pg_user)
    with owner.transaction() as conn:
        for table in ("rate_limits", "failed_attempts", "account_lockouts", "token_blacklist", "resource_daily_ledger"):
            conn.execute(sql.SQL("GRANT SELECT,INSERT,UPDATE,DELETE ON {} TO {}").format(sql.Identifier(table), role))
        conn.execute(sql.SQL("GRANT USAGE,SELECT ON SEQUENCE public.token_blacklist_id_seq TO {}").format(role))
        conn.execute(sql.SQL("GRANT USAGE,SELECT ON SEQUENCE public.resource_daily_ledger_id_seq TO {}").format(role))


def test_migrated_authnz_startup_extras_serve_without_owner_authority(migrated_runtime, pg_restricted_backend):
    """Normal startup must admit a complete migrated schema without replaying owner DDL."""
    owner, runtime_dir, env, _ids = migrated_runtime
    owner_dir = runtime_dir.parent / "startup-owner"
    owner_dir.mkdir()
    owner_env = _runtime_env(owner_dir, _dsn(owner.config), backend="postgres", mode="multi_user")
    assert _run_runtime(owner_dir, owner_env, _STARTUP_SCRIPT)["complete"]  # nosec B101 -- real owner provisioning precondition.
    _grant_required_auth_runtime(owner, pg_restricted_backend.config)
    before = _snapshot(owner)
    result = _run_runtime(runtime_dir, env, _STARTUP_SCRIPT)
    assert result["complete"], result  # nosec B101 -- actual normal startup on an ordinary serving role.
    assert _snapshot(owner) == before  # nosec B101 -- serving startup cannot migrate or rewrite owner state.


@pytest.fixture
def migrated_startup(migrated_runtime, pg_restricted_backend):
    """Complete the real owner extras before testing a separate DML-only role."""
    owner, runtime_dir, _env, _ids = migrated_runtime
    owner_dir = runtime_dir.parent / "startup-provision"
    owner_dir.mkdir()
    owner_env = _runtime_env(owner_dir, _dsn(owner.config), backend="postgres", mode="multi_user")
    assert _run_runtime(owner_dir, owner_env, _STARTUP_SCRIPT)["complete"]  # nosec B101 -- real owner provisioning.
    _grant_required_auth_runtime(owner, pg_restricted_backend.config)
    return migrated_runtime


def _startup_snapshot(owner):
    """Include migration backfill rows so read-only readiness cannot silently repair them."""
    return {
        **_snapshot(owner),
        "sessions": owner.execute("SELECT * FROM public.sessions ORDER BY id").rows,
        "registration": owner.execute("SELECT * FROM public.registration_codes ORDER BY id").rows,
        "stt": owner.execute("SELECT * FROM public.org_stt_settings ORDER BY org_id").rows,
    }


@pytest.mark.parametrize(
    "drift,reason",
    [
        ("ALTER TABLE public.sessions DROP COLUMN refresh_jti", "AUTHNZ_CORE_SCHEMA_NOT_READY"),
        ("ALTER TABLE public.sessions ALTER COLUMN last_activity DROP DEFAULT", "AUTHNZ_CORE_SCHEMA_NOT_READY"),
        (
            "INSERT INTO public.sessions(user_id,token_hash,expires_at,last_activity) SELECT id,'x01-unbackfilled',CURRENT_TIMESTAMP,NULL FROM public.users WHERE username='runtime-user'",
            "AUTHNZ_CORE_SCHEMA_NOT_READY",
        ),
        ("ALTER TABLE public.sessions DROP CONSTRAINT sessions_user_id_fkey", "AUTHNZ_CORE_SCHEMA_NOT_READY"),
        ("DROP INDEX public.idx_sessions_access_jti", "AUTHNZ_CORE_SCHEMA_NOT_READY"),
        (
            "ALTER TABLE public.org_stt_settings ALTER COLUMN delete_audio_after_success SET DEFAULT FALSE",
            "AUTHNZ_PG_STT_SCHEMA_NOT_READY",
        ),
        ("ALTER TABLE public.org_stt_settings ALTER COLUMN redact_pii DROP NOT NULL", "AUTHNZ_PG_STT_SCHEMA_NOT_READY"),
        (
            "ALTER TABLE public.org_stt_settings ALTER COLUMN audio_retention_hours DROP DEFAULT; ALTER TABLE public.org_stt_settings ALTER COLUMN audio_retention_hours TYPE INTEGER USING audio_retention_hours::integer; ALTER TABLE public.org_stt_settings ALTER COLUMN audio_retention_hours SET DEFAULT 0",
            "AUTHNZ_PG_STT_SCHEMA_NOT_READY",
        ),
        (
            "ALTER TABLE public.org_stt_settings DROP CONSTRAINT org_stt_settings_updated_by_fkey",
            "AUTHNZ_PG_STT_SCHEMA_NOT_READY",
        ),
        (
            "ALTER TABLE public.registration_codes ADD COLUMN IF NOT EXISTS uses INTEGER DEFAULT 0; INSERT INTO public.registration_codes(code,uses,times_used) VALUES ('x01-legacy-uses',3,0)",
            "AUTHNZ_CORE_SCHEMA_NOT_READY",
        ),
        (
            "ALTER TABLE public.registration_codes DROP CONSTRAINT registration_codes_org_id_fkey",
            "AUTHNZ_CORE_SCHEMA_NOT_READY",
        ),
        (
            "ALTER TABLE public.users ALTER COLUMN last_login TYPE TIMESTAMP USING last_login AT TIME ZONE 'UTC'",
            "AUTHNZ_USER_TIMESTAMPS_NOT_READY",
        ),
        ("DROP TABLE public.share_audit_log", "AUTHNZ_PG_SHARING_SCHEMA_NOT_READY"),
        (
            "DELETE FROM public.role_permissions WHERE role_id = (SELECT id FROM public.roles WHERE name='user') AND permission_id = (SELECT id FROM public.permissions WHERE name='prompts.read')",
            "AUTHNZ_CORE_SCHEMA_NOT_READY",
        ),
    ],
    ids=[
        "session-refresh",
        "session-default",
        "session-backfill",
        "session-fk",
        "session-index",
        "stt-privacy-default",
        "stt-not-null",
        "stt-type",
        "stt-fk",
        "registration-backfill",
        "registration-org-fk",
        "user-timestamp",
        "sharing",
        "mcp-default-grant",
    ],
)
@pytest.mark.asyncio
async def test_nonowner_startup_refuses_required_drift_without_repair(migrated_startup, drift, reason):
    """Every required startup family fails closed without owner rights or in-place repair."""
    owner, runtime_dir, env, _ids = migrated_startup
    async with unmanaged_authnz_pg_connection(owner.config.pg_database) as conn:
        await conn.execute(drift)
    before = _startup_snapshot(owner)
    result = _run_runtime(runtime_dir, env, _STARTUP_SCRIPT)
    assert result == {"complete": False, "error": reason}  # nosec B101 -- real fail-closed startup reason.
    assert _startup_snapshot(owner) == before  # nosec B101 -- malformed rows/schema remain untouched.


@pytest.mark.asyncio
async def test_nonowner_startup_keeps_optional_migrations_optional(migrated_startup):
    """The serving admission must not promote optional owner extras into fatal requirements."""
    owner, runtime_dir, env, _ids = migrated_startup
    async with unmanaged_authnz_pg_connection(owner.config.pg_database) as conn:
        await conn.execute("DROP TABLE public.privilege_snapshots")
    before = _startup_snapshot(owner)
    result = _run_runtime(runtime_dir, env, _STARTUP_SCRIPT)
    assert result == {"complete": True}  # nosec B101 -- optional extra remains optional.
    assert _startup_snapshot(owner) == before  # nosec B101 -- no optional owner migration replay.


@pytest.mark.parametrize(
    "drift",
    [
        "ALTER TABLE public.sessions ALTER COLUMN is_active SET DEFAULT FALSE",
        "ALTER TABLE public.sessions ALTER COLUMN is_revoked SET DEFAULT TRUE",
        "ALTER TABLE public.sessions ALTER COLUMN created_at DROP DEFAULT",
        "ALTER TABLE public.sessions ALTER COLUMN id DROP DEFAULT",
        "ALTER TABLE public.org_invites ALTER COLUMN is_active SET DEFAULT FALSE",
        "ALTER TABLE public.org_invites ALTER COLUMN uses_count DROP DEFAULT",
        "ALTER TABLE public.password_history ALTER COLUMN created_at DROP DEFAULT",
        "ALTER TABLE public.audit_logs ALTER COLUMN id DROP DEFAULT",
        "ALTER TABLE public.password_history ALTER COLUMN id DROP DEFAULT",
        "ALTER TABLE public.org_invites ALTER COLUMN id DROP DEFAULT",
        "ALTER TABLE public.org_invite_redemptions ALTER COLUMN id DROP DEFAULT",
        "ALTER TABLE public.sessions ALTER COLUMN token_hash DROP NOT NULL",
        "ALTER TABLE public.org_invites ALTER COLUMN expires_at DROP NOT NULL",
    ],
    ids=[
        "inactive-session",
        "revoked-session",
        "session-created-default",
        "session-generated-id",
        "inactive-invite",
        "invite-usage-default",
        "password-history-timestamp",
        "audit-generated-id",
        "password-generated-id",
        "invite-generated-id",
        "redemption-generated-id",
        "session-token-nullability",
        "invite-expiry-nullability",
    ],
)
@pytest.mark.asyncio
async def test_nonowner_startup_refuses_session_insert_default_drift(migrated_startup, drift):
    """Login omits these columns and must not admit broken owner-provisioned defaults."""
    owner, runtime_dir, env, _ids = migrated_startup
    async with unmanaged_authnz_pg_connection(owner.config.pg_database) as conn:
        await conn.execute(drift)
    before = _startup_snapshot(owner)
    result = _run_runtime(runtime_dir, env, _STARTUP_SCRIPT)
    assert result == {"complete": False, "error": "AUTHNZ_CORE_SCHEMA_NOT_READY"}  # nosec B101 -- actual broken session-default refusal.
    assert _startup_snapshot(owner) == before  # nosec B101 -- nonowner cannot repair owner defaults.


_STT_REPO_SCRIPT = """
import asyncio
import json
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.repos.org_stt_settings_repo import AuthnzOrgSttSettingsRepo

async def main():
    result = {'complete': False}
    try:
        pool = await get_db_pool()
        if DIRECT:
            async with pool.transaction() as conn:
                await AuthnzOrgSttSettingsRepo(conn).ensure_tables()
        else:
            await AuthnzOrgSttSettingsRepo(pool).ensure_tables()
        result['complete'] = True
    except Exception as error:
        result['error'] = type(error).__name__
    finally:
        await reset_db_pool()
    print('RUNTIME_RESULT=' + json.dumps(result))

asyncio.run(main())
"""


@pytest.mark.parametrize("direct", [False, True], ids=["pool", "connection"])
def test_org_stt_repository_serves_migrated_schema_without_owner_authority(migrated_startup, direct):
    """The endpoint repository must reuse migrated privacy settings without DDL."""
    owner, runtime_dir, env, _ids = migrated_startup
    before = _startup_snapshot(owner)
    result = _run_runtime(runtime_dir, env, f"DIRECT={direct!r}\n" + _STT_REPO_SCRIPT)
    assert result == {"complete": True}  # nosec B101 -- actual serving repository admission.
    assert _startup_snapshot(owner) == before  # nosec B101 -- endpoint readiness cannot migrate settings.


@pytest.mark.parametrize(
    "drift",
    [
        "ALTER TABLE public.org_stt_settings ALTER COLUMN delete_audio_after_success SET DEFAULT FALSE",
        "ALTER TABLE public.org_stt_settings DROP COLUMN redact_categories_json",
    ],
    ids=["privacy-default", "missing-column"],
)
@pytest.mark.asyncio
async def test_org_stt_repository_refuses_schema_drift_without_repair(migrated_startup, drift):
    """Request-time admission rejects privacy/schema drift without a DDL fallback."""
    owner, runtime_dir, env, _ids = migrated_startup
    async with unmanaged_authnz_pg_connection(owner.config.pg_database) as conn:
        await conn.execute(drift)
    before = _startup_snapshot(owner)
    result = _run_runtime(runtime_dir, env, "DIRECT=False\n" + _STT_REPO_SCRIPT)
    assert result == {"complete": False, "error": "TransactionError"}  # nosec B101 -- pool preserves its normal readiness-error translation.
    assert _startup_snapshot(owner) == before  # nosec B101 -- request does not repair owner schema.


_MONITORING_REPO_SCRIPT = """
import asyncio
import json
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.repos.admin_monitoring_repo import AuthnzAdminMonitoringRepo

async def main():
    result = {'complete': False}
    try:
        pool = await get_db_pool()
        repo = AuthnzAdminMonitoringRepo(pool)
        await repo.ensure_schema_ready_once()
        if ROUND_TRIP:
            rule = await repo.create_rule(metric='queue_depth', operator='gte', threshold=2.5,
                duration_minutes=1, severity='warning', enabled=True, created_by_user_id=ADMIN_ID)
            assert (await repo.get_rule(rule['id']))['threshold'] == 2.5
            state = await repo.upsert_alert_state(alert_identity='x01-runtime-monitor',
                assigned_to_user_id=ADMIN_ID, updated_by_user_id=ADMIN_ID)
            assert state['assigned_to_user_id'] == ADMIN_ID
            event = await repo.append_alert_event(alert_identity='x01-runtime-monitor',
                action='assigned', actor_user_id=ADMIN_ID, details_json='{}')
            assert (await repo.list_alert_events(alert_identity='x01-runtime-monitor'))[0]['id'] == event['id']
            assert await repo.delete_rule(rule['id'])
            assert await repo.clear_state_and_events() == {'deleted_states': 1, 'deleted_events': 1}
        else:
            await repo.list_rules()
        await AuthnzAdminMonitoringRepo(pool).ensure_schema_ready_once()
        result['complete'] = True
    except Exception as error:
        result['error'] = type(error).__name__
    finally:
        await reset_db_pool()
    print('RUNTIME_RESULT=' + json.dumps(result))

asyncio.run(main())
"""


@pytest.fixture
def migrated_monitoring(migrated_runtime, pg_restricted_backend):
    """Provision with the real owner, then expose only DML to the fixture serving role."""
    owner, runtime_dir, env, ids = migrated_runtime
    owner_dir = runtime_dir.parent / "monitoring-owner"
    owner_dir.mkdir()
    owner_env = _runtime_env(owner_dir, _dsn(owner.config), backend="postgres", mode="multi_user")
    assert _run_runtime(owner_dir, owner_env, "ROUND_TRIP=False\n" + _MONITORING_REPO_SCRIPT)["complete"]  # nosec B101 -- actual owner provisioning.
    role = sql.Identifier(pg_restricted_backend.config.pg_user)
    with owner.transaction() as conn:
        for table in ("admin_alert_rules", "admin_alert_state", "admin_alert_events"):
            conn.execute(sql.SQL("GRANT SELECT,INSERT,UPDATE,DELETE ON {} TO {}").format(sql.Identifier(table), role))
        for sequence in ("admin_alert_rules_id_seq", "admin_alert_events_id_seq"):
            conn.execute(sql.SQL("GRANT USAGE,SELECT ON SEQUENCE {} TO {}").format(sql.Identifier(sequence), role))
    return owner, runtime_dir, env, ids


def test_nonowner_monitoring_serves_without_schema_creation(migrated_monitoring):
    """Router/request readiness permits real monitoring writes without owner authority."""
    owner, runtime_dir, env, ids = migrated_monitoring
    before = _snapshot(owner)
    result = _run_runtime(runtime_dir, env, f"ROUND_TRIP=True\nADMIN_ID={ids['admin']}\n" + _MONITORING_REPO_SCRIPT)
    assert result == {"complete": True}, result  # nosec B101 -- actual restricted-role round trip.
    assert _snapshot(owner) == before  # nosec B101 -- readiness cannot replay owner migrations.


@pytest.mark.parametrize(
    "drift",
    ["DROP TABLE public.admin_alert_events", "ALTER TABLE public.admin_alert_rules DROP COLUMN threshold"],
    ids=["table", "column"],
)
def test_nonowner_monitoring_refuses_missing_schema_without_repair(migrated_monitoring, drift):
    """A missing monitoring family must fail closed, never gain serving DDL rights."""
    owner, runtime_dir, env, _ids = migrated_monitoring
    owner.execute(drift)
    before = _snapshot(owner)
    result = _run_runtime(runtime_dir, env, "ROUND_TRIP=False\n" + _MONITORING_REPO_SCRIPT)
    assert not result["complete"], result  # nosec B101 -- actual malformed-schema refusal.
    assert _snapshot(owner) == before  # nosec B101 -- serving refusal preserves owner schema.


def test_owner_startup_provisions_monitoring_before_serving(migrated_runtime):
    """Canonical owner startup must create every table required by the router startup."""
    owner, runtime_dir, _env, _ids = migrated_runtime
    owner_dir = runtime_dir.parent / "monitoring-startup-owner"
    owner_dir.mkdir()
    owner_env = _runtime_env(owner_dir, _dsn(owner.config), backend="postgres", mode="multi_user")
    assert _run_runtime(owner_dir, owner_env, _STARTUP_SCRIPT)["complete"]  # nosec B101 -- actual canonical owner startup.
    names = {
        row["tablename"] for row in owner.execute("SELECT tablename FROM pg_tables WHERE schemaname='public'").rows
    }
    assert {"admin_alert_rules", "admin_alert_state", "admin_alert_events"} <= names  # nosec B101 -- required owner-provisioned router schema.


_LOCKOUT_REPO_SCRIPT = """
import asyncio
import json
from datetime import datetime, timezone
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.repos.rate_limits_repo import AuthnzRateLimitsRepo
from tldw_Server_API.app.core.AuthNZ.lockout_tracker import LockoutTracker

async def main():
    result = {'complete': False}
    tracker = None
    try:
        pool = await get_db_pool()
        repo = AuthnzRateLimitsRepo(pool)
        if TRACKER_ONLY:
            tracker = LockoutTracker(db_pool=pool)
            await tracker.initialize()
        else:
            await repo.ensure_schema()
        if ROUND_TRIP:
            now = datetime.now(timezone.utc)
            assert await repo.get_active_lockout(identifier='x01-lockout', now=now) is None
            blocked = await repo.record_failed_attempt_and_lockout(identifier='x01-lockout', attempt_type='login',
                now=now, lockout_threshold=1, lockout_duration_minutes=5)
            assert blocked['is_locked'] is True
            assert await repo.get_active_lockout(identifier='x01-lockout', now=now) == blocked['lockout_expires']
            assert await repo.get_active_lockout(identifier='x01-lockout', attempt_type='password_reset', now=now) is None
            await repo.reset_failed_attempts_and_lockout(identifier='x01-lockout', attempt_type='login')
            assert await repo.get_active_lockout(identifier='x01-lockout', now=now) is None
        result['complete'] = True
    except Exception as error:
        result['error'] = type(error).__name__
    finally:
        if tracker is not None:
            result['initialized'] = tracker._initialized
        await reset_db_pool()
    print('RUNTIME_RESULT=' + json.dumps(result))

asyncio.run(main())
"""


@pytest.fixture
def migrated_lockouts(migrated_startup, pg_restricted_backend):
    """Use the actual owner helper and grant only DML to the official runtime role."""
    owner, runtime_dir, env, ids = migrated_startup
    owner_dir = runtime_dir.parent / "lockout-owner"
    owner_dir.mkdir()
    owner_env = _runtime_env(owner_dir, _dsn(owner.config), backend="postgres", mode="multi_user")
    assert _run_runtime(owner_dir, owner_env, "TRACKER_ONLY=False\nROUND_TRIP=False\n" + _LOCKOUT_REPO_SCRIPT)[
        "complete"
    ]  # nosec B101 -- actual owner provisioning.
    _grant_required_auth_runtime(owner, pg_restricted_backend.config)
    return owner, runtime_dir, env, ids


def test_nonowner_lockouts_serve_and_preserve_attempt_scope(migrated_lockouts):
    """A fully provisioned serving role can enforce and clear a real lockout without DDL."""
    owner, runtime_dir, env, _ids = migrated_lockouts
    before = _snapshot(owner)
    result = _run_runtime(runtime_dir, env, "TRACKER_ONLY=False\nROUND_TRIP=True\n" + _LOCKOUT_REPO_SCRIPT)
    assert result == {"complete": True}, result  # nosec B101 -- actual restricted lockout CRUD and distinct attempt scope.
    assert _snapshot(owner) == before  # nosec B101 -- serving bootstrap does not change owner schema.


@pytest.mark.parametrize(
    "drift",
    ["DROP TABLE public.account_lockouts", "ALTER TABLE public.failed_attempts DROP COLUMN attempt_count"],
    ids=["table", "column"],
)
def test_nonowner_lockout_initialization_refuses_missing_schema(migrated_lockouts, drift):
    """Required security storage cannot be hidden by an initialized=True flag."""
    owner, runtime_dir, env, _ids = migrated_lockouts
    owner.execute(drift)
    before = _snapshot(owner)
    result = _run_runtime(runtime_dir, env, "TRACKER_ONLY=True\nROUND_TRIP=False\n" + _LOCKOUT_REPO_SCRIPT)
    assert not result["complete"] and result["initialized"] is False, result  # nosec B101 -- actual initializer must refuse missing lockout state.
    startup = _run_runtime(runtime_dir, env, _STARTUP_SCRIPT)
    assert startup == {"complete": False, "error": "AUTHNZ_PG_LOCKOUT_SCHEMA_NOT_READY"}, startup  # nosec B101 -- required login security readiness is sanitized and fail-closed.
    assert _snapshot(owner) == before  # nosec B101 -- no serving-role repair or metadata mutation.


def test_owner_startup_provisions_lockouts_before_serving(migrated_runtime):
    """Canonical owner startup includes security tables needed before the login handler."""
    owner, runtime_dir, _env, _ids = migrated_runtime
    owner_dir = runtime_dir.parent / "lockout-startup-owner"
    owner_dir.mkdir()
    owner_env = _runtime_env(owner_dir, _dsn(owner.config), backend="postgres", mode="multi_user")
    assert _run_runtime(owner_dir, owner_env, _STARTUP_SCRIPT)["complete"]  # nosec B101 -- normal owner startup completes.
    names = {
        row["tablename"] for row in owner.execute("SELECT tablename FROM pg_tables WHERE schemaname='public'").rows
    }
    assert {"rate_limits", "failed_attempts", "account_lockouts"} <= names  # nosec B101 -- owner must provision required login protection.


_BLACKLIST_RUNTIME_SCRIPT = """
import asyncio
import json
from datetime import datetime, timedelta, timezone
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.token_blacklist import TokenBlacklist
from tldw_Server_API.app.core.AuthNZ.repos.token_blacklist_repo import AuthnzTokenBlacklistRepo

async def main():
    result = {'complete': False}
    service = None
    try:
        pool = await get_db_pool()
        if OWNER_PROVISION:
            await AuthnzTokenBlacklistRepo(pool).ensure_schema()
        else:
            service = TokenBlacklist(db_pool=pool)
            await service.initialize()
            if ROUND_TRIP:
                assert await service.is_blacklisted('x01-unrevoked', strict=True) is False
                assert await service.revoke_token(jti='x01-revoked', expires_at=datetime.now(timezone.utc)+timedelta(minutes=5),
                    user_id=ADMIN_ID, token_type='access')
                reopened = TokenBlacklist(db_pool=pool)
                assert await reopened.is_blacklisted('x01-revoked', strict=True) is True
                assert await reopened.is_blacklisted('x01-unrevoked', strict=True) is False
                from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
                from tldw_Server_API.app.core.AuthNZ.session_manager import get_session_manager
                jwt_service = get_jwt_service()
                token = jwt_service.create_access_token(user_id=ADMIN_ID, username='runtime-admin', role='admin')
                claims = jwt_service.decode_access_token(token)
                sessions = await get_session_manager()
                assert await sessions.is_token_blacklisted(token, claims['jti'], strict=True) is False
                assert await service.revoke_token(jti=claims['jti'],
                    expires_at=datetime.fromtimestamp(claims['exp'], timezone.utc), user_id=ADMIN_ID, token_type='access')
                assert await sessions.is_token_blacklisted(token, claims['jti'], strict=True) is True
        result['complete'] = True
    except Exception as error:
        result['error'] = type(error).__name__
    finally:
        if service is not None:
            result['initialized'] = service._initialized
        await reset_db_pool()
    print('RUNTIME_RESULT=' + json.dumps(result))

asyncio.run(main())
"""


@pytest.fixture
def migrated_blacklist(migrated_startup, pg_restricted_backend):
    """Owner provisions the canonical repo before granting runtime-only access."""
    owner, runtime_dir, env, ids = migrated_startup
    owner_dir = runtime_dir.parent / "blacklist-owner"
    owner_dir.mkdir()
    owner_env = _runtime_env(owner_dir, _dsn(owner.config), backend="postgres", mode="multi_user")
    assert _run_runtime(owner_dir, owner_env, "OWNER_PROVISION=True\nROUND_TRIP=False\n" + _BLACKLIST_RUNTIME_SCRIPT)[
        "complete"
    ]  # nosec B101 -- actual owner provisioning.
    role = sql.Identifier(pg_restricted_backend.config.pg_user)
    with owner.transaction() as conn:
        conn.execute(sql.SQL("GRANT SELECT,INSERT,UPDATE,DELETE ON public.token_blacklist TO {}").format(role))
        conn.execute(sql.SQL("GRANT USAGE,SELECT ON SEQUENCE public.token_blacklist_id_seq TO {}").format(role))
    return owner, runtime_dir, env, ids


def test_nonowner_blacklist_accepts_unrevoked_and_retains_revocation(migrated_blacklist):
    """Required fresh-token checks work without DDL while persisted revocation stays enforced."""
    owner, runtime_dir, env, ids = migrated_blacklist
    before = _snapshot(owner)
    result = _run_runtime(
        runtime_dir,
        env,
        f"OWNER_PROVISION=False\nROUND_TRIP=True\nADMIN_ID={ids['admin']}\n" + _BLACKLIST_RUNTIME_SCRIPT,
    )
    assert result == {"complete": True, "initialized": True}, result  # nosec B101 -- actual fresh/revoked/read-after-reopen service behavior.
    assert _snapshot(owner) == before  # nosec B101 -- serving checks cannot modify owner schema.


def test_owner_startup_provisions_blacklist_before_serving(migrated_runtime):
    """Required revocation storage must exist before admitting the serving role."""
    owner, runtime_dir, _env, _ids = migrated_runtime
    owner_dir = runtime_dir.parent / "blacklist-startup-owner"
    owner_dir.mkdir()
    owner_env = _runtime_env(owner_dir, _dsn(owner.config), backend="postgres", mode="multi_user")
    assert _run_runtime(owner_dir, owner_env, _STARTUP_SCRIPT)["complete"]  # nosec B101 -- actual canonical owner startup.
    names = {
        row["tablename"] for row in owner.execute("SELECT tablename FROM pg_tables WHERE schemaname='public'").rows
    }
    assert "token_blacklist" in names  # nosec B101 -- canonical owner startup must provision revocation storage.


@pytest.mark.parametrize(
    "drift",
    ["DROP TABLE public.token_blacklist", "ALTER TABLE public.token_blacklist DROP COLUMN expires_at"],
    ids=["table", "column"],
)
def test_nonowner_blacklist_refuses_missing_schema_before_initialization(migrated_blacklist, drift):
    """Revocation storage failures must remain fail-closed without runtime DDL."""
    owner, runtime_dir, env, _ids = migrated_blacklist
    owner.execute(drift)
    before = _snapshot(owner)
    result = _run_runtime(runtime_dir, env, "OWNER_PROVISION=False\nROUND_TRIP=False\n" + _BLACKLIST_RUNTIME_SCRIPT)
    assert not result["complete"] and result["initialized"] is False, result  # nosec B101 -- actual revocation service must refuse missing storage.
    startup = _run_runtime(runtime_dir, env, _STARTUP_SCRIPT)
    assert startup == {"complete": False, "error": "AUTHNZ_PG_TOKEN_BLACKLIST_NOT_READY"}, startup  # nosec B101 -- canonical startup refuses before authentication.
    assert _snapshot(owner) == before  # nosec B101 -- refusal cannot repair the owner schema.


_DAILY_LEDGER_SCRIPT = """
import asyncio
import json
from datetime import datetime, timezone
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.DB_Management.Resource_Daily_Ledger import LedgerEntry, ResourceDailyLedger

async def main():
    result = {'complete': False}
    ledger = None
    try:
        ledger = ResourceDailyLedger(db_pool=await get_db_pool())
        await ledger.initialize()
        if ROUND_TRIP:
            now = datetime.now(timezone.utc)
            entry = LedgerEntry('user', 'x01-runtime', 'media_mb', 5, 'x01-retained', now)
            assert await ledger.add(entry) is True
            assert await ledger.add(entry) is False
            reopened = ResourceDailyLedger(db_pool=ledger.db_pool)
            assert await reopened.total_for_day('user', 'x01-runtime', 'media_mb') == 5
            denied = await reopened.consume_if_within_cap(
                LedgerEntry('user', 'x01-runtime', 'media_mb', 7, 'x01-denied', now), daily_cap=10)
            assert denied.allowed is False and denied.inserted is False and denied.used == 5
            async def consume(op):
                return await reopened.consume_if_within_cap(
                    LedgerEntry('user', 'x01-runtime', 'media_mb', 1, op, now), daily_cap=6)
            results = await asyncio.gather(consume('x01-first'), consume('x01-second'))
            assert sum(row.allowed for row in results) == 1
            assert await reopened.total_for_day('user', 'x01-runtime', 'media_mb') == 6
        result['complete'] = True
    except Exception as error:
        result['error'] = type(error).__name__
    finally:
        if ledger is not None:
            result['initialized'] = ledger._initialized
        await reset_db_pool()
    print('RUNTIME_RESULT=' + json.dumps(result))

asyncio.run(main())
"""


@pytest.fixture
def migrated_daily_ledger(migrated_startup, pg_restricted_backend):
    """Canonical service provisions as owner before an ordinary accounting writer."""
    owner, runtime_dir, env, ids = migrated_startup
    owner_dir = runtime_dir.parent / "daily-ledger-owner"
    owner_dir.mkdir()
    owner_env = _runtime_env(owner_dir, _dsn(owner.config), backend="postgres", mode="multi_user")
    assert _run_runtime(owner_dir, owner_env, "ROUND_TRIP=False\n" + _DAILY_LEDGER_SCRIPT)["complete"]  # nosec B101 -- real owner provisioning precondition.
    role = sql.Identifier(pg_restricted_backend.config.pg_user)
    with owner.transaction() as conn:
        conn.execute(sql.SQL("GRANT SELECT,INSERT,UPDATE,DELETE ON public.resource_daily_ledger TO {}").format(role))
        conn.execute(sql.SQL("GRANT USAGE,SELECT ON SEQUENCE public.resource_daily_ledger_id_seq TO {}").format(role))
    return owner, runtime_dir, env, ids


def test_nonowner_daily_ledger_preserves_idempotency_and_atomic_cap(migrated_daily_ledger):
    """Ordinary accounting must initialize read-only, persist once, and enforce concurrent caps."""
    owner, runtime_dir, env, _ids = migrated_daily_ledger
    before = _snapshot(owner)
    result = _run_runtime(runtime_dir, env, "ROUND_TRIP=True\n" + _DAILY_LEDGER_SCRIPT)
    assert result == {"complete": True, "initialized": True}, result  # nosec B101 -- real restricted accounting and concurrent cap behavior.
    assert _snapshot(owner) == before  # nosec B101 -- runtime accounting never changes schema.


def test_owner_startup_provisions_daily_ledger_before_serving(migrated_runtime):
    """Required accounting storage must exist before admitting ordinary uploads."""
    owner, runtime_dir, _env, _ids = migrated_runtime
    owner_dir = runtime_dir.parent / "daily-ledger-startup-owner"
    owner_dir.mkdir()
    owner_env = _runtime_env(owner_dir, _dsn(owner.config), backend="postgres", mode="multi_user")
    assert _run_runtime(owner_dir, owner_env, _STARTUP_SCRIPT)["complete"]  # nosec B101 -- actual canonical owner startup.
    names = {
        row["tablename"] for row in owner.execute("SELECT tablename FROM pg_tables WHERE schemaname='public'").rows
    }
    assert "resource_daily_ledger" in names  # nosec B101 -- owner startup must provision accounting.


@pytest.mark.parametrize(
    "drift",
    [
        ("DROP TABLE public.resource_daily_ledger",),
        ("ALTER TABLE public.resource_daily_ledger DROP COLUMN occurred_at",),
        ("DROP INDEX public.uq_ledger_op",),
        ("DROP INDEX public.uq_ledger_op", "CREATE INDEX uq_ledger_op ON public.resource_daily_ledger(op_id)"),
        (
            "DROP INDEX public.uq_ledger_op",
            "ALTER TABLE public.resource_daily_ledger ADD CONSTRAINT uq_ledger_op "
            "UNIQUE(day_utc, entity_scope, entity_value, category, op_id) DEFERRABLE INITIALLY IMMEDIATE",
        ),
    ],
    ids=["table", "column", "unique-index", "wrong-index", "deferrable-index"],
)
def test_nonowner_daily_ledger_refuses_incomplete_schema(migrated_daily_ledger, drift):
    """Admission refuses incomplete columns or lost idempotency without repairing them."""
    owner, runtime_dir, env, _ids = migrated_daily_ledger
    for statement in drift:
        owner.execute(statement)
    before = _snapshot(owner)
    result = _run_runtime(runtime_dir, env, "ROUND_TRIP=False\n" + _DAILY_LEDGER_SCRIPT)
    assert not result["complete"] and result["initialized"] is False, result  # nosec B101 -- incomplete accounting cannot become initialized.
    startup = _run_runtime(runtime_dir, env, _STARTUP_SCRIPT)
    assert startup == {"complete": False, "error": "AUTHNZ_PG_DAILY_LEDGER_NOT_READY"}, startup  # nosec B101 -- fail closed before upload.
    assert _snapshot(owner) == before  # nosec B101 -- failed runtime validation cannot repair owner state.


_NATIVE_QUOTA_STARTUP_SCRIPT = """
import asyncio
import json
from contextlib import AsyncExitStack
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.DB_Management.storage_quota_reservation_store import NativeQuotaOwner, NativeQuotaReservationStore
from tldw_Server_API.app.services.startup_auth import _ensure_pg_extras

async def main():
    result = {'complete': False}
    try:
        pool = await get_db_pool()
        async with AsyncExitStack() as occupied:
            for _ in range(4):
                await occupied.enter_async_context(pool.pool.acquire())
            async with asyncio.timeout(15):
                await _ensure_pg_extras(pool)
                if USER_ID is not None:
                    identity = NativeQuotaOwner(
                        user_id=USER_ID, owner_key='native-key:x01-quota',
                        user_uuid=str(await pool.fetchval('SELECT uuid FROM users WHERE id = ?', USER_ID)),
                    )
                    async with NativeQuotaReservationStore(pool).transaction(identity) as store:
                        await store.user(identity)
                        admitted = await store.insert(identity, 'x01-native-quota', 8)
                        assert admitted.state == 'active'
                        await store.insert(identity, 'x01-native-quota-second', 13)
                        assert await store.usage(user_id=USER_ID) == {'bytes': 21, 'count': 2}
                        released = await store.release(identity, 'x01-native-quota')
                        assert released.state == 'released'
                        assert await store.usage(user_id=USER_ID) == {'bytes': 13, 'count': 1}
                        await store.release(identity, 'x01-native-quota-second')
                        assert await store.usage(user_id=USER_ID) == {'bytes': 0, 'count': 0}
                    result['quota'] = ['active', 'released']
            result['complete'] = True
    except Exception as error:
        result['error'] = type(error).__name__
        seen = set()
        while error is not None and id(error) not in seen:
            seen.add(id(error))
            if type(error).__name__ == 'SchemaReadinessError':
                result['error'] = 'SchemaReadinessError'
                break
            error = error.__cause__ or error.__context__
    finally:
        await reset_db_pool()
    print('RUNTIME_RESULT=' + json.dumps(result))

asyncio.run(main())
"""


def test_nonowner_startup_retains_real_native_quota_roundtrip(migrated_startup):
    owner, runtime_dir, env, ids = migrated_startup
    before = _startup_snapshot(owner)
    script = f"USER_ID={ids['user']!r}\n" + _NATIVE_QUOTA_STARTUP_SCRIPT
    assert _run_runtime(runtime_dir, env, script) == {"complete": True, "quota": ["active", "released"]}  # nosec B101 -- real DML-only role and one free pool slot.
    assert _startup_snapshot(owner) == before  # nosec B101 -- readiness never changes owner schema or account state.


@pytest.mark.parametrize("drift", [
    "DROP TABLE public.native_chat_quota_reservations",
    "ALTER TABLE public.native_chat_quota_reservations DROP COLUMN owner_key",
    "ALTER TABLE public.native_chat_quota_reservations ALTER COLUMN size_bytes TYPE INTEGER",
    "ALTER TABLE public.native_chat_quota_reservations ALTER COLUMN owner_key TYPE VARCHAR(1)",
    "ALTER TABLE public.native_chat_quota_reservations ALTER COLUMN workspace_id SET NOT NULL",
    "ALTER TABLE public.native_chat_quota_reservations ALTER COLUMN size_bytes DROP NOT NULL",
    "ALTER TABLE public.native_chat_quota_reservations DROP CONSTRAINT native_chat_quota_reservations_pkey",
    "DROP INDEX public.native_chat_quota_user",
    "DROP INDEX public.native_chat_quota_user; CREATE UNIQUE INDEX native_chat_quota_user ON public.native_chat_quota_reservations(user_id,user_uuid,state)",
    "DROP INDEX public.native_chat_quota_org; CREATE INDEX native_chat_quota_org ON public.native_chat_quota_reservations(team_id,state)",
    "ALTER TABLE public.native_chat_quota_reservations DROP CONSTRAINT native_chat_quota_reservations_scope_type_check",
    "ALTER TABLE public.native_chat_quota_reservations DROP CONSTRAINT native_chat_quota_reservations_state_check",
    "ALTER TABLE public.native_chat_quota_reservations DROP CONSTRAINT native_chat_quota_reservations_size_bytes_check",
    "ALTER TABLE public.native_chat_quota_reservations DROP CONSTRAINT native_chat_quota_reservations_check",
    "ALTER TABLE public.native_chat_quota_reservations DROP CONSTRAINT native_chat_quota_reservations_check1",
    "ALTER TABLE public.native_chat_quota_reservations DROP CONSTRAINT native_chat_quota_reservations_pkey; ALTER TABLE public.native_chat_quota_reservations ADD PRIMARY KEY (user_id,candidate_key) DEFERRABLE INITIALLY DEFERRED",
], ids=["table", "column", "integer-type", "text-type", "nullable", "not-null", "primary", "index", "unique-index", "wrong-index", "scope", "state", "size", "team-org", "workspace", "deferred-primary"])
@pytest.mark.asyncio
async def test_nonowner_startup_refuses_native_quota_drift(migrated_startup, drift):
    owner, runtime_dir, env, _ids = migrated_startup
    async with unmanaged_authnz_pg_connection(owner.config.pg_database) as conn:
        await conn.execute(drift)
    before = _startup_snapshot(owner)
    result = _run_runtime(runtime_dir, env, "USER_ID=None\n" + _NATIVE_QUOTA_STARTUP_SCRIPT)
    assert result == {"complete": False, "error": "SchemaReadinessError"}  # nosec B101 -- absence or weakened quota enforcement cannot be ready.
    assert _startup_snapshot(owner) == before  # nosec B101 -- an ordinary serving principal never repairs the owner schema.


_AUDIO_PROFILE_OWNER_SCRIPT = """
import asyncio
import json
import traceback
from datetime import datetime, timezone
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.Usage import audio_quota

async def main():
    result = {'complete': False, 'phase': 'owner-pool'}
    try:
        pool = await get_db_pool()
        result['phase'] = 'owner-audio-schema'
        await audio_quota._ensure_tables(pool)
        result['phase'] = 'owner-legacy-row'
        await pool.execute('INSERT INTO audio_usage_daily (user_id, day, minutes_used, jobs_started) VALUES ($1, $2, 2.5, 1)',
            USER_ID, datetime.now(timezone.utc).date())
        result['complete'] = True
    except Exception as error:
        result['error'] = type(error).__name__
        result['functions'] = [frame.name for frame in traceback.extract_tb(error.__traceback__)[-5:]]
    finally:
        await reset_db_pool()
    print('RUNTIME_RESULT=' + json.dumps(result))

asyncio.run(main())
"""

_AUDIO_PROFILE_SCRIPT = """
import asyncio
import json
import traceback
from httpx import ASGITransport, AsyncClient
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.Usage import audio_quota

async def main():
    result = {'complete': False, 'phase': 'cold-reporting'}
    try:
        assert await audio_quota.get_daily_minutes_used(USER_ID) == EXPECTED
        assert await audio_quota.get_monthly_minutes_used(USER_ID) == EXPECTED
        pool = await get_db_pool()
        count = await pool.fetchval("SELECT COUNT(*) FROM resource_daily_ledger WHERE entity_scope='user' AND entity_value=$1 AND category='minutes'", str(USER_ID))
        assert count == (1 if EXPECTED else 0)
        result['phase'] = 'real-login'
        from tldw_Server_API.app.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url='http://localhost') as client:
            login = await client.post('/api/v1/auth/login', data={'username':'runtime-user','password':'X!a7-fixture-account'})
            result['loginStatus'] = login.status_code
            assert login.status_code == 200, login.status_code
            headers = {'Authorization':'Bearer '+login.json()['access_token'], 'X-TLDW-Expected-User-ID': str(USER_ID)}
            before = dict(await pool.fetchone('SELECT id,role,is_active,is_verified,updated_at,profile_version FROM users WHERE id=$1', USER_ID))
            result['phase'] = 'real-profile'
            response = await client.get('/api/v1/users/me/profile', headers=headers)
            result['profileStatus'] = response.status_code
            assert response.status_code == 200, response.status_code
            profile = response.json()
            assert profile['user']['id'] == USER_ID and profile['user']['role'] == 'user'
            assert profile['quotas']['audio']['daily_minutes_used'] == EXPECTED
            assert dict(await pool.fetchone('SELECT id,role,is_active,is_verified,updated_at,profile_version FROM users WHERE id=$1', USER_ID)) == before
        result['complete'] = True
    except Exception as error:
        result['error'] = type(error).__name__
        result['functions'] = [frame.name for frame in traceback.extract_tb(error.__traceback__)[-5:]]
    finally:
        from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import shutdown_all_audit_services
        await shutdown_all_audit_services()
        await reset_db_pool()
    print('RUNTIME_RESULT=' + json.dumps(result))

asyncio.run(main())
"""


@pytest.mark.parametrize("legacy_present", [True, False], ids=["legacy-usage", "no-legacy-table"])
def test_nonowner_profile_preserves_cold_audio_reporting(migrated_daily_ledger, pg_restricted_backend, legacy_present):
    """Cold reporting must retain legacy minutes without request-time owner DDL."""
    owner, runtime_dir, env, ids = migrated_daily_ledger
    if legacy_present:
        prepare = runtime_dir.parent / "audio-owner"
        prepare.mkdir()
        owner_env = _runtime_env(prepare, _dsn(owner.config), backend="postgres", mode="multi_user")
        prepared = _run_runtime(prepare, owner_env, f"USER_ID={ids['user']}\n" + _AUDIO_PROFILE_OWNER_SCRIPT)
        assert prepared["complete"], prepared  # nosec B101 -- real canonical owner schema and legacy row.
        with owner.transaction() as conn:
            conn.execute(
                sql.SQL("GRANT SELECT,INSERT,UPDATE,DELETE ON audio_usage_daily,audio_user_tiers TO {}").format(
                    sql.Identifier(pg_restricted_backend.config.pg_user)
                )
            )
    before = _snapshot(owner)
    script = f"USER_ID={ids['user']}\nEXPECTED={2.5 if legacy_present else 0.0}\n" + _AUDIO_PROFILE_SCRIPT
    result = _run_runtime(runtime_dir, env, script)
    assert result["complete"], result  # nosec B101 -- actual cold accounting and authenticated profile chain.
    if legacy_present:
        again = _run_runtime(runtime_dir, env, script)
        assert again["complete"], again  # nosec B101 -- real new-process idempotence, no cache reset.
    after = _snapshot(owner)
    assert {k: v for k, v in before.items() if k != "users"} == {
        k: v for k, v in after.items() if k != "users"
    }  # nosec B101 -- login may change auth timestamps, not schema or grants.


_CONNECTOR_POLICY_OWNER_SCRIPT = """
import asyncio
import json
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.DB_Management.Connectors_DB import ensure_connectors_tables
async def main():
    try:
        pool = await get_db_pool()
        await ensure_connectors_tables(pool, is_postgres=True)
        await pool.execute("INSERT INTO public.org_connector_policy (org_id, enabled_providers, denied_remote_paths, allowed_export_formats, quotas_per_role) VALUES ($1, $2, $3, $4, $5)",
            41, 'notion', '/private/*', 'md', json.dumps({'user': {'max_jobs_per_day': 3}}))
        print('RUNTIME_RESULT=' + json.dumps({'complete': True}))
    finally:
        await reset_db_pool()
asyncio.run(main())
"""

_CONNECTOR_POLICY_RUNTIME_SCRIPT = """
import asyncio
import json
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.External_Sources.connectors_service import get_policy
async def main():
    result = {'complete': False}
    try:
        async with (await get_db_pool()).transaction() as conn:
            policy = await get_policy(conn, 41)
            assert not DENIED
            if PRESENT:
                assert policy['enabled_providers'] == ['notion'] and policy['denied_remote_paths'] == ['/private/*']
                assert policy['quotas_per_role'] == {'user': {'max_jobs_per_day':3}}
                assert await get_policy(conn, 42) == {}
            else:
                assert policy == {}
        result['complete'] = True
    except Exception as error:
        result['error'] = type(error).__name__
        cause = error.__cause__ or error.__context__
        result['denied'] = type(error).__name__ == 'InsufficientPrivilegeError' or type(cause).__name__ == 'InsufficientPrivilegeError'
        result['complete'] = bool(DENIED and result['denied'])
    finally:
        await reset_db_pool()
    print('RUNTIME_RESULT=' + json.dumps(result))
asyncio.run(main())
"""


@pytest.mark.parametrize("policy_state", ["present", "absent", "denied"])
def test_nonowner_connector_policy_reads_without_schema_provisioning(
    migrated_runtime, pg_restricted_backend, policy_state
):
    """Policy reads preserve configured restrictions and never hide access denials."""
    owner, runtime_dir, env, _ids = migrated_runtime
    if policy_state != "absent":
        prepare = runtime_dir.parent / "connector-policy-owner"
        prepare.mkdir()
        owner_env = _runtime_env(prepare, _dsn(owner.config), backend="postgres", mode="multi_user")
        assert _run_runtime(prepare, owner_env, _CONNECTOR_POLICY_OWNER_SCRIPT)[
            "complete"
        ]  # nosec B101 -- canonical owner provisioning.
        if policy_state == "present":
            with owner.transaction() as conn:
                conn.execute(
                    sql.SQL("GRANT SELECT ON public.org_connector_policy TO {}").format(
                        sql.Identifier(pg_restricted_backend.config.pg_user)
                    )
                )
    before = _snapshot(owner)
    result = _run_runtime(
        runtime_dir,
        env,
        f"PRESENT={policy_state != 'absent'}\nDENIED={policy_state == 'denied'}\n" + _CONNECTOR_POLICY_RUNTIME_SCRIPT,
    )
    assert result["complete"], result  # nosec B101 -- real configured, absent and denied policy paths.
    assert _snapshot(owner) == before  # nosec B101 -- no read-time DDL, backfill, user or grant mutation.


_CONNECTOR_POLICY_SAVE_SCRIPT = """
import asyncio
import json
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.External_Sources.connectors_service import upsert_policy
async def main():
    result = {'complete': False}
    try:
        pool = await get_db_pool()
        for maximum in (3, 5):
            policy = await upsert_policy(pool, 41, {'enabled_providers':['notion'], 'quotas_per_role': {'user': {'max_jobs_per_day': maximum}}})
            assert policy['quotas_per_role'] == {'user': {'max_jobs_per_day': maximum}}
        result['complete'] = True
    except Exception as error:
        result['error'] = type(error).__name__
        cause = error.__cause__ or error.__context__
        result['cause'] = type(cause).__name__ if cause else None
    finally:
        await reset_db_pool()
    print('RUNTIME_RESULT=' + json.dumps(result))
asyncio.run(main())
"""


def test_postgres_policy_save_and_update_preserve_nonempty_quotas(migrated_runtime):
    """The normal policy save caller round-trips a real nonempty JSON quota map."""
    owner, runtime_dir, _env, _ids = migrated_runtime
    prepare = runtime_dir.parent / "connector-policy-save-owner"
    prepare.mkdir()
    owner_env = _runtime_env(prepare, _dsn(owner.config), backend="postgres", mode="multi_user")
    result = _run_runtime(prepare, owner_env, _CONNECTOR_POLICY_SAVE_SCRIPT)
    assert result["complete"], result  # nosec B101 -- actual PostgreSQL save and update contract.
