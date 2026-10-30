"""Postgres storage invariants for provider credentials (TASK-13950.17 PR A1, characterization)."""

import base64

import pytest

from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
    ensure_byok_oauth_state_pg,
    ensure_byok_validation_runs_table_pg,
    ensure_llm_provider_overrides_pg,
    ensure_org_provider_secrets_pg,
    ensure_user_provider_secrets_pg,
)
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.tests.AuthNZ_Postgres.test_byok_oauth_endpoints_pg import _create_postgres_runtime_scope
from tldw_Server_API.tests.helpers.provider_credential_invariants import (
    assert_org_secret_encrypted_and_scope_bound,
    assert_user_secret_encrypted_and_owner_scoped,
)

pytestmark = pytest.mark.integration

PROVIDER_TABLES = (
    "byok_oauth_state",
    "byok_validation_runs",
    "llm_provider_overrides",
    "org_provider_secrets",
    "user_provider_secrets",
)


@pytest.fixture
def byok_key(monkeypatch):
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", base64.b64encode(b"k" * 32).decode("ascii"))
    reset_settings()
    yield
    reset_settings()


@pytest.mark.asyncio
async def test_user_secret_is_encrypted_at_rest_and_owner_scoped_postgres(test_db_pool, byok_key) -> None:
    owner_id, _, _ = await _create_postgres_runtime_scope(test_db_pool)
    other_id, _, _ = await _create_postgres_runtime_scope(test_db_pool)
    await assert_user_secret_encrypted_and_owner_scoped(test_db_pool, owner_id=owner_id, other_id=other_id)


@pytest.mark.asyncio
async def test_org_secret_is_encrypted_at_rest_and_scope_bound_postgres(test_db_pool, byok_key) -> None:
    _, org_id, _ = await _create_postgres_runtime_scope(test_db_pool)
    _, other_org_id, _ = await _create_postgres_runtime_scope(test_db_pool)
    await assert_org_secret_encrypted_and_scope_bound(test_db_pool, org_id=org_id, other_org_id=other_org_id)


@pytest.mark.asyncio
async def test_provider_credential_tables_keep_their_rls_posture_postgres(test_db_pool) -> None:
    """Pins an absence: RLS is off and no policy exists on these tables today.

    The move cannot change that, because the DDL stays in the AuthNZ migrations. Isolation is
    the repositories' owner predicates, which the two scope tests above prove. The three owned
    tables are listed in Helper_Scripts/ci/rls_coverage_baseline.txt. Adding RLS to them is a
    deliberate change that edits this test.
    """
    for ensure in (
        ensure_user_provider_secrets_pg,
        ensure_org_provider_secrets_pg,
        ensure_byok_oauth_state_pg,
        ensure_llm_provider_overrides_pg,
        ensure_byok_validation_runs_table_pg,
    ):
        assert await ensure(test_db_pool) is True
    rows = await test_db_pool.fetchall(
        "SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class WHERE relkind = 'r' AND relname IN "
        "('byok_oauth_state', 'byok_validation_runs', 'llm_provider_overrides', 'org_provider_secrets', 'user_provider_secrets')"
    )
    assert {row["relname"]: (row["relrowsecurity"], row["relforcerowsecurity"]) for row in rows} == dict.fromkeys(PROVIDER_TABLES, (False, False))
    assert (
        await test_db_pool.fetchall(
            "SELECT tablename FROM pg_policies WHERE tablename IN "
            "('byok_oauth_state', 'byok_validation_runs', 'llm_provider_overrides', 'org_provider_secrets', 'user_provider_secrets')"
        )
        == []
    )
