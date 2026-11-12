"""PostgreSQL: llm_usage_log.credential_source exists, is written, and the quota counters skip ``user`` rows (TASK-13955)."""

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    TrustedMembershipReason,
    TrustedMembershipWriteContext,
)
from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import ensure_usage_tables_pg
from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import AuthnzUsageRepo
from tldw_Server_API.tests.helpers.authnz_seed import ensure_test_user, unmanaged_authnz_pg_connection


@pytest.mark.asyncio
async def test_credential_source_is_stored_and_user_rows_are_not_counted_pg(test_db_pool) -> None:
    """The column is added idempotently; a ``user`` row is stored but not summed."""
    assert await ensure_usage_tables_pg(test_db_pool)
    assert await ensure_usage_tables_pg(test_db_pool)
    columns = await test_db_pool.fetch(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = 'llm_usage_log'"
    )
    assert "credential_source" in {str(row["column_name"]) for row in columns}

    user_id = await ensure_test_user(test_db_pool, "byok-quota-pg")
    repo = AuthnzUsageRepo(test_db_pool)
    for source, tokens, cost in (("user", 1000, 9.0), ("server_default", 30, 0.3), (None, 5, 0.05)):
        await repo.insert_llm_usage_log(
            user_id=user_id, key_id=None, endpoint="internal", operation="chat", provider="openai",
            model="gpt-4o", status=200, latency_ms=1, prompt_tokens=tokens, completion_tokens=0,
            total_tokens=tokens, prompt_cost_usd=cost, completion_cost_usd=0.0, total_cost_usd=cost,
            credential_source=source,
        )

    stored = await test_db_pool.fetch(
        "SELECT credential_source FROM llm_usage_log WHERE user_id = $1 ORDER BY id", user_id
    )
    assert [row["credential_source"] for row in stored] == ["user", "server_default", None]
    since = datetime.now(timezone.utc) - timedelta(days=1)
    assert await repo.sum_user_llm_tokens_since(user_id=user_id, since=since) == 35.0
    assert await repo.sum_user_llm_cost_since(user_id=user_id, since=since) == pytest.approx(0.35)


@pytest.mark.asyncio
async def test_postgres_names_the_missing_column_so_the_counters_fall_back_pg(test_db_pool) -> None:
    """PostgreSQL's undefined-column error names ``credential_source``, which is how the counters recognise it."""
    from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import missing_credential_source

    assert await ensure_usage_tables_pg(test_db_pool)
    with pytest.raises(Exception) as missing:  # noqa: PT011 - whatever the driver raises
        await test_db_pool.fetchval("SELECT COUNT(credential_source) FROM usage_log")  # usage_log has no such column
    assert missing_credential_source(missing.value, "a PostgreSQL test")


@pytest.mark.asyncio
async def test_a_usage_table_without_the_column_keeps_the_row_and_counts_it_pg(test_db_pool) -> None:
    """On a PostgreSQL usage table from before the column, an own-key row is written without its source and counts."""
    from tldw_Server_API.app.core.AuthNZ.database import DatabasePool

    assert await ensure_usage_tables_pg(test_db_pool)
    user_id = await ensure_test_user(test_db_pool, "byok-unmigrated-pg")
    schema = f"byok_unmigrated_{uuid.uuid4().hex[:8]}"
    # This test's own copy of the table as an older release left it: no column (and no id, which no statement names).
    async with unmanaged_authnz_pg_connection() as conn:
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        await conn.execute(f'CREATE TABLE "{schema}".llm_usage_log (LIKE public.llm_usage_log)')
        await conn.execute(f'ALTER TABLE "{schema}".llm_usage_log DROP COLUMN id, DROP COLUMN credential_source')
    url = f"{test_db_pool.settings.DATABASE_URL}?search_path={schema}"
    unmigrated = DatabasePool(test_db_pool.settings.model_copy(update={"DATABASE_URL": url}))
    try:
        repo = AuthnzUsageRepo(unmigrated)
        await repo.insert_llm_usage_log(
            user_id=user_id, key_id=None, endpoint="internal", operation="chat", provider="openai",
            model="gpt-4o", status=200, latency_ms=1, prompt_tokens=1000, completion_tokens=0,
            total_tokens=1000, prompt_cost_usd=9.0, completion_cost_usd=0.0, total_cost_usd=9.0,
            cached_input_tokens=3, credential_source="user",
        )
        # The retry is the full statement without the column, not the pre-088 fallback, which drops cache tokens.
        assert await unmigrated.fetchval("SELECT cached_input_tokens FROM llm_usage_log") == 3
        since = datetime.now(timezone.utc) - timedelta(days=1)
        assert await repo.sum_user_llm_tokens_since(user_id=user_id, since=since) == 1000.0  # the row counts
    finally:
        await unmigrated.close()
        async with unmanaged_authnz_pg_connection() as conn:
            await conn.execute(f'DROP SCHEMA "{schema}" CASCADE')  # this test's schema and table only


@pytest.mark.asyncio
async def test_the_hosted_plans_llm_tokens_leave_out_a_members_own_key_pg(test_db_pool, monkeypatch) -> None:
    """The org's monthly LLM tokens count every source but the member's own key on PostgreSQL (TASK-13955)."""
    from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
    from tldw_Server_API.app.core.Billing.enforcement import BillingEnforcer

    assert await ensure_usage_tables_pg(test_db_pool)
    user_id = await ensure_test_user(test_db_pool, "byok-plan-pg")
    orgs = AuthnzOrgsTeamsRepo(test_db_pool)
    org = await orgs.create_organization_with_owner_membership(
        name="byok-plan-pg",
        owner_user_id=user_id,
        context=TrustedMembershipWriteContext(
            trusted_reason=TrustedMembershipReason.BOOTSTRAP,
        ),
    )
    org_id = int(org["id"])
    repo = AuthnzUsageRepo(test_db_pool)
    for source, tokens in (("user", 1000), ("user_unmetered", 40), ("server_default", 300), (None, 5)):
        await repo.insert_llm_usage_log(
            user_id=user_id, key_id=None, endpoint="internal", operation="chat", provider="openai",
            model="gpt-4o", status=200, latency_ms=1, prompt_tokens=tokens, completion_tokens=0,
            total_tokens=tokens, prompt_cost_usd=0.0, completion_cost_usd=0.0, total_cost_usd=0.0,
            credential_source=source,
        )

    async def _pool():
        return test_db_pool

    monkeypatch.setattr("tldw_Server_API.app.core.AuthNZ.database.get_db_pool", _pool)
    assert await BillingEnforcer()._get_llm_tokens_month(org_id=org_id) == 345
