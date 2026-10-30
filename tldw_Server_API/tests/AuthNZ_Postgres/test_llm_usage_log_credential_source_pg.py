"""PostgreSQL: llm_usage_log.credential_source exists, is written, and the quota counters skip ``user`` rows (TASK-13955)."""

from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import ensure_usage_tables_pg
from tldw_Server_API.app.core.AuthNZ.repos.usage_repo import AuthnzUsageRepo
from tldw_Server_API.tests.helpers.authnz_seed import ensure_test_user


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
async def test_the_hosted_plans_llm_tokens_leave_out_a_members_own_key_pg(test_db_pool, monkeypatch) -> None:
    """The org's monthly LLM tokens count every source but the member's own key on PostgreSQL (TASK-13955)."""
    from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
    from tldw_Server_API.app.core.Billing.enforcement import BillingEnforcer

    assert await ensure_usage_tables_pg(test_db_pool)
    user_id = await ensure_test_user(test_db_pool, "byok-plan-pg")
    orgs = AuthnzOrgsTeamsRepo(test_db_pool)
    org_id = int((await orgs.create_organization(name="byok-plan-pg", owner_user_id=user_id))["id"])
    await orgs.add_org_member(org_id=org_id, user_id=user_id)
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
