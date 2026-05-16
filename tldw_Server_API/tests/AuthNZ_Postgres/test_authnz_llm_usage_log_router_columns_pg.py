from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_llm_usage_log_has_router_analytics_columns_pg(test_db_pool):
    from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import ensure_usage_tables_pg

    assert await ensure_usage_tables_pg(test_db_pool)

    col_rows = await test_db_pool.fetch(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'llm_usage_log'
        """
    )
    cols = {str(row["column_name"]) for row in col_rows}
    assert {
        "remote_ip",
        "user_agent",
        "token_name",
        "conversation_id",
        "cached_input_tokens",
        "cache_write_input_tokens",
        "cache_read_input_tokens",
        "billable_input_tokens",
        "reasoning_tokens",
        "choice_count",
        "estimate_source",
        "prompt_fingerprint",
        "prompt_fingerprint_version",
        "world_book_fingerprint",
        "raw_usage_metadata_json",
    }.issubset(cols)

    idx_rows = await test_db_pool.fetch(
        """
        SELECT indexname
        FROM pg_indexes
        WHERE schemaname = 'public' AND tablename = 'llm_usage_log'
        """
    )
    idx_names = {str(row["indexname"]) for row in idx_rows}
    assert "idx_llm_usage_log_remote_ip_ts" in idx_names
    assert "idx_llm_usage_log_token_name_ts" in idx_names
