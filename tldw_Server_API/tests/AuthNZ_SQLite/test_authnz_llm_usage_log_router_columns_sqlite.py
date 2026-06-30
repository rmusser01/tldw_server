from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_llm_usage_log_has_router_analytics_columns_sqlite(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    cols = {row["name"] for row in await pool.fetchall("PRAGMA table_info(llm_usage_log)")}
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

    indexes = {row["name"] for row in await pool.fetchall("PRAGMA index_list(llm_usage_log)")}
    assert "idx_llm_usage_log_remote_ip_ts" in indexes
    assert "idx_llm_usage_log_token_name_ts" in indexes
