from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.services import admin_router_analytics_service


def _setup_env(tmp_path) -> None:
    os.environ["AUTH_MODE"] = "single_user"
    os.environ["SINGLE_USER_API_KEY"] = "unit-test-router-analytics"
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path / 'users_test_router_analytics_service.db'}"


async def _ensure_router_usage_seed_rows() -> int:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.tests.helpers.authnz_seed import ensure_test_user

    pool = await get_db_pool()
    # The users table is created by UsersDB inside ensure_test_user below.
    # Hand-rolling a cut-down one here is rejected by
    # profile_user_write_guard: a users table without the profile-version
    # columns would silently break the machinery that depends on them.
    await pool.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_usage_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            user_id INTEGER,
            key_id INTEGER,
            endpoint TEXT,
            operation TEXT,
            provider TEXT,
            model TEXT,
            status INTEGER,
            latency_ms INTEGER,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER,
            prompt_cost_usd REAL,
            completion_cost_usd REAL,
            total_cost_usd REAL,
            currency TEXT,
            estimated INTEGER,
            request_id TEXT,
            remote_ip TEXT,
            user_agent TEXT,
            token_name TEXT,
            conversation_id TEXT
        )
        """
    )
    await pool.execute(
        """
        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            key_hash TEXT,
            key_prefix TEXT,
            name TEXT,
            status TEXT DEFAULT 'active'
        )
        """
    )

    cols = {row["name"] for row in await pool.fetchall("PRAGMA table_info(llm_usage_log)")}
    if "remote_ip" not in cols:
        await pool.execute("ALTER TABLE llm_usage_log ADD COLUMN remote_ip TEXT")
    if "user_agent" not in cols:
        await pool.execute("ALTER TABLE llm_usage_log ADD COLUMN user_agent TEXT")
    if "token_name" not in cols:
        await pool.execute("ALTER TABLE llm_usage_log ADD COLUMN token_name TEXT")
    if "conversation_id" not in cols:
        await pool.execute("ALTER TABLE llm_usage_log ADD COLUMN conversation_id TEXT")

    key_cols = {row["name"] for row in await pool.fetchall("PRAGMA table_info(api_keys)")}
    if "llm_budget_day_tokens" not in key_cols:
        await pool.execute("ALTER TABLE api_keys ADD COLUMN llm_budget_day_tokens INTEGER")
    if "llm_budget_month_tokens" not in key_cols:
        await pool.execute("ALTER TABLE api_keys ADD COLUMN llm_budget_month_tokens INTEGER")
    if "llm_budget_day_usd" not in key_cols:
        await pool.execute("ALTER TABLE api_keys ADD COLUMN llm_budget_day_usd REAL")
    if "llm_budget_month_usd" not in key_cols:
        await pool.execute("ALTER TABLE api_keys ADD COLUMN llm_budget_month_usd REAL")

    user_id = await ensure_test_user(
        pool, "router_analytics_user", "router_analytics_user@example.com"
    )
    await pool.execute(
        """
        INSERT OR REPLACE INTO api_keys (
            id, user_id, key_hash, name, status,
            llm_budget_day_tokens, llm_budget_month_tokens, llm_budget_day_usd, llm_budget_month_usd
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        11,
        user_id,
        "hash-11",
        "Admin",
        "active",
        100,
        1000,
        1.0,
        10.0,
    )
    await pool.execute(
        """
        INSERT OR REPLACE INTO api_keys (
            id, user_id, key_hash, name, status,
            llm_budget_day_tokens, llm_budget_month_tokens, llm_budget_day_usd, llm_budget_month_usd
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        12,
        user_id,
        "hash-12",
        "Ops",
        "active",
        30,
        100,
        0.05,
        1.0,
    )
    await pool.execute(
        """
        INSERT OR REPLACE INTO api_keys (
            id, user_id, key_hash, name, status
        ) VALUES (?, ?, ?, ?, ?)
        """,
        13,
        user_id,
        "hash-13",
        "NoBudget",
        "active",
    )

    # Fixed UTC timestamps keep range math deterministic for tests.
    await pool.execute(
        """
        INSERT INTO llm_usage_log (
            ts, user_id, key_id, endpoint, operation, provider, model, status, latency_ms,
            prompt_tokens, completion_tokens, total_tokens, prompt_cost_usd, completion_cost_usd, total_cost_usd,
            currency, estimated, request_id, remote_ip, user_agent, token_name, conversation_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        "2026-03-01 10:20:00",
        user_id,
        11,
        "/api/v1/chat/completions",
        "chat",
        "openai",
        "gpt-4o-mini",
        200,
        1000,
        10,
        20,
        30,
        0.01,
        0.02,
        0.03,
        "USD",
        0,
        "req-1",
        "127.0.0.1",
        "curl/8.8.0",
        "Admin",
        "conv-1",
    )
    await pool.execute(
        """
        INSERT INTO llm_usage_log (
            ts, user_id, key_id, endpoint, operation, provider, model, status, latency_ms,
            prompt_tokens, completion_tokens, total_tokens, prompt_cost_usd, completion_cost_usd, total_cost_usd,
            currency, estimated, request_id, remote_ip, user_agent, token_name, conversation_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        "2026-03-01 10:25:00",
        user_id,
        12,
        "/api/v1/chat/completions",
        "chat",
        "groq",
        "llama-3.3-70b",
        200,
        500,
        30,
        10,
        40,
        0.03,
        0.01,
        0.04,
        "USD",
        0,
        "req-2",
        "10.0.0.5",
        "python-httpx/1.0",
        "Ops",
        "conv-2",
    )
    await pool.execute(
        """
        INSERT INTO llm_usage_log (
            ts, user_id, key_id, endpoint, operation, provider, model, status, latency_ms,
            prompt_tokens, completion_tokens, total_tokens, prompt_cost_usd, completion_cost_usd, total_cost_usd,
            currency, estimated, request_id, remote_ip, user_agent, token_name, conversation_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        "2026-03-01 10:26:00",
        user_id,
        12,
        "/api/v1/chat/completions",
        "chat",
        "groq",
        "llama-3.3-70b",
        500,
        200,
        5,
        0,
        5,
        0.0,
        0.0,
        0.0,
        "USD",
        1,
        "req-3",
        "10.0.0.5",
        "python-httpx/1.0",
        "Ops",
        "conv-2",
    )
    await pool.execute(
        """
        INSERT INTO llm_usage_log (
            ts, user_id, key_id, endpoint, operation, provider, model, status, latency_ms,
            prompt_tokens, completion_tokens, total_tokens, prompt_cost_usd, completion_cost_usd, total_cost_usd,
            currency, estimated, request_id, remote_ip, user_agent, token_name, conversation_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        "2026-03-01 10:27:00",
        user_id,
        13,
        "/api/v1/chat/completions",
        "chat",
        "anthropic",
        "claude-3.5",
        503,
        300,
        7,
        2,
        9,
        0.0,
        0.0,
        0.0,
        "USD",
        1,
        "req-4",
        None,
        None,
        None,
        "conv-3",
    )
    return user_id


def _single_user_principal(user_id: int) -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=user_id,
        subject="single_user",
        roles=["admin"],
        permissions=["*"],
        is_admin=True,
    )


def test_build_usage_where_clause_org_scope_uses_exists_predicate():
    start = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
    end = datetime(2026, 3, 1, 11, 0, tzinfo=timezone.utc)
    join_clause, where_clause, params = admin_router_analytics_service._build_usage_where_clause(
        is_pg=False,
        start=start,
        end=end,
        provider=None,
        model=None,
        token_id=None,
        org_ids=[101, 202],
    )

    assert join_clause == ""
    assert "EXISTS (SELECT 1 FROM org_members om" in where_clause
    assert "JOIN org_members" not in where_clause
    assert params[-2:] == [101, 202]


@pytest.mark.asyncio
async def test_router_analytics_status_and_breakdowns_sqlite(monkeypatch, tmp_path):
    _setup_env(tmp_path)
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    await reset_db_pool()
    reset_settings()
    await reset_session_manager()
    user_id = await _ensure_router_usage_seed_rows()
    pool = await get_db_pool()
    principal = _single_user_principal(user_id)

    monkeypatch.setattr(
        admin_router_analytics_service,
        "_utcnow",
        lambda: datetime(2026, 3, 1, 10, 30, tzinfo=timezone.utc),
    )

    status = await admin_router_analytics_service.get_router_analytics_status(
        principal=principal,
        db=pool,
        range_value="1h",
        org_id=None,
        provider=None,
        model=None,
        token_id=None,
        granularity=None,
    )

    assert status.kpis.requests == 4
    assert status.kpis.prompt_tokens == 52
    assert status.kpis.generated_tokens == 32
    assert status.kpis.total_tokens == 84
    assert status.kpis.avg_latency_ms == pytest.approx(500.0)
    assert status.kpis.avg_gen_toks_per_s == pytest.approx(16.0)
    assert status.providers_available == 3
    assert status.providers_online == 2
    assert len(status.series) == 4

    breakdowns = await admin_router_analytics_service.get_router_analytics_status_breakdowns(
        principal=principal,
        db=pool,
        range_value="1h",
        org_id=None,
        provider=None,
        model=None,
        token_id=None,
        granularity=None,
    )
    providers = {row.key: row.requests for row in breakdowns.providers}
    assert providers["groq"] == 2
    assert providers["openai"] == 1
    assert providers["anthropic"] == 1
    assert any(row.key == "unknown" and row.requests == 1 for row in breakdowns.remote_ips)
    assert any(row.key == "unknown" and row.requests == 1 for row in breakdowns.user_agents)
    assert any(row.key == "unknown" and row.requests == 1 for row in breakdowns.token_names)

    meta = await admin_router_analytics_service.get_router_analytics_meta(
        principal=principal,
        db=pool,
    )
    provider_values = {option.value for option in meta.providers}
    token_values = {option.value for option in meta.tokens}
    token_labels_by_id = {option.key_id: option.label for option in meta.tokens}
    assert {"openai", "groq", "anthropic"} <= provider_values
    assert {"11", "12", "13"} <= token_values
    assert token_labels_by_id[11] == "Admin"
    assert token_labels_by_id[12] == "Ops"


@pytest.mark.asyncio
async def test_router_analytics_status_honors_provider_and_token_filters(monkeypatch, tmp_path):
    _setup_env(tmp_path)
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    await reset_db_pool()
    reset_settings()
    await reset_session_manager()
    user_id = await _ensure_router_usage_seed_rows()
    pool = await get_db_pool()
    principal = _single_user_principal(user_id)

    monkeypatch.setattr(
        admin_router_analytics_service,
        "_utcnow",
        lambda: datetime(2026, 3, 1, 10, 30, tzinfo=timezone.utc),
    )

    by_provider = await admin_router_analytics_service.get_router_analytics_status(
        principal=principal,
        db=pool,
        range_value="1h",
        org_id=None,
        provider="groq",
        model=None,
        token_id=None,
        granularity=None,
    )
    assert by_provider.kpis.requests == 2
    assert by_provider.kpis.prompt_tokens == 35

    by_token = await admin_router_analytics_service.get_router_analytics_status(
        principal=principal,
        db=pool,
        range_value="1h",
        org_id=None,
        provider=None,
        model=None,
        token_id=11,
        granularity=None,
    )
    assert by_token.kpis.requests == 1
    assert by_token.kpis.total_tokens == 30


@pytest.mark.asyncio
async def test_router_analytics_quota_returns_key_budget_utilization(monkeypatch, tmp_path):
    _setup_env(tmp_path)
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    await reset_db_pool()
    reset_settings()
    await reset_session_manager()
    user_id = await _ensure_router_usage_seed_rows()
    pool = await get_db_pool()
    principal = _single_user_principal(user_id)

    monkeypatch.setattr(
        admin_router_analytics_service,
        "_utcnow",
        lambda: datetime(2026, 3, 1, 10, 30, tzinfo=timezone.utc),
    )

    quota = await admin_router_analytics_service.get_router_analytics_quota(
        principal=principal,
        db=pool,
        range_value="1h",
        org_id=None,
        provider=None,
        model=None,
        token_id=None,
        granularity=None,
    )

    assert quota.summary.keys_total >= 2
    assert quota.summary.keys_over_budget >= 1
    keyed = {row.key_id: row for row in quota.items}
    assert 12 in keyed
    assert keyed[12].over_budget is True
    assert keyed[12].day_tokens is not None
    assert keyed[12].day_tokens.exceeded is True
    assert keyed[12].day_tokens.utilization_pct == pytest.approx(150.0)


@pytest.mark.asyncio
async def test_router_analytics_providers_returns_provider_health_breakdown(monkeypatch, tmp_path):
    _setup_env(tmp_path)
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    await reset_db_pool()
    reset_settings()
    await reset_session_manager()
    user_id = await _ensure_router_usage_seed_rows()
    pool = await get_db_pool()
    principal = _single_user_principal(user_id)

    monkeypatch.setattr(
        admin_router_analytics_service,
        "_utcnow",
        lambda: datetime(2026, 3, 1, 10, 30, tzinfo=timezone.utc),
    )

    providers = await admin_router_analytics_service.get_router_analytics_providers(
        principal=principal,
        db=pool,
        range_value="1h",
        org_id=None,
        provider=None,
        model=None,
        token_id=None,
        granularity=None,
    )

    assert providers.summary.providers_total == 3
    assert providers.summary.providers_online == 2
    assert providers.summary.failover_events == 2

    keyed = {row.provider: row for row in providers.items}
    assert keyed["groq"].requests == 2
    assert keyed["groq"].errors == 1
    assert keyed["groq"].online is True
    assert keyed["groq"].success_rate_pct == pytest.approx(50.0)
    assert keyed["groq"].avg_latency_ms == pytest.approx(350.0)

    assert keyed["anthropic"].requests == 1
    assert keyed["anthropic"].online is False
    assert keyed["anthropic"].success_rate_pct == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_router_analytics_access_returns_token_ip_and_user_agent_breakdown(monkeypatch, tmp_path):
    _setup_env(tmp_path)
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    await reset_db_pool()
    reset_settings()
    await reset_session_manager()
    user_id = await _ensure_router_usage_seed_rows()
    pool = await get_db_pool()
    principal = _single_user_principal(user_id)

    monkeypatch.setattr(
        admin_router_analytics_service,
        "_utcnow",
        lambda: datetime(2026, 3, 1, 10, 30, tzinfo=timezone.utc),
    )

    access = await admin_router_analytics_service.get_router_analytics_access(
        principal=principal,
        db=pool,
        range_value="1h",
        org_id=None,
        provider=None,
        model=None,
        token_id=None,
        granularity=None,
    )

    assert access.summary.token_names_total >= 3
    assert access.summary.remote_ips_total >= 2
    assert access.summary.user_agents_total >= 2
    assert access.summary.anonymous_requests == 1

    token_names = {row.key: row.requests for row in access.token_names}
    assert token_names["Ops"] == 2
    assert token_names["unknown"] == 1

    remote_ips = {row.key: row.requests for row in access.remote_ips}
    assert remote_ips["10.0.0.5"] == 2
    assert remote_ips["unknown"] == 1

    user_agents = {row.key: row.requests for row in access.user_agents}
    assert user_agents["python-httpx/1.0"] == 2
    assert user_agents["unknown"] == 1


@pytest.mark.asyncio
async def test_router_analytics_network_returns_ip_endpoint_and_operation_breakdown(monkeypatch, tmp_path):
    _setup_env(tmp_path)
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    await reset_db_pool()
    reset_settings()
    await reset_session_manager()
    user_id = await _ensure_router_usage_seed_rows()
    pool = await get_db_pool()
    principal = _single_user_principal(user_id)

    monkeypatch.setattr(
        admin_router_analytics_service,
        "_utcnow",
        lambda: datetime(2026, 3, 1, 10, 30, tzinfo=timezone.utc),
    )

    network = await admin_router_analytics_service.get_router_analytics_network(
        principal=principal,
        db=pool,
        range_value="1h",
        org_id=None,
        provider=None,
        model=None,
        token_id=None,
        granularity=None,
    )

    assert network.summary.remote_ips_total >= 2
    assert network.summary.endpoints_total >= 1
    assert network.summary.operations_total >= 1
    assert network.summary.error_requests == 2

    remote_ips = {row.key: row.requests for row in network.remote_ips}
    assert remote_ips["10.0.0.5"] == 2
    assert remote_ips["unknown"] == 1

    endpoints = {row.key: row.requests for row in network.endpoints}
    assert endpoints["/api/v1/chat/completions"] == 4

    operations = {row.key: row.requests for row in network.operations}
    assert operations["chat"] == 4


@pytest.mark.asyncio
async def test_router_analytics_models_returns_model_health_breakdown(monkeypatch, tmp_path):
    _setup_env(tmp_path)
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    await reset_db_pool()
    reset_settings()
    await reset_session_manager()
    user_id = await _ensure_router_usage_seed_rows()
    pool = await get_db_pool()
    principal = _single_user_principal(user_id)

    monkeypatch.setattr(
        admin_router_analytics_service,
        "_utcnow",
        lambda: datetime(2026, 3, 1, 10, 30, tzinfo=timezone.utc),
    )

    models_payload = await admin_router_analytics_service.get_router_analytics_models(
        principal=principal,
        db=pool,
        range_value="1h",
        org_id=None,
        provider=None,
        model=None,
        token_id=None,
        granularity=None,
    )

    assert models_payload.summary.models_total >= 3
    assert models_payload.summary.models_online == 2
    assert models_payload.summary.providers_total == 3
    assert models_payload.summary.error_requests == 2

    keyed = {(row.provider, row.model): row for row in models_payload.items}
    assert keyed[("groq", "llama-3.3-70b")].requests == 2
    assert keyed[("groq", "llama-3.3-70b")].errors == 1
    assert keyed[("groq", "llama-3.3-70b")].online is True
    assert keyed[("groq", "llama-3.3-70b")].success_rate_pct == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_router_analytics_conversations_returns_conversation_breakdown(monkeypatch, tmp_path):
    _setup_env(tmp_path)
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    await reset_db_pool()
    reset_settings()
    await reset_session_manager()
    user_id = await _ensure_router_usage_seed_rows()
    pool = await get_db_pool()
    principal = _single_user_principal(user_id)

    monkeypatch.setattr(
        admin_router_analytics_service,
        "_utcnow",
        lambda: datetime(2026, 3, 1, 10, 30, tzinfo=timezone.utc),
    )

    conversations_payload = await admin_router_analytics_service.get_router_analytics_conversations(
        principal=principal,
        db=pool,
        range_value="1h",
        org_id=None,
        provider=None,
        model=None,
        token_id=None,
        granularity=None,
    )

    assert conversations_payload.summary.conversations_total == 3
    assert conversations_payload.summary.active_conversations == 2
    assert conversations_payload.summary.avg_requests_per_conversation == pytest.approx(4.0 / 3.0)
    assert conversations_payload.summary.error_requests == 2

    keyed = {row.conversation_id: row for row in conversations_payload.items}
    assert keyed["conv-2"].requests == 2
    assert keyed["conv-2"].errors == 1
    assert keyed["conv-2"].success_rate_pct == pytest.approx(50.0)
    assert keyed["conv-2"].avg_latency_ms == pytest.approx(350.0)
    assert keyed["conv-3"].success_rate_pct == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_router_analytics_log_returns_recent_rows(monkeypatch, tmp_path):
    _setup_env(tmp_path)
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    await reset_db_pool()
    reset_settings()
    await reset_session_manager()
    user_id = await _ensure_router_usage_seed_rows()
    pool = await get_db_pool()
    principal = _single_user_principal(user_id)

    monkeypatch.setattr(
        admin_router_analytics_service,
        "_utcnow",
        lambda: datetime(2026, 3, 1, 10, 30, tzinfo=timezone.utc),
    )

    log_payload = await admin_router_analytics_service.get_router_analytics_log(
        principal=principal,
        db=pool,
        range_value="1h",
        org_id=None,
        provider=None,
        model=None,
        token_id=None,
        granularity=None,
    )

    assert log_payload.summary.requests_total == 4
    assert log_payload.summary.error_requests == 2
    assert log_payload.summary.estimated_requests == 2
    assert log_payload.summary.request_ids_total == 4
    assert len(log_payload.items) == 4
    assert log_payload.items[0].request_id == "req-4"
    assert log_payload.items[0].status == 503
    assert log_payload.items[0].error is True
    assert log_payload.items[0].estimated is True


@pytest.mark.asyncio
async def test_router_analytics_log_falls_back_for_legacy_schema(monkeypatch, tmp_path):
    _setup_env(tmp_path)
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    await reset_db_pool()
    reset_settings()
    await reset_session_manager()
    user_id = await _ensure_router_usage_seed_rows()
    pool = await get_db_pool()
    principal = _single_user_principal(user_id)

    original_fetch_rows = admin_router_analytics_service._fetch_rows

    async def _fetch_rows_legacy_query_error(db_obj, sql: str, params: list[object]):
        if "TRIM(conversation_id)" in sql:
            raise ValueError("no such column: conversation_id")
        return await original_fetch_rows(db_obj, sql, params)

    monkeypatch.setattr(admin_router_analytics_service, "_fetch_rows", _fetch_rows_legacy_query_error)

    monkeypatch.setattr(
        admin_router_analytics_service,
        "_utcnow",
        lambda: datetime(2026, 3, 1, 10, 30, tzinfo=timezone.utc),
    )

    log_payload = await admin_router_analytics_service.get_router_analytics_log(
        principal=principal,
        db=pool,
        range_value="1h",
        org_id=None,
        provider=None,
        model=None,
        token_id=None,
        granularity=None,
    )

    assert log_payload.summary.requests_total == 4
    assert log_payload.partial is True
    assert log_payload.warnings is not None
    assert "legacy_llm_usage_log_schema_missing_router_dimensions" in log_payload.warnings
    assert len(log_payload.items) == 4
    assert log_payload.items[0].conversation_id == "unknown"
    assert log_payload.items[0].token_name == "unknown"
    assert log_payload.items[0].remote_ip == "unknown"
    assert log_payload.items[0].user_agent == "unknown"
