from __future__ import annotations

import contextlib
import json
import os
import uuid
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Usage.usage_tracker import log_llm_usage


async def _ensure_llm_tables(pool):
    if pool.pool:
        await pool.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_usage_log (
                id SERIAL PRIMARY KEY,
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
                prompt_cost_usd DOUBLE PRECISION,
                completion_cost_usd DOUBLE PRECISION,
                total_cost_usd DOUBLE PRECISION,
                currency TEXT,
                estimated BOOLEAN,
                request_id TEXT,
                remote_ip TEXT,
                user_agent TEXT,
                token_name TEXT,
                conversation_id TEXT
            )
            """
        )
    else:
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


async def _ensure_llm_cache_columns(pool):
    columns = [
        ("cached_input_tokens", "INTEGER"),
        ("cache_write_input_tokens", "INTEGER"),
        ("cache_read_input_tokens", "INTEGER"),
        ("billable_input_tokens", "INTEGER"),
        ("reasoning_tokens", "INTEGER"),
        ("choice_count", "INTEGER"),
        ("estimate_source", "TEXT"),
        ("prompt_fingerprint", "TEXT"),
        ("prompt_fingerprint_version", "TEXT"),
        ("world_book_fingerprint", "TEXT"),
        ("raw_usage_metadata_json", "TEXT"),
    ]
    for column, column_type in columns:
        if pool.pool:
            await pool.execute(f"ALTER TABLE llm_usage_log ADD COLUMN IF NOT EXISTS {column} {column_type}")
        else:
            with contextlib.suppress(Exception):
                await pool.execute(f"ALTER TABLE llm_usage_log ADD COLUMN {column} {column_type}")


@pytest.mark.asyncio
async def test_usage_tracker_inserts_sqlite(monkeypatch):
    # Force SQLite single-user temp DB
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "ut-key-" + uuid.uuid4().hex)
    dburl = f"sqlite:///./Databases/users_test_ut_{uuid.uuid4().hex}.sqlite"
    monkeypatch.setenv("DATABASE_URL", dburl)

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager

    reset_settings()
    await reset_db_pool()
    await reset_session_manager()

    pool = await get_db_pool()
    await _ensure_llm_tables(pool)

    # Insert a usage row
    await log_llm_usage(
        user_id=1,
        key_id=None,
        endpoint="POST:/api/v1/chat/completions",
        operation="chat",
        provider="openai",
        model="gpt-3.5-turbo",
        status=200,
        latency_ms=123,
        prompt_tokens=1000,
        completion_tokens=500,
        request_id="req-xyz",
    )

    # Verify row exists with costs populated
    if pool.pool:
        row = await pool.fetchone("SELECT prompt_tokens, completion_tokens, total_cost_usd FROM llm_usage_log WHERE request_id = $1", "req-xyz")
    else:
        row = await pool.fetchone("SELECT prompt_tokens, completion_tokens, total_cost_usd FROM llm_usage_log WHERE request_id = ?", "req-xyz")

    assert row is not None
    pt = int(row["prompt_tokens"]) if isinstance(row, dict) else int(row[0])
    ct = int(row["completion_tokens"]) if isinstance(row, dict) else int(row[1])
    cost = float(row["total_cost_usd"]) if isinstance(row, dict) else float(row[2])
    assert pt == 1000 and ct == 500
    assert cost > 0.0


@pytest.mark.asyncio
async def test_usage_tracker_persists_normalized_cache_fields(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "ut-key-" + uuid.uuid4().hex)
    monkeypatch.setenv(
        "PRICING_OVERRIDES",
        json.dumps(
            {
                "anthropic": {
                    "claude-3-sonnet": {
                        "prompt": 0.010,
                        "completion": 0.030,
                        "cache_read": 0.001,
                        "cache_write": 0.005,
                    }
                }
            }
        ),
    )
    dburl = f"sqlite:///./Databases/users_test_ut_{uuid.uuid4().hex}.sqlite"
    monkeypatch.setenv("DATABASE_URL", dburl)

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.Usage.pricing_catalog import reset_pricing_catalog

    reset_settings()
    reset_pricing_catalog()
    await reset_db_pool()
    await reset_session_manager()

    pool = await get_db_pool()
    await _ensure_llm_tables(pool)
    await _ensure_llm_cache_columns(pool)

    await log_llm_usage(
        user_id=1,
        key_id=None,
        endpoint="POST:/api/v1/chat/completions",
        operation="chat",
        provider="anthropic",
        model="claude-3-sonnet",
        status=200,
        latency_ms=123,
        prompt_tokens=100,
        completion_tokens=25,
        total_tokens=125,
        request_id="req-cache-fields",
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 25,
            "cache_creation_input_tokens": 10,
            "cache_read_input_tokens": 70,
            "api_key": "sk-secret",
        },
        choice_count=2,
        estimate_source="provider_usage",
        prompt_fingerprint="prompt-v1:abc",
        prompt_fingerprint_version="prompt-v1",
        world_book_fingerprint="world-v1:def",
    )

    if pool.pool:
        row = await pool.fetchone(
            """
            SELECT cached_input_tokens, cache_write_input_tokens, cache_read_input_tokens,
                   billable_input_tokens, reasoning_tokens, choice_count, estimate_source,
                   prompt_fingerprint, prompt_fingerprint_version, world_book_fingerprint,
                   raw_usage_metadata_json, prompt_cost_usd, completion_cost_usd, total_cost_usd
            FROM llm_usage_log WHERE request_id = $1
            """,
            "req-cache-fields",
        )
    else:
        row = await pool.fetchone(
            """
            SELECT cached_input_tokens, cache_write_input_tokens, cache_read_input_tokens,
                   billable_input_tokens, reasoning_tokens, choice_count, estimate_source,
                   prompt_fingerprint, prompt_fingerprint_version, world_book_fingerprint,
                   raw_usage_metadata_json, prompt_cost_usd, completion_cost_usd, total_cost_usd
            FROM llm_usage_log WHERE request_id = ?
            """,
            "req-cache-fields",
        )

    assert row is not None
    assert int(row["cached_input_tokens"]) == 70
    assert int(row["cache_write_input_tokens"]) == 10
    assert int(row["cache_read_input_tokens"]) == 70
    assert int(row["billable_input_tokens"]) == 20
    assert int(row["reasoning_tokens"]) == 0
    assert int(row["choice_count"]) == 2
    assert row["estimate_source"] == "provider_usage"
    assert row["prompt_fingerprint"] == "prompt-v1:abc"
    assert row["prompt_fingerprint_version"] == "prompt-v1"
    assert row["world_book_fingerprint"] == "world-v1:def"
    assert "sk-secret" not in row["raw_usage_metadata_json"]
    assert "cache_read_input_tokens" in row["raw_usage_metadata_json"]
    assert float(row["prompt_cost_usd"]) == pytest.approx(((20 * 0.010) + (70 * 0.001) + (10 * 0.005)) / 1000.0)
    assert float(row["completion_cost_usd"]) == pytest.approx((25 * 0.030) / 1000.0)
    assert float(row["total_cost_usd"]) == pytest.approx(
        float(row["prompt_cost_usd"]) + float(row["completion_cost_usd"])
    )
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)
    reset_pricing_catalog()


@pytest.mark.asyncio
async def test_log_llm_usage_estimate_source_distinguishes_missing_usage_from_provider_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "ut-key-" + uuid.uuid4().hex)
    dburl = f"sqlite:///./Databases/users_test_ut_{uuid.uuid4().hex}.sqlite"
    monkeypatch.setenv("DATABASE_URL", dburl)

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager

    reset_settings()
    await reset_db_pool()
    await reset_session_manager()

    pool = await get_db_pool()
    await _ensure_llm_tables(pool)
    await _ensure_llm_cache_columns(pool)

    await log_llm_usage(
        user_id=1,
        key_id=None,
        endpoint="POST:/api/v1/chat/completions",
        operation="chat",
        provider="openai",
        model="gpt-4o-mini",
        status=200,
        latency_ms=10,
        prompt_tokens=8,
        completion_tokens=2,
        usage_metadata=None,
        estimated=None,
        estimate_source=None,
        request_id="req-missing-usage-source",
    )
    await log_llm_usage(
        user_id=1,
        key_id=None,
        endpoint="POST:/api/v1/chat/completions",
        operation="chat",
        provider="openai",
        model="gpt-4o-mini",
        status=200,
        latency_ms=10,
        prompt_tokens=8,
        completion_tokens=2,
        usage_metadata={"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
        estimated=None,
        estimate_source=None,
        request_id="req-provider-usage-source",
    )

    if pool.pool:
        rows = await pool.fetchall(
            """
            SELECT request_id, estimate_source, estimated
            FROM llm_usage_log
            WHERE request_id IN ($1, $2)
            ORDER BY request_id
            """,
            "req-missing-usage-source",
            "req-provider-usage-source",
        )
    else:
        rows = await pool.fetchall(
            """
            SELECT request_id, estimate_source, estimated
            FROM llm_usage_log
            WHERE request_id IN (?, ?)
            ORDER BY request_id
            """,
            "req-missing-usage-source",
            "req-provider-usage-source",
        )

    sources = {row["request_id"]: row["estimate_source"] for row in rows}
    assert sources == {
        "req-missing-usage-source": "missing_usage",
        "req-provider-usage-source": "provider_usage",
    }
    estimated_flags = {row["request_id"]: bool(row["estimated"]) for row in rows}
    assert estimated_flags == {
        "req-missing-usage-source": True,
        "req-provider-usage-source": False,
    }


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []

    def debug(self, message: str, *args, **kwargs) -> None:
        self.debugs.append(message)


def _safe_usage_settings() -> SimpleNamespace:
    return SimpleNamespace(
        LLM_USAGE_ENABLED=True,
        USAGE_LOG_DISABLE_META=True,
        PII_REDACT_LOGS=False,
    )


@pytest.mark.asyncio
async def test_tokens_daily_ledger_init_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.core.Usage import usage_tracker as usage_tracker_module

    class _FailingLedger:
        async def initialize(self) -> None:
            raise RuntimeError("ledger init failed at /private/ledger.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(usage_tracker_module, "_tokens_daily_ledger", None)
    monkeypatch.setattr(usage_tracker_module, "ResourceDailyLedger", _FailingLedger)
    monkeypatch.setattr(usage_tracker_module, "LedgerEntry", object())
    monkeypatch.setattr(usage_tracker_module, "logger", logger_stub)

    ledger = await usage_tracker_module._get_tokens_daily_ledger()

    assert ledger is None
    assert logger_stub.debugs == ["LLM usage ResourceDailyLedger init failed; tokens/day caps disabled"]
    assert "ledger init failed" not in str(logger_stub.debugs)
    assert "/private/ledger.db" not in str(logger_stub.debugs)


@pytest.mark.asyncio
async def test_log_llm_usage_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.core.Usage import usage_tracker as usage_tracker_module

    async def _fail_get_db_pool():
        raise RuntimeError("usage DB failed at /private/llm-usage.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(usage_tracker_module, "get_settings", _safe_usage_settings)
    monkeypatch.setattr(usage_tracker_module, "get_db_pool", _fail_get_db_pool)
    monkeypatch.setattr(usage_tracker_module, "logger", logger_stub)

    await usage_tracker_module.log_llm_usage(
        user_id=1,
        key_id=None,
        endpoint="POST:/api/v1/chat/completions",
        operation="chat",
        provider="test",
        model="test-model",
        status=200,
        latency_ms=1,
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        request_id="raw-request-id",
    )

    assert logger_stub.debugs == ["LLM usage logging skipped/failed"]
    assert "usage DB failed" not in str(logger_stub.debugs)
    assert "/private/llm-usage.db" not in str(logger_stub.debugs)
    assert "raw-request-id" not in str(logger_stub.debugs)


@pytest.mark.asyncio
async def test_log_llm_usage_persists_router_enrichment(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "ut-key-" + uuid.uuid4().hex)
    monkeypatch.setenv("PII_REDACT_LOGS", "false")
    monkeypatch.setenv("USAGE_LOG_DISABLE_META", "false")
    dburl = f"sqlite:///./Databases/users_test_ut_{uuid.uuid4().hex}.sqlite"
    monkeypatch.setenv("DATABASE_URL", dburl)

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager

    reset_settings()
    await reset_db_pool()
    await reset_session_manager()

    pool = await get_db_pool()
    await _ensure_llm_tables(pool)

    await log_llm_usage(
        user_id=1,
        key_id=1,
        endpoint="POST:/api/v1/chat/completions",
        operation="chat",
        provider="openai",
        model="gpt-4o-mini",
        status=200,
        latency_ms=120,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        request_id="req-enrich",
        remote_ip="127.0.0.1",
        user_agent="pytest-agent/1.0",
        token_name="Admin",  # nosec B106
        conversation_id="conv-1",
    )

    if pool.pool:
        row = await pool.fetchone(
            "SELECT remote_ip, user_agent, token_name, conversation_id FROM llm_usage_log WHERE request_id = $1",
            "req-enrich",
        )
    else:
        row = await pool.fetchone(
            "SELECT remote_ip, user_agent, token_name, conversation_id FROM llm_usage_log WHERE request_id = ?",
            "req-enrich",
        )

    assert row is not None
    if isinstance(row, dict):
        assert row["remote_ip"] == "127.0.0.1"
        assert row["user_agent"] == "pytest-agent/1.0"
        assert row["token_name"] == "Admin"
        assert row["conversation_id"] == "conv-1"
    else:
        assert row[0] == "127.0.0.1"
        assert row[1] == "pytest-agent/1.0"
        assert row[2] == "Admin"
        assert row[3] == "conv-1"


@pytest.mark.asyncio
async def test_log_llm_usage_derives_token_name_from_key(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "ut-key-" + uuid.uuid4().hex)
    monkeypatch.setenv("PII_REDACT_LOGS", "false")
    monkeypatch.setenv("USAGE_LOG_DISABLE_META", "false")
    dburl = f"sqlite:///./Databases/users_test_ut_{uuid.uuid4().hex}.sqlite"
    monkeypatch.setenv("DATABASE_URL", dburl)

    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool, get_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager

    reset_settings()
    await reset_db_pool()
    await reset_session_manager()

    pool = await get_db_pool()
    await _ensure_llm_tables(pool)

    if pool.pool:
        await pool.execute(
            """
            INSERT INTO users (id, username, email, password_hash)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (id) DO NOTHING
            """,
            1,
            "usage-test-user",
            "usage-test-user@example.com",
            "hash",
        )
    else:
        await pool.execute(
            """
            INSERT OR IGNORE INTO users (id, username, email, password_hash)
            VALUES (?, ?, ?, ?)
            """,
            1,
            "usage-test-user",
            "usage-test-user@example.com",
            "hash",
        )

    key_hash = "kh-" + uuid.uuid4().hex
    if pool.pool:
        await pool.execute(
            "INSERT INTO api_keys (user_id, key_hash, name, scope) VALUES ($1, $2, $3, $4)",
            1,
            key_hash,
            "DerivedName",
            "read",
        )
        key_id = await pool.fetchval("SELECT id FROM api_keys WHERE key_hash = $1", key_hash)
    else:
        await pool.execute(
            "INSERT INTO api_keys (user_id, key_hash, name, scope) VALUES (?, ?, ?, ?)",
            1,
            key_hash,
            "DerivedName",
            "read",
        )
        key_id = await pool.fetchval("SELECT id FROM api_keys WHERE key_hash = ?", key_hash)
    assert key_id is not None

    await log_llm_usage(
        user_id=1,
        key_id=int(key_id),
        endpoint="POST:/api/v1/chat/completions",
        operation="chat",
        provider="openai",
        model="gpt-4o-mini",
        status=200,
        latency_ms=90,
        prompt_tokens=5,
        completion_tokens=2,
        total_tokens=7,
        request_id="req-derive-name",
    )

    if pool.pool:
        row = await pool.fetchone("SELECT token_name FROM llm_usage_log WHERE request_id = $1", "req-derive-name")
    else:
        row = await pool.fetchone("SELECT token_name FROM llm_usage_log WHERE request_id = ?", "req-derive-name")

    assert row is not None
    if isinstance(row, dict):
        assert row["token_name"] == "DerivedName"
    else:
        assert row[0] == "DerivedName"
