import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from tldw_Server_API.app.core.AuthNZ.monitoring import AuthNZMonitor
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
from tldw_Server_API.app.core.AuthNZ.migrations import (
    ensure_authnz_tables,
    migration_072_create_identity_providers_table,
    migration_073_create_federated_identities_table,
    migration_074_create_federated_managed_grants_table,
)
from tldw_Server_API.app.core.AuthNZ.scheduler import AuthNZScheduler


pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_metrics_summary_uses_boolean_revoked_filter(monkeypatch):
    pool = await get_db_pool()
    if getattr(pool, "pool", None) is None and getattr(pool, "db_path", None):
        ensure_authnz_tables(Path(pool.db_path))
        with sqlite3.connect(pool.db_path) as conn:
            migration_072_create_identity_providers_table(conn)
            migration_073_create_federated_identities_table(conn)
            migration_074_create_federated_managed_grants_table(conn)

    # Insert a dedicated user and related records for this test
    uname = f"metrics_user_{uuid.uuid4().hex[:8]}"
    email = f"{uname}@example.com"
    await pool.execute(
        """
        INSERT INTO users (username, email, password_hash, is_active)
        VALUES (?, ?, ?, 1)
        """,
        (uname, email, "hash"),
    )
    user_row = await pool.fetchone("SELECT id FROM users WHERE username = ?", uname)
    user_id = user_row["id"] if isinstance(user_row, dict) else user_row[0]

    expires_at = (datetime.utcnow() + timedelta(hours=1)).isoformat()
    token_hash = f"metrics_token_hash_{uuid.uuid4().hex[:8]}"
    refresh_hash = f"metrics_refresh_hash_{uuid.uuid4().hex[:8]}"

    await pool.execute(
        """
        INSERT INTO sessions (
            user_id, token_hash, refresh_token_hash, encrypted_token, encrypted_refresh,
            expires_at, refresh_expires_at, ip_address, user_agent, device_id,
            is_active, is_revoked
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 0)
        """,
        (
            user_id,
            token_hash,
            refresh_hash,
            "enc_token",
            "enc_refresh",
            expires_at,
            None,
            "127.0.0.1",
            "pytest-agent",
            "device-id",
        ),
    )

    await pool.execute(
        """
        INSERT INTO api_keys (user_id, key_hash, key_prefix, scope, status)
        VALUES (?, ?, ?, ?, 'active')
        """,
        (
            user_id,
            f"metrics_api_key_hash_{uuid.uuid4().hex}",
            f"metrics_key_{uuid.uuid4().hex[:10]}",
            "read",
        ),
    )

    await pool.execute(
        """
        INSERT INTO audit_logs (user_id, action, details, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            user_id,
            "metric_auth_success",
            "{}",
            datetime.utcnow().isoformat(),
        ),
    )

    original_fetchone = pool.fetchone

    async def wrapped_fetchone(query: str, *args):
        assert "is_revoked = 0" not in query
        return await original_fetchone(query, *args)

    monkeypatch.setattr(pool, "fetchone", wrapped_fetchone)

    monitor = AuthNZMonitor()
    summary = await monitor.get_metrics_summary()

    assert summary["sessions"]["active"] >= 1
    assert summary["authentication"]["successful"] >= 1

    await pool.execute("DELETE FROM audit_logs WHERE user_id = ?", (user_id,))
    await pool.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
    await pool.execute("DELETE FROM api_keys WHERE user_id = ?", (user_id,))
    await pool.execute("DELETE FROM users WHERE id = ?", (user_id,))


class _NoopDispatcher:
    async def dispatch(self, **kwargs):
        return True


@pytest.mark.asyncio
async def test_scheduler_monitor_queries_sqlite(monkeypatch):
    """Scheduler monitoring helpers should run on SQLite without SQL errors."""
    await reset_db_pool()
    scheduler = AuthNZScheduler()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.scheduler.get_security_alert_dispatcher",
        lambda: _NoopDispatcher(),
    )

    await scheduler._monitor_auth_failures()
    await scheduler._monitor_api_usage()
    await reset_db_pool()
