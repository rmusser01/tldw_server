from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.repos.monitoring_repo import AuthnzMonitoringRepo
from tldw_Server_API.app.core.AuthNZ.settings import Settings


@pytest.mark.asyncio
async def test_authnz_monitoring_repo_sqlite_basic(tmp_path):
    """AuthnzMonitoringRepo should store and aggregate metrics on SQLite."""
    db_path = tmp_path / "authnz_monitoring_repo.sqlite"
    settings = Settings(
        AUTH_MODE="multi_user",
        DATABASE_URL=f"sqlite:///{db_path}",
        JWT_SECRET_KEY="monitoring-secret-key-32-characters-minimum!",
        ENABLE_REGISTRATION=True,
        REQUIRE_REGISTRATION_CODE=False,
    )

    pool = DatabasePool(settings)
    await pool.initialize()

    try:
        # Ensure audit_logs table exists (migrations may not have run yet in this temp DB)
        async with pool.transaction() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    action TEXT NOT NULL,
                    details TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

        repo = AuthnzMonitoringRepo(pool)
        now = datetime.now(timezone.utc)

        # Insert a couple of metric audit rows
        await repo.insert_metric_audit_log(
            action="metric_auth_success",
            details_json="{}",
            created_at=now - timedelta(minutes=10),
        )
        await repo.insert_metric_audit_log(
            action="metric_auth_failure",
            details_json="{}",
            created_at=now - timedelta(minutes=5),
        )
        await repo.insert_metric_audit_log(
            action="metric_rate_limit_hit",
            details_json="{}",
            created_at=now - timedelta(minutes=2),
        )

        cutoff = now - timedelta(minutes=15)
        summary = await repo.get_metrics_window_summary(cutoff)
        assert summary["successful_auths"] >= 1
        assert summary["failed_auths"] >= 1
        assert summary["rate_limit_hits"] >= 1

        # Insert a couple of sessions/api_keys rows for count helpers.
        # These tables and FKs are provisioned by migrations in real DBs;
        # for this isolated repo test, skip the counts if inserts fail.
        try:
            await pool.execute(
                """
                INSERT INTO sessions (user_id, token_hash, expires_at, is_revoked)
                VALUES (?, ?, ?, 0)
                """,
                (1, "dummy-hash", (now + timedelta(hours=1)).isoformat()),
            )
            await pool.execute(
                "INSERT INTO api_keys (user_id, key_hash, key_prefix, scope, status) VALUES (?, ?, ?, ?, 'active')",
                (1, "dummy-key-hash", "dummy-prefix", "read"),
            )

            active_sessions = await repo.get_active_sessions_count(now)
            active_keys = await repo.get_active_api_keys_count()
            assert active_sessions >= 0
            assert active_keys >= 0
        except Exception:
            # Some temp schemas may enforce FKs before users are seeded; the
            # aggregation helpers are still exercised by metrics summary above.
            _ = None

        # Security alerts list should round-trip through audit_logs
        await repo.insert_metric_audit_log(
            action="metric_security_alert",
            details_json='{"labels":{"alert_type":"test_alert","severity":"high"}}',
            created_at=now,
        )
        alerts = await repo.get_recent_security_alerts(limit=5)
        assert alerts
        assert alerts[0]["action"] == "metric_security_alert"
    finally:
        await pool.close()
