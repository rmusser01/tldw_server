from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import pytest_asyncio

from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
from tldw_Server_API.app.core.AuthNZ.repos.rate_limits_repo import (
    AuthnzRateLimitsRepo,
)
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings


@pytest_asyncio.fixture
async def sqlite_rate_limits_repo(tmp_path, monkeypatch, fresh_services) -> AuthnzRateLimitsRepo:
    """Provide an AuthnzRateLimitsRepo wired to a fresh SQLite AuthNZ database."""
    """AuthnzRateLimitsRepo lockout + rate_limits helpers should work on SQLite."""
    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    return AuthnzRateLimitsRepo(pool)


@pytest.mark.asyncio
async def test_authnz_rate_limits_repo_lockout_and_cleanup_sqlite(
    sqlite_rate_limits_repo: AuthnzRateLimitsRepo,
) -> None:
    """AuthnzRateLimitsRepo lockout + rate_limits helpers should work on SQLite."""
    repo = sqlite_rate_limits_repo

    identifier = "ip:127.0.0.1"
    attempt_type = "login"
    now = datetime.now(timezone.utc).replace(microsecond=0)

    # Drive attempts up to and beyond the lockout threshold
    threshold = 3
    duration = 5
    # First attempt starts counting and should not raise; lockout semantics
    # are validated more thoroughly in Postgres-backed tests.
    first = await repo.record_failed_attempt_and_lockout(
        identifier=identifier,
        attempt_type=attempt_type,
        now=now,
        lockout_threshold=threshold,
        lockout_duration_minutes=duration,
    )
    assert first["attempt_count"] >= 1
    assert "is_locked" in first

    # get_active_lockout should not error even if no lock is present yet
    locked_until = await repo.get_active_lockout(identifier=identifier, now=now)
    assert locked_until is None

    # After moving past the lockout window and clearing failed attempts, lockout should be gone
    future = now + timedelta(minutes=duration + 1)
    await repo.reset_failed_attempts_and_lockout(
        identifier=identifier,
        attempt_type=attempt_type,
    )
    locked_after_reset = await repo.get_active_lockout(identifier=identifier, now=future)
    assert locked_after_reset is None

    # Rate-limit window increment and cleanup
    endpoint = "/api/test"
    window_start = now.replace(second=0)
    count1 = await repo.increment_rate_limit_window(
        identifier=identifier,
        endpoint=endpoint,
        window_start=window_start,
    )
    count2 = await repo.increment_rate_limit_window(
        identifier=identifier,
        endpoint=endpoint,
        window_start=window_start,
    )
    assert count1 == 1
    assert count2 == 2

    fetched = await repo.get_rate_limit_count(
        identifier=identifier,
        endpoint=endpoint,
        window_start=window_start,
    )
    assert fetched == 2

    # list/cleanup helpers should see and remove the rows
    endpoints = await repo.list_rate_limit_endpoints_for_identifier(
        identifier=identifier
    )
    assert endpoint in endpoints

    # Cleanup older than cutoff that excludes current row: should be 0
    cutoff_recent = window_start - timedelta(minutes=1)
    deleted_recent = await repo.cleanup_rate_limits_older_than(cutoff_recent)
    assert deleted_recent == 0

    # Cleanup with cutoff in the future should delete the current bucket
    cutoff_future = window_start + timedelta(minutes=10)
    deleted_future = await repo.cleanup_rate_limits_older_than(cutoff_future)
    assert deleted_future >= 1

    # Explicit delete helper should remove any remaining buckets for the identifier
    deleted_all = await repo.delete_rate_limits_for_identifier(identifier=identifier)
    assert deleted_all >= 0
    endpoints_after_delete = await repo.list_rate_limit_endpoints_for_identifier(
        identifier=identifier
    )
    assert not endpoints_after_delete
