from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.AuthNZ.repos.rate_limits_repo import (
    AuthnzRateLimitsRepo,
)
from tldw_Server_API.app.core.AuthNZ.rate_limiter import RateLimiter
from tldw_Server_API.app.core.AuthNZ.settings import Settings


pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_authnz_rate_limits_repo_lockout_postgres(test_db_pool):
    """AuthnzRateLimitsRepo lockout helpers should behave consistently on Postgres via RateLimiter."""
    pool = test_db_pool

    settings = Settings(
        AUTH_MODE="multi_user",
        JWT_SECRET_KEY="x" * 64,
        MAX_LOGIN_ATTEMPTS=3,
        LOCKOUT_DURATION_MINUTES=5,
    )

    limiter = RateLimiter(db_pool=pool, settings=settings)
    await limiter.initialize()

    identifier = "pg-lockout-user"

    # Drive identifier to lockout via RateLimiter (DB-backed path)
    last_result = None
    for _ in range(settings.MAX_LOGIN_ATTEMPTS):
        last_result = await limiter.record_failed_attempt(
            identifier=identifier,
            attempt_type="login",
        )

    assert last_result is not None
    assert last_result["is_locked"] is True
    assert last_result["attempt_count"] == settings.MAX_LOGIN_ATTEMPTS

    # check_lockout should see an active lock
    is_locked, locked_until = await limiter.check_lockout(identifier)
    assert is_locked is True
    assert locked_until is not None

    # Reset via repo and confirm lockout is cleared
    repo = AuthnzRateLimitsRepo(pool)
    await repo.reset_failed_attempts_and_lockout(
        identifier=identifier,
        attempt_type="login",
    )

    # Use a pre-expiry timestamp so a lingering lockout would still be visible
    pre_expiry = locked_until - timedelta(seconds=1)
    locked_after_reset = await repo.get_active_lockout(
        identifier=identifier,
        now=pre_expiry,
    )
    assert locked_after_reset is None


@pytest.mark.asyncio
async def test_authnz_rate_limits_repo_lockout_scoped_by_attempt_type_postgres(test_db_pool):
    """Lockout visibility should remain scoped to the matching attempt type."""
    pool = test_db_pool

    settings = Settings(
        AUTH_MODE="multi_user",
        JWT_SECRET_KEY="x" * 64,
        MAX_LOGIN_ATTEMPTS=3,
        LOCKOUT_DURATION_MINUTES=5,
    )

    limiter = RateLimiter(db_pool=pool, settings=settings)
    await limiter.initialize()

    identifier = "pg-lockout-scope-user"
    for _ in range(settings.MAX_LOGIN_ATTEMPTS):
        await limiter.record_failed_attempt(
            identifier=identifier,
            attempt_type="login",
        )

    login_locked, _ = await limiter.check_lockout(identifier, attempt_type="login")
    reset_locked, _ = await limiter.check_lockout(
        identifier,
        attempt_type="password_reset",
    )

    assert login_locked is True
    assert reset_locked is False


@pytest.mark.asyncio
async def test_authnz_rate_limits_repo_rate_window_postgres(test_db_pool):
    """AuthnzRateLimitsRepo rate_limits window helpers should behave consistently on Postgres."""
    pool = test_db_pool
    repo = AuthnzRateLimitsRepo(pool)

    identifier = "pg-rate-user"
    endpoint = "/api/pg-rate"
    now = datetime.now(timezone.utc).replace(microsecond=0)
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

    endpoints = await repo.list_rate_limit_endpoints_for_identifier(
        identifier=identifier
    )
    assert endpoint in endpoints

    cutoff_recent = window_start - timedelta(minutes=1)
    deleted_recent = await repo.cleanup_rate_limits_older_than(cutoff_recent)
    assert deleted_recent == 0

    cutoff_future = window_start + timedelta(minutes=10)
    deleted_future = await repo.cleanup_rate_limits_older_than(cutoff_future)
    assert deleted_future >= 1
