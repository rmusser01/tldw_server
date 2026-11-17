"""AuthNZ limiter facade: lockout tracking via LockoutTracker + RG-compatible no-op rate limits.

**Phase 2 Deprecation Notice**:
Rate-limit enforcement is fully delegated to Resource Governor (RG).
``check_rate_limit()`` and ``check_user_rate_limit()`` are no-ops and retained
only for compatibility signatures.

Lockout logic has been extracted to ``lockout_tracker.py``.  This module
preserves the public API (``RateLimiter``, ``get_rate_limiter``,
``check_rate_limit``) for backward compatibility while delegating lockout
methods to ``LockoutTracker``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.lockout_tracker import (
    LockoutTracker,
    get_lockout_tracker,
    reset_lockout_tracker,
)
from tldw_Server_API.app.core.AuthNZ.settings import Settings, get_settings


class RateLimiter:
    """
    AuthNZ limiter facade.

    - Rate limiting is delegated to Resource Governor at ingress.
    - Lockout tracking is delegated to ``LockoutTracker``.
    - This class preserves compatibility signatures for existing callers.
    """

    def __init__(
        self,
        db_pool: DatabasePool | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.db_pool = db_pool
        self.enabled = True
        self._initialized = False
        self._lockout: LockoutTracker | None = None

    async def initialize(self) -> None:
        if self._initialized:
            return
        if not self.db_pool:
            self.db_pool = await get_db_pool()
        # Initialize the lockout tracker with the same pool
        self._lockout = LockoutTracker(db_pool=self.db_pool, settings=self.settings)
        await self._lockout.initialize()
        self._initialized = True

    def _get_lockout_tracker(self) -> LockoutTracker:
        if self._lockout is not None:
            return self._lockout
        # Fallback to global singleton when not explicitly initialized
        return get_lockout_tracker()

    async def check_rate_limit(
        self,
        identifier: str,
        endpoint: str,
        limit: int | None = None,
        window_minutes: int | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """No-op rate limit check (RG handles ingress limits)."""
        return True, {"rate_limit_source": "resource_governor"}

    async def check_user_rate_limit(
        self,
        user_id: int,
        endpoint: str,
        limit: int | None = None,
        window_minutes: int | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """No-op per-user rate limit check (RG handles ingress limits)."""
        return True, {"rate_limit_source": "resource_governor"}

    async def record_failed_attempt(
        self,
        identifier: str,
        attempt_type: str = "login",
        lockout_threshold: int | None = None,
        lockout_duration_minutes: int | None = None,
        *,
        conn: Any | None = None,
    ) -> dict[str, Any]:
        """Delegate to LockoutTracker."""
        tracker = self._get_lockout_tracker()
        return await tracker.record_failed_attempt(
            identifier,
            attempt_type=attempt_type,
            lockout_threshold=lockout_threshold,
            lockout_duration_minutes=lockout_duration_minutes,
            conn=conn,
        )

    async def check_lockout(self, identifier: str, attempt_type: str = "login") -> tuple[bool, datetime | None]:
        """Delegate to LockoutTracker."""
        tracker = self._get_lockout_tracker()
        return await tracker.check_lockout(identifier, attempt_type=attempt_type)

    async def reset_failed_attempts(
        self, identifier: str, attempt_type: str = "login", *, conn: Any | None = None
    ) -> None:
        """Delegate to LockoutTracker."""
        tracker = self._get_lockout_tracker()
        await tracker.reset_failed_attempts(identifier, attempt_type=attempt_type, conn=conn)

    async def get_usage_stats(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"rate_limit_source": "resource_governor"}


_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


async def reset_rate_limiter() -> None:
    """Reset the process-global RateLimiter and its lockout singleton."""
    global _rate_limiter
    limiter = _rate_limiter
    _rate_limiter = None
    if limiter is not None:
        limiter._lockout = None
        limiter.db_pool = None
        limiter._initialized = False
    await reset_lockout_tracker()


async def check_rate_limit(
    identifier: str,
    endpoint: str,
    limit: int | None = None,
    window_minutes: int | None = None,
) -> tuple[bool, dict[str, Any]]:
    limiter = get_rate_limiter()
    return await limiter.check_rate_limit(identifier, endpoint, limit=limit, window_minutes=window_minutes)
