"""Account lockout tracking for brute-force protection.

This module provides dedicated lockout tracking that is independent of rate
limiting.  It records failed authentication attempts and triggers account
lockouts when a threshold is exceeded.

Previously this logic lived in ``rate_limiter.py`` alongside (now no-op) rate
limit methods.  Extracting it here makes the security function explicit and
allows the legacy rate limiter shim to be retired without losing brute-force
protection.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.repos.rate_limits_repo import AuthnzRateLimitsRepo
from tldw_Server_API.app.core.AuthNZ.settings import Settings, get_settings


class LockoutTracker:
    """
    DB-backed account lockout tracker.

    Tracks failed authentication attempts and triggers/checks account lockouts.
    This is a security function (brute-force protection), not rate limiting.
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
        self._repo: AuthnzRateLimitsRepo | None = None

    async def initialize(self) -> None:
        if self._initialized:
            return
        if not self.db_pool:
            self.db_pool = await get_db_pool()
        if self._repo is None:
            self._repo = AuthnzRateLimitsRepo(self.db_pool)
        try:
            await self._repo.ensure_schema()
        except Exception:
            logger.warning("LockoutTracker schema ensure warning")
        self._initialized = True

    def _get_repo(self) -> AuthnzRateLimitsRepo:
        if not self.db_pool:
            raise RuntimeError("LockoutTracker database pool is not initialized")
        if self._repo is None:
            self._repo = AuthnzRateLimitsRepo(self.db_pool)
        return self._repo

    async def record_failed_attempt(
        self,
        identifier: str,
        attempt_type: str = "login",
        lockout_threshold: int | None = None,
        lockout_duration_minutes: int | None = None,
    ) -> dict[str, Any]:
        """Record a failed authentication attempt and check for lockout."""
        if not self._initialized:
            await self.initialize()

        lockout_threshold = lockout_threshold or self.settings.MAX_LOGIN_ATTEMPTS
        lockout_duration_minutes = lockout_duration_minutes or self.settings.LOCKOUT_DURATION_MINUTES

        now = datetime.now(timezone.utc)
        repo = self._get_repo()
        result = await repo.record_failed_attempt_and_lockout(
            identifier=identifier,
            attempt_type=attempt_type,
            now=now,
            lockout_threshold=int(lockout_threshold),
            lockout_duration_minutes=int(lockout_duration_minutes),
        )

        attempt_count = int(result.get("attempt_count", 0))
        is_locked = bool(result.get("is_locked", False))
        lockout_expires_dt = result.get("lockout_expires")

        if is_locked and lockout_expires_dt is not None:
            if self.settings.PII_REDACT_LOGS:
                logger.warning("Account locked after failed attempts [redacted]")
            else:
                logger.warning(
                    f"Account locked for {identifier} after {attempt_count} failed attempts"
                )
            return {
                "attempt_count": attempt_count,
                "is_locked": True,
                "lockout_expires": lockout_expires_dt.isoformat(),
                "remaining_attempts": 0,
            }

        return {
            "attempt_count": attempt_count,
            "is_locked": False,
            "remaining_attempts": max(0, int(lockout_threshold) - attempt_count),
        }

    async def check_lockout(self, identifier: str, attempt_type: str = "login") -> tuple[bool, datetime | None]:
        """Check whether an account is currently locked out."""
        if not self._initialized:
            await self.initialize()
        repo = self._get_repo()
        locked_until = await repo.get_active_lockout(
            identifier=identifier,
            attempt_type=attempt_type,
            now=datetime.now(timezone.utc),
        )
        if locked_until is not None:
            return True, locked_until
        return False, None

    async def reset_failed_attempts(self, identifier: str, attempt_type: str = "login") -> None:
        """Clear failed attempt counters and lockout for an identifier and attempt type."""
        if not self._initialized:
            await self.initialize()
        repo = self._get_repo()
        await repo.reset_failed_attempts_and_lockout(
            identifier=identifier,
            attempt_type=attempt_type,
        )


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_lockout_tracker: LockoutTracker | None = None


def get_lockout_tracker() -> LockoutTracker:
    """Return the process-global LockoutTracker instance."""
    global _lockout_tracker
    if _lockout_tracker is None:
        _lockout_tracker = LockoutTracker()
    return _lockout_tracker


async def reset_lockout_tracker() -> None:
    """Reset the process-global LockoutTracker singleton.

    Primarily used by tests that create isolated databases per test and need to
    avoid stale repository/database pool references between app lifecycles.
    """
    global _lockout_tracker
    tracker = _lockout_tracker
    _lockout_tracker = None
    if tracker is not None:
        tracker._repo = None
        tracker.db_pool = None
        tracker._initialized = False
