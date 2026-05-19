from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.AuthNZ import lockout_tracker as lockout_tracker_module
from tldw_Server_API.app.core.AuthNZ.lockout_tracker import LockoutTracker
from tldw_Server_API.app.core.AuthNZ.rate_limiter import RateLimiter


class _ControlledDatetime(lockout_tracker_module.datetime):
    current = lockout_tracker_module.datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    @classmethod
    def utcnow(cls):
        return cls.current

    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return cls.current
        try:
            return cls.current.astimezone(tz)
        except Exception:
            return cls.current


class _StubRepo:
    def __init__(self) -> None:
        self._attempts: dict[tuple[str, str], dict[str, object]] = {}
        self._lockouts: dict[tuple[str, str], datetime] = {}

    async def ensure_schema(self) -> None:
        return None

    async def record_failed_attempt_and_lockout(
        self,
        *,
        identifier: str,
        attempt_type: str,
        now: datetime,
        lockout_threshold: int,
        lockout_duration_minutes: int,
    ) -> dict[str, object]:
        key = (identifier, attempt_type)
        entry = self._attempts.get(key)
        window_delta = timedelta(minutes=lockout_duration_minutes)
        if entry is None or entry["window_start"] + window_delta < now:
            attempt_count = 1
            window_start = now
        else:
            attempt_count = int(entry["attempt_count"]) + 1
            window_start = entry["window_start"]
        self._attempts[key] = {"attempt_count": attempt_count, "window_start": window_start}

        is_locked = attempt_count >= lockout_threshold
        lockout_expires = None
        if is_locked:
            lockout_expires = now + window_delta
            self._lockouts[key] = lockout_expires

        return {
            "attempt_count": attempt_count,
            "is_locked": is_locked,
            "lockout_expires": lockout_expires,
        }

    async def get_active_lockout(self, *, identifier: str, attempt_type: str = "login", now: datetime):
        key = (identifier, attempt_type)
        locked_until = self._lockouts.get(key)
        if locked_until and locked_until > now:
            return locked_until
        if locked_until:
            self._lockouts.pop(key, None)
        return None

    async def reset_failed_attempts_and_lockout(self, *, identifier: str, attempt_type: str) -> None:
        self._attempts.pop((identifier, attempt_type), None)
        self._lockouts.pop((identifier, attempt_type), None)


@pytest.mark.asyncio
async def test_lockout_recovers_after_window(monkeypatch):
    original_datetime = lockout_tracker_module.datetime
    monkeypatch.setattr(lockout_tracker_module, "datetime", _ControlledDatetime)

    settings = SimpleNamespace(
        MAX_LOGIN_ATTEMPTS=3,
        LOCKOUT_DURATION_MINUTES=5,
        PII_REDACT_LOGS=False,
    )
    # Set up a LockoutTracker with stub internals
    tracker = LockoutTracker(settings=settings)
    tracker.db_pool = object()
    tracker._repo = _StubRepo()
    tracker._initialized = True

    # Wire it into the RateLimiter for backward-compat testing
    limiter = RateLimiter(settings=settings)
    limiter.db_pool = object()
    limiter._initialized = True
    limiter._lockout = tracker

    identifier = "ip:127.0.0.1"

    for _ in range(settings.MAX_LOGIN_ATTEMPTS):
        result = await limiter.record_failed_attempt(identifier, attempt_type="login")

    assert result["is_locked"] is True

    _ControlledDatetime.current = original_datetime(2024, 1, 1, 0, 6, 0, tzinfo=timezone.utc)
    is_locked, _ = await limiter.check_lockout(identifier)
    assert is_locked is False

    result_after = await limiter.record_failed_attempt(identifier, attempt_type="login")
    assert result_after["is_locked"] is False
    assert result_after["attempt_count"] == 1
    assert result_after["remaining_attempts"] == settings.MAX_LOGIN_ATTEMPTS - 1


@pytest.mark.asyncio
async def test_reset_failed_attempts_only_clears_matching_attempt_type():
    settings = SimpleNamespace(
        MAX_LOGIN_ATTEMPTS=3,
        LOCKOUT_DURATION_MINUTES=5,
        PII_REDACT_LOGS=False,
    )
    tracker = LockoutTracker(settings=settings)
    tracker.db_pool = object()
    tracker._repo = _StubRepo()
    tracker._initialized = True

    limiter = RateLimiter(settings=settings)
    limiter.db_pool = object()
    limiter._initialized = True
    limiter._lockout = tracker

    identifier = "user:rate-reset"

    for _ in range(settings.MAX_LOGIN_ATTEMPTS):
        await limiter.record_failed_attempt(identifier, attempt_type="login")
        await limiter.record_failed_attempt(identifier, attempt_type="password_reset")

    await limiter.reset_failed_attempts(identifier, attempt_type="login")

    login_locked, _ = await limiter.check_lockout(identifier, attempt_type="login")
    reset_locked, _ = await limiter.check_lockout(
        identifier,
        attempt_type="password_reset",
    )

    assert login_locked is False
    assert reset_locked is True
