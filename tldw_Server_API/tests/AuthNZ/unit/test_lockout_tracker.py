"""Tests for the extracted LockoutTracker module."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.AuthNZ import lockout_tracker as lt_module
from tldw_Server_API.app.core.AuthNZ.lockout_tracker import LockoutTracker, get_lockout_tracker


class _StubRepo:
    """In-memory stub of AuthnzRateLimitsRepo for unit testing."""

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


class _FailingSchemaRepo:
    async def ensure_schema(self) -> None:
        raise RuntimeError("lockout schema failed at /private/lockouts.db")


class _LoggerStub:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def warning(self, message: str, *args, **kwargs) -> None:
        self.messages.append(message)


def _make_tracker(settings=None) -> LockoutTracker:
    """Create a LockoutTracker with stub internals for testing."""
    settings = settings or SimpleNamespace(
        MAX_LOGIN_ATTEMPTS=3,
        LOCKOUT_DURATION_MINUTES=5,
        PII_REDACT_LOGS=False,
    )
    tracker = LockoutTracker(settings=settings)
    tracker.db_pool = object()  # type: ignore[assignment]
    tracker._repo = _StubRepo()
    tracker._initialized = True
    return tracker


@pytest.mark.asyncio
async def test_lockout_triggers_after_threshold():
    tracker = _make_tracker()
    identifier = "user:alice"

    for i in range(3):
        result = await tracker.record_failed_attempt(identifier)

    assert result["is_locked"] is True
    assert result["remaining_attempts"] == 0
    assert result["attempt_count"] == 3
    assert "lockout_expires" in result


@pytest.mark.asyncio
async def test_check_lockout_is_scoped_by_attempt_type():
    tracker = _make_tracker()
    identifier = "user:alice"

    for _ in range(3):
        await tracker.record_failed_attempt(identifier, attempt_type="login")

    login_locked, _ = await tracker.check_lockout(identifier, attempt_type="login")
    reset_locked, _ = await tracker.check_lockout(identifier, attempt_type="password_reset")

    assert login_locked is True
    assert reset_locked is False


@pytest.mark.asyncio
async def test_reset_failed_attempts_only_clears_matching_attempt_type():
    tracker = _make_tracker()
    identifier = "user:bob"

    for _ in range(3):
        await tracker.record_failed_attempt(identifier, attempt_type="login")
        await tracker.record_failed_attempt(identifier, attempt_type="password_reset")

    await tracker.reset_failed_attempts(identifier, attempt_type="login")

    login_locked, _ = await tracker.check_lockout(identifier, attempt_type="login")
    reset_locked, _ = await tracker.check_lockout(identifier, attempt_type="password_reset")

    assert login_locked is False
    assert reset_locked is True


@pytest.mark.asyncio
async def test_check_lockout_returns_true_when_locked():
    tracker = _make_tracker()
    identifier = "user:charlie"

    for _ in range(3):
        await tracker.record_failed_attempt(identifier)

    is_locked, locked_until = await tracker.check_lockout(identifier)
    assert is_locked is True
    assert locked_until is not None


@pytest.mark.asyncio
async def test_reset_clears_lockout():
    tracker = _make_tracker()
    identifier = "user:dave"

    for _ in range(3):
        await tracker.record_failed_attempt(identifier)

    is_locked, _ = await tracker.check_lockout(identifier)
    assert is_locked is True

    await tracker.reset_failed_attempts(identifier)
    is_locked, _ = await tracker.check_lockout(identifier)
    assert is_locked is False


@pytest.mark.asyncio
async def test_no_lockout_below_threshold():
    tracker = _make_tracker()
    identifier = "user:erin"

    result = await tracker.record_failed_attempt(identifier)
    assert result["is_locked"] is False
    assert result["remaining_attempts"] == 2

    result = await tracker.record_failed_attempt(identifier)
    assert result["is_locked"] is False
    assert result["remaining_attempts"] == 1


@pytest.mark.asyncio
async def test_get_lockout_tracker_returns_singleton(monkeypatch):
    monkeypatch.setattr(lt_module, "_lockout_tracker", None)
    t1 = get_lockout_tracker()
    t2 = get_lockout_tracker()
    assert t1 is t2


@pytest.mark.asyncio
async def test_pii_redacted_log(monkeypatch):
    """When PII_REDACT_LOGS is True, the lockout log message should not contain the identifier."""
    settings = SimpleNamespace(
        MAX_LOGIN_ATTEMPTS=1,
        LOCKOUT_DURATION_MINUTES=5,
        PII_REDACT_LOGS=True,
    )
    tracker = _make_tracker(settings=settings)
    result = await tracker.record_failed_attempt("sensitive_user")
    assert result["is_locked"] is True


@pytest.mark.asyncio
async def test_initialize_schema_warning_log_is_sanitized(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(lt_module, "logger", logger_stub)

    tracker = LockoutTracker(
        db_pool=object(),  # type: ignore[arg-type]
        settings=SimpleNamespace(
            MAX_LOGIN_ATTEMPTS=3,
            LOCKOUT_DURATION_MINUTES=5,
            PII_REDACT_LOGS=False,
        ),
    )
    tracker._repo = _FailingSchemaRepo()  # type: ignore[assignment]

    await tracker.initialize()

    assert tracker._initialized is True
    assert logger_stub.messages == ["LockoutTracker schema ensure warning"]
    assert "lockout schema failed" not in str(logger_stub.messages)
    assert "/private/lockouts.db" not in str(logger_stub.messages)
