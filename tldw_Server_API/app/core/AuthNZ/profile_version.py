"""Sanitized profile-version values and failures shared with UserProfiles."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

_PROFILE_TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})?$"
)
_POSTGRES_CONCURRENCY_SQLSTATES = frozenset({"40P01", "40001"})


class ProfileVersionError(RuntimeError):
    """Base class for transport-neutral profile-version failures."""

    code = "profile_version_failed"


class ProfileVersionNotFound(ProfileVersionError):
    """The target user or its durable profile-version anchor is absent."""

    code = "profile_update_not_found"

    def __init__(self) -> None:
        super().__init__("Target profile was not found")


class ProfileVersionInvalid(ProfileVersionError):
    """A stored or supplied profile-version value is invalid."""

    code = "profile_version_invalid"

    def __init__(self) -> None:
        super().__init__("Stored profile version is invalid")


class ProfileVersionReadFailed(ProfileVersionError):
    """The complete profile-version snapshot could not be read."""

    code = "profile_version_read_failed"

    def __init__(self, *, sqlstate: str | None = None) -> None:
        super().__init__("Profile version could not be read")
        if (
            type(sqlstate) is str
            and sqlstate in _POSTGRES_CONCURRENCY_SQLSTATES
        ):
            self.sqlstate = sqlstate

    @classmethod
    def from_storage_error(
        cls,
        error: BaseException,
    ) -> ProfileVersionReadFailed:
        """Preserve only a safe PostgreSQL conflict signal from storage errors."""
        return cls(sqlstate=_postgres_concurrency_sqlstate(error))


def normalize_profile_version(value: Any, *, allow_naive: bool = False) -> datetime:
    """Return one aware UTC timestamp or fail with a sanitized domain error."""
    if type(value) is datetime:
        parsed = value
    elif type(value) is str:
        candidate = value.strip()
        if not _PROFILE_TIMESTAMP_PATTERN.fullmatch(candidate):
            raise ProfileVersionInvalid()
        try:
            parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        except ValueError:
            raise ProfileVersionInvalid() from None
    else:
        raise ProfileVersionInvalid() from None

    try:
        parsed_tzinfo = parsed.tzinfo
    except Exception:  # noqa: BLE001 - tzinfo implementations are untrusted
        raise ProfileVersionInvalid() from None
    if parsed_tzinfo is None:
        if not allow_naive:
            raise ProfileVersionInvalid()
        parsed = parsed.replace(tzinfo=timezone.utc)
    try:
        return parsed.astimezone(timezone.utc)
    except Exception:  # noqa: BLE001 - tzinfo implementations are untrusted
        raise ProfileVersionInvalid() from None


def compute_touch_value(clock_now_utc: Any, version_floor: Any) -> datetime:
    """Compute the exact monotonic value for the final profile-version touch."""
    now = normalize_profile_version(clock_now_utc)
    floor = normalize_profile_version(version_floor)
    try:
        next_floor = floor + timedelta(microseconds=1)
    except OverflowError:
        raise ProfileVersionInvalid() from None
    return max(now, next_floor)


def _postgres_concurrency_sqlstate(error: BaseException) -> str | None:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and len(seen) < 32:
        identity = id(current)
        if identity in seen:
            break
        seen.add(identity)
        sqlstate = _exception_attribute(current, "sqlstate")
        pgcode = _exception_attribute(current, "pgcode")
        for candidate in (sqlstate, pgcode):
            if (
                type(candidate) is str
                and candidate in _POSTGRES_CONCURRENCY_SQLSTATES
            ):
                return candidate
        cause = _exception_attribute(current, "__cause__")
        context = _exception_attribute(current, "__context__")
        suppress_context = _exception_attribute(current, "__suppress_context__")
        if isinstance(cause, BaseException):
            current = cause
        elif suppress_context is not True and isinstance(context, BaseException):
            current = context
        else:
            current = None
    return None


def _exception_attribute(error: BaseException, name: str) -> Any:
    try:
        return getattr(error, name, None)
    except Exception:  # noqa: BLE001 - backend exceptions are untrusted
        return None


__all__ = [
    "ProfileVersionError",
    "ProfileVersionInvalid",
    "ProfileVersionNotFound",
    "ProfileVersionReadFailed",
    "compute_touch_value",
    "normalize_profile_version",
]
