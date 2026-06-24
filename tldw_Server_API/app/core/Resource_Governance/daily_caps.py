from __future__ import annotations

"""
Daily-cap helpers for ResourceGovernor categories.

v1.1 introduces durable per-day caps (e.g., tokens-per-day) backed by the
generic ResourceDailyLedger DAL. Governors consult these helpers when a policy
defines a ``daily_cap`` for a category.

These helpers are best-effort and fail open when the ledger is unavailable to
avoid breaking request flows during upgrades.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from sqlite3 import Error as SQLiteError
from typing import Any

from loguru import logger

try:  # pragma: no cover - asyncpg may be absent in minimal installs
    import asyncpg
except ImportError:  # pragma: no cover - optional dependency guard
    asyncpg = None  # type: ignore[assignment]

try:  # pragma: no cover - AuthNZ exceptions are optional during early import tests
    from tldw_Server_API.app.core.AuthNZ.exceptions import (
        DatabaseError,
        DatabaseLockError,
        TransactionError,
    )
except ImportError:  # pragma: no cover - safe fallback for partial imports
    DatabaseError = DatabaseLockError = TransactionError = RuntimeError  # type: ignore[misc,assignment]

try:  # pragma: no cover - DAL optional in early startup/tests
    from tldw_Server_API.app.core.DB_Management.Resource_Daily_Ledger import (  # type: ignore
        LedgerEntry,
        ResourceDailyLedger,
    )
except ImportError:  # pragma: no cover - safe fallback
    LedgerEntry = None  # type: ignore
    ResourceDailyLedger = None  # type: ignore

_daily_ledger: ResourceDailyLedger | None = None  # type: ignore[name-defined]
_daily_ledger_lock = asyncio.Lock()
_ASYNC_PG_EXCEPTIONS = (asyncpg.PostgresError,) if asyncpg is not None else ()
_LEDGER_NONCRITICAL_EXCEPTIONS = (
    OSError,
    RuntimeError,
    SQLiteError,
    TypeError,
    ValueError,
    DatabaseError,
    DatabaseLockError,
    TransactionError,
    *_ASYNC_PG_EXCEPTIONS,
)


async def _get_ledger() -> ResourceDailyLedger | None:
    global _daily_ledger
    if ResourceDailyLedger is None:
        return None
    if _daily_ledger is not None:
        return _daily_ledger
    async with _daily_ledger_lock:
        if _daily_ledger is not None:
            return _daily_ledger
        try:
            ledger = ResourceDailyLedger()  # type: ignore[call-arg]
            await ledger.initialize()
            _daily_ledger = ledger
            return ledger
        except _LEDGER_NONCRITICAL_EXCEPTIONS as exc:  # pragma: no cover - defensive
            logger.debug(f"RG daily caps: ledger init failed; caps disabled: {exc}")
            _daily_ledger = None
            return None


def _seconds_until_next_utc_day(now_dt: datetime | None = None) -> int:
    dt = now_dt or datetime.now(timezone.utc)
    dt = dt.astimezone(timezone.utc)
    tomorrow = dt.date() + timedelta(days=1)
    next_midnight = datetime.combine(tomorrow, datetime.min.time(), tzinfo=timezone.utc)
    try:
        return max(1, int((next_midnight - dt).total_seconds()))
    except (OverflowError, TypeError, ValueError):
        return 60 * 60 * 24


async def check_daily_cap(
    *,
    entity_scope: str,
    entity_value: str,
    category: str,
    daily_cap: int,
    units: int,
    day_utc: str | None = None,
) -> tuple[bool, int, dict[str, Any]]:
    """
    Check whether an entity has remaining daily headroom for the given category.

    Returns (allowed, retry_after_seconds, details). When the ledger is
    unavailable or daily_cap <= 0, this returns (True, 0, {}).
    """
    try:
        cap = int(daily_cap or 0)
        if cap <= 0:
            return True, 0, {}
    except (TypeError, ValueError):
        return True, 0, {}

    ledger = await _get_ledger()
    if ledger is None:
        return True, 0, {}

    try:
        used = await ledger.total_for_day(
            entity_scope=str(entity_scope),
            entity_value=str(entity_value),
            category=str(category),
            day_utc=day_utc,
        )
        remaining = max(0, cap - int(used or 0))
        allowed = remaining >= int(units or 0)
        retry_after = _seconds_until_next_utc_day()
        details = {
            "daily_cap": cap,
            "daily_used": int(used or 0),
            "daily_remaining": int(remaining),
            "daily_reset_seconds": int(retry_after),
        }
        return bool(allowed), int(retry_after), details
    except _LEDGER_NONCRITICAL_EXCEPTIONS as exc:  # pragma: no cover - defensive
        logger.debug(f"RG daily caps: check failed for {entity_scope}:{entity_value}:{category}: {exc}")
        return True, 0, {}


async def consume_daily_cap(
    *,
    entity_scope: str,
    entity_value: str,
    category: str,
    daily_cap: int,
    units: int,
    op_id: str,
    day_utc: str | None = None,
) -> tuple[bool, int, dict[str, Any]]:
    """
    Idempotently consume daily-cap headroom for a reservation.

    The durable ledger's unique op index prevents double-counting the same
    reservation. The ledger DAL serializes the cap check and insert at the
    database level so multiple app workers cannot overspend the same cap.
    """
    try:
        cap = int(daily_cap or 0)
        units_i = max(0, int(units or 0))
        op = str(op_id or "").strip()
        if cap <= 0 or units_i <= 0 or not op:
            return True, 0, {}
    except (TypeError, ValueError):
        return True, 0, {}

    ledger = await _get_ledger()
    if ledger is None or LedgerEntry is None:
        return True, 0, {}

    try:
        result = await ledger.consume_if_within_cap(
            LedgerEntry(
                entity_scope=str(entity_scope),
                entity_value=str(entity_value),
                category=str(category),
                units=units_i,
                op_id=op,
                occurred_at=datetime.now(timezone.utc),
            ),
            daily_cap=cap,
            day_utc=day_utc,
        )
        retry_after = _seconds_until_next_utc_day()
        used = int(result.used or 0)
        return bool(result.allowed), int(retry_after), {
            "daily_cap": cap,
            "daily_used": used,
            "daily_remaining": max(0, cap - used),
            "daily_reset_seconds": int(retry_after),
        }
    except _LEDGER_NONCRITICAL_EXCEPTIONS as exc:  # pragma: no cover - defensive
        logger.debug(f"RG daily caps: consume failed for {entity_scope}:{entity_value}:{category}: {exc}")
        return True, 0, {}
