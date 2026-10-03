from __future__ import annotations

"""
Workflows daily-ledger helpers (RG v1.1).

These helpers provide best-effort, idempotent accounting of workflow runs into
the shared ResourceDailyLedger so ResourceGovernor can enforce
``workflows_runs.daily_cap`` policies.

All functions fail open when the ledger or AuthNZ DB is unavailable.
"""

import asyncio
from datetime import datetime, timezone
from sqlite3 import Error as SQLiteError

from loguru import logger

try:  # pragma: no cover - DAL optional during early startup/tests
    from tldw_Server_API.app.core.DB_Management.Resource_Daily_Ledger import (  # type: ignore
        LedgerEntry,
        ResourceDailyLedger,
    )
except ImportError:  # pragma: no cover - safe fallback
    LedgerEntry = None  # type: ignore
    ResourceDailyLedger = None  # type: ignore

_WORKFLOWS_LEDGER_NONCRITICAL_EXCEPTIONS = (
    OSError,
    RuntimeError,
    SQLiteError,
    TimeoutError,
    TypeError,
    ValueError,
)


_WORKFLOWS_CATEGORY = "workflows_runs"

_workflows_daily_ledger: ResourceDailyLedger | None = None  # type: ignore[name-defined]
_workflows_daily_ledger_lock = asyncio.Lock()


async def get_workflows_daily_ledger() -> ResourceDailyLedger | None:
    """Lazily initialize the shared ResourceDailyLedger for workflows."""
    global _workflows_daily_ledger
    if ResourceDailyLedger is None:
        return None
    if _workflows_daily_ledger is not None:
        return _workflows_daily_ledger
    async with _workflows_daily_ledger_lock:
        if _workflows_daily_ledger is not None:
            return _workflows_daily_ledger
        try:
            ledger = ResourceDailyLedger()  # type: ignore[call-arg]
            await ledger.initialize()
            _workflows_daily_ledger = ledger
            return ledger
        except _WORKFLOWS_LEDGER_NONCRITICAL_EXCEPTIONS as exc:  # pragma: no cover - defensive
            logger.debug(f"Workflows: ResourceDailyLedger init failed: {exc}")
            _workflows_daily_ledger = None
            return None


def workflows_ledger_category() -> str:
    """Return the ledger category name used for workflow runs."""
    return _WORKFLOWS_CATEGORY


async def record_workflow_run(
    *,
    entity_scope: str,
    entity_value: str,
    run_id: str,
    units: int = 1,
    occurred_at: datetime | None = None,
) -> bool:
    """
    Shadow-write a workflow run into the daily ledger.

    Returns True if inserted; False if already present or ledger unavailable.
    """
    if ResourceDailyLedger is None or LedgerEntry is None:
        return False
    ledger = await get_workflows_daily_ledger()
    if ledger is None:
        return False

    ts = occurred_at or datetime.now(timezone.utc)
    try:
        entry = LedgerEntry(  # type: ignore[call-arg]
            entity_scope=str(entity_scope),
            entity_value=str(entity_value),
            category=_WORKFLOWS_CATEGORY,
            units=max(0, int(units)),
            op_id=str(run_id),
            occurred_at=ts,
        )
        return bool(await ledger.add(entry))
    except _WORKFLOWS_LEDGER_NONCRITICAL_EXCEPTIONS as exc:  # pragma: no cover - defensive
        logger.debug(f"Workflows: ledger.add failed for run_id={run_id}: {exc}")
        return False


async def consume_workflow_run_if_within_cap(
    *,
    entity_scope: str,
    entity_value: str,
    run_id: str,
    daily_cap: int | None,
) -> tuple[bool, int]:
    """Atomically admit and record one workflow run against the daily cap (Qodo Q17).

    The admission check and the ledger write are the single atomic
    ``ResourceDailyLedger.add_if_within_daily_cap`` operation, so two
    concurrent runs cannot both pass against the same remaining slot.

    ``daily_cap`` of ``None`` means unlimited: always admits, and still
    records (gate the check, never the record). Fails open (admits, does
    not record) when the ledger is unavailable or errors.

    Returns ``(allowed, remaining)``: ``remaining`` is the cap headroom
    after admission (or before, when refused), for 429 header reporting.
    """
    if ResourceDailyLedger is None or LedgerEntry is None:
        return True, 0
    ledger = await get_workflows_daily_ledger()
    if ledger is None:
        return True, 0

    entry = LedgerEntry(  # type: ignore[call-arg]
        entity_scope=str(entity_scope),
        entity_value=str(entity_value),
        category=_WORKFLOWS_CATEGORY,
        units=1,
        op_id=str(run_id),
        occurred_at=datetime.now(timezone.utc),
    )
    if daily_cap is None:
        try:
            await ledger.add(entry)
        except _WORKFLOWS_LEDGER_NONCRITICAL_EXCEPTIONS as exc:  # pragma: no cover - defensive
            logger.debug(f"Workflows: ledger.add failed for run_id={run_id}: {exc}")
        return True, 0
    try:
        allowed, remaining = await ledger.add_if_within_daily_cap(entry, int(daily_cap))
        return bool(allowed), int(remaining)
    except _WORKFLOWS_LEDGER_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(f"Workflows: atomic cap consume failed for run_id={run_id}; failing open: {exc}")
        return True, 0
