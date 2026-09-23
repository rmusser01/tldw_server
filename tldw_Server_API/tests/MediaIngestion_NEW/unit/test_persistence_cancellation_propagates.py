"""Regression guard for TASK-13295 -- cancellation must not be swallowed as noncritical.

`_PERSISTENCE_NONCRITICAL_EXCEPTIONS` was written to replace `except Exception:` and
satisfy ruff BLE001, but it listed `asyncio.CancelledError` first. That class derives
from BaseException, not Exception, so the bare handler it replaced never caught it --
the remediation widened what is suppressed and introduced a defect the lint rule cannot
see.

Effect on the ~140 sites using the tuple, several of which wrap awaits: a client
disconnecting mid-`POST /media/add` is absorbed, the coroutine keeps writing to the
media DB for a client that is gone, and `Task.cancel()` never completes, so graceful
shutdown blocks on that task.

The suppression is also asserted directly against the tuple object rather than only
through one call path, because the tuple is the shared thing -- the ratchet in
tests/lint/test_noncritical_exception_tuples.py covers the rest of the codebase.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing import persistence

pytestmark = pytest.mark.unit


def test_the_tuple_does_not_catch_cancellation() -> None:
    """The shared tuple itself, independent of any call path."""
    with pytest.raises(asyncio.CancelledError):
        with contextlib.suppress(persistence._PERSISTENCE_NONCRITICAL_EXCEPTIONS):
            raise asyncio.CancelledError


def test_no_member_of_the_tuple_derives_from_baseexception_only() -> None:
    """Assert the whole tuple, not just cancellation, so a sibling cannot creep in.

    The tuple object is inspected directly because the tuple *is* the defect: its
    membership is the shared thing every one of the ~140 handlers in this module
    resolves. The awaiting-call-path test below covers the consequence.
    """
    offenders = [
        exc.__name__
        for exc in persistence._PERSISTENCE_NONCRITICAL_EXCEPTIONS
        if not issubclass(exc, Exception)
    ]
    assert not offenders, (
        "these members are BaseException-derived, so `except Exception:` never caught "
        f"them and listing them changed behaviour: {offenders}"
    )


@pytest.mark.asyncio
async def test_cancellation_during_ledger_init_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end through a real awaiting call path in persistence.py.

    `_get_media_ingestion_daily_ledger` awaits `ledger.initialize()` inside a
    try/except guarded by the tuple. With CancelledError in the tuple the cancellation
    is logged at debug level and the function returns None, so the caller proceeds as
    if the ledger were merely unavailable.
    """
    monkeypatch.setattr(persistence, "_media_ingestion_daily_ledger", None)

    from tldw_Server_API.app.core.DB_Management import Resource_Daily_Ledger

    class _CancellingLedger:
        async def initialize(self) -> None:
            raise asyncio.CancelledError

    monkeypatch.setattr(
        Resource_Daily_Ledger, "ResourceDailyLedger", _CancellingLedger
    )

    with pytest.raises(asyncio.CancelledError):
        await persistence._get_media_ingestion_daily_ledger()


@pytest.mark.asyncio
async def test_ordinary_failures_are_still_absorbed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: the tuple must keep doing its actual job."""
    monkeypatch.setattr(persistence, "_media_ingestion_daily_ledger", None)

    from tldw_Server_API.app.core.DB_Management import Resource_Daily_Ledger

    class _FailingLedger:
        async def initialize(self) -> None:
            raise RuntimeError("ledger backend down")

    monkeypatch.setattr(
        Resource_Daily_Ledger, "ResourceDailyLedger", _FailingLedger
    )

    assert await persistence._get_media_ingestion_daily_ledger() is None
