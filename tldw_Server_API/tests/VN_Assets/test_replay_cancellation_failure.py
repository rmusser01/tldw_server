"""A drained failing replay preserves requested cancellation and native failures."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.tests.VN_Assets.test_cancelled_storage_handoff import (
    handoff as handoff,
)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_count", [0, 1, 2])
async def test_replay_cancel_wins_over_drained_operation_failure(
    handoff: SimpleNamespace, cancel_count: int,
) -> None:
    """Cancellation wins after rollback/close; uncancelled native errors still propagate."""
    repo = handoff.service.repo
    owner = repo.db.get_connection()
    started, release = threading.Event(), threading.Event()
    observed: list[tuple[threading.Thread, sqlite3.Connection]] = []
    original_batch = repo.get_batch(handoff.payload["batch_id"])

    def delayed_failure() -> dict[str, Any] | None:
        """Hold a real operation-owned write and then fail its transaction."""
        observed.append((threading.current_thread(), repo.db.get_connection()))
        with repo.db.transaction() as connection:
            connection.execute(
                "UPDATE vn_asset_batches SET completed_count=99 WHERE id=?", (handoff.payload["batch_id"],),
            )
            started.set()
            assert release.wait(3), "failure drain release timed out"
            raise OSError("failing replay transition")

    task = asyncio.create_task(repo.run_worker_replay_operation(delayed_failure))
    try:
        assert await asyncio.to_thread(started.wait, 3)
        for _request in range(cancel_count):
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        release.set()
        expected_error = asyncio.CancelledError if cancel_count else OSError
        with pytest.raises(expected_error):
            await task
        assert task.cancelled() is bool(cancel_count)
        assert repo.get_batch(handoff.payload["batch_id"]) == original_batch
        assert owner.execute("SELECT 1").fetchone()[0] == 1
        thread, connection = observed[0]
        assert not thread.is_alive()
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
    finally:
        release.set()
        if not task.done():
            await task
