"""Application lifespan must release owned resources when ASGI lifespan is cancelled."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize("exit_kind", ["normal", "cancelled", "body_error"])
async def test_main_lifespan_releases_worker_and_preserves_exit(
    monkeypatch: pytest.MonkeyPatch, exit_kind: str,
) -> None:
    """Shutdown must run even when the server cancels the suspended lifespan yield."""
    from tldw_Server_API.app import main
    from tldw_Server_API.app.services import (
        lifespan_shutdown_sequence,
        lifespan_startup_sequence,
    )

    stop = threading.Event()
    worker = threading.Thread(target=stop.wait, name="uat287-owned-worker", daemon=False)
    handles = SimpleNamespace(db_pool=object(), session_manager=object(), heavy_startup_handles=object())
    entered = asyncio.Event()

    async def startup(**_kwargs: object) -> SimpleNamespace:
        worker.start()
        return handles

    async def shutdown(**_kwargs: object) -> None:
        # Exercise an actual suspension during cleanup of a cancelled task.
        await asyncio.sleep(0)
        stop.set()
        worker.join(timeout=1)

    shutdown_spy = AsyncMock(side_effect=shutdown)
    monkeypatch.setattr(lifespan_startup_sequence, "run_lifespan_startup_sequence", startup)
    monkeypatch.setattr(lifespan_shutdown_sequence, "run_lifespan_shutdown_sequence", shutdown_spy)
    monkeypatch.setattr(main, "_run_startup_config_validation", lambda: None)

    async def serve() -> None:
        async with main.lifespan(FastAPI()):
            entered.set()
            if exit_kind == "cancelled":
                await asyncio.Event().wait()
            if exit_kind == "body_error":
                raise RuntimeError("simulated lifespan body failure")

    task = asyncio.create_task(serve())
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        if exit_kind == "cancelled":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif exit_kind == "body_error":
            with pytest.raises(RuntimeError, match="simulated lifespan body failure"):
                await task
        else:
            await task

        assert not worker.is_alive(), "lifespan skipped shutdown and left its owned worker running"
        shutdown_spy.assert_awaited_once()
        assert shutdown_spy.call_args.kwargs["db_pool"] is handles.db_pool
        assert shutdown_spy.call_args.kwargs["session_manager"] is handles.session_manager
    finally:
        # Red regressions must not hang the pytest interpreter itself.
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        stop.set()
        if worker.ident is not None:
            worker.join(timeout=1)
