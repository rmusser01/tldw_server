"""Run LLM usage accounting on one event loop (LLM usage accounting spec §5).

The AuthNZ pool is bound to one loop, and on Postgres using it from another
loop recreates it. Accounting writes are therefore scheduled on the server's
loop from wherever the LLM call ran, and never through ``asyncio.run``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

_accounting_loop: asyncio.AbstractEventLoop | None = None
_pending: set[Any] = set()


def set_accounting_loop(loop: asyncio.AbstractEventLoop | None) -> None:
    """Record the loop accounting runs on (the server's loop, set at startup)."""
    global _accounting_loop
    _accounting_loop = loop


def get_accounting_loop() -> asyncio.AbstractEventLoop | None:
    """The accounting loop, or None when none was set, it has closed, or it is not running."""
    loop = _accounting_loop
    # A stopped loop would accept a threadsafe hand-off and never run it.
    if loop is None or loop.is_closed() or not loop.is_running():
        return None
    return loop


def _on_done(future: Any) -> None:
    """Forget a finished write and log its failure."""
    _pending.discard(future)
    if future.cancelled():
        return
    error = future.exception()
    if error is not None:
        logger.bind(exception_type=type(error).__name__).warning("LLM usage accounting write failed")


def submit_accounting(make_coro: Callable[[], Awaitable[Any]]) -> bool:
    """Schedule ``make_coro()`` on the accounting loop from any thread; never raises.

    Fire-and-forget. Returns False when there is no loop to run it on.
    """
    coro: Any = None
    try:
        try:
            running: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        target = get_accounting_loop() or running
        if target is None:
            return False
        coro = make_coro()
        future = target.create_task(coro) if running is target else asyncio.run_coroutine_threadsafe(coro, target)
        _pending.add(future)
        future.add_done_callback(_on_done)
        return True
    except Exception as error:  # noqa: BLE001 - accounting must never break an LLM call
        close = getattr(coro, "close", None)
        if callable(close):
            close()
        logger.bind(exception_type=type(error).__name__).warning("LLM usage accounting could not be scheduled")
        return False


async def drain_accounting(timeout: float = 5.0) -> None:
    """Wait until scheduled writes have finished (tests and shutdown)."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while _pending and loop.time() < deadline:
        await asyncio.sleep(0.01)
