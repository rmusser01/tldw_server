"""Cancellation-aware bridge from async scraping into synchronous extraction."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import partial
from threading import Event
from typing import Any, TypeVar

from .extraction.dependencies import cancellation_checkpoint_scope

_T = TypeVar("_T")


def _raise_if_cancelled(cancelled: Event) -> None:
    if cancelled.is_set():
        raise asyncio.CancelledError


async def run_extraction_in_thread(
    func: Callable[..., _T],
    /,
    *args: Any,
    **kwargs: Any,
) -> _T:
    """Offload extraction and notify its cooperative checkpoints on cancellation."""

    cancelled = Event()
    checkpoint = partial(_raise_if_cancelled, cancelled)
    with cancellation_checkpoint_scope(checkpoint):
        try:
            return await asyncio.to_thread(func, *args, **kwargs)
        except asyncio.CancelledError:
            cancelled.set()
            raise
