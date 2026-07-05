"""Cancellation helpers for Web_Scraping runtime boundaries."""

from __future__ import annotations

import asyncio


def is_cancellation(exc: BaseException) -> bool:
    """Return True when an exception represents task cancellation."""

    return isinstance(exc, asyncio.CancelledError)
