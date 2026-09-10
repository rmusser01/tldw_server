"""Deprecated compatibility re-export; implementation lives in Web_Scraping.preflight."""

from __future__ import annotations

from tldw_Server_API.app.core.Web_Scraping.preflight.analyzers.rate_limit_profiler import (
    BLOCKING_STATUS_CODES,
    BURST_COUNT,
    DEFAULT_DELAY,
    GENTLE_PROBE_COUNT,
    profile_rate_limits,
)

__all__ = [
    "GENTLE_PROBE_COUNT",
    "BURST_COUNT",
    "DEFAULT_DELAY",
    "BLOCKING_STATUS_CODES",
    "profile_rate_limits",
]
