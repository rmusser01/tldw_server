"""Shared bounded retry-delay policy for extraction strategies."""

from __future__ import annotations

import math
import os

_DEFAULT_RETRY_MAX_DELAY_MS = 30_000.0


def cap_retry_delay(delay_seconds: float) -> float:
    """Cap a complete base-plus-jitter delay by the configured maximum."""

    raw = os.getenv("EXTRACTOR_RETRY_MAX_DELAY_MS")
    try:
        maximum_ms = _DEFAULT_RETRY_MAX_DELAY_MS if raw is None else float(raw)
    except (TypeError, ValueError):
        maximum_ms = _DEFAULT_RETRY_MAX_DELAY_MS
    if not math.isfinite(maximum_ms) or maximum_ms < 0.0:
        maximum_ms = _DEFAULT_RETRY_MAX_DELAY_MS
    return min(max(0.0, delay_seconds), maximum_ms / 1000.0)
