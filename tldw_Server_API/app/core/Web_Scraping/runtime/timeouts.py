"""Timeout and budget contracts for Web_Scraping runtime operations."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimeTimeouts:
    """Runtime timeout values in seconds."""

    fetch_timeout_s: float | None = None
    browser_timeout_s: float | None = None
    preflight_timeout_s: float | None = None

    def __post_init__(self) -> None:
        for field_name in ("fetch_timeout_s", "browser_timeout_s", "preflight_timeout_s"):
            value = getattr(self, field_name)
            if value is None:
                continue
            normalized = float(value)
            if not math.isfinite(normalized):
                raise ValueError(f"{field_name} must be finite")
            if normalized < 0:
                raise ValueError(f"{field_name} must be non-negative")
            object.__setattr__(self, field_name, normalized)
