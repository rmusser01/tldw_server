"""Shared exception policy for best-effort lifecycle metadata operations."""

from __future__ import annotations

LIFECYCLE_GUARD_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
