"""Shared VN Play service errors."""

from __future__ import annotations


class VNPlayError(Exception):
    """Base error raised by VN Play runtime services."""


class VNPlayNotFoundError(VNPlayError):
    """Raised when a VN Play resource cannot be found for the current owner."""


class VNPlayConflictError(VNPlayError):
    """Raised when the requested turn cannot be applied to current session state."""


class VNPlayTurnError(VNPlayError):
    """Raised when a turn attempt fails after being accepted."""
