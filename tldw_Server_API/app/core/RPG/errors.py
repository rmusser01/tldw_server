"""Compatibility exports for RPG domain exceptions."""

from __future__ import annotations

from tldw_Server_API.app.core.exceptions import (
    RPGConflictError,
    RPGError,
    RPGNotFoundError,
    RPGValidationError,
)

__all__ = [
    "RPGConflictError",
    "RPGError",
    "RPGNotFoundError",
    "RPGValidationError",
]
