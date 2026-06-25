from __future__ import annotations


class RPGError(Exception):
    """Base exception for RPG runtime errors."""


class RPGNotFoundError(RPGError):
    """Raised when an RPG resource cannot be found."""


class RPGValidationError(RPGError):
    """Raised when RPG input fails domain validation."""


class RPGConflictError(RPGError):
    """Raised when an RPG write conflicts with current state."""
