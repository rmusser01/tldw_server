"""Compatibility exports for the governed function-integrity analyzer."""

from __future__ import annotations

from ...preflight.analyzers.integrity_analyzer import (
    FUNCTION_SUSPICION_MAP,
    FUNCTIONS_TO_CHECK,
    analyze_function_integrity,
)

__all__ = [
    "FUNCTIONS_TO_CHECK",
    "FUNCTION_SUSPICION_MAP",
    "analyze_function_integrity",
]
