"""Compatibility shim for the canonical governed preflight aggregate APIs."""

from __future__ import annotations

from ..preflight.runner import gather_analysis, run_analysis

__all__ = ["gather_analysis", "run_analysis"]
