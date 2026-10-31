"""Compatibility exports for the governed honeypot analyzer."""

from __future__ import annotations

from ...preflight.analyzers.behavioral_detector import (
    HONEYPOT_THRESHOLD,
    ScanDepth,
    detect_honeypots,
)

__all__ = ["HONEYPOT_THRESHOLD", "ScanDepth", "detect_honeypots"]
