"""Compatibility exports for the governed fingerprint analyzer."""

from __future__ import annotations

from ...preflight.analyzers.fingerprint_analyzer import (
    JS_PROBE_SCRIPT,
    KNOWN_BOT_DETECTION_SCRIPTS,
    KNOWN_BOT_GLOBAL_OBJECTS,
    analyze_fingerprinting,
)

__all__ = [
    "JS_PROBE_SCRIPT",
    "KNOWN_BOT_DETECTION_SCRIPTS",
    "KNOWN_BOT_GLOBAL_OBJECTS",
    "analyze_fingerprinting",
]
