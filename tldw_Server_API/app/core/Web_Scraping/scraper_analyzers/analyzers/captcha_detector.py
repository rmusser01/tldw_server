"""Compatibility exports for the governed CAPTCHA analyzer."""

from __future__ import annotations

from ...preflight.analyzers.captcha_detector import (
    CAPTCHA_FINGERPRINTS,
    detect_captcha,
)

__all__ = ["CAPTCHA_FINGERPRINTS", "detect_captcha"]
