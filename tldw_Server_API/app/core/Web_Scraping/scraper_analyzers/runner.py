"""Compatibility shim for the canonical governed preflight runner."""

from __future__ import annotations

from ..preflight.runner import (
    AnalysisOutput,
    ScanDepth,
    analyze_fingerprinting,
    analyze_function_integrity,
    analyze_js_rendering,
    analyze_tls_fingerprint,
    check_robots_txt,
    detect_captcha,
    detect_honeypots,
    detect_waf,
    gather_analysis,
    profile_rate_limits,
    run_analysis,
)

__all__ = [
    "AnalysisOutput",
    "ScanDepth",
    "analyze_fingerprinting",
    "analyze_function_integrity",
    "analyze_js_rendering",
    "analyze_tls_fingerprint",
    "check_robots_txt",
    "detect_captcha",
    "detect_honeypots",
    "detect_waf",
    "gather_analysis",
    "profile_rate_limits",
    "run_analysis",
]
