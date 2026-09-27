"""Deprecated compatibility re-export; implementation lives in Web_Scraping.preflight."""

from __future__ import annotations

from tldw_Server_API.app.core.Web_Scraping.preflight.analyzers.waf_detector import (
    detect_waf,
)

__all__ = ["detect_waf"]
