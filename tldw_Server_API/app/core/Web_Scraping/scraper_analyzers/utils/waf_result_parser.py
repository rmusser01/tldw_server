"""Deprecated compatibility re-export; implementation lives in Web_Scraping.preflight."""

from __future__ import annotations

from tldw_Server_API.app.core.Web_Scraping.preflight.utils.waf_result_parser import (
    ANSI_RE,
    GENERIC_PHRASES,
    clean_text,
    parse_wafw00f_output,
)

__all__ = ["ANSI_RE", "GENERIC_PHRASES", "clean_text", "parse_wafw00f_output"]
