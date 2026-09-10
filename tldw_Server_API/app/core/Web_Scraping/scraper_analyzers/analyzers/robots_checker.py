"""Deprecated compatibility re-export; implementation lives in Web_Scraping.preflight."""

from __future__ import annotations

from tldw_Server_API.app.core.Web_Scraping.preflight.analyzers.robots_checker import (
    check_robots_txt,
)

__all__ = ["check_robots_txt"]
