"""Deprecated compatibility re-export; implementation lives in Web_Scraping.preflight."""

from __future__ import annotations

from tldw_Server_API.app.core.Web_Scraping.preflight.utils.impersonate_target import (
    get_impersonate_target,
)

__all__ = ["get_impersonate_target"]
