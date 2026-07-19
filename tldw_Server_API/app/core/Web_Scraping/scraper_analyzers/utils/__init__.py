"""Deprecated compatibility re-export; implementation lives in Web_Scraping.preflight."""

from __future__ import annotations

from tldw_Server_API.app.core.Web_Scraping.preflight.utils.browser_identities import (
    MODERN_BROWSER_IDENTITIES,
)
from tldw_Server_API.app.core.Web_Scraping.preflight.utils.impersonate_target import (
    get_impersonate_target,
)
from tldw_Server_API.app.core.Web_Scraping.preflight.utils.waf_result_parser import (
    parse_wafw00f_output,
)

__all__ = ["MODERN_BROWSER_IDENTITIES", "get_impersonate_target", "parse_wafw00f_output"]
