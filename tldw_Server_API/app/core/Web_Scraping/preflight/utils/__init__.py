"""Shared utilities for the scraping analyzers."""

from __future__ import annotations

from .browser_identities import MODERN_BROWSER_IDENTITIES
from .impersonate_target import get_impersonate_target
from .waf_result_parser import parse_wafw00f_output

__all__ = ["MODERN_BROWSER_IDENTITIES", "get_impersonate_target", "parse_wafw00f_output"]
