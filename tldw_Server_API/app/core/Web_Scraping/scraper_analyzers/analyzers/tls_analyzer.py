"""Deprecated compatibility re-export; implementation lives in Web_Scraping.preflight."""

from __future__ import annotations

from tldw_Server_API.app.core.Web_Scraping.preflight.analyzers.tls_analyzer import (
    analyze_tls_fingerprint,
)

__all__ = ["analyze_tls_fingerprint"]
