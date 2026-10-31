"""Deprecated compatibility re-export; implementation lives in Web_Scraping.preflight."""

from __future__ import annotations

from tldw_Server_API.app.core.Web_Scraping.preflight.recommendations.recommender import (
    generate_recommendations,
)

__all__ = ["generate_recommendations"]
