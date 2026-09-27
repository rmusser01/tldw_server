"""Deprecated compatibility re-export; implementation lives in Web_Scraping.preflight."""

from __future__ import annotations

from tldw_Server_API.app.core.Web_Scraping.preflight.scoring.scoring_engine import (
    calculate_difficulty_score,
)

__all__ = ["calculate_difficulty_score"]
