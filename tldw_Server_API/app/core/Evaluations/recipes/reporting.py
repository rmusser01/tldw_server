"""Reporting models for recipe evaluations."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tldw_Server_API.app.core.Evaluations.evaluation_recipe_schemas import (
    ConfidenceSummary,
    RecommendationSlot,
    RecipeRunRecord,
)


class RecipeRunReport(BaseModel):
    """A lightweight report payload for a recipe run."""

    run: RecipeRunRecord
    confidence_summary: ConfidenceSummary | None = None
    recommendation_slots: dict[str, RecommendationSlot] = Field(default_factory=dict)
