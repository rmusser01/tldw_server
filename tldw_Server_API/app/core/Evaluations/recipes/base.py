"""Shared recipe interfaces and lightweight built-in recipe definitions."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

from tldw_Server_API.app.core.Evaluations.evaluation_recipe_schemas import RecipeManifest


class RecipeDefinition(ABC):
    """Base interface for a recipe definition."""

    manifest: RecipeManifest

    @property
    def recipe_id(self) -> str:
        return self.manifest.recipe_id

    def get_manifest(self) -> RecipeManifest:
        return self.manifest.model_copy(deep=True)


@dataclass(frozen=True)
class StaticRecipeDefinition(RecipeDefinition):
    """Simple recipe definition that only exposes a manifest."""

    manifest: RecipeManifest
