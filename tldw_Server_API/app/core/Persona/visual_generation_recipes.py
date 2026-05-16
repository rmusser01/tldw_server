"""Recipe-backed Persona Visual generation request helpers.

This module keeps bundled starter production recipes as bounded, trace-safe
intent metadata over the existing generated-candidate Jobs flow. It validates
starter/output pairs, composes the effective image-generation prompt, and
normalizes correlation IDs without executing providers or mutating visual packs.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any

from tldw_Server_API.app.core.Persona.visual_starter_fixtures import (
    DEFAULT_PERSONA_VISUAL_STARTER_PACK_ID,
    DEFAULT_PERSONA_VISUAL_STARTER_PACKS,
    LEGACY_PERSONA_VISUAL_STARTER_PACK_ID,
    PersonaVisualStarterPack,
)


MAX_PERSONA_VISUAL_GENERATION_PROMPT_LENGTH = 4000
MAX_PERSONA_VISUAL_GENERATION_REQUEST_ID_LENGTH = 120
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,119}$")
_SECRET_MARKERS = ("api_key", "apikey", "bearer ", "password", "secret", "token=", "sk-")


@dataclass(frozen=True)
class PersonaVisualRecipeGenerationIntent:
    """Resolved recipe generation request metadata for one queued job."""

    request_id: str
    effective_prompt: str
    recipe_intent: dict[str, Any]


def normalize_persona_visual_generation_request_id(request_id: str | None) -> str:
    """Return a trace-safe request identifier or generate a bounded one."""
    if request_id is None or str(request_id).strip() == "":
        return f"pvgen:{uuid.uuid4().hex}"
    normalized = str(request_id).strip()
    lower = normalized.lower()
    if (
        len(normalized) > MAX_PERSONA_VISUAL_GENERATION_REQUEST_ID_LENGTH
        or not _REQUEST_ID_RE.match(normalized)
        or "/" in normalized
        or "\\" in normalized
        or any(marker in lower for marker in _SECRET_MARKERS)
    ):
        raise ValueError("invalid_request_id")
    return normalized


def build_persona_visual_recipe_generation_intent(
    *,
    starter_pack_id: str | None,
    recipe_output: str | None,
    user_prompt: str,
    request_id: str,
    starter_packs: tuple[PersonaVisualStarterPack, ...] = DEFAULT_PERSONA_VISUAL_STARTER_PACKS,
) -> PersonaVisualRecipeGenerationIntent | None:
    """Validate optional recipe fields and return effective prompt metadata."""
    starter_id = str(starter_pack_id or "").strip()
    output_id = str(recipe_output or "").strip()
    if not starter_id and not output_id:
        return None
    if starter_id and not output_id:
        raise ValueError("recipe_output_required_with_starter_pack_id")
    if output_id and not starter_id:
        raise ValueError("starter_pack_id_required_with_recipe_output")

    starter = _starter_by_id(starter_id, starter_packs=starter_packs)
    recipe = starter.production_recipe
    animation_outputs = tuple(str(item).strip() for item in recipe.animation_outputs)
    if output_id not in animation_outputs:
        raise ValueError("recipe_output_not_found")

    normalized_prompt = str(user_prompt or "").strip()
    if not normalized_prompt:
        raise ValueError("prompt is required")

    review_checks = [str(item).strip() for item in recipe.review_checks if str(item).strip()]
    recipe_intent = {
        "starter_pack_id": starter.id,
        "recipe_output": output_id,
        "correlation_id": request_id,
        "user_prompt": normalized_prompt,
        "identity_brief": str(recipe.identity_brief).strip(),
        "neutral_anchor": str(recipe.neutral_anchor).strip(),
        "static_sheet": str(recipe.static_sheet).strip(),
        "review_checks": review_checks,
    }
    effective_prompt = _compose_effective_prompt(
        starter=starter,
        recipe_output=output_id,
        recipe_intent=recipe_intent,
    )
    if len(effective_prompt) > MAX_PERSONA_VISUAL_GENERATION_PROMPT_LENGTH:
        raise ValueError("recipe_prompt_too_long")
    return PersonaVisualRecipeGenerationIntent(
        request_id=request_id,
        effective_prompt=effective_prompt,
        recipe_intent=recipe_intent,
    )


def _starter_by_id(
    starter_pack_id: str,
    *,
    starter_packs: tuple[PersonaVisualStarterPack, ...],
) -> PersonaVisualStarterPack:
    starter_id = str(starter_pack_id or "").strip()
    if starter_id == LEGACY_PERSONA_VISUAL_STARTER_PACK_ID:
        starter_id = DEFAULT_PERSONA_VISUAL_STARTER_PACK_ID
    for starter in starter_packs:
        if starter.id == starter_id:
            return starter
    raise ValueError("starter_pack_not_found")


def _compose_effective_prompt(
    *,
    starter: PersonaVisualStarterPack,
    recipe_output: str,
    recipe_intent: dict[str, Any],
) -> str:
    review_checks = ", ".join(recipe_intent["review_checks"])
    return "\n".join(
        (
            "Generate a Persona Visual asset candidate from a bundled starter production recipe.",
            f"Starter pack: {starter.id}",
            f"Recipe output: {recipe_output}",
            f"Identity brief: {recipe_intent['identity_brief']}",
            f"Neutral anchor: {recipe_intent['neutral_anchor']}",
            f"Static sheet guidance: {recipe_intent['static_sheet']}",
            f"Review checks: {review_checks}",
            f"User direction: {recipe_intent['user_prompt']}",
        )
    )


__all__ = [
    "MAX_PERSONA_VISUAL_GENERATION_PROMPT_LENGTH",
    "MAX_PERSONA_VISUAL_GENERATION_REQUEST_ID_LENGTH",
    "PersonaVisualRecipeGenerationIntent",
    "build_persona_visual_recipe_generation_intent",
    "normalize_persona_visual_generation_request_id",
]
