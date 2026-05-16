"""Trace-safe provenance metadata for Persona Visual generated candidates.

Generation jobs may carry prompt, provider, recipe, and request metadata that is
useful during human review but unsafe to echo directly. This module builds and
normalizes the small candidate provenance payload stored with review candidates:
stable IDs, bounded labels, and recipe summary fields only. Raw prompts and
unknown keys are intentionally excluded.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


PERSONA_VISUAL_CANDIDATE_PROVENANCE_SCHEMA_VERSION = 1
PERSONA_VISUAL_CANDIDATE_PROVENANCE_ID_TEXT_LIMIT = 128
PERSONA_VISUAL_CANDIDATE_PROVENANCE_SUMMARY_TEXT_LIMIT = 240
PERSONA_VISUAL_CANDIDATE_PROVENANCE_REVIEW_CHECK_LIMIT = 12
PERSONA_VISUAL_CANDIDATE_PROVENANCE_REVIEW_CHECK_TEXT_LIMIT = 120

_ALLOWED_GENERATION_MODES = frozenset({"prompt_only", "recipe_backed"})
_SECRET_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer ",
    "password",
    "secret",
    "sk-",
    "token",
    "x-api-key",
)
_PATH_MARKERS = (
    "/home/",
    "/private/",
    "/users/",
    "\\",
)
_TOP_LEVEL_TEXT_FIELDS = {
    "request_id": PERSONA_VISUAL_CANDIDATE_PROVENANCE_ID_TEXT_LIMIT,
    "job_id": PERSONA_VISUAL_CANDIDATE_PROVENANCE_ID_TEXT_LIMIT,
    "backend": PERSONA_VISUAL_CANDIDATE_PROVENANCE_ID_TEXT_LIMIT,
    "target_state": PERSONA_VISUAL_CANDIDATE_PROVENANCE_ID_TEXT_LIMIT,
}
_RECIPE_TEXT_FIELDS = {
    "starter_pack_id": PERSONA_VISUAL_CANDIDATE_PROVENANCE_ID_TEXT_LIMIT,
    "recipe_output": PERSONA_VISUAL_CANDIDATE_PROVENANCE_ID_TEXT_LIMIT,
    "correlation_id": PERSONA_VISUAL_CANDIDATE_PROVENANCE_ID_TEXT_LIMIT,
    "identity_brief": PERSONA_VISUAL_CANDIDATE_PROVENANCE_SUMMARY_TEXT_LIMIT,
    "neutral_anchor": PERSONA_VISUAL_CANDIDATE_PROVENANCE_SUMMARY_TEXT_LIMIT,
    "static_sheet": PERSONA_VISUAL_CANDIDATE_PROVENANCE_SUMMARY_TEXT_LIMIT,
}


def build_persona_visual_candidate_provenance(
    *,
    request_id: str | None,
    job_id: str | int | None,
    backend: str | None,
    target_state: str | None,
    recipe_intent: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build normalized provenance for a generated Persona Visual candidate."""
    recipe_payload = dict(recipe_intent) if isinstance(recipe_intent, Mapping) else None
    raw_provenance: dict[str, Any] = {
        "generation_mode": "recipe_backed" if recipe_payload else "prompt_only",
        "request_id": request_id,
        "job_id": job_id,
        "backend": backend,
        "target_state": target_state,
    }
    if recipe_payload:
        user_prompt = str(recipe_payload.get("user_prompt") or "").strip()
        raw_provenance["recipe"] = {
            "starter_pack_id": recipe_payload.get("starter_pack_id"),
            "recipe_output": recipe_payload.get("recipe_output"),
            "correlation_id": recipe_payload.get("correlation_id") or request_id,
            "identity_brief": recipe_payload.get("identity_brief"),
            "neutral_anchor": recipe_payload.get("neutral_anchor"),
            "static_sheet": recipe_payload.get("static_sheet"),
            "review_checks": recipe_payload.get("review_checks"),
            "user_prompt_included": bool(user_prompt),
        }
    return normalize_persona_visual_candidate_provenance(raw_provenance)


def normalize_persona_visual_candidate_provenance(value: Any) -> dict[str, Any]:
    """Return bounded candidate provenance with only review-safe keys."""
    if not isinstance(value, Mapping):
        return {}

    normalized: dict[str, Any] = {}
    mode = _normalize_generation_mode(value.get("generation_mode"), value.get("recipe"))
    if mode:
        normalized["generation_mode"] = mode

    for field_name, text_limit in _TOP_LEVEL_TEXT_FIELDS.items():
        text_value = _safe_provenance_text(value.get(field_name), max_length=text_limit)
        if text_value is not None:
            normalized[field_name] = text_value

    recipe = _normalize_recipe_provenance(value.get("recipe"))
    if recipe:
        normalized["recipe"] = recipe
        normalized.setdefault("generation_mode", "recipe_backed")

    if not normalized:
        return {}
    return {
        "schema_version": PERSONA_VISUAL_CANDIDATE_PROVENANCE_SCHEMA_VERSION,
        **normalized,
    }


def _normalize_generation_mode(value: Any, recipe: Any) -> str | None:
    mode = str(value or "").strip()
    if mode in _ALLOWED_GENERATION_MODES:
        return mode
    if isinstance(recipe, Mapping) and recipe:
        return "recipe_backed"
    return None


def _normalize_recipe_provenance(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    recipe: dict[str, Any] = {}
    for field_name, text_limit in _RECIPE_TEXT_FIELDS.items():
        text_value = _safe_provenance_text(value.get(field_name), max_length=text_limit)
        if text_value is not None:
            recipe[field_name] = text_value
    review_checks = _normalize_review_checks(value.get("review_checks"))
    if review_checks:
        recipe["review_checks"] = review_checks
    if "user_prompt_included" in value:
        recipe["user_prompt_included"] = bool(value.get("user_prompt_included"))
    return recipe


def _normalize_review_checks(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    checks: list[str] = []
    for item in value[:PERSONA_VISUAL_CANDIDATE_PROVENANCE_REVIEW_CHECK_LIMIT]:
        text_value = _safe_provenance_text(
            item,
            max_length=PERSONA_VISUAL_CANDIDATE_PROVENANCE_REVIEW_CHECK_TEXT_LIMIT,
        )
        if text_value is not None:
            checks.append(text_value)
    return checks


def _safe_provenance_text(value: Any, *, max_length: int) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lower_text = text.lower()
    if any(marker in lower_text for marker in _SECRET_MARKERS):
        return "[redacted]"
    if any(marker in lower_text for marker in _PATH_MARKERS):
        return "[redacted]"
    collapsed = " ".join(text.split())
    if len(collapsed) > max_length:
        return collapsed[:max_length]
    return collapsed
