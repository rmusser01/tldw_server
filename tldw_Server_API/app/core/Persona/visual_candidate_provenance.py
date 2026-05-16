"""Trace-safe provenance metadata for Persona Visual generated candidates.

Generation jobs may carry prompt, provider, recipe, and request metadata that is
useful during human review but unsafe to echo directly. This module builds and
normalizes the small candidate provenance payload stored with review candidates:
stable IDs, bounded labels, and recipe summary fields only. Raw prompts and
unknown keys are intentionally excluded.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


PERSONA_VISUAL_CANDIDATE_PROVENANCE_SCHEMA_VERSION = 1
PERSONA_VISUAL_CANDIDATE_PROVENANCE_ID_TEXT_LIMIT = 128
PERSONA_VISUAL_CANDIDATE_PROVENANCE_SUMMARY_TEXT_LIMIT = 240
PERSONA_VISUAL_CANDIDATE_PROVENANCE_REVIEW_CHECK_LIMIT = 12
PERSONA_VISUAL_CANDIDATE_PROVENANCE_REVIEW_CHECK_TEXT_LIMIT = 120

_ALLOWED_GENERATION_MODES = frozenset({"prompt_only", "recipe_backed"})
_SECRET_VALUE_PATTERNS = (
    re.compile(r"\bBearer\b(?:\s*[:=]\s*\S+|\s+[A-Za-z0-9+/_=-]{12,})"),
    re.compile(r"\bauthorization\b\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"\bx\s*[-_ ]?\s*api\s*[-_ ]?\s*key\b\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"\bapi\s*[-_ ]?\s*key\b\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"\b(?:access|refresh|id|auth)\s*[-_ ]?\s*token\b\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"\bclient\s*[-_ ]?\s*secret\b\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"\bsecret\s*[-_ ]?\s*(?:key|token)\b\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"\bpassword\b\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9_-]{7,}\b"),
)
_PATH_PATTERNS = (
    re.compile(r"(?:^|\s)/(?:home|private|users)/\S+", re.IGNORECASE),
    re.compile(r"\b[A-Za-z]:\\\S+"),
    re.compile(r"\\\\[^\s\\]+\\\S+"),
)
_TOKENISH_VALUE_PATTERN = re.compile(r"^[A-Za-z0-9+/_=-]{40,}$")
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
    """Return a supported generation mode, inferring recipe-backed when needed."""
    mode = str(value or "").strip()
    if mode in _ALLOWED_GENERATION_MODES:
        return mode
    if isinstance(recipe, Mapping) and recipe:
        return "recipe_backed"
    return None


def _normalize_recipe_provenance(value: Any) -> dict[str, Any]:
    """Normalize the recipe subsection to the bounded review-safe fields."""
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
    """Return bounded review-check strings after safety normalization."""
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
    """Return normalized text or a redaction marker for unsafe provenance values."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    collapsed = " ".join(text.split())
    if _contains_secret_value(text, collapsed):
        return "[redacted]"
    if any(pattern.search(collapsed) for pattern in _PATH_PATTERNS):
        return "[redacted]"
    if len(collapsed) > max_length:
        return collapsed[:max_length]
    return collapsed


def _contains_secret_value(text: str, collapsed: str) -> bool:
    """Detect explicit auth-key shapes and high-entropy standalone tokens."""
    if any(pattern.search(text) or pattern.search(collapsed) for pattern in _SECRET_VALUE_PATTERNS):
        return True
    return _looks_like_single_token_secret(collapsed)


def _looks_like_single_token_secret(value: str) -> bool:
    """Return true for long standalone values that resemble opaque credentials."""
    if not _TOKENISH_VALUE_PATTERN.fullmatch(value):
        return False
    character_classes = (
        any(char.islower() for char in value),
        any(char.isupper() for char in value),
        any(char.isdigit() for char in value),
        any(char in "+/_=-" for char in value),
    )
    return sum(character_classes) >= 3 and len(set(value)) >= 16
