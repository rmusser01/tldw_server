"""Deterministic persona buddy derivation and overlay resolution core."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

PERSONA_BUDDY_DERIVATION_VERSION = 1

_SPECIES = ("cat", "duck", "owl", "ghost", "robot", "capybara")
_SILHOUETTES = {
    "cat": ("cat_round",),
    "duck": ("duck_round",),
    "owl": ("owl_round",),
    "ghost": ("ghost_round",),
    "robot": ("robot_round",),
    "capybara": ("capybara_round",),
}
_PALETTES = ("moss", "ember", "sky", "ink")
_BEHAVIOR_FAMILIES = ("steady", "curious", "playful", "measured")
_EXPRESSION_PROFILES = ("warm", "focused", "calm", "bright")
# Accessory and eye-style compatibility is opt-in per species. Unlisted species
# fall back to universal defaults via .get() (accessories: {None}, eyes: "dot").
_ACCESSORY_COMPATIBILITY = {
    "owl": {None, "scarf", "halo"},
    "robot": {None, "antenna", "visor"},
}
_EYE_STYLE_COMPATIBILITY = {
    "owl": {"dot", "sleepy"},
    "robot": {"dot", "visor"},
}
_DEFAULT_ACCESSORY_BY_SPECIES = {"owl": None, "robot": "antenna"}
_DEFAULT_EYE_STYLE_BY_SPECIES = {"owl": "dot", "robot": "dot"}


def _normalize_stable_identity_value(value: Any) -> str:
    """Normalize stable identity values for deterministic fingerprinting."""
    if value is None:
        return ""
    return str(value)


def build_persona_buddy_source_fingerprint(profile: dict[str, Any]) -> str:
    """Build a stable source fingerprint from persona-authoritative fields."""
    stable_payload = {
        "id": _normalize_stable_identity_value(profile.get("id")),
        "name": _normalize_stable_identity_value(profile.get("name")),
        "origin_character_id": _normalize_stable_identity_value(profile.get("origin_character_id")),
        "origin_character_name": _normalize_stable_identity_value(
            profile.get("origin_character_name")
        ),
        "origin_character_snapshot_at": _normalize_stable_identity_value(
            profile.get("origin_character_snapshot_at")
        ),
    }
    raw = json.dumps(stable_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _pick_catalog_value(values: tuple[str, ...], digest: str, offset: int) -> str:
    index = int(digest[offset : offset + 8], 16) % len(values)
    return values[index]


def derive_persona_buddy_core(profile: dict[str, Any]) -> dict[str, Any]:
    """Derive a deterministic buddy core from stable persona fields."""
    digest = build_persona_buddy_source_fingerprint(profile)
    species_id = _pick_catalog_value(_SPECIES, digest, 0)
    return {
        "species_id": species_id,
        "silhouette_id": _pick_catalog_value(_SILHOUETTES[species_id], digest, 8),
        "palette_id": _pick_catalog_value(_PALETTES, digest, 16),
        "behavior_family": _pick_catalog_value(_BEHAVIOR_FAMILIES, digest, 24),
        "expression_profile": _pick_catalog_value(_EXPRESSION_PROFILES, digest, 32),
    }


def normalize_persona_buddy_overlay_preferences(
    overlay_preferences: dict[str, Any] | None,
) -> dict[str, Any]:
    """Normalize overlay preferences to known optional keys."""
    overlay = overlay_preferences or {}
    accessory_id = overlay.get("accessory_id")
    eye_style = overlay.get("eye_style")
    return {
        "accessory_id": None if accessory_id is None else str(accessory_id),
        "eye_style": None if eye_style is None else str(eye_style),
    }


def resolve_persona_buddy_profile(
    *, derived_core: dict[str, Any], overlay_preferences: dict[str, Any] | None
) -> dict[str, Any]:
    """Resolve a complete buddy profile with compatibility-aware overlay fallback."""
    normalized_overlay = normalize_persona_buddy_overlay_preferences(overlay_preferences)
    species_id = str(derived_core["species_id"])
    overlay_accessory_id = normalized_overlay.get("accessory_id")
    overlay_eye_style = normalized_overlay.get("eye_style")
    compatibility_status = "exact"

    allowed_accessories = _ACCESSORY_COMPATIBILITY.get(species_id, {None})
    allowed_eye_styles = _EYE_STYLE_COMPATIBILITY.get(species_id, {"dot"})
    default_accessory_id = _DEFAULT_ACCESSORY_BY_SPECIES.get(species_id)
    default_eye_style = _DEFAULT_EYE_STYLE_BY_SPECIES.get(species_id, "dot")

    if overlay_accessory_id is None:
        accessory_id = default_accessory_id
    elif overlay_accessory_id in allowed_accessories:
        accessory_id = overlay_accessory_id
    else:
        accessory_id = default_accessory_id
        compatibility_status = "fallback_applied"

    if overlay_eye_style is None:
        eye_style = default_eye_style
    elif overlay_eye_style in allowed_eye_styles:
        eye_style = overlay_eye_style
    else:
        eye_style = default_eye_style
        compatibility_status = "fallback_applied"

    return {
        "derivation_version": PERSONA_BUDDY_DERIVATION_VERSION,
        "species_id": species_id,
        "silhouette_id": str(derived_core["silhouette_id"]),
        "palette_id": str(derived_core["palette_id"]),
        "behavior_family": str(derived_core["behavior_family"]),
        "expression_profile": str(derived_core["expression_profile"]),
        "accessory_id": accessory_id,
        "eye_style": eye_style,
        "compatibility_status": compatibility_status,
    }


def build_persona_buddy_summary(
    *,
    persona_name: str,
    role_summary: str | None,
    buddy_row: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a compact, render-oriented buddy summary from an existing Track A row."""
    safe_persona_name = str(persona_name or "").strip()
    safe_role_summary = str(role_summary or "").strip() or None

    summary: dict[str, Any] = {
        "has_buddy": False,
        "persona_name": safe_persona_name,
        "role_summary": safe_role_summary,
        "visual": None,
    }
    if not isinstance(buddy_row, dict):
        return summary

    resolved_profile = buddy_row.get("resolved_profile")
    if not isinstance(resolved_profile, dict):
        return summary

    species_id = str(resolved_profile.get("species_id") or "").strip()
    silhouette_id = str(resolved_profile.get("silhouette_id") or "").strip()
    palette_id = str(resolved_profile.get("palette_id") or "").strip()
    if not species_id or not silhouette_id or not palette_id:
        return summary

    summary["has_buddy"] = True
    summary["visual"] = {
        "species_id": species_id,
        "silhouette_id": silhouette_id,
        "palette_id": palette_id,
        "accessory_id": None
        if resolved_profile.get("accessory_id") is None
        else str(resolved_profile.get("accessory_id")),
        "eye_style": None
        if resolved_profile.get("eye_style") is None
        else str(resolved_profile.get("eye_style")),
        "expression_profile": None
        if resolved_profile.get("expression_profile") is None
        else str(resolved_profile.get("expression_profile")),
    }
    return summary


def ensure_persona_buddy_for_profile(
    db: "CharactersRAGDB",
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Ensure buddy row exists and is current for a persona profile."""
    persona_id = str(profile.get("id") or "").strip()
    user_id = str(profile.get("user_id") or "").strip()
    if not persona_id or not user_id:
        raise ValueError("profile must include id and user_id")

    source_fingerprint = build_persona_buddy_source_fingerprint(profile)
    current = db.get_persona_buddy(
        persona_id=persona_id,
        user_id=user_id,
        include_deleted_personas=True,
    )
    if (
        current
        and int(current.get("derivation_version", 0)) == PERSONA_BUDDY_DERIVATION_VERSION
        and str(current.get("source_fingerprint") or "") == source_fingerprint
    ):
        return current

    overlay_preferences = {}
    if current and isinstance(current.get("overlay_preferences"), dict):
        overlay_preferences = current["overlay_preferences"]
    derived_core = derive_persona_buddy_core(profile)
    return db.upsert_persona_buddy(
        persona_id=persona_id,
        user_id=user_id,
        derivation_version=PERSONA_BUDDY_DERIVATION_VERSION,
        source_fingerprint=source_fingerprint,
        derived_core=derived_core,
        overlay_preferences=overlay_preferences,
    )
