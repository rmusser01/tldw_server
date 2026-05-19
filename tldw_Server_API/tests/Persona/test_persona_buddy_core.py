"""Core deterministic derivation tests for persona buddy helpers."""

import pytest

from tldw_Server_API.app.core.Persona.buddy import (
    PERSONA_BUDDY_DERIVATION_VERSION,
    build_persona_buddy_source_fingerprint,
    derive_persona_buddy_core,
    normalize_persona_buddy_overlay_preferences,
    resolve_persona_buddy_profile,
)

pytestmark = pytest.mark.unit


def test_build_persona_buddy_core_ignores_system_prompt_voice_defaults_and_setup():
    base_profile = {
        "id": "persona-alpha",
        "name": "Research Owl",
        "origin_character_id": 42,
        "origin_character_name": "Archivist",
        "origin_character_snapshot_at": "2026-03-07T10:00:00Z",
    }
    mutable_only_changes = {
        **base_profile,
        "system_prompt": "Totally rewritten prompt",
        "voice_defaults": {"tts_voice": "af_heart"},
        "setup": {"status": "completed"},
    }

    assert derive_persona_buddy_core(base_profile) == derive_persona_buddy_core(mutable_only_changes)


def test_resolve_persona_buddy_profile_falls_back_when_overlay_is_incompatible():
    derived_core = {
        "species_id": "owl",
        "silhouette_id": "owl_round",
        "palette_id": "moss",
        "behavior_family": "steady",
        "expression_profile": "warm",
    }

    resolved = resolve_persona_buddy_profile(
        derived_core=derived_core,
        overlay_preferences={"accessory_id": "tinyduck", "eye_style": "spiral"},
    )

    assert resolved["compatibility_status"] == "fallback_applied"
    assert resolved["accessory_id"] is None
    assert resolved["eye_style"] == "dot"


def test_resolve_persona_buddy_profile_defaults_without_overlay_are_exact():
    derived_core = {
        "species_id": "owl",
        "silhouette_id": "owl_round",
        "palette_id": "moss",
        "behavior_family": "steady",
        "expression_profile": "warm",
    }

    resolved = resolve_persona_buddy_profile(
        derived_core=derived_core,
        overlay_preferences=None,
    )

    assert resolved["compatibility_status"] == "exact"
    assert resolved["accessory_id"] is None
    assert resolved["eye_style"] == "dot"


def test_resolve_persona_buddy_profile_preserves_compatible_overlay_as_exact():
    derived_core = {
        "species_id": "owl",
        "silhouette_id": "owl_round",
        "palette_id": "moss",
        "behavior_family": "steady",
        "expression_profile": "warm",
    }

    resolved = resolve_persona_buddy_profile(
        derived_core=derived_core,
        overlay_preferences={"accessory_id": "scarf", "eye_style": "sleepy"},
    )

    assert resolved["compatibility_status"] == "exact"
    assert resolved["accessory_id"] == "scarf"
    assert resolved["eye_style"] == "sleepy"


def test_build_persona_buddy_source_fingerprint_changes_for_name_or_origin_changes():
    base = {
        "id": "persona-alpha",
        "name": "Research Owl",
        "origin_character_id": 42,
        "origin_character_name": "Archivist",
        "origin_character_snapshot_at": "2026-03-07T10:00:00Z",
    }
    renamed = {**base, "name": "Renamed Persona"}

    assert build_persona_buddy_source_fingerprint(base) != build_persona_buddy_source_fingerprint(
        renamed
    )


def test_build_persona_buddy_source_fingerprint_normalizes_semantically_equivalent_stable_inputs():
    base = {
        "id": "persona-alpha",
        "name": "Research Owl",
        "origin_character_id": 42,
        "origin_character_name": "Archivist",
        "origin_character_snapshot_at": "2026-03-07T10:00:00Z",
    }
    equivalent = {
        **base,
        "origin_character_id": "42",
    }

    assert build_persona_buddy_source_fingerprint(base) == build_persona_buddy_source_fingerprint(
        equivalent
    )


def test_derive_persona_buddy_core_golden_output_for_derivation_version():
    profile = {
        "id": "persona-alpha",
        "name": "Research Owl",
        "origin_character_id": 42,
        "origin_character_name": "Archivist",
        "origin_character_snapshot_at": "2026-03-07T10:00:00Z",
    }

    assert PERSONA_BUDDY_DERIVATION_VERSION == 1
    assert derive_persona_buddy_core(profile) == {
        "species_id": "duck",
        "silhouette_id": "duck_round",
        "palette_id": "ink",
        "behavior_family": "curious",
        "expression_profile": "focused",
    }


def test_normalize_persona_buddy_overlay_preferences_casts_to_contract_shape():
    normalized = normalize_persona_buddy_overlay_preferences(
        {"accessory_id": 7, "eye_style": False}
    )

    assert normalized == {
        "accessory_id": "7",
        "eye_style": "False",
    }
