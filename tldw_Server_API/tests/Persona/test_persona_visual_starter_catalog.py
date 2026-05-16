from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.persona import (
    PersonaVisualStarterPackResponse,
    PersonaVisualStarterProductionRecipeResponse,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Persona.visual_manifest_assets import (
    collect_visual_manifest_asset_ids,
)
from tldw_Server_API.app.core.Persona.visual_starter_catalog import (
    PersonaVisualStarterCatalogError,
    PersonaVisualStarterCatalogService,
)
from tldw_Server_API.app.core.Persona.visual_starter_fixtures import (
    DEFAULT_PERSONA_VISUAL_STARTER_PACK_ID,
    DEFAULT_PERSONA_VISUAL_STARTER_PACK_IDS,
    DEFAULT_PERSONA_VISUAL_STARTER_PACKS,
    LEGACY_PERSONA_VISUAL_STARTER_PACK_ID,
    PersonaVisualStarterAsset,
    PersonaVisualStarterPack,
    PersonaVisualStarterProductionRecipe,
)
from tldw_Server_API.app.core.Persona.visual_starter_recipe_taxonomy import (
    BUDDY_VISUAL_ANIMATION_OUTPUT_IDS,
    BUDDY_VISUAL_EXPECTED_ASSET_GROUP_IDS,
    BUDDY_VISUAL_STATIC_SOURCE_ASSET_GROUP_IDS,
)


pytestmark = pytest.mark.unit

_BASIC_STARTER_PACK_IDS = {
    "research-buddy-basic",
    "migu-marker-basic",
    "minimal-helper-basic",
}
_REQUIRED_BUDDY_STATES = {"idle", "listening", "thinking", "speaking", "error"}
_BASIC_PRODUCTION_GUIDANCE_EXPECTATIONS = {
    "research-buddy-basic": {
        "identity": ("monitor", "antenna"),
        "neutral": ("rounded screen", "compact body"),
        "state_delta": ("mouth", "accent marks"),
    },
    "migu-marker-basic": {
        "identity": ("marker-line", "cyan twin tails"),
        "neutral": ("cream oval face", "gray body"),
        "state_delta": ("hair bob", "teal"),
    },
    "minimal-helper-basic": {
        "identity": ("geometric", "green diamond"),
        "neutral": ("centered diamond", "stub limbs"),
        "state_delta": ("signal icons", "red error"),
    },
}


def _png_bytes(width: int = 2, height: int = 2) -> bytes:
    buffer = BytesIO()
    Image.new("RGBA", (width, height), (48, 96, 160, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


def _manifest_frame_asset_ids(manifest: dict[str, Any]) -> set[str]:
    asset_ids: set[str] = set()
    for animation in manifest.get("animations", {}).values():
        if not isinstance(animation, dict):
            continue
        for frame in animation.get("frames", []):
            if isinstance(frame, dict) and frame.get("asset_id"):
                asset_ids.add(str(frame["asset_id"]))
    return asset_ids


def _state_frame_asset_id(manifest: dict[str, Any], state: str) -> str:
    animation_id = manifest["states"][state]["animation_id"]
    frame = manifest["animations"][animation_id]["frames"][0]
    return str(frame["asset_id"])


def _pack_fixture(starter_pack_id: str) -> PersonaVisualStarterPack:
    for starter in DEFAULT_PERSONA_VISUAL_STARTER_PACKS:
        if starter.id == starter_pack_id:
            return starter
    raise AssertionError(f"Unknown starter pack fixture: {starter_pack_id}")


def _assert_recipe_shape(recipe: dict[str, Any], *, expected_output: str) -> None:
    assert set(recipe) == {
        "identity_brief",
        "neutral_anchor",
        "static_sheet",
        "animation_outputs",
        "review_checks",
    }
    assert isinstance(recipe["identity_brief"], str)
    assert recipe["identity_brief"].strip()
    assert isinstance(recipe["neutral_anchor"], str)
    assert "neutral" in recipe["neutral_anchor"].lower()
    assert isinstance(recipe["static_sheet"], str)
    assert recipe["static_sheet"].strip()
    assert expected_output in recipe["animation_outputs"]
    assert "neutral_identity_consistency" in recipe["review_checks"]


def _valid_recipe_payload() -> dict[str, object]:
    return {
        "identity_brief": "Keep the starter identity consistent.",
        "neutral_anchor": "Start from a neutral front-facing anchor.",
        "static_sheet": "Author the required static sheet.",
        "animation_outputs": ["required_state_loops"],
        "review_checks": ["neutral_identity_consistency"],
    }


@pytest.fixture()
def db_instance(tmp_path: Path) -> Iterator[CharactersRAGDB]:
    db = CharactersRAGDB(
        tmp_path / "persona_visual_starter_catalog.sqlite",
        "persona-visual-starter-test",
    )
    yield db
    db.close_connection()


@pytest.fixture(autouse=True)
def visual_storage_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "visuals"

    def _fake_visuals_dir(user_id: str) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        return root

    monkeypatch.setattr(
        DatabasePaths,
        "get_user_persona_visuals_dir",
        staticmethod(_fake_visuals_dir),
    )
    return root


def test_starter_catalog_lists_bundled_packs_with_basic_tier_art_ready(
    db_instance: CharactersRAGDB,
) -> None:
    service = PersonaVisualStarterCatalogService(db_instance)

    starters = service.list_starter_packs()

    assert [starter["id"] for starter in starters] == list(DEFAULT_PERSONA_VISUAL_STARTER_PACK_IDS)
    assert LEGACY_PERSONA_VISUAL_STARTER_PACK_ID not in {starter["id"] for starter in starters}
    starter = starters[0]
    assert starter["title"] == "Research Buddy Basic"
    assert starter["renderer_type"] == "sprite_frames"
    assert starter["manifest_version"] == 1
    assert starter["asset_count"] >= 1
    assert starter["total_bytes"] > 0
    assert _REQUIRED_BUDDY_STATES.issubset(set(starter["states_offered"]))
    basic_starters = [starter for starter in starters if starter["id"] in _BASIC_STARTER_PACK_IDS]
    scaffold_starters = [starter for starter in starters if starter["id"] not in _BASIC_STARTER_PACK_IDS]
    assert {starter["production_status"] for starter in basic_starters} == {"art_ready"}
    assert all("catalog:art-ready" in starter["tags"] for starter in basic_starters)
    assert all("scaffold" not in starter["description"].lower() for starter in basic_starters)
    assert {starter["production_status"] for starter in scaffold_starters} == {"scaffold"}
    assert all("catalog:scaffold" in starter["tags"] for starter in scaffold_starters)
    assert all("scaffold" in starter["description"].lower() for starter in scaffold_starters)
    assert any("tier:intricate" in starter["tags"] for starter in starters)
    assert any("tool.notes_search" in starter["states_offered"] for starter in starters)
    assert {starter["complexity_tier"] for starter in starters} == {
        "basic",
        "intermediate",
        "intricate",
    }
    assert all(starter["neutral_anchor_required"] for starter in starters)
    assert all("neutral_anchor" in starter["expected_asset_groups"] for starter in starters)
    assert all(starter["animation_coverage_notes"] for starter in starters)
    assert all(starter["production_recipe"] for starter in starters)


def test_get_starter_pack_returns_isolated_manifest_preview(
    db_instance: CharactersRAGDB,
) -> None:
    service = PersonaVisualStarterCatalogService(db_instance)
    first = service.get_starter_pack(DEFAULT_PERSONA_VISUAL_STARTER_PACK_ID)
    first["manifest"]["states"]["idle"]["animation_id"] = "mutated"
    first["expected_asset_groups"].append("mutated")
    first["production_recipe"]["animation_outputs"].append("mutated")

    second = service.get_starter_pack(DEFAULT_PERSONA_VISUAL_STARTER_PACK_ID)

    assert second["manifest"]["states"]["idle"]["animation_id"] == "idle-loop"
    assert "mutated" not in second["expected_asset_groups"]
    assert "mutated" not in second["production_recipe"]["animation_outputs"]


@pytest.mark.parametrize(
    (
        "starter_pack_id",
        "complexity_tier",
        "production_status",
        "required_group",
        "expected_output",
    ),
    (
        (
            "research-buddy-basic",
            "basic",
            "art_ready",
            "required_state_loops",
            "required_state_loops",
        ),
        (
            "study-desk-intermediate",
            "intermediate",
            "scaffold",
            "static_talking_sheet",
            "required_state_loops",
        ),
        (
            "lofi-study-intricate",
            "intricate",
            "scaffold",
            "animation_atlas",
            "animation_atlas",
        ),
    ),
)
def test_starter_pack_reports_production_readiness_metadata(
    db_instance: CharactersRAGDB,
    starter_pack_id: str,
    complexity_tier: str,
    production_status: str,
    required_group: str,
    expected_output: str,
) -> None:
    service = PersonaVisualStarterCatalogService(db_instance)

    detail = service.get_starter_pack(starter_pack_id)

    assert detail["production_status"] == production_status
    assert detail["complexity_tier"] == complexity_tier
    assert detail["neutral_anchor_required"] is True
    assert required_group in detail["expected_asset_groups"]
    if production_status == "art_ready":
        assert all("reviewed" in note.lower() for note in detail["animation_coverage_notes"])
    else:
        assert all("scaffold" in note.lower() for note in detail["animation_coverage_notes"])
    _assert_recipe_shape(detail["production_recipe"], expected_output=expected_output)


@pytest.mark.parametrize("starter_pack_id", sorted(_BASIC_STARTER_PACK_IDS))
def test_basic_starter_packs_use_reviewed_multi_frame_state_assets(
    db_instance: CharactersRAGDB,
    starter_pack_id: str,
) -> None:
    service = PersonaVisualStarterCatalogService(db_instance)

    detail = service.get_starter_pack(starter_pack_id)
    fixture = _pack_fixture(starter_pack_id)
    asset_by_key = {asset.asset_key: asset for asset in fixture.assets}

    assert detail["production_status"] == "art_ready"
    assert detail["asset_count"] == 12
    assert {asset["asset_role"] for asset in detail["assets"]} == {
        "frame",
        "preview",
        "still_pose",
    }
    assert {"neutral-anchor", "preview"}.issubset(asset_by_key)

    for state in _REQUIRED_BUDDY_STATES:
        animation_id = detail["manifest"]["states"][state]["animation_id"]
        frames = detail["manifest"]["animations"][animation_id]["frames"]
        assert len(frames) == 2
        assert all(str(frame["asset_id"]).startswith(f"{state}-") for frame in frames)

    frame_asset_ids = _manifest_frame_asset_ids(detail["manifest"])
    assert len(frame_asset_ids) == 10
    assert "neutral-anchor" not in frame_asset_ids
    assert "preview" not in frame_asset_ids

    for asset in fixture.assets:
        image = Image.open(BytesIO(asset.content))
        assert image.mode == "RGBA"
        assert image.size == (96, 96)
        assert image.getpixel((0, 0))[3] == 0
        assert image.getbbox() is not None


@pytest.mark.parametrize("starter_pack_id", sorted(_BASIC_STARTER_PACK_IDS))
def test_basic_starter_packs_expose_design_specific_recreation_guidance(
    db_instance: CharactersRAGDB,
    starter_pack_id: str,
) -> None:
    service = PersonaVisualStarterCatalogService(db_instance)

    recipe = service.get_starter_pack(starter_pack_id)["production_recipe"]

    expectations = _BASIC_PRODUCTION_GUIDANCE_EXPECTATIONS[starter_pack_id]
    identity_brief = recipe["identity_brief"].lower()
    neutral_anchor = recipe["neutral_anchor"].lower()
    state_delta_guidance = recipe["static_sheet"].lower()
    for snippet in expectations["identity"]:
        assert snippet in identity_brief
    for snippet in expectations["neutral"]:
        assert snippet in neutral_anchor
    for snippet in expectations["state_delta"]:
        assert snippet in state_delta_guidance


def test_default_starter_production_recipes_use_pipeline_taxonomy(
    db_instance: CharactersRAGDB,
) -> None:
    service = PersonaVisualStarterCatalogService(db_instance)

    for detail in (service.get_starter_pack(starter_id) for starter_id in DEFAULT_PERSONA_VISUAL_STARTER_PACK_IDS):
        expected_groups = set(detail["expected_asset_groups"])
        animation_outputs = set(detail["production_recipe"]["animation_outputs"])

        assert expected_groups <= BUDDY_VISUAL_EXPECTED_ASSET_GROUP_IDS
        assert animation_outputs <= BUDDY_VISUAL_ANIMATION_OUTPUT_IDS
        assert animation_outputs <= expected_groups
        assert not (animation_outputs & BUDDY_VISUAL_STATIC_SOURCE_ASSET_GROUP_IDS)


def test_list_starter_packs_rejects_static_source_animation_outputs(
    db_instance: CharactersRAGDB,
) -> None:
    malformed_recipe = replace(
        DEFAULT_PERSONA_VISUAL_STARTER_PACKS[0].production_recipe,
        animation_outputs=("static_talking_sheet",),
    )
    malformed = replace(
        DEFAULT_PERSONA_VISUAL_STARTER_PACKS[0],
        production_recipe=malformed_recipe,
    )
    service = PersonaVisualStarterCatalogService(db_instance, starter_packs=(malformed,))

    with pytest.raises(PersonaVisualStarterCatalogError) as exc_info:
        service.list_starter_packs()

    assert exc_info.value.code == "invalid_starter_fixture"
    assert exc_info.value.details["field_name"] == "production_recipe.animation_outputs"


def test_list_starter_packs_rejects_recipe_outputs_missing_expected_groups(
    db_instance: CharactersRAGDB,
) -> None:
    malformed_recipe = replace(
        DEFAULT_PERSONA_VISUAL_STARTER_PACKS[0].production_recipe,
        animation_outputs=("custom_state_variants",),
    )
    malformed = replace(
        DEFAULT_PERSONA_VISUAL_STARTER_PACKS[0],
        production_recipe=malformed_recipe,
    )
    service = PersonaVisualStarterCatalogService(db_instance, starter_packs=(malformed,))

    with pytest.raises(PersonaVisualStarterCatalogError) as exc_info:
        service.list_starter_packs()

    assert exc_info.value.code == "invalid_starter_fixture"
    assert exc_info.value.details["field_name"] == "production_recipe.animation_outputs"
    assert exc_info.value.details["invalid_outputs"] == ["custom_state_variants"]


@pytest.mark.parametrize(
    "starter_pack_id",
    (
        "study-desk-intermediate",
        "tool-helper-intermediate",
        "object-creature-intermediate",
        "lofi-study-intricate",
        "action-guide-intricate",
        "elaborate-persona-intricate",
    ),
)
def test_static_talking_and_reaction_sheets_are_source_material_not_animation_output(
    db_instance: CharactersRAGDB,
    starter_pack_id: str,
) -> None:
    service = PersonaVisualStarterCatalogService(db_instance)

    detail = service.get_starter_pack(starter_pack_id)

    assert "static_talking_sheet" in detail["expected_asset_groups"]
    assert "static_reaction_sheet" in detail["expected_asset_groups"]
    assert "static" in detail["production_recipe"]["static_sheet"].lower()
    assert "static_talking_sheet" not in detail["production_recipe"]["animation_outputs"]
    assert "static_reaction_sheet" not in detail["production_recipe"]["animation_outputs"]


@pytest.mark.parametrize(
    ("field_name", "value"),
    (
        ("complexity_tier", "basic "),
        ("production_status", " scaffold"),
        ("expected_asset_groups", None),
        ("expected_asset_groups", "neutral_anchor"),
        ("expected_asset_groups", ("neutral_anchor", "")),
        ("animation_coverage_notes", None),
        ("animation_coverage_notes", "Scaffold fixture only."),
        ("animation_coverage_notes", ("Scaffold fixture only.", "")),
        ("neutral_anchor_required", "false"),
        ("neutral_anchor_required", 1),
        ("production_recipe", {"identity_brief": "not immutable"}),
        (
            "production_recipe",
            PersonaVisualStarterProductionRecipe(
                identity_brief="Identity",
                neutral_anchor="Neutral anchor",
                static_sheet="Static sheet",
                animation_outputs=(),
            ),
        ),
        (
            "production_recipe",
            PersonaVisualStarterProductionRecipe(
                identity_brief="Identity",
                neutral_anchor="Neutral anchor",
                static_sheet="Static sheet",
                animation_outputs=("required_state_loops",),
                review_checks=("transparent_background",),
            ),
        ),
    ),
)
def test_list_starter_packs_rejects_malformed_production_metadata(
    db_instance: CharactersRAGDB,
    field_name: str,
    value: Any,
) -> None:
    malformed = replace(
        DEFAULT_PERSONA_VISUAL_STARTER_PACKS[0],
        **{field_name: value},
    )
    service = PersonaVisualStarterCatalogService(db_instance, starter_packs=(malformed,))

    with pytest.raises(PersonaVisualStarterCatalogError) as exc_info:
        service.list_starter_packs()

    assert exc_info.value.code == "invalid_starter_fixture"
    assert exc_info.value.details["starter_pack_id"] == malformed.id


@pytest.mark.parametrize(
    "patch",
    (
        {"identity_brief": ""},
        {"neutral_anchor": "x" * 321},
        {"animation_outputs": []},
        {"animation_outputs": ["required_state_loops"] * 13},
        {"animation_outputs": ["x" * 321]},
        {"review_checks": ["transparent_background"]},
    ),
)
def test_production_recipe_response_enforces_catalog_bounds(patch: dict[str, object]) -> None:
    payload = _valid_recipe_payload()
    payload.update(patch)

    with pytest.raises(ValidationError):
        PersonaVisualStarterProductionRecipeResponse.model_validate(payload)


@pytest.mark.parametrize(
    "animation_outputs",
    (
        ["static_talking_sheet"],
        ["static_reaction_sheet"],
        ["identity_brief"],
        ["neutral_anchor"],
        ["model_sheet"],
        ["unknown_output"],
    ),
)
def test_production_recipe_response_rejects_non_animation_outputs(
    animation_outputs: list[str],
) -> None:
    payload = _valid_recipe_payload()
    payload["animation_outputs"] = animation_outputs

    with pytest.raises(ValidationError) as exc_info:
        PersonaVisualStarterProductionRecipeResponse.model_validate(payload)

    error_message = str(exc_info.value)
    for animation_output in animation_outputs:
        assert animation_output in error_message


def test_starter_pack_response_rejects_unknown_expected_asset_groups() -> None:
    with pytest.raises(ValidationError) as exc_info:
        PersonaVisualStarterPackResponse.model_validate(
            {
                "id": "starter",
                "title": "Starter",
                "description": "Starter fixture",
                "renderer_type": "sprite_frames",
                "expected_asset_groups": ["neutral_anchor", "unknown_group"],
                "production_recipe": _valid_recipe_payload(),
            }
        )

    assert "unknown_group" in str(exc_info.value)


def test_starter_pack_response_rejects_recipe_outputs_missing_expected_groups() -> None:
    payload = {
        "id": "starter",
        "title": "Starter",
        "description": "Starter fixture",
        "renderer_type": "sprite_frames",
        "expected_asset_groups": ["neutral_anchor", "required_state_loops"],
        "production_recipe": {
            **_valid_recipe_payload(),
            "animation_outputs": ["required_state_loops", "custom_state_variants"],
        },
    }

    with pytest.raises(ValidationError) as exc_info:
        PersonaVisualStarterPackResponse.model_validate(payload)

    assert "custom_state_variants" in str(exc_info.value)


def test_get_starter_pack_accepts_legacy_research_buddy_alias(
    db_instance: CharactersRAGDB,
) -> None:
    service = PersonaVisualStarterCatalogService(db_instance)

    detail = service.get_starter_pack(LEGACY_PERSONA_VISUAL_STARTER_PACK_ID)

    assert detail["id"] == DEFAULT_PERSONA_VISUAL_STARTER_PACK_ID
    assert detail["title"] == "Research Buddy Basic"


@pytest.mark.parametrize(
    ("starter_pack_id", "custom_states"),
    (
        (
            "action-guide-intricate",
            ("reaction.anticipation", "reaction.success"),
        ),
        (
            "elaborate-persona-intricate",
            ("mood.focused", "tool.media_import"),
        ),
    ),
)
def test_multi_custom_state_scaffolds_use_distinct_variant_assets(
    db_instance: CharactersRAGDB,
    starter_pack_id: str,
    custom_states: tuple[str, ...],
) -> None:
    service = PersonaVisualStarterCatalogService(db_instance)

    detail = service.get_starter_pack(starter_pack_id)
    state_asset_ids = {_state_frame_asset_id(detail["manifest"], state) for state in custom_states}

    assert len(state_asset_ids) == len(custom_states)


def test_copy_starter_pack_to_persona_creates_inactive_user_owned_draft(
    db_instance: CharactersRAGDB,
    visual_storage_root: Path,
) -> None:
    user_id = "user-1"
    persona_id = db_instance.create_persona_profile({"user_id": user_id, "name": "Target"})
    active_pack = db_instance.create_persona_visual_pack(
        persona_id=persona_id,
        user_id=user_id,
        title="Existing active visual",
        status="draft",
    )
    db_instance.activate_persona_visual_pack(
        persona_id=persona_id,
        user_id=user_id,
        pack_id=active_pack["id"],
    )
    service = PersonaVisualStarterCatalogService(db_instance)

    copied = service.copy_starter_pack_to_persona(
        starter_pack_id=DEFAULT_PERSONA_VISUAL_STARTER_PACK_ID,
        persona_id=persona_id,
        user_id=user_id,
    )

    assert copied["status"] == "draft"
    assert copied["persona_id"] == persona_id
    assert copied["user_id"] == user_id
    assert copied["title"] == "Research Buddy Basic"
    assert copied["provenance"] == "imported"
    assert copied["active_at"] is None
    assert (
        db_instance.get_active_persona_visual_pack(
            persona_id=persona_id,
            user_id=user_id,
        )["id"]
        == active_pack["id"]
    )

    copied_assets = db_instance.list_persona_visual_assets(
        pack_id=copied["id"],
        persona_id=persona_id,
        user_id=user_id,
    )
    assert copied["assets"] == copied_assets
    assert len(copied_assets) == 12
    assert {asset["provenance"] for asset in copied_assets} == {"imported"}
    assert (visual_storage_root / persona_id / copied["id"]).is_dir()

    copied_asset_ids = {str(asset["id"]) for asset in copied_assets}
    referenced_asset_ids = collect_visual_manifest_asset_ids(copied["manifest"])
    assert referenced_asset_ids < copied_asset_ids
    assert len(referenced_asset_ids) == 10
    assert "neutral" not in str(copied["manifest"])


@pytest.mark.parametrize("starter_pack_id", DEFAULT_PERSONA_VISUAL_STARTER_PACK_IDS)
def test_copy_every_default_scaffold_creates_inactive_user_owned_draft(
    db_instance: CharactersRAGDB,
    starter_pack_id: str,
) -> None:
    user_id = "user-1"
    persona_id = db_instance.create_persona_profile({"user_id": user_id, "name": f"Target {starter_pack_id}"})
    service = PersonaVisualStarterCatalogService(db_instance)
    starter_detail = service.get_starter_pack(starter_pack_id)
    fixture_asset_keys = {str(asset["asset_key"]) for asset in starter_detail["assets"]}

    copied = service.copy_starter_pack_to_persona(
        starter_pack_id=starter_pack_id,
        persona_id=persona_id,
        user_id=user_id,
    )

    assert copied["status"] == "draft"
    assert copied["active_at"] is None
    assert copied["title"] == starter_detail["title"]
    assert {"idle", "listening", "thinking", "speaking", "error"}.issubset(set(copied["manifest"]["states"]))
    copied_assets = db_instance.list_persona_visual_assets(
        pack_id=copied["id"],
        persona_id=persona_id,
        user_id=user_id,
    )
    copied_asset_ids = {str(asset["id"]) for asset in copied_assets}
    referenced_asset_ids = collect_visual_manifest_asset_ids(copied["manifest"])
    if starter_pack_id in _BASIC_STARTER_PACK_IDS:
        assert referenced_asset_ids < copied_asset_ids
        assert len(referenced_asset_ids) == 10
    else:
        assert referenced_asset_ids == copied_asset_ids
    frame_asset_ids = _manifest_frame_asset_ids(copied["manifest"])
    assert frame_asset_ids == referenced_asset_ids
    assert not (fixture_asset_keys & frame_asset_ids)


def test_copy_legacy_research_buddy_alias_creates_default_draft(
    db_instance: CharactersRAGDB,
) -> None:
    user_id = "user-1"
    persona_id = db_instance.create_persona_profile({"user_id": user_id, "name": "Target"})
    service = PersonaVisualStarterCatalogService(db_instance)

    copied = service.copy_starter_pack_to_persona(
        starter_pack_id=LEGACY_PERSONA_VISUAL_STARTER_PACK_ID,
        persona_id=persona_id,
        user_id=user_id,
    )

    assert copied["status"] == "draft"
    assert copied["title"] == "Research Buddy Basic"


def test_copy_starter_pack_rejects_malformed_fixture_manifest(
    db_instance: CharactersRAGDB,
) -> None:
    user_id = "user-1"
    persona_id = db_instance.create_persona_profile({"user_id": user_id, "name": "Target"})
    malformed = PersonaVisualStarterPack(
        id="malformed",
        title="Malformed starter",
        description="Invalid test fixture",
        renderer_type="sprite_frames",
        manifest={
            "manifest_version": 1,
            "renderer_type": "sprite_frames",
            "states": {"idle": {"animation_id": "idle"}},
            "animations": {
                "idle": {
                    "frames": [{"asset_id": "missing_asset", "duration_ms": 100}],
                    "frame_rate": 1,
                }
            },
        },
        assets=(
            PersonaVisualStarterAsset(
                asset_key="starter_idle",
                filename="idle.png",
                mime_type="image/png",
                content=_png_bytes(),
                asset_role="frame",
            ),
        ),
    )
    service = PersonaVisualStarterCatalogService(db_instance, starter_packs=(malformed,))

    with pytest.raises(PersonaVisualStarterCatalogError) as exc_info:
        service.copy_starter_pack_to_persona(
            starter_pack_id="malformed",
            persona_id=persona_id,
            user_id=user_id,
        )

    assert exc_info.value.code == "invalid_starter_manifest"
    assert db_instance.list_persona_visual_packs(persona_id=persona_id, user_id=user_id) == []


def test_list_starter_packs_rejects_invalid_fixture_renderer_type(
    db_instance: CharactersRAGDB,
) -> None:
    malformed = PersonaVisualStarterPack(
        id="bad-renderer",
        title="Bad renderer",
        description="Invalid test fixture",
        renderer_type="unknown_renderer",
        manifest={
            "manifest_version": 1,
            "renderer_type": "unknown_renderer",
            "states": {"idle": {"animation_id": "idle"}},
            "animations": {
                "idle": {
                    "frames": [{"asset_id": "starter_idle", "duration_ms": 100}],
                    "frame_rate": 1,
                }
            },
        },
        assets=(
            PersonaVisualStarterAsset(
                asset_key="starter_idle",
                filename="idle.png",
                mime_type="image/png",
                content=_png_bytes(),
                asset_role="frame",
            ),
        ),
    )
    service = PersonaVisualStarterCatalogService(db_instance, starter_packs=(malformed,))

    with pytest.raises(PersonaVisualStarterCatalogError) as exc_info:
        service.list_starter_packs()

    assert exc_info.value.code == "invalid_starter_fixture"


def test_list_starter_packs_rejects_duplicate_fixture_asset_keys(
    db_instance: CharactersRAGDB,
) -> None:
    malformed = PersonaVisualStarterPack(
        id="duplicate-asset-key",
        title="Duplicate asset key",
        description="Invalid test fixture",
        renderer_type="sprite_frames",
        manifest={
            "manifest_version": 1,
            "renderer_type": "sprite_frames",
            "states": {"idle": {"animation_id": "idle"}},
            "animations": {
                "idle": {
                    "frames": [{"asset_id": "starter_idle", "duration_ms": 100}],
                    "frame_rate": 1,
                }
            },
        },
        assets=(
            PersonaVisualStarterAsset(
                asset_key="starter_idle",
                filename="idle-one.png",
                mime_type="image/png",
                content=_png_bytes(),
                asset_role="frame",
            ),
            PersonaVisualStarterAsset(
                asset_key="starter_idle",
                filename="idle-two.png",
                mime_type="image/png",
                content=_png_bytes(),
                asset_role="frame",
            ),
        ),
    )
    service = PersonaVisualStarterCatalogService(db_instance, starter_packs=(malformed,))

    with pytest.raises(PersonaVisualStarterCatalogError) as exc_info:
        service.list_starter_packs()

    assert exc_info.value.code == "invalid_starter_asset"
    assert exc_info.value.details["asset_key"] == "starter_idle"


def test_get_starter_pack_rejects_invalid_fixture_asset_role(
    db_instance: CharactersRAGDB,
) -> None:
    malformed = PersonaVisualStarterPack(
        id="bad-asset-role",
        title="Bad asset role",
        description="Invalid test fixture",
        renderer_type="sprite_frames",
        manifest={
            "manifest_version": 1,
            "renderer_type": "sprite_frames",
            "states": {"idle": {"animation_id": "idle"}},
            "animations": {
                "idle": {
                    "frames": [{"asset_id": "starter_idle", "duration_ms": 100}],
                    "frame_rate": 1,
                }
            },
        },
        assets=(
            PersonaVisualStarterAsset(
                asset_key="starter_idle",
                filename="idle.png",
                mime_type="image/png",
                content=_png_bytes(),
                asset_role="unsupported_role",
            ),
        ),
    )
    service = PersonaVisualStarterCatalogService(db_instance, starter_packs=(malformed,))

    with pytest.raises(PersonaVisualStarterCatalogError) as exc_info:
        service.get_starter_pack("bad-asset-role")

    assert exc_info.value.code == "invalid_starter_fixture"


def test_copy_starter_pack_cleans_up_when_manifest_update_returns_none(
    db_instance: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = "user-1"
    persona_id = db_instance.create_persona_profile({"user_id": user_id, "name": "Target"})
    service = PersonaVisualStarterCatalogService(db_instance)
    monkeypatch.setattr(db_instance, "update_persona_visual_pack_manifest", lambda **_: None)

    with pytest.raises(PersonaVisualStarterCatalogError) as exc_info:
        service.copy_starter_pack_to_persona(
            starter_pack_id=DEFAULT_PERSONA_VISUAL_STARTER_PACK_ID,
            persona_id=persona_id,
            user_id=user_id,
        )

    assert exc_info.value.code == "starter_copy_failed"
    assert db_instance.list_persona_visual_packs(persona_id=persona_id, user_id=user_id) == []


def test_copy_starter_pack_cleans_up_when_status_update_returns_none(
    db_instance: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user_id = "user-1"
    persona_id = db_instance.create_persona_profile({"user_id": user_id, "name": "Target"})
    service = PersonaVisualStarterCatalogService(db_instance)
    monkeypatch.setattr(db_instance, "update_persona_visual_pack_status", lambda **_: None)

    with pytest.raises(PersonaVisualStarterCatalogError) as exc_info:
        service.copy_starter_pack_to_persona(
            starter_pack_id=DEFAULT_PERSONA_VISUAL_STARTER_PACK_ID,
            persona_id=persona_id,
            user_id=user_id,
        )

    assert exc_info.value.code == "starter_copy_failed"
    assert db_instance.list_persona_visual_packs(persona_id=persona_id, user_id=user_id) == []
