import pytest

from tldw_Server_API.app.core.Persona.visual_renderer_capabilities import (
    get_persona_visual_renderer_capability,
    list_persona_visual_renderer_capabilities,
)
from tldw_Server_API.app.core.Persona.visuals import (
    MAX_FRAMES_PER_ANIMATION,
    PersonaVisualManifestError,
    REQUIRED_VISUAL_STATES,
    validate_visual_manifest,
)


pytestmark = pytest.mark.unit


def _activatable_manifest() -> dict:
    return {
        "manifest_version": 1,
        "renderer_type": "sprite_frames",
        "states": {
            "idle": {"animation_id": "idle"},
            "listening": {"animation_id": "listen"},
            "thinking": {"animation_id": "think"},
            "speaking": {"animation_id": "speak"},
            "error": {"animation_id": "error"},
        },
        "animations": {
            "idle": {"asset_ids": ["asset-idle"], "frame_rate": 1, "loop": True},
            "listen": {"asset_ids": ["asset-listen"], "frame_rate": 8, "loop": True},
            "think": {"asset_ids": ["asset-think"], "frame_rate": 8, "loop": True},
            "speak": {"asset_ids": ["asset-speak"], "frame_rate": 12, "loop": True},
            "error": {"asset_ids": ["asset-error"], "frame_rate": 1, "loop": False},
        },
        "fallbacks": {"tool_running": ["thinking", "idle"]},
    }


def test_renderer_capability_registry_lists_only_sprite_frames_in_v1() -> None:
    capabilities = list_persona_visual_renderer_capabilities()

    assert [cap.renderer_type for cap in capabilities] == ["sprite_frames"]
    capability = capabilities[0]
    assert capability.display_name == "Sprite frames"
    assert capability.manifest_versions == (1,)
    assert capability.can_validate is True
    assert capability.can_activate is True
    assert capability.buddy_runtime_supported is True
    assert capability.import_supported is True
    assert capability.export_supported is True
    assert capability.disabled_reason is None


def test_renderer_capability_lookup_rejects_unknown_or_future_renderers() -> None:
    assert get_persona_visual_renderer_capability("sprite_frames") is not None
    assert get_persona_visual_renderer_capability(" sprite_frames ") is None
    assert get_persona_visual_renderer_capability("live2d") is None
    assert get_persona_visual_renderer_capability("sprite_sheet") is None
    assert get_persona_visual_renderer_capability("not_real") is None


def test_valid_manifest_resolves_required_states_and_normalizes_frames() -> None:
    result = validate_visual_manifest(
        _activatable_manifest(),
        available_asset_ids={
            "asset-idle",
            "asset-listen",
            "asset-think",
            "asset-speak",
            "asset-error",
        },
        require_activatable=True,
    )

    assert set(result.resolved_required_states) == REQUIRED_VISUAL_STATES
    assert result.manifest["animations"]["idle"]["frames"] == [
        {"asset_id": "asset-idle"}
    ]


@pytest.mark.parametrize("renderer_type", ["live2d", "static_image", "sprite_sheet", "not_real"])
def test_manifest_rejects_unsupported_renderer_types(renderer_type: str) -> None:
    manifest = _activatable_manifest()
    manifest["renderer_type"] = renderer_type

    with pytest.raises(PersonaVisualManifestError, match="unsupported renderer_type"):
        validate_visual_manifest(
            manifest,
            available_asset_ids={
                "asset-idle",
                "asset-listen",
                "asset-think",
                "asset-speak",
                "asset-error",
            },
            require_activatable=True,
        )


def test_manifest_rejects_whitespace_padded_renderer_type() -> None:
    manifest = _activatable_manifest()
    manifest["renderer_type"] = " sprite_frames "

    with pytest.raises(PersonaVisualManifestError, match="unsupported renderer_type"):
        validate_visual_manifest(
            manifest,
            available_asset_ids={
                "asset-idle",
                "asset-listen",
                "asset-think",
                "asset-speak",
                "asset-error",
            },
            require_activatable=True,
        )


def test_activation_rejects_missing_required_state() -> None:
    manifest = {
        "manifest_version": 1,
        "renderer_type": "sprite_frames",
        "states": {"idle": {"animation_id": "idle"}},
        "animations": {"idle": {"asset_ids": ["asset-idle"], "frame_rate": 1}},
        "fallbacks": {},
    }

    with pytest.raises(PersonaVisualManifestError, match="listening"):
        validate_visual_manifest(
            manifest,
            available_asset_ids={"asset-idle"},
            require_activatable=True,
        )


def test_rejects_fallback_cycles() -> None:
    manifest = {
        "manifest_version": 1,
        "renderer_type": "sprite_frames",
        "states": {"idle": {"animation_id": "idle"}},
        "animations": {"idle": {"asset_ids": ["asset-idle"], "frame_rate": 1}},
        "fallbacks": {"thinking": ["tool_running"], "tool_running": ["thinking"]},
    }

    with pytest.raises(PersonaVisualManifestError, match="cycle"):
        validate_visual_manifest(
            manifest,
            available_asset_ids={"asset-idle"},
            require_activatable=False,
        )


def test_accepts_sprite_sheet_regions_and_preview_frame() -> None:
    manifest = {
        "manifest_version": 1,
        "renderer_type": "sprite_frames",
        "states": {"idle": {"animation_id": "idle"}},
        "animations": {
            "idle": {
                "frames": [
                    {
                        "asset_id": "sheet-asset",
                        "region": {"x": 0, "y": 0, "width": 128, "height": 128},
                        "duration_ms": 120,
                    },
                    {
                        "asset_id": "sheet-asset",
                        "region": {"x": 128, "y": 0, "width": 128, "height": 128},
                        "duration_ms": 120,
                    },
                ],
                "frame_rate": 8,
                "preview_frame": 1,
            }
        },
    }

    result = validate_visual_manifest(
        manifest,
        available_asset_ids={"sheet-asset"},
        available_asset_dimensions={"sheet-asset": (256, 128)},
        require_activatable=False,
    )

    assert result.manifest["animations"]["idle"]["frames"][1]["region"]["x"] == 128
    assert result.manifest["animations"]["idle"]["preview_frame"] == 1


def test_rejects_sprite_sheet_regions_outside_asset_bounds() -> None:
    manifest = {
        "manifest_version": 1,
        "renderer_type": "sprite_frames",
        "states": {"idle": {"animation_id": "idle"}},
        "animations": {
            "idle": {
                "frames": [
                    {
                        "asset_id": "sheet-asset",
                        "region": {"x": 200, "y": 0, "width": 128, "height": 128},
                    }
                ],
                "frame_rate": 8,
            }
        },
    }

    with pytest.raises(PersonaVisualManifestError, match="region"):
        validate_visual_manifest(
            manifest,
            available_asset_ids={"sheet-asset"},
            available_asset_dimensions={"sheet-asset": (256, 128)},
            require_activatable=False,
        )


def test_rejects_preview_frame_out_of_range() -> None:
    manifest = {
        "manifest_version": 1,
        "renderer_type": "sprite_frames",
        "states": {"idle": {"animation_id": "idle"}},
        "animations": {
            "idle": {
                "asset_ids": ["asset-idle"],
                "frame_rate": 1,
                "preview_frame": 2,
            }
        },
    }

    with pytest.raises(PersonaVisualManifestError, match="preview_frame"):
        validate_visual_manifest(
            manifest,
            available_asset_ids={"asset-idle"},
            require_activatable=False,
        )


def test_rejects_too_many_frames() -> None:
    manifest = {
        "manifest_version": 1,
        "renderer_type": "sprite_frames",
        "states": {"idle": {"animation_id": "idle"}},
        "animations": {
            "idle": {
                "asset_ids": [
                    f"asset-{index}"
                    for index in range(MAX_FRAMES_PER_ANIMATION + 1)
                ],
                "frame_rate": 1,
            }
        },
    }

    with pytest.raises(PersonaVisualManifestError, match="240"):
        validate_visual_manifest(
            manifest,
            available_asset_ids={
                f"asset-{index}" for index in range(MAX_FRAMES_PER_ANIMATION + 1)
            },
            require_activatable=False,
        )


def test_rejects_invalid_authored_triggers() -> None:
    manifest = _activatable_manifest()
    manifest["authored_triggers"] = [
        {
            "id": "bad-trigger",
            "source": "unknown",
            "match": "notes",
            "state": "tool_running",
            "duration_ms": 2500,
            "priority": 20,
        }
    ]

    with pytest.raises(PersonaVisualManifestError, match="source"):
        validate_visual_manifest(
            manifest,
            available_asset_ids={
                "asset-idle",
                "asset-listen",
                "asset-think",
                "asset-speak",
                "asset-error",
            },
            require_activatable=True,
        )


def test_rejects_unknown_assets() -> None:
    manifest = {
        "manifest_version": 1,
        "renderer_type": "sprite_frames",
        "states": {"idle": {"animation_id": "idle"}},
        "animations": {"idle": {"asset_ids": ["missing-asset"], "frame_rate": 1}},
    }

    with pytest.raises(PersonaVisualManifestError, match="missing-asset"):
        validate_visual_manifest(
            manifest,
            available_asset_ids=set(),
            require_activatable=False,
        )
