import pytest

from tldw_Server_API.app.core.Persona.visual_renderer_capabilities import (
    PersonaVisualRendererCapability,
    get_persona_visual_renderer_capability,
    list_persona_visual_renderer_capabilities,
)
from tldw_Server_API.app.core.Persona import visuals as visuals_module
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


def test_renderer_capability_registry_exposes_v2_metadata_without_enabling_live2d() -> None:
    capabilities = list_persona_visual_renderer_capabilities()

    assert [cap.renderer_type for cap in capabilities] == ["sprite_frames", "live2d"]
    by_renderer = {cap.renderer_type: cap for cap in capabilities}
    sprite_frames = by_renderer["sprite_frames"]
    assert sprite_frames.display_name == "Sprite frames"
    assert sprite_frames.manifest_versions == (1,)
    assert sprite_frames.renderer_contract_versions == (1,)
    assert "frame" in sprite_frames.supported_asset_roles
    assert "sprite_sheet" in sprite_frames.supported_asset_roles
    assert sprite_frames.required_role_categories == ()
    assert sprite_frames.can_validate is True
    assert sprite_frames.can_activate is True
    assert sprite_frames.buddy_runtime_supported is True
    assert sprite_frames.import_supported is True
    assert sprite_frames.export_supported is True
    assert sprite_frames.setup_status == "supported"
    assert sprite_frames.setup_blockers == ()
    assert sprite_frames.disabled_reason is None

    live2d = by_renderer["live2d"]
    assert live2d.manifest_versions == (2,)
    assert live2d.renderer_contract_versions == (1,)
    assert "live2d_model_manifest" in live2d.supported_asset_roles
    assert live2d.required_role_categories == ("fallback_preview", "source_manifest")
    assert live2d.role_category_map["source_manifest"] == ("live2d_model_manifest",)
    assert live2d.can_validate is False
    assert live2d.can_activate is False
    assert live2d.buddy_runtime_supported is False
    assert live2d.import_supported is False
    assert live2d.export_supported is False
    assert live2d.requires_static_fallback is True
    assert live2d.setup_status == "unsupported_renderer"
    assert "runtime_adapter_not_implemented" in live2d.setup_blockers
    assert live2d.disabled_reason == "runtime_adapter_not_implemented"


def test_renderer_capability_lookup_keeps_live2d_non_activatable() -> None:
    assert get_persona_visual_renderer_capability("sprite_frames") is not None
    assert get_persona_visual_renderer_capability(" sprite_frames ") is None
    live2d = get_persona_visual_renderer_capability("live2d")
    assert live2d is not None
    assert live2d.can_validate is False
    assert live2d.can_activate is False
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


def test_manifest_rejects_validation_only_capability_for_activation(monkeypatch) -> None:
    manifest = _activatable_manifest()
    validation_only_capability = PersonaVisualRendererCapability(
        renderer_type="sprite_frames",
        display_name="Sprite frames",
        manifest_versions=(1,),
        can_validate=True,
        can_activate=False,
        buddy_runtime_supported=True,
        import_supported=True,
        export_supported=True,
    )
    monkeypatch.setattr(
        visuals_module,
        "get_persona_visual_renderer_capability",
        lambda renderer_type: validation_only_capability,
    )

    result = validate_visual_manifest(
        manifest,
        available_asset_ids={
            "asset-idle",
            "asset-listen",
            "asset-think",
            "asset-speak",
            "asset-error",
        },
        require_activatable=False,
    )
    assert result.manifest["renderer_type"] == "sprite_frames"

    with pytest.raises(
        PersonaVisualManifestError,
        match="unsupported renderer_type for activation",
    ):
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


def test_manifest_error_sanitizes_renderer_type_for_messages() -> None:
    manifest = _activatable_manifest()
    manifest["renderer_type"] = "live2d\nsecret"

    with pytest.raises(PersonaVisualManifestError) as exc_info:
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

    message = str(exc_info.value)
    assert "\n" not in message
    assert "live2d\\nsecret" in message


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


def test_activatable_manifest_accepts_sprite_sheet_regions_with_known_dimensions() -> None:
    manifest = _activatable_manifest()
    for index, animation in enumerate(manifest["animations"].values()):
        animation.pop("asset_ids", None)
        animation["frames"] = [
            {
                "asset_id": "sheet-asset",
                "region": {
                    "x": index * 64,
                    "y": 0,
                    "width": 64,
                    "height": 64,
                },
                "duration_ms": 120,
            }
        ]

    result = validate_visual_manifest(
        manifest,
        available_asset_ids={"sheet-asset"},
        available_asset_dimensions={"sheet-asset": (512, 128)},
        require_activatable=True,
    )

    assert set(result.resolved_required_states) == REQUIRED_VISUAL_STATES
    assert result.manifest["renderer_type"] == "sprite_frames"
    for animation in result.manifest["animations"].values():
        frame = animation["frames"][0]
        assert frame["asset_id"] == "sheet-asset"
        assert frame["region"]["width"] == 64
        assert frame["region"]["height"] == 64


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


def test_accepts_sprite_sheet_regions_without_dimensions_until_asset_metadata_exists() -> None:
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

    result = validate_visual_manifest(
        manifest,
        available_asset_ids={"sheet-asset"},
        available_asset_dimensions={},
        require_activatable=False,
    )

    assert result.manifest["animations"]["idle"]["frames"][0]["region"] == {
        "x": 200,
        "y": 0,
        "width": 128,
        "height": 128,
    }


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
