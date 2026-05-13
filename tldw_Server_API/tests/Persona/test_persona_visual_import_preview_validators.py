from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Persona.visual_import_preview_validators import (
    PersonaVisualImportPreviewAsset,
    preview_renderer_import,
)


pytestmark = pytest.mark.unit


def _live2d_manifest() -> dict[str, object]:
    return {
        "manifest_version": 2,
        "renderer_type": "live2d",
        "renderer_contract_version": 1,
        "renderer_assets": {
            "fallback_preview_asset_id": "asset-fallback",
            "source_manifest_asset_id": "asset-model",
        },
        "states": {"idle": {"animation_id": "idle"}},
        "animations": {"idle": {"renderer_action": {"motion_group": "Idle"}}},
    }


def test_preview_renderer_import_reports_live2d_as_blocked_fixture_only() -> None:
    result = preview_renderer_import(
        manifest=_live2d_manifest(),
        assets=[
            PersonaVisualImportPreviewAsset(
                source_asset_id="asset-fallback",
                asset_role="fallback_preview",
                mime_type="image/png",
                width=256,
                height=256,
            ),
            PersonaVisualImportPreviewAsset(
                source_asset_id="asset-model",
                asset_role="live2d_model_manifest",
                mime_type="application/json",
            ),
        ],
    )

    assert result.status == "unsupported_renderer"
    assert result.renderer_type == "live2d"
    assert result.manifest_version == 2
    assert result.can_commit is False
    assert result.activation_eligible is False
    assert "runtime_adapter_not_implemented" in result.blockers
    assert result.normalized_role_categories == {
        "fallback_preview": ["asset-fallback"],
        "source_manifest": ["asset-model"],
    }


def test_preview_renderer_import_reports_missing_required_role_categories() -> None:
    result = preview_renderer_import(
        manifest=_live2d_manifest(),
        assets=[
            PersonaVisualImportPreviewAsset(
                source_asset_id="asset-model",
                asset_role="live2d_model_manifest",
                mime_type="application/json",
            ),
        ],
    )

    assert result.status == "unsupported_renderer"
    assert "missing_required_role_category:fallback_preview" in result.blockers
    assert "runtime_adapter_not_implemented" in result.blockers
    assert result.normalized_role_categories == {
        "fallback_preview": [],
        "source_manifest": ["asset-model"],
    }


def test_preview_renderer_import_reports_unknown_renderer_without_asset_writes() -> None:
    manifest = _live2d_manifest()
    manifest["renderer_type"] = "not_real"

    result = preview_renderer_import(
        manifest=manifest,
        assets=[
            PersonaVisualImportPreviewAsset(
                source_asset_id="asset-model",
                asset_role="renderer_metadata",
                mime_type="application/json",
            )
        ],
    )

    assert result.status == "unsupported_renderer"
    assert result.can_commit is False
    assert result.activation_eligible is False
    assert result.blockers == ["unknown_renderer:not_real"]
    assert result.warnings == []


def test_preview_renderer_import_sanitizes_unknown_renderer_blocker() -> None:
    unsafe_renderer_type = "not_real\n\tsecret\\path" + ("x" * 120)
    manifest = _live2d_manifest()
    manifest["renderer_type"] = unsafe_renderer_type

    result = preview_renderer_import(manifest=manifest, assets=[])

    assert result.renderer_type == unsafe_renderer_type
    assert len(result.blockers) == 1
    blocker = result.blockers[0]
    assert "\n" not in blocker
    assert "\t" not in blocker
    assert "\r" not in blocker
    assert "not_real\\n\\tsecret\\\\path" in blocker
    assert blocker.endswith("...")


def test_preview_renderer_import_keeps_sprite_frames_supported() -> None:
    result = preview_renderer_import(
        manifest={
            "manifest_version": 1,
            "renderer_type": "sprite_frames",
            "renderer_contract_version": 1,
        },
        assets=[
            PersonaVisualImportPreviewAsset(
                source_asset_id="asset-idle",
                asset_role="frame",
                mime_type="image/png",
                width=64,
                height=64,
            )
        ],
    )

    assert result.status == "supported"
    assert result.can_commit is True
    assert result.activation_eligible is True
    assert result.blockers == []
    assert result.normalized_role_categories["frame"] == ["asset-idle"]


def test_preview_renderer_import_blocks_missing_required_versions() -> None:
    result = preview_renderer_import(
        manifest={"renderer_type": "sprite_frames"},
        assets=[
            PersonaVisualImportPreviewAsset(
                source_asset_id="asset-idle",
                asset_role="frame",
                mime_type="image/png",
            )
        ],
    )

    assert result.status == "invalid_renderer_assets"
    assert result.can_commit is False
    assert result.activation_eligible is False
    assert "unsupported_manifest_version:None" in result.blockers
    assert "unsupported_renderer_contract_version:None" in result.blockers


def test_preview_renderer_import_rejects_non_integer_version_values() -> None:
    result = preview_renderer_import(
        manifest={
            "manifest_version": 1.9,
            "renderer_type": "sprite_frames",
            "renderer_contract_version": 1.9,
        },
        assets=[
            PersonaVisualImportPreviewAsset(
                source_asset_id="asset-idle",
                asset_role="frame",
                mime_type="image/png",
            )
        ],
    )

    assert result.manifest_version is None
    assert result.renderer_contract_version is None
    assert result.status == "invalid_renderer_assets"
    assert result.can_commit is False
    assert "unsupported_manifest_version:None" in result.blockers
    assert "unsupported_renderer_contract_version:None" in result.blockers


def test_preview_renderer_import_ignores_unexpected_asset_objects() -> None:
    result = preview_renderer_import(
        manifest={
            "manifest_version": 1,
            "renderer_type": "sprite_frames",
            "renderer_contract_version": 1,
        },
        assets=[object()],  # type: ignore[list-item]
    )

    assert result.status == "supported"
    assert result.normalized_role_categories == {}
