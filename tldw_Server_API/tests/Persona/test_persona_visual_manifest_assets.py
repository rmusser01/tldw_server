from copy import deepcopy

from tldw_Server_API.app.core.Persona.visual_manifest_assets import (
    collect_visual_manifest_asset_ids,
    remap_visual_manifest_assets,
)


def test_collect_visual_manifest_asset_ids_reads_all_supported_references() -> None:
    manifest = {
        "manifest_version": 1,
        "renderer_type": "sprite_frames",
        "states": {"idle": {"animation_id": "idle"}},
        "animations": {
            "idle": {
                "frames": [{"asset_id": "asset-frame"}],
                "asset_ids": ["asset-sheet"],
                "preview_asset_id": "asset-preview",
            }
        },
    }

    assert collect_visual_manifest_asset_ids(manifest) == {
        "asset-frame",
        "asset-sheet",
        "asset-preview",
    }


def test_remap_visual_manifest_assets_returns_copy_with_supported_references_remapped() -> None:
    manifest = {
        "manifest_version": 1,
        "renderer_type": "sprite_frames",
        "states": {"idle": {"animation_id": "idle"}},
        "animations": {
            "idle": {
                "frames": [{"asset_id": "source-a"}, {"asset_id": "source-b"}],
                "asset_ids": ["source-a", "source-c"],
                "preview_asset_id": "source-b",
            }
        },
    }
    original = deepcopy(manifest)

    remapped = remap_visual_manifest_assets(
        manifest,
        {"source-a": "target-a", "source-b": "target-b"},
    )

    assert manifest == original
    animation = remapped["animations"]["idle"]
    assert animation["frames"] == [
        {"asset_id": "target-a"},
        {"asset_id": "target-b"},
    ]
    assert animation["asset_ids"] == ["target-a", "source-c"]
    assert animation["preview_asset_id"] == "target-b"
