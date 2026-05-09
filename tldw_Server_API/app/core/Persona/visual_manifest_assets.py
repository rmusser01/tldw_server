from __future__ import annotations

from copy import deepcopy
from typing import Any


def collect_visual_manifest_asset_ids(manifest: dict[str, Any]) -> set[str]:
    asset_ids: set[str] = set()
    animations = manifest.get("animations")
    if not isinstance(animations, dict):
        return asset_ids
    for animation in animations.values():
        if not isinstance(animation, dict):
            continue
        frames = animation.get("frames")
        if isinstance(frames, list):
            for frame in frames:
                if isinstance(frame, dict):
                    asset_id = str(frame.get("asset_id") or "").strip()
                    if asset_id:
                        asset_ids.add(asset_id)
        listed_asset_ids = animation.get("asset_ids")
        if isinstance(listed_asset_ids, list):
            for asset_id in listed_asset_ids:
                normalized = str(asset_id or "").strip()
                if normalized:
                    asset_ids.add(normalized)
        preview_asset_id = str(animation.get("preview_asset_id") or "").strip()
        if preview_asset_id:
            asset_ids.add(preview_asset_id)
    return asset_ids


def remap_visual_manifest_assets(
    manifest: dict[str, Any],
    asset_id_map: dict[str, str],
) -> dict[str, Any]:
    remapped = deepcopy(manifest)
    animations = remapped.get("animations")
    if not isinstance(animations, dict):
        return remapped
    for animation in animations.values():
        if not isinstance(animation, dict):
            continue
        frames = animation.get("frames")
        if isinstance(frames, list):
            for frame in frames:
                if not isinstance(frame, dict):
                    continue
                asset_id = str(frame.get("asset_id") or "")
                if asset_id in asset_id_map:
                    frame["asset_id"] = asset_id_map[asset_id]
        asset_ids = animation.get("asset_ids")
        if isinstance(asset_ids, list):
            animation["asset_ids"] = [
                asset_id_map.get(str(asset_id), asset_id)
                for asset_id in asset_ids
            ]
        preview_asset_id = str(animation.get("preview_asset_id") or "")
        if preview_asset_id in asset_id_map:
            animation["preview_asset_id"] = asset_id_map[preview_asset_id]
    return remapped
