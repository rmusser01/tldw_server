"""Utilities for locating and rewriting Persona Visual manifest asset IDs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _remap_asset_id(asset_id: Any, asset_id_map: dict[str, str]) -> Any:
    """Normalize an asset ID and substitute it when a duplicate mapping exists."""
    normalized = str(asset_id or "").strip()
    if normalized in asset_id_map:
        return asset_id_map[normalized]
    return normalized or asset_id


def collect_visual_manifest_asset_ids(manifest: dict[str, Any]) -> set[str]:
    """Return normalized asset IDs referenced by supported manifest fields."""
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
    """Return a copy of a manifest with supported asset references remapped."""
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
                if "asset_id" not in frame:
                    continue
                frame["asset_id"] = _remap_asset_id(frame.get("asset_id"), asset_id_map)
        asset_ids = animation.get("asset_ids")
        if isinstance(asset_ids, list):
            animation["asset_ids"] = [
                _remap_asset_id(asset_id, asset_id_map)
                for asset_id in asset_ids
            ]
        if "preview_asset_id" in animation:
            animation["preview_asset_id"] = _remap_asset_id(
                animation.get("preview_asset_id"),
                asset_id_map,
            )
    return remapped
