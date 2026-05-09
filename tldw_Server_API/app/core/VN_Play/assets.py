"""Approved-manifest asset resolution for VN Play visual directives."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from tldw_Server_API.app.core.VN_Play.models import VisualDirectiveResolution


_ASSET_TYPE_COLLECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "background": ("backgrounds", "background"),
    "backgrounds": ("backgrounds", "background"),
    "sprite": ("sprites", "sprite"),
    "sprites": ("sprites", "sprite"),
    "depth": ("depth_companions", "depth_companion", "depth"),
    "depth_companion": ("depth_companions", "depth_companion", "depth"),
    "depth_companions": ("depth_companions", "depth_companion", "depth"),
    "cg": ("cgs", "cg"),
    "cgs": ("cgs", "cg"),
}


def resolve_visual_directive(
    manifest: Mapping[str, Any],
    directive: Mapping[str, Any],
    *,
    seed: str,
) -> VisualDirectiveResolution:
    """Resolve one visual directive against an approved VN asset manifest."""
    directive_dict = dict(directive)
    candidates = [
        item
        for item in _iter_manifest_items(manifest, directive_dict.get("asset_type"))
        if _is_approved_item(item)
        and _matches_slot_key(item, directive_dict)
        and _matches_labels(item, directive_dict)
    ]
    if not candidates:
        return VisualDirectiveResolution(
            applied=False,
            reason="asset_not_found",
            directive=directive_dict,
        )

    selected = sorted(candidates, key=lambda item: _candidate_sort_key(item, seed))[0]
    return VisualDirectiveResolution(
        applied=True,
        item=dict(selected),
        directive=directive_dict,
    )


def resolve_scene_directives(
    manifest: Mapping[str, Any],
    directives: Sequence[Mapping[str, Any]],
    *,
    seed: str,
) -> list[VisualDirectiveResolution]:
    """Resolve a set of visual directives with deterministic per-directive ordering."""
    return [
        resolve_visual_directive(manifest, directive, seed=f"{seed}:{index}")
        for index, directive in enumerate(directives)
    ]


def _iter_manifest_items(
    manifest: Mapping[str, Any],
    asset_type: Any,
) -> list[dict[str, Any]]:
    assets = manifest.get("assets", {})
    if not isinstance(assets, Mapping):
        return []

    collection_keys = _collection_keys_for_asset_type(asset_type)
    if collection_keys is not None:
        items: list[dict[str, Any]] = []
        for collection_key in collection_keys:
            items.extend(_list_of_dicts(assets.get(collection_key, [])))
        return items

    all_items: list[dict[str, Any]] = []
    for items in assets.values():
        all_items.extend(_list_of_dicts(items))
    return all_items


def _collection_keys_for_asset_type(asset_type: Any) -> list[str] | None:
    if not isinstance(asset_type, str) or not asset_type.strip():
        return None

    normalized = asset_type.strip().lower()
    aliases = _ASSET_TYPE_COLLECTION_ALIASES.get(normalized, (asset_type,))
    return list(dict.fromkeys((*aliases, asset_type)))


def _is_approved_item(item: Mapping[str, Any]) -> bool:
    if item.get("approved") is False:
        return False
    review_status = item.get("review_status")
    if isinstance(review_status, str) and review_status != "approved":
        return False
    status = item.get("status")
    if isinstance(status, str) and status not in {"approved", "ready"}:
        return False
    return True


def _matches_slot_key(item: Mapping[str, Any], directive: Mapping[str, Any]) -> bool:
    slot_key = directive.get("slot_key")
    if not isinstance(slot_key, str) or not slot_key:
        return True
    return item.get("slot_key") == slot_key


def _matches_labels(item: Mapping[str, Any], directive: Mapping[str, Any]) -> bool:
    requested = directive.get("labels", {})
    if not isinstance(requested, Mapping) or not requested:
        return True
    labels = item.get("labels", {})
    if not isinstance(labels, Mapping):
        return False
    return all(labels.get(key) == value for key, value in requested.items())


def _candidate_sort_key(item: Mapping[str, Any], seed: str) -> tuple[int, str]:
    preferred_rank = 0 if bool(item.get("preferred")) else 1
    identity = "|".join(
        [
            seed,
            str(item.get("slot_key", "")),
            str(item.get("item_id", "")),
            str(item.get("variant_index", "")),
        ]
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return (preferred_rank, digest)


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]
