"""Approved-only VN asset manifest builder."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from tldw_Server_API.app.core.VN_Assets.constants import (
    ASSET_TYPE_BACKGROUND,
    ASSET_TYPE_CG,
    ASSET_TYPE_DEPTH_COMPANION,
    ASSET_TYPE_SPRITE,
    ITEM_REVIEW_STATUS_APPROVED,
    MANIFEST_SCHEMA_VERSION,
)
from tldw_Server_API.app.core.VN_Assets.models import VNAssetItem, VNAssetPack, VNAssetSlot


ASSET_COLLECTIONS = {
    ASSET_TYPE_BACKGROUND: "backgrounds",
    ASSET_TYPE_DEPTH_COMPANION: "depth_companions",
    ASSET_TYPE_SPRITE: "sprites",
    ASSET_TYPE_CG: "cgs",
}


def build_manifest(
    *,
    pack: VNAssetPack,
    slots: Iterable[VNAssetSlot],
    items: Iterable[VNAssetItem],
    content_url_builder: Callable[[VNAssetPack, VNAssetItem], str] | None = None,
) -> dict[str, Any]:
    """Build a manifest containing approved items only."""
    slot_list = list(slots)
    slot_by_id = {
        slot.id: slot
        for slot in slot_list
        if slot.id is not None
    }
    assets: dict[str, list[dict[str, Any]]] = {
        collection_name: []
        for collection_name in ASSET_COLLECTIONS.values()
    }

    approved_items = sorted(
        (
            item
            for item in items
            if item.review_status == ITEM_REVIEW_STATUS_APPROVED and item.slot_id in slot_by_id
        ),
        key=lambda item: (slot_by_id[item.slot_id].slot_key, item.variant_index, item.id),
    )
    approved_background_item_ids = {
        item.id
        for item in approved_items
        if item.id is not None and slot_by_id[item.slot_id].asset_type == ASSET_TYPE_BACKGROUND
    }
    approved_depth_by_parent_item_id = {
        item.parent_item_id: item
        for item in approved_items
        if (
            slot_by_id[item.slot_id].asset_type == ASSET_TYPE_DEPTH_COMPANION
            and item.parent_item_id is not None
            and item.parent_item_id in approved_background_item_ids
        )
    }
    depth_slots_by_parent_slot_id = {
        slot.depends_on_slot_id
        for slot in slot_list
        if slot.asset_type == ASSET_TYPE_DEPTH_COMPANION and slot.depends_on_slot_id is not None
    }
    depth_slots_by_parent_slot_key = {
        slot.depends_on_slot_key
        for slot in slot_list
        if slot.asset_type == ASSET_TYPE_DEPTH_COMPANION and slot.depends_on_slot_key is not None
    }

    for item in approved_items:
        slot = slot_by_id[item.slot_id]
        collection_name = ASSET_COLLECTIONS.get(slot.asset_type)
        if collection_name is None:
            continue

        asset = _base_asset_entry(
            item,
            slot,
            content_url=_content_url(pack, item, content_url_builder),
        )
        if slot.asset_type == ASSET_TYPE_SPRITE:
            asset.update(
                {
                    "anchor": item.anchor or {"x": 0.5, "y": 1.0},
                    "scale_hint": item.scale_hint if item.scale_hint is not None else 1.0,
                    "has_alpha": bool(item.has_alpha) if item.has_alpha is not None else False,
                    "crop_box": item.crop_box,
                    "trim_status": item.trim_status,
                }
            )
        elif slot.asset_type == ASSET_TYPE_BACKGROUND:
            depth_item = approved_depth_by_parent_item_id.get(item.id)
            if depth_item is not None:
                asset["depth_companion_item_id"] = depth_item.id
                asset["depth_companion_status"] = "available"
            elif (
                slot.id in depth_slots_by_parent_slot_id
                or slot.slot_key in depth_slots_by_parent_slot_key
            ):
                asset["depth_companion_status"] = "unavailable"
        elif slot.asset_type == ASSET_TYPE_DEPTH_COMPANION:
            if item.parent_item_id not in approved_background_item_ids:
                continue
            asset.update(
                {
                    "parent_item_id": item.parent_item_id,
                    "depth_kind": item.depth_kind,
                }
            )
        assets[collection_name].append(asset)

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "pack_id": pack.id,
        "title": pack.title,
        "primary_character_id": pack.primary_character_id,
        "content_rating": pack.content_rating,
        "assets": assets,
    }


def _base_asset_entry(
    item: VNAssetItem,
    slot: VNAssetSlot,
    *,
    content_url: str,
) -> dict[str, Any]:
    return {
        "item_id": item.id,
        "slot_id": item.slot_id,
        "slot_key": slot.slot_key,
        "asset_type": slot.asset_type,
        "variant_index": item.variant_index,
        "content_url": content_url,
        "generated_file_id": item.generated_file_id,
        "file_artifact_id": item.file_artifact_id,
        "storage_ref": item.storage_ref,
        "mime_type": item.mime_type,
        "width": item.width,
        "height": item.height,
        "labels": slot.labels,
        "preferred": item.preferred,
    }


def _content_url(
    pack: VNAssetPack,
    item: VNAssetItem,
    content_url_builder: Callable[[VNAssetPack, VNAssetItem], str] | None,
) -> str:
    if content_url_builder is not None:
        return content_url_builder(pack, item)
    return f"/api/v1/vn/vn-assets/packs/{pack.id}/items/{item.id}/content"
