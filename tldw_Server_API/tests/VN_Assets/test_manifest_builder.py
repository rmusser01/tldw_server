from tldw_Server_API.app.core.VN_Assets.manifest import build_manifest
from tldw_Server_API.app.core.VN_Assets.models import (
    VNAssetItem,
    VNAssetPack,
    VNAssetSlot,
)


def _pack() -> VNAssetPack:
    return VNAssetPack(
        id=1,
        owner_user_id=42,
        title="Starter Pack",
        primary_character_id=7,
    )


def _sprite_slot() -> VNAssetSlot:
    return VNAssetSlot(
        id=10,
        pack_id=1,
        asset_type="sprite",
        slot_key="sprite.primary.neutral",
        labels={"expression": "neutral", "pose": "front"},
        variant_count=1,
    )


def _background_slot() -> VNAssetSlot:
    return VNAssetSlot(
        id=20,
        pack_id=1,
        asset_type="background",
        slot_key="background.interior.day",
        labels={"location": "interior", "time": "day"},
        variant_count=1,
    )


def _depth_slot() -> VNAssetSlot:
    return VNAssetSlot(
        id=21,
        pack_id=1,
        asset_type="depth_companion",
        slot_key="depth_companion.interior.day",
        labels={"location": "interior", "time": "day"},
        variant_count=0,
        required_for_runtime=False,
        depends_on_slot_id=20,
    )


def _cg_slot() -> VNAssetSlot:
    return VNAssetSlot(
        id=30,
        pack_id=1,
        asset_type="cg",
        slot_key="cg.opening",
        labels={"scenario": "opening"},
        variant_count=1,
    )


def _item(
    item_id: int,
    review_status: str,
    *,
    slot_id: int = 10,
    preferred: bool = False,
    width: int = 768,
    height: int = 1024,
) -> VNAssetItem:
    return VNAssetItem(
        id=item_id,
        pack_id=1,
        slot_id=slot_id,
        variant_index=0,
        review_status=review_status,
        generated_file_id=1000 + item_id,
        mime_type="image/png",
        width=width,
        height=height,
        preferred=preferred,
    )


def test_manifest_exposes_approved_items_only() -> None:
    approved_item = _item(1, "approved")
    draft_item = _item(2, "draft")
    hidden_item = _item(3, "hidden")
    rejected_item = _item(4, "rejected")

    manifest = build_manifest(
        pack=_pack(),
        slots=[_sprite_slot()],
        items=[approved_item, draft_item, hidden_item, rejected_item],
    )

    assert [item["item_id"] for item in manifest["assets"]["sprites"]] == [approved_item.id]


def test_manifest_uses_runtime_contract_shape_and_content_urls() -> None:
    approved_item = _item(1, "approved", preferred=True)

    manifest = build_manifest(
        pack=_pack(),
        slots=[_sprite_slot()],
        items=[approved_item],
    )
    sprite = manifest["assets"]["sprites"][0]

    assert manifest["schema_version"] == "vn_asset_manifest.v1"
    assert manifest["pack_id"] == 1
    assert manifest["title"] == "Starter Pack"
    assert manifest["primary_character_id"] == 7
    assert manifest["content_rating"] == "general"
    assert set(manifest["assets"]) == {
        "sprites",
        "backgrounds",
        "depth_companions",
        "cgs",
    }
    assert sprite["content_url"] == "/api/v1/vn/vn-assets/packs/1/items/1/content"
    assert sprite["labels"] == {"expression": "neutral", "pose": "front"}
    assert sprite["preferred"] is True
    assert sprite["crop_box"] is None
    assert sprite["trim_status"] == "unknown"


def test_manifest_defaults_sprite_runtime_metadata() -> None:
    approved_sprite_without_anchor = _item(1, "approved")

    manifest = build_manifest(
        pack=_pack(),
        slots=[_sprite_slot()],
        items=[approved_sprite_without_anchor],
    )
    sprite = manifest["assets"]["sprites"][0]

    assert sprite["anchor"] == {"x": 0.5, "y": 1.0}
    assert sprite["scale_hint"] == 1.0
    assert sprite["has_alpha"] is False


def test_manifest_includes_background_depth_linkage_and_cgs() -> None:
    background = _item(
        2,
        "approved",
        slot_id=20,
        width=1280,
        height=720,
    )
    depth = VNAssetItem(
        id=3,
        pack_id=1,
        slot_id=21,
        variant_index=0,
        review_status="approved",
        generated_file_id=1003,
        mime_type="image/png",
        width=1280,
        height=720,
        depth_kind="prompted",
        parent_item_id=2,
    )
    cg = _item(4, "approved", slot_id=30, width=1280, height=720)

    manifest = build_manifest(
        pack=_pack(),
        slots=[_background_slot(), _depth_slot(), _cg_slot()],
        items=[background, depth, cg],
    )

    assert manifest["assets"]["backgrounds"][0]["depth_companion_item_id"] == 3
    assert manifest["assets"]["backgrounds"][0]["depth_companion_status"] == "available"
    assert manifest["assets"]["depth_companions"][0]["parent_item_id"] == 2
    assert manifest["assets"]["depth_companions"][0]["depth_kind"] == "prompted"
    assert manifest["assets"]["cgs"][0]["item_id"] == 4


def test_manifest_omits_depth_companion_when_parent_background_is_filtered_out() -> None:
    draft_background = _item(2, "draft", slot_id=20)
    depth = VNAssetItem(
        id=3,
        pack_id=1,
        slot_id=21,
        variant_index=0,
        review_status="approved",
        generated_file_id=1003,
        mime_type="image/png",
        depth_kind="prompted",
        parent_item_id=2,
    )

    manifest = build_manifest(
        pack=_pack(),
        slots=[_background_slot(), _depth_slot()],
        items=[draft_background, depth],
    )

    assert manifest["assets"]["backgrounds"] == []
    assert manifest["assets"]["depth_companions"] == []


def test_manifest_reports_unavailable_depth_when_requested_without_approved_item() -> None:
    background = _item(2, "approved", slot_id=20)

    manifest = build_manifest(
        pack=_pack(),
        slots=[_background_slot(), _depth_slot()],
        items=[background],
    )

    assert manifest["assets"]["backgrounds"][0]["depth_companion_status"] == "unavailable"
