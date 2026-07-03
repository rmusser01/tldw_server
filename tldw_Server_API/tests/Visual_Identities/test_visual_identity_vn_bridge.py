from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.Visual_Identities.vn_bridge import (
    build_vn_visual_identity_source_context,
)

pytestmark = pytest.mark.unit

OWNER_USER_ID = 1


class FakeVNAssetPacksRepository:
    def __init__(self) -> None:
        self.items = {
            29: {"id": 29, "pack_id": 7, "slot_id": 11},
        }
        self.slots = {
            11: {"id": 11, "pack_id": 7, "slot_key": "happy", "asset_type": "sprite"},
            12: {"id": 12, "pack_id": 7, "slot_key": "sad", "asset_type": "sprite"},
        }
        self.packs = {
            7: {"id": 7, "owner_user_id": OWNER_USER_ID},
            8: {"id": 8, "owner_user_id": OWNER_USER_ID},
        }

    def get_item(self, item_id: int) -> dict[str, Any] | None:
        return self.items.get(item_id)

    def get_slot(self, slot_id: int) -> dict[str, Any] | None:
        return self.slots.get(slot_id)

    def get_pack(self, pack_id: int) -> dict[str, Any] | None:
        return self.packs.get(pack_id)


@pytest.fixture
def vn_repository() -> FakeVNAssetPacksRepository:
    return FakeVNAssetPacksRepository()


def test_vn_bridge_derives_trusted_context_from_generated_file(
    vn_repository: FakeVNAssetPacksRepository,
) -> None:
    context = build_vn_visual_identity_source_context(
        user_id=OWNER_USER_ID,
        vn_repository=vn_repository,
        generated_file_record={
            "id": 42,
            "source_feature": "vn_assets",
            "source_ref": "vn_asset_item:29",
            "mime_type": "image/webp",
            "original_filename": "maya_happy.webp",
        },
        requested_context={
            "source_feature": "client-lie",
            "generated_file_id": 999,
            "vn_item_id": 29,
            "vn_slot_label": "Happy",
        },
    )

    assert context["source_feature"] == "vn_assets"
    assert context["generated_file_id"] == 42
    assert context["filename"] == "maya_happy.webp"
    assert context["source_ref"] == "vn_asset_item:29"
    assert context["vn_item_id"] == 29
    assert context["vn_slot_label"] == "Happy"


def test_vn_bridge_verifies_structural_vn_ids_before_persisting(
    vn_repository: FakeVNAssetPacksRepository,
) -> None:
    context = build_vn_visual_identity_source_context(
        user_id=OWNER_USER_ID,
        vn_repository=vn_repository,
        generated_file_record={
            "id": 42,
            "source_feature": "vn_assets",
            "source_ref": "vn_asset_item:29",
            "mime_type": "image/webp",
            "original_filename": "maya_happy.webp",
        },
        requested_context={
            "vn_item_id": 29,
            "vn_pack_id": 7,
            "vn_slot_id": 11,
            "vn_slot_key": "happy",
            "vn_asset_type": "sprite",
        },
    )

    assert context["vn_item_id"] == 29
    assert context["vn_pack_id"] == 7
    assert context["vn_slot_id"] == 11
    assert context["vn_slot_key"] == "happy"
    assert context["vn_asset_type"] == "sprite"


def test_vn_bridge_accepts_digit_string_structural_ids(
    vn_repository: FakeVNAssetPacksRepository,
) -> None:
    context = build_vn_visual_identity_source_context(
        user_id=str(OWNER_USER_ID),
        vn_repository=vn_repository,
        generated_file_record={
            "id": "42",
            "source_feature": "vn_assets",
            "source_ref": "vn_asset_item:29",
        },
        requested_context={
            "vn_item_id": "29",
            "vn_pack_id": "7",
            "vn_slot_id": "11",
        },
    )

    assert context["generated_file_id"] == 42
    assert context["vn_item_id"] == 29
    assert context["vn_pack_id"] == 7
    assert context["vn_slot_id"] == 11


@pytest.mark.parametrize(
    "source_feature",
    [
        None,
        "",
        "   ",
    ],
)
def test_vn_bridge_rejects_missing_or_blank_source_feature(
    source_feature: str | None,
    vn_repository: FakeVNAssetPacksRepository,
) -> None:
    generated_file_record = {
        "id": 42,
        "source_ref": "vn_asset_item:29",
    }
    if source_feature is not None:
        generated_file_record["source_feature"] = source_feature

    with pytest.raises(ValueError, match="vn_generated_file_context_mismatch"):
        build_vn_visual_identity_source_context(
            user_id=OWNER_USER_ID,
            vn_repository=vn_repository,
            generated_file_record=generated_file_record,
        )


@pytest.mark.parametrize(
    "source_ref",
    [
        None,
        "",
        "vn_asset_item",
        "vn_asset_item:not-an-int",
        "image_gen:29",
    ],
)
def test_vn_bridge_rejects_missing_or_malformed_source_ref(
    source_ref: str | None,
    vn_repository: FakeVNAssetPacksRepository,
) -> None:
    generated_file_record = {
        "id": 42,
        "source_feature": "vn_assets",
    }
    if source_ref is not None:
        generated_file_record["source_ref"] = source_ref

    with pytest.raises(ValueError, match="vn_generated_file_context_mismatch"):
        build_vn_visual_identity_source_context(
            user_id=OWNER_USER_ID,
            vn_repository=vn_repository,
            generated_file_record=generated_file_record,
            requested_context={"vn_slot_label": "Happy"},
        )


def test_vn_bridge_rejects_item_source_ref_mismatch(
    vn_repository: FakeVNAssetPacksRepository,
) -> None:
    with pytest.raises(ValueError, match="vn_generated_file_context_mismatch"):
        build_vn_visual_identity_source_context(
            user_id=OWNER_USER_ID,
            vn_repository=vn_repository,
            generated_file_record={
                "id": 42,
                "source_feature": "vn_assets",
                "source_ref": "vn_asset_item:29",
            },
            requested_context={"vn_item_id": 30},
        )


@pytest.mark.parametrize(
    "requested_context",
    [
        {"vn_item_id": 29, "vn_pack_id": 8},
        {"vn_item_id": 29, "vn_slot_id": 12},
        {"vn_item_id": 29, "vn_slot_key": "sad"},
        {"vn_item_id": 29, "vn_asset_type": "background"},
        {"vn_item_id": 29.9},
        {"vn_item_id": 29, "vn_pack_id": 7.9},
        {"vn_item_id": 29, "vn_slot_id": 11.1},
    ],
)
def test_vn_bridge_rejects_unverified_structural_hints(
    requested_context: dict[str, object],
    vn_repository: FakeVNAssetPacksRepository,
) -> None:
    with pytest.raises(ValueError, match="vn_generated_file_context_mismatch"):
        build_vn_visual_identity_source_context(
            user_id=OWNER_USER_ID,
            vn_repository=vn_repository,
            generated_file_record={
                "id": 42,
                "source_feature": "vn_assets",
                "source_ref": "vn_asset_item:29",
            },
            requested_context=requested_context,
        )


def test_vn_bridge_rejects_cross_user_pack_owner(
    vn_repository: FakeVNAssetPacksRepository,
) -> None:
    vn_repository.packs[7]["owner_user_id"] = OWNER_USER_ID + 1

    with pytest.raises(ValueError, match="vn_generated_file_context_mismatch"):
        build_vn_visual_identity_source_context(
            user_id=OWNER_USER_ID,
            vn_repository=vn_repository,
            generated_file_record={
                "id": 42,
                "source_feature": "vn_assets",
                "source_ref": "vn_asset_item:29",
            },
        )
