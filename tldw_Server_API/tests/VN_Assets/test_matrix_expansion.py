import pytest

from tldw_Server_API.app.core.VN_Assets.matrix import expand_starter_matrix


def test_starter_matrix_expansion_is_deterministic() -> None:
    slots_a = expand_starter_matrix(primary_character_id=7, variant_count=2)
    slots_b = expand_starter_matrix(primary_character_id=7, variant_count=2)

    assert [slot.slot_key for slot in slots_a] == [slot.slot_key for slot in slots_b]
    assert all(slot.variant_count == 2 for slot in slots_a if slot.asset_type != "depth_companion")


def test_starter_matrix_includes_spec_asset_types_without_ui_slots() -> None:
    slots = expand_starter_matrix(primary_character_id=7, variant_count=2)
    asset_types = {slot.asset_type for slot in slots}

    assert {"sprite", "background", "depth_companion", "cg"}.issubset(asset_types)
    assert "ui" not in asset_types


def test_starter_matrix_adds_lazy_depth_slots_per_background_dependency() -> None:
    slots = expand_starter_matrix(primary_character_id=7, variant_count=2)
    background_slot_keys = {
        slot.slot_key
        for slot in slots
        if slot.asset_type == "background"
    }
    depth_slots = [
        slot
        for slot in slots
        if slot.asset_type == "depth_companion"
    ]

    assert len(depth_slots) == len(background_slot_keys)
    assert all(slot.variant_count == 0 for slot in depth_slots)
    assert all(slot.required_for_runtime is False for slot in depth_slots)
    assert {slot.depends_on_slot_key for slot in depth_slots} == background_slot_keys


def test_matrix_expansion_enforces_default_item_limit() -> None:
    with pytest.raises(ValueError, match="vn_asset_pack_item_limit_exceeded"):
        expand_starter_matrix(
            primary_character_id=7,
            variant_count=99,
            max_items=300,
            max_variants_per_slot=99,
        )


def test_matrix_expansion_enforces_default_slot_variant_limit() -> None:
    with pytest.raises(ValueError, match="vn_asset_slot_variant_limit_exceeded"):
        expand_starter_matrix(primary_character_id=7, variant_count=7)
