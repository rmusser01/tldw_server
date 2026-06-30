"""Deterministic starter matrix expansion for VN asset packs."""

from __future__ import annotations

from dataclasses import dataclass

from tldw_Server_API.app.core.VN_Assets.constants import (
    ASSET_TYPE_BACKGROUND,
    ASSET_TYPE_CG,
    ASSET_TYPE_DEPTH_COMPANION,
    ASSET_TYPE_SPRITE,
    DEFAULT_VN_ASSET_PACK_ITEM_LIMIT,
    DEFAULT_VN_ASSET_SLOT_VARIANT_LIMIT,
    ERROR_ITEM_LIMIT_EXCEEDED,
    ERROR_SLOT_VARIANT_LIMIT_EXCEEDED,
)
from tldw_Server_API.app.core.VN_Assets.models import VNAssetSlot


@dataclass(frozen=True, slots=True)
class StarterMatrixDimension:
    asset_type: str
    slot_name: str
    required_for_runtime: bool = True
    lazy_depth: bool = False


STARTER_MATRIX_DIMENSIONS: tuple[StarterMatrixDimension, ...] = (
    StarterMatrixDimension(ASSET_TYPE_BACKGROUND, "interior"),
    StarterMatrixDimension(ASSET_TYPE_BACKGROUND, "exterior"),
    StarterMatrixDimension(ASSET_TYPE_SPRITE, "primary_neutral"),
    StarterMatrixDimension(ASSET_TYPE_SPRITE, "primary_happy"),
    StarterMatrixDimension(ASSET_TYPE_SPRITE, "primary_concerned"),
    StarterMatrixDimension(ASSET_TYPE_CG, "opening"),
)


def expand_starter_matrix(
    *,
    primary_character_id: int,
    variant_count: int = 1,
    max_items: int = DEFAULT_VN_ASSET_PACK_ITEM_LIMIT,
    max_variants_per_slot: int = DEFAULT_VN_ASSET_SLOT_VARIANT_LIMIT,
) -> list[VNAssetSlot]:
    """Expand the starter matrix into deterministic planned slot definitions."""
    if primary_character_id < 1:
        raise ValueError("primary_character_id_required")
    if variant_count < 1:
        raise ValueError("variant_count_must_be_positive")
    if max_variants_per_slot < 1:
        raise ValueError("max_variants_per_slot_must_be_positive")
    if variant_count > max_variants_per_slot:
        raise ValueError(ERROR_SLOT_VARIANT_LIMIT_EXCEEDED)
    if max_items < 1:
        raise ValueError("max_items_must_be_positive")

    dimensions = sorted(
        STARTER_MATRIX_DIMENSIONS,
        key=lambda dimension: (dimension.asset_type, dimension.slot_name),
    )
    planned_item_count = sum(
        0 if dimension.lazy_depth else variant_count
        for dimension in dimensions
    )
    if planned_item_count > max_items:
        raise ValueError(ERROR_ITEM_LIMIT_EXCEEDED)

    slots: list[VNAssetSlot] = []
    for dimension in dimensions:
        slot_variant_count = 0 if dimension.lazy_depth else variant_count
        slot_key = _slot_key(primary_character_id, dimension)
        slots.append(
            VNAssetSlot(
                asset_type=dimension.asset_type,
                slot_key=slot_key,
                labels={
                    "matrix": "starter",
                    "primary_character_id": primary_character_id,
                    "slot_name": dimension.slot_name,
                },
                variant_count=slot_variant_count,
                required_for_runtime=dimension.required_for_runtime,
            )
        )
        if dimension.asset_type == ASSET_TYPE_BACKGROUND:
            depth_dimension = StarterMatrixDimension(
                ASSET_TYPE_DEPTH_COMPANION,
                dimension.slot_name,
                required_for_runtime=False,
                lazy_depth=True,
            )
            slots.append(
                VNAssetSlot(
                    asset_type=depth_dimension.asset_type,
                    slot_key=_slot_key(primary_character_id, depth_dimension),
                    labels={
                        "matrix": "starter",
                        "primary_character_id": primary_character_id,
                        "slot_name": depth_dimension.slot_name,
                        "parent_slot_key": slot_key,
                    },
                    variant_count=0,
                    required_for_runtime=False,
                    depends_on_slot_key=slot_key,
                )
            )
    return slots


def _slot_key(primary_character_id: int, dimension: StarterMatrixDimension) -> str:
    return f"{dimension.asset_type}.{primary_character_id}.{dimension.slot_name}"
