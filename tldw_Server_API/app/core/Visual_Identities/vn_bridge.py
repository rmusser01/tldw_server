"""VN asset provenance bridge for Visual Identity generated-file imports."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Protocol

from tldw_Server_API.app.core.VN_Assets.storage import (
    SOURCE_FEATURE_VN_ASSETS,
    vn_asset_source_ref,
)
from tldw_Server_API.app.core.Visual_Identities.source_context import canonicalize_source_context

VN_SOURCE_FEATURE = SOURCE_FEATURE_VN_ASSETS
VN_GENERATED_FILE_CONTEXT_MISMATCH = "vn_generated_file_context_mismatch"

_VN_ITEM_SOURCE_REF_RE = re.compile(r"^vn_asset_item:(\d+)$")
_TRUSTED_GENERATED_FILE_KEYS = {
    "source_feature",
    "generated_file_id",
    "filename",
    "mime_type",
    "source_ref",
}
_STRUCTURAL_VN_KEYS = {
    "vn_item_id",
    "vn_pack_id",
    "vn_slot_id",
    "vn_slot_key",
    "vn_asset_type",
}


class VNAssetPacksRepositoryProtocol(Protocol):
    """Repository methods needed to verify VN asset provenance."""

    def get_item(self, item_id: int) -> Mapping[str, Any] | None:
        """Return a VN asset item by ID."""

    def get_slot(self, slot_id: int) -> Mapping[str, Any] | None:
        """Return a VN asset slot by ID."""

    def get_pack(self, pack_id: int) -> Mapping[str, Any] | None:
        """Return a VN asset pack by ID."""


def build_vn_visual_identity_source_context(
    *,
    user_id: int | str,
    vn_repository: VNAssetPacksRepositoryProtocol,
    generated_file_record: Mapping[str, Any],
    requested_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve trusted Visual Identity source context for a VN generated file."""
    requested = canonicalize_source_context(requested_context or {})
    source_feature = str(generated_file_record.get("source_feature") or VN_SOURCE_FEATURE).strip()
    if source_feature.lower() != VN_SOURCE_FEATURE:
        _raise_context_mismatch()

    source_ref = _optional_string(generated_file_record.get("source_ref"))
    requested_item_id = _coerce_optional_positive_int(requested.get("vn_item_id"))
    derived_item_id = _parse_vn_item_id_from_source_ref(source_ref)
    vn_item_id = requested_item_id if requested_item_id is not None else derived_item_id

    if vn_item_id is not None and source_ref != vn_asset_source_ref(vn_item_id):
        _raise_context_mismatch()

    item: Mapping[str, Any] | None = None
    slot: Mapping[str, Any] | None = None
    pack: Mapping[str, Any] | None = None
    if vn_item_id is not None:
        item, slot, pack = _load_verified_item_context(
            user_id=user_id,
            vn_repository=vn_repository,
            item_id=vn_item_id,
        )
    elif any(key in requested for key in _STRUCTURAL_VN_KEYS):
        _raise_context_mismatch()

    context = {
        key: value
        for key, value in requested.items()
        if key not in _TRUSTED_GENERATED_FILE_KEYS and key not in _STRUCTURAL_VN_KEYS
    }
    context.update(
        {
            "source_feature": source_feature.lower(),
            "generated_file_id": _coerce_optional_positive_int(generated_file_record.get("id")),
            "filename": _filename_from_generated_file(generated_file_record),
            "mime_type": _optional_string(generated_file_record.get("mime_type")),
            "source_ref": source_ref,
        }
    )

    if vn_item_id is not None:
        context["vn_item_id"] = vn_item_id
        _copy_verified_structural_hint(
            context,
            requested=requested,
            key="vn_pack_id",
            expected=_coerce_required_positive_int(pack.get("id") if pack is not None else None),
        )
        _copy_verified_structural_hint(
            context,
            requested=requested,
            key="vn_slot_id",
            expected=_coerce_required_positive_int(slot.get("id") if slot is not None else None),
        )
        _copy_verified_structural_hint(
            context,
            requested=requested,
            key="vn_slot_key",
            expected=_optional_string(slot.get("slot_key") if slot is not None else None),
        )
        _copy_verified_structural_hint(
            context,
            requested=requested,
            key="vn_asset_type",
            expected=_optional_string(slot.get("asset_type") if slot is not None else None),
        )

    return canonicalize_source_context(context)


def _load_verified_item_context(
    *,
    user_id: int | str,
    vn_repository: VNAssetPacksRepositoryProtocol,
    item_id: int,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    item = vn_repository.get_item(item_id)
    if item is None:
        _raise_context_mismatch()

    item_pack_id = _coerce_required_positive_int(item.get("pack_id"))
    item_slot_id = _coerce_required_positive_int(item.get("slot_id"))
    slot = vn_repository.get_slot(item_slot_id)
    pack = vn_repository.get_pack(item_pack_id)
    if slot is None or pack is None:
        _raise_context_mismatch()

    slot_pack_id = _coerce_required_positive_int(slot.get("pack_id"))
    pack_id = _coerce_required_positive_int(pack.get("id"))
    if slot_pack_id != item_pack_id or pack_id != item_pack_id:
        _raise_context_mismatch()

    pack_owner_user_id = _coerce_required_positive_int(pack.get("owner_user_id"))
    if pack_owner_user_id != _coerce_required_positive_int(user_id):
        _raise_context_mismatch()
    return item, slot, pack


def _copy_verified_structural_hint(
    context: dict[str, Any],
    *,
    requested: Mapping[str, Any],
    key: str,
    expected: Any,
) -> None:
    if key not in requested:
        return

    requested_value = requested[key]
    if isinstance(expected, int):
        requested_value = _coerce_optional_positive_int(requested_value)
    else:
        requested_value = _optional_string(requested_value)
    if requested_value != expected:
        _raise_context_mismatch()
    context[key] = expected


def _parse_vn_item_id_from_source_ref(source_ref: str | None) -> int | None:
    if not source_ref:
        return None
    match = _VN_ITEM_SOURCE_REF_RE.fullmatch(source_ref)
    if match is None:
        return None
    return _coerce_required_positive_int(match.group(1))


def _filename_from_generated_file(generated_file_record: Mapping[str, Any]) -> str | None:
    for key in ("original_filename", "filename"):
        filename = _optional_string(generated_file_record.get(key))
        if filename:
            return filename
    return None


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _coerce_optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        _raise_context_mismatch()
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        _raise_context_mismatch()
    if normalized < 1:
        _raise_context_mismatch()
    return normalized


def _coerce_required_positive_int(value: Any) -> int:
    normalized = _coerce_optional_positive_int(value)
    if normalized is None:
        _raise_context_mismatch()
    return normalized


def _raise_context_mismatch() -> None:
    raise ValueError(VN_GENERATED_FILE_CONTEXT_MISMATCH)


__all__ = [
    "VN_GENERATED_FILE_CONTEXT_MISMATCH",
    "VN_SOURCE_FEATURE",
    "VNAssetPacksRepositoryProtocol",
    "build_vn_visual_identity_source_context",
]
