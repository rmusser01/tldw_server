"""Async VN pack import preview validator and planner."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.DB_Management.VNAssetPacks_DB import VNAssetPacksRepository
from tldw_Server_API.app.core.VN_Assets.portability.archive import (
    normalize_member_name,
    validate_archive_members,
)
from tldw_Server_API.app.core.VN_Assets.portability.conflicts import (
    build_update_existing_plan,
    detect_conflicts,
)
from tldw_Server_API.app.core.VN_Assets.portability.constants import (
    ASSET_BYTES_STATUS_MISSING,
    ASSET_BYTES_STATUS_PRESENT,
    CHECKSUMS_PATH,
    MANIFEST_PATH,
    VNPACK_SCHEMA_VERSION,
)
from tldw_Server_API.app.core.VN_Assets.portability.fingerprints import (
    canonical_payload_fingerprint,
    sha256_file,
    sha256_stream,
)


class VNPackImportPreviewer:
    """Validate a VN pack archive and produce an immutable preview plan."""

    def __init__(self, *, repo: VNAssetPacksRepository | None = None) -> None:
        self.repo = repo

    async def create_preview(
        self,
        *,
        archive_path: Path,
        owner_user_id: int,
        progress: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        archive_path = Path(archive_path)
        self._progress(progress, "validating_archive", {"archive_path": str(archive_path)})
        validate_archive_members(archive_path)
        archive_sha256 = sha256_file(archive_path)

        with zipfile.ZipFile(archive_path, "r") as archive:
            members = _archive_members_by_normalized_name(archive)
            manifest = _read_required_json(archive, members, MANIFEST_PATH)
            _require_mapping(manifest, MANIFEST_PATH)
            if manifest.get("schema_version") != VNPACK_SCHEMA_VERSION:
                raise ValueError("unsupported_vnpack_schema")

            checksums = _read_required_json(archive, members, CHECKSUMS_PATH)
            _require_mapping(checksums, CHECKSUMS_PATH)
            self._progress(progress, "validating_checksums", {"member_count": len(members)})
            _validate_checksums(archive, members, checksums)

            pack_payload = _read_required_json(archive, members, "metadata/pack.json")
            slots_payload = _read_required_json(archive, members, "metadata/slots.json")
            items_payload = _read_required_json(archive, members, "metadata/items.json")
            character_payload = _read_optional_json(archive, members, "metadata/character.json")

            pack = _section_record(pack_payload, key="pack", path="metadata/pack.json")
            slots = _section_list(slots_payload, key="slots", path="metadata/slots.json")
            items = _section_list(items_payload, key="items", path="metadata/items.json")
            character = (
                _section_record(character_payload, key="character", path="metadata/character.json")
                if character_payload is not None
                else None
            )

            self._progress(progress, "validating_assets", {"item_count": len(items)})
            asset_summary = _validate_items_and_assets(archive, members, slots=slots, items=items)

        bundle_summary = _bundle_summary(
            manifest=manifest,
            pack=pack,
            slots=slots,
            items=items,
            asset_summary=asset_summary,
            character=character,
        )
        validation_warnings = _validation_warnings(items)
        required_choices = _required_choices(manifest=manifest, pack=pack, character=character)
        preview_fingerprint = _preview_fingerprint(
            manifest=manifest,
            pack=pack,
            slots=slots,
            items=items,
            character=character,
        )
        self._progress(progress, "detecting_conflicts", {"fingerprint": preview_fingerprint})
        conflicts = detect_conflicts(
            repo=self.repo,
            owner_user_id=int(owner_user_id),
            manifest=manifest,
            pack=pack,
            slots=slots,
            items=items,
            character=character,
        )
        update_existing_plan = build_update_existing_plan(
            repo=self.repo,
            owner_user_id=int(owner_user_id),
            manifest=manifest,
            pack=pack,
            slots=slots,
            items=items,
        )
        proposed_plan = {
            "target_mode": "create_new",
            "trust_modes": ["trusted_restore", "untrusted_import"],
            "default_trust_mode": "untrusted_import",
            "character_resolution_required": bool(required_choices),
            "missing_asset_policy": "import_hidden_metadata_only",
            "update_identity_rules": {
                "slots": "asset_type+slot_key",
                "items": [
                    "source_item_fingerprint",
                    "slot+asset_checksum",
                    "slot+variant_index_requires_selection",
                ],
            },
            "update_existing": update_existing_plan,
        }
        quota_estimate = {
            "asset_bytes": asset_summary["present_asset_bytes"],
            "present_asset_items": asset_summary["present_asset_items"],
            "missing_asset_items": asset_summary["missing_asset_items"],
        }
        self._progress(progress, "completed", {"archive_sha256": archive_sha256})
        return {
            "status": "completed",
            "archive_sha256": archive_sha256,
            "canonical_payload_fingerprint": preview_fingerprint,
            "schema_version": VNPACK_SCHEMA_VERSION,
            "bundle_summary": bundle_summary,
            "validation_warnings": validation_warnings,
            "conflicts": conflicts,
            "proposed_plan": proposed_plan,
            "quota_estimate": quota_estimate,
            "required_choices": required_choices,
        }

    def _progress(
        self,
        progress: Callable[[str, dict[str, Any]], None] | None,
        stage: str,
        payload: dict[str, Any],
    ) -> None:
        if progress is not None:
            progress(stage, payload)


def _archive_members_by_normalized_name(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        members[normalize_member_name(getattr(info, "orig_filename", info.filename))] = info
    return members


def _read_required_json(
    archive: zipfile.ZipFile,
    members: Mapping[str, zipfile.ZipInfo],
    path: str,
) -> Any:
    if path not in members:
        raise ValueError(f"missing_required_archive_member: {path}")
    return _read_json(archive, members[path], path=path)


def _read_optional_json(
    archive: zipfile.ZipFile,
    members: Mapping[str, zipfile.ZipInfo],
    path: str,
) -> Any:
    if path not in members:
        return None
    return _read_json(archive, members[path], path=path)


def _read_json(archive: zipfile.ZipFile, info: zipfile.ZipInfo, *, path: str) -> Any:
    try:
        return json.loads(archive.read(info).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"malformed_metadata: {path}") from exc


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"malformed_metadata: {path}")
    return value


def _section_record(value: Any, *, key: str, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not isinstance(value.get(key), Mapping):
        raise ValueError(f"malformed_metadata: {path}")
    return dict(value[key])


def _section_list(value: Any, *, key: str, path: str) -> list[dict[str, Any]]:
    if not isinstance(value, Mapping) or not isinstance(value.get(key), list):
        raise ValueError(f"malformed_metadata: {path}")
    records: list[dict[str, Any]] = []
    for item in value[key]:
        if not isinstance(item, Mapping):
            raise ValueError(f"malformed_metadata: {path}")
        records.append(dict(item))
    return records


def _validate_checksums(
    archive: zipfile.ZipFile,
    members: Mapping[str, zipfile.ZipInfo],
    checksums: Mapping[str, Any],
) -> None:
    for path, expected in sorted(checksums.items()):
        normalized_path = normalize_member_name(str(path))
        if normalized_path == CHECKSUMS_PATH:
            continue
        if normalized_path not in members:
            raise ValueError(f"checksum_member_missing: {normalized_path}")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"checksum_malformed: {normalized_path}")
        with archive.open(members[normalized_path]) as stream:
            actual = sha256_stream(stream)
        if actual != expected:
            raise ValueError(f"checksum_mismatch: {normalized_path}")


def _validate_items_and_assets(
    archive: zipfile.ZipFile,
    members: Mapping[str, zipfile.ZipInfo],
    *,
    slots: list[dict[str, Any]],
    items: list[dict[str, Any]],
) -> dict[str, int]:
    slot_ids = {int(slot["source_slot_id"]) for slot in slots if slot.get("source_slot_id") not in (None, "")}
    present_asset_items = 0
    missing_asset_items = 0
    present_asset_bytes = 0

    for item in items:
        if item.get("source_slot_id") not in (None, "") and int(item["source_slot_id"]) not in slot_ids:
            raise ValueError("malformed_metadata: metadata/items.json")
        status = str(item.get("asset_bytes_status") or "").strip()
        asset_path = item.get("asset_path")
        if status == ASSET_BYTES_STATUS_PRESENT:
            if not asset_path:
                raise ValueError("missing_asset_file")
            normalized_path = normalize_member_name(str(asset_path))
            if normalized_path not in members:
                raise ValueError(f"missing_asset_file: {normalized_path}")
            asset_sha256 = item.get("asset_sha256")
            asset_info = members[normalized_path]
            if asset_sha256:
                with archive.open(asset_info) as stream:
                    if sha256_stream(stream) != str(asset_sha256):
                        raise ValueError(f"asset_checksum_mismatch: {normalized_path}")
            present_asset_items += 1
            present_asset_bytes += int(asset_info.file_size)
        else:
            item["asset_bytes_status"] = ASSET_BYTES_STATUS_MISSING
            item.pop("asset_path", None)
            missing_asset_items += 1

    required_impacted = _required_slots_impacted_by_missing_bytes(slots=slots, items=items)
    return {
        "present_asset_items": present_asset_items,
        "missing_asset_items": missing_asset_items,
        "present_asset_bytes": present_asset_bytes,
        "required_slots_impacted_by_missing_bytes": required_impacted,
    }


def _bundle_summary(
    *,
    manifest: Mapping[str, Any],
    pack: Mapping[str, Any],
    slots: list[dict[str, Any]],
    items: list[dict[str, Any]],
    asset_summary: Mapping[str, int],
    character: Mapping[str, Any] | None,
) -> dict[str, Any]:
    review_counts: dict[str, int] = {}
    for item in items:
        status = str(item.get("review_status") or "unknown")
        review_counts[status] = review_counts.get(status, 0) + 1
    return {
        "pack_title": pack.get("title") or manifest.get("pack_title"),
        "content_rating": pack.get("content_rating") or manifest.get("content_rating"),
        "slot_count": len(slots),
        "item_count": len(items),
        "review_counts": review_counts,
        "assets_with_bytes": asset_summary["present_asset_items"],
        "missing_asset_items": asset_summary["missing_asset_items"],
        "required_slots_impacted_by_missing_bytes": asset_summary[
            "required_slots_impacted_by_missing_bytes"
        ],
        "include_character": character is not None,
        "include_world_books": bool(manifest.get("include_world_books")),
    }


def _validation_warnings(items: list[Mapping[str, Any]]) -> list[str]:
    warnings: list[str] = []
    for item in items:
        if item.get("asset_bytes_status") == ASSET_BYTES_STATUS_MISSING:
            warnings.append(
                "missing_asset_bytes:"
                f"{item.get('asset_type')}:{item.get('slot_key')}:variant:{int(item.get('variant_index') or 0)}"
            )
    return warnings


def _required_choices(
    *,
    manifest: Mapping[str, Any],
    pack: Mapping[str, Any],
    character: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    if character is not None:
        return [
            {
                "choice_id": "primary_character",
                "resource": "character",
                "source_character_id": pack.get("primary_character_id"),
                "source_character_name": character.get("name"),
                "allowed_actions": ["import_included_character", "link_existing_character", "fail_import"],
                "default_action": "import_included_character",
                "required": True,
            }
        ]
    if manifest.get("include_character"):
        raise ValueError("missing_required_character_payload")
    return [
        {
            "choice_id": "primary_character",
            "resource": "character",
            "source_character_id": pack.get("primary_character_id"),
            "allowed_actions": ["link_existing_character", "fail_import"],
            "default_action": "link_existing_character",
            "required": True,
        }
    ]


def _preview_fingerprint(
    *,
    manifest: Mapping[str, Any],
    pack: Mapping[str, Any],
    slots: list[dict[str, Any]],
    items: list[dict[str, Any]],
    character: Mapping[str, Any] | None,
) -> str:
    manifest_fingerprint = manifest.get("canonical_payload_fingerprint")
    if isinstance(manifest_fingerprint, str) and len(manifest_fingerprint) == 64:
        return manifest_fingerprint
    payload: dict[str, Any] = {"pack": pack, "slots": slots, "items": items}
    if character is not None:
        payload["character"] = character
    return canonical_payload_fingerprint(payload)


def _required_slots_impacted_by_missing_bytes(
    *,
    slots: list[dict[str, Any]],
    items: list[dict[str, Any]],
) -> int:
    by_source_slot_id: dict[int, list[dict[str, Any]]] = {}
    by_slot_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in items:
        if item.get("source_slot_id") not in (None, ""):
            by_source_slot_id.setdefault(int(item["source_slot_id"]), []).append(item)
        by_slot_key.setdefault((str(item.get("asset_type")), str(item.get("slot_key"))), []).append(item)

    impacted = 0
    for slot in slots:
        if not bool(slot.get("required_for_runtime", True)):
            continue
        slot_items = by_source_slot_id.get(int(slot["source_slot_id"])) if slot.get("source_slot_id") not in (None, "") else None
        if slot_items is None:
            slot_items = by_slot_key.get((str(slot.get("asset_type")), str(slot.get("slot_key"))), [])
        if slot_items and all(item.get("asset_bytes_status") == ASSET_BYTES_STATUS_MISSING for item in slot_items):
            impacted += 1
    return impacted
