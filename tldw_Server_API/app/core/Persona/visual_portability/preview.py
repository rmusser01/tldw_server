"""Persona visual pack import preview validator."""

from __future__ import annotations

import json
import re
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.Persona.visual_import_preview_validators import (
    preview_renderer_import,
)
from tldw_Server_API.app.core.Persona.visuals import (
    PersonaVisualManifestError,
    validate_visual_manifest,
)

from .archive import normalize_member_name, validate_archive_members
from .constants import (
    ASSET_BYTES_STATUS_MISSING,
    ASSET_BYTES_STATUS_PRESENT,
    CHECKSUMS_PATH,
    MANIFEST_PATH,
    PERSONA_VISUAL_PACK_SCHEMA_VERSION,
    TRUST_MODE_TRUSTED_RESTORE,
    TRUST_MODE_UNTRUSTED_IMPORT,
)
from .fingerprints import canonical_payload_fingerprint, sha256_file, sha256_stream

_INTEGER_TEXT_RE = re.compile(r"^[+-]?[0-9]+$")


class PersonaVisualPackImportPreviewer:
    """Validate a persona visual pack archive and produce a review plan."""

    def create_preview(
        self,
        *,
        archive_path: Path,
        owner_user_id: str,
        target_persona_id: str | None = None,
        target_packs: Sequence[Mapping[str, Any]] | None = None,
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
            if manifest.get("schema_version") != PERSONA_VISUAL_PACK_SCHEMA_VERSION:
                raise ValueError("unsupported_persona_visual_pack_schema")

            checksums = _read_required_json(archive, members, CHECKSUMS_PATH)
            _require_mapping(checksums, CHECKSUMS_PATH)
            self._progress(progress, "validating_checksums", {"member_count": len(members)})
            _validate_checksums(archive, members, checksums)

            pack_payload = _read_required_json(archive, members, "metadata/pack.json")
            assets_payload = _read_required_json(archive, members, "metadata/assets.json")
            pack = _section_record(pack_payload, key="pack", path="metadata/pack.json")
            assets = _section_list(assets_payload, key="assets", path="metadata/assets.json")

            self._progress(progress, "validating_assets", {"asset_count": len(assets)})
            asset_summary = _validate_assets(archive, members, assets=assets)
            visual_manifest = pack.get("visual_manifest")
            if not isinstance(visual_manifest, dict):
                raise ValueError("malformed_metadata: metadata/pack.json")
            renderer_import_preview: dict[str, Any] | None = None
            resolved_required_states: Mapping[str, str] = {}
            if _uses_renderer_import_preview(visual_manifest):
                renderer_import_preview = preview_renderer_import(
                    manifest=visual_manifest,
                    assets=assets,
                ).to_dict()
            else:
                available_asset_ids = {
                    str(asset["source_asset_id"])
                    for asset in assets
                    if asset.get("source_asset_id") not in (None, "")
                }
                available_dimensions = {
                    str(asset["source_asset_id"]): (int(asset["width"]), int(asset["height"]))
                    for asset in assets
                    if asset.get("source_asset_id") not in (None, "")
                    and asset.get("width") is not None
                    and asset.get("height") is not None
                }
                try:
                    manifest_validation = validate_visual_manifest(
                        visual_manifest,
                        available_asset_ids=available_asset_ids,
                        available_asset_dimensions=available_dimensions,
                        require_activatable=False,
                    )
                except PersonaVisualManifestError as exc:
                    raise ValueError("malformed_visual_manifest") from exc
                resolved_required_states = manifest_validation.resolved_required_states

        validation_warnings = _validation_warnings(assets)
        source_persona_id = str(pack.get("source_persona_id") or "")
        conflicts = _target_pack_conflicts(pack=pack, target_packs=target_packs)
        replaceable_pack_ids = _replaceable_pack_ids(conflicts)
        target_modes = ["create_new"]
        if replaceable_pack_ids:
            target_modes.append("replace_draft")
        required_choices = _required_choices(
            source_persona_id=source_persona_id,
            target_persona_id=target_persona_id,
            conflicts=conflicts,
            replaceable_pack_ids=replaceable_pack_ids,
        )
        preview_fingerprint = _preview_fingerprint(
            manifest=manifest,
            pack=pack,
            assets=assets,
        )
        proposed_plan = {
            "target_mode": "create_new",
            "target_modes": target_modes,
            "trust_modes": [
                TRUST_MODE_TRUSTED_RESTORE,
                TRUST_MODE_UNTRUSTED_IMPORT,
            ],
            "default_trust_mode": TRUST_MODE_UNTRUSTED_IMPORT,
            "default_target_mode": "create_new",
            "review_before_commit": True,
            "default_target_persona_id": target_persona_id or source_persona_id or None,
            "missing_asset_policy": "import_metadata_only_until_bytes_supplied",
            "replaceable_pack_ids": replaceable_pack_ids,
            "update_identity_rules": {
                "assets": [
                    "source_asset_id",
                    "asset_role+checksum_sha256",
                ],
                "manifest": "state_and_animation_ids",
            },
        }
        preview_status = "completed"
        if renderer_import_preview is not None:
            proposed_plan["renderer_import_preview"] = renderer_import_preview
            proposed_plan["commit_eligible"] = bool(renderer_import_preview.get("can_commit"))
            proposed_plan["activation_eligible"] = bool(
                renderer_import_preview.get("activation_eligible")
            )
            proposed_plan["commit_blockers"] = list(renderer_import_preview.get("blockers") or [])
            if not proposed_plan["commit_eligible"]:
                preview_status = "blocked"
        quota_estimate = {
            "asset_bytes": asset_summary["present_asset_bytes"],
            "present_asset_items": asset_summary["present_asset_items"],
            "missing_asset_items": asset_summary["missing_asset_items"],
        }
        bundle_summary = _bundle_summary(
            manifest=manifest,
            pack=pack,
            assets=assets,
            asset_summary=asset_summary,
            resolved_required_states=resolved_required_states,
        )
        self._progress(progress, "completed", {"archive_sha256": archive_sha256})
        return {
            "status": preview_status,
            "archive_sha256": archive_sha256,
            "canonical_payload_fingerprint": preview_fingerprint,
            "schema_version": PERSONA_VISUAL_PACK_SCHEMA_VERSION,
            "bundle_summary": bundle_summary,
            "validation_warnings": validation_warnings,
            "conflicts": conflicts,
            "proposed_plan": proposed_plan,
            "quota_estimate": quota_estimate,
            "required_choices": required_choices,
            "target_warnings": _target_warnings(
                source_persona_id=source_persona_id,
                target_persona_id=target_persona_id,
            ),
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


def _uses_renderer_import_preview(visual_manifest: Mapping[str, Any]) -> bool:
    """Return whether archive preview should use renderer capability diagnostics."""

    manifest_version = _coerce_manifest_version(visual_manifest.get("manifest_version"))
    return manifest_version is not None and manifest_version != 1


def _coerce_manifest_version(value: Any) -> int | None:
    """Normalize JSON manifest version values used for preview-path routing."""

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        stripped = value.strip()
        return int(stripped) if _INTEGER_TEXT_RE.fullmatch(stripped) else None
    return None


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


def _validate_assets(
    archive: zipfile.ZipFile,
    members: Mapping[str, zipfile.ZipInfo],
    *,
    assets: list[dict[str, Any]],
) -> dict[str, int]:
    present_asset_items = 0
    missing_asset_items = 0
    present_asset_bytes = 0

    for asset in assets:
        if not asset.get("source_asset_id"):
            raise ValueError("malformed_metadata: metadata/assets.json")
        status = str(asset.get("asset_bytes_status") or "").strip()
        asset_path = asset.get("asset_path")
        if status == ASSET_BYTES_STATUS_PRESENT:
            if not asset_path:
                raise ValueError("missing_asset_file")
            normalized_path = normalize_member_name(str(asset_path))
            if normalized_path not in members:
                raise ValueError(f"missing_asset_file: {normalized_path}")
            asset_info = members[normalized_path]
            asset_sha256 = asset.get("asset_sha256")
            if asset_sha256:
                with archive.open(asset_info) as stream:
                    if sha256_stream(stream) != str(asset_sha256):
                        raise ValueError(f"asset_checksum_mismatch: {normalized_path}")
            present_asset_items += 1
            present_asset_bytes += int(asset_info.file_size)
        else:
            asset["asset_bytes_status"] = ASSET_BYTES_STATUS_MISSING
            asset.pop("asset_path", None)
            missing_asset_items += 1

    return {
        "present_asset_items": present_asset_items,
        "missing_asset_items": missing_asset_items,
        "present_asset_bytes": present_asset_bytes,
    }


def _bundle_summary(
    *,
    manifest: Mapping[str, Any],
    pack: Mapping[str, Any],
    assets: list[dict[str, Any]],
    asset_summary: Mapping[str, int],
    resolved_required_states: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "pack_title": pack.get("title") or manifest.get("pack_title"),
        "renderer_type": pack.get("renderer_type") or manifest.get("renderer_type"),
        "source_persona_id": pack.get("source_persona_id"),
        "asset_count": len(assets),
        "assets_with_bytes": asset_summary["present_asset_items"],
        "missing_asset_items": asset_summary["missing_asset_items"],
        "state_count": len((pack.get("visual_manifest") or {}).get("states", {})),
        "animation_count": len((pack.get("visual_manifest") or {}).get("animations", {})),
        "resolved_required_states": dict(resolved_required_states),
        "assets": [
            {
                "source_asset_id": asset.get("source_asset_id"),
                "asset_role": asset.get("asset_role"),
                "asset_bytes_status": asset.get("asset_bytes_status"),
                "mime_type": asset.get("mime_type"),
                "width": asset.get("width"),
                "height": asset.get("height"),
            }
            for asset in assets
        ],
    }


def _validation_warnings(assets: list[Mapping[str, Any]]) -> list[str]:
    warnings: list[str] = []
    for asset in assets:
        if asset.get("asset_bytes_status") == ASSET_BYTES_STATUS_MISSING:
            warnings.append(
                "missing_asset_bytes:"
                f"{asset.get('asset_role')}:{asset.get('source_asset_id')}"
            )
    return warnings


_REPLACEABLE_TARGET_PACK_STATUSES = frozenset({"draft", "review", "failed"})


def _target_pack_conflicts(
    *,
    pack: Mapping[str, Any],
    target_packs: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Return title-match conflicts against the current target persona packs."""
    incoming_title = str(pack.get("title") or "").strip()
    if not incoming_title:
        return []

    conflicts: list[dict[str, Any]] = []
    for target_pack in target_packs or []:
        target_title = str(target_pack.get("title") or "").strip()
        if target_title.casefold() != incoming_title.casefold():
            continue
        pack_id = str(target_pack.get("id") or "").strip()
        if not pack_id:
            continue
        pack_status = str(target_pack.get("status") or "").strip() or "unknown"
        allowed_choices = ["create_new"]
        if pack_status in _REPLACEABLE_TARGET_PACK_STATUSES:
            allowed_choices.append("replace_draft")
        conflicts.append(
            {
                "conflict_id": f"target_pack_title_match:{pack_id}",
                "type": "target_pack_title_match",
                "severity": "warning",
                "message": (
                    f"Target persona already has a {pack_status} visual pack "
                    f"named {incoming_title}."
                ),
                "pack_id": pack_id,
                "pack_title": target_title,
                "pack_status": pack_status,
                "allowed_choices": allowed_choices,
            }
        )

    return sorted(conflicts, key=lambda conflict: str(conflict["conflict_id"]))


def _replaceable_pack_ids(conflicts: Sequence[Mapping[str, Any]]) -> list[str]:
    """List conflict pack ids that can be selected for replace-draft import."""
    replaceable: list[str] = []
    for conflict in conflicts:
        allowed = conflict.get("allowed_choices")
        if not isinstance(allowed, (list, tuple, set)) or "replace_draft" not in allowed:
            continue
        pack_id = str(conflict.get("pack_id") or "").strip()
        if pack_id:
            replaceable.append(pack_id)
    return replaceable


def _required_choices(
    *,
    source_persona_id: str,
    target_persona_id: str | None,
    conflicts: Sequence[Mapping[str, Any]],
    replaceable_pack_ids: Sequence[str],
) -> list[dict[str, Any]]:
    choices: list[dict[str, Any]] = []
    if target_persona_id:
        if conflicts:
            choices.append(
                {
                    "choice_id": "import_target_mode",
                    "reason": "target_pack_conflicts",
                    "default_target_mode": "create_new",
                    "allowed_target_modes": (
                        ["create_new", "replace_draft"]
                        if replaceable_pack_ids
                        else ["create_new"]
                    ),
                    "replaceable_pack_ids": list(replaceable_pack_ids),
                }
            )
        return choices
    choices.append(
        {
            "choice_id": "target_persona",
            "resource": "persona",
            "source_persona_id": source_persona_id or None,
            "allowed_actions": ["import_to_source_persona", "select_existing_persona"],
            "default_action": "import_to_source_persona",
            "required": True,
        }
    )
    return choices


def _target_warnings(
    *,
    source_persona_id: str,
    target_persona_id: str | None,
) -> list[str]:
    if target_persona_id and source_persona_id and target_persona_id != source_persona_id:
        return ["target_persona_differs_from_source"]
    return []


def _preview_fingerprint(
    *,
    manifest: Mapping[str, Any],
    pack: Mapping[str, Any],
    assets: list[dict[str, Any]],
) -> str:
    manifest_fingerprint = manifest.get("canonical_payload_fingerprint")
    if isinstance(manifest_fingerprint, str) and len(manifest_fingerprint) == 64:
        return manifest_fingerprint
    return canonical_payload_fingerprint({"pack": pack, "assets": assets})
