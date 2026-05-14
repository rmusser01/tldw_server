"""Persona visual pack import commit executor."""

from __future__ import annotations

import contextlib
import json
import zipfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.DB_Management.PersonaVisualPortability_DB import (
    PersonaVisualPortabilityRepository,
)
from tldw_Server_API.app.core.Persona.visual_portability.archive import normalize_member_name
from tldw_Server_API.app.core.Persona.visual_portability.constants import (
    ASSET_BYTES_STATUS_PRESENT,
    TRUST_MODE_TRUSTED_RESTORE,
    TRUST_MODE_UNTRUSTED_IMPORT,
)
from tldw_Server_API.app.core.Persona.visual_portability.fingerprints import sha256_file
from tldw_Server_API.app.core.Persona.visual_portability.preview import (
    PersonaVisualPackImportPreviewer,
    _archive_members_by_normalized_name,
    _read_required_json,
    _section_list,
    _section_record,
)
from tldw_Server_API.app.core.Persona.visual_manifest_assets import remap_visual_manifest_assets
from tldw_Server_API.app.core.Persona.visual_service import PersonaVisualService
from tldw_Server_API.app.core.Persona.visuals import validate_visual_manifest


_REPLACEABLE_IMPORT_TARGET_STATUSES = frozenset({"draft", "review", "failed"})


class PersonaVisualPackImporter:
    """Commit a completed persona visual pack preview into a new draft pack."""

    def __init__(
        self,
        *,
        db: Any,
        repo: PersonaVisualPortabilityRepository,
        user_id: str,
    ) -> None:
        self.db = db
        self.repo = repo
        self.user_id = str(user_id)
        self.service = PersonaVisualService(db)

    def import_preview(
        self,
        *,
        preview_id: str,
        target_persona_id: str,
        trust_mode: str,
        target_mode: str = "create_new",
        target_pack_id: str | None = None,
        title: str | None = None,
        conflict_choice_explicit: bool = False,
        progress: Any | None = None,
    ) -> dict[str, Any]:
        target_mode_value = str(target_mode or "create_new").strip()
        if target_mode_value not in {"create_new", "replace_draft"}:
            raise ValueError("unsupported_import_target_mode")
        if trust_mode not in {TRUST_MODE_TRUSTED_RESTORE, TRUST_MODE_UNTRUSTED_IMPORT}:
            raise ValueError("unsupported_import_trust_mode")

        preview = self.repo.get_import_preview(str(preview_id), owner_user_id=self.user_id)
        if preview is None:
            raise ValueError("import_preview_not_found")
        if target_mode_value != "replace_draft" and target_pack_id:
            raise ValueError("target_pack_id_requires_replace_draft")

        archive_path = Path(str(preview.get("archive_path") or ""))
        self._validate_preview_ready(preview=preview, archive_path=archive_path)
        self._progress(progress, "revalidating_preview", {"preview_id": str(preview_id)})
        target_packs = self.db.list_persona_visual_packs(
            persona_id=target_persona_id,
            user_id=self.user_id,
        )
        revalidated = PersonaVisualPackImportPreviewer().create_preview(
            archive_path=archive_path,
            owner_user_id=self.user_id,
            target_persona_id=target_persona_id,
            target_packs=target_packs,
        )
        expected_fingerprint = str(preview.get("canonical_payload_fingerprint") or "")
        if expected_fingerprint and revalidated["canonical_payload_fingerprint"] != expected_fingerprint:
            raise ValueError("import_archive_fingerprint_changed")
        _validate_preview_commit_allowed(revalidated)
        revalidated_conflicts = revalidated.get("conflicts")
        current_conflicts = revalidated_conflicts if isinstance(revalidated_conflicts, list) else []
        if current_conflicts and not conflict_choice_explicit:
            raise ValueError("import_conflict_choice_required")
        replacement_pack = None
        if target_mode_value == "replace_draft":
            replacement_pack = self._replacement_pack_or_error(
                target_persona_id=target_persona_id,
                target_pack_id=target_pack_id,
                preview_conflicts=current_conflicts,
            )

        with zipfile.ZipFile(archive_path, "r") as archive:
            members = _archive_members_by_normalized_name(archive)
            pack_payload = _read_required_json(archive, members, "metadata/pack.json")
            assets_payload = _read_required_json(archive, members, "metadata/assets.json")
            pack = _section_record(pack_payload, key="pack", path="metadata/pack.json")
            assets = _section_list(assets_payload, key="assets", path="metadata/assets.json")

            self._progress(progress, "creating_pack", {"target_persona_id": target_persona_id})
            pack_title = _import_pack_title(title=title, pack=pack)
            created_pack = self.db.create_persona_visual_pack(
                persona_id=target_persona_id,
                user_id=self.user_id,
                title=pack_title,
                renderer_type=str(pack.get("renderer_type") or "sprite_frames"),
                manifest={
                    "manifest_version": 1,
                    "renderer_type": str(pack.get("renderer_type") or "sprite_frames"),
                    "states": {},
                    "animations": {},
                },
                provenance="imported",
            )

            id_maps: dict[str, Any] = {"assets": {}, "packs": {}}
            if pack.get("source_pack_id") not in (None, ""):
                id_maps["packs"][str(pack["source_pack_id"])] = str(created_pack["id"])

            imported_assets = []
            for asset in assets:
                if asset.get("asset_bytes_status") != ASSET_BYTES_STATUS_PRESENT:
                    continue
                asset_path = normalize_member_name(str(asset.get("asset_path") or ""))
                if asset_path not in members:
                    raise ValueError(f"missing_asset_file: {asset_path}")
                content = archive.read(members[asset_path])
                imported = self.service.create_asset_from_upload(
                    persona_id=target_persona_id,
                    user_id=self.user_id,
                    pack_id=str(created_pack["id"]),
                    content=content,
                    mime_type=str(asset.get("mime_type") or "application/octet-stream"),
                    original_filename=asset.get("original_filename"),
                    asset_role=str(asset.get("asset_role") or "frame"),
                    provenance="imported",
                )
                source_asset_id = str(asset.get("source_asset_id") or "")
                if source_asset_id:
                    id_maps["assets"][source_asset_id] = str(imported["id"])
                imported_assets.append(imported)

        visual_manifest = pack.get("visual_manifest") if isinstance(pack.get("visual_manifest"), dict) else {}
        remapped_manifest = remap_visual_manifest_assets(visual_manifest, id_maps["assets"])
        asset_ids = {str(asset["id"]) for asset in imported_assets}
        asset_dimensions = {
            str(asset["id"]): (int(asset["width"]), int(asset["height"]))
            for asset in imported_assets
            if asset.get("width") is not None and asset.get("height") is not None
        }
        validation = validate_visual_manifest(
            remapped_manifest,
            available_asset_ids=asset_ids,
            available_asset_dimensions=asset_dimensions,
            require_activatable=False,
        )
        updated_pack = self.db.update_persona_visual_pack_manifest(
            pack_id=str(created_pack["id"]),
            persona_id=target_persona_id,
            user_id=self.user_id,
            manifest=validation.manifest,
            expected_version=int(created_pack["version"]),
        )
        replaced_pack_id = None
        if replacement_pack is not None:
            replaced_pack_id = str(replacement_pack["id"])
            self._progress(
                progress,
                "replacing_draft",
                {"target_pack_id": replaced_pack_id, "pack_id": str(created_pack["id"])},
            )
            try:
                replaced = self.db.soft_delete_persona_visual_pack_with_assets(
                    pack_id=replaced_pack_id,
                    persona_id=target_persona_id,
                    user_id=self.user_id,
                    expected_version=int(replacement_pack["version"]),
                    allowed_statuses=_REPLACEABLE_IMPORT_TARGET_STATUSES,
                )
            except Exception:
                self._cleanup_created_pack(
                    pack_id=str(created_pack["id"]),
                    target_persona_id=target_persona_id,
                )
                raise
            if not replaced:
                self._cleanup_created_pack(
                    pack_id=str(created_pack["id"]),
                    target_persona_id=target_persona_id,
                )
                raise ValueError("import_target_pack_not_replaceable")
        self._progress(
            progress,
            "completed",
            {"pack_id": str(created_pack["id"]), "asset_count": len(imported_assets)},
        )
        return {
            "status": "imported",
            "preview_id": str(preview_id),
            "pack_id": str(created_pack["id"]),
            "target_mode": target_mode_value,
            "replaced_pack_id": replaced_pack_id,
            "pack": updated_pack,
            "id_maps": id_maps,
            "created_records": {
                "pack_id": str(created_pack["id"]),
                "asset_ids": [str(asset["id"]) for asset in imported_assets],
                "replaced_pack_id": replaced_pack_id,
            },
        }

    def _replacement_pack_or_error(
        self,
        *,
        target_persona_id: str,
        target_pack_id: str | None,
        preview_conflicts: Any,
    ) -> dict[str, Any]:
        """Validate that the selected target pack is currently replaceable."""
        target_pack_id_value = str(target_pack_id or "").strip()
        if not target_pack_id_value:
            raise ValueError("target_pack_id_required")
        if target_pack_id_value not in _replaceable_pack_ids_from_conflicts(preview_conflicts):
            raise ValueError("import_target_pack_not_replaceable")
        replacement_pack = self.db.get_persona_visual_pack(
            pack_id=target_pack_id_value,
            persona_id=target_persona_id,
            user_id=self.user_id,
        )
        if replacement_pack is None:
            raise ValueError("import_target_pack_not_found")
        if str(replacement_pack.get("status") or "") not in _REPLACEABLE_IMPORT_TARGET_STATUSES:
            raise ValueError("import_target_pack_not_replaceable")
        return replacement_pack

    def _cleanup_created_pack(self, *, pack_id: str, target_persona_id: str) -> None:
        """Best-effort cleanup for a newly imported pack after commit failure."""
        with contextlib.suppress(Exception):
            self.db.soft_delete_persona_visual_pack_with_assets(
                pack_id=pack_id,
                persona_id=target_persona_id,
                user_id=self.user_id,
            )

    def _validate_preview_ready(self, *, preview: dict[str, Any], archive_path: Path) -> None:
        _validate_preview_commit_allowed(preview)
        if not archive_path.is_file():
            raise ValueError("import_archive_not_found")
        expires_at = _parse_datetime(preview.get("expires_at"))
        if expires_at is not None and expires_at <= datetime.now(timezone.utc):
            raise ValueError("import_preview_expired")
        expected_sha = str(preview.get("archive_sha256") or "")
        if expected_sha and sha256_file(archive_path) != expected_sha:
            raise ValueError("import_archive_checksum_changed")

    def _progress(self, progress: Any | None, stage: str, payload: dict[str, Any]) -> None:
        if progress is not None:
            progress(stage, payload)


def _import_preview_json_field(row: Mapping[str, Any], key: str, default: Any) -> Any:
    """Read a JSON-backed import preview column with a typed fallback."""
    value = row.get(key)
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def _import_preview_proposed_plan(preview: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return proposed-plan metadata from either a row or fresh preview payload."""
    proposed_plan = preview.get("proposed_plan")
    if isinstance(proposed_plan, Mapping):
        return proposed_plan
    stored_plan = _import_preview_json_field(preview, "proposed_plan_json", {})
    if isinstance(stored_plan, Mapping):
        return stored_plan
    return {}


def _validate_preview_commit_allowed(preview: Mapping[str, Any]) -> None:
    """Reject previews that are not currently eligible for import commit."""
    proposed_plan = _import_preview_proposed_plan(preview)
    status_value = str(preview.get("status") or "").strip()
    if status_value == "blocked" or proposed_plan.get("commit_eligible") is False:
        raise ValueError("import_preview_not_commit_eligible")
    if status_value != "completed":
        raise ValueError("import_preview_not_completed")


def _replaceable_pack_ids_from_conflicts(conflicts: Any) -> set[str]:
    """Extract target pack ids that current conflict metadata allows replacing."""
    if not isinstance(conflicts, list):
        return set()
    replaceable: set[str] = set()
    for conflict in conflicts:
        if not isinstance(conflict, dict):
            continue
        allowed_choices = conflict.get("allowed_choices")
        if not isinstance(allowed_choices, list) or "replace_draft" not in allowed_choices:
            continue
        pack_id = str(conflict.get("pack_id") or "").strip()
        if pack_id:
            replaceable.add(pack_id)
    return replaceable


def _import_pack_title(*, title: str | None, pack: Mapping[str, Any]) -> str:
    """Resolve the imported pack title from the explicit override or archive metadata."""
    title_value = str(title or "").strip()
    if title_value:
        return title_value
    pack_title = str(pack.get("title") or "").strip()
    return pack_title or "Imported visual pack"


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed
