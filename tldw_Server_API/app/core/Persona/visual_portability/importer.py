"""Persona visual pack import commit executor."""

from __future__ import annotations

import zipfile
from copy import deepcopy
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
from tldw_Server_API.app.core.Persona.visual_service import PersonaVisualService
from tldw_Server_API.app.core.Persona.visuals import validate_visual_manifest


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
        progress: Any | None = None,
    ) -> dict[str, Any]:
        if target_mode != "create_new":
            raise ValueError("unsupported_import_target_mode")
        if trust_mode not in {TRUST_MODE_TRUSTED_RESTORE, TRUST_MODE_UNTRUSTED_IMPORT}:
            raise ValueError("unsupported_import_trust_mode")

        preview = self.repo.get_import_preview(str(preview_id), owner_user_id=self.user_id)
        if preview is None:
            raise ValueError("import_preview_not_found")
        archive_path = Path(str(preview.get("archive_path") or ""))
        self._validate_preview_ready(preview=preview, archive_path=archive_path)
        self._progress(progress, "revalidating_preview", {"preview_id": str(preview_id)})
        revalidated = PersonaVisualPackImportPreviewer().create_preview(
            archive_path=archive_path,
            owner_user_id=self.user_id,
            target_persona_id=target_persona_id,
        )
        expected_fingerprint = str(preview.get("canonical_payload_fingerprint") or "")
        if expected_fingerprint and revalidated["canonical_payload_fingerprint"] != expected_fingerprint:
            raise ValueError("import_archive_fingerprint_changed")

        with zipfile.ZipFile(archive_path, "r") as archive:
            members = _archive_members_by_normalized_name(archive)
            pack_payload = _read_required_json(archive, members, "metadata/pack.json")
            assets_payload = _read_required_json(archive, members, "metadata/assets.json")
            pack = _section_record(pack_payload, key="pack", path="metadata/pack.json")
            assets = _section_list(assets_payload, key="assets", path="metadata/assets.json")

            self._progress(progress, "creating_pack", {"target_persona_id": target_persona_id})
            created_pack = self.db.create_persona_visual_pack(
                persona_id=target_persona_id,
                user_id=self.user_id,
                title=str(pack.get("title") or "Imported visual pack"),
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
        remapped_manifest = _remap_visual_manifest_assets(visual_manifest, id_maps["assets"])
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
        self._progress(
            progress,
            "completed",
            {"pack_id": str(created_pack["id"]), "asset_count": len(imported_assets)},
        )
        return {
            "status": "imported",
            "preview_id": str(preview_id),
            "pack_id": str(created_pack["id"]),
            "pack": updated_pack,
            "id_maps": id_maps,
            "created_records": {
                "pack_id": str(created_pack["id"]),
                "asset_ids": [str(asset["id"]) for asset in imported_assets],
            },
        }

    def _validate_preview_ready(self, *, preview: dict[str, Any], archive_path: Path) -> None:
        if str(preview.get("status") or "") != "completed":
            raise ValueError("import_preview_not_completed")
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


def _remap_visual_manifest_assets(
    manifest: dict[str, Any],
    asset_id_map: dict[str, str],
) -> dict[str, Any]:
    remapped = deepcopy(manifest)
    animations = remapped.get("animations")
    if not isinstance(animations, dict):
        return remapped
    for animation in animations.values():
        if not isinstance(animation, dict):
            continue
        frames = animation.get("frames")
        if isinstance(frames, list):
            for frame in frames:
                if not isinstance(frame, dict):
                    continue
                asset_id = str(frame.get("asset_id") or "")
                if asset_id in asset_id_map:
                    frame["asset_id"] = asset_id_map[asset_id]
        asset_ids = animation.get("asset_ids")
        if isinstance(asset_ids, list):
            animation["asset_ids"] = [
                asset_id_map.get(str(asset_id), asset_id)
                for asset_id in asset_ids
            ]
        preview_asset_id = str(animation.get("preview_asset_id") or "")
        if preview_asset_id in asset_id_map:
            animation["preview_asset_id"] = asset_id_map[preview_asset_id]
    return remapped


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
