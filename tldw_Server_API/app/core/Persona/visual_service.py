from __future__ import annotations

import hashlib
import io
import uuid
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from loguru import logger
from PIL import Image, UnidentifiedImageError

from tldw_Server_API.app.core.DB_Management.db_path_utils import (
    DatabasePaths,
    normalize_output_storage_filename,
)
from tldw_Server_API.app.core.exceptions import InvalidStoragePathError
from tldw_Server_API.app.core.Persona.companion_behavior import (
    CompanionBehaviorValidationError,
    normalize_companion_behavior,
)
from tldw_Server_API.app.core.Persona.visual_asset_constraints import (
    ALLOWED_VISUAL_MIME_TYPES,
    MAX_VISUAL_IMAGE_DIMENSION,
    MAX_VISUAL_RASTER_FRAMES,
    VISUAL_MIME_EXTENSIONS,
)
from tldw_Server_API.app.core.Persona.visual_manifest_assets import (
    collect_visual_manifest_asset_ids,
    remap_visual_manifest_assets,
)
from tldw_Server_API.app.core.Persona.visual_portability.fingerprints import (
    build_persona_visual_pack_fingerprint,
)
from tldw_Server_API.app.core.Persona.visual_renderer_capabilities import (
    get_persona_visual_renderer_capability,
)
from tldw_Server_API.app.core.Persona.visuals import (
    PersonaVisualManifestError,
    resolved_visual_state_ids,
    validate_sprite_static_coverage,
    validate_visual_manifest,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


MAX_VISUAL_UPLOAD_BYTES = 10_485_760
VISUAL_STORAGE_PREFIX = "persona_visuals"

_MIME_EXTENSIONS = VISUAL_MIME_EXTENSIONS
_IMAGE_VALIDATION_ERRORS = (
    OSError,
    ValueError,
    UnidentifiedImageError,
)
_REVIEWABLE_CANDIDATE_STATUSES = frozenset({"review"})


class PersonaVisualServiceError(Exception):
    """Service-level persona visual failure with stable API-facing codes."""

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


class PersonaVisualService:
    """Core service for persona visual-pack asset storage and activation."""

    def __init__(self, db: CharactersRAGDB) -> None:
        self._db = db

    def create_asset_from_upload(
        self,
        *,
        persona_id: str,
        user_id: str,
        pack_id: str,
        content: bytes,
        mime_type: str,
        original_filename: str | None,
        asset_role: str = "frame",
        provenance: str = "uploaded",
    ) -> dict[str, Any]:
        normalized_mime = self._normalize_mime_type(mime_type)
        content_bytes = bytes(content or b"")
        if len(content_bytes) > MAX_VISUAL_UPLOAD_BYTES:
            raise PersonaVisualServiceError(
                "upload_too_large",
                f"Persona visual upload exceeds {MAX_VISUAL_UPLOAD_BYTES} bytes.",
            )

        width, height = self._validate_image_bytes(
            content_bytes,
            mime_type=normalized_mime,
        )
        pack = self._db.get_persona_visual_pack(
            pack_id=pack_id,
            persona_id=persona_id,
            user_id=user_id,
        )
        if not pack:
            raise PersonaVisualServiceError(
                "pack_not_found",
                "Persona visual pack not found for user.",
                details={"pack_id": pack_id},
            )

        asset_id = uuid.uuid4().hex
        extension = _MIME_EXTENSIONS[normalized_mime]
        storage_key, storage_path = self._build_storage_target(
            user_id=user_id,
            persona_id=persona_id,
            pack_id=pack_id,
            asset_id=asset_id,
            extension=extension,
        )
        checksum = hashlib.sha256(content_bytes).hexdigest()

        storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage_path.write_bytes(content_bytes)
        try:
            asset = self._db.create_persona_visual_asset(
                asset_id=asset_id,
                pack_id=pack_id,
                persona_id=persona_id,
                user_id=user_id,
                asset_role=asset_role,
                storage_key=storage_key,
                original_filename=original_filename,
                mime_type=normalized_mime,
                byte_size=len(content_bytes),
                checksum_sha256=checksum,
                width=width,
                height=height,
                provenance=provenance,
            )
        except Exception:
            storage_path.unlink(missing_ok=True)
            raise

        asset["storage_path"] = str(storage_path)
        return asset

    def create_generated_asset(
        self,
        *,
        persona_id: str,
        user_id: str,
        pack_id: str,
        content: bytes,
        mime_type: str,
        original_filename: str | None = None,
    ) -> dict[str, Any]:
        return self.create_asset_from_upload(
            persona_id=persona_id,
            user_id=user_id,
            pack_id=pack_id,
            content=content,
            mime_type=mime_type,
            original_filename=original_filename,
            asset_role="generated_candidate",
            provenance="generated",
        )

    def activate_pack(
        self,
        *,
        persona_id: str,
        user_id: str,
        pack_id: str,
        expected_version: int,
        reviewed_fingerprint: str,
    ) -> dict[str, Any]:
        pack = self._db.get_persona_visual_pack(
            pack_id=pack_id,
            persona_id=persona_id,
            user_id=user_id,
        )
        if not pack:
            raise PersonaVisualServiceError(
                "pack_not_found",
                "Persona visual pack not found for user.",
                details={"pack_id": pack_id},
            )
        if int(pack["version"]) != int(expected_version):
            raise PersonaVisualServiceError(
                "activation_conflict",
                "Persona visual pack version changed before activation.",
                details={"pack_id": pack_id},
            )
        assets = self._db.list_persona_visual_assets(
            pack_id=pack_id,
            persona_id=persona_id,
            user_id=user_id,
        )
        fingerprint = self._validate_and_fingerprint(pack=pack, assets=assets, user_id=user_id)
        if fingerprint != str(reviewed_fingerprint or ""):
            raise PersonaVisualServiceError(
                "stale_review",
                "Persona visual pack review no longer matches the pack payload.",
                details={"pack_id": pack_id},
            )
        return self._db.activate_persona_visual_pack(
            persona_id=persona_id,
            user_id=user_id,
            pack_id=pack_id,
            expected_version=int(expected_version),
            reviewed_fingerprint=fingerprint,
        )

    def review_pack(
        self,
        *,
        pack_id: str,
        user_id: str,
        reviewer_user_id: str,
        expected_version: int,
    ) -> dict[str, Any]:
        """Validate an inactive pack and bind a review to its exact fingerprint."""
        pack = self._db.get_persona_visual_pack_for_user(pack_id=pack_id, user_id=user_id)
        if not pack:
            raise PersonaVisualServiceError(
                "pack_not_found",
                "Persona visual pack not found for user.",
                details={"pack_id": pack_id},
            )
        if int(pack["version"]) != int(expected_version):
            raise PersonaVisualServiceError(
                "review_conflict",
                "Persona visual pack version changed before review.",
                details={"pack_id": pack_id},
            )
        assets = self._db.list_persona_visual_assets(
            pack_id=pack_id,
            persona_id=str(pack["persona_id"]),
            user_id=user_id,
        )
        fingerprint = self._validate_and_fingerprint(pack=pack, assets=assets, user_id=user_id)
        return self._db.create_persona_visual_pack_review(
            pack_id=pack_id,
            user_id=user_id,
            reviewer_user_id=reviewer_user_id,
            fingerprint=fingerprint,
            expected_pack_version=int(expected_version),
        )

    def fork_pack_revision(
        self,
        *,
        pack_id: str,
        user_id: str,
        manifest: dict[str, Any],
        companion_behavior: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Copy a pack and its assets into a new editable inactive revision."""
        source = self._db.get_persona_visual_pack_for_user(pack_id=pack_id, user_id=user_id)
        if not source:
            raise PersonaVisualServiceError(
                "pack_not_found",
                "Persona visual pack not found for user.",
                details={"pack_id": pack_id},
            )
        persona_id = str(source["persona_id"])
        source_assets = self._db.list_persona_visual_assets(
            pack_id=pack_id,
            persona_id=persona_id,
            user_id=user_id,
        )
        target = self._db.create_persona_visual_pack(
            persona_id=persona_id,
            user_id=user_id,
            title=str(source["title"]),
            renderer_type=str(source["renderer_type"]),
            status="failed",
            parent_pack_id=pack_id,
            revision_number=int(source.get("revision_number") or 1) + 1,
            provenance=str(source.get("provenance") or "uploaded"),
            manifest={
                "manifest_version": 1,
                "renderer_type": str(source["renderer_type"]),
                "states": {},
                "animations": {},
            },
        )
        copied_paths: list[Path] = []
        try:
            asset_id_map: dict[str, str] = {}
            copied_assets: list[dict[str, Any]] = []
            for source_asset in source_assets:
                source_path = self._asset_storage_path(
                    user_id=user_id,
                    storage_key=str(source_asset.get("storage_key") or ""),
                )
                copied = self.create_asset_from_upload(
                    persona_id=persona_id,
                    user_id=user_id,
                    pack_id=str(target["id"]),
                    content=source_path.read_bytes(),
                    mime_type=str(source_asset.get("mime_type") or ""),
                    original_filename=source_asset.get("original_filename"),
                    asset_role=str(source_asset.get("asset_role") or "frame"),
                    provenance=str(source_asset.get("provenance") or "uploaded"),
                )
                copied_paths.append(Path(str(copied["storage_path"])))
                asset_id_map[str(source_asset["id"])] = str(copied["id"])
                copied_assets.append(copied)
            remapped_manifest = remap_visual_manifest_assets(deepcopy(manifest), asset_id_map)
            validation = validate_visual_manifest(
                remapped_manifest,
                available_asset_ids={str(asset["id"]) for asset in copied_assets},
                available_asset_dimensions={
                    str(asset["id"]): (int(asset["width"]), int(asset["height"]))
                    for asset in copied_assets
                    if asset.get("width") is not None and asset.get("height") is not None
                },
                require_activatable=False,
            )
            behavior = normalize_companion_behavior(
                companion_behavior,
                resolvable_state_ids=resolved_visual_state_ids(validation.manifest),
            )
            updated = self._db.update_persona_visual_pack_payload(
                pack_id=str(target["id"]),
                user_id=user_id,
                manifest=validation.manifest,
                companion_behavior=behavior,
                expected_version=int(target["version"]),
            )
            finalized = self._db.update_persona_visual_pack_status(
                pack_id=str(target["id"]),
                persona_id=persona_id,
                user_id=user_id,
                status="draft",
                expected_version=int(updated["version"]),
            )
        except Exception:
            self._db.soft_delete_persona_visual_pack_with_assets(
                pack_id=str(target["id"]),
                persona_id=persona_id,
                user_id=user_id,
            )
            for path in copied_paths:
                path.unlink(missing_ok=True)
            raise
        finalized["assets"] = self._db.list_persona_visual_assets(
            pack_id=str(target["id"]), persona_id=persona_id, user_id=user_id
        )
        finalized["assets_by_id"] = {
            str(asset["id"]): asset for asset in finalized["assets"]
        }
        return finalized

    def deactivate_pack(
        self,
        *,
        persona_id: str,
        user_id: str,
    ) -> bool:
        return self._db.deactivate_persona_visual_pack(
            persona_id=persona_id,
            user_id=user_id,
        )

    def duplicate_pack_to_persona(
        self,
        *,
        source_persona_id: str,
        user_id: str,
        pack_id: str,
        target_persona_id: str,
        title: str | None = None,
    ) -> dict[str, Any]:
        if str(source_persona_id) == str(target_persona_id):
            raise PersonaVisualServiceError(
                "same_persona_target_unsupported",
                "Persona visual packs can only be duplicated to a different persona in V1.",
            )

        source_pack = self._db.get_persona_visual_pack(
            pack_id=pack_id,
            persona_id=source_persona_id,
            user_id=user_id,
        )
        if not source_pack:
            raise PersonaVisualServiceError(
                "pack_not_found",
                "Persona visual pack not found for user.",
                details={"pack_id": pack_id},
            )

        target_persona = self._db.get_persona_profile(
            persona_id=target_persona_id,
            user_id=user_id,
        )
        if not target_persona:
            raise PersonaVisualServiceError(
                "target_persona_not_found",
                "Target persona not found for user.",
                details={"target_persona_id": target_persona_id},
            )

        source_manifest = source_pack.get("manifest") if isinstance(source_pack.get("manifest"), dict) else {}
        source_assets = self._db.list_persona_visual_assets(
            pack_id=pack_id,
            persona_id=source_persona_id,
            user_id=user_id,
        )
        source_assets_by_id = {str(asset["id"]): asset for asset in source_assets}
        referenced_asset_ids = collect_visual_manifest_asset_ids(source_manifest)
        missing_asset_ids = sorted(referenced_asset_ids - set(source_assets_by_id))
        if missing_asset_ids:
            raise PersonaVisualServiceError(
                "invalid_manifest",
                "Persona visual manifest references assets that are not in the source pack.",
                details={"asset_ids": missing_asset_ids},
            )

        asset_ids_to_copy = set(source_assets_by_id)
        preflight_assets: list[tuple[dict[str, Any], Path]] = []
        for asset_id in sorted(asset_ids_to_copy):
            asset = source_assets_by_id[asset_id]
            source_path = self._asset_storage_path(
                user_id=user_id,
                storage_key=str(asset.get("storage_key") or ""),
            )
            if not source_path.is_file():
                raise PersonaVisualServiceError(
                    "source_asset_missing",
                    "Persona visual source asset file is missing.",
                    details={"asset_id": asset_id},
                )
            checksum = self._sha256_file(source_path)
            if checksum != str(asset.get("checksum_sha256") or ""):
                raise PersonaVisualServiceError(
                    "source_asset_checksum_mismatch",
                    "Persona visual source asset checksum does not match metadata.",
                    details={"asset_id": asset_id},
                )
            preflight_assets.append((asset, source_path))

        title_value = str(title or "").strip()
        if not title_value:
            title_value = f"Copy of {source_pack['title']}"
        renderer_type = str(source_pack.get("renderer_type") or "sprite_frames")
        target_pack = self._db.create_persona_visual_pack(
            persona_id=target_persona_id,
            user_id=user_id,
            title=title_value,
            renderer_type=renderer_type,
            status="failed",
            parent_pack_id=pack_id,
            parent_persona_id=source_persona_id,
            provenance="mixed",
            companion_behavior=deepcopy(source_pack.get("companion_behavior")),
            manifest={
                "manifest_version": 1,
                "renderer_type": renderer_type,
                "states": {},
                "animations": {},
            },
        )
        copied_file_paths: list[Path] = []
        try:
            asset_id_map: dict[str, str] = {}
            copied_assets: list[dict[str, Any]] = []
            for source_asset, source_path in preflight_assets:
                copied = self.create_asset_from_upload(
                    persona_id=target_persona_id,
                    user_id=user_id,
                    pack_id=str(target_pack["id"]),
                    content=source_path.read_bytes(),
                    mime_type=str(source_asset.get("mime_type") or "application/octet-stream"),
                    original_filename=source_asset.get("original_filename"),
                    asset_role=str(source_asset.get("asset_role") or "frame"),
                    provenance="mixed",
                )
                if copied.get("storage_path"):
                    copied_file_paths.append(Path(str(copied["storage_path"])))
                asset_id_map[str(source_asset["id"])] = str(copied["id"])
                copied_assets.append(copied)

            remapped_manifest = remap_visual_manifest_assets(source_manifest, asset_id_map)
            asset_ids = {str(asset["id"]) for asset in copied_assets}
            asset_dimensions = {
                str(asset["id"]): (int(asset["width"]), int(asset["height"]))
                for asset in copied_assets
                if asset.get("width") is not None and asset.get("height") is not None
            }
            try:
                validation = validate_visual_manifest(
                    remapped_manifest,
                    available_asset_ids=asset_ids,
                    available_asset_dimensions=asset_dimensions,
                    require_activatable=False,
                )
            except PersonaVisualManifestError as exc:
                raise PersonaVisualServiceError(
                    "invalid_manifest",
                    str(exc),
                    details={"pack_id": pack_id},
                ) from exc
            updated_pack = self._db.update_persona_visual_pack_manifest(
                pack_id=str(target_pack["id"]),
                persona_id=target_persona_id,
                user_id=user_id,
                manifest=validation.manifest,
                expected_version=int(target_pack["version"]),
            )
            finalized = self._db.update_persona_visual_pack_status(
                pack_id=str(target_pack["id"]),
                persona_id=target_persona_id,
                user_id=user_id,
                status="draft",
                expected_version=int(updated_pack["version"]),
            )
        except Exception:
            try:
                self._db.soft_delete_persona_visual_pack_with_assets(
                    pack_id=str(target_pack["id"]),
                    persona_id=target_persona_id,
                    user_id=user_id,
                )
            except Exception as cleanup_error:  # pragma: no cover - defensive cleanup logging
                logger.warning(
                    "Failed to soft-delete partially duplicated persona visual pack {}: {}",
                    target_pack.get("id"),
                    cleanup_error,
                )
            for path in copied_file_paths:
                try:
                    path.unlink(missing_ok=True)
                except OSError as cleanup_error:  # pragma: no cover - defensive cleanup logging
                    logger.warning(
                        "Failed to unlink partially duplicated persona visual asset {}: {}",
                        path,
                        cleanup_error,
                    )
            raise

        assets = self._db.list_persona_visual_assets(
            pack_id=str(target_pack["id"]),
            persona_id=target_persona_id,
            user_id=user_id,
        )
        finalized["assets"] = assets
        finalized["assets_by_id"] = {str(asset["id"]): asset for asset in assets}
        return finalized

    def list_candidates(
        self,
        *,
        persona_id: str,
        user_id: str,
        pack_id: str,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        return self._db.list_persona_visual_candidates(
            pack_id=pack_id,
            persona_id=persona_id,
            user_id=user_id,
            status=status,
        )

    def accept_candidate(
        self,
        *,
        persona_id: str,
        user_id: str,
        pack_id: str,
        candidate_id: str,
    ) -> dict[str, Any]:
        candidate = self._db.get_persona_visual_candidate(
            candidate_id=candidate_id,
            pack_id=pack_id,
            persona_id=persona_id,
            user_id=user_id,
        )
        if not candidate:
            raise PersonaVisualServiceError(
                "candidate_not_found",
                "Persona visual candidate not found for user.",
                details={"candidate_id": candidate_id},
            )
        self._ensure_candidate_reviewable(candidate)
        pack = self._db.get_persona_visual_pack(
            pack_id=pack_id,
            persona_id=persona_id,
            user_id=user_id,
        )
        if not pack:
            raise PersonaVisualServiceError(
                "pack_not_found",
                "Persona visual pack not found for user.",
                details={"pack_id": pack_id},
            )

        merged_manifest = self._merge_candidate_patch(
            pack.get("manifest") if isinstance(pack.get("manifest"), dict) else {},
            (
                candidate.get("proposed_manifest_patch")
                if isinstance(candidate.get("proposed_manifest_patch"), dict)
                else {}
            ),
        )
        assets = self._db.list_persona_visual_assets(
            pack_id=pack_id,
            persona_id=persona_id,
            user_id=user_id,
        )
        asset_ids = {str(asset["id"]) for asset in assets}
        asset_dimensions = {
            str(asset["id"]): (int(asset["width"]), int(asset["height"]))
            for asset in assets
            if asset.get("width") is not None and asset.get("height") is not None
        }
        try:
            validation = validate_visual_manifest(
                merged_manifest,
                available_asset_ids=asset_ids,
                available_asset_dimensions=asset_dimensions,
                require_activatable=False,
            )
        except PersonaVisualManifestError as exc:
            raise PersonaVisualServiceError(
                "invalid_manifest",
                str(exc),
                details={"candidate_id": candidate_id, "pack_id": pack_id},
            ) from exc

        self._db.update_persona_visual_pack_manifest(
            pack_id=pack_id,
            persona_id=persona_id,
            user_id=user_id,
            manifest=validation.manifest,
            expected_version=int(pack["version"]),
        )
        updated = self._db.update_persona_visual_candidate_status(
            candidate_id=candidate_id,
            pack_id=pack_id,
            persona_id=persona_id,
            user_id=user_id,
            status="accepted",
            expected_statuses=set(_REVIEWABLE_CANDIDATE_STATUSES),
        )
        if not updated:
            latest = self._db.get_persona_visual_candidate(
                candidate_id=candidate_id,
                pack_id=pack_id,
                persona_id=persona_id,
                user_id=user_id,
            )
            if latest:
                self._raise_candidate_status_conflict(latest)
            raise PersonaVisualServiceError(
                "candidate_not_found",
                "Persona visual candidate not found for user.",
                details={"candidate_id": candidate_id},
            )
        return updated

    def reject_candidate(
        self,
        *,
        persona_id: str,
        user_id: str,
        pack_id: str,
        candidate_id: str,
        status: str = "rejected",
        failure_reason: str | None = None,
    ) -> dict[str, Any]:
        candidate = self._db.get_persona_visual_candidate(
            candidate_id=candidate_id,
            pack_id=pack_id,
            persona_id=persona_id,
            user_id=user_id,
        )
        if not candidate:
            raise PersonaVisualServiceError(
                "candidate_not_found",
                "Persona visual candidate not found for user.",
                details={"candidate_id": candidate_id},
            )
        self._ensure_candidate_reviewable(candidate)
        updated = self._db.update_persona_visual_candidate_status(
            candidate_id=candidate_id,
            pack_id=pack_id,
            persona_id=persona_id,
            user_id=user_id,
            status=status,
            failure_reason=failure_reason,
            expected_statuses=set(_REVIEWABLE_CANDIDATE_STATUSES),
        )
        if not updated:
            latest = self._db.get_persona_visual_candidate(
                candidate_id=candidate_id,
                pack_id=pack_id,
                persona_id=persona_id,
                user_id=user_id,
            )
            if latest:
                self._raise_candidate_status_conflict(latest)
            raise PersonaVisualServiceError(
                "candidate_not_found",
                "Persona visual candidate not found for user.",
                details={"candidate_id": candidate_id},
            )
        return updated

    def _ensure_candidate_reviewable(self, candidate: dict[str, Any]) -> None:
        """Raise when a generated visual candidate has already reached a terminal state."""
        if str(candidate.get("status") or "").strip() in _REVIEWABLE_CANDIDATE_STATUSES:
            return
        self._raise_candidate_status_conflict(candidate)

    def _raise_candidate_status_conflict(self, candidate: dict[str, Any]) -> None:
        """Raise a typed service error for terminal candidate review transitions."""
        raise PersonaVisualServiceError(
            "candidate_status_conflict",
            "Persona visual candidate has already reached a terminal review status.",
            details={
                "candidate_id": str(candidate.get("id") or ""),
                "status": str(candidate.get("status") or ""),
            },
        )

    @staticmethod
    def _merge_candidate_patch(
        manifest: dict[str, Any],
        patch: dict[str, Any],
    ) -> dict[str, Any]:
        merged = deepcopy(manifest if isinstance(manifest, dict) else {})
        merged.setdefault("manifest_version", 1)
        merged.setdefault("renderer_type", "sprite_frames")
        merged.setdefault("states", {})
        merged.setdefault("animations", {})
        merged.setdefault("fallbacks", {})
        merged.setdefault("authored_triggers", [])

        for field_name in ("states", "animations", "fallbacks"):
            patch_value = patch.get(field_name)
            if not isinstance(patch_value, dict):
                continue
            target = merged.get(field_name)
            if not isinstance(target, dict):
                target = {}
                merged[field_name] = target
            for key, value in patch_value.items():
                if not isinstance(key, str) or not key:
                    continue
                if value is None:
                    continue
                target[key] = deepcopy(value)

        patch_triggers = patch.get("authored_triggers")
        if isinstance(patch_triggers, list):
            current_triggers = merged.get("authored_triggers")
            if not isinstance(current_triggers, list):
                current_triggers = []
            merged_triggers: list[Any] = []
            trigger_index_by_id: dict[str, int] = {}
            for trigger in [*deepcopy(current_triggers), *deepcopy(patch_triggers)]:
                if not isinstance(trigger, dict):
                    merged_triggers.append(trigger)
                    continue
                trigger_id = str(trigger.get("id") or "").strip()
                if not trigger_id:
                    merged_triggers.append(trigger)
                    continue
                existing_index = trigger_index_by_id.get(trigger_id)
                if existing_index is None:
                    trigger_index_by_id[trigger_id] = len(merged_triggers)
                    merged_triggers.append(trigger)
                    continue
                merged_triggers[existing_index] = trigger
            merged["authored_triggers"] = merged_triggers
        return merged

    @staticmethod
    def _normalize_mime_type(mime_type: str) -> str:
        normalized = str(mime_type or "").strip().lower()
        if normalized not in ALLOWED_VISUAL_MIME_TYPES:
            raise PersonaVisualServiceError(
                "unsupported_mime_type",
                f"Unsupported persona visual MIME type: {mime_type}",
            )
        return normalized

    @staticmethod
    def _validate_image_bytes(content: bytes, *, mime_type: str) -> tuple[int, int]:
        width, height, detected_mime, _frame_count = PersonaVisualService._probe_image_bytes(
            content
        )
        if detected_mime != mime_type:
            raise PersonaVisualServiceError(
                "mime_mismatch",
                f"Persona visual upload MIME mismatch: expected {mime_type}, detected {detected_mime}.",
            )
        return width, height

    @staticmethod
    def _probe_image_bytes(content: bytes) -> tuple[int, int, str, int]:
        """Decode bounded raster metadata from protected bytes."""
        if not content:
            raise PersonaVisualServiceError("invalid_image", "Persona visual upload is empty.")
        try:
            with Image.open(io.BytesIO(content)) as image:
                width, height = image.size
                detected_mime = str(Image.MIME.get(image.format or "") or "").lower()
                frame_count = int(getattr(image, "n_frames", 1) or 1)
                if frame_count > MAX_VISUAL_RASTER_FRAMES:
                    raise PersonaVisualServiceError(
                        "too_many_image_frames",
                        f"Persona visual raster may contain at most {MAX_VISUAL_RASTER_FRAMES} frames.",
                    )
                image.verify()
        except PersonaVisualServiceError:
            raise
        except _IMAGE_VALIDATION_ERRORS as exc:
            raise PersonaVisualServiceError(
                "invalid_image",
                "Persona visual upload is not a valid raster image.",
            ) from exc

        if detected_mime not in ALLOWED_VISUAL_MIME_TYPES:
            raise PersonaVisualServiceError(
                "unsupported_mime_type",
                f"Unsupported persona visual MIME type: {detected_mime}",
            )
        if width > MAX_VISUAL_IMAGE_DIMENSION or height > MAX_VISUAL_IMAGE_DIMENSION:
            raise PersonaVisualServiceError(
                "image_too_large",
                f"Persona visual image dimensions must be <= {MAX_VISUAL_IMAGE_DIMENSION}.",
            )
        return int(width), int(height), detected_mime, frame_count

    def _validate_and_fingerprint(
        self,
        *,
        pack: dict[str, Any],
        assets: list[dict[str, Any]],
        user_id: str,
    ) -> str:
        manifest = pack.get("manifest")
        row_renderer = str(pack.get("renderer_type") or "")
        row_version = pack.get("manifest_version")
        if (
            not isinstance(manifest, dict)
            or manifest.get("renderer_type") != row_renderer
            or isinstance(manifest.get("manifest_version"), bool)
            or not isinstance(manifest.get("manifest_version"), int)
            or manifest.get("manifest_version") != row_version
        ):
            raise PersonaVisualServiceError(
                "invalid_renderer_contract",
                "Persona visual pack renderer metadata does not match its manifest.",
                details={"pack_id": str(pack["id"])},
            )
        capability = get_persona_visual_renderer_capability(row_renderer)
        if (
            capability is None
            or row_version not in capability.manifest_versions
            or not capability.can_validate
            or not capability.can_activate
        ):
            raise PersonaVisualServiceError(
                "unsupported_renderer",
                "Persona visual pack renderer cannot be activated.",
                details={"pack_id": str(pack["id"]), "renderer_type": row_renderer},
            )
        asset_ids = {str(asset["id"]) for asset in assets}
        dimensions = {
            str(asset["id"]): (int(asset["width"]), int(asset["height"]))
            for asset in assets
            if asset.get("width") is not None and asset.get("height") is not None
        }
        try:
            manifest_validation = validate_visual_manifest(
                manifest,
                available_asset_ids=asset_ids,
                available_asset_dimensions=dimensions,
                require_activatable=True,
            )
        except PersonaVisualManifestError as exc:
            raise PersonaVisualServiceError(
                "invalid_manifest", str(exc), details={"pack_id": str(pack["id"])}
            ) from exc
        try:
            behavior = normalize_companion_behavior(
                pack.get("companion_behavior"),
                resolvable_state_ids=resolved_visual_state_ids(manifest_validation.manifest),
            )
        except CompanionBehaviorValidationError as exc:
            raise PersonaVisualServiceError(
                "invalid_companion_behavior",
                str(exc),
                details={"pack_id": str(pack["id"])},
            ) from exc

        reachable_ids = collect_visual_manifest_asset_ids(manifest_validation.manifest)
        assets_by_id = {str(asset["id"]): asset for asset in assets}
        missing = sorted(reachable_ids - set(assets_by_id))
        if missing:
            raise PersonaVisualServiceError(
                "invalid_manifest",
                "Persona visual manifest references missing pack assets.",
                details={"pack_id": str(pack["id"]), "asset_ids": missing},
            )
        reachable_assets = [assets_by_id[asset_id] for asset_id in sorted(reachable_ids)]
        enriched = self._probe_reachable_assets(reachable_assets, user_id=user_id)
        static_validation = validate_sprite_static_coverage(
            manifest_validation.manifest,
            enriched,
        )
        if not static_validation.is_valid:
            raise PersonaVisualServiceError(
                "invalid_static_coverage",
                "Persona visual pack lacks one-frame PNG coverage for every built-in state.",
                details={"pack_id": str(pack["id"]), "errors": list(static_validation.errors)},
            )
        normalized_pack = {
            **pack,
            "manifest": manifest_validation.manifest,
            "manifest_version": manifest_validation.manifest["manifest_version"],
            "companion_behavior": behavior,
        }
        return build_persona_visual_pack_fingerprint(normalized_pack, reachable_assets)

    def _probe_reachable_assets(
        self,
        assets: list[dict[str, Any]],
        *,
        user_id: str,
    ) -> tuple[MappingProxyType[str, Any], ...]:
        enriched: list[MappingProxyType[str, Any]] = []
        for asset in assets:
            path = self._asset_storage_path(
                user_id=user_id,
                storage_key=str(asset.get("storage_key") or ""),
            )
            if not path.is_file():
                raise PersonaVisualServiceError(
                    "source_asset_missing",
                    "Persona visual source asset file is missing.",
                    details={"asset_id": str(asset["id"])},
                )
            if path.stat().st_size > MAX_VISUAL_UPLOAD_BYTES:
                raise PersonaVisualServiceError(
                    "upload_too_large",
                    f"Persona visual asset exceeds {MAX_VISUAL_UPLOAD_BYTES} bytes.",
                    details={"asset_id": str(asset["id"])},
                )
            content = path.read_bytes()
            if len(content) > MAX_VISUAL_UPLOAD_BYTES:
                raise PersonaVisualServiceError(
                    "upload_too_large",
                    f"Persona visual asset exceeds {MAX_VISUAL_UPLOAD_BYTES} bytes.",
                    details={"asset_id": str(asset["id"])},
                )
            if hashlib.sha256(content).hexdigest() != str(asset.get("checksum_sha256") or ""):
                raise PersonaVisualServiceError(
                    "asset_checksum_mismatch",
                    "Persona visual asset checksum does not match protected bytes.",
                    details={"asset_id": str(asset["id"])},
                )
            _width, _height, detected_mime, frame_count = self._probe_image_bytes(content)
            enriched.append(
                MappingProxyType(
                    {
                        **asset,
                        "detected_mime_type": detected_mime,
                        "decoded_frame_count": frame_count,
                    }
                )
            )
        return tuple(enriched)

    @staticmethod
    def _safe_storage_component(value: str, *, prefix: str) -> str:
        raw = str(value or "").strip()
        try:
            return normalize_output_storage_filename(
                raw,
                allow_absolute=False,
                reject_relative_with_separators=True,
                expand_user=False,
            )
        except InvalidStoragePathError:
            digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
            return f"{prefix}_{digest}"

    def _build_storage_target(
        self,
        *,
        user_id: str,
        persona_id: str,
        pack_id: str,
        asset_id: str,
        extension: str,
    ) -> tuple[str, Path]:
        visuals_dir = DatabasePaths.get_user_persona_visuals_dir(user_id)
        base = visuals_dir.resolve(strict=False)
        safe_persona_id = self._safe_storage_component(persona_id, prefix="persona")
        safe_pack_id = self._safe_storage_component(pack_id, prefix="pack")
        safe_asset_name = self._safe_storage_component(f"{asset_id}{extension}", prefix="asset")
        target_path = (base / safe_persona_id / safe_pack_id / safe_asset_name).resolve(strict=False)
        if not target_path.is_relative_to(base):
            raise PersonaVisualServiceError(
                "invalid_storage_path",
                "Persona visual storage path escapes the user visual directory.",
            )
        storage_key = f"{VISUAL_STORAGE_PREFIX}/{safe_persona_id}/{safe_pack_id}/{safe_asset_name}"
        return storage_key, target_path

    def _asset_storage_path(self, *, user_id: str, storage_key: str) -> Path:
        prefix = f"{VISUAL_STORAGE_PREFIX}/"
        relative_key = storage_key[len(prefix) :] if storage_key.startswith(prefix) else storage_key
        relative_path = Path(*Path(relative_key).parts)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise PersonaVisualServiceError(
                "invalid_storage_path",
                "Persona visual storage path escapes the user visual directory.",
            )
        base = DatabasePaths.get_user_persona_visuals_dir(user_id).resolve(strict=False)
        target_path = (base / relative_path).resolve(strict=False)
        if not target_path.is_relative_to(base):
            raise PersonaVisualServiceError(
                "invalid_storage_path",
                "Persona visual storage path escapes the user visual directory.",
            )
        return target_path

    @staticmethod
    def _sha256_file(path: Path) -> str:
        """Hash a source asset without keeping every duplicate asset in memory."""
        digest = hashlib.sha256()
        with path.open("rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


__all__ = [
    "ALLOWED_VISUAL_MIME_TYPES",
    "MAX_VISUAL_IMAGE_DIMENSION",
    "MAX_VISUAL_UPLOAD_BYTES",
    "PersonaVisualService",
    "PersonaVisualServiceError",
]
