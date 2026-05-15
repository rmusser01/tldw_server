from __future__ import annotations

import hashlib
import io
import uuid
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from PIL import Image, UnidentifiedImageError

from tldw_Server_API.app.core.DB_Management.db_path_utils import (
    DatabasePaths,
    normalize_output_storage_filename,
)
from tldw_Server_API.app.core.exceptions import InvalidStoragePathError
from tldw_Server_API.app.core.Persona.visual_asset_constraints import (
    ALLOWED_VISUAL_MIME_TYPES,
    MAX_VISUAL_IMAGE_DIMENSION,
    VISUAL_MIME_EXTENSIONS,
)
from tldw_Server_API.app.core.Persona.visual_manifest_assets import (
    collect_visual_manifest_asset_ids,
    remap_visual_manifest_assets,
)
from tldw_Server_API.app.core.Persona.visual_starter_catalog import (
    get_persona_visual_starter_pack,
)
from tldw_Server_API.app.core.Persona.visuals import (
    PersonaVisualManifestError,
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
                pack.get("manifest") or {},
                available_asset_ids=asset_ids,
                available_asset_dimensions=asset_dimensions,
                require_activatable=True,
            )
        except PersonaVisualManifestError as exc:
            raise PersonaVisualServiceError(
                "invalid_manifest",
                str(exc),
                details={"pack_id": pack_id},
            ) from exc

        self._db.update_persona_visual_pack_manifest(
            pack_id=pack_id,
            persona_id=persona_id,
            user_id=user_id,
            manifest=validation.manifest,
            expected_version=int(pack["version"]),
        )
        return self._db.activate_persona_visual_pack(
            persona_id=persona_id,
            user_id=user_id,
            pack_id=pack_id,
        )

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

        preflight_assets: list[tuple[dict[str, Any], Path]] = []
        for asset_id in sorted(referenced_asset_ids):
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

    def copy_starter_pack_to_persona(
        self,
        *,
        starter_pack_id: str,
        user_id: str,
        target_persona_id: str,
        title: str | None = None,
    ) -> dict[str, Any]:
        """Copy an immutable bundled starter pack into a user-owned draft pack."""
        starter_pack = get_persona_visual_starter_pack(starter_pack_id)
        if starter_pack is None:
            raise PersonaVisualServiceError(
                "starter_pack_not_found",
                "Persona visual starter pack not found.",
                details={"starter_pack_id": starter_pack_id},
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

        if not isinstance(starter_pack.manifest, dict):
            raise PersonaVisualServiceError(
                "invalid_starter_pack",
                "Persona visual starter manifest must be an object.",
                details={"starter_pack_id": starter_pack.id},
            )
        source_manifest = deepcopy(starter_pack.manifest)
        seen_asset_ids: set[str] = set()
        duplicate_asset_ids: set[str] = set()
        starter_assets_by_id: dict[str, Any] = {}
        for asset in starter_pack.assets:
            source_asset_id = str(asset.id or "").strip()
            if not source_asset_id:
                raise PersonaVisualServiceError(
                    "invalid_starter_pack",
                    "Persona visual starter pack contains a bundled asset without an ID.",
                    details={"starter_pack_id": starter_pack.id},
                )
            if source_asset_id in seen_asset_ids:
                duplicate_asset_ids.add(source_asset_id)
            seen_asset_ids.add(source_asset_id)
            starter_assets_by_id[source_asset_id] = asset
        if duplicate_asset_ids:
            raise PersonaVisualServiceError(
                "invalid_starter_pack",
                "Persona visual starter pack contains duplicate bundled asset IDs.",
                details={"starter_pack_id": starter_pack.id, "asset_ids": sorted(duplicate_asset_ids)},
            )
        referenced_asset_ids = collect_visual_manifest_asset_ids(source_manifest)
        missing_asset_ids = sorted(referenced_asset_ids - set(starter_assets_by_id))
        if missing_asset_ids:
            raise PersonaVisualServiceError(
                "invalid_starter_pack",
                "Persona visual starter manifest references bundled assets that are missing.",
                details={"starter_pack_id": starter_pack.id, "asset_ids": missing_asset_ids},
            )
        if not referenced_asset_ids:
            raise PersonaVisualServiceError(
                "invalid_starter_pack",
                "Persona visual starter manifest does not reference any bundled assets.",
                details={"starter_pack_id": starter_pack.id},
            )
        try:
            validate_visual_manifest(
                source_manifest,
                available_asset_ids=referenced_asset_ids,
                require_activatable=False,
            )
        except PersonaVisualManifestError as exc:
            raise PersonaVisualServiceError(
                "invalid_starter_pack",
                str(exc),
                details={"starter_pack_id": starter_pack.id},
            ) from exc

        title_value = str(title or "").strip() or starter_pack.title
        target_pack = self._db.create_persona_visual_pack(
            persona_id=target_persona_id,
            user_id=user_id,
            title=title_value,
            renderer_type=starter_pack.renderer_type,
            status="failed",
            provenance="imported",
            manifest={
                "manifest_version": starter_pack.manifest_version,
                "renderer_type": starter_pack.renderer_type,
                "states": {},
                "animations": {},
            },
        )
        copied_file_paths: list[Path] = []
        try:
            asset_id_map: dict[str, str] = {}
            copied_assets: list[dict[str, Any]] = []
            for source_asset_id in sorted(referenced_asset_ids):
                source_asset = starter_assets_by_id[source_asset_id]
                copied = self.create_asset_from_upload(
                    persona_id=target_persona_id,
                    user_id=user_id,
                    pack_id=str(target_pack["id"]),
                    content=source_asset.content,
                    mime_type=source_asset.mime_type,
                    original_filename=source_asset.filename,
                    asset_role=source_asset.asset_role,
                    provenance="imported",
                )
                if copied.get("storage_path"):
                    copied_file_paths.append(Path(str(copied["storage_path"])))
                asset_id_map[source_asset_id] = str(copied["id"])
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
                    "invalid_starter_pack",
                    str(exc),
                    details={"starter_pack_id": starter_pack.id},
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
                    "Failed to soft-delete partially copied persona visual starter pack {}: {}",
                    target_pack.get("id"),
                    cleanup_error,
                )
            for path in copied_file_paths:
                try:
                    path.unlink(missing_ok=True)
                except OSError as cleanup_error:  # pragma: no cover - defensive cleanup logging
                    logger.warning(
                        "Failed to unlink partially copied persona visual starter asset {}: {}",
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
            candidate.get("proposed_manifest_patch")
            if isinstance(candidate.get("proposed_manifest_patch"), dict)
            else {},
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
        )
        if not updated:
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
        updated = self._db.update_persona_visual_candidate_status(
            candidate_id=candidate_id,
            pack_id=pack_id,
            persona_id=persona_id,
            user_id=user_id,
            status=status,
            failure_reason=failure_reason,
        )
        if not updated:
            raise PersonaVisualServiceError(
                "candidate_not_found",
                "Persona visual candidate not found for user.",
                details={"candidate_id": candidate_id},
            )
        return updated

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
        if not content:
            raise PersonaVisualServiceError("invalid_image", "Persona visual upload is empty.")
        try:
            with Image.open(io.BytesIO(content)) as image:
                width, height = image.size
                detected_mime = Image.MIME.get(image.format or "")
                image.verify()
        except _IMAGE_VALIDATION_ERRORS as exc:
            raise PersonaVisualServiceError(
                "invalid_image",
                "Persona visual upload is not a valid raster image.",
            ) from exc

        if detected_mime and detected_mime.lower() != mime_type:
            raise PersonaVisualServiceError(
                "mime_mismatch",
                f"Persona visual upload MIME mismatch: expected {mime_type}, detected {detected_mime.lower()}.",
            )
        if width > MAX_VISUAL_IMAGE_DIMENSION or height > MAX_VISUAL_IMAGE_DIMENSION:
            raise PersonaVisualServiceError(
                "image_too_large",
                f"Persona visual image dimensions must be <= {MAX_VISUAL_IMAGE_DIMENSION}.",
            )
        return int(width), int(height)

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
        relative_key = storage_key[len(prefix):] if storage_key.startswith(prefix) else storage_key
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
