from __future__ import annotations

import hashlib
import io
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image, UnidentifiedImageError

from tldw_Server_API.app.core.DB_Management.db_path_utils import (
    DatabasePaths,
    normalize_output_storage_filename,
)
from tldw_Server_API.app.core.exceptions import InvalidStoragePathError
from tldw_Server_API.app.core.Persona.visuals import (
    PersonaVisualManifestError,
    validate_visual_manifest,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


ALLOWED_VISUAL_MIME_TYPES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/gif",
    }
)
MAX_VISUAL_UPLOAD_BYTES = 10_485_760
MAX_VISUAL_IMAGE_DIMENSION = 4096
VISUAL_STORAGE_PREFIX = "persona_visuals"

_MIME_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
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
                provenance="uploaded",
            )
        except Exception:
            storage_path.unlink(missing_ok=True)
            raise

        asset["storage_path"] = str(storage_path)
        return asset

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


__all__ = [
    "ALLOWED_VISUAL_MIME_TYPES",
    "MAX_VISUAL_IMAGE_DIMENSION",
    "MAX_VISUAL_UPLOAD_BYTES",
    "PersonaVisualService",
    "PersonaVisualServiceError",
]
