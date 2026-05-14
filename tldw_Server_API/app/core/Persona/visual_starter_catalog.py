"""Service for copying bundled Persona Visual starter packs into user drafts.

Bundled starter packs are immutable fixtures used for first-run visual Buddy
setup. This service lists those fixtures and creates normal user-owned draft
packs by validating fixture manifests, copying fixture assets through
``PersonaVisualService``, and remapping manifest asset references. It never
activates a pack automatically and never stores global mutable pack rows.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from tldw_Server_API.app.core.Persona.visual_manifest_assets import (
    collect_visual_manifest_asset_ids,
    remap_visual_manifest_assets,
)
from tldw_Server_API.app.core.Persona.visual_service import (
    PersonaVisualService,
    PersonaVisualServiceError,
)
from tldw_Server_API.app.core.Persona.visual_starter_fixtures import (
    DEFAULT_PERSONA_VISUAL_STARTER_PACKS,
    PersonaVisualStarterPack,
)
from tldw_Server_API.app.core.Persona.visuals import (
    PersonaVisualManifestError,
    validate_visual_manifest,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


class PersonaVisualStarterCatalogError(Exception):
    """Stable service error for bundled Persona Visual starter-pack operations."""

    def __init__(self, code: str, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


class PersonaVisualStarterCatalogService:
    """List bundled starter packs and copy them into user-owned draft packs."""

    def __init__(
        self,
        db: CharactersRAGDB,
        *,
        visual_service: PersonaVisualService | None = None,
        starter_packs: tuple[PersonaVisualStarterPack, ...] = DEFAULT_PERSONA_VISUAL_STARTER_PACKS,
    ) -> None:
        self._db = db
        self._visual_service = visual_service or PersonaVisualService(db)
        self._starter_packs = self._index_starter_packs(starter_packs)

    def list_starter_packs(self) -> list[dict[str, Any]]:
        """Return safe summaries for all bundled starter packs."""
        return [self._starter_summary(starter) for starter in self._starter_packs.values()]

    def get_starter_pack(self, starter_pack_id: str) -> dict[str, Any]:
        """Return one starter pack summary with a manifest preview."""
        starter = self._get_starter(starter_pack_id)
        detail = self._starter_summary(starter)
        detail["manifest"] = deepcopy(starter.manifest)
        detail["assets"] = [
            {
                "asset_key": asset.asset_key,
                "filename": asset.filename,
                "mime_type": asset.mime_type,
                "asset_role": asset.asset_role,
                "byte_size": len(asset.content),
            }
            for asset in starter.assets
        ]
        return detail

    def copy_starter_pack_to_persona(
        self,
        *,
        starter_pack_id: str,
        persona_id: str,
        user_id: str,
        title: str | None = None,
    ) -> dict[str, Any]:
        """Copy a bundled starter into one user's persona as an inactive draft."""
        starter = self._get_starter(starter_pack_id)
        persona_id_value = str(persona_id or "").strip()
        user_id_value = str(user_id or "").strip()
        if not persona_id_value:
            raise PersonaVisualStarterCatalogError(
                "persona_id_required",
                "Target persona_id is required.",
            )
        if not user_id_value:
            raise PersonaVisualStarterCatalogError(
                "user_id_required",
                "User id is required.",
            )

        target_persona = self._db.get_persona_profile(
            persona_id=persona_id_value,
            user_id=user_id_value,
        )
        if not target_persona:
            raise PersonaVisualStarterCatalogError(
                "target_persona_not_found",
                "Target persona not found for user.",
                details={"persona_id": persona_id_value},
            )

        self._validate_starter_fixture(starter)
        title_value = str(title or "").strip() or starter.title
        target_pack = self._db.create_persona_visual_pack(
            persona_id=persona_id_value,
            user_id=user_id_value,
            title=title_value,
            renderer_type=starter.renderer_type,
            status="failed",
            provenance="imported",
            manifest={
                "manifest_version": 1,
                "renderer_type": starter.renderer_type,
                "states": {},
                "animations": {},
            },
        )
        copied_file_paths: list[Path] = []
        try:
            asset_id_map: dict[str, str] = {}
            copied_assets: list[dict[str, Any]] = []
            for starter_asset in starter.assets:
                try:
                    copied = self._visual_service.create_asset_from_upload(
                        persona_id=persona_id_value,
                        user_id=user_id_value,
                        pack_id=str(target_pack["id"]),
                        content=starter_asset.content,
                        mime_type=starter_asset.mime_type,
                        original_filename=starter_asset.filename,
                        asset_role=starter_asset.asset_role,
                        provenance="imported",
                    )
                except PersonaVisualServiceError as exc:
                    raise PersonaVisualStarterCatalogError(
                        "invalid_starter_asset",
                        str(exc),
                        details={"asset_key": starter_asset.asset_key, **exc.details},
                    ) from exc
                if copied.get("storage_path"):
                    copied_file_paths.append(Path(str(copied["storage_path"])))
                asset_id_map[starter_asset.asset_key] = str(copied["id"])
                copied_assets.append(copied)

            remapped_manifest = remap_visual_manifest_assets(starter.manifest, asset_id_map)
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
                    require_activatable=True,
                )
            except PersonaVisualManifestError as exc:
                raise PersonaVisualStarterCatalogError(
                    "invalid_starter_manifest",
                    str(exc),
                    details={"starter_pack_id": starter.id},
                ) from exc

            updated_pack = self._db.update_persona_visual_pack_manifest(
                pack_id=str(target_pack["id"]),
                persona_id=persona_id_value,
                user_id=user_id_value,
                manifest=validation.manifest,
                expected_version=int(target_pack["version"]),
            )
            finalized = self._db.update_persona_visual_pack_status(
                pack_id=str(target_pack["id"]),
                persona_id=persona_id_value,
                user_id=user_id_value,
                status="draft",
                expected_version=int(updated_pack["version"]) if updated_pack else None,
            )
        except Exception:
            self._cleanup_partial_pack(
                target_pack_id=str(target_pack["id"]),
                persona_id=persona_id_value,
                user_id=user_id_value,
                copied_file_paths=copied_file_paths,
            )
            raise

        assets = self._db.list_persona_visual_assets(
            pack_id=str(target_pack["id"]),
            persona_id=persona_id_value,
            user_id=user_id_value,
        )
        finalized["assets"] = assets
        finalized["assets_by_id"] = {str(asset["id"]): asset for asset in assets}
        return finalized

    @staticmethod
    def _index_starter_packs(
        starter_packs: tuple[PersonaVisualStarterPack, ...],
    ) -> dict[str, PersonaVisualStarterPack]:
        indexed: dict[str, PersonaVisualStarterPack] = {}
        for starter in starter_packs:
            starter_id = str(starter.id or "").strip()
            if not starter_id:
                raise PersonaVisualStarterCatalogError(
                    "invalid_starter_fixture",
                    "Bundled starter pack id is required.",
                )
            if starter_id in indexed:
                raise PersonaVisualStarterCatalogError(
                    "duplicate_starter_fixture",
                    "Bundled starter pack ids must be unique.",
                    details={"starter_pack_id": starter_id},
                )
            indexed[starter_id] = starter
        return indexed

    def _get_starter(self, starter_pack_id: str) -> PersonaVisualStarterPack:
        starter_id = str(starter_pack_id or "").strip()
        starter = self._starter_packs.get(starter_id)
        if starter is None:
            raise PersonaVisualStarterCatalogError(
                "starter_pack_not_found",
                "Persona Visual starter pack not found.",
                details={"starter_pack_id": starter_id},
            )
        return starter

    @staticmethod
    def _starter_summary(starter: PersonaVisualStarterPack) -> dict[str, Any]:
        states = starter.manifest.get("states") if isinstance(starter.manifest, dict) else {}
        return {
            "id": starter.id,
            "title": starter.title,
            "description": starter.description,
            "renderer_type": starter.renderer_type,
            "manifest_version": int(starter.manifest.get("manifest_version", 1) or 1),
            "states_offered": sorted(states) if isinstance(states, dict) else [],
            "asset_count": len(starter.assets),
            "total_bytes": sum(len(asset.content) for asset in starter.assets),
            "tags": list(starter.tags),
            "license_label": starter.license_label,
        }

    @staticmethod
    def _validate_starter_fixture(starter: PersonaVisualStarterPack) -> None:
        asset_keys: set[str] = set()
        for asset in starter.assets:
            key = str(asset.asset_key or "").strip()
            if not key:
                raise PersonaVisualStarterCatalogError(
                    "invalid_starter_asset",
                    "Bundled starter asset keys must be non-empty.",
                    details={"starter_pack_id": starter.id},
                )
            if key in asset_keys:
                raise PersonaVisualStarterCatalogError(
                    "invalid_starter_asset",
                    "Bundled starter asset keys must be unique.",
                    details={"starter_pack_id": starter.id, "asset_key": key},
                )
            if not asset.content:
                raise PersonaVisualStarterCatalogError(
                    "invalid_starter_asset",
                    "Bundled starter asset content is empty.",
                    details={"starter_pack_id": starter.id, "asset_key": key},
                )
            asset_keys.add(key)

        referenced_asset_ids = collect_visual_manifest_asset_ids(starter.manifest)
        missing_asset_ids = sorted(referenced_asset_ids - asset_keys)
        if missing_asset_ids:
            raise PersonaVisualStarterCatalogError(
                "invalid_starter_manifest",
                "Bundled starter manifest references assets missing from the fixture.",
                details={"starter_pack_id": starter.id, "asset_ids": missing_asset_ids},
            )
        try:
            validate_visual_manifest(
                deepcopy(starter.manifest),
                available_asset_ids=asset_keys,
                require_activatable=True,
            )
        except PersonaVisualManifestError as exc:
            raise PersonaVisualStarterCatalogError(
                "invalid_starter_manifest",
                str(exc),
                details={"starter_pack_id": starter.id},
            ) from exc

    def _cleanup_partial_pack(
        self,
        *,
        target_pack_id: str,
        persona_id: str,
        user_id: str,
        copied_file_paths: list[Path],
    ) -> None:
        try:
            self._db.soft_delete_persona_visual_pack_with_assets(
                pack_id=target_pack_id,
                persona_id=persona_id,
                user_id=user_id,
            )
        except Exception as cleanup_error:  # pragma: no cover - defensive cleanup logging
            logger.warning(
                "Failed to soft-delete partial starter visual pack {}: {}",
                target_pack_id,
                cleanup_error,
            )
        for path in copied_file_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError as cleanup_error:  # pragma: no cover - defensive cleanup logging
                logger.warning(
                    "Failed to unlink partial starter visual asset {}: {}",
                    path,
                    cleanup_error,
                )


__all__ = [
    "PersonaVisualStarterCatalogError",
    "PersonaVisualStarterCatalogService",
]
