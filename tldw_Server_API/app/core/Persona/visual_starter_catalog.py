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
from tldw_Server_API.app.core.Persona.visual_renderer_capabilities import (
    get_persona_visual_renderer_capability,
)
from tldw_Server_API.app.core.Persona.visual_service import (
    PersonaVisualService,
    PersonaVisualServiceError,
)
from tldw_Server_API.app.core.Persona.visual_starter_fixtures import (
    DEFAULT_PERSONA_VISUAL_STARTER_PACK_ID,
    DEFAULT_PERSONA_VISUAL_STARTER_PACKS,
    LEGACY_PERSONA_VISUAL_STARTER_PACK_ID,
    PersonaVisualStarterPack,
    PersonaVisualStarterProductionRecipe,
)
from tldw_Server_API.app.core.Persona.visuals import (
    PersonaVisualManifestError,
    validate_visual_manifest,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


_ALLOWED_STARTER_RESPONSE_ASSET_ROLES = frozenset(
    {
        "frame",
        "still_pose",
        "sprite_sheet",
        "preview",
        "generated_candidate",
    }
)
_ALLOWED_STARTER_COMPLEXITY_TIERS = frozenset({"basic", "intermediate", "intricate"})
_ALLOWED_STARTER_PRODUCTION_STATUSES = frozenset({"scaffold", "art_ready"})
_MAX_STARTER_RECIPE_TEXT_LENGTH = 320
_MAX_STARTER_RECIPE_ITEMS = 12


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
        starter_packs = tuple(self._starter_packs.values())
        for starter in starter_packs:
            self._validate_starter_fixture(starter)
        return [self._starter_summary(starter) for starter in starter_packs]

    def get_starter_pack(self, starter_pack_id: str) -> dict[str, Any]:
        """Return one starter pack summary with a manifest preview."""
        starter = self._get_starter(starter_pack_id)
        self._validate_starter_fixture(starter)
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

            remapped_manifest = remap_visual_manifest_assets(deepcopy(starter.manifest), asset_id_map)
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
            if not updated_pack:
                raise PersonaVisualStarterCatalogError(
                    "starter_copy_failed",
                    "Failed to persist remapped starter manifest.",
                    details={
                        "starter_pack_id": starter.id,
                        "target_pack_id": str(target_pack["id"]),
                    },
                )
            finalized = self._db.update_persona_visual_pack_status(
                pack_id=str(target_pack["id"]),
                persona_id=persona_id_value,
                user_id=user_id_value,
                status="draft",
                expected_version=int(updated_pack["version"]),
            )
            if not finalized:
                raise PersonaVisualStarterCatalogError(
                    "starter_copy_failed",
                    "Failed to transition copied starter pack to draft.",
                    details={
                        "starter_pack_id": starter.id,
                        "target_pack_id": str(target_pack["id"]),
                    },
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
        if (
            starter_id == LEGACY_PERSONA_VISUAL_STARTER_PACK_ID
            and starter_id not in self._starter_packs
        ):
            starter_id = DEFAULT_PERSONA_VISUAL_STARTER_PACK_ID
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
        complexity_tier = PersonaVisualStarterCatalogService._starter_metadata_text(
            starter.complexity_tier,
            field_name="complexity_tier",
            starter_id=starter.id,
        )
        production_status = PersonaVisualStarterCatalogService._starter_metadata_text(
            starter.production_status,
            field_name="production_status",
            starter_id=starter.id,
        )
        expected_asset_groups = PersonaVisualStarterCatalogService._starter_metadata_tuple(
            starter.expected_asset_groups,
            field_name="expected_asset_groups",
            starter_id=starter.id,
        )
        animation_coverage_notes = PersonaVisualStarterCatalogService._starter_metadata_tuple(
            starter.animation_coverage_notes,
            field_name="animation_coverage_notes",
            starter_id=starter.id,
        )
        neutral_anchor_required = PersonaVisualStarterCatalogService._starter_metadata_bool(
            starter.neutral_anchor_required,
            field_name="neutral_anchor_required",
            starter_id=starter.id,
        )
        production_recipe = PersonaVisualStarterCatalogService._starter_production_recipe(
            starter.production_recipe,
            starter_id=starter.id,
        )
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
            "complexity_tier": complexity_tier,
            "production_status": production_status,
            "neutral_anchor_required": neutral_anchor_required,
            "expected_asset_groups": list(expected_asset_groups),
            "animation_coverage_notes": list(animation_coverage_notes),
            "production_recipe": production_recipe,
        }

    @staticmethod
    def _validate_starter_fixture(starter: PersonaVisualStarterPack) -> None:
        renderer_type = str(starter.renderer_type or "").strip()
        capability = get_persona_visual_renderer_capability(renderer_type)
        if capability is None or not capability.can_activate:
            raise PersonaVisualStarterCatalogError(
                "invalid_starter_fixture",
                "Bundled starter renderer_type is not supported for starter packs.",
                details={"starter_pack_id": starter.id, "renderer_type": renderer_type},
            )
        if not isinstance(starter.manifest, dict):
            raise PersonaVisualStarterCatalogError(
                "invalid_starter_fixture",
                "Bundled starter manifest must be an object.",
                details={"starter_pack_id": starter.id},
            )
        manifest_renderer_type = str(starter.manifest.get("renderer_type") or "").strip()
        if manifest_renderer_type != renderer_type:
            raise PersonaVisualStarterCatalogError(
                "invalid_starter_fixture",
                "Bundled starter renderer_type must match its manifest.",
                details={
                    "starter_pack_id": starter.id,
                    "renderer_type": renderer_type,
                    "manifest_renderer_type": manifest_renderer_type,
                },
            )

        complexity_tier = PersonaVisualStarterCatalogService._starter_metadata_text(
            starter.complexity_tier,
            field_name="complexity_tier",
            starter_id=starter.id,
        )
        if complexity_tier not in _ALLOWED_STARTER_COMPLEXITY_TIERS:
            raise PersonaVisualStarterCatalogError(
                "invalid_starter_fixture",
                "Bundled starter complexity_tier is not supported.",
                details={
                    "starter_pack_id": starter.id,
                    "complexity_tier": complexity_tier,
                },
            )
        production_status = PersonaVisualStarterCatalogService._starter_metadata_text(
            starter.production_status,
            field_name="production_status",
            starter_id=starter.id,
        )
        if production_status not in _ALLOWED_STARTER_PRODUCTION_STATUSES:
            raise PersonaVisualStarterCatalogError(
                "invalid_starter_fixture",
                "Bundled starter production_status is not supported.",
                details={
                    "starter_pack_id": starter.id,
                    "production_status": production_status,
                },
            )
        expected_asset_groups = PersonaVisualStarterCatalogService._starter_metadata_tuple(
            starter.expected_asset_groups,
            field_name="expected_asset_groups",
            starter_id=starter.id,
        )
        neutral_anchor_required = PersonaVisualStarterCatalogService._starter_metadata_bool(
            starter.neutral_anchor_required,
            field_name="neutral_anchor_required",
            starter_id=starter.id,
        )
        if (
            neutral_anchor_required
            and "neutral_anchor" not in expected_asset_groups
        ):
            raise PersonaVisualStarterCatalogError(
                "invalid_starter_fixture",
                "Bundled starter neutral-anchor metadata is inconsistent.",
                details={"starter_pack_id": starter.id},
            )
        animation_coverage_notes = PersonaVisualStarterCatalogService._starter_metadata_tuple(
            starter.animation_coverage_notes,
            field_name="animation_coverage_notes",
            starter_id=starter.id,
        )
        if not animation_coverage_notes:
            raise PersonaVisualStarterCatalogError(
                "invalid_starter_fixture",
                "Bundled starter animation coverage notes are required.",
                details={"starter_pack_id": starter.id},
            )
        PersonaVisualStarterCatalogService._starter_production_recipe(
            starter.production_recipe,
            starter_id=starter.id,
        )

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
            asset_role = str(asset.asset_role or "").strip()
            if (
                asset_role not in capability.supported_asset_roles
                or asset_role not in _ALLOWED_STARTER_RESPONSE_ASSET_ROLES
            ):
                raise PersonaVisualStarterCatalogError(
                    "invalid_starter_fixture",
                    "Bundled starter asset_role is not supported for starter-pack responses.",
                    details={
                        "starter_pack_id": starter.id,
                        "asset_key": key,
                        "asset_role": asset_role,
                    },
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

    @staticmethod
    def _starter_metadata_text(value: object, *, field_name: str, starter_id: str) -> str:
        """Return one canonical starter metadata string or fail fixture validation."""
        if not isinstance(value, str):
            raise PersonaVisualStarterCatalogError(
                "invalid_starter_fixture",
                "Bundled starter metadata text must be a canonical non-empty string.",
                details={"starter_pack_id": starter_id, "field_name": field_name},
            )
        normalized = value.strip()
        if not normalized or normalized != value:
            raise PersonaVisualStarterCatalogError(
                "invalid_starter_fixture",
                "Bundled starter metadata text must be a canonical non-empty string.",
                details={"starter_pack_id": starter_id, "field_name": field_name},
            )
        return normalized

    @staticmethod
    def _starter_metadata_bool(value: object, *, field_name: str, starter_id: str) -> bool:
        """Return one starter metadata boolean or fail fixture validation."""
        if not isinstance(value, bool):
            raise PersonaVisualStarterCatalogError(
                "invalid_starter_fixture",
                "Bundled starter metadata boolean must be a boolean.",
                details={"starter_pack_id": starter_id, "field_name": field_name},
            )
        return value

    @staticmethod
    def _starter_metadata_tuple(
        value: object,
        *,
        field_name: str,
        starter_id: str,
    ) -> tuple[str, ...]:
        """Return canonical immutable starter metadata entries or fail validation."""
        if not isinstance(value, tuple):
            raise PersonaVisualStarterCatalogError(
                "invalid_starter_fixture",
                "Bundled starter metadata lists must be immutable tuples.",
                details={"starter_pack_id": starter_id, "field_name": field_name},
            )
        items: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise PersonaVisualStarterCatalogError(
                    "invalid_starter_fixture",
                    "Bundled starter metadata entries must be canonical non-empty strings.",
                    details={"starter_pack_id": starter_id, "field_name": field_name},
                )
            normalized = item.strip()
            if not normalized or normalized != item:
                raise PersonaVisualStarterCatalogError(
                    "invalid_starter_fixture",
                    "Bundled starter metadata entries must be canonical non-empty strings.",
                    details={"starter_pack_id": starter_id, "field_name": field_name},
                )
            items.append(normalized)
        return tuple(items)

    @staticmethod
    def _starter_production_recipe(
        value: object,
        *,
        starter_id: str,
    ) -> dict[str, Any]:
        """Return validated starter production-recipe metadata for API output."""
        if not isinstance(value, PersonaVisualStarterProductionRecipe):
            raise PersonaVisualStarterCatalogError(
                "invalid_starter_fixture",
                "Bundled starter production_recipe must be immutable recipe metadata.",
                details={"starter_pack_id": starter_id, "field_name": "production_recipe"},
            )

        identity_brief = PersonaVisualStarterCatalogService._starter_recipe_text(
            value.identity_brief,
            field_name="production_recipe.identity_brief",
            starter_id=starter_id,
        )
        neutral_anchor = PersonaVisualStarterCatalogService._starter_recipe_text(
            value.neutral_anchor,
            field_name="production_recipe.neutral_anchor",
            starter_id=starter_id,
        )
        static_sheet = PersonaVisualStarterCatalogService._starter_recipe_text(
            value.static_sheet,
            field_name="production_recipe.static_sheet",
            starter_id=starter_id,
        )
        animation_outputs = PersonaVisualStarterCatalogService._starter_recipe_tuple(
            value.animation_outputs,
            field_name="production_recipe.animation_outputs",
            starter_id=starter_id,
        )
        review_checks = PersonaVisualStarterCatalogService._starter_recipe_tuple(
            value.review_checks,
            field_name="production_recipe.review_checks",
            starter_id=starter_id,
        )
        if "neutral_identity_consistency" not in review_checks:
            raise PersonaVisualStarterCatalogError(
                "invalid_starter_fixture",
                "Bundled starter production recipe must review neutral identity consistency.",
                details={
                    "starter_pack_id": starter_id,
                    "field_name": "production_recipe.review_checks",
                },
            )
        return {
            "identity_brief": identity_brief,
            "neutral_anchor": neutral_anchor,
            "static_sheet": static_sheet,
            "animation_outputs": list(animation_outputs),
            "review_checks": list(review_checks),
        }

    @staticmethod
    def _starter_recipe_text(value: object, *, field_name: str, starter_id: str) -> str:
        """Return bounded production-recipe text or fail fixture validation."""
        text = PersonaVisualStarterCatalogService._starter_metadata_text(
            value,
            field_name=field_name,
            starter_id=starter_id,
        )
        if len(text) > _MAX_STARTER_RECIPE_TEXT_LENGTH:
            raise PersonaVisualStarterCatalogError(
                "invalid_starter_fixture",
                "Bundled starter production recipe text is too long.",
                details={"starter_pack_id": starter_id, "field_name": field_name},
            )
        return text

    @staticmethod
    def _starter_recipe_tuple(
        value: object,
        *,
        field_name: str,
        starter_id: str,
    ) -> tuple[str, ...]:
        """Return bounded production-recipe entries or fail fixture validation."""
        items = PersonaVisualStarterCatalogService._starter_metadata_tuple(
            value,
            field_name=field_name,
            starter_id=starter_id,
        )
        if not items or len(items) > _MAX_STARTER_RECIPE_ITEMS:
            raise PersonaVisualStarterCatalogError(
                "invalid_starter_fixture",
                "Bundled starter production recipe entries must be bounded and non-empty.",
                details={"starter_pack_id": starter_id, "field_name": field_name},
            )
        if any(len(item) > _MAX_STARTER_RECIPE_TEXT_LENGTH for item in items):
            raise PersonaVisualStarterCatalogError(
                "invalid_starter_fixture",
                "Bundled starter production recipe entry text is too long.",
                details={"starter_pack_id": starter_id, "field_name": field_name},
            )
        return items

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
