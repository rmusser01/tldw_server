"""Service layer for visual identity expression pack activation and resolution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VisualIdentity_DB import VisualIdentityRepository
from tldw_Server_API.app.core.Visual_Identities.expression_slots import normalize_expression_key


ActorId = int | str


@dataclass(frozen=True)
class VisualIdentityActivationResult:
    """Result returned after a draft has been activated into a pack version."""

    draft_id: int
    pack_id: int
    pack_version_id: int
    asset_ids: tuple[int, ...]
    binding_id: int | None = None


@dataclass(frozen=True)
class VisualIdentityResolvedAsset:
    """Resolved visual identity asset metadata for a message expression."""

    actor_kind: str
    actor_id: ActorId
    pack_id: int | None
    pack_version_id: int | None
    expression_key: str | None
    requested_expression_key: str | None
    asset_id: int | None
    storage_relpath: str | None
    fallback_reason: str
    is_animated: bool = False
    content_type: str | None = None
    asset_url: str | None = None
    role_id: str | None = None
    role_label: str | None = None
    resolution_source: str = "binding"


class VisualIdentityService:
    """Coordinate visual identity pack activation and expression resolution."""

    def __init__(
        self,
        db: CharactersRAGDB,
        owner_user_id: int,
        jobs_manager: Any | None = None,
    ) -> None:
        self.db = db
        self.owner_user_id = int(owner_user_id)
        self.jobs_manager = jobs_manager
        self.repository = VisualIdentityRepository.initialized(db)

    def create_pack(
        self,
        *,
        title: str,
        description: str = "",
        default_expression_key: str = "neutral",
        source_kind: str = "manual",
        source_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create an active pack shell without activating a version."""
        normalized_default = normalize_expression_key(default_expression_key) or "neutral"
        return self.repository.create_pack(
            owner_user_id=self.owner_user_id,
            title=title,
            description=description,
            status="active",
            active_version_id=None,
            default_expression_key=normalized_default,
            source_kind=source_kind,
            source_context=source_context,
        )

    def activate_draft(
        self,
        *,
        draft_id: int,
        actor_kind: str | None = None,
        actor_id: ActorId | None = None,
    ) -> VisualIdentityActivationResult:
        """Activate a ready draft and optionally bind it to an owned actor."""
        if (actor_kind is None) != (actor_id is None):
            raise ValueError("visual_identity_binding_actor_required")
        normalized_actor_id = (
            self._validate_actor_for_binding(actor_kind, actor_id)
            if actor_kind is not None and actor_id is not None
            else None
        )
        activation = self.repository.activate_draft_as_version(
            owner_user_id=self.owner_user_id,
            draft_id=draft_id,
            actor_kind=actor_kind,
            actor_id=normalized_actor_id,
        )
        binding = activation.get("binding")
        return VisualIdentityActivationResult(
            draft_id=int(draft_id),
            pack_id=int(activation["pack"]["id"]),
            pack_version_id=int(activation["pack_version"]["id"]),
            asset_ids=tuple(int(asset["id"]) for asset in activation["assets"]),
            binding_id=int(binding["id"]) if binding is not None else None,
        )

    def resolve_expression_asset(
        self,
        actor_kind: str,
        actor_id: ActorId,
        requested_expression_key: str,
        manual_override_expression_key: str | None = None,
        mood_expression_key: str | None = None,
        *,
        role_id: str | None = None,
        role_label: str | None = None,
        override_pack_id: int | None = None,
        override_pack_version_id: int | None = None,
        allow_override_fallback: bool = False,
    ) -> VisualIdentityResolvedAsset:
        """Resolve the best asset for a requested actor expression."""
        normalized_actor_id = self._validate_actor_for_binding(actor_kind, actor_id)
        normalized_requested = normalize_expression_key(requested_expression_key)
        if override_pack_version_id is not None and override_pack_id is None:
            raise ValueError("pack_not_found")
        if override_pack_id is not None:
            return self._resolve_override_expression_asset(
                actor_kind=actor_kind,
                actor_id=normalized_actor_id,
                requested_expression_key=normalized_requested,
                role_id=role_id,
                role_label=role_label,
                override_pack_id=override_pack_id,
                override_pack_version_id=override_pack_version_id,
                allow_override_fallback=allow_override_fallback,
                manual_override_expression_key=manual_override_expression_key,
                mood_expression_key=mood_expression_key,
            )

        binding = self.repository.resolve_active_binding(
            owner_user_id=self.owner_user_id,
            actor_kind=actor_kind,
            actor_id=normalized_actor_id,
        )
        if binding is None:
            legacy = self._resolve_legacy_character_mood(
                actor_kind=actor_kind,
                actor_id=normalized_actor_id,
                manual_override_expression_key=normalize_expression_key(
                    manual_override_expression_key or ""
                ),
                requested_expression_key=normalized_requested,
                mood_expression_key=normalize_expression_key(mood_expression_key or ""),
                default_expression_key="neutral",
                role_id=role_id,
                role_label=role_label,
            )
            if legacy is not None:
                return legacy
            return self._placeholder(
                actor_kind=actor_kind,
                actor_id=normalized_actor_id,
                requested_expression_key=normalized_requested,
                role_id=role_id,
                role_label=role_label,
            )

        pack_version_id = int(binding["active_version_id"])
        assets: dict[str, dict[str, Any]] = {}
        for asset in self.repository.list_assets_for_version(
            pack_version_id,
            owner_user_id=self.owner_user_id,
        ):
            assets.setdefault(str(asset["expression_key"]), asset)
        candidates = [
            ("manual_override", normalize_expression_key(manual_override_expression_key or "")),
            ("requested", normalized_requested),
            ("mood", normalize_expression_key(mood_expression_key or "")),
            (
                "pack_default",
                normalize_expression_key(str(binding.get("pack_default_expression_key") or "")),
            ),
        ]
        for fallback_reason, expression_key in candidates:
            if not expression_key:
                continue
            asset = assets.get(expression_key)
            if asset is not None:
                return self._resolved_asset(
                    actor_kind=actor_kind,
                    actor_id=normalized_actor_id,
                    requested_expression_key=normalized_requested,
                    binding=binding,
                    asset=asset,
                    expression_key=expression_key,
                    fallback_reason=fallback_reason,
                    resolution_source=(
                        "binding" if fallback_reason == "requested" else "binding_fallback"
                    ),
                    role_id=role_id,
                    role_label=role_label,
                )

        for alias in ("neutral", "default", "normal"):
            lookup_keys = (alias, normalize_expression_key(alias))
            for lookup_key in lookup_keys:
                if not lookup_key:
                    continue
                asset = assets.get(lookup_key)
                if asset is None:
                    continue
                return self._resolved_asset(
                    actor_kind=actor_kind,
                    actor_id=normalized_actor_id,
                    requested_expression_key=normalized_requested,
                    binding=binding,
                    asset=asset,
                    expression_key=str(asset["expression_key"]),
                    fallback_reason="neutral_alias",
                    resolution_source="binding_fallback",
                    role_id=role_id,
                    role_label=role_label,
                )

        legacy = self._resolve_legacy_character_mood(
            actor_kind=actor_kind,
            actor_id=normalized_actor_id,
            manual_override_expression_key=normalize_expression_key(
                manual_override_expression_key or ""
            ),
            requested_expression_key=normalized_requested,
            mood_expression_key=normalize_expression_key(mood_expression_key or ""),
            default_expression_key=normalize_expression_key(
                str(binding.get("pack_default_expression_key") or "")
            )
            or "neutral",
            binding=binding,
            role_id=role_id,
            role_label=role_label,
        )
        if legacy is not None:
            return legacy

        return self._placeholder(
            actor_kind=actor_kind,
            actor_id=normalized_actor_id,
            requested_expression_key=normalized_requested,
            role_id=role_id,
            role_label=role_label,
        )

    def _resolve_override_expression_asset(
        self,
        *,
        actor_kind: str,
        actor_id: ActorId,
        requested_expression_key: str | None,
        role_id: str | None,
        role_label: str | None,
        override_pack_id: int,
        override_pack_version_id: int | None,
        allow_override_fallback: bool,
        manual_override_expression_key: str | None,
        mood_expression_key: str | None,
    ) -> VisualIdentityResolvedAsset:
        """Resolve against an explicit pack/version override before actor binding fallback."""
        pack, version = self._require_owned_override_version(
            pack_id=override_pack_id,
            pack_version_id=override_pack_version_id,
        )
        assets: dict[str, dict[str, Any]] = {}
        for asset in self.repository.list_assets_for_version(
            int(version["id"]),
            owner_user_id=self.owner_user_id,
        ):
            assets.setdefault(str(asset["expression_key"]), asset)

        candidates = [("requested", requested_expression_key)]
        if allow_override_fallback:
            candidates.append(
                (
                    "pack_default",
                    normalize_expression_key(
                        str(
                            version.get("default_expression_key")
                            or pack.get("default_expression_key")
                            or ""
                        )
                    ),
                )
            )
            candidates.extend(("neutral_alias", alias) for alias in ("neutral", "default", "normal"))

        for fallback_reason, expression_key in candidates:
            if not expression_key:
                continue
            asset = assets.get(expression_key)
            if asset is None:
                continue
            return self._resolved_version_asset(
                actor_kind=actor_kind,
                actor_id=actor_id,
                requested_expression_key=requested_expression_key,
                pack_id=int(pack["id"]),
                pack_version_id=int(version["id"]),
                asset=asset,
                expression_key=str(asset["expression_key"]),
                fallback_reason=(
                    fallback_reason
                    if fallback_reason == "requested"
                    else f"override_expression_missing:{fallback_reason}"
                ),
                resolution_source=(
                    "override" if fallback_reason == "requested" else "override_fallback"
                ),
                role_id=role_id,
                role_label=role_label,
            )

        if not allow_override_fallback:
            raise ValueError("override_expression_missing")

        fallback = self.resolve_expression_asset(
            actor_kind=actor_kind,
            actor_id=actor_id,
            requested_expression_key=requested_expression_key or "",
            manual_override_expression_key=manual_override_expression_key,
            mood_expression_key=mood_expression_key,
            role_id=role_id,
            role_label=role_label,
        )
        return replace(
            fallback,
            fallback_reason=f"override_expression_missing:{fallback.fallback_reason}",
            resolution_source=self._override_fallback_source(fallback.resolution_source),
            role_id=role_id,
            role_label=role_label,
        )

    def _require_owned_override_version(
        self,
        *,
        pack_id: int,
        pack_version_id: int | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return an owned override pack and version or raise a typed resolver error."""
        try:
            normalized_pack_id = int(pack_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("pack_not_found") from exc

        pack = self.repository.get_pack(normalized_pack_id, owner_user_id=self.owner_user_id)
        if pack is None:
            raise ValueError("pack_not_found")

        if pack_version_id is None:
            raise ValueError("pack_version_not_found")
        else:
            try:
                normalized_pack_version_id = int(pack_version_id)
            except (TypeError, ValueError) as exc:
                raise ValueError("pack_version_not_found") from exc

        version = self.repository.get_pack_version(
            normalized_pack_version_id,
            owner_user_id=self.owner_user_id,
        )
        if version is None:
            raise ValueError("pack_version_not_found")
        if int(version["pack_id"]) != normalized_pack_id:
            raise ValueError("pack_version_mismatch")
        return pack, version

    def _override_fallback_source(self, resolution_source: str) -> str:
        """Map normal resolver source into the matching override fallback source."""
        if resolution_source in {"binding", "binding_fallback"}:
            return "override_binding_fallback"
        if resolution_source == "legacy_character_mood":
            return "override_legacy_fallback"
        return "override_placeholder_fallback"

    def _resolve_legacy_character_mood(
        self,
        *,
        actor_kind: str,
        actor_id: ActorId,
        manual_override_expression_key: str | None,
        requested_expression_key: str | None,
        mood_expression_key: str | None,
        default_expression_key: str | None,
        binding: dict[str, Any] | None = None,
        role_id: str | None = None,
        role_label: str | None = None,
    ) -> VisualIdentityResolvedAsset | None:
        """Resolve legacy character mood image metadata without copying bytes."""
        if actor_kind != "character":
            return None
        try:
            character_id = int(actor_id)
        except (TypeError, ValueError):
            return None

        character = self.db.get_character_card_by_id(character_id)
        if character is None:
            return None

        mood_images = self._legacy_mood_images(character)
        if not mood_images:
            return None

        normalized_images: dict[str, str] = {}
        for raw_key, raw_value in mood_images.items():
            expression_key = normalize_expression_key(str(raw_key))
            if not expression_key or not isinstance(raw_value, str):
                continue
            image_url = raw_value.strip()
            if image_url:
                normalized_images.setdefault(expression_key, image_url)

        for expression_key in (
            manual_override_expression_key,
            requested_expression_key,
            mood_expression_key,
            default_expression_key,
        ):
            if not expression_key:
                continue
            image_url = normalized_images.get(expression_key)
            if image_url:
                return VisualIdentityResolvedAsset(
                    actor_kind=actor_kind,
                    actor_id=actor_id,
                    pack_id=int(binding["pack_id"]) if binding is not None else None,
                    pack_version_id=(
                        int(binding["active_version_id"]) if binding is not None else None
                    ),
                    expression_key=expression_key,
                    requested_expression_key=requested_expression_key,
                    asset_id=None,
                    storage_relpath=None,
                    fallback_reason="legacy_character_mood",
                    is_animated=False,
                    content_type=None,
                    asset_url=image_url,
                    role_id=role_id,
                    role_label=role_label,
                    resolution_source="legacy_character_mood",
                )
        return None

    def _legacy_mood_images(self, character: dict[str, Any]) -> dict[str, Any]:
        """Extract supported legacy mood image maps from a character card."""
        extensions = character.get("extensions")
        if isinstance(extensions, dict):
            tldw_extension = extensions.get("tldw")
            if isinstance(tldw_extension, dict):
                for key in ("mood_images", "moodImages"):
                    value = tldw_extension.get(key)
                    if isinstance(value, dict):
                        return value
            for key in ("mood_images", "moodImages"):
                value = extensions.get(key)
                if isinstance(value, dict):
                    return value
        for key in ("mood_images", "moodImages"):
            value = character.get(key)
            if isinstance(value, dict):
                return value
        return {}

    def _validate_actor_for_binding(
        self,
        actor_kind: str,
        actor_id: ActorId,
    ) -> ActorId:
        """Validate that an actor exists and is owned by the current user."""
        if actor_kind == "character":
            try:
                character_id = int(actor_id)
            except (TypeError, ValueError) as exc:
                raise ValueError("visual_identity_character_not_found") from exc
            if self.db.get_character_card_by_id(character_id) is None:
                raise ValueError("visual_identity_character_not_found")
            return character_id

        if actor_kind == "persona":
            persona_id = str(actor_id).strip()
            if not persona_id:
                raise ValueError("visual_identity_persona_not_found")
            persona = self.db.get_persona_profile(
                persona_id,
                user_id=str(self.owner_user_id),
                include_deleted=False,
            )
            if persona is None:
                raise ValueError("visual_identity_persona_not_found")
            return persona_id

        raise ValueError("visual_identity_actor_kind_invalid")

    def _resolved_asset(
        self,
        *,
        actor_kind: str,
        actor_id: ActorId,
        requested_expression_key: str | None,
        binding: dict[str, Any],
        asset: dict[str, Any],
        expression_key: str,
        fallback_reason: str,
        resolution_source: str,
        role_id: str | None,
        role_label: str | None,
    ) -> VisualIdentityResolvedAsset:
        """Build a resolved asset result from a selected version asset row."""
        return self._resolved_version_asset(
            actor_kind=actor_kind,
            actor_id=actor_id,
            requested_expression_key=requested_expression_key,
            pack_id=int(binding["pack_id"]),
            pack_version_id=int(binding["active_version_id"]),
            asset=asset,
            expression_key=expression_key,
            fallback_reason=fallback_reason,
            resolution_source=resolution_source,
            role_id=role_id,
            role_label=role_label,
        )

    def _resolved_version_asset(
        self,
        *,
        actor_kind: str,
        actor_id: ActorId,
        requested_expression_key: str | None,
        pack_id: int,
        pack_version_id: int,
        asset: dict[str, Any],
        expression_key: str,
        fallback_reason: str,
        resolution_source: str,
        role_id: str | None,
        role_label: str | None,
    ) -> VisualIdentityResolvedAsset:
        """Build a resolved asset result from a selected pack version asset row."""
        return VisualIdentityResolvedAsset(
            actor_kind=actor_kind,
            actor_id=actor_id,
            pack_id=pack_id,
            pack_version_id=pack_version_id,
            expression_key=expression_key,
            requested_expression_key=requested_expression_key,
            asset_id=int(asset["id"]),
            storage_relpath=str(asset["storage_relpath"]),
            fallback_reason=fallback_reason,
            is_animated=bool(asset.get("is_animated")),
            content_type=str(asset["content_type"]),
            asset_url=None,
            role_id=role_id,
            role_label=role_label,
            resolution_source=resolution_source,
        )

    def _placeholder(
        self,
        *,
        actor_kind: str,
        actor_id: ActorId,
        requested_expression_key: str | None,
        role_id: str | None = None,
        role_label: str | None = None,
    ) -> VisualIdentityResolvedAsset:
        """Build an explicit placeholder result for unresolved visual identities."""
        return VisualIdentityResolvedAsset(
            actor_kind=actor_kind,
            actor_id=actor_id,
            pack_id=None,
            pack_version_id=None,
            expression_key=None,
            requested_expression_key=requested_expression_key,
            asset_id=None,
            storage_relpath=None,
            fallback_reason="placeholder",
            is_animated=False,
            content_type=None,
            asset_url=None,
            role_id=role_id,
            role_label=role_label,
            resolution_source="placeholder",
        )
