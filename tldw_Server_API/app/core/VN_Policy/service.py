"""VN policy and generation profile business logic."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.VNPolicy_DB import (
    BuiltinVNPolicyProfileStore,
    SyncVNPolicyProfileStoreAdapter,
    VNProfileSnapshotRepository,
    VNPolicyProfileStore,
)

_MATURE_RATINGS = {"suggestive", "mature", "adult", "explicit", "nsfw"}
_KNOWN_METADATA_STATUSES = {
    "adult",
    "minor",
    "missing",
    "unknown_or_ambiguous",
    "conflicting",
    "imported_untrusted",
}


class VNPolicyService:
    """Coordinate VN policy profiles, generation profiles, and decisions."""

    def __init__(
        self,
        db: CharactersRAGDB,
        *,
        owner_user_id: int,
        profile_store: VNPolicyProfileStore | BuiltinVNPolicyProfileStore | SyncVNPolicyProfileStoreAdapter | None = None,
    ):
        self._db = db
        self._snapshot_repo: VNProfileSnapshotRepository | None = None
        self.profile_store = profile_store or BuiltinVNPolicyProfileStore()
        self.owner_user_id = owner_user_id

    @property
    def snapshot_repo(self) -> VNProfileSnapshotRepository:
        """Lazily initialize per-user snapshot storage only when snapshots are needed."""
        if self._snapshot_repo is None:
            self._snapshot_repo = VNProfileSnapshotRepository.initialized(self._db)
        return self._snapshot_repo

    async def list_policy_profiles(self, *, limit: int, offset: int) -> tuple[list[dict[str, Any]], int]:
        """List usable policy profiles."""
        return await self.profile_store.list_policy_profiles(limit=limit, offset=offset)

    async def list_generation_profiles(self, *, limit: int, offset: int) -> tuple[list[dict[str, Any]], int]:
        """List usable generation profiles."""
        return await self.profile_store.list_generation_profiles(limit=limit, offset=offset)

    async def get_policy_profile(self, profile_id: str) -> dict[str, Any] | None:
        """Return a usable policy profile."""
        return await self.profile_store.get_policy_profile(profile_id)

    async def get_generation_profile(self, profile_id: str) -> dict[str, Any] | None:
        """Return a usable generation profile."""
        return await self.profile_store.get_generation_profile(profile_id)

    async def evaluate_character_safety_metadata(
        self,
        *,
        content_rating: str,
        metadata_status: str,
        policy_profile_id: str,
    ) -> dict[str, Any]:
        """Evaluate the character safety metadata portion of a VN policy request."""
        profile = await self.profile_store.get_policy_profile(policy_profile_id)
        if profile is None:
            raise ValueError("policy_profile_not_found")
        return evaluate_character_safety_definition(
            profile_definition=profile["definition"],
            policy_profile_id=policy_profile_id,
            content_rating=content_rating,
            metadata_status=metadata_status,
        )

    async def create_policy_profile(
        self,
        *,
        profile_id: str,
        display_name: str,
        definition: Mapping[str, Any],
        description: str | None,
        created_by_user_id: int | None,
    ) -> dict[str, Any]:
        """Create a global VN policy profile definition."""
        return await self.profile_store.create_policy_profile(
            profile_id=profile_id,
            display_name=display_name,
            description=description,
            definition=definition,
            created_by_user_id=created_by_user_id,
        )

    async def update_policy_profile(
        self,
        profile_id: str,
        *,
        display_name: str | None,
        description: str | None,
        definition: Mapping[str, Any] | None,
        updated_by_user_id: int | None,
    ) -> dict[str, Any]:
        """Update a global VN policy profile definition."""
        return await self.profile_store.update_policy_profile(
            profile_id,
            display_name=display_name,
            description=description,
            definition=definition,
            updated_by_user_id=updated_by_user_id,
        )

    async def disable_policy_profile(self, profile_id: str, *, updated_by_user_id: int | None) -> None:
        """Disable a global VN policy profile definition."""
        await self.profile_store.disable_policy_profile(profile_id, updated_by_user_id=updated_by_user_id)

    async def create_generation_profile(
        self,
        *,
        profile_id: str,
        display_name: str,
        definition: Mapping[str, Any],
        description: str | None,
        created_by_user_id: int | None,
    ) -> dict[str, Any]:
        """Create a global VN generation profile definition."""
        return await self.profile_store.create_generation_profile(
            profile_id=profile_id,
            display_name=display_name,
            description=description,
            definition=definition,
            created_by_user_id=created_by_user_id,
        )

    async def update_generation_profile(
        self,
        profile_id: str,
        *,
        display_name: str | None,
        description: str | None,
        definition: Mapping[str, Any],
        updated_by_user_id: int | None,
    ) -> dict[str, Any]:
        """Update a global VN generation profile definition."""
        return await self.profile_store.update_generation_profile(
            profile_id,
            display_name=display_name,
            description=description,
            definition=definition,
            updated_by_user_id=updated_by_user_id,
        )

    async def disable_generation_profile(self, profile_id: str, *, updated_by_user_id: int | None) -> None:
        """Disable a global VN generation profile definition."""
        await self.profile_store.disable_generation_profile(profile_id, updated_by_user_id=updated_by_user_id)

    async def evaluate(
        self,
        *,
        target_type: str,
        target_id: int | None,
        policy_profile_id: str,
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Evaluate a VN target context without mutating runtime state."""
        raw_owner = context.get("target_owner_user_id")
        if raw_owner is not None:
            try:
                target_owner_user_id = int(raw_owner)
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid_target_owner_user_id") from exc
            if target_owner_user_id != self.owner_user_id:
                raise ValueError("target_not_found")
        if target_id is not None:
            raise ValueError("target_resolution_unavailable")

        character_safety = context.get("character_safety")
        metadata_status = "missing"
        if isinstance(character_safety, Mapping):
            metadata_status = str(character_safety.get("metadata_status") or "missing")
        result = await self.evaluate_character_safety_metadata(
            content_rating=str(context.get("content_rating") or "general"),
            metadata_status=metadata_status,
            policy_profile_id=policy_profile_id,
        )
        result["target_type"] = target_type
        result["target_id"] = target_id
        return result

    async def create_profile_snapshots(
        self,
        *,
        resource_type: str,
        resource_id: int | None,
        policy_profile_id: str,
        generation_profile_id: str,
    ) -> dict[str, int]:
        """Snapshot effective policy and generation profile definitions for a user resource."""
        policy_profile = await self.profile_store.get_policy_profile(policy_profile_id)
        generation_profile = await self.profile_store.get_generation_profile(generation_profile_id)
        if policy_profile is None:
            raise ValueError("policy_profile_not_found")
        if generation_profile is None:
            raise ValueError("generation_profile_not_found")

        policy_snapshot = self.snapshot_repo.create_profile_snapshot(
            owner_user_id=self.owner_user_id,
            snapshot_type="policy",
            profile_id=str(policy_profile["profile_id"]),
            profile_version=int(policy_profile["version"]),
            resource_type=resource_type,
            resource_id=resource_id,
            definition=policy_profile["definition"],
        )
        generation_snapshot = self.snapshot_repo.create_profile_snapshot(
            owner_user_id=self.owner_user_id,
            snapshot_type="generation",
            profile_id=str(generation_profile["profile_id"]),
            profile_version=int(generation_profile["version"]),
            resource_type=resource_type,
            resource_id=resource_id,
            definition=generation_profile["definition"],
        )
        return {
            "policy_snapshot_id": int(policy_snapshot["id"]),
            "generation_snapshot_id": int(generation_snapshot["id"]),
        }


def evaluate_character_safety_definition(
    *,
    profile_definition: Mapping[str, Any],
    policy_profile_id: str,
    content_rating: str,
    metadata_status: str,
) -> dict[str, Any]:
    """Evaluate character safety metadata against an already-resolved policy definition."""
    normalized_status = normalize_character_safety_metadata_status(metadata_status)
    if normalized_status == "adult":
        return _decision_payload(policy_profile_id, "allow", [])
    if normalized_status == "minor" and _is_mature_rating(content_rating):
        return _decision_payload(
            policy_profile_id,
            "block",
            [
                _reason(
                    "character_safety_minor_disallowed",
                    "error",
                    "Mature VN requests require adult character metadata.",
                    requires_acknowledgement=False,
                )
            ],
        )
    if normalized_status == "minor":
        return _decision_payload(policy_profile_id, "allow", [])

    action = _profile_action(profile_definition, normalized_status, content_rating)
    acknowledgement_required = bool(profile_definition.get("acknowledgement_required_for_warnings", True))
    reason = _reason(
        f"character_safety_{normalized_status}",
        "warning" if action == "warn" else "error",
        _message_for_status(normalized_status),
        requires_acknowledgement=action == "warn" and acknowledgement_required,
    )
    return _decision_payload(policy_profile_id, action, [reason])


def _profile_action(profile_definition: Mapping[str, Any], status: str, content_rating: str) -> str:
    character_safety = profile_definition.get("character_safety")
    if not isinstance(character_safety, Mapping):
        return "block"
    status_rules = character_safety.get(status)
    if not isinstance(status_rules, Mapping):
        return "block"
    normalized_rating = str(content_rating or "general").strip().lower()
    raw_action = status_rules.get(normalized_rating) or status_rules.get("default")
    if raw_action is None and _is_mature_rating(normalized_rating):
        raw_action = status_rules.get("mature")
    if raw_action is None:
        raw_action = status_rules.get("general")
    action = str(raw_action or "block").strip().lower()
    return action if action in {"allow", "warn", "block"} else "block"


def _decision_payload(profile_id: str, decision: str, reasons: list[dict[str, Any]]) -> dict[str, Any]:
    blocked = decision == "block"
    requires_acknowledgement = any(bool(reason.get("requires_acknowledgement")) for reason in reasons)
    remediation = [
        "Add character safety metadata or acknowledge the warning for this request."
        for reason in reasons
        if reason.get("requires_acknowledgement")
    ]
    return {
        "decision": decision,
        "profile_id": profile_id,
        "reasons": reasons,
        "blocked": blocked,
        "requires_acknowledgement": requires_acknowledgement,
        "remediation": remediation,
    }


def _reason(
    code: str,
    severity: str,
    message: str,
    *,
    requires_acknowledgement: bool,
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "requires_acknowledgement": requires_acknowledgement,
    }


def normalize_character_safety_metadata_status(status: str) -> str:
    """Normalize character safety metadata labels into policy matrix statuses."""
    normalized = str(status or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"unknown", "ambiguous", "unknown_ambiguous"}:
        normalized = "unknown_or_ambiguous"
    if normalized in {"imported_untrusted", "untrusted_import", "imported_without_trusted_provenance"}:
        normalized = "imported_untrusted"
    if normalized not in _KNOWN_METADATA_STATUSES:
        return "unknown_or_ambiguous"
    return normalized


def _is_mature_rating(content_rating: str) -> bool:
    return str(content_rating or "").strip().lower() in _MATURE_RATINGS


def _message_for_status(status: str) -> str:
    messages = {
        "missing": "Character safety metadata is missing.",
        "unknown_or_ambiguous": "Character safety metadata is unknown or ambiguous.",
        "conflicting": "Character safety metadata is conflicting.",
        "imported_untrusted": "Imported character safety metadata is not trusted.",
    }
    return messages.get(status, "Character safety metadata requires review.")
