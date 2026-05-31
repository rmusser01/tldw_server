"""Gateway profile-management helpers for standalone MCP profile stores."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal
from uuid import uuid4

from loguru import logger

from mcp_unified.interfaces.storage import (
    AuditStore,
    ProfileAssignmentStore,
    ProfileStore,
)
from mcp_unified.profiles.defaults import (
    GATEWAY_DEFAULT_ASSIGNMENT_ID,
    load_gateway_default_assignment,
)
from mcp_unified.profiles.models import MCPProfile
from mcp_unified.profiles.presets import duplicate_builtin_preset, get_builtin_preset
from mcp_unified.profiles.store import (
    ProfileAssignmentStoreUnavailableError,
    ProfileStoreUnavailableError,
)
from mcp_unified.storage.models import AuditEvent, ProfileAssignment


@dataclass(frozen=True, slots=True)
class GatewayProfileStoreMetadata:
    """User-facing metadata describing the active profile store."""

    kind: Literal["memory", "sqlite"]
    persistent: bool

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-safe store metadata payload."""
        return {"kind": self.kind, "persistent": self.persistent}


class GatewayProfileManagementError(RuntimeError):
    """Domain error for expected gateway profile-management failures."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str,
        profile_id: str | None = None,
        preset_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.profile_id = profile_id
        self.preset_id = preset_id

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe error payload."""
        payload: dict[str, Any] = {
            "ok": False,
            "error": str(self),
            "reason_code": self.reason_code,
        }
        if self.profile_id is not None:
            payload["profile_id"] = self.profile_id
        if self.preset_id is not None:
            payload["preset_id"] = self.preset_id
        return payload


class GatewayProfileManager:
    """Manage editable gateway profiles, presets, and default assignments."""

    def __init__(
        self,
        *,
        profile_store: ProfileStore,
        assignment_store: ProfileAssignmentStore,
        store_metadata: GatewayProfileStoreMetadata,
        audit_store: AuditStore | None = None,
        fallback_default_profile_id: str | None = None,
    ) -> None:
        self.profile_store = profile_store
        self.assignment_store = assignment_store
        self.store_metadata = store_metadata
        self.audit_store = audit_store
        self.fallback_default_profile_id = fallback_default_profile_id

    async def list_profiles(self) -> dict[str, Any]:
        """List all profiles with store metadata."""
        try:
            profiles = await self.profile_store.list_profiles()
        except ProfileStoreUnavailableError as exc:
            raise self._error(
                "Profile store unavailable",
                reason_code="profile_store_unavailable",
            ) from exc
        return {
            "ok": True,
            "profiles": [self._dump_profile(profile) for profile in profiles],
            "store": self.store_metadata.to_payload(),
        }

    async def show_profile(self, profile_id: str) -> dict[str, Any]:
        """Return one profile by id with store metadata."""
        normalized_profile_id = self._require_text(
            profile_id,
            field="profile_id",
        )
        profile = await self._get_profile(normalized_profile_id)
        if profile is None:
            await self._audit_expected_failure(
                "profile.show_failed",
                reason_code="profile_not_found",
                profile_id=normalized_profile_id,
                target_type="profile",
                target_id=normalized_profile_id,
            )
            raise self._error(
                f"Profile not found: {normalized_profile_id}",
                reason_code="profile_not_found",
                profile_id=normalized_profile_id,
            )
        return {
            "ok": True,
            "profile": self._dump_profile(profile),
            "store": self.store_metadata.to_payload(),
        }

    async def duplicate_preset(
        self,
        preset_id: str,
        *,
        profile_id: str | None = None,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Duplicate a built-in preset into the editable profile store."""
        normalized_preset_id = self._require_text(preset_id, field="preset_id")
        normalized_profile_id = self._optional_text(profile_id, field="profile_id")
        normalized_name = self._optional_text(name, field="name")

        preset = get_builtin_preset(normalized_preset_id)
        if preset is None:
            await self._audit_expected_failure(
                "profile.duplication_failed",
                reason_code="preset_not_found",
                preset_id=normalized_preset_id,
                target_type="profile_preset",
                target_id=normalized_preset_id,
            )
            raise self._error(
                f"Preset not found: {normalized_preset_id}",
                reason_code="preset_not_found",
                preset_id=normalized_preset_id,
            )

        target_profile_id = normalized_profile_id or preset.id
        if await self._get_profile(target_profile_id) is not None:
            await self._audit_expected_failure(
                "profile.duplication_failed",
                reason_code="profile_already_exists",
                profile_id=target_profile_id,
                preset_id=normalized_preset_id,
                target_type="profile",
                target_id=target_profile_id,
            )
            raise self._error(
                f"Profile already exists: {target_profile_id}",
                reason_code="profile_already_exists",
                profile_id=target_profile_id,
                preset_id=normalized_preset_id,
            )

        profile = duplicate_builtin_preset(
            normalized_preset_id,
            profile_id=target_profile_id,
            name=normalized_name,
        )
        try:
            stored = await self.profile_store.upsert_profile(profile)
        except ProfileStoreUnavailableError as exc:
            raise self._error(
                "Profile store unavailable",
                reason_code="profile_store_unavailable",
            ) from exc

        await self._append_audit_event(
            "profile.duplicated_from_preset",
            profile_id=stored.id,
            target_type="profile",
            target_id=stored.id,
            payload={
                "profile_id": stored.id,
                "preset_id": stored.preset_id,
                "preset_version": stored.preset_version,
            },
        )
        return {
            "ok": True,
            "profile": self._dump_profile(stored),
            "store": self.store_metadata.to_payload(),
        }

    async def get_default_profile(self) -> dict[str, Any]:
        """Return the active gateway default profile."""
        assignment = await self._load_default_assignment()
        default_source = "assignment" if assignment is not None else None
        resolved_profile_id = assignment.profile_id if assignment is not None else None
        if resolved_profile_id is None and self.fallback_default_profile_id is not None:
            resolved_profile_id = self.fallback_default_profile_id
            default_source = "fallback_default_profile_id"

        if resolved_profile_id is None:
            raise self._error(
                "Default profile is not configured",
                reason_code="default_profile_not_configured",
            )

        profile = await self._get_profile(resolved_profile_id)
        if profile is None:
            await self._audit_expected_failure(
                "profile.default_read_failed",
                reason_code="profile_not_found",
                profile_id=resolved_profile_id,
                assignment_id=assignment.id if assignment is not None else None,
                target_type="profile_assignment" if assignment is not None else "profile",
                target_id=assignment.id if assignment is not None else resolved_profile_id,
            )
            raise self._error(
                f"Profile not found: {resolved_profile_id}",
                reason_code="profile_not_found",
                profile_id=resolved_profile_id,
            )
        if not profile.enabled:
            await self._audit_expected_failure(
                "profile.default_read_failed",
                reason_code="profile_disabled",
                profile_id=resolved_profile_id,
                assignment_id=assignment.id if assignment is not None else None,
                target_type="profile_assignment" if assignment is not None else "profile",
                target_id=assignment.id if assignment is not None else resolved_profile_id,
            )
            raise self._error(
                f"Profile disabled: {resolved_profile_id}",
                reason_code="profile_disabled",
                profile_id=resolved_profile_id,
            )

        return {
            "ok": True,
            "profile": self._dump_profile(profile),
            "assignment": self._dump_assignment(assignment),
            "default": {
                "source": default_source,
                "profile_id": resolved_profile_id,
                "assignment_id": assignment.id if assignment is not None else None,
            },
            "store": self.store_metadata.to_payload(),
        }

    async def set_default_profile(self, profile_id: str) -> dict[str, Any]:
        """Set the gateway default profile assignment."""
        normalized_profile_id = self._require_text(
            profile_id,
            field="profile_id",
        )
        profile = await self._get_profile(normalized_profile_id)
        if profile is None:
            await self._audit_expected_failure(
                "profile.default_change_failed",
                reason_code="profile_not_found",
                profile_id=normalized_profile_id,
                target_type="profile",
                target_id=normalized_profile_id,
            )
            raise self._error(
                f"Profile not found: {normalized_profile_id}",
                reason_code="profile_not_found",
                profile_id=normalized_profile_id,
            )
        if not profile.enabled:
            await self._audit_expected_failure(
                "profile.default_change_failed",
                reason_code="profile_disabled",
                profile_id=normalized_profile_id,
                target_type="profile",
                target_id=normalized_profile_id,
            )
            raise self._error(
                f"Profile disabled: {normalized_profile_id}",
                reason_code="profile_disabled",
                profile_id=normalized_profile_id,
            )

        existing = await self._get_assignment(GATEWAY_DEFAULT_ASSIGNMENT_ID)
        current_default = await self._load_default_assignment()
        now = datetime.now(timezone.utc)
        if current_default is not None and current_default.updated_at >= now:
            now = current_default.updated_at + timedelta(microseconds=1)
        assignment = ProfileAssignment(
            id=GATEWAY_DEFAULT_ASSIGNMENT_ID,
            profile_id=normalized_profile_id,
            is_default=True,
            enabled=True,
            provenance={
                "source": "gateway_profile_manager",
                "profile_id": normalized_profile_id,
            },
            created_at=existing.created_at if existing is not None else now,
            updated_at=now,
        )
        try:
            stored_assignment = await self.assignment_store.upsert_assignment(assignment)
        except ProfileAssignmentStoreUnavailableError as exc:
            raise self._error(
                "Profile assignment store unavailable",
                reason_code="assignment_store_unavailable",
            ) from exc

        await self._append_audit_event(
            "profile.default_changed",
            profile_id=normalized_profile_id,
            target_type="profile_assignment",
            target_id=stored_assignment.id,
            payload={
                "profile_id": normalized_profile_id,
                "assignment_id": stored_assignment.id,
                "previous_profile_id": existing.profile_id if existing is not None else None,
            },
        )
        return {
            "ok": True,
            "profile": self._dump_profile(profile),
            "assignment": self._dump_assignment(stored_assignment),
            "default": {
                "source": "assignment",
                "profile_id": normalized_profile_id,
                "assignment_id": stored_assignment.id,
            },
            "store": self.store_metadata.to_payload(),
        }

    async def _append_audit_event(
        self,
        event_type: str,
        *,
        profile_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        """Append an audit event when an audit store is configured."""
        if self.audit_store is None:
            return
        event = AuditEvent(
            id=f"gateway-profile-{uuid4().hex}",
            event_type=event_type,
            profile_id=profile_id,
            target_type=target_type,
            target_id=target_id,
            payload=dict(payload or {}),
            provenance={"source": "gateway_profile_manager"},
        )
        try:
            await self.audit_store.append_event(event)
        except Exception:  # noqa: BLE001
            # Audit logging is best-effort and must not fail profile mutations.
            logger.opt(exception=True).warning(
                "Gateway profile audit append failed for {event_type}",
                event_type=event_type,
                target_type=target_type,
                target_id=target_id,
            )

    async def _audit_expected_failure(
        self,
        event_type: str,
        *,
        reason_code: str,
        profile_id: str | None = None,
        preset_id: str | None = None,
        assignment_id: str | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {"reason_code": reason_code}
        if profile_id is not None:
            payload["profile_id"] = profile_id
        if preset_id is not None:
            payload["preset_id"] = preset_id
        if assignment_id is not None:
            payload["assignment_id"] = assignment_id
        await self._append_audit_event(
            event_type,
            profile_id=profile_id,
            target_type=target_type,
            target_id=target_id,
            payload=payload,
        )

    async def _get_profile(self, profile_id: str) -> MCPProfile | None:
        try:
            return await self.profile_store.get_profile(profile_id)
        except ProfileStoreUnavailableError as exc:
            raise self._error(
                "Profile store unavailable",
                reason_code="profile_store_unavailable",
                profile_id=profile_id,
            ) from exc

    async def _get_assignment(
        self,
        assignment_id: str,
    ) -> ProfileAssignment | None:
        try:
            return await self.assignment_store.get_assignment(assignment_id)
        except ProfileAssignmentStoreUnavailableError as exc:
            raise self._error(
                "Profile assignment store unavailable",
                reason_code="assignment_store_unavailable",
            ) from exc

    async def _load_default_assignment(self) -> ProfileAssignment | None:
        try:
            return await load_gateway_default_assignment(self.assignment_store)
        except ProfileAssignmentStoreUnavailableError as exc:
            raise self._error(
                "Profile assignment store unavailable",
                reason_code="assignment_store_unavailable",
            ) from exc

    @staticmethod
    def _dump_profile(profile: MCPProfile) -> dict[str, Any]:
        return profile.model_dump(mode="json")

    @staticmethod
    def _dump_assignment(
        assignment: ProfileAssignment | None,
    ) -> dict[str, Any] | None:
        if assignment is None:
            return None
        return assignment.model_dump(mode="json")

    @classmethod
    def _require_text(cls, value: str, *, field: str) -> str:
        normalized = cls._optional_text(value, field=field)
        if normalized is None:
            raise cls._error(
                f"Invalid profile request: {field} is required",
                reason_code="invalid_profile_request",
            )
        return normalized

    @staticmethod
    def _optional_text(value: str | None, *, field: str) -> str | None:
        del field
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise GatewayProfileManagementError(
                "Invalid profile request",
                reason_code="invalid_profile_request",
            )
        return normalized

    @staticmethod
    def _error(
        message: str,
        *,
        reason_code: str,
        profile_id: str | None = None,
        preset_id: str | None = None,
    ) -> GatewayProfileManagementError:
        return GatewayProfileManagementError(
            message,
            reason_code=reason_code,
            profile_id=profile_id,
            preset_id=preset_id,
        )
