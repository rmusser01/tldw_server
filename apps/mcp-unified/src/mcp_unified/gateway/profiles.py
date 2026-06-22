"""Gateway profile-management helpers for standalone MCP profile stores."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal
from uuid import uuid4

from loguru import logger

from mcp_unified.gateway.profile_governance import (
    AllowPermissionChangeGovernor,
    PermissionChangeDecision,
    PermissionChangeGovernor,
    PermissionChangeRequest,
)
from mcp_unified.interfaces.file_policy_actions import (
    FILE_POLICY_ADMIN_ACTIONS,
    FILE_POLICY_DESTRUCTIVE_ACTIONS,
    FILE_POLICY_EXFILTRATION_ACTIONS,
    FILE_POLICY_LOCK_ACTIONS,
)
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
from mcp_unified.profiles.path_grants import (
    PATH_GRANT_AUTHORING_KEYS,
    compile_policy_path_grants,
)
from mcp_unified.profiles.presets import duplicate_builtin_preset, get_builtin_preset
from mcp_unified.profiles.store import (
    ProfileAlreadyExistsError,
    ProfileAssignmentStoreUnavailableError,
    ProfileStoreUnavailableError,
)
from mcp_unified.storage.models import AuditEvent, ProfileAssignment

_PROFILE_PATCH_FIELDS = frozenset(
    {"name", "description", "enabled", "metadata", "policy_document"}
)
_POLICY_PATCH_FIELDS = frozenset(
    {
        "allowed_tools",
        "denied_tools",
        "capabilities",
        "denied_capabilities",
        "tool_patterns",
        "module_patterns",
        "risk_classes",
        "resource_constraints",
        "path_grants",
        *PATH_GRANT_AUTHORING_KEYS,
    }
)
_PATH_GRANT_POLICY_FIELDS = frozenset({"path_grants", *PATH_GRANT_AUTHORING_KEYS})


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
        permission_governor: PermissionChangeGovernor | None = None,
    ) -> None:
        self.profile_store = profile_store
        self.assignment_store = assignment_store
        self.store_metadata = store_metadata
        self.audit_store = audit_store
        self.fallback_default_profile_id = fallback_default_profile_id
        self.permission_governor = permission_governor or AllowPermissionChangeGovernor()

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
        governance_request = self._permission_request_for_profile(
            "profile.duplicate_from_preset",
            profile,
            changed_fields=("policy_document", "profile"),
        )
        governance_decision = await self._enforce_permission_change(governance_request)
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
                "governance": self._governance_audit_payload(
                    governance_request,
                    governance_decision,
                ),
            },
        )
        return {
            "ok": True,
            "profile": self._dump_profile(stored),
            "store": self.store_metadata.to_payload(),
        }

    async def create_profile(
        self,
        profile_document: MCPProfile | Mapping[str, Any],
    ) -> dict[str, Any]:
        """Create a user-editable gateway profile."""
        try:
            profile = (
                profile_document.model_copy(deep=True)
                if isinstance(profile_document, MCPProfile)
                else MCPProfile.model_validate(profile_document)
            )
        except (TypeError, ValueError) as exc:
            raise self._error(
                "Invalid profile request",
                reason_code="invalid_profile_request",
            ) from exc

        normalized_profile_id = self._require_text(profile.id, field="profile_id")
        normalized_name = self._require_text(profile.name, field="name")
        profile = profile.model_copy(
            update={"id": normalized_profile_id, "name": normalized_name},
            deep=True,
        )

        effective_default_id = await self._effective_default_profile_id()
        if not profile.enabled and profile.id == effective_default_id:
            await self._audit_expected_failure(
                "profile.create_failed",
                reason_code="profile_is_default",
                profile_id=profile.id,
                target_type="profile",
                target_id=profile.id,
            )
            raise self._error(
                f"Profile is the effective default: {profile.id}",
                reason_code="profile_is_default",
                profile_id=profile.id,
            )

        now = datetime.now(timezone.utc)
        profile = profile.model_copy(update={"updated_at": now}, deep=True)
        self._validate_path_grant_policy_payload(
            self._policy_payload(profile.policy_document),
            reason_code="invalid_profile_request",
        )
        governance_request = self._permission_request_for_profile(
            "profile.create",
            profile,
            changed_fields=self._profile_create_fields(profile),
        )
        governance_decision = await self._enforce_permission_change(governance_request)
        try:
            stored = await self.profile_store.create_profile(profile)
        except ProfileAlreadyExistsError as exc:
            await self._audit_expected_failure(
                "profile.create_failed",
                reason_code="profile_already_exists",
                profile_id=exc.profile_id,
                target_type="profile",
                target_id=exc.profile_id,
            )
            raise self._error(
                f"Profile already exists: {exc.profile_id}",
                reason_code="profile_already_exists",
                profile_id=exc.profile_id,
            ) from exc
        except ProfileStoreUnavailableError as exc:
            raise self._error(
                "Profile store unavailable",
                reason_code="profile_store_unavailable",
                profile_id=profile.id,
            ) from exc

        await self._append_audit_event(
            "profile.created",
            profile_id=stored.id,
            target_type="profile",
            target_id=stored.id,
            payload={
                "profile_id": stored.id,
                "governance": self._governance_audit_payload(
                    governance_request,
                    governance_decision,
                ),
            },
        )
        profile_payload = self._dump_profile(stored)
        profile_payload["created_at"] = stored.created_at.isoformat()
        profile_payload["updated_at"] = stored.updated_at.isoformat()
        return {
            "ok": True,
            "profile": profile_payload,
            "store": self.store_metadata.to_payload(),
        }

    async def patch_profile(
        self,
        profile_id: str,
        patch_document: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Apply a limited semantic patch to a stored gateway profile."""
        normalized_profile_id = self._require_text(profile_id, field="profile_id")
        try:
            patch = self._validate_patch_document(patch_document)
        except GatewayProfileManagementError:
            await self._audit_expected_failure(
                "profile.patch_failed",
                reason_code="invalid_profile_patch",
                profile_id=normalized_profile_id,
                target_type="profile",
                target_id=normalized_profile_id,
            )
            raise

        profile = await self._get_profile(normalized_profile_id)
        if profile is None:
            await self._audit_expected_failure(
                "profile.patch_failed",
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

        if (
            patch.get("enabled") is False
            and normalized_profile_id == await self._effective_default_profile_id()
        ):
            await self._audit_expected_failure(
                "profile.patch_failed",
                reason_code="profile_is_default",
                profile_id=normalized_profile_id,
                target_type="profile",
                target_id=normalized_profile_id,
            )
            raise self._error(
                f"Profile is the effective default: {normalized_profile_id}",
                reason_code="profile_is_default",
                profile_id=normalized_profile_id,
            )

        try:
            updated, changed_fields = self._apply_profile_patch(profile, patch)
        except GatewayProfileManagementError:
            await self._audit_expected_failure(
                "profile.patch_failed",
                reason_code="invalid_profile_patch",
                profile_id=normalized_profile_id,
                target_type="profile",
                target_id=normalized_profile_id,
            )
            raise

        governance_request = self._permission_request_for_profile(
            "profile.patch",
            updated,
            changed_fields=changed_fields,
            policy_fields=self._patch_policy_fields(patch),
        )
        governance_decision = await self._enforce_permission_change(governance_request)
        updated = updated.model_copy(
            update={"updated_at": datetime.now(timezone.utc)},
            deep=True,
        )
        try:
            stored = await self.profile_store.upsert_profile(updated)
        except ProfileStoreUnavailableError as exc:
            raise self._error(
                "Profile store unavailable",
                reason_code="profile_store_unavailable",
                profile_id=normalized_profile_id,
            ) from exc

        await self._append_audit_event(
            "profile.patched",
            profile_id=stored.id,
            target_type="profile",
            target_id=stored.id,
            payload={
                "profile_id": stored.id,
                "changed_fields": list(changed_fields),
                "governance": self._governance_audit_payload(
                    governance_request,
                    governance_decision,
                ),
            },
        )
        profile_payload = self._dump_profile(stored)
        profile_payload["created_at"] = stored.created_at.isoformat()
        profile_payload["updated_at"] = stored.updated_at.isoformat()
        return {
            "ok": True,
            "profile": profile_payload,
            "store": self.store_metadata.to_payload(),
        }

    async def delete_profile(self, profile_id: str) -> dict[str, Any]:
        """Delete an unassigned non-default profile."""
        normalized_profile_id = self._require_text(profile_id, field="profile_id")
        effective_default_id = await self._effective_default_profile_id()
        if normalized_profile_id == effective_default_id:
            await self._audit_expected_failure(
                "profile.delete_failed",
                reason_code="profile_is_default",
                profile_id=normalized_profile_id,
                target_type="profile",
                target_id=normalized_profile_id,
            )
            raise self._error(
                f"Profile is the effective default: {normalized_profile_id}",
                reason_code="profile_is_default",
                profile_id=normalized_profile_id,
            )

        profile = await self._get_profile(normalized_profile_id)
        if profile is None:
            await self._audit_expected_failure(
                "profile.delete_failed",
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

        try:
            assignments = await self.assignment_store.list_assignments(
                profile_id=normalized_profile_id,
            )
        except ProfileAssignmentStoreUnavailableError as exc:
            raise self._error(
                "Profile assignment store unavailable",
                reason_code="assignment_store_unavailable",
            ) from exc
        if assignments:
            await self._audit_expected_failure(
                "profile.delete_failed",
                reason_code="profile_has_assignments",
                profile_id=normalized_profile_id,
                target_type="profile",
                target_id=normalized_profile_id,
            )
            raise self._error(
                f"Profile has assignments: {normalized_profile_id}",
                reason_code="profile_has_assignments",
                profile_id=normalized_profile_id,
            )

        governance_request = self._permission_request_for_profile(
            "profile.delete",
            profile,
            changed_fields=("profile",),
        )
        governance_decision = await self._enforce_permission_change(governance_request)
        guarded_delete = getattr(
            self.profile_store,
            "delete_profile_if_unassigned",
            None,
        )
        if callable(guarded_delete):
            try:
                result = await guarded_delete(
                    normalized_profile_id,
                    effective_default_profile_id=effective_default_id,
                )
            except ProfileStoreUnavailableError as exc:
                await self._audit_expected_failure(
                    "profile.delete_failed",
                    reason_code="profile_store_unavailable",
                    profile_id=normalized_profile_id,
                    target_type="profile",
                    target_id=normalized_profile_id,
                )
                raise self._error(
                    "Profile store unavailable",
                    reason_code="profile_store_unavailable",
                    profile_id=normalized_profile_id,
                ) from exc
            except RuntimeError as exc:
                await self._audit_expected_failure(
                    "profile.delete_failed",
                    reason_code="profile_store_unavailable",
                    profile_id=normalized_profile_id,
                    target_type="profile",
                    target_id=normalized_profile_id,
                )
                raise self._error(
                    "Profile store unavailable",
                    reason_code="profile_store_unavailable",
                    profile_id=normalized_profile_id,
                ) from exc
        elif not self.store_metadata.persistent:
            result = await self._manager_guarded_delete(normalized_profile_id)
        else:
            await self._audit_expected_failure(
                "profile.delete_failed",
                reason_code="profile_store_unavailable",
                profile_id=normalized_profile_id,
                target_type="profile",
                target_id=normalized_profile_id,
            )
            raise self._error(
                "Profile store unavailable",
                reason_code="profile_store_unavailable",
                profile_id=normalized_profile_id,
            )

        if result == "deleted":
            await self._append_audit_event(
                "profile.deleted",
                profile_id=normalized_profile_id,
                target_type="profile",
                target_id=normalized_profile_id,
                payload={
                    "profile_id": normalized_profile_id,
                    "governance": self._governance_audit_payload(
                        governance_request,
                        governance_decision,
                    ),
                },
            )
            return {
                "ok": True,
                "profile_id": normalized_profile_id,
                "store": self.store_metadata.to_payload(),
            }
        if result == "not_found":
            await self._audit_expected_failure(
                "profile.delete_failed",
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
        if result == "has_assignments":
            await self._audit_expected_failure(
                "profile.delete_failed",
                reason_code="profile_has_assignments",
                profile_id=normalized_profile_id,
                target_type="profile",
                target_id=normalized_profile_id,
            )
            raise self._error(
                f"Profile has assignments: {normalized_profile_id}",
                reason_code="profile_has_assignments",
                profile_id=normalized_profile_id,
            )
        if result == "is_default":
            await self._audit_expected_failure(
                "profile.delete_failed",
                reason_code="profile_is_default",
                profile_id=normalized_profile_id,
                target_type="profile",
                target_id=normalized_profile_id,
            )
            raise self._error(
                f"Profile is the effective default: {normalized_profile_id}",
                reason_code="profile_is_default",
                profile_id=normalized_profile_id,
            )

        await self._audit_expected_failure(
            "profile.delete_failed",
            reason_code="unexpected_delete_result",
            profile_id=normalized_profile_id,
            target_type="profile",
            target_id=normalized_profile_id,
            details={"delete_result": result},
        )
        raise self._error(
            f"Unexpected guarded delete status {result!r} for profile {normalized_profile_id}",
            reason_code="unexpected_delete_result",
            profile_id=normalized_profile_id,
        )

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
        governance_request = self._permission_request_for_profile(
            "profile.default_change",
            profile,
            changed_fields=("default_profile",),
            target_type="profile_assignment",
            target_id=GATEWAY_DEFAULT_ASSIGNMENT_ID,
            extra_risk_flags=("default_profile_change",),
        )
        governance_decision = await self._enforce_permission_change(governance_request)
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
                "governance": self._governance_audit_payload(
                    governance_request,
                    governance_decision,
                ),
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
        details: Mapping[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {"reason_code": reason_code}
        if profile_id is not None:
            payload["profile_id"] = profile_id
        if preset_id is not None:
            payload["preset_id"] = preset_id
        if assignment_id is not None:
            payload["assignment_id"] = assignment_id
        if details:
            payload.update(details)
        await self._append_audit_event(
            event_type,
            profile_id=profile_id,
            target_type=target_type,
            target_id=target_id,
            payload=payload,
        )

    async def _enforce_permission_change(
        self,
        request: PermissionChangeRequest,
    ) -> PermissionChangeDecision:
        """Evaluate and enforce a profile permission-change governance request."""
        try:
            decision = await self.permission_governor.evaluate_permission_change(request)
            if not isinstance(decision, PermissionChangeDecision):
                raise TypeError("permission governor returned an invalid decision")
        except Exception as exc:  # noqa: BLE001
            logger.opt(exception=True).warning(
                "Gateway profile permission governor failed closed for {action}",
                action=request.action,
                target_type=request.target_type,
                target_id=request.target_id,
            )
            decision = PermissionChangeDecision(
                outcome="deny",
                reason_code="permission_change_governor_error",
            )
            await self._audit_blocked_permission_change(
                request,
                decision,
                reason_code="permission_change_denied",
            )
            raise self._error(
                "Permission change denied",
                reason_code="permission_change_denied",
                profile_id=request.profile_id,
            ) from exc

        if decision.outcome == "allow":
            return decision

        reason_code = (
            "permission_change_requires_approval"
            if decision.outcome == "ask"
            else "permission_change_denied"
        )
        await self._audit_blocked_permission_change(
            request,
            decision,
            reason_code=reason_code,
        )
        message = (
            "Permission change requires approval"
            if decision.outcome == "ask"
            else "Permission change denied"
        )
        raise self._error(
            message,
            reason_code=reason_code,
            profile_id=request.profile_id,
        )

    async def _audit_blocked_permission_change(
        self,
        request: PermissionChangeRequest,
        decision: PermissionChangeDecision,
        *,
        reason_code: str,
    ) -> None:
        payload = self._governance_audit_payload(request, decision)
        payload["reason_code"] = reason_code
        await self._append_audit_event(
            "profile.permission_change_blocked",
            profile_id=request.profile_id,
            target_type=request.target_type,
            target_id=request.target_id,
            payload=payload,
        )

    def _permission_request_for_profile(
        self,
        action: str,
        profile: MCPProfile,
        *,
        changed_fields: tuple[str, ...],
        policy_fields: tuple[str, ...] | None = None,
        target_type: str = "profile",
        target_id: str | None = None,
        extra_risk_flags: tuple[str, ...] = (),
    ) -> PermissionChangeRequest:
        policy_payload = self._policy_payload(profile.policy_document)
        effective_policy_fields = (
            tuple(sorted(policy_fields))
            if policy_fields is not None
            else self._non_empty_policy_fields(policy_payload)
        )
        risk_flags = self._permission_change_risk_flags(
            action=action,
            profile=profile,
            policy_payload=policy_payload,
            policy_fields=effective_policy_fields,
            extra_risk_flags=extra_risk_flags,
        )
        return PermissionChangeRequest(
            action=action,
            profile_id=profile.id,
            target_type=target_type,
            target_id=target_id or profile.id,
            changed_fields=tuple(sorted(set(changed_fields))),
            policy_fields=effective_policy_fields,
            risk_flags=risk_flags,
        )

    def _permission_change_risk_flags(
        self,
        *,
        action: str,
        profile: MCPProfile,
        policy_payload: Mapping[str, Any],
        policy_fields: tuple[str, ...],
        extra_risk_flags: tuple[str, ...],
    ) -> tuple[str, ...]:
        flags = set(extra_risk_flags)
        if profile.enabled and action in {
            "profile.create",
            "profile.duplicate_from_preset",
        }:
            flags.add("profile_enabled")
        if policy_fields:
            flags.add("policy_changed")
        if any(field in policy_fields for field in {"allowed_tools", "tool_patterns", "module_patterns"}):
            flags.add("tool_policy_changed")
        if any(field in policy_fields for field in {"path_grants", *PATH_GRANT_AUTHORING_KEYS}):
            flags.add("path_grants_changed")
            compiled = compile_policy_path_grants(policy_payload)
            allowed_path_grant_actions = {
                str(action)
                for grant in compiled.path_grants
                if grant.get("effect") == "allow"
                for action in grant.get("actions", ())
            }
            if allowed_path_grant_actions.intersection({"edit", "write"}):
                flags.add("path_grants_write_or_edit")
            if allowed_path_grant_actions.intersection(FILE_POLICY_DESTRUCTIVE_ACTIONS):
                flags.add("path_grants_destructive")
            if allowed_path_grant_actions.intersection(FILE_POLICY_EXFILTRATION_ACTIONS):
                flags.add("path_grants_exfiltration")
            if allowed_path_grant_actions.intersection(FILE_POLICY_ADMIN_ACTIONS):
                flags.add("path_grants_admin")
            if allowed_path_grant_actions.intersection(FILE_POLICY_LOCK_ACTIONS):
                flags.add("path_grants_lock")
        if self._has_wildcard_tool_policy(policy_payload):
            flags.add("wildcard_tool_policy")
        return tuple(sorted(flags))

    def _governance_audit_payload(
        self,
        request: PermissionChangeRequest,
        decision: PermissionChangeDecision,
    ) -> dict[str, Any]:
        decision_payload: dict[str, Any] = {
            "outcome": decision.outcome,
            "reason_code": decision.reason_code,
        }
        if decision.metadata:
            decision_payload["metadata_keys"] = sorted(str(key) for key in decision.metadata)
        payload: dict[str, Any] = {
            "action": request.action,
            "target_type": request.target_type,
            "target_id": request.target_id,
            "changed_fields": self._audit_changed_fields(request.changed_fields),
            "policy_fields": list(request.policy_fields),
            "risk_flags": list(request.risk_flags),
            "decision": decision_payload,
        }
        if request.profile_id is not None:
            payload["profile_id"] = request.profile_id
        return payload

    @staticmethod
    def _audit_changed_fields(changed_fields: tuple[str, ...]) -> list[str]:
        return [
            "policy" if field == "policy_document" else field
            for field in changed_fields
        ]

    @classmethod
    def _profile_create_fields(cls, profile: MCPProfile) -> tuple[str, ...]:
        fields = {"name"}
        if profile.description:
            fields.add("description")
        if profile.enabled:
            fields.add("enabled")
        if profile.metadata:
            fields.add("metadata")
        if cls._non_empty_policy_fields(cls._policy_payload(profile.policy_document)):
            fields.add("policy_document")
        return tuple(sorted(fields))

    @staticmethod
    def _patch_policy_fields(patch: Mapping[str, Any]) -> tuple[str, ...]:
        policy_patch = patch.get("policy_document")
        if not isinstance(policy_patch, Mapping):
            return ()
        return tuple(sorted(str(key) for key in policy_patch))

    @staticmethod
    def _non_empty_policy_fields(policy_payload: Mapping[str, Any]) -> tuple[str, ...]:
        return tuple(
            sorted(
                str(key)
                for key, value in policy_payload.items()
                if GatewayProfileManager._has_non_empty_policy_value(value)
            )
        )

    @staticmethod
    def _has_non_empty_policy_value(value: Any) -> bool:
        if value is None or value == "":
            return False
        if isinstance(value, Collection) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            return len(value) > 0
        return True

    @staticmethod
    def _policy_payload(policy_document: object | None) -> dict[str, Any]:
        if policy_document is None:
            return {}
        model_dump = getattr(policy_document, "model_dump", None)
        if callable(model_dump):
            payload = model_dump(mode="json")
            return payload if isinstance(payload, dict) else {}
        dict_method = getattr(policy_document, "dict", None)
        if callable(dict_method):
            payload = dict_method()
            return payload if isinstance(payload, dict) else {}
        if isinstance(policy_document, Mapping):
            return dict(policy_document)
        return {}

    @staticmethod
    def _has_wildcard_tool_policy(policy_payload: Mapping[str, Any]) -> bool:
        for field in ("allowed_tools", "tool_patterns", "module_patterns"):
            values = policy_payload.get(field)
            if isinstance(values, str):
                values = [values]
            if not isinstance(values, (list, tuple, set)):
                continue
            if any(isinstance(value, str) and "*" in value for value in values):
                return True
        return False

    def _validate_path_grant_policy_payload(
        self,
        policy_payload: Mapping[str, Any],
        *,
        reason_code: str,
    ) -> None:
        if not any(field in policy_payload for field in _PATH_GRANT_POLICY_FIELDS):
            return

        raw_path_grants = policy_payload.get("path_grants")
        if "path_grants" in policy_payload and (
            not isinstance(raw_path_grants, list)
            or any(not isinstance(item, Mapping) for item in raw_path_grants)
        ):
            raise self._error("Invalid profile request", reason_code=reason_code)

        compiled_path_grants = compile_policy_path_grants(policy_payload)
        if compiled_path_grants.has_errors:
            raise self._error("Invalid profile request", reason_code=reason_code)

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

    async def _effective_default_profile_id(self) -> str | None:
        assignment = await self._load_default_assignment()
        if assignment is not None:
            return assignment.profile_id
        return self.fallback_default_profile_id

    async def _manager_guarded_delete(self, profile_id: str) -> str:
        profile = await self._get_profile(profile_id)
        if profile is None:
            return "not_found"
        try:
            assignments = await self.assignment_store.list_assignments(
                profile_id=profile_id,
            )
        except ProfileAssignmentStoreUnavailableError as exc:
            raise self._error(
                "Profile assignment store unavailable",
                reason_code="assignment_store_unavailable",
            ) from exc
        if assignments:
            return "has_assignments"
        try:
            deleted = await self.profile_store.delete_profile(profile_id)
        except ProfileStoreUnavailableError as exc:
            raise self._error(
                "Profile store unavailable",
                reason_code="profile_store_unavailable",
                profile_id=profile_id,
            ) from exc
        return "deleted" if deleted else "not_found"

    def _validate_patch_document(self, patch: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(patch, Mapping):
            raise self._error(
                "Invalid profile patch",
                reason_code="invalid_profile_patch",
            )
        normalized = dict(patch)
        if not normalized:
            raise self._error(
                "Invalid profile patch",
                reason_code="invalid_profile_patch",
            )
        unsupported_fields = set(normalized) - _PROFILE_PATCH_FIELDS
        if unsupported_fields:
            raise self._error(
                "Invalid profile patch",
                reason_code="invalid_profile_patch",
            )
        policy_patch = normalized.get("policy_document")
        if "policy_document" in normalized:
            if not isinstance(policy_patch, Mapping):
                raise self._error(
                    "Invalid profile patch",
                    reason_code="invalid_profile_patch",
                )
            policy_patch = dict(policy_patch)
            unsupported_policy_fields = set(policy_patch) - _POLICY_PATCH_FIELDS
            if unsupported_policy_fields:
                raise self._error(
                    "Invalid profile patch",
                    reason_code="invalid_profile_patch",
                )
            self._validate_path_grant_policy_payload(
                policy_patch,
                reason_code="invalid_profile_patch",
            )
            if not policy_patch and set(normalized) == {"policy_document"}:
                raise self._error(
                    "Invalid profile patch",
                    reason_code="invalid_profile_patch",
                )
            normalized["policy_document"] = policy_patch
        if "name" in normalized:
            name = normalized["name"]
            if not isinstance(name, str) or not name.strip():
                raise self._error(
                    "Invalid profile patch",
                    reason_code="invalid_profile_patch",
                )
            normalized["name"] = name.strip()
        return normalized

    def _apply_profile_patch(
        self,
        profile: MCPProfile,
        patch: Mapping[str, Any],
    ) -> tuple[MCPProfile, tuple[str, ...]]:
        before = profile.model_dump(mode="json", exclude={"updated_at"})
        profile_payload = profile.model_dump(mode="python")
        for field, value in patch.items():
            if field == "policy_document":
                policy_payload = (
                    profile.policy_document.model_dump(mode="python")
                    if profile.policy_document is not None
                    else {}
                )
                policy_payload.update(dict(value))
                profile_payload["policy_document"] = policy_payload
            else:
                profile_payload[field] = value
        try:
            updated = MCPProfile.model_validate(profile_payload)
        except (TypeError, ValueError) as exc:
            raise self._error(
                "Invalid profile patch",
                reason_code="invalid_profile_patch",
            ) from exc

        after = updated.model_dump(mode="json", exclude={"updated_at"})
        if after == before:
            raise self._error(
                "Invalid profile patch",
                reason_code="invalid_profile_patch",
            )
        return updated, tuple(sorted(patch))

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
    def _optional_text(value: object | None, *, field: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise GatewayProfileManagementError(
                f"Invalid profile request: {field} must be text",
                reason_code="invalid_profile_request",
            )
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
