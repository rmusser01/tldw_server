"""Gateway config snapshot import/export helpers for standalone MCP stores."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import parse_qsl, urlsplit
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mcp_unified.interfaces.storage import (
    AuditStore,
    CredentialGrantStore,
    ExternalRegistryStore,
    ProfileAssignmentStore,
    ProfileStore,
)
from mcp_unified.profiles.defaults import GATEWAY_DEFAULT_ASSIGNMENT_ID
from mcp_unified.profiles.models import MCPProfile
from mcp_unified.storage.models import (
    AuditEvent,
    CredentialGrant,
    ExternalServerDefinition,
    ProfileAssignment,
)

from .credential_grants import (
    GatewayCredentialGrantManagementError,
    reject_secret_looking_metadata,
)

SNAPSHOT_SCHEMA = "mcp_unified.gateway.config_snapshot"
SNAPSHOT_VERSION = 1
_SECRET_QUERY_MARKERS = ("secret", "token", "password")
_SECRET_QUERY_EXACT_MATCHES = {
    "api_key",
    "authorization",
    "headers",
    "env",
    "credential_value",
}


class GatewayConfigSnapshotManagementError(RuntimeError):
    """Domain error for expected gateway config snapshot failures."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str,
        applied_actions: list[dict[str, str]] | None = None,
        failed_actions: list[dict[str, str]] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.applied_actions = list(applied_actions or [])
        self.failed_actions = list(failed_actions or [])

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe error payload."""

        payload: dict[str, Any] = {
            "ok": False,
            "error": str(self),
            "reason_code": self.reason_code,
        }
        if self.applied_actions:
            payload["applied_actions"] = [dict(action) for action in self.applied_actions]
        if self.failed_actions:
            payload["failed_actions"] = [dict(action) for action in self.failed_actions]
        return payload


class GatewayConfigSnapshot(BaseModel):
    """Versioned gateway config snapshot without embedded secret values."""

    model_config = ConfigDict(extra="forbid")

    schema: Literal["mcp_unified.gateway.config_snapshot"] = SNAPSHOT_SCHEMA
    version: Literal[1] = SNAPSHOT_VERSION
    profiles: list[MCPProfile] = Field(default_factory=list)
    default_assignment: ProfileAssignment | None = None
    external_servers: list[ExternalServerDefinition] = Field(default_factory=list)
    credential_grants: list[CredentialGrant] = Field(default_factory=list)

    @field_validator("profiles", mode="after")
    @classmethod
    def _sort_profiles(cls, value: list[MCPProfile]) -> list[MCPProfile]:
        """Return copy-isolated profiles sorted by id."""

        return [item.model_copy(deep=True) for item in sorted(value, key=lambda item: item.id)]

    @field_validator("external_servers", mode="after")
    @classmethod
    def _sort_external_servers(
        cls,
        value: list[ExternalServerDefinition],
    ) -> list[ExternalServerDefinition]:
        """Return copy-isolated external servers sorted by id."""

        return [item.model_copy(deep=True) for item in sorted(value, key=lambda item: item.id)]

    @field_validator("credential_grants", mode="after")
    @classmethod
    def _sort_credential_grants(
        cls,
        value: list[CredentialGrant],
    ) -> list[CredentialGrant]:
        """Return copy-isolated credential grants sorted by id."""

        return [item.model_copy(deep=True) for item in sorted(value, key=lambda item: item.id)]

    @model_validator(mode="after")
    def _validate_secret_safety(self) -> GatewayConfigSnapshot:
        """Reject snapshot fields that can carry inline secret material."""

        for profile in self.profiles:
            _reject_secret_material(profile.metadata)
            _reject_secret_material(profile.provenance)
        if self.default_assignment is not None:
            if self.default_assignment.id != GATEWAY_DEFAULT_ASSIGNMENT_ID:
                raise ValueError(
                    "Snapshot default assignment id must be gateway default assignment id"
                )
            _reject_secret_material(self.default_assignment.provenance)
        for server in self.external_servers:
            _reject_secret_material(server.metadata)
            _reject_secret_material(server.provenance)
            _reject_external_server_inline_secrets(server)
        for grant in self.credential_grants:
            _reject_secret_material(grant.metadata)
            _reject_secret_material(grant.provenance)
        return self


@dataclass(frozen=True, slots=True)
class _SnapshotAction:
    """One planned snapshot import mutation."""

    action: str
    target_id: str

    def to_payload(self) -> dict[str, str]:
        """Return a JSON-safe action summary."""

        return {"action": self.action, "target_id": self.target_id}


class GatewayConfigSnapshotManager:
    """Export and import gateway config snapshots through store protocols."""

    def __init__(
        self,
        *,
        profile_store: ProfileStore,
        assignment_store: ProfileAssignmentStore,
        external_registry_store: ExternalRegistryStore,
        credential_grant_store: CredentialGrantStore,
        audit_store: AuditStore | None = None,
    ) -> None:
        self.profile_store = profile_store
        self.assignment_store = assignment_store
        self.external_registry_store = external_registry_store
        self.credential_grant_store = credential_grant_store
        self.audit_store = audit_store

    async def export_snapshot(self) -> GatewayConfigSnapshot:
        """Export a deterministic config snapshot from current stores."""

        profiles = await self.profile_store.list_profiles()
        default_assignment = await self.assignment_store.get_assignment(
            GATEWAY_DEFAULT_ASSIGNMENT_ID
        )
        external_servers = await self.external_registry_store.list_server_definitions()
        credential_grants = await self.credential_grant_store.list_grants()
        return GatewayConfigSnapshot(
            profiles=profiles,
            default_assignment=default_assignment,
            external_servers=external_servers,
            credential_grants=credential_grants,
        )

    async def import_snapshot(
        self,
        snapshot_document: GatewayConfigSnapshot | Mapping[str, Any],
        *,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Validate and import a config snapshot with non-destructive upserts.

        The manager validates the full snapshot before the first write. Once
        writes begin, arbitrary injected stores are best-effort rather than
        transactionally atomic, so failures report applied and failed action ids.
        """

        snapshot = self._coerce_snapshot(snapshot_document)
        await self._validate_references(snapshot)
        plan = self._build_plan(snapshot)
        if dry_run:
            return {"ok": True, "dry_run": True, "plan": plan}

        await self._append_audit_event(
            "config_snapshot.import_started",
            payload={"plan": plan},
        )
        applied_actions: list[dict[str, str]] = []
        failed_actions: list[dict[str, str]] = []
        actions = self._mutation_actions(snapshot)
        for action, mutation in actions:
            action_payload = action.to_payload()
            try:
                await mutation()
            except Exception as exc:  # noqa: BLE001
                logger.opt(exception=True).warning(
                    "Gateway config snapshot import action failed",
                    action=action.action,
                    target_id=action.target_id,
                )
                failed_actions.append(
                    {
                        **action_payload,
                        "reason_code": exc.__class__.__name__,
                    }
                )
                await self._append_audit_event(
                    "config_snapshot.import_failed",
                    payload={
                        "applied_actions": applied_actions,
                        "failed_actions": failed_actions,
                    },
                )
                raise GatewayConfigSnapshotManagementError(
                    "Gateway config snapshot import failed",
                    reason_code="config_snapshot_import_failed",
                    applied_actions=applied_actions,
                    failed_actions=failed_actions,
                ) from exc
            applied_actions.append(action_payload)

        await self._append_audit_event(
            "config_snapshot.import_completed",
            payload={
                "applied_actions": applied_actions,
                "counts": plan["counts"],
            },
        )
        return {
            "ok": True,
            "dry_run": False,
            "plan": plan,
            "applied_actions": applied_actions,
            "failed_actions": [],
        }

    @staticmethod
    def _coerce_snapshot(
        snapshot_document: GatewayConfigSnapshot | Mapping[str, Any],
    ) -> GatewayConfigSnapshot:
        if isinstance(snapshot_document, GatewayConfigSnapshot):
            return snapshot_document.model_copy(deep=True)
        try:
            return GatewayConfigSnapshot.model_validate(snapshot_document)
        except Exception as exc:  # noqa: BLE001
            raise GatewayConfigSnapshotManagementError(
                "Invalid gateway config snapshot",
                reason_code="invalid_config_snapshot",
            ) from exc

    async def _validate_references(self, snapshot: GatewayConfigSnapshot) -> None:
        incoming_profile_ids = {profile.id for profile in snapshot.profiles}
        incoming_servers = {server.id: server for server in snapshot.external_servers}
        current_profile_ids = {profile.id for profile in await self.profile_store.list_profiles()}
        current_servers = {
            server.id: server
            for server in await self.external_registry_store.list_server_definitions()
        }

        if snapshot.default_assignment is not None:
            self._require_gateway_default_assignment_id(snapshot.default_assignment.id)
            self._require_profile_reference(
                snapshot.default_assignment.profile_id,
                incoming_profile_ids=incoming_profile_ids,
                current_profile_ids=current_profile_ids,
            )

        for grant in snapshot.credential_grants:
            self._require_profile_reference(
                grant.profile_id,
                incoming_profile_ids=incoming_profile_ids,
                current_profile_ids=current_profile_ids,
            )
            if grant.external_server_id is None:
                continue
            server = incoming_servers.get(grant.external_server_id) or current_servers.get(
                grant.external_server_id
            )
            if server is None:
                raise GatewayConfigSnapshotManagementError(
                    f"External server not found: {grant.external_server_id}",
                    reason_code="config_snapshot_invalid_reference",
                )
            if _credential_slot_missing(server, grant.credential_slot):
                raise GatewayConfigSnapshotManagementError(
                    "Credential grant references a missing external server slot",
                    reason_code="config_snapshot_invalid_reference",
                )

    @staticmethod
    def _require_profile_reference(
        profile_id: str,
        *,
        incoming_profile_ids: set[str],
        current_profile_ids: set[str],
    ) -> None:
        if profile_id in incoming_profile_ids or profile_id in current_profile_ids:
            return
        raise GatewayConfigSnapshotManagementError(
            f"Profile not found: {profile_id}",
            reason_code="config_snapshot_invalid_reference",
        )

    @staticmethod
    def _require_gateway_default_assignment_id(assignment_id: str) -> None:
        if assignment_id == GATEWAY_DEFAULT_ASSIGNMENT_ID:
            return
        raise GatewayConfigSnapshotManagementError(
            "Gateway config snapshot default assignment id is invalid",
            reason_code="invalid_config_snapshot",
        )

    @staticmethod
    def _build_plan(snapshot: GatewayConfigSnapshot) -> dict[str, Any]:
        actions = [action.to_payload() for action, _mutation in _planned_actions(snapshot)]
        return {
            "actions": actions,
            "counts": {
                "credential_grants": len(snapshot.credential_grants),
                "default_assignment": 1 if snapshot.default_assignment is not None else 0,
                "external_servers": len(snapshot.external_servers),
                "profiles": len(snapshot.profiles),
            },
        }

    def _mutation_actions(
        self,
        snapshot: GatewayConfigSnapshot,
    ) -> list[tuple[_SnapshotAction, Any]]:
        actions: list[tuple[_SnapshotAction, Any]] = []
        for profile in snapshot.profiles:
            stored_profile = profile.model_copy(deep=True)
            actions.append(
                (
                    _SnapshotAction("upsert_profile", profile.id),
                    lambda stored_profile=stored_profile: self.profile_store.upsert_profile(
                        stored_profile
                    ),
                )
            )
        if snapshot.default_assignment is not None:
            assignment = snapshot.default_assignment.model_copy(
                update={"id": GATEWAY_DEFAULT_ASSIGNMENT_ID},
                deep=True,
            )
            actions.append(
                (
                    _SnapshotAction(
                        "set_default_assignment",
                        GATEWAY_DEFAULT_ASSIGNMENT_ID,
                    ),
                    lambda assignment=assignment: self.assignment_store.upsert_assignment(
                        assignment
                    ),
                )
            )
        for server in snapshot.external_servers:
            stored_server = server.model_copy(deep=True)
            actions.append(
                (
                    _SnapshotAction("upsert_external_server", server.id),
                    lambda stored_server=stored_server: self.external_registry_store.upsert_server(
                        stored_server
                    ),
                )
            )
        for grant in snapshot.credential_grants:
            stored_grant = grant.model_copy(deep=True)
            actions.append(
                (
                    _SnapshotAction("upsert_credential_grant", grant.id),
                    lambda stored_grant=stored_grant: self.credential_grant_store.upsert_grant(
                        stored_grant
                    ),
                )
            )
        return actions

    async def _append_audit_event(
        self,
        event_type: str,
        *,
        payload: Mapping[str, Any],
    ) -> None:
        if self.audit_store is None:
            return
        event = AuditEvent(
            id=str(uuid4()),
            event_type=event_type,
            target_type="config_snapshot",
            payload=dict(payload),
            provenance={"source": "gateway_config_snapshot_manager"},
        )
        try:
            await self.audit_store.append_event(event)
        except Exception:  # noqa: BLE001
            logger.opt(exception=True).warning(
                "Gateway config snapshot audit append failed",
                event_type=event_type,
            )


def _planned_actions(
    snapshot: GatewayConfigSnapshot,
) -> list[tuple[_SnapshotAction, None]]:
    actions: list[tuple[_SnapshotAction, None]] = [
        (_SnapshotAction("upsert_profile", profile.id), None)
        for profile in snapshot.profiles
    ]
    if snapshot.default_assignment is not None:
        actions.append(
            (
                _SnapshotAction(
                    "set_default_assignment",
                    GATEWAY_DEFAULT_ASSIGNMENT_ID,
                ),
                None,
            )
        )
    actions.extend(
        (_SnapshotAction("upsert_external_server", server.id), None)
        for server in snapshot.external_servers
    )
    actions.extend(
        (_SnapshotAction("upsert_credential_grant", grant.id), None)
        for grant in snapshot.credential_grants
    )
    return actions


def _reject_secret_material(value: Any) -> None:
    try:
        reject_secret_looking_metadata(value)
    except GatewayCredentialGrantManagementError as exc:
        raise ValueError(str(exc)) from exc


def _credential_slot_missing(
    server: ExternalServerDefinition,
    credential_slot: str,
) -> bool:
    slots = server.credential_slots
    if not isinstance(slots, list | tuple | set):
        return True
    return not slots or credential_slot not in slots


def _reject_external_server_inline_secrets(server: ExternalServerDefinition) -> None:
    command = server.command
    if command is None:
        command_parts: tuple[str, ...] | list[str] = ()
    elif not isinstance(command, list | tuple):
        raise ValueError("External server command must be a list of strings")
    else:
        command_parts = command

    for command_part in command_parts:
        if not isinstance(command_part, str):
            raise ValueError("External server command must be a list of strings")
        if _command_part_contains_inline_secret(command_part):
            raise ValueError("External server command must not contain secret material")
    if server.url:
        parsed = urlsplit(server.url)
        if parsed.username or parsed.password:
            raise ValueError("External server URL must not contain secret material")
        for key, _value in parse_qsl(parsed.query, keep_blank_values=True):
            if _is_secret_key(key):
                raise ValueError("External server URL must not contain secret material")


def _command_part_contains_inline_secret(command_part: str) -> bool:
    text = command_part.strip()
    if "=" not in text:
        return False
    key = text.split("=", 1)[0].strip().lstrip("-").replace("-", "_")
    return _is_secret_key(key)


def _is_secret_key(key: str) -> bool:
    key_text = key.strip().lower()
    return key_text in _SECRET_QUERY_EXACT_MATCHES or any(
        marker in key_text for marker in _SECRET_QUERY_MARKERS
    )


__all__ = [
    "GatewayConfigSnapshot",
    "GatewayConfigSnapshotManagementError",
    "GatewayConfigSnapshotManager",
    "SNAPSHOT_SCHEMA",
    "SNAPSHOT_VERSION",
]
