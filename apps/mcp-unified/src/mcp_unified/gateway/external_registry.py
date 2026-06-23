"""Gateway external registry management helpers for standalone MCP stores."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import urlparse
from uuid import uuid4

from loguru import logger

from mcp_unified.interfaces.storage import (
    AuditStore,
    CredentialGrantStore,
    ExternalRegistryStore,
    ExternalRegistryStoreUnavailableError,
    ExternalServerAlreadyExistsError,
)
from mcp_unified.storage.models import AuditEvent, ExternalServerDefinition

_SERVER_ID_RE = re.compile(r"^[a-z0-9_-]+$")
_PATCH_FIELDS = frozenset(
    {
        "name",
        "transport",
        "command",
        "url",
        "cwd",
        "env_allowlist",
        "credential_slots",
        "metadata",
        "provenance",
        "enabled",
        "auto_start",
    }
)
_TEXT_LIST_FIELDS = ("command", "env_allowlist", "credential_slots")


@dataclass(frozen=True, slots=True)
class GatewayStoreMetadata:
    """User-facing metadata describing a gateway management store."""

    kind: Literal["memory", "sqlite"]
    persistent: bool

    def to_payload(self) -> dict[str, Any]:
        return {"kind": self.kind, "persistent": self.persistent}


class GatewayExternalRegistryManagementError(RuntimeError):
    """Domain error for expected gateway external-registry failures."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str,
        server_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.server_id = server_id

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe error payload."""
        payload: dict[str, Any] = {
            "ok": False,
            "error": str(self),
            "reason_code": self.reason_code,
        }
        if self.server_id is not None:
            payload["server_id"] = self.server_id
        return payload


class GatewayExternalRegistryManager:
    """Manage stored external MCP server definitions for the package gateway."""

    def __init__(
        self,
        *,
        external_registry_store: ExternalRegistryStore,
        store_metadata: GatewayStoreMetadata,
        credential_grant_store: CredentialGrantStore | None = None,
        audit_store: AuditStore | None = None,
    ) -> None:
        self.external_registry_store = external_registry_store
        self.store_metadata = store_metadata
        self.credential_grant_store = credential_grant_store
        self.audit_store = audit_store

    async def list_servers(self, enabled: bool | None = None) -> dict[str, Any]:
        """List external server definitions with store metadata."""
        try:
            servers = await self.external_registry_store.list_server_definitions(
                enabled=enabled,
            )
        except ExternalRegistryStoreUnavailableError as exc:
            raise self._error(
                "External registry store unavailable",
                reason_code="external_registry_store_unavailable",
            ) from exc
        return {
            "ok": True,
            "servers": [
                self._dump_server(server)
                for server in sorted(servers, key=lambda item: item.id)
            ],
            "store": self.store_metadata.to_payload(),
        }

    async def show_server(self, server_id: str) -> dict[str, Any]:
        """Return one external server definition by id with store metadata."""
        try:
            normalized_server_id = self._normalize_server_id(
                server_id,
                reason_code="invalid_external_server_request",
            )
            server = await self._get_server(normalized_server_id)
        except GatewayExternalRegistryManagementError as exc:
            await self._audit_expected_failure(
                "external_server.show_failed",
                reason_code=exc.reason_code,
                server_id=exc.server_id,
                target_id=exc.server_id,
            )
            raise
        if server is None:
            await self._audit_expected_failure(
                "external_server.show_failed",
                reason_code="external_server_not_found",
                server_id=normalized_server_id,
                target_id=normalized_server_id,
            )
            raise self._error(
                f"External server not found: {normalized_server_id}",
                reason_code="external_server_not_found",
                server_id=normalized_server_id,
            )
        return {
            "ok": True,
            "server": self._dump_server(server),
            "store": self.store_metadata.to_payload(),
        }

    async def create_server(
        self,
        server_document: ExternalServerDefinition | Mapping[str, Any],
    ) -> dict[str, Any]:
        """Create a stored external MCP server definition."""
        server_id_for_audit: str | None = None
        try:
            server = self._coerce_create_document(server_document)
            server_id_for_audit = server.id
            self._validate_enabled_websocket_url(
                server,
                reason_code="invalid_external_server_request",
            )
        except GatewayExternalRegistryManagementError as exc:
            await self._audit_expected_failure(
                "external_server.create_failed",
                reason_code=exc.reason_code,
                server_id=exc.server_id,
                target_id=exc.server_id,
            )
            raise

        server = server.model_copy(
            update={"updated_at": datetime.now(timezone.utc)},
            deep=True,
        )
        try:
            stored = await self.external_registry_store.create_server(server)
        except ExternalServerAlreadyExistsError as exc:
            await self._audit_expected_failure(
                "external_server.create_failed",
                reason_code="external_server_already_exists",
                server_id=exc.server_id,
                target_id=exc.server_id,
            )
            raise self._error(
                f"External server already exists: {exc.server_id}",
                reason_code="external_server_already_exists",
                server_id=exc.server_id,
            ) from exc
        except ExternalRegistryStoreUnavailableError as exc:
            await self._audit_expected_failure(
                "external_server.create_failed",
                reason_code="external_registry_store_unavailable",
                server_id=server_id_for_audit,
                target_id=server_id_for_audit,
            )
            raise self._error(
                "External registry store unavailable",
                reason_code="external_registry_store_unavailable",
                server_id=server_id_for_audit,
            ) from exc

        await self._append_audit_event(
            "external_server.created",
            target_id=stored.id,
            payload={"server_id": stored.id},
        )
        return {
            "ok": True,
            "server": self._dump_server(stored),
            "store": self.store_metadata.to_payload(),
        }

    async def patch_server(
        self,
        server_id: str,
        patch_document: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Apply a replace-style semantic patch to an external server definition."""
        try:
            normalized_server_id = self._normalize_server_id(
                server_id,
                reason_code="invalid_external_server_patch",
            )
            patch = self._validate_patch_document(patch_document)
        except GatewayExternalRegistryManagementError as exc:
            await self._audit_expected_failure(
                "external_server.patch_failed",
                reason_code=exc.reason_code,
                server_id=exc.server_id,
                target_id=exc.server_id,
            )
            raise

        server = await self._get_server_for_mutation(
            normalized_server_id,
            event_type="external_server.patch_failed",
        )
        try:
            updated, changed_fields = self._apply_patch(server, patch)
            self._validate_enabled_websocket_url(
                updated,
                reason_code="invalid_external_server_patch",
            )
            await self._validate_credential_slot_patch(server, updated, patch)
        except GatewayExternalRegistryManagementError as exc:
            await self._audit_expected_failure(
                "external_server.patch_failed",
                reason_code=exc.reason_code,
                server_id=normalized_server_id,
                target_id=normalized_server_id,
            )
            raise

        updated = updated.model_copy(
            update={"updated_at": datetime.now(timezone.utc)},
            deep=True,
        )
        try:
            stored = await self.external_registry_store.update_server(updated)
        except ExternalRegistryStoreUnavailableError as exc:
            await self._audit_expected_failure(
                "external_server.patch_failed",
                reason_code="external_registry_store_unavailable",
                server_id=normalized_server_id,
                target_id=normalized_server_id,
            )
            raise self._error(
                "External registry store unavailable",
                reason_code="external_registry_store_unavailable",
                server_id=normalized_server_id,
            ) from exc

        if stored is None:
            await self._audit_expected_failure(
                "external_server.patch_failed",
                reason_code="external_server_not_found",
                server_id=normalized_server_id,
                target_id=normalized_server_id,
            )
            raise self._error(
                f"External server not found: {normalized_server_id}",
                reason_code="external_server_not_found",
                server_id=normalized_server_id,
            )

        await self._append_audit_event(
            "external_server.patched",
            target_id=stored.id,
            payload={
                "server_id": stored.id,
                "changed_fields": list(changed_fields),
            },
        )
        return {
            "ok": True,
            "server": self._dump_server(stored),
            "store": self.store_metadata.to_payload(),
        }

    async def delete_server(self, server_id: str) -> dict[str, Any]:
        """Delete an external server definition when credential grants allow it."""
        try:
            normalized_server_id = self._normalize_server_id(
                server_id,
                reason_code="invalid_external_server_request",
            )
        except GatewayExternalRegistryManagementError as exc:
            await self._audit_expected_failure(
                "external_server.delete_failed",
                reason_code=exc.reason_code,
                server_id=exc.server_id,
                target_id=exc.server_id,
            )
            raise

        server = await self._get_server_for_mutation(
            normalized_server_id,
            event_type="external_server.delete_failed",
        )
        del server
        try:
            await self._require_no_enabled_credential_grants(
                normalized_server_id,
                missing_store_reason_code="credential_grant_store_unavailable",
            )
        except GatewayExternalRegistryManagementError as exc:
            await self._audit_expected_failure(
                "external_server.delete_failed",
                reason_code=exc.reason_code,
                server_id=normalized_server_id,
                target_id=normalized_server_id,
            )
            raise

        try:
            deleted = await self.external_registry_store.delete_server(
                normalized_server_id,
            )
        except ExternalRegistryStoreUnavailableError as exc:
            await self._audit_expected_failure(
                "external_server.delete_failed",
                reason_code="external_registry_store_unavailable",
                server_id=normalized_server_id,
                target_id=normalized_server_id,
            )
            raise self._error(
                "External registry store unavailable",
                reason_code="external_registry_store_unavailable",
                server_id=normalized_server_id,
            ) from exc

        if not deleted:
            await self._audit_expected_failure(
                "external_server.delete_failed",
                reason_code="external_server_not_found",
                server_id=normalized_server_id,
                target_id=normalized_server_id,
            )
            raise self._error(
                f"External server not found: {normalized_server_id}",
                reason_code="external_server_not_found",
                server_id=normalized_server_id,
            )

        await self._append_audit_event(
            "external_server.deleted",
            target_id=normalized_server_id,
            payload={"server_id": normalized_server_id},
        )
        return {
            "ok": True,
            "server_id": normalized_server_id,
            "store": self.store_metadata.to_payload(),
        }

    def _coerce_create_document(
        self,
        server_document: ExternalServerDefinition | Mapping[str, Any],
    ) -> ExternalServerDefinition:
        try:
            server = (
                server_document.model_copy(deep=True)
                if isinstance(server_document, ExternalServerDefinition)
                else ExternalServerDefinition.model_validate(server_document)
            )
            normalized_server_id = self._normalize_server_id(
                server.id,
                reason_code="invalid_external_server_request",
            )
            normalized_name = self._require_text(
                server.name,
                field="name",
                reason_code="invalid_external_server_request",
                server_id=normalized_server_id,
            )
            return self._normalize_server_lists(
                server.model_copy(
                    update={"id": normalized_server_id, "name": normalized_name},
                    deep=True,
                )
            )
        except GatewayExternalRegistryManagementError:
            raise
        except (TypeError, ValueError) as exc:
            raise self._error(
                "Invalid external server request",
                reason_code="invalid_external_server_request",
            ) from exc

    def _validate_patch_document(self, patch_document: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(patch_document, Mapping):
            raise self._error(
                "Invalid external server patch",
                reason_code="invalid_external_server_patch",
            )
        patch = dict(patch_document)
        unsupported_fields = set(patch) - _PATCH_FIELDS
        if not patch or unsupported_fields:
            raise self._error(
                "Invalid external server patch",
                reason_code="invalid_external_server_patch",
            )
        if "name" in patch:
            patch["name"] = self._require_text(
                patch["name"],
                field="name",
                reason_code="invalid_external_server_patch",
            )
        return patch

    def _apply_patch(
        self,
        server: ExternalServerDefinition,
        patch: Mapping[str, Any],
    ) -> tuple[ExternalServerDefinition, tuple[str, ...]]:
        payload = server.model_dump(mode="python")
        payload.update(dict(patch))
        try:
            updated = ExternalServerDefinition.model_validate(payload)
        except (TypeError, ValueError) as exc:
            raise self._error(
                "Invalid external server patch",
                reason_code="invalid_external_server_patch",
                server_id=server.id,
            ) from exc
        return self._normalize_server_lists(updated), tuple(sorted(patch))

    async def _validate_credential_slot_patch(
        self,
        current: ExternalServerDefinition,
        updated: ExternalServerDefinition,
        patch: Mapping[str, Any],
    ) -> None:
        if "credential_slots" not in patch:
            return
        current_slots = set(self._normalize_text_list(current.credential_slots))
        updated_slots = set(self._normalize_text_list(updated.credential_slots))
        if current_slots.issubset(updated_slots):
            return
        if updated.enabled:
            raise self._error(
                f"Credential slot relaxation requires disabling server: {current.id}",
                reason_code="credential_slot_change_requires_disabled_server",
                server_id=current.id,
            )
        await self._require_no_enabled_credential_grants(
            current.id,
            missing_store_reason_code="credential_grant_store_unavailable",
        )

    async def _require_no_enabled_credential_grants(
        self,
        server_id: str,
        *,
        missing_store_reason_code: str,
    ) -> None:
        if self.credential_grant_store is None:
            raise self._error(
                "Credential grant store unavailable",
                reason_code=missing_store_reason_code,
                server_id=server_id,
            )
        try:
            grants = await self.credential_grant_store.list_grants(
                external_server_id=server_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.opt(exception=True).warning(
                "Gateway external registry credential grant lookup failed",
                server_id=server_id,
            )
            raise self._error(
                "Credential grant store unavailable",
                reason_code=missing_store_reason_code,
                server_id=server_id,
            ) from exc
        if any(grant.enabled for grant in grants):
            raise self._error(
                f"External server has credential grants: {server_id}",
                reason_code="external_server_has_credential_grants",
                server_id=server_id,
            )

    def _validate_enabled_websocket_url(
        self,
        server: ExternalServerDefinition,
        *,
        reason_code: str,
    ) -> None:
        if not server.enabled or server.transport != "websocket":
            return
        scheme = urlparse(server.url or "").scheme
        if scheme not in {"ws", "wss"}:
            raise self._error(
                "Enabled websocket external servers require ws:// or wss:// URLs",
                reason_code=reason_code,
                server_id=server.id,
            )

    def _normalize_server_lists(
        self,
        server: ExternalServerDefinition,
    ) -> ExternalServerDefinition:
        updates = {
            field: self._normalize_text_list(getattr(server, field))
            for field in _TEXT_LIST_FIELDS
        }
        return server.model_copy(update=updates, deep=True)

    @staticmethod
    def _normalize_text_list(values: list[str]) -> list[str]:
        return [item.strip() for item in values if item.strip()]

    async def _get_server(
        self,
        server_id: str,
    ) -> ExternalServerDefinition | None:
        try:
            return await self.external_registry_store.get_server(server_id)
        except ExternalRegistryStoreUnavailableError as exc:
            raise self._error(
                "External registry store unavailable",
                reason_code="external_registry_store_unavailable",
                server_id=server_id,
            ) from exc

    async def _get_server_for_mutation(
        self,
        server_id: str,
        *,
        event_type: str,
    ) -> ExternalServerDefinition:
        try:
            server = await self._get_server(server_id)
        except GatewayExternalRegistryManagementError as exc:
            await self._audit_expected_failure(
                event_type,
                reason_code=exc.reason_code,
                server_id=server_id,
                target_id=server_id,
            )
            raise
        if server is None:
            await self._audit_expected_failure(
                event_type,
                reason_code="external_server_not_found",
                server_id=server_id,
                target_id=server_id,
            )
            raise self._error(
                f"External server not found: {server_id}",
                reason_code="external_server_not_found",
                server_id=server_id,
            )
        return server

    async def _append_audit_event(
        self,
        event_type: str,
        *,
        target_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        """Append an audit event when an audit store is configured."""
        if self.audit_store is None:
            return
        event = AuditEvent(
            id=str(uuid4()),
            event_type=event_type,
            target_type="external_server" if target_id is not None else None,
            target_id=target_id,
            payload=dict(payload or {}),
            provenance={"source": "gateway_external_registry_manager"},
            created_at=datetime.now(timezone.utc),
        )
        try:
            await self.audit_store.append_event(event)
        except Exception:  # noqa: BLE001
            # Audit logging is best-effort and must not fail registry mutations.
            logger.opt(exception=True).warning(
                "Gateway external registry audit append failed for {event_type}",
                event_type=event_type,
                target_id=target_id,
            )

    async def _audit_expected_failure(
        self,
        event_type: str,
        *,
        reason_code: str,
        server_id: str | None = None,
        target_id: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {"reason_code": reason_code}
        if server_id is not None:
            payload["server_id"] = server_id
        await self._append_audit_event(
            event_type,
            target_id=target_id,
            payload=payload,
        )

    def _normalize_server_id(
        self,
        value: str,
        *,
        reason_code: str,
    ) -> str:
        text = self._require_text(
            value,
            field="server_id",
            reason_code=reason_code,
        )
        if not _SERVER_ID_RE.fullmatch(text):
            raise self._error(
                "Invalid external server id",
                reason_code=reason_code,
                server_id=text,
            )
        return text

    def _require_text(
        self,
        value: Any,
        *,
        field: str,
        reason_code: str,
        server_id: str | None = None,
    ) -> str:
        if not isinstance(value, str):
            raise self._error(
                f"Invalid {field}",
                reason_code=reason_code,
                server_id=server_id,
            )
        text = value.strip()
        if not text:
            raise self._error(
                f"Invalid {field}",
                reason_code=reason_code,
                server_id=server_id,
            )
        return text

    @staticmethod
    def _dump_server(server: ExternalServerDefinition) -> dict[str, Any]:
        return server.model_dump(mode="json")

    @staticmethod
    def _error(
        message: str,
        *,
        reason_code: str,
        server_id: str | None = None,
    ) -> GatewayExternalRegistryManagementError:
        return GatewayExternalRegistryManagementError(
            message,
            reason_code=reason_code,
            server_id=server_id,
        )
