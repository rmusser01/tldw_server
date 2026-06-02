"""Gateway credential-grant management helpers for standalone MCP stores."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from loguru import logger

from mcp_unified.gateway.external_registry import GatewayStoreMetadata
from mcp_unified.interfaces.storage import (
    AuditStore,
    CredentialGrantAlreadyExistsError,
    CredentialGrantStore,
    ExternalRegistryStore,
    ProfileStore,
)
from mcp_unified.storage.models import AuditEvent, CredentialGrant

_PATCH_FIELDS = frozenset(
    {
        "broker_id",
        "credential_slot",
        "external_server_id",
        "scopes",
        "metadata",
        "provenance",
        "enabled",
    }
)
_SECRET_KEY_MARKERS = ("secret", "token", "password")
_SECRET_KEY_EXACT_MATCHES = {
    "api_key",
    "authorization",
    "headers",
    "env",
    "credential_value",
}
CREDENTIAL_GRANT_SENSITIVE_MATERIAL_REJECTED_REASON = "_".join(
    ("credential", "grant", "secret", "material", "rejected")
)
CREDENTIAL_GRANT_SENSITIVE_MATERIAL_REJECTED_ERROR = " ".join(
    ("Credential grant metadata must not contain", "secret material")
)


class GatewayCredentialGrantManagementError(RuntimeError):
    """Domain error for expected gateway credential-grant failures."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str,
        grant_id: str | None = None,
        profile_id: str | None = None,
        external_server_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.grant_id = grant_id
        self.profile_id = profile_id
        self.external_server_id = external_server_id

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe error payload."""

        payload: dict[str, Any] = {
            "ok": False,
            "error": str(self),
            "reason_code": self.reason_code,
        }
        if self.grant_id is not None:
            payload["grant_id"] = self.grant_id
        if self.profile_id is not None:
            payload["profile_id"] = self.profile_id
        if self.external_server_id is not None:
            payload["external_server_id"] = self.external_server_id
        return payload


class GatewayCredentialGrantManager:
    """Manage stored credential broker grant metadata for the package gateway."""

    def __init__(
        self,
        *,
        credential_grant_store: CredentialGrantStore,
        store_metadata: GatewayStoreMetadata,
        profile_store: ProfileStore | None = None,
        external_registry_store: ExternalRegistryStore | None = None,
        audit_store: AuditStore | None = None,
    ) -> None:
        self.credential_grant_store = credential_grant_store
        self.profile_store = profile_store
        self.external_registry_store = external_registry_store
        self.audit_store = audit_store
        self.store_metadata = store_metadata

    async def list_grants(
        self,
        *,
        profile_id: str | None = None,
        external_server_id: str | None = None,
    ) -> dict[str, Any]:
        """List credential grants with store metadata."""

        try:
            grants = await self.credential_grant_store.list_grants(
                profile_id=self._normalize_optional_text(profile_id),
                external_server_id=self._normalize_optional_text(external_server_id),
            )
        except Exception as exc:  # noqa: BLE001
            logger.opt(exception=True).warning("Gateway credential grant list failed")
            raise self._error(
                "Credential grant store unavailable",
                reason_code="credential_grant_store_unavailable",
            ) from exc
        return {
            "ok": True,
            "grants": [
                self._dump_grant(grant)
                for grant in sorted(grants, key=lambda item: item.id)
            ],
            "store": self.store_metadata.to_payload(),
        }

    async def show_grant(self, grant_id: str) -> dict[str, Any]:
        """Return one credential grant by id with store metadata."""

        normalized_grant_id = self._normalize_grant_id(
            grant_id,
            reason_code="invalid_credential_grant_request",
        )
        grant = await self._get_grant(normalized_grant_id)
        if grant is None:
            raise self._error(
                f"Credential grant not found: {normalized_grant_id}",
                reason_code="credential_grant_not_found",
                grant_id=normalized_grant_id,
            )
        return {
            "ok": True,
            "grant": self._dump_grant(grant),
            "store": self.store_metadata.to_payload(),
        }

    async def create_grant(
        self,
        grant_document: CredentialGrant | Mapping[str, Any],
    ) -> dict[str, Any]:
        """Create credential grant metadata without replacing existing grants."""

        grant = self._coerce_create_document(grant_document)
        await self._validate_references(grant)
        try:
            stored = await self.credential_grant_store.create_grant(grant)
        except CredentialGrantAlreadyExistsError as exc:
            raise self._error(
                f"Credential grant already exists: {exc.grant_id}",
                reason_code="credential_grant_already_exists",
                grant_id=exc.grant_id,
            ) from exc
        except Exception as exc:  # noqa: BLE001
            logger.opt(exception=True).warning(
                "Gateway credential grant create failed",
                grant_id=grant.id,
            )
            raise self._error(
                "Credential grant store unavailable",
                reason_code="credential_grant_store_unavailable",
                grant_id=grant.id,
            ) from exc

        await self._append_audit_event(
            "credential_grant.created",
            target_id=stored.id,
            profile_id=stored.profile_id,
            payload={"grant_id": stored.id, "profile_id": stored.profile_id},
        )
        return {
            "ok": True,
            "grant": self._dump_grant(stored),
            "store": self.store_metadata.to_payload(),
        }

    async def patch_grant(
        self,
        grant_id: str,
        patch_document: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Apply an allowed metadata patch to a credential grant."""

        normalized_grant_id = self._normalize_grant_id(
            grant_id,
            reason_code="invalid_credential_grant_patch",
        )
        patch = self._validate_patch_document(patch_document)
        current = await self._get_grant_for_mutation(normalized_grant_id)
        updated, changed_fields = self._apply_patch(current, patch)
        await self._validate_references(updated)
        try:
            stored = await self.credential_grant_store.upsert_grant(updated)
        except Exception as exc:  # noqa: BLE001
            logger.opt(exception=True).warning(
                "Gateway credential grant patch failed",
                grant_id=normalized_grant_id,
            )
            raise self._error(
                "Credential grant store unavailable",
                reason_code="credential_grant_store_unavailable",
                grant_id=normalized_grant_id,
            ) from exc

        await self._append_audit_event(
            "credential_grant.patched",
            target_id=stored.id,
            profile_id=stored.profile_id,
            payload={
                "grant_id": stored.id,
                "profile_id": stored.profile_id,
                "changed_fields": list(changed_fields),
            },
        )
        return {
            "ok": True,
            "grant": self._dump_grant(stored),
            "store": self.store_metadata.to_payload(),
        }

    async def delete_grant(self, grant_id: str) -> dict[str, Any]:
        """Delete credential grant metadata by id."""

        normalized_grant_id = self._normalize_grant_id(
            grant_id,
            reason_code="invalid_credential_grant_request",
        )
        current = await self._get_grant_for_mutation(normalized_grant_id)
        try:
            deleted = await self.credential_grant_store.delete_grant(
                normalized_grant_id,
            )
        except Exception as exc:  # noqa: BLE001
            logger.opt(exception=True).warning(
                "Gateway credential grant delete failed",
                grant_id=normalized_grant_id,
            )
            raise self._error(
                "Credential grant store unavailable",
                reason_code="credential_grant_store_unavailable",
                grant_id=normalized_grant_id,
            ) from exc
        if not deleted:
            raise self._error(
                f"Credential grant not found: {normalized_grant_id}",
                reason_code="credential_grant_not_found",
                grant_id=normalized_grant_id,
            )

        await self._append_audit_event(
            "credential_grant.deleted",
            target_id=normalized_grant_id,
            profile_id=current.profile_id,
            payload={
                "grant_id": normalized_grant_id,
                "profile_id": current.profile_id,
            },
        )
        return {
            "ok": True,
            "grant_id": normalized_grant_id,
            "store": self.store_metadata.to_payload(),
        }

    def _coerce_create_document(
        self,
        grant_document: CredentialGrant | Mapping[str, Any],
    ) -> CredentialGrant:
        try:
            grant = (
                grant_document.model_copy(deep=True)
                if isinstance(grant_document, CredentialGrant)
                else CredentialGrant.model_validate(grant_document)
            )
            grant = self._normalize_grant(grant)
            self._reject_secret_material(grant.metadata, grant_id=grant.id)
            self._reject_secret_material(grant.provenance, grant_id=grant.id)
            return grant.model_copy(
                update={"updated_at": datetime.now(timezone.utc)},
                deep=True,
            )
        except GatewayCredentialGrantManagementError:
            raise
        except (TypeError, ValueError) as exc:
            raise self._error(
                "Invalid credential grant request",
                reason_code="invalid_credential_grant_request",
            ) from exc

    def _validate_patch_document(self, patch_document: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(patch_document, Mapping):
            raise self._error(
                "Invalid credential grant patch",
                reason_code="invalid_credential_grant_patch",
            )
        patch = dict(patch_document)
        unsupported_fields = set(patch) - _PATCH_FIELDS
        if not patch or unsupported_fields:
            raise self._error(
                "Invalid credential grant patch",
                reason_code="invalid_credential_grant_patch",
            )
        for field_name in ("broker_id", "credential_slot"):
            if field_name in patch:
                patch[field_name] = self._require_text(
                    patch[field_name],
                    field=field_name,
                    reason_code="invalid_credential_grant_patch",
                )
        if "external_server_id" in patch:
            patch["external_server_id"] = self._normalize_optional_text(
                patch["external_server_id"]
            )
        if "scopes" in patch:
            patch["scopes"] = self._normalize_text_list(
                patch["scopes"],
                reason_code="invalid_credential_grant_patch",
            )
        if "metadata" in patch:
            self._reject_secret_material(patch["metadata"])
        if "provenance" in patch:
            self._reject_secret_material(patch["provenance"])
        return patch

    def _apply_patch(
        self,
        grant: CredentialGrant,
        patch: Mapping[str, Any],
    ) -> tuple[CredentialGrant, tuple[str, ...]]:
        payload = grant.model_dump(mode="python")
        payload.update(dict(patch))
        payload["updated_at"] = datetime.now(timezone.utc)
        try:
            updated = CredentialGrant.model_validate(payload)
        except (TypeError, ValueError) as exc:
            raise self._error(
                "Invalid credential grant patch",
                reason_code="invalid_credential_grant_patch",
                grant_id=grant.id,
            ) from exc
        return self._normalize_grant(updated), tuple(sorted(patch))

    def _normalize_grant(self, grant: CredentialGrant) -> CredentialGrant:
        normalized_grant_id = self._normalize_grant_id(
            grant.id,
            reason_code="invalid_credential_grant_request",
        )
        profile_id = self._require_text(
            grant.profile_id,
            field="profile_id",
            reason_code="invalid_credential_grant_request",
            grant_id=normalized_grant_id,
        )
        broker_id = self._require_text(
            grant.broker_id,
            field="broker_id",
            reason_code="invalid_credential_grant_request",
            grant_id=normalized_grant_id,
        )
        credential_slot = self._require_text(
            grant.credential_slot,
            field="credential_slot",
            reason_code="invalid_credential_grant_request",
            grant_id=normalized_grant_id,
        )
        return grant.model_copy(
            update={
                "id": normalized_grant_id,
                "profile_id": profile_id,
                "broker_id": broker_id,
                "credential_slot": credential_slot,
                "external_server_id": self._normalize_optional_text(
                    grant.external_server_id,
                ),
                "scopes": self._normalize_text_list(
                    grant.scopes,
                    reason_code="invalid_credential_grant_request",
                ),
            },
            deep=True,
        )

    async def _validate_references(self, grant: CredentialGrant) -> None:
        if self.profile_store is not None:
            try:
                profile = await self.profile_store.get_profile(grant.profile_id)
            except Exception as exc:  # noqa: BLE001
                raise self._error(
                    "Profile store unavailable",
                    reason_code="profile_store_unavailable",
                    grant_id=grant.id,
                    profile_id=grant.profile_id,
                ) from exc
            if profile is None:
                raise self._error(
                    f"Profile not found: {grant.profile_id}",
                    reason_code="profile_not_found",
                    grant_id=grant.id,
                    profile_id=grant.profile_id,
                )

        if self.external_registry_store is not None and grant.external_server_id:
            try:
                server = await self.external_registry_store.get_server(
                    grant.external_server_id,
                )
            except Exception as exc:  # noqa: BLE001
                raise self._error(
                    "External registry store unavailable",
                    reason_code="external_registry_store_unavailable",
                    grant_id=grant.id,
                    external_server_id=grant.external_server_id,
                ) from exc
            if server is None:
                raise self._error(
                    f"External server not found: {grant.external_server_id}",
                    reason_code="external_server_not_found",
                    grant_id=grant.id,
                    external_server_id=grant.external_server_id,
                )

    async def _get_grant(self, grant_id: str) -> CredentialGrant | None:
        try:
            return await self.credential_grant_store.get_grant(grant_id)
        except Exception as exc:  # noqa: BLE001
            raise self._error(
                "Credential grant store unavailable",
                reason_code="credential_grant_store_unavailable",
                grant_id=grant_id,
            ) from exc

    async def _get_grant_for_mutation(self, grant_id: str) -> CredentialGrant:
        grant = await self._get_grant(grant_id)
        if grant is None:
            raise self._error(
                f"Credential grant not found: {grant_id}",
                reason_code="credential_grant_not_found",
                grant_id=grant_id,
            )
        return grant

    async def _append_audit_event(
        self,
        event_type: str,
        *,
        target_id: str | None = None,
        profile_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        """Append an audit event when an audit store is configured."""

        if self.audit_store is None:
            return
        event = AuditEvent(
            id=str(uuid4()),
            event_type=event_type,
            profile_id=profile_id,
            target_type="credential_grant" if target_id is not None else None,
            target_id=target_id,
            payload=dict(payload or {}),
            provenance={"source": "gateway_credential_grant_manager"},
            created_at=datetime.now(timezone.utc),
        )
        try:
            await self.audit_store.append_event(event)
        except Exception:  # noqa: BLE001
            logger.opt(exception=True).warning(
                "Gateway credential grant audit append failed for {event_type}",
                event_type=event_type,
                target_id=target_id,
            )

    def _reject_secret_material(
        self,
        value: Any,
        *,
        grant_id: str | None = None,
    ) -> None:
        if _contains_secret_key(value):
            raise self._error(
                CREDENTIAL_GRANT_SENSITIVE_MATERIAL_REJECTED_ERROR,
                reason_code=CREDENTIAL_GRANT_SENSITIVE_MATERIAL_REJECTED_REASON,
                grant_id=grant_id,
            )

    def _normalize_grant_id(
        self,
        value: str,
        *,
        reason_code: str,
    ) -> str:
        return self._require_text(value, field="grant_id", reason_code=reason_code)

    def _require_text(
        self,
        value: Any,
        *,
        field: str,
        reason_code: str,
        grant_id: str | None = None,
    ) -> str:
        if not isinstance(value, str):
            raise self._error(
                f"Invalid {field}",
                reason_code=reason_code,
                grant_id=grant_id,
            )
        text = value.strip()
        if not text:
            raise self._error(
                f"Invalid {field}",
                reason_code=reason_code,
                grant_id=grant_id,
            )
        return text

    @staticmethod
    def _normalize_optional_text(value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            return None
        text = value.strip()
        return text or None

    def _normalize_text_list(self, values: Any, *, reason_code: str) -> list[str]:
        if values is None:
            return []
        if not isinstance(values, list | tuple):
            raise self._error(
                "Invalid credential grant list field",
                reason_code=reason_code,
            )
        return [
            item.strip()
            for item in values
            if isinstance(item, str) and item.strip()
        ]

    @staticmethod
    def _dump_grant(grant: CredentialGrant) -> dict[str, Any]:
        return grant.model_dump(mode="json")

    @staticmethod
    def _error(
        message: str,
        *,
        reason_code: str,
        grant_id: str | None = None,
        profile_id: str | None = None,
        external_server_id: str | None = None,
    ) -> GatewayCredentialGrantManagementError:
        return GatewayCredentialGrantManagementError(
            message,
            reason_code=reason_code,
            grant_id=grant_id,
            profile_id=profile_id,
            external_server_id=external_server_id,
        )


def reject_secret_looking_metadata(value: Any) -> None:
    """Reject mappings containing secret-looking keys."""

    if _contains_secret_key(value):
        raise GatewayCredentialGrantManagementError(
            CREDENTIAL_GRANT_SENSITIVE_MATERIAL_REJECTED_ERROR,
            reason_code=CREDENTIAL_GRANT_SENSITIVE_MATERIAL_REJECTED_REASON,
        )


def _contains_secret_key(value: Any) -> bool:
    """Return whether a JSON-like value contains secret-looking keys."""

    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            key_text = str(key).strip().lower()
            if key_text in _SECRET_KEY_EXACT_MATCHES or any(
                marker in key_text for marker in _SECRET_KEY_MARKERS
            ):
                return True
            if _contains_secret_key(nested_value):
                return True
    elif isinstance(value, list | tuple):
        return any(_contains_secret_key(item) for item in value)
    return False


__all__ = [
    "CREDENTIAL_GRANT_SENSITIVE_MATERIAL_REJECTED_ERROR",
    "CREDENTIAL_GRANT_SENSITIVE_MATERIAL_REJECTED_REASON",
    "GatewayCredentialGrantManagementError",
    "GatewayCredentialGrantManager",
    "reject_secret_looking_metadata",
]
