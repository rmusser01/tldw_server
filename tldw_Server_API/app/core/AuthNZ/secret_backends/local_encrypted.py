from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from tldw_Server_API.app.core.AuthNZ.repos.managed_secret_refs_repo import (
    ManagedSecretRefsRepo,
)
from tldw_Server_API.app.core.AuthNZ.secret_backends.base import SecretBackend
from tldw_Server_API.app.core.Provider_Credentials.repos.org_provider_secrets_repo import (
    AuthnzOrgProviderSecretsRepo,
)
from tldw_Server_API.app.core.Provider_Credentials.repos.user_provider_secrets_repo import (
    AuthnzUserProviderSecretsRepo,
)
from tldw_Server_API.app.core.Provider_Credentials.user_provider_secrets import (
    build_managed_secret_backend_ref,
    decrypt_byok_payload,
    dumps_envelope,
    encrypt_byok_payload,
    key_hint_for_api_key,
    loads_envelope,
    normalize_provider_name,
    normalize_secret_owner_scope_type,
)


class LocalEncryptedSecretBackend(SecretBackend):
    """Secret backend backed by the existing encrypted BYOK tables."""

    backend_name = "local_encrypted_v1"
    display_name = "Local Encrypted"
    capabilities = {
        "store_ref": True,
        "resolve_for_use": True,
        "rotate_if_supported": True,
        "describe_status": True,
        "delete_ref": True,
    }
    default_ephemeral_ttl_seconds = 300

    def __init__(self, *, db_pool, ephemeral_ttl_seconds: int | None = None) -> None:
        super().__init__(db_pool=db_pool)
        self.ephemeral_ttl_seconds = max(60, int(ephemeral_ttl_seconds or self.default_ephemeral_ttl_seconds))

    async def ensure_tables(self) -> None:
        managed_repo = ManagedSecretRefsRepo(self.db_pool)
        await managed_repo.ensure_tables()
        await managed_repo.ensure_backend_registration(
            name=self.backend_name,
            display_name=self.display_name,
            capabilities=self.capabilities,
        )

    @staticmethod
    def _validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict) or not payload:
            raise ValueError("Local encrypted secret payload must be a non-empty object")
        return dict(payload)

    async def _store_underlying_secret(
        self,
        *,
        owner_scope_type: str,
        owner_scope_id: int,
        provider_key: str,
        payload: dict[str, Any],
        updated_at: datetime,
        created_by: int | None,
        updated_by: int | None,
    ) -> None:
        encrypted_blob = dumps_envelope(encrypt_byok_payload(payload))
        api_key = payload.get("api_key")
        key_hint = key_hint_for_api_key(str(api_key or "")) or None
        metadata = {
            "backend_name": self.backend_name,
            "provider_key": provider_key,
        }

        if owner_scope_type == "user":
            repo = AuthnzUserProviderSecretsRepo(self.db_pool)
            await repo.ensure_tables()
            await repo.upsert_secret(
                user_id=int(owner_scope_id),
                provider=provider_key,
                encrypted_blob=encrypted_blob,
                key_hint=key_hint,
                metadata=metadata,
                updated_at=updated_at,
                created_by=created_by,
                updated_by=updated_by,
            )
            return

        repo = AuthnzOrgProviderSecretsRepo(self.db_pool)
        await repo.ensure_tables()
        await repo.upsert_secret(
            scope_type=owner_scope_type,
            scope_id=int(owner_scope_id),
            provider=provider_key,
            encrypted_blob=encrypted_blob,
            key_hint=key_hint,
            metadata=metadata,
            updated_at=updated_at,
            created_by=created_by,
            updated_by=updated_by,
        )

    async def store_ref(
        self,
        *,
        owner_scope_type: str,
        owner_scope_id: int,
        provider_key: str,
        payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        display_name: str | None = None,
        created_by: int | None = None,
        updated_by: int | None = None,
    ) -> dict[str, Any]:
        await self.ensure_tables()
        scope_type = normalize_secret_owner_scope_type(owner_scope_type)
        provider = normalize_provider_name(provider_key)
        normalized_payload = self._validate_payload(payload)
        now = datetime.now(timezone.utc)

        await self._store_underlying_secret(
            owner_scope_type=scope_type,
            owner_scope_id=int(owner_scope_id),
            provider_key=provider,
            payload=normalized_payload,
            updated_at=now,
            created_by=created_by,
            updated_by=updated_by,
        )

        managed_repo = ManagedSecretRefsRepo(self.db_pool)
        return await managed_repo.upsert_ref(
            backend_name=self.backend_name,
            owner_scope_type=scope_type,
            owner_scope_id=int(owner_scope_id),
            provider_key=provider,
            backend_ref=build_managed_secret_backend_ref(scope_type, int(owner_scope_id), provider),
            metadata=metadata,
            display_name=display_name,
            status="active",
            created_by=created_by,
            updated_by=updated_by,
        )

    async def _fetch_underlying_secret_row(self, ref: dict[str, Any]) -> tuple[dict[str, Any] | None, Any]:
        scope_type = normalize_secret_owner_scope_type(str(ref.get("owner_scope_type") or ""))
        scope_id = int(ref["owner_scope_id"])
        provider = normalize_provider_name(str(ref.get("provider_key") or ""))

        if scope_type == "user":
            repo = AuthnzUserProviderSecretsRepo(self.db_pool)
            await repo.ensure_tables()
            return await repo.fetch_secret_for_user(scope_id, provider), repo

        repo = AuthnzOrgProviderSecretsRepo(self.db_pool)
        await repo.ensure_tables()
        return await repo.fetch_secret(scope_type, scope_id, provider), repo

    async def resolve_for_use(self, secret_ref_id: int) -> dict[str, Any]:
        await self.ensure_tables()
        managed_repo = ManagedSecretRefsRepo(self.db_pool)
        ref = await managed_repo.get_ref(int(secret_ref_id))
        if not ref:
            raise ValueError("Managed secret ref is not available")

        secret_row, repo = await self._fetch_underlying_secret_row(ref)
        encrypted_blob = str(secret_row.get("encrypted_blob") or "") if isinstance(secret_row, dict) else ""
        if not encrypted_blob:
            raise ValueError("Managed secret payload is not available")

        payload = decrypt_byok_payload(loads_envelope(encrypted_blob))
        resolved_at = datetime.now(timezone.utc)
        expires_at = resolved_at + timedelta(seconds=self.ephemeral_ttl_seconds)

        if isinstance(repo, AuthnzUserProviderSecretsRepo):
            await repo.touch_last_used(int(ref["owner_scope_id"]), str(ref["provider_key"]), resolved_at)
        else:
            await repo.touch_last_used(
                str(ref["owner_scope_type"]),
                int(ref["owner_scope_id"]),
                str(ref["provider_key"]),
                resolved_at,
            )
        await managed_repo.touch_last_resolved(
            int(secret_ref_id),
            resolved_at=resolved_at,
            expires_at=expires_at,
        )

        return {
            "id": int(ref["id"]),
            "backend_name": self.backend_name,
            "owner_scope_type": str(ref["owner_scope_type"]),
            "owner_scope_id": int(ref["owner_scope_id"]),
            "provider_key": str(ref["provider_key"]),
            "material": payload,
            "resolved_at": resolved_at,
            "expires_at": expires_at,
        }

    async def rotate_if_supported(
        self,
        secret_ref_id: int,
        *,
        payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        updated_by: int | None = None,
    ) -> dict[str, Any]:
        managed_repo = ManagedSecretRefsRepo(self.db_pool)
        ref = await managed_repo.get_ref(int(secret_ref_id))
        if not ref:
            raise ValueError("Managed secret ref is not available")
        return await self.store_ref(
            owner_scope_type=str(ref["owner_scope_type"]),
            owner_scope_id=int(ref["owner_scope_id"]),
            provider_key=str(ref["provider_key"]),
            payload=payload,
            metadata=metadata if metadata is not None else ref.get("metadata"),
            display_name=ref.get("display_name"),
            created_by=ref.get("created_by"),
            updated_by=updated_by,
        )

    async def describe_status(self, secret_ref_id: int) -> dict[str, Any]:
        managed_repo = ManagedSecretRefsRepo(self.db_pool)
        ref = await managed_repo.get_ref(int(secret_ref_id), include_revoked=True)
        if not ref:
            return {"state": "missing", "backend_name": self.backend_name}
        if ref.get("revoked_at"):
            return {"state": "revoked", "backend_name": self.backend_name, "id": int(ref["id"])}

        secret_row, _repo = await self._fetch_underlying_secret_row(ref)
        encrypted_blob = str(secret_row.get("encrypted_blob") or "") if isinstance(secret_row, dict) else ""
        state = "ready" if encrypted_blob else "missing"
        return {
            "id": int(ref["id"]),
            "backend_name": self.backend_name,
            "owner_scope_type": str(ref["owner_scope_type"]),
            "owner_scope_id": int(ref["owner_scope_id"]),
            "provider_key": str(ref["provider_key"]),
            "state": state,
        }

    async def delete_ref(
        self,
        secret_ref_id: int,
        *,
        revoked_by: int | None = None,
    ) -> bool:
        managed_repo = ManagedSecretRefsRepo(self.db_pool)
        ref = await managed_repo.get_ref(int(secret_ref_id))
        if not ref:
            return False

        provider = str(ref["provider_key"])
        scope_type = str(ref["owner_scope_type"])
        scope_id = int(ref["owner_scope_id"])
        revoked_at = datetime.now(timezone.utc)

        if scope_type == "user":
            repo = AuthnzUserProviderSecretsRepo(self.db_pool)
            await repo.ensure_tables()
            await repo.delete_secret(scope_id, provider, revoked_by=revoked_by, revoked_at=revoked_at)
        else:
            repo = AuthnzOrgProviderSecretsRepo(self.db_pool)
            await repo.ensure_tables()
            await repo.delete_secret(
                scope_type,
                scope_id,
                provider,
                revoked_by=revoked_by,
                revoked_at=revoked_at,
            )
        return await managed_repo.delete_ref(
            int(secret_ref_id),
            revoked_by=revoked_by,
            revoked_at=revoked_at,
        )
