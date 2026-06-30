"""Service for creating, validating, and revoking share tokens."""
from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timezone
from typing import Any

import bcrypt

from tldw_Server_API.app.core.AuthNZ.repos.shared_workspace_repo import SharedWorkspaceRepo


class ShareTokenService:
    """Manages token-based sharing links with expiry, password, and use limits."""

    _PROTOTYPE_RESOURCE_TYPE = "prototype_workspace"
    _PROTOTYPE_RESOURCE_ALIASES = {"prototype_workspace", "prototype"}
    _PROTOTYPE_STORAGE_PREFIX = "prototype_workspace::"
    _SUPPORTED_RESOURCE_TYPES = {"chatbook", "workspace", "prototype_workspace"}

    def __init__(self, repo: SharedWorkspaceRepo) -> None:
        self._repo = repo

    async def generate_token(
        self,
        *,
        resource_type: str,
        resource_id: str,
        owner_user_id: int,
        access_level: str = "view_chat",
        allow_clone: bool = True,
        password: str | None = None,
        max_uses: int | None = None,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        normalized_resource_type, normalized_resource_id = self._normalize_resource_identity_for_write(
            resource_type=resource_type,
            resource_id=resource_id,
        )
        raw_token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
        token_prefix = raw_token[:8]

        password_hash: str | None = None
        if password:
            password_hash = bcrypt.hashpw(
                password.encode("utf-8"),
                bcrypt.gensalt(),
            ).decode("utf-8")

        record = await self._repo.create_token(
            token_hash=token_hash,
            token_prefix=token_prefix,
            resource_type=normalized_resource_type,
            resource_id=normalized_resource_id,
            owner_user_id=owner_user_id,
            access_level=access_level,
            allow_clone=allow_clone,
            password_hash=password_hash,
            max_uses=max_uses,
            expires_at=expires_at,
        )

        # Return raw token only once — never stored server-side
        record = self._hydrate_legacy_resource_identity(record)
        record["raw_token"] = raw_token
        return record

    def _normalize_resource_identity_for_write(self, *, resource_type: str, resource_id: str) -> tuple[str, str]:
        normalized_type = str(resource_type or "").strip().lower()
        normalized_id = str(resource_id or "").strip()
        if not normalized_type or not normalized_id:
            raise ValueError("resource_type and resource_id are required")
        if normalized_type in self._PROTOTYPE_RESOURCE_ALIASES:
            normalized_type = self._PROTOTYPE_RESOURCE_TYPE
        if normalized_type not in self._SUPPORTED_RESOURCE_TYPES:
            raise ValueError(f"Unsupported resource_type: {resource_type}")
        if (
            normalized_type == self._PROTOTYPE_RESOURCE_TYPE
            and normalized_id.startswith(self._PROTOTYPE_STORAGE_PREFIX)
        ):
            normalized_id = normalized_id[len(self._PROTOTYPE_STORAGE_PREFIX):]
        if not normalized_id:
            raise ValueError("resource_id is required")
        return normalized_type, normalized_id

    def _hydrate_legacy_resource_identity(self, record: dict[str, Any]) -> dict[str, Any]:
        hydrated = dict(record)
        stored_type = str(hydrated.get("resource_type") or "").strip().lower()
        stored_id = str(hydrated.get("resource_id") or "").strip()
        if stored_type == "workspace" and stored_id.startswith(self._PROTOTYPE_STORAGE_PREFIX):
            hydrated["resource_type"] = self._PROTOTYPE_RESOURCE_TYPE
            hydrated["resource_id"] = stored_id[len(self._PROTOTYPE_STORAGE_PREFIX):]
        return hydrated

    async def validate_token(
        self,
        raw_token: str,
        *,
        allow_exhausted: bool = False,
    ) -> dict[str, Any] | None:
        prefix = raw_token[:8]
        token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()

        candidates = await self._repo.find_tokens_by_prefix(prefix)
        if not candidates:
            return None

        for candidate in candidates:
            stored_hash = candidate.get("token_hash", "")
            if not hmac.compare_digest(token_hash, stored_hash):
                continue

            # Defense-in-depth: check revocation even though SQL filters it
            if candidate.get("is_revoked") or candidate.get("revoked_at"):
                return None

            # Check expiry
            expires_at = candidate.get("expires_at")
            if expires_at:
                if isinstance(expires_at, str):
                    try:
                        exp_dt = datetime.fromisoformat(expires_at)
                    except ValueError:
                        return None
                else:
                    exp_dt = expires_at
                if exp_dt.tzinfo is None:
                    exp_dt = exp_dt.replace(tzinfo=timezone.utc)
                if exp_dt < datetime.now(timezone.utc):
                    return None

            # Check use count
            max_uses = candidate.get("max_uses")
            candidate_with_state = dict(candidate)
            if max_uses is not None and candidate.get("use_count", 0) >= max_uses:
                if not allow_exhausted:
                    return None
                candidate_with_state["is_use_exhausted"] = True
            else:
                candidate_with_state["is_use_exhausted"] = False

            return self._hydrate_legacy_resource_identity(candidate_with_state)

        return None

    async def verify_password(self, token_record: dict[str, Any], password: str) -> bool:
        stored_hash = token_record.get("password_hash")
        if not stored_hash:
            return True  # No password required
        return bcrypt.checkpw(
            password.encode("utf-8"),
            stored_hash.encode("utf-8"),
        )

    async def use_token(self, token_id: int) -> None:
        await self._repo.increment_token_use_count(token_id)

    async def claim_token_use(self, token_id: int) -> bool:
        return await self._repo.claim_token_use(token_id)

    async def release_token_use(self, token_id: int) -> None:
        await self._repo.release_token_use(token_id)

    async def revoke_token(self, token_id: int) -> bool:
        return await self._repo.revoke_token(token_id)

    async def list_tokens(self, owner_user_id: int) -> list[dict[str, Any]]:
        tokens = await self._repo.list_tokens_for_user(owner_user_id)
        # Strip sensitive fields
        for idx, t in enumerate(tokens):
            t.pop("token_hash", None)
            t.pop("password_hash", None)
            tokens[idx] = self._hydrate_legacy_resource_identity(t)
        return tokens

    async def revoke_tokens_for_resource(
        self, resource_type: str, resource_id: str, owner_user_id: int
    ) -> int:
        normalized_resource_type, normalized_resource_id = self._normalize_resource_identity_for_write(
            resource_type=resource_type,
            resource_id=resource_id,
        )
        revoked = await self._repo.revoke_tokens_for_resource(
            normalized_resource_type,
            normalized_resource_id,
            owner_user_id,
        )
        if normalized_resource_type == self._PROTOTYPE_RESOURCE_TYPE:
            revoked += await self._repo.revoke_tokens_for_resource(
                "workspace",
                f"{self._PROTOTYPE_STORAGE_PREFIX}{normalized_resource_id}",
                owner_user_id,
            )
        return revoked

    def get_prototype_workspace_id(self, token_record: dict[str, Any]) -> str | None:
        normalized = self._hydrate_legacy_resource_identity(token_record)
        resource_type = str(normalized.get("resource_type") or "").strip().lower()
        resource_id = str(normalized.get("resource_id") or "").strip()
        if resource_type == self._PROTOTYPE_RESOURCE_TYPE and resource_id:
            return resource_id
        return None
