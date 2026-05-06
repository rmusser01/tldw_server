"""Brokered preview handles for prototype workspaces and sessions."""
from __future__ import annotations

import hashlib
import hmac
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, ClassVar

from tldw_Server_API.app.core.AuthNZ.repos.prototype_workspaces_repo import (
    PrototypeWorkspacesRepo,
)
from tldw_Server_API.app.core.AuthNZ.settings import get_settings

from .models import (
    PrototypePreviewHandleRecord,
    PrototypePreviewScope,
    PrototypePreviewStatus,
    actor_key_for_session,
    preview_scope_id,
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_iso8601(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _resolve_stable_signing_secret(signing_secret: str | None) -> str:
    settings = get_settings()
    candidates = (
        signing_secret,
        os.getenv("PROTOTYPE_PREVIEW_SIGNING_SECRET"),
        getattr(settings, "JWT_SECRET_KEY", None),
        getattr(settings, "SINGLE_USER_API_KEY", None),
        os.getenv("JWT_SECRET_KEY"),
        os.getenv("SINGLE_USER_API_KEY"),
    )
    for candidate in candidates:
        value = str(candidate or "").strip()
        if value:
            return value
    raise RuntimeError(
        "Prototype preview grants require a stable signing secret; set "
        "PROTOTYPE_PREVIEW_SIGNING_SECRET, JWT_SECRET_KEY, or SINGLE_USER_API_KEY"
    )


class PrototypePreviewHandleNotFound(RuntimeError):
    """Raised when a preview handle cannot be found or renewed."""


class PrototypePreviewBroker:
    """Issue opaque preview handles and short-lived signed grants."""

    _records: ClassVar[dict[str, PrototypePreviewHandleRecord]] = {}
    _active_scope_handles: ClassVar[dict[str, str]] = {}
    _lock: ClassVar[threading.RLock] = threading.RLock()

    def __init__(
        self,
        *,
        repo: PrototypeWorkspacesRepo,
        base_preview_path: str = "/api/v1/prototype-previews",
        grant_ttl_seconds: int = 5 * 60,
        signing_secret: str | None = None,
    ) -> None:
        self._repo = repo
        self._base_preview_path = str(base_preview_path or "/api/v1/prototype-previews").rstrip("/")
        self._grant_ttl_seconds = max(int(grant_ttl_seconds), 30)
        self._signing_secret = _resolve_stable_signing_secret(signing_secret)

    async def issue_preview_grant(
        self,
        *,
        prototype_workspace_id: str,
        snapshot_id: str,
        runtime_target_url: str,
        prototype_session_id: str | None = None,
        runtime_policy_profile: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        workspace = await self._repo.get_workspace(prototype_workspace_id)
        if not workspace:
            raise RuntimeError("prototype workspace not found")
        if workspace.get("is_archived"):
            raise RuntimeError("prototype workspace is archived")
        if not str(snapshot_id or "").strip():
            raise ValueError("snapshot_id is required")
        if not str(runtime_target_url or "").strip():
            raise ValueError("runtime_target_url is required")

        session: dict[str, Any] | None = None
        preview_scope = PrototypePreviewScope.CANONICAL
        actor_key = f"workspace:{prototype_workspace_id}"
        resolved_runtime_policy = runtime_policy_profile or "canonical_preview"

        if prototype_session_id:
            preview_scope = PrototypePreviewScope.SESSION
            session = await self._repo.get_session(prototype_session_id)
            if not session or session.get("prototype_workspace_id") != prototype_workspace_id:
                raise RuntimeError("prototype session not found in workspace")
            if session.get("is_revoked"):
                raise RuntimeError("revoked session cannot receive preview grants")
            expires_at = _normalize_iso8601(session.get("expires_at"))
            if expires_at and expires_at <= _utc_now():
                raise RuntimeError("expired session cannot receive preview grants")
            await self._assert_session_actor_active(session)
            actor_key = actor_key_for_session(
                actor_type=str(session.get("actor_type") or ""),
                actor_user_id=session.get("actor_user_id"),
                actor_shared_actor_id=session.get("actor_shared_actor_id"),
            )
            resolved_runtime_policy = await self._resolve_runtime_policy_profile(
                workspace=workspace,
                session=session,
                explicit_profile=runtime_policy_profile,
            )

        scope_id = preview_scope_id(
            preview_scope=preview_scope.value,
            prototype_workspace_id=prototype_workspace_id,
            prototype_session_id=prototype_session_id,
        )
        handle_id = f"pph_{uuid.uuid4().hex}"
        now = _utc_now()
        record = PrototypePreviewHandleRecord(
            handle_id=handle_id,
            preview_scope=preview_scope,
            scope_id=scope_id,
            prototype_workspace_id=prototype_workspace_id,
            prototype_session_id=prototype_session_id,
            actor_key=actor_key,
            target_ref=str(runtime_target_url),
            runtime_policy_profile=resolved_runtime_policy,
            metadata={
                "snapshot_id": str(snapshot_id),
                **(metadata or {}),
            },
            created_at=now.isoformat(),
        )
        previous_handle_id: str | None = None

        with self._lock:
            previous_handle_id = self._active_scope_handles.get(scope_id)
            self._revoke_scope_locked(scope_id=scope_id, revoked_at=now.isoformat())
            self._records[handle_id] = record
            self._active_scope_handles[scope_id] = handle_id

        try:
            if prototype_session_id:
                updated = await self._repo.update_session_state(
                    prototype_session_id,
                    preview_handle=handle_id,
                    preview_status=PrototypePreviewStatus.READY.value,
                    last_activity_at=now.isoformat(),
                )
                if not updated:
                    raise RuntimeError("failed to persist preview handle")
            else:
                updated_workspace = await self._repo.update_workspace_state(
                    prototype_workspace_id,
                    canonical_preview_status=PrototypePreviewStatus.READY.value,
                )
                if not updated_workspace:
                    raise RuntimeError("failed to persist canonical preview state")
        except Exception:
            with self._lock:
                failed_record = self._records.get(handle_id)
                if failed_record is not None:
                    failed_record.is_active = False
                    failed_record.revoked_at = _utc_now().isoformat()
                if self._active_scope_handles.get(scope_id) == handle_id:
                    self._active_scope_handles.pop(scope_id, None)
                if previous_handle_id:
                    previous_record = self._records.get(previous_handle_id)
                    if previous_record is not None:
                        previous_record.is_active = True
                        previous_record.revoked_at = None
                        self._active_scope_handles[scope_id] = previous_handle_id
            raise

        token, expires_at = self._mint_grant_token(
            preview_handle=handle_id,
            actor_key=actor_key,
        )
        return {
            "preview_handle": handle_id,
            "preview_scope": preview_scope.value,
            "prototype_workspace_id": prototype_workspace_id,
            "prototype_session_id": prototype_session_id,
            "snapshot_id": str(snapshot_id),
            "preview_url": f"{self._base_preview_path}/{handle_id}?exp={expires_at}&token={token}",
            "expires_at": datetime.fromtimestamp(expires_at, timezone.utc).isoformat(),
            "token": token,
            "runtime_policy_profile": resolved_runtime_policy,
        }

    async def revoke_preview_handle(self, preview_handle: str) -> bool:
        handle_id = str(preview_handle or "").strip()
        if not handle_id:
            return False
        with self._lock:
            record = self._records.get(handle_id)
            if not record or not record.is_active:
                return False
            record.is_active = False
            record.revoked_at = _utc_now().isoformat()
            if self._active_scope_handles.get(record.scope_id) == handle_id:
                self._active_scope_handles.pop(record.scope_id, None)
        return True

    def get_preview_record(self, preview_handle: str) -> dict[str, Any] | None:
        handle_id = str(preview_handle or "").strip()
        if not handle_id:
            return None
        with self._lock:
            record = self._records.get(handle_id)
            if not record:
                return None
            return self._record_to_dict(record)

    async def renew_preview_grant(self, preview_handle: str) -> dict[str, Any]:
        handle_id = str(preview_handle or "").strip()
        if not handle_id:
            raise PrototypePreviewHandleNotFound("preview handle is required")

        with self._lock:
            record = self._records.get(handle_id)
            if not record or not record.is_active:
                raise PrototypePreviewHandleNotFound("preview handle not found")
            record_dict = self._record_to_dict(record)

        if not await self._is_grant_still_authorized(record):
            raise RuntimeError("preview handle is no longer authorized")

        token, expires_at = self._mint_grant_token(
            preview_handle=handle_id,
            actor_key=str(record_dict["actor_key"]),
        )
        return {
            "preview_handle": handle_id,
            "preview_scope": record_dict["preview_scope"],
            "prototype_workspace_id": record_dict["prototype_workspace_id"],
            "prototype_session_id": record_dict["prototype_session_id"],
            "snapshot_id": (record_dict.get("metadata") or {}).get("snapshot_id"),
            "preview_url": f"{self._base_preview_path}/{handle_id}?exp={expires_at}&token={token}",
            "expires_at": datetime.fromtimestamp(expires_at, timezone.utc).isoformat(),
            "token": token,
            "runtime_policy_profile": str(record_dict["runtime_policy_profile"]),
        }

    async def validate_preview_grant(
        self,
        *,
        preview_handle: str,
        token: str,
        exp: int,
        actor_key: str,
    ) -> dict[str, Any] | None:
        handle_id = str(preview_handle or "").strip()
        expected_actor_key = str(actor_key or "").strip()
        now = int(time.time())
        if not handle_id or not token or exp <= now or not expected_actor_key:
            return None

        with self._lock:
            record = self._records.get(handle_id)
            if not record or not record.is_active:
                return None
            if record.actor_key != expected_actor_key:
                return None
            expected_sig = self._build_signature(
                preview_handle=handle_id,
                actor_key=expected_actor_key,
                exp=exp,
            )
            if not hmac.compare_digest(token, expected_sig):
                return None
            record_dict = self._record_to_dict(record)
        if not await self._is_grant_still_authorized(record):
            return None
        return record_dict

    async def resolve_preview_target(
        self,
        *,
        preview_handle: str,
        token: str,
        exp: int,
        actor_key: str,
    ) -> dict[str, Any] | None:
        record = await self.validate_preview_grant(
            preview_handle=preview_handle,
            token=token,
            exp=exp,
            actor_key=actor_key,
        )
        if not record:
            return None
        return {
            **record,
            "runtime_target_url": record.get("target_ref"),
        }

    async def _is_grant_still_authorized(self, record: PrototypePreviewHandleRecord) -> bool:
        if record.preview_scope == PrototypePreviewScope.CANONICAL:
            workspace = await self._repo.get_workspace(record.prototype_workspace_id)
            return bool(workspace) and not bool(workspace.get("is_archived"))

        if not record.prototype_session_id:
            return False
        session = await self._repo.get_session(record.prototype_session_id)
        if not session or session.get("is_revoked"):
            return False
        session_expires_at = _normalize_iso8601(session.get("expires_at"))
        if session_expires_at and session_expires_at <= _utc_now():
            return False
        return await self._session_actor_is_active(session)

    async def _assert_session_actor_active(self, session: dict[str, Any]) -> None:
        if not await self._session_actor_is_active(session):
            actor_type = str(session.get("actor_type") or "").strip().lower()
            if actor_type == "external_collaborator":
                raise RuntimeError("revoked shared actor cannot receive preview grants")
            raise RuntimeError("inactive session actor cannot receive preview grants")

    async def _session_actor_is_active(self, session: dict[str, Any]) -> bool:
        actor_type = str(session.get("actor_type") or "").strip().lower()
        if actor_type != "external_collaborator":
            return True

        shared_actor_id = session.get("actor_shared_actor_id")
        actor = await self._repo.get_shared_actor(str(shared_actor_id or ""))
        if not actor:
            return False
        if actor.get("is_revoked"):
            return False
        expires_at = _normalize_iso8601(actor.get("expires_at"))
        return not (expires_at and expires_at <= _utc_now())

    async def _resolve_runtime_policy_profile(
        self,
        *,
        workspace: dict[str, Any],
        session: dict[str, Any],
        explicit_profile: str | None,
    ) -> str:
        if explicit_profile:
            return str(explicit_profile)

        actor_type = str(session.get("actor_type") or "").strip().lower()
        if actor_type == "external_collaborator":
            actor = await self._repo.get_shared_actor(str(session.get("actor_shared_actor_id") or ""))
            if actor and actor.get("runtime_policy_profile"):
                return str(actor["runtime_policy_profile"])

        runtime_policy = workspace.get("runtime_policy") or {}
        if actor_type == "owner":
            return str(runtime_policy.get("owner_profile") or "owner_collab")
        if actor_type == "internal_collaborator":
            return str(runtime_policy.get("internal_collaborator_profile") or "internal_collab")
        return str(runtime_policy.get("external_collaborator_profile") or "locked_collab")

    def _mint_grant_token(self, *, preview_handle: str, actor_key: str) -> tuple[str, int]:
        exp = int(time.time()) + self._grant_ttl_seconds
        token = self._build_signature(
            preview_handle=preview_handle,
            actor_key=actor_key,
            exp=exp,
        )
        return token, exp

    def _build_signature(self, *, preview_handle: str, actor_key: str, exp: int) -> str:
        msg = f"{preview_handle}:{actor_key}:{int(exp)}".encode()
        return hmac.new(
            self._signing_secret.encode(),
            msg,
            hashlib.sha256,
        ).hexdigest()

    @classmethod
    def _revoke_scope_locked(cls, *, scope_id: str, revoked_at: str) -> None:
        existing_handle = cls._active_scope_handles.get(scope_id)
        if not existing_handle:
            return
        record = cls._records.get(existing_handle)
        if not record:
            cls._active_scope_handles.pop(scope_id, None)
            return
        record.is_active = False
        record.revoked_at = revoked_at
        cls._active_scope_handles.pop(scope_id, None)

    @staticmethod
    def _record_to_dict(record: PrototypePreviewHandleRecord) -> dict[str, Any]:
        return {
            "preview_handle": record.handle_id,
            "preview_scope": record.preview_scope.value,
            "scope_id": record.scope_id,
            "prototype_workspace_id": record.prototype_workspace_id,
            "prototype_session_id": record.prototype_session_id,
            "actor_key": record.actor_key,
            "target_ref": record.target_ref,
            "runtime_policy_profile": record.runtime_policy_profile,
            "metadata": dict(record.metadata),
            "is_active": bool(record.is_active),
            "created_at": record.created_at,
            "revoked_at": record.revoked_at,
        }
