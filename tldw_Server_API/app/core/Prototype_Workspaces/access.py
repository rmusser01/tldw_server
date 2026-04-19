"""Prototype workspace access helpers for external private-link collaborators."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from tldw_Server_API.app.core.AuthNZ.repos.prototype_workspaces_repo import (
    PrototypeWorkspacesRepo,
)
from tldw_Server_API.app.core.AuthNZ.settings import get_settings

PROTOTYPE_SHARED_ACTOR_COOKIE = "prototype_shared_actor"
DEFAULT_EXTERNAL_RUNTIME_POLICY_PROFILE = "locked_collab"
DEFAULT_EXTERNAL_DISPLAY_NAME = "External Collaborator"
_FALLBACK_SIGNING_SECRET = secrets.token_urlsafe(32)


class PrototypeAccessError(RuntimeError):
    """Raised when a prototype private-link exchange cannot be completed."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class PrototypeExternalAccessContext:
    shared_actor_id: str
    actor_type: str
    session_token: str
    runtime_policy_profile: str
    resume_cookie_value: str
    is_resume: bool


class PrototypeAccessService:
    """Create or resume external prototype collaborators for private-link exchange."""

    def __init__(
        self,
        repo: PrototypeWorkspacesRepo,
        *,
        session_ttl_seconds: int = 30 * 60,
        signing_secret: str | None = None,
    ) -> None:
        self._repo = repo
        self._session_ttl_seconds = max(int(session_ttl_seconds), 60)
        settings = get_settings()
        configured_signing_secret = (
            getattr(settings, "JWT_SECRET_KEY", None)
            or getattr(settings, "SINGLE_USER_API_KEY", None)
        )
        self._signing_secret = (
            signing_secret
            or configured_signing_secret
            or os.getenv("JWT_SECRET_KEY")
            or os.getenv("SINGLE_USER_API_KEY")
            or _FALLBACK_SIGNING_SECRET
        )

    async def exchange_external_collaborator(
        self,
        *,
        prototype_workspace_id: str,
        share_link_id: int,
        display_name: str | None,
        resume_cookie_value: str | None,
        allow_create: bool = True,
        expires_at: str | None = None,
    ) -> PrototypeExternalAccessContext:
        workspace = await self._repo.get_workspace(prototype_workspace_id)
        if not workspace:
            raise PrototypeAccessError("workspace_not_found", "Prototype workspace not found")
        if workspace.get("is_archived"):
            raise PrototypeAccessError("workspace_archived", "Prototype workspace is archived")

        runtime_policy = workspace.get("runtime_policy") or {}
        share_policy = workspace.get("share_policy") or {}
        runtime_policy_profile = str(
            runtime_policy.get("external_collaborator_profile")
            or runtime_policy.get("external_runtime_policy_profile")
            or DEFAULT_EXTERNAL_RUNTIME_POLICY_PROFILE
        )
        quota_policy = runtime_policy.get("external_quota_policy")
        if not isinstance(quota_policy, dict):
            quota_policy = {}

        allow_resume = _to_bool(share_policy.get("allow_browser_session_resume", True))

        actor: dict[str, Any] | None = None
        binding_secret: str | None = None
        is_resume = False
        candidate = await self._resolve_resume_candidate(
            prototype_workspace_id=prototype_workspace_id,
            share_link_id=share_link_id,
            allow_resume=allow_resume,
            resume_cookie_value=resume_cookie_value,
        )
        if candidate:
            next_binding_secret = secrets.token_urlsafe(24)
            rotated = await self._repo.rotate_shared_actor_binding(
                candidate["id"],
                new_session_binding_id=next_binding_secret,
            )
            if rotated and _is_active_resume_candidate(
                rotated,
                prototype_workspace_id=prototype_workspace_id,
                share_link_id=share_link_id,
            ):
                actor = rotated
                is_resume = True
                binding_secret = str(rotated.get("session_binding_id") or next_binding_secret)

        if actor is None:
            if not allow_create:
                raise PrototypeAccessError(
                    "resume_required",
                    "Prototype share link has exhausted new-collaborator uses",
                )
            binding_secret = secrets.token_urlsafe(24)
            actor = await self._repo.create_shared_actor(
                prototype_workspace_id=prototype_workspace_id,
                share_link_id=share_link_id,
                display_name=_normalize_display_name(display_name),
                session_binding_id=binding_secret,
                runtime_policy_profile=runtime_policy_profile,
                quota_policy=quota_policy,
                expires_at=expires_at,
            )
        if not binding_secret:
            binding_secret = str(actor.get("session_binding_id") or "")

        session_token = self._mint_session_token(
            shared_actor_id=actor["id"],
            prototype_workspace_id=prototype_workspace_id,
            share_link_id=share_link_id,
            runtime_policy_profile=runtime_policy_profile,
        )
        resume_cookie = self.encode_resume_cookie(
            shared_actor_id=actor["id"],
            binding_secret=binding_secret,
        )
        return PrototypeExternalAccessContext(
            shared_actor_id=actor["id"],
            actor_type="external_collaborator",
            session_token=session_token,
            runtime_policy_profile=runtime_policy_profile,
            resume_cookie_value=resume_cookie,
            is_resume=is_resume,
        )

    async def can_resume_external_collaborator(
        self,
        *,
        prototype_workspace_id: str,
        share_link_id: int,
        resume_cookie_value: str | None,
    ) -> bool:
        workspace = await self._repo.get_workspace(prototype_workspace_id)
        if not workspace or workspace.get("is_archived"):
            return False
        share_policy = workspace.get("share_policy") or {}
        allow_resume = _to_bool(share_policy.get("allow_browser_session_resume", True))
        candidate = await self._resolve_resume_candidate(
            prototype_workspace_id=prototype_workspace_id,
            share_link_id=share_link_id,
            allow_resume=allow_resume,
            resume_cookie_value=resume_cookie_value,
        )
        return candidate is not None

    def encode_resume_cookie(self, *, shared_actor_id: str, binding_secret: str) -> str:
        payload = {
            "shared_actor_id": shared_actor_id,
            "binding_secret": binding_secret,
            "nonce": secrets.token_urlsafe(8),
        }
        payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        payload_b64 = _b64url(payload_json)
        signature = hmac.new(
            self._signing_secret.encode("utf-8"),
            payload_b64.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        signature_b64 = _b64url(signature)
        return f"ptca.{payload_b64}.{signature_b64}"

    def decode_resume_cookie(self, value: str | None) -> dict[str, str] | None:
        if not value:
            return None
        parts = str(value).split(".")
        if len(parts) != 3 or parts[0] != "ptca":
            return None
        payload_b64, signature_b64 = parts[1], parts[2]
        expected_sig = hmac.new(
            self._signing_secret.encode("utf-8"),
            payload_b64.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        expected_sig_b64 = _b64url(expected_sig)
        if not hmac.compare_digest(expected_sig_b64, signature_b64):
            return None
        try:
            payload_raw = _b64url_decode(payload_b64)
            payload = json.loads(payload_raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        shared_actor_id = str(payload.get("shared_actor_id") or "").strip()
        binding_secret = str(payload.get("binding_secret") or "").strip()
        if not shared_actor_id or not binding_secret:
            return None
        return {"shared_actor_id": shared_actor_id, "binding_secret": binding_secret}

    async def _resolve_resume_candidate(
        self,
        *,
        prototype_workspace_id: str,
        share_link_id: int,
        allow_resume: bool,
        resume_cookie_value: str | None,
    ) -> dict[str, Any] | None:
        if not allow_resume or not resume_cookie_value:
            return None
        resume_binding = self.decode_resume_cookie(resume_cookie_value)
        if not resume_binding:
            return None
        candidate = await self._repo.get_shared_actor(resume_binding["shared_actor_id"])
        if not (
            _is_active_resume_candidate(
                candidate,
                prototype_workspace_id=prototype_workspace_id,
                share_link_id=share_link_id,
            )
            and candidate
            and candidate.get("session_binding_id")
            and hmac.compare_digest(
                str(candidate.get("session_binding_id")),
                resume_binding["binding_secret"],
            )
        ):
            return None
        return candidate

    def _mint_session_token(
        self,
        *,
        shared_actor_id: str,
        prototype_workspace_id: str,
        share_link_id: int,
        runtime_policy_profile: str,
    ) -> str:
        issued_at = int(time.time())
        expires_at = issued_at + self._session_ttl_seconds
        payload = {
            "sub": shared_actor_id,
            "workspace_id": prototype_workspace_id,
            "share_link_id": int(share_link_id),
            "actor_type": "external_collaborator",
            "runtime_policy_profile": runtime_policy_profile,
            "iat": issued_at,
            "exp": expires_at,
            "nonce": secrets.token_urlsafe(12),
        }
        payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        payload_b64 = _b64url(payload_json)
        signature = hmac.new(
            self._signing_secret.encode("utf-8"),
            payload_b64.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        signature_b64 = _b64url(signature)
        return f"ptc.{payload_b64}.{signature_b64}"


def _normalize_display_name(display_name: str | None) -> str:
    value = str(display_name or "").strip()
    if not value:
        return DEFAULT_EXTERNAL_DISPLAY_NAME
    return value[:120]


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _parse_optional_ts(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _is_active_resume_candidate(
    actor: dict[str, Any] | None,
    *,
    prototype_workspace_id: str,
    share_link_id: int,
) -> bool:
    if not actor:
        return False
    if actor.get("prototype_workspace_id") != prototype_workspace_id:
        return False
    if int(actor.get("share_link_id") or 0) != int(share_link_id):
        return False
    if actor.get("is_revoked") or actor.get("revoked_at"):
        return False
    expires_at = _parse_optional_ts(actor.get("expires_at"))
    if expires_at is not None and expires_at <= datetime.now(timezone.utc):
        return False
    return True


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(raw: str) -> bytes:
    padded = raw + ("=" * ((4 - len(raw) % 4) % 4))
    return base64.urlsafe_b64decode(padded.encode("ascii"))
