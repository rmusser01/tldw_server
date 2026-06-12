"""Process-local TTL-bound policy grant store for the standalone MCP gateway."""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Mapping
from typing import Any

from .models import PolicyGrant, PolicyGrantStore, validate_grant_request


class InMemoryPolicyGrantStore:
    """Thread-safe, process-local TTL-bound policy grant store."""

    def __init__(
        self,
        *,
        token_bytes: int = 24,
        sweep_interval: int = 64,
        max_sweep_entries: int = 512,
    ) -> None:
        self._token_bytes = max(16, token_bytes)
        self._sweep_interval = max(1, sweep_interval)
        self._max_sweep_entries = max(1, max_sweep_entries)
        self._operation_count = 0
        self._grants: dict[str, PolicyGrant] = {}
        self._lock = threading.RLock()

    def create_grant(
        self,
        *,
        profile_id: str,
        grant_type: str,
        subject_type: str,
        value: str,
        ttl_seconds: int,
        actions: tuple[str, ...] = (),
        effect: str = "allow",
        session_id: str | None = None,
        user_id: str | None = None,
        granted_by: str | None = None,
        reason: str | None = None,
    ) -> PolicyGrant:
        """Create one validated grant with a generated opaque id."""

        normalized_profile_id, grant_type, subject_type, normalized_value = validate_grant_request(
            profile_id=profile_id,
            grant_type=grant_type,
            subject_type=subject_type,
            value=value,
        )
        now = time.time()
        ttl = max(1, int(ttl_seconds))
        grant = PolicyGrant(
            grant_id=secrets.token_urlsafe(self._token_bytes),
            profile_id=normalized_profile_id,
            grant_type=grant_type,
            subject_type=subject_type,
            value=normalized_value,
            expires_at=now + ttl,
            ttl_seconds=ttl,
            actions=tuple(actions),
            effect=effect,
            session_id=session_id,
            user_id=user_id,
            granted_by=granted_by,
            reason=reason,
        )
        with self._lock:
            self._maybe_sweep_expired_locked(now=now)
            self._grants[grant.grant_id] = grant
        return grant

    def list_active_grants(
        self,
        *,
        profile_id: str | None = None,
        grant_type: str | None = None,
    ) -> list[PolicyGrant]:
        """Return active grants filtered by profile and grant type."""

        now = time.time()
        with self._lock:
            self._maybe_sweep_expired_locked(now=now)
            return [
                grant
                for grant in self._grants.values()
                if grant.is_active(now)
                and (profile_id is None or grant.profile_id == profile_id)
                and (grant_type is None or grant.grant_type == grant_type)
            ]

    def revoke_grant(self, grant_id: str) -> PolicyGrant | None:
        """Remove one grant by id, returning it when it was still active."""

        now = time.time()
        with self._lock:
            grant = self._grants.pop(grant_id, None)
        if grant is None or not grant.is_active(now):
            return None
        return grant

    def find_active_grant(
        self,
        *,
        profile_id: str,
        grant_type: str,
        subject_type: str,
        value: str,
        session_id: str | None = None,
    ) -> PolicyGrant | None:
        """Return one active grant matching the normalized subject, if any."""

        try:
            normalized_profile_id, _, _, normalized_value = validate_grant_request(
                profile_id=profile_id,
                grant_type=grant_type,
                subject_type=subject_type,
                value=value,
            )
        except ValueError:
            return None
        now = time.time()
        best: PolicyGrant | None = None
        with self._lock:
            self._maybe_sweep_expired_locked(now=now)
            for grant in self._grants.values():
                if not grant.is_active(now):
                    continue
                if (
                    grant.profile_id != normalized_profile_id
                    or grant.grant_type != grant_type
                    or grant.subject_type != subject_type
                    or grant.value != normalized_value
                ):
                    continue
                if not grant.matches_session(session_id):
                    continue
                if best is None or (best.session_id is None and grant.session_id is not None):
                    best = grant
        return best

    def _maybe_sweep_expired_locked(self, *, now: float) -> None:
        self._operation_count += 1
        if self._operation_count % self._sweep_interval != 0 and len(self._grants) <= self._max_sweep_entries:
            return
        # Rotate visited entries to the end so successive bounded sweeps
        # eventually visit every entry, not just the head of the dict.
        for _ in range(min(self._max_sweep_entries, len(self._grants))):
            grant_id, grant = next(iter(self._grants.items()))
            if grant.is_active(now):
                self._grants.pop(grant_id)
                self._grants[grant_id] = grant
            else:
                del self._grants[grant_id]


def create_policy_grant_store(settings: Mapping[str, Any] | None = None) -> PolicyGrantStore:
    """Create the configured policy grant store backend.

    Defaults to the process-local memory store. Unsupported backends fail
    closed so future persistent backends cannot silently downgrade operator
    intent.
    """

    raw_backend = (settings or {}).get("grant_store_backend")
    if raw_backend is None:
        return InMemoryPolicyGrantStore()
    backend = str(raw_backend).strip().lower()
    if backend in {"memory", "in_memory"}:
        return InMemoryPolicyGrantStore()
    if backend == "sqlite":
        sqlite_path = (settings or {}).get("grant_store_sqlite_path")
        if sqlite_path is None or not str(sqlite_path).strip():
            raise ValueError("grant_store_sqlite_path is required for sqlite policy grant store")
        try:
            from .sqlite import SQLitePolicyGrantStore
        except ModuleNotFoundError as exc:
            if exc.name == "sqlalchemy":
                raise ImportError(
                    "SQLitePolicyGrantStore requires the mcp-unified sqlite extra. "
                    "Install mcp-unified[sqlite] or mcp-unified[gateway]."
                ) from exc
            raise

        return SQLitePolicyGrantStore(
            str(sqlite_path).strip(),
            timeout_seconds=float((settings or {}).get("grant_store_sqlite_timeout_seconds", 30.0)),
            cleanup_interval=int((settings or {}).get("grant_store_cleanup_interval", 64)),
            cleanup_limit=int((settings or {}).get("grant_store_cleanup_limit", 512)),
        )
    raise ValueError(f"unsupported policy grant_store_backend: {raw_backend!r}")


__all__ = [
    "InMemoryPolicyGrantStore",
    "create_policy_grant_store",
]
