from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.config import settings as app_settings
from tldw_Server_API.app.core.DB_Management.sqlite_policy import (
    configure_sqlite_connection,
)

from .models import RunPhase, RunStatus, RuntimeType
from .utils import coerce_optional_nonempty_string

_SANDBOX_STORE_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    json.JSONDecodeError,
    sqlite3.Error,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_optional_iso_datetime(value: Any) -> str | None:
    """Serialize datetime-like values to ISO text without leaking 'None' strings."""
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    return text


def _parse_optional_iso_datetime(value: Any) -> datetime | None:
    """Parse ISO datetime text defensively; return None on empty/invalid input."""
    text = _coerce_optional_iso_datetime(value)
    if text is None:
        return None
    try:
        return datetime.fromisoformat(text)
    except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
        return None


def _normalize_string_dict(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    if value is None:
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        loaded = json.loads(text)
    except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
        return {}
    if not isinstance(loaded, dict):
        return {}
    return {str(k): str(v) for k, v in loaded.items()}


class IdempotencyConflict(Exception):
    def __init__(self, original_id: str, key: str | None = None, created_at: float | None = None, message: str = "Idempotency conflict") -> None:
        super().__init__(message)
        self.original_id = original_id
        self.key = key
        # created_at is expressed as epoch seconds (float) at the store layer
        self.created_at = created_at


class ClusterStoreUnavailable(RuntimeError):
    """Raised when cluster store backend is requested but unavailable."""


class SandboxStore:
    """Abstract store for runs, idempotency, and usage counters."""

    def check_idempotency(self, endpoint: str, user_id: Any, key: str | None, body: dict[str, Any]) -> dict[str, Any] | None:
        raise NotImplementedError

    def store_idempotency(self, endpoint: str, user_id: Any, key: str | None, body: dict[str, Any], object_id: str, response: dict[str, Any]) -> None:
        raise NotImplementedError

    def put_run(self, user_id: Any, st: RunStatus) -> None:
        raise NotImplementedError

    def get_run(self, run_id: str) -> RunStatus | None:
        raise NotImplementedError

    def update_run(self, st: RunStatus) -> None:
        raise NotImplementedError

    # Durable claim fencing helpers for run execution dispatch.
    def try_claim_run(self, run_id: str, *, worker_id: str, lease_seconds: int = 30) -> RunStatus | None:
        raise NotImplementedError

    def renew_run_claim(self, run_id: str, *, worker_id: str, lease_seconds: int = 30) -> bool:
        raise NotImplementedError

    def release_run_claim(self, run_id: str, *, worker_id: str) -> bool:
        raise NotImplementedError

    # Atomically transition a claimed queued run to starting if active-run limit allows.
    def try_admit_run_start(
        self,
        run_id: str,
        *,
        worker_id: str,
        max_active_runs: int,
        lease_seconds: int = 30,
        max_active_per_user: int = 0,
        max_active_per_persona: int = 0,
        max_active_per_workspace: int = 0,
        max_active_per_workspace_group: int = 0,
    ) -> RunStatus | None:
        raise NotImplementedError

    def get_run_owner(self, run_id: str) -> str | None:
        raise NotImplementedError

    # Session metadata APIs (owner/runtime/workspace) for cross-process durability.
    def put_session(
        self,
        user_id: Any,
        *,
        session_id: str,
        runtime: str | None,
        base_image: str | None,
        cpu_limit: float | None,
        memory_mb: int | None,
        timeout_sec: int | None,
        network_policy: str | None,
        env: dict[str, str] | None,
        labels: dict[str, str] | None,
        trust_level: str | None,
        expires_at_iso: str | None,
        workspace_path: str | None,
        persona_id: str | None = None,
        workspace_id: str | None = None,
        workspace_group_id: str | None = None,
        scope_snapshot_id: str | None = None,
    ) -> None:
        raise NotImplementedError

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def get_session_owner(self, session_id: str) -> str | None:
        raise NotImplementedError

    def list_workspace_paths_for_user_workspace(self, *, user_id: str, workspace_id: str) -> list[str]:
        raise NotImplementedError

    def delete_session(self, session_id: str) -> bool:
        raise NotImplementedError

    # ACP control-plane metadata for cross-process/cross-node session rehydration.
    def put_acp_session_control(
        self,
        *,
        session_id: str,
        user_id: Any,
        sandbox_session_id: str | None,
        run_id: str | None,
        ssh_host: str | None = None,
        ssh_port: int | None = None,
        ssh_user: str | None = None,
        ssh_private_key: str | None = None,
        persona_id: str | None = None,
        workspace_id: str | None = None,
        workspace_group_id: str | None = None,
        scope_snapshot_id: str | None = None,
    ) -> None:
        raise NotImplementedError

    def get_acp_session_control(self, session_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def delete_acp_session_control(self, session_id: str) -> bool:
        raise NotImplementedError

    # Virtualization session control metadata for persisted VM/session bookkeeping.
    def put_vz_session_control(
        self,
        *,
        session_id: str,
        runtime: str,
        vm_id: str,
        template_id: str | None,
        workspace_mount: str | None,
        agent_ready: bool,
        helper_instance_id: str | None = None,
        helper_started_at: str | None = None,
    ) -> None:
        raise NotImplementedError

    def get_vz_session_control(self, session_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    def delete_vz_session_control(self, session_id: str) -> bool:
        raise NotImplementedError

    def list_vz_session_controls(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def get_user_artifact_bytes(self, user_id: str) -> int:
        return 0

    def try_reserve_user_artifact_bytes(self, user_id: str, delta: int, cap_bytes: int) -> bool:
        if int(delta or 0) <= 0:
            self.increment_user_artifact_bytes(user_id, int(delta or 0))
            return True
        current = int(self.get_user_artifact_bytes(user_id))
        if current + int(delta) > int(cap_bytes):
            return False
        self.increment_user_artifact_bytes(user_id, int(delta))
        return True

    def increment_user_artifact_bytes(self, user_id: str, delta: int) -> None:
        pass

    # Admin listing APIs
    def list_runs(
        self,
        *,
        image_digest: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        persona_id: str | None = None,
        workspace_id: str | None = None,
        workspace_group_id: str | None = None,
        scope_snapshot_id: str | None = None,
        phase: str | None = None,
        started_at_from: str | None = None,
        started_at_to: str | None = None,
        limit: int = 50,
        offset: int = 0,
        sort_desc: bool = True,
    ) -> list[dict]:
        """Return a list of run summary rows as dicts suitable for admin list endpoints.

        Each dict contains: id, user_id, spec_version, runtime, base_image, session_id,
        persona_id, workspace_id, workspace_group_id, scope_snapshot_id, phase,
        exit_code, started_at, finished_at, message, image_digest, policy_hash
        """
        raise NotImplementedError

    def count_runs(
        self,
        *,
        image_digest: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        persona_id: str | None = None,
        workspace_id: str | None = None,
        workspace_group_id: str | None = None,
        scope_snapshot_id: str | None = None,
        phase: str | None = None,
        started_at_from: str | None = None,
        started_at_to: str | None = None,
    ) -> int:
        raise NotImplementedError

    # Admin: Idempotency listing
    def list_idempotency(
        self,
        *,
        endpoint: str | None = None,
        user_id: str | None = None,
        key: str | None = None,
        created_at_from: str | None = None,
        created_at_to: str | None = None,
        limit: int = 50,
        offset: int = 0,
        sort_desc: bool = True,
    ) -> list[dict]:
        raise NotImplementedError

    def count_idempotency(
        self,
        *,
        endpoint: str | None = None,
        user_id: str | None = None,
        key: str | None = None,
        created_at_from: str | None = None,
        created_at_to: str | None = None,
    ) -> int:
        raise NotImplementedError

    # Admin: Usage aggregates per user
    def list_usage(
        self,
        *,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
        sort_desc: bool = True,
    ) -> list[dict]:
        raise NotImplementedError

    def count_usage(
        self,
        *,
        user_id: str | None = None,
    ) -> int:
        raise NotImplementedError

    def _coerce_created_at(self, value: str | int | float) -> float:
        """Coerce created_at filter to epoch seconds.

        Accepts ISO-8601 strings (including trailing 'Z'), ints, or floats.
        Raises ValueError if not parseable.
        """
        txt = str(value).strip()
        if txt.endswith("Z"):
            txt = txt[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(txt).timestamp()
        except ValueError:
            try:
                return float(txt)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid created_at filter: {value!r}") from None

    # Durable run queue methods for cross-restart persistence.
    def enqueue_run(self, run_id: str, user_id: str, priority: int = 0) -> None:
        """Add a run to the persistent queue."""
        raise NotImplementedError

    def dequeue_run(self, worker_id: str) -> dict | None:
        """Remove and return the highest-priority queued run, or None.

        Returns a dict with keys: run_id, user_id, priority, enqueued_at.
        Within the same priority, FIFO order (earliest enqueued_at first).
        """
        raise NotImplementedError

    def remove_from_queue(self, run_id: str) -> bool:
        """Remove a specific run from the queue. Returns True if it was present."""
        raise NotImplementedError

    # Optional: TTL GC for idempotency
    def gc_idempotency(self) -> int:
        """Garbage-collect expired idempotency records.

        Returns the number of records deleted. Default implementation does
        nothing and returns 0; concrete backends may override.
        """
        return 0


class InMemoryStore(SandboxStore):
    def __init__(self, idem_ttl_sec: int = 600) -> None:
        self.idem_ttl_sec = idem_ttl_sec
        self._idem: dict[tuple[str, str, str], tuple[float, str, dict[str, Any], str]] = {}
        self._runs: dict[str, RunStatus] = {}
        self._owners: dict[str, str] = {}
        self._sessions: dict[str, dict[str, Any]] = {}
        self._acp_sessions: dict[str, dict[str, Any]] = {}
        self._vz_sessions: dict[str, dict[str, Any]] = {}
        self._user_bytes: dict[str, int] = {}
        self._run_queue: list[dict[str, Any]] = []
        self._lock = threading.RLock()

    def _fp(self, body: dict[str, Any]) -> str:
        try:
            canon = json.dumps(body, sort_keys=True, separators=(",", ":"))
        except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
            canon = str(body)
        import hashlib
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()

    def _user_key(self, user_id: Any) -> str:
        try:
            return str(user_id)
        except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
            return ""

    def _gc_idem(self) -> int:
        now = time.time()
        expired = [k for k, (ts, _fp, _resp, _oid) in self._idem.items() if now - ts > self.idem_ttl_sec]
        for k in expired:
            self._idem.pop(k, None)
        return len(expired)

    def check_idempotency(self, endpoint: str, user_id: Any, key: str | None, body: dict[str, Any]) -> dict[str, Any] | None:
        if not key:
            return None
        with self._lock:
            self._gc_idem()
            idx = (endpoint, self._user_key(user_id), key)
            rec = self._idem.get(idx)
            if not rec:
                return None
            ts, fp_saved, resp, obj_id = rec
            fp_new = self._fp(body)
            if fp_new == fp_saved:
                return resp
            # include key and created_at (epoch seconds) for richer error details upstream
            raise IdempotencyConflict(obj_id, key=key, created_at=ts)

    def store_idempotency(self, endpoint: str, user_id: Any, key: str | None, body: dict[str, Any], object_id: str, response: dict[str, Any]) -> None:
        if not key:
            return
        with self._lock:
            idx = (endpoint, self._user_key(user_id), key)
            if idx not in self._idem:
                self._idem[idx] = (time.time(), self._fp(body), response, object_id)

    def gc_idempotency(self) -> int:
        with self._lock:
            return self._gc_idem()

    def put_run(self, user_id: Any, st: RunStatus) -> None:
        with self._lock:
            self._runs[st.id] = st
            self._owners[st.id] = self._user_key(user_id)

    def get_run(self, run_id: str) -> RunStatus | None:
        with self._lock:
            return self._runs.get(run_id)

    def update_run(self, st: RunStatus) -> None:
        if st.phase in (RunPhase.completed, RunPhase.failed, RunPhase.killed, RunPhase.timed_out):
            st.claim_owner = None
            st.claim_expires_at = None
        with self._lock:
            self._runs[st.id] = st

    def try_claim_run(self, run_id: str, *, worker_id: str, lease_seconds: int = 30) -> RunStatus | None:
        wid = str(worker_id or "").strip()
        if not wid:
            return None
        ttl = max(1, int(lease_seconds or 0))
        now = datetime.now(timezone.utc)
        expires = now + timedelta(seconds=ttl)
        with self._lock:
            st = self._runs.get(str(run_id))
            if st is None or st.phase != RunPhase.queued:
                return None
            owner = str(getattr(st, "claim_owner", "") or "").strip()
            exp = getattr(st, "claim_expires_at", None)
            if owner and owner != wid and isinstance(exp, datetime) and exp > now:
                return None
            st.claim_owner = wid
            st.claim_expires_at = expires
            self._runs[st.id] = st
            return st

    def renew_run_claim(self, run_id: str, *, worker_id: str, lease_seconds: int = 30) -> bool:
        wid = str(worker_id or "").strip()
        if not wid:
            return False
        ttl = max(1, int(lease_seconds or 0))
        with self._lock:
            st = self._runs.get(str(run_id))
            if st is None:
                return False
            owner = str(getattr(st, "claim_owner", "") or "").strip()
            if owner != wid:
                return False
            st.claim_expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)
            self._runs[st.id] = st
            return True

    def release_run_claim(self, run_id: str, *, worker_id: str) -> bool:
        wid = str(worker_id or "").strip()
        if not wid:
            return False
        with self._lock:
            st = self._runs.get(str(run_id))
            if st is None:
                return False
            owner = str(getattr(st, "claim_owner", "") or "").strip()
            if owner != wid:
                return False
            st.claim_owner = None
            st.claim_expires_at = None
            self._runs[st.id] = st
            return True

    def try_admit_run_start(
        self,
        run_id: str,
        *,
        worker_id: str,
        max_active_runs: int,
        lease_seconds: int = 30,
        max_active_per_user: int = 0,
        max_active_per_persona: int = 0,
        max_active_per_workspace: int = 0,
        max_active_per_workspace_group: int = 0,
    ) -> RunStatus | None:
        wid = str(worker_id or "").strip()
        if not wid:
            return None
        limit = max(1, int(max_active_runs or 0))
        ttl = max(1, int(lease_seconds or 0))
        per_user_limit = max(0, int(max_active_per_user or 0))
        per_persona_limit = max(0, int(max_active_per_persona or 0))
        per_workspace_limit = max(0, int(max_active_per_workspace or 0))
        per_workspace_group_limit = max(0, int(max_active_per_workspace_group or 0))
        now = datetime.now(timezone.utc)
        with self._lock:
            st = self._runs.get(str(run_id))
            if st is None or st.phase != RunPhase.queued:
                return None
            owner = str(getattr(st, "claim_owner", "") or "").strip()
            if owner != wid:
                return None
            exp = getattr(st, "claim_expires_at", None)
            if isinstance(exp, datetime) and exp <= now:
                return None
            target_user = self._owners.get(st.id)
            target_persona = getattr(st, "persona_id", None)
            target_workspace = getattr(st, "workspace_id", None)
            target_workspace_group = getattr(st, "workspace_group_id", None)
            active = 0
            active_user = 0
            active_persona = 0
            active_workspace = 0
            active_workspace_group = 0
            for rs in self._runs.values():
                is_active = False
                if rs.phase == RunPhase.running:
                    is_active = True
                elif rs.phase == RunPhase.starting and getattr(rs, "started_at", None) is not None:
                    is_active = True
                if not is_active:
                    continue
                active += 1
                if target_user and self._owners.get(rs.id) == target_user:
                    active_user += 1
                if target_persona and getattr(rs, "persona_id", None) == target_persona:
                    active_persona += 1
                if target_workspace and getattr(rs, "workspace_id", None) == target_workspace:
                    active_workspace += 1
                if target_workspace_group and getattr(rs, "workspace_group_id", None) == target_workspace_group:
                    active_workspace_group += 1
            if active >= limit:
                return None
            if per_user_limit > 0 and target_user and active_user >= per_user_limit:
                return None
            if per_persona_limit > 0 and target_persona and active_persona >= per_persona_limit:
                return None
            if per_workspace_limit > 0 and target_workspace and active_workspace >= per_workspace_limit:
                return None
            if per_workspace_group_limit > 0 and target_workspace_group and active_workspace_group >= per_workspace_group_limit:
                return None
            st.phase = RunPhase.starting
            st.started_at = now
            st.finished_at = None
            st.exit_code = None
            st.claim_expires_at = now + timedelta(seconds=ttl)
            self._runs[st.id] = st
            return st

    def get_run_owner(self, run_id: str) -> str | None:
        with self._lock:
            return self._owners.get(run_id)

    def put_session(
        self,
        user_id: Any,
        *,
        session_id: str,
        runtime: str | None,
        base_image: str | None,
        cpu_limit: float | None,
        memory_mb: int | None,
        timeout_sec: int | None,
        network_policy: str | None,
        env: dict[str, str] | None,
        labels: dict[str, str] | None,
        trust_level: str | None,
        expires_at_iso: str | None,
        workspace_path: str | None,
        persona_id: str | None = None,
        workspace_id: str | None = None,
        workspace_group_id: str | None = None,
        scope_snapshot_id: str | None = None,
    ) -> None:
        with self._lock:
            self._sessions[str(session_id)] = {
                "id": str(session_id),
                "user_id": self._user_key(user_id),
                "runtime": runtime,
                "base_image": base_image,
                "cpu_limit": (float(cpu_limit) if cpu_limit is not None else None),
                "memory_mb": (int(memory_mb) if memory_mb is not None else None),
                "timeout_sec": (int(timeout_sec) if timeout_sec is not None else None),
                "network_policy": (str(network_policy) if network_policy is not None else None),
                "env": _normalize_string_dict(env),
                "labels": _normalize_string_dict(labels),
                "trust_level": (str(trust_level) if trust_level is not None else None),
                "expires_at": expires_at_iso,
                "workspace_path": workspace_path,
                "persona_id": (str(persona_id) if persona_id is not None else None),
                "workspace_id": (str(workspace_id) if workspace_id is not None else None),
                "workspace_group_id": (str(workspace_group_id) if workspace_group_id is not None else None),
                "scope_snapshot_id": (str(scope_snapshot_id) if scope_snapshot_id is not None else None),
            }

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._sessions.get(str(session_id))
            return dict(row) if isinstance(row, dict) else None

    def get_session_owner(self, session_id: str) -> str | None:
        with self._lock:
            row = self._sessions.get(str(session_id))
            if not isinstance(row, dict):
                return None
            owner = row.get("user_id")
            return str(owner) if owner is not None else None

    def list_workspace_paths_for_user_workspace(self, *, user_id: str, workspace_id: str) -> list[str]:
        owner_key = self._user_key(user_id)
        workspace_key = str(workspace_id or "").strip()
        if not owner_key or not workspace_key:
            return []
        now = datetime.now(timezone.utc)
        paths: list[str] = []
        with self._lock:
            for row in self._sessions.values():
                if not isinstance(row, dict):
                    continue
                if str(row.get("user_id") or "") != owner_key:
                    continue
                if str(row.get("workspace_id") or "") != workspace_key:
                    continue
                expires_at = _parse_optional_iso_datetime(row.get("expires_at"))
                if expires_at is not None and expires_at <= now:
                    continue
                workspace_path = str(row.get("workspace_path") or "").strip()
                if workspace_path:
                    paths.append(workspace_path)
        return paths

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(str(session_id), None) is not None

    def put_acp_session_control(
        self,
        *,
        session_id: str,
        user_id: Any,
        sandbox_session_id: str | None,
        run_id: str | None,
        ssh_host: str | None = None,
        ssh_port: int | None = None,
        ssh_user: str | None = None,
        ssh_private_key: str | None = None,
        persona_id: str | None = None,
        workspace_id: str | None = None,
        workspace_group_id: str | None = None,
        scope_snapshot_id: str | None = None,
    ) -> None:
        now_ts = time.time()
        with self._lock:
            existing = self._acp_sessions.get(str(session_id), {})
            created_at = existing.get("created_at", now_ts)
            self._acp_sessions[str(session_id)] = {
                "id": str(session_id),
                "user_id": self._user_key(user_id),
                "sandbox_session_id": (str(sandbox_session_id) if sandbox_session_id is not None else None),
                "run_id": (str(run_id) if run_id is not None else None),
                "ssh_host": (str(ssh_host) if ssh_host is not None else None),
                "ssh_port": (int(ssh_port) if ssh_port is not None else None),
                "ssh_user": (str(ssh_user) if ssh_user is not None else None),
                "ssh_private_key": (str(ssh_private_key) if ssh_private_key is not None else None),
                "persona_id": (str(persona_id) if persona_id is not None else None),
                "workspace_id": (str(workspace_id) if workspace_id is not None else None),
                "workspace_group_id": (str(workspace_group_id) if workspace_group_id is not None else None),
                "scope_snapshot_id": (str(scope_snapshot_id) if scope_snapshot_id is not None else None),
                "created_at": float(created_at),
                "updated_at": float(now_ts),
            }

    def get_acp_session_control(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._acp_sessions.get(str(session_id))
            return dict(row) if isinstance(row, dict) else None

    def delete_acp_session_control(self, session_id: str) -> bool:
        with self._lock:
            return self._acp_sessions.pop(str(session_id), None) is not None

    def put_vz_session_control(
        self,
        *,
        session_id: str,
        runtime: str,
        vm_id: str,
        template_id: str | None,
        workspace_mount: str | None,
        agent_ready: bool,
        helper_instance_id: str | None = None,
        helper_started_at: str | None = None,
    ) -> None:
        now_ts = time.time()
        with self._lock:
            existing = self._vz_sessions.get(str(session_id), {})
            created_at = existing.get("created_at", now_ts)
            self._vz_sessions[str(session_id)] = {
                "id": str(session_id),
                "runtime": str(runtime),
                "vm_id": str(vm_id),
                "template_id": (str(template_id) if template_id is not None else None),
                "workspace_mount": (str(workspace_mount) if workspace_mount is not None else None),
                "agent_ready": bool(agent_ready),
                "helper_instance_id": coerce_optional_nonempty_string(helper_instance_id),
                "helper_started_at": coerce_optional_nonempty_string(helper_started_at),
                "created_at": float(created_at),
                "updated_at": float(now_ts),
            }

    def get_vz_session_control(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._vz_sessions.get(str(session_id))
            return dict(row) if isinstance(row, dict) else None

    def delete_vz_session_control(self, session_id: str) -> bool:
        with self._lock:
            return self._vz_sessions.pop(str(session_id), None) is not None

    def list_vz_session_controls(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                dict(row)
                for row in self._vz_sessions.values()
                if isinstance(row, dict)
            ]

    def get_user_artifact_bytes(self, user_id: str) -> int:
        with self._lock:
            return int(self._user_bytes.get(user_id, 0))

    def increment_user_artifact_bytes(self, user_id: str, delta: int) -> None:
        with self._lock:
            cur = int(self._user_bytes.get(user_id, 0))
            self._user_bytes[user_id] = max(0, cur + int(delta))

    def try_reserve_user_artifact_bytes(self, user_id: str, delta: int, cap_bytes: int) -> bool:
        d = int(delta or 0)
        if d <= 0:
            self.increment_user_artifact_bytes(user_id, d)
            return True
        with self._lock:
            cur = int(self._user_bytes.get(user_id, 0))
            if cur + d > int(cap_bytes):
                return False
            self._user_bytes[user_id] = cur + d
            return True

    def list_runs(
        self,
        *,
        image_digest: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        persona_id: str | None = None,
        workspace_id: str | None = None,
        workspace_group_id: str | None = None,
        scope_snapshot_id: str | None = None,
        phase: str | None = None,
        started_at_from: str | None = None,
        started_at_to: str | None = None,
        limit: int = 50,
        offset: int = 0,
        sort_desc: bool = True,
    ) -> list[dict]:
        from datetime import datetime
        with self._lock:
            rows = []
            for st in self._runs.values():
                if image_digest and (st.image_digest or None) != image_digest:
                    continue
                if user_id and self._owners.get(st.id) != user_id:
                    continue
                if session_id and (getattr(st, "session_id", None) != session_id):
                    continue
                if persona_id and (getattr(st, "persona_id", None) != persona_id):
                    continue
                if workspace_id and (getattr(st, "workspace_id", None) != workspace_id):
                    continue
                if workspace_group_id and (getattr(st, "workspace_group_id", None) != workspace_group_id):
                    continue
                if scope_snapshot_id and (getattr(st, "scope_snapshot_id", None) != scope_snapshot_id):
                    continue
                if phase and st.phase.value != phase:
                    continue
                sa = st.started_at
                if started_at_from:
                    try:
                        dt_from = datetime.fromisoformat(started_at_from)
                        if not (sa and sa >= dt_from):
                            continue
                    except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                        pass
                if started_at_to:
                    try:
                        dt_to = datetime.fromisoformat(started_at_to)
                        if not (sa and sa <= dt_to):
                            continue
                    except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                        pass
                rows.append({
                    "id": st.id,
                    "user_id": self._owners.get(st.id),
                    "spec_version": st.spec_version,
                    "runtime": (st.runtime.value if st.runtime else None),
                    "runtime_version": getattr(st, "runtime_version", None),
                    "base_image": st.base_image,
                    "session_id": getattr(st, "session_id", None),
                    "persona_id": getattr(st, "persona_id", None),
                    "workspace_id": getattr(st, "workspace_id", None),
                    "workspace_group_id": getattr(st, "workspace_group_id", None),
                    "scope_snapshot_id": getattr(st, "scope_snapshot_id", None),
                    "phase": st.phase.value,
                    "exit_code": st.exit_code,
                    "started_at": (st.started_at.isoformat() if st.started_at else None),
                    "finished_at": (st.finished_at.isoformat() if st.finished_at else None),
                    "message": st.message,
                    "image_digest": st.image_digest,
                    "policy_hash": st.policy_hash,
                    "resource_usage": st.resource_usage,
                })
        def _key(r: dict):
            return r.get("started_at") or ""
        rows.sort(key=_key, reverse=bool(sort_desc))
        return rows[offset: offset + limit]

    def count_runs(
        self,
        *,
        image_digest: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        persona_id: str | None = None,
        workspace_id: str | None = None,
        workspace_group_id: str | None = None,
        scope_snapshot_id: str | None = None,
        phase: str | None = None,
        started_at_from: str | None = None,
        started_at_to: str | None = None,
    ) -> int:
        return len(self.list_runs(
            image_digest=image_digest,
            user_id=user_id,
            session_id=session_id,
            persona_id=persona_id,
            workspace_id=workspace_id,
            workspace_group_id=workspace_group_id,
            scope_snapshot_id=scope_snapshot_id,
            phase=phase,
            started_at_from=started_at_from,
            started_at_to=started_at_to,
            limit=10**9,
            offset=0,
            sort_desc=True,
        ))

    def list_idempotency(
        self,
        *,
        endpoint: str | None = None,
        user_id: str | None = None,
        key: str | None = None,
        created_at_from: str | None = None,
        created_at_to: str | None = None,
        limit: int = 50,
        offset: int = 0,
        sort_desc: bool = True,
    ) -> list[dict]:
        with self._lock:
            rows = []
            for (ep, uid, k), (ts, fp, _resp, oid) in self._idem.items():
                if endpoint and ep != endpoint:
                    continue
                if user_id and uid != user_id:
                    continue
                if key and k != key:
                    continue
                if created_at_from:
                    try:
                        from datetime import datetime
                        dt_from = datetime.fromisoformat(created_at_from)
                        if ts < dt_from.timestamp():
                            continue
                    except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                        pass
                if created_at_to:
                    try:
                        from datetime import datetime
                        dt_to = datetime.fromisoformat(created_at_to)
                        if ts > dt_to.timestamp():
                            continue
                    except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                        pass
                from datetime import datetime, timezone
                rows.append({
                    "endpoint": ep,
                    "user_id": uid,
                    "key": k,
                    "fingerprint": fp,
                    "object_id": oid,
                    "created_at": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                })
        rows.sort(key=lambda r: r.get("created_at") or "", reverse=bool(sort_desc))
        return rows[offset: offset + limit]

    def count_idempotency(
        self,
        *,
        endpoint: str | None = None,
        user_id: str | None = None,
        key: str | None = None,
        created_at_from: str | None = None,
        created_at_to: str | None = None,
    ) -> int:
        return len(self.list_idempotency(
            endpoint=endpoint,
            user_id=user_id,
            key=key,
            created_at_from=created_at_from,
            created_at_to=created_at_to,
            limit=10**9,
            offset=0,
            sort_desc=True,
        ))

    def list_usage(
        self,
        *,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
        sort_desc: bool = True,
    ) -> list[dict]:
        # Aggregate runs_count and log_bytes from runs; artifact_bytes from _user_bytes
        with self._lock:
            users = set(self._owners.values())
            if user_id:
                users = {u for u in users if u == user_id}
            items: list[dict] = []
            for uid in sorted(users):
                runs = [r for r_id, r in self._runs.items() if self._owners.get(r_id) == uid]
                runs_count = len(runs)
                log_bytes = 0
                for st in runs:
                    try:
                        if st.resource_usage and isinstance(st.resource_usage.get("log_bytes"), int):
                            log_bytes += int(st.resource_usage.get("log_bytes") or 0)
                    except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                        continue
                art_bytes = int(self._user_bytes.get(uid, 0))
                items.append({
                    "user_id": uid,
                    "runs_count": int(runs_count),
                    "log_bytes": int(log_bytes),
                    "artifact_bytes": int(art_bytes),
                })
        items.sort(key=lambda r: r.get("user_id") or "", reverse=bool(sort_desc))
        return items[offset: offset + limit]

    # -- Durable run queue (in-memory) ------------------------------------------
    def enqueue_run(self, run_id: str, user_id: str, priority: int = 0) -> None:
        with self._lock:
            self._run_queue.append({
                "run_id": str(run_id),
                "user_id": str(user_id),
                "priority": int(priority),
                "enqueued_at": time.time(),
            })
            # Keep sorted: highest priority first, then earliest enqueued_at
            self._run_queue.sort(key=lambda e: (-e["priority"], e["enqueued_at"]))

    def dequeue_run(self, worker_id: str) -> dict | None:
        with self._lock:
            if not self._run_queue:
                return None
            return self._run_queue.pop(0)

    def remove_from_queue(self, run_id: str) -> bool:
        rid = str(run_id)
        with self._lock:
            before = len(self._run_queue)
            self._run_queue = [e for e in self._run_queue if e["run_id"] != rid]
            return len(self._run_queue) < before

    def count_usage(
        self,
        *,
        user_id: str | None = None,
    ) -> int:
        return len(self.list_usage(user_id=user_id, limit=10**9, offset=0, sort_desc=True))


class SQLiteStore(SandboxStore):
    def __init__(self, db_path: str | None = None, idem_ttl_sec: int = 600) -> None:
        self.idem_ttl_sec = idem_ttl_sec
        if not db_path:
            try:
                proj = getattr(app_settings, "PROJECT_ROOT", ".")
            except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                proj = "."
            db_path = str(Path(str(proj)) / "tmp_dir" / "sandbox" / "meta" / "sandbox_store.db")
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

        # Delegate run queue SQL to DB_Management abstraction
        from tldw_Server_API.app.core.DB_Management.Sandbox_Queue_DB import SQLiteRunQueueDB
        self._queue_db = SQLiteRunQueueDB(conn_factory=self._conn, lock=self._lock)

    def _conn(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        configure_sqlite_connection(con)
        con.row_factory = sqlite3.Row
        return con

    def _init_db(self) -> None:
        """
        Initialize the SQLite database schema for sandbox storage and apply a migration to add the `resource_usage` column if it is missing.

        Creates the tables used by the store: `sandbox_runs` (run metadata, including `resource_usage`), `sandbox_idempotency` (idempotency records keyed by endpoint, user_key, and key), and `sandbox_usage` (per-user artifact byte usage). After creating tables, attempts to add the `resource_usage` column to `sandbox_runs` for backfill compatibility; ignores only the specific "column already exists" error and re-raises any other migration failure.
        """
        with self._conn() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS sandbox_runs (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    spec_version TEXT,
                    runtime TEXT,
                    runtime_version TEXT,
                    base_image TEXT,
                    session_id TEXT,
                    persona_id TEXT,
                    workspace_id TEXT,
                    workspace_group_id TEXT,
                    scope_snapshot_id TEXT,
                    claim_owner TEXT,
                    claim_expires_at TEXT,
                    phase TEXT,
                    exit_code INTEGER,
                    started_at TEXT,
                    finished_at TEXT,
                    message TEXT,
                    image_digest TEXT,
                    policy_hash TEXT,
                    resource_usage TEXT
                );
                CREATE TABLE IF NOT EXISTS sandbox_idempotency (
                    endpoint TEXT,
                    user_key TEXT,
                    key TEXT,
                    fingerprint TEXT,
                    object_id TEXT,
                    response_body TEXT,
                    created_at REAL,
                    PRIMARY KEY (endpoint, user_key, key)
                );
                CREATE TABLE IF NOT EXISTS sandbox_usage (
                    user_id TEXT PRIMARY KEY,
                    artifact_bytes INTEGER
                );
                CREATE TABLE IF NOT EXISTS sandbox_sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    runtime TEXT,
                    base_image TEXT,
                    cpu_limit REAL,
                    memory_mb INTEGER,
                    timeout_sec INTEGER,
                    network_policy TEXT,
                    env TEXT,
                    labels TEXT,
                    trust_level TEXT,
                    persona_id TEXT,
                    workspace_id TEXT,
                    workspace_group_id TEXT,
                    scope_snapshot_id TEXT,
                    expires_at TEXT,
                    workspace_path TEXT,
                    created_at REAL,
                    updated_at REAL
                );
                CREATE TABLE IF NOT EXISTS sandbox_acp_sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    sandbox_session_id TEXT,
                    run_id TEXT,
                    ssh_host TEXT,
                    ssh_port INTEGER,
                    ssh_user TEXT,
                    ssh_private_key TEXT,
                    persona_id TEXT,
                    workspace_id TEXT,
                    workspace_group_id TEXT,
                    scope_snapshot_id TEXT,
                    created_at REAL,
                    updated_at REAL
                );
                CREATE TABLE IF NOT EXISTS run_queue (
                    run_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    priority INTEGER DEFAULT 0,
                    enqueued_at TEXT NOT NULL DEFAULT (datetime('now'))
                );
                CREATE TABLE IF NOT EXISTS sandbox_vz_sessions (
                    id TEXT PRIMARY KEY,
                    runtime TEXT,
                    vm_id TEXT,
                    template_id TEXT,
                    workspace_mount TEXT,
                    agent_ready INTEGER,
                    helper_instance_id TEXT,
                    helper_started_at TEXT,
                    created_at REAL,
                    updated_at REAL
                );
                """
            )
            # Backfill migrations for older schemas.
            def _ensure_sqlite_column(table: str, column: str, coltype: str) -> None:
                try:
                    con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
                except sqlite3.OperationalError as e:
                    msg = str(e).lower()
                    if (
                        "duplicate" in msg
                        or "already exists" in msg
                        or "duplicate column" in msg
                    ):
                        logger.debug(
                            "SQLite migration: {}.{} already exists; skipping ALTER TABLE",
                            table,
                            column,
                        )
                        return
                    logger.exception(
                        "SQLite migration failed adding {}.{}",
                        table,
                        column,
                    )
                    raise

            _ensure_sqlite_column("sandbox_runs", "resource_usage", "TEXT")
            _ensure_sqlite_column("sandbox_runs", "runtime_version", "TEXT")
            _ensure_sqlite_column("sandbox_runs", "session_id", "TEXT")
            _ensure_sqlite_column("sandbox_runs", "persona_id", "TEXT")
            _ensure_sqlite_column("sandbox_runs", "workspace_id", "TEXT")
            _ensure_sqlite_column("sandbox_runs", "workspace_group_id", "TEXT")
            _ensure_sqlite_column("sandbox_runs", "scope_snapshot_id", "TEXT")
            _ensure_sqlite_column("sandbox_runs", "claim_owner", "TEXT")
            _ensure_sqlite_column("sandbox_runs", "claim_expires_at", "TEXT")
            _ensure_sqlite_column("sandbox_sessions", "cpu_limit", "REAL")
            _ensure_sqlite_column("sandbox_sessions", "memory_mb", "INTEGER")
            _ensure_sqlite_column("sandbox_sessions", "timeout_sec", "INTEGER")
            _ensure_sqlite_column("sandbox_sessions", "network_policy", "TEXT")
            _ensure_sqlite_column("sandbox_sessions", "env", "TEXT")
            _ensure_sqlite_column("sandbox_sessions", "labels", "TEXT")
            _ensure_sqlite_column("sandbox_sessions", "trust_level", "TEXT")
            _ensure_sqlite_column("sandbox_sessions", "persona_id", "TEXT")
            _ensure_sqlite_column("sandbox_sessions", "workspace_id", "TEXT")
            _ensure_sqlite_column("sandbox_sessions", "workspace_group_id", "TEXT")
            _ensure_sqlite_column("sandbox_sessions", "scope_snapshot_id", "TEXT")
            _ensure_sqlite_column("sandbox_acp_sessions", "sandbox_session_id", "TEXT")
            _ensure_sqlite_column("sandbox_acp_sessions", "run_id", "TEXT")
            _ensure_sqlite_column("sandbox_acp_sessions", "ssh_host", "TEXT")
            _ensure_sqlite_column("sandbox_acp_sessions", "ssh_port", "INTEGER")
            _ensure_sqlite_column("sandbox_acp_sessions", "ssh_user", "TEXT")
            _ensure_sqlite_column("sandbox_acp_sessions", "ssh_private_key", "TEXT")
            _ensure_sqlite_column("sandbox_acp_sessions", "persona_id", "TEXT")
            _ensure_sqlite_column("sandbox_acp_sessions", "workspace_id", "TEXT")
            _ensure_sqlite_column("sandbox_acp_sessions", "workspace_group_id", "TEXT")
            _ensure_sqlite_column("sandbox_acp_sessions", "scope_snapshot_id", "TEXT")
            _ensure_sqlite_column("sandbox_vz_sessions", "helper_instance_id", "TEXT")
            _ensure_sqlite_column("sandbox_vz_sessions", "helper_started_at", "TEXT")

    def _fp(self, body: dict[str, Any]) -> str:
        """
        Compute a stable SHA-256 fingerprint for a JSON-like body.

        The function canonicalizes `body` to a deterministic JSON string (keys sorted, compact separators) and falls back to `str(body)` if serialization fails, then returns the SHA-256 hex digest of that string.

        Parameters:
            body (Dict[str, Any]): The JSON-like object to fingerprint; should be JSON-serializable when possible.

        Returns:
            str: Hexadecimal SHA-256 digest of the canonicalized representation.
        """
        try:
            canon = json.dumps(body, sort_keys=True, separators=(",", ":"))
        except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
            canon = str(body)
        import hashlib
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()

    def _user_key(self, user_id: Any) -> str:
        try:
            return str(user_id)
        except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
            return ""

    def _gc_idem(self, con: sqlite3.Connection) -> int:
        try:
            ttl = max(1, int(self.idem_ttl_sec))
        except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
            ttl = 600
        cutoff = time.time() - ttl
        cur = con.execute("SELECT COUNT(*) FROM sandbox_idempotency WHERE created_at < ?", (cutoff,))
        row = cur.fetchone()
        n = int(row[0]) if row else 0
        con.execute("DELETE FROM sandbox_idempotency WHERE created_at < ?", (cutoff,))
        return n

    def check_idempotency(self, endpoint: str, user_id: Any, key: str | None, body: dict[str, Any]) -> dict[str, Any] | None:
        if not key:
            return None
        with self._lock, self._conn() as con:
            self._gc_idem(con)
            cur = con.execute(
                "SELECT fingerprint, response_body, object_id, created_at FROM sandbox_idempotency WHERE endpoint=? AND user_key=? AND key=?",
                (endpoint, self._user_key(user_id), key),
            )
            row = cur.fetchone()
            if not row:
                return None
            fp_new = self._fp(body)
            if row["fingerprint"] == fp_new:
                try:
                    return json.loads(row["response_body"]) if row["response_body"] else None
                except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                    return None
            # include key and created_at (epoch seconds) from the row for richer error details upstream
            try:
                ct = float(row["created_at"]) if row["created_at"] is not None else None
            except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                ct = None
            raise IdempotencyConflict(row["object_id"], key=key, created_at=ct)

    def store_idempotency(self, endpoint: str, user_id: Any, key: str | None, body: dict[str, Any], object_id: str, response: dict[str, Any]) -> None:
        if not key:
            return
        with self._lock, self._conn() as con:
            try:
                con.execute(
                    "INSERT OR IGNORE INTO sandbox_idempotency(endpoint,user_key,key,fingerprint,object_id,response_body,created_at) VALUES (?,?,?,?,?,?,?)",
                    (
                        endpoint,
                        self._user_key(user_id),
                        key,
                        self._fp(body),
                        object_id,
                        json.dumps(response, ensure_ascii=False),
                        time.time(),
                    ),
                )
            except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"idempotency store failed: {e}")

    def gc_idempotency(self) -> int:
        """One-shot TTL GC for idempotency rows; returns number of deleted rows."""
        with self._lock, self._conn() as con:
            try:
                return self._gc_idem(con)
            except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                return 0

    def put_run(self, user_id: Any, st: RunStatus) -> None:
        """
        Persist a RunStatus for the given user, replacing any existing record with the same run id.

        Stores the run fields (id, user_id, spec_version, runtime, base_image, phase, exit_code,
        started_at, finished_at, message, image_digest, policy_hash) into the backend. Timestamps
        are stored as ISO 8601 strings; if `resource_usage` is a dict it is serialized to JSON and stored,
        otherwise `resource_usage` is stored as null.

        Parameters:
            user_id: Identifier of the user who owns the run; converted to a canonical string for storage.
            st (RunStatus): RunStatus object to persist.
        """
        with self._lock, self._conn() as con:
            con.execute(
                (
                    "REPLACE INTO sandbox_runs("
                    "id,user_id,spec_version,runtime,runtime_version,base_image,"
                    "session_id,persona_id,workspace_id,workspace_group_id,scope_snapshot_id,"
                    "claim_owner,claim_expires_at,"
                    "phase,exit_code,started_at,finished_at,message,image_digest,policy_hash,resource_usage"
                    ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
                ),
                (
                    st.id,
                    self._user_key(user_id),
                    st.spec_version,
                    (st.runtime.value if st.runtime else None),
                    (st.runtime_version if getattr(st, "runtime_version", None) else None),
                    st.base_image,
                    (str(st.session_id) if getattr(st, "session_id", None) is not None else None),
                    (str(st.persona_id) if getattr(st, "persona_id", None) is not None else None),
                    (str(st.workspace_id) if getattr(st, "workspace_id", None) is not None else None),
                    (str(st.workspace_group_id) if getattr(st, "workspace_group_id", None) is not None else None),
                    (str(st.scope_snapshot_id) if getattr(st, "scope_snapshot_id", None) is not None else None),
                    (str(st.claim_owner) if getattr(st, "claim_owner", None) is not None else None),
                    _coerce_optional_iso_datetime(getattr(st, "claim_expires_at", None)),
                    st.phase.value,
                    st.exit_code,
                    (st.started_at.isoformat() if st.started_at else None),
                    (st.finished_at.isoformat() if st.finished_at else None),
                    st.message,
                    st.image_digest,
                    st.policy_hash,
                    (json.dumps(st.resource_usage) if isinstance(st.resource_usage, dict) else None),
                ),
            )

    def get_run(self, run_id: str) -> RunStatus | None:
        """
        Retrieve a RunStatus by its run identifier.

        Parses stored fields (including ISO timestamps, enum fields, and JSON-encoded `resource_usage`) and returns a populated RunStatus object.

        Returns:
            `RunStatus` for the given `run_id`, or `None` if no record exists or if stored data cannot be parsed.
        """
        with self._lock, self._conn() as con:
            cur = con.execute("SELECT * FROM sandbox_runs WHERE id=?", (run_id,))
            row = cur.fetchone()
            if not row:
                return None
            row_dict = dict(row)
            try:
                ru = None
                try:
                    ru = json.loads(row_dict.get("resource_usage")) if row_dict.get("resource_usage") else None
                except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                    ru = None
                st = RunStatus(
                    id=row_dict.get("id"),
                    phase=RunPhase(row_dict.get("phase")),
                    spec_version=row_dict.get("spec_version"),
                    runtime=(RuntimeType(row_dict.get("runtime")) if row_dict.get("runtime") else None),
                    runtime_version=row_dict.get("runtime_version"),
                    base_image=row_dict.get("base_image"),
                    image_digest=row_dict.get("image_digest"),
                    policy_hash=row_dict.get("policy_hash"),
                    exit_code=row_dict.get("exit_code"),
                    started_at=(datetime.fromisoformat(row_dict.get("started_at")) if row_dict.get("started_at") else None),
                    finished_at=(datetime.fromisoformat(row_dict.get("finished_at")) if row_dict.get("finished_at") else None),
                    message=row_dict.get("message"),
                    resource_usage=ru,
                    session_id=row_dict.get("session_id"),
                    persona_id=row_dict.get("persona_id"),
                    workspace_id=row_dict.get("workspace_id"),
                    workspace_group_id=row_dict.get("workspace_group_id"),
                    scope_snapshot_id=row_dict.get("scope_snapshot_id"),
                    claim_owner=row_dict.get("claim_owner"),
                    claim_expires_at=_parse_optional_iso_datetime(row_dict.get("claim_expires_at")),
                )
                return st
            except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                return None

    def update_run(self, st: RunStatus) -> None:
        claim_owner = (str(st.claim_owner) if getattr(st, "claim_owner", None) is not None else None)
        claim_expires_at = _coerce_optional_iso_datetime(getattr(st, "claim_expires_at", None))
        if st.phase in (RunPhase.completed, RunPhase.failed, RunPhase.killed, RunPhase.timed_out):
            claim_owner = None
            claim_expires_at = None
        with self._lock, self._conn() as con:
            cur = con.execute(
                (
                    "UPDATE sandbox_runs SET "
                    "spec_version=?, runtime=?, runtime_version=?, base_image=?, phase=?, exit_code=?, "
                    "session_id=?, persona_id=?, workspace_id=?, workspace_group_id=?, scope_snapshot_id=?, "
                    "claim_owner=?, claim_expires_at=?, "
                    "started_at=?, finished_at=?, message=?, image_digest=?, policy_hash=?, resource_usage=? "
                    "WHERE id=?"
                ),
                (
                    st.spec_version,
                    (st.runtime.value if st.runtime else None),
                    (st.runtime_version if getattr(st, "runtime_version", None) else None),
                    st.base_image,
                    st.phase.value,
                    st.exit_code,
                    (str(st.session_id) if getattr(st, "session_id", None) is not None else None),
                    (str(st.persona_id) if getattr(st, "persona_id", None) is not None else None),
                    (str(st.workspace_id) if getattr(st, "workspace_id", None) is not None else None),
                    (str(st.workspace_group_id) if getattr(st, "workspace_group_id", None) is not None else None),
                    (str(st.scope_snapshot_id) if getattr(st, "scope_snapshot_id", None) is not None else None),
                    claim_owner,
                    claim_expires_at,
                    (st.started_at.isoformat() if st.started_at else None),
                    (st.finished_at.isoformat() if st.finished_at else None),
                    st.message,
                    st.image_digest,
                    st.policy_hash,
                    (json.dumps(st.resource_usage) if isinstance(st.resource_usage, dict) else None),
                    st.id,
                ),
            )
            try:
                updated = int(getattr(cur, "rowcount", 0))
            except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                updated = 0
            if updated <= 0:
                logger.debug("SQLiteStore.update_run skipped missing run_id={}", st.id)

    def try_claim_run(self, run_id: str, *, worker_id: str, lease_seconds: int = 30) -> RunStatus | None:
        wid = str(worker_id or "").strip()
        if not wid:
            return None
        ttl = max(1, int(lease_seconds or 0))
        now_iso = datetime.now(timezone.utc).isoformat()
        exp_iso = (datetime.now(timezone.utc) + timedelta(seconds=ttl)).isoformat()
        with self._lock, self._conn() as con:
            cur = con.execute(
                (
                    "UPDATE sandbox_runs SET claim_owner=?, claim_expires_at=? "
                    "WHERE id=? AND phase=? AND "
                    "(claim_owner IS NULL OR claim_expires_at IS NULL OR claim_expires_at <= ? OR claim_owner = ?)"
                ),
                (
                    wid,
                    exp_iso,
                    str(run_id),
                    RunPhase.queued.value,
                    now_iso,
                    wid,
                ),
            )
            try:
                updated = int(getattr(cur, "rowcount", 0))
            except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                updated = 0
        if updated <= 0:
            return None
        return self.get_run(str(run_id))

    def renew_run_claim(self, run_id: str, *, worker_id: str, lease_seconds: int = 30) -> bool:
        wid = str(worker_id or "").strip()
        if not wid:
            return False
        ttl = max(1, int(lease_seconds or 0))
        exp_iso = (datetime.now(timezone.utc) + timedelta(seconds=ttl)).isoformat()
        with self._lock, self._conn() as con:
            cur = con.execute(
                "UPDATE sandbox_runs SET claim_expires_at=? WHERE id=? AND claim_owner=?",
                (exp_iso, str(run_id), wid),
            )
            try:
                updated = int(getattr(cur, "rowcount", 0))
            except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                updated = 0
        return updated > 0

    def release_run_claim(self, run_id: str, *, worker_id: str) -> bool:
        wid = str(worker_id or "").strip()
        if not wid:
            return False
        with self._lock, self._conn() as con:
            cur = con.execute(
                "UPDATE sandbox_runs SET claim_owner=NULL, claim_expires_at=NULL WHERE id=? AND claim_owner=?",
                (str(run_id), wid),
            )
            try:
                updated = int(getattr(cur, "rowcount", 0))
            except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                updated = 0
        return updated > 0

    def try_admit_run_start(
        self,
        run_id: str,
        *,
        worker_id: str,
        max_active_runs: int,
        lease_seconds: int = 30,
        max_active_per_user: int = 0,
        max_active_per_persona: int = 0,
        max_active_per_workspace: int = 0,
        max_active_per_workspace_group: int = 0,
    ) -> RunStatus | None:
        wid = str(worker_id or "").strip()
        if not wid:
            return None
        limit = max(1, int(max_active_runs or 0))
        ttl = max(1, int(lease_seconds or 0))
        per_user_limit = max(0, int(max_active_per_user or 0))
        per_persona_limit = max(0, int(max_active_per_persona or 0))
        per_workspace_limit = max(0, int(max_active_per_workspace or 0))
        per_workspace_group_limit = max(0, int(max_active_per_workspace_group or 0))
        now_iso = datetime.now(timezone.utc).isoformat()
        exp_iso = (datetime.now(timezone.utc) + timedelta(seconds=ttl)).isoformat()
        with self._lock, self._conn() as con:
            try:
                con.execute("BEGIN IMMEDIATE")
                cur_target = con.execute(
                    (
                        "SELECT user_id, persona_id, workspace_id, workspace_group_id "
                        "FROM sandbox_runs "
                        "WHERE id=? AND phase=? AND claim_owner=? "
                        "AND (claim_expires_at IS NULL OR claim_expires_at > ?)"
                    ),
                    (
                        str(run_id),
                        RunPhase.queued.value,
                        wid,
                        now_iso,
                    ),
                )
                target = cur_target.fetchone()
                if not target:
                    con.rollback()
                    return None
                cur_active = con.execute(
                    (
                        "SELECT COUNT(*) AS c FROM sandbox_runs "
                        "WHERE phase=? OR (phase=? AND started_at IS NOT NULL)"
                    ),
                    (RunPhase.running.value, RunPhase.starting.value),
                )
                row = cur_active.fetchone()
                active = int(row["c"]) if row and row["c"] is not None else 0
                if active >= limit:
                    con.rollback()
                    return None
                target_user = target["user_id"] if target and target["user_id"] is not None else None
                target_persona = target["persona_id"] if target and target["persona_id"] is not None else None
                target_workspace = target["workspace_id"] if target and target["workspace_id"] is not None else None
                target_workspace_group = target["workspace_group_id"] if target and target["workspace_group_id"] is not None else None
                if per_user_limit > 0 and target_user:
                    cur_user = con.execute(
                        (
                            "SELECT COUNT(*) AS c FROM sandbox_runs "
                            "WHERE user_id=? AND (phase=? OR (phase=? AND started_at IS NOT NULL))"
                        ),
                        (target_user, RunPhase.running.value, RunPhase.starting.value),
                    )
                    row_user = cur_user.fetchone()
                    active_user = int(row_user["c"]) if row_user and row_user["c"] is not None else 0
                    if active_user >= per_user_limit:
                        con.rollback()
                        return None
                if per_persona_limit > 0 and target_persona:
                    cur_persona = con.execute(
                        (
                            "SELECT COUNT(*) AS c FROM sandbox_runs "
                            "WHERE persona_id=? AND (phase=? OR (phase=? AND started_at IS NOT NULL))"
                        ),
                        (target_persona, RunPhase.running.value, RunPhase.starting.value),
                    )
                    row_persona = cur_persona.fetchone()
                    active_persona = int(row_persona["c"]) if row_persona and row_persona["c"] is not None else 0
                    if active_persona >= per_persona_limit:
                        con.rollback()
                        return None
                if per_workspace_limit > 0 and target_workspace:
                    cur_workspace = con.execute(
                        (
                            "SELECT COUNT(*) AS c FROM sandbox_runs "
                            "WHERE workspace_id=? AND (phase=? OR (phase=? AND started_at IS NOT NULL))"
                        ),
                        (target_workspace, RunPhase.running.value, RunPhase.starting.value),
                    )
                    row_workspace = cur_workspace.fetchone()
                    active_workspace = int(row_workspace["c"]) if row_workspace and row_workspace["c"] is not None else 0
                    if active_workspace >= per_workspace_limit:
                        con.rollback()
                        return None
                if per_workspace_group_limit > 0 and target_workspace_group:
                    cur_workspace_group = con.execute(
                        (
                            "SELECT COUNT(*) AS c FROM sandbox_runs "
                            "WHERE workspace_group_id=? AND (phase=? OR (phase=? AND started_at IS NOT NULL))"
                        ),
                        (target_workspace_group, RunPhase.running.value, RunPhase.starting.value),
                    )
                    row_workspace_group = cur_workspace_group.fetchone()
                    active_workspace_group = int(row_workspace_group["c"]) if row_workspace_group and row_workspace_group["c"] is not None else 0
                    if active_workspace_group >= per_workspace_group_limit:
                        con.rollback()
                        return None
                cur = con.execute(
                    (
                        "UPDATE sandbox_runs SET phase=?, started_at=?, finished_at=NULL, exit_code=NULL, "
                        "claim_expires_at=? "
                        "WHERE id=? AND phase=? AND claim_owner=? "
                        "AND (claim_expires_at IS NULL OR claim_expires_at > ?)"
                    ),
                    (
                        RunPhase.starting.value,
                        now_iso,
                        exp_iso,
                        str(run_id),
                        RunPhase.queued.value,
                        wid,
                        now_iso,
                    ),
                )
                updated = int(getattr(cur, "rowcount", 0) or 0)
                if updated <= 0:
                    con.rollback()
                    return None
                con.commit()
            except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                with contextlib.suppress(_SANDBOX_STORE_NONCRITICAL_EXCEPTIONS):
                    con.rollback()
                return None
        return self.get_run(str(run_id))

    def get_run_owner(self, run_id: str) -> str | None:
        with self._lock, self._conn() as con:
            cur = con.execute("SELECT user_id FROM sandbox_runs WHERE id=?", (run_id,))
            row = cur.fetchone()
            return (row["user_id"] if row else None)

    def put_session(
        self,
        user_id: Any,
        *,
        session_id: str,
        runtime: str | None,
        base_image: str | None,
        cpu_limit: float | None,
        memory_mb: int | None,
        timeout_sec: int | None,
        network_policy: str | None,
        env: dict[str, str] | None,
        labels: dict[str, str] | None,
        trust_level: str | None,
        expires_at_iso: str | None,
        workspace_path: str | None,
        persona_id: str | None = None,
        workspace_id: str | None = None,
        workspace_group_id: str | None = None,
        scope_snapshot_id: str | None = None,
    ) -> None:
        now_ts = time.time()
        with self._lock, self._conn() as con:
            con.execute(
                (
                    "INSERT INTO sandbox_sessions("
                    "id,user_id,runtime,base_image,cpu_limit,memory_mb,timeout_sec,network_policy,env,labels,trust_level,"
                    "persona_id,workspace_id,workspace_group_id,scope_snapshot_id,expires_at,workspace_path,created_at,updated_at"
                    ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(id) DO UPDATE SET "
                    "user_id=excluded.user_id,"
                    "runtime=excluded.runtime,"
                    "base_image=excluded.base_image,"
                    "cpu_limit=excluded.cpu_limit,"
                    "memory_mb=excluded.memory_mb,"
                    "timeout_sec=excluded.timeout_sec,"
                    "network_policy=excluded.network_policy,"
                    "env=excluded.env,"
                    "labels=excluded.labels,"
                    "trust_level=excluded.trust_level,"
                    "persona_id=excluded.persona_id,"
                    "workspace_id=excluded.workspace_id,"
                    "workspace_group_id=excluded.workspace_group_id,"
                    "scope_snapshot_id=excluded.scope_snapshot_id,"
                    "expires_at=excluded.expires_at,"
                    "workspace_path=excluded.workspace_path,"
                    "updated_at=excluded.updated_at"
                ),
                (
                    str(session_id),
                    self._user_key(user_id),
                    (str(runtime) if runtime is not None else None),
                    (str(base_image) if base_image is not None else None),
                    (float(cpu_limit) if cpu_limit is not None else None),
                    (int(memory_mb) if memory_mb is not None else None),
                    (int(timeout_sec) if timeout_sec is not None else None),
                    (str(network_policy) if network_policy is not None else None),
                    (json.dumps(_normalize_string_dict(env)) if env is not None else None),
                    (json.dumps(_normalize_string_dict(labels)) if labels is not None else None),
                    (str(trust_level) if trust_level is not None else None),
                    (str(persona_id) if persona_id is not None else None),
                    (str(workspace_id) if workspace_id is not None else None),
                    (str(workspace_group_id) if workspace_group_id is not None else None),
                    (str(scope_snapshot_id) if scope_snapshot_id is not None else None),
                    (str(expires_at_iso) if expires_at_iso is not None else None),
                    (str(workspace_path) if workspace_path is not None else None),
                    float(now_ts),
                    float(now_ts),
                ),
            )

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock, self._conn() as con:
            cur = con.execute(
                (
                    "SELECT id,user_id,runtime,base_image,cpu_limit,memory_mb,timeout_sec,network_policy,env,labels,trust_level,"
                    "persona_id,workspace_id,workspace_group_id,scope_snapshot_id,"
                    "expires_at,workspace_path,created_at,updated_at "
                    "FROM sandbox_sessions WHERE id=?"
                ),
                (str(session_id),),
            )
            row = cur.fetchone()
            if not row:
                return None
            row_dict = dict(row)
            return {
                "id": row_dict.get("id"),
                "user_id": row_dict.get("user_id"),
                "runtime": row_dict.get("runtime"),
                "base_image": row_dict.get("base_image"),
                "cpu_limit": row_dict.get("cpu_limit"),
                "memory_mb": row_dict.get("memory_mb"),
                "timeout_sec": row_dict.get("timeout_sec"),
                "network_policy": row_dict.get("network_policy"),
                "env": _normalize_string_dict(row_dict.get("env")),
                "labels": _normalize_string_dict(row_dict.get("labels")),
                "trust_level": row_dict.get("trust_level"),
                "persona_id": row_dict.get("persona_id"),
                "workspace_id": row_dict.get("workspace_id"),
                "workspace_group_id": row_dict.get("workspace_group_id"),
                "scope_snapshot_id": row_dict.get("scope_snapshot_id"),
                "expires_at": row_dict.get("expires_at"),
                "workspace_path": row_dict.get("workspace_path"),
                "created_at": row_dict.get("created_at"),
                "updated_at": row_dict.get("updated_at"),
            }

    def get_session_owner(self, session_id: str) -> str | None:
        row = self.get_session(str(session_id))
        if not isinstance(row, dict):
            return None
        owner = row.get("user_id")
        return str(owner) if owner is not None else None

    def list_workspace_paths_for_user_workspace(self, *, user_id: str, workspace_id: str) -> list[str]:
        owner_key = self._user_key(user_id)
        workspace_key = str(workspace_id or "").strip()
        if not owner_key or not workspace_key:
            return []
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock, self._conn() as con:
            cur = con.execute(
                (
                    "SELECT workspace_path FROM sandbox_sessions "
                    "WHERE user_id=? AND workspace_id=? AND workspace_path IS NOT NULL "
                    "AND TRIM(workspace_path) != '' "
                    "AND (expires_at IS NULL OR expires_at = '' OR expires_at > ?)"
                ),
                (owner_key, workspace_key, now_iso),
            )
            return [
                str(row["workspace_path"]).strip()
                for row in cur.fetchall()
                if str(row["workspace_path"] or "").strip()
            ]

    def delete_session(self, session_id: str) -> bool:
        with self._lock, self._conn() as con:
            cur = con.execute("DELETE FROM sandbox_sessions WHERE id=?", (str(session_id),))
            try:
                deleted = int(getattr(cur, "rowcount", 0))
            except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                deleted = 0
            return deleted > 0

    def put_acp_session_control(
        self,
        *,
        session_id: str,
        user_id: Any,
        sandbox_session_id: str | None,
        run_id: str | None,
        ssh_host: str | None = None,
        ssh_port: int | None = None,
        ssh_user: str | None = None,
        ssh_private_key: str | None = None,
        persona_id: str | None = None,
        workspace_id: str | None = None,
        workspace_group_id: str | None = None,
        scope_snapshot_id: str | None = None,
    ) -> None:
        now_ts = time.time()
        with self._lock, self._conn() as con:
            con.execute(
                (
                    "INSERT INTO sandbox_acp_sessions("
                    "id,user_id,sandbox_session_id,run_id,ssh_host,ssh_port,ssh_user,ssh_private_key,"
                    "persona_id,workspace_id,workspace_group_id,scope_snapshot_id,created_at,updated_at"
                    ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(id) DO UPDATE SET "
                    "user_id=excluded.user_id,"
                    "sandbox_session_id=excluded.sandbox_session_id,"
                    "run_id=excluded.run_id,"
                    "ssh_host=excluded.ssh_host,"
                    "ssh_port=excluded.ssh_port,"
                    "ssh_user=excluded.ssh_user,"
                    "ssh_private_key=excluded.ssh_private_key,"
                    "persona_id=excluded.persona_id,"
                    "workspace_id=excluded.workspace_id,"
                    "workspace_group_id=excluded.workspace_group_id,"
                    "scope_snapshot_id=excluded.scope_snapshot_id,"
                    "updated_at=excluded.updated_at"
                ),
                (
                    str(session_id),
                    self._user_key(user_id),
                    (str(sandbox_session_id) if sandbox_session_id is not None else None),
                    (str(run_id) if run_id is not None else None),
                    (str(ssh_host) if ssh_host is not None else None),
                    (int(ssh_port) if ssh_port is not None else None),
                    (str(ssh_user) if ssh_user is not None else None),
                    (str(ssh_private_key) if ssh_private_key is not None else None),
                    (str(persona_id) if persona_id is not None else None),
                    (str(workspace_id) if workspace_id is not None else None),
                    (str(workspace_group_id) if workspace_group_id is not None else None),
                    (str(scope_snapshot_id) if scope_snapshot_id is not None else None),
                    float(now_ts),
                    float(now_ts),
                ),
            )

    def get_acp_session_control(self, session_id: str) -> dict[str, Any] | None:
        with self._lock, self._conn() as con:
            cur = con.execute(
                (
                    "SELECT id,user_id,sandbox_session_id,run_id,ssh_host,ssh_port,ssh_user,ssh_private_key,"
                    "persona_id,workspace_id,workspace_group_id,scope_snapshot_id,created_at,updated_at "
                    "FROM sandbox_acp_sessions WHERE id=?"
                ),
                (str(session_id),),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def delete_acp_session_control(self, session_id: str) -> bool:
        with self._lock, self._conn() as con:
            cur = con.execute("DELETE FROM sandbox_acp_sessions WHERE id=?", (str(session_id),))
            try:
                deleted = int(getattr(cur, "rowcount", 0))
            except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                deleted = 0
            return deleted > 0

    def put_vz_session_control(
        self,
        *,
        session_id: str,
        runtime: str,
        vm_id: str,
        template_id: str | None,
        workspace_mount: str | None,
        agent_ready: bool,
        helper_instance_id: str | None = None,
        helper_started_at: str | None = None,
    ) -> None:
        now_ts = time.time()
        with self._lock, self._conn() as con:
            con.execute(
                (
                    "INSERT INTO sandbox_vz_sessions("
                    "id,runtime,vm_id,template_id,workspace_mount,agent_ready,"
                    "helper_instance_id,helper_started_at,created_at,updated_at"
                    ") VALUES (?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(id) DO UPDATE SET "
                    "runtime=excluded.runtime,"
                    "vm_id=excluded.vm_id,"
                    "template_id=excluded.template_id,"
                    "workspace_mount=excluded.workspace_mount,"
                    "agent_ready=excluded.agent_ready,"
                    "helper_instance_id=excluded.helper_instance_id,"
                    "helper_started_at=excluded.helper_started_at,"
                    "updated_at=excluded.updated_at"
                ),
                (
                    str(session_id),
                    str(runtime),
                    str(vm_id),
                    (str(template_id) if template_id is not None else None),
                    (str(workspace_mount) if workspace_mount is not None else None),
                    1 if agent_ready else 0,
                    coerce_optional_nonempty_string(helper_instance_id),
                    coerce_optional_nonempty_string(helper_started_at),
                    float(now_ts),
                    float(now_ts),
                ),
            )

    def get_vz_session_control(self, session_id: str) -> dict[str, Any] | None:
        with self._lock, self._conn() as con:
            cur = con.execute(
                (
                    "SELECT id,runtime,vm_id,template_id,workspace_mount,agent_ready,"
                    "helper_instance_id,helper_started_at,created_at,updated_at "
                    "FROM sandbox_vz_sessions WHERE id=?"
                ),
                (str(session_id),),
            )
            row = cur.fetchone()
            if not row:
                return None
            row_dict = dict(row)
            row_dict["agent_ready"] = bool(row_dict.get("agent_ready"))
            return row_dict

    def delete_vz_session_control(self, session_id: str) -> bool:
        with self._lock, self._conn() as con:
            cur = con.execute("DELETE FROM sandbox_vz_sessions WHERE id=?", (str(session_id),))
            try:
                deleted = int(getattr(cur, "rowcount", 0))
            except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                deleted = 0
            return deleted > 0

    def list_vz_session_controls(self) -> list[dict[str, Any]]:
        with self._lock, self._conn() as con:
            cur = con.execute(
                "SELECT id,runtime,vm_id,template_id,workspace_mount,agent_ready,"
                "helper_instance_id,helper_started_at,created_at,updated_at "
                "FROM sandbox_vz_sessions ORDER BY id"
            )
            rows = [dict(row) for row in cur.fetchall() or []]
            for row in rows:
                row["agent_ready"] = bool(row.get("agent_ready"))
            return rows

    def get_user_artifact_bytes(self, user_id: str) -> int:
        with self._lock, self._conn() as con:
            cur = con.execute("SELECT artifact_bytes FROM sandbox_usage WHERE user_id=?", (user_id,))
            row = cur.fetchone()
            return int(row["artifact_bytes"]) if row and row["artifact_bytes"] is not None else 0

    def increment_user_artifact_bytes(self, user_id: str, delta: int) -> None:
        with self._lock, self._conn() as con:
            cur = con.execute("SELECT artifact_bytes FROM sandbox_usage WHERE user_id=?", (user_id,))
            row = cur.fetchone()
            cur_val = int(row["artifact_bytes"]) if row and row["artifact_bytes"] is not None else 0
            new_val = max(0, cur_val + int(delta))
            con.execute(
                "REPLACE INTO sandbox_usage(user_id, artifact_bytes) VALUES (?,?)",
                (user_id, new_val),
            )

    def try_reserve_user_artifact_bytes(self, user_id: str, delta: int, cap_bytes: int) -> bool:
        d = int(delta or 0)
        if d <= 0:
            self.increment_user_artifact_bytes(user_id, d)
            return True
        with self._lock, self._conn() as con:
            con.execute("BEGIN IMMEDIATE")
            cur = con.execute("SELECT artifact_bytes FROM sandbox_usage WHERE user_id=?", (user_id,))
            row = cur.fetchone()
            cur_val = int(row["artifact_bytes"]) if row and row["artifact_bytes"] is not None else 0
            if cur_val + d > int(cap_bytes):
                con.rollback()
                return False
            con.execute(
                "REPLACE INTO sandbox_usage(user_id, artifact_bytes) VALUES (?,?)",
                (user_id, cur_val + d),
            )
            return True

    def list_runs(
        self,
        *,
        image_digest: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        persona_id: str | None = None,
        workspace_id: str | None = None,
        workspace_group_id: str | None = None,
        scope_snapshot_id: str | None = None,
        phase: str | None = None,
        started_at_from: str | None = None,
        started_at_to: str | None = None,
        limit: int = 50,
        offset: int = 0,
        sort_desc: bool = True,
    ) -> list[dict]:
        order = "DESC" if sort_desc else "ASC"
        where = ["1=1"]
        params: list[Any] = []
        if image_digest:
            where.append("image_digest = ?")
            params.append(image_digest)
        if user_id:
            where.append("user_id = ?")
            params.append(user_id)
        if session_id:
            where.append("session_id = ?")
            params.append(session_id)
        if persona_id:
            where.append("persona_id = ?")
            params.append(persona_id)
        if workspace_id:
            where.append("workspace_id = ?")
            params.append(workspace_id)
        if workspace_group_id:
            where.append("workspace_group_id = ?")
            params.append(workspace_group_id)
        if scope_snapshot_id:
            where.append("scope_snapshot_id = ?")
            params.append(scope_snapshot_id)
        if phase:
            where.append("phase = ?")
            params.append(phase)
        if started_at_from:
            where.append("started_at >= ?")
            params.append(started_at_from)
        if started_at_to:
            where.append("started_at <= ?")
            params.append(started_at_to)
        sql = (
            "SELECT id,user_id,spec_version,runtime,runtime_version,base_image,session_id,persona_id,workspace_id,workspace_group_id,scope_snapshot_id,phase,exit_code,started_at,finished_at,message,image_digest,policy_hash,resource_usage "  # nosec B608
            f"FROM sandbox_runs WHERE {' AND '.join(where)} ORDER BY started_at {order} LIMIT ? OFFSET ?"
        )
        params.extend([int(limit), int(offset)])
        with self._lock, self._conn() as con:
            cur = con.execute(sql, tuple(params))
            items: list[dict] = []
            for row in cur.fetchall():
                row_dict = dict(row)
                resource_usage = None
                if row_dict.get("resource_usage"):
                    try:
                        loaded = json.loads(row_dict.get("resource_usage"))
                        resource_usage = loaded if isinstance(loaded, dict) else None
                    except (TypeError, ValueError) as exc:
                        logger.debug(
                            "SQLiteStore.list_runs ignored malformed resource_usage for run_id={}: {}",
                            row_dict.get("id"),
                            exc,
                        )
                        resource_usage = None
                items.append({
                    "id": row_dict.get("id"),
                    "user_id": row_dict.get("user_id"),
                    "spec_version": row_dict.get("spec_version"),
                    "runtime": row_dict.get("runtime"),
                    "runtime_version": row_dict.get("runtime_version"),
                    "base_image": row_dict.get("base_image"),
                    "session_id": row_dict.get("session_id"),
                    "persona_id": row_dict.get("persona_id"),
                    "workspace_id": row_dict.get("workspace_id"),
                    "workspace_group_id": row_dict.get("workspace_group_id"),
                    "scope_snapshot_id": row_dict.get("scope_snapshot_id"),
                    "phase": row_dict.get("phase"),
                    "exit_code": row_dict.get("exit_code"),
                    "started_at": row_dict.get("started_at"),
                    "finished_at": row_dict.get("finished_at"),
                    "message": row_dict.get("message"),
                    "image_digest": row_dict.get("image_digest"),
                    "policy_hash": row_dict.get("policy_hash"),
                    "resource_usage": resource_usage,
                })
            return items

    def count_runs(
        self,
        *,
        image_digest: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        persona_id: str | None = None,
        workspace_id: str | None = None,
        workspace_group_id: str | None = None,
        scope_snapshot_id: str | None = None,
        phase: str | None = None,
        started_at_from: str | None = None,
        started_at_to: str | None = None,
    ) -> int:
        where = ["1=1"]
        params: list[Any] = []
        if image_digest:
            where.append("image_digest = ?")
            params.append(image_digest)
        if user_id:
            where.append("user_id = ?")
            params.append(user_id)
        if session_id:
            where.append("session_id = ?")
            params.append(session_id)
        if persona_id:
            where.append("persona_id = ?")
            params.append(persona_id)
        if workspace_id:
            where.append("workspace_id = ?")
            params.append(workspace_id)
        if workspace_group_id:
            where.append("workspace_group_id = ?")
            params.append(workspace_group_id)
        if scope_snapshot_id:
            where.append("scope_snapshot_id = ?")
            params.append(scope_snapshot_id)
        if phase:
            where.append("phase = ?")
            params.append(phase)
        if started_at_from:
            where.append("started_at >= ?")
            params.append(started_at_from)
        if started_at_to:
            where.append("started_at <= ?")
            params.append(started_at_to)
        sql = f"SELECT COUNT(*) AS c FROM sandbox_runs WHERE {' AND '.join(where)}"  # nosec B608
        with self._lock, self._conn() as con:
            cur = con.execute(sql, tuple(params))
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def list_idempotency(
        self,
        *,
        endpoint: str | None = None,
        user_id: str | None = None,
        key: str | None = None,
        created_at_from: str | None = None,
        created_at_to: str | None = None,
        limit: int = 50,
        offset: int = 0,
        sort_desc: bool = True,
    ) -> list[dict]:
        order = "DESC" if sort_desc else "ASC"
        where = ["1=1"]
        params: list[Any] = []
        if endpoint:
            where.append("endpoint = ?")
            params.append(endpoint)
        if user_id:
            where.append("user_key = ?")
            params.append(user_id)
        if key:
            where.append("key = ?")
            params.append(key)
        if created_at_from is not None:
            where.append("created_at >= ?")
            params.append(self._coerce_created_at(created_at_from))
        if created_at_to is not None:
            where.append("created_at <= ?")
            params.append(self._coerce_created_at(created_at_to))
        sql = (
            "SELECT endpoint,user_key,key,fingerprint,object_id,created_at FROM sandbox_idempotency "  # nosec B608
            f"WHERE {' AND '.join(where)} ORDER BY created_at {order} LIMIT ? OFFSET ?"
        )
        params.extend([int(limit), int(offset)])
        items: list[dict] = []
        with self._lock, self._conn() as con:
            cur = con.execute(sql, tuple(params))
            for row in cur.fetchall():
                try:
                    from datetime import datetime, timezone
                    iso_ct = datetime.fromtimestamp(float(row["created_at"]), tz=timezone.utc).isoformat()
                except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                    iso_ct = None
                items.append({
                    "endpoint": row["endpoint"],
                    "user_id": row["user_key"],
                    "key": row["key"],
                    "fingerprint": row["fingerprint"],
                    "object_id": row["object_id"],
                    "created_at": iso_ct,
                })
        return items

    def count_idempotency(
        self,
        *,
        endpoint: str | None = None,
        user_id: str | None = None,
        key: str | None = None,
        created_at_from: str | None = None,
        created_at_to: str | None = None,
    ) -> int:
        where = ["1=1"]
        params: list[Any] = []
        if endpoint:
            where.append("endpoint = ?")
            params.append(endpoint)
        if user_id:
            where.append("user_key = ?")
            params.append(user_id)
        if key:
            where.append("key = ?")
            params.append(key)
        if created_at_from is not None:
            where.append("created_at >= ?")
            params.append(self._coerce_created_at(created_at_from))
        if created_at_to is not None:
            where.append("created_at <= ?")
            params.append(self._coerce_created_at(created_at_to))
        sql = f"SELECT COUNT(*) FROM sandbox_idempotency WHERE {' AND '.join(where)}"  # nosec B608
        with self._lock, self._conn() as con:
            cur = con.execute(sql, tuple(params))
            row = cur.fetchone()
            return int(row[0]) if row else 0

    def list_usage(
        self,
        *,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
        sort_desc: bool = True,
    ) -> list[dict]:
        # Aggregate in Python to avoid JSON1 dependence
        with self._lock, self._conn() as con:
            # Gather artifact bytes per user
            art: dict[str, int] = {}
            cur = con.execute("SELECT user_id, artifact_bytes FROM sandbox_usage")
            for row in cur.fetchall():
                u = row["user_id"]
                if user_id and u != user_id:
                    continue
                art[u] = int(row["artifact_bytes"]) if row["artifact_bytes"] is not None else 0
            # Gather runs and log_bytes from runs table
            cur2 = con.execute("SELECT user_id, resource_usage FROM sandbox_runs")
            agg: dict[str, dict] = {}
            for row in cur2.fetchall():
                u = row["user_id"]
                if not u:
                    continue
                if user_id and u != user_id:
                    continue
                rs = agg.setdefault(u, {"runs_count": 0, "log_bytes": 0})
                rs["runs_count"] += 1
                try:
                    import json as _json
                    ru = _json.loads(row["resource_usage"]) if row["resource_usage"] else None
                    if ru and isinstance(ru.get("log_bytes"), int):
                        rs["log_bytes"] += int(ru.get("log_bytes") or 0)
                except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                    pass
            # Build items
            users = set(art.keys()) | set(agg.keys())
            items: list[dict] = []
            for u in users:
                items.append({
                    "user_id": u,
                    "runs_count": int((agg.get(u) or {}).get("runs_count", 0)),
                    "log_bytes": int((agg.get(u) or {}).get("log_bytes", 0)),
                    "artifact_bytes": int(art.get(u, 0)),
                })
        items.sort(key=lambda r: r.get("user_id") or "", reverse=bool(sort_desc))
        return items[offset: offset + limit]

    # -- Durable run queue (SQLite) — delegated to DB_Management -----------------
    def enqueue_run(self, run_id: str, user_id: str, priority: int = 0) -> None:
        try:
            self._queue_db.enqueue(run_id, user_id, priority)
        except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"enqueue_run failed (sqlite): {e}")

    def dequeue_run(self, worker_id: str) -> dict | None:
        try:
            return self._queue_db.dequeue(worker_id)
        except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"dequeue_run failed (sqlite): {e}")
            return None

    def remove_from_queue(self, run_id: str) -> bool:
        try:
            return self._queue_db.remove(run_id)
        except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"remove_from_queue failed (sqlite): {e}")
            return False

    def count_usage(
        self,
        *,
        user_id: str | None = None,
    ) -> int:
        return len(self.list_usage(user_id=user_id, limit=10**9, offset=0, sort_desc=True))


class PostgresStore(SandboxStore):
    """Cluster/durable store backed by Postgres.

    - Requires psycopg (v3).
    - Uses per-operation connections (no extra pooling dependency).
    - Stores datetimes as ISO-8601 TEXT for parity with SQLite paths.
    """

    def __init__(self, dsn: str, idem_ttl_sec: int = 600) -> None:
        self.idem_ttl_sec = int(idem_ttl_sec)
        self.dsn = dsn
        self._lock = threading.RLock()
        try:
            import psycopg  # noqa: F401
            from psycopg.rows import dict_row  # noqa: F401
        except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS as e:  # pragma: no cover
            raise RuntimeError("psycopg is required for PostgresStore") from e
        self._init_db()

        # Delegate run queue SQL to DB_Management abstraction
        from tldw_Server_API.app.core.DB_Management.Sandbox_Queue_DB import PostgresRunQueueDB
        self._queue_db = PostgresRunQueueDB(conn_factory=self._conn, lock=self._lock)

    def _conn(self):
        import psycopg
        from psycopg.rows import dict_row
        return psycopg.connect(self.dsn, autocommit=True, row_factory=dict_row)

    def _init_db(self) -> None:
        with self._conn() as con, con.cursor() as cur:
            cur.execute(
                """
                    CREATE TABLE IF NOT EXISTS sandbox_runs (
                        id TEXT PRIMARY KEY,
                        user_id TEXT,
                        spec_version TEXT,
                        runtime TEXT,
                        runtime_version TEXT,
                        base_image TEXT,
                        session_id TEXT,
                        persona_id TEXT,
                        workspace_id TEXT,
                        workspace_group_id TEXT,
                        scope_snapshot_id TEXT,
                        claim_owner TEXT,
                        claim_expires_at TEXT,
                        phase TEXT,
                        exit_code INTEGER,
                        started_at TEXT,
                        finished_at TEXT,
                        message TEXT,
                        image_digest TEXT,
                        policy_hash TEXT,
                        resource_usage JSONB
                    );
                    """
            )
            cur.execute(
                """
                    CREATE TABLE IF NOT EXISTS sandbox_idempotency (
                        endpoint TEXT,
                        user_key TEXT,
                        key TEXT,
                        fingerprint TEXT,
                        object_id TEXT,
                        response_body JSONB,
                        created_at DOUBLE PRECISION,
                        PRIMARY KEY (endpoint, user_key, key)
                    );
                    """
            )
            cur.execute(
                """
                    CREATE TABLE IF NOT EXISTS sandbox_usage (
                        user_id TEXT PRIMARY KEY,
                        artifact_bytes BIGINT
                    );
                    """
            )
            cur.execute(
                """
                    CREATE TABLE IF NOT EXISTS sandbox_sessions (
                        id TEXT PRIMARY KEY,
                        user_id TEXT,
                        runtime TEXT,
                        base_image TEXT,
                        cpu_limit DOUBLE PRECISION,
                        memory_mb INTEGER,
                        timeout_sec INTEGER,
                        network_policy TEXT,
                        env TEXT,
                        labels TEXT,
                        trust_level TEXT,
                        persona_id TEXT,
                        workspace_id TEXT,
                        workspace_group_id TEXT,
                        scope_snapshot_id TEXT,
                        expires_at TEXT,
                        workspace_path TEXT,
                        created_at DOUBLE PRECISION,
                        updated_at DOUBLE PRECISION
                    );
                    """
            )
            cur.execute(
                """
                    CREATE TABLE IF NOT EXISTS sandbox_vz_sessions (
                        id TEXT PRIMARY KEY,
                        runtime TEXT,
                        vm_id TEXT,
                        template_id TEXT,
                        workspace_mount TEXT,
                        agent_ready BOOLEAN,
                        helper_instance_id TEXT,
                        helper_started_at TEXT,
                        created_at DOUBLE PRECISION,
                        updated_at DOUBLE PRECISION
                    );
                    """
            )
            cur.execute(
                """
                    CREATE TABLE IF NOT EXISTS sandbox_acp_sessions (
                        id TEXT PRIMARY KEY,
                        user_id TEXT,
                        sandbox_session_id TEXT,
                        run_id TEXT,
                        ssh_host TEXT,
                        ssh_port INTEGER,
                        ssh_user TEXT,
                        ssh_private_key TEXT,
                        persona_id TEXT,
                        workspace_id TEXT,
                        workspace_group_id TEXT,
                        scope_snapshot_id TEXT,
                        created_at DOUBLE PRECISION,
                        updated_at DOUBLE PRECISION
                    );
                    """
            )
            cur.execute(
                """
                    CREATE TABLE IF NOT EXISTS run_queue (
                        run_id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        priority INTEGER DEFAULT 0,
                        enqueued_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
                    );
                    """
            )
            # Migrations: ensure new columns exist
            def _ensure_column(table: str, col: str, coltype: str) -> None:
                try:
                    cur.execute(
                        """
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name=%s AND column_name=%s
                            """,
                        (table, col),
                    )
                    if cur.fetchone():
                        return
                    cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")
                except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS as exc:
                    logger.exception(f"Postgres migration failed while adding {table}.{col}")
                    raise ClusterStoreUnavailable(
                        f"Postgres migration failed while adding required column {table}.{col}"
                    ) from exc

            _ensure_column("sandbox_runs", "resource_usage", "JSONB")
            _ensure_column("sandbox_runs", "runtime_version", "TEXT")
            _ensure_column("sandbox_runs", "session_id", "TEXT")
            _ensure_column("sandbox_runs", "persona_id", "TEXT")
            _ensure_column("sandbox_runs", "workspace_id", "TEXT")
            _ensure_column("sandbox_runs", "workspace_group_id", "TEXT")
            _ensure_column("sandbox_runs", "scope_snapshot_id", "TEXT")
            _ensure_column("sandbox_runs", "claim_owner", "TEXT")
            _ensure_column("sandbox_runs", "claim_expires_at", "TEXT")
            _ensure_column("sandbox_sessions", "cpu_limit", "DOUBLE PRECISION")
            _ensure_column("sandbox_sessions", "memory_mb", "INTEGER")
            _ensure_column("sandbox_sessions", "timeout_sec", "INTEGER")
            _ensure_column("sandbox_sessions", "network_policy", "TEXT")
            _ensure_column("sandbox_sessions", "env", "TEXT")
            _ensure_column("sandbox_sessions", "labels", "TEXT")
            _ensure_column("sandbox_sessions", "trust_level", "TEXT")
            _ensure_column("sandbox_sessions", "persona_id", "TEXT")
            _ensure_column("sandbox_sessions", "workspace_id", "TEXT")
            _ensure_column("sandbox_sessions", "workspace_group_id", "TEXT")
            _ensure_column("sandbox_sessions", "scope_snapshot_id", "TEXT")
            _ensure_column("sandbox_acp_sessions", "sandbox_session_id", "TEXT")
            _ensure_column("sandbox_acp_sessions", "run_id", "TEXT")
            _ensure_column("sandbox_acp_sessions", "ssh_host", "TEXT")
            _ensure_column("sandbox_acp_sessions", "ssh_port", "INTEGER")
            _ensure_column("sandbox_acp_sessions", "ssh_user", "TEXT")
            _ensure_column("sandbox_acp_sessions", "ssh_private_key", "TEXT")
            _ensure_column("sandbox_acp_sessions", "persona_id", "TEXT")
            _ensure_column("sandbox_acp_sessions", "workspace_id", "TEXT")
            _ensure_column("sandbox_acp_sessions", "workspace_group_id", "TEXT")
            _ensure_column("sandbox_acp_sessions", "scope_snapshot_id", "TEXT")
            _ensure_column("sandbox_vz_sessions", "helper_instance_id", "TEXT")
            _ensure_column("sandbox_vz_sessions", "helper_started_at", "TEXT")

    def _fp(self, body: dict[str, Any]) -> str:
        try:
            canon = json.dumps(body, sort_keys=True, separators=(",", ":"))
        except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
            canon = str(body)
        import hashlib
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()

    def _user_key(self, user_id: Any) -> str:
        try:
            return str(user_id)
        except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
            return ""

    def check_idempotency(self, endpoint: str, user_id: Any, key: str | None, body: dict[str, Any]) -> dict[str, Any] | None:
        if not key:
            return None
        with self._lock, self._conn() as con, con.cursor() as cur:
            # TTL GC
            try:
                ttl = max(1, int(self.idem_ttl_sec))
            except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                ttl = 600
            cutoff = time.time() - ttl
            with contextlib.suppress(_SANDBOX_STORE_NONCRITICAL_EXCEPTIONS):
                cur.execute("DELETE FROM sandbox_idempotency WHERE created_at < %s", (cutoff,))
            cur.execute(
                """
                    SELECT fingerprint, response_body, object_id, created_at
                    FROM sandbox_idempotency
                    WHERE endpoint=%s AND user_key=%s AND key=%s
                    """,
                (endpoint, self._user_key(user_id), key),
            )
            row = cur.fetchone()
            if not row:
                return None
            fp_new = self._fp(body)
            if row.get("fingerprint") == fp_new:
                try:
                    return row.get("response_body")
                except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                    return None
            # Conflict: include created_at epoch seconds
            ct = None
            try:
                ct = float(row.get("created_at")) if row.get("created_at") is not None else None
            except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                ct = None
            raise IdempotencyConflict(row.get("object_id") or "", key=key, created_at=ct)

    def store_idempotency(self, endpoint: str, user_id: Any, key: str | None, body: dict[str, Any], object_id: str, response: dict[str, Any]) -> None:
        if not key:
            return
        with self._lock, self._conn() as con:
            with con.cursor() as cur:
                try:
                    cur.execute(
                        """
                        INSERT INTO sandbox_idempotency(endpoint,user_key,key,fingerprint,object_id,response_body,created_at)
                        VALUES (%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (endpoint, user_key, key) DO NOTHING
                        """,
                        (
                            endpoint,
                            self._user_key(user_id),
                            key,
                            self._fp(body),
                            object_id,
                            json.dumps(response, ensure_ascii=False),
                            time.time(),
                        ),
                    )
                except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS as e:
                    logger.debug(f"idempotency store failed (pg): {e}")

    def put_run(self, user_id: Any, st: RunStatus) -> None:
        with self._lock, self._conn() as con:
            with con.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sandbox_runs (
                        id,user_id,spec_version,runtime,runtime_version,base_image,
                        session_id,persona_id,workspace_id,workspace_group_id,scope_snapshot_id,
                        claim_owner,claim_expires_at,
                        phase,exit_code,started_at,finished_at,message,image_digest,policy_hash,resource_usage
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (id) DO UPDATE SET
                        user_id=EXCLUDED.user_id,
                        spec_version=EXCLUDED.spec_version,
                        runtime=EXCLUDED.runtime,
                        runtime_version=EXCLUDED.runtime_version,
                        base_image=EXCLUDED.base_image,
                        session_id=EXCLUDED.session_id,
                        persona_id=EXCLUDED.persona_id,
                        workspace_id=EXCLUDED.workspace_id,
                        workspace_group_id=EXCLUDED.workspace_group_id,
                        scope_snapshot_id=EXCLUDED.scope_snapshot_id,
                        claim_owner=EXCLUDED.claim_owner,
                        claim_expires_at=EXCLUDED.claim_expires_at,
                        phase=EXCLUDED.phase,
                        exit_code=EXCLUDED.exit_code,
                        started_at=EXCLUDED.started_at,
                        finished_at=EXCLUDED.finished_at,
                        message=EXCLUDED.message,
                        image_digest=EXCLUDED.image_digest,
                        policy_hash=EXCLUDED.policy_hash,
                        resource_usage=EXCLUDED.resource_usage
                    """,
                    (
                        st.id,
                        self._user_key(user_id),
                        st.spec_version,
                        (st.runtime.value if st.runtime else None),
                        (st.runtime_version if getattr(st, "runtime_version", None) else None),
                        st.base_image,
                        (str(st.session_id) if getattr(st, "session_id", None) is not None else None),
                        (str(st.persona_id) if getattr(st, "persona_id", None) is not None else None),
                        (str(st.workspace_id) if getattr(st, "workspace_id", None) is not None else None),
                        (str(st.workspace_group_id) if getattr(st, "workspace_group_id", None) is not None else None),
                        (str(st.scope_snapshot_id) if getattr(st, "scope_snapshot_id", None) is not None else None),
                        (str(st.claim_owner) if getattr(st, "claim_owner", None) is not None else None),
                        _coerce_optional_iso_datetime(getattr(st, "claim_expires_at", None)),
                        st.phase.value,
                        st.exit_code,
                        (st.started_at.isoformat() if st.started_at else None),
                        (st.finished_at.isoformat() if st.finished_at else None),
                        st.message,
                        st.image_digest,
                        st.policy_hash,
                        (json.dumps(st.resource_usage) if isinstance(st.resource_usage, dict) else None),
                    ),
                )

    def get_run(self, run_id: str) -> RunStatus | None:
        with self._lock, self._conn() as con, con.cursor() as cur:
            cur.execute("SELECT * FROM sandbox_runs WHERE id=%s", (run_id,))
            row = cur.fetchone()
            if not row:
                return None
            try:
                ru = None
                try:
                    ru = row.get("resource_usage") if row.get("resource_usage") else None
                    if isinstance(ru, str):
                        ru = json.loads(ru)
                except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                    ru = None
                st = RunStatus(
                    id=row.get("id"),
                    phase=RunPhase(row.get("phase")),
                    spec_version=row.get("spec_version"),
                    runtime=(RuntimeType(row.get("runtime")) if row.get("runtime") else None),
                    runtime_version=row.get("runtime_version"),
                    base_image=row.get("base_image"),
                    image_digest=row.get("image_digest"),
                    policy_hash=row.get("policy_hash"),
                    exit_code=row.get("exit_code"),
                    started_at=(datetime.fromisoformat(row.get("started_at")) if row.get("started_at") else None),
                    finished_at=(datetime.fromisoformat(row.get("finished_at")) if row.get("finished_at") else None),
                    session_id=row.get("session_id"),
                    persona_id=row.get("persona_id"),
                    workspace_id=row.get("workspace_id"),
                    workspace_group_id=row.get("workspace_group_id"),
                    scope_snapshot_id=row.get("scope_snapshot_id"),
                    claim_owner=row.get("claim_owner"),
                    claim_expires_at=_parse_optional_iso_datetime(row.get("claim_expires_at")),
                )
                st.message = row.get("message")
                st.resource_usage = ru if isinstance(ru, dict) else None
                return st
            except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"pg get_run parse error: {e}")
                return None

    def update_run(self, st: RunStatus) -> None:
        claim_owner = (str(st.claim_owner) if getattr(st, "claim_owner", None) is not None else None)
        claim_expires_at = _coerce_optional_iso_datetime(getattr(st, "claim_expires_at", None))
        if st.phase in (RunPhase.completed, RunPhase.failed, RunPhase.killed, RunPhase.timed_out):
            claim_owner = None
            claim_expires_at = None
        with self._lock, self._conn() as con:
            with con.cursor() as cur:
                cur.execute(
                    """
                    UPDATE sandbox_runs SET
                        spec_version=%s,
                        runtime=%s,
                        runtime_version=%s,
                        base_image=%s,
                        session_id=%s,
                        persona_id=%s,
                        workspace_id=%s,
                        workspace_group_id=%s,
                        scope_snapshot_id=%s,
                        claim_owner=%s,
                        claim_expires_at=%s,
                        phase=%s,
                        exit_code=%s,
                        started_at=%s,
                        finished_at=%s,
                        message=%s,
                        image_digest=%s,
                        policy_hash=%s,
                        resource_usage=%s
                    WHERE id=%s
                    """,
                    (
                        st.spec_version,
                        (st.runtime.value if st.runtime else None),
                        (st.runtime_version if getattr(st, "runtime_version", None) else None),
                        st.base_image,
                        (str(st.session_id) if getattr(st, "session_id", None) is not None else None),
                        (str(st.persona_id) if getattr(st, "persona_id", None) is not None else None),
                        (str(st.workspace_id) if getattr(st, "workspace_id", None) is not None else None),
                        (str(st.workspace_group_id) if getattr(st, "workspace_group_id", None) is not None else None),
                        (str(st.scope_snapshot_id) if getattr(st, "scope_snapshot_id", None) is not None else None),
                        claim_owner,
                        claim_expires_at,
                        st.phase.value,
                        st.exit_code,
                        (st.started_at.isoformat() if st.started_at else None),
                        (st.finished_at.isoformat() if st.finished_at else None),
                        st.message,
                        st.image_digest,
                        st.policy_hash,
                        (json.dumps(st.resource_usage) if isinstance(st.resource_usage, dict) else None),
                        st.id,
                    ),
                )
                try:
                    updated = int(getattr(cur, "rowcount", 0))
                except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                    updated = 0
                if updated <= 0:
                    logger.debug("PostgresStore.update_run skipped missing run_id={}", st.id)

    def try_claim_run(self, run_id: str, *, worker_id: str, lease_seconds: int = 30) -> RunStatus | None:
        wid = str(worker_id or "").strip()
        if not wid:
            return None
        ttl = max(1, int(lease_seconds or 0))
        now_iso = datetime.now(timezone.utc).isoformat()
        exp_iso = (datetime.now(timezone.utc) + timedelta(seconds=ttl)).isoformat()
        with self._lock, self._conn() as con, con.cursor() as cur:
            cur.execute(
                """
                UPDATE sandbox_runs
                SET claim_owner=%s, claim_expires_at=%s
                WHERE id=%s
                  AND phase=%s
                  AND (
                    claim_owner IS NULL
                    OR claim_expires_at IS NULL
                    OR claim_expires_at::timestamptz <= %s::timestamptz
                    OR claim_owner = %s
                  )
                """,
                (
                    wid,
                    exp_iso,
                    str(run_id),
                    RunPhase.queued.value,
                    now_iso,
                    wid,
                ),
            )
            try:
                updated = int(getattr(cur, "rowcount", 0))
            except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                updated = 0
        if updated <= 0:
            return None
        return self.get_run(str(run_id))

    def renew_run_claim(self, run_id: str, *, worker_id: str, lease_seconds: int = 30) -> bool:
        wid = str(worker_id or "").strip()
        if not wid:
            return False
        ttl = max(1, int(lease_seconds or 0))
        exp_iso = (datetime.now(timezone.utc) + timedelta(seconds=ttl)).isoformat()
        with self._lock, self._conn() as con, con.cursor() as cur:
            cur.execute(
                "UPDATE sandbox_runs SET claim_expires_at=%s WHERE id=%s AND claim_owner=%s",
                (exp_iso, str(run_id), wid),
            )
            try:
                updated = int(getattr(cur, "rowcount", 0))
            except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                updated = 0
        return updated > 0

    def release_run_claim(self, run_id: str, *, worker_id: str) -> bool:
        wid = str(worker_id or "").strip()
        if not wid:
            return False
        with self._lock, self._conn() as con, con.cursor() as cur:
            cur.execute(
                "UPDATE sandbox_runs SET claim_owner=NULL, claim_expires_at=NULL WHERE id=%s AND claim_owner=%s",
                (str(run_id), wid),
            )
            try:
                updated = int(getattr(cur, "rowcount", 0))
            except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                updated = 0
        return updated > 0

    def try_admit_run_start(
        self,
        run_id: str,
        *,
        worker_id: str,
        max_active_runs: int,
        lease_seconds: int = 30,
        max_active_per_user: int = 0,
        max_active_per_persona: int = 0,
        max_active_per_workspace: int = 0,
        max_active_per_workspace_group: int = 0,
    ) -> RunStatus | None:
        wid = str(worker_id or "").strip()
        if not wid:
            return None
        limit = max(1, int(max_active_runs or 0))
        ttl = max(1, int(lease_seconds or 0))
        per_user_limit = max(0, int(max_active_per_user or 0))
        per_persona_limit = max(0, int(max_active_per_persona or 0))
        per_workspace_limit = max(0, int(max_active_per_workspace or 0))
        per_workspace_group_limit = max(0, int(max_active_per_workspace_group or 0))
        now_iso = datetime.now(timezone.utc).isoformat()
        exp_iso = (datetime.now(timezone.utc) + timedelta(seconds=ttl)).isoformat()
        with self._lock, self._conn() as con, con.cursor() as cur:
            try:
                cur.execute("BEGIN")
                # Serialize active-slot admission across nodes in cluster mode.
                cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("sandbox_active_run_slot_admission",))
                cur.execute(
                    (
                        "SELECT user_id, persona_id, workspace_id, workspace_group_id "
                        "FROM sandbox_runs "
                        "WHERE id=%s AND phase=%s AND claim_owner=%s "
                        "AND (claim_expires_at IS NULL OR claim_expires_at::timestamptz > %s::timestamptz)"
                    ),
                    (str(run_id), RunPhase.queued.value, wid, now_iso),
                )
                target = cur.fetchone() or {}
                if not target:
                    cur.execute("ROLLBACK")
                    return None
                cur.execute(
                    (
                        "SELECT COUNT(*) AS c FROM sandbox_runs "
                        "WHERE phase=%s OR (phase=%s AND started_at IS NOT NULL)"
                    ),
                    (RunPhase.running.value, RunPhase.starting.value),
                )
                row = cur.fetchone() or {}
                active = int(row.get("c") or 0)
                if active >= limit:
                    cur.execute("ROLLBACK")
                    return None
                target_user = target.get("user_id")
                target_persona = target.get("persona_id")
                target_workspace = target.get("workspace_id")
                target_workspace_group = target.get("workspace_group_id")
                if per_user_limit > 0 and target_user:
                    cur.execute(
                        (
                            "SELECT COUNT(*) AS c FROM sandbox_runs "
                            "WHERE user_id=%s AND (phase=%s OR (phase=%s AND started_at IS NOT NULL))"
                        ),
                        (target_user, RunPhase.running.value, RunPhase.starting.value),
                    )
                    row_user = cur.fetchone() or {}
                    active_user = int(row_user.get("c") or 0)
                    if active_user >= per_user_limit:
                        cur.execute("ROLLBACK")
                        return None
                if per_persona_limit > 0 and target_persona:
                    cur.execute(
                        (
                            "SELECT COUNT(*) AS c FROM sandbox_runs "
                            "WHERE persona_id=%s AND (phase=%s OR (phase=%s AND started_at IS NOT NULL))"
                        ),
                        (target_persona, RunPhase.running.value, RunPhase.starting.value),
                    )
                    row_persona = cur.fetchone() or {}
                    active_persona = int(row_persona.get("c") or 0)
                    if active_persona >= per_persona_limit:
                        cur.execute("ROLLBACK")
                        return None
                if per_workspace_limit > 0 and target_workspace:
                    cur.execute(
                        (
                            "SELECT COUNT(*) AS c FROM sandbox_runs "
                            "WHERE workspace_id=%s AND (phase=%s OR (phase=%s AND started_at IS NOT NULL))"
                        ),
                        (target_workspace, RunPhase.running.value, RunPhase.starting.value),
                    )
                    row_workspace = cur.fetchone() or {}
                    active_workspace = int(row_workspace.get("c") or 0)
                    if active_workspace >= per_workspace_limit:
                        cur.execute("ROLLBACK")
                        return None
                if per_workspace_group_limit > 0 and target_workspace_group:
                    cur.execute(
                        (
                            "SELECT COUNT(*) AS c FROM sandbox_runs "
                            "WHERE workspace_group_id=%s AND (phase=%s OR (phase=%s AND started_at IS NOT NULL))"
                        ),
                        (target_workspace_group, RunPhase.running.value, RunPhase.starting.value),
                    )
                    row_workspace_group = cur.fetchone() or {}
                    active_workspace_group = int(row_workspace_group.get("c") or 0)
                    if active_workspace_group >= per_workspace_group_limit:
                        cur.execute("ROLLBACK")
                        return None
                cur.execute(
                    """
                    UPDATE sandbox_runs
                    SET phase=%s,
                        started_at=%s,
                        finished_at=NULL,
                        exit_code=NULL,
                        claim_expires_at=%s
                    WHERE id=%s
                      AND phase=%s
                      AND claim_owner=%s
                      AND (claim_expires_at IS NULL OR claim_expires_at::timestamptz > %s::timestamptz)
                    """,
                    (
                        RunPhase.starting.value,
                        now_iso,
                        exp_iso,
                        str(run_id),
                        RunPhase.queued.value,
                        wid,
                        now_iso,
                    ),
                )
                updated = int(getattr(cur, "rowcount", 0) or 0)
                if updated <= 0:
                    cur.execute("ROLLBACK")
                    return None
                cur.execute("COMMIT")
            except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                with contextlib.suppress(_SANDBOX_STORE_NONCRITICAL_EXCEPTIONS):
                    cur.execute("ROLLBACK")
                return None
        return self.get_run(str(run_id))

    def get_run_owner(self, run_id: str) -> str | None:
        with self._lock, self._conn() as con, con.cursor() as cur:
            cur.execute("SELECT user_id FROM sandbox_runs WHERE id=%s", (run_id,))
            row = cur.fetchone()
            if row and (row.get("user_id") is not None):
                return str(row.get("user_id"))
            return None

    def put_session(
        self,
        user_id: Any,
        *,
        session_id: str,
        runtime: str | None,
        base_image: str | None,
        cpu_limit: float | None,
        memory_mb: int | None,
        timeout_sec: int | None,
        network_policy: str | None,
        env: dict[str, str] | None,
        labels: dict[str, str] | None,
        trust_level: str | None,
        expires_at_iso: str | None,
        workspace_path: str | None,
        persona_id: str | None = None,
        workspace_id: str | None = None,
        workspace_group_id: str | None = None,
        scope_snapshot_id: str | None = None,
    ) -> None:
        now_ts = time.time()
        with self._lock, self._conn() as con:
            with con.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sandbox_sessions(
                        id,user_id,runtime,base_image,cpu_limit,memory_mb,timeout_sec,network_policy,env,labels,trust_level,
                        persona_id,workspace_id,workspace_group_id,scope_snapshot_id,expires_at,workspace_path,created_at,updated_at
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (id) DO UPDATE SET
                        user_id=EXCLUDED.user_id,
                        runtime=EXCLUDED.runtime,
                        base_image=EXCLUDED.base_image,
                        cpu_limit=EXCLUDED.cpu_limit,
                        memory_mb=EXCLUDED.memory_mb,
                        timeout_sec=EXCLUDED.timeout_sec,
                        network_policy=EXCLUDED.network_policy,
                        env=EXCLUDED.env,
                        labels=EXCLUDED.labels,
                        trust_level=EXCLUDED.trust_level,
                        persona_id=EXCLUDED.persona_id,
                        workspace_id=EXCLUDED.workspace_id,
                        workspace_group_id=EXCLUDED.workspace_group_id,
                        scope_snapshot_id=EXCLUDED.scope_snapshot_id,
                        expires_at=EXCLUDED.expires_at,
                        workspace_path=EXCLUDED.workspace_path,
                        updated_at=EXCLUDED.updated_at
                    """,
                    (
                        str(session_id),
                        self._user_key(user_id),
                        (str(runtime) if runtime is not None else None),
                        (str(base_image) if base_image is not None else None),
                        (float(cpu_limit) if cpu_limit is not None else None),
                        (int(memory_mb) if memory_mb is not None else None),
                        (int(timeout_sec) if timeout_sec is not None else None),
                        (str(network_policy) if network_policy is not None else None),
                        (json.dumps(_normalize_string_dict(env)) if env is not None else None),
                        (json.dumps(_normalize_string_dict(labels)) if labels is not None else None),
                        (str(trust_level) if trust_level is not None else None),
                        (str(persona_id) if persona_id is not None else None),
                        (str(workspace_id) if workspace_id is not None else None),
                        (str(workspace_group_id) if workspace_group_id is not None else None),
                        (str(scope_snapshot_id) if scope_snapshot_id is not None else None),
                        (str(expires_at_iso) if expires_at_iso is not None else None),
                        (str(workspace_path) if workspace_path is not None else None),
                        float(now_ts),
                        float(now_ts),
                    ),
                )

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock, self._conn() as con, con.cursor() as cur:
            cur.execute(
                (
                    "SELECT id,user_id,runtime,base_image,cpu_limit,memory_mb,timeout_sec,network_policy,env,labels,trust_level,"
                    "persona_id,workspace_id,workspace_group_id,scope_snapshot_id,"
                    "expires_at,workspace_path,created_at,updated_at "
                    "FROM sandbox_sessions WHERE id=%s"
                ),
                (str(session_id),),
            )
            row = cur.fetchone()
            if not isinstance(row, dict):
                return None
            result = dict(row)
            result["env"] = _normalize_string_dict(result.get("env"))
            result["labels"] = _normalize_string_dict(result.get("labels"))
            return result

    def get_session_owner(self, session_id: str) -> str | None:
        row = self.get_session(str(session_id))
        if not isinstance(row, dict):
            return None
        owner = row.get("user_id")
        return str(owner) if owner is not None else None

    def list_workspace_paths_for_user_workspace(self, *, user_id: str, workspace_id: str) -> list[str]:
        owner_key = self._user_key(user_id)
        workspace_key = str(workspace_id or "").strip()
        if not owner_key or not workspace_key:
            return []
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock, self._conn() as con, con.cursor() as cur:
            cur.execute(
                (
                    "SELECT workspace_path FROM sandbox_sessions "
                    "WHERE user_id=%s AND workspace_id=%s AND workspace_path IS NOT NULL "
                    "AND BTRIM(workspace_path) <> '' "
                    "AND (expires_at IS NULL OR expires_at = '' OR expires_at::timestamptz > %s::timestamptz)"
                ),
                (owner_key, workspace_key, now_iso),
            )
            rows = cur.fetchall() or []
            return [
                str(row.get("workspace_path")).strip()
                for row in rows
                if str((row or {}).get("workspace_path") or "").strip()
            ]

    def delete_session(self, session_id: str) -> bool:
        with self._lock, self._conn() as con, con.cursor() as cur:
            cur.execute("DELETE FROM sandbox_sessions WHERE id=%s", (str(session_id),))
            try:
                deleted = int(getattr(cur, "rowcount", 0))
            except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                deleted = 0
            return deleted > 0

    def put_acp_session_control(
        self,
        *,
        session_id: str,
        user_id: Any,
        sandbox_session_id: str | None,
        run_id: str | None,
        ssh_host: str | None = None,
        ssh_port: int | None = None,
        ssh_user: str | None = None,
        ssh_private_key: str | None = None,
        persona_id: str | None = None,
        workspace_id: str | None = None,
        workspace_group_id: str | None = None,
        scope_snapshot_id: str | None = None,
    ) -> None:
        now_ts = time.time()
        with self._lock, self._conn() as con:
            with con.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sandbox_acp_sessions(
                        id,user_id,sandbox_session_id,run_id,ssh_host,ssh_port,ssh_user,ssh_private_key,
                        persona_id,workspace_id,workspace_group_id,scope_snapshot_id,created_at,updated_at
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (id) DO UPDATE SET
                        user_id=EXCLUDED.user_id,
                        sandbox_session_id=EXCLUDED.sandbox_session_id,
                        run_id=EXCLUDED.run_id,
                        ssh_host=EXCLUDED.ssh_host,
                        ssh_port=EXCLUDED.ssh_port,
                        ssh_user=EXCLUDED.ssh_user,
                        ssh_private_key=EXCLUDED.ssh_private_key,
                        persona_id=EXCLUDED.persona_id,
                        workspace_id=EXCLUDED.workspace_id,
                        workspace_group_id=EXCLUDED.workspace_group_id,
                        scope_snapshot_id=EXCLUDED.scope_snapshot_id,
                        updated_at=EXCLUDED.updated_at
                    """,
                    (
                        str(session_id),
                        self._user_key(user_id),
                        (str(sandbox_session_id) if sandbox_session_id is not None else None),
                        (str(run_id) if run_id is not None else None),
                        (str(ssh_host) if ssh_host is not None else None),
                        (int(ssh_port) if ssh_port is not None else None),
                        (str(ssh_user) if ssh_user is not None else None),
                        (str(ssh_private_key) if ssh_private_key is not None else None),
                        (str(persona_id) if persona_id is not None else None),
                        (str(workspace_id) if workspace_id is not None else None),
                        (str(workspace_group_id) if workspace_group_id is not None else None),
                        (str(scope_snapshot_id) if scope_snapshot_id is not None else None),
                        float(now_ts),
                        float(now_ts),
                    ),
                )

    def get_acp_session_control(self, session_id: str) -> dict[str, Any] | None:
        with self._lock, self._conn() as con, con.cursor() as cur:
            cur.execute(
                (
                    "SELECT id,user_id,sandbox_session_id,run_id,ssh_host,ssh_port,ssh_user,ssh_private_key,"
                    "persona_id,workspace_id,workspace_group_id,scope_snapshot_id,created_at,updated_at "
                    "FROM sandbox_acp_sessions WHERE id=%s"
                ),
                (str(session_id),),
            )
            row = cur.fetchone()
            return dict(row) if isinstance(row, dict) else None

    def delete_acp_session_control(self, session_id: str) -> bool:
        with self._lock, self._conn() as con, con.cursor() as cur:
            cur.execute("DELETE FROM sandbox_acp_sessions WHERE id=%s", (str(session_id),))
            try:
                deleted = int(getattr(cur, "rowcount", 0))
            except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                deleted = 0
            return deleted > 0

    def put_vz_session_control(
        self,
        *,
        session_id: str,
        runtime: str,
        vm_id: str,
        template_id: str | None,
        workspace_mount: str | None,
        agent_ready: bool,
        helper_instance_id: str | None = None,
        helper_started_at: str | None = None,
    ) -> None:
        now_ts = time.time()
        with self._lock, self._conn() as con:
            with con.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sandbox_vz_sessions(
                        id,runtime,vm_id,template_id,workspace_mount,agent_ready,
                        helper_instance_id,helper_started_at,created_at,updated_at
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (id) DO UPDATE SET
                        runtime=EXCLUDED.runtime,
                        vm_id=EXCLUDED.vm_id,
                        template_id=EXCLUDED.template_id,
                        workspace_mount=EXCLUDED.workspace_mount,
                        agent_ready=EXCLUDED.agent_ready,
                        helper_instance_id=EXCLUDED.helper_instance_id,
                        helper_started_at=EXCLUDED.helper_started_at,
                        updated_at=EXCLUDED.updated_at
                    """,
                    (
                        str(session_id),
                        str(runtime),
                        str(vm_id),
                        (str(template_id) if template_id is not None else None),
                        (str(workspace_mount) if workspace_mount is not None else None),
                        bool(agent_ready),
                        coerce_optional_nonempty_string(helper_instance_id),
                        coerce_optional_nonempty_string(helper_started_at),
                        float(now_ts),
                        float(now_ts),
                    ),
                )

    def get_vz_session_control(self, session_id: str) -> dict[str, Any] | None:
        with self._lock, self._conn() as con, con.cursor() as cur:
            cur.execute(
                (
                    "SELECT id,runtime,vm_id,template_id,workspace_mount,agent_ready,"
                    "helper_instance_id,helper_started_at,created_at,updated_at "
                    "FROM sandbox_vz_sessions WHERE id=%s"
                ),
                (str(session_id),),
            )
            row = cur.fetchone()
            return dict(row) if isinstance(row, dict) else None

    def delete_vz_session_control(self, session_id: str) -> bool:
        with self._lock, self._conn() as con, con.cursor() as cur:
            cur.execute("DELETE FROM sandbox_vz_sessions WHERE id=%s", (str(session_id),))
            try:
                deleted = int(getattr(cur, "rowcount", 0))
            except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                deleted = 0
            return deleted > 0

    def list_vz_session_controls(self) -> list[dict[str, Any]]:
        with self._lock, self._conn() as con, con.cursor() as cur:
            cur.execute(
                "SELECT id,runtime,vm_id,template_id,workspace_mount,agent_ready,"
                "helper_instance_id,helper_started_at,created_at,updated_at "
                "FROM sandbox_vz_sessions ORDER BY id"
            )
            return [dict(row) for row in cur.fetchall() or [] if isinstance(row, dict)]

    def get_user_artifact_bytes(self, user_id: str) -> int:
        with self._lock, self._conn() as con, con.cursor() as cur:
            cur.execute("SELECT artifact_bytes FROM sandbox_usage WHERE user_id=%s", (user_id,))
            row = cur.fetchone()
            if not row:
                return 0
            try:
                return int(row.get("artifact_bytes") or 0)
            except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                return 0

    def increment_user_artifact_bytes(self, user_id: str, delta: int) -> None:
        if not user_id:
            return
        d = int(delta or 0)
        with self._lock, self._conn() as con:
            with con.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sandbox_usage(user_id, artifact_bytes) VALUES (%s, GREATEST(0, %s))
                    ON CONFLICT (user_id) DO UPDATE
                    SET artifact_bytes = GREATEST(0, COALESCE(sandbox_usage.artifact_bytes, 0) + EXCLUDED.artifact_bytes)
                    """,
                    (user_id, d),
                )

    def try_reserve_user_artifact_bytes(self, user_id: str, delta: int, cap_bytes: int) -> bool:
        if not user_id:
            return False
        d = int(delta or 0)
        if d <= 0:
            self.increment_user_artifact_bytes(user_id, d)
            return True
        with self._lock, self._conn() as con:
            with con.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sandbox_usage(user_id, artifact_bytes)
                    SELECT %s, %s
                    WHERE %s <= %s
                    ON CONFLICT (user_id) DO UPDATE
                    SET artifact_bytes = COALESCE(sandbox_usage.artifact_bytes, 0) + EXCLUDED.artifact_bytes
                    WHERE COALESCE(sandbox_usage.artifact_bytes, 0) + EXCLUDED.artifact_bytes <= %s
                    RETURNING artifact_bytes
                    """,
                    (user_id, d, d, int(cap_bytes), int(cap_bytes)),
                )
                return cur.fetchone() is not None

    def list_runs(
        self,
        *,
        image_digest: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        persona_id: str | None = None,
        workspace_id: str | None = None,
        workspace_group_id: str | None = None,
        scope_snapshot_id: str | None = None,
        phase: str | None = None,
        started_at_from: str | None = None,
        started_at_to: str | None = None,
        limit: int = 50,
        offset: int = 0,
        sort_desc: bool = True,
    ) -> list[dict]:
        order = "DESC" if sort_desc else "ASC"
        where = ["1=1"]
        params: list[Any] = []
        if image_digest:
            where.append("image_digest = %s")
            params.append(image_digest)
        if user_id:
            where.append("user_id = %s")
            params.append(user_id)
        if session_id:
            where.append("session_id = %s")
            params.append(session_id)
        if persona_id:
            where.append("persona_id = %s")
            params.append(persona_id)
        if workspace_id:
            where.append("workspace_id = %s")
            params.append(workspace_id)
        if workspace_group_id:
            where.append("workspace_group_id = %s")
            params.append(workspace_group_id)
        if scope_snapshot_id:
            where.append("scope_snapshot_id = %s")
            params.append(scope_snapshot_id)
        if phase:
            where.append("phase = %s")
            params.append(phase)
        if started_at_from:
            where.append("started_at >= %s")
            params.append(started_at_from)
        if started_at_to:
            where.append("started_at <= %s")
            params.append(started_at_to)
        sql = (
            "SELECT id,user_id,spec_version,runtime,runtime_version,base_image,session_id,persona_id,workspace_id,workspace_group_id,scope_snapshot_id,phase,exit_code,started_at,finished_at,message,image_digest,policy_hash,resource_usage "  # nosec B608
            f"FROM sandbox_runs WHERE {' AND '.join(where)} ORDER BY started_at {order} LIMIT %s OFFSET %s"
        )
        params.extend([int(limit), int(offset)])
        with self._lock, self._conn() as con, con.cursor() as cur:
            cur.execute(sql, tuple(params))
            items: list[dict] = []
            for row in cur.fetchall() or []:
                resource_usage = row.get("resource_usage")
                if isinstance(resource_usage, str):
                    try:
                        loaded = json.loads(resource_usage)
                        resource_usage = loaded if isinstance(loaded, dict) else None
                    except (TypeError, ValueError) as exc:
                        logger.debug(
                            "PostgresStore.list_runs ignored malformed resource_usage for run_id={}: {}",
                            row.get("id"),
                            exc,
                        )
                        resource_usage = None
                elif not isinstance(resource_usage, dict):
                    resource_usage = None
                items.append({
                    "id": row.get("id"),
                    "user_id": row.get("user_id"),
                    "spec_version": row.get("spec_version"),
                    "runtime": row.get("runtime"),
                    "runtime_version": row.get("runtime_version"),
                    "base_image": row.get("base_image"),
                    "session_id": row.get("session_id"),
                    "persona_id": row.get("persona_id"),
                    "workspace_id": row.get("workspace_id"),
                    "workspace_group_id": row.get("workspace_group_id"),
                    "scope_snapshot_id": row.get("scope_snapshot_id"),
                    "phase": row.get("phase"),
                    "exit_code": row.get("exit_code"),
                    "started_at": row.get("started_at"),
                    "finished_at": row.get("finished_at"),
                    "message": row.get("message"),
                    "image_digest": row.get("image_digest"),
                    "policy_hash": row.get("policy_hash"),
                    "resource_usage": resource_usage,
                })
            return items

    def count_runs(
        self,
        *,
        image_digest: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        persona_id: str | None = None,
        workspace_id: str | None = None,
        workspace_group_id: str | None = None,
        scope_snapshot_id: str | None = None,
        phase: str | None = None,
        started_at_from: str | None = None,
        started_at_to: str | None = None,
    ) -> int:
        where = ["1=1"]
        params: list[Any] = []
        if image_digest:
            where.append("image_digest = %s")
            params.append(image_digest)
        if user_id:
            where.append("user_id = %s")
            params.append(user_id)
        if session_id:
            where.append("session_id = %s")
            params.append(session_id)
        if persona_id:
            where.append("persona_id = %s")
            params.append(persona_id)
        if workspace_id:
            where.append("workspace_id = %s")
            params.append(workspace_id)
        if workspace_group_id:
            where.append("workspace_group_id = %s")
            params.append(workspace_group_id)
        if scope_snapshot_id:
            where.append("scope_snapshot_id = %s")
            params.append(scope_snapshot_id)
        if phase:
            where.append("phase = %s")
            params.append(phase)
        if started_at_from:
            where.append("started_at >= %s")
            params.append(started_at_from)
        if started_at_to:
            where.append("started_at <= %s")
            params.append(started_at_to)
        sql = f"SELECT COUNT(*) AS c FROM sandbox_runs WHERE {' AND '.join(where)}"  # nosec B608
        with self._lock, self._conn() as con, con.cursor() as cur:
            cur.execute(sql, tuple(params))
            row = cur.fetchone()
            try:
                return int(list(row.values())[0]) if row else 0
            except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                return 0

    def list_idempotency(
        self,
        *,
        endpoint: str | None = None,
        user_id: str | None = None,
        key: str | None = None,
        created_at_from: str | None = None,
        created_at_to: str | None = None,
        limit: int = 50,
        offset: int = 0,
        sort_desc: bool = True,
    ) -> list[dict]:
        order = "DESC" if sort_desc else "ASC"
        where = ["1=1"]
        params: list[Any] = []
        if endpoint:
            where.append("endpoint = %s")
            params.append(endpoint)
        if user_id:
            where.append("user_key = %s")
            params.append(user_id)
        if key:
            where.append("key = %s")
            params.append(key)
        if created_at_from:
            where.append("created_at >= %s")
            params.append(self._coerce_created_at(created_at_from))
        if created_at_to:
            where.append("created_at <= %s")
            params.append(self._coerce_created_at(created_at_to))
        sql = (
            "SELECT endpoint,user_key,key,fingerprint,object_id,created_at FROM sandbox_idempotency "  # nosec B608
            f"WHERE {' AND '.join(where)} ORDER BY created_at {order} LIMIT %s OFFSET %s"
        )
        params.extend([int(limit), int(offset)])
        with self._lock, self._conn() as con, con.cursor() as cur:
            cur.execute(sql, tuple(params))
            items: list[dict] = []
            for row in cur.fetchall() or []:
                iso_ct = None
                try:
                    if row.get("created_at") is not None:
                        iso_ct = datetime.fromtimestamp(float(row.get("created_at")), tz=timezone.utc).isoformat()
                except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                    iso_ct = None
                items.append({
                    "endpoint": row.get("endpoint"),
                    "user_id": row.get("user_key"),
                    "key": row.get("key"),
                    "fingerprint": row.get("fingerprint"),
                    "object_id": row.get("object_id"),
                    "created_at": iso_ct,
                })
            return items

    def count_idempotency(
        self,
        *,
        endpoint: str | None = None,
        user_id: str | None = None,
        key: str | None = None,
        created_at_from: str | None = None,
        created_at_to: str | None = None,
    ) -> int:
        where = ["1=1"]
        params: list[Any] = []
        if endpoint:
            where.append("endpoint = %s")
            params.append(endpoint)
        if user_id:
            where.append("user_key = %s")
            params.append(user_id)
        if key:
            where.append("key = %s")
            params.append(key)
        if created_at_from:
            where.append("created_at >= %s")
            params.append(self._coerce_created_at(created_at_from))
        if created_at_to:
            where.append("created_at <= %s")
            params.append(self._coerce_created_at(created_at_to))
        sql = f"SELECT COUNT(*) AS c FROM sandbox_idempotency WHERE {' AND '.join(where)}"  # nosec B608
        with self._lock, self._conn() as con, con.cursor() as cur:
            cur.execute(sql, tuple(params))
            row = cur.fetchone()
            try:
                return int(list(row.values())[0]) if row else 0
            except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                return 0

    def list_usage(
        self,
        *,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
        sort_desc: bool = True,
    ) -> list[dict]:
        with self._lock, self._conn() as con, con.cursor() as cur:
            cur.execute("SELECT user_id, artifact_bytes FROM sandbox_usage")
            usage_rows = {r.get("user_id"): int(r.get("artifact_bytes") or 0) for r in (cur.fetchall() or [])}
            cur.execute("SELECT user_id, resource_usage FROM sandbox_runs")
            agg: dict[str, dict] = {}
            for row in cur.fetchall() or []:
                u = row.get("user_id")
                if not u:
                    continue
                if user_id and u != user_id:
                    continue
                rs = agg.setdefault(u, {"runs_count": 0, "log_bytes": 0})
                rs["runs_count"] += 1
                try:
                    ru = row.get("resource_usage")
                    if isinstance(ru, str):
                        ru = json.loads(ru)
                    if ru and isinstance(ru.get("log_bytes"), int):
                        rs["log_bytes"] += int(ru.get("log_bytes") or 0)
                except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
                    pass
            users = set(usage_rows.keys()) | set(agg.keys())
            items: list[dict] = []
            for u in sorted(users, reverse=bool(sort_desc)):
                if user_id and u != user_id:
                    continue
                items.append({
                    "user_id": u,
                    "runs_count": int((agg.get(u) or {}).get("runs_count", 0)),
                    "log_bytes": int((agg.get(u) or {}).get("log_bytes", 0)),
                    "artifact_bytes": int(usage_rows.get(u, 0)),
                })
            return items[offset: offset + limit]

    # -- Durable run queue (Postgres) — delegated to DB_Management ----------------
    def enqueue_run(self, run_id: str, user_id: str, priority: int = 0) -> None:
        try:
            self._queue_db.enqueue(run_id, user_id, priority)
        except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"enqueue_run failed (pg): {e}")

    def dequeue_run(self, worker_id: str) -> dict | None:
        try:
            return self._queue_db.dequeue(worker_id)
        except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"dequeue_run failed (pg): {e}")
            return None

    def remove_from_queue(self, run_id: str) -> bool:
        try:
            return self._queue_db.remove(run_id)
        except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"remove_from_queue failed (pg): {e}")
            return False

    def count_usage(
        self,
        *,
        user_id: str | None = None,
    ) -> int:
        return len(self.list_usage(user_id=user_id, limit=10**9, offset=0, sort_desc=True))


def _resolve_pg_dsn() -> str | None:
    # Prefer explicit SANDBOX_STORE_PG_DSN, then env, then DATABASE_URL
    try:
        dsn = getattr(app_settings, "SANDBOX_STORE_PG_DSN", None)
    except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
        dsn = None
    dsn = dsn or os.getenv("SANDBOX_STORE_PG_DSN") or os.getenv("SANDBOX_PG_DSN")
    if not dsn:
        try:
            dsn = getattr(app_settings, "DATABASE_URL", None)
        except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
            dsn = None
    if not dsn:
        return None
    dsn_str = str(dsn)
    # Ignore sqlite URLs for cluster mode; require a real Postgres DSN.
    if dsn_str.strip().lower().startswith("sqlite"):
        return None
    return dsn_str


def _require_cluster_ready() -> str:
    dsn = _resolve_pg_dsn()
    if not dsn:
        raise ClusterStoreUnavailable(
            "SANDBOX_STORE_BACKEND=cluster requires a non-sqlite Postgres DSN "
            "(set SANDBOX_STORE_PG_DSN or DATABASE_URL)."
        )
    try:
        import psycopg  # noqa: F401
    except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS as exc:
        raise ClusterStoreUnavailable(
            "SANDBOX_STORE_BACKEND=cluster requires psycopg to be installed."
        ) from exc
    return dsn


def get_store() -> SandboxStore:
    backend = None
    try:
        backend = str(getattr(app_settings, "SANDBOX_STORE_BACKEND", "memory")).strip().lower()
    except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
        backend = "memory"
    if backend == "memory":
        ttl = int(getattr(app_settings, "SANDBOX_IDEMPOTENCY_TTL_SEC", 600))
        return InMemoryStore(idem_ttl_sec=ttl)
    if backend == "cluster":
        ttl = int(getattr(app_settings, "SANDBOX_IDEMPOTENCY_TTL_SEC", 600))
        dsn = _require_cluster_ready()
        return PostgresStore(dsn=dsn, idem_ttl_sec=ttl)
    # Default sqlite
    ttl = int(getattr(app_settings, "SANDBOX_IDEMPOTENCY_TTL_SEC", 600))
    try:
        db_path = getattr(app_settings, "SANDBOX_STORE_DB_PATH", None)
    except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
        db_path = None
    return SQLiteStore(db_path=db_path, idem_ttl_sec=ttl)


def get_store_mode() -> str:
    """Return the effective store mode for feature discovery.

    Values: memory | sqlite | cluster | unknown
    """
    try:
        backend = str(getattr(app_settings, "SANDBOX_STORE_BACKEND", "memory")).strip().lower()
    except _SANDBOX_STORE_NONCRITICAL_EXCEPTIONS:
        backend = "memory"
    if backend == "cluster":
        _ = _require_cluster_ready()
        return "cluster"
    if backend in {"memory", "sqlite"}:
        return backend
    return "unknown"
