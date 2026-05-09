from __future__ import annotations

import contextlib
import json
import os
import shutil
import stat
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.config import settings as app_settings
from tldw_Server_API.app.core.testing import is_truthy

from .models import RunPhase, RunSpec, RunStatus, RuntimeType, Session, SessionSpec, TrustLevel
from .policy import SandboxPolicy, SandboxPolicyConfig
from .store import IdempotencyConflict as StoreIdemConflict
from .store import get_store

_SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS = (
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
)

_OWNER_ONLY_DIR_MODE = stat.S_IRWXU


def _coerce_optional_nonempty_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


class IdempotencyConflict(Exception):
    def __init__(self, original_id: str, key: str | None = None, prior_created_at: str | None = None, message: str = "Idempotency conflict") -> None:
        super().__init__(message)
        self.original_id = original_id
        self.key = key
        # ISO 8601 timestamp string (UTC) preferred at orchestrator/api layers
        self.prior_created_at = prior_created_at


class SessionActiveRunsConflict(Exception):
    def __init__(self, session_id: str, active_runs: int, message: str = "session_has_active_runs") -> None:
        super().__init__(message)
        self.session_id = str(session_id)
        self.active_runs = max(0, int(active_runs))


def _fingerprint_body(body: dict[str, Any]) -> str:
    try:
        canon = json.dumps(body, sort_keys=True, separators=(",", ":"))
    except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
        # Fallback: string-ify unsafely
        canon = str(body)
    import hashlib

    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


@dataclass
class _IdemRecord:
    key: str
    endpoint: str
    user_key: str
    fingerprint: str
    object_id: str
    response_body: dict[str, Any]
    created_at: float


class SandboxOrchestrator:
    """In-memory orchestrator with idempotency and a simple run queue.

    Not production-grade; intended to be replaced by a pluggable backend.
    """

    def __init__(self, policy: SandboxPolicy | None = None) -> None:
        cfg = SandboxPolicyConfig.from_settings()
        self.policy = policy or SandboxPolicy(cfg)
        self._lock = threading.RLock()
        self._sessions: dict[str, Session] = {}
        # Store backend for runs/idempotency/usage and durable run queue
        self._store = get_store()
        # In-memory run queue caches (L1 over durable store queue).
        # These provide fast access for non-critical operations like wait
        # time estimation and fallback counting when the store is unavailable.
        self._queue: list[tuple[str, float]] = []
        self._enqueue_index: dict[str, float] = {}
        self._queue_meta: dict[str, dict[str, str | None]] = {}
        try:
            self._idem_ttl_sec = int(getattr(app_settings, "SANDBOX_IDEMPOTENCY_TTL_SEC", 600))
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
            self._idem_ttl_sec = 600
        # Queue policy
        try:
            import os as _os
            self._queue_max = int(_os.getenv("SANDBOX_QUEUE_MAX_LENGTH") or getattr(app_settings, "SANDBOX_QUEUE_MAX_LENGTH", 100))
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
            self._queue_max = 100
        try:
            import os as _os
            self._queue_ttl = int(_os.getenv("SANDBOX_QUEUE_TTL_SEC") or getattr(app_settings, "SANDBOX_QUEUE_TTL_SEC", 120))
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
            self._queue_ttl = 120
        self._artifact_gc_lock = threading.RLock()
        self._artifact_gc_last_run_monotonic = 0.0
        self._session_roots: dict[str, str] = {}
        self._artifacts: dict[str, dict[str, bytes]] = {}

    # -----------------
    # Idempotency Core
    # -----------------
    def _user_key(self, user_id: Any) -> str:
        try:
            return str(user_id)
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
            return ""

    def _effective_int_limit(self, env_key: str, settings_attr: str, default: int) -> int:
        try:
            import os as _os
            raw = _os.getenv(env_key)
            if raw is None:
                raw = getattr(app_settings, settings_attr, default)
            return int(raw)
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
            return int(default)

    def _count_queued_runs(
        self,
        *,
        user_id: str | None = None,
        persona_id: str | None = None,
        workspace_id: str | None = None,
        workspace_group_id: str | None = None,
    ) -> int:
        try:
            return int(self._store.count_runs(
                user_id=user_id,
                persona_id=persona_id,
                workspace_id=workspace_id,
                workspace_group_id=workspace_group_id,
                phase=RunPhase.queued.value,
            ))
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
            with self._lock:
                if user_id is None and persona_id is None and workspace_id is None and workspace_group_id is None:
                    return len(self._queue)
                count = 0
                for meta in self._queue_meta.values():
                    if user_id is not None and meta.get("user_id") != user_id:
                        continue
                    if persona_id is not None and meta.get("persona_id") != persona_id:
                        continue
                    if workspace_id is not None and meta.get("workspace_id") != workspace_id:
                        continue
                    if workspace_group_id is not None and meta.get("workspace_group_id") != workspace_group_id:
                        continue
                    count += 1
                return count



    def _check_idem(self, endpoint: str, user_id: Any, idem_key: str | None, body: dict[str, Any]) -> dict[str, Any] | None:
        try:
            return self._store.check_idempotency(endpoint, user_id, idem_key, body)
        except StoreIdemConflict as e:
            # Convert store-level epoch seconds into ISO 8601 for API surfaces
            iso_ct: str | None = None
            try:
                if getattr(e, "created_at", None) is not None:
                    from datetime import datetime, timezone
                    iso_ct = datetime.fromtimestamp(float(e.created_at), tz=timezone.utc).isoformat()
            except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                iso_ct = None
            raise IdempotencyConflict(e.original_id, key=getattr(e, "key", None), prior_created_at=iso_ct) from e

    def _store_idem(self, endpoint: str, user_id: Any, idem_key: str | None, body: dict[str, Any], object_id: str, response: dict[str, Any]) -> None:
        self._store.store_idempotency(endpoint, user_id, idem_key, body, object_id, response)

    def _session_from_record(self, record: dict[str, Any]) -> Session | None:
        if not isinstance(record, dict):
            return None
        sid = str(record.get("id") or "").strip()
        if not sid:
            return None
        runtime = self.policy.cfg.default_runtime
        runtime_raw = record.get("runtime")
        if runtime_raw:
            try:
                runtime = RuntimeType(str(runtime_raw))
            except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                runtime = self.policy.cfg.default_runtime
        expires_at = None
        expires_raw = record.get("expires_at")
        if expires_raw:
            try:
                expires_at = datetime.fromisoformat(str(expires_raw))
            except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                expires_at = None
        cpu_limit = None
        if record.get("cpu_limit") is not None:
            try:
                cpu_limit = float(record.get("cpu_limit"))
            except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                cpu_limit = None
        memory_mb = None
        if record.get("memory_mb") is not None:
            try:
                memory_mb = int(record.get("memory_mb"))
            except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                memory_mb = None
        timeout_sec = 300
        if record.get("timeout_sec") is not None:
            try:
                timeout_sec = int(record.get("timeout_sec"))
            except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                timeout_sec = 300
        trust_level = None
        trust_raw = record.get("trust_level")
        if trust_raw:
            try:
                trust_level = TrustLevel(str(trust_raw))
            except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                trust_level = None
        return Session(
            id=sid,
            runtime=runtime,
            base_image=(str(record.get("base_image")) if record.get("base_image") is not None else None),
            expires_at=expires_at,
            cpu_limit=cpu_limit,
            memory_mb=memory_mb,
            timeout_sec=timeout_sec,
            network_policy=(str(record.get("network_policy") or "deny_all")),
            env=(
                {str(k): str(v) for k, v in (record.get("env") or {}).items()}
                if isinstance(record.get("env"), dict)
                else {}
            ),
            labels=(
                {str(k): str(v) for k, v in (record.get("labels") or {}).items()}
                if isinstance(record.get("labels"), dict)
                else {}
            ),
            trust_level=trust_level,
            persona_id=(str(record.get("persona_id")) if record.get("persona_id") is not None else None),
            workspace_id=(str(record.get("workspace_id")) if record.get("workspace_id") is not None else None),
            workspace_group_id=(str(record.get("workspace_group_id")) if record.get("workspace_group_id") is not None else None),
            scope_snapshot_id=(str(record.get("scope_snapshot_id")) if record.get("scope_snapshot_id") is not None else None),
        )

    def _drop_cached_session(self, session_id: str) -> None:
        sid = str(session_id or "").strip()
        if not sid:
            return
        with self._lock:
            self._sessions.pop(sid, None)
            self._session_roots.pop(sid, None)

    def get_session(self, session_id: str, *, allow_cache_on_store_error: bool = True) -> Session | None:
        sid = str(session_id or "").strip()
        if not sid:
            return None
        with self._lock:
            cached = self._sessions.get(sid)
        if cached is not None:
            # Validate cache against shared store so cross-node deletes invalidate
            # stale in-process session state.
            try:
                record = self._store.get_session(sid)
            except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"store.get_session failed during cache validation: {e}")
                if allow_cache_on_store_error:
                    return cached
                return None
            if not isinstance(record, dict):
                self._drop_cached_session(sid)
                return None
            sess = self._session_from_record(record)
            if sess is None:
                self._drop_cached_session(sid)
                return None
            with self._lock:
                self._sessions[sid] = sess
            return sess
        try:
            record = self._store.get_session(sid)
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"store.get_session failed: {e}")
            return None
        sess = self._session_from_record(record or {})
        if sess is not None:
            with self._lock:
                self._sessions[sid] = sess
        return sess

    # -----------------
    # Sessions
    # -----------------
    def create_session(self, user_id: Any, spec: SessionSpec, spec_version: str, idem_key: str | None, body: dict[str, Any]) -> Session:
        # Check idempotency storage first
        stored = self._check_idem("sessions", user_id, idem_key, body)
        if stored is not None:
            sid = stored.get("id")
            if sid:
                existing = self.get_session(str(sid))
                if existing is not None:
                    owner = self.get_session_owner(str(sid)) or self._user_key(user_id)
                    with contextlib.suppress(_SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS):
                        self._ensure_workspace(owner, str(sid))
                    return existing
            # Fallback reconstruction from idempotency payload.
            runtime = spec.runtime or self.policy.cfg.default_runtime
            runtime_raw = stored.get("runtime")
            if runtime_raw:
                try:
                    runtime = RuntimeType(str(runtime_raw))
                except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                    runtime = spec.runtime or self.policy.cfg.default_runtime
            expires_at = None
            expires_raw = stored.get("expires_at")
            if expires_raw:
                try:
                    expires_at = datetime.fromisoformat(str(expires_raw))
                except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                    expires_at = None
            sid_str = str(stored.get("id", "") or "")
            sess = Session(
                id=sid_str,
                runtime=runtime,
                base_image=(stored.get("base_image") if stored.get("base_image") is not None else spec.base_image),
                expires_at=expires_at,
                cpu_limit=(
                    float(stored.get("cpu_limit"))
                    if stored.get("cpu_limit") is not None
                    else spec.cpu_limit
                ),
                memory_mb=(
                    int(stored.get("memory_mb"))
                    if stored.get("memory_mb") is not None
                    else spec.memory_mb
                ),
                timeout_sec=(
                    int(stored.get("timeout_sec"))
                    if stored.get("timeout_sec") is not None
                    else spec.timeout_sec
                ),
                network_policy=(
                    str(stored.get("network_policy"))
                    if stored.get("network_policy") is not None
                    else spec.network_policy
                ),
                env=(
                    {str(k): str(v) for k, v in (stored.get("env") or {}).items()}
                    if isinstance(stored.get("env"), dict)
                    else dict(spec.env or {})
                ),
                labels=(
                    {str(k): str(v) for k, v in (stored.get("labels") or {}).items()}
                    if isinstance(stored.get("labels"), dict)
                    else dict(spec.labels or {})
                ),
                trust_level=(
                    TrustLevel(str(stored.get("trust_level")))
                    if stored.get("trust_level") is not None
                    else spec.trust_level
                ),
                persona_id=(
                    str(stored.get("persona_id"))
                    if stored.get("persona_id") is not None
                    else spec.persona_id
                ),
                workspace_id=(
                    str(stored.get("workspace_id"))
                    if stored.get("workspace_id") is not None
                    else spec.workspace_id
                ),
                workspace_group_id=(
                    str(stored.get("workspace_group_id"))
                    if stored.get("workspace_group_id") is not None
                    else spec.workspace_group_id
                ),
                scope_snapshot_id=(
                    str(stored.get("scope_snapshot_id"))
                    if stored.get("scope_snapshot_id") is not None
                    else spec.scope_snapshot_id
                ),
            )
            owner = self.get_session_owner(sid_str) or self._user_key(user_id)
            ws_path = None
            with contextlib.suppress(_SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS):
                ws_path = self._ensure_workspace(owner, sid_str)
            with contextlib.suppress(_SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS):
                self._store.put_session(
                    owner,
                    session_id=sid_str,
                    runtime=sess.runtime.value if sess.runtime else None,
                    base_image=sess.base_image,
                    cpu_limit=sess.cpu_limit,
                    memory_mb=sess.memory_mb,
                    timeout_sec=sess.timeout_sec,
                    network_policy=sess.network_policy,
                    env=sess.env,
                    labels=sess.labels,
                    trust_level=(sess.trust_level.value if sess.trust_level else None),
                    expires_at_iso=(sess.expires_at.isoformat() if sess.expires_at else None),
                    workspace_path=ws_path,
                    persona_id=sess.persona_id,
                    workspace_id=sess.workspace_id,
                    workspace_group_id=sess.workspace_group_id,
                    scope_snapshot_id=sess.scope_snapshot_id,
                )
            with self._lock:
                self._sessions[sid_str] = sess
            return sess

        # Create a new session (workspace optional in scaffold)
        sid = str(uuid.uuid4())
        expires_at: datetime | None = None
        try:
            ttl_sec = int(getattr(app_settings, "SANDBOX_SESSION_TTL_SEC", 0))
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
            ttl_sec = 0
        if ttl_sec and ttl_sec > 0:
            expires_at = datetime.utcnow() + timedelta(seconds=int(ttl_sec))
        sess = Session(
            id=sid,
            runtime=spec.runtime or self.policy.cfg.default_runtime,
            base_image=spec.base_image,
            expires_at=expires_at,
            cpu_limit=spec.cpu_limit,
            memory_mb=spec.memory_mb,
            timeout_sec=spec.timeout_sec,
            network_policy=spec.network_policy,
            env=dict(spec.env or {}),
            labels=dict(spec.labels or {}),
            trust_level=spec.trust_level,
            persona_id=spec.persona_id,
            workspace_id=spec.workspace_id,
            workspace_group_id=spec.workspace_group_id,
            scope_snapshot_id=spec.scope_snapshot_id,
        )
        owner = self._user_key(user_id)
        ws_path = None
        with contextlib.suppress(_SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS):
            ws_path = self._ensure_workspace(owner, sid)
        with contextlib.suppress(_SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS):
            self._store.put_session(
                owner,
                session_id=sid,
                runtime=sess.runtime.value if sess.runtime else None,
                base_image=sess.base_image,
                cpu_limit=sess.cpu_limit,
                memory_mb=sess.memory_mb,
                timeout_sec=sess.timeout_sec,
                network_policy=sess.network_policy,
                env=sess.env,
                labels=sess.labels,
                trust_level=(sess.trust_level.value if sess.trust_level else None),
                expires_at_iso=(sess.expires_at.isoformat() if sess.expires_at else None),
                workspace_path=ws_path,
                persona_id=sess.persona_id,
                workspace_id=sess.workspace_id,
                workspace_group_id=sess.workspace_group_id,
                scope_snapshot_id=sess.scope_snapshot_id,
            )
        with self._lock:
            self._sessions[sid] = sess
        # Store idempotent response body
        resp = {
            "id": sid,
            "runtime": sess.runtime.value,
            "base_image": sess.base_image,
            "cpu_limit": sess.cpu_limit,
            "memory_mb": sess.memory_mb,
            "timeout_sec": sess.timeout_sec,
            "network_policy": sess.network_policy,
            "env": dict(sess.env or {}),
            "labels": dict(sess.labels or {}),
            "trust_level": (sess.trust_level.value if sess.trust_level else None),
            "expires_at": (sess.expires_at.isoformat() if sess.expires_at else None),
            "persona_id": sess.persona_id,
            "workspace_id": sess.workspace_id,
            "workspace_group_id": sess.workspace_group_id,
            "scope_snapshot_id": sess.scope_snapshot_id,
        }
        self._store_idem("sessions", user_id, idem_key, body, sid, resp)
        return sess

    # -----------------
    # Runs
    # -----------------
    def enqueue_run(self, user_id: Any, spec: RunSpec, spec_version: str, idem_key: str | None, body: dict[str, Any]) -> RunStatus:
        # Check idempotency
        stored = self._check_idem("runs", user_id, idem_key, body)
        if stored is not None:
            rid = stored.get("id", "")
            # Return stored status if available
            try:
                st = self._store.get_run(rid)
                if st:
                    return st
            except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                pass
            # Otherwise synthesize minimal queued status
            return RunStatus(
                id=rid,
                phase=RunPhase.queued,
                spec_version=spec_version,
                runtime=spec.runtime,
                base_image=spec.base_image,
                session_id=spec.session_id,
                persona_id=spec.persona_id,
                workspace_id=spec.workspace_id,
                workspace_group_id=spec.workspace_group_id,
                scope_snapshot_id=spec.scope_snapshot_id,
            )

        # Enforce queue capacity: prune TTL then check max length
        self._prune_queue_ttl()
        owner_key = self._user_key(user_id)
        persona_key = str(spec.persona_id).strip() if spec.persona_id else None
        workspace_key = str(spec.workspace_id).strip() if spec.workspace_id else None
        workspace_group_key = str(spec.workspace_group_id).strip() if spec.workspace_group_id else None
        with self._lock:
            # Read effective queue capacity at call time to honor per-test env overrides
            effective_queue_max = self._effective_int_limit(
                "SANDBOX_QUEUE_MAX_LENGTH",
                "SANDBOX_QUEUE_MAX_LENGTH",
                100,
            )
            retry_after = max(
                1,
                self._effective_int_limit(
                    "SANDBOX_QUEUE_TTL_SEC",
                    "SANDBOX_QUEUE_TTL_SEC",
                    120,
                ),
            )
            # If max length is <= 0, treat as no capacity (force backpressure)
            queued_global = self._count_queued_runs()
            if effective_queue_max <= 0 or queued_global >= effective_queue_max:
                raise QueueFull(
                    retry_after=retry_after,
                    reason="queue_capacity_exceeded",
                    quota_scope="global",
                    limit=effective_queue_max,
                )

            per_user_limit = self._effective_int_limit(
                "SANDBOX_QUEUE_MAX_PER_USER",
                "SANDBOX_QUEUE_MAX_PER_USER",
                0,
            )
            if per_user_limit > 0:
                queued_for_user = self._count_queued_runs(user_id=owner_key)
                if queued_for_user >= per_user_limit:
                    raise QueueFull(
                        retry_after=retry_after,
                        reason="user_queue_quota_exceeded",
                        quota_scope="user_id",
                        limit=per_user_limit,
                    )

            per_persona_limit = self._effective_int_limit(
                "SANDBOX_QUEUE_MAX_PER_PERSONA",
                "SANDBOX_QUEUE_MAX_PER_PERSONA",
                0,
            )
            if per_persona_limit > 0 and persona_key:
                queued_for_persona = self._count_queued_runs(persona_id=persona_key)
                if queued_for_persona >= per_persona_limit:
                    raise QueueFull(
                        retry_after=retry_after,
                        reason="persona_queue_quota_exceeded",
                        quota_scope="persona_id",
                        limit=per_persona_limit,
                    )

            per_workspace_limit = self._effective_int_limit(
                "SANDBOX_QUEUE_MAX_PER_WORKSPACE",
                "SANDBOX_QUEUE_MAX_PER_WORKSPACE",
                0,
            )
            if per_workspace_limit > 0 and workspace_key:
                queued_for_workspace = self._count_queued_runs(workspace_id=workspace_key)
                if queued_for_workspace >= per_workspace_limit:
                    raise QueueFull(
                        retry_after=retry_after,
                        reason="workspace_queue_quota_exceeded",
                        quota_scope="workspace_id",
                        limit=per_workspace_limit,
                    )

            per_workspace_group_limit = self._effective_int_limit(
                "SANDBOX_QUEUE_MAX_PER_WORKSPACE_GROUP",
                "SANDBOX_QUEUE_MAX_PER_WORKSPACE_GROUP",
                0,
            )
            if per_workspace_group_limit > 0 and workspace_group_key:
                queued_for_workspace_group = self._count_queued_runs(workspace_group_id=workspace_group_key)
                if queued_for_workspace_group >= per_workspace_group_limit:
                    raise QueueFull(
                        retry_after=retry_after,
                        reason="workspace_group_queue_quota_exceeded",
                        quota_scope="workspace_group_id",
                        limit=per_workspace_group_limit,
                    )

        # Create new run in queued state
        rid = str(uuid.uuid4())
        status = RunStatus(
            id=rid,
            phase=RunPhase.queued,
            spec_version=spec_version,
            runtime=spec.runtime,
            base_image=spec.base_image,
            session_id=spec.session_id,
            persona_id=spec.persona_id,
            workspace_id=spec.workspace_id,
            workspace_group_id=spec.workspace_group_id,
            scope_snapshot_id=spec.scope_snapshot_id,
        )
        # Optional: estimated start time based on queue length and a per-run estimate
        try:
            from datetime import datetime, timedelta
            per_run = int(getattr(app_settings, "SANDBOX_QUEUE_ESTIMATED_WAIT_PER_RUN_SEC", 5))
            with self._lock:
                q_len = len(self._queue)
            status.estimated_start_time = datetime.utcnow() + timedelta(seconds=max(0, q_len) * max(0, per_run))
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
            pass

        # Enqueue and persist
        with self._lock:
            ts = time.time()
            self._queue.append((rid, ts))
            self._enqueue_index[rid] = ts
            self._queue_meta[rid] = {
                "user_id": owner_key,
                "persona_id": persona_key,
                "workspace_id": workspace_key,
                "workspace_group_id": workspace_group_key,
            }
        try:
            self._store.put_run(user_id, status)
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"store.put_run failed: {e}")
        # Persist to durable queue for cross-restart recovery
        try:
            self._store.enqueue_run(rid, owner_key, priority=0)
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"store.enqueue_run failed: {e}")
        self._store_idem("runs", user_id, idem_key, body, rid, {
            "id": rid,
            "phase": status.phase.value,
            "spec_version": spec_version,
            "runtime": spec.runtime.value if spec.runtime else None,
            "base_image": spec.base_image,
            "session_id": spec.session_id,
            "persona_id": spec.persona_id,
            "workspace_id": spec.workspace_id,
            "workspace_group_id": spec.workspace_group_id,
            "scope_snapshot_id": spec.scope_snapshot_id,
            "exit_code": status.exit_code,
        })
        return status

    def _prune_queue_ttl(self) -> None:
        """Drop queued runs older than TTL and mark them expired."""
        try:
            ttl = max(1, int(self._queue_ttl))
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
            ttl = 120
        now = time.time()
        expired: list[str] = []
        with self._lock:
            kept: list[tuple[str, float]] = []
            for rid, ts in self._queue:
                if now - ts > ttl:
                    expired.append(rid)
                else:
                    kept.append((rid, ts))
            self._queue = kept
            for rid in expired:
                with contextlib.suppress(_SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS):
                    self._enqueue_index.pop(rid, None)
                    self._queue_meta.pop(rid, None)
        if not expired:
            return
        # Remove expired entries from durable queue
        for rid in expired:
            with contextlib.suppress(_SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS):
                self._store.remove_from_queue(rid)
        from datetime import datetime
        for rid in expired:
            try:
                st = self._store.get_run(rid)
                if st and st.phase == RunPhase.queued:
                    st.phase = RunPhase.failed
                    st.message = "queue_ttl_expired"
                    st.finished_at = datetime.utcnow()
                    self._store.update_run(st)
                    # Metrics: TTL expiry, include runtime label if available
                    try:
                        from tldw_Server_API.app.core.Metrics import increment_counter as _inc
                        rt_label = None
                        try:
                            rt_label = st.runtime.value if getattr(st, "runtime", None) else None
                        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                            rt_label = None
                        labels = {"component": "sandbox", "reason": "queue_ttl_expired"}
                        if rt_label:
                            labels["runtime"] = rt_label
                        _inc("sandbox_queue_ttl_expired_total", labels=labels)
                    except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                        pass
                    try:
                        from .streams import get_hub
                        get_hub().publish_event(rid, "end", {"exit_code": None, "reason": "queue_ttl_expired"})
                    except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                        pass
            except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                continue

    def get_enqueue_time(self, run_id: str) -> float | None:
        with self._lock:
            return self._enqueue_index.get(run_id)

    # -----------------
    # Lookups (stubs)
    # -----------------
    def get_run(self, run_id: str) -> RunStatus | None:
        try:
            return self._store.get_run(run_id)
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"store.get_run failed: {e}")
            return None

    def update_run(self, run_id: str, status: RunStatus) -> None:
        try:
            self._store.update_run(status)
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"store.update_run failed: {e}")
        # Cleanup enqueue index and durable queue when leaving queued phase
        try:
            if status.phase != RunPhase.queued:
                with self._lock:
                    self._enqueue_index.pop(run_id, None)
                    self._queue_meta.pop(run_id, None)
                    if self._queue:
                        self._queue = [(rid, ts) for (rid, ts) in self._queue if rid != run_id]
                with contextlib.suppress(_SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS):
                    self._store.remove_from_queue(run_id)
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
            pass

    def try_claim_run(self, run_id: str, *, worker_id: str, lease_seconds: int = 30) -> RunStatus | None:
        try:
            return self._store.try_claim_run(str(run_id), worker_id=str(worker_id), lease_seconds=int(lease_seconds))
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"store.try_claim_run failed: {e}")
            return None

    def renew_run_claim(self, run_id: str, *, worker_id: str, lease_seconds: int = 30) -> bool:
        try:
            return bool(self._store.renew_run_claim(str(run_id), worker_id=str(worker_id), lease_seconds=int(lease_seconds)))
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"store.renew_run_claim failed: {e}")
            return False

    def release_run_claim(self, run_id: str, *, worker_id: str) -> bool:
        try:
            return bool(self._store.release_run_claim(str(run_id), worker_id=str(worker_id)))
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"store.release_run_claim failed: {e}")
            return False

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
        try:
            return self._store.try_admit_run_start(
                str(run_id),
                worker_id=str(worker_id),
                max_active_runs=int(max_active_runs),
                lease_seconds=int(lease_seconds),
                max_active_per_user=int(max_active_per_user),
                max_active_per_persona=int(max_active_per_persona),
                max_active_per_workspace=int(max_active_per_workspace),
                max_active_per_workspace_group=int(max_active_per_workspace_group),
            )
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"store.try_admit_run_start failed: {e}")
            return None

    def get_run_owner(self, run_id: str) -> str | None:
        try:
            return self._store.get_run_owner(run_id)
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"store.get_run_owner failed: {e}")
            return None

    def get_session_owner(self, session_id: str) -> str | None:
        try:
            return self._store.get_session_owner(str(session_id))
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"store.get_session_owner failed: {e}")
            return None

    def _prune_expired_sessions(self) -> None:
        now = datetime.utcnow()
        expired: list[str] = []
        with self._lock:
            for sid, sess in list(self._sessions.items()):
                if sess.expires_at and sess.expires_at <= now:
                    expired.append(sid)
        for sid in expired:
            try:
                self.destroy_session(sid)
            except SessionActiveRunsConflict:
                continue
            except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                continue

    def destroy_session(self, session_id: str, *, remove_workspace_tree: bool = True) -> bool:
        sid = str(session_id or "").strip()
        if not sid:
            return False
        active_runs = (
            self.count_runs(session_id=sid, phase=RunPhase.queued.value)
            + self.count_runs(session_id=sid, phase=RunPhase.starting.value)
            + self.count_runs(session_id=sid, phase=RunPhase.running.value)
        )
        if active_runs > 0:
            raise SessionActiveRunsConflict(session_id=sid, active_runs=active_runs)
        run_rows: list[dict] = []
        offset = 0
        page_size = 500
        while True:
            rows = self.list_runs(session_id=sid, limit=page_size, offset=offset, sort_desc=False)
            if not rows:
                break
            run_rows.extend(rows)
            if len(rows) < page_size:
                break
            offset += page_size
        ws_path = None
        removed = False
        store_row = None
        with contextlib.suppress(_SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS):
            store_row = self._store.get_session(sid)
        with self._lock:
            if sid in self._sessions:
                self._sessions.pop(sid, None)
                removed = True
            ws_path = self._session_roots.pop(sid, None)
        if not ws_path and isinstance(store_row, dict):
            try:
                ws_candidate = store_row.get("workspace_path")
                if ws_candidate:
                    ws_path = str(ws_candidate)
            except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                ws_path = None
        store_removed = False
        with contextlib.suppress(_SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS):
            store_removed = bool(self._store.delete_session(sid))
        for row in run_rows:
            try:
                rid = str(row.get("id") or "").strip()
            except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                rid = ""
            if not rid:
                continue
            try:
                owner = str(row.get("user_id") or "").strip() or "unknown"
            except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                owner = "unknown"
            with contextlib.suppress(_SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS):
                self._remove_run_artifacts(rid, owner=owner, decrement_usage=True)
        if remove_workspace_tree and ws_path:
            with contextlib.suppress(_SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS):
                ws_path_obj = Path(str(ws_path))
                session_root = ws_path_obj.parent if ws_path_obj.name == "workspace" else ws_path_obj
                shutil.rmtree(session_root, ignore_errors=True)
        return bool(removed or store_removed)

    # -----------------
    # Artifacts
    # -----------------
    def _artifact_root(self) -> Path:
        root = os.getenv("SANDBOX_SHARED_ARTIFACTS_DIR")
        if not root:
            try:
                root = getattr(app_settings, "SANDBOX_ROOT_DIR", None)
            except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                root = None
        if not root:
            try:
                proj = getattr(app_settings, "PROJECT_ROOT", ".")
            except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                proj = "."
            root = Path(str(proj)) / "tmp_dir" / "sandbox"
        return Path(str(root))

    def _effective_artifact_ttl_hours(self) -> int:
        try:
            raw = os.getenv("SANDBOX_ARTIFACT_TTL_HOURS")
            if raw is None:
                raw = getattr(app_settings, "SANDBOX_ARTIFACT_TTL_HOURS", 24)
            return max(0, int(raw))
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
            return 24

    def _effective_artifact_janitor_interval_sec(self) -> int:
        try:
            raw = os.getenv("SANDBOX_ARTIFACT_JANITOR_INTERVAL_SEC")
            if raw is None:
                raw = getattr(app_settings, "SANDBOX_ARTIFACT_JANITOR_INTERVAL_SEC", 30)
            return max(0, int(raw))
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
            return 30

    @staticmethod
    def _artifact_dir_stats(art_dir: Path) -> tuple[int, int]:
        files = 0
        total_bytes = 0
        for root, _dirs, names in os.walk(art_dir):
            for fn in names:
                files += 1
                full = Path(root) / fn
                with contextlib.suppress(_SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS):
                    total_bytes += int(full.stat().st_size)
        return files, total_bytes

    def _remove_run_artifacts(
        self,
        run_id: str,
        *,
        owner: str,
        decrement_usage: bool = True,
    ) -> dict[str, int]:
        art_dir = self._artifact_dir(owner or "unknown", run_id)
        if not art_dir.exists():
            with self._lock:
                self._artifacts.pop(run_id, None)
            return {"removed_runs": 0, "removed_files": 0, "removed_bytes": 0}
        removed_files, removed_bytes = self._artifact_dir_stats(art_dir)
        with contextlib.suppress(_SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS):
            shutil.rmtree(art_dir, ignore_errors=True)
        if art_dir.exists():
            return {"removed_runs": 0, "removed_files": 0, "removed_bytes": 0}
        with self._lock:
            self._artifacts.pop(run_id, None)
        if decrement_usage and removed_bytes > 0:
            with contextlib.suppress(_SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS):
                self._store.increment_user_artifact_bytes(owner or "unknown", -int(removed_bytes))
        return {
            "removed_runs": 1,
            "removed_files": int(removed_files),
            "removed_bytes": int(removed_bytes),
        }

    def _maybe_prune_expired_artifacts(self) -> None:
        with contextlib.suppress(_SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS):
            self.prune_expired_artifacts(force=False)

    def prune_expired_artifacts(
        self,
        *,
        force: bool = False,
        now_utc: datetime | None = None,
    ) -> dict[str, int]:
        start_monotonic = time.monotonic()
        ttl_hours = self._effective_artifact_ttl_hours()
        interval_sec = self._effective_artifact_janitor_interval_sec()
        now_monotonic = time.monotonic()
        with self._artifact_gc_lock:
            if not force and interval_sec > 0 and (now_monotonic - self._artifact_gc_last_run_monotonic) < interval_sec:
                return {"removed_runs": 0, "removed_files": 0, "removed_bytes": 0}
            self._artifact_gc_last_run_monotonic = now_monotonic

        now = now_utc or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        cutoff = now - timedelta(hours=ttl_hours)
        removed_runs = 0
        removed_files = 0
        removed_bytes = 0
        offset = 0
        page_size = 500
        while True:
            try:
                rows = self._store.list_runs(limit=page_size, offset=offset, sort_desc=False)
            except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                rows = []
            if not rows:
                break
            for row in rows:
                try:
                    phase = str(row.get("phase") or "")
                except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                    phase = ""
                if phase not in {
                    RunPhase.completed.value,
                    RunPhase.failed.value,
                    RunPhase.killed.value,
                    RunPhase.timed_out.value,
                }:
                    continue
                try:
                    rid = str(row.get("id") or "").strip()
                except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                    rid = ""
                if not rid:
                    continue
                terminal_iso = row.get("finished_at") or row.get("started_at")
                terminal_dt = None
                try:
                    if isinstance(terminal_iso, datetime):
                        terminal_dt = terminal_iso
                    elif terminal_iso:
                        terminal_dt = datetime.fromisoformat(str(terminal_iso))
                except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                    terminal_dt = None
                if terminal_dt is not None and terminal_dt.tzinfo is None:
                    terminal_dt = terminal_dt.replace(tzinfo=timezone.utc)
                if terminal_dt is None or terminal_dt > cutoff:
                    continue
                try:
                    owner = str(row.get("user_id") or "").strip() or "unknown"
                except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                    owner = "unknown"
                removed = self._remove_run_artifacts(rid, owner=owner, decrement_usage=True)
                removed_runs += int(removed.get("removed_runs", 0) or 0)
                removed_files += int(removed.get("removed_files", 0) or 0)
                removed_bytes += int(removed.get("removed_bytes", 0) or 0)
            if len(rows) < page_size:
                break
            offset += page_size
        summary = {
            "removed_runs": int(removed_runs),
            "removed_files": int(removed_files),
            "removed_bytes": int(removed_bytes),
        }
        try:
            from tldw_Server_API.app.core.Metrics import increment_counter as _inc
            from tldw_Server_API.app.core.Metrics import observe_histogram as _obs

            labels = {"mode": ("force" if force else "opportunistic")}
            _inc("sandbox_artifact_janitor_runs_total", labels=labels)
            if removed_runs > 0:
                _inc("sandbox_artifact_janitor_removed_runs_total", value=float(removed_runs), labels=labels)
            if removed_files > 0:
                _inc("sandbox_artifact_janitor_removed_files_total", value=float(removed_files), labels=labels)
            if removed_bytes > 0:
                _inc("sandbox_artifact_janitor_removed_bytes_total", value=float(removed_bytes), labels=labels)
            _obs(
                "sandbox_artifact_janitor_duration_ms",
                value=max(0.0, (time.monotonic() - start_monotonic) * 1000.0),
                labels=labels,
            )
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
            pass
        return summary

    def reconcile_artifact_usage(self) -> dict[str, int]:
        start_monotonic = time.monotonic()
        disk_by_user: dict[str, int] = {}
        root = self._artifact_root()
        if root.exists():
            for user_dir in root.iterdir():
                try:
                    if not user_dir.is_dir():
                        continue
                    runs_root = user_dir / "runs"
                    if not runs_root.exists() or not runs_root.is_dir():
                        continue
                    user_total = 0
                    for run_dir in runs_root.iterdir():
                        if not run_dir.is_dir():
                            continue
                        art_dir = run_dir / "artifacts"
                        if not art_dir.exists() or not art_dir.is_dir():
                            continue
                        for walk_root, _dirs, files in os.walk(art_dir):
                            for fn in files:
                                full = Path(walk_root) / fn
                                with contextlib.suppress(_SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS):
                                    user_total += int(full.stat().st_size)
                    disk_by_user[str(user_dir.name)] = int(user_total)
                except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                    continue

        usage_users: set[str] = set(disk_by_user.keys())
        offset = 0
        page_size = 500
        while True:
            try:
                rows = self._store.list_usage(limit=page_size, offset=offset, sort_desc=False)
            except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                rows = []
            if not rows:
                break
            for row in rows:
                try:
                    uid = str(row.get("user_id") or "").strip()
                    if uid:
                        usage_users.add(uid)
                except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                    continue
            if len(rows) < page_size:
                break
            offset += page_size

        corrected_users = 0
        corrected_bytes = 0
        scanned_users = 0
        for uid in sorted(usage_users):
            scanned_users += 1
            disk_bytes = int(disk_by_user.get(uid, 0))
            try:
                stored_bytes = int(self._store.get_user_artifact_bytes(uid))
            except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                stored_bytes = 0
            delta = int(disk_bytes - stored_bytes)
            if delta != 0:
                with contextlib.suppress(_SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS):
                    self._store.increment_user_artifact_bytes(uid, delta)
                corrected_users += 1
                corrected_bytes += abs(delta)

        summary = {
            "scanned_users": int(scanned_users),
            "corrected_users": int(corrected_users),
            "corrected_bytes": int(corrected_bytes),
            "disk_users": int(len(disk_by_user)),
        }
        try:
            from tldw_Server_API.app.core.Metrics import increment_counter as _inc
            from tldw_Server_API.app.core.Metrics import observe_histogram as _obs

            _inc("sandbox_artifact_reconcile_runs_total")
            if corrected_users > 0:
                _inc("sandbox_artifact_reconcile_corrected_users_total", value=float(corrected_users))
            if corrected_bytes > 0:
                _inc("sandbox_artifact_reconcile_corrected_bytes_total", value=float(corrected_bytes))
            _obs(
                "sandbox_artifact_reconcile_duration_ms",
                value=max(0.0, (time.monotonic() - start_monotonic) * 1000.0),
            )
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
            pass
        return summary

    def _artifact_dir(self, user_id: str, run_id: str) -> Path:
        return self._artifact_root() / user_id / "runs" / run_id / "artifacts"

    def _safe_rel(self, p: str) -> str:
        p = p.replace("\\", "/").lstrip("/")
        # prevent path traversal
        parts: list[str] = []
        for comp in p.split('/'):
            if comp in ("", "."):
                continue
            if comp == "..":
                parts.append("_")
            else:
                parts.append(comp)
        return "/".join(parts)

    def store_artifacts(self, run_id: str, items: dict[str, bytes]) -> None:
        # Enforce caps and persist to filesystem under run's artifacts directory
        owner = None
        try:
            owner = self._store.get_run_owner(run_id)
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
            owner = None
        owner = owner or "unknown"
        # Caps (bytes)
        try:
            cap_run = int(getattr(app_settings, "SANDBOX_MAX_ARTIFACT_BYTES_PER_RUN_MB", 32)) * 1024 * 1024
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
            cap_run = 32 * 1024 * 1024
        try:
            cap_user = int(getattr(app_settings, "SANDBOX_MAX_ARTIFACT_BYTES_PER_USER_MB", 128)) * 1024 * 1024
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
            cap_user = 128 * 1024 * 1024

        selected: dict[str, bytes] = {}
        total_run = 0
        # Deterministic order
        for path in sorted(items.keys()):
            data = items[path]
            sz = len(data)
            if total_run + sz > cap_run:
                break
            try:
                admitted = bool(self._store.try_reserve_user_artifact_bytes(owner, sz, cap_user))
            except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                admitted = False
            if not admitted:
                break
            selected[path] = data
            total_run += sz

        # Persist selected to FS and memory map for backward compatibility
        art_dir = self._artifact_dir(owner, run_id)
        with contextlib.suppress(_SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS):
            art_dir.mkdir(parents=True, exist_ok=True)
        persisted: dict[str, bytes] = {}
        for path, data in selected.items():
            rel = self._safe_rel(path)
            full = art_dir / rel
            try:
                full.parent.mkdir(parents=True, exist_ok=True)
                with open(full, "wb") as f:
                    f.write(data)
            except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"Failed to persist artifact {rel}: {e}")
                with contextlib.suppress(_SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS):
                    self._store.increment_user_artifact_bytes(owner, -len(data))
                continue
            persisted[path] = data

        with self._lock:
            self._artifacts[run_id] = persisted

    def list_artifacts(self, run_id: str) -> dict[str, int]:
        self._maybe_prune_expired_artifacts()
        # Try filesystem, fallback to memory
        owner = None
        try:
            owner = self._store.get_run_owner(run_id)
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
            owner = None
        art_dir = self._artifact_dir((owner or "unknown"), run_id)
        result: dict[str, int] = {}
        if art_dir.exists():
            for root, _dirs, files in os.walk(art_dir):
                for fn in files:
                    full = Path(root) / fn
                    rel = str(full.relative_to(art_dir)).replace(os.sep, "/")
                    try:
                        result[rel] = full.stat().st_size
                    except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                        result[rel] = 0
            if result:
                return result
        with self._lock:
            mapping = self._artifacts.get(run_id) or {}
            return {k: len(v) for k, v in mapping.items()}

    def get_artifact(self, run_id: str, path: str) -> bytes | None:
        self._maybe_prune_expired_artifacts()
        owner = None
        try:
            owner = self._store.get_run_owner(run_id)
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
            owner = None
        art_dir = self._artifact_dir((owner or "unknown"), run_id)
        if path:
            rel = self._safe_rel(path)
            full = art_dir / rel
            try:
                if full.exists():
                    with open(full, "rb") as f:
                        return f.read()
            except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                pass
        with self._lock:
            mapping = self._artifacts.get(run_id) or {}
            return mapping.get(path)

    def get_artifact_path(self, run_id: str, path: str) -> Path | None:
        self._maybe_prune_expired_artifacts()
        owner = None
        try:
            owner = self._store.get_run_owner(run_id)
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
            owner = None
        art_dir = self._artifact_dir((owner or "unknown"), run_id)
        rel = self._safe_rel(path)
        full = art_dir / rel
        try:
            if full.exists() and full.is_file():
                return full
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
            return None
        return None

    # -----------------
    # Workspaces
    # -----------------
    def _workspace_path(self, user_id: Any, session_id: str) -> Path:
        try:
            root = getattr(app_settings, "SANDBOX_ROOT_DIR", None)
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
            root = None
        if not root:
            try:
                proj = getattr(app_settings, "PROJECT_ROOT", ".")
            except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
                proj = "."
            root = Path(str(proj)) / "tmp_dir" / "sandbox"
        return Path(str(root)) / str(user_id) / "sessions" / str(session_id) / "workspace"

    def _ensure_workspace(self, user_id: Any, session_id: str) -> str:
        ws = self._workspace_path(user_id, session_id)
        ws.mkdir(parents=True, exist_ok=True)
        try:
            bind_workspace = is_truthy(
                str(
                    os.getenv("SANDBOX_DOCKER_BIND_WORKSPACE")
                    or getattr(app_settings, "SANDBOX_DOCKER_BIND_WORKSPACE", "")
                ).strip().lower()
            )
            if bind_workspace:
                os.chmod(ws, _OWNER_ONLY_DIR_MODE)
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS:
            pass
        with self._lock:
            self._session_roots[session_id] = str(ws)
        return str(ws)

    def get_session_workspace_path(self, session_id: str, *, allow_cache_on_store_error: bool = True) -> str | None:
        sid = str(session_id or "").strip()
        if not sid:
            return None
        with contextlib.suppress(_SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS):
            self._prune_expired_sessions()
        try:
            row = self._store.get_session(sid)
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"store.get_session workspace lookup failed: {e}")
            if not allow_cache_on_store_error:
                return None
            with self._lock:
                ws = self._session_roots.get(sid)
                return ws
        if not isinstance(row, dict):
            self._drop_cached_session(sid)
            return None
        ws_path = row.get("workspace_path")
        if not ws_path:
            owner = row.get("user_id")
            if owner:
                ws_path = str(self._workspace_path(owner, sid))
        if not ws_path:
            self._drop_cached_session(sid)
            return None
        ws_str = str(ws_path)
        with contextlib.suppress(_SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS):
            Path(ws_str).mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._session_roots[sid] = ws_str
        return ws_str

    def get_session_workspace_path_for_user(self, session_id: str, user_id: str) -> str | None:
        sid = str(session_id or "").strip()
        owner_key = str(user_id or "").strip()
        if not sid or not owner_key:
            return None
        owner = self.get_session_owner(sid)
        if str(owner or "").strip() != owner_key:
            return None
        return self.get_session_workspace_path(sid)

    def list_workspace_paths_for_user_workspace(self, *, user_id: str, workspace_id: str) -> list[str]:
        owner_key = str(user_id or "").strip()
        workspace_key = str(workspace_id or "").strip()
        if not owner_key or not workspace_key:
            return []
        with contextlib.suppress(_SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS):
            self._prune_expired_sessions()
        try:
            return self._store.list_workspace_paths_for_user_workspace(
                user_id=owner_key,
                workspace_id=workspace_key,
            )
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(
                "store.list_workspace_paths_for_user_workspace failed: {}",
                exc,
            )
            return []

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
        self._store.put_vz_session_control(
            session_id=str(session_id),
            runtime=str(runtime),
            vm_id=str(vm_id),
            template_id=(str(template_id) if template_id is not None else None),
            workspace_mount=(str(workspace_mount) if workspace_mount is not None else None),
            agent_ready=bool(agent_ready),
            helper_instance_id=_coerce_optional_nonempty_string(helper_instance_id),
            helper_started_at=_coerce_optional_nonempty_string(helper_started_at),
        )

    def get_vz_session_control(self, session_id: str) -> dict[str, Any] | None:
        return self._store.get_vz_session_control(str(session_id))

    def delete_vz_session_control(self, session_id: str) -> bool:
        return bool(self._store.delete_vz_session_control(str(session_id)))

    def list_vz_session_controls(self) -> list[dict[str, Any]]:
        rows = self._store.list_vz_session_controls()
        return [dict(row) for row in rows if isinstance(row, dict)]

    # -----------------
    # Admin listing helpers
    # -----------------
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
        try:
            return self._store.list_runs(
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
                limit=limit,
                offset=offset,
                sort_desc=sort_desc,
            )
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"store.list_runs failed: {e}")
            return []

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
        try:
            return int(self._store.count_runs(
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
            ))
        except _SANDBOX_ORCH_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"store.count_runs failed: {e}")
            return 0


class QueueFull(Exception):
    def __init__(
        self,
        retry_after: int = 30,
        reason: str = "queue_full",
        quota_scope: str | None = None,
        limit: int | None = None,
    ) -> None:
        super().__init__("queue_full")
        self.retry_after = int(retry_after)
        self.reason = str(reason or "queue_full")
        self.quota_scope = str(quota_scope) if quota_scope else None
        self.limit = int(limit) if limit is not None else None
