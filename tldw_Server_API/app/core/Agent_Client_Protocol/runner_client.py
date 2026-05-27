from __future__ import annotations

import asyncio
import fnmatch
import json
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from tldw_Server_API.app.core.Agent_Client_Protocol.config import (
    ACPRunnerConfig,
    load_acp_runner_config,
)
from tldw_Server_API.app.core.Agent_Client_Protocol.permission_tiers import determine_permission_tier
from tldw_Server_API.app.core.Agent_Client_Protocol.stdio_client import (
    ACPMessage,
    ACPResponseError,
    ACPStdioClient,
)
from tldw_Server_API.app.services.acp_runtime_policy_service import (
    ACPRuntimePolicyService,
    ACPRuntimePolicySnapshot,
)

# Permission timeout in seconds (5 minutes)
PERMISSION_TIMEOUT_SECONDS = 300

_ACP_GOVERNANCE_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)


def _policy_match_patterns(policy_document: dict[str, Any], *keys: str) -> list[str]:
    patterns: list[str] = []
    for key in keys:
        value = policy_document.get(key)
        if isinstance(value, str):
            candidate = value.strip()
            if candidate:
                patterns.append(candidate)
        elif isinstance(value, (list, tuple, set)):
            for entry in value:
                candidate = str(entry or "").strip()
                if candidate:
                    patterns.append(candidate)
    seen: set[str] = set()
    out: list[str] = []
    for pattern in patterns:
        if pattern in seen:
            continue
        seen.add(pattern)
        out.append(pattern)
    return out


def _tool_matches_policy(tool_name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(tool_name, pattern) for pattern in patterns)


def _resolve_runtime_permission_outcome(
    snapshot: ACPRuntimePolicySnapshot | None,
    tool_name: str,
) -> dict[str, Any]:
    if snapshot is None:
        return {
            "action": "deny",
            "deny_reason": "policy_snapshot_unavailable",
            "approval_requirement": "denied",
            "policy_snapshot_fingerprint": None,
            "provenance_summary": {},
        }

    policy_document = dict(snapshot.resolved_policy_document or {})
    denied_patterns = _policy_match_patterns(policy_document, "denied_tools")
    if denied_patterns and _tool_matches_policy(tool_name, denied_patterns):
        return {
            "action": "deny",
            "deny_reason": "tool_denied_by_policy",
            "approval_requirement": "denied",
            "policy_snapshot_fingerprint": snapshot.policy_snapshot_fingerprint,
            "provenance_summary": dict(snapshot.policy_provenance_summary or {}),
        }

    allowed_patterns = _policy_match_patterns(
        policy_document,
        "allowed_tools",
        "tool_names",
        "tool_patterns",
    )
    if not allowed_patterns or not _tool_matches_policy(tool_name, allowed_patterns):
        return {
            "action": "deny",
            "deny_reason": "tool_not_allowed_by_policy",
            "approval_requirement": "denied",
            "policy_snapshot_fingerprint": snapshot.policy_snapshot_fingerprint,
            "provenance_summary": dict(snapshot.policy_provenance_summary or {}),
        }

    # Check tool_tier_overrides -- Bash prefix patterns like Bash(git:*) -> auto
    overrides = policy_document.get("tool_tier_overrides", {})
    if isinstance(overrides, dict):
        for pattern, tier in overrides.items():
            if fnmatch.fnmatch(tool_name, pattern):
                tier_lower = str(tier or "").strip().lower()
                if tier_lower == "auto":
                    return {
                        "action": "approve",
                        "approval_requirement": "allow",
                        "policy_snapshot_fingerprint": snapshot.policy_snapshot_fingerprint,
                        "provenance_summary": dict(snapshot.policy_provenance_summary or {}),
                    }
                # Any non-auto tier (individual, batch, etc.) requires approval
                return {
                    "action": "prompt",
                    "approval_requirement": "approval_required",
                    "governance_reason": f"tool_tier_override_{tier_lower}",
                    "policy_snapshot_fingerprint": snapshot.policy_snapshot_fingerprint,
                    "provenance_summary": dict(snapshot.policy_provenance_summary or {}),
                }

    approval_mode = str(
        (snapshot.approval_summary or {}).get("mode")
        or policy_document.get("approval_mode")
        or ""
    ).strip().lower()
    if approval_mode in {"require_approval", "approval_required", "ask", "ask_on_broaden"}:
        return {
            "action": "prompt",
            "approval_requirement": "approval_required",
            "governance_reason": "policy_approval_required",
            "policy_snapshot_fingerprint": snapshot.policy_snapshot_fingerprint,
            "provenance_summary": dict(snapshot.policy_provenance_summary or {}),
        }

    return {
        "action": "approve",
        "approval_requirement": "allow",
        "policy_snapshot_fingerprint": snapshot.policy_snapshot_fingerprint,
        "provenance_summary": dict(snapshot.policy_provenance_summary or {}),
    }


@dataclass
class PendingPermission:
    """Tracks a pending permission request."""
    request_id: str
    session_id: str
    tool_name: str
    tool_arguments: dict[str, Any]
    acp_message_id: Any  # The original ACP message ID for responding
    approval_requirement: str | None = None
    governance_reason: str | None = None
    deny_reason: str | None = None
    provenance_summary: dict[str, Any] = field(default_factory=dict)
    runtime_narrowing_reason: str | None = None
    policy_snapshot_fingerprint: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    # Future is created when permission request is processed, not at dataclass init
    # This avoids the deprecated asyncio.get_event_loop() call
    future: asyncio.Future | None = field(default=None)


@dataclass
class SessionWebSocketRegistry:
    """Tracks WebSocket connections and state per session."""
    session_id: str
    websockets: set[Callable[[dict[str, Any]], Coroutine[Any, Any, None]]] = field(default_factory=set)
    pending_permissions: dict[str, PendingPermission] = field(default_factory=dict)
    # Tiers that are batch-approved for this session
    batch_approved_tiers: set[str] = field(default_factory=set)


def _pending_permission_to_metadata(pending: PendingPermission | None) -> dict[str, Any]:
    if pending is None:
        return {}
    return {
        "tool_name": pending.tool_name,
        "approval_requirement": pending.approval_requirement,
        "governance_reason": pending.governance_reason,
        "deny_reason": pending.deny_reason,
        "runtime_narrowing_reason": pending.runtime_narrowing_reason,
        "policy_snapshot_fingerprint": pending.policy_snapshot_fingerprint,
    }


class ACPRuntimePolicySupportMixin:
    """Shared runtime policy snapshot caching and pending-permission helpers."""

    _runtime_policy_service: ACPRuntimePolicyService
    _runtime_policy_snapshots: dict[str, ACPRuntimePolicySnapshot]
    _runtime_policy_refresh_tasks: dict[str, asyncio.Task[ACPRuntimePolicySnapshot | None]]
    _runtime_policy_refresh_lock: asyncio.Lock
    _ws_registry: dict[str, SessionWebSocketRegistry]
    _runtime_policy_log_label: str

    async def _get_acp_session_store(self) -> Any:
        from tldw_Server_API.app.services.admin_acp_sessions_service import get_acp_session_store

        return await get_acp_session_store()

    async def _refresh_runtime_policy_snapshot(
        self,
        session_id: str,
        *,
        session_store: Any | None = None,
        session_record: Any | None = None,
    ) -> ACPRuntimePolicySnapshot | None:
        label = str(getattr(self, "_runtime_policy_log_label", "ACP runtime policy"))

        async def _build() -> ACPRuntimePolicySnapshot | None:
            store = session_store or await self._get_acp_session_store()
            record = session_record or await store.get_session(session_id)
            if record is None:
                return None
            try:
                snapshot = await self._runtime_policy_service.build_snapshot(
                    session_record=record,
                    user_id=int(getattr(record, "user_id")),
                )
                await self._runtime_policy_service.persist_snapshot(
                    session_store=store,
                    snapshot=snapshot,
                )
                self._runtime_policy_snapshots[str(session_id)] = snapshot
                return snapshot
            except Exception as exc:
                logger.warning("{} refresh failed for session {}: {}", label, session_id, exc)
                self._runtime_policy_snapshots.pop(str(session_id), None)
                try:
                    await store.update_policy_snapshot_state(
                        session_id,
                        policy_snapshot_version=None,
                        policy_snapshot_fingerprint=None,
                        policy_snapshot_refreshed_at=None,
                        policy_summary=None,
                        policy_provenance_summary=None,
                        policy_refresh_error=str(exc),
                    )
                except Exception as persist_exc:
                    logger.warning(
                        "Failed to persist {} refresh error for session {}: {}",
                        label.lower(),
                        session_id,
                        persist_exc,
                    )
                return None

        async with self._runtime_policy_refresh_lock:
            refresh_task = self._runtime_policy_refresh_tasks.get(str(session_id))
            if refresh_task is None or refresh_task.done():
                refresh_task = asyncio.create_task(_build())
                self._runtime_policy_refresh_tasks[str(session_id)] = refresh_task
        try:
            return await refresh_task
        finally:
            async with self._runtime_policy_refresh_lock:
                current = self._runtime_policy_refresh_tasks.get(str(session_id))
                if current is refresh_task and refresh_task.done():
                    self._runtime_policy_refresh_tasks.pop(str(session_id), None)

    async def _get_runtime_policy_snapshot(
        self,
        session_id: str,
        *,
        force_refresh: bool = False,
    ) -> ACPRuntimePolicySnapshot | None:
        store = await self._get_acp_session_store()
        record = await store.get_session(session_id)
        if record is None:
            return None

        cached = self._runtime_policy_snapshots.get(str(session_id))
        persisted_fingerprint = getattr(record, "policy_snapshot_fingerprint", None)
        if (
            not force_refresh
            and cached is not None
            and (not persisted_fingerprint or persisted_fingerprint == cached.policy_snapshot_fingerprint)
        ):
            return cached

        return await self._refresh_runtime_policy_snapshot(
            session_id,
            session_store=store,
            session_record=record,
        )

    def _clear_runtime_policy_session_state(self, session_id: str) -> None:
        self._runtime_policy_snapshots.pop(str(session_id), None)
        refresh_task = self._runtime_policy_refresh_tasks.pop(str(session_id), None)
        if refresh_task is not None and not refresh_task.done():
            refresh_task.cancel()

    def _should_force_runtime_policy_refresh(self, *, tier: str) -> bool:
        return str(tier or "").strip().lower() != "auto"

    def get_pending_permission_metadata(self, session_id: str, request_id: str) -> dict[str, Any]:
        registry = self._ws_registry.get(str(session_id))
        if registry is None:
            return {}
        pending = registry.pending_permissions.get(str(request_id))
        return _pending_permission_to_metadata(pending)


class ACPGovernanceDeniedError(ACPResponseError):
    """Raised when ACP governance blocks prompt execution."""

    def __init__(
        self,
        message: str = "Prompt blocked by governance policy",
        *,
        governance: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.governance = governance or {}


class ACPGovernanceCoordinator:
    """Shared governance checks and approval outcome merge logic for ACP."""

    def __init__(self) -> None:
        self._service: Any | None = None
        self._store: Any | None = None
        self._lock = asyncio.Lock()

    @staticmethod
    def _decision_action(decision: dict[str, Any] | None) -> str:
        if not isinstance(decision, dict):
            return ""
        action = str(decision.get("action") or decision.get("status") or "").strip().lower()
        return action

    @staticmethod
    def _decision_rollout_mode(decision: dict[str, Any] | None) -> str:
        """Return normalized rollout mode from governance decision payload."""
        if not isinstance(decision, dict):
            return "enforce"
        candidate = str(decision.get("rollout_mode") or "").strip().lower()
        return candidate if candidate in {"off", "shadow", "enforce"} else "enforce"

    @classmethod
    def is_denied(cls, decision: dict[str, Any] | None) -> bool:
        return cls._decision_action(decision) == "deny"

    @classmethod
    def is_denied_with_enforcement(cls, decision: dict[str, Any] | None) -> bool:
        """Return True only when governance denies and rollout mode is enforce."""
        return cls.is_denied(decision) and cls._decision_rollout_mode(decision) == "enforce"

    @classmethod
    def resolve_permission_outcome(
        cls,
        *,
        tier: str,
        batch_tier_approved: bool,
        governance: dict[str, Any] | None,
    ) -> str:
        """Return one of: approve | deny | prompt."""
        action = cls._decision_action(governance)
        rollout_mode = cls._decision_rollout_mode(governance)

        if action == "deny" and rollout_mode == "enforce":
            return "deny"
        if action == "require_approval" and rollout_mode in {"shadow", "enforce"}:
            return "prompt"
        if tier == "auto":
            return "approve"
        if tier == "batch" and batch_tier_approved:
            return "approve"
        return "prompt"

    @classmethod
    def _serialize_decision(cls, decision: Any) -> dict[str, Any]:
        if decision is None:
            return {}
        if isinstance(decision, dict):
            return {str(k): v for k, v in decision.items()}
        dump = getattr(decision, "model_dump", None)
        if callable(dump):
            try:
                dumped = dump()
                if isinstance(dumped, dict):
                    return {str(k): v for k, v in dumped.items()}
            except _ACP_GOVERNANCE_NONCRITICAL_EXCEPTIONS:
                pass
        payload: dict[str, Any] = {}
        for key in ("action", "status", "category", "category_source", "fallback_reason", "matched_rules"):
            value = getattr(decision, key, None)
            if value is not None:
                payload[key] = value
        return payload

    async def _ensure_service(self) -> Any | None:
        if self._service is not None:
            return self._service

        async with self._lock:
            if self._service is not None:
                return self._service
            try:
                from tldw_Server_API.app.core.Governance.service import GovernanceService
                from tldw_Server_API.app.core.Governance.store import GovernanceStore
                from tldw_Server_API.app.core.MCP_unified.config import get_config
            except _ACP_GOVERNANCE_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug("ACP governance coordinator unavailable (import failure): {}", exc)
                return None

            try:
                cfg = get_config()
                configured_path = getattr(cfg, "governance_db_path", None)
                sqlite_path = str(configured_path or "Databases/governance.db")
                db_path = Path(sqlite_path).expanduser()
                db_path.parent.mkdir(parents=True, exist_ok=True)

                self._store = GovernanceStore(sqlite_path=str(db_path))
                await self._store.ensure_schema()
                self._service = GovernanceService(store=self._store)
                return self._service
            except _ACP_GOVERNANCE_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug("ACP governance coordinator disabled (service init failure): {}", exc)
                self._service = None
                self._store = None
                return None

    async def validate_change(
        self,
        *,
        surface: str,
        summary: str,
        category: str,
        metadata: dict[str, Any],
    ) -> dict[str, Any] | None:
        service = await self._ensure_service()
        if service is None:
            return None

        try:
            decision = await service.validate_change(
                surface=surface,
                summary=summary,
                category=category,
                metadata=metadata,
            )
            return self._serialize_decision(decision)
        except _ACP_GOVERNANCE_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("ACP governance validation failed open: {}", exc)
            return None


def _merge_runtime_and_governance_permission_outcome(
    *,
    tier: str,
    batch_tier_approved: bool,
    runtime_outcome: dict[str, Any],
    governance: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(runtime_outcome or {})
    runtime_action = str(merged.get("action") or "deny").strip().lower() or "deny"
    if runtime_action == "deny":
        merged["approval_requirement"] = "denied"
        return merged

    approval_outcome = ACPGovernanceCoordinator.resolve_permission_outcome(
        tier=tier,
        batch_tier_approved=batch_tier_approved,
        governance=governance,
    )
    if approval_outcome == "deny":
        merged["action"] = "deny"
        merged["approval_requirement"] = "denied"
        merged["deny_reason"] = "governance_denied"
        merged["governance_reason"] = "governance_denied"
        return merged

    if runtime_action == "prompt" or approval_outcome == "prompt":
        merged["action"] = "prompt"
        merged["approval_requirement"] = "approval_required"
        if approval_outcome == "prompt" and runtime_action != "prompt":
            merged["governance_reason"] = "governance_approval_required"
        return merged

    merged["action"] = "approve"
    merged["approval_requirement"] = "allow"
    merged.pop("deny_reason", None)
    return merged


class ACPRunnerClient(ACPRuntimePolicySupportMixin):
    def __init__(self, config: ACPRunnerConfig) -> None:
        self.config = config
        self._client = ACPStdioClient(
            command=config.command,
            args=config.args,
            env=config.env,
            cwd=config.cwd,
        )
        self._client.set_notification_handler(self._handle_notification)
        self._client.set_request_handler(self._handle_request)
        self._updates: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        self._session_owners: dict[str, int] = {}
        self._agent_capabilities: dict[str, Any] = {}
        # WebSocket registry per session
        self._ws_registry: dict[str, SessionWebSocketRegistry] = {}
        self._ws_registry_lock = asyncio.Lock()
        self._governance = ACPGovernanceCoordinator()
        self._runtime_policy_log_label = "ACP runtime policy"
        self._runtime_policy_service = ACPRuntimePolicyService()
        self._runtime_policy_snapshots: dict[str, ACPRuntimePolicySnapshot] = {}
        self._runtime_policy_refresh_tasks: dict[str, asyncio.Task[ACPRuntimePolicySnapshot | None]] = {}
        self._runtime_policy_refresh_lock = asyncio.Lock()

    @classmethod
    def from_config(cls) -> ACPRunnerClient:
        return cls(load_acp_runner_config())

    @property
    def agent_capabilities(self) -> dict[str, Any]:
        return self._agent_capabilities

    @property
    def is_running(self) -> bool:
        return self._client.is_running

    async def start(self) -> None:
        if not self._client.is_running:
            await self._client.start()
        if self.config.startup_timeout_sec > 0:
            await asyncio.wait_for(self.initialize(), timeout=self.config.startup_timeout_sec)
        else:
            await self.initialize()

    async def initialize(self) -> dict[str, Any]:
        response = await self._client.call(
            "initialize",
            {
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": {"readTextFile": False, "writeTextFile": False},
                    "terminal": False,
                },
                "clientInfo": {
                    "name": "tldw-server",
                    "title": "TLDW Server",
                    "version": "0.1.0",
                },
            },
        )
        result = response.result or {}
        self._agent_capabilities = result.get("agentCapabilities", {}) or {}
        return result

    async def create_session(
        self,
        cwd: str,
        mcp_servers: list[dict[str, Any]] | None = None,
        agent_type: str | None = None,
        user_id: int | None = None,
        session_env: dict[str, str] | None = None,
    ) -> str:
        params: dict[str, Any] = {
            "cwd": cwd,
            "mcpServers": mcp_servers if mcp_servers is not None else [],
        }
        if agent_type:
            params["agentType"] = agent_type
        if session_env:
            params["env"] = dict(session_env)
        try:
            response = await self._client.call("session/new", params)
        except ACPResponseError as exc:
            if not agent_type or "invalid params" not in str(exc).lower():
                raise
            fallback_params = dict(params)
            fallback_params.pop("agentType", None)
            logger.debug(
                "ACP session/new rejected agentType for this runner; retrying without agentType"
            )
            response = await self._client.call("session/new", fallback_params)
        result = response.result or {}
        session_id = result.get("sessionId")
        if not session_id:
            raise ACPResponseError("Missing sessionId in response")
        if user_id is not None:
            self._session_owners[str(session_id)] = int(user_id)
        return session_id

    async def list_agents(self) -> dict[str, Any]:
        response = await self._client.call("agent/list", {})
        return response.result or {}

    @staticmethod
    def _safe_json_summary(payload: Any, *, max_chars: int = 1200) -> str:
        try:
            rendered = json.dumps(payload or {}, sort_keys=True, default=str)
        except _ACP_GOVERNANCE_NONCRITICAL_EXCEPTIONS:
            rendered = str(payload)
        if len(rendered) > max_chars:
            return rendered[:max_chars]
        return rendered

    @staticmethod
    def _resolve_tool_category(tool_name: str) -> str:
        if isinstance(tool_name, str) and "." in tool_name:
            prefix = tool_name.split(".", 1)[0].strip().lower()
            if prefix:
                return prefix
        return "acp"

    @staticmethod
    def _resolve_governance_rollout_mode(metadata: dict[str, Any] | None = None) -> str:
        """Resolve ACP rollout mode with optional per-request metadata override."""
        raw_mode = None
        if isinstance(metadata, dict):
            raw_mode = metadata.get("governance_rollout_mode")
        try:
            from tldw_Server_API.app.core import config as app_config

            return app_config.resolve_governance_rollout_mode(
                str(raw_mode) if raw_mode is not None else None
            )
        except _ACP_GOVERNANCE_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("ACP governance rollout mode resolution failed; defaulting to off: {}", exc)
            candidate = str(raw_mode or "").strip().lower()
            return candidate if candidate in {"off", "shadow", "enforce"} else "off"

    @staticmethod
    def _record_governance_check(
        *,
        surface: str,
        category: str,
        status: str,
        rollout_mode: str,
    ) -> None:
        """Emit ACP governance metric via the shared MCP metrics collector."""
        try:
            from tldw_Server_API.app.core.MCP_unified.monitoring.metrics import get_metrics_collector

            get_metrics_collector().record_governance_check(
                surface=surface,
                category=category,
                status=status,
                rollout_mode=rollout_mode,
            )
        except _ACP_GOVERNANCE_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("ACP governance metric emit failed open: {}", exc)

    def _build_governance_metadata(
        self,
        session_id: str,
        *,
        metadata: dict[str, Any] | None = None,
        user_id: int | None = None,
    ) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        if isinstance(metadata, dict):
            merged.update(metadata)
        merged.setdefault("session_id", str(session_id))
        owner = self._session_owners.get(str(session_id))
        if owner is not None:
            merged.setdefault("user_id", int(owner))
        if user_id is not None:
            merged.setdefault("user_id", int(user_id))
        return merged

    async def check_prompt_governance(
        self,
        session_id: str,
        prompt: list[dict[str, Any]],
        *,
        user_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        merged_metadata = self._build_governance_metadata(
            session_id,
            metadata=metadata,
            user_id=user_id,
        )
        rollout_mode = self._resolve_governance_rollout_mode(merged_metadata)
        if rollout_mode == "off":
            self._record_governance_check(
                surface="acp_prompt",
                category="acp",
                status="unknown",
                rollout_mode=rollout_mode,
            )
            return {
                "action": "allow",
                "status": "allow",
                "category": "acp",
                "rollout_mode": rollout_mode,
            }

        decision = await self._governance.validate_change(
            surface="acp_prompt",
            summary=f"session={session_id}; prompt={self._safe_json_summary(prompt)}",
            category="acp",
            metadata=merged_metadata,
        )
        payload = dict(decision or {})
        payload.setdefault("rollout_mode", rollout_mode)
        status = str(payload.get("action") or payload.get("status") or "unknown").strip().lower() or "unknown"
        self._record_governance_check(
            surface="acp_prompt",
            category="acp",
            status=status,
            rollout_mode=rollout_mode,
        )
        return payload

    async def check_permission_governance(
        self,
        session_id: str,
        tool_name: str,
        tool_arguments: dict[str, Any],
        *,
        tier: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        merged_metadata = self._build_governance_metadata(
            session_id,
            metadata=metadata,
        )
        merged_metadata.setdefault("permission_tier", tier)
        merged_metadata.setdefault("tool_name", tool_name)
        category = self._resolve_tool_category(tool_name)
        rollout_mode = self._resolve_governance_rollout_mode(merged_metadata)
        if rollout_mode == "off":
            self._record_governance_check(
                surface="acp_permission",
                category=category,
                status="unknown",
                rollout_mode=rollout_mode,
            )
            return {
                "action": "allow",
                "status": "allow",
                "category": category,
                "rollout_mode": rollout_mode,
            }

        decision = await self._governance.validate_change(
            surface="acp_permission",
            summary=(
                f"session={session_id}; tier={tier}; tool={tool_name}; "
                f"input={self._safe_json_summary(tool_arguments)}"
            ),
            category=category,
            metadata=merged_metadata,
        )
        payload = dict(decision or {})
        payload.setdefault("rollout_mode", rollout_mode)
        status = str(payload.get("action") or payload.get("status") or "unknown").strip().lower() or "unknown"
        self._record_governance_check(
            surface="acp_permission",
            category=category,
            status=status,
            rollout_mode=rollout_mode,
        )
        return payload

    async def prompt(self, session_id: str, prompt: list[dict[str, Any]]) -> dict[str, Any]:
        governance = await self.check_prompt_governance(session_id, prompt)
        if self._governance.is_denied_with_enforcement(governance):
            raise ACPGovernanceDeniedError(governance=governance or {})

        response = await self._client.call(
            "session/prompt",
            {
                "sessionId": session_id,
                "prompt": prompt,
            },
        )
        return response.result or {}

    async def cancel(self, session_id: str) -> None:
        await self._client.notify("session/cancel", {"sessionId": session_id})

    async def close_session(self, session_id: str) -> None:
        await self._client.call("_tldw/session/close", {"sessionId": session_id})
        self._updates.pop(session_id, None)
        self._session_owners.pop(str(session_id), None)
        self._clear_runtime_policy_session_state(session_id)

    def pop_updates(self, session_id: str, limit: int = 100) -> list[dict[str, Any]]:
        updates = []
        queue = self._updates.get(session_id)
        if not queue:
            return updates
        for _ in range(min(limit, len(queue))):
            updates.append(queue.popleft())
        return updates

    async def verify_session_access(self, session_id: str, user_id: int) -> bool:
        """Verify that the given user owns the ACP session.

        Returns False when the session is unknown or ownership does not match.
        """
        owner = self._session_owners.get(str(session_id))
        if owner is None:
            return False
        return int(owner) == int(user_id)

    async def shutdown(self) -> None:
        await self._client.close()

    # -------------------------------------------------------------------------
    # WebSocket Registry Management
    # -------------------------------------------------------------------------

    async def register_websocket(
        self,
        session_id: str,
        send_callback: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
    ) -> None:
        """Register a WebSocket send callback for a session."""
        async with self._ws_registry_lock:
            if session_id not in self._ws_registry:
                self._ws_registry[session_id] = SessionWebSocketRegistry(session_id=session_id)
            self._ws_registry[session_id].websockets.add(send_callback)
            logger.debug("Registered WebSocket for session {}", session_id)

    async def unregister_websocket(
        self,
        session_id: str,
        send_callback: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
    ) -> None:
        """Unregister a WebSocket send callback for a session."""
        async with self._ws_registry_lock:
            if session_id in self._ws_registry:
                self._ws_registry[session_id].websockets.discard(send_callback)
                logger.debug("Unregistered WebSocket for session {}", session_id)

    def has_websocket_connections(self, session_id: str) -> bool:
        """Check if a session has any active WebSocket connections."""
        registry = self._ws_registry.get(session_id)
        return registry is not None and len(registry.websockets) > 0

    async def _broadcast_to_session(self, session_id: str, message: dict[str, Any]) -> None:
        """Broadcast a message to all WebSockets connected to a session."""
        registry = self._ws_registry.get(session_id)
        if not registry:
            return

        failed_callbacks = []
        for callback in list(registry.websockets):
            try:
                await callback(message)
            except Exception as e:
                logger.warning("Failed to send to WebSocket for session {}: {}", session_id, e)
                failed_callbacks.append(callback)

        # Remove failed callbacks
        for callback in failed_callbacks:
            registry.websockets.discard(callback)

    # -------------------------------------------------------------------------
    # Permission Management
    # -------------------------------------------------------------------------

    async def respond_to_permission(
        self,
        session_id: str,
        request_id: str,
        approved: bool,
        batch_approve_tier: str | None = None,
    ) -> bool:
        """Respond to a pending permission request.

        Returns True if the permission was found and responded to.
        """
        registry = self._ws_registry.get(session_id)
        if not registry:
            logger.warning("No registry for session {} when responding to permission", session_id)
            return False

        pending = registry.pending_permissions.pop(request_id, None)
        if not pending:
            logger.warning("Permission request {} not found for session {}", request_id, session_id)
            return False

        # Handle batch approval tier
        if approved and batch_approve_tier:
            registry.batch_approved_tiers.add(batch_approve_tier)
            logger.info("Batch-approved tier {} for session {}", batch_approve_tier, session_id)

        # Resolve the permission future
        outcome = "approved" if approved else "denied"
        if not pending.future.done():
            pending.future.set_result({"outcome": outcome})

        return True

    def _determine_permission_tier(self, tool_name: str) -> str:
        """Determine the permission tier for a tool based on policies then heuristics."""
        return determine_permission_tier(tool_name)

    # -------------------------------------------------------------------------
    # Notification and Request Handlers
    # -------------------------------------------------------------------------

    async def _handle_notification(self, msg: ACPMessage) -> None:
        if msg.method != "session/update":
            return
        params = msg.params or {}
        session_id = params.get("sessionId")
        if not session_id:
            return

        # Queue update for polling clients
        self._updates[session_id].append(params)

        # Broadcast to WebSocket clients
        update_message = {
            "type": "update",
            "session_id": session_id,
            "update_type": params.get("type", "unknown"),
            "data": params,
        }
        await self._broadcast_to_session(session_id, update_message)

    async def _handle_request(self, msg: ACPMessage) -> ACPMessage:
        if msg.method != "session/request_permission":
            return ACPMessage(jsonrpc="2.0", id=msg.id, error={"code": -32601, "message": "method not found"})

        params = msg.params or {}
        session_id = params.get("sessionId", "")
        tool_name = params.get("tool", {}).get("name", "unknown")
        tool_arguments = params.get("tool", {}).get("input", {})

        # Determine permission tier
        tier = self._determine_permission_tier(tool_name)
        registry = self._ws_registry.get(session_id)
        batch_tier_approved = bool(registry and tier in registry.batch_approved_tiers)
        runtime_snapshot = await self._get_runtime_policy_snapshot(
            session_id,
            force_refresh=self._should_force_runtime_policy_refresh(tier=tier),
        )
        runtime_outcome = _resolve_runtime_permission_outcome(runtime_snapshot, tool_name)
        governance = await self.check_permission_governance(
            session_id,
            tool_name,
            tool_arguments,
            tier=tier,
        )
        permission_outcome = _merge_runtime_and_governance_permission_outcome(
            tier=tier,
            batch_tier_approved=batch_tier_approved,
            runtime_outcome=runtime_outcome,
            governance=governance,
        )

        if permission_outcome["action"] == "deny":
            outcome_payload: dict[str, Any] = {
                "outcome": "denied",
                "deny_reason": permission_outcome.get("deny_reason"),
                "policy_snapshot_fingerprint": permission_outcome.get("policy_snapshot_fingerprint"),
                "provenance_summary": permission_outcome.get("provenance_summary"),
            }
            if isinstance(governance, dict) and governance:
                outcome_payload["governance"] = governance
            return ACPMessage(
                jsonrpc="2.0",
                id=msg.id,
                result={"outcome": outcome_payload},
            )

        if permission_outcome["action"] == "approve":
            logger.debug("Auto-approving {} (runtime policy)", tool_name)
            return ACPMessage(
                jsonrpc="2.0",
                id=msg.id,
                result={"outcome": {"outcome": "approved"}},
            )

        # Check if we have WebSocket connections to ask for permission
        if not self.has_websocket_connections(session_id):
            logger.warning("ACP permission request auto-cancelled (no WebSocket connections)")
            return ACPMessage(
                jsonrpc="2.0",
                id=msg.id,
                result={"outcome": {"outcome": "cancelled"}},
            )

        # Create pending permission with future for async response
        request_id = str(uuid.uuid4())
        pending = PendingPermission(
            request_id=request_id,
            session_id=session_id,
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            acp_message_id=msg.id,
            approval_requirement=permission_outcome.get("approval_requirement"),
            governance_reason=permission_outcome.get("governance_reason"),
            deny_reason=permission_outcome.get("deny_reason"),
            provenance_summary=dict(permission_outcome.get("provenance_summary") or {}),
            runtime_narrowing_reason=permission_outcome.get("runtime_narrowing_reason"),
            policy_snapshot_fingerprint=permission_outcome.get("policy_snapshot_fingerprint"),
            future=asyncio.get_running_loop().create_future(),
        )

        # Register pending permission
        async with self._ws_registry_lock:
            if session_id not in self._ws_registry:
                self._ws_registry[session_id] = SessionWebSocketRegistry(session_id=session_id)
            self._ws_registry[session_id].pending_permissions[request_id] = pending

        # Broadcast permission request to WebSocket clients
        permission_message = {
            "type": "permission_request",
            "request_id": request_id,
            "session_id": session_id,
            "tool_name": tool_name,
            "tool_arguments": tool_arguments,
            "tier": tier,
            "approval_requirement": permission_outcome.get("approval_requirement"),
            "governance_reason": permission_outcome.get("governance_reason"),
            "provenance_summary": permission_outcome.get("provenance_summary"),
            "runtime_narrowing_reason": permission_outcome.get("runtime_narrowing_reason"),
            "policy_snapshot_fingerprint": permission_outcome.get("policy_snapshot_fingerprint"),
            "timeout_seconds": PERMISSION_TIMEOUT_SECONDS,
        }
        if isinstance(governance, dict) and governance:
            permission_message["governance"] = governance
        await self._broadcast_to_session(session_id, permission_message)

        # Re-check connections after broadcast - all might have failed
        if not self.has_websocket_connections(session_id):
            logger.warning("ACP permission request auto-cancelled (all WebSocket connections failed during broadcast)")
            # Clean up pending permission
            async with self._ws_registry_lock:
                if session_id in self._ws_registry:
                    self._ws_registry[session_id].pending_permissions.pop(request_id, None)
            return ACPMessage(
                jsonrpc="2.0",
                id=msg.id,
                result={"outcome": {"outcome": "cancelled"}},
            )

        # Wait for permission response with timeout
        try:
            result = await asyncio.wait_for(
                pending.future,
                timeout=PERMISSION_TIMEOUT_SECONDS,
            )
            logger.info("Permission {} for tool {} result: {}", request_id, tool_name, result)
            return ACPMessage(
                jsonrpc="2.0",
                id=msg.id,
                result={"outcome": result},
            )
        except asyncio.TimeoutError:
            logger.warning("Permission request {} timed out for tool {}", request_id, tool_name)
            # Clean up pending permission
            async with self._ws_registry_lock:
                if session_id in self._ws_registry:
                    self._ws_registry[session_id].pending_permissions.pop(request_id, None)
            return ACPMessage(
                jsonrpc="2.0",
                id=msg.id,
                result={"outcome": {"outcome": "cancelled"}},
            )


_runner_client: ACPRunnerClient | None = None
_runner_lock = asyncio.Lock()
_sandbox_client: Any | None = None


async def get_runner_client() -> ACPRunnerClient:
    global _runner_client
    global _sandbox_client
    async with _runner_lock:
        try:
            from tldw_Server_API.app.core.Agent_Client_Protocol.config import load_acp_sandbox_config
            sb_cfg = load_acp_sandbox_config()
        except Exception:
            sb_cfg = None
        if sb_cfg and getattr(sb_cfg, "enabled", False):
            if _sandbox_client is None or not getattr(_sandbox_client, "is_running", True):
                from tldw_Server_API.app.core.Agent_Client_Protocol.sandbox_runner_client import (
                    ACPSandboxRunnerManager,
                )
                _sandbox_client = ACPSandboxRunnerManager(sb_cfg)
                await _sandbox_client.start()
            return _sandbox_client  # type: ignore[return-value]
        if _runner_client is None or not _runner_client.is_running:
            _runner_client = ACPRunnerClient.from_config()
            await _runner_client.start()
        return _runner_client
