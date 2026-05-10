from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import socket
import uuid
from collections import defaultdict, deque
from collections.abc import Coroutine
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from tldw_Server_API.app.api.v1.endpoints import sandbox as sandbox_ep
from tldw_Server_API.app.core.Agent_Client_Protocol.config import ACPSandboxConfig, load_acp_sandbox_config
from tldw_Server_API.app.core.Agent_Client_Protocol.runner_client import (
    ACPGovernanceCoordinator,
    ACPGovernanceDeniedError,
    ACPRuntimePolicySupportMixin,
    PERMISSION_TIMEOUT_SECONDS,
    PendingPermission,
    SessionWebSocketRegistry,
    _merge_runtime_and_governance_permission_outcome,
    _resolve_runtime_permission_outcome,
)
from tldw_Server_API.app.core.Agent_Client_Protocol.permission_tiers import determine_permission_tier
from tldw_Server_API.app.core.Agent_Client_Protocol.stdio_client import ACPMessage, ACPResponseError
from tldw_Server_API.app.core.Agent_Client_Protocol.stream_client import ACPStreamClient
from tldw_Server_API.app.core.config import settings as app_settings
from tldw_Server_API.app.core.Sandbox.models import RunSpec, RuntimeType, SessionSpec, TrustLevel
from tldw_Server_API.app.core.Sandbox.policy import SandboxPolicy
from tldw_Server_API.app.core.Sandbox.runtime_capabilities import RuntimePreflightResult, collect_runtime_preflights
from tldw_Server_API.app.core.Sandbox.runners.lima_runner import LimaRunner
from tldw_Server_API.app.core.Sandbox.streams import get_hub
from tldw_Server_API.app.core.testing import is_truthy
from tldw_Server_API.app.services.acp_runtime_policy_service import (
    ACPRuntimePolicyService,
    ACPRuntimePolicySnapshot,
)

_ACP_SANDBOX_NONCRITICAL_EXCEPTIONS = (
    ACPResponseError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    json.JSONDecodeError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)


def _is_self_referential_agent_command(command: str) -> bool:
    """Return True when ACP runner is configured to launch itself as downstream."""
    normalized = (command or "").strip()
    if not normalized:
        return False
    return Path(normalized).name == "tldw-agent-acp"


@dataclass
class SandboxSessionHandle:
    session_id: str
    user_id: int
    sandbox_session_id: str
    run_id: str
    client: ACPStreamClient
    reader_task: asyncio.Task
    ssh_user: str | None = None
    ssh_host: str | None = None
    ssh_port: int | None = None
    ssh_private_key: str | None = None
    agent_capabilities: dict[str, Any] | None = None
    persona_id: str | None = None
    workspace_id: str | None = None
    workspace_group_id: str | None = None
    scope_snapshot_id: str | None = None


class ACPSandboxRunnerManager(ACPRuntimePolicySupportMixin):
    def __init__(self, config: ACPSandboxConfig | None = None) -> None:
        self.config = config or load_acp_sandbox_config()
        self._sessions: dict[str, SandboxSessionHandle] = {}
        self._sessions_lock = asyncio.Lock()
        self._updates: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        self._agent_capabilities: dict[str, Any] = {}
        self._ws_registry: dict[str, SessionWebSocketRegistry] = {}
        self._ws_registry_lock = asyncio.Lock()
        self._ssh_ports_in_use: set[int] = set()
        self._ssh_ports_lock = asyncio.Lock()
        self._governance = ACPGovernanceCoordinator()
        self._runtime_policy_log_label = "ACP sandbox runtime policy"
        self._runtime_policy_service = ACPRuntimePolicyService()
        self._runtime_policy_snapshots: dict[str, ACPRuntimePolicySnapshot] = {}
        self._runtime_policy_refresh_tasks: dict[str, asyncio.Task[ACPRuntimePolicySnapshot | None]] = {}
        self._runtime_policy_refresh_lock = asyncio.Lock()

    def _control_record_from_handle(self, sess: SandboxSessionHandle) -> dict[str, Any]:
        return {
            "id": sess.session_id,
            "user_id": int(sess.user_id),
            "sandbox_session_id": sess.sandbox_session_id,
            "run_id": sess.run_id,
            "ssh_host": sess.ssh_host,
            "ssh_port": sess.ssh_port,
            "ssh_user": sess.ssh_user,
            "ssh_private_key": sess.ssh_private_key,
            "persona_id": sess.persona_id,
            "workspace_id": sess.workspace_id,
            "workspace_group_id": sess.workspace_group_id,
            "scope_snapshot_id": sess.scope_snapshot_id,
        }

    def _get_sandbox_store(self) -> Any | None:
        try:
            sandbox_service = sandbox_ep._service  # type: ignore[attr-defined]
            orch = getattr(sandbox_service, "_orch", None)
            return getattr(orch, "_store", None)
        except _ACP_SANDBOX_NONCRITICAL_EXCEPTIONS:
            return None

    def _store_control_record(self, sess: SandboxSessionHandle, *, required: bool = False) -> bool:
        store = self._get_sandbox_store()
        if store is None:
            if required:
                raise ACPResponseError("ACP sandbox control metadata store is unavailable")
            return False
        putter = getattr(store, "put_acp_session_control", None)
        if not callable(putter):
            if required:
                raise ACPResponseError("ACP sandbox control metadata store does not support ACP session control")
            return False
        control = self._control_record_from_handle(sess)
        try:
            putter(
                session_id=str(control.get("id") or ""),
                user_id=control.get("user_id"),
                sandbox_session_id=control.get("sandbox_session_id"),
                run_id=control.get("run_id"),
                ssh_host=control.get("ssh_host"),
                ssh_port=control.get("ssh_port"),
                ssh_user=control.get("ssh_user"),
                ssh_private_key=control.get("ssh_private_key"),
                persona_id=control.get("persona_id"),
                workspace_id=control.get("workspace_id"),
                workspace_group_id=control.get("workspace_group_id"),
                scope_snapshot_id=control.get("scope_snapshot_id"),
            )
            return True
        except _ACP_SANDBOX_NONCRITICAL_EXCEPTIONS as exc:
            if required:
                raise ACPResponseError("Failed to persist ACP sandbox session control metadata") from exc
            return False

    def _load_control_record(self, session_id: str) -> dict[str, Any] | None:
        store = self._get_sandbox_store()
        if store is None:
            return None
        getter = getattr(store, "get_acp_session_control", None)
        if not callable(getter):
            return None
        try:
            row = getter(str(session_id))
        except _ACP_SANDBOX_NONCRITICAL_EXCEPTIONS:
            return None
        return dict(row) if isinstance(row, dict) else None

    def _delete_control_record(self, session_id: str) -> bool:
        store = self._get_sandbox_store()
        if store is None:
            return False
        deleter = getattr(store, "delete_acp_session_control", None)
        if not callable(deleter):
            return False
        try:
            return bool(deleter(str(session_id)))
        except _ACP_SANDBOX_NONCRITICAL_EXCEPTIONS:
            return False

    @property
    def agent_capabilities(self) -> dict[str, Any]:
        return self._agent_capabilities

    @property
    def is_running(self) -> bool:
        return True

    @staticmethod
    def _safe_json_summary(payload: Any, *, max_chars: int = 1200) -> str:
        try:
            rendered = json.dumps(payload or {}, sort_keys=True, default=str)
        except _ACP_SANDBOX_NONCRITICAL_EXCEPTIONS:
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
        """Resolve ACP sandbox rollout mode with optional per-request override."""
        raw_mode = None
        if isinstance(metadata, dict):
            raw_mode = metadata.get("governance_rollout_mode")
        try:
            from tldw_Server_API.app.core import config as app_config

            return app_config.resolve_governance_rollout_mode(
                str(raw_mode) if raw_mode is not None else None
            )
        except _ACP_SANDBOX_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("ACP sandbox rollout mode resolution failed; defaulting to off: {}", exc)
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
        """Emit ACP sandbox governance metrics using shared MCP collector."""
        try:
            from tldw_Server_API.app.core.MCP_unified.monitoring.metrics import get_metrics_collector

            get_metrics_collector().record_governance_check(
                surface=surface,
                category=category,
                status=status,
                rollout_mode=rollout_mode,
            )
        except _ACP_SANDBOX_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("ACP sandbox governance metric emit failed open: {}", exc)

    async def _build_governance_metadata(
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
        control = await self._get_session_control_record(session_id)
        if isinstance(control, dict):
            owner = control.get("user_id")
            if owner is not None:
                try:
                    merged.setdefault("user_id", int(owner))
                except _ACP_SANDBOX_NONCRITICAL_EXCEPTIONS:
                    pass
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
        merged_metadata = await self._build_governance_metadata(
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
        merged_metadata = await self._build_governance_metadata(
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

    async def start(self) -> None:
        return

    async def shutdown(self) -> None:
        async with self._sessions_lock:
            sessions = list(self._sessions.values())
        for sess in sessions:
            await self.close_session(sess.session_id)

    def _validate_lima_strict_runtime_requirements(
        self,
        *,
        preflight: RuntimePreflightResult | None = None,
    ) -> None:
        runtime = str(self.config.runtime or "").strip().lower()
        if runtime != RuntimeType.lima.value:
            return
        network_policy = str(self.config.network_policy or "deny_all").strip().lower()
        if network_policy not in {"deny_all", "allowlist"}:
            raise ACPResponseError(
                "ACP lima strict policy requires network_policy to be deny_all or allowlist"
            )
        preflight = preflight or LimaRunner().preflight(network_policy=network_policy)
        if preflight.available:
            return
        reasons = list(preflight.reasons or [])
        raise ACPResponseError(
            f"ACP lima strict policy requirements not satisfied: {', '.join(reasons) if reasons else 'unknown'}"
        )

    def _configured_runtime(self) -> RuntimeType:
        runtime_raw = str(self.config.runtime or "").strip().lower()
        try:
            return RuntimeType(runtime_raw)
        except ValueError as exc:
            raise ACPResponseError(f"Unsupported ACP sandbox runtime: {runtime_raw or 'unknown'}") from exc

    def _validate_runtime_requirements(self, runtime: RuntimeType) -> None:
        network_policy = str(self.config.network_policy or "deny_all").strip().lower() or "deny_all"
        runtime_preflights = collect_runtime_preflights(network_policy=network_policy)
        preflight = runtime_preflights.get(runtime)
        if runtime == RuntimeType.lima:
            self._validate_lima_strict_runtime_requirements(preflight=preflight)
        if preflight is None:
            return
        if not preflight.available:
            reasons = list(preflight.reasons or [])
            raise ACPResponseError(
                f"ACP {runtime.value} runtime requirements not satisfied: "
                f"{', '.join(reasons) if reasons else 'unknown'}"
            )
        try:
            SandboxPolicy._require_trust_level_supported(
                runtime,
                TrustLevel.standard,
                runtime_preflights={runtime: preflight},
            )
        except SandboxPolicy.PolicyUnsupported as exc:
            reasons = list(getattr(exc, "reasons", []) or [])
            detail = ", ".join(reasons) if reasons else getattr(exc, "requirement", "unsupported")
            raise ACPResponseError(
                f"ACP {runtime.value} runtime requirements not satisfied: {detail}"
            ) from exc

    # -------------------------------------------------------------------------
    # Session lifecycle
    # -------------------------------------------------------------------------

    async def create_session(
        self,
        cwd: str,
        mcp_servers: list[dict[str, Any]] | None = None,
        agent_type: str | None = None,
        user_id: int | None = None,
        persona_id: str | None = None,
        workspace_id: str | None = None,
        workspace_group_id: str | None = None,
        scope_snapshot_id: str | None = None,
        session_env: dict[str, str] | None = None,
    ) -> str:
        if not self.config.enabled:
            raise ACPResponseError("ACP sandbox mode is not enabled")
        if not self.config.agent_command:
            raise ACPResponseError("ACP_SANDBOX_AGENT_COMMAND is required for sandbox mode")
        if _is_self_referential_agent_command(self.config.agent_command):
            raise ACPResponseError(
                "ACP_SANDBOX_AGENT_COMMAND cannot be tldw-agent-acp. "
                "Set it to a downstream ACP-compatible coding agent command "
                "(for example: claude, codex, or opencode)."
            )
        if user_id is None:
            raise ACPResponseError("ACP sandbox sessions require an authenticated user_id")
        if cwd and cwd != "/workspace":
            logger.warning("ACP sandbox ignores host cwd '{}' (using /workspace inside container)", cwd)
        try:
            env_bg = os.getenv("SANDBOX_BACKGROUND_EXECUTION")
            if env_bg is not None:
                background = is_truthy(env_bg)
            else:
                background = bool(getattr(app_settings, "SANDBOX_BACKGROUND_EXECUTION", False))
        except _ACP_SANDBOX_NONCRITICAL_EXCEPTIONS:
            background = False
        if not background:
            raise ACPResponseError("SANDBOX_BACKGROUND_EXECUTION must be enabled for ACP sandbox sessions")
        try:
            env_exec = os.getenv("SANDBOX_ENABLE_EXECUTION")
            if env_exec is not None:
                execute_enabled = is_truthy(env_exec)
            else:
                execute_enabled = bool(getattr(app_settings, "SANDBOX_ENABLE_EXECUTION", False))
        except _ACP_SANDBOX_NONCRITICAL_EXCEPTIONS:
            execute_enabled = False
        if not execute_enabled:
            raise ACPResponseError("SANDBOX_ENABLE_EXECUTION must be enabled for ACP sandbox sessions")
        runtime = self._configured_runtime()
        self._validate_runtime_requirements(runtime)

        sandbox_service = sandbox_ep._service  # type: ignore[attr-defined]

        sandbox_session = None
        status = None
        ssh_port: int | None = None
        ssh_key_priv: str | None = None
        ssh_key_pub: str | None = None
        acp_session_id: str | None = None

        try:
            # Create sandbox session
            sess_spec = SessionSpec(
                runtime=runtime,
                base_image=self.config.base_image,
                network_policy=self.config.network_policy or "deny_all",
                trust_level=TrustLevel.standard,
                persona_id=persona_id,
                workspace_id=workspace_id,
                workspace_group_id=workspace_group_id,
                scope_snapshot_id=scope_snapshot_id,
            )
            sandbox_session = sandbox_service.create_session(
                user_id=user_id,
                spec=sess_spec,
                spec_version="1.0",
                idem_key=None,
                raw_body={
                    "runtime": self.config.runtime,
                    "base_image": self.config.base_image,
                    "persona_id": persona_id,
                    "workspace_id": workspace_id,
                    "workspace_group_id": workspace_group_id,
                    "scope_snapshot_id": scope_snapshot_id,
                },
            )

            if self.config.ssh_enabled:
                ssh_key_priv, ssh_key_pub = self._generate_ssh_keypair()
                ssh_port = await self._allocate_ssh_port()

            merged_agent_env = {
                **dict(self.config.agent_env or {}),
                **dict(session_env or {}),
            }
            env = {
                "ACP_SSH_USER": self.config.ssh_user,
                "ACP_AGENT_COMMAND": self.config.agent_command,
                "ACP_AGENT_ARGS_JSON": json.dumps(self.config.agent_args),
                "ACP_AGENT_ENV_JSON": json.dumps(merged_agent_env),
                "ACP_WORKSPACE_ROOT": "/workspace",
                "ACP_RUNTIME_HOME": "/workspace/.acp-home",
            }
            if self.config.ssh_enabled and ssh_key_pub:
                env["ACP_SSH_AUTHORIZED_KEY"] = ssh_key_pub
                env["ACP_SSH_PORT"] = str(int(self.config.ssh_container_port))

            port_mappings: list[dict[str, str | int]] = []
            if self.config.ssh_enabled and ssh_port is not None:
                port_mappings = [{
                    "host_ip": self.config.ssh_host,
                    "host_port": ssh_port,
                    "container_port": int(self.config.ssh_container_port),
                }]

            run_spec = RunSpec(
                session_id=sandbox_session.id,
                runtime=runtime,
                base_image=self.config.base_image,
                command=["/usr/local/bin/tldw-acp-entrypoint"],
                env=env,
                interactive=True,
                run_as_root=bool(self.config.run_as_root),
                read_only_root=bool(self.config.read_only_root),
                network_policy=self.config.network_policy or "deny_all",
                trust_level=TrustLevel.standard,
                port_mappings=port_mappings,
                persona_id=persona_id,
                workspace_id=workspace_id,
                workspace_group_id=workspace_group_id,
                scope_snapshot_id=scope_snapshot_id,
            )

            status = sandbox_service.start_run_scaffold(
                user_id=user_id,
                spec=run_spec,
                spec_version="1.0",
                idem_key=None,
                raw_body={
                    "runtime": self.config.runtime,
                    "base_image": self.config.base_image,
                    "persona_id": persona_id,
                    "workspace_id": workspace_id,
                    "workspace_group_id": workspace_group_id,
                    "scope_snapshot_id": scope_snapshot_id,
                },
            )
        except _ACP_SANDBOX_NONCRITICAL_EXCEPTIONS:
            if status is not None:
                with contextlib.suppress(_ACP_SANDBOX_NONCRITICAL_EXCEPTIONS):
                    sandbox_service.cancel_run(status.id)
            if sandbox_session is not None:
                with contextlib.suppress(_ACP_SANDBOX_NONCRITICAL_EXCEPTIONS):
                    sandbox_service.destroy_session(sandbox_session.id)
            if ssh_port is not None:
                await self._release_ssh_port(ssh_port)
            raise

        hub = get_hub()
        q = hub.subscribe_with_buffer(status.id)

        async def _send_bytes(data: bytes) -> None:
            try:
                hub.push_stdin(status.id, data)
            except _ACP_SANDBOX_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"ACP sandbox stdin push failed: {e}")

        client = ACPStreamClient(send_bytes=_send_bytes)
        client.set_notification_handler(self._handle_notification)
        client.set_request_handler(self._handle_request)

        reader_task = asyncio.create_task(self._reader_loop(status.id, q, client))
        session_registered = False

        try:
            await client.start()
            init_result = await client.call(
                "initialize",
                {
                    "protocolVersion": 1,
                    "clientCapabilities": {
                        "fs": {"readTextFile": True, "writeTextFile": True},
                        "terminal": True,
                    },
                    "clientInfo": {
                        "name": "tldw-server",
                        "title": "TLDW Server",
                        "version": "0.1.0",
                    },
                },
            )
            self._agent_capabilities = init_result.result.get("agentCapabilities", {}) if init_result.result else {}

            # Always use container workspace root for ACP sessions
            session_new = await client.call(
                "session/new",
                {
                    "cwd": "/workspace",
                    "mcpServers": mcp_servers or [],
                    "agentType": agent_type,
                    **({"env": dict(session_env)} if session_env else {}),
                },
            )
            session_id = (session_new.result or {}).get("sessionId")
            if not session_id:
                raise ACPResponseError("Missing sessionId in ACP sandbox response")
            acp_session_id = str(session_id)

            handle = SandboxSessionHandle(
                session_id=session_id,
                user_id=user_id,
                sandbox_session_id=sandbox_session.id,
                run_id=status.id,
                persona_id=persona_id,
                workspace_id=workspace_id,
                workspace_group_id=workspace_group_id,
                scope_snapshot_id=scope_snapshot_id,
                client=client,
                reader_task=reader_task,
                ssh_user=self.config.ssh_user if self.config.ssh_enabled else None,
                ssh_host=self.config.ssh_host if self.config.ssh_enabled else None,
                ssh_port=ssh_port if self.config.ssh_enabled else None,
                ssh_private_key=ssh_key_priv if self.config.ssh_enabled else None,
                agent_capabilities=self._agent_capabilities,
            )

            async with self._sessions_lock:
                self._sessions[session_id] = handle

            self._store_control_record(handle, required=True)
            session_registered = True
            return session_id
        except _ACP_SANDBOX_NONCRITICAL_EXCEPTIONS:
            with contextlib.suppress(_ACP_SANDBOX_NONCRITICAL_EXCEPTIONS):
                reader_task.cancel()
            with contextlib.suppress(_ACP_SANDBOX_NONCRITICAL_EXCEPTIONS):
                await client.close()
            if acp_session_id:
                async with self._sessions_lock:
                    self._sessions.pop(acp_session_id, None)
                self._updates.pop(acp_session_id, None)
                self._delete_control_record(acp_session_id)
            with contextlib.suppress(_ACP_SANDBOX_NONCRITICAL_EXCEPTIONS):
                sandbox_service.cancel_run(status.id)
            with contextlib.suppress(_ACP_SANDBOX_NONCRITICAL_EXCEPTIONS):
                sandbox_service.destroy_session(sandbox_session.id)
            raise
        finally:
            if ssh_port is not None and not session_registered:
                with contextlib.suppress(_ACP_SANDBOX_NONCRITICAL_EXCEPTIONS):
                    await self._release_ssh_port(ssh_port)

    async def prompt(self, session_id: str, prompt: list[dict[str, Any]]) -> dict[str, Any]:
        governance = await self.check_prompt_governance(session_id, prompt)
        if self._governance.is_denied_with_enforcement(governance):
            raise ACPGovernanceDeniedError(governance=governance or {})

        sess = await self._get_session(session_id)
        response = await sess.client.call(
            "session/prompt",
            {
                "sessionId": session_id,
                "prompt": prompt,
            },
        )
        return response.result or {}

    async def cancel(self, session_id: str) -> None:
        sess = await self._get_session(session_id)
        await sess.client.notify("session/cancel", {"sessionId": session_id})

    async def close_session(self, session_id: str) -> None:
        sess = await self._get_session(session_id, required=False)
        control = self._load_control_record(session_id) if not sess else self._control_record_from_handle(sess)
        if not sess and not control:
            return
        if sess:
            with contextlib.suppress(_ACP_SANDBOX_NONCRITICAL_EXCEPTIONS):
                await sess.client.call("_tldw/session/close", {"sessionId": session_id})
            try:
                if sess.reader_task:
                    sess.reader_task.cancel()
            except _ACP_SANDBOX_NONCRITICAL_EXCEPTIONS:
                pass
            with contextlib.suppress(_ACP_SANDBOX_NONCRITICAL_EXCEPTIONS):
                await sess.client.close()
        run_id = str((control or {}).get("run_id") or "")
        sandbox_session_id = str((control or {}).get("sandbox_session_id") or "")
        ssh_port_raw = (control or {}).get("ssh_port")
        try:
            ssh_port = int(ssh_port_raw) if ssh_port_raw is not None else None
        except _ACP_SANDBOX_NONCRITICAL_EXCEPTIONS:
            ssh_port = None
        try:
            if run_id:
                sandbox_ep._service.cancel_run(run_id)  # type: ignore[attr-defined]
        except _ACP_SANDBOX_NONCRITICAL_EXCEPTIONS:
            pass
        try:
            if sandbox_session_id:
                sandbox_ep._service.destroy_session(sandbox_session_id)  # type: ignore[attr-defined]
        except _ACP_SANDBOX_NONCRITICAL_EXCEPTIONS:
            pass
        if ssh_port:
            await self._release_ssh_port(ssh_port)
        async with self._sessions_lock:
            self._sessions.pop(session_id, None)
        self._updates.pop(session_id, None)
        self._clear_runtime_policy_session_state(session_id)
        self._delete_control_record(session_id)

    def pop_updates(self, session_id: str, limit: int = 100) -> list[dict[str, Any]]:
        updates = []
        queue = self._updates.get(session_id)
        if not queue:
            return updates
        for _ in range(min(limit, len(queue))):
            updates.append(queue.popleft())
        return updates

    async def verify_session_access(self, session_id: str, user_id: int) -> bool:
        """Verify that the given user owns the sandbox-backed ACP session."""
        control = await self._get_session_control_record(session_id)
        if not control:
            return False
        owner = control.get("user_id")
        if owner is None:
            return False
        try:
            return int(owner) == int(user_id)
        except _ACP_SANDBOX_NONCRITICAL_EXCEPTIONS:
            return False

    # -------------------------------------------------------------------------
    # SSH metadata
    # -------------------------------------------------------------------------
    async def get_ssh_info(self, session_id: str, user_id: int | None = None) -> tuple[str, int, str, str] | None:
        control = await self._get_session_control_record(session_id)
        if not control:
            return None
        owner = control.get("user_id")
        if user_id is not None and owner is not None:
            try:
                if int(owner) != int(user_id):
                    return None
            except _ACP_SANDBOX_NONCRITICAL_EXCEPTIONS:
                return None
        ssh_host = control.get("ssh_host")
        ssh_port = control.get("ssh_port")
        ssh_user = control.get("ssh_user")
        ssh_private_key = control.get("ssh_private_key")
        if not ssh_host or ssh_port is None or not ssh_user or not ssh_private_key:
            return None
        try:
            return str(ssh_host), int(ssh_port), str(ssh_user), str(ssh_private_key)
        except _ACP_SANDBOX_NONCRITICAL_EXCEPTIONS:
            return None

    async def get_session_metadata(self, session_id: str, user_id: int | None = None) -> dict[str, Any] | None:
        control = await self._get_session_control_record(session_id)
        if not control:
            return None
        owner = control.get("user_id")
        if user_id is not None and owner is not None:
            try:
                if int(owner) != int(user_id):
                    return None
            except _ACP_SANDBOX_NONCRITICAL_EXCEPTIONS:
                return None
        session_has_ssh = bool(control.get("ssh_user"))
        return {
            "sandbox_session_id": control.get("sandbox_session_id"),
            "sandbox_run_id": control.get("run_id"),
            "ssh_ws_url": f"/api/v1/acp/sessions/{session_id}/ssh" if session_has_ssh else None,
            "ssh_user": control.get("ssh_user"),
            "persona_id": control.get("persona_id"),
            "workspace_id": control.get("workspace_id"),
            "workspace_group_id": control.get("workspace_group_id"),
            "scope_snapshot_id": control.get("scope_snapshot_id"),
        }

    # -------------------------------------------------------------------------
    # WebSocket registry / permissions (adapted from ACPRunnerClient)
    # -------------------------------------------------------------------------

    async def register_websocket(
        self,
        session_id: str,
        send_callback: Callable[[dict[str, Any]], Coroutine[Any, Any, None]],
    ) -> None:
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
        async with self._ws_registry_lock:
            if session_id in self._ws_registry:
                self._ws_registry[session_id].websockets.discard(send_callback)
                logger.debug("Unregistered WebSocket for session {}", session_id)

    def has_websocket_connections(self, session_id: str) -> bool:
        registry = self._ws_registry.get(session_id)
        return registry is not None and len(registry.websockets) > 0

    async def _broadcast_to_session(self, session_id: str, message: dict[str, Any]) -> None:
        registry = self._ws_registry.get(session_id)
        if not registry:
            return

        failed_callbacks = []
        for callback in list(registry.websockets):
            try:
                await callback(message)
            except _ACP_SANDBOX_NONCRITICAL_EXCEPTIONS as e:
                logger.warning("Failed to send to WebSocket for session {}: {}", session_id, e)
                failed_callbacks.append(callback)

        for callback in failed_callbacks:
            registry.websockets.discard(callback)

    async def respond_to_permission(
        self,
        session_id: str,
        request_id: str,
        approved: bool,
        batch_approve_tier: str | None = None,
    ) -> bool:
        registry = self._ws_registry.get(session_id)
        if not registry:
            logger.warning("No registry for session {} when responding to permission", session_id)
            return False

        pending = registry.pending_permissions.pop(request_id, None)
        if not pending:
            logger.warning("Permission request {} not found for session {}", request_id, session_id)
            return False

        if approved and batch_approve_tier:
            registry.batch_approved_tiers.add(batch_approve_tier)
            logger.info("Batch-approved tier {} for session {}", batch_approve_tier, session_id)

        outcome = "approved" if approved else "denied"
        if not pending.future.done():
            pending.future.set_result({"outcome": outcome})

        return True

    def _determine_permission_tier(self, tool_name: str) -> str:
        return determine_permission_tier(tool_name)

    async def _handle_notification(self, msg: ACPMessage) -> None:
        if msg.method != "session/update":
            return
        params = msg.params or {}
        session_id = params.get("sessionId")
        if not session_id:
            return

        self._updates[session_id].append(params)

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

        if not self.has_websocket_connections(session_id):
            logger.warning("ACP permission request auto-cancelled (no WebSocket connections)")
            return ACPMessage(
                jsonrpc="2.0",
                id=msg.id,
                result={"outcome": {"outcome": "cancelled"}},
            )

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

        async with self._ws_registry_lock:
            if session_id not in self._ws_registry:
                self._ws_registry[session_id] = SessionWebSocketRegistry(session_id=session_id)
            self._ws_registry[session_id].pending_permissions[request_id] = pending

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

        if not self.has_websocket_connections(session_id):
            logger.warning("ACP permission request auto-cancelled (all WebSocket connections failed during broadcast)")
            async with self._ws_registry_lock:
                if session_id in self._ws_registry:
                    self._ws_registry[session_id].pending_permissions.pop(request_id, None)
            return ACPMessage(
                jsonrpc="2.0",
                id=msg.id,
                result={"outcome": {"outcome": "cancelled"}},
            )

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
            async with self._ws_registry_lock:
                if session_id in self._ws_registry:
                    self._ws_registry[session_id].pending_permissions.pop(request_id, None)
            return ACPMessage(
                jsonrpc="2.0",
                id=msg.id,
                result={"outcome": {"outcome": "cancelled"}},
            )

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    async def _reader_loop(self, run_id: str, q: asyncio.Queue, client: ACPStreamClient) -> None:
        try:
            while True:
                frame = await q.get()
                if not isinstance(frame, dict):
                    continue
                ftype = frame.get("type")
                if ftype in {"stdout", "stderr"}:
                    enc = frame.get("encoding") or "utf8"
                    data_field = frame.get("data")
                    if not isinstance(data_field, str):
                        continue
                    try:
                        raw = base64.b64decode(data_field) if enc == "base64" else data_field.encode("utf-8")
                    except _ACP_SANDBOX_NONCRITICAL_EXCEPTIONS:
                        raw = b""
                    if not raw:
                        continue
                    if ftype == "stdout":
                        await client.feed_bytes(raw)
                    else:
                        logger.debug(f"ACP sandbox stderr: {data_field}")
                elif ftype == "event" and frame.get("event") == "end":
                    await client.close()
                    return
        except asyncio.CancelledError:
            return
        except _ACP_SANDBOX_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"ACP sandbox reader loop error: {e}")

    async def _get_session_control_record(self, session_id: str) -> dict[str, Any] | None:
        async with self._sessions_lock:
            sess = self._sessions.get(session_id)
        if sess:
            return self._control_record_from_handle(sess)
        return self._load_control_record(session_id)

    async def _attach_to_existing_session(self, control: dict[str, Any]) -> SandboxSessionHandle | None:
        session_id = str(control.get("id") or "")
        run_id = str(control.get("run_id") or "")
        sandbox_session_id = str(control.get("sandbox_session_id") or "")
        if not session_id or not run_id or not sandbox_session_id:
            return None
        try:
            user_id = int(control.get("user_id"))
        except _ACP_SANDBOX_NONCRITICAL_EXCEPTIONS:
            return None

        hub = get_hub()
        q = hub.subscribe_with_buffer(run_id)

        async def _send_bytes(data: bytes) -> None:
            try:
                hub.push_stdin(run_id, data)
            except _ACP_SANDBOX_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"ACP sandbox stdin push failed (rehydrate): {e}")

        client = ACPStreamClient(send_bytes=_send_bytes)
        client.set_notification_handler(self._handle_notification)
        client.set_request_handler(self._handle_request)
        reader_task = asyncio.create_task(self._reader_loop(run_id, q, client))
        try:
            await client.start()
            with contextlib.suppress(_ACP_SANDBOX_NONCRITICAL_EXCEPTIONS):
                init_result = await client.call(
                    "initialize",
                    {
                        "protocolVersion": 1,
                        "clientCapabilities": {
                            "fs": {"readTextFile": True, "writeTextFile": True},
                            "terminal": True,
                        },
                        "clientInfo": {
                            "name": "tldw-server",
                            "title": "TLDW Server",
                            "version": "0.1.0",
                        },
                    },
                )
                caps = init_result.result.get("agentCapabilities", {}) if init_result.result else {}
                if isinstance(caps, dict) and caps:
                    self._agent_capabilities = caps
        except _ACP_SANDBOX_NONCRITICAL_EXCEPTIONS:
            with contextlib.suppress(_ACP_SANDBOX_NONCRITICAL_EXCEPTIONS):
                reader_task.cancel()
            with contextlib.suppress(_ACP_SANDBOX_NONCRITICAL_EXCEPTIONS):
                await client.close()
            return None

        ssh_port_raw = control.get("ssh_port")
        try:
            ssh_port = int(ssh_port_raw) if ssh_port_raw is not None else None
        except _ACP_SANDBOX_NONCRITICAL_EXCEPTIONS:
            ssh_port = None
        if ssh_port is not None:
            async with self._ssh_ports_lock:
                self._ssh_ports_in_use.add(ssh_port)

        return SandboxSessionHandle(
            session_id=session_id,
            user_id=user_id,
            sandbox_session_id=sandbox_session_id,
            run_id=run_id,
            client=client,
            reader_task=reader_task,
            ssh_user=(str(control.get("ssh_user")) if control.get("ssh_user") is not None else None),
            ssh_host=(str(control.get("ssh_host")) if control.get("ssh_host") is not None else None),
            ssh_port=ssh_port,
            ssh_private_key=(
                str(control.get("ssh_private_key"))
                if control.get("ssh_private_key") is not None
                else None
            ),
            agent_capabilities=self._agent_capabilities,
            persona_id=(str(control.get("persona_id")) if control.get("persona_id") is not None else None),
            workspace_id=(str(control.get("workspace_id")) if control.get("workspace_id") is not None else None),
            workspace_group_id=(
                str(control.get("workspace_group_id"))
                if control.get("workspace_group_id") is not None
                else None
            ),
            scope_snapshot_id=(
                str(control.get("scope_snapshot_id"))
                if control.get("scope_snapshot_id") is not None
                else None
            ),
        )

    async def _get_session(self, session_id: str, required: bool = True) -> SandboxSessionHandle | None:
        async with self._sessions_lock:
            sess = self._sessions.get(session_id)
        if sess:
            return sess

        control = self._load_control_record(session_id)
        if control:
            restored = await self._attach_to_existing_session(control)
            if restored is not None:
                existing: SandboxSessionHandle | None = None
                async with self._sessions_lock:
                    existing = self._sessions.get(session_id)
                    if existing is None:
                        self._sessions[session_id] = restored
                        existing = restored
                if existing is not restored:
                    with contextlib.suppress(_ACP_SANDBOX_NONCRITICAL_EXCEPTIONS):
                        restored.reader_task.cancel()
                    with contextlib.suppress(_ACP_SANDBOX_NONCRITICAL_EXCEPTIONS):
                        await restored.client.close()
                else:
                    self._store_control_record(existing)
                return existing

        if required:
            raise ACPResponseError("Unknown session")
        return None

    def _generate_ssh_keypair(self) -> tuple[str, str]:
        try:
            import asyncssh  # type: ignore

            key = asyncssh.generate_private_key("ssh-ed25519")
            priv = key.export_private_key().decode("utf-8")
            pub = key.export_public_key().decode("utf-8")
            return priv, pub
        except _ACP_SANDBOX_NONCRITICAL_EXCEPTIONS:
            try:
                import shutil
                import subprocess  # nosec B404 - controlled fallback to local ssh-keygen binary
                import tempfile
                from pathlib import Path

                ssh_keygen = shutil.which("ssh-keygen")
                if not ssh_keygen:
                    raise ACPResponseError("ssh-keygen is unavailable")

                with tempfile.TemporaryDirectory(prefix="acp_ssh_key_") as tmpdir:
                    key_path = Path(tmpdir) / "id_ed25519"
                    subprocess.run(
                        [ssh_keygen, "-q", "-t", "ed25519", "-N", "", "-f", str(key_path)],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )  # nosec B603 - fixed argv, no shell, temp path is generated by the server
                    priv = key_path.read_text(encoding="utf-8")
                    pub = key_path.with_suffix(".pub").read_text(encoding="utf-8").strip()
                    return priv, pub
            except _ACP_SANDBOX_NONCRITICAL_EXCEPTIONS as e:
                raise ACPResponseError(f"Failed to generate SSH key: {e}") from e

    async def _allocate_ssh_port(self) -> int:
        async with self._ssh_ports_lock:
            for port in range(self.config.ssh_port_min, self.config.ssh_port_max + 1):
                if port in self._ssh_ports_in_use:
                    continue
                if not self._port_available(self.config.ssh_host, port):
                    continue
                self._ssh_ports_in_use.add(port)
                return port
        raise ACPResponseError("No available SSH ports")

    async def _release_ssh_port(self, port: int) -> None:
        async with self._ssh_ports_lock:
            self._ssh_ports_in_use.discard(port)

    def _port_available(self, host: str, port: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((host, port))
                return True
        except _ACP_SANDBOX_NONCRITICAL_EXCEPTIONS:
            return False
