"""Tests for protocol-side MCP tool-use reporting capture."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import re
import time
from enum import Enum
from types import SimpleNamespace
from typing import Any

import pytest
from loguru import logger
from mcp_unified.interfaces.runtime import ToolHookCallContext, ToolHookDecision
from mcp_unified.tool_hooks import ConfiguredToolCallHookManager, ToolHookRegistration
from mcp_unified.tool_use_reporting.builders import classify_tool_use_exception
from mcp_unified.tool_use_reporting.models import MAX_FILE_POLICY_DECISIONS, ToolUseEvent

from tldw_Server_API.app.core.MCP_unified.auth.rate_limiter import RateLimitExceeded
from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.run_command_module import RunCommandModule
from tldw_Server_API.app.core.MCP_unified.protocol import (
    ErrorCode,
    GovernanceDeniedError,
    InvalidParamsException,
    MCPProtocol,
    RequestContext,
)


class _RecordingToolUseRecorder:
    def __init__(self) -> None:
        self.events: list[ToolUseEvent] = []

    async def record_tool_use(self, event: ToolUseEvent) -> None:
        self.events.append(event)


class _FailingToolUseRecorder:
    def __init__(self) -> None:
        self.called = False

    async def record_tool_use(self, event: ToolUseEvent) -> None:
        self.called = True
        raise RuntimeError(f"do not leak {event.requested_tool_name}")


class _DenyingToolHookManager:
    async def before_tool_call(self, _context: ToolHookCallContext) -> ToolHookDecision:
        return ToolHookDecision(
            action="deny",
            reason_code="blocked_by_profile_hook",
            message="blocked by hook",
            metadata={
                "hook_id": "profile-policy",
                "hook_order": 10,
                "path": "/Users/example/private.txt",
            },
        )

    async def after_tool_call(self, _context: ToolHookCallContext) -> None:
        return None


class _FailingPostToolHookManager:
    async def before_tool_call(self, _context: ToolHookCallContext) -> None:
        return None

    async def after_tool_call(self, _context: ToolHookCallContext) -> None:
        raise RuntimeError("post hook failed for /Users/example/private.txt")


class _RecordingPostToolHookManager:
    def __init__(self) -> None:
        self.after_contexts: list[ToolHookCallContext] = []

    async def before_tool_call(self, _context: ToolHookCallContext) -> None:
        return None

    async def after_tool_call(self, context: ToolHookCallContext) -> None:
        self.after_contexts.append(context)


class _AllowAllRbac:
    async def check_permission(self, *_args: Any, **_kwargs: Any) -> bool:
        return True


class _NoopRateLimiter:
    async def check_rate_limit(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _ToolOnlyRateLimiter:
    async def check_rate_limit(self, key: str, *_args: Any, **_kwargs: Any) -> None:
        if ":tool:" in key:
            raise RateLimitExceeded(1)


class _NoopMetrics:
    def __getattr__(self, _name: str):
        return lambda *args, **kwargs: None


class _RecordingMetrics(_NoopMetrics):
    def __init__(self) -> None:
        self.module_operations: list[bool] = []

    def record_module_operation(self, **kwargs: Any) -> None:
        self.module_operations.append(bool(kwargs["success"]))


class _FailingObserverMetrics(_NoopMetrics):
    def record_module_operation(self, **_kwargs: Any) -> None:
        raise RuntimeError("private metrics detail")

    def record_idempotency_hit(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("private hit detail")

    def record_idempotency_miss(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("private miss detail")


class ExoticObserverError(Exception):
    pass


class _OriginalToolError(Exception):
    pass


class _ExoticObserverMetrics(_NoopMetrics):
    def __init__(self, target: str) -> None:
        self.target = target

    def record_module_operation(self, **_kwargs: Any) -> None:
        if self.target == "module_metrics":
            raise ExoticObserverError("private module metrics detail")

    def record_tool_invalid_params(self, *_args: Any, **_kwargs: Any) -> None:
        if self.target == "invalid_params_metrics":
            raise ExoticObserverError("private invalid params detail")

    def record_idempotency_hit(self, *_args: Any, **_kwargs: Any) -> None:
        if self.target == "idempotency_metrics":
            raise ExoticObserverError("private hit detail")

    def record_idempotency_miss(self, *_args: Any, **_kwargs: Any) -> None:
        if self.target == "idempotency_metrics":
            raise ExoticObserverError("private miss detail")


class _StaticEffectivePolicyResolver:
    """Test double that returns a fixed effective policy for any context."""

    def __init__(self, policy: dict[str, Any] | None = None) -> None:
        self.policy = policy

    async def resolve_for_context(
        self,
        *,
        user_id: str | None,
        metadata: dict[str, Any],
    ) -> dict[str, Any] | None:
        del user_id, metadata
        return self.policy


class _AllowApprovalEvaluator:
    """Test double that always allows tool calls through approval evaluation."""

    async def evaluate_tool_call(self, **_kwargs: Any) -> dict[str, Any]:
        return {"status": "allow", "reason": "test_allow"}


class _NoopExternalAccessEvaluator:
    """Test double with no external server credential grants."""

    async def resolve_for_sources(
        self,
        *,
        sources: list[dict[str, Any]],
        effective_policy: dict[str, Any] | None,
    ) -> dict[str, Any]:
        del sources, effective_policy
        return {"servers": []}


class _FilePolicyPathScopeEnforcer:
    """Test double that returns an allowed redacted file-policy decision."""

    async def evaluate_tool_call(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "enabled": True,
            "within_scope": True,
            "reason": None,
            "force_approval": False,
            "normalized_paths": ["/Users/me/workspace/private/story.txt"],
            "scope_payload": {
                "path_scope_mode": "workspace_root",
                "workspace_root": "/Users/me/workspace",
                "scope_root": "/Users/me/workspace",
                "normalized_paths": ["/Users/me/workspace/private/story.txt"],
                "path_decisions": [
                    {
                        "requested_action": "edit",
                        "normalized_path": "private/story.txt",
                        "grant_outcome": "allowed",
                        "grant_source": "path_grants",
                        "matched_grant_prefix": "private",
                        "matched_grant_effect": "allow",
                        "reason_code": None,
                        "redacted": True,
                    }
                ],
            },
        }


class _DenyFilePolicyPathScopeEnforcer:
    """Test double that returns a denied redacted file-policy decision."""

    async def evaluate_tool_call(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "enabled": True,
            "within_scope": False,
            "reason": "path_action_denied",
            "force_approval": False,
            "normalized_paths": ["/Users/me/workspace/private/story.txt"],
            "scope_payload": {
                "path_scope_mode": "workspace_root",
                "workspace_root": "/Users/me/workspace",
                "scope_root": "/Users/me/workspace",
                "normalized_paths": ["/Users/me/workspace/private/story.txt"],
                "path_decisions": [
                    {
                        "requested_action": "edit",
                        "normalized_path": "private/story.txt",
                        "grant_outcome": "denied",
                        "grant_source": "path_grants",
                        "matched_grant_prefix": "private",
                        "matched_grant_effect": "deny",
                        "reason_code": "path_action_denied",
                        "redacted": True,
                    }
                ],
            },
        }


class _Span:
    def set_attribute(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _Telemetry:
    def trace_context(self, *_args: Any, **_kwargs: Any):
        return contextlib.nullcontext(_Span())


class _RecordingSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, Any] = {}

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value


class _RecordingTelemetry:
    def __init__(self) -> None:
        self.spans: list[_RecordingSpan] = []

    def trace_context(self, *_args: Any, **_kwargs: Any):
        span = _RecordingSpan()
        self.spans.append(span)
        return contextlib.nullcontext(span)


class _ToolModule:
    name = "test_module"

    def __init__(self, *, write: bool = False, fail: BaseException | None = None) -> None:
        self.write = write
        self.fail = fail
        self.calls = 0
        self.config = SimpleNamespace(timeout_seconds=5)

    async def get_tools(self) -> list[dict[str, Any]]:
        return [self._tool_def("test.write" if self.write else "test.read")]

    async def get_tool_def(self, tool_name: str) -> dict[str, Any]:
        return self._tool_def(tool_name)

    def _tool_def(self, tool_name: str) -> dict[str, Any]:
        properties: dict[str, Any] = {"value": {"type": "string"}}
        if self.write:
            properties["idempotencyKey"] = {"type": "string"}
        return {
            "name": tool_name,
            "description": "",
            "inputSchema": {"type": "object", "properties": properties},
            "metadata": {
                "category": "management" if self.write else "read",
                "eval": {
                    "tool_prompt_id": f"mcp.{tool_name}.v1",
                    "tool_prompt_version": "2026.06.06",
                    "action_family": "write" if self.write else "read",
                    "result_kind": "json",
                    "prompt_variant": "builtin",
                },
            },
        }

    def sanitize_input(self, args: Any) -> Any:
        return args

    def validate_tool_arguments(self, _tool_name: str, _tool_args: dict[str, Any]) -> None:
        return None

    def is_write_tool_call(
        self,
        _tool_name: str,
        _tool_args: dict[str, Any],
        *,
        tool_def: dict[str, Any] | None = None,
    ) -> bool:
        return self.write

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: RequestContext | None = None,
    ) -> dict[str, Any]:
        del context
        if self.fail is not None:
            raise self.fail
        self.calls += 1
        return {"ok": True, "tool": tool_name, "call": self.calls, "arguments": arguments}

    async def execute_with_circuit_breaker(self, func, *args: Any, **kwargs: Any) -> Any:
        return await func(*args, **kwargs)


class _ExpectedFailureWriteModule(_ToolModule):
    def __init__(self) -> None:
        super().__init__(write=True)

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: RequestContext | None = None,
    ) -> dict[str, Any]:
        del tool_name, arguments, context
        from tldw_Server_API.app.core.MCP_unified.execution_outcomes import (
            ExpectedToolFailure,
            ExpectedToolFailureReason,
        )

        self.calls += 1
        raise ExpectedToolFailure(ExpectedToolFailureReason.IDEMPOTENCY_UNAVAILABLE)


class _FailClosedToolModule(_ToolModule):
    def _tool_def(self, tool_name: str) -> dict[str, Any]:
        tool_def = super()._tool_def(tool_name)
        tool_def["metadata"]["rate_limit_fail_closed"] = True
        return tool_def


class _ExpectedFailureBreakerModule(BaseModule):
    def __init__(
        self,
        failure: BaseException,
        *,
        threshold: int = 1,
        recovery_timeout: int = 60,
    ) -> None:
        self.failure = failure
        self.calls = 0
        super().__init__(
            ModuleConfig(
                name="test_module",
                circuit_breaker_threshold=threshold,
                circuit_breaker_timeout=recovery_timeout,
            )
        )

    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"ok": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        return [await self.get_tool_def("test.read")]

    async def get_tool_def(self, tool_name: str) -> dict[str, Any]:
        return {
            "name": tool_name,
            "description": "",
            "inputSchema": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
            "metadata": {
                "category": "read",
                "eval": {
                    "tool_prompt_id": "mcp.test.read.v1",
                    "tool_prompt_version": "2026.06.06",
                    "action_family": "read",
                    "result_kind": "json",
                    "prompt_variant": "builtin",
                },
            },
        }

    def sanitize_input(self, args: Any) -> Any:
        return args

    def validate_tool_arguments(
        self,
        _tool_name: str,
        _tool_args: dict[str, Any],
    ) -> None:
        return None

    def is_write_tool_call(
        self,
        _tool_name: str,
        _tool_args: dict[str, Any],
        *,
        tool_def: dict[str, Any] | None = None,
    ) -> bool:
        del tool_def
        return False

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: RequestContext | None = None,
    ) -> dict[str, Any]:
        del tool_name, arguments, context
        self.calls += 1
        raise self.failure


class _UnavailableRateLimiter:
    async def check_rate_limit(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("SENTINEL_RATE_LIMIT_BACKEND_SECRET")


class _MemoryRedis:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}

    async def get(self, key: str) -> Any:
        return self.values.get(key)

    async def set(
        self,
        key: str,
        value: Any,
        *,
        nx: bool = False,
        ex: int | None = None,
    ) -> bool:
        del ex
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    @staticmethod
    def _bytes(value: Any) -> bytes:
        return value if isinstance(value, bytes) else str(value).encode("utf-8")

    async def eval(self, script: str, numkeys: int, *values: Any) -> Any:
        if numkeys == 2 and len(values) == 4 and "return {1, result}" in script:
            binding_key, result_key, arguments_hash, _ttl = values
            binding = self.values.get(binding_key)
            if binding is None:
                return [-2]
            if binding != self._bytes(arguments_hash):
                return [-1]
            result = self.values.get(result_key)
            return [0] if result is None else [1, result]

        if numkeys == 2 and len(values) == 4:
            binding_key, result_key, arguments_hash, _ttl = values
            existing = self.values.get(binding_key)
            if existing is not None:
                return 1 if existing == self._bytes(arguments_hash) else -1
            if result_key in self.values:
                return -2
            self.values[binding_key] = self._bytes(arguments_hash)
            return 2

        if numkeys == 3 and len(values) == 7:
            binding_key, result_key, lock_key, arguments_hash, encoded, _ttl, token = values
            if self.values.get(binding_key) != self._bytes(arguments_hash):
                return 0
            if self.values.get(lock_key) != self._bytes(token):
                return -1
            existing = self.values.get(result_key)
            if existing is not None and existing != self._bytes(encoded):
                return -2
            self.values[result_key] = encoded
            return 1

        if numkeys == 1 and len(values) == 3:
            binding_key, arguments_hash, _ttl = values
            return int(self.values.get(binding_key) == self._bytes(arguments_hash))

        if numkeys != 1 or len(values) != 2:
            raise AssertionError("Unexpected Redis Lua operation")
        key, token = values
        if self.values.get(key) != token:
            return 0
        del self.values[key]
        return 1

    async def expire(self, _key: str, _ttl: int) -> bool:
        return True


class _FilesystemPayloadModule(_ToolModule):
    name = "filesystem"

    def __init__(self) -> None:
        super().__init__(write=True)

    async def get_tools(self) -> list[dict[str, Any]]:
        return [self._tool_def("fs.patch")]

    def _tool_def(self, tool_name: str) -> dict[str, Any]:
        return {
            "name": tool_name,
            "description": "",
            "inputSchema": {"type": "object", "properties": {"diff": {"type": "string"}}},
            "metadata": {
                "category": "management",
                "eval": {
                    "tool_prompt_id": "mcp.fs.patch.v1",
                    "tool_prompt_version": "2026.06.04",
                    "task_families": ["filesystem_edit"],
                    "expected_result_kind": "structured_filesystem_edit",
                    "prompt_variant": "builtin",
                },
            },
        }

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: RequestContext | None = None,
    ) -> dict[str, Any]:
        del tool_name, arguments, context
        return {
            "path": "/Users/me/private/story.txt",
            "content": "SECRET FILE CONTENT",
            "read_receipt": "receipt-token-secret",
            "diff": "--- a/private/story.txt\n+++ b/private/story.txt\n",
            "sha256_before": "a" * 64,
            "sha256_after": "b" * 64,
            "lock_lease_id": "lease-token-secret",
            "eval": {
                "tool_name": "fs.patch",
                "tool_prompt_id": "mcp.fs.patch.v1",
                "tool_prompt_version": "2026.06.04",
                "action_family": "filesystem_edit",
                "result_kind": "structured_filesystem_edit",
                "path_filter_used": True,
                "truncated": False,
            },
        }


class _Registry:
    def __init__(self, module: _ToolModule) -> None:
        self.module = module

    async def find_module_for_tool(self, _tool_name: str) -> _ToolModule:
        return self.module

    def get_module_id_for_tool(self, _tool_name: str) -> str:
        return self.module.name

    async def get_all_modules(self) -> dict[str, Any]:
        return {self.module.name: self.module}


class _FilesystemListModule(_ToolModule):
    """Minimal fs.list test double used by governed run-command telemetry tests."""

    name = "filesystem"

    async def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "fs.list",
                "description": "List files.",
                "inputSchema": {"type": "object", "properties": {"path": {"type": "string"}}},
                "metadata": {
                    "category": "read",
                    "eval": {
                        "tool_prompt_id": "mcp.fs.list.v1",
                        "tool_prompt_version": "2026.06.06",
                        "action_family": "filesystem_read",
                        "result_kind": "json",
                        "prompt_variant": "builtin",
                    },
                },
            }
        ]

    async def get_tool_def(self, tool_name: str) -> dict[str, Any]:
        return (await self.get_tools())[0] if tool_name == "fs.list" else await super().get_tool_def(tool_name)

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: RequestContext | None = None,
    ) -> dict[str, Any]:
        del tool_name, context
        self.calls += 1
        return {
            "path": arguments.get("path") or ".",
            "entries": [{"name": "notes.txt", "type": "file"}],
        }


class _RunCommandRegistry:
    """Registry test double that routes run-command and fs.list tools."""

    def __init__(self, run_module: RunCommandModule, fs_module: _FilesystemListModule) -> None:
        self.run_module = run_module
        self.fs_module = fs_module

    async def find_module_for_tool(self, tool_name: str) -> Any:
        if tool_name in {"run", "bash", "shell", "powershell", "pwsh"}:
            return self.run_module
        if tool_name == "fs.list":
            return self.fs_module
        return None

    def get_module_id_for_tool(self, tool_name: str) -> str | None:
        if tool_name in {"run", "bash", "shell", "powershell", "pwsh"}:
            return "run_command"
        if tool_name == "fs.list":
            return "filesystem"
        return None

    async def get_all_modules(self) -> dict[str, Any]:
        return {
            "run_command": self.run_module,
            "filesystem": self.fs_module,
        }


def _protocol(
    *,
    module: _ToolModule | None = None,
    recorder: Any | None = None,
    rate_limiter: Any | None = None,
    telemetry_provider: Any | None = None,
    effective_policy: dict[str, Any] | None = None,
    path_scope_enforcer: Any | None = None,
    hook_manager: Any | None = None,
) -> tuple[MCPProtocol, Any]:
    recorder = recorder or _RecordingToolUseRecorder()
    module = module or _ToolModule()
    deps = SimpleNamespace(
        module_registry=_Registry(module),
        rbac_policy=_AllowAllRbac(),
        rate_limiter=rate_limiter or _NoopRateLimiter(),
        metrics_collector=_NoopMetrics(),
        telemetry_provider=telemetry_provider or _Telemetry(),
        tool_catalog_provider=object(),
        effective_policy_resolver=_StaticEffectivePolicyResolver(effective_policy),
        approval_evaluator=_AllowApprovalEvaluator(),
        path_scope_enforcer=path_scope_enforcer or _FilePolicyPathScopeEnforcer(),
        external_access_evaluator=_NoopExternalAccessEvaluator(),
        redis_client_factory=lambda **kwargs: None,
        tool_use_recorder=recorder,
        tool_call_hook_manager=hook_manager,
    )
    protocol = MCPProtocol(dependencies=deps)
    return protocol, recorder


def _run_command_protocol() -> tuple[MCPProtocol, _RecordingToolUseRecorder]:
    """Build a protocol wired to run-command plus a filesystem backend double."""

    recorder = _RecordingToolUseRecorder()
    run_module = RunCommandModule(ModuleConfig(name="run", settings={}))
    fs_module = _FilesystemListModule()
    registry = _RunCommandRegistry(run_module, fs_module)
    deps = SimpleNamespace(
        module_registry=registry,
        rbac_policy=_AllowAllRbac(),
        rate_limiter=_NoopRateLimiter(),
        metrics_collector=_NoopMetrics(),
        telemetry_provider=_Telemetry(),
        tool_catalog_provider=object(),
        effective_policy_resolver=_StaticEffectivePolicyResolver(None),
        approval_evaluator=_AllowApprovalEvaluator(),
        path_scope_enforcer=_FilePolicyPathScopeEnforcer(),
        external_access_evaluator=_NoopExternalAccessEvaluator(),
        redis_client_factory=lambda **kwargs: None,
        tool_use_recorder=recorder,
        tool_call_hook_manager=None,
    )
    protocol = MCPProtocol(dependencies=deps)
    run_module.config.settings["protocol"] = protocol
    return protocol, recorder


def _request_context(
    metadata: dict[str, Any] | None = None,
    *,
    request_id: str = "tool-use-reporting",
) -> RequestContext:
    """Build a request context with stable defaults for reporting assertions."""

    return RequestContext(
        request_id=request_id,
        user_id="user-1",
        client_id="client-1",
        metadata=metadata or {},
    )


def _assert_exact_expected_failure_payload(
    payload: dict[str, Any],
    *,
    reason_code: str,
    message: str,
    module: str = "test_module",
    tool: str = "test.read",
) -> None:
    execution_eval = payload["eval"]
    assert payload == {
        "content": [
            {
                "type": "json",
                "json": {
                    "status": "failed",
                    "reason_code": reason_code,
                    "message": message,
                },
            }
        ],
        "isError": True,
        "module": module,
        "tool": tool,
        "eval": execution_eval,
    }


def test_tool_use_file_policy_decisions_bounds_copied_entries() -> None:
    decisions = [
        {
            "requested_action": "edit",
            "normalized_path": f"private/story-{index}.txt",
            "grant_outcome": "allowed",
            "redacted": True,
        }
        for index in range(MAX_FILE_POLICY_DECISIONS + 5)
    ]

    copied = MCPProtocol._tool_use_file_policy_decisions({"path_decisions": decisions})

    assert len(copied) == MAX_FILE_POLICY_DECISIONS
    assert copied[-1]["normalized_path"] == (f"private/story-{MAX_FILE_POLICY_DECISIONS - 1}.txt")


def test_protocol_derives_grant_outcome_from_all_file_policy_decisions() -> None:
    protocol, _ = _protocol()

    denied_event = protocol._build_tool_use_event(
        context=_request_context(),
        requested_tool_name="fs.patch",
        effective_tool_name="fs.patch",
        status="denied",
        execution_origin="failed_before_execution",
        duration_ms=0,
        scope_payload={
            "path_decisions": [
                {"grant_outcome": "allowed", "normalized_path": "private/ok.txt"},
                {"grant_outcome": "denied", "normalized_path": "private/blocked.txt"},
            ],
        },
    )
    not_granted_event = protocol._build_tool_use_event(
        context=_request_context(),
        requested_tool_name="fs.patch",
        effective_tool_name="fs.patch",
        status="denied",
        execution_origin="failed_before_execution",
        duration_ms=0,
        scope_payload={
            "path_decisions": [
                {"grant_outcome": "allowed", "normalized_path": "private/ok.txt"},
                {"grant_outcome": "not_granted", "normalized_path": "private/missing.txt"},
            ],
        },
    )

    assert denied_event.grant_outcome == "denied"
    assert not_granted_event.grant_outcome == "not_granted"


def test_protocol_eval_metadata_grant_outcome_overrides_derived_decisions() -> None:
    protocol, _ = _protocol()

    event = protocol._build_tool_use_event(
        context=_request_context(),
        requested_tool_name="fs.patch",
        effective_tool_name="fs.patch",
        status="denied",
        execution_origin="failed_before_execution",
        duration_ms=0,
        tool_def={"metadata": {"eval": {"grant_outcome": "metadata_override"}}},
        scope_payload={
            "path_decisions": [
                {"grant_outcome": "denied", "normalized_path": "private/blocked.txt"},
            ],
        },
    )

    assert event.grant_outcome == "metadata_override"


def test_protocol_file_policy_presence_ignores_empty_containers() -> None:
    protocol, _ = _protocol()

    event = protocol._build_tool_use_event(
        context=_request_context(),
        requested_tool_name="fs.patch",
        effective_tool_name="fs.patch",
        status="success",
        execution_origin="executed",
        duration_ms=0,
        payload={
            "expected_sha256_by_path": {},
            "sha256_after": [],
            "lock_lease_id_by_path": {},
        },
        tool_args={
            "expected_sha256_by_path": [],
            "lock_lease_id_by_path": {},
        },
    )

    assert event.file_policy_sha256_before_present is False
    assert event.file_policy_sha256_after_present is False
    assert event.file_policy_lock_lease_present is False


def test_protocol_consumes_hook_results_per_tool_use_event() -> None:
    protocol, _ = _protocol()
    context = _request_context(
        metadata={
            "mcp_tool_hook_results": [
                {
                    "phase": "pre",
                    "hook_id": "policy-hook",
                    "hook_order": 4,
                    "action": "deny",
                    "status": "deny",
                    "reason_code": "blocked",
                }
            ]
        }
    )

    first_event = protocol._build_tool_use_event(
        context=context,
        requested_tool_name="test.read",
        status="denied",
        execution_origin="failed_before_execution",
        duration_ms=0,
    )
    second_event = protocol._build_tool_use_event(
        context=context,
        requested_tool_name="test.read",
        status="success",
        execution_origin="executed",
        duration_ms=0,
    )

    assert len(first_event.tool_hook_results) == 1
    assert first_event.tool_hook_results[0].hook_id == "policy-hook"
    assert "mcp_tool_hook_results" not in context.metadata
    assert len(second_event.tool_hook_results) == 0


@pytest.mark.asyncio
async def test_tool_execution_reporter_records_process_request_failure_directly() -> None:
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPRequest
    from tldw_Server_API.app.core.MCP_unified.tool_execution.reporting import (
        ToolExecutionReporter,
    )

    recorder = _RecordingToolUseRecorder()
    reporter = ToolExecutionReporter(
        recorder=recorder,
        metrics=_NoopMetrics(),
        tool_name_re=re.compile(r"^[A-Za-z0-9_.:-]{1,100}$"),
        noncritical_exceptions=(Exception,),
    )
    context = _request_context(metadata={"profile_id": "architect"})
    request = MCPRequest(
        method="tools/call",
        params={"name": "test.read", "arguments": {"value": "ok"}},
        id="req-reporter",
    )

    await reporter.record_process_request_failure(
        request=request,
        context=context,
        status="denied",
        reason_code="permission_denied",
        start_ts=0,
    )

    event = recorder.events[-1]
    assert event.runtime_surface == "protocol"
    assert event.requested_tool_name == "test.read"
    assert event.status == "denied"
    assert event.reason_code == "permission_denied"
    assert event.execution_origin == "failed_before_execution"
    assert event.profile_id == "architect"


@pytest.mark.asyncio
async def test_protocol_records_successful_tool_use_event() -> None:
    protocol, recorder = _protocol()

    response = await protocol._handle_tools_call(
        {"name": "test.read", "arguments": {"value": "ok"}},
        _request_context(metadata={"profile_id": "architect", "model_id": "gpt-4.1"}),
    )

    assert response["tool"] == "test.read"
    event = recorder.events[-1]
    assert event.runtime_surface == "protocol"
    assert event.requested_tool_name == "test.read"
    assert event.effective_tool_name == "test.read"
    assert event.profile_id == "architect"
    assert event.model_id == "gpt-4.1"
    assert event.tool_prompt_id == "mcp.test.read.v1"
    assert event.status == "success"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_command_nested_backend_events_are_correlated_and_marked_nested() -> None:
    protocol, recorder = _run_command_protocol()
    unsafe_request_id = "run/request id @ unsafe " + ("x" * 128)
    expected_correlation_id = f"run-{hashlib.sha256(unsafe_request_id.encode('utf-8')).hexdigest()[:32]}"

    response = await protocol._handle_tools_call(
        {"name": "run", "arguments": {"command": "ls"}},
        _request_context(metadata={"profile_id": "architect"}, request_id=unsafe_request_id),
    )

    assert response["tool"] == "run"
    assert [event.requested_tool_name for event in recorder.events] == ["fs.list", "run"]

    nested_event = recorder.events[0]
    assert nested_event.requested_tool_name == "fs.list"
    assert nested_event.nested is True
    assert nested_event.correlation_id == expected_correlation_id
    assert nested_event.profile_id == "architect"
    assert nested_event.tool_prompt_id == "mcp.fs.list.v1"

    outer_event = recorder.events[1]
    assert outer_event.requested_tool_name == "run"
    assert outer_event.nested is False
    assert outer_event.correlation_id is None
    dumped = "\n".join(event.model_dump_json() for event in recorder.events)
    assert '"command":"ls"' not in dumped
    assert unsafe_request_id not in dumped


@pytest.mark.asyncio
async def test_protocol_records_filesystem_eval_metadata_without_sensitive_payload() -> None:
    protocol, recorder = _protocol(module=_FilesystemPayloadModule())

    response = await protocol._handle_tools_call(
        {
            "name": "fs.patch",
            "arguments": {
                "diff": "--- a/private/story.txt\n+++ b/private/story.txt\n",
                "read_receipt_by_path": {"private/story.txt": "receipt-token-secret"},
            },
        },
        _request_context(metadata={"profile_id": "backend-engineer"}),
    )

    assert response["content"][0]["json"]["content"] == "SECRET FILE CONTENT"
    event = recorder.events[-1]
    dumped = event.model_dump_json()
    assert event.requested_tool_name == "fs.patch"
    assert event.action_family == "filesystem_edit"
    assert event.result_kind == "structured_filesystem_edit"
    assert event.path_filter_used is True
    assert event.truncated is False
    assert event.profile_id == "backend-engineer"
    assert "SECRET FILE CONTENT" not in dumped
    assert "receipt-token-secret" not in dumped
    assert "lease-token-secret" not in dumped
    assert "aaaaaaaaaaaaaaaa" not in dumped
    assert "bbbbbbbbbbbbbbbb" not in dumped
    assert "/Users/me" not in dumped
    assert "--- a/private" not in dumped


@pytest.mark.asyncio
async def test_protocol_records_file_policy_decision_metadata() -> None:
    protocol, recorder = _protocol(
        module=_FilesystemPayloadModule(),
        effective_policy={
            "enabled": True,
            "allowed_tools": ["fs.patch"],
            "denied_tools": [],
            "policy_document": {"path_scope_mode": "workspace_root"},
        },
        path_scope_enforcer=_FilePolicyPathScopeEnforcer(),
    )

    await protocol._handle_tools_call(
        {
            "name": "fs.patch",
            "arguments": {
                "diff": "--- a/private/story.txt\n+++ b/private/story.txt\n",
                "read_receipt_by_path": {"private/story.txt": "receipt-token-secret"},
            },
        },
        _request_context(
            metadata={
                "profile_id": "backend-engineer",
                "mcp_policy_context_enabled": True,
            }
        ),
    )

    event = recorder.events[-1]
    assert event.requested_tool_name == "fs.patch"
    assert event.file_policy_sha256_before_present is True
    assert event.file_policy_sha256_after_present is True
    assert event.file_policy_lock_lease_present is True
    assert len(event.file_policy_decisions) == 1
    decision = event.file_policy_decisions[0]
    assert decision.requested_action == "edit"
    assert decision.normalized_path == "private/story.txt"
    assert decision.grant_outcome == "allowed"
    assert decision.grant_source == "path_grants"
    assert decision.matched_grant_prefix == "private"
    assert decision.matched_grant_effect == "allow"
    assert decision.redacted is True
    dumped = event.model_dump_json()
    assert "/Users/me" not in dumped
    assert "receipt-token-secret" not in dumped
    assert "lease-token-secret" not in dumped


@pytest.mark.asyncio
async def test_protocol_records_denied_file_policy_decision_metadata() -> None:
    protocol, recorder = _protocol(
        module=_FilesystemPayloadModule(),
        effective_policy={
            "enabled": True,
            "allowed_tools": ["fs.patch"],
            "denied_tools": [],
            "policy_document": {"path_scope_mode": "workspace_root"},
        },
        path_scope_enforcer=_DenyFilePolicyPathScopeEnforcer(),
    )

    with pytest.raises(GovernanceDeniedError):
        await protocol._handle_tools_call(
            {
                "name": "fs.patch",
                "arguments": {
                    "diff": "--- a/private/story.txt\n+++ b/private/story.txt\n",
                    "expected_sha256_by_path": {"private/story.txt": "c" * 64},
                    "lock_lease_id_by_path": {"private/story.txt": "lease-token-secret"},
                    "read_receipt_by_path": {"private/story.txt": "receipt-token-secret"},
                },
            },
            _request_context(
                metadata={
                    "profile_id": "backend-engineer",
                    "mcp_policy_context_enabled": True,
                }
            ),
        )

    event = recorder.events[-1]
    assert event.execution_origin == "failed_before_execution"
    assert event.status == "denied"
    assert event.reason_code == "path_action_denied"
    assert event.grant_outcome == "denied"
    assert event.file_policy_sha256_before_present is True
    assert event.file_policy_sha256_after_present is False
    assert event.file_policy_lock_lease_present is True
    assert len(event.file_policy_decisions) == 1
    decision = event.file_policy_decisions[0]
    assert decision.normalized_path == "private/story.txt"
    assert decision.grant_outcome == "denied"
    assert decision.matched_grant_effect == "deny"
    assert decision.reason_code == "path_action_denied"
    dumped = event.model_dump_json()
    assert "/Users/me" not in dumped
    assert "cccccccccccccccc" not in dumped
    assert "lease-token-secret" not in dumped
    assert "receipt-token-secret" not in dumped


@pytest.mark.asyncio
async def test_protocol_records_prepare_denial_without_raw_error() -> None:
    protocol, recorder = _protocol()

    with pytest.raises(PermissionError):
        await protocol._handle_tools_call(
            {"name": "test.read", "arguments": {"path": "/Users/me/secret.txt"}},
            _request_context(metadata={"allowed_tools": ["other.tool"]}),
        )

    event = recorder.events[-1]
    assert event.execution_origin == "failed_before_execution"
    assert event.status == "denied"
    assert event.reason_code == "permission_denied"
    assert "/Users/me" not in event.model_dump_json()


@pytest.mark.asyncio
async def test_protocol_records_pre_hook_denial_metadata_without_raw_payload() -> None:
    protocol, recorder = _protocol(hook_manager=_DenyingToolHookManager())

    with pytest.raises(GovernanceDeniedError):
        await protocol._handle_tools_call(
            {"name": "test.read", "arguments": {"path": "/Users/example/private.txt"}},
            _request_context(metadata={"profile_id": "backend-engineer"}),
        )

    event = recorder.events[-1]
    assert event.execution_origin == "failed_before_execution"
    assert event.status == "denied"
    assert event.reason_code == "blocked_by_profile_hook"
    assert len(event.tool_hook_results) == 1
    hook_result = event.tool_hook_results[0]
    assert hook_result.phase == "pre"
    assert hook_result.hook_id == "profile-policy"
    assert hook_result.hook_order == 10
    assert hook_result.action == "deny"
    assert hook_result.status == "deny"
    assert hook_result.reason_code == "blocked_by_profile_hook"
    dumped = event.model_dump_json()
    assert "/Users/example" not in dumped
    assert "blocked by hook" not in dumped


@pytest.mark.asyncio
async def test_protocol_records_pre_hook_failure_order_without_executing_tool() -> None:
    async def fail_pre_hook(_context: ToolHookCallContext) -> ToolHookDecision:
        raise RuntimeError("policy backend failed")

    hook_manager = ConfiguredToolCallHookManager(
        [ToolHookRegistration(hook_id="policy-backend", before=fail_pre_hook, order=12)]
    )
    module = _ToolModule()
    protocol, recorder = _protocol(module=module, hook_manager=hook_manager)

    with pytest.raises(GovernanceDeniedError):
        await protocol._handle_tools_call(
            {"name": "test.read", "arguments": {"value": "ok"}},
            _request_context(metadata={"profile_id": "backend-engineer"}),
        )

    assert module.calls == 0
    event = recorder.events[-1]
    assert event.execution_origin == "failed_before_execution"
    assert event.status == "denied"
    assert event.reason_code == "tool_hook_unavailable"
    assert len(event.tool_hook_results) == 1
    hook_result = event.tool_hook_results[0]
    assert hook_result.phase == "pre"
    assert hook_result.hook_id == "policy-backend"
    assert hook_result.hook_order == 12
    assert hook_result.status == "deny"
    assert hook_result.reason_code == "tool_hook_unavailable"
    assert hook_result.error_type == "ToolHookExecutionError"


@pytest.mark.asyncio
async def test_protocol_records_post_hook_failure_without_changing_success() -> None:
    protocol, recorder = _protocol(hook_manager=_FailingPostToolHookManager())

    response = await protocol._handle_tools_call(
        {"name": "test.read", "arguments": {"value": "ok"}},
        _request_context(metadata={"profile_id": "backend-engineer"}),
    )

    assert response["tool"] == "test.read"
    event = recorder.events[-1]
    assert event.status == "success"
    assert len(event.tool_hook_results) == 1
    hook_result = event.tool_hook_results[0]
    assert hook_result.phase == "post"
    assert hook_result.status == "error"
    assert hook_result.action == "deny"
    assert hook_result.reason_code == "tool_hook_unavailable"
    assert hook_result.error_type == "RuntimeError"
    assert "/Users/example" not in event.model_dump_json()


@pytest.mark.asyncio
async def test_protocol_records_early_process_request_tool_name_error() -> None:
    protocol, recorder = _protocol()

    response = await protocol.process_request(
        {
            "jsonrpc": "2.0",
            "id": "req-1",
            "method": "tools/call",
            "params": {"name": "../secret", "arguments": {}},
        },
        _request_context(),
    )

    assert response.error.code == ErrorCode.INVALID_PARAMS
    event = recorder.events[-1]
    assert event.runtime_surface == "protocol"
    assert event.requested_tool_name == "unknown"
    assert event.status == "invalid_params"
    assert event.execution_origin == "failed_before_execution"


@pytest.mark.asyncio
async def test_protocol_recorder_failure_does_not_change_tool_response() -> None:
    recorder = _FailingToolUseRecorder()
    protocol, _ = _protocol(recorder=recorder)

    response = await protocol._handle_tools_call(
        {"name": "test.read", "arguments": {}},
        _request_context(),
    )

    assert response["tool"] == "test.read"
    assert recorder.called is True


@pytest.mark.asyncio
async def test_protocol_event_build_failure_preserves_process_request_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, recorder = _protocol()

    def fail_event(**_kwargs: Any) -> ToolUseEvent:
        raise ValueError("telemetry build failed")

    monkeypatch.setattr(protocol, "_build_tool_use_event", fail_event)

    response = await protocol.process_request(
        {
            "jsonrpc": "2.0",
            "id": "req-1",
            "method": "tools/call",
            "params": {"name": "../secret", "arguments": {}},
        },
        _request_context(),
    )

    assert response.error.code == ErrorCode.INVALID_PARAMS
    assert "Invalid tool name" in response.error.message
    assert recorder.events == []


@pytest.mark.asyncio
async def test_protocol_event_build_failure_preserves_prepare_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, recorder = _protocol()

    def fail_event(**_kwargs: Any) -> ToolUseEvent:
        raise ValueError("telemetry build failed")

    monkeypatch.setattr(protocol, "_build_tool_use_event", fail_event)

    with pytest.raises(PermissionError):
        await protocol._handle_tools_call(
            {"name": "test.read", "arguments": {"path": "/Users/me/secret.txt"}},
            _request_context(metadata={"allowed_tools": ["other.tool"]}),
        )

    assert recorder.events == []


@pytest.mark.asyncio
async def test_protocol_event_build_failure_preserves_tool_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, recorder = _protocol()

    def fail_event(**_kwargs: Any) -> ToolUseEvent:
        raise ValueError("telemetry build failed")

    monkeypatch.setattr(protocol, "_build_tool_use_event", fail_event)

    response = await protocol._handle_tools_call(
        {"name": "test.read", "arguments": {}},
        _request_context(),
    )

    assert response["tool"] == "test.read"
    assert recorder.events == []


@pytest.mark.asyncio
async def test_restoring_protocol_tool_use_event_builder_does_not_recurse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, recorder = _protocol()
    original_build_event = protocol._build_tool_use_event

    def fail_event(**_kwargs: Any) -> ToolUseEvent:
        raise ValueError("telemetry build failed")

    monkeypatch.setattr(protocol, "_build_tool_use_event", fail_event)
    protocol._build_tool_use_event = original_build_event

    response = await protocol._handle_tools_call(
        {"name": "test.read", "arguments": {}},
        _request_context(),
    )

    assert response["tool"] == "test.read"
    assert len(recorder.events) == 1
    assert recorder.events[0].requested_tool_name == "test.read"


@pytest.mark.asyncio
async def test_protocol_event_build_failure_preserves_tool_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, recorder = _protocol(module=_ToolModule(fail=ValueError("original failure")))

    def fail_event(**_kwargs: Any) -> ToolUseEvent:
        raise RuntimeError("telemetry build failed")

    monkeypatch.setattr(protocol, "_build_tool_use_event", fail_event)

    with pytest.raises(InvalidParamsException, match="original failure"):
        await protocol._handle_tools_call(
            {"name": "test.read", "arguments": {}},
            _request_context(),
        )

    assert recorder.events == []


@pytest.mark.asyncio
async def test_protocol_skips_when_tool_use_already_observed() -> None:
    protocol, recorder = _protocol()

    response = await protocol._handle_tools_call(
        {"name": "test.read", "arguments": {}},
        _request_context(metadata={"mcp_tool_use_observed": True}),
    )

    assert response["tool"] == "test.read"
    assert recorder.events == []


@pytest.mark.asyncio
async def test_protocol_records_execution_failure_without_raw_arguments() -> None:
    protocol, recorder = _protocol(module=_ToolModule(fail=ValueError("bad /Users/me/secret.txt")))

    with pytest.raises(InvalidParamsException):
        await protocol._handle_tools_call(
            {"name": "test.read", "arguments": {"path": "/Users/me/secret.txt"}},
            _request_context(),
        )

    event = recorder.events[-1]
    assert event.status == "invalid_params"
    assert event.reason_code == "invalid_params"
    assert "/Users/me" not in event.model_dump_json()


@pytest.mark.asyncio
async def test_protocol_records_unavailable_failure_origin_as_unavailable() -> None:
    protocol, recorder = _protocol(module=_ToolModule(fail=LookupError("missing tool")))

    with pytest.raises(LookupError):
        await protocol._handle_tools_call(
            {"name": "test.read", "arguments": {}},
            _request_context(),
        )

    event = recorder.events[-1]
    assert event.status == "unavailable"
    assert event.reason_code == "tool_unavailable"
    assert event.execution_origin == "unavailable"


@pytest.mark.asyncio
async def test_protocol_records_idempotency_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_DISABLE_WRITE_TOOLS", "false")
    from tldw_Server_API.app.core.MCP_unified.config import get_config

    get_config.cache_clear()  # type: ignore[attr-defined]
    module = _ToolModule(write=True)
    hooks = _RecordingPostToolHookManager()
    protocol, recorder = _protocol(module=module, hook_manager=hooks)
    metrics = _RecordingMetrics()
    protocol.metrics = metrics
    audit_calls: list[str] = []

    def _record_audit(*_args: Any, **kwargs: Any) -> None:
        audit_calls.append(str(kwargs["status"]))

    monkeypatch.setattr(protocol._tool_execution_reporter, "audit_tool_event", _record_audit)
    context = _request_context()
    params = {
        "name": "test.write",
        "arguments": {"value": "A"},
        "idempotencyKey": "idem-1",
    }

    first = await protocol._handle_tools_call(params, context)
    second = await protocol._handle_tools_call(params, context)

    assert first["content"] == second["content"]
    assert module.calls == 1
    assert metrics.module_operations == [True]
    assert audit_calls == ["success"]
    assert len(hooks.after_contexts) == 1
    cached_events = [event for event in recorder.events if event.execution_origin == "cached"]
    assert len(cached_events) == 1
    assert cached_events[0].idempotency_replay is True


@pytest.mark.asyncio
async def test_idempotent_success_survives_metrics_audit_and_reporting_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCP_DISABLE_WRITE_TOOLS", "false")
    from tldw_Server_API.app.core.MCP_unified.config import get_config

    get_config.cache_clear()  # type: ignore[attr-defined]
    module = _ToolModule(write=True)
    protocol, _recorder = _protocol(module=module)
    protocol.metrics = _FailingObserverMetrics()

    def _fail_audit(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("private audit detail")

    def _fail_event(**_kwargs: Any) -> ToolUseEvent:
        raise RuntimeError("private reporting detail")

    monkeypatch.setattr(protocol._tool_execution_reporter, "audit_tool_event", _fail_audit)
    monkeypatch.setattr(protocol._tool_execution_reporter, "build_event", _fail_event)
    params = {
        "name": "test.write",
        "arguments": {"value": "A"},
        "idempotencyKey": "observer-failures",
    }

    first = await protocol._handle_tools_call(params, _request_context())
    replay = await protocol._handle_tools_call(params, _request_context())

    assert first["content"] == replay["content"]
    assert module.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "observer",
    ["module_metrics", "audit", "post_hook", "idempotency_metrics", "reporting"],
)
async def test_exotic_success_observer_cannot_replace_committed_replay(
    monkeypatch: pytest.MonkeyPatch,
    observer: str,
) -> None:
    monkeypatch.setenv("MCP_DISABLE_WRITE_TOOLS", "false")
    from tldw_Server_API.app.core.MCP_unified.config import get_config

    get_config.cache_clear()  # type: ignore[attr-defined]
    module = _ToolModule(write=True)
    protocol, _recorder = _protocol(module=module)
    protocol.metrics = _ExoticObserverMetrics(observer)

    if observer == "audit":
        monkeypatch.setattr(
            protocol._tool_execution_reporter,
            "audit_tool_event",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(ExoticObserverError("private audit detail")),
        )
    if observer == "reporting":
        monkeypatch.setattr(
            protocol._tool_execution_reporter,
            "build_event",
            lambda **_kwargs: (_ for _ in ()).throw(ExoticObserverError("private reporting detail")),
        )
    if observer == "post_hook":

        async def _fail_post_hook(**_kwargs: Any) -> None:
            raise ExoticObserverError("private post-hook detail")

        protocol._run_post_tool_hooks = _fail_post_hook

    params = {
        "name": "test.write",
        "arguments": {"value": "A"},
        "idempotencyKey": f"exotic-success-{observer}",
    }

    first = await protocol._handle_tools_call(params, _request_context())
    replay = await protocol._handle_tools_call(params, _request_context())

    assert first["content"] == replay["content"]
    assert module.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("observer", ["module_metrics", "audit", "post_hook", "reporting"])
async def test_exotic_failure_observer_cannot_replace_original_failure(
    monkeypatch: pytest.MonkeyPatch,
    observer: str,
) -> None:
    original = _OriginalToolError("original module failure")
    protocol, _recorder = _protocol(module=_ToolModule(fail=original))
    protocol.metrics = _ExoticObserverMetrics(observer)

    if observer == "audit":
        monkeypatch.setattr(
            protocol._tool_execution_reporter,
            "audit_tool_event",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(ExoticObserverError("private audit detail")),
        )
    if observer == "reporting":
        monkeypatch.setattr(
            protocol._tool_execution_reporter,
            "build_event",
            lambda **_kwargs: (_ for _ in ()).throw(ExoticObserverError("private reporting detail")),
        )
    if observer == "post_hook":

        async def _fail_post_hook(**_kwargs: Any) -> None:
            raise ExoticObserverError("private post-hook detail")

        protocol._run_post_tool_hooks = _fail_post_hook

    with pytest.raises(_OriginalToolError) as caught:
        await protocol._handle_tools_call(
            {"name": "test.read", "arguments": {"value": "A"}},
            _request_context(),
        )

    assert caught.value is original


@pytest.mark.asyncio
async def test_exotic_invalid_params_metric_cannot_replace_original_failure() -> None:
    protocol, _recorder = _protocol(module=_ToolModule(fail=ValueError("original invalid")))
    protocol.metrics = _ExoticObserverMetrics("invalid_params_metrics")

    with pytest.raises(InvalidParamsException, match="original invalid"):
        await protocol._handle_tools_call(
            {"name": "test.read", "arguments": {"value": "A"}},
            _request_context(),
        )


def test_reporting_classifier_recognizes_expected_failure_shape_without_message_access() -> None:
    from tldw_Server_API.app.core.MCP_unified.execution_outcomes import (
        ExpectedToolFailure,
        ExpectedToolFailureReason,
    )

    failure = ExpectedToolFailure(ExpectedToolFailureReason.IDEMPOTENCY_UNAVAILABLE)

    assert classify_tool_use_exception(failure) == (
        "error",
        "idempotency_unavailable",
    )


def test_reporting_classifier_ignores_throwing_expected_failure_descriptor() -> None:
    class _ThrowingExpectedFailureShape(LookupError):
        @property
        def reason(self) -> object:
            raise LookupError("SENTINEL_CLASSIFIER_SECRET")

        def __str__(self) -> str:
            raise AssertionError("classifier must not render exception text")

    assert classify_tool_use_exception(_ThrowingExpectedFailureShape()) == (
        "unavailable",
        "tool_unavailable",
    )


def test_reporting_classifier_propagates_descriptor_cancellation() -> None:
    cancellation = asyncio.CancelledError()

    class _CancellingExpectedFailureShape(LookupError):
        @property
        def reason(self) -> object:
            raise cancellation

        def __str__(self) -> str:
            raise AssertionError("classifier must not render exception text")

    with pytest.raises(asyncio.CancelledError) as caught:
        classify_tool_use_exception(_CancellingExpectedFailureShape())

    assert caught.value is cancellation


def test_reporting_classifier_rejects_spoofed_and_mismatched_expected_shapes() -> None:
    from tldw_Server_API.app.core.MCP_unified.execution_outcomes import (
        ExpectedToolFailureReason,
    )

    class _SpoofedExpectedFailureShape(LookupError):
        reason = object()
        reason_code = "idempotency_unavailable"
        public_message = "Idempotent execution is temporarily unavailable."
        breaker_action = "ignore"

        def __str__(self) -> str:
            raise AssertionError("classifier must not render exception text")

    class _MismatchedExpectedFailureShape(LookupError):
        reason = ExpectedToolFailureReason.IDEMPOTENCY_UNAVAILABLE
        reason_code = "dependency_unavailable"
        public_message = reason.public_message
        breaker_action = reason.breaker_action

        def __str__(self) -> str:
            raise AssertionError("classifier must not render exception text")

    expected_fallback = ("unavailable", "tool_unavailable")
    assert classify_tool_use_exception(_SpoofedExpectedFailureShape()) == expected_fallback
    assert classify_tool_use_exception(_MismatchedExpectedFailureShape()) == expected_fallback


def test_reporting_classifier_sanitizes_valid_host_neutral_expected_failure_shape() -> None:
    class _ForeignExpectedReason(Enum):
        DEPENDENCY = (
            "private reason/SENTINEL_CLASSIFIER_SECRET",
            "A bounded public message.",
            "record_failure",
        )

        @property
        def reason_code(self) -> str:
            return self.value[0]

        @property
        def public_message(self) -> str:
            return self.value[1]

        @property
        def breaker_action(self) -> str:
            return self.value[2]

    class _ForeignExpectedFailure(Exception):
        reason = _ForeignExpectedReason.DEPENDENCY
        reason_code = reason.reason_code
        public_message = reason.public_message
        breaker_action = reason.breaker_action

        def __str__(self) -> str:
            raise AssertionError("classifier must not render exception text")

    assert classify_tool_use_exception(_ForeignExpectedFailure()) == (
        "error",
        "unknown",
    )


@pytest.mark.parametrize("cache_backend", ["local", "redis"])
@pytest.mark.asyncio
async def test_expected_write_failure_returns_exact_tool_error_without_result_cache(
    monkeypatch: pytest.MonkeyPatch,
    cache_backend: str,
) -> None:
    monkeypatch.setenv("MCP_DISABLE_WRITE_TOOLS", "false")
    from tldw_Server_API.app.core.MCP_unified.config import get_config

    get_config.cache_clear()  # type: ignore[attr-defined]
    module = _ExpectedFailureWriteModule()
    protocol, _ = _protocol(module=module)
    redis = _MemoryRedis()
    protocol._idempotency._redis_attempted = True
    protocol._idempotency._redis_ready = cache_backend == "redis"
    protocol._idempotency._redis_client = redis if cache_backend == "redis" else None
    request = {
        "jsonrpc": "2.0",
        "id": f"expected-failure-{cache_backend}",
        "method": "tools/call",
        "params": {
            "name": "test.write",
            "arguments": {"value": "A"},
            "idempotencyKey": "expected-failure-key",
        },
    }

    try:
        first = await protocol.process_request(request, _request_context())
        second = await protocol.process_request(request, _request_context())
    finally:
        get_config.cache_clear()  # type: ignore[attr-defined]

    assert first is not None
    assert second is not None
    assert first.error is None
    assert second.error is None
    assert module.calls == 2
    assert protocol._idempotency._local_cache == {}
    assert not any(key.startswith("mcp:idemp:result:") for key in redis.values)

    for response in (first, second):
        payload = response.result
        execution_eval = payload["eval"]
        assert payload == {
            "content": [
                {
                    "type": "json",
                    "json": {
                        "status": "failed",
                        "reason_code": "idempotency_unavailable",
                        "message": "Idempotent execution is temporarily unavailable.",
                    },
                }
            ],
            "isError": True,
            "module": "test_module",
            "tool": "test.write",
            "eval": execution_eval,
        }


@pytest.mark.asyncio
async def test_fail_closed_rate_limit_expected_failure_is_observed_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.execution_outcomes import (
        ExpectedToolFailureReason,
    )

    module = _FailClosedToolModule()
    hooks = _RecordingPostToolHookManager()
    protocol, recorder = _protocol(
        module=module,
        rate_limiter=_UnavailableRateLimiter(),
        hook_manager=hooks,
    )
    metrics = _RecordingMetrics()
    protocol.metrics = metrics
    audit_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        protocol._tool_execution_reporter,
        "audit_tool_event",
        lambda *_args, **kwargs: audit_calls.append(kwargs),
    )

    result = await protocol._handle_tools_call(
        {"name": "test.read", "arguments": {"value": "A"}},
        _request_context(),
    )

    reason = ExpectedToolFailureReason.RATE_LIMIT_UNAVAILABLE
    _assert_exact_expected_failure_payload(
        result,
        reason_code=reason.reason_code,
        message=reason.public_message,
    )
    assert module.calls == 0
    assert metrics.module_operations == []
    assert [call["status"] for call in audit_calls] == ["failure"]
    assert [call["reason_code"] for call in audit_calls] == [reason.reason_code]
    assert [context.status for context in hooks.after_contexts] == ["failure"]
    assert len(recorder.events) == 1
    assert recorder.events[0].status == "error"
    assert recorder.events[0].reason_code == reason.reason_code
    assert recorder.events[0].execution_origin == "failed_before_execution"


@pytest.mark.parametrize(
    "reason_name",
    ["IDEMPOTENCY_IN_PROGRESS", "IDEMPOTENCY_UNAVAILABLE"],
)
@pytest.mark.asyncio
async def test_expected_idempotency_failure_is_observed_before_module_execution(
    monkeypatch: pytest.MonkeyPatch,
    reason_name: str,
) -> None:
    monkeypatch.setenv("MCP_DISABLE_WRITE_TOOLS", "false")
    from tldw_Server_API.app.core.MCP_unified.config import get_config
    from tldw_Server_API.app.core.MCP_unified.execution_outcomes import (
        ExpectedToolFailure,
        ExpectedToolFailureReason,
    )

    get_config.cache_clear()  # type: ignore[attr-defined]
    module = _ToolModule(write=True)
    hooks = _RecordingPostToolHookManager()
    protocol, recorder = _protocol(module=module, hook_manager=hooks)
    metrics = _RecordingMetrics()
    protocol.metrics = metrics
    audit_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        protocol._tool_execution_reporter,
        "audit_tool_event",
        lambda *_args, **kwargs: audit_calls.append(kwargs),
    )
    reason = ExpectedToolFailureReason[reason_name]

    async def _fail_idempotency(*_args: Any, **_kwargs: Any) -> Any:
        raise ExpectedToolFailure(reason)

    monkeypatch.setattr(protocol._idempotency, "execute", _fail_idempotency)
    try:
        result = await protocol._handle_tools_call(
            {
                "name": "test.write",
                "arguments": {"value": "A"},
                "idempotencyKey": f"expected-{reason.reason_code}",
            },
            _request_context(),
        )
    finally:
        get_config.cache_clear()  # type: ignore[attr-defined]

    _assert_exact_expected_failure_payload(
        result,
        reason_code=reason.reason_code,
        message=reason.public_message,
        tool="test.write",
    )
    assert module.calls == 0
    assert metrics.module_operations == []
    assert [call["reason_code"] for call in audit_calls] == [reason.reason_code]
    assert [context.status for context in hooks.after_contexts] == ["failure"]
    assert len(recorder.events) == 1
    assert recorder.events[0].status == "error"
    assert recorder.events[0].reason_code == reason.reason_code
    assert recorder.events[0].execution_origin == "failed_before_execution"


@pytest.mark.asyncio
async def test_second_live_binding_expected_failure_remains_before_module_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.execution_outcomes import (
        ExpectedToolFailure,
        ExpectedToolFailureReason,
    )

    module = _ToolModule()
    hooks = _RecordingPostToolHookManager()
    protocol, recorder = _protocol(module=module, hook_manager=hooks)
    metrics = _RecordingMetrics()
    protocol.metrics = metrics
    audit_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        protocol._tool_execution_reporter,
        "audit_tool_event",
        lambda *_args, **kwargs: audit_calls.append(kwargs),
    )
    prepared = await protocol.prepare_tool_call(
        params={"name": "test.read", "arguments": {"value": "A"}},
        context=_request_context(),
    )
    original_verify = protocol._tool_execution_security.verify_prepared_tool_call
    verify_calls = 0

    async def _fail_second_verification(*args: Any, **kwargs: Any) -> None:
        nonlocal verify_calls
        verify_calls += 1
        if verify_calls == 2:
            raise ExpectedToolFailure(ExpectedToolFailureReason.STALE_PREPARED_CALL)
        await original_verify(*args, **kwargs)

    monkeypatch.setattr(
        protocol._tool_execution_security,
        "verify_prepared_tool_call",
        _fail_second_verification,
    )

    result = await protocol.execute_prepared_tool_call(prepared)

    reason = ExpectedToolFailureReason.STALE_PREPARED_CALL
    _assert_exact_expected_failure_payload(
        result,
        reason_code=reason.reason_code,
        message=reason.public_message,
    )
    assert verify_calls == 2
    assert module.calls == 0
    assert metrics.module_operations == []
    assert [call["reason_code"] for call in audit_calls] == [reason.reason_code]
    assert [context.status for context in hooks.after_contexts] == ["failure"]
    assert len(recorder.events) == 1
    assert recorder.events[0].status == "error"
    assert recorder.events[0].reason_code == reason.reason_code
    assert recorder.events[0].execution_origin == "failed_before_execution"


@pytest.mark.asyncio
async def test_ignored_expected_module_failure_observes_once_and_leaves_breaker_neutral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.execution_outcomes import (
        ExpectedToolFailure,
        ExpectedToolFailureReason,
    )

    reason = ExpectedToolFailureReason.IDEMPOTENCY_UNAVAILABLE
    module = _ExpectedFailureBreakerModule(ExpectedToolFailure(reason))
    hooks = _RecordingPostToolHookManager()
    protocol, recorder = _protocol(module=module, hook_manager=hooks)
    metrics = _RecordingMetrics()
    protocol.metrics = metrics
    audit_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        protocol._tool_execution_reporter,
        "audit_tool_event",
        lambda *_args, **kwargs: audit_calls.append(kwargs),
    )
    breaker = module._circuit_breaker
    breaker_before = (
        breaker.failure_count,
        breaker.success_count,
        breaker._state,
        breaker._current_recovery_timeout,
    )

    result = await protocol._handle_tools_call(
        {"name": "test.read", "arguments": {"value": "A"}},
        _request_context(),
    )

    _assert_exact_expected_failure_payload(
        result,
        reason_code=reason.reason_code,
        message=reason.public_message,
    )
    assert module.calls == 1
    assert metrics.module_operations == [False]
    assert [call["reason_code"] for call in audit_calls] == [reason.reason_code]
    assert [context.status for context in hooks.after_contexts] == ["failure"]
    assert len(recorder.events) == 1
    assert recorder.events[0].status == "error"
    assert recorder.events[0].reason_code == reason.reason_code
    assert recorder.events[0].execution_origin == "executed"
    assert (
        breaker.failure_count,
        breaker.success_count,
        breaker._state,
        breaker._current_recovery_timeout,
    ) == breaker_before


@pytest.mark.asyncio
async def test_counted_dependency_failure_observes_and_reopens_breaker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.execution_outcomes import (
        ExpectedToolFailure,
        ExpectedToolFailureReason,
    )

    reason = ExpectedToolFailureReason.DEPENDENCY_UNAVAILABLE
    module = _ExpectedFailureBreakerModule(
        ExpectedToolFailure(reason),
        recovery_timeout=1,
    )
    hooks = _RecordingPostToolHookManager()
    protocol, recorder = _protocol(module=module, hook_manager=hooks)
    metrics = _RecordingMetrics()
    protocol.metrics = metrics
    audit_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        protocol._tool_execution_reporter,
        "audit_tool_event",
        lambda *_args, **kwargs: audit_calls.append(kwargs),
    )

    first = await protocol._handle_tools_call(
        {"name": "test.read", "arguments": {"value": "A"}},
        _request_context(request_id="counted-first"),
    )
    breaker = module._circuit_breaker
    assert breaker.failure_count == 1
    assert breaker._state == "open"
    breaker._opened_at = time.time() - 2.0
    second = await protocol._handle_tools_call(
        {"name": "test.read", "arguments": {"value": "A"}},
        _request_context(request_id="counted-second"),
    )

    for result in (first, second):
        _assert_exact_expected_failure_payload(
            result,
            reason_code=reason.reason_code,
            message=reason.public_message,
        )
    assert module.calls == 2
    assert breaker.failure_count == 2
    assert breaker._state == "open"
    assert breaker._current_recovery_timeout == 2.0
    assert metrics.module_operations == [False, False]
    assert [call["reason_code"] for call in audit_calls] == [
        reason.reason_code,
        reason.reason_code,
    ]
    assert [context.status for context in hooks.after_contexts] == [
        "failure",
        "failure",
    ]
    assert len(recorder.events) == 2
    assert all(event.status == "error" for event in recorder.events)
    assert all(event.reason_code == reason.reason_code for event in recorder.events)
    assert all(event.execution_origin == "executed" for event in recorder.events)


@pytest.mark.asyncio
async def test_unexpected_module_failure_keeps_generic_error_and_breaker_counting() -> None:
    original = RuntimeError("SENTINEL_UNEXPECTED_PRIVATE_DETAIL")
    module = _ExpectedFailureBreakerModule(original)
    protocol, recorder = _protocol(module=module)
    metrics = _RecordingMetrics()
    protocol.metrics = metrics

    with pytest.raises(RuntimeError, match="^tool_execution_error$"):
        await protocol._handle_tools_call(
            {"name": "test.read", "arguments": {"value": "A"}},
            _request_context(),
        )

    assert module.calls == 1
    assert module._circuit_breaker.failure_count == 1
    assert module._circuit_breaker._state == "open"
    assert metrics.module_operations == [False]
    assert len(recorder.events) == 1
    assert recorder.events[0].status == "error"
    assert recorder.events[0].execution_origin == "executed"


@pytest.mark.parametrize("cache_backend", ["local", "redis"])
@pytest.mark.asyncio
async def test_pre_success_cancellation_propagates_without_envelope_or_result_cache(
    monkeypatch: pytest.MonkeyPatch,
    cache_backend: str,
) -> None:
    monkeypatch.setenv("MCP_DISABLE_WRITE_TOOLS", "false")
    from tldw_Server_API.app.core.MCP_unified.config import get_config

    get_config.cache_clear()  # type: ignore[attr-defined]
    cancellation = asyncio.CancelledError("SENTINEL_CANCELLATION_PRIVATE_DETAIL")
    module = _ToolModule(write=True, fail=cancellation)
    protocol, recorder = _protocol(module=module)
    redis = _MemoryRedis()
    protocol._idempotency._redis_attempted = True
    protocol._idempotency._redis_ready = cache_backend == "redis"
    protocol._idempotency._redis_client = redis if cache_backend == "redis" else None
    params = {
        "name": "test.write",
        "arguments": {"value": "A"},
        "idempotencyKey": f"cancel-{cache_backend}",
    }

    try:
        with pytest.raises(asyncio.CancelledError) as caught:
            await protocol._handle_tools_call(params, _request_context())
    finally:
        get_config.cache_clear()  # type: ignore[attr-defined]

    assert caught.value is cancellation
    assert protocol._idempotency._local_cache == {}
    assert not any(key.startswith("mcp:idemp:result:") for key in redis.values)
    assert recorder.events == []


@pytest.mark.parametrize("observer", ["module_metrics", "audit", "post_hook", "reporting"])
@pytest.mark.asyncio
async def test_exotic_observer_failure_cannot_replace_expected_failure_payload(
    monkeypatch: pytest.MonkeyPatch,
    observer: str,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.execution_outcomes import (
        ExpectedToolFailure,
        ExpectedToolFailureReason,
    )

    calls: list[str] = []
    reason = ExpectedToolFailureReason.IDEMPOTENCY_UNAVAILABLE
    module = _ExpectedFailureBreakerModule(ExpectedToolFailure(reason))
    protocol, _recorder = _protocol(module=module)

    class _ExpectedObserverMetrics(_NoopMetrics):
        def record_module_operation(self, **_kwargs: Any) -> None:
            calls.append("module_metrics")
            if observer == "module_metrics":
                raise ExoticObserverError("SENTINEL_EXPECTED_OBSERVER_SECRET")

    protocol.metrics = _ExpectedObserverMetrics()

    def _audit(*_args: Any, **_kwargs: Any) -> None:
        calls.append("audit")
        if observer == "audit":
            raise ExoticObserverError("SENTINEL_EXPECTED_OBSERVER_SECRET")

    async def _post_hook(**_kwargs: Any) -> None:
        calls.append("post_hook")
        if observer == "post_hook":
            raise ExoticObserverError("SENTINEL_EXPECTED_OBSERVER_SECRET")

    original_build_event = protocol._tool_execution_reporter.build_event

    def _build_event(**kwargs: Any) -> ToolUseEvent:
        calls.append("reporting")
        if observer == "reporting":
            raise ExoticObserverError("SENTINEL_EXPECTED_OBSERVER_SECRET")
        return original_build_event(**kwargs)

    monkeypatch.setattr(protocol._tool_execution_reporter, "audit_tool_event", _audit)
    monkeypatch.setattr(protocol._tool_execution_reporter, "build_event", _build_event)
    protocol._run_post_tool_hooks = _post_hook

    result = await protocol._handle_tools_call(
        {"name": "test.read", "arguments": {"value": "A"}},
        _request_context(),
    )

    _assert_exact_expected_failure_payload(
        result,
        reason_code=reason.reason_code,
        message=reason.public_message,
    )
    assert observer in calls


@pytest.mark.asyncio
async def test_expected_failure_observers_receive_detached_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified.execution_outcomes import (
        ExpectedToolFailure,
        ExpectedToolFailureReason,
    )

    reason = ExpectedToolFailureReason.IDEMPOTENCY_UNAVAILABLE
    module = _ExpectedFailureBreakerModule(ExpectedToolFailure(reason))
    protocol, recorder = _protocol(module=module)
    observed: list[dict[str, Any]] = []

    async def _mutate_post_hook(**kwargs: Any) -> None:
        observed.append(kwargs)
        kwargs["tool_def"]["metadata"]["category"] = "SENTINEL_MUTATED_CATEGORY"

    protocol._run_post_tool_hooks = _mutate_post_hook

    result = await protocol._handle_tools_call(
        {"name": "test.read", "arguments": {"value": "A"}},
        _request_context(),
    )

    _assert_exact_expected_failure_payload(
        result,
        reason_code=reason.reason_code,
        message=reason.public_message,
    )
    assert len(observed) == 1
    assert len(recorder.events) == 1
    assert recorder.events[0].category == "read"


@pytest.mark.asyncio
async def test_expected_failure_sentinel_is_absent_from_all_observable_surfaces() -> None:
    from tldw_Server_API.app.core.MCP_unified.execution_outcomes import (
        ExpectedToolFailure,
        ExpectedToolFailureReason,
    )

    sentinel = "SENTINEL_EXPECTED_FAILURE_INTERNAL_SECRET"
    failure = ExpectedToolFailure(ExpectedToolFailureReason.DEPENDENCY_UNAVAILABLE)
    failure.__cause__ = RuntimeError(sentinel)
    module = _ExpectedFailureBreakerModule(failure)
    hooks = _RecordingPostToolHookManager()
    telemetry = _RecordingTelemetry()
    protocol, recorder = _protocol(
        module=module,
        hook_manager=hooks,
        telemetry_provider=telemetry,
    )
    captured: list[Any] = []
    sink_id = logger.add(lambda message: captured.append(message.record), level="DEBUG")
    try:
        result = await protocol._handle_tools_call(
            {"name": "test.read", "arguments": {"value": "A"}},
            _request_context(),
        )
    finally:
        logger.remove(sink_id)

    reason = ExpectedToolFailureReason.DEPENDENCY_UNAVAILABLE
    _assert_exact_expected_failure_payload(
        result,
        reason_code=reason.reason_code,
        message=reason.public_message,
    )
    audit_records = [record for record in captured if record["extra"].get("audit")]
    assert any(record["extra"].get("reason_code") == reason.reason_code for record in audit_records)
    surfaces = repr(
        {
            "payload_and_eval": result,
            "events": recorder.events,
            "telemetry": [span.attributes for span in telemetry.spans],
            "hooks": hooks.after_contexts,
            "audit": audit_records,
            "logs": captured,
        }
    )
    assert sentinel not in surfaces
    assert "ExpectedToolFailure" not in repr(result)


@pytest.mark.asyncio
async def test_protocol_process_request_does_not_double_record_handler_rate_limit() -> None:
    protocol, recorder = _protocol(rate_limiter=_ToolOnlyRateLimiter())

    with pytest.raises(RateLimitExceeded):
        await protocol.process_request(
            {
                "jsonrpc": "2.0",
                "id": "req-1",
                "method": "tools/call",
                "params": {"name": "test.read", "arguments": {}},
            },
            _request_context(),
        )

    rate_limited_events = [event for event in recorder.events if event.status == "rate_limited"]
    assert len(rate_limited_events) == 1
    assert rate_limited_events[0].execution_origin == "failed_before_execution"


@pytest.mark.asyncio
async def test_tools_call_coarse_authorization_denial_records_denied_tool_use() -> None:
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPRequest

    recorder = _RecordingToolUseRecorder()
    protocol, _ = _protocol(recorder=recorder)

    async def deny_request(_request: MCPRequest, _context: RequestContext) -> bool:
        return False

    protocol._check_authorization = deny_request  # type: ignore[method-assign]
    context = RequestContext(request_id="coarse-deny", user_id="u1", client_id="c1")
    request = MCPRequest(
        method="tools/call",
        params={"name": "test.read", "arguments": {"value": "x"}},
        id="coarse-deny",
    )

    response = await protocol.process_request(request, context)

    assert response.error is not None
    assert response.error.code == ErrorCode.AUTHORIZATION_ERROR
    assert len(recorder.events) == 1
    event = recorder.events[-1]
    assert event.runtime_surface == "protocol"
    assert event.requested_tool_name == "test.read"
    assert event.status == "denied"
    assert event.reason_code == "permission_denied"
    assert event.execution_origin == "failed_before_execution"


@pytest.mark.asyncio
async def test_tools_call_deep_authorization_denial_records_prepare_failure() -> None:
    recorder = _RecordingToolUseRecorder()
    protocol, _ = _protocol(recorder=recorder)

    async def allow_request(_request: Any, _context: RequestContext) -> bool:
        return True

    async def deny_prepare(
        *,
        params: dict[str, Any],
        context: RequestContext,
        idempotency_key: str | None = None,
    ) -> Any:
        del params, context, idempotency_key
        raise PermissionError("Permission denied for tool: test.read")

    protocol._check_authorization = allow_request  # type: ignore[method-assign]
    protocol.prepare_tool_call = deny_prepare  # type: ignore[method-assign]
    context = RequestContext(request_id="deep-deny", user_id="u1", client_id="c1")

    response = await protocol.process_request(
        {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": "test.read", "arguments": {"value": "x"}},
            "id": "deep-deny",
        },
        context,
    )

    assert response.error is not None
    assert response.error.code == ErrorCode.AUTHORIZATION_ERROR
    assert len(recorder.events) == 1
    event = recorder.events[-1]
    assert event.runtime_surface == "protocol"
    assert event.requested_tool_name == "test.read"
    assert event.status == "denied"
    assert event.reason_code == "permission_denied"
    assert event.execution_origin == "failed_before_execution"


@pytest.mark.asyncio
async def test_restoring_public_prepare_tool_call_does_not_recurse() -> None:
    protocol, _ = _protocol()
    original_prepare = protocol.prepare_tool_call

    async def patched_prepare(
        *,
        params: dict[str, Any],
        context: RequestContext,
        idempotency_key: str | None = None,
    ) -> Any:
        return await original_prepare(
            params=params,
            context=context,
            idempotency_key=idempotency_key,
        )

    protocol.prepare_tool_call = patched_prepare  # type: ignore[method-assign]
    protocol.prepare_tool_call = original_prepare  # type: ignore[method-assign]

    result = await protocol._handle_tools_call(
        {"name": "test.read", "arguments": {"value": "x"}},
        RequestContext(request_id="restore-prepare", user_id="u1", client_id="c1"),
    )

    assert result["tool"] == "test.read"


@pytest.mark.asyncio
async def test_replaced_tool_name_regex_syncs_reporting_and_security_helpers() -> None:
    protocol, _ = _protocol()
    protocol._tool_name_re = re.compile(r"^test:[A-Za-z]+$")
    context = RequestContext(request_id="regex-sync", user_id="u1", client_id="c1")

    assert protocol._safe_tool_use_name("test.read") == "unknown"
    with pytest.raises(InvalidParamsException, match="Invalid tool name"):
        await protocol.prepare_tool_call(
            params={"name": "test.read", "arguments": {"value": "x"}},
            context=context,
        )

    assert protocol._safe_tool_use_name("test:read") == "test:read"
