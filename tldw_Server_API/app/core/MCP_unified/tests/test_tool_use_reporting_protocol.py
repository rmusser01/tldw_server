"""Tests for protocol-side MCP tool-use reporting capture."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any

import pytest
from mcp_unified.interfaces.runtime import ToolHookCallContext, ToolHookDecision
from mcp_unified.tool_hooks import ConfiguredToolCallHookManager, ToolHookRegistration
from mcp_unified.tool_use_reporting.models import MAX_FILE_POLICY_DECISIONS, ToolUseEvent

from tldw_Server_API.app.core.MCP_unified.auth.rate_limiter import RateLimitExceeded
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


def _protocol(
    *,
    module: _ToolModule | None = None,
    recorder: Any | None = None,
    rate_limiter: Any | None = None,
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
        telemetry_provider=_Telemetry(),
        tool_catalog_provider=object(),
        effective_policy_resolver=_StaticEffectivePolicyResolver(effective_policy),
        approval_evaluator=_AllowApprovalEvaluator(),
        path_scope_enforcer=path_scope_enforcer or _FilePolicyPathScopeEnforcer(),
        redis_client_factory=lambda **kwargs: None,
        tool_use_recorder=recorder,
        tool_call_hook_manager=hook_manager,
    )
    protocol = MCPProtocol(dependencies=deps)
    return protocol, recorder


def _request_context(metadata: dict[str, Any] | None = None) -> RequestContext:
    return RequestContext(
        request_id="tool-use-reporting",
        user_id="user-1",
        client_id="client-1",
        metadata=metadata or {},
    )


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
    assert copied[-1]["normalized_path"] == (
        f"private/story-{MAX_FILE_POLICY_DECISIONS - 1}.txt"
    )


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

    assert response.error.code == ErrorCode.INTERNAL_ERROR
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

    assert response.error.code == ErrorCode.INTERNAL_ERROR
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
    protocol, recorder = _protocol(module=module)
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
    assert recorder.events[-1].execution_origin == "cached"
    assert recorder.events[-1].idempotency_replay is True


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

    rate_limited_events = [
        event for event in recorder.events if event.status == "rate_limited"
    ]
    assert len(rate_limited_events) == 1
    assert rate_limited_events[0].execution_origin == "failed_before_execution"
