from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

import pytest
from mcp_unified.interfaces.runtime import ToolHookCallContext, ToolHookDecision

from tldw_Server_API.app.core.MCP_unified.protocol import (
    GovernanceDeniedError,
    MCPProtocol,
    MCPRequest,
    RequestContext,
)


class _AllowAllRBAC:
    async def check_permission(self, *_args: Any, **_kwargs: Any) -> bool:
        return True


class _NoopRateLimiter:
    async def check_rate_limit(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _NoopMetrics:
    def __getattr__(self, _name: str) -> Callable[..., None]:
        return lambda *args, **kwargs: None


class _Span:
    def set_attribute(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _Telemetry:
    def trace_context(self, *_args: Any, **_kwargs: Any) -> contextlib.AbstractContextManager[_Span]:
        return contextlib.nullcontext(_Span())


class _StaticEffectivePolicyResolver:
    async def resolve_for_context(
        self,
        *,
        user_id: str | None,
        metadata: dict[str, Any],
    ) -> dict[str, Any] | None:
        del user_id, metadata
        return None


class _AllowApprovalEvaluator:
    async def evaluate_tool_call(self, **_kwargs: Any) -> dict[str, Any]:
        return {"status": "allow", "reason": "test_allow"}


class _AllowPathScopeEnforcer:
    async def evaluate_tool_call(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "enabled": False,
            "within_scope": True,
            "reason": None,
            "force_approval": False,
            "normalized_paths": [],
            "scope_payload": None,
        }


class _NoopExternalAccessEvaluator:
    async def resolve_for_sources(
        self,
        *,
        sources: list[dict[str, Any]],
        effective_policy: dict[str, Any] | None,
    ) -> dict[str, Any]:
        del sources, effective_policy
        return {"servers": []}


class _ToolModuleStub:
    name = "stub"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.executions = 0

    async def get_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "stub.echo",
                "description": "Echo test arguments.",
                "inputSchema": {"type": "object"},
                "metadata": {"category": "read"},
            }
        ]

    async def get_tool_def(self, tool_name: str) -> dict[str, Any]:
        return {
            "name": tool_name,
            "description": "Echo test arguments.",
            "inputSchema": {"type": "object"},
            "metadata": {"category": "read"},
        }

    def sanitize_input(self, args: dict[str, Any]) -> dict[str, Any]:
        sanitized = dict(args)
        for key, value in list(sanitized.items()):
            if isinstance(value, str):
                sanitized[key] = value.replace("\0", "")
        return sanitized

    def validate_tool_arguments(self, _tool_name: str, _tool_args: dict[str, Any]) -> None:
        return None

    def is_write_tool_call(
        self,
        _tool_name: str,
        _arguments: dict[str, Any],
        *,
        tool_def: dict[str, Any] | None = None,
    ) -> bool:
        del tool_def
        return False

    async def execute_with_circuit_breaker(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return await func(*args, **kwargs)

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: RequestContext | None = None,
    ) -> dict[str, Any]:
        del context
        self.executions += 1
        if self.fail:
            raise RuntimeError("module exploded")
        return {"ok": True, "tool": tool_name, "arguments": arguments}


class _SandboxModuleStub(_ToolModuleStub):
    """Sandbox-like module stub used to verify secret-bearing hook metadata."""

    name = "sandbox"

    async def get_tools(self) -> list[dict[str, Any]]:
        """Return a minimal sandbox.run tool definition."""

        return [
            {
                "name": "sandbox.run",
                "description": "Run a sandbox command.",
                "inputSchema": {"type": "object"},
                "metadata": {"category": "write"},
            }
        ]

    def is_write_tool_call(
        self,
        _tool_name: str,
        _arguments: dict[str, Any],
        *,
        tool_def: dict[str, Any] | None = None,
    ) -> bool:
        """Classify sandbox.run as write-capable for protocol governance."""

        del tool_def
        return True


class _WriteToolModuleStub(_ToolModuleStub):
    """Effectful module used to exercise idempotent observer ordering."""

    async def get_tools(self) -> list[dict[str, Any]]:
        return [await self.get_tool_def("stub.write")]

    async def get_tool_def(self, tool_name: str) -> dict[str, Any]:
        return {
            "name": tool_name,
            "description": "Write test arguments.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "idempotencyKey": {"type": "string"},
                },
            },
            "metadata": {"category": "management"},
        }

    def is_write_tool_call(
        self,
        _tool_name: str,
        _arguments: dict[str, Any],
        *,
        tool_def: dict[str, Any] | None = None,
    ) -> bool:
        del tool_def
        return True


class _RegistryStub:
    def __init__(self, module: _ToolModuleStub) -> None:
        self.module = module

    async def find_module_for_tool(self, _tool_name: str) -> _ToolModuleStub:
        return self.module

    def get_module_id_for_tool(self, _tool_name: str) -> str:
        return self.module.name


class _RecordingToolHookManager:
    def __init__(
        self,
        *,
        before_decision: ToolHookDecision | dict[str, Any] | None = None,
        after_decision: ToolHookDecision | dict[str, Any] | None = None,
    ) -> None:
        self.before_decision = before_decision
        self.after_decision = after_decision
        self.before_contexts: list[ToolHookCallContext] = []
        self.after_contexts: list[ToolHookCallContext] = []

    async def before_tool_call(self, context: ToolHookCallContext) -> ToolHookDecision | dict[str, Any] | None:
        self.before_contexts.append(context)
        return self.before_decision

    async def after_tool_call(self, context: ToolHookCallContext) -> ToolHookDecision | dict[str, Any] | None:
        self.after_contexts.append(context)
        return self.after_decision


class _MutatingToolHookManager(_RecordingToolHookManager):
    async def before_tool_call(self, context: ToolHookCallContext) -> ToolHookDecision | dict[str, Any] | None:
        self.before_contexts.append(context)
        if isinstance(context.tool_args, dict):
            nested_args = context.tool_args.get("nested")
            if isinstance(nested_args, dict):
                nested_args["value"] = "mutated-by-hook"
        nested_metadata = context.metadata.get("nested")
        if isinstance(nested_metadata, dict):
            nested_metadata["value"] = "mutated-by-hook"
        return self.before_decision


class _FailingPreToolHookManager(_RecordingToolHookManager):
    async def before_tool_call(self, context: ToolHookCallContext) -> ToolHookDecision | dict[str, Any] | None:
        self.before_contexts.append(context)
        raise RuntimeError("pre-hook unavailable")


class _FailingPostToolHookManager(_RecordingToolHookManager):
    async def after_tool_call(self, context: ToolHookCallContext) -> ToolHookDecision | dict[str, Any] | None:
        self.after_contexts.append(context)
        raise RuntimeError("post-hook unavailable")


def _protocol(
    *,
    module: _ToolModuleStub | None = None,
    hook_manager: _RecordingToolHookManager | None = None,
) -> tuple[MCPProtocol, _ToolModuleStub, _RecordingToolHookManager]:
    module = module or _ToolModuleStub()
    hook_manager = hook_manager or _RecordingToolHookManager()
    deps = SimpleNamespace(
        module_registry=_RegistryStub(module),
        rbac_policy=_AllowAllRBAC(),
        rate_limiter=_NoopRateLimiter(),
        metrics_collector=_NoopMetrics(),
        telemetry_provider=_Telemetry(),
        tool_catalog_provider=object(),
        effective_policy_resolver=_StaticEffectivePolicyResolver(),
        approval_evaluator=_AllowApprovalEvaluator(),
        path_scope_enforcer=_AllowPathScopeEnforcer(),
        external_access_evaluator=_NoopExternalAccessEvaluator(),
        redis_client_factory=lambda **_kwargs: None,
        tool_call_hook_manager=hook_manager,
    )
    return MCPProtocol(dependencies=deps), module, hook_manager


def _context(metadata: dict[str, Any] | None = None) -> RequestContext:
    return RequestContext(
        request_id="hook-test",
        user_id="user-1",
        client_id="client-1",
        session_id="session-1",
        metadata=metadata or {},
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pre_hook_allow_observes_sanitized_metadata_and_post_hook_observes_success() -> None:
    protocol, module, hooks = _protocol()

    result = await protocol._handle_tools_call(
        {"name": "stub.echo", "arguments": {"target": "alpha\0"}},
        _context(),
    )

    assert result["tool"] == "stub.echo"
    assert module.executions == 1
    assert len(hooks.before_contexts) == 1
    assert hooks.before_contexts[0].phase == "pre"
    assert hooks.before_contexts[0].tool_name == "stub.echo"
    assert hooks.before_contexts[0].module_id == "stub"
    assert hooks.before_contexts[0].tool_args == {"target": "alpha"}
    assert hooks.before_contexts[0].arguments_hash
    assert len(hooks.after_contexts) == 1
    assert hooks.after_contexts[0].phase == "post"
    assert hooks.after_contexts[0].status == "success"
    assert hooks.after_contexts[0].error_type is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hook_context_mutation_cannot_mutate_prepared_tool_arguments() -> None:
    hooks = _MutatingToolHookManager()
    protocol, module, _ = _protocol(hook_manager=hooks)
    context = _context(metadata={"nested": {"value": "original"}})

    result = await protocol._handle_tools_call(
        {
            "name": "stub.echo",
            "arguments": {"nested": {"value": "original"}},
        },
        context,
    )

    assert module.executions == 1
    assert result["content"][0]["json"]["arguments"] == {"nested": {"value": "original"}}
    assert context.metadata == {"nested": {"value": "original"}}
    assert len(hooks.before_contexts) == 1
    assert hooks.before_contexts[0].tool_args == {"nested": {"value": "mutated-by-hook"}}
    assert hooks.before_contexts[0].metadata == {"nested": {"value": "mutated-by-hook"}}


@pytest.mark.unit
async def test_sandbox_run_env_is_redacted_in_hook_context_but_preserved_for_execution() -> None:
    """Hook contexts should redact sandbox env values without mutating execution args."""

    hooks = _RecordingToolHookManager()
    protocol, module, _ = _protocol(module=_SandboxModuleStub(), hook_manager=hooks)

    result = await protocol._handle_tools_call(
        {
            "name": "sandbox.run",
            "arguments": {
                "command": ["python", "-V"],
                "env": {"API_TOKEN": "super-secret", "PUBLIC_FLAG": "enabled"},
            },
        },
        _context(),
    )

    assert module.executions == 1
    assert result["content"][0]["json"]["arguments"]["env"] == {
        "API_TOKEN": "super-secret",
        "PUBLIC_FLAG": "enabled",
    }
    assert hooks.before_contexts[0].tool_args == {
        "command": ["python", "-V"],
        "env": {"API_TOKEN": "[redacted]", "PUBLIC_FLAG": "[redacted]"},
    }
    assert hooks.after_contexts[0].tool_args == hooks.before_contexts[0].tool_args


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pre_hook_deny_maps_to_authorization_error_and_skips_execution() -> None:
    hooks = _RecordingToolHookManager(
        before_decision=ToolHookDecision(
            action="deny",
            reason_code="blocked_by_test_hook",
            message="blocked /Users/example/private.txt token=secret",
            metadata={
                "hook_id": "local-policy",
                "hook_order": "7",
                "path": "/Users/example/private.txt",
                "token": "secret",
            },
        )
    )
    protocol, module, _ = _protocol(hook_manager=hooks)
    request = MCPRequest(
        method="tools/call",
        params={"name": "stub.echo", "arguments": {"target": "beta"}},
        id="hook-deny",
    )

    response = await protocol.process_request(request, _context())

    assert module.executions == 0
    assert response.error is not None
    assert response.error.code == -32001
    assert isinstance(response.error.data, dict)
    assert response.error.data["governance"]["hook"] == {
        "phase": "pre",
        "action": "deny",
        "status": "deny",
        "reason_code": "blocked_by_test_hook",
        "hook_id": "local-policy",
        "hook_order": 7,
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pre_hook_summary_strips_untrusted_composite_hook_payload_fields() -> None:
    hooks = _RecordingToolHookManager(
        before_decision=ToolHookDecision(
            action="deny",
            reason_code="blocked_by_composite_hook",
            metadata={
                "hook_results": [
                    {
                        "phase": "pre",
                        "hook_id": "profile-policy",
                        "hook_order": 8,
                        "action": "deny",
                        "status": "deny",
                        "reason_code": "blocked_by_composite_hook",
                        "message": "raw /Users/example/private.txt should not be retained",
                        "metadata": {"path": "/Users/example/private.txt"},
                        "raw_args": {"token": "super-secret"},
                    }
                ]
            },
        )
    )
    protocol, _, _ = _protocol(hook_manager=hooks)
    context = _context()

    with pytest.raises(GovernanceDeniedError):
        await protocol._run_pre_tool_hooks(
            tool_name="stub.echo",
            tool_args={"target": "beta"},
            module_id="stub",
            tool_def={"metadata": {"category": "read"}},
            is_write=False,
            arguments_hash="abc123",
            context=context,
            scope_payload=None,
        )

    assert context.metadata["mcp_tool_hook_results"] == [
        {
            "phase": "pre",
            "hook_id": "profile-policy",
            "hook_order": 8,
            "action": "deny",
            "status": "deny",
            "reason_code": "blocked_by_composite_hook",
        }
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replacing_protocol_hook_manager_updates_hook_execution_helper() -> None:
    protocol, module, original_hooks = _protocol()
    replacement_hooks = _RecordingToolHookManager(
        before_decision=ToolHookDecision(
            action="deny",
            reason_code="blocked_after_replacement",
        )
    )
    protocol._tool_call_hook_manager = replacement_hooks

    with pytest.raises(GovernanceDeniedError):
        await protocol._handle_tools_call(
            {"name": "stub.echo", "arguments": {"target": "beta"}},
            _context(),
        )

    assert module.executions == 0
    assert original_hooks.before_contexts == []
    assert len(replacement_hooks.before_contexts) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pre_hook_ask_maps_to_approval_error_and_skips_execution() -> None:
    hooks = _RecordingToolHookManager(
        before_decision=ToolHookDecision(
            action="ask",
            reason_code="operator_review_required",
            message="needs approval for /Users/example/private.txt token=secret",
            metadata={
                "hook_id": "approval-policy",
                "hook_order": "3",
                "path": "/Users/example/private.txt",
                "token": "secret",
            },
        )
    )
    protocol, module, _ = _protocol(hook_manager=hooks)
    request = MCPRequest(
        method="tools/call",
        params={"name": "stub.echo", "arguments": {"target": "gamma"}},
        id="hook-ask",
    )

    response = await protocol.process_request(request, _context())

    assert module.executions == 0
    assert response.error is not None
    assert response.error.code == -32001
    assert isinstance(response.error.data, dict)
    assert response.error.data["approval"]["source"] == "tool_hook"
    assert response.error.data["approval"]["reason_code"] == "operator_review_required"
    assert response.error.data["approval"]["message"] is None
    assert response.error.data["approval"]["hook"] == {
        "phase": "pre",
        "action": "ask",
        "status": "ask",
        "reason_code": "operator_review_required",
        "hook_id": "approval-policy",
        "hook_order": 3,
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pre_hook_failure_fails_closed_and_skips_execution() -> None:
    hooks = _FailingPreToolHookManager()
    protocol, module, _ = _protocol(hook_manager=hooks)
    request = MCPRequest(
        method="tools/call",
        params={"name": "stub.echo", "arguments": {"target": "zeta"}},
        id="hook-failure",
    )

    response = await protocol.process_request(request, _context())

    assert module.executions == 0
    assert response.error is not None
    assert response.error.code == -32001
    assert isinstance(response.error.data, dict)
    assert response.error.data["governance"]["reason_code"] == "tool_hook_unavailable"
    assert response.error.data["governance"]["hook"]["error_type"] == "RuntimeError"
    assert len(hooks.before_contexts) == 1
    assert hooks.before_contexts[0].phase == "pre"
    assert hooks.before_contexts[0].tool_name == "stub.echo"
    assert hooks.before_contexts[0].module_id == "stub"
    assert hooks.after_contexts == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_post_hook_failure_preserves_success_result() -> None:
    hooks = _FailingPostToolHookManager()
    protocol, module, _ = _protocol(hook_manager=hooks)

    result = await protocol._handle_tools_call(
        {"name": "stub.echo", "arguments": {"target": "eta"}},
        _context(),
    )

    assert module.executions == 1
    assert result["tool"] == "stub.echo"
    assert result["content"][0]["json"]["arguments"] == {"target": "eta"}
    assert len(hooks.after_contexts) == 1
    assert hooks.after_contexts[0].phase == "post"
    assert hooks.after_contexts[0].tool_name == "stub.echo"
    assert hooks.after_contexts[0].module_id == "stub"
    assert hooks.after_contexts[0].status == "success"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_idempotent_success_is_committed_before_raising_post_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCP_DISABLE_WRITE_TOOLS", "false")
    from tldw_Server_API.app.core.MCP_unified.config import get_config

    get_config.cache_clear()  # type: ignore[attr-defined]
    protocol, module, _hooks = _protocol(module=_WriteToolModuleStub())
    post_hook_calls = 0

    async def _raise_after_success(**_kwargs: Any) -> None:
        nonlocal post_hook_calls
        post_hook_calls += 1
        raise RuntimeError("private post-hook detail")

    protocol._run_post_tool_hooks = _raise_after_success
    params = {
        "name": "stub.write",
        "arguments": {"target": "eta"},
        "idempotencyKey": "post-hook-raise",
    }

    first = await protocol._handle_tools_call(params, _context())
    replay = await protocol._handle_tools_call(params, _context())

    assert first["content"] == replay["content"]
    assert module.executions == 1
    assert post_hook_calls == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_post_hook_cancellation_after_commit_replays_without_module_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCP_DISABLE_WRITE_TOOLS", "false")
    from tldw_Server_API.app.core.MCP_unified.config import get_config

    get_config.cache_clear()  # type: ignore[attr-defined]
    protocol, module, _hooks = _protocol(module=_WriteToolModuleStub())
    cancellation = asyncio.CancelledError("post-hook cancellation")
    post_hook_calls = 0

    async def _cancel_after_success(**_kwargs: Any) -> None:
        nonlocal post_hook_calls
        post_hook_calls += 1
        raise cancellation

    protocol._run_post_tool_hooks = _cancel_after_success
    params = {
        "name": "stub.write",
        "arguments": {"target": "theta"},
        "idempotencyKey": "post-hook-cancel",
    }

    with pytest.raises(asyncio.CancelledError) as caught:
        await protocol._handle_tools_call(params, _context())
    replay = await protocol._handle_tools_call(params, _context())

    assert caught.value is cancellation
    assert replay["tool"] == "stub.write"
    assert module.executions == 1
    assert post_hook_calls == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replacing_protocol_post_hook_wrapper_controls_runtime_post_hooks() -> None:
    protocol, module, hooks = _protocol()
    wrapper_calls: list[dict[str, Any]] = []

    async def replacement_post_hook_wrapper(**kwargs: Any) -> None:
        wrapper_calls.append(kwargs)

    protocol._run_post_tool_hooks = replacement_post_hook_wrapper

    result = await protocol._handle_tools_call(
        {"name": "stub.echo", "arguments": {"target": "theta"}},
        _context(),
    )

    assert module.executions == 1
    assert result["tool"] == "stub.echo"
    assert len(wrapper_calls) == 1
    assert wrapper_calls[0]["tool_name"] == "stub.echo"
    assert wrapper_calls[0]["status"] == "success"
    assert hooks.after_contexts == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_post_hook_observes_failure_without_converting_it_to_success() -> None:
    hooks = _RecordingToolHookManager(after_decision=ToolHookDecision(action="allow"))
    protocol, module, _ = _protocol(module=_ToolModuleStub(fail=True), hook_manager=hooks)

    with pytest.raises(RuntimeError, match="tool_execution_error"):
        await protocol._handle_tools_call(
            {"name": "stub.echo", "arguments": {"target": "delta"}},
            _context(),
        )

    assert module.executions == 1
    assert len(hooks.after_contexts) == 1
    assert hooks.after_contexts[0].phase == "post"
    assert hooks.after_contexts[0].status == "failure"
    assert hooks.after_contexts[0].error_type == "RuntimeError"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_existing_context_policy_denial_takes_precedence_over_hooks() -> None:
    protocol, module, hooks = _protocol()

    with pytest.raises(PermissionError, match="not allowed by execution context"):
        await protocol._handle_tools_call(
            {"name": "stub.echo", "arguments": {"target": "epsilon"}},
            _context(metadata={"allowed_tools": ["other.tool"]}),
        )

    assert module.executions == 0
    assert hooks.before_contexts == []
    assert hooks.after_contexts == []
