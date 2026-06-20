from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, MCPRequest, RequestContext


class _AllowAllRBAC:
    async def check_permission(self, *_args, **_kwargs):
        return True


class _ToolModuleStub:
    name = "stub"

    async def get_tools(self):
        return [{"name": "stub.echo", "description": "", "inputSchema": {"type": "object"}}]

    async def execute_tool(self, tool_name, arguments, context=None):
        return {"ok": True, "tool": tool_name, "arguments": arguments}

    async def execute_with_circuit_breaker(self, func, *args, **kwargs):
        return await func(*args, **kwargs)

    def sanitize_input(self, args):
        return args

    def validate_tool_arguments(self, tool_name, tool_args):
        return None

    def is_write_tool_def(self, tool_def):
        return False

    async def get_tool_def(self, tool_name):
        return {"name": tool_name, "metadata": {"category": "read"}}


class _GovernanceModuleStub:
    name = "governance"

    async def get_tools(self):
        return [{"name": "governance.query_knowledge", "description": "", "inputSchema": {"type": "object"}}]

    async def execute_tool(self, tool_name, arguments, context=None):
        return {"ok": True, "tool": tool_name}

    async def execute_with_circuit_breaker(self, func, *args, **kwargs):
        return await func(*args, **kwargs)

    def sanitize_input(self, args):
        return args

    def validate_tool_arguments(self, tool_name, tool_args):
        return None

    def is_write_tool_def(self, tool_def):
        return False

    async def get_tool_def(self, tool_name):
        return {"name": tool_name, "metadata": {"category": "governance"}}


class _RegistryStub:
    def __init__(self, module) -> None:
        self._module = module

    async def find_module_for_tool(self, _tool_name):
        return self._module

    def get_module_id_for_tool(self, _tool_name):
        return getattr(self._module, "name", "stub")


@dataclass
class _Decision:
    action: str
    status: str
    category: str
    category_source: str = "explicit"
    fallback_reason: str | None = None
    matched_rules: tuple[str, ...] = ()


class _GovernanceServiceStub:
    def __init__(self, action: str = "allow") -> None:
        self.action = action
        self.called = False
        self.last_validate: dict[str, Any] | None = None

    async def validate_change(self, **kwargs: Any):
        self.called = True
        self.last_validate = kwargs
        return _Decision(
            action=self.action,
            status=self.action,
            category=str(kwargs.get("category") or "general"),
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_non_governance_tool_invokes_preflight():
    proto = MCPProtocol()
    proto.rbac_policy = _AllowAllRBAC()
    proto.module_registry = _RegistryStub(_ToolModuleStub())

    service = _GovernanceServiceStub(action="warn")

    async def _fake_ensure_service():
        return service

    proto._ensure_governance_service = _fake_ensure_service  # type: ignore[attr-defined]

    ctx = RequestContext(
        request_id="gov-preflight-1",
        user_id="1",
        metadata={"governance_rollout_mode": "enforce"},
    )
    result = await proto._handle_tools_call({"name": "stub.echo", "arguments": {"x": 1}}, ctx)

    assert result["tool"] == "stub.echo"
    assert service.called is True
    assert service.last_validate is not None
    assert service.last_validate["surface"] == "mcp_tool"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_governance_tools_bypass_preflight_but_keep_rbac():
    proto = MCPProtocol()
    proto.rbac_policy = _AllowAllRBAC()
    proto.module_registry = _RegistryStub(_GovernanceModuleStub())

    service = _GovernanceServiceStub(action="deny")

    async def _fake_ensure_service():
        return service

    proto._ensure_governance_service = _fake_ensure_service  # type: ignore[attr-defined]

    rbac_checked = {"tool": 0}

    async def _wrapped_has_tool_permission(context, tool_name, **kwargs):
        rbac_checked["tool"] += 1
        return True

    proto._has_tool_permission = _wrapped_has_tool_permission  # type: ignore[assignment]

    ctx = RequestContext(request_id="gov-preflight-2", user_id="1", metadata={})
    result = await proto._handle_tools_call({"name": "governance.query_knowledge", "arguments": {}}, ctx)

    assert result["tool"] == "governance.query_knowledge"
    assert service.called is False
    assert rbac_checked["tool"] > 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_wire_compat_adds_governance_details_in_error_data_only():
    proto = MCPProtocol()
    proto.rbac_policy = _AllowAllRBAC()
    proto.module_registry = _RegistryStub(_ToolModuleStub())

    service = _GovernanceServiceStub(action="deny")

    async def _fake_ensure_service():
        return service

    proto._ensure_governance_service = _fake_ensure_service  # type: ignore[attr-defined]

    req = MCPRequest(method="tools/call", params={"name": "stub.echo", "arguments": {"x": 1}}, id="gov-deny")
    ctx = RequestContext(
        request_id="gov-preflight-3",
        user_id="1",
        metadata={"governance_rollout_mode": "enforce"},
    )

    resp = await proto.process_request(req, ctx)

    assert resp.error is not None
    assert resp.error.code == -32001
    assert isinstance(resp.error.data, dict)
    assert "governance" in resp.error.data


@pytest.mark.unit
@pytest.mark.asyncio
async def test_policy_resolver_runtime_failure_fails_closed_before_governance_preflight():
    from tldw_Server_API.app.core.MCP_unified.adapters.tldw_policy import (
        TldwEffectivePolicyResolver,
    )

    class _FailingHubResolver:
        async def resolve_for_context(self, *, user_id, metadata):
            raise RuntimeError("resolver unavailable")

    async def _resolver_factory():
        return _FailingHubResolver()

    proto = MCPProtocol()
    proto.rbac_policy = _AllowAllRBAC()
    proto.module_registry = _RegistryStub(_ToolModuleStub())
    proto.dependencies.effective_policy_resolver = TldwEffectivePolicyResolver(
        resolver_factory=_resolver_factory,
    )

    service = _GovernanceServiceStub(action="allow")

    async def _fake_ensure_service():
        return service

    proto._ensure_governance_service = _fake_ensure_service  # type: ignore[attr-defined]

    req = MCPRequest(method="tools/call", params={"name": "stub.echo", "arguments": {"x": 1}}, id="policy-fail")
    ctx = RequestContext(
        request_id="policy-fail",
        user_id="1",
        metadata={
            "mcp_policy_context_enabled": True,
            "governance_rollout_mode": "enforce",
        },
    )

    resp = await proto.process_request(req, ctx)

    assert resp.error is not None
    assert resp.error.code == -32001
    assert "MCP Hub policy" in resp.error.message
    assert ctx.metadata["_mcp_effective_tool_policy"]["resolution_error"] == "policy_resolution_failed"
    assert service.called is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shadow_rollout_deny_does_not_block_and_records_metrics(monkeypatch):
    proto = MCPProtocol()
    proto.rbac_policy = _AllowAllRBAC()
    proto.module_registry = _RegistryStub(_ToolModuleStub())

    service = _GovernanceServiceStub(action="deny")

    async def _fake_ensure_service():
        return service

    proto._ensure_governance_service = _fake_ensure_service  # type: ignore[attr-defined]
    monkeypatch.setattr(
        proto,
        "_resolve_governance_rollout_mode",
        lambda _metadata=None: "shadow",
        raising=False,
    )

    metric_calls: list[dict[str, str]] = []
    monkeypatch.setattr(
        proto.metrics,
        "record_governance_check",
        lambda **kwargs: metric_calls.append({k: str(v) for k, v in kwargs.items()}),
    )

    ctx = RequestContext(request_id="gov-shadow-1", user_id="1", metadata={})
    result = await proto._handle_tools_call({"name": "stub.echo", "arguments": {"x": 1}}, ctx)

    assert result["tool"] == "stub.echo"
    assert service.called is True
    assert metric_calls
    assert metric_calls[-1]["rollout_mode"] == "shadow"
    assert metric_calls[-1]["status"] == "deny"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_off_rollout_skips_governance_service_and_records_metrics(monkeypatch):
    proto = MCPProtocol()
    proto.rbac_policy = _AllowAllRBAC()
    proto.module_registry = _RegistryStub(_ToolModuleStub())

    service = _GovernanceServiceStub(action="deny")

    async def _fake_ensure_service():
        return service

    proto._ensure_governance_service = _fake_ensure_service  # type: ignore[attr-defined]
    monkeypatch.setattr(
        proto,
        "_resolve_governance_rollout_mode",
        lambda _metadata=None: "off",
        raising=False,
    )

    metric_calls: list[dict[str, str]] = []
    monkeypatch.setattr(
        proto.metrics,
        "record_governance_check",
        lambda **kwargs: metric_calls.append({k: str(v) for k, v in kwargs.items()}),
    )

    ctx = RequestContext(request_id="gov-off-1", user_id="1", metadata={})
    result = await proto._handle_tools_call({"name": "stub.echo", "arguments": {"x": 1}}, ctx)

    assert result["tool"] == "stub.echo"
    assert service.called is False
    assert metric_calls
    assert metric_calls[-1]["rollout_mode"] == "off"
    assert metric_calls[-1]["status"] == "unknown"
