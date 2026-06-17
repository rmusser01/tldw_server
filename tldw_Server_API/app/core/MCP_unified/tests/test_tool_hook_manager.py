"""Tests for host-neutral MCP tool-call hook manager primitives."""

from __future__ import annotations

from typing import Any

import pytest

from mcp_unified.interfaces.runtime import ToolHookCallContext, ToolHookDecision
from mcp_unified.tool_hooks import (
    ConfiguredToolCallHookManager,
    ToolHookExecutionError,
    ToolHookRegistration,
)


def _context(*, phase: str = "pre") -> ToolHookCallContext:
    return ToolHookCallContext(
        phase="post" if phase == "post" else "pre",
        tool_name="fs.patch",
        module_id="filesystem",
        is_write=True,
        tool_category="management",
        arguments_hash="abc123",
        request_id="request-1",
        user_id="user-1",
        client_id="client-1",
        session_id="session-1",
        tool_args={"path": "docs/story.md"},
        metadata={"profile_id": "backend-engineer"},
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_configured_manager_runs_pre_hooks_in_order_until_blocking_decision() -> None:
    calls: list[str] = []

    async def allow(_context: ToolHookCallContext) -> ToolHookDecision:
        calls.append("allow")
        return ToolHookDecision(action="allow", reason_code="first_clear")

    async def deny(_context: ToolHookCallContext) -> ToolHookDecision:
        calls.append("deny")
        return ToolHookDecision(action="deny", reason_code="blocked_by_policy")

    async def late(_context: ToolHookCallContext) -> ToolHookDecision:
        calls.append("late")
        return ToolHookDecision(action="allow")

    manager = ConfiguredToolCallHookManager(
        [
            ToolHookRegistration(hook_id="late-hook", before=late, order=30),
            ToolHookRegistration(hook_id="allow-hook", before=allow, order=10),
            ToolHookRegistration(hook_id="deny-hook", before=deny, order=20),
        ]
    )

    decision = await manager.before_tool_call(_context())

    assert calls == ["allow", "deny"]
    assert decision.action == "deny"
    assert decision.reason_code == "blocked_by_policy"
    assert decision.metadata["hook_id"] == "deny-hook"
    assert decision.metadata["hook_order"] == 20
    assert [item["hook_id"] for item in decision.metadata["hook_results"]] == [
        "allow-hook",
        "deny-hook",
    ]
    assert [item["action"] for item in decision.metadata["hook_results"]] == [
        "allow",
        "deny",
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_configured_manager_skips_disabled_and_wrong_phase_hooks() -> None:
    calls: list[str] = []

    async def record(context: ToolHookCallContext) -> ToolHookDecision:
        calls.append(context.phase)
        return ToolHookDecision(action="allow")

    manager = ConfiguredToolCallHookManager(
        [
            ToolHookRegistration(hook_id="disabled", before=record, enabled=False),
            ToolHookRegistration(hook_id="post-only", before=record, phases=("post",)),
            ToolHookRegistration(hook_id="enabled", before=record, phases=("pre",)),
        ]
    )

    decision = await manager.before_tool_call(_context())

    assert calls == ["pre"]
    assert decision.action == "allow"
    assert [item["hook_id"] for item in decision.metadata["hook_results"]] == ["enabled"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_configured_manager_pre_hook_failure_identifies_failed_hook() -> None:
    async def explode(_context: ToolHookCallContext) -> ToolHookDecision:
        raise RuntimeError("hook backend unavailable")

    manager = ConfiguredToolCallHookManager(
        [ToolHookRegistration(hook_id="policy-service", before=explode)]
    )

    with pytest.raises(ToolHookExecutionError) as exc_info:
        await manager.before_tool_call(_context())

    assert exc_info.value.hook_id == "policy-service"
    assert exc_info.value.phase == "pre"
    assert exc_info.value.error_type == "RuntimeError"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_configured_manager_post_hooks_continue_after_failure_and_report_error() -> None:
    calls: list[str] = []

    async def first(_context: ToolHookCallContext) -> ToolHookDecision:
        calls.append("first")
        return ToolHookDecision(action="allow", reason_code="observed")

    async def explode(_context: ToolHookCallContext) -> ToolHookDecision:
        calls.append("explode")
        raise RuntimeError("post hook backend unavailable")

    async def last(_context: ToolHookCallContext) -> dict[str, Any]:
        calls.append("last")
        return {"action": "allow", "reason_code": "finished"}

    manager = ConfiguredToolCallHookManager(
        [
            ToolHookRegistration(hook_id="first", after=first, order=10),
            ToolHookRegistration(hook_id="explode", after=explode, order=20),
            ToolHookRegistration(hook_id="last", after=last, order=30),
        ]
    )

    decision = await manager.after_tool_call(_context(phase="post"))

    assert calls == ["first", "explode", "last"]
    assert decision.action == "allow"
    assert decision.metadata["hook_results"] == [
        {
            "phase": "post",
            "hook_id": "first",
            "hook_order": 10,
            "action": "allow",
            "status": "allow",
            "reason_code": "observed",
        },
        {
            "phase": "post",
            "hook_id": "explode",
            "hook_order": 20,
            "action": "deny",
            "status": "error",
            "reason_code": "tool_hook_unavailable",
            "error_type": "RuntimeError",
        },
        {
            "phase": "post",
            "hook_id": "last",
            "hook_order": 30,
            "action": "allow",
            "status": "allow",
            "reason_code": "finished",
        },
    ]
