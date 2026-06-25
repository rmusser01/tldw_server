from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext


@dataclass
class _Prepared:
    name: str
    context: RequestContext


class _Reporter:
    def __init__(self) -> None:
        self.prepare_failures: list[dict[str, Any]] = []

    def should_record(self, context: RequestContext) -> bool:
        return context.metadata.get("mcp_tool_use_observed") is not True

    async def record_prepare_failure(
        self,
        *,
        context: RequestContext,
        params: dict[str, Any],
        exc: Exception,
        start_ts: float,
    ) -> None:
        del start_ts
        self.prepare_failures.append(
            {
                "request_id": context.request_id,
                "name": params.get("name"),
                "error_type": exc.__class__.__name__,
            }
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_execution_coordinator_delegates_prepare_then_execute() -> None:
    from tldw_Server_API.app.core.MCP_unified.tool_execution.coordinator import (
        ToolExecutionCoordinator,
    )

    calls: list[str] = []
    context = RequestContext(request_id="coord-ok", user_id="u1", client_id="c1")
    prepared = _Prepared(name="demo.echo", context=context)

    async def prepare_impl(
        *,
        params: dict[str, Any],
        context: RequestContext,
        idempotency_key: str | None = None,
    ) -> _Prepared:
        assert params == {"name": "demo.echo", "arguments": {"value": "x"}}
        assert idempotency_key is None
        calls.append("prepare")
        return prepared

    async def execute_impl(prepared_call: _Prepared) -> dict[str, Any]:
        assert prepared_call is prepared
        calls.append("execute")
        return {"content": [{"type": "text", "text": "ok"}], "tool": prepared_call.name}

    coordinator = ToolExecutionCoordinator(
        prepare_tool_call_impl=prepare_impl,
        execute_prepared_tool_call_impl=execute_impl,
        reporter=_Reporter(),
    )

    result = await coordinator.handle_tools_call(
        {"name": "demo.echo", "arguments": {"value": "x"}},
        context,
    )

    assert calls == ["prepare", "execute"]
    assert result["tool"] == "demo.echo"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_execution_coordinator_reports_prepare_failure_before_reraising() -> None:
    from tldw_Server_API.app.core.MCP_unified.protocol import InvalidParamsException
    from tldw_Server_API.app.core.MCP_unified.tool_execution.coordinator import (
        ToolExecutionCoordinator,
    )

    reporter = _Reporter()
    context = RequestContext(request_id="coord-fail", user_id="u1", client_id="c1")

    async def prepare_impl(
        *,
        params: dict[str, Any],
        context: RequestContext,
        idempotency_key: str | None = None,
    ) -> _Prepared:
        del params, context, idempotency_key
        raise InvalidParamsException("bad arguments")

    async def execute_impl(prepared_call: _Prepared) -> dict[str, Any]:
        del prepared_call
        raise AssertionError("execute should not run")

    coordinator = ToolExecutionCoordinator(
        prepare_tool_call_impl=prepare_impl,
        execute_prepared_tool_call_impl=execute_impl,
        reporter=reporter,
    )

    with pytest.raises(InvalidParamsException):
        await coordinator.handle_tools_call({"name": "demo.echo"}, context)

    assert reporter.prepare_failures == [
        {
            "request_id": "coord-fail",
            "name": "demo.echo",
            "error_type": "InvalidParamsException",
        }
    ]
