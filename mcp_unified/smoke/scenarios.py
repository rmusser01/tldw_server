"""Built-in MCP smoke scenario runners."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Literal

from mcp_unified.smoke.client import McpSmokeClient, McpSmokeClientError
from mcp_unified.smoke.reporting import (
    SmokeReport,
    SmokeStepReport,
    redact_detail,
    summarize_result,
)
from mcp_unified.smoke.transports import McpSmokeTransport

ScenarioMode = Literal["best_effort", "strict"]

_METHOD_NOT_FOUND = -32601
_INVALID_REQUEST = -32600
_POLICY_DENIED = -32001


async def run_baseline_scenario(
    transport: McpSmokeTransport,
    *,
    mode: ScenarioMode = "best_effort",
    safe_tool_name: str = "echo.search",
    safe_tool_arguments: dict[str, object] | None = None,
    safe_resource_uri: str | None = None,
    safe_prompt_name: str | None = None,
    safe_prompt_arguments: dict[str, object] | None = None,
) -> SmokeReport:
    """Run the ordered baseline MCP JSON-RPC smoke scenario."""

    if mode not in {"best_effort", "strict"}:
        raise ValueError("mode must be 'best_effort' or 'strict'")

    started_at = datetime.now(UTC).isoformat()
    report = SmokeReport(
        transport=transport.__class__.__name__,
        started_at=started_at,
        metadata={"mode": mode},
    )
    started = perf_counter()
    client = McpSmokeClient(transport)
    state: dict[str, Any] = {
        "capabilities": {},
        "tools": [],
        "safe_tool_name": safe_tool_name,
        "safe_tool_arguments": (
            safe_tool_arguments
            if safe_tool_arguments is not None
            else {"query": "smoke"}
        ),
        "safe_resource_uri": safe_resource_uri,
        "safe_prompt_name": safe_prompt_name,
        "safe_prompt_arguments": (
            safe_prompt_arguments
            if safe_prompt_arguments is not None
            else {"topic": "smoke"}
        ),
    }

    await transport.start()
    try:
        await _record_step(report, "initialize", lambda: _step_initialize(client, state))
        await _record_step(
            report,
            "notifications/initialized",
            lambda: _step_initialized_notification(client),
        )
        await _record_step(report, "ping", client.ping, method="ping")
        await _record_step(report, "tools/list", lambda: _step_tools_list(client, state))
        await _record_step(
            report,
            "tools/call",
            lambda: _step_safe_tool_call(client, state, mode),
        )
        await _record_step(
            report,
            "tools/call:unknown",
            lambda: _step_unknown_tool(transport),
        )
        await _record_step(
            report,
            "profile-filtered visibility",
            _step_profile_filtered_visibility,
        )
        await _record_step(
            report,
            "resources",
            lambda: _step_resources(client, state, mode),
        )
        await _record_step(
            report,
            "prompts",
            lambda: _step_prompts(client, state, mode),
        )
        await _record_step(
            report,
            "json-rpc batch",
            lambda: _step_jsonrpc_batch(transport),
        )
        await _record_step(
            report,
            "malformed request",
            lambda: _step_malformed_request(transport),
        )
        await _record_step(
            report,
            "policy denial",
            lambda: _step_policy_denial(transport),
        )
    finally:
        await transport.close()
        report.elapsed_ms = _elapsed_ms(started)
        report.ok = all(step.ok for step in report.steps)

    return report


async def _record_step(
    report: SmokeReport,
    name: str,
    action: Callable[[], Awaitable[object]],
    *,
    method: str | None = None,
) -> None:
    started = perf_counter()
    try:
        outcome = await action()
    except McpSmokeClientError as exc:
        report.steps.append(
            SmokeStepReport(
                name=name,
                ok=False,
                method=method,
                elapsed_ms=_elapsed_ms(started),
                reason_code="jsonrpc_error",
                error_code=_client_error_code(exc),
                detail=redact_detail({"message": str(exc), "error": exc.error}),
            )
        )
        return
    except Exception as exc:  # noqa: BLE001 - smoke reports must capture transport failures.
        report.steps.append(
            SmokeStepReport(
                name=name,
                ok=False,
                method=method,
                elapsed_ms=_elapsed_ms(started),
                reason_code="exception",
                detail=redact_detail(
                    {"type": exc.__class__.__name__, "message": str(exc)}
                ),
            )
        )
        return

    report.steps.append(_outcome_to_step(name, outcome, started, method=method))


async def _step_initialize(client: McpSmokeClient, state: dict[str, Any]) -> object:
    result = await client.initialize()
    if not isinstance(result, dict):
        return _failure("invalid_initialize_result", detail=result)
    server_info = result.get("serverInfo")
    if not isinstance(server_info, dict) or not server_info.get("name"):
        return _failure("missing_server_info", detail=result)
    capabilities = result.get("capabilities")
    if not isinstance(capabilities, dict):
        return _failure("missing_capabilities", detail=result)
    state["capabilities"] = capabilities
    return result


async def _step_initialized_notification(client: McpSmokeClient) -> object:
    await client.notify("notifications/initialized")
    try:
        ping = await client.ping()
    except McpSmokeClientError as exc:
        return _failure(
            "followup_ping_failed",
            error_code=_client_error_code(exc),
            detail={"message": str(exc), "error": exc.error},
        )
    if not _is_successful_ping_result(ping):
        return _failure("followup_ping_failed", detail=ping)
    return {"followup_ping": ping}


async def _step_tools_list(client: McpSmokeClient, state: dict[str, Any]) -> object:
    result = await client.list_tools()
    if not isinstance(result, dict) or not isinstance(result.get("tools"), list):
        return _failure("invalid_tools_list", detail=result)
    tools = result["tools"]
    for tool in tools:
        if not isinstance(tool, dict):
            return _failure("invalid_tool_descriptor", detail=tool)
        if not isinstance(tool.get("name"), str) or not tool["name"]:
            return _failure("invalid_tool_descriptor", detail=tool)
        if not isinstance(tool.get("description"), str):
            return _failure("invalid_tool_descriptor", detail=tool)
        if not isinstance(tool.get("inputSchema"), dict):
            return _failure("invalid_tool_descriptor", detail=tool)
    state["tools"] = tools
    return result


async def _step_safe_tool_call(
    client: McpSmokeClient,
    state: dict[str, Any],
    mode: ScenarioMode,
) -> object:
    safe_tool_name = state["safe_tool_name"]
    tool_names = {
        tool.get("name")
        for tool in state.get("tools", [])
        if isinstance(tool, dict)
    }
    if safe_tool_name not in tool_names:
        if mode == "strict":
            return _failure("safe_tool_unavailable", detail={"tool": safe_tool_name})
        return _skip("safe_tool_unavailable", detail={"tool": safe_tool_name})
    result = await client.call_tool(safe_tool_name, state["safe_tool_arguments"])
    if not isinstance(result, dict) or not isinstance(result.get("content"), list):
        return _failure("invalid_tool_call_result", detail=result)
    return result


async def _step_unknown_tool(transport: McpSmokeTransport) -> object:
    response = await transport.request(
        {
            "jsonrpc": "2.0",
            "id": "smoke-unknown-tool",
            "method": "tools/call",
            "params": {
                "name": "smoke.missing_tool",
                "arguments": {},
            },
        }
    )
    return _expect_jsonrpc_error(
        response,
        expected_code=_METHOD_NOT_FOUND,
        reason_code="unknown_tool_not_rejected",
    )


async def _step_profile_filtered_visibility() -> object:
    return _skip("not_configured")


async def _step_resources(
    client: McpSmokeClient,
    state: dict[str, Any],
    mode: ScenarioMode,
) -> object:
    if not _capability_available(state, "resources"):
        if mode == "strict":
            return _failure("required_capability_unavailable", detail={"capability": "resources"})
        return _skip("capability_unavailable", detail={"capability": "resources"})
    result = await client.list_resources()
    if not isinstance(result, dict) or not isinstance(result.get("resources"), list):
        return _failure("invalid_resources_list", detail=result)
    resources = result["resources"]
    if not resources:
        return result
    uri = state.get("safe_resource_uri") or _first_named_value(resources, "uri")
    if not isinstance(uri, str):
        return _failure("invalid_resource_descriptor", detail=resources[0])
    read_result = await client.read_resource(uri)
    if not isinstance(read_result, dict) or not isinstance(read_result.get("contents"), list):
        return _failure("invalid_resource_read_result", detail=read_result)
    return {"list": result, "read": read_result}


async def _step_prompts(
    client: McpSmokeClient,
    state: dict[str, Any],
    mode: ScenarioMode,
) -> object:
    if not _capability_available(state, "prompts"):
        if mode == "strict":
            return _failure("required_capability_unavailable", detail={"capability": "prompts"})
        return _skip("capability_unavailable", detail={"capability": "prompts"})
    result = await client.list_prompts()
    if not isinstance(result, dict) or not isinstance(result.get("prompts"), list):
        return _failure("invalid_prompts_list", detail=result)
    prompts = result["prompts"]
    if not prompts:
        return result
    name = state.get("safe_prompt_name") or _first_named_value(prompts, "name")
    if not isinstance(name, str):
        return _failure("invalid_prompt_descriptor", detail=prompts[0])
    prompt_result = await client.get_prompt(name, state["safe_prompt_arguments"])
    if not isinstance(prompt_result, dict) or not isinstance(prompt_result.get("messages"), list):
        return _failure("invalid_prompt_get_result", detail=prompt_result)
    return {"list": result, "get": prompt_result}


async def _step_jsonrpc_batch(transport: McpSmokeTransport) -> object:
    response = await transport.request(
        [
            {"jsonrpc": "2.0", "id": "smoke-batch-ping", "method": "ping"},
            {"jsonrpc": "2.0", "id": "smoke-batch-tools", "method": "tools/list"},
        ]
    )
    if not isinstance(response, list):
        return _failure("invalid_batch_response", detail=response)
    responses = {
        item.get("id"): item
        for item in response
        if isinstance(item, dict)
    }
    if set(responses) != {"smoke-batch-ping", "smoke-batch-tools"}:
        return _failure("invalid_batch_correlation", detail=response)
    ping_validation = _validate_jsonrpc_success_response(
        responses["smoke-batch-ping"],
        expected_id="smoke-batch-ping",
    )
    if ping_validation is not None:
        return _failure(ping_validation, detail=response)
    tools_validation = _validate_jsonrpc_success_response(
        responses["smoke-batch-tools"],
        expected_id="smoke-batch-tools",
    )
    if tools_validation is not None:
        return _failure(tools_validation, detail=response)
    if not _is_successful_ping_result(responses["smoke-batch-ping"]["result"]):
        return _failure("invalid_batch_ping_result", detail=response)
    tools_result = responses["smoke-batch-tools"]["result"]
    if not isinstance(tools_result, dict) or not isinstance(tools_result.get("tools"), list):
        return _failure("invalid_batch_tools_result", detail=response)
    return {"responses": response}


async def _step_malformed_request(transport: McpSmokeTransport) -> object:
    response = await transport.request(
        {
            "jsonrpc": "2.0",
            "id": "smoke-malformed",
            "params": {},
        }
    )
    return _expect_jsonrpc_error(
        response,
        expected_code=_INVALID_REQUEST,
        reason_code="malformed_request_not_rejected",
    )


async def _step_policy_denial(transport: McpSmokeTransport) -> object:
    denied_tool_name = _denied_tool_name(transport)
    if denied_tool_name is None:
        return _skip("not_configured")
    response = await transport.request(
        {
            "jsonrpc": "2.0",
            "id": "smoke-policy-denial",
            "method": "tools/call",
            "params": {
                "name": denied_tool_name,
                "arguments": {},
            },
        }
    )
    return _expect_jsonrpc_error(
        response,
        expected_code=_POLICY_DENIED,
        reason_code="policy_denial_not_enforced",
    )


def _outcome_to_step(
    name: str,
    outcome: object,
    started: float,
    *,
    method: str | None,
) -> SmokeStepReport:
    if isinstance(outcome, _StepOutcome):
        return SmokeStepReport(
            name=name,
            ok=outcome.ok,
            method=method,
            elapsed_ms=_elapsed_ms(started),
            result_summary=summarize_result(outcome.detail) if outcome.ok else None,
            error_code=outcome.error_code,
            reason_code=outcome.reason_code,
            detail=redact_detail(outcome.detail),
        )
    return SmokeStepReport(
        name=name,
        ok=True,
        method=method,
        elapsed_ms=_elapsed_ms(started),
        result_summary=summarize_result(outcome),
    )


class _StepOutcome:
    def __init__(
        self,
        *,
        ok: bool,
        reason_code: str,
        detail: object | None = None,
        error_code: int | None = None,
    ) -> None:
        self.ok = ok
        self.reason_code = reason_code
        self.detail = detail
        self.error_code = error_code


def _failure(
    reason_code: str,
    *,
    detail: object | None = None,
    error_code: int | None = None,
) -> _StepOutcome:
    return _StepOutcome(
        ok=False,
        reason_code=reason_code,
        detail=detail,
        error_code=error_code,
    )


def _skip(reason_code: str, *, detail: object | None = None) -> _StepOutcome:
    return _StepOutcome(ok=True, reason_code=reason_code, detail=detail)


def _expect_jsonrpc_error(
    response: object,
    *,
    expected_code: int,
    reason_code: str,
) -> object:
    code = _jsonrpc_error_code(response)
    if code != expected_code:
        return _failure(reason_code, detail=response, error_code=code)
    return response


def _validate_jsonrpc_success_response(
    response: object,
    *,
    expected_id: str,
) -> str | None:
    if not isinstance(response, dict):
        return "invalid_batch_item"
    if response.get("jsonrpc") != "2.0":
        return "invalid_batch_item"
    if response.get("id") != expected_id:
        return "invalid_batch_correlation"
    has_result = "result" in response
    has_error = "error" in response
    if has_result == has_error:
        return "invalid_batch_item"
    if has_error:
        return "batch_item_failed"
    return None


def _jsonrpc_error_code(response: object) -> int | None:
    if not isinstance(response, dict):
        return None
    error = response.get("error")
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    if isinstance(code, int):
        return code
    return None


def _client_error_code(exc: McpSmokeClientError) -> int | None:
    if isinstance(exc.error, dict):
        code = exc.error.get("code")
        if isinstance(code, int):
            return code
    return None


def _is_successful_ping_result(result: object) -> bool:
    return result == {"pong": True} or result == {}


def _capability_available(state: dict[str, Any], capability_name: str) -> bool:
    capabilities = state.get("capabilities")
    if not isinstance(capabilities, dict):
        return False
    capability = capabilities.get(capability_name)
    return isinstance(capability, dict) and capability.get("available") is True


def _first_named_value(items: list[object], key: str) -> object | None:
    first = items[0] if items else None
    if isinstance(first, dict):
        return first.get(key)
    return None


def _denied_tool_name(transport: McpSmokeTransport) -> str | None:
    direct = getattr(transport, "denied_tool_name", None)
    if isinstance(direct, str) and direct:
        return direct
    runtime = getattr(transport, "runtime", None)
    runtime_tool = getattr(runtime, "denied_tool_name", None)
    if isinstance(runtime_tool, str) and runtime_tool:
        return runtime_tool
    return None


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 3)


__all__ = ["ScenarioMode", "run_baseline_scenario"]
