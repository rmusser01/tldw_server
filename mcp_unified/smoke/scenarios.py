"""Built-in MCP smoke scenario runners."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from mcp_unified.smoke.client import McpSmokeClient
from mcp_unified.smoke.exceptions import McpSmokeClientError, McpSmokeTransportError
from mcp_unified.smoke.reporting import (
    SmokeReport,
    SmokeStepReport,
    redact_detail,
    summarize_result,
)
from mcp_unified.smoke.types import McpSmokeTransport

ScenarioMode = Literal["best_effort", "strict"]

_METHOD_NOT_FOUND = -32601
_INVALID_REQUEST = -32600
_INVALID_PARAMS = -32602
_POLICY_DENIED = -32001
_DEFAULT_INPUT_ARTIFACT_PATH = "input/product-brief.md"
_DEFAULT_OUTPUT_ARTIFACT_PATH = "output/user-stories.md"
_DEFAULT_ARTIFACT_READ_ARGUMENTS = {"path": "{input_path}"}
_DEFAULT_ARTIFACT_SUMMARIZE_ARGUMENTS = {"source_path": "{input_path}"}
_DEFAULT_ARTIFACT_WRITE_ARGUMENTS = {
    "path": "{output_path}",
    "content": "{derived_content}",
}
_DEFAULT_ARTIFACT_STAT_ARGUMENTS = {"path": "{output_path}"}
_PRODUCT_BRIEF_MARKDOWN = """# Portable Research Notebook

Build a local-first research notebook for analysts who collect video, audio,
web pages, and PDFs. Users need fast search, cited answers, exportable notes,
and safe automation that cannot touch files outside their workspace.
"""


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
        server_info = state.get("server_info")
        if isinstance(server_info, dict):
            report.metadata["server_info"] = server_info
        await transport.close()
        report.elapsed_ms = _elapsed_ms(started)
        report.ok = all(step.ok for step in report.steps)

    return report


async def run_real_world_scenario(
    transport: McpSmokeTransport,
    *,
    mode: ScenarioMode = "best_effort",
    artifact_dir: str | Path | None = None,
    artifact_read_tool_name: str = "artifact.read",
    artifact_summarize_tool_name: str = "artifact.summarize",
    artifact_write_tool_name: str = "artifact.write",
    artifact_stat_tool_name: str = "artifact.stat",
    artifact_read_arguments: dict[str, object] | None = None,
    artifact_summarize_arguments: dict[str, object] | None = None,
    artifact_write_arguments: dict[str, object] | None = None,
    artifact_stat_arguments: dict[str, object] | None = None,
    real_llm_provider: str | None = None,
    real_llm_api_key_env: str | None = None,
    real_llm_base_url: str | None = None,
    real_llm_model: str | None = None,
    real_llm_http_client: Any | None = None,
) -> SmokeReport:
    """Run the real-world MCP UAT scenario with isolated artifacts."""

    if mode not in {"best_effort", "strict"}:
        raise ValueError("mode must be 'best_effort' or 'strict'")

    artifact_root = _prepare_artifact_root(artifact_dir)
    started_at = datetime.now(UTC).isoformat()
    report = SmokeReport(
        transport=transport.__class__.__name__,
        started_at=started_at,
        metadata={
            "mode": mode,
            "scenario": "real-world",
            "artifacts": {
                "input_path": _DEFAULT_INPUT_ARTIFACT_PATH,
                "output_path": _DEFAULT_OUTPUT_ARTIFACT_PATH,
            },
        },
    )
    started = perf_counter()
    client = McpSmokeClient(transport)
    state: dict[str, Any] = {
        "capabilities": {},
        "tools": [],
        "artifact_root": artifact_root,
        "input_path": _DEFAULT_INPUT_ARTIFACT_PATH,
        "output_path": _DEFAULT_OUTPUT_ARTIFACT_PATH,
        "artifact_read_tool_name": artifact_read_tool_name,
        "artifact_summarize_tool_name": artifact_summarize_tool_name,
        "artifact_write_tool_name": artifact_write_tool_name,
        "artifact_stat_tool_name": artifact_stat_tool_name,
        "artifact_read_arguments": artifact_read_arguments
        if artifact_read_arguments is not None
        else dict(_DEFAULT_ARTIFACT_READ_ARGUMENTS),
        "artifact_summarize_arguments": artifact_summarize_arguments
        if artifact_summarize_arguments is not None
        else dict(_DEFAULT_ARTIFACT_SUMMARIZE_ARGUMENTS),
        "artifact_write_arguments": artifact_write_arguments
        if artifact_write_arguments is not None
        else dict(_DEFAULT_ARTIFACT_WRITE_ARGUMENTS),
        "artifact_stat_arguments": artifact_stat_arguments
        if artifact_stat_arguments is not None
        else dict(_DEFAULT_ARTIFACT_STAT_ARGUMENTS),
        "safe_prompt_name": None,
        "safe_prompt_arguments": {"topic": "real-world MCP UAT"},
        "real_llm_provider": real_llm_provider,
        "real_llm_api_key_env": real_llm_api_key_env,
        "real_llm_base_url": real_llm_base_url,
        "real_llm_model": real_llm_model,
        "real_llm_http_client": real_llm_http_client,
    }

    await transport.start()
    try:
        await _record_step(report, "initialize", lambda: _step_initialize(client, state))
        await _record_step(
            report,
            "notifications/initialized",
            lambda: _step_initialized_notification(client),
        )
        await _record_step(report, "tools/list", lambda: _step_tools_list(client, state))
        await _record_step(
            report,
            "artifact setup",
            lambda: _step_artifact_setup(state),
        )
        await _record_step(
            report,
            "artifact read",
            lambda: _step_artifact_read(client, state, mode),
            method="tools/call",
        )
        await _record_step(
            report,
            "prompts",
            lambda: _step_prompts(client, state, mode),
        )
        await _record_step(
            report,
            "artifact summarize",
            lambda: _step_artifact_summarize(client, state, mode),
            method="tools/call",
        )
        await _record_step(
            report,
            "artifact write",
            lambda: _step_artifact_write(client, state, mode),
            method="tools/call",
        )
        await _record_step(
            report,
            "artifact verification",
            lambda: _step_artifact_verification(client, state, mode),
            method="tools/call",
        )
        await _record_step(
            report,
            "real LLM",
            lambda: _step_real_llm(state, mode),
        )
    finally:
        server_info = state.get("server_info")
        if isinstance(server_info, dict):
            report.metadata["server_info"] = server_info
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
    """Run one smoke scenario step and append its redacted outcome to the report."""

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
    except McpSmokeTransportError as exc:
        report.steps.append(
            SmokeStepReport(
                name=name,
                ok=False,
                method=method or exc.method,
                elapsed_ms=_elapsed_ms(started),
                reason_code=exc.reason_code,
                detail=redact_detail(
                    {"type": exc.__class__.__name__, "message": str(exc)}
                ),
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
    state["server_info"] = server_info
    return result


async def _step_initialized_notification(client: McpSmokeClient) -> object:
    notification_response = await client.notify("notifications/initialized")
    if notification_response is not None:
        return _failure("notification_returned_response", detail=notification_response)
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
    return _expect_unknown_tool_error(
        response,
        expected_id="smoke-unknown-tool",
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
    expected_ids = {"smoke-batch-ping", "smoke-batch-tools"}
    if len(response) != len(expected_ids):
        return _failure("invalid_batch_response_count", detail=response)
    if not all(isinstance(item, dict) for item in response):
        return _failure("invalid_batch_item", detail=response)
    response_ids = [item.get("id") for item in response]
    if len(response_ids) != len(set(response_ids)):
        return _failure("duplicate_batch_id", detail=response)
    responses = {
        item.get("id"): item
        for item in response
        if isinstance(item, dict)
    }
    if set(responses) != expected_ids:
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
        expected_id="smoke-malformed",
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
        expected_id="smoke-policy-denial",
        expected_code=_POLICY_DENIED,
        reason_code="policy_denial_not_enforced",
    )


async def _step_artifact_setup(state: dict[str, Any]) -> object:
    artifact_root = state.get("artifact_root")
    if not isinstance(artifact_root, Path):
        return _failure("artifact_setup_failed", detail={"message": "missing artifact root"})
    input_path = state["input_path"]
    output_path = state["output_path"]
    input_target = artifact_root / input_path
    output_target = artifact_root / output_path
    try:
        input_target.parent.mkdir(parents=True, exist_ok=True)
        output_target.parent.mkdir(parents=True, exist_ok=True)
        input_target.write_text(_PRODUCT_BRIEF_MARKDOWN, encoding="utf-8")
    except OSError as exc:
        return _failure(
            "artifact_setup_failed",
            detail={"type": exc.__class__.__name__, "message": str(exc)},
        )
    state["seeded_artifact_bytes"] = len(_PRODUCT_BRIEF_MARKDOWN.encode("utf-8"))
    return {
        "input_path": input_path,
        "output_path": output_path,
        "input_bytes": state["seeded_artifact_bytes"],
    }


async def _step_artifact_read(
    client: McpSmokeClient,
    state: dict[str, Any],
    mode: ScenarioMode,
) -> object:
    tool_name = state["artifact_read_tool_name"]
    unavailable = _unavailable_tool_outcome(tool_name, state, mode)
    if unavailable is not None:
        return unavailable
    arguments = _render_artifact_arguments(state["artifact_read_arguments"], state)
    result = await client.call_tool(tool_name, arguments)
    if not isinstance(result, dict) or not isinstance(result.get("content"), list):
        return _failure("invalid_artifact_read_result", detail=result)
    text = _extract_tool_text(result)
    if text is None:
        return _failure("invalid_artifact_read_result", detail=result)
    state["artifact_read_text"] = text
    return _artifact_step_detail(result, path=state["input_path"])


async def _step_artifact_summarize(
    client: McpSmokeClient,
    state: dict[str, Any],
    mode: ScenarioMode,
) -> object:
    tool_name = state["artifact_summarize_tool_name"]
    unavailable = _unavailable_tool_outcome(tool_name, state, mode)
    if unavailable is not None:
        return unavailable
    arguments = _render_artifact_arguments(state["artifact_summarize_arguments"], state)
    result = await client.call_tool(tool_name, arguments)
    if not isinstance(result, dict) or not isinstance(result.get("content"), list):
        return _failure("invalid_artifact_summary_result", detail=result)
    summary = _extract_structured_text(result, "summary_markdown") or _extract_tool_text(
        result
    )
    if summary is None:
        return _failure("invalid_artifact_summary_result", detail=result)
    state["derived_content"] = summary
    return _artifact_step_detail(result, path=state["input_path"])


async def _step_artifact_write(
    client: McpSmokeClient,
    state: dict[str, Any],
    mode: ScenarioMode,
) -> object:
    if "derived_content" not in state:
        if mode == "strict":
            return _failure("artifact_input_unavailable")
        return _skip("artifact_input_unavailable")
    tool_name = state["artifact_write_tool_name"]
    unavailable = _unavailable_tool_outcome(tool_name, state, mode)
    if unavailable is not None:
        return unavailable
    arguments = _render_artifact_arguments(state["artifact_write_arguments"], state)
    result = await client.call_tool(tool_name, arguments)
    if not isinstance(result, dict) or not isinstance(result.get("content"), list):
        return _failure("invalid_artifact_write_result", detail=result)
    return _artifact_step_detail(result, path=state["output_path"])


async def _step_artifact_verification(
    client: McpSmokeClient,
    state: dict[str, Any],
    mode: ScenarioMode,
) -> object:
    tool_name = state["artifact_stat_tool_name"]
    unavailable = _unavailable_tool_outcome(tool_name, state, mode)
    if unavailable is not None:
        return unavailable
    arguments = _render_artifact_arguments(state["artifact_stat_arguments"], state)
    result = await client.call_tool(tool_name, arguments)
    if not isinstance(result, dict) or not isinstance(result.get("content"), list):
        return _failure("invalid_artifact_verification_result", detail=result)
    structured = result.get("structuredContent")
    if not isinstance(structured, dict):
        return _failure("invalid_artifact_verification_result", detail=result)
    if structured.get("exists") is not True:
        return _failure("artifact_verification_failed", detail=result)
    if structured.get("path") != state["output_path"]:
        return _failure("artifact_verification_failed", detail=result)
    return _artifact_step_detail(result, path=state["output_path"])


async def _step_real_llm(state: dict[str, Any], mode: ScenarioMode) -> object:
    provider = state.get("real_llm_provider")
    api_key_env = state.get("real_llm_api_key_env")
    if provider is None and api_key_env is None:
        return _skip("not_configured")
    if not isinstance(provider, str) or not provider.strip():
        return _missing_llm_configuration(mode, "real_llm_provider")
    if provider != "openai-compatible":
        return _failure(
            "real_llm_provider_unsupported",
            detail={"provider": provider},
        )
    if not isinstance(api_key_env, str) or not api_key_env.strip():
        return _missing_llm_configuration(mode, "real_llm_api_key_env")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        if mode == "strict":
            return _failure("real_llm_env_missing", detail={"env": api_key_env})
        return _skip("real_llm_env_missing", detail={"env": api_key_env})

    from mcp_unified.smoke.real_llm import call_openai_compatible

    try:
        result = await call_openai_compatible(
            api_key=api_key,
            prompt=_llm_prompt(state),
            base_url=state.get("real_llm_base_url"),
            model=state.get("real_llm_model"),
            http_client=state.get("real_llm_http_client"),
        )
    except Exception as exc:  # noqa: BLE001 - reports must capture provider failures.
        return _failure(
            "real_llm_call_failed",
            detail={"type": exc.__class__.__name__, "message": str(exc)},
        )
    return result


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
    expected_id: str,
    expected_code: int,
    reason_code: str,
) -> object:
    validation = _validate_jsonrpc_error_response(
        response,
        expected_id=expected_id,
        expected_code=expected_code,
    )
    if validation is not None:
        reason, code = validation
        if reason == "unexpected_error_code":
            return _failure(reason_code, detail=response, error_code=code)
        return _failure(reason, detail=response, error_code=code)
    return response


def _expect_unknown_tool_error(
    response: object,
    *,
    expected_id: str,
    reason_code: str,
) -> object:
    validation = _validate_jsonrpc_error_response(
        response,
        expected_id=expected_id,
        expected_code=_METHOD_NOT_FOUND,
    )
    if validation is None:
        return response

    reason, code = validation
    if (
        reason == "unexpected_error_code"
        and code == _INVALID_PARAMS
        and _jsonrpc_error_message_indicates_unknown_tool(response)
    ):
        return response
    if reason == "unexpected_error_code":
        return _failure(reason_code, detail=response, error_code=code)
    return _failure(reason, detail=response, error_code=code)


def _validate_jsonrpc_error_response(
    response: object,
    *,
    expected_id: str,
    expected_code: int,
) -> tuple[str, int | None] | None:
    if not isinstance(response, dict):
        return ("malformed_error_envelope", None)
    code = _jsonrpc_error_code(response)
    if response.get("jsonrpc") != "2.0":
        return ("malformed_error_envelope", code)
    if response.get("id") != expected_id:
        return ("malformed_error_envelope", code)
    has_result = "result" in response
    has_error = "error" in response
    if has_result or not has_error:
        return ("malformed_error_envelope", code)
    error = response.get("error")
    if not isinstance(error, dict):
        return ("malformed_error_envelope", None)
    if not isinstance(code, int):
        return ("malformed_error_envelope", None)
    message = error.get("message")
    if not isinstance(message, str) or not message.strip():
        return ("malformed_error_envelope", code)
    if code != expected_code:
        return ("unexpected_error_code", code)
    return None


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


def _jsonrpc_error_message_indicates_unknown_tool(response: object) -> bool:
    if not isinstance(response, dict):
        return False
    error = response.get("error")
    if not isinstance(error, dict):
        return False
    message = error.get("message")
    if not isinstance(message, str):
        return False
    normalized = message.lower()
    return (
        "unknown tool" in normalized
        or "missing tool" in normalized
        or "tool not found" in normalized
    )


def _client_error_code(exc: McpSmokeClientError) -> int | None:
    if isinstance(exc.error, dict):
        code = exc.error.get("code")
        if isinstance(code, int):
            return code
    return None


def _is_successful_ping_result(result: object) -> bool:
    return isinstance(result, dict) and result.get("pong") is True


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


def _prepare_artifact_root(artifact_dir: str | Path | None) -> Path:
    if artifact_dir is None:
        root = Path(tempfile.mkdtemp(prefix="mcp-smoke-real-world-"))
    else:
        root = Path(artifact_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _unavailable_tool_outcome(
    tool_name: object,
    state: dict[str, Any],
    mode: ScenarioMode,
) -> _StepOutcome | None:
    if isinstance(tool_name, str) and tool_name in _available_tool_names(state):
        return None
    detail = {"tool": tool_name if isinstance(tool_name, str) else str(tool_name)}
    if mode == "strict":
        return _failure("required_tool_unavailable", detail=detail)
    return _skip("required_tool_unavailable", detail=detail)


def _available_tool_names(state: dict[str, Any]) -> set[str]:
    tools = state.get("tools")
    if not isinstance(tools, list):
        return set()
    return {
        tool["name"]
        for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("name"), str)
    }


def _render_artifact_arguments(
    template: dict[str, object],
    state: dict[str, Any],
) -> dict[str, object]:
    rendered = _render_template_value(template, state)
    if not isinstance(rendered, dict):  # pragma: no cover - type guard for callers.
        raise TypeError("artifact argument template must render to an object")
    return rendered


def _render_template_value(value: object, state: dict[str, Any]) -> object:
    if isinstance(value, str):
        output = value
        for key in (
            "input_path",
            "output_path",
            "derived_content",
            "artifact_read_text",
        ):
            replacement = state.get(key, "")
            if isinstance(replacement, str):
                output = output.replace("{" + key + "}", replacement)
        return output
    if isinstance(value, dict):
        return {key: _render_template_value(nested, state) for key, nested in value.items()}
    if isinstance(value, list):
        return [_render_template_value(item, state) for item in value]
    return value


def _extract_tool_text(result: dict[str, object]) -> str | None:
    structured_text = _extract_structured_text(result, "text")
    if structured_text is not None:
        return structured_text
    content = result.get("content")
    if not isinstance(content, list):
        return None
    for item in content:
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            return item["text"]
    return None


def _extract_structured_text(result: dict[str, object], key: str) -> str | None:
    structured = result.get("structuredContent")
    if isinstance(structured, dict) and isinstance(structured.get(key), str):
        return structured[key]
    return None


def _artifact_step_detail(result: dict[str, object], *, path: object) -> dict[str, object]:
    detail: dict[str, object] = {"path": path}
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        for key in ("exists", "sha256", "source_path"):
            if key in structured:
                detail[key] = structured[key]
        if "bytes" in structured:
            detail["byte_count"] = structured["bytes"]
    content = result.get("content")
    if isinstance(content, list):
        detail["content_blocks"] = len(content)
    return detail


def _missing_llm_configuration(mode: ScenarioMode, field: str) -> _StepOutcome:
    if mode == "strict":
        return _failure("real_llm_not_configured", detail={"field": field})
    return _skip("not_configured", detail={"field": field})


def _llm_prompt(state: dict[str, Any]) -> str:
    artifact_name = state.get("output_path", "output/user-stories.md")
    return (
        "Return compact JSON with keys status and note. "
        "Status must be ok if this smoke test can inspect the derived artifact "
        f"named {artifact_name!r}. Keep the note under 80 characters."
    )


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 3)


__all__ = ["ScenarioMode", "run_baseline_scenario", "run_real_world_scenario"]
