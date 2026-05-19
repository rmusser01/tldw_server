"""Session-aware run-first presentation helpers for ACP MCP tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_llm_caller import (
    mcp_tools_to_openai_format,
)

_RUN_TOOL_NAME = "run"
_RUN_FIRST_PROMPT_FRAGMENT = (
    "ACP run-first guidance: prefer `run(command)` for exploratory or multi-step "
    "work. Use typed MCP tools when `run` is unavailable or when a direct "
    "structured tool is the clearer governed path."
)


@dataclass(frozen=True)
class ACPToolPresentation:
    """Resolved tool presentation for one ACP session."""

    openai_tools: list[dict[str, Any]]
    effective_tool_names: list[str]
    prompt_fragment: str | None
    presentation_variant: str
    eligible: bool
    ineligible_reason: str | None


def _matches_provider_allowlist(
    provider_key: str | None,
    provider_allowlist: list[str] | None,
    *,
    fail_closed_when_empty: bool = False,
) -> bool:
    if provider_allowlist is None:
        return True

    candidate = str(provider_key or "").strip().lower()
    if not candidate:
        return False

    normalized_allowlist = {
        str(item or "").strip().lower() for item in provider_allowlist if str(item or "").strip()
    }
    if not normalized_allowlist:
        return not fail_closed_when_empty
    return candidate in normalized_allowlist


def _shape_description(description: Any, *, eligible: bool, is_run: bool) -> str:
    text = str(description or "").strip()
    if not eligible:
        return text

    if is_run:
        if text:
            return f"{text} Preferred first tool for multi-step work."
        return "Preferred first tool for multi-step work."

    if text:
        return f"Fallback tool: {text}"
    return "Fallback tool for specialized use."


def _normalize_tools(tools: list[dict[str, Any]] | None) -> tuple[list[dict[str, Any]], list[str]]:
    normalized_tools: list[dict[str, Any]] = []
    effective_tool_names: list[str] = []
    seen_names: set[str] = set()

    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        raw_name = tool.get("name")
        if not isinstance(raw_name, str):
            continue
        tool_name = raw_name.strip()
        if not tool_name or tool_name in seen_names:
            continue
        seen_names.add(tool_name)
        effective_tool_names.append(tool_name)
        raw_schema = tool.get("inputSchema")
        normalized_tools.append(
            {
                **tool,
                "description": str(tool.get("description") or ""),
                "inputSchema": raw_schema if isinstance(raw_schema, dict) else {"type": "object"},
            }
        )

    return normalized_tools, effective_tool_names


def present_acp_tools(
    *,
    session_id: str,
    tools: list[dict[str, Any]] | None,
    rollout_mode: str = "off",
    provider_key: str | None = None,
    provider_allowlist: list[str] | None = None,
    presentation_variant: str = "acp_phase2a_v1",
) -> ACPToolPresentation:
    """Resolve one ACP session's tool presentation for the LLM surface."""

    del session_id

    normalized_tools, effective_tool_names = _normalize_tools(tools)
    rollout_mode_name = str(rollout_mode or "").strip().lower()
    rollout_enabled = rollout_mode_name in {"gated", "default_on", "on", "enabled", "true", "1"}
    run_present = _RUN_TOOL_NAME in effective_tool_names
    provider_allowed = _matches_provider_allowlist(
        provider_key,
        provider_allowlist,
        fail_closed_when_empty=rollout_mode_name == "default_on",
    )
    eligible = rollout_enabled and run_present and provider_allowed

    if not effective_tool_names:
        ineligible_reason = "no_tools"
    elif not rollout_enabled:
        ineligible_reason = "rollout_off"
    elif not run_present:
        ineligible_reason = "run_missing"
    elif not provider_allowed:
        ineligible_reason = "provider_not_in_rollout_allowlist"
    else:
        ineligible_reason = None

    if eligible:
        ordered_tools = sorted(
            normalized_tools,
            key=lambda tool: str(tool.get("name") or "").strip() != _RUN_TOOL_NAME,
        )
    else:
        ordered_tools = normalized_tools

    presented_mcp_tools = [
        {
            **tool,
            "description": _shape_description(
                tool.get("description"),
                eligible=eligible,
                is_run=str(tool.get("name") or "").strip() == _RUN_TOOL_NAME,
            ),
        }
        for tool in ordered_tools
    ]

    return ACPToolPresentation(
        openai_tools=mcp_tools_to_openai_format(presented_mcp_tools),
        effective_tool_names=[str(tool.get("name") or "").strip() for tool in ordered_tools],
        prompt_fragment=_RUN_FIRST_PROMPT_FRAGMENT if eligible else None,
        presentation_variant=presentation_variant,
        eligible=eligible,
        ineligible_reason=ineligible_reason,
    )
