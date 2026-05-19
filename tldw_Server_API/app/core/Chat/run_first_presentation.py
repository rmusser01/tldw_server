"""Run-first presentation helpers for chat tool assembly.

This module keeps the rollout-specific tool shaping logic separate from the
request plumbing so chat_service can resolve one session-scoped effective tool
set and reuse it for both model-facing visibility and local auto-exec policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

_RUN_TOOL_NAME = "run"
_WILDCARD_TOKEN = "*"  # nosec B105 - wildcard token, not a secret
_RUN_FIRST_PROMPT_FRAGMENT = (
    "Run-first guidance: prefer `run(command)` for exploratory or multi-step work. "
    "Use typed tools when `run` is unavailable or when a direct structured tool "
    "is the clearer governed path."
)


@dataclass(frozen=True)
class ChatRunFirstPresentation:
    """Resolved run-first tool presentation for a chat session."""

    llm_tools: list[dict[str, Any]]
    effective_tool_names: list[str]
    prompt_fragment: str | None
    presentation_variant: str
    eligible: bool
    ineligible_reason: str | None
    tool_choice: str | None = None


def _matches_allow_catalog(tool_name: str, allow_catalog: list[str] | None) -> bool:
    if allow_catalog is None:
        return True
    if not allow_catalog:
        return False
    for pattern in allow_catalog:
        token = str(pattern or "").strip()
        if not token:
            continue
        if token == _WILDCARD_TOKEN:
            return True
        if token.endswith("*"):
            if tool_name.startswith(token[:-1]):
                return True
            continue
        if token == tool_name:
            return True
    return False


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


def _tool_name(tool: dict[str, Any]) -> str | None:
    if not isinstance(tool, dict):
        return None
    tool_type = str(tool.get("type") or "").strip().lower()
    if tool_type == "function":
        function = tool.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str):
                cleaned = name.strip()
                return cleaned or None
        return None

    if "function_declarations" in tool:
        declarations = tool.get("function_declarations")
        if isinstance(declarations, list) and declarations:
            first = declarations[0]
            if isinstance(first, dict):
                name = first.get("name")
                if isinstance(name, str):
                    cleaned = name.strip()
                    return cleaned or None
    return None


def tool_names_from_definitions(tool: dict[str, Any]) -> list[str]:
    if not isinstance(tool, dict):
        return []

    tool_type = str(tool.get("type") or "").strip().lower()
    if tool_type == "function":
        name = _tool_name(tool)
        return [name] if name else []

    declarations = tool.get("function_declarations")
    if isinstance(declarations, list):
        names: list[str] = []
        seen: set[str] = set()
        for declaration in declarations:
            if not isinstance(declaration, dict):
                continue
            raw_name = declaration.get("name")
            if not isinstance(raw_name, str):
                continue
            cleaned = raw_name.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            names.append(cleaned)
        return names

    return []


def _shape_description(description: Any, *, eligible: bool, is_run: bool) -> str | None:
    if not isinstance(description, str):
        description = ""
    text = description.strip()
    if not eligible:
        return text or None

    if is_run:
        if text:
            return f"{text} Preferred first tool for multi-step work."
        return "Preferred first tool for multi-step work."

    if text:
        return f"Fallback tool: {text}"
    return "Fallback tool for specialized use."


def _present_tool(tool: dict[str, Any], *, eligible: bool, is_run: bool) -> dict[str, Any]:
    presented = dict(tool)
    if "function" in presented and isinstance(presented["function"], dict):
        function = dict(presented["function"])
        function["description"] = _shape_description(
            function.get("description"),
            eligible=eligible,
            is_run=is_run,
        )
        presented["function"] = function
    elif "function_declarations" in presented and isinstance(presented["function_declarations"], list):
        declarations: list[dict[str, Any]] = []
        for declaration in presented["function_declarations"]:
            if not isinstance(declaration, dict):
                continue
            copied = dict(declaration)
            copied["description"] = _shape_description(
                copied.get("description"),
                eligible=eligible,
                is_run=str(copied.get("name") or "").strip() == _RUN_TOOL_NAME,
            )
            declarations.append(copied)
        if eligible:
            declarations.sort(
                key=lambda declaration: str(declaration.get("name") or "").strip() != _RUN_TOOL_NAME
            )
        presented["function_declarations"] = declarations
    return presented


def present_chat_tools(
    *,
    tools: list[dict[str, Any]] | None,
    allow_catalog: list[str] | None,
    rollout_mode: str = "off",
    provider_key: str | None = None,
    provider_allowlist: list[str] | None = None,
    streaming: bool = False,
    presentation_variant: str = "chat_phase2a_v1",
) -> ChatRunFirstPresentation:
    """Resolve chat tool visibility and run-first presentation for one session."""

    del streaming

    raw_tools = list(tools or [])
    filtered_tools: list[dict[str, Any]] = []
    effective_tool_names: list[str] = []
    seen_names: set[str] = set()

    for tool in raw_tools:
        tool_names = tool_names_from_definitions(tool)
        if not tool_names:
            continue

        allowed_names = [
            tool_name
            for tool_name in tool_names
            if _matches_allow_catalog(tool_name, allow_catalog) and tool_name not in seen_names
        ]
        if not allowed_names:
            continue

        if "function_declarations" in tool and isinstance(tool.get("function_declarations"), list):
            filtered_declarations: list[dict[str, Any]] = []
            for declaration in tool.get("function_declarations") or []:
                if not isinstance(declaration, dict):
                    continue
                raw_name = declaration.get("name")
                if not isinstance(raw_name, str):
                    continue
                cleaned = raw_name.strip()
                if cleaned not in allowed_names:
                    continue
                filtered_declarations.append(dict(declaration))
                seen_names.add(cleaned)
                effective_tool_names.append(cleaned)

            if filtered_declarations:
                copied_tool = dict(tool)
                copied_tool["function_declarations"] = filtered_declarations
                filtered_tools.append(copied_tool)
            continue

        tool_name = allowed_names[0]
        seen_names.add(tool_name)
        filtered_tools.append(tool)
        effective_tool_names.append(tool_name)

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
        ineligible_reason = "no_effective_tools"
    elif not rollout_enabled:
        ineligible_reason = "rollout_off"
    elif not run_present:
        ineligible_reason = "run_missing_after_filtering"
    elif not provider_allowed:
        ineligible_reason = "provider_not_in_rollout_allowlist"
    else:
        ineligible_reason = None

    if not eligible:
        return ChatRunFirstPresentation(
            llm_tools=filtered_tools,
            effective_tool_names=effective_tool_names,
            prompt_fragment=None,
            presentation_variant=presentation_variant,
            eligible=False,
            ineligible_reason=ineligible_reason,
            tool_choice=None,
        )

    run_tool = [tool for tool in filtered_tools if _tool_name(tool) == _RUN_TOOL_NAME]
    other_tools = [tool for tool in filtered_tools if _tool_name(tool) != _RUN_TOOL_NAME]
    presented_tools = [
        _present_tool(tool, eligible=True, is_run=True) for tool in run_tool
    ]
    presented_tools.extend(
        _present_tool(tool, eligible=True, is_run=False) for tool in other_tools
    )
    presented_effective_names: list[str] = []
    for tool in presented_tools:
        for tool_name in tool_names_from_definitions(tool):
            if tool_name not in presented_effective_names:
                presented_effective_names.append(tool_name)

    return ChatRunFirstPresentation(
        llm_tools=presented_tools,
        effective_tool_names=presented_effective_names,
        prompt_fragment=_RUN_FIRST_PROMPT_FRAGMENT,
        presentation_variant=presentation_variant,
        eligible=True,
        ineligible_reason=None,
        tool_choice=None,
    )
