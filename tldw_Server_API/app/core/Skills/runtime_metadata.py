"""Helpers for exposing read-only skill runtime declaration metadata."""

from __future__ import annotations

from collections.abc import Sequence

SkillRuntimeMetadataDict = dict[str, str | bool | int | None]


def _normalize_allowed_tool_declarations(allowed_tools: Sequence[str] | str | None) -> list[str]:
    """Return declared tool strings without treating a single string as a sequence."""
    if allowed_tools is None:
        return []
    if isinstance(allowed_tools, str):
        return [allowed_tools] if allowed_tools else []
    return [tool for tool in allowed_tools if isinstance(tool, str)]


def build_skill_runtime_metadata(
    *,
    context: str | None,
    allowed_tools: Sequence[str] | str | None,
    model: str | None,
    disable_model_invocation: bool | None,
) -> SkillRuntimeMetadataDict:
    """Derive structured runtime declarations from existing skill metadata."""
    execution_mode = "fork" if context == "fork" else "inline"
    declared_tool_count = len(_normalize_allowed_tool_declarations(allowed_tools))

    return {
        "execution_mode": execution_mode,
        "test_run_may_call_model": execution_mode == "fork",
        "declares_tools": declared_tool_count > 0,
        "declared_tool_count": declared_tool_count,
        "model_override": model,
        "auto_invocation_enabled": not bool(disable_model_invocation),
    }
