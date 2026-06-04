"""Shared helpers for MCP tool observability metadata."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeAlias


ToolEvalValue: TypeAlias = str | list[str]
ToolEvalMetadata: TypeAlias = dict[str, dict[str, ToolEvalValue]]
ExecutionEvalValue: TypeAlias = str | bool | int | float
ExecutionEvalMetadata: TypeAlias = dict[str, ExecutionEvalValue]


def _clean_list(values: Iterable[str]) -> list[str]:
    """Return stripped, non-blank values while preserving input order."""
    return [cleaned for value in values if (cleaned := str(value).strip())]


def _required_string(value: str, field_name: str) -> str:
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be blank")
    return cleaned


def _optional_string(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def build_tool_eval_metadata(
    *,
    tool_prompt_id: str,
    tool_prompt_version: str,
    task_families: Iterable[str],
    expected_result_kind: str,
    success_signals: Iterable[str],
    prompt_variant: str = "builtin",
) -> ToolEvalMetadata:
    """Build stable evaluation metadata for an MCP tool definition."""
    return {
        "eval": {
            "tool_prompt_id": _required_string(tool_prompt_id, "tool_prompt_id"),
            "tool_prompt_version": _required_string(
                tool_prompt_version, "tool_prompt_version"
            ),
            "task_families": _clean_list(task_families),
            "expected_result_kind": _required_string(
                expected_result_kind, "expected_result_kind"
            ),
            "success_signals": _clean_list(success_signals),
            "prompt_variant": _optional_string(prompt_variant) or "builtin",
        }
    }


def build_execution_eval_metadata(
    *,
    tool_name: str,
    tool_prompt_id: str,
    tool_prompt_version: str,
    action_family: str,
    result_kind: str,
    profile_id: str | None = None,
    path_filter_used: bool | None = None,
    truncated: bool | None = False,
    reason_code: str | None = None,
    duration_ms: float | None = None,
) -> ExecutionEvalMetadata:
    """Build non-sensitive scalar metadata for an MCP tool execution result."""
    metadata: ExecutionEvalMetadata = {
        "tool_name": _required_string(tool_name, "tool_name"),
        "tool_prompt_id": _required_string(tool_prompt_id, "tool_prompt_id"),
        "tool_prompt_version": _required_string(
            tool_prompt_version, "tool_prompt_version"
        ),
        "action_family": _required_string(action_family, "action_family"),
        "result_kind": _required_string(result_kind, "result_kind"),
    }

    clean_profile_id = _optional_string(profile_id)
    if clean_profile_id is not None:
        metadata["profile_id"] = clean_profile_id
    if path_filter_used is not None:
        metadata["path_filter_used"] = bool(path_filter_used)
    if truncated is not None:
        metadata["truncated"] = bool(truncated)
    clean_reason_code = _optional_string(reason_code)
    if clean_reason_code is not None:
        metadata["reason_code"] = clean_reason_code
    if duration_ms is not None:
        metadata["duration_ms"] = float(duration_ms)

    return metadata
