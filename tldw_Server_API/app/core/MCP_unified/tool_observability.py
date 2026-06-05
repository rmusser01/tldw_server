"""Shared helpers for MCP tool observability metadata."""

from __future__ import annotations

from collections.abc import Iterable
import re
from typing import Any, TypeAlias


ToolEvalValue: TypeAlias = str | list[str]
ToolEvalMetadata: TypeAlias = dict[str, dict[str, ToolEvalValue]]
ExecutionEvalValue: TypeAlias = str | bool | int | float
ExecutionEvalMetadata: TypeAlias = dict[str, ExecutionEvalValue]

DEFAULT_TOOL_PROMPT_VERSION = "2026.06.04"
_TOOL_PROMPT_ID_INVALID_CHARS = re.compile(r"[^a-z0-9_.-]+")
_WRITE_CATEGORIES = frozenset({"ingestion", "management", "write", "mutation"})


def _clean_list(values: Iterable[str]) -> list[str]:
    """Return stripped, non-blank values while preserving input order."""
    return [cleaned for value in values if isinstance(value, str) and (cleaned := value.strip())]


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


def _metadata_dict(tool_def: dict[str, Any] | None) -> dict[str, Any]:
    metadata = (tool_def or {}).get("metadata") if isinstance(tool_def, dict) else None
    return dict(metadata) if isinstance(metadata, dict) else {}


def _clean_tool_prompt_name(tool_name: str) -> str:
    cleaned = _TOOL_PROMPT_ID_INVALID_CHARS.sub("_", tool_name.strip().lower())
    cleaned = cleaned.strip("._-")
    return cleaned or "unknown"


def _first_tool_family(tool_name: str) -> str:
    cleaned = _clean_tool_prompt_name(tool_name)
    family = cleaned.split(".", 1)[0].strip()
    return family or "general"


def _infer_category(tool_name: str, metadata: dict[str, Any]) -> str:
    category = _optional_string(str(metadata.get("category"))) if metadata.get("category") is not None else None
    return category or _first_tool_family(tool_name)


def _infer_prompt_variant(metadata: dict[str, Any], prompt_variant: str | None) -> str:
    explicit = _optional_string(prompt_variant)
    if explicit is not None:
        return explicit
    if metadata.get("federated") is True or any(
        key in metadata for key in ("external_server_id", "server_id", "upstream_tool")
    ):
        return "external_federated"
    if metadata.get("canonical_tool"):
        return "alias"
    return "builtin"


def _infer_success_signals(metadata: dict[str, Any], category: str) -> list[str]:
    signals = ["completed_without_error"]
    if bool(metadata.get("readOnlyHint")):
        signals.insert(0, "avoided_mutation")
    if bool(metadata.get("write_capable")) or bool(metadata.get("is_write")) or category.lower() in _WRITE_CATEGORIES:
        signals.append("completed_requested_mutation")
    return signals


def _clean_existing_tool_eval(value: Any) -> dict[str, ToolEvalValue] | None:
    if not isinstance(value, dict):
        return None

    tool_prompt_id = _optional_string(value.get("tool_prompt_id"))
    tool_prompt_version = _optional_string(value.get("tool_prompt_version"))
    expected_result_kind = _optional_string(value.get("expected_result_kind"))
    if tool_prompt_id is None or tool_prompt_version is None or expected_result_kind is None:
        return None

    raw_task_families = value.get("task_families")
    task_families = _clean_list(raw_task_families) if isinstance(raw_task_families, list) else []
    raw_success_signals = value.get("success_signals")
    success_signals = _clean_list(raw_success_signals) if isinstance(raw_success_signals, list) else []

    return {
        "tool_prompt_id": tool_prompt_id,
        "tool_prompt_version": tool_prompt_version,
        "task_families": task_families,
        "expected_result_kind": expected_result_kind,
        "success_signals": success_signals,
        "prompt_variant": _optional_string(value.get("prompt_variant")) or "builtin",
    }


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
            "tool_prompt_version": _required_string(tool_prompt_version, "tool_prompt_version"),
            "task_families": _clean_list(task_families),
            "expected_result_kind": _required_string(expected_result_kind, "expected_result_kind"),
            "success_signals": _clean_list(success_signals),
            "prompt_variant": _optional_string(prompt_variant) or "builtin",
        }
    }


def infer_tool_eval_metadata(
    *,
    tool_name: str,
    metadata: dict[str, Any] | None = None,
    prompt_variant: str | None = None,
) -> ToolEvalMetadata:
    """Infer stable evaluation metadata for a tool definition."""
    cleaned_tool_name = _required_string(tool_name, "tool_name")
    metadata = dict(metadata or {})
    category = _infer_category(cleaned_tool_name, metadata)
    return build_tool_eval_metadata(
        tool_prompt_id=f"mcp.{_clean_tool_prompt_name(cleaned_tool_name)}.v1",
        tool_prompt_version=DEFAULT_TOOL_PROMPT_VERSION,
        task_families=[category],
        expected_result_kind=f"{category}_result",
        success_signals=_infer_success_signals(metadata, category),
        prompt_variant=_infer_prompt_variant(metadata, prompt_variant),
    )


def ensure_tool_definition_eval_metadata(
    tool_def: dict[str, Any],
    *,
    prompt_variant: str | None = None,
) -> dict[str, Any]:
    """Return a tool definition copy with safe evaluation metadata attached."""
    if not isinstance(tool_def, dict):
        return tool_def

    normalized = dict(tool_def)
    tool_name = _required_string(str(normalized.get("name") or ""), "tool_name")
    metadata = _metadata_dict(normalized)
    existing_eval = _clean_existing_tool_eval(metadata.get("eval"))
    if existing_eval is None:
        metadata.update(
            infer_tool_eval_metadata(
                tool_name=tool_name,
                metadata=metadata,
                prompt_variant=prompt_variant,
            )
        )
    else:
        metadata["eval"] = existing_eval
    normalized["metadata"] = metadata
    return normalized


def execution_eval_metadata_from_tool_definition(
    *,
    tool_name: str,
    tool_def: dict[str, Any] | None,
    profile_id: str | None = None,
    path_filter_used: bool | None = None,
    truncated: bool | None = False,
    reason_code: str | None = None,
    duration_ms: float | None = None,
) -> ExecutionEvalMetadata:
    """Build execution eval metadata from the matching tool definition metadata."""
    normalized_tool_def = ensure_tool_definition_eval_metadata(
        tool_def if isinstance(tool_def, dict) else {"name": tool_name}
    )
    definition_eval = _metadata_dict(normalized_tool_def).get("eval")
    if not isinstance(definition_eval, dict):
        definition_eval = infer_tool_eval_metadata(tool_name=tool_name)["eval"]

    task_families = definition_eval.get("task_families")
    action_family = _clean_list(task_families) if isinstance(task_families, list) else []
    return build_execution_eval_metadata(
        tool_name=tool_name,
        tool_prompt_id=str(definition_eval.get("tool_prompt_id") or f"mcp.{_clean_tool_prompt_name(tool_name)}.v1"),
        tool_prompt_version=str(definition_eval.get("tool_prompt_version") or DEFAULT_TOOL_PROMPT_VERSION),
        action_family=action_family[0] if action_family else _first_tool_family(tool_name),
        result_kind=str(definition_eval.get("expected_result_kind") or f"{_first_tool_family(tool_name)}_result"),
        profile_id=profile_id,
        path_filter_used=path_filter_used,
        truncated=truncated,
        reason_code=reason_code,
        duration_ms=duration_ms,
    )


def attach_execution_eval_metadata(
    result: Any,
    *,
    tool_name: str,
    tool_def: dict[str, Any] | None,
    profile_id: str | None = None,
    path_filter_used: bool | None = None,
    truncated: bool | None = False,
    reason_code: str | None = None,
    duration_ms: float | None = None,
) -> Any:
    """Attach safe eval metadata to a structured tool result copy when absent."""
    if not isinstance(result, dict) or isinstance(result.get("eval"), dict):
        return result

    enriched = dict(result)
    enriched["eval"] = execution_eval_metadata_from_tool_definition(
        tool_name=tool_name,
        tool_def=tool_def,
        profile_id=profile_id,
        path_filter_used=path_filter_used,
        truncated=truncated,
        reason_code=reason_code,
        duration_ms=duration_ms,
    )
    return enriched


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
        "tool_prompt_version": _required_string(tool_prompt_version, "tool_prompt_version"),
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
