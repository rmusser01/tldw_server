"""Internal models for staged MCP tool execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolResolution:
    tool_name: str
    tool_args: Any
    module: Any
    module_id: str | None
    tool_def: dict[str, Any] | None
    is_write: bool | None


@dataclass(slots=True)
class PolicyEvaluation:
    effective_policy: dict[str, Any] | None
    external_access_result: dict[str, Any] = field(default_factory=dict)
    path_scope_result: dict[str, Any] = field(default_factory=dict)
    scope_payload: dict[str, Any] | None = None
    within_effective_policy: bool = True
    within_resolved_scope: bool = True
