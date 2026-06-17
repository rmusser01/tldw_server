"""Host-neutral models for configurable MCP tool-call hooks."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from mcp_unified.interfaces.runtime import (
    ToolCallHookManager,
    ToolHookCallContext,
    ToolHookDecision,
    ToolHookPhase,
)

ToolHookResult: TypeAlias = ToolHookDecision | dict[str, Any] | None
ToolHookCallback: TypeAlias = Callable[
    [ToolHookCallContext],
    ToolHookResult | Awaitable[ToolHookResult],
]


@dataclass(frozen=True, slots=True)
class ToolHookRegistration:
    """One configured hook registration for the package-level hook manager."""

    hook_id: str
    hook: ToolCallHookManager | None = None
    before: ToolHookCallback | None = None
    after: ToolHookCallback | None = None
    order: int = 0
    enabled: bool = True
    phases: tuple[ToolHookPhase, ...] = field(default=("pre", "post"))

    def __post_init__(self) -> None:
        """Validate registration shape early for deterministic runtime behavior."""

        hook_id = str(self.hook_id).strip()
        if not hook_id:
            raise ValueError("hook_id is required")
        object.__setattr__(self, "hook_id", hook_id)
        object.__setattr__(self, "order", int(self.order))

        phases: list[ToolHookPhase] = []
        for phase in self.phases:
            if phase not in {"pre", "post"}:
                raise ValueError(f"Unsupported MCP tool hook phase: {phase!r}")
            if phase not in phases:
                phases.append(phase)
        object.__setattr__(self, "phases", tuple(phases))

        if self.hook is None and self.before is None and self.after is None:
            raise ValueError("ToolHookRegistration requires hook, before, or after")


class ToolHookExecutionError(RuntimeError):
    """Raised when a pre-tool hook fails and execution must fail closed."""

    def __init__(
        self,
        *,
        hook_id: str,
        phase: ToolHookPhase,
        error_type: str,
    ) -> None:
        super().__init__(f"MCP tool hook failed: {phase}:{hook_id}:{error_type}")
        self.hook_id = hook_id
        self.phase = phase
        self.error_type = error_type


__all__ = [
    "ToolHookCallback",
    "ToolHookExecutionError",
    "ToolHookRegistration",
    "ToolHookResult",
]
