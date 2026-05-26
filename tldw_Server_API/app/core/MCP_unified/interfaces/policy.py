from __future__ import annotations

from typing import Any, Protocol


class EffectivePolicyResolver(Protocol):
    async def resolve_for_context(
        self,
        *,
        user_id: str | None,
        metadata: dict[str, Any],
    ) -> dict[str, Any] | None: ...


class ApprovalEvaluator(Protocol):
    async def evaluate_tool_call(
        self,
        *,
        effective_policy: dict[str, Any],
        tool_name: str,
        tool_args: Any,
        context: Any,
        tool_def: dict[str, Any] | None,
        is_write: bool | None,
        within_effective_policy: bool,
        force_approval: bool = False,
        approval_reason: str | None = None,
        scope_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


class PathScopeEnforcer(Protocol):
    async def evaluate_tool_call(
        self,
        *,
        effective_policy: dict[str, Any] | None,
        context: Any,
        tool_name: str,
        tool_args: Any,
        tool_def: dict[str, Any] | None,
    ) -> dict[str, Any]: ...


class ExternalAccessEvaluator(Protocol):
    async def resolve_for_sources(
        self,
        *,
        sources: list[dict[str, Any]],
        effective_policy: dict[str, Any],
    ) -> dict[str, Any]: ...
