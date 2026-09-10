"""Explicit dependencies for the MCP tool execution pipeline."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from .reporting import ToolExecutionReporter

if TYPE_CHECKING:
    from .models import IdempotencyExecutionPolicy, IdempotencyRunResult


class IdempotencyExecutor(Protocol):
    """Narrow idempotency dependency consumed by the execution runtime."""

    async def execute(
        self,
        cache_key: str,
        arguments_hash: str,
        execute_fn: Callable[[], Awaitable[dict[str, Any]]],
        *,
        policy: IdempotencyExecutionPolicy,
    ) -> IdempotencyRunResult: ...

    async def shutdown(self) -> None: ...


@dataclass(frozen=True, slots=True)
class CompatibilityCallbackLedgerEntry:
    """Document a temporary protocol compatibility callback during extraction."""

    callback: str
    current_owner: str
    target_owner: str
    removal_stage: str
    parity_test: str


@dataclass(slots=True)
class ToolExecutionDependencies:
    """Dependency bundle shared by extracted MCP tool-execution stages."""

    module_registry: Any
    rbac_policy: Any
    rate_limiter: Any
    metrics: Any
    telemetry: Any
    hook_manager: Any
    tool_use_recorder: Any
    idempotency: IdempotencyExecutor
    config_provider: Callable[[], Any]
    effective_policy_resolver: Any
    path_scope_enforcer: Any
    approval_evaluator: Any
    external_access_evaluator: Any
    reporter: ToolExecutionReporter
    api_key_scope_normalizer: Any = None
