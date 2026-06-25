"""Explicit dependencies for the MCP tool execution pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .reporting import ToolExecutionReporter


@dataclass(frozen=True, slots=True)
class CompatibilityCallbackLedgerEntry:
    callback: str
    current_owner: str
    target_owner: str
    removal_stage: str
    parity_test: str


@dataclass(slots=True)
class ToolExecutionDependencies:
    module_registry: Any
    rbac_policy: Any
    rate_limiter: Any
    metrics: Any
    telemetry: Any
    hook_manager: Any
    tool_use_recorder: Any
    idempotency: Any
    config_provider: Callable[[], Any]
    effective_policy_resolver: Any
    path_scope_enforcer: Any
    approval_evaluator: Any
    external_access_evaluator: Any
    reporter: ToolExecutionReporter
    api_key_scope_normalizer: Any = None
