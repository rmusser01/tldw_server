"""Internal models for staged MCP tool execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class CanonicalJsonSnapshot:
    """Immutable canonical bytes and their lowercase SHA-256 digest."""

    encoded: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class IdempotencyExecutionPolicy:
    """All bounded idempotency decisions fixed during preparation."""

    inject_argument: bool
    ttl_seconds: int
    contention_wait_seconds: int
    finalize_seconds: int
    lock_ttl_seconds: int
    max_entries: int
    max_result_bytes: int


@dataclass(frozen=True, slots=True)
class PreparedExecutionPolicy:
    """Immutable security-relevant decisions for one prepared tool call."""

    version: Literal[1]
    effect: Literal["read", "write"]
    rate_limit_category: str
    rate_limit_fail_closed: bool
    idempotency: IdempotencyExecutionPolicy


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
