"""Framework-neutral moderation execution data and fixed safety ceilings.

This module performs no I/O and never compiles or executes a regular expression.
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, fields
from typing import Literal

from tldw_Server_API.app.core.exception_types import ModerationExecutionError
from tldw_Server_API.app.core.Moderation.models import ModerationEvaluationResult

BaseOperation = Literal["evaluate", "redact", "redact_with_count"]
Operation = BaseOperation | Literal["supervised_evaluate", "custom_substitute", "redact_many"]
OPERATIONS = frozenset({"evaluate", "redact", "redact_with_count", "supervised_evaluate", "custom_substitute", "redact_many"})
MAX_TEXT_FIELDS = 4096


@dataclass(frozen=True)
class ExecutionBounds:
    """Per-operation safety configuration, which may only tighten the ceilings."""

    timeout_s: float = 2.0
    max_workers: int = 4
    max_input_chars: int = 1_000_000
    max_output_chars: int = 1_000_000
    max_rules: int = 256
    max_pattern_chars: int = 4_096
    max_replacement_chars: int = 4_096
    max_policy_bytes: int = 256 * 1024
    max_request_bytes: int = 32 * 1024 * 1024
    max_response_bytes: int = 8 * 1024 * 1024

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            valid_type = type(value) in (int, float) if item.name == "timeout_s" else type(value) is int
            if not valid_type or value <= 0 or value > item.default or not math.isfinite(value):
                raise ValueError("invalid moderation execution bound")


@dataclass(frozen=True)
class FailureMetadata:
    """Allowlisted failure context, including lengths rejected at admission."""

    operation: Operation
    phase: str | None
    input_chars: int

    def __post_init__(self) -> None:
        if (
            type(self.operation) is not str
            or self.operation not in OPERATIONS
            or self.phase is not None
            and type(self.phase) is not str
            or type(self.input_chars) is not int
            or self.input_chars < 0
        ):
            raise ValueError("invalid moderation failure metadata")
        if self.phase not in {"input", "output"}:
            object.__setattr__(self, "phase", None)


@dataclass
class ExecutionScope:
    """A shared operation deadline, cancellation signal and report-once claim."""

    deadline: float
    cancelled: threading.Event
    correlation_id: str
    _reported: bool = field(default=False, init=False, repr=False)
    _report_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def claim_report(self) -> bool:
        """Atomically claim the one incident report for this operation."""
        with self._report_lock:
            if self._reported:
                return False
            self._reported = True
            return True


_CURRENT_SCOPE: ContextVar[ExecutionScope | None] = ContextVar("moderation_execution_scope", default=None)


@contextmanager
def execution_scope(*, timeout_s: float = 2.0) -> Iterator[ExecutionScope]:
    """Inherit an outer scope rather than refreshing its deadline or claim."""
    ExecutionBounds(timeout_s=timeout_s)
    existing = _CURRENT_SCOPE.get()
    if existing is not None:
        yield existing
        return
    scope = ExecutionScope(time.monotonic() + timeout_s, threading.Event(), uuid.uuid4().hex)
    token = _CURRENT_SCOPE.set(scope)
    try:
        yield scope
    finally:
        _CURRENT_SCOPE.reset(token)


@dataclass(frozen=True)
class OutputBudget:
    """Check prospective character allocations before slices, expansion or joins."""

    max_chars: int

    def __post_init__(self) -> None:
        if type(self.max_chars) is not int or not 0 <= self.max_chars <= 1_000_000:
            raise ValueError("invalid moderation output budget")

    def check(self, chars: int) -> None:
        """Reject growth without embedding content in the exception."""
        if chars > self.max_chars:
            raise ModerationExecutionError("size_exceeded")


@dataclass(frozen=True)
class RegexRuleSnapshot:
    """Ordered Guardian matching metadata without parent-only policy details."""

    pattern: str
    flags: int
    action: str | None
    phase: str


@dataclass(frozen=True)
class SubstitutionRuleSnapshot:
    """One sequential stdlib template-substitution rule."""

    pattern: str
    flags: int
    replacement: str


@dataclass(frozen=True)
class SupervisedBatchResult:
    """Selected snapshot index/span and optional batch redaction."""

    winner_index: int | None
    match_span: tuple[int, int] | None
    redacted_text: str | None


@dataclass(frozen=True)
class SubstitutionBatchResult:
    """Sequential substitution output, count and ordered skipped indices."""

    text: str
    count: int
    skipped_rule_indices: tuple[int, ...]


@dataclass(frozen=True)
class WorkerRequest:
    """A data-only operation snapshot, independently validated at the wire."""

    operation: Operation
    text: str | tuple[str, ...]
    phase: str | None
    policy: dict
    limits: dict
    bounds: ExecutionBounds


def request_input_chars(request: WorkerRequest) -> int:
    """Count submitted characters, not independent fields, for safe metadata."""
    return sum(map(len, request.text)) if isinstance(request.text, tuple) else len(request.text)


@dataclass(frozen=True)
class WorkerResponse:
    """A typed operation result or allowlisted content-free failure."""

    # None is reserved for failures before request decoding (e.g. limit failure).
    operation: Operation | None
    result: (
        ModerationEvaluationResult | str | list[str] | tuple[str, int] | SupervisedBatchResult | SubstitutionBatchResult | None
    ) = None
    reason: str | None = None
    resource_limits: dict[str, bool] | None = None
