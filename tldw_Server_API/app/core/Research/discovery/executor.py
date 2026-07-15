"""Sequential, accounted execution boundary for research discovery plans.

Discovery adapters are statically registered, product-owned, trusted in-process
code. ``BoundDispatch`` is an accounting and API boundary, not a Python sandbox;
adapters that require protection from reflective closure access need process
isolation, which is outside this module's scope.
"""

from __future__ import annotations

import asyncio
import copy
import email.utils
import math
import re
import time
import uuid
import weakref
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, cast

from .contracts import (
    CREDENTIALED_ROUTE_SKIP_REASON,
    MAX_PAGINATION_CURSOR,
    AccessRoute,
    AttributionMatch,
    BudgetCeilings,
    CredentialRequirement,
    DeferredNumericCSVQueryBinding,
    DiscoveryPlan,
    DispatchAllowance,
    DispatchIntent,
    ExecutionMode,
    JSONBodyPair,
    OperationKind,
    PlannedBudgetAllowance,
    PlannedDispatchGroup,
    PlannedLogicalAttempt,
    PredicateOperator,
    QueryPair,
    RouteLimits,
    SkippedCode,
    SkippedStatus,
    SkippedTarget,
    SourcePredicate,
    SourceRouteReference,
    canonical_policy_digest,
    derive_plan_allowance,
    evaluate_source_predicate,
)
from .gateway import (
    DiscoveryGatewayError,
    DiscoveryGatewayResponse,
    DiscoveryGatewayTrace,
    reconstruct_redirect_intent,
)
from .planner import expected_dispatch_group_id, expected_logical_attempt_id
from .registry import DiscoveryRegistry

PolicyActivityCheck = Callable[[str, str], bool]
DispatchIDFactory = Callable[[], str]
MonotonicClock = Callable[[], int | float]
CancellationCheck = Callable[[], bool]
_GATEWAY_ERROR_CODES = frozenset({"request_rejected", "policy_inactive", "hop_failed", "invalid_hop_response"})
_EXECUTION_STOP_CODES = frozenset(
    {
        "aggregate_deadline_exceeded",
        "execution_cancelled",
        "cancellation_check_failed",
        "execution_clock_invalid",
    }
)
_ADAPTER_ERROR_CODES = frozenset(
    {
        "provider_rate_limited",
        "provider_response_rejected",
        "provider_payload_invalid",
        "provider_parse_limit_exceeded",
        "provider_parse_deadline_exceeded",
    }
)
_DELTA_SECONDS_RE = re.compile(r"[0-9]+\Z")


class ExecutionState(str, Enum):
    """Physical lifecycle states plus the logical-only valid-empty state."""

    RESERVED = "reserved"
    DISPATCHING = "dispatching"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    INDETERMINATE_AFTER_DISPATCH = "indeterminate_after_dispatch"
    VALID_EMPTY = "valid_empty"


PhysicalDispatchState = ExecutionState
LogicalOutcomeState = ExecutionState


class DiscoveryExecutionError(ValueError):
    """Stable executor failure containing only a sanitized code."""

    __slots__ = ("code",)

    def __init__(self, code: str) -> None:
        if type(code) is not str or not code:
            raise TypeError("execution_error_code_must_be_nonempty_string")
        self.code = code
        super().__init__(code)


def _valid_retry_after(value: object) -> bool:
    """Return whether one value is delta-seconds or strict IMF-fixdate."""
    if type(value) is not str:
        return False
    if _DELTA_SECONDS_RE.fullmatch(value) is not None:
        return True
    try:
        parsed = email.utils.parsedate_to_datetime(value)
        return email.utils.format_datetime(parsed, usegmt=True) == value
    except (TypeError, ValueError):
        return False


class DiscoveryAdapterError(ValueError):
    """Stable adapter failure containing only allowlisted metadata."""

    __slots__ = (
        "code",
        "retry_after",
        "__weakref__",
    )

    def __init__(self, code: str, *, retry_after: str | None = None) -> None:
        if type(code) is not str:
            raise TypeError("adapter_error_code_must_be_string")
        if code not in _ADAPTER_ERROR_CODES:
            raise ValueError("adapter_error_code_invalid")
        if retry_after is not None:
            if code != "provider_rate_limited":
                raise ValueError("retry_after_requires_rate_limit")
            if not _valid_retry_after(retry_after):
                raise ValueError("retry_after_invalid")
        self.code = code
        self.retry_after = retry_after
        super().__init__(code)
        _ADAPTER_ERROR_SEALS[self] = (code, retry_after)


_ADAPTER_ERROR_SEALS: weakref.WeakKeyDictionary[
    DiscoveryAdapterError,
    tuple[str, str | None],
] = weakref.WeakKeyDictionary()


def _trusted_adapter_error(error: BaseException) -> tuple[str, str | None] | None:
    """Snapshot one exact, unmodified adapter failure."""
    if type(error) is not DiscoveryAdapterError:
        return None
    try:
        code = error.code
        retry_after = error.retry_after
        if (
            type(code) is not str
            or (retry_after is not None and type(retry_after) is not str)
            or _ADAPTER_ERROR_SEALS.get(error) != (code, retry_after)
            or error.args != (code,)
            or code not in _ADAPTER_ERROR_CODES
            or (retry_after is not None and not _valid_retry_after(retry_after))
            or (code != "provider_rate_limited" and retry_after is not None)
        ):
            return None
    except Exception:  # noqa: BLE001 - malformed adapter failures fail closed.
        return None
    return code, retry_after


@dataclass(frozen=True, slots=True)
class NumericCursor:
    """A bounded nonnegative numeric pagination cursor."""

    value: int

    def __post_init__(self) -> None:
        if type(self.value) is not int or not 0 <= self.value <= MAX_PAGINATION_CURSOR:
            raise ValueError("cursor_must_be_bounded_nonnegative_integer")


@dataclass(frozen=True, slots=True)
class NumericCSVBindingValues:
    """Positive numeric values for one declared deferred CSV binding."""

    binding_id: str
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.binding_id) is not str or not self.binding_id:
            raise ValueError("binding_id_must_be_nonempty")
        if type(self.values) is not tuple or not self.values:
            raise ValueError("binding_values_must_be_nonempty_tuple")
        if any(type(value) is not int or value <= 0 for value in self.values):
            raise ValueError("binding_values_must_be_positive_integers")


@dataclass(frozen=True, slots=True)
class AttemptJournalRecord:
    """Current immutable state of one unique physical dispatch."""

    dispatch_id: str
    dispatch_group_id: str
    route_id: str
    operation_kind: OperationKind
    state: ExecutionState


@dataclass(frozen=True, slots=True)
class DispatchAccounting:
    """Cumulative journal accounting and live physical ceiling."""

    created: int
    debited: int
    released: int
    outstanding: int
    physical_ceiling: int


@dataclass(frozen=True, slots=True)
class DiscoveryExecutionUsage:
    """Immutable physical work and runtime counters for one execution."""

    physical_records: tuple[AttemptJournalRecord, ...]
    accounting: DispatchAccounting
    route_attempts: int
    pages: int
    redirects: int
    retries: int
    possible_duplicate_work: bool


@dataclass(slots=True)
class _ContinuationUsage:
    """Mutable execution-wide continuation counters."""

    redirects: int = 0
    retries: int = 0


class _ExecutionControl:
    """Latched cancellation and aggregate deadline shared by all groups."""

    __slots__ = ("_cancellation_check", "_clock", "_deadline", "_last_clock", "_stop_code")

    def __init__(
        self,
        *,
        max_wall_time_ms: int,
        monotonic_clock: MonotonicClock,
        cancellation_check: CancellationCheck | None,
    ) -> None:
        self._cancellation_check = cancellation_check
        self._clock = monotonic_clock
        self._deadline: float | None = None
        self._last_clock: float | None = None
        self._stop_code: str | None = None
        try:
            self._check_cancellation()
            started_at = self._read_clock()
            try:
                deadline = started_at + (max_wall_time_ms / 1000)
            except ArithmeticError:
                self._latch("execution_clock_invalid")
            if not math.isfinite(deadline):
                self._latch("execution_clock_invalid")
            self._deadline = deadline
            if started_at >= deadline:
                self._latch("aggregate_deadline_exceeded")
        except DiscoveryExecutionError:
            pass

    @property
    def stop_code(self) -> str | None:
        """Return the first execution-wide stop code, if one has latched."""
        return self._stop_code

    def _latch(self, code: str) -> None:
        if self._stop_code is None:
            self._stop_code = code
        raise DiscoveryExecutionError(self._stop_code)

    def latch(self, code: str) -> None:
        """Latch one executor-owned stop discovered outside a clock read."""
        if code not in _EXECUTION_STOP_CODES:
            raise RuntimeError("invalid_execution_stop_code")
        self._latch(code)

    def _check_cancellation(self) -> None:
        if self._cancellation_check is None:
            return
        try:
            cancelled = self._cancellation_check()
        except BaseException:  # noqa: BLE001 - callback failures are typed stops.
            self._latch("cancellation_check_failed")
        if type(cancelled) is not bool:
            self._latch("cancellation_check_failed")
        if cancelled:
            self._latch("execution_cancelled")

    def _read_clock(self) -> float:
        try:
            value = self._clock()
            if type(value) not in {int, float}:
                self._latch("execution_clock_invalid")
            current = float(value)
            if not math.isfinite(current):
                self._latch("execution_clock_invalid")
            if self._last_clock is not None and current < self._last_clock:
                self._latch("execution_clock_invalid")
        except DiscoveryExecutionError:
            raise
        except BaseException:  # noqa: BLE001 - clock failures are typed stops.
            self._latch("execution_clock_invalid")
        self._last_clock = current
        return current

    def checkpoint(self) -> float:
        """Validate one boundary and return time left to the absolute deadline."""
        if self._stop_code is not None:
            raise DiscoveryExecutionError(self._stop_code)
        self._check_cancellation()
        current = self._read_clock()
        if self._deadline is None:
            self._latch("execution_clock_invalid")
        remaining = self._deadline - current
        if remaining <= 0:
            self._latch("aggregate_deadline_exceeded")
        return remaining


_DEBITED_STATES = frozenset(
    {
        ExecutionState.DISPATCHING,
        ExecutionState.SUCCEEDED,
        ExecutionState.FAILED,
        ExecutionState.TIMED_OUT,
        ExecutionState.INDETERMINATE_AFTER_DISPATCH,
    }
)
_RELEASED_STATES = frozenset({ExecutionState.CANCELLED, ExecutionState.SKIPPED})


class AttemptJournal:
    """In-memory physical journal enforcing reservation/debit invariants."""

    __slots__ = ("_physical_ceiling", "_records")

    def __init__(self, *, physical_ceiling: int) -> None:
        if type(physical_ceiling) is not int or physical_ceiling < 0:
            raise DiscoveryExecutionError("physical_ceiling_must_be_nonnegative_integer")
        self._physical_ceiling = physical_ceiling
        self._records: dict[str, AttemptJournalRecord] = {}

    @property
    def physical_ceiling(self) -> int:
        """Return the immutable ceiling fixed when this journal was created."""
        return self._physical_ceiling

    @property
    def records(self) -> tuple[AttemptJournalRecord, ...]:
        """Return current records in reservation order."""
        return tuple(self._records.values())

    @property
    def accounting(self) -> DispatchAccounting:
        """Return counts satisfying the journal conservation law."""
        states = tuple(record.state for record in self._records.values())
        accounting = DispatchAccounting(
            created=len(states),
            debited=sum(state in _DEBITED_STATES for state in states),
            released=sum(state in _RELEASED_STATES for state in states),
            outstanding=sum(state is ExecutionState.RESERVED for state in states),
            physical_ceiling=self.physical_ceiling,
        )
        if accounting.created != accounting.debited + accounting.released + accounting.outstanding:
            raise RuntimeError("journal_accounting_invariant_violated")
        if accounting.debited + accounting.outstanding > accounting.physical_ceiling:
            raise RuntimeError("journal_physical_ceiling_violated")
        return accounting

    def is_pristine(self, *, physical_ceiling: int) -> bool:
        """Return whether this exact journal is empty for one new execution."""
        if (
            type(self) is not AttemptJournal
            or type(physical_ceiling) is not int
            or physical_ceiling < 0
            or type(self._physical_ceiling) is not int
            or self._physical_ceiling != physical_ceiling
            or type(self._records) is not dict
            or self._records
        ):
            return False
        try:
            accounting = self.accounting
        except Exception:  # noqa: BLE001 - corrupted injected state fails closed.
            return False
        return (
            type(accounting) is DispatchAccounting
            and all(
                type(value) is int
                for value in (
                    accounting.created,
                    accounting.debited,
                    accounting.released,
                    accounting.outstanding,
                    accounting.physical_ceiling,
                )
            )
            and accounting == DispatchAccounting(0, 0, 0, 0, physical_ceiling)
        )

    def reserve(
        self,
        *,
        dispatch_id: str,
        dispatch_group_id: str,
        route_id: str,
        operation_kind: OperationKind,
    ) -> AttemptJournalRecord:
        """Reserve one unique physical dispatch immediately before use."""
        if dispatch_id in self._records:
            raise DiscoveryExecutionError("duplicate_dispatch_id")
        if any(type(value) is not str or not value for value in (dispatch_id, dispatch_group_id, route_id)):
            raise DiscoveryExecutionError("journal_identifiers_must_be_nonempty")
        if not isinstance(operation_kind, OperationKind):
            raise DiscoveryExecutionError("operation_kind_must_be_typed")
        accounting = self.accounting
        if accounting.debited + accounting.outstanding >= self.physical_ceiling:
            raise DiscoveryExecutionError("physical_dispatch_ceiling_exhausted")
        record = AttemptJournalRecord(
            dispatch_id,
            dispatch_group_id,
            route_id,
            operation_kind,
            ExecutionState.RESERVED,
        )
        self._records[dispatch_id] = record
        return record

    def mark_dispatching(self, dispatch_id: str) -> AttemptJournalRecord:
        """Cross the nonrefundable debit boundary."""
        return self._transition(dispatch_id, ExecutionState.RESERVED, ExecutionState.DISPATCHING)

    def mark_succeeded(self, dispatch_id: str) -> AttemptJournalRecord:
        """Record a definitive successful response."""
        return self._transition(dispatch_id, ExecutionState.DISPATCHING, ExecutionState.SUCCEEDED)

    def mark_failed(self, dispatch_id: str) -> AttemptJournalRecord:
        """Record a definitive dispatched failure."""
        return self._transition(dispatch_id, ExecutionState.DISPATCHING, ExecutionState.FAILED)

    def mark_timed_out(self, dispatch_id: str) -> AttemptJournalRecord:
        """Record a dispatched operation whose deadline expired."""
        return self._transition(dispatch_id, ExecutionState.DISPATCHING, ExecutionState.TIMED_OUT)

    def mark_indeterminate_after_dispatch(self, dispatch_id: str) -> AttemptJournalRecord:
        """Record dispatched work without a definitive outcome."""
        return self._transition(
            dispatch_id,
            ExecutionState.DISPATCHING,
            ExecutionState.INDETERMINATE_AFTER_DISPATCH,
        )

    def release(self, dispatch_id: str, state: ExecutionState) -> AttemptJournalRecord:
        """Release definitely unused reserved capacity."""
        if state not in _RELEASED_STATES:
            raise DiscoveryExecutionError("release_requires_unused_terminal_state")
        return self._transition(dispatch_id, ExecutionState.RESERVED, state)

    def _transition(
        self,
        dispatch_id: str,
        expected: ExecutionState,
        target: ExecutionState,
    ) -> AttemptJournalRecord:
        record = self._records.get(dispatch_id)
        if record is None:
            raise DiscoveryExecutionError("unknown_dispatch_id")
        if record.state is not expected:
            raise DiscoveryExecutionError("invalid_dispatch_state_transition")
        updated = replace(record, state=target)
        self._records[dispatch_id] = updated
        _ = self.accounting
        return updated


def _validated_journal_accounting(
    journal: AttemptJournal,
    expected_physical_ceiling: int,
) -> DispatchAccounting:
    """Return one exact accounting snapshot bound to the trusted plan ceiling."""
    if (
        type(journal) is not AttemptJournal
        or type(expected_physical_ceiling) is not int
        or expected_physical_ceiling < 0
    ):
        raise DiscoveryExecutionError("journal_accounting_invalid")
    try:
        private_ceiling = journal._physical_ceiling
        records = journal._records
    except Exception:  # noqa: BLE001 - corrupted runtime state fails closed.
        raise DiscoveryExecutionError("journal_accounting_invalid") from None
    if type(private_ceiling) is not int or type(records) is not dict:
        raise DiscoveryExecutionError("journal_accounting_invalid")
    if private_ceiling != expected_physical_ceiling:
        raise DiscoveryExecutionError("journal_ceiling_mismatch")
    try:
        accounting = journal.accounting
    except Exception:  # noqa: BLE001 - corrupted runtime state fails closed.
        raise DiscoveryExecutionError("journal_accounting_invalid") from None
    if type(accounting) is not DispatchAccounting:
        raise DiscoveryExecutionError("journal_accounting_invalid")
    values = (
        accounting.created,
        accounting.debited,
        accounting.released,
        accounting.outstanding,
        accounting.physical_ceiling,
    )
    if any(type(value) is not int or value < 0 for value in values):
        raise DiscoveryExecutionError("journal_accounting_invalid")
    if accounting.physical_ceiling != expected_physical_ceiling:
        raise DiscoveryExecutionError("journal_ceiling_mismatch")
    if (
        accounting.created != accounting.debited + accounting.released + accounting.outstanding
        or accounting.debited + accounting.outstanding > expected_physical_ceiling
    ):
        raise DiscoveryExecutionError("journal_accounting_invalid")
    return accounting


def _journal_lineage_snapshot(
    journal: AttemptJournal,
    expected_physical_ceiling: int,
) -> tuple[
    tuple[tuple[str, AttemptJournalRecord], ...],
    tuple[tuple[str, str, str, OperationKind, ExecutionState], ...],
    DispatchAccounting,
]:
    """Capture exact ordered journal identity and immutable record values."""
    accounting = _validated_journal_accounting(journal, expected_physical_ceiling)
    try:
        items = tuple(journal._records.items())
        record_values = tuple(
            (
                record.dispatch_id,
                record.dispatch_group_id,
                record.route_id,
                record.operation_kind,
                record.state,
            )
            for _key, record in items
        )
    except Exception:  # noqa: BLE001 - corrupted runtime state fails closed.
        raise DiscoveryExecutionError("journal_accounting_invalid") from None
    for (key, record), values in zip(items, record_values):
        dispatch_id, dispatch_group_id, route_id, operation_kind, state = values
        if (
            type(key) is not str
            or not key
            or type(record) is not AttemptJournalRecord
            or type(dispatch_id) is not str
            or not dispatch_id
            or key != dispatch_id
            or type(dispatch_group_id) is not str
            or not dispatch_group_id
            or type(route_id) is not str
            or not route_id
            or type(operation_kind) is not OperationKind
            or type(state) is not ExecutionState
        ):
            raise DiscoveryExecutionError("journal_accounting_invalid")
    return items, record_values, accounting


class _JournalLineageGuard:
    """Executor-owned monotonic view of one exact attempt journal."""

    __slots__ = (
        "__journal",
        "__physical_ceiling",
        "__expected_items",
        "__expected_record_values",
        "__expected_accounting",
    )

    def __init__(self, journal: AttemptJournal, physical_ceiling: int) -> None:
        self.__journal = journal
        self.__physical_ceiling = physical_ceiling
        self.__expected_items: tuple[tuple[str, AttemptJournalRecord], ...] = ()
        self.__expected_record_values: tuple[tuple[str, str, str, OperationKind, ExecutionState], ...] = ()
        self.__expected_accounting = DispatchAccounting(0, 0, 0, 0, physical_ceiling)
        self.__refresh()

    def validate(self) -> DispatchAccounting:
        """Reject any journal transition not synchronously performed here."""
        items, record_values, accounting = _journal_lineage_snapshot(
            self.__journal,
            self.__physical_ceiling,
        )
        if (
            len(items) != len(self.__expected_items)
            or any(
                current_key != expected_key or current_record is not expected_record
                for (current_key, current_record), (expected_key, expected_record) in zip(
                    items,
                    self.__expected_items,
                )
            )
            or record_values != self.__expected_record_values
            or accounting != self.__expected_accounting
        ):
            raise DiscoveryExecutionError("journal_lineage_mismatch")
        return accounting

    def reserve(
        self,
        *,
        dispatch_id: str,
        dispatch_group_id: str,
        route_id: str,
        operation_kind: OperationKind,
    ) -> AttemptJournalRecord:
        """Validate, reserve, and synchronously advance expected lineage."""
        return self.__mutate(
            lambda: self.__journal.reserve(
                dispatch_id=dispatch_id,
                dispatch_group_id=dispatch_group_id,
                route_id=route_id,
                operation_kind=operation_kind,
            )
        )

    def mark_dispatching(self, dispatch_id: str) -> AttemptJournalRecord:
        return self.__mutate(lambda: self.__journal.mark_dispatching(dispatch_id))

    def mark_succeeded(self, dispatch_id: str) -> AttemptJournalRecord:
        return self.__mutate(lambda: self.__journal.mark_succeeded(dispatch_id))

    def mark_failed(self, dispatch_id: str) -> AttemptJournalRecord:
        return self.__mutate(lambda: self.__journal.mark_failed(dispatch_id))

    def mark_timed_out(self, dispatch_id: str) -> AttemptJournalRecord:
        return self.__mutate(lambda: self.__journal.mark_timed_out(dispatch_id))

    def mark_indeterminate_after_dispatch(self, dispatch_id: str) -> AttemptJournalRecord:
        return self.__mutate(lambda: self.__journal.mark_indeterminate_after_dispatch(dispatch_id))

    def release(self, dispatch_id: str, state: ExecutionState) -> AttemptJournalRecord:
        return self.__mutate(lambda: self.__journal.release(dispatch_id, state))

    def final_snapshot(self) -> tuple[tuple[AttemptJournalRecord, ...], DispatchAccounting]:
        """Atomically validate and return the executor-owned final snapshot."""
        accounting = self.validate()
        return tuple(record for _key, record in self.__expected_items), accounting

    def __mutate(self, operation: Callable[[], AttemptJournalRecord]) -> AttemptJournalRecord:
        self.validate()
        record = operation()
        if type(record) is not AttemptJournalRecord:
            raise DiscoveryExecutionError("journal_accounting_invalid")
        self.__refresh()
        return record

    def __refresh(self) -> None:
        items, record_values, accounting = _journal_lineage_snapshot(
            self.__journal,
            self.__physical_ceiling,
        )
        self.__expected_items = items
        self.__expected_record_values = record_values
        self.__expected_accounting = accounting


def _freeze_candidate_value(value: object) -> object:
    """Copy JSON-like candidate data into recursively immutable values."""
    value_type = type(value)
    if value is None or value_type in {str, int, bool}:
        return value
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError("candidate_record_invalid")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("candidate_record_invalid")
            frozen[key] = _freeze_candidate_value(item)
        return MappingProxyType(frozen)
    if value_type in {list, tuple}:
        return tuple(_freeze_candidate_value(item) for item in value)
    raise ValueError("candidate_record_invalid")


def _freeze_candidate_record(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Fail closed without exposing exceptions from caller-owned values."""
    try:
        frozen = _freeze_candidate_value(record)
    except Exception:  # noqa: BLE001 - candidate objects are untrusted data.
        raise ValueError("candidate_record_invalid") from None
    if not isinstance(frozen, Mapping):
        raise ValueError("candidate_record_invalid")
    return frozen


@dataclass(frozen=True, slots=True)
class DiscoveryCandidate:
    """One adapter-produced candidate before attribution."""

    candidate_id: str
    record: Mapping[str, Any]

    def __post_init__(self) -> None:
        if type(self.candidate_id) is not str or not self.candidate_id:
            raise ValueError("candidate_id_must_be_nonempty")
        if not isinstance(self.record, Mapping):
            raise TypeError("candidate_record_must_be_mapping")
        object.__setattr__(self, "record", _freeze_candidate_record(self.record))


@dataclass(frozen=True, slots=True)
class DiscoveryAdapterResult:
    """Immutable candidates returned by one adapter invocation."""

    candidates: tuple[DiscoveryCandidate, ...]

    def __post_init__(self) -> None:
        if type(self.candidates) is not tuple or any(type(item) is not DiscoveryCandidate for item in self.candidates):
            raise TypeError("adapter_candidates_must_be_typed_tuple")


def _snapshot_adapter_result(value: object) -> DiscoveryAdapterResult | None:
    """Reconstruct adapter output without retaining adapter-owned objects."""
    if type(value) is not DiscoveryAdapterResult:
        return None
    try:
        if type(value.candidates) is not tuple:
            return None
        candidates = tuple(
            DiscoveryCandidate(candidate.candidate_id, candidate.record)
            for candidate in value.candidates
            if type(candidate) is DiscoveryCandidate
        )
        if len(candidates) != len(value.candidates):
            return None
        return DiscoveryAdapterResult(candidates)
    except Exception:  # noqa: BLE001 - malformed adapter values fail closed.
        return None


@dataclass(frozen=True, slots=True)
class AttributedDiscoveryCandidate:
    """One committed candidate with executor-owned attribution."""

    candidate_id: str
    record: Mapping[str, Any]
    catalog_source_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LogicalAttemptOutcome:
    """Deterministic result for one planned logical attempt."""

    logical_attempt_id: str
    catalog_source_id: str
    state: ExecutionState
    code: str | None = None
    retry_after: str | None = None

    def __post_init__(self) -> None:
        if self.retry_after is None:
            return
        if (
            type(self.retry_after) is not str
            or not _valid_retry_after(self.retry_after)
            or self.code != "provider_rate_limited"
            or self.state is not ExecutionState.FAILED
        ):
            raise ValueError("retry_after_outcome_invalid")


@dataclass(frozen=True, slots=True)
class DiscoveryExecutionResult:
    """Committed candidates and logical outcomes from one execution."""

    candidates: tuple[AttributedDiscoveryCandidate, ...]
    logical_outcomes: tuple[LogicalAttemptOutcome, ...]
    skipped: tuple[SkippedTarget, ...]
    truncated_candidates: int
    usage: DiscoveryExecutionUsage


class BoundDispatch(Protocol):
    """Accounted physical dispatch API exposed to trusted adapters."""

    async def __call__(
        self,
        intent: DispatchIntent,
        *,
        cursor: NumericCursor | None = None,
        bindings: tuple[NumericCSVBindingValues, ...] = (),
    ) -> DiscoveryGatewayResponse:
        """Dispatch one exact planned intent."""


DiscoveryAdapter = Callable[[PlannedDispatchGroup, BoundDispatch], Awaitable[DiscoveryAdapterResult]]


DiscoveryGateway = Callable[..., Awaitable[DiscoveryGatewayResponse]]


def _snapshot(value: object) -> object:
    """Convert a planned group into strict immutable primitives."""
    value_type = type(value)
    if value is None or value_type in {str, int, bool}:
        return (value_type, value)
    if value_type is tuple:
        return (tuple, tuple(_snapshot(item) for item in value))
    if isinstance(value, Enum):
        return (value_type, value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return (value_type, tuple((field.name, _snapshot(getattr(value, field.name))) for field in fields(value)))
    raise DiscoveryExecutionError("untrusted_planned_value")


def _same_snapshot(value: object, expected: object) -> bool:
    try:
        return _snapshot(value) == expected
    except Exception:  # noqa: BLE001 - hostile adapter mutations fail closed.
        return False


_PLAN_ENUM_TYPES = frozenset(
    {
        ExecutionMode,
        OperationKind,
        PredicateOperator,
        SkippedCode,
        SkippedStatus,
    }
)
_PLAN_DATACLASS_TYPES = frozenset(
    {
        QueryPair,
        JSONBodyPair,
        DeferredNumericCSVQueryBinding,
        RouteLimits,
        SourcePredicate,
        DispatchIntent,
        DispatchAllowance,
        PlannedLogicalAttempt,
        PlannedDispatchGroup,
        SkippedTarget,
        BudgetCeilings,
        PlannedBudgetAllowance,
        DiscoveryPlan,
    }
)


def _reconstruct_plan_value(value: object) -> object:
    """Recursively reconstruct one exact allowlisted plan contract value."""
    value_type = type(value)
    if value is None or value_type in {str, int, bool}:
        return value
    if value_type is tuple:
        return tuple(_reconstruct_plan_value(item) for item in value)
    if value_type in _PLAN_ENUM_TYPES:
        enum_value = cast(Enum, value).value
        if type(enum_value) not in {str, int, bool}:
            raise TypeError("invalid_plan_enum_value")
        return value_type(enum_value)
    if value_type not in _PLAN_DATACLASS_TYPES:
        raise TypeError("unsupported_plan_contract")
    if value_type is DiscoveryPlan:
        raise TypeError("discovery_plan_requires_explicit_rebuild")
    init_values = {
        field.name: _reconstruct_plan_value(getattr(value, field.name)) for field in fields(value) if field.init
    }
    return value_type(**init_values)


def _covering_ceilings(
    ceilings: BudgetCeilings,
    allowance: PlannedBudgetAllowance,
) -> BudgetCeilings:
    """Build validation ceilings while preserving stricter live runtime caps."""
    return BudgetCeilings(
        max(ceilings.max_route_attempts, allowance.route_attempts),
        max(ceilings.max_physical_dispatches, allowance.physical_dispatches),
        max(ceilings.max_pages_per_route, allowance.max_pages_per_route),
        max(ceilings.max_redirects, allowance.redirects),
        max(ceilings.max_retries, allowance.retries),
        max(ceilings.max_wall_time_ms, allowance.aggregate_wall_time_ms),
        max(ceilings.max_results, allowance.returned_results),
    )


def _rebuild_trusted_plan(plan: DiscoveryPlan) -> DiscoveryPlan:
    """Copy and reconstruct caller-owned plan state before any adapter runs."""
    try:
        caller_snapshot = _snapshot(plan)
        copied = copy.deepcopy(plan)
    except Exception:  # noqa: BLE001 - caller-owned plan copying fails closed.
        raise DiscoveryExecutionError("plan_snapshot_failed") from None
    if not _same_snapshot(copied, caller_snapshot):
        raise DiscoveryExecutionError("plan_snapshot_failed")

    try:
        if type(copied) is not DiscoveryPlan:
            raise TypeError("invalid_plan_type")
        init_values = {
            field.name: _reconstruct_plan_value(getattr(copied, field.name)) for field in fields(copied) if field.init
        }
        caller_plan_digest = init_values.get("plan_digest")
        if type(caller_plan_digest) is not str or not caller_plan_digest:
            raise ValueError("missing_plan_digest")
        live_ceilings = init_values["ceilings"]
        copied_allowance = _reconstruct_plan_value(copied.allowance)
        if type(live_ceilings) is not BudgetCeilings or type(copied_allowance) is not PlannedBudgetAllowance:
            raise TypeError("invalid_plan_budget_contract")
        dispatch_groups = init_values["dispatch_groups"]
        result_limit = init_values["result_limit"]
        derived_allowance = derive_plan_allowance(dispatch_groups, result_limit)
        if _snapshot(copied_allowance) != _snapshot(derived_allowance):
            raise ValueError("plan_allowance_mismatch")
        try:
            trusted = DiscoveryPlan(**init_values)
        except ValueError as error:
            code = error.args[0] if len(error.args) == 1 else None
            if type(code) is not str or not code.startswith("budget_exceeded:"):
                raise
            init_values["ceilings"] = _covering_ceilings(live_ceilings, derived_allowance)
            trusted = DiscoveryPlan(**init_values)
            object.__setattr__(trusted, "ceilings", live_ceilings)
        _validate_plan_identifiers(trusted)
    except Exception:  # noqa: BLE001 - invalid plan internals fail closed.
        raise DiscoveryExecutionError("plan_validation_failed") from None
    return trusted


def _validate_plan_identifiers(plan: DiscoveryPlan) -> None:
    """Require every caller-supplied deterministic ID to match its payload."""
    for group in plan.dispatch_groups:
        group_id = expected_dispatch_group_id(group)
        if group.dispatch_group_id != group_id:
            raise ValueError("dispatch_group_id_mismatch")
        if any(
            attempt.logical_attempt_id != expected_logical_attempt_id(attempt, group_id)
            for attempt in group.logical_attempts
        ):
            raise ValueError("logical_attempt_id_mismatch")


def _canonical_source_link(
    registry: DiscoveryRegistry,
    source_id: object,
    route_id: object,
) -> tuple[SourceRouteReference, tuple[int, str, int]] | None:
    """Return one canonical source-route reference and its declared order."""
    if type(source_id) is not str or type(route_id) is not str:
        return None
    try:
        source = registry.get_source(source_id)
        if (
            type(source.catalog_source_id) is not str
            or source.catalog_source_id != source_id
            or type(source.priority) is not int
            or type(source.route_references) is not tuple
        ):
            return None
        references = tuple(
            (index, reference)
            for index, reference in enumerate(source.route_references)
            if type(reference) is SourceRouteReference
            and type(reference.route_id) is str
            and reference.route_id == route_id
        )
    except Exception:  # noqa: BLE001 - mutated registry source data fails closed.
        return None
    if len(references) != 1:
        return None
    route_index, reference = references[0]
    return reference, (source.priority, source.catalog_source_id, route_index)


def _validate_plan_catalog_references(plan: DiscoveryPlan, registry: DiscoveryRegistry) -> None:
    """Bind all logical and skipped targets to current canonical catalog links."""
    try:
        group_order: list[tuple[int, str, int]] = []
        seen_source_routes: set[tuple[str, str]] = set()
        for group in plan.dispatch_groups:
            if type(group.logical_attempts) is not tuple:
                raise ValueError("invalid_logical_attempts")
            logical_order: list[tuple[int, str]] = []
            group_candidates: list[tuple[int, str, int]] = []
            for attempt in group.logical_attempts:
                link = _canonical_source_link(
                    registry,
                    attempt.catalog_source_id,
                    group.route_id,
                )
                if link is None:
                    raise ValueError("logical_attempt_catalog_mismatch")
                reference, order_key = link
                if reference is None or _snapshot(reference.source_predicate) != _snapshot(attempt.source_predicate):
                    raise ValueError("logical_attempt_catalog_mismatch")
                source_route = (attempt.catalog_source_id, group.route_id)
                if source_route in seen_source_routes:
                    raise ValueError("duplicate_source_route_target")
                seen_source_routes.add(source_route)
                logical_order.append(order_key[:2])
                group_candidates.append(order_key)
            if tuple(logical_order) != tuple(sorted(logical_order)):
                raise ValueError("logical_attempt_order_mismatch")
            group_order.append(min(group_candidates))
        if tuple(group_order) != tuple(sorted(group_order)):
            raise ValueError("dispatch_group_order_mismatch")
        skipped_order: list[tuple[int, str, int]] = []
        for skipped in plan.skipped:
            if type(skipped) is not SkippedTarget:
                raise ValueError("invalid_skipped_target")
            link = _canonical_source_link(
                registry,
                skipped.requested_source_id,
                skipped.route_id,
            )
            if link is None:
                raise ValueError("skipped_target_catalog_mismatch")
            route = registry.get_route(skipped.route_id)
            if route.route_id != skipped.route_id:
                raise ValueError("skipped_target_route_mismatch")
            if route.credential_requirement is CredentialRequirement.NONE:
                if skipped.status is not SkippedStatus.SKIPPED or skipped.code is not SkippedCode.ROUTE_NOT_READY:
                    raise ValueError("credentialless_skipped_target_semantics_mismatch")
            elif route.credential_requirement is CredentialRequirement.API_KEY:
                if (
                    skipped.status is not SkippedStatus.UNAVAILABLE
                    or skipped.code is not SkippedCode.CREDENTIALED_OUT_OF_SCOPE
                    or skipped.reason != CREDENTIALED_ROUTE_SKIP_REASON
                ):
                    raise ValueError("credentialed_skipped_target_semantics_mismatch")
            else:
                raise ValueError("invalid_route_credential_requirement")
            source_route = (skipped.requested_source_id, skipped.route_id)
            if source_route in seen_source_routes:
                raise ValueError("duplicate_source_route_target")
            seen_source_routes.add(source_route)
            skipped_order.append(link[1])
        if tuple(skipped_order) != tuple(sorted(skipped_order)):
            raise ValueError("skipped_target_order_mismatch")
    except Exception:  # noqa: BLE001 - invalid catalog links fail closed.
        raise DiscoveryExecutionError("plan_validation_failed") from None


def _new_dispatch_id() -> str:
    return str(uuid.uuid4())


def _policy_active(check: PolicyActivityCheck, route_id: str, digest: str) -> bool:
    try:
        return check(route_id, digest) is True
    except Exception:  # noqa: BLE001 - policy checks fail closed.
        return False


def _gateway_error_metadata(error: DiscoveryGatewayError) -> tuple[str, bool, bool] | None:
    """Read only exact, public gateway error scalars."""
    try:
        code = error.code
        retryable = error.retryable
        timed_out = error.timed_out
    except Exception:  # noqa: BLE001 - hostile subclasses fail closed.
        return None
    if (
        type(code) is not str
        or code not in _GATEWAY_ERROR_CODES
        or type(retryable) is not bool
        or type(timed_out) is not bool
    ):
        return None
    return code, retryable, timed_out


def _group_matches_route(group: PlannedDispatchGroup, route: AccessRoute) -> bool:
    try:
        limits = route.policy.limits
        allowance = group.allowance
        return (
            group.route_id == route.route_id
            and group.backend_id == route.backend_id
            and group.adapter_id == route.adapter_id
            and group.adapter_version == route.adapter_version
            and group.fallback_order == route.fallback_order
            and group.policy_digest == route.policy.policy_digest == canonical_policy_digest(route.policy)
            and group.limits == limits
            and allowance.physical_dispatches == route.max_physical_dispatches
            and allowance.pages == limits.max_pages
            and allowance.redirects == limits.max_redirects
            and allowance.retries == limits.max_retries
        )
    except Exception:  # noqa: BLE001 - mutated registry objects fail closed.
        return False


def _response_matches(response: object, route: AccessRoute, intent: DispatchIntent) -> bool:
    if type(response) is not DiscoveryGatewayResponse or type(response.trace) is not DiscoveryGatewayTrace:
        return False
    trace = response.trace
    return (
        type(trace.route_id) is str
        and trace.route_id == route.route_id
        and type(trace.policy_digest) is str
        and trace.policy_digest == intent.policy_digest
        and type(trace.method) is str
        and trace.method == intent.method
        and type(trace.path) is str
        and trace.path == intent.path
        and type(trace.query_keys) is tuple
        and all(type(key) is str for key in trace.query_keys)
        and trace.query_keys == tuple(pair.name for pair in intent.query_pairs)
        and type(trace.status_code) is int
        and type(response.status_code) is int
        and trace.status_code == response.status_code
    )


class _GroupExecutionController:
    """Executor-owned state for one trusted dispatch group."""

    def __init__(
        self,
        group: PlannedDispatchGroup,
        *,
        registry: DiscoveryRegistry,
        journal_guard: _JournalLineageGuard,
        execution_control: _ExecutionControl,
        gateway: DiscoveryGateway,
        policy_is_active: PolicyActivityCheck,
        dispatch_id_factory: DispatchIDFactory,
        max_pages_per_route: int,
        max_redirects: int,
        max_retries: int,
        continuation_usage: _ContinuationUsage,
    ) -> None:
        self._trusted_group = group
        self._group_snapshot = _snapshot(group)
        self._exposed_group = copy.deepcopy(group)
        if not _same_snapshot(self._exposed_group, self._group_snapshot):
            raise DiscoveryExecutionError("bound_group_copy_failed")
        self._intent_bindings = {
            id(exposed): (trusted, _snapshot(trusted), index)
            for index, (trusted, exposed) in enumerate(zip(group.intents, self._exposed_group.intents))
        }
        self._registry = registry
        self._journal_guard = journal_guard
        self._execution_control = execution_control
        self._gateway = gateway
        self._policy_is_active = policy_is_active
        self._dispatch_id_factory = dispatch_id_factory
        self._max_pages_per_route = max_pages_per_route
        self._max_redirects = max_redirects
        self._max_retries = max_retries
        self._continuation_usage = continuation_usage
        self._used: set[int] = set()
        self._completed_searches: set[int] = set()
        self._successful_searches: set[int] = set()
        self._seen_cursors: dict[int, set[int]] = {}
        self.physical_dispatches = 0
        self.pages = 0
        self.redirects = 0
        self.retries = 0
        self.closed = False
        self.failure_code: str | None = None
        self._owner_task: asyncio.Task[Any] | None = None

    @property
    def exposed_group(self) -> PlannedDispatchGroup:
        return self._exposed_group

    @property
    def intact(self) -> bool:
        return _same_snapshot(self._trusted_group, self._group_snapshot) and _same_snapshot(
            self._exposed_group,
            self._group_snapshot,
        )

    @property
    def has_successful_search(self) -> bool:
        return bool(self._successful_searches)

    @property
    def has_completed_search(self) -> bool:
        return bool(self._completed_searches)

    def close(self) -> None:
        self.closed = True

    def bind_owner_task(self) -> None:
        """Bind dispatch use to the task directly awaiting the adapter."""
        owner_task = asyncio.current_task()
        if owner_task is None:
            self._reject("adapter_task_unavailable")
        self._owner_task = owner_task

    def _reject(self, code: str) -> None:
        if not self.closed and self.failure_code is None:
            self.failure_code = code
        raise DiscoveryExecutionError(code)

    def validate_journal_lineage(self) -> DispatchAccounting:
        """Validate the shared execution journal and retain the failure code."""
        try:
            return self._journal_guard.validate()
        except DiscoveryExecutionError as error:
            self._reject(error.code)

    def _execution_checkpoint(self) -> float:
        """Apply the shared execution stop controls at one trusted boundary."""
        try:
            return self._execution_control.checkpoint()
        except DiscoveryExecutionError as error:
            self._reject(error.code)

    def _latch_execution_stop(self, code: str) -> None:
        """Retain an externally observed aggregate stop on this controller."""
        try:
            self._execution_control.latch(code)
        except DiscoveryExecutionError as error:
            self._reject(error.code)

    async def _call_gateway(self, route: AccessRoute, intent: DispatchIntent) -> DiscoveryGatewayResponse:
        """Await the gateway seam inside an executor-owned coroutine task."""
        return await self._gateway(route, intent, is_policy_active=self._policy_is_active)

    async def _cancel_and_drain_gateway(
        self,
        gateway_task: asyncio.Task[DiscoveryGatewayResponse],
        *,
        tolerate_cancellation: bool = False,
    ) -> None:
        """Cancel one child and consume its terminal result without leaking it."""
        while not gateway_task.done():
            gateway_task.cancel()
            try:
                await asyncio.wait({gateway_task})
            except asyncio.CancelledError:
                if not tolerate_cancellation:
                    raise
        try:
            gateway_task.result()
        except BaseException:  # noqa: BLE001 - cancelled child must be consumed.
            pass

    def _integrity_route(self) -> AccessRoute:
        """Return a fresh route after journal and bound-plan validation."""
        self.validate_journal_lineage()
        if not self.intact:
            self._reject("bound_plan_mutated")
        try:
            route = self._registry.get_route(self._trusted_group.route_id)
        except Exception:  # noqa: BLE001 - registry lookup fails closed.
            self._reject("registry_mismatch")
        if not _group_matches_route(self._trusted_group, route):
            self._reject("registry_mismatch")
        return route

    def _validate_after_policy_callback(self) -> tuple[AccessRoute, float]:
        """Revalidate all mutable state after policy and control callbacks."""
        remaining_seconds = self._execution_checkpoint()
        return self._integrity_route(), remaining_seconds

    def _validated_route(self) -> tuple[AccessRoute, float]:
        """Return a current allowed route and remaining aggregate time."""
        route = self._integrity_route()
        active = _policy_active(self._policy_is_active, route.route_id, self._trusted_group.policy_digest)
        route, remaining_seconds = self._validate_after_policy_callback()
        if not active:
            self._reject("dispatch_policy_inactive")
        return route, remaining_seconds

    def _release_unused_reservation(self, dispatch_id: str, state: ExecutionState) -> None:
        """Best-effort release without masking the failure being propagated."""
        try:
            self._journal_guard.release(dispatch_id, state)
        except BaseException:  # noqa: BLE001 - the original failure must win.
            pass

    def _effective_intent(
        self,
        route: AccessRoute,
        trusted_intent: DispatchIntent,
        intent_index: int,
        cursor: NumericCursor | None,
        bindings: tuple[NumericCSVBindingValues, ...],
    ) -> tuple[DispatchIntent, int | None]:
        declared_bindings = trusted_intent.query_bindings
        if cursor is not None and (type(bindings) is not tuple or bindings):
            self._reject("cursor_and_bindings_conflict")
        if declared_bindings:
            if not self._successful_searches:
                self._reject("search_not_ready")
            if type(bindings) is not tuple or not bindings:
                self._reject("binding_values_required")
            if any(type(values) is not NumericCSVBindingValues for values in bindings):
                self._reject("binding_values_mismatch")
            binding_ids = tuple(values.binding_id for values in bindings)
            declared_ids = tuple(declaration.binding_id for declaration in declared_bindings)
            if (
                any(type(binding_id) is not str for binding_id in binding_ids)
                or len(set(binding_ids)) != len(binding_ids)
                or set(binding_ids) != set(declared_ids)
            ):
                self._reject("binding_values_mismatch")
            provided = {values.binding_id: values for values in bindings}
            grounded_pairs = []
            for declaration in declared_bindings:
                values = provided[declaration.binding_id].values
                if (
                    type(values) is not tuple
                    or not values
                    or any(type(value) is not int or value <= 0 for value in values)
                ):
                    self._reject("binding_values_mismatch")
                if len(values) > declaration.max_items or any(
                    len(str(value)) > declaration.max_item_chars for value in values
                ):
                    self._reject("binding_values_limit_exceeded")
                grounded_pairs.append(QueryPair(declaration.query_name, ",".join(str(value) for value in values)))
            trusted_intent = replace(
                trusted_intent,
                query_pairs=trusted_intent.query_pairs + tuple(grounded_pairs),
                query_bindings=(),
            )
        elif type(bindings) is not tuple or bindings:
            self._reject("bindings_not_allowed")
        if cursor is None:
            if intent_index in self._used:
                self._reject("dispatch_intent_already_used")
            self._used.add(intent_index)
            if trusted_intent.operation_kind is not OperationKind.SEARCH:
                return trusted_intent, None
            query_key = route.policy.pagination_query_key
            body_key = route.policy.pagination_json_body_key
            if type(query_key) is str and body_key is None:
                values = tuple(pair.value for pair in trusted_intent.query_pairs if pair.name == query_key)
                valid = (
                    len(values) == 1
                    and type(values[0]) is str
                    and values[0].isascii()
                    and values[0].isdecimal()
                    and len(values[0]) <= len(str(MAX_PAGINATION_CURSOR))
                    and int(values[0]) <= MAX_PAGINATION_CURSOR
                )
            elif query_key is None and type(body_key) is str:
                values = tuple(pair.value for pair in trusted_intent.json_body_pairs if pair.name == body_key)
                valid = len(values) == 1 and type(values[0]) is int and 0 <= values[0] <= MAX_PAGINATION_CURSOR
            else:
                values = ()
                valid = False
            if not valid:
                self._reject("pagination_query_invalid")
            return trusted_intent, int(values[0])
        if (
            type(cursor) is not NumericCursor
            or type(cursor.value) is not int
            or not 0 <= cursor.value <= MAX_PAGINATION_CURSOR
        ):
            self._reject("invalid_pagination_cursor")
        if trusted_intent.operation_kind is not OperationKind.SEARCH:
            self._reject("cursor_not_allowed")
        if intent_index not in self._successful_searches:
            self._reject("search_not_ready")
        if cursor.value in self._seen_cursors.get(intent_index, set()):
            self._reject("pagination_cursor_repeated")
        query_key = route.policy.pagination_query_key
        body_key = route.policy.pagination_json_body_key
        if type(query_key) is str and body_key is None:
            matching = tuple(index for index, pair in enumerate(trusted_intent.query_pairs) if pair.name == query_key)
            if len(matching) != 1:
                self._reject("pagination_query_invalid")
            try:
                query_pairs = tuple(
                    QueryPair(pair.name, str(cursor.value)) if index == matching[0] else pair
                    for index, pair in enumerate(trusted_intent.query_pairs)
                )
                effective_intent = replace(trusted_intent, query_pairs=query_pairs)
            except Exception:  # noqa: BLE001 - cursor reconstruction must latch a typed failure.
                self._reject("invalid_pagination_cursor")
            return effective_intent, cursor.value
        if query_key is None and type(body_key) is str:
            matching = tuple(
                index for index, pair in enumerate(trusted_intent.json_body_pairs) if pair.name == body_key
            )
            if len(matching) != 1:
                self._reject("pagination_query_invalid")
            try:
                json_body_pairs = tuple(
                    JSONBodyPair(pair.name, cursor.value) if index == matching[0] else pair
                    for index, pair in enumerate(trusted_intent.json_body_pairs)
                )
                effective_intent = replace(trusted_intent, json_body_pairs=json_body_pairs)
            except Exception:  # noqa: BLE001 - cursor reconstruction must latch a typed failure.
                self._reject("invalid_pagination_cursor")
            return effective_intent, cursor.value
        self._reject("pagination_query_invalid")

    async def __call__(
        self,
        intent: DispatchIntent,
        *,
        cursor: NumericCursor | None = None,
        bindings: tuple[NumericCSVBindingValues, ...] = (),
    ) -> DiscoveryGatewayResponse:
        if self._owner_task is None or asyncio.current_task() is not self._owner_task:
            self._reject("dispatch_task_mismatch")
        if self.closed:
            self._reject("dispatch_capability_closed")
        if self.failure_code is not None:
            raise DiscoveryExecutionError(self.failure_code)
        self._execution_checkpoint()
        binding = self._intent_bindings.get(id(intent))
        if binding is None:
            self._reject("dispatch_intent_not_bound_to_group")
        trusted_intent, intent_snapshot, intent_index = binding
        if not _same_snapshot(intent, intent_snapshot):
            self._reject("bound_plan_mutated")
        route, _remaining_seconds = self._validated_route()
        effective_intent, page_cursor = self._effective_intent(
            route,
            trusted_intent,
            intent_index,
            cursor,
            bindings,
        )
        is_page = effective_intent.operation_kind is OperationKind.SEARCH
        if is_page and (self.pages >= self._trusted_group.allowance.pages or self.pages >= self._max_pages_per_route):
            self._reject("page_ceiling_exhausted")
        first_hop = True
        pending_continuation: str | None = None
        while True:
            self._execution_checkpoint()
            if pending_continuation is not None:
                route, _remaining_seconds = self._validated_route()
            if self.physical_dispatches >= self._trusted_group.allowance.physical_dispatches:
                self._reject("group_physical_dispatch_ceiling_exhausted")
            accounting = self.validate_journal_lineage()
            if accounting.debited + accounting.outstanding >= accounting.physical_ceiling:
                self._reject("physical_dispatch_ceiling_exhausted")
            try:
                dispatch_id = self._dispatch_id_factory()
            except Exception:  # noqa: BLE001 - ID factories cannot expose details.
                self._execution_checkpoint()
                self._reject("dispatch_id_factory_failed")
            self._execution_checkpoint()
            if type(dispatch_id) is not str or not dispatch_id:
                self._reject("dispatch_id_factory_failed")
            try:
                self._journal_guard.reserve(
                    dispatch_id=dispatch_id,
                    dispatch_group_id=self._trusted_group.dispatch_group_id,
                    route_id=self._trusted_group.route_id,
                    operation_kind=trusted_intent.operation_kind,
                )
            except DiscoveryExecutionError as error:
                self._reject(error.code)
            try:
                route, remaining_seconds = self._validated_route()
            except DiscoveryExecutionError as error:
                release_state = (
                    ExecutionState.CANCELLED
                    if error.code in {"execution_cancelled", "cancellation_check_failed"}
                    else ExecutionState.SKIPPED
                )
                self._release_unused_reservation(dispatch_id, release_state)
                raise
            except asyncio.CancelledError:
                self._release_unused_reservation(dispatch_id, ExecutionState.CANCELLED)
                raise
            except BaseException:
                self._release_unused_reservation(dispatch_id, ExecutionState.SKIPPED)
                raise
            self._journal_guard.mark_dispatching(dispatch_id)
            self.physical_dispatches += 1
            if pending_continuation == "retry":
                self.retries += 1
                self._continuation_usage.retries += 1
            elif pending_continuation == "redirect":
                self.redirects += 1
                self._continuation_usage.redirects += 1
            pending_continuation = None
            if first_hop and is_page:
                self.pages += 1
                self._seen_cursors.setdefault(intent_index, set()).add(page_cursor)  # type: ignore[arg-type]
            first_hop = False
            aggregate_timed_out = False
            try:
                gateway_task = asyncio.create_task(self._call_gateway(route, effective_intent))
                done, _pending = await asyncio.wait(
                    {gateway_task},
                    timeout=remaining_seconds,
                )
                if not done:
                    await self._cancel_and_drain_gateway(gateway_task)
                    self.validate_journal_lineage()
                    self._journal_guard.mark_timed_out(dispatch_id)
                    aggregate_timed_out = True
                else:
                    response = gateway_task.result()
            except asyncio.CancelledError:
                await self._cancel_and_drain_gateway(
                    gateway_task,
                    tolerate_cancellation=True,
                )
                try:
                    self._journal_guard.mark_indeterminate_after_dispatch(dispatch_id)
                except DiscoveryExecutionError:
                    pass
                raise
            except DiscoveryGatewayError as error:
                self.validate_journal_lineage()
                metadata = _gateway_error_metadata(error)
                if metadata is None:
                    self._journal_guard.mark_failed(dispatch_id)
                    self._execution_checkpoint()
                    self._reject("gateway_error_invalid")
                error_code, retryable, timed_out = metadata
                if timed_out:
                    self._journal_guard.mark_timed_out(dispatch_id)
                else:
                    self._journal_guard.mark_failed(dispatch_id)
                self._execution_checkpoint()
                if retryable:
                    if effective_intent.method not in {"GET", "HEAD"}:
                        self._reject("gateway_retry_not_allowed")
                    if (
                        self.retries >= self._trusted_group.allowance.retries
                        or self._continuation_usage.retries >= self._max_retries
                    ):
                        self._reject("gateway_retry_exhausted")
                    pending_continuation = "retry"
                    continue
                self._reject("gateway_timed_out" if timed_out else f"gateway_{error_code}")
            except (TimeoutError, asyncio.TimeoutError):
                self.validate_journal_lineage()
                self._journal_guard.mark_timed_out(dispatch_id)
                self._execution_checkpoint()
                self._reject("gateway_timed_out")
            except Exception:  # noqa: BLE001 - transport/provider details stay private.
                self.validate_journal_lineage()
                self._journal_guard.mark_failed(dispatch_id)
                self._execution_checkpoint()
                self._reject("gateway_failed")
            if aggregate_timed_out:
                self._execution_checkpoint()
                self._latch_execution_stop("aggregate_deadline_exceeded")
            self.validate_journal_lineage()
            if not _response_matches(response, route, effective_intent):
                self._journal_guard.mark_failed(dispatch_id)
                self._execution_checkpoint()
                self._reject("gateway_response_mismatch")
            self._journal_guard.mark_succeeded(dispatch_id)
            self._execution_checkpoint()
            if 300 <= response.status_code < 400:
                location = response.redirect_location
                redirected = (
                    reconstruct_redirect_intent(route, effective_intent, location) if type(location) is str else None
                )
                if redirected is None:
                    self._reject("gateway_redirect_invalid")
                if (
                    self.redirects >= self._trusted_group.allowance.redirects
                    or self._continuation_usage.redirects >= self._max_redirects
                ):
                    self._reject("gateway_redirect_exhausted")
                pending_continuation = "redirect"
                effective_intent = redirected
                continue
            break
        if is_page:
            self._completed_searches.add(intent_index)
            if 200 <= response.status_code < 300:
                self._successful_searches.add(intent_index)
        return response


def _adapter_dispatch(controller: _GroupExecutionController) -> BoundDispatch:
    async def dispatch(
        intent: DispatchIntent,
        *,
        cursor: NumericCursor | None = None,
        bindings: tuple[NumericCSVBindingValues, ...] = (),
    ) -> DiscoveryGatewayResponse:
        return await controller(intent, cursor=cursor, bindings=bindings)

    return dispatch


def _outcomes(
    group: PlannedDispatchGroup,
    state: ExecutionState,
    code: str | None = None,
    retry_after: str | None = None,
) -> list[LogicalAttemptOutcome]:
    return [
        LogicalAttemptOutcome(
            attempt.logical_attempt_id,
            attempt.catalog_source_id,
            state,
            code,
            retry_after,
        )
        for attempt in group.logical_attempts
    ]


def _failure_state(code: str) -> ExecutionState:
    if code in {
        "gateway_timed_out",
        "aggregate_deadline_exceeded",
        "provider_parse_deadline_exceeded",
    }:
        return ExecutionState.TIMED_OUT
    if code in {"execution_cancelled", "cancellation_check_failed"}:
        return ExecutionState.CANCELLED
    if code == "indeterminate_after_dispatch":
        return ExecutionState.INDETERMINATE_AFTER_DISPATCH
    return ExecutionState.FAILED


async def execute_discovery_plan(
    plan: DiscoveryPlan,
    *,
    registry: DiscoveryRegistry,
    adapters: Mapping[str, DiscoveryAdapter],
    gateway: DiscoveryGateway,
    policy_is_active: PolicyActivityCheck,
    dispatch_id_factory: DispatchIDFactory = _new_dispatch_id,
    journal: AttemptJournal | None = None,
    monotonic_clock: MonotonicClock = time.monotonic,
    cancellation_check: CancellationCheck | None = None,
) -> DiscoveryExecutionResult:
    """Execute frozen groups sequentially through closed capabilities."""
    if not isinstance(plan, DiscoveryPlan) or not isinstance(registry, DiscoveryRegistry):
        raise DiscoveryExecutionError("typed_plan_and_registry_required")
    trusted_plan = _rebuild_trusted_plan(plan)
    if (
        trusted_plan.catalog_version != registry.catalog_version
        or trusted_plan.registry_version != registry.registry_version
    ):
        raise DiscoveryExecutionError("plan_registry_version_mismatch")
    _validate_plan_catalog_references(trusted_plan, registry)
    expected_physical_ceiling = trusted_plan.ceilings.max_physical_dispatches
    if journal is None:
        active_journal = AttemptJournal(physical_ceiling=expected_physical_ceiling)
    else:
        if type(journal) is not AttemptJournal:
            raise DiscoveryExecutionError("invalid_injected_journal")
        try:
            injected_ceiling = journal.physical_ceiling
        except Exception:  # noqa: BLE001 - corrupted injected state fails closed.
            raise DiscoveryExecutionError("journal_ceiling_mismatch") from None
        if type(injected_ceiling) is not int or injected_ceiling != expected_physical_ceiling:
            raise DiscoveryExecutionError("journal_ceiling_mismatch")
        if not journal.is_pristine(physical_ceiling=expected_physical_ceiling):
            raise DiscoveryExecutionError("journal_not_pristine")
        active_journal = journal
    journal_guard = _JournalLineageGuard(active_journal, expected_physical_ceiling)
    execution_control = _ExecutionControl(
        max_wall_time_ms=trusted_plan.ceilings.max_wall_time_ms,
        monotonic_clock=monotonic_clock,
        cancellation_check=cancellation_check,
    )
    returned_result_cap = min(
        trusted_plan.allowance.returned_results,
        trusted_plan.ceilings.max_results,
    )

    committed: list[AttributedDiscoveryCandidate] = []
    outcomes: list[LogicalAttemptOutcome] = []
    truncated = 0
    route_attempts = 0
    pages = 0
    continuation_usage = _ContinuationUsage()
    for trusted_group in trusted_plan.dispatch_groups:
        stop_code = execution_control.stop_code
        if stop_code is not None:
            outcomes.extend(_outcomes(trusted_group, _failure_state(stop_code), stop_code))
            continue
        if trusted_group.fallback_order > 0:
            outcomes.extend(_outcomes(trusted_group, ExecutionState.SKIPPED, "fallback_not_executed"))
            continue
        try:
            route = registry.get_route(trusted_group.route_id)
        except Exception:  # noqa: BLE001 - one invalid group cannot abort later groups.
            outcomes.extend(_outcomes(trusted_group, ExecutionState.FAILED, "registry_mismatch"))
            continue
        if not _group_matches_route(trusted_group, route):
            outcomes.extend(_outcomes(trusted_group, ExecutionState.FAILED, "registry_mismatch"))
            continue
        try:
            adapter = adapters[trusted_group.adapter_id]
        except Exception:  # noqa: BLE001 - malformed adapter maps fail closed per group.
            outcomes.extend(_outcomes(trusted_group, ExecutionState.FAILED, "missing_adapter"))
            continue
        if not callable(adapter):
            outcomes.extend(_outcomes(trusted_group, ExecutionState.FAILED, "missing_adapter"))
            continue

        group_route_attempts = len(trusted_group.logical_attempts)
        if route_attempts + group_route_attempts > trusted_plan.ceilings.max_route_attempts:
            outcomes.extend(_outcomes(trusted_group, ExecutionState.FAILED, "route_attempt_ceiling_exhausted"))
            continue

        try:
            controller = _GroupExecutionController(
                trusted_group,
                registry=registry,
                journal_guard=journal_guard,
                execution_control=execution_control,
                gateway=gateway,
                policy_is_active=policy_is_active,
                dispatch_id_factory=dispatch_id_factory,
                max_pages_per_route=trusted_plan.ceilings.max_pages_per_route,
                max_redirects=trusted_plan.ceilings.max_redirects,
                max_retries=trusted_plan.ceilings.max_retries,
                continuation_usage=continuation_usage,
            )
        except DiscoveryExecutionError as error:
            outcomes.extend(_outcomes(trusted_group, ExecutionState.FAILED, error.code))
            continue
        adapter_result: object = None
        adapter_error: str | None = None
        adapter_retry_after: str | None = None
        route_attempts += group_route_attempts
        try:
            controller.bind_owner_task()
            try:
                raw_adapter_result = await adapter(controller.exposed_group, _adapter_dispatch(controller))
            except asyncio.CancelledError:
                try:
                    journal_guard.validate()
                except DiscoveryExecutionError:
                    pass
                raise
            except Exception:
                controller.validate_journal_lineage()
                controller._execution_checkpoint()
                raise
            except BaseException:
                try:
                    journal_guard.validate()
                except DiscoveryExecutionError:
                    pass
                raise
            else:
                controller.validate_journal_lineage()
                controller._execution_checkpoint()
                adapter_result = _snapshot_adapter_result(raw_adapter_result)
        except DiscoveryAdapterError as error:
            trusted_error = _trusted_adapter_error(error)
            if trusted_error is None or controller.failure_code is not None or not controller.has_completed_search:
                adapter_error = controller.failure_code or "adapter_failed"
            else:
                adapter_error, adapter_retry_after = trusted_error
        except DiscoveryExecutionError:
            adapter_error = controller.failure_code or "adapter_failed"
        except Exception:  # noqa: BLE001 - adapter/provider details stay private.
            adapter_error = controller.failure_code or "adapter_failed"
        finally:
            controller.close()
            pages += controller.pages
        journal_guard.validate()

        failure_code = controller.failure_code
        if failure_code is None and not controller.intact:
            failure_code = "bound_plan_mutated"
        if failure_code is None:
            try:
                current_route = registry.get_route(trusted_group.route_id)
            except Exception:  # noqa: BLE001 - post-adapter registry checks fail closed.
                current_route = None
            if current_route is None or not _group_matches_route(trusted_group, current_route):
                failure_code = "registry_mismatch"
        if failure_code is None and (adapter_error is None or controller.has_completed_search):
            policy_active = _policy_active(
                policy_is_active,
                trusted_group.route_id,
                trusted_group.policy_digest,
            )
            try:
                controller._validate_after_policy_callback()
            except DiscoveryExecutionError as error:
                failure_code = error.code
            else:
                if not policy_active:
                    failure_code = "dispatch_policy_inactive"
        if failure_code is None:
            failure_code = adapter_error
        if failure_code is None and type(adapter_result) is not DiscoveryAdapterResult:
            failure_code = "malformed_adapter_result"
        if (
            failure_code is None
            and isinstance(adapter_result, DiscoveryAdapterResult)
            and not controller.has_successful_search
        ):
            failure_code = "missing_search_dispatch"
        if failure_code is not None:
            retry_after = (
                adapter_retry_after if controller.failure_code is None and failure_code == adapter_error else None
            )
            outcomes.extend(
                _outcomes(
                    trusted_group,
                    _failure_state(failure_code),
                    failure_code,
                    retry_after,
                )
            )
            continue

        adapter_result = cast(DiscoveryAdapterResult, adapter_result)
        matched_sources: set[str] = set()
        route_candidates = adapter_result.candidates[: trusted_group.limits.max_results]
        group_candidates: list[AttributedDiscoveryCandidate] = []
        group_truncated = max(0, len(adapter_result.candidates) - len(route_candidates))
        attribution_failed = False
        for candidate in route_candidates:
            try:
                source_ids = tuple(
                    attempt.catalog_source_id
                    for attempt in trusted_group.logical_attempts
                    if attempt.source_predicate is None
                    or evaluate_source_predicate(attempt.source_predicate, candidate.record) is AttributionMatch.MATCH
                )
            except Exception:  # noqa: BLE001 - malformed candidate data fails the group closed.
                attribution_failed = True
                break
            if not source_ids:
                continue
            matched_sources.update(source_ids)
            if len(committed) + len(group_candidates) >= returned_result_cap:
                group_truncated += 1
                continue
            group_candidates.append(AttributedDiscoveryCandidate(candidate.candidate_id, candidate.record, source_ids))
        if attribution_failed:
            outcomes.extend(_outcomes(trusted_group, ExecutionState.FAILED, "candidate_attribution_failed"))
            continue
        policy_active = _policy_active(
            policy_is_active,
            trusted_group.route_id,
            trusted_group.policy_digest,
        )
        try:
            controller._validate_after_policy_callback()
        except DiscoveryExecutionError as error:
            outcomes.extend(_outcomes(trusted_group, _failure_state(error.code), error.code))
            continue
        if not policy_active:
            outcomes.extend(_outcomes(trusted_group, ExecutionState.FAILED, "dispatch_policy_inactive"))
            continue
        committed.extend(group_candidates)
        truncated += group_truncated
        outcomes.extend(
            LogicalAttemptOutcome(
                attempt.logical_attempt_id,
                attempt.catalog_source_id,
                (
                    ExecutionState.SUCCEEDED
                    if attempt.catalog_source_id in matched_sources
                    else ExecutionState.VALID_EMPTY
                ),
            )
            for attempt in trusted_group.logical_attempts
        )

    physical_records, final_accounting = journal_guard.final_snapshot()
    usage = DiscoveryExecutionUsage(
        physical_records=physical_records,
        accounting=final_accounting,
        route_attempts=route_attempts,
        pages=pages,
        redirects=continuation_usage.redirects,
        retries=continuation_usage.retries,
        possible_duplicate_work=continuation_usage.retries > 0,
    )
    return DiscoveryExecutionResult(
        tuple(committed),
        tuple(outcomes),
        trusted_plan.skipped,
        truncated,
        usage,
    )
