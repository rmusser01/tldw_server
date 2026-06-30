# circuit_breaker.py
"""
Unified circuit breaker module for the tldw_server project.

Provides a single, shared implementation of the circuit breaker pattern
used across all modules (Embeddings, Evaluations, TTS, RAG, Chat, MCP,
WebSearch). Supports both sync and async callers, counter and rolling-window
failure detection, exponential backoff on recovery timeout, serial half-open
probes, state-change callbacks, and Prometheus-style metrics emission.

Design decisions
----------------
- **Shared-state aware** -- supports optional DB-backed persistence for
  multi-worker/process deployments with optimistic locking.
- **Serial half-open probes** (``half_open_max_calls=1`` by default).
- **Always starts CLOSED.**
- **Original exceptions re-raised as-is** -- the breaker records failures
  internally but never wraps the original error.
- **State-change callbacks** for modules that need them (TTS health
  monitoring, RAG coordinator).
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from enum import Enum, IntEnum
from functools import wraps
from typing import Any, Callable, TypeVar

from loguru import logger

from tldw_Server_API.app.core.DB_Management.Circuit_Breaker_Registry_DB import (
    CircuitBreakerOptimisticLockError as _StoreOptimisticLockError,
)
from tldw_Server_API.app.core.DB_Management.Circuit_Breaker_Registry_DB import (
    CircuitBreakerRegistryDB as _RegistryDB,
)
from tldw_Server_API.app.core.DB_Management.Circuit_Breaker_Registry_DB import (
    CircuitBreakerStoredState as _StoredState,
)
from tldw_Server_API.app.core.testing import is_explicit_pytest_runtime

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Metrics helpers -- graceful no-op if metrics_manager is unavailable
# ---------------------------------------------------------------------------

try:
    from tldw_Server_API.app.core.Metrics.metrics_manager import (
        increment_counter as _increment_counter,
    )
    from tldw_Server_API.app.core.Metrics.metrics_manager import (
        set_gauge as _set_gauge,
    )
except Exception:  # noqa: BLE001
    def _increment_counter(metric_name: str, value: float = 1, labels: dict[str, str] | None = None) -> None:  # type: ignore[misc]
        pass

    def _set_gauge(metric_name: str, value: float = 0, labels: dict[str, str] | None = None) -> None:  # type: ignore[misc]
        pass


# ---------------------------------------------------------------------------
# Metrics configuration
# ---------------------------------------------------------------------------

# Label validation allow-lists (unknown labels normalised to ``"other"``)
_VALID_CATEGORIES: set[str] = {
    "", "embeddings", "evaluations", "tts", "rag", "chat", "mcp",
    "websearch",
}
_VALID_REASONS: set[str] = {
    "", "unknown", "threshold", "half_open_failure", "timeout",
}

_warned_labels: set[str] = set()


def _validate_label(value: str, valid_set: set[str], kind: str) -> str:
    """Return *value* if known, else ``"other"`` with a one-shot warning."""
    if value in valid_set:
        return value
    key = f"{kind}:{value}"
    if key not in _warned_labels:
        _warned_labels.add(key)
        logger.warning("Unknown circuit-breaker {} label {!r}, normalising to 'other'", kind, value)
    return "other"


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class CircuitState(IntEnum):
    """Circuit breaker states (int values for gauge metric compatibility)."""
    CLOSED = 0
    OPEN = 1
    HALF_OPEN = 2


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Configuration for a :class:`CircuitBreaker` instance.

    This dataclass is frozen (immutable).  Use :meth:`CircuitBreaker.update_config`
    to replace the config at runtime.
    """

    failure_threshold: int = 5
    """Consecutive failures (counter mode) before tripping to OPEN."""

    success_threshold: int = 2
    """Consecutive successes in HALF_OPEN before returning to CLOSED."""

    recovery_timeout: float = 60.0
    """Seconds to wait in OPEN before transitioning to HALF_OPEN."""

    half_open_max_calls: int = 1
    """Max concurrent probe calls allowed in HALF_OPEN (serial by default)."""

    expected_exception: type | tuple = Exception
    """Exception type(s) that count as circuit-breaker failures.
    Exceptions *not* matching this are always re-raised without recording."""

    # Rolling-window mode (disabled when window_size == 0)
    window_size: int = 0
    """Rolling-window size.  ``0`` = counter mode; ``>0`` = rolling-window."""

    failure_rate_threshold: float = 0.5
    """Failure rate (0..1) at which the breaker trips in rolling-window mode."""

    # Minimum calls before tripping
    min_calls: int = 0
    """Minimum number of calls in the rolling window before the breaker can
    trip.  ``0`` means use *window_size* as minimum (backward-compat)."""

    # Call timeout (async only)
    call_timeout: float | None = None
    """If set, wraps ``call_async`` in ``asyncio.wait_for()`` with this timeout."""

    # Exponential backoff on recovery timeout
    backoff_factor: float = 1.0
    """Multiplier applied to *recovery_timeout* after each HALF_OPEN → OPEN
    transition.  ``1.0`` = no backoff; ``2.0`` = double each time."""

    max_recovery_timeout: float = 300.0
    """Upper bound for the recovery timeout after backoff."""

    # Metrics labels
    category: str = ""
    """Arbitrary label emitted with every metric (e.g. ``"embeddings"``)."""

    service: str = ""
    """Service label for metrics; falls back to the breaker's *name*."""

    operation: str = "call"
    """Operation label for success/failure counters."""

    # Metrics opt-out
    emit_metrics: bool = True
    """Set to ``False`` to suppress all metric emissions for this breaker."""


class CircuitBreakerOpenError(Exception):
    """Raised when a call is rejected because the circuit is OPEN.

    Attributes carry enough context for callers to build rich error
    responses (HTTP 503, retry-after headers, etc.).
    """

    def __init__(
        self,
        message: str,
        *,
        breaker_name: str = "",
        category: str = "",
        service: str = "",
        recovery_timeout: float = 0.0,
        failure_count: int = 0,
        recovery_at: float | None = None,
    ):
        super().__init__(message)
        self.breaker_name = breaker_name
        self.category = category
        self.service = service
        self.recovery_timeout = recovery_timeout
        self.failure_count = failure_count
        self.recovery_at = recovery_at


class PersistMutationKind(str, Enum):
    """Contract enum for persisted breaker mutations (stage 2 implementation target).

    Values represent logical mutations applied to local in-memory breaker state
    that may later be merged into shared storage after optimistic-lock conflicts.
    """

    SUCCESS = "success"
    FAILURE = "failure"
    TRANSITION = "transition"
    RESET = "reset"
    SLOT_ACQUIRE = "slot_acquire"
    SLOT_RELEASE = "slot_release"
    SNAPSHOT_SYNC = "snapshot_sync"


@dataclass(frozen=True)
class PersistMutation:
    """Immutable persistence mutation contract used by merge/retry logic.

    Stage 1 intentionally defines this structure without changing runtime
    behavior. Stage 2 will use it to apply deterministic conflict merges.
    """

    kind: PersistMutationKind
    observed_version: int
    observed_state: CircuitState
    observed_at: float


MERGE_POLICY_CONTRACT_V1 = """
Circuit-breaker optimistic-lock merge contract (Stage 1 design freeze):

1. OPEN transitions are sticky across conflicts:
   A stale writer must not overwrite a newer OPEN transition with CLOSED/HALF_OPEN.
2. Counters merge as operation deltas, not whole-snapshot replacement:
   SUCCESS/FAILURE updates apply relative increments against the latest persisted row.
3. Timestamps are monotonic and conflict-safe:
   last_state_change_time and last_failure_time use max(local, persisted) semantics.
4. Retry is bounded:
   merge/retry loops must use a finite attempt budget and fail safe when exhausted.
5. Determinism:
   given the same persisted row and mutation sequence, merge output is deterministic.
""".strip()


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """Unified circuit breaker supporting sync and async callers.

    Parameters are supplied via a :class:`CircuitBreakerConfig` dataclass
    (or as individual keyword arguments for backward compatibility).

    Two failure-detection modes are supported:

    * **Counter mode** (``config.window_size == 0``): the circuit opens after
      ``failure_threshold`` consecutive failures.  A single success resets
      the counter.
    * **Rolling-window mode** (``config.window_size > 0``): a fixed-size
      deque records recent call outcomes and the breaker trips when the
      failure *rate* reaches ``failure_rate_threshold`` **and** the window
      is full.
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        *,
        on_state_change: list[Callable] | None = None,
        state_store: _RegistryDB | None = None,
        store_sync_interval: float = 0.0,
        # Convenience kwargs that populate a config when *config* is None:
        failure_threshold: int | None = None,
        recovery_timeout: float | None = None,
        expected_exception: type | tuple | None = None,
        success_threshold: int | None = None,
        half_open_max_calls: int | None = None,
        category: str | None = None,
        service: str | None = None,
        operation: str | None = None,
        window_size: int | None = None,
        failure_rate_threshold: float | None = None,
        backoff_factor: float | None = None,
        max_recovery_timeout: float | None = None,
    ):
        if config is None:
            kwargs: dict[str, Any] = {}
            if failure_threshold is not None:
                kwargs["failure_threshold"] = failure_threshold
            if recovery_timeout is not None:
                kwargs["recovery_timeout"] = recovery_timeout
            if expected_exception is not None:
                kwargs["expected_exception"] = expected_exception
            if success_threshold is not None:
                kwargs["success_threshold"] = success_threshold
            if half_open_max_calls is not None:
                kwargs["half_open_max_calls"] = half_open_max_calls
            if category is not None:
                kwargs["category"] = category
            if service is not None:
                kwargs["service"] = service
            if operation is not None:
                kwargs["operation"] = operation
            if window_size is not None:
                kwargs["window_size"] = window_size
            if failure_rate_threshold is not None:
                kwargs["failure_rate_threshold"] = failure_rate_threshold
            if backoff_factor is not None:
                kwargs["backoff_factor"] = backoff_factor
            if max_recovery_timeout is not None:
                kwargs["max_recovery_timeout"] = max_recovery_timeout
            config = CircuitBreakerConfig(**kwargs)

        self.name = name
        self.config = config
        if state_store is not None and config.window_size > 0:
            raise ValueError(
                "Persistent rolling-window circuit breakers are not supported "
                "until window state is stored in the shared registry"
            )

        # Resolve metrics labels
        self._category = config.category or ""
        self._service = config.service or name
        self._operation = config.operation or "call"

        # State
        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0
        self._last_failure_time: float | None = None
        self._half_open_calls: int = 0
        self._last_trip_failure_count: int = 0

        # Rolling window (only used when window_size > 0)
        self._window: deque[bool] | None = None
        if config.window_size > 0:
            self._window = deque(maxlen=config.window_size)

        # State-change tracking
        self._last_state_change_time: float = time.time()

        # Exponential backoff state
        self._current_recovery_timeout: float = config.recovery_timeout

        # Callbacks: list[Callable[[CircuitBreaker, CircuitState, CircuitState], None]]
        self.on_state_change: list[Callable] = list(on_state_change or [])

        # Thread safety (sync path)
        self._lock = threading.RLock()

        # Optional persistent shared storage state
        self._state_store: _RegistryDB | None = state_store
        self._state_store_version: int = 0
        self._store_sync_interval: float = max(0.0, float(store_sync_interval))
        self._last_store_sync: float = 0.0
        try:
            parsed_retries = int(os.getenv("CIRCUIT_BREAKER_PERSIST_MAX_RETRIES", "4"))
        except ValueError:
            parsed_retries = 4
        self._persist_retry_attempts: int = max(1, parsed_retries)
        try:
            parsed_lease_ttl = float(
                os.getenv("CIRCUIT_BREAKER_HALF_OPEN_LEASE_TTL_SECONDS", "120.0")
            )
        except ValueError:
            parsed_lease_ttl = 120.0
        self._half_open_lease_ttl_seconds: float = max(1.0, parsed_lease_ttl)
        self._lease_owner_id: str = f"{os.getpid()}:{uuid.uuid4().hex}"

        if self._state_store is not None:
            self._sync_from_store_locked(force=True)
            if self._state_store_version == 0:
                self._persist_state_locked(PersistMutationKind.SNAPSHOT_SYNC)

        logger.debug(
            "Circuit breaker '{}' initialized: failure_threshold={}, "
            "recovery_timeout={}s, window_size={}",
            name,
            config.failure_threshold,
            config.recovery_timeout,
            config.window_size,
        )

    # -- convenience properties ---------------------------------------------

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._sync_from_store_locked()
            self._maybe_transition_to_half_open()
            return self._state

    @property
    def failure_count(self) -> int:
        with self._lock:
            self._sync_from_store_locked()
            return self._failure_count

    @property
    def success_count(self) -> int:
        with self._lock:
            self._sync_from_store_locked()
            return self._success_count

    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        return self.state == CircuitState.HALF_OPEN

    @property
    def last_state_change_time(self) -> float:
        """Timestamp of the most recent state transition."""
        with self._lock:
            self._sync_from_store_locked()
            return self._last_state_change_time

    @property
    def last_failure_time(self) -> float | None:
        """Timestamp of the most recent recorded failure."""
        with self._lock:
            self._sync_from_store_locked()
            return self._last_failure_time

    @property
    def current_recovery_timeout(self) -> float:
        """Current recovery timeout (may differ from config if backoff active)."""
        with self._lock:
            self._sync_from_store_locked()
            return self._current_recovery_timeout

    @property
    def half_open_calls(self) -> int:
        """Number of currently active half-open probe calls."""
        with self._lock:
            self._sync_from_store_locked()
            return self._half_open_calls

    def _to_stored_state_locked(self) -> _StoredState:
        return _StoredState(
            state=int(self._state.value),
            failure_count=int(self._failure_count),
            success_count=int(self._success_count),
            last_failure_time=self._last_failure_time,
            last_state_change_time=float(self._last_state_change_time),
            half_open_calls=int(self._half_open_calls),
            current_recovery_timeout=float(self._current_recovery_timeout),
            last_trip_failure_count=int(self._last_trip_failure_count),
        )

    def _apply_stored_state_locked(self, stored: _StoredState, *, version: int) -> None:
        try:
            state = CircuitState(int(stored.state))
        except Exception:
            logger.warning(
                "Ignoring persisted circuit breaker state with unknown value {} for '{}'",
                stored.state,
                self.name,
            )
            return
        self._state = state
        self._failure_count = int(stored.failure_count)
        self._success_count = int(stored.success_count)
        self._last_failure_time = stored.last_failure_time
        self._last_state_change_time = float(stored.last_state_change_time)
        self._half_open_calls = int(stored.half_open_calls)
        self._current_recovery_timeout = float(stored.current_recovery_timeout)
        self._last_trip_failure_count = int(stored.last_trip_failure_count)
        self._state_store_version = int(version)

    def _sync_from_store_locked(self, *, force: bool = False) -> None:
        if self._state_store is None:
            return
        now = time.time()
        if (
            not force
            and self._store_sync_interval > 0
            and now - self._last_store_sync < self._store_sync_interval
        ):
            return
        row = self._state_store.load(self.name)
        self._last_store_sync = now
        if row is None:
            return
        stored, version = row
        if version > self._state_store_version:
            self._apply_stored_state_locked(stored, version=version)

    @staticmethod
    def _max_optional_time(a: float | None, b: float | None) -> float | None:
        if a is None:
            return b
        if b is None:
            return a
        return max(a, b)

    def _merge_persist_conflict_locked(
        self,
        *,
        latest: _StoredState,
        local: _StoredState,
        mutation: PersistMutation,
    ) -> _StoredState:
        """Merge *local* mutation into *latest* persisted state after a conflict.

        This method is deterministic and intentionally conservative:
        - Local SUCCESS/FAILURE operations are re-applied as deltas.
        - OPEN state is sticky and cannot be overwritten by a stale non-OPEN snapshot.
        """
        latest_state = CircuitState(int(latest.state))
        local_state = CircuitState(int(local.state))

        if mutation.kind == PersistMutationKind.FAILURE:
            state = latest_state
            failure_count = int(latest.failure_count)
            success_count = int(latest.success_count)
            half_open_calls = int(latest.half_open_calls)
            current_recovery_timeout = max(
                float(latest.current_recovery_timeout),
                float(local.current_recovery_timeout),
            )
            last_trip_failure_count = max(
                int(latest.last_trip_failure_count),
                int(local.last_trip_failure_count),
            )
            last_failure_time = self._max_optional_time(
                latest.last_failure_time,
                local.last_failure_time,
            )
            last_state_change_time = max(
                float(latest.last_state_change_time),
                float(local.last_state_change_time),
            )

            if state == CircuitState.HALF_OPEN:
                # Any failure in HALF_OPEN re-opens the breaker.
                pre_trip = max(1, int(latest.failure_count) + 1)
                state = CircuitState.OPEN
                failure_count = 0
                success_count = 0
                half_open_calls = 0
                last_trip_failure_count = max(last_trip_failure_count, pre_trip)
            elif state == CircuitState.CLOSED:
                failure_count = int(latest.failure_count) + 1
                should_open = local_state == CircuitState.OPEN
                if not should_open and self._window is None:
                    should_open = failure_count >= self.config.failure_threshold
                if should_open:
                    state = CircuitState.OPEN
                    last_trip_failure_count = max(last_trip_failure_count, failure_count)
                    failure_count = 0
                    success_count = 0
                    half_open_calls = 0
            elif state == CircuitState.OPEN:
                # Keep OPEN sticky; retain latest counters and only refresh metadata.
                half_open_calls = 0

            return _StoredState(
                state=int(state.value),
                failure_count=int(failure_count),
                success_count=int(success_count),
                last_failure_time=last_failure_time,
                last_state_change_time=float(last_state_change_time),
                half_open_calls=int(half_open_calls),
                current_recovery_timeout=float(current_recovery_timeout),
                last_trip_failure_count=int(last_trip_failure_count),
            )

        if mutation.kind == PersistMutationKind.SUCCESS:
            state = latest_state
            failure_count = int(latest.failure_count)
            success_count = int(latest.success_count)
            half_open_calls = int(latest.half_open_calls)
            current_recovery_timeout = max(
                float(latest.current_recovery_timeout),
                float(local.current_recovery_timeout),
            )
            last_trip_failure_count = max(
                int(latest.last_trip_failure_count),
                int(local.last_trip_failure_count),
            )
            last_failure_time = self._max_optional_time(
                latest.last_failure_time,
                local.last_failure_time,
            )
            last_state_change_time = max(
                float(latest.last_state_change_time),
                float(local.last_state_change_time),
            )

            if state == CircuitState.HALF_OPEN:
                success_count = int(latest.success_count) + 1
                if success_count >= self.config.success_threshold:
                    state = CircuitState.CLOSED
                    failure_count = 0
                    success_count = 0
                    half_open_calls = 0
                    current_recovery_timeout = self.config.recovery_timeout
            elif state == CircuitState.CLOSED:
                if self._window is None:
                    failure_count = 0
            elif state == CircuitState.OPEN:
                # Successes are ignored while OPEN to preserve OPEN stickiness.
                half_open_calls = 0

            return _StoredState(
                state=int(state.value),
                failure_count=int(failure_count),
                success_count=int(success_count),
                last_failure_time=last_failure_time,
                last_state_change_time=float(last_state_change_time),
                half_open_calls=int(half_open_calls),
                current_recovery_timeout=float(current_recovery_timeout),
                last_trip_failure_count=int(last_trip_failure_count),
            )

        # Snapshot/transition/reset path: prefer local snapshot unless it would
        # clobber a newer OPEN state.
        if latest_state == CircuitState.OPEN and local_state != CircuitState.OPEN:
            return _StoredState(
                state=int(latest_state.value),
                failure_count=int(latest.failure_count),
                success_count=int(latest.success_count),
                last_failure_time=latest.last_failure_time,
                last_state_change_time=max(
                    float(latest.last_state_change_time),
                    float(local.last_state_change_time),
                ),
                half_open_calls=int(latest.half_open_calls),
                current_recovery_timeout=float(latest.current_recovery_timeout),
                last_trip_failure_count=max(
                    int(latest.last_trip_failure_count),
                    int(local.last_trip_failure_count),
                ),
            )

        return _StoredState(
            state=int(local.state),
            failure_count=int(local.failure_count),
            success_count=int(local.success_count),
            last_failure_time=local.last_failure_time,
            last_state_change_time=max(
                float(latest.last_state_change_time),
                float(local.last_state_change_time),
            ),
            half_open_calls=int(local.half_open_calls),
            current_recovery_timeout=float(local.current_recovery_timeout),
            last_trip_failure_count=int(local.last_trip_failure_count),
        )

    def _emit_persist_conflict(
        self,
        *,
        mutation: PersistMutation,
        attempt: int,
        current_version: int | None,
    ) -> None:
        if self.config.emit_metrics:
            labels = self._metric_labels("persist_conflict")
            labels["mutation"] = mutation.kind.value
            _increment_counter(
                "circuit_breaker_persist_conflicts_total",
                labels=labels,
            )
        logger.debug(
            "Circuit breaker '{}' persistence conflict on attempt {}/{} "
            "(mutation={}, expected_version={}, current_version={})",
            self.name,
            attempt,
            self._persist_retry_attempts,
            mutation.kind.value,
            mutation.observed_version,
            current_version,
        )

    def _persist_state_locked(
        self,
        mutation_kind: PersistMutationKind = PersistMutationKind.SNAPSHOT_SYNC,
    ) -> None:
        if self._state_store is None:
            return

        pending_state = self._to_stored_state_locked()
        mutation = PersistMutation(
            kind=mutation_kind,
            observed_version=self._state_store_version,
            observed_state=self._state,
            observed_at=time.time(),
        )

        max_attempts = self._persist_retry_attempts
        for attempt in range(1, max_attempts + 1):
            expected_version = self._state_store_version
            try:
                new_version = self._state_store.upsert(
                    self.name,
                    pending_state,
                    expected_version=expected_version,
                )
                self._state_store_version = new_version
                self._last_store_sync = time.time()
                return
            except _StoreOptimisticLockError as exc:
                self._emit_persist_conflict(
                    mutation=mutation,
                    attempt=attempt,
                    current_version=exc.current_version,
                )
                latest = self._state_store.load(self.name)
                if latest is None:
                    # Row disappeared between load/upsert; retry as a fresh insert.
                    self._state_store_version = 0
                    continue

                latest_state, latest_version = latest
                if attempt >= max_attempts:
                    self._apply_stored_state_locked(
                        latest_state,
                        version=latest_version,
                    )
                    logger.warning(
                        "Circuit breaker '{}' persistence retries exhausted; "
                        "adopting latest shared state at version {}",
                        self.name,
                        latest_version,
                    )
                    return

                merged_state = self._merge_persist_conflict_locked(
                    latest=latest_state,
                    local=pending_state,
                    mutation=mutation,
                )
                self._apply_stored_state_locked(
                    merged_state,
                    version=latest_version,
                )
                pending_state = merged_state

    def _attach_state_store(self, state_store: _RegistryDB, *, sync_interval: float = 0.0) -> None:
        with self._lock:
            if self.config.window_size > 0:
                raise ValueError(
                    "Persistent rolling-window circuit breakers are not supported "
                    "until window state is stored in the shared registry"
                )
            self._state_store = state_store
            self._store_sync_interval = max(0.0, float(sync_interval))
            self._state_store_version = 0
            self._last_store_sync = 0.0
            self._sync_from_store_locked(force=True)
            if self._state_store_version == 0:
                self._persist_state_locked(PersistMutationKind.SNAPSHOT_SYNC)

    # -- check-pattern API (for Chat, WebSearch) ----------------------------

    def can_attempt(self) -> bool:
        """Return *True* if a call would be allowed right now.

        Does **not** consume a half-open slot; use :meth:`call` /
        :meth:`call_async` for gated execution, or pair ``can_attempt``
        with ``record_success`` / ``record_failure`` manually.
        """
        with self._lock:
            self._sync_from_store_locked()
            self._maybe_transition_to_half_open()
            if self._state == CircuitState.OPEN:
                return False
            if self._state == CircuitState.HALF_OPEN:
                if self._state_store is not None:
                    try:
                        active = self._state_store.count_active_probe_leases(self.name)
                        return active < self.config.half_open_max_calls
                    except Exception as exc:  # noqa: BLE001
                        logger.debug(
                            "Circuit breaker '{}' failed reading distributed probe leases: {}",
                            self.name,
                            exc,
                        )
                return self._half_open_calls < self.config.half_open_max_calls
            return True

    def record_success(self) -> None:
        """Manually record a success (check-pattern API)."""
        with self._lock:
            self._sync_from_store_locked()
            self._on_success()

    def record_failure(self, error: Exception | None = None) -> None:
        """Manually record a failure (check-pattern API)."""
        with self._lock:
            self._sync_from_store_locked()
            self._on_failure(error)

    # -- sync execution -----------------------------------------------------

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute *func* through the circuit breaker (sync).

        Raises :class:`CircuitBreakerOpenError` when the circuit is OPEN
        (or HALF_OPEN with no probe slots left).  On failure, the original
        exception is re-raised after being recorded.
        """
        with self._lock:
            self._sync_from_store_locked()
            self._maybe_transition_to_half_open()
            self._guard_or_raise()
            acquired_slot, lease_id = self._acquire_half_open_slot()
            if self._state == CircuitState.HALF_OPEN and not acquired_slot:
                self._raise_half_open_probe_limit_locked()

        try:
            result = func(*args, **kwargs)
        except BaseException as exc:
            if isinstance(exc, self.config.expected_exception):
                with self._lock:
                    self._on_failure(exc)
            raise
        else:
            with self._lock:
                self._on_success()
            return result
        finally:
            if acquired_slot:
                with self._lock:
                    self._release_half_open_slot_locked(lease_id)

    # -- async execution ----------------------------------------------------

    async def call_async(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute *func* through the circuit breaker (async).

        Works identically to :meth:`call` but awaits the wrapped function.
        If ``config.call_timeout`` is set, wraps the call in
        ``asyncio.wait_for()``.
        """
        # Use threading lock for state checks (fast, no I/O).
        with self._lock:
            self._sync_from_store_locked()
            self._maybe_transition_to_half_open()
            self._guard_or_raise()
            acquired_slot, lease_id = self._acquire_half_open_slot()
            if self._state == CircuitState.HALF_OPEN and not acquired_slot:
                self._raise_half_open_probe_limit_locked()

        try:
            if self.config.call_timeout is not None:
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.call_timeout,
                )
            else:
                result = await func(*args, **kwargs)
        except asyncio.TimeoutError:
            with self._lock:
                self._on_failure(None)
                self._emit_timeout()
            raise
        except BaseException as exc:
            if isinstance(exc, self.config.expected_exception):
                with self._lock:
                    self._on_failure(exc)
            raise
        else:
            with self._lock:
                self._on_success()
            return result
        finally:
            if acquired_slot:
                with self._lock:
                    self._release_half_open_slot_locked(lease_id)

    # -- admin / introspection ----------------------------------------------

    def reset(self) -> None:
        """Manually reset to CLOSED (e.g. after a deploy or config change)."""
        with self._lock:
            self._sync_from_store_locked()
            old = self._state
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._last_trip_failure_count = 0
            self._last_failure_time = None
            self._last_state_change_time = time.time()
            self._current_recovery_timeout = self.config.recovery_timeout
            if self._window is not None:
                self._window.clear()
            self._clear_probe_leases_locked()
            self._emit_state_gauge()
            if old != CircuitState.CLOSED:
                self._fire_callbacks(old, CircuitState.CLOSED)
            self._persist_state_locked(PersistMutationKind.RESET)
            logger.info("Circuit breaker '{}' manually reset", self.name)

    def force_open(self) -> None:
        """Force the breaker into OPEN state (e.g. for testing or manual intervention)."""
        with self._lock:
            if self._state != CircuitState.OPEN:
                self._transition(CircuitState.OPEN, reason="unknown")

    def force_half_open(self) -> None:
        """Force the breaker into HALF_OPEN state (e.g. for testing)."""
        with self._lock:
            if self._state != CircuitState.HALF_OPEN:
                self._transition(CircuitState.HALF_OPEN)

    def update_config(self, new_config: CircuitBreakerConfig) -> None:
        """Replace the breaker's config with *new_config* under lock.

        Since ``CircuitBreakerConfig`` is frozen, this replaces the entire
        object.  Re-derives metrics labels and adjusts the rolling window
        if ``window_size`` changed.
        """
        with self._lock:
            old_config = self.config
            self.config = new_config
            self._category = new_config.category or ""
            self._service = new_config.service or self.name
            self._operation = new_config.operation or "call"
            self._current_recovery_timeout = new_config.recovery_timeout

            # Adjust window if window_size changed
            if new_config.window_size > 0:
                if self._window is None or self._window.maxlen != new_config.window_size:
                    self._window = deque(maxlen=new_config.window_size)
            elif new_config.window_size == 0:
                self._window = None

            logger.info(
                "Circuit breaker '{}' config updated: {} -> {}",
                self.name,
                old_config,
                new_config,
            )

    def get_status(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the breaker's state."""
        with self._lock:
            self._sync_from_store_locked()
            self._maybe_transition_to_half_open()
            result: dict[str, Any] = {
                "name": self.name,
                "state": self._state.name,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure_time": self._last_failure_time,
                "last_state_change_time": self._last_state_change_time,
                "half_open_calls": self._half_open_calls,
                "current_recovery_timeout": self._current_recovery_timeout,
                "settings": {
                    "failure_threshold": self.config.failure_threshold,
                    "recovery_timeout": self.config.recovery_timeout,
                    "success_threshold": self.config.success_threshold,
                    "half_open_max_calls": self.config.half_open_max_calls,
                    "window_size": self.config.window_size,
                    "failure_rate_threshold": self.config.failure_rate_threshold,
                    "backoff_factor": self.config.backoff_factor,
                    "max_recovery_timeout": self.config.max_recovery_timeout,
                    "min_calls": self.config.min_calls,
                    "call_timeout": self.config.call_timeout,
                    "emit_metrics": self.config.emit_metrics,
                },
            }
            if self._window is not None:
                result["window"] = list(self._window)
                result["failure_rate"] = (
                    sum(1 for r in self._window if not r) / len(self._window)
                    if self._window
                    else 0.0
                )
            return result

    # -- internal state machine ---------------------------------------------

    def _maybe_transition_to_half_open(self) -> None:
        """If OPEN and recovery timeout elapsed, move to HALF_OPEN.

        Must be called while ``self._lock`` is held.
        """
        if (
            self._state == CircuitState.OPEN
            and self._last_failure_time is not None
            and time.time() - self._last_failure_time >= self._current_recovery_timeout
        ):
            self._transition(CircuitState.HALF_OPEN)

    def _guard_or_raise(self) -> None:
        """Raise :class:`CircuitBreakerOpenError` if the call should be rejected.

        Must be called while ``self._lock`` is held.
        """
        if self._state == CircuitState.OPEN:
            self._emit_rejection()
            recovery_at = (
                self._last_failure_time + self._current_recovery_timeout
                if self._last_failure_time else None
            )
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is OPEN",
                breaker_name=self.name,
                category=self._category,
                service=self._service,
                recovery_timeout=self._current_recovery_timeout,
                failure_count=self._last_trip_failure_count,
                recovery_at=recovery_at,
            )
        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.config.half_open_max_calls:
                self._emit_rejection()
                recovery_at = (
                    self._last_failure_time + self._current_recovery_timeout
                    if self._last_failure_time else None
                )
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' HALF_OPEN probe limit reached",
                    breaker_name=self.name,
                    category=self._category,
                    service=self._service,
                    recovery_timeout=self._current_recovery_timeout,
                    failure_count=self._failure_count,
                    recovery_at=recovery_at,
                )

    def _raise_half_open_probe_limit_locked(self) -> None:
        """Raise a standard HALF_OPEN probe-limit rejection error.

        Must be called while ``self._lock`` is held.
        """
        self._emit_rejection()
        recovery_at = (
            self._last_failure_time + self._current_recovery_timeout
            if self._last_failure_time else None
        )
        raise CircuitBreakerOpenError(
            f"Circuit breaker '{self.name}' HALF_OPEN probe limit reached",
            breaker_name=self.name,
            category=self._category,
            service=self._service,
            recovery_timeout=self._current_recovery_timeout,
            failure_count=self._failure_count,
            recovery_at=recovery_at,
        )

    def _clear_probe_leases_locked(self) -> None:
        """Best-effort cleanup of distributed HALF_OPEN probe leases."""
        if self._state_store is None:
            return
        try:
            self._state_store.clear_probe_leases(self.name)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Circuit breaker '{}' failed clearing probe leases: {}",
                self.name,
                exc,
            )

    def _acquire_half_open_slot(self) -> tuple[bool, str | None]:
        """Try to acquire a half-open probe slot.

        Returns ``(acquired, lease_id)`` where ``lease_id`` is only set when
        distributed lease coordination is enabled.
        """
        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.config.half_open_max_calls:
                return False, None

            lease_id: str | None = None
            if self._state_store is not None:
                try:
                    lease = self._state_store.acquire_probe_lease(
                        self.name,
                        max_calls=self.config.half_open_max_calls,
                        ttl_seconds=self._half_open_lease_ttl_seconds,
                        owner_id=self._lease_owner_id,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Circuit breaker '{}' failed to acquire distributed probe lease: {}",
                        self.name,
                        exc,
                    )
                    return False, None
                if lease is None:
                    return False, None
                lease_id = lease.lease_id

            self._half_open_calls += 1
            return True, lease_id
        return False, None

    def _release_half_open_slot_locked(self, lease_id: str | None) -> None:
        """Release a local and distributed half-open probe slot."""
        if self._half_open_calls > 0:
            self._half_open_calls -= 1
        if lease_id is None or self._state_store is None:
            return
        try:
            self._state_store.release_probe_lease(self.name, lease_id)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "Circuit breaker '{}' failed to release probe lease {}: {}",
                self.name,
                lease_id,
                exc,
            )

    def _on_success(self) -> None:
        """Record a successful call.  Must be called while ``self._lock`` is held."""
        self._emit_success()
        transitioned = False

        if self._window is not None:
            self._window.append(True)

        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._transition(CircuitState.CLOSED)
                transitioned = True
        elif self._state == CircuitState.CLOSED:
            # In counter mode a success resets the failure counter
            if self._window is None:
                self._failure_count = 0
        if not transitioned:
            self._persist_state_locked(PersistMutationKind.SUCCESS)

    def _on_failure(self, error: Exception | None = None) -> None:
        """Record a failed call.  Must be called while ``self._lock`` is held."""
        self._emit_failure(error)
        self._last_failure_time = time.time()
        transitioned = False

        if self._window is not None:
            self._window.append(False)

        self._failure_count += 1

        if self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open re-opens the circuit
            self._transition(CircuitState.OPEN, reason="half_open_failure")
            transitioned = True
        elif self._state == CircuitState.CLOSED:
            if self._should_trip():
                self._transition(CircuitState.OPEN, reason="threshold")
                transitioned = True
        if not transitioned:
            self._persist_state_locked(PersistMutationKind.FAILURE)

    def _should_trip(self) -> bool:
        """Decide whether the breaker should trip from CLOSED → OPEN.

        Must be called while ``self._lock`` is held.
        """
        if self._window is not None:
            # Rolling-window mode
            effective_min = self.config.min_calls or self.config.window_size
            if len(self._window) >= effective_min:
                rate = sum(1 for r in self._window if not r) / len(self._window)
                return rate >= self.config.failure_rate_threshold
            return False
        # Counter mode
        return self._failure_count >= self.config.failure_threshold

    def _transition(self, new_state: CircuitState, *, reason: str = "") -> None:
        """Perform a state transition.  Must be called while ``self._lock`` is held."""
        old = self._state
        if old == new_state:
            return

        self._state = new_state
        self._last_state_change_time = time.time()

        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._current_recovery_timeout = self.config.recovery_timeout
            if self._window is not None:
                self._window.clear()
            self._clear_probe_leases_locked()
            logger.info("Circuit breaker '{}' CLOSED", self.name)

        elif new_state == CircuitState.OPEN:
            if old == CircuitState.HALF_OPEN and self.config.backoff_factor > 1.0:
                self._current_recovery_timeout = min(
                    self._current_recovery_timeout * self.config.backoff_factor,
                    self.config.max_recovery_timeout,
                )
            self._last_trip_failure_count = self._failure_count
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._last_failure_time = time.time()
            self._clear_probe_leases_locked()
            self._emit_trip(reason)
            logger.warning(
                "Circuit breaker '{}' OPEN — retry in {:.1f}s",
                self.name,
                self._current_recovery_timeout,
            )

        elif new_state == CircuitState.HALF_OPEN:
            self._success_count = 0
            self._failure_count = 0
            self._half_open_calls = 0
            logger.info("Circuit breaker '{}' HALF_OPEN (probing)", self.name)

        self._emit_state_gauge()
        self._fire_callbacks(old, new_state)
        self._persist_state_locked(PersistMutationKind.TRANSITION)

    # -- callbacks ----------------------------------------------------------

    def _fire_callbacks(self, old: CircuitState, new: CircuitState) -> None:
        for cb in self.on_state_change:
            try:
                cb(self, old, new)
            except Exception:  # noqa: BLE001
                logger.opt(exception=True).warning(
                    "State-change callback error in breaker '{}'", self.name
                )

    # -- metrics emission ---------------------------------------------------

    def _metric_labels(self, operation: str) -> dict[str, str]:
        return {
            "category": self._category,
            "service": self._service,
            "operation": operation,
        }

    def _emit_state_gauge(self) -> None:
        if not self.config.emit_metrics:
            return
        labels = self._metric_labels("state_change")
        _set_gauge(
            "circuit_breaker_state",
            float(self._state.value),
            labels=labels,
        )

    def _emit_trip(self, reason: str) -> None:
        if not self.config.emit_metrics:
            return
        validated_reason = _validate_label(reason or "unknown", _VALID_REASONS, "reason")
        labels = {
            "category": self._category,
            "service": self._service,
            "reason": validated_reason,
        }
        _increment_counter("circuit_breaker_trips_total", labels=labels)

    def _emit_rejection(self) -> None:
        if not self.config.emit_metrics:
            return
        labels = self._metric_labels(self._operation)
        _increment_counter(
            "circuit_breaker_rejections_total",
            labels=labels,
        )

    def _emit_success(self) -> None:
        if not self.config.emit_metrics:
            return
        labels = self._metric_labels(self._operation)
        _increment_counter(
            "circuit_breaker_successes_total",
            labels=labels,
        )

    def _emit_failure(self, error: Exception | None = None) -> None:
        if not self.config.emit_metrics:
            return
        outcome = type(error).__name__ if error else "error"
        labels = {
            "category": self._category,
            "service": self._service,
            "operation": self._operation,
            "outcome": outcome,
        }
        _increment_counter(
            "circuit_breaker_failures_total",
            labels=labels,
        )

    def _emit_timeout(self) -> None:
        """Emit the ``circuit_breaker_timeouts_total`` counter."""
        if not self.config.emit_metrics:
            return
        _increment_counter(
            "circuit_breaker_timeouts_total",
            labels=self._metric_labels(self._operation),
        )

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<CircuitBreaker name={self.name!r} state={self._state.name} "
            f"failures={self._failure_count}>"
        )


# ---------------------------------------------------------------------------
# CircuitBreakerRegistry
# ---------------------------------------------------------------------------

def _should_enable_registry_persistence(explicit: bool | None) -> bool:
    if explicit is not None:
        return bool(explicit)
    mode = str(os.getenv("CIRCUIT_BREAKER_REGISTRY_MODE", "auto")).strip().lower()
    if mode in {"memory", "inmemory", "off", "disabled", "none"}:
        return False
    if mode in {"persistent", "shared", "db", "database", "on", "enabled"}:
        return True
    # Auto mode: keep tests fast/isolated while using shared persistence in runtime.
    return not is_explicit_pytest_runtime()


class CircuitBreakerRegistry:
    """Thread-safe registry for managing named :class:`CircuitBreaker` instances."""

    def __init__(
        self,
        *,
        persistence_enabled: bool | None = None,
        db_path: str | None = None,
        store_sync_interval: float = 0.0,
    ) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()
        self._store_sync_interval = max(0.0, float(store_sync_interval))
        self._db: _RegistryDB | None = None
        self._persistence_enabled = _should_enable_registry_persistence(
            persistence_enabled,
        )
        if self._persistence_enabled:
            try:
                self._db = _RegistryDB(db_path=db_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Circuit breaker registry persistence unavailable, falling back to in-memory mode: {}",
                    exc,
                )
                self._persistence_enabled = False

    @property
    def persistence_enabled(self) -> bool:
        return bool(self._db is not None)

    def get_or_create(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        **kwargs: Any,
    ) -> CircuitBreaker:
        """Return an existing breaker or create and register a new one.

        If a breaker with *name* already exists and *config* is provided,
        a warning is logged if the configs differ.
        """
        with self._lock:
            if name in self._breakers:
                existing = self._breakers[name]
                if config is not None and config != existing.config:
                    logger.warning(
                        "get_or_create('{}') called with different config; "
                        "returning existing breaker. Old: {}, New: {}",
                        name, existing.config, config,
                    )
                return existing
            breaker = CircuitBreaker(
                name,
                config=config,
                state_store=self._db,
                store_sync_interval=self._store_sync_interval,
                **kwargs,
            )
            self._breakers[name] = breaker
            return breaker

    def register(self, breaker: CircuitBreaker) -> None:
        """Register a pre-built breaker (replaces any existing one with the same name)."""
        with self._lock:
            if self._db is not None:
                breaker._attach_state_store(
                    self._db,
                    sync_interval=self._store_sync_interval,
                )
            self._breakers[breaker.name] = breaker

    def get(self, name: str) -> CircuitBreaker | None:
        """Look up a breaker by *name* (or ``None``)."""
        return self._breakers.get(name)

    def remove(self, name: str) -> bool:
        """Remove a breaker by *name*.  Returns ``True`` if it was found."""
        with self._lock:
            removed = self._breakers.pop(name, None) is not None
            if self._db is not None:
                self._db.delete(name)
            return removed

    def get_all_status(self) -> dict[str, dict[str, Any]]:
        """Return ``{name: status_dict}`` for every registered breaker."""
        with self._lock:
            statuses = {
                name: breaker.get_status()
                for name, breaker in self._breakers.items()
            }
            if self._db is not None:
                for breaker_name, (stored, _) in self._db.load_all().items():
                    if breaker_name in statuses:
                        continue
                    state_name = CircuitState(int(stored.state)).name
                    statuses[breaker_name] = {
                        "name": breaker_name,
                        "state": state_name,
                        "failure_count": stored.failure_count,
                        "success_count": stored.success_count,
                        "last_failure_time": stored.last_failure_time,
                        "last_state_change_time": stored.last_state_change_time,
                        "half_open_calls": stored.half_open_calls,
                        "current_recovery_timeout": stored.current_recovery_timeout,
                        "settings": {},
                    }
            return statuses

    def reset_all(self) -> None:
        """Reset every registered breaker to CLOSED."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
            if self._db is not None:
                for breaker_name in self._db.load_all():
                    if breaker_name not in self._breakers:
                        self._db.delete(breaker_name)

    def clear(self) -> None:
        """Remove all registered breakers from the registry."""
        with self._lock:
            self._breakers.clear()
            if self._db is not None:
                self._db.clear()

    def reset(self, name: str) -> bool:
        """Reset a single breaker by name.  Returns ``True`` if found."""
        breaker = self._breakers.get(name)
        if breaker is not None:
            breaker.reset()
            return True
        if self._db is not None:
            row = self._db.load(name)
            if row is not None:
                self._db.delete(name)
        return False


# Global singleton registry
registry = CircuitBreakerRegistry()


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def circuit_breaker(
    name: str | None = None,
    config: CircuitBreakerConfig | None = None,
    *,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: type | tuple = Exception,
    success_threshold: int = 2,
    half_open_max_calls: int = 1,
    category: str = "",
    service: str = "",
    operation: str = "call",
) -> Callable:
    """Decorator that wraps a function with a circuit breaker.

    The breaker is registered in the global :data:`registry`.  Auto-detects
    sync vs async callables.

    Example::

        @circuit_breaker(name="openai_api", failure_threshold=3)
        async def call_openai_api():
            ...
    """
    def decorator(func: Callable) -> Callable:
        breaker_name = name or f"{func.__module__}.{func.__qualname__}"
        cfg = config or CircuitBreakerConfig(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
            success_threshold=success_threshold,
            half_open_max_calls=half_open_max_calls,
            category=category,
            service=service or breaker_name,
            operation=operation,
        )
        breaker = registry.get_or_create(breaker_name, config=cfg)

        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: Any, **kw: Any) -> Any:
                return await breaker.call_async(func, *args, **kw)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args: Any, **kw: Any) -> Any:
                return breaker.call(func, *args, **kw)
            return sync_wrapper

    return decorator


# ---------------------------------------------------------------------------
# Public API summary
# ---------------------------------------------------------------------------

__all__ = [
    "CircuitBreakerConfig",
    "CircuitBreakerOpenError",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "CircuitState",
    "circuit_breaker",
    "registry",
]
