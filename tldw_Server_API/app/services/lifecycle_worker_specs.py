"""Declarative lifecycle worker specifications and validation helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

from tldw_Server_API.app.core.testing import env_flag_enabled
from tldw_Server_API.app.services.lifecycle_workers import ShutdownPhase


@dataclass(frozen=True)
class WorkerLifecycleContext:
    """Base context passed to lifecycle worker providers."""

    app: Any
    settings: Mapping[str, Any]
    test_mode: bool
    route_enabled: Callable[..., bool]
    logger: Any
    startup_guard_exceptions: tuple[type[BaseException], ...]
    import_exceptions: tuple[type[BaseException], ...]
    sidecar_mode: bool = False


StopEventWorkerFactory = Callable[
    [WorkerLifecycleContext, asyncio.Event],
    Awaitable[Any],
]
ShutdownCallbackFactory = Callable[
    [WorkerLifecycleContext],
    Callable[[], Awaitable[None]],
]


def always_enabled(_context: WorkerLifecycleContext) -> bool:
    """Default worker enablement predicate."""

    return True


class WorkerStrategy(str, Enum):
    """Worker startup and shutdown strategy."""

    STOP_EVENT_TASK = "stop_event_task"
    CALLBACK_ONLY = "callback_only"


class WorkerFailurePolicy(str, Enum):
    """Startup failure behavior for a worker spec."""

    SKIP = "skip"
    ABORT = "abort"


@dataclass(frozen=True)
class WorkerSpec:
    """Declarative contract for one lifecycle-managed worker."""

    name: str
    task_name: str
    category: str
    phase: ShutdownPhase
    timeout_sec: float = 5.0
    enabled: Callable[[WorkerLifecycleContext], bool] = always_enabled
    strategy: WorkerStrategy = WorkerStrategy.STOP_EVENT_TASK
    factory: StopEventWorkerFactory | None = None
    depends_on: tuple[str, ...] = ()
    shutdown_callback_factory: ShutdownCallbackFactory | None = None
    diagnostic_name: str | None = None
    failure_policy: WorkerFailurePolicy = WorkerFailurePolicy.SKIP


def stop_event_worker_spec(
    *,
    name: str,
    worker_service: Callable[[Any], Awaitable[Any]],
    category: str,
    phase: ShutdownPhase,
    enabled: Callable[[WorkerLifecycleContext], bool] = always_enabled,
    timeout_sec: float = 5.0,
) -> WorkerSpec:
    """Build a standard stop-event task worker spec."""

    def _factory(
        _context: WorkerLifecycleContext,
        stop_event: asyncio.Event,
    ) -> Awaitable[Any]:
        return worker_service(stop_event)

    return WorkerSpec(
        name=name,
        task_name=name,
        category=category,
        phase=phase,
        timeout_sec=timeout_sec,
        enabled=enabled,
        factory=_factory,
    )


def route_enabled_predicate(
    flag_key: str,
    route_key: str,
    **route_kwargs: object,
) -> Callable[[WorkerLifecycleContext], bool]:
    """Return a worker predicate backed by the lifecycle route gate."""

    def _enabled(context: WorkerLifecycleContext) -> bool:
        return env_flag_enabled(flag_key) and bool(
            context.route_enabled(route_key, **route_kwargs)
        )

    return _enabled


@dataclass(frozen=True)
class WorkerSpecGraph:
    """Validated worker spec graph indexed by worker name."""

    specs: tuple[WorkerSpec, ...]
    specs_by_name: Mapping[str, WorkerSpec]


class WorkerSpecValidationError(ValueError):
    """Raised when lifecycle worker specs are internally inconsistent."""


def validate_worker_spec_graph(specs: Sequence[WorkerSpec]) -> WorkerSpecGraph:
    """Validate worker specs without evaluating enablement predicates."""

    spec_tuple = tuple(specs)
    specs_by_name = _validate_unique_names(spec_tuple)
    _validate_diagnostic_names(spec_tuple)
    for spec in spec_tuple:
        _validate_phase(spec)
        _validate_enabled_predicate(spec)
        _validate_strategy_requirements(spec)
        _validate_failure_policy(spec)
        _validate_dependency_shape(spec)
        _validate_dependencies(spec, specs_by_name)
    _validate_acyclic(spec_tuple, specs_by_name)
    return WorkerSpecGraph(
        specs=spec_tuple,
        specs_by_name=MappingProxyType(specs_by_name),
    )


def validate_enabled_worker_dependencies(
    graph: WorkerSpecGraph,
    enabled_names: set[str],
) -> None:
    """Reject enabled specs whose validated dependencies are not also enabled."""

    for worker_name in enabled_names:
        spec = graph.specs_by_name.get(worker_name)
        if spec is None:
            raise WorkerSpecValidationError(f"Enabled worker {worker_name!r} is not in spec graph")
        disabled_dependencies = [
            dependency_name
            for dependency_name in spec.depends_on
            if dependency_name not in enabled_names
        ]
        if disabled_dependencies:
            dependencies = ", ".join(disabled_dependencies)
            raise WorkerSpecValidationError(
                f"Enabled worker {spec.name!r} depends on disabled worker(s): {dependencies}"
            )


def _validate_unique_names(specs: tuple[WorkerSpec, ...]) -> dict[str, WorkerSpec]:
    """Return specs keyed by name after rejecting duplicate worker identities."""

    specs_by_name: dict[str, WorkerSpec] = {}
    for spec in specs:
        if spec.name in specs_by_name:
            raise WorkerSpecValidationError(f"duplicate worker spec name: {spec.name}")
        specs_by_name[spec.name] = spec
    return specs_by_name


def _validate_diagnostic_names(specs: tuple[WorkerSpec, ...]) -> None:
    """Reject diagnostic names that would publish ambiguous lifecycle metadata."""

    diagnostic_names: dict[str, str] = {}
    for spec in specs:
        diagnostic_name = spec.diagnostic_name or spec.name
        owner = diagnostic_names.get(diagnostic_name)
        if owner is not None:
            raise WorkerSpecValidationError(
                f"Duplicate diagnostic name {diagnostic_name!r} "
                f"for worker specs {owner!r} and {spec.name!r}"
            )
        diagnostic_names[diagnostic_name] = spec.name


def _validate_phase(spec: WorkerSpec) -> None:
    """Require specs to use explicit shutdown phases instead of raw values."""

    if not isinstance(spec.phase, ShutdownPhase):
        raise WorkerSpecValidationError(
            f"Worker spec {spec.name!r} phase must be a ShutdownPhase value, "
            f"got {spec.phase!r}"
        )


def _validate_enabled_predicate(spec: WorkerSpec) -> None:
    """Require enablement to be an explicit predicate callable."""

    if not callable(spec.enabled):
        raise WorkerSpecValidationError(
            f"Worker spec {spec.name!r} enabled must be callable"
        )


def _validate_strategy_requirements(spec: WorkerSpec) -> None:
    """Validate the factory fields required by a worker startup strategy."""

    if not isinstance(spec.strategy, WorkerStrategy):
        raise WorkerSpecValidationError(
            f"Worker spec {spec.name!r} strategy must be a WorkerStrategy value, "
            f"got {spec.strategy!r}"
        )

    if spec.strategy is WorkerStrategy.STOP_EVENT_TASK:
        if spec.factory is None:
            raise WorkerSpecValidationError(
                f"Worker spec {spec.name!r} uses stop_event_task and requires factory"
            )
        if not callable(spec.factory):
            raise WorkerSpecValidationError(
                f"Worker spec {spec.name!r} factory must be callable"
            )
        if spec.shutdown_callback_factory is not None:
            raise WorkerSpecValidationError(
                f"Worker spec {spec.name!r} uses stop_event_task and must not define "
                "shutdown_callback_factory"
            )
    if spec.strategy is WorkerStrategy.CALLBACK_ONLY:
        if spec.factory is not None:
            raise WorkerSpecValidationError(
                f"Worker spec {spec.name!r} uses callback_only and must not define factory"
            )
        if spec.shutdown_callback_factory is None:
            raise WorkerSpecValidationError(
                f"Worker spec {spec.name!r} uses callback_only and requires "
                "shutdown_callback_factory"
            )
        if not callable(spec.shutdown_callback_factory):
            raise WorkerSpecValidationError(
                f"Worker spec {spec.name!r} shutdown_callback_factory must be callable"
            )


def _validate_failure_policy(spec: WorkerSpec) -> None:
    """Require each spec to declare a known startup failure policy."""

    if not isinstance(spec.failure_policy, WorkerFailurePolicy):
        raise WorkerSpecValidationError(
            f"Worker spec {spec.name!r} failure_policy must be a WorkerFailurePolicy "
            f"value, got {spec.failure_policy!r}"
        )


def _validate_dependency_shape(spec: WorkerSpec) -> None:
    """Require dependencies to be an immutable tuple of worker names."""

    if not isinstance(spec.depends_on, tuple):
        raise WorkerSpecValidationError(
            f"Worker spec {spec.name!r} depends_on must be a tuple of worker names"
        )


def _validate_dependencies(
    spec: WorkerSpec,
    specs_by_name: Mapping[str, WorkerSpec],
) -> None:
    """Reject dependencies that do not exist in the collected spec graph."""

    for dependency_name in spec.depends_on:
        if dependency_name not in specs_by_name:
            raise WorkerSpecValidationError(
                f"Worker spec {spec.name!r} depends on unknown worker {dependency_name!r}"
            )


def _validate_acyclic(
    specs: tuple[WorkerSpec, ...],
    specs_by_name: Mapping[str, WorkerSpec],
) -> None:
    """Reject dependency cycles so startup order and rollback remain deterministic."""

    visited: set[str] = set()
    visiting: list[str] = []

    def visit(spec: WorkerSpec) -> None:
        if spec.name in visited:
            return
        if spec.name in visiting:
            cycle_start = visiting.index(spec.name)
            cycle = [*visiting[cycle_start:], spec.name]
            raise WorkerSpecValidationError(
                f"Worker spec dependency cycle detected: {' -> '.join(cycle)}"
            )

        visiting.append(spec.name)
        for dependency_name in spec.depends_on:
            visit(specs_by_name[dependency_name])
        visiting.pop()
        visited.add(spec.name)

    for spec in specs:
        visit(spec)
