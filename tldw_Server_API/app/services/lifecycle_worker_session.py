"""Runtime lifecycle worker session state and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, cast

from loguru import logger

from tldw_Server_API.app.services.lifecycle_exceptions import LIFECYCLE_GUARD_EXCEPTIONS
from tldw_Server_API.app.services.lifecycle_worker_specs import (
    WorkerSpec,
    WorkerSpecGraph,
)
from tldw_Server_API.app.services.lifecycle_workers import (
    ManagedWorker,
    ShutdownPhase,
    publish_worker_inventory,
)

_STOPPED_NAMES_ATTR_BY_PHASE = {
    ShutdownPhase.JOB_POLLER_QUIESCE: "_tldw_shutdown_quiesced_job_poller_names",
    ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN: ("_tldw_shutdown_stopped_background_worker_names"),
    ShutdownPhase.POST_WORKER_SHUTDOWN: "_tldw_shutdown_stopped_post_worker_names",
}


@dataclass
class WorkerLifecycleSession:
    """Track declarative worker handles and shutdown diagnostics for one lifespan."""

    app: Any
    graph: WorkerSpecGraph
    handles_by_name: dict[str, ManagedWorker] = field(default_factory=dict)
    enabled_names: set[str] = field(default_factory=set)
    disabled_names: set[str] = field(default_factory=set)
    startup_failures: dict[str, str] = field(default_factory=dict)
    stopped_names_by_phase: dict[ShutdownPhase, set[str]] = field(default_factory=dict)
    stopped_or_quiesced_names: set[str] = field(default_factory=set)

    def register_handle(self, spec: WorkerSpec, handle: ManagedWorker) -> None:
        """Register a started worker handle under its spec name."""

        self.handles_by_name[spec.name] = handle
        self.enabled_names.add(spec.name)
        self.disabled_names.discard(spec.name)

    def mark_disabled(self, name: str) -> None:
        """Record that a worker spec was disabled during startup."""

        self.disabled_names.add(name)
        self.enabled_names.discard(name)

    def mark_startup_failure(self, name: str, exc: BaseException) -> None:
        """Record a worker startup failure for diagnostics."""

        self.startup_failures[name] = str(exc)

    def mark_stopped(self, name: str, phase: ShutdownPhase | str) -> None:
        """Record a worker as stopped or quiesced in a shutdown phase."""

        shutdown_phase = _normalize_shutdown_phase(phase)
        self.stopped_names_by_phase.setdefault(shutdown_phase, set()).add(name)
        self.stopped_or_quiesced_names.add(name)

    def handles_for_phase(self, phase: ShutdownPhase | str) -> list[ManagedWorker]:
        """Return unstopped handles owned by one shutdown phase."""

        shutdown_phase = _normalize_shutdown_phase(phase)
        return [
            handle
            for name, handle in self.handles_by_name.items()
            if name not in self.stopped_or_quiesced_names
            and _normalize_shutdown_phase(handle.shutdown_phase) is shutdown_phase
        ]

    def publish_inventory(self) -> None:
        """Publish full and compatibility worker inventories to app state."""

        publish_worker_inventory(self.app, self._inventory_handles())

    def publish_stopped_names(self, phase: ShutdownPhase | str) -> None:
        """Publish stopped or quiesced worker names for one shutdown phase."""

        shutdown_phase = _normalize_shutdown_phase(phase)
        attr_name = _STOPPED_NAMES_ATTR_BY_PHASE[shutdown_phase]
        stopped_names = [
            self._diagnostic_name_for(name)
            for name in sorted(self.stopped_names_by_phase.get(shutdown_phase, set()))
        ]
        try:
            setattr(self.app.state, attr_name, stopped_names)
        except LIFECYCLE_GUARD_EXCEPTIONS as exc:
            logger.debug("Lifecycle worker metadata publication skipped for {}: {}", attr_name, exc)

    def _inventory_handles(self) -> list[ManagedWorker]:
        return [
            _handle_with_spec_diagnostics(handle, self.graph.specs_by_name.get(name))
            for name, handle in self.handles_by_name.items()
        ]

    def _diagnostic_name_for(self, name: str) -> str:
        spec = self.graph.specs_by_name.get(name)
        if spec is None:
            return name
        return spec.diagnostic_name or spec.name


def _normalize_shutdown_phase(phase: ShutdownPhase | str) -> ShutdownPhase:
    if isinstance(phase, ShutdownPhase):
        return phase
    return ShutdownPhase(str(phase))


def _handle_with_spec_diagnostics(
    handle: ManagedWorker,
    spec: WorkerSpec | None,
) -> ManagedWorker:
    if spec is None:
        return handle
    task: Any = handle.task
    if task is None:
        task = _DiagnosticTaskName(spec.task_name)
    return replace(
        handle,
        name=spec.diagnostic_name or spec.name,
        task=cast(Any, task),
    )


class _DiagnosticTaskName:
    def __init__(self, task_name: str) -> None:
        self._task_name = task_name

    def get_name(self) -> str:
        return self._task_name
