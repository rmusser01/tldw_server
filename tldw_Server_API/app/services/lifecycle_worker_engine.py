"""Declarative lifecycle worker startup and shutdown orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from loguru import logger

from tldw_Server_API.app.services.lifecycle_worker_session import WorkerLifecycleSession
from tldw_Server_API.app.services.lifecycle_worker_specs import (
    WorkerFailurePolicy,
    WorkerLifecycleContext,
    WorkerSpec,
    WorkerSpecGraph,
    WorkerStrategy,
    validate_enabled_worker_dependencies,
    validate_worker_spec_graph,
)
from tldw_Server_API.app.services.lifecycle_workers import ManagedWorker, ShutdownPhase


class LifecycleWorkerEngine:
    """Start and stop declarative lifecycle workers."""

    async def start(
        self,
        context: WorkerLifecycleContext,
        specs: Sequence[WorkerSpec],
    ) -> WorkerLifecycleSession:
        """Validate, enable, start, register, and publish lifecycle workers."""

        graph = validate_worker_spec_graph(specs)
        session = WorkerLifecycleSession(context.app, graph)
        enabled_names = self._evaluate_enabled(context, session, graph)
        validate_enabled_worker_dependencies(graph, enabled_names)
        try:
            for spec in self._startup_order(graph, enabled_names):
                await self._start_one(context, session, spec)
        except Exception:
            await self._cleanup_started_handles(session)
            raise
        session.publish_inventory()
        return session

    async def stop_phase(
        self,
        session: WorkerLifecycleSession,
        phase: ShutdownPhase | str,
    ) -> None:
        """Stop unstopped workers in one shutdown phase by reverse dependency order."""

        shutdown_phase = _normalize_shutdown_phase(phase)
        for batch in self._reverse_dependency_batches(session, shutdown_phase):
            await asyncio.gather(
                *(self._stop_one(session, shutdown_phase, handle) for handle in batch)
            )
        session.publish_stopped_names(shutdown_phase)
        session.publish_inventory()

    def _evaluate_enabled(
        self,
        context: WorkerLifecycleContext,
        session: WorkerLifecycleSession,
        graph: WorkerSpecGraph,
    ) -> set[str]:
        enabled_names: set[str] = set()
        for spec in graph.specs:
            try:
                enabled = spec.enabled(context)
            except Exception as exc:
                session.mark_startup_failure(spec.name, exc)
                if spec.failure_policy is WorkerFailurePolicy.ABORT:
                    raise
                continue
            if enabled:
                enabled_names.add(spec.name)
            else:
                session.mark_disabled(spec.name)
        return enabled_names

    def _startup_order(
        self,
        graph: WorkerSpecGraph,
        enabled_names: set[str],
    ) -> list[WorkerSpec]:
        ordered: list[WorkerSpec] = []
        visited: set[str] = set()

        def visit(spec: WorkerSpec) -> None:
            if spec.name in visited:
                return
            for dependency_name in spec.depends_on:
                if dependency_name in enabled_names:
                    visit(graph.specs_by_name[dependency_name])
            visited.add(spec.name)
            ordered.append(spec)

        for spec in graph.specs:
            if spec.name in enabled_names:
                visit(spec)
        return ordered

    async def _start_one(
        self,
        context: WorkerLifecycleContext,
        session: WorkerLifecycleSession,
        spec: WorkerSpec,
    ) -> None:
        try:
            self._validate_started_dependencies(session, spec)
            handle = await self._create_handle(context, spec)
        except Exception as exc:
            session.mark_startup_failure(spec.name, exc)
            if spec.failure_policy is WorkerFailurePolicy.ABORT:
                raise
            logger.warning("Lifecycle worker {} startup skipped after failure: {}", spec.name, exc)
            return
        session.register_handle(spec, handle)

    def _validate_started_dependencies(
        self,
        session: WorkerLifecycleSession,
        spec: WorkerSpec,
    ) -> None:
        for dependency_name in spec.depends_on:
            if dependency_name not in session.handles_by_name:
                raise RuntimeError(
                    f"Worker {spec.name!r} dependency {dependency_name!r} did not start"
                )

    async def _create_handle(
        self,
        context: WorkerLifecycleContext,
        spec: WorkerSpec,
    ) -> ManagedWorker:
        if spec.strategy is WorkerStrategy.STOP_EVENT_TASK:
            if spec.factory is None:
                raise RuntimeError(f"Worker spec {spec.name!r} has no stop-event factory")
            stop_event = asyncio.Event()
            task = asyncio.create_task(
                spec.factory(context, stop_event),
                name=spec.task_name,
            )
            await asyncio.sleep(0)
            if task.done():
                if task.cancelled():
                    raise RuntimeError(f"Worker {spec.name!r} startup task was cancelled")
                task_exception = task.exception()
                if task_exception is not None:
                    raise task_exception
            return ManagedWorker(
                name=spec.name,
                task=task,
                stop_event=stop_event,
                timeout_sec=spec.timeout_sec,
                category=spec.category,
                shutdown_phase=spec.phase,
            )

        if spec.shutdown_callback_factory is None:
            raise RuntimeError(f"Worker spec {spec.name!r} has no shutdown callback factory")
        return ManagedWorker(
            name=spec.name,
            task=None,
            stop_event=None,
            shutdown_callback=spec.shutdown_callback_factory(context),
            timeout_sec=spec.timeout_sec,
            category=spec.category,
            shutdown_phase=spec.phase,
        )

    def _reverse_dependency_batches(
        self,
        session: WorkerLifecycleSession,
        phase: ShutdownPhase,
    ) -> list[list[ManagedWorker]]:
        active_names = {
            name
            for name, handle in session.handles_by_name.items()
            if name not in session.stopped_or_quiesced_names
            and _normalize_shutdown_phase(handle.shutdown_phase) is phase
        }
        return self._reverse_dependency_batches_for_names(session, active_names)

    def _reverse_dependency_batches_for_names(
        self,
        session: WorkerLifecycleSession,
        active_names: set[str],
    ) -> list[list[ManagedWorker]]:
        remaining = set(active_names)
        batches: list[list[ManagedWorker]] = []

        while remaining:
            batch_names = [
                spec.name
                for spec in session.graph.specs
                if spec.name in remaining
                and not self._has_remaining_dependent(spec.name, remaining, session.graph)
            ]
            if not batch_names:
                batch_names = [
                    spec.name for spec in session.graph.specs if spec.name in remaining
                ]
            batches.append([session.handles_by_name[name] for name in batch_names])
            remaining.difference_update(batch_names)

        return batches

    async def _cleanup_started_handles(self, session: WorkerLifecycleSession) -> None:
        active_names = set(session.handles_by_name) - session.stopped_or_quiesced_names
        for batch in self._reverse_dependency_batches_for_names(session, active_names):
            try:
                await asyncio.gather(
                    *(
                        self._stop_one(
                            session,
                            _normalize_shutdown_phase(handle.shutdown_phase),
                            handle,
                        )
                        for handle in batch
                    )
                )
            except Exception as exc:  # noqa: BLE001 - cleanup must not hide startup aborts.
                logger.warning(
                    "Lifecycle worker startup cleanup failed while stopping registered "
                    "workers: {}",
                    exc,
                )

    def _has_remaining_dependent(
        self,
        name: str,
        remaining: set[str],
        graph: WorkerSpecGraph,
    ) -> bool:
        return any(
            spec.name in remaining and name in spec.depends_on
            for spec in graph.specs
        )

    async def _stop_one(
        self,
        session: WorkerLifecycleSession,
        phase: ShutdownPhase,
        handle: ManagedWorker,
    ) -> None:
        stopped = await self._request_and_await_stop(handle)
        if stopped:
            session.mark_stopped(handle.name, phase)

    async def _request_and_await_stop(self, handle: ManagedWorker) -> bool:
        if handle.stop_event is not None:
            handle.stop_event.set()

        if handle.shutdown_callback is not None:
            return await self._run_shutdown_callback(handle)

        if handle.task is None:
            return handle.stop_event is not None

        return await self._await_task_shutdown(handle)

    async def _run_shutdown_callback(self, handle: ManagedWorker) -> bool:
        if handle.shutdown_callback is None:
            return True
        try:
            await asyncio.wait_for(handle.shutdown_callback(), timeout=handle.timeout_sec)
            return True
        except asyncio.TimeoutError:
            logger.warning(
                "App Shutdown: Timed out waiting for lifecycle worker {} shutdown callback "
                "after {}s",
                handle.name,
                handle.timeout_sec,
            )
        except asyncio.CancelledError:
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                raise
            logger.warning(
                "App Shutdown: Lifecycle worker {} shutdown callback was cancelled",
                handle.name,
            )
        except Exception as exc:  # noqa: BLE001 - shutdown hooks must not block teardown.
            logger.warning(
                "App Shutdown: Lifecycle worker {} shutdown callback failed: {}",
                handle.name,
                exc,
            )
        return False

    async def _await_task_shutdown(self, handle: ManagedWorker) -> bool:
        if handle.task is None:
            return True
        try:
            await asyncio.wait_for(asyncio.shield(handle.task), timeout=handle.timeout_sec)
        except asyncio.CancelledError:
            return bool(handle.task.done())
        except asyncio.TimeoutError:
            logger.warning(
                "App Shutdown: Timed out waiting for lifecycle worker {} after {}s; "
                "cancelling",
                handle.name,
                handle.timeout_sec,
            )
            await self._cancel_task(handle)
        except Exception as exc:  # noqa: BLE001 - worker failures must not block shutdown.
            logger.warning(
                "App Shutdown: Lifecycle worker {} exited during shutdown: {}",
                handle.name,
                exc,
            )
        return bool(handle.task.done())

    async def _cancel_task(self, handle: ManagedWorker) -> None:
        if handle.task is None:
            return
        try:
            handle.task.cancel()
        except Exception as exc:  # noqa: BLE001 - cancel hooks can raise arbitrary errors.
            logger.warning(
                "App Shutdown: Lifecycle worker {} cancel request failed: {}",
                handle.name,
                exc,
            )
            return
        try:
            await asyncio.wait_for(handle.task, timeout=1.0)
        except asyncio.CancelledError:
            pass
        except asyncio.TimeoutError:
            logger.warning(
                "App Shutdown: Lifecycle worker {} did not cancel within 1.0s after timeout",
                handle.name,
            )
        except Exception as exc:  # noqa: BLE001 - worker failures must not block shutdown.
            logger.warning(
                "App Shutdown: Lifecycle worker {} raised after cancellation: {}",
                handle.name,
                exc,
            )


def _normalize_shutdown_phase(phase: ShutdownPhase | str) -> ShutdownPhase:
    if isinstance(phase, ShutdownPhase):
        return phase
    return ShutdownPhase(str(phase))
