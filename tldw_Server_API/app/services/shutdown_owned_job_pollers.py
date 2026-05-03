"""
Shutdown-owned job poller and timing helpers extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from typing import Any

from fastapi import FastAPI
from loguru import logger

from tldw_Server_API.app.services.lifecycle_workers import (
    ManagedWorker,
    ShutdownPhase,
    publish_worker_inventory,
)

DEFAULT_GUARD_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


ManagedJobPoller = ManagedWorker


def publish_shutdown_job_poller_inventory(
    app: FastAPI,
    handles: Sequence[ManagedJobPoller],
) -> None:
    """Expose shutdown-owned job poller metadata on app.state.

    Invalid handle shapes are programmer errors and should surface before
    inventory publication can leave stale diagnostics on app.state.
    """
    publish_worker_inventory(app, handles)


def register_owned_job_poller(
    app: FastAPI,
    handles: list[ManagedJobPoller],
    *,
    name: str,
    task: asyncio.Task[Any] | None,
    stop_event: asyncio.Event | None = None,
    timeout_sec: float = 5.0,
    publish_inventory=publish_shutdown_job_poller_inventory,
) -> None:
    """Register one shutdown-owned job poller and refresh app-state inventory."""
    if task is None:
        return
    handles.append(
        ManagedWorker(
            name=name,
            task=task,
            stop_event=stop_event,
            timeout_sec=timeout_sec,
            shutdown_phase=ShutdownPhase.JOB_POLLER_QUIESCE,
        )
    )
    publish_inventory(app, handles)


def _is_job_poller_quiesce(handle: ManagedJobPoller) -> bool:
    """Return whether a worker belongs to the job-poller quiesce phase.

    ShutdownPhase is a str enum, so equality preserves compatibility with
    legacy string phase values already stored on a handle.
    """
    return handle.shutdown_phase == ShutdownPhase.JOB_POLLER_QUIESCE


def replace_owned_job_poller_inventory(
    app: FastAPI,
    handles: list[ManagedJobPoller],
    *,
    registrations: list[tuple[str, asyncio.Task[Any] | None, asyncio.Event | None, float]],
    register_owned_job_poller_fn=register_owned_job_poller,
    publish_inventory=publish_shutdown_job_poller_inventory,
) -> None:
    """Replace the managed job-poller inventory with the current owned poller set."""
    handles[:] = [
        handle
        for handle in handles
        if not _is_job_poller_quiesce(handle)
    ]
    publish_inventory(app, handles)
    for name, task, stop_event, timeout_sec in registrations:
        register_owned_job_poller_fn(
            app,
            handles,
            name=name,
            task=task,
            stop_event=stop_event,
            timeout_sec=timeout_sec,
        )


def record_shutdown_timing_segment(
    app: FastAPI,
    segment: str,
    duration_ms: int,
    *,
    logger_obj: Any = logger,
    guard_exceptions: tuple[type[BaseException], ...] = DEFAULT_GUARD_EXCEPTIONS,
    **extra: object,
) -> None:
    """Store one shutdown timing segment and emit a consistent log line."""
    payload = {"segment": segment, "duration_ms": max(int(duration_ms), 0), **extra}
    segments = getattr(app.state, "_tldw_shutdown_timing_segments", None)
    if not isinstance(segments, list):
        segments = []
        try:
            app.state._tldw_shutdown_timing_segments = segments
        except guard_exceptions:
            return
    segments.append(payload)
    extra_text = " ".join(f"{key}={value}" for key, value in extra.items())
    if extra_text:
        logger_obj.info(
            f"App Shutdown Timing: segment={segment} duration_ms={payload['duration_ms']} {extra_text}"
        )
    else:
        logger_obj.info(f"App Shutdown Timing: segment={segment} duration_ms={payload['duration_ms']}")


@contextmanager
def timed_shutdown_segment(
    app: FastAPI,
    segment: str,
    *,
    monotonic=time.monotonic,
    record_shutdown_timing_segment=record_shutdown_timing_segment,
    **extra: object,
) -> Iterator[None]:
    """Measure a shutdown block with monotonic time and record it on app.state."""
    started = monotonic()
    try:
        yield
    finally:
        duration_ms = int((monotonic() - started) * 1000)
        record_shutdown_timing_segment(app, segment, duration_ms, **extra)


def record_shutdown_timing_total(
    app: FastAPI,
    duration_ms: int,
    *,
    record_shutdown_timing_segment=record_shutdown_timing_segment,
    logger_obj: Any = logger,
    guard_exceptions: tuple[type[BaseException], ...] = DEFAULT_GUARD_EXCEPTIONS,
) -> None:
    """Record total teardown time and summarize the slowest non-total segment."""
    segments = getattr(app.state, "_tldw_shutdown_timing_segments", [])
    non_total_segments = [
        entry
        for entry in segments
        if isinstance(entry, dict) and entry.get("segment") != "total app teardown"
    ]
    if non_total_segments:
        slowest = max(non_total_segments, key=lambda entry: int(entry.get("duration_ms", 0)))
        slowest_segment = str(slowest.get("segment", ""))
        slowest_duration_ms = int(slowest.get("duration_ms", 0))
    else:
        slowest_segment = "total app teardown"
        slowest_duration_ms = max(int(duration_ms), 0)
    record_shutdown_timing_segment(app, "total app teardown", duration_ms)
    summary = {
        "duration_ms": max(int(duration_ms), 0),
        "slowest_segment": slowest_segment,
        "slowest_duration_ms": slowest_duration_ms,
    }
    try:
        app.state._tldw_shutdown_timing_total = summary
    except guard_exceptions:
        pass
    logger_obj.info(
        "App Shutdown Timing: total duration_ms={} slowest_segment={} slowest_duration_ms={}",
        summary["duration_ms"],
        summary["slowest_segment"],
        summary["slowest_duration_ms"],
    )


async def stop_registered_job_pollers(
    app: FastAPI,
    handles: list[ManagedJobPoller],
    *,
    logger_obj: Any = logger,
    guard_exceptions: tuple[type[BaseException], ...] = DEFAULT_GUARD_EXCEPTIONS,
    asyncio_module=asyncio,
) -> None:
    """Stop registered job pollers, preferring explicit stop events."""

    async def _await_job_poller_shutdown(handle: ManagedJobPoller) -> bool:
        task = handle.task
        if task is None:
            return True
        try:
            await asyncio_module.wait_for(asyncio_module.shield(task), timeout=handle.timeout_sec)
        except asyncio.CancelledError:
            return bool(task.done())
        except asyncio.TimeoutError:
            logger_obj.warning(
                "App Shutdown: Timed out waiting for job poller {} after {}s; cancelling",
                handle.name,
                handle.timeout_sec,
            )
            task.cancel()
            try:
                await asyncio_module.wait_for(task, timeout=1.0)
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                logger_obj.warning(
                    "App Shutdown: Job poller {} did not cancel within 1.0s after timeout",
                    handle.name,
                )
            except Exception as exc:  # noqa: BLE001 - worker failures must not block shutdown.
                logger_obj.warning(
                    "App Shutdown: Job poller {} raised after cancellation: {}",
                    handle.name,
                    exc,
                )
        except guard_exceptions as exc:
            logger_obj.debug(
                "App Shutdown: Job poller stop guard triggered for {}: {}",
                handle.name,
                exc,
            )
        except Exception as exc:  # noqa: BLE001 - worker failures must not block shutdown.
            logger_obj.warning(
                "App Shutdown: Job poller {} exited during shutdown: {}",
                handle.name,
                exc,
            )
        return bool(task.done())

    for handle in handles:
        if handle.stop_event is not None:
            handle.stop_event.set()
        elif handle.task is not None:
            with suppress(*guard_exceptions):
                handle.task.cancel()

    quiesce_results = await asyncio_module.gather(
        *(_await_job_poller_shutdown(handle) for handle in handles),
        return_exceptions=False,
    )
    try:
        app.state._tldw_shutdown_quiesced_job_poller_names = [
            handle.name
            for handle, quiesced in zip(handles, quiesce_results)
            if quiesced
        ]
    except guard_exceptions:
        pass


async def quiesce_owned_job_pollers_for_shutdown(
    app: FastAPI,
    handles: list[ManagedJobPoller],
    *,
    wait_for_leases_sec: int | float,
    count_active_processing: Any,
    stop_registered_job_pollers=stop_registered_job_pollers,
    record_shutdown_timing_segment=record_shutdown_timing_segment,
    timed_shutdown_segment=timed_shutdown_segment,
    guard_exceptions: tuple[type[BaseException], ...] = DEFAULT_GUARD_EXCEPTIONS,
    monotonic=time.monotonic,
    asyncio_module=asyncio,
) -> None:
    """Optionally wait for active leases, then quiesce owned job pollers."""
    lease_wait_started = monotonic()
    initial_active = 0
    if wait_for_leases_sec > 0 and handles:
        try:
            initial_active = max(int(count_active_processing()), 0)
        except guard_exceptions:
            initial_active = 0
        if initial_active > 0:
            deadline = monotonic() + float(wait_for_leases_sec)
            active = initial_active
            while active > 0:
                remaining_sec = deadline - monotonic()
                if remaining_sec <= 0:
                    break
                await asyncio_module.sleep(min(0.5, remaining_sec))
                try:
                    active = max(int(count_active_processing()), 0)
                except guard_exceptions:
                    active = 0
            duration_ms = int((monotonic() - lease_wait_started) * 1000)
            record_shutdown_timing_segment(
                app,
                "optional_lease_wait",
                duration_ms,
                skipped=False,
                initial_active=initial_active,
                wait_for_leases_sec=float(wait_for_leases_sec),
            )
        else:
            record_shutdown_timing_segment(
                app,
                "optional_lease_wait",
                0,
                skipped=True,
                initial_active=0,
                wait_for_leases_sec=float(wait_for_leases_sec),
            )
    else:
        record_shutdown_timing_segment(
            app,
            "optional_lease_wait",
            0,
            skipped=True,
            initial_active=0,
            wait_for_leases_sec=float(wait_for_leases_sec),
        )

    with timed_shutdown_segment(app, "job_poller_quiesce", poller_count=len(handles)):
        await stop_registered_job_pollers(app, handles)
