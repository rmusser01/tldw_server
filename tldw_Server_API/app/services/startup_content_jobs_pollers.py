"""
Content-oriented jobs poller startup helpers extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from tldw_Server_API.app.services.lifecycle_worker_specs import (
    ShutdownPhase,
    WorkerLifecycleContext,
    WorkerSpec,
    route_enabled_predicate,
    stop_event_worker_spec,
)
from tldw_Server_API.app.services.lifecycle_workers import WorkerRegistry

_STARTUP_GUARD_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass
class ContentJobsPollerHandles:
    """Startup-owned content jobs poller handles used later in shutdown flow."""

    audio_jobs_stop_event: Any | None = None
    audio_jobs_task: Any | None = None
    audiobook_jobs_stop_event: Any | None = None
    audiobook_jobs_task: Any | None = None
    presentation_render_jobs_stop_event: Any | None = None
    presentation_render_jobs_task: Any | None = None
    media_ingest_jobs_stop_event: Any | None = None
    media_ingest_jobs_task: Any | None = None
    media_ingest_heavy_jobs_stop_event: Any | None = None
    media_ingest_heavy_jobs_task: Any | None = None
    reading_digest_jobs_stop_event: Any | None = None
    reading_digest_jobs_task: Any | None = None
    llamacpp_acquisition_jobs_stop_event: Any | None = None
    llamacpp_acquisition_jobs_task: Any | None = None
    vn_asset_jobs_stop_event: Any | None = None
    vn_asset_jobs_task: Any | None = None
    vn_asset_generation_jobs_stop_event: Any | None = None
    vn_asset_generation_jobs_task: Any | None = None
    companion_reflection_jobs_stop_event: Any | None = None
    companion_reflection_jobs_task: Any | None = None


def provide_content_jobs_worker_specs(
    _context: WorkerLifecycleContext | None = None,
) -> tuple[WorkerSpec, ...]:
    """Return declarative specs for content-oriented jobs pollers."""

    return (
        stop_event_worker_spec(
            name="audio_jobs_task",
            worker_service=_run_audio_jobs_worker_service,
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            enabled=route_enabled_predicate("AUDIO_JOBS_WORKER_ENABLED", "audio-jobs"),
        ),
        stop_event_worker_spec(
            name="audiobook_jobs_task",
            worker_service=_run_audiobook_jobs_worker_service,
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            enabled=route_enabled_predicate(
                "AUDIOBOOK_JOBS_WORKER_ENABLED",
                "audiobooks",
            ),
        ),
        stop_event_worker_spec(
            name="presentation_render_jobs_task",
            worker_service=_run_presentation_render_jobs_worker_service,
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            enabled=route_enabled_predicate(
                "PRESENTATION_RENDER_JOBS_WORKER_ENABLED",
                "slides",
            ),
        ),
        stop_event_worker_spec(
            name="media_ingest_jobs_task",
            worker_service=_run_media_ingest_jobs_worker_service,
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            enabled=route_enabled_predicate(
                "MEDIA_INGEST_JOBS_WORKER_ENABLED",
                "media",
            ),
        ),
        stop_event_worker_spec(
            name="media_ingest_heavy_jobs_task",
            worker_service=_run_media_ingest_heavy_jobs_worker_service,
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            enabled=route_enabled_predicate(
                "MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED",
                "media-ingest-heavy-jobs",
                default_stable=False,
            ),
        ),
        stop_event_worker_spec(
            name="reading_digest_jobs_task",
            worker_service=_run_reading_digest_jobs_worker_service,
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            enabled=route_enabled_predicate(
                "READING_DIGEST_JOBS_WORKER_ENABLED",
                "reading",
            ),
        ),
        stop_event_worker_spec(
            name="vn_asset_jobs_task",
            worker_service=_run_vn_asset_jobs_worker_service,
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            enabled=route_enabled_predicate(
                "VN_ASSET_JOBS_WORKER_ENABLED",
                "vn-assets",
                default_stable=True,
            ),
        ),
        stop_event_worker_spec(
            name="vn_asset_generation_jobs_task",
            worker_service=_run_vn_asset_generation_jobs_worker_service,
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            enabled=route_enabled_predicate(
                "VN_ASSET_GENERATION_JOBS_WORKER_ENABLED",
                "vn-assets-generation",
                default_stable=True,
            ),
        ),
        stop_event_worker_spec(
            name="companion_reflection_jobs_task",
            worker_service=_run_companion_reflection_jobs_worker_service,
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            enabled=route_enabled_predicate(
                "COMPANION_REFLECTION_JOBS_WORKER_ENABLED",
                "companion",
            ),
        ),
    )


async def start_content_jobs_pollers(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    should_start_worker: Callable[..., bool],
    worker_inventory: WorkerRegistry | None = None,
) -> ContentJobsPollerHandles:
    """Start content jobs pollers and return their handles."""

    audio_jobs_stop_event, audio_jobs_task = await _start_audio_jobs_worker(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        should_start_worker=should_start_worker,
        worker_inventory=worker_inventory,
    )
    audiobook_jobs_stop_event, audiobook_jobs_task = await _start_audiobook_jobs_worker(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        should_start_worker=should_start_worker,
        worker_inventory=worker_inventory,
    )
    presentation_render_jobs_stop_event, presentation_render_jobs_task = (
        await _start_presentation_render_jobs_worker(
            app=app,
            owned_job_pollers=owned_job_pollers,
            register_owned_job_poller=register_owned_job_poller,
            should_start_worker=should_start_worker,
            worker_inventory=worker_inventory,
        )
    )
    (
        media_ingest_jobs_stop_event,
        media_ingest_jobs_task,
        media_ingest_heavy_jobs_stop_event,
        media_ingest_heavy_jobs_task,
    ) = await _start_media_ingest_jobs_workers(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        should_start_worker=should_start_worker,
        worker_inventory=worker_inventory,
    )
    reading_digest_jobs_stop_event, reading_digest_jobs_task = await _start_reading_digest_jobs_worker(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        should_start_worker=should_start_worker,
        worker_inventory=worker_inventory,
    )
    llamacpp_acquisition_jobs_stop_event, llamacpp_acquisition_jobs_task = (
        await _start_llamacpp_acquisition_jobs_worker(
            app=app,
            owned_job_pollers=owned_job_pollers,
            register_owned_job_poller=register_owned_job_poller,
            should_start_worker=should_start_worker,
            worker_inventory=worker_inventory,
        )
    )
    (
        vn_asset_jobs_stop_event,
        vn_asset_jobs_task,
        vn_asset_generation_jobs_stop_event,
        vn_asset_generation_jobs_task,
    ) = await _start_vn_asset_jobs_workers(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        should_start_worker=should_start_worker,
        worker_inventory=worker_inventory,
    )
    companion_reflection_jobs_stop_event, companion_reflection_jobs_task = (
        await _start_companion_reflection_jobs_worker(
            app=app,
            owned_job_pollers=owned_job_pollers,
            register_owned_job_poller=register_owned_job_poller,
            should_start_worker=should_start_worker,
            worker_inventory=worker_inventory,
        )
    )
    return ContentJobsPollerHandles(
        audio_jobs_stop_event=audio_jobs_stop_event,
        audio_jobs_task=audio_jobs_task,
        audiobook_jobs_stop_event=audiobook_jobs_stop_event,
        audiobook_jobs_task=audiobook_jobs_task,
        presentation_render_jobs_stop_event=presentation_render_jobs_stop_event,
        presentation_render_jobs_task=presentation_render_jobs_task,
        media_ingest_jobs_stop_event=media_ingest_jobs_stop_event,
        media_ingest_jobs_task=media_ingest_jobs_task,
        media_ingest_heavy_jobs_stop_event=media_ingest_heavy_jobs_stop_event,
        media_ingest_heavy_jobs_task=media_ingest_heavy_jobs_task,
        reading_digest_jobs_stop_event=reading_digest_jobs_stop_event,
        reading_digest_jobs_task=reading_digest_jobs_task,
        llamacpp_acquisition_jobs_stop_event=llamacpp_acquisition_jobs_stop_event,
        llamacpp_acquisition_jobs_task=llamacpp_acquisition_jobs_task,
        vn_asset_jobs_stop_event=vn_asset_jobs_stop_event,
        vn_asset_jobs_task=vn_asset_jobs_task,
        vn_asset_generation_jobs_stop_event=vn_asset_generation_jobs_stop_event,
        vn_asset_generation_jobs_task=vn_asset_generation_jobs_task,
        companion_reflection_jobs_stop_event=companion_reflection_jobs_stop_event,
        companion_reflection_jobs_task=companion_reflection_jobs_task,
    )


def _make_event() -> Any:
    return asyncio.Event()


def _create_task(awaitable: Any) -> Any:
    return asyncio.create_task(awaitable)


async def _register_jobs_worker_with_inventory(
    worker_inventory: WorkerRegistry,
    *,
    name: str,
    coroutine_factory: Callable[[Any], Any],
) -> tuple[Any, Any]:
    """Register a content jobs poller with the shared lifecycle inventory."""

    task, stop_event = await worker_inventory.register_custom(
        name=name,
        task_name=name,
        coroutine_factory=coroutine_factory,
        timeout_sec=5.0,
        category="jobs",
        shutdown_phase=ShutdownPhase.JOB_POLLER_QUIESCE,
    )
    return stop_event, task


async def _start_audio_jobs_worker(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    should_start_worker: Callable[..., bool],
    worker_inventory: WorkerRegistry | None = None,
) -> tuple[Any | None, Any | None]:
    """Start the Audio jobs poller and return its shutdown handles."""

    try:
        enabled = should_start_worker("AUDIO_JOBS_WORKER_ENABLED", "audio-jobs")
        if not enabled:
            logger.info("Audio Jobs worker disabled by flag (AUDIO_JOBS_WORKER_ENABLED)")
            return None, None

        if worker_inventory is not None:
            stop_event, task = await _register_jobs_worker_with_inventory(
                worker_inventory,
                name="audio_jobs_task",
                coroutine_factory=_run_audio_jobs_worker_service,
            )
            logger.info("Audio Jobs worker started with explicit stop_event signal")
            return stop_event, task

        stop_event = _make_event()
        task = _create_task(_run_audio_jobs_worker_service(stop_event))
        logger.info("Audio Jobs worker started with explicit stop_event signal")
        register_owned_job_poller(
            app,
            owned_job_pollers,
            name="audio_jobs_task",
            task=task,
            stop_event=stop_event,
        )
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Audio Jobs worker: {exc}")
        return None, None


async def _start_audiobook_jobs_worker(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    should_start_worker: Callable[..., bool],
    worker_inventory: WorkerRegistry | None = None,
) -> tuple[Any | None, Any | None]:
    """Start the Audiobook jobs poller and return its shutdown handles."""

    try:
        enabled = should_start_worker("AUDIOBOOK_JOBS_WORKER_ENABLED", "audiobooks")
        if not enabled:
            logger.info("Audiobook Jobs worker disabled by flag (AUDIOBOOK_JOBS_WORKER_ENABLED)")
            return None, None

        if worker_inventory is not None:
            stop_event, task = await _register_jobs_worker_with_inventory(
                worker_inventory,
                name="audiobook_jobs_task",
                coroutine_factory=_run_audiobook_jobs_worker_service,
            )
            logger.info("Audiobook Jobs worker started with explicit stop_event signal")
            return stop_event, task

        stop_event = _make_event()
        task = _create_task(_run_audiobook_jobs_worker_service(stop_event))
        logger.info("Audiobook Jobs worker started with explicit stop_event signal")
        register_owned_job_poller(
            app,
            owned_job_pollers,
            name="audiobook_jobs_task",
            task=task,
            stop_event=stop_event,
        )
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Audiobook Jobs worker: {exc}")
        return None, None


async def _start_presentation_render_jobs_worker(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    should_start_worker: Callable[..., bool],
    worker_inventory: WorkerRegistry | None = None,
) -> tuple[Any | None, Any | None]:
    """Start the presentation-render jobs poller and return its shutdown handles."""

    try:
        enabled = should_start_worker("PRESENTATION_RENDER_JOBS_WORKER_ENABLED", "slides")
        if not enabled:
            logger.info(
                "Presentation Render Jobs worker disabled by flag (PRESENTATION_RENDER_JOBS_WORKER_ENABLED)"
            )
            return None, None

        if worker_inventory is not None:
            stop_event, task = await _register_jobs_worker_with_inventory(
                worker_inventory,
                name="presentation_render_jobs_task",
                coroutine_factory=_run_presentation_render_jobs_worker_service,
            )
            logger.info("Presentation Render Jobs worker started with explicit stop_event signal")
            return stop_event, task

        stop_event = _make_event()
        task = _create_task(_run_presentation_render_jobs_worker_service(stop_event))
        logger.info("Presentation Render Jobs worker started with explicit stop_event signal")
        register_owned_job_poller(
            app,
            owned_job_pollers,
            name="presentation_render_jobs_task",
            task=task,
            stop_event=stop_event,
        )
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Presentation Render Jobs worker: {exc}")
        return None, None


async def _start_media_ingest_jobs_workers(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    should_start_worker: Callable[..., bool],
    worker_inventory: WorkerRegistry | None = None,
) -> tuple[Any | None, Any | None, Any | None, Any | None]:
    """Start media ingest jobs pollers and return their shutdown handles."""

    media_ingest_jobs_stop_event = None
    media_ingest_jobs_task = None
    media_ingest_heavy_jobs_stop_event = None
    media_ingest_heavy_jobs_task = None

    try:
        enabled = should_start_worker("MEDIA_INGEST_JOBS_WORKER_ENABLED", "media")
        if enabled:
            if worker_inventory is not None:
                media_ingest_jobs_stop_event, media_ingest_jobs_task = await _register_jobs_worker_with_inventory(
                    worker_inventory,
                    name="media_ingest_jobs_task",
                    coroutine_factory=_run_media_ingest_jobs_worker_service,
                )
            else:
                media_ingest_jobs_stop_event = _make_event()
                media_ingest_jobs_task = _create_task(
                    _run_media_ingest_jobs_worker_service(media_ingest_jobs_stop_event)
                )
            logger.info("Media Ingest Jobs worker started with explicit stop_event signal")
            if worker_inventory is None:
                register_owned_job_poller(
                    app,
                    owned_job_pollers,
                    name="media_ingest_jobs_task",
                    task=media_ingest_jobs_task,
                    stop_event=media_ingest_jobs_stop_event,
                )
        else:
            logger.info("Media Ingest Jobs worker disabled by flag (MEDIA_INGEST_JOBS_WORKER_ENABLED)")
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        _safe_cancel_task(media_ingest_jobs_task)
        logger.warning(f"Failed to start Media Ingest Jobs worker: {exc}")
        return None, None, None, None

    try:
        heavy_enabled = should_start_worker(
            "MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED",
            "media-ingest-heavy-jobs",
            default_stable=False,
        )
        if heavy_enabled:
            if worker_inventory is not None:
                media_ingest_heavy_jobs_stop_event, media_ingest_heavy_jobs_task = (
                    await _register_jobs_worker_with_inventory(
                        worker_inventory,
                        name="media_ingest_heavy_jobs_task",
                        coroutine_factory=_run_media_ingest_heavy_jobs_worker_service,
                    )
                )
            else:
                media_ingest_heavy_jobs_stop_event = _make_event()
                media_ingest_heavy_jobs_task = _create_task(
                    _run_media_ingest_heavy_jobs_worker_service(media_ingest_heavy_jobs_stop_event)
                )
            logger.info("Media Ingest Heavy Jobs worker started with explicit stop_event signal")
            if worker_inventory is None:
                register_owned_job_poller(
                    app,
                    owned_job_pollers,
                    name="media_ingest_heavy_jobs_task",
                    task=media_ingest_heavy_jobs_task,
                    stop_event=media_ingest_heavy_jobs_stop_event,
                )
        else:
            logger.info(
                "Media Ingest Heavy Jobs worker disabled by flag (MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED)"
            )
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        _safe_cancel_task(media_ingest_heavy_jobs_task)
        logger.warning(f"Failed to start Media Ingest Heavy Jobs worker: {exc}")
        media_ingest_heavy_jobs_stop_event = None
        media_ingest_heavy_jobs_task = None

    return (
        media_ingest_jobs_stop_event,
        media_ingest_jobs_task,
        media_ingest_heavy_jobs_stop_event,
        media_ingest_heavy_jobs_task,
    )


def _safe_cancel_task(task: Any | None) -> None:
    if task is None:
        return
    try:
        task.cancel()
    except _STARTUP_GUARD_EXCEPTIONS:
        pass


async def _start_reading_digest_jobs_worker(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    should_start_worker: Callable[..., bool],
    worker_inventory: WorkerRegistry | None = None,
) -> tuple[Any | None, Any | None]:
    """Start the reading digest jobs poller and return its shutdown handles."""

    try:
        enabled = should_start_worker("READING_DIGEST_JOBS_WORKER_ENABLED", "reading")
        if not enabled:
            logger.info("Reading digest Jobs worker disabled by flag (READING_DIGEST_JOBS_WORKER_ENABLED)")
            return None, None

        if worker_inventory is not None:
            stop_event, task = await _register_jobs_worker_with_inventory(
                worker_inventory,
                name="reading_digest_jobs_task",
                coroutine_factory=_run_reading_digest_jobs_worker_service,
            )
            logger.info("Reading digest Jobs worker started with explicit stop_event signal")
            return stop_event, task

        stop_event = _make_event()
        task = _create_task(_run_reading_digest_jobs_worker_service(stop_event))
        logger.info("Reading digest Jobs worker started with explicit stop_event signal")
        register_owned_job_poller(
            app,
            owned_job_pollers,
            name="reading_digest_jobs_task",
            task=task,
            stop_event=stop_event,
        )
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Reading digest Jobs worker: {exc}")
        return None, None


async def _start_llamacpp_acquisition_jobs_worker(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    should_start_worker: Callable[..., bool],
    worker_inventory: WorkerRegistry | None = None,
) -> tuple[Any | None, Any | None]:
    """Start the llama.cpp acquisition jobs poller and return its shutdown handles."""

    task = None
    try:
        enabled = should_start_worker(
            "LLAMACPP_ACQUISITION_JOBS_WORKER_ENABLED",
            "llamacpp-acquisition",
        )
        if not enabled:
            logger.info(
                "llama.cpp Acquisition Jobs worker disabled by flag "
                "(LLAMACPP_ACQUISITION_JOBS_WORKER_ENABLED)"
            )
            return None, None

        if worker_inventory is not None:
            stop_event, task = await _register_jobs_worker_with_inventory(
                worker_inventory,
                name="llamacpp_acquisition_jobs_task",
                coroutine_factory=_run_llamacpp_acquisition_jobs_worker_service,
            )
            logger.info("llama.cpp Acquisition Jobs worker started with explicit stop_event signal")
            return stop_event, task

        stop_event = _make_event()
        task = _create_task(_run_llamacpp_acquisition_jobs_worker_service(stop_event))
        logger.info("llama.cpp Acquisition Jobs worker started with explicit stop_event signal")
        register_owned_job_poller(
            app,
            owned_job_pollers,
            name="llamacpp_acquisition_jobs_task",
            task=task,
            stop_event=stop_event,
        )
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        _safe_cancel_task(task)
        logger.warning(f"Failed to start llama.cpp Acquisition Jobs worker: {exc}")
        return None, None


async def _start_vn_asset_jobs_workers(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    should_start_worker: Callable[..., bool],
    worker_inventory: WorkerRegistry | None = None,
) -> tuple[Any | None, Any | None, Any | None, Any | None]:
    """Start VN asset jobs pollers and return their shutdown handles."""

    vn_asset_jobs_stop_event = None
    vn_asset_jobs_task = None
    vn_asset_generation_jobs_stop_event = None
    vn_asset_generation_jobs_task = None

    try:
        enabled = should_start_worker(
            "VN_ASSET_JOBS_WORKER_ENABLED",
            "vn-assets",
            default_stable=True,
        )
        if enabled:
            if worker_inventory is not None:
                vn_asset_jobs_stop_event, vn_asset_jobs_task = await _register_jobs_worker_with_inventory(
                    worker_inventory,
                    name="vn_asset_jobs_task",
                    coroutine_factory=_run_vn_asset_jobs_worker_service,
                )
            else:
                vn_asset_jobs_stop_event = _make_event()
                vn_asset_jobs_task = _create_task(
                    _run_vn_asset_jobs_worker_service(vn_asset_jobs_stop_event)
                )
            logger.info("VN asset Jobs worker started with explicit stop_event signal")
            if worker_inventory is None:
                register_owned_job_poller(
                    app,
                    owned_job_pollers,
                    name="vn_asset_jobs_task",
                    task=vn_asset_jobs_task,
                    stop_event=vn_asset_jobs_stop_event,
                )
        else:
            logger.info("VN asset Jobs worker disabled by flag (VN_ASSET_JOBS_WORKER_ENABLED)")
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        _safe_cancel_task(vn_asset_jobs_task)
        logger.warning(f"Failed to start VN asset Jobs worker: {exc}")
        return None, None, None, None

    try:
        generation_enabled = should_start_worker(
            "VN_ASSET_GENERATION_JOBS_WORKER_ENABLED",
            "vn-assets-generation",
            default_stable=True,
        )
        if generation_enabled:
            if worker_inventory is not None:
                vn_asset_generation_jobs_stop_event, vn_asset_generation_jobs_task = (
                    await _register_jobs_worker_with_inventory(
                        worker_inventory,
                        name="vn_asset_generation_jobs_task",
                        coroutine_factory=_run_vn_asset_generation_jobs_worker_service,
                    )
                )
            else:
                vn_asset_generation_jobs_stop_event = _make_event()
                vn_asset_generation_jobs_task = _create_task(
                    _run_vn_asset_generation_jobs_worker_service(
                        vn_asset_generation_jobs_stop_event
                    )
                )
            logger.info("VN asset generation Jobs worker started with explicit stop_event signal")
            if worker_inventory is None:
                register_owned_job_poller(
                    app,
                    owned_job_pollers,
                    name="vn_asset_generation_jobs_task",
                    task=vn_asset_generation_jobs_task,
                    stop_event=vn_asset_generation_jobs_stop_event,
                )
        else:
            logger.info(
                "VN asset generation Jobs worker disabled by flag "
                "(VN_ASSET_GENERATION_JOBS_WORKER_ENABLED)"
            )
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        _safe_cancel_task(vn_asset_generation_jobs_task)
        logger.warning(f"Failed to start VN asset generation Jobs worker: {exc}")
        vn_asset_generation_jobs_stop_event = None
        vn_asset_generation_jobs_task = None

    return (
        vn_asset_jobs_stop_event,
        vn_asset_jobs_task,
        vn_asset_generation_jobs_stop_event,
        vn_asset_generation_jobs_task,
    )


async def _start_companion_reflection_jobs_worker(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    should_start_worker: Callable[..., bool],
    worker_inventory: WorkerRegistry | None = None,
) -> tuple[Any | None, Any | None]:
    """Start the companion reflection jobs poller and return its shutdown handles."""

    try:
        enabled = should_start_worker("COMPANION_REFLECTION_JOBS_WORKER_ENABLED", "companion")
        if not enabled:
            logger.info(
                "Companion reflection Jobs worker disabled by flag (COMPANION_REFLECTION_JOBS_WORKER_ENABLED)"
            )
            return None, None

        if worker_inventory is not None:
            stop_event, task = await _register_jobs_worker_with_inventory(
                worker_inventory,
                name="companion_reflection_jobs_task",
                coroutine_factory=_run_companion_reflection_jobs_worker_service,
            )
            logger.info("Companion reflection Jobs worker started with explicit stop_event signal")
            return stop_event, task

        stop_event = _make_event()
        task = _create_task(_run_companion_reflection_jobs_worker_service(stop_event))
        logger.info("Companion reflection Jobs worker started with explicit stop_event signal")
        register_owned_job_poller(
            app,
            owned_job_pollers,
            name="companion_reflection_jobs_task",
            task=task,
            stop_event=stop_event,
        )
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Companion reflection Jobs worker: {exc}")
        return None, None


def _run_audio_jobs_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.audio_jobs_worker import (
        run_audio_jobs_worker as _run_audio_jobs_worker,
    )

    return _run_audio_jobs_worker(stop_event)


def _run_audiobook_jobs_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.audiobook_jobs_worker import (
        run_audiobook_jobs_worker as _run_audiobook_jobs_worker,
    )

    return _run_audiobook_jobs_worker(stop_event)


def _run_presentation_render_jobs_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.presentation_render_jobs_worker import (
        run_presentation_render_jobs_worker as _run_presentation_render_jobs_worker,
    )

    return _run_presentation_render_jobs_worker(stop_event)


def _run_media_ingest_jobs_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.media_ingest_jobs_worker import (
        run_media_ingest_jobs_worker as _run_media_ingest_jobs_worker,
    )

    return _run_media_ingest_jobs_worker(stop_event)


def _run_media_ingest_heavy_jobs_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.media_ingest_jobs_worker import (
        run_media_ingest_heavy_jobs_worker as _run_media_ingest_heavy_jobs_worker,
    )

    return _run_media_ingest_heavy_jobs_worker(stop_event)


def _run_reading_digest_jobs_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.core.Collections.reading_digest_jobs_worker import (
        run_reading_digest_jobs_worker as _run_reading_digest_jobs_worker,
    )

    return _run_reading_digest_jobs_worker(stop_event)


def _run_llamacpp_acquisition_jobs_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.llamacpp_acquisition_jobs_worker import (
        run_llamacpp_acquisition_jobs_worker as _run_llamacpp_acquisition_jobs_worker,
    )

    return _run_llamacpp_acquisition_jobs_worker(stop_event)


def _run_vn_asset_jobs_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.vn_asset_jobs_worker import (
        run_vn_asset_jobs_worker as _run_vn_asset_jobs_worker,
    )

    return _run_vn_asset_jobs_worker(stop_event)


def _run_vn_asset_generation_jobs_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.vn_asset_jobs_worker import (
        run_vn_asset_generation_jobs_worker as _run_vn_asset_generation_jobs_worker,
    )

    return _run_vn_asset_generation_jobs_worker(stop_event)


def _run_companion_reflection_jobs_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.core.Personalization.companion_reflection_jobs_worker import (
        run_companion_reflection_jobs_worker as _run_companion_reflection_jobs_worker,
    )

    return _run_companion_reflection_jobs_worker(stop_event)
