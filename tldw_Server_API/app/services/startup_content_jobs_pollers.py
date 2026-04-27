"""
Content-oriented jobs poller startup helpers extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

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
    companion_reflection_jobs_stop_event: Any | None = None
    companion_reflection_jobs_task: Any | None = None


async def start_content_jobs_pollers(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    should_start_worker: Callable[..., bool],
) -> ContentJobsPollerHandles:
    """Start content jobs pollers and return their handles."""

    audio_jobs_stop_event, audio_jobs_task = await _start_audio_jobs_worker(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        should_start_worker=should_start_worker,
    )
    audiobook_jobs_stop_event, audiobook_jobs_task = await _start_audiobook_jobs_worker(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        should_start_worker=should_start_worker,
    )
    presentation_render_jobs_stop_event, presentation_render_jobs_task = (
        await _start_presentation_render_jobs_worker(
            app=app,
            owned_job_pollers=owned_job_pollers,
            register_owned_job_poller=register_owned_job_poller,
            should_start_worker=should_start_worker,
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
    )
    reading_digest_jobs_stop_event, reading_digest_jobs_task = await _start_reading_digest_jobs_worker(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        should_start_worker=should_start_worker,
    )
    companion_reflection_jobs_stop_event, companion_reflection_jobs_task = (
        await _start_companion_reflection_jobs_worker(
            app=app,
            owned_job_pollers=owned_job_pollers,
            register_owned_job_poller=register_owned_job_poller,
            should_start_worker=should_start_worker,
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
        companion_reflection_jobs_stop_event=companion_reflection_jobs_stop_event,
        companion_reflection_jobs_task=companion_reflection_jobs_task,
    )


def _make_event() -> Any:
    return asyncio.Event()


def _create_task(awaitable: Any) -> Any:
    return asyncio.create_task(awaitable)


async def _start_audio_jobs_worker(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    should_start_worker: Callable[..., bool],
) -> tuple[Any | None, Any | None]:
    try:
        enabled = should_start_worker("AUDIO_JOBS_WORKER_ENABLED", "audio-jobs")
        if not enabled:
            logger.info("Audio Jobs worker disabled by flag (AUDIO_JOBS_WORKER_ENABLED)")
            return None, None

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
) -> tuple[Any | None, Any | None]:
    try:
        enabled = should_start_worker("AUDIOBOOK_JOBS_WORKER_ENABLED", "audiobooks")
        if not enabled:
            logger.info("Audiobook Jobs worker disabled by flag (AUDIOBOOK_JOBS_WORKER_ENABLED)")
            return None, None

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
) -> tuple[Any | None, Any | None]:
    try:
        enabled = should_start_worker("PRESENTATION_RENDER_JOBS_WORKER_ENABLED", "slides")
        if not enabled:
            logger.info(
                "Presentation Render Jobs worker disabled by flag (PRESENTATION_RENDER_JOBS_WORKER_ENABLED)"
            )
            return None, None

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
) -> tuple[Any | None, Any | None, Any | None, Any | None]:
    media_ingest_jobs_stop_event = None
    media_ingest_jobs_task = None
    media_ingest_heavy_jobs_stop_event = None
    media_ingest_heavy_jobs_task = None

    try:
        enabled = should_start_worker("MEDIA_INGEST_JOBS_WORKER_ENABLED", "media")
        if enabled:
            media_ingest_jobs_stop_event = _make_event()
            media_ingest_jobs_task = _create_task(
                _run_media_ingest_jobs_worker_service(media_ingest_jobs_stop_event)
            )
            logger.info("Media Ingest Jobs worker started with explicit stop_event signal")
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
            media_ingest_heavy_jobs_stop_event = _make_event()
            media_ingest_heavy_jobs_task = _create_task(
                _run_media_ingest_heavy_jobs_worker_service(media_ingest_heavy_jobs_stop_event)
            )
            logger.info("Media Ingest Heavy Jobs worker started with explicit stop_event signal")
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
) -> tuple[Any | None, Any | None]:
    try:
        enabled = should_start_worker("READING_DIGEST_JOBS_WORKER_ENABLED", "reading")
        if not enabled:
            logger.info("Reading digest Jobs worker disabled by flag (READING_DIGEST_JOBS_WORKER_ENABLED)")
            return None, None

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


async def _start_companion_reflection_jobs_worker(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    should_start_worker: Callable[..., bool],
) -> tuple[Any | None, Any | None]:
    try:
        enabled = should_start_worker("COMPANION_REFLECTION_JOBS_WORKER_ENABLED", "companion")
        if not enabled:
            logger.info(
                "Companion reflection Jobs worker disabled by flag (COMPANION_REFLECTION_JOBS_WORKER_ENABLED)"
            )
            return None, None

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


def _run_companion_reflection_jobs_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.core.Personalization.companion_reflection_jobs_worker import (
        run_companion_reflection_jobs_worker as _run_companion_reflection_jobs_worker,
    )

    return _run_companion_reflection_jobs_worker(stop_event)
