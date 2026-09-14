"""
Content-oriented jobs poller startup helpers extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
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
from tldw_Server_API.app.services.worker_startup_policy import should_start_inprocess_worker

_STARTUP_GUARD_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
_STANDALONE_HANDLER_SHUTDOWN_GRACE_SECONDS = 5.0
_STANDALONE_RETRY_SECONDS = 60.0
_STANDALONE_WORKER_SHUTDOWN_TIMEOUT_SECONDS = 15.0
_STANDALONE_COORDINATION_GENERATION_ENV = "SLIDES_STANDALONE_COORDINATION_GENERATION"
_STANDALONE_MAX_COORDINATION_GENERATION = (1 << 63) - 1
_STANDALONE_VALIDATION_POOL_ATTR = "standalone_html_validation_pool"
_STANDALONE_VALIDATION_POOL_LOCK_ATTR = "standalone_html_validation_pool_lock"
_STANDALONE_VALIDATION_POOL_WORKER_OWNED_ATTR = "standalone_html_validation_pool_worker_owned"
_STANDALONE_TRANSPORT_CONTEXT_ATTR = "standalone_html_transport_context"


@dataclass
class _StandaloneHtmlGenerationRuntime:
    """Private source-free lifecycle wiring for reconciliation and Task 8."""

    reconciler: Any
    local_only: bool
    job_manager: Any | None = None
    keyring: Any | None = None
    registry: Any | None = None
    digest_snapshot_loader: Any | None = None
    current_config_loader: Any | None = None
    provider_api_key_loader: Any | None = None
    validation_pool: Any | None = None
    admission_gate: Any | None = None
    validator_available: bool = False
    config_epoch: str | None = None


@dataclass
class _StandaloneHtmlAdmissionGate:
    open: bool = False


@dataclass
class ContentJobsPollerHandles:
    """Startup-owned content jobs poller handles used later in shutdown flow."""

    audio_jobs_stop_event: Any | None = None
    audio_jobs_task: Any | None = None
    audiobook_jobs_stop_event: Any | None = None
    audiobook_jobs_task: Any | None = None
    audio_studio_jobs_stop_event: Any | None = None
    audio_studio_jobs_task: Any | None = None
    presentation_render_jobs_stop_event: Any | None = None
    presentation_render_jobs_task: Any | None = None
    research_workspace_output_jobs_stop_event: Any | None = None
    research_workspace_output_jobs_task: Any | None = None
    media_ingest_jobs_stop_event: Any | None = None
    media_ingest_jobs_task: Any | None = None
    media_ingest_heavy_jobs_stop_event: Any | None = None
    media_ingest_heavy_jobs_task: Any | None = None
    reading_digest_jobs_stop_event: Any | None = None
    reading_digest_jobs_task: Any | None = None
    chat_macros_jobs_stop_event: Any | None = None
    chat_macros_jobs_task: Any | None = None
    llamacpp_acquisition_jobs_stop_event: Any | None = None
    llamacpp_acquisition_jobs_task: Any | None = None
    visual_identity_jobs_stop_event: Any | None = None
    visual_identity_jobs_task: Any | None = None
    vn_asset_jobs_stop_event: Any | None = None
    vn_asset_jobs_task: Any | None = None
    vn_asset_generation_jobs_stop_event: Any | None = None
    vn_asset_generation_jobs_task: Any | None = None
    companion_reflection_jobs_stop_event: Any | None = None
    companion_reflection_jobs_task: Any | None = None


def media_ingest_worker_predicate(
    flag_key: str,
    route_key: str,
    *,
    default_stable: bool,
) -> Callable[[WorkerLifecycleContext], bool]:
    """Return an in-process media ingest worker predicate."""

    def _enabled(context: WorkerLifecycleContext) -> bool:
        return should_start_inprocess_worker(
            flag_key,
            route_key,
            sidecar_mode=context.sidecar_mode,
            default_stable=default_stable,
            test_mode=context.test_mode,
            route_enabled=context.route_enabled,
        )

    return _enabled


def _standalone_html_worker_enabled(context: WorkerLifecycleContext) -> bool:
    """Keep cleanup/handler ownership aligned with the existing Slides route."""

    return bool(context.route_enabled("slides"))


def _standalone_html_worker_factory(
    context: WorkerLifecycleContext,
    stop_event: asyncio.Event,
) -> Any:
    return _run_standalone_html_generation_jobs_service(context, stop_event)


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
        WorkerSpec(
            name="standalone_html_generation_jobs_task",
            task_name="standalone_html_generation_jobs_task",
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            timeout_sec=_STANDALONE_WORKER_SHUTDOWN_TIMEOUT_SECONDS,
            enabled=_standalone_html_worker_enabled,
            factory=_standalone_html_worker_factory,
        ),
        stop_event_worker_spec(
            name="research_workspace_output_jobs_task",
            worker_service=_run_research_workspace_output_jobs_worker_service,
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            enabled=route_enabled_predicate(
                "RESEARCH_WORKSPACE_OUTPUT_JOBS_WORKER_ENABLED",
                "research-workspace-output-jobs",
                default_stable=True,
            ),
        ),
        stop_event_worker_spec(
            name="media_ingest_jobs_task",
            worker_service=_run_media_ingest_jobs_worker_service,
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            enabled=media_ingest_worker_predicate(
                "MEDIA_INGEST_JOBS_WORKER_ENABLED",
                "media",
                default_stable=True,
            ),
        ),
        stop_event_worker_spec(
            name="media_ingest_heavy_jobs_task",
            worker_service=_run_media_ingest_heavy_jobs_worker_service,
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            enabled=media_ingest_worker_predicate(
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
            name="chat_macros_jobs_task",
            worker_service=_run_chat_macros_jobs_worker_service,
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            enabled=route_enabled_predicate(
                "CHAT_MACROS_JOBS_WORKER_ENABLED",
                "chat-macros",
            ),
        ),
        stop_event_worker_spec(
            name="llamacpp_acquisition_jobs_task",
            worker_service=_run_llamacpp_acquisition_jobs_worker_service,
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            enabled=route_enabled_predicate(
                "LLAMACPP_ACQUISITION_JOBS_WORKER_ENABLED",
                "llamacpp-acquisition",
            ),
        ),
        stop_event_worker_spec(
            name="visual_identity_jobs_task",
            worker_service=_run_visual_identity_jobs_worker_service,
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            enabled=route_enabled_predicate(
                "VISUAL_IDENTITY_JOBS_WORKER_ENABLED",
                "visual-identities",
                default_stable=True,
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
        stop_event_worker_spec(
            name="scheduled_tasks_recurring_question_jobs_task",
            worker_service=_run_scheduled_tasks_recurring_question_jobs_worker_service,
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            enabled=route_enabled_predicate(
                "SCHEDULED_TASKS_RECURRING_QUESTION_WORKER_ENABLED",
                "scheduled-tasks-recurring-question",
                default_stable=False,
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
    audio_studio_jobs_stop_event, audio_studio_jobs_task = await _start_audio_studio_jobs_worker(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        should_start_worker=should_start_worker,
        worker_inventory=worker_inventory,
    )
    presentation_render_jobs_stop_event, presentation_render_jobs_task = await _start_presentation_render_jobs_worker(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        should_start_worker=should_start_worker,
        worker_inventory=worker_inventory,
    )
    research_workspace_output_jobs_stop_event, research_workspace_output_jobs_task = (
        await _start_research_workspace_output_jobs_worker(
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
    chat_macros_jobs_stop_event, chat_macros_jobs_task = await _start_chat_macros_jobs_worker(
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
    visual_identity_jobs_stop_event, visual_identity_jobs_task = await _start_visual_identity_jobs_worker(
        app=app,
        owned_job_pollers=owned_job_pollers,
        register_owned_job_poller=register_owned_job_poller,
        should_start_worker=should_start_worker,
        worker_inventory=worker_inventory,
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
        audio_studio_jobs_stop_event=audio_studio_jobs_stop_event,
        audio_studio_jobs_task=audio_studio_jobs_task,
        presentation_render_jobs_stop_event=presentation_render_jobs_stop_event,
        presentation_render_jobs_task=presentation_render_jobs_task,
        research_workspace_output_jobs_stop_event=research_workspace_output_jobs_stop_event,
        research_workspace_output_jobs_task=research_workspace_output_jobs_task,
        media_ingest_jobs_stop_event=media_ingest_jobs_stop_event,
        media_ingest_jobs_task=media_ingest_jobs_task,
        media_ingest_heavy_jobs_stop_event=media_ingest_heavy_jobs_stop_event,
        media_ingest_heavy_jobs_task=media_ingest_heavy_jobs_task,
        reading_digest_jobs_stop_event=reading_digest_jobs_stop_event,
        reading_digest_jobs_task=reading_digest_jobs_task,
        chat_macros_jobs_stop_event=chat_macros_jobs_stop_event,
        chat_macros_jobs_task=chat_macros_jobs_task,
        llamacpp_acquisition_jobs_stop_event=llamacpp_acquisition_jobs_stop_event,
        llamacpp_acquisition_jobs_task=llamacpp_acquisition_jobs_task,
        visual_identity_jobs_stop_event=visual_identity_jobs_stop_event,
        visual_identity_jobs_task=visual_identity_jobs_task,
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


async def _start_audio_studio_jobs_worker(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    should_start_worker: Callable[..., bool],
    worker_inventory: WorkerRegistry | None = None,
) -> tuple[Any | None, Any | None]:
    """Start the Audio Studio jobs poller and return its shutdown handles."""

    try:
        enabled = should_start_worker("AUDIO_STUDIO_JOBS_WORKER_ENABLED", "audio-studio")
        if not enabled:
            logger.info("Audio Studio Jobs worker disabled by flag (AUDIO_STUDIO_JOBS_WORKER_ENABLED)")
            return None, None

        if worker_inventory is not None:
            stop_event, task = await _register_jobs_worker_with_inventory(
                worker_inventory,
                name="audio_studio_jobs_task",
                coroutine_factory=_run_audio_studio_jobs_worker_service,
            )
            logger.info("Audio Studio Jobs worker started with explicit stop_event signal")
            return stop_event, task

        stop_event = _make_event()
        task = _create_task(_run_audio_studio_jobs_worker_service(stop_event))
        logger.info("Audio Studio Jobs worker started with explicit stop_event signal")
        register_owned_job_poller(
            app,
            owned_job_pollers,
            name="audio_studio_jobs_task",
            task=task,
            stop_event=stop_event,
        )
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Audio Studio Jobs worker: {exc}")
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
            logger.info("Presentation Render Jobs worker disabled by flag (PRESENTATION_RENDER_JOBS_WORKER_ENABLED)")
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


async def _start_research_workspace_output_jobs_worker(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    should_start_worker: Callable[..., bool],
    worker_inventory: WorkerRegistry | None = None,
) -> tuple[Any | None, Any | None]:
    """Start the Research Workspace output jobs poller and return its shutdown handles."""

    try:
        enabled = should_start_worker(
            "RESEARCH_WORKSPACE_OUTPUT_JOBS_WORKER_ENABLED",
            "research-workspace-output-jobs",
            default_stable=True,
        )
        if not enabled:
            logger.info(
                "Research Workspace Output Jobs worker disabled by flag "
                "(RESEARCH_WORKSPACE_OUTPUT_JOBS_WORKER_ENABLED)"
            )
            return None, None

        if worker_inventory is not None:
            stop_event, task = await _register_jobs_worker_with_inventory(
                worker_inventory,
                name="research_workspace_output_jobs_task",
                coroutine_factory=_run_research_workspace_output_jobs_worker_service,
            )
            logger.info("Research Workspace Output Jobs worker started with explicit stop_event signal")
            return stop_event, task

        stop_event = _make_event()
        task = _create_task(_run_research_workspace_output_jobs_worker_service(stop_event))
        logger.info("Research Workspace Output Jobs worker started with explicit stop_event signal")
        register_owned_job_poller(
            app,
            owned_job_pollers,
            name="research_workspace_output_jobs_task",
            task=task,
            stop_event=stop_event,
        )
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start Research Workspace Output Jobs worker: {exc}")
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
            logger.info("Media Ingest Heavy Jobs worker disabled by flag (MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED)")
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


async def _start_chat_macros_jobs_worker(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    should_start_worker: Callable[..., bool],
    worker_inventory: WorkerRegistry | None = None,
) -> tuple[Any | None, Any | None]:
    """Start the chat macro jobs poller and return its shutdown handles."""

    task = None
    try:
        enabled = should_start_worker("CHAT_MACROS_JOBS_WORKER_ENABLED", "chat-macros")
        if not enabled:
            logger.info("Chat macro Jobs worker disabled by flag (CHAT_MACROS_JOBS_WORKER_ENABLED)")
            return None, None

        if worker_inventory is not None:
            stop_event, task = await _register_jobs_worker_with_inventory(
                worker_inventory,
                name="chat_macros_jobs_task",
                coroutine_factory=_run_chat_macros_jobs_worker_service,
            )
            logger.info("Chat macro Jobs worker started with explicit stop_event signal")
            return stop_event, task

        stop_event = _make_event()
        task = _create_task(_run_chat_macros_jobs_worker_service(stop_event))
        logger.info("Chat macro Jobs worker started with explicit stop_event signal")
        register_owned_job_poller(
            app,
            owned_job_pollers,
            name="chat_macros_jobs_task",
            task=task,
            stop_event=stop_event,
        )
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        _safe_cancel_task(task)
        logger.warning(f"Failed to start Chat macro Jobs worker: {exc}")
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
                "llama.cpp Acquisition Jobs worker disabled by flag " "(LLAMACPP_ACQUISITION_JOBS_WORKER_ENABLED)"
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


async def _start_visual_identity_jobs_worker(
    *,
    app: Any,
    owned_job_pollers: list[Any],
    register_owned_job_poller: Callable[..., None],
    should_start_worker: Callable[..., bool],
    worker_inventory: WorkerRegistry | None = None,
) -> tuple[Any | None, Any | None]:
    """Start the Visual Identity jobs poller and return its shutdown handles."""

    task = None
    try:
        enabled = should_start_worker(
            "VISUAL_IDENTITY_JOBS_WORKER_ENABLED",
            "visual-identities",
            default_stable=True,
        )
        if not enabled:
            logger.info(
                "Visual Identity Jobs worker disabled by flag "
                "(VISUAL_IDENTITY_JOBS_WORKER_ENABLED)"
            )
            return None, None

        if worker_inventory is not None:
            stop_event, task = await _register_jobs_worker_with_inventory(
                worker_inventory,
                name="visual_identity_jobs_task",
                coroutine_factory=_run_visual_identity_jobs_worker_service,
            )
            logger.info("Visual Identity Jobs worker started with explicit stop_event signal")
            return stop_event, task

        stop_event = _make_event()
        task = _create_task(_run_visual_identity_jobs_worker_service(stop_event))
        logger.info("Visual Identity Jobs worker started with explicit stop_event signal")
        register_owned_job_poller(
            app,
            owned_job_pollers,
            name="visual_identity_jobs_task",
            task=task,
            stop_event=stop_event,
        )
        return stop_event, task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        _safe_cancel_task(task)
        logger.warning(f"Failed to start Visual Identity Jobs worker: {exc}")
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
                vn_asset_jobs_task = _create_task(_run_vn_asset_jobs_worker_service(vn_asset_jobs_stop_event))
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
                    _run_vn_asset_generation_jobs_worker_service(vn_asset_generation_jobs_stop_event)
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
            logger.info("VN asset generation Jobs worker disabled by flag " "(VN_ASSET_GENERATION_JOBS_WORKER_ENABLED)")
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
            logger.info("Companion reflection Jobs worker disabled by flag (COMPANION_REFLECTION_JOBS_WORKER_ENABLED)")
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


def _run_audio_studio_jobs_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.core.Audio_Studio.jobs_worker import (
        run_audio_studio_jobs_worker as _run_audio_studio_jobs_worker,
    )

    return _run_audio_studio_jobs_worker(stop_event)


def _run_presentation_render_jobs_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.presentation_render_jobs_worker import (
        run_presentation_render_jobs_worker as _run_presentation_render_jobs_worker,
    )

    return _run_presentation_render_jobs_worker(stop_event)


def _run_research_workspace_output_jobs_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.research_workspace_output_jobs_worker import (
        run_research_workspace_output_jobs_worker as _run_research_workspace_output_jobs_worker,
    )

    return _run_research_workspace_output_jobs_worker(stop_event)


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


def _run_chat_macros_jobs_worker_service(stop_event: Any) -> Any:
    """Build the chat macro worker coroutine bound to its lifecycle stop event."""
    from tldw_Server_API.app.services.chat_macros_jobs_worker import (
        run_chat_macros_jobs_worker as _run_chat_macros_jobs_worker,
    )

    return _run_chat_macros_jobs_worker(stop_event)


def _run_llamacpp_acquisition_jobs_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.llamacpp_acquisition_jobs_worker import (
        run_llamacpp_acquisition_jobs_worker as _run_llamacpp_acquisition_jobs_worker,
    )

    return _run_llamacpp_acquisition_jobs_worker(stop_event)


def _run_visual_identity_jobs_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.visual_identity_jobs_worker import (
        run_visual_identity_jobs_worker as _run_visual_identity_jobs_worker,
    )

    return _run_visual_identity_jobs_worker(stop_event)


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


def _run_scheduled_tasks_recurring_question_jobs_worker_service(stop_event: Any) -> Any:
    from tldw_Server_API.app.services.scheduled_task_recurring_question_worker import (
        run_recurring_question_jobs_worker as _run_recurring_question_jobs_worker,
    )

    return _run_recurring_question_jobs_worker(stop_event)


def _standalone_html_jobs_manager() -> Any:
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    db_url = (os.getenv("JOBS_DB_URL") or "").strip()
    if not db_url:
        return JobManager()
    return JobManager(
        backend="postgres" if db_url.startswith("postgres") else None,
        db_url=db_url,
    )


def _standalone_html_coordination_epoch(
    *,
    static_config: Any,
    current_key_id: str,
) -> str:
    """Derive one stable source-free epoch over the complete boot authority."""

    target = static_config.target
    prompt = static_config.prompt
    allowed_targets = sorted(
        (asdict(candidate) for candidate in static_config.allowed_targets),
        key=lambda candidate: (
            candidate["provider"],
            candidate["model"],
            candidate["adapter_id"],
            candidate["endpoint_identity"],
        ),
    )

    manifest = json.dumps(
        {
            "current_key_id": current_key_id,
            "protocol": "slides.standalone_html.reconciliation.v1",
            "static_policy": {
                "feature_enabled": static_config.feature_enabled,
                "egress_enabled": static_config.egress_enabled,
                "enabled": static_config.enabled,
                "disabled_reason": static_config.disabled_reason,
                "generation_config_revision": static_config.generation_config_revision,
                "target": asdict(target) if target is not None else None,
                "prompt": (
                    {
                        "sha256": prompt.sha256,
                        "contract_version": prompt.contract_version,
                        "byte_count": prompt.byte_count,
                    }
                    if prompt is not None
                    else None
                ),
                "allowed_targets": allowed_targets,
                "input_limits": asdict(static_config.input_limits),
                "output_limits": asdict(static_config.output_limits),
                "provider_limits": asdict(static_config.provider_limits),
            },
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(manifest.encode("ascii")).hexdigest()
    generation = _standalone_html_coordination_generation()
    if generation == 0:
        return "sha256:" + digest
    return f"v1:g{generation}:sha256:{digest}"


def _minimum_dataclass(boot_value: Any, live_value: Any) -> Any:
    """Return the boot dataclass with every numeric field narrowed live."""

    return replace(
        boot_value,
        **{
            item.name: min(
                getattr(boot_value, item.name),
                getattr(live_value, item.name),
            )
            for item in fields(boot_value)
        },
    )


def _restrict_standalone_html_config(boot_config: Any, live_config: Any) -> Any:
    """Allow live controls to revoke boot authority but never broaden it."""

    from tldw_Server_API.app.core.Slides.standalone_html_config import (
        SlidesStandaloneHtmlConfig,
    )

    if not isinstance(boot_config, SlidesStandaloneHtmlConfig) or not isinstance(
        live_config,
        SlidesStandaloneHtmlConfig,
    ):
        raise TypeError("standalone HTML configuration snapshot is invalid")
    feature_enabled = boot_config.feature_enabled and live_config.feature_enabled
    egress_enabled = boot_config.egress_enabled and live_config.egress_enabled
    allowed_live = frozenset(live_config.allowed_targets)
    allowed_targets = tuple(target for target in boot_config.allowed_targets if target in allowed_live)
    target_allowed = boot_config.target is not None and boot_config.target in allowed_targets
    enabled = boot_config.enabled and live_config.enabled and feature_enabled and egress_enabled and target_allowed
    disabled_reason = boot_config.disabled_reason
    if not feature_enabled:
        disabled_reason = "feature_disabled"
    elif not egress_enabled:
        disabled_reason = "egress_disabled"
    elif boot_config.enabled and not target_allowed:
        disabled_reason = "default_model_not_allowed"
    elif not live_config.enabled:
        disabled_reason = live_config.disabled_reason
    elif enabled:
        disabled_reason = None
    return replace(
        boot_config,
        feature_enabled=feature_enabled,
        egress_enabled=egress_enabled,
        enabled=enabled,
        disabled_reason=disabled_reason,
        target=boot_config.target if enabled else None,
        prompt=boot_config.prompt if enabled else None,
        generation_config_revision=(boot_config.generation_config_revision if enabled else None),
        _revision_manifest=(boot_config.revision_manifest if enabled else ""),
        allowed_targets=allowed_targets,
        input_limits=_minimum_dataclass(
            boot_config.input_limits,
            live_config.input_limits,
        ),
        output_limits=_minimum_dataclass(
            boot_config.output_limits,
            live_config.output_limits,
        ),
        provider_limits=_minimum_dataclass(
            boot_config.provider_limits,
            live_config.provider_limits,
        ),
    )


def _standalone_html_coordination_generation() -> int:
    """Load the canonical monotonic generation used to fence rolling deploys."""

    raw = os.getenv(_STANDALONE_COORDINATION_GENERATION_ENV)
    if raw is None:
        return 0
    if not raw or not raw.isascii() or not raw.isdigit() or (len(raw) > 1 and raw.startswith("0")) or len(raw) > 19:
        raise ValueError("standalone HTML coordination generation is invalid")
    generation = int(raw)
    if generation > _STANDALONE_MAX_COORDINATION_GENERATION:
        raise ValueError("standalone HTML coordination generation is invalid")
    return generation


def _local_only_standalone_runtime(*, job_manager: Any, base_dir: Any) -> _StandaloneHtmlGenerationRuntime:
    from tldw_Server_API.app.core.Slides.standalone_html_reconciler import (
        FencedStandaloneHtmlReconciler,
    )

    return _StandaloneHtmlGenerationRuntime(
        reconciler=FencedStandaloneHtmlReconciler(
            job_manager=job_manager,
            user_db_base_dir=base_dir,
            config_epoch="standalone-html-local-cleanup-v1",
            holder_uuid=str(uuid.uuid4()),
        ),
        local_only=True,
        job_manager=job_manager,
    )


async def _build_standalone_html_generation_runtime(
    _context: WorkerLifecycleContext,
) -> _StandaloneHtmlGenerationRuntime:
    """Build full generation wiring or a persistent local-expiry fallback."""

    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

    base_dir = DatabasePaths.resolve_user_db_base_dir()
    try:
        job_manager = _standalone_html_jobs_manager()
    except Exception:  # noqa: BLE001 - local expiry needs no Jobs implementation detail
        return _local_only_standalone_runtime(job_manager=object(), base_dir=base_dir)
    try:
        from tldw_Server_API.app.core.config import (
            load_comprehensive_config,
            refresh_config_cache,
        )
        from tldw_Server_API.app.core.LLM_Calls.adapter_utils import (
            resolve_provider_api_key_from_config,
        )
        from tldw_Server_API.app.core.Slides import standalone_html_validator
        from tldw_Server_API.app.core.Slides.standalone_html_config import (
            StandaloneHtmlGenerationAvailability,
            load_standalone_html_config,
        )
        from tldw_Server_API.app.core.Slides.standalone_html_reconciler import (
            FencedStandaloneHtmlReconciler,
        )
        from tldw_Server_API.app.core.Slides.standalone_html_registry import (
            DigestKeyUnavailableError,
            JobManagerDigestKeyRegistryStore,
            StandaloneHtmlHmacKeyring,
            StandaloneHtmlKeyRegistry,
        )

        keyring = StandaloneHtmlHmacKeyring.from_env()
        all_available = StandaloneHtmlGenerationAvailability(
            digest_key_available=True,
            worker_handler_registered=True,
            reconciler_admission_ready=True,
            validator_available=True,
        )
        static_config = load_standalone_html_config(
            load_comprehensive_config(),
            availability=all_available,
        )
        config_epoch = _standalone_html_coordination_epoch(
            static_config=static_config,
            current_key_id=keyring.configured_current_key_id,
        )
        registry = StandaloneHtmlKeyRegistry(
            store=JobManagerDigestKeyRegistryStore(job_manager),
            keyring=keyring,
        )
        before = await registry.snapshot()
        snapshot = await registry.activate_configured_current(
            expected_current_key_id=before.current_key_id,
            expected_config_epoch=before.config_epoch,
            new_config_epoch=config_epoch,
            now=datetime.now(timezone.utc),
        )
        snapshot.require_generation_ready()
        reconciler = FencedStandaloneHtmlReconciler(
            job_manager=job_manager,
            user_db_base_dir=base_dir,
            config_epoch=config_epoch,
            holder_uuid=str(uuid.uuid4()),
        )
        gate = _StandaloneHtmlAdmissionGate()
        validator_available = bool(
            standalone_html_validator.html5lib is not None and standalone_html_validator.tinycss2 is not None
        )

        async def digest_snapshot_loader():
            if not gate.open or not reconciler.admission_ready():
                raise DigestKeyUnavailableError("generation digest key unavailable")
            current = await registry.snapshot()
            current.require_generation_ready()
            if current.config_epoch != config_epoch:
                raise DigestKeyUnavailableError("generation digest key unavailable")
            return current

        def current_config_loader():
            refresh_config_cache()
            app = getattr(_context, "app", None)
            availability = StandaloneHtmlGenerationAvailability(
                digest_key_available=True,
                worker_handler_registered=(
                    getattr(
                        getattr(app, "state", None),
                        "standalone_html_generation_worker_registered",
                        False,
                    )
                    is True
                ),
                reconciler_admission_ready=(
                    gate.open
                    and reconciler.admission_ready()
                    and getattr(
                        getattr(app, "state", None),
                        "standalone_html_reconciler_admission_ready",
                        False,
                    )
                    is True
                ),
                validator_available=validator_available,
            )
            return _restrict_standalone_html_config(
                static_config,
                load_standalone_html_config(
                    load_comprehensive_config(),
                    availability=availability,
                ),
            )

        def provider_api_key_loader(target: Any) -> str | None:
            return resolve_provider_api_key_from_config(
                target.provider,
                load_comprehensive_config(),
            )

        return _StandaloneHtmlGenerationRuntime(
            reconciler=reconciler,
            local_only=False,
            job_manager=job_manager,
            keyring=keyring,
            registry=registry,
            digest_snapshot_loader=digest_snapshot_loader,
            current_config_loader=current_config_loader,
            provider_api_key_loader=provider_api_key_loader,
            admission_gate=gate,
            validator_available=validator_available,
            config_epoch=config_epoch,
        )
    except Exception as exc:  # noqa: BLE001 - never expose key/config/store details
        logger.warning(
            "Standalone HTML generation remains closed; error_type={}",
            type(exc).__name__,
        )
        return _local_only_standalone_runtime(
            job_manager=job_manager,
            base_dir=base_dir,
        )


async def _get_worker_owned_validation_pool(app: Any) -> Any:
    from tldw_Server_API.app.core.Slides.standalone_html_validation_pool import (
        StandaloneHtmlValidationPool,
    )

    state = app.state
    lock = getattr(state, _STANDALONE_VALIDATION_POOL_LOCK_ATTR, None)
    if lock is None:
        lock = asyncio.Lock()
        setattr(state, _STANDALONE_VALIDATION_POOL_LOCK_ATTR, lock)
    async with lock:
        pool = getattr(state, _STANDALONE_VALIDATION_POOL_ATTR, None)
        if pool is None:
            pool = StandaloneHtmlValidationPool()
            setattr(state, _STANDALONE_VALIDATION_POOL_ATTR, pool)
        setattr(state, _STANDALONE_VALIDATION_POOL_WORKER_OWNED_ATTR, True)
        return pool


async def _close_worker_owned_validation_pool(app: Any) -> None:
    state = app.state
    if getattr(state, _STANDALONE_VALIDATION_POOL_WORKER_OWNED_ATTR, False) is not True:
        return
    pool = getattr(state, _STANDALONE_VALIDATION_POOL_ATTR, None)
    try:
        if pool is not None:
            await pool.close()
    finally:
        for attr_name in (
            _STANDALONE_VALIDATION_POOL_ATTR,
            _STANDALONE_VALIDATION_POOL_LOCK_ATTR,
            _STANDALONE_VALIDATION_POOL_WORKER_OWNED_ATTR,
        ):
            with contextlib.suppress(AttributeError, KeyError):
                delattr(state, attr_name)


async def _run_standalone_html_generation_handler(
    runtime: _StandaloneHtmlGenerationRuntime,
    stop_event: asyncio.Event,
) -> None:
    from tldw_Server_API.app.services.standalone_html_generation_jobs_worker import (
        run_standalone_html_generation_jobs_worker,
    )

    await run_standalone_html_generation_jobs_worker(
        keyring=runtime.keyring,
        digest_snapshot_loader=runtime.digest_snapshot_loader,
        validation_pool=runtime.validation_pool,
        current_config_loader=runtime.current_config_loader,
        provider_api_key_loader=runtime.provider_api_key_loader,
        stop_event=stop_event,
    )


async def _run_reconciliation_batch(
    call: Callable[[], Any],
    *,
    on_cancel: Callable[[], None] | None = None,
) -> Any:
    """Finish an in-flight sync batch even if lifecycle cancellation arrives."""

    task = asyncio.create_task(asyncio.to_thread(call))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        if on_cancel is not None:
            on_cancel()
        with contextlib.suppress(Exception):
            await task
        raise


async def _interruptible_wait(stop_event: asyncio.Event, timeout: float) -> None:
    if timeout <= 0:
        await asyncio.sleep(0)
        return
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(stop_event.wait(), timeout=timeout)


def _close_standalone_html_admission(app: Any, runtime: Any | None) -> None:
    gate = getattr(runtime, "admission_gate", None)
    if gate is not None:
        gate.open = False
    app.state.standalone_html_generation_worker_registered = False
    app.state.standalone_html_reconciler_admission_ready = False


def _publish_standalone_html_transport_context(app: Any, runtime: Any) -> None:
    """Publish only the full source-free lifecycle context used by REST transport."""

    if getattr(runtime, "local_only", True):
        return
    setattr(app.state, _STANDALONE_TRANSPORT_CONTEXT_ATTR, runtime)


def _clear_standalone_html_transport_context(app: Any, runtime: Any | None) -> None:
    """Remove the context only when it is still owned by this lifecycle."""

    current = getattr(app.state, _STANDALONE_TRANSPORT_CONTEXT_ATTR, None)
    if current is not runtime:
        return
    with contextlib.suppress(AttributeError, KeyError):
        delattr(app.state, _STANDALONE_TRANSPORT_CONTEXT_ATTR)


def _standalone_html_handler_done(
    app: Any,
    runtime: Any,
    task: asyncio.Task[Any],
) -> None:
    """Close admission as soon as the handler exits, without waiting for polling."""

    _close_standalone_html_admission(app, runtime)
    if not task.cancelled():
        with contextlib.suppress(Exception):
            task.exception()


async def _cleanup_standalone_html_generation_runtime(
    *,
    app: Any,
    runtime: Any | None,
    handler_task: asyncio.Task[Any] | None,
    handler_stop_event: asyncio.Event | None,
) -> None:
    """Close admission, stop Task 8, release fencing, then close its pool."""

    _close_standalone_html_admission(app, runtime)
    if handler_stop_event is not None:
        handler_stop_event.set()
    try:
        if handler_task is not None:
            done, _pending = await asyncio.wait(
                {handler_task},
                timeout=_STANDALONE_HANDLER_SHUTDOWN_GRACE_SECONDS,
            )
            if handler_task not in done:
                handler_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await handler_task
    finally:
        try:
            if runtime is not None:
                await _run_reconciliation_batch(runtime.reconciler.release)
        finally:
            try:
                _clear_standalone_html_transport_context(app, runtime)
            finally:
                await _close_worker_owned_validation_pool(app)


async def _run_cancellation_safe_standalone_cleanup(
    *,
    app: Any,
    runtime: Any | None,
    handler_task: asyncio.Task[Any] | None,
    handler_stop_event: asyncio.Event | None,
) -> None:
    cleanup_task = asyncio.create_task(
        _cleanup_standalone_html_generation_runtime(
            app=app,
            runtime=runtime,
            handler_task=handler_task,
            handler_stop_event=handler_stop_event,
        )
    )
    cancelled = False
    while not cleanup_task.done():
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            cancelled = True
    cleanup_task.result()
    if cancelled:
        raise asyncio.CancelledError


async def _run_standalone_html_generation_jobs_service(
    context: WorkerLifecycleContext,
    stop_event: asyncio.Event,
) -> None:
    """Own reconciliation, handler admission, validator pool, and shutdown."""

    app = context.app
    app.state.standalone_html_generation_worker_registered = False
    app.state.standalone_html_reconciler_admission_ready = False
    stale_context = getattr(app.state, _STANDALONE_TRANSPORT_CONTEXT_ATTR, None)
    if stale_context is not None:
        _close_standalone_html_admission(app, stale_context)
        _clear_standalone_html_transport_context(app, stale_context)
    runtime: _StandaloneHtmlGenerationRuntime | Any | None = None
    handler_task: asyncio.Task[Any] | None = None
    handler_stop_event: asyncio.Event | None = None
    draining_local_expiry = False

    def begin_shutdown(*, cancel_handler: bool = False) -> None:
        _close_standalone_html_admission(app, runtime)
        if handler_stop_event is not None:
            handler_stop_event.set()
        if cancel_handler and handler_task is not None and not handler_task.done():
            handler_task.cancel()

    async def watch_for_stop() -> None:
        await stop_event.wait()
        begin_shutdown()

    shutdown_watcher = asyncio.create_task(watch_for_stop())
    try:
        while not stop_event.is_set():
            try:
                if runtime is None:
                    runtime = await _build_standalone_html_generation_runtime(context)
                    _publish_standalone_html_transport_context(app, runtime)
                    if stop_event.is_set():
                        break
                if getattr(runtime, "local_only", False):
                    local_result = await _run_reconciliation_batch(
                        runtime.reconciler.run_local_expiry_batch,
                        on_cancel=lambda: begin_shutdown(cancel_handler=True),
                    )
                    if stop_event.is_set():
                        break
                    if getattr(local_result, "local_sweep_state", "blocked") == "progressed":
                        await _interruptible_wait(stop_event, 0.0)
                        continue
                    await _interruptible_wait(stop_event, _STANDALONE_RETRY_SECONDS)
                    if not stop_event.is_set():
                        candidate = await _build_standalone_html_generation_runtime(context)
                        if stop_event.is_set():
                            break
                        if not getattr(candidate, "local_only", False):
                            await _run_reconciliation_batch(
                                runtime.reconciler.release,
                                on_cancel=lambda: begin_shutdown(cancel_handler=True),
                            )
                            if stop_event.is_set():
                                break
                            runtime = candidate
                            _publish_standalone_html_transport_context(app, runtime)
                    continue

                if draining_local_expiry:
                    local_result = await _run_reconciliation_batch(
                        runtime.reconciler.run_local_expiry_batch,
                        on_cancel=lambda: begin_shutdown(cancel_handler=True),
                    )
                    if stop_event.is_set():
                        break
                    if getattr(local_result, "local_sweep_state", "blocked") == "progressed":
                        await _interruptible_wait(stop_event, 0.0)
                        continue
                    draining_local_expiry = False
                    await _interruptible_wait(stop_event, _STANDALONE_RETRY_SECONDS)
                    continue

                result = await _run_reconciliation_batch(
                    runtime.reconciler.run_batch,
                    on_cancel=lambda: begin_shutdown(cancel_handler=True),
                )
                if stop_event.is_set():
                    break
                if handler_task is not None and handler_task.done():
                    _close_standalone_html_admission(app, runtime)
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        handler_task.result()
                    handler_task = None
                    handler_stop_event = None

                if result.startup_ready and handler_task is None and not stop_event.is_set():
                    runtime.validation_pool = await _get_worker_owned_validation_pool(app)
                    if stop_event.is_set():
                        break
                    handler_stop_event = asyncio.Event()
                    handler_task = asyncio.create_task(
                        _run_standalone_html_generation_handler(
                            runtime,
                            handler_stop_event,
                        ),
                        name="standalone_html_generation_handler",
                    )
                    handler_task.add_done_callback(
                        lambda task, active_runtime=runtime: _standalone_html_handler_done(
                            app,
                            active_runtime,
                            task,
                        )
                    )
                    await asyncio.sleep(0)
                    if handler_task.done():
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            handler_task.result()
                        handler_task = None
                        handler_stop_event = None
                    if stop_event.is_set():
                        break

                if result.startup_ready and handler_task is not None and not stop_event.is_set():
                    if runtime.admission_gate is not None:
                        runtime.admission_gate.open = True
                    app.state.standalone_html_generation_worker_registered = True
                    app.state.standalone_html_reconciler_admission_ready = True
                else:
                    if runtime.admission_gate is not None:
                        runtime.admission_gate.open = False
                    app.state.standalone_html_reconciler_admission_ready = False

                if not result.jobs_available and getattr(result, "local_sweep_state", "not_run") == "progressed":
                    draining_local_expiry = True

                delay = (
                    0.0
                    if draining_local_expiry or (result.leader and result.jobs_available and not result.completed_pass)
                    else _STANDALONE_RETRY_SECONDS
                )
                await _interruptible_wait(stop_event, delay)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep retrying, redacted
                if runtime is not None and getattr(runtime, "admission_gate", None) is not None:
                    runtime.admission_gate.open = False
                app.state.standalone_html_reconciler_admission_ready = False
                logger.warning(
                    "Standalone HTML lifecycle iteration failed closed; error_type={}",
                    type(exc).__name__,
                )
                await _interruptible_wait(stop_event, _STANDALONE_RETRY_SECONDS)
    finally:
        begin_shutdown()
        shutdown_watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await shutdown_watcher
        await _run_cancellation_safe_standalone_cleanup(
            app=app,
            runtime=runtime,
            handler_task=handler_task,
            handler_stop_event=handler_stop_event,
        )
