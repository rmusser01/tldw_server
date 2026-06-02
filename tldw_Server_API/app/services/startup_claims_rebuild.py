"""
Claims rebuild startup helpers extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from loguru import logger

from tldw_Server_API.app.services.lifecycle_worker_specs import (
    WorkerLifecycleContext,
    WorkerSpec,
)
from tldw_Server_API.app.services.lifecycle_workers import (
    ShutdownPhase,
    start_stop_event_worker,
)

_STARTUP_GUARD_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def provide_claims_rebuild_worker_specs(
    _context: WorkerLifecycleContext | None = None,
) -> tuple[WorkerSpec, ...]:
    """Return the declarative spec for the claims rebuild worker."""

    return (
        WorkerSpec(
            name="claims_rebuild",
            task_name="claims_task",
            category="claims",
            phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            enabled=_claims_rebuild_worker_enabled,
            factory=lambda context, stop_event: _run_claims_rebuild_loop(
                context.settings,
                stop_event=stop_event,
                interval_sec=int(context.settings.get("CLAIMS_REBUILD_INTERVAL_SEC", 3600)),
                policy=str(context.settings.get("CLAIMS_REBUILD_POLICY", "missing")).lower(),
            ),
        ),
    )


def _claims_rebuild_worker_enabled(context: WorkerLifecycleContext) -> bool:
    return bool(context.settings.get("CLAIMS_REBUILD_ENABLED", False))


async def start_claims_rebuild_worker(
    app_settings: Mapping[str, Any],
    *,
    worker_inventory: Any | None = None,
) -> Any | None:
    """Start the claims rebuild worker when enabled."""
    try:
        enabled = bool(app_settings.get("CLAIMS_REBUILD_ENABLED", False))
        if not enabled:
            logger.info("Claims rebuild worker disabled by settings")
            return None

        interval_sec = int(app_settings.get("CLAIMS_REBUILD_INTERVAL_SEC", 3600))
        policy = str(app_settings.get("CLAIMS_REBUILD_POLICY", "missing")).lower()

        if worker_inventory is not None:
            task, _registered_stop_event = await start_stop_event_worker(
                worker_inventory,
                name="claims_rebuild",
                task_name="claims_task",
                coroutine_factory=lambda registered_stop_event: _run_claims_rebuild_loop(
                    app_settings,
                    stop_event=registered_stop_event,
                    interval_sec=interval_sec,
                    policy=policy,
                ),
                category="claims",
                shutdown_phase=ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
            )
            return task

        stop_event = asyncio.Event()
        task = asyncio.create_task(
            _run_claims_rebuild_loop(
                app_settings,
                stop_event=stop_event,
                interval_sec=interval_sec,
                policy=policy,
            )
        )
        task._tldw_claims_rebuild_stop_event = stop_event
        return task
    except _STARTUP_GUARD_EXCEPTIONS as exc:
        logger.warning(f"Failed to start claims rebuild worker: {exc}")
        return None


async def _run_claims_rebuild_loop(
    app_settings: Mapping[str, Any],
    *,
    stop_event: asyncio.Event,
    interval_sec: int,
    policy: str,
) -> None:
    """Run claims rebuild iterations until the lifecycle stop event is set.

    The loop performs one bounded rebuild scan per interval and exits when the
    caller-owned stop event is signaled by either the managed lifecycle worker
    registry or the legacy direct-task shutdown path.
    """
    logger.info(f"Starting claims rebuild worker (every {interval_sec}s, policy={policy})")
    service = _get_claims_rebuild_service()
    while not stop_event.is_set():
        try:
            run_claims_rebuild_iteration(app_settings, service, policy=policy)
        except _STARTUP_GUARD_EXCEPTIONS as exc:
            logger.warning(f"Claims rebuild loop error: {exc}")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_sec)
        except asyncio.TimeoutError:
            continue


def run_claims_rebuild_iteration(
    app_settings: Mapping[str, Any],
    service: Any,
    *,
    policy: str,
) -> None:
    """Submit one claims rebuild batch for the configured default user."""
    with _claims_rebuild_db_session(app_settings) as (_, db_path, db):
        media_ids = _list_claims_rebuild_media_ids(
            db,
            policy=policy,
            stale_days=int(app_settings.get("CLAIMS_STALE_DAYS", 7)),
            compare_media_last_modified=False,
            limit=25,
        )
        for media_id in media_ids:
            service.submit(media_id=media_id, db_path=db_path)


@contextmanager
def _claims_rebuild_db_session(
    app_settings: Mapping[str, Any],
) -> Iterator[tuple[int, str, Any]]:
    """Yield one managed Media DB session for the claims rebuild worker loop."""
    user_id = int(app_settings.get("SINGLE_USER_FIXED_ID", "1"))
    db_path = str(_get_user_media_db_path(user_id))
    client_id = str(app_settings.get("SERVER_CLIENT_ID", "SERVER_API_V1"))
    with _managed_media_database(
        client_id=client_id,
        db_path=db_path,
        initialize=False,
    ) as db:
        yield user_id, db_path, db


def _get_claims_rebuild_service() -> Any:
    from tldw_Server_API.app.core.Claims_Extraction.claims_rebuild_service import (
        get_claims_rebuild_service as _get_claims_svc,
    )

    return _get_claims_svc()


def _get_user_media_db_path(user_id: int) -> Any:
    from tldw_Server_API.app.core.DB_Management.db_path_utils import (
        get_user_media_db_path,
    )

    return get_user_media_db_path(user_id)


def _managed_media_database(*, client_id: str, db_path: str, initialize: bool) -> Any:
    from tldw_Server_API.app.core.DB_Management.media_db.api import (
        managed_media_database,
    )

    return managed_media_database(
        client_id=client_id,
        db_path=db_path,
        initialize=initialize,
    )


def _list_claims_rebuild_media_ids(
    db: Any,
    *,
    policy: str,
    stale_days: int,
    compare_media_last_modified: bool,
    limit: int,
) -> Any:
    from tldw_Server_API.app.core.Claims_Extraction.claims_service import (
        list_claims_rebuild_media_ids,
    )

    return list_claims_rebuild_media_ids(
        db,
        policy=policy,
        stale_days=stale_days,
        compare_media_last_modified=compare_media_last_modified,
        limit=limit,
    )
