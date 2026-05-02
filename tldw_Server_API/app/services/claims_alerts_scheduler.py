"""
Claims alerts scheduler.

Runs periodic evaluation of claims alert rules and dispatches notifications.
Enable via env:
  - CLAIMS_ALERTS_SCHEDULER_ENABLED=true
  - CLAIMS_ALERTS_EVAL_INTERVAL_SEC=300
  - CLAIMS_ALERTS_WINDOW_SEC=3600
  - CLAIMS_ALERTS_BASELINE_SEC=86400
"""

from __future__ import annotations

import asyncio
import os
from typing import Callable

from loguru import logger

from tldw_Server_API.app.core.Claims_Extraction.claims_service import (
    evaluate_claims_alerts_for_scheduler,
    send_claims_alert_email_digest_for_scheduler,
)
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.DB_Manager import content_db_settings
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.api import managed_media_database
from tldw_Server_API.app.core.testing import is_truthy

_CLAIMS_ALERTS_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    KeyError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)

def _enumerate_sqlite_user_ids() -> list[int]:
    try:
        base = DatabasePaths.get_user_db_base_dir()
    except _CLAIMS_ALERTS_NONCRITICAL_EXCEPTIONS as exc:
        logger.bind(error_type=type(exc).__name__).debug(
            "claims_alerts: failed to resolve user db base dir"
        )
        return []
    user_ids: list[int] = []
    for entry in base.iterdir():
        if not entry.is_dir():
            continue
        try:
            user_id = int(entry.name)
        except (TypeError, ValueError):
            logger.debug(f"claims_alerts: skipping non-int user dir {entry.name}")
            continue
        db_path = entry / DatabasePaths.MEDIA_DB_NAME
        if db_path.exists():
            user_ids.append(user_id)
    if not user_ids:
        try:
            user_ids = [DatabasePaths.get_single_user_id()]
        except _CLAIMS_ALERTS_NONCRITICAL_EXCEPTIONS as exc:
            logger.bind(error_type=type(exc).__name__).debug(
                "claims_alerts: failed to derive single_user_id"
            )
            user_ids = []
    return sorted(set(user_ids))


async def run_claims_alerts_once(
    *,
    evaluator: Callable[..., dict] | None = None,
    window_sec: int | None = None,
    baseline_sec: int | None = None,
) -> int:
    eval_fn = evaluator or evaluate_claims_alerts_for_scheduler
    window_val = int(window_sec or settings.get("CLAIMS_ALERTS_WINDOW_SEC", 3600))
    baseline_val = int(baseline_sec or settings.get("CLAIMS_ALERTS_BASELINE_SEC", 86400))
    processed = 0
    if content_db_settings.backend_type == BackendType.POSTGRESQL:
        try:
            with managed_media_database(
                client_id=str(settings.get("SERVER_CLIENT_ID", "SERVER_API_V1")),
                suppress_init_exceptions=_CLAIMS_ALERTS_NONCRITICAL_EXCEPTIONS,
                suppress_close_exceptions=_CLAIMS_ALERTS_NONCRITICAL_EXCEPTIONS,
            ) as db:
                user_ids = db.list_claims_monitoring_user_ids()
                for user_id in user_ids:
                    try:
                        await asyncio.to_thread(
                            eval_fn,
                            target_user_id=str(user_id),
                            window_sec=window_val,
                            baseline_sec=baseline_val,
                            db=db,
                        )
                        await send_claims_alert_email_digest_for_scheduler(
                            target_user_id=str(user_id),
                            db=db,
                        )
                        processed += 1
                    except _CLAIMS_ALERTS_NONCRITICAL_EXCEPTIONS as exc:
                        logger.bind(error_type=type(exc).__name__).warning(
                            f"claims_alerts: evaluation failed for user {user_id}"
                        )
        except _CLAIMS_ALERTS_NONCRITICAL_EXCEPTIONS as exc:
            logger.bind(error_type=type(exc).__name__).warning(
                "claims_alerts: failed to create media db"
            )
            return 0
    else:
        user_ids = _enumerate_sqlite_user_ids()
        for user_id in user_ids:
            try:
                with managed_media_database(
                    client_id=str(settings.get("SERVER_CLIENT_ID", "SERVER_API_V1")),
                    db_path=str(DatabasePaths.get_media_db_path(int(user_id))),
                    suppress_init_exceptions=_CLAIMS_ALERTS_NONCRITICAL_EXCEPTIONS,
                    suppress_close_exceptions=_CLAIMS_ALERTS_NONCRITICAL_EXCEPTIONS,
                ) as user_db:
                    await asyncio.to_thread(
                        eval_fn,
                        target_user_id=str(user_id),
                        window_sec=window_val,
                        baseline_sec=baseline_val,
                        db=user_db,
                    )
                    await send_claims_alert_email_digest_for_scheduler(
                        target_user_id=str(user_id),
                        db=user_db,
                    )
                    processed += 1
            except _CLAIMS_ALERTS_NONCRITICAL_EXCEPTIONS as exc:
                logger.bind(error_type=type(exc).__name__).warning(
                    f"claims_alerts: evaluation failed for user {user_id}"
                )
    return processed


async def start_claims_alerts_scheduler() -> asyncio.Task | None:
    enabled = is_truthy(os.getenv("CLAIMS_ALERTS_SCHEDULER_ENABLED")) or bool(
        settings.get("CLAIMS_ALERTS_SCHEDULER_ENABLED", False)
    )
    if not enabled:
        logger.info("Claims alerts scheduler disabled (CLAIMS_ALERTS_SCHEDULER_ENABLED != true)")
        return None
    try:
        interval = int(os.getenv("CLAIMS_ALERTS_EVAL_INTERVAL_SEC") or settings.get("CLAIMS_ALERTS_EVAL_INTERVAL_SEC", 300))
    except (TypeError, ValueError):
        interval = 300

    async def _runner() -> None:
        await asyncio.sleep(min(5, interval))
        while True:
            try:
                await run_claims_alerts_once()
            except _CLAIMS_ALERTS_NONCRITICAL_EXCEPTIONS as exc:
                logger.bind(error_type=type(exc).__name__).warning(
                    "Claims alerts scheduler loop error"
                )
            await asyncio.sleep(interval)

    task = asyncio.create_task(_runner(), name="claims_alerts_scheduler")
    logger.info(f"Claims alerts scheduler started (interval={interval}s)")
    return task
