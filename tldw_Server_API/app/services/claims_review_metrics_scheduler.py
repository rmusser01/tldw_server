"""
Claims review metrics scheduler.

Aggregates daily review log activity into claims_review_extractor_metrics_daily.
Enable via env:
  - CLAIMS_REVIEW_METRICS_SCHEDULER_ENABLED=true
  - CLAIMS_REVIEW_METRICS_INTERVAL_SEC=86400
  - CLAIMS_REVIEW_METRICS_LOOKBACK_DAYS=2
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Callable

from loguru import logger

from tldw_Server_API.app.core.Claims_Extraction.claims_service import (
    aggregate_claims_review_extractor_metrics_daily,
)
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.DB_Manager import content_db_settings
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.api import managed_media_database
from tldw_Server_API.app.core.testing import is_truthy

_CLAIMS_REVIEW_METRICS_NONCRITICAL_EXCEPTIONS = (
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
    except _CLAIMS_REVIEW_METRICS_NONCRITICAL_EXCEPTIONS as exc:
        logger.bind(error_type=type(exc).__name__).debug(
            "claims_review_metrics: failed to resolve user db base dir"
        )
        return []
    user_ids: list[int] = []
    for entry in base.iterdir():
        if not entry.is_dir():
            continue
        try:
            user_id = int(entry.name)
        except (TypeError, ValueError):
            logger.debug(f"claims_review_metrics: skipping non-int user dir {entry.name}")
            continue
        db_path = entry / DatabasePaths.MEDIA_DB_NAME
        if db_path.exists():
            user_ids.append(user_id)
    if not user_ids:
        try:
            user_ids = [DatabasePaths.get_single_user_id()]
        except _CLAIMS_REVIEW_METRICS_NONCRITICAL_EXCEPTIONS as exc:
            logger.bind(error_type=type(exc).__name__).debug(
                "claims_review_metrics: failed to derive single_user_id"
            )
            user_ids = []
    return sorted(set(user_ids))


async def run_claims_review_metrics_once(
    *,
    aggregator: Callable[..., int] | None = None,
    lookback_days: int | None = None,
    report_date: str | None = None,
    db: Any | None = None,
    target_user_id: str | None = None,
) -> int:
    agg_fn = aggregator or aggregate_claims_review_extractor_metrics_daily
    try:
        lookback_val = int(
            lookback_days if lookback_days is not None else settings.get("CLAIMS_REVIEW_METRICS_LOOKBACK_DAYS", 2)
        )
    except (TypeError, ValueError):
        lookback_val = 2

    if db is not None:
        return agg_fn(
            db=db,
            target_user_id=target_user_id,
            report_date=report_date,
            lookback_days=lookback_val,
        )

    processed = 0
    if content_db_settings.backend_type == BackendType.POSTGRESQL:
        try:
            with managed_media_database(
                client_id=str(settings.get("SERVER_CLIENT_ID", "SERVER_API_V1")),
                suppress_init_exceptions=_CLAIMS_REVIEW_METRICS_NONCRITICAL_EXCEPTIONS,
                suppress_close_exceptions=_CLAIMS_REVIEW_METRICS_NONCRITICAL_EXCEPTIONS,
            ) as media_db:
                user_ids = media_db.list_claims_review_user_ids()
                if not user_ids:
                    try:
                        user_ids = [str(DatabasePaths.get_single_user_id())]
                    except _CLAIMS_REVIEW_METRICS_NONCRITICAL_EXCEPTIONS:
                        user_ids = []
                for user_id in user_ids:
                    try:
                        processed += await asyncio.to_thread(
                            agg_fn,
                            db=media_db,
                            target_user_id=str(user_id),
                            report_date=report_date,
                            lookback_days=lookback_val,
                        )
                    except _CLAIMS_REVIEW_METRICS_NONCRITICAL_EXCEPTIONS as exc:
                        logger.bind(error_type=type(exc).__name__).warning(
                            f"claims_review_metrics: aggregation failed for user {user_id}"
                        )
        except _CLAIMS_REVIEW_METRICS_NONCRITICAL_EXCEPTIONS as exc:
            logger.bind(error_type=type(exc).__name__).warning(
                "claims_review_metrics: failed to create media db"
            )
            return 0
    else:
        user_ids = _enumerate_sqlite_user_ids()
        for user_id in user_ids:
            try:
                with managed_media_database(
                    client_id=str(settings.get("SERVER_CLIENT_ID", "SERVER_API_V1")),
                    db_path=str(DatabasePaths.get_media_db_path(int(user_id))),
                    suppress_init_exceptions=_CLAIMS_REVIEW_METRICS_NONCRITICAL_EXCEPTIONS,
                    suppress_close_exceptions=_CLAIMS_REVIEW_METRICS_NONCRITICAL_EXCEPTIONS,
                ) as user_db:
                    processed += await asyncio.to_thread(
                        agg_fn,
                        db=user_db,
                        target_user_id=str(user_id),
                        report_date=report_date,
                        lookback_days=lookback_val,
                    )
            except _CLAIMS_REVIEW_METRICS_NONCRITICAL_EXCEPTIONS as exc:
                logger.bind(error_type=type(exc).__name__).warning(
                    f"claims_review_metrics: aggregation failed for user {user_id}"
                )
    return processed


async def start_claims_review_metrics_scheduler() -> asyncio.Task | None:
    enabled = is_truthy(os.getenv("CLAIMS_REVIEW_METRICS_SCHEDULER_ENABLED")) or bool(
        settings.get("CLAIMS_REVIEW_METRICS_SCHEDULER_ENABLED", False)
    )
    if not enabled:
        logger.info("Claims review metrics scheduler disabled (CLAIMS_REVIEW_METRICS_SCHEDULER_ENABLED != true)")
        return None
    try:
        interval = int(
            os.getenv("CLAIMS_REVIEW_METRICS_INTERVAL_SEC")
            or settings.get("CLAIMS_REVIEW_METRICS_INTERVAL_SEC", 86400)
        )
    except (TypeError, ValueError):
        interval = 86400

    async def _runner() -> None:
        await asyncio.sleep(min(5, interval))
        while True:
            try:
                await run_claims_review_metrics_once()
            except _CLAIMS_REVIEW_METRICS_NONCRITICAL_EXCEPTIONS as exc:
                logger.bind(error_type=type(exc).__name__).warning(
                    "Claims review metrics scheduler loop error"
                )
            await asyncio.sleep(interval)

    task = asyncio.create_task(_runner(), name="claims_review_metrics_scheduler")
    logger.info(f"Claims review metrics scheduler started (interval={interval}s)")
    return task
