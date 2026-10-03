"""Recurring Claims metrics production through shared Jobs."""

from __future__ import annotations

import asyncio
import contextlib
import math
import os
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from types import MappingProxyType
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from tldw_Server_API.app.core.Claims_Extraction import claims_jobs
from tldw_Server_API.app.core.Claims_Extraction.claims_job_contracts import (
    CLAIMS_JOBS_DEFAULT_QUEUE,
    is_routable_claims_owner_id_text,
)
from tldw_Server_API.app.core.Claims_Extraction.claims_service import (
    _parse_iso_date,
    aggregate_claims_review_extractor_metrics_daily,
)
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.DB_Manager import content_db_settings
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.api import managed_media_database
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.app.core.Jobs.operations.contracts import NoTransitionReason, OperationOutcome
from tldw_Server_API.app.core.Utils.coercion import parse_bool


@dataclass(frozen=True)
class ReviewMetricsSchedulerConfig:
    """Immutable routing and window settings for one scheduler lifecycle."""

    enabled: bool
    mode: str
    interval_seconds: int
    lookback_days: int
    single_user_mode: bool
    job_settings: Mapping[str, Any]


def _setting(key: str, default: Any, source: Mapping[str, Any]) -> Any:
    return os.environ.get(key, source.get(key, default))


def _positive_setting(key: str, default: int, source: Mapping[str, Any]) -> int:
    raw = _setting(key, default, source)
    try:
        if isinstance(raw, bool) or not isinstance(raw, (str, int)):
            raise ValueError("invalid numeric setting")
        value = int(raw)
        if value <= 0:
            raise ValueError("non-positive setting")
        return value
    except (TypeError, ValueError):
        logger.bind(setting=key, normalized=default).warning("Claims review metrics setting defaulted")
        return default


class _SafeIntervalTrigger(IntervalTrigger):
    """End recurring work safely at the datetime representability boundary."""

    def get_next_fire_time(self, previous_fire_time: datetime | None, now: datetime) -> datetime | None:
        try:
            return super().get_next_fire_time(previous_fire_time, now)
        except (OverflowError, ValueError, OSError):
            logger.warning("Claims review metrics interval exceeds supported dates")
            return None


def _interval_trigger(interval: int, now: datetime) -> IntervalTrigger:
    first = now + timedelta(seconds=min(5, interval))
    trigger = _SafeIntervalTrigger(seconds=interval, start_date=first, timezone=timezone.utc)
    first_fire = trigger.get_next_fire_time(None, now)
    if first_fire is None or trigger.get_next_fire_time(first_fire, first_fire) is None:
        raise ValueError("interval exceeds supported dates")
    return trigger


def resolve_scheduler_config(source: Mapping[str, Any] | None = None) -> ReviewMetricsSchedulerConfig:
    """Snapshot configuration with environment precedence and safe bounds."""
    source = settings if source is None else source
    enabled = parse_bool(str(_setting("CLAIMS_REVIEW_METRICS_SCHEDULER_ENABLED", False, source)), default=False)
    interval = _positive_setting("CLAIMS_REVIEW_METRICS_INTERVAL_SEC", 86400, source)
    if interval < 60:
        interval = 60
        logger.bind(setting="CLAIMS_REVIEW_METRICS_INTERVAL_SEC", normalized=interval).warning(
            "Claims review metrics interval raised to minimum"
        )
    lookback = _positive_setting("CLAIMS_REVIEW_METRICS_LOOKBACK_DAYS", 2, source)
    if lookback > 366:
        lookback = 366
        logger.bind(setting="CLAIMS_REVIEW_METRICS_LOOKBACK_DAYS", normalized=lookback).warning(
            "Claims review metrics lookback capped"
        )
    try:
        _interval_trigger(interval, datetime.now(timezone.utc))
    except (ValueError, OverflowError, OSError):
        enabled = False
        logger.bind(setting="CLAIMS_REVIEW_METRICS_INTERVAL_SEC").warning(
            "Claims review metrics scheduler disabled: unsupported interval"
        )
    global_jobs = claims_jobs.claims_jobs_enabled({
        "CLAIMS_JOBS_ENABLED": _setting("CLAIMS_JOBS_ENABLED", False, source),
    })
    metrics_jobs = claims_jobs.claims_review_metrics_jobs_enabled({
        "CLAIMS_JOBS_ENABLED": True,
        "CLAIMS_REVIEW_METRICS_JOBS_ENABLED": _setting("CLAIMS_REVIEW_METRICS_JOBS_ENABLED", False, source),
    })
    if metrics_jobs and not global_jobs:
        logger.warning("Claims review metrics Jobs flag requires global Claims Jobs; using local mode")
    snapshot = {
        key: _setting(key, default, source)
        for key, default in (
            ("CLAIMS_JOBS_QUEUE", CLAIMS_JOBS_DEFAULT_QUEUE),
            ("CLAIMS_JOBS_MAX_RETRIES_REVIEW_METRICS", 3),
        )
    }
    return ReviewMetricsSchedulerConfig(
        enabled,
        "jobs" if metrics_jobs and global_jobs else "local",
        interval,
        lookback,
        str(_setting("AUTH_MODE", "single_user", source)).lower() == "single_user",
        MappingProxyType(snapshot),
    )


def capture_window(now: datetime, interval: int, lookback: int) -> tuple[str, date, date]:
    """Derive a canonical slot and inclusive UTC window from one aware time."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("window timestamp must be timezone aware")
    now = now.astimezone(timezone.utc)
    slot_epoch = math.floor(now.timestamp()) // interval * interval
    scheduled = datetime.fromtimestamp(slot_epoch, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return scheduled, now.date() - timedelta(days=lookback - 1), now.date()


def _enumerate_sqlite_user_ids(*, single_user_mode: bool | None = None) -> list[int]:
    """Discover canonical owner directories without creating databases."""
    try:
        base = DatabasePaths.resolve_user_db_base_dir()
    except Exception as exc:  # noqa: BLE001 - sanitize resolver failures at the discovery boundary.
        logger.bind(error_type=type(exc).__name__).debug("claims_review_metrics: failed to resolve user db base dir")
        return []
    ids = set()
    try:
        for entry in base.iterdir():
            if is_routable_claims_owner_id_text(entry.name) and entry.is_dir():
                if (entry / DatabasePaths.MEDIA_DB_NAME).is_file():
                    ids.add(int(entry.name))
    except FileNotFoundError:
        pass
    if single_user_mode is None:
        single_user_mode = (
            str(os.getenv("AUTH_MODE", settings.get("AUTH_MODE", "single_user"))).lower() == "single_user"
        )
    if not ids and single_user_mode:
        try:
            owner = str(DatabasePaths.get_single_user_id())
            if is_routable_claims_owner_id_text(owner):
                ids.add(int(owner))
        except Exception as exc:  # noqa: BLE001 - sanitize the optional fixed-owner fallback.
            logger.bind(error_type=type(exc).__name__).debug("claims_review_metrics: failed to derive single_user_id")
    return sorted(ids)


def _postgres_owner_page(start_date: date, end_date: date, after: str | None) -> list[str]:
    with scoped_context(user_id=None, is_admin=True):
        with managed_media_database(client_id="claims_review_metrics_discovery", existing_only=True) as db:
            return db.list_claims_review_user_ids_page(
                start_date=start_date, end_date=end_date, after_user_id=after, limit=100
            )


async def _iter_owner_ids(
    config: ReviewMetricsSchedulerConfig, start_date: date, end_date: date, stop_event: asyncio.Event
) -> AsyncIterator[str]:
    """Yield bounded owner pages with synchronous IO off the event loop."""
    if stop_event.is_set():
        return
    if content_db_settings.backend_type == BackendType.SQLITE:
        ids = await asyncio.to_thread(_enumerate_sqlite_user_ids, single_user_mode=config.single_user_mode)
        for owner in ids:
            if stop_event.is_set():
                return
            yield str(owner)
        return
    after = None
    found = False
    page_number = 0
    while not stop_event.is_set():
        page = await asyncio.to_thread(_postgres_owner_page, start_date, end_date, after)
        if not page:
            break
        page_number += 1
        if after is not None and page[-1] <= after:
            raise ValueError("owner discovery cursor did not advance")
        for owner_position, owner in enumerate(page, start=1):
            if stop_event.is_set():
                return
            if is_routable_claims_owner_id_text(owner):
                found = True
                yield owner
            else:
                logger.bind(
                    operation="discover_review_metrics_owners",
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                    page_number=page_number,
                    owner_position=owner_position,
                ).warning("Claims review metrics skipped invalid owner")
        after = page[-1]
        if len(page) < 100:
            break
        await asyncio.sleep(0)
    if not found and config.single_user_mode and not stop_event.is_set():
        owner = str(await asyncio.to_thread(DatabasePaths.get_single_user_id))
        if is_routable_claims_owner_id_text(owner):
            yield owner


def _aggregate_owner(
    owner: str, start_date: date, end_date: date, aggregator: Callable[..., int] | None = None, **legacy_args: Any
) -> int:
    from tldw_Server_API.app.core.Claims_Extraction.claims_review_metrics import aggregate_claims_review_metrics_window

    db_args = {}
    if content_db_settings.backend_type == BackendType.SQLITE:
        owner_dir = DatabasePaths.resolve_user_base_directory(int(owner))
        db_args["db_path"] = str(owner_dir / DatabasePaths.MEDIA_DB_NAME)
    with scoped_context(user_id=int(owner), is_admin=True):
        with managed_media_database(client_id="claims_review_metrics", existing_only=True, **db_args) as db:
            if aggregator is not None:
                return aggregator(db=db, target_user_id=owner, **legacy_args)
            return aggregate_claims_review_metrics_window(
                db=db, owner_user_id=owner, start_date=start_date, end_date=end_date
            )


async def run_claims_review_metrics_once(
    *,
    aggregator: Callable[..., int] | None = None,
    lookback_days: int | None = None,
    report_date: str | None = None,
    db: Any | None = None,
    target_user_id: str | None = None,
) -> int:
    """Preserve the local and caller-owned database compatibility entry point."""
    if db is not None:
        return (aggregator or aggregate_claims_review_extractor_metrics_daily)(
            db=db, target_user_id=target_user_id, report_date=report_date, lookback_days=lookback_days
        )
    config = resolve_scheduler_config()
    try:
        lookback = max(1, min(366, int(lookback_days))) if lookback_days is not None else config.lookback_days
    except (TypeError, ValueError):
        lookback = 2
    _, start, end = capture_window(datetime.now(timezone.utc), config.interval_seconds, lookback)
    parsed_report_date = _parse_iso_date(report_date)
    if parsed_report_date is not None:
        start = end = parsed_report_date
    written = 0
    try:
        async for owner in _iter_owner_ids(config, start, end, asyncio.Event()):
            try:
                written += await asyncio.to_thread(
                    _aggregate_owner, owner, start, end, aggregator, report_date=report_date, lookback_days=lookback
                )
            except Exception as exc:  # noqa: BLE001 - isolate owner failures in the compatibility entry point.
                logger.bind(error_type=type(exc).__name__).warning(
                    "claims_review_metrics: aggregation failed for user {}", owner
                )
    except Exception as exc:  # noqa: BLE001 - keep discovery errors within this background run.
        logger.bind(error_type=type(exc).__name__).warning("claims_review_metrics: failed to create media db")
    return written


async def _retry_delay(stop_event: asyncio.Event, seconds: float) -> None:
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(stop_event.wait(), seconds)


async def run_review_metrics_callback(
    config: ReviewMetricsSchedulerConfig,
    *,
    now: datetime | None = None,
    stop_event: asyncio.Event | None = None,
    job_manager: Any = None,
) -> dict[str, Any]:
    """Fan out one captured window with bounded transient admission retries."""
    from tldw_Server_API.app.core.Claims_Extraction.claims_job_handlers import (
        _is_transient_review_metrics_storage_error,
    )

    stop = stop_event if stop_event is not None else asyncio.Event()
    scheduled, start, end = capture_window(
        now or datetime.now(timezone.utc), config.interval_seconds, config.lookback_days
    )
    summary = {
        "mode": config.mode,
        "scheduled_for": scheduled,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "discovered": 0,
        "accepted": 0,
        "deduplicated": 0,
        "failed": 0,
    }
    try:
        async for owner in _iter_owner_ids(config, start, end, stop):
            if stop.is_set():
                break
            summary["discovered"] += 1
            if config.mode == "local":
                try:
                    await asyncio.to_thread(_aggregate_owner, owner, start, end)
                except FileNotFoundError:
                    pass
                except Exception as exc:  # noqa: BLE001 - one owner must not prevent later aggregation.
                    summary["failed"] += 1
                    logger.bind(owner_user_id=owner, error_type=type(exc).__name__).warning(
                        "Claims review metrics local aggregation failed"
                    )
                await asyncio.sleep(0)
                continue
            for attempt in range(3):
                if stop.is_set():
                    break
                retryable = False
                try:
                    if job_manager is None:
                        job_manager = await asyncio.to_thread(claims_jobs.jobs_manager_from_env)
                    if stop.is_set():
                        break
                    result = await asyncio.to_thread(
                        claims_jobs.enqueue_claims_review_metrics,
                        owner_user_id=owner,
                        scheduled_for=scheduled,
                        start_date=start.isoformat(),
                        end_date=end.isoformat(),
                        interval_seconds=config.interval_seconds,
                        job_manager=job_manager,
                        settings_obj=config.job_settings,
                    )
                    if result.outcome is OperationOutcome.APPLIED:
                        summary["accepted"] += 1
                        break
                    if (
                        result.outcome is OperationOutcome.NO_TRANSITION
                        and result.no_transition_reason is NoTransitionReason.IDEMPOTENT_EXISTING
                    ):
                        summary["deduplicated"] += 1
                        break
                    retryable = result.outcome is OperationOutcome.BACKEND_CONFLICT
                except Exception as exc:  # noqa: BLE001 - classify and sanitize storage/admission failures.
                    retryable = _is_transient_review_metrics_storage_error(exc)
                    logger.bind(owner_user_id=owner, error_type=type(exc).__name__).warning(
                        "Claims review metrics admission failed"
                    )
                if retryable and attempt < 2:
                    await _retry_delay(stop, (0.25, 1.0)[attempt])
                    continue
                summary["failed"] += 1
                break
            await asyncio.sleep(0)
    except Exception as exc:  # noqa: BLE001 - discovery failures wait for the next scheduler callback.
        logger.bind(error_type=type(exc).__name__).warning("Claims review metrics discovery failed")
        summary["discovery_failed"] = True
    logger.bind(**summary).info("Claims review metrics callback completed")
    return summary


async def start_claims_review_metrics_scheduler() -> asyncio.Task | None:
    """Register deferred UTC work and return its lifecycle-owned task."""
    config = resolve_scheduler_config()
    if not config.enabled:
        logger.info("Claims review metrics scheduler disabled")
        return None
    trigger = _interval_trigger(config.interval_seconds, datetime.now(timezone.utc))

    async def runner() -> None:
        stop = asyncio.Event()
        active: set[asyncio.Task] = set()
        scheduler = AsyncIOScheduler(timezone=timezone.utc)

        async def callback() -> None:
            task = asyncio.current_task()
            if task is not None:
                active.add(task)
            try:
                await run_review_metrics_callback(config, stop_event=stop)
            except asyncio.CancelledError:
                stop.set()
                raise
            except Exception as exc:  # noqa: BLE001 - sanitize unanticipated background callback failures.
                logger.bind(error_type=type(exc).__name__).warning("Claims review metrics scheduler callback failed")
            finally:
                if task is not None:
                    active.discard(task)

        scheduler.add_job(
            callback,
            trigger=trigger,
            id="claims_review_metrics",
            max_instances=1,
            coalesce=True,
            misfire_grace_time=None,
        )
        scheduler.start()
        try:
            await stop.wait()
        finally:
            stop.set()
            try:
                scheduler.pause()
                if active:
                    _, pending = await asyncio.wait(tuple(active), timeout=5)
                    for task in pending:
                        task.cancel()
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)
            finally:
                scheduler.shutdown(wait=False)

    return asyncio.create_task(runner(), name="claims_review_metrics_scheduler")
