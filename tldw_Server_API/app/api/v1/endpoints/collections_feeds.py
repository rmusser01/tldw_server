from __future__ import annotations

import contextlib
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Path, Query
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.Watchlists_DB_Deps import get_watchlists_db_for_user
from tldw_Server_API.app.api.v1.schemas.collections_feeds_schemas import (
    CollectionsFeed,
    CollectionsFeedCreateRequest,
    CollectionsFeedsListResponse,
    CollectionsFeedUpdateRequest,
)
from tldw_Server_API.app.api.v1.schemas.pagination import PagePaginationMeta
from tldw_Server_API.app.api.v1.utils.pagination import build_page_pagination_meta
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, User
from tldw_Server_API.app.core.DB_Management.Watchlists_DB import WatchlistsDatabase
from tldw_Server_API.app.core.Personalization import (
    record_watchlist_source_created,
    record_watchlist_source_deleted,
    record_watchlist_source_updated,
)

FEED_ORIGIN = "feed"
_COLLECTIONS_COMPANION_SURFACE = "api.collections"
_FEED_JOB_KEYS = ("collections_feed_job_id", "collections_job_id")
_DEFAULT_HOURLY_CRON = "0 * * * *"
_DEFAULT_DAILY_CRON = "0 0 * * *"

_COLLECTIONS_FEEDS_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    json.JSONDecodeError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    sqlite3.Error,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)

router = APIRouter(prefix="/collections/feeds", tags=["collections-feeds"])

_RSS_EXTENSIONS = {".xml", ".rss", ".atom", ".feed"}
_RSS_PATH_HINTS = {"/feed", "/rss", "/atom", "/feeds", "/index.xml"}


def _detect_source_type(url: str) -> str:
    """Detect source type from URL heuristics. Returns 'rss' or 'site'."""
    try:
        parsed = urlparse(str(url))
        path = (parsed.path or "").lower().rstrip("/")
        for ext in _RSS_EXTENSIONS:
            if path.endswith(ext):
                return "rss"
        for hint in _RSS_PATH_HINTS:
            if path.endswith(hint) or hint + "." in path:
                return "rss"
        query = (parsed.query or "").lower()
        if "format=rss" in query or "format=atom" in query or "feed=" in query:
            return "rss"
    except _COLLECTIONS_FEEDS_NONCRITICAL_EXCEPTIONS:
        pass
    return "rss"


def _parse_settings(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except _COLLECTIONS_FEEDS_NONCRITICAL_EXCEPTIONS:
        return {}


def _is_feed_source(settings: dict[str, Any]) -> bool:
    if settings.get("collections_origin") == FEED_ORIGIN:
        return True
    nested = settings.get("collections")
    return isinstance(nested, dict) and nested.get("origin") == FEED_ORIGIN


def _extract_job_id(settings: dict[str, Any]) -> int | None:
    for key in _FEED_JOB_KEYS:
        val = settings.get(key)
        if val is None:
            continue
        try:
            return int(val)
        except _COLLECTIONS_FEEDS_NONCRITICAL_EXCEPTIONS:
            continue
    nested = settings.get("collections")
    if isinstance(nested, dict) and nested.get("job_id") is not None:
        try:
            return int(nested.get("job_id"))
        except _COLLECTIONS_FEEDS_NONCRITICAL_EXCEPTIONS:
            return None
    return None


def _sanitize_settings(settings: dict[str, Any]) -> dict[str, Any] | None:
    if not settings:
        return None
    cleaned = dict(settings)
    cleaned.pop("collections_origin", None)
    for key in _FEED_JOB_KEYS:
        cleaned.pop(key, None)
    nested = cleaned.get("collections")
    if isinstance(nested, dict):
        nested = dict(nested)
        nested.pop("origin", None)
        nested.pop("job_id", None)
        if nested:
            cleaned["collections"] = nested
        else:
            cleaned.pop("collections", None)
    return cleaned or None


def _default_name(url: str) -> str:
    try:
        host = urlparse(url).hostname
    except _COLLECTIONS_FEEDS_NONCRITICAL_EXCEPTIONS:
        host = None
    return host or url or "Feed"


def _normalize_tz(tz: str | None) -> str:
    if not tz or tz.upper() == "UTC":
        return "UTC"
    t = tz.strip().upper()
    if t.startswith("UTC+") or t.startswith("UTC-"):
        try:
            sign = 1 if t[3] == "+" else -1
            hours = int(t[4:])
            etc_offset = -sign * hours
            return f"Etc/GMT{('+' if etc_offset > 0 else '')}{etc_offset}" if etc_offset != 0 else "Etc/GMT"
        except _COLLECTIONS_FEEDS_NONCRITICAL_EXCEPTIONS:
            return "UTC"
    return tz


def _compute_next_run(cron: str | None, timezone: str | None) -> str | None:
    if not cron:
        return None
    try:
        from apscheduler.triggers.cron import CronTrigger
        tz = _normalize_tz(timezone) or "UTC"
        trigger = CronTrigger.from_crontab(cron, timezone=tz)
        from datetime import datetime
        now = datetime.now(trigger.timezone)
        nxt = trigger.get_next_fire_time(None, now)
        return nxt.isoformat() if nxt else None
    except _COLLECTIONS_FEEDS_NONCRITICAL_EXCEPTIONS:
        return None


def _register_schedule(db: WatchlistsDatabase, job_row, *, current_user: User) -> None:
    if not getattr(job_row, "schedule_expr", None):
        return
    try:
        from tldw_Server_API.app.services.workflows_scheduler import get_workflows_scheduler
        svc = get_workflows_scheduler()
        sid = svc.create(
            tenant_id=str(getattr(current_user, "tenant_id", "default")),
            user_id=str(current_user.id),
            workflow_id=None,
            name=f"watchlist:{job_row.id}:{job_row.name}",
            cron=job_row.schedule_expr,
            timezone=_normalize_tz(job_row.schedule_timezone) or "UTC",
            inputs={"watchlist_job_id": job_row.id},
            run_mode="async",
            validation_mode="block",
            enabled=bool(job_row.active),
            concurrency_mode="queue",
            misfire_grace_sec=300,
            coalesce=True,
        )
        db.set_job_schedule_id(job_row.id, sid)
        return
    except _COLLECTIONS_FEEDS_NONCRITICAL_EXCEPTIONS:
        logger.debug("Collections feeds schedule registration failed")
    try:
        if job_row.schedule_expr:
            from uuid import uuid4

            from tldw_Server_API.app.core.DB_Management.Workflows_Scheduler_DB import WorkflowsSchedulerDB
            sid = uuid4().hex
            wfdb = WorkflowsSchedulerDB(user_id=int(current_user.id))
            wfdb.create_schedule(
                id=sid,
                tenant_id=str(getattr(current_user, "tenant_id", "default")),
                user_id=str(current_user.id),
                workflow_id=None,
                name=f"watchlist:{job_row.id}:{job_row.name}",
                cron=job_row.schedule_expr,
                timezone=_normalize_tz(job_row.schedule_timezone) or "UTC",
                inputs={"watchlist_job_id": job_row.id},
                run_mode="async",
                validation_mode="block",
                enabled=bool(job_row.active),
                concurrency_mode="queue",
                misfire_grace_sec=300,
                coalesce=True,
            )
            db.set_job_schedule_id(job_row.id, sid)
    except _COLLECTIONS_FEEDS_NONCRITICAL_EXCEPTIONS:
        logger.debug("Collections feeds schedule DB fallback failed")


def _derive_health_status(source_row) -> str:
    """Derive health status from source row fields."""
    consec_errors = int(getattr(source_row, "consec_errors", 0) or 0)
    active = bool(getattr(source_row, "active", 1))
    if not active:
        return "disabled"
    if consec_errors >= 5:
        return "failing"
    if consec_errors >= 1:
        return "degraded"
    return "healthy"


def _extract_promoted_at(settings: dict[str, Any], job_row=None) -> str | None:
    """Extract promoted_at from job output_prefs schedule config."""
    if job_row is not None:
        try:
            import json as _json
            prefs = _json.loads(getattr(job_row, "output_prefs_json", None) or "{}")
            schedule_cfg = prefs.get("collections_schedule")
            if isinstance(schedule_cfg, dict) and schedule_cfg.get("promoted"):
                return schedule_cfg.get("promoted_at")
        except (ValueError, TypeError, AttributeError):
            pass
    return None


def _to_feed_response(source_row, *, job_row=None, settings: dict[str, Any] | None = None) -> CollectionsFeed:
    settings = settings if settings is not None else _parse_settings(source_row.settings_json)
    job_id = _extract_job_id(settings)
    return CollectionsFeed(
        id=int(source_row.id),
        name=source_row.name,
        url=source_row.url,
        source_type=str(source_row.source_type or "rss"),
        origin=FEED_ORIGIN,
        tags=list(source_row.tags or []),
        active=bool(source_row.active),
        settings=_sanitize_settings(settings),
        last_scraped_at=source_row.last_scraped_at,
        etag=source_row.etag,
        last_modified=source_row.last_modified,
        defer_until=source_row.defer_until,
        status=source_row.status,
        consec_not_modified=source_row.consec_not_modified,
        consec_errors=int(getattr(source_row, "consec_errors", 0) or 0),
        health_status=_derive_health_status(source_row),
        promoted_at=_extract_promoted_at(settings, job_row=job_row),
        created_at=source_row.created_at,
        updated_at=source_row.updated_at,
        job_id=job_row.id if job_row is not None else job_id,
        schedule_expr=getattr(job_row, "schedule_expr", None),
        timezone=getattr(job_row, "schedule_timezone", None),
        job_active=getattr(job_row, "active", None),
        next_run_at=getattr(job_row, "next_run_at", None),
        wf_schedule_id=getattr(job_row, "wf_schedule_id", None),
    )


def _load_job(db: WatchlistsDatabase, settings: dict[str, Any]):
    job_id = _extract_job_id(settings)
    if not job_id:
        return None
    try:
        return db.get_job(int(job_id))
    except _COLLECTIONS_FEEDS_NONCRITICAL_EXCEPTIONS:
        return None


def _merge_settings(existing: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in (updates or {}).items():
        merged[key] = value
    merged["collections_origin"] = FEED_ORIGIN
    for key in _FEED_JOB_KEYS:
        if key in existing:
            merged[key] = existing.get(key)
    existing_nested = existing.get("collections") if isinstance(existing.get("collections"), dict) else {}
    update_nested = merged.get("collections") if isinstance(merged.get("collections"), dict) else {}
    nested = dict(existing_nested)
    nested.update(update_nested)
    if nested:
        nested["origin"] = FEED_ORIGIN
        if "job_id" in existing_nested:
            nested["job_id"] = existing_nested.get("job_id")
        merged["collections"] = nested
    return merged


def _ensure_collections_schedule(
    output_prefs: dict[str, Any],
    *,
    mode: str,
    daily_expr: str,
    promote_after_hours: int,
) -> dict[str, Any]:
    if not isinstance(output_prefs, dict):
        output_prefs = {}
    schedule_cfg = output_prefs.get("collections_schedule")
    if not isinstance(schedule_cfg, dict):
        schedule_cfg = {}
    schedule_cfg["mode"] = mode
    if mode == "hourly_then_daily":
        schedule_cfg.setdefault("daily_expr", daily_expr)
        schedule_cfg.setdefault("promote_after_hours", promote_after_hours)
        schedule_cfg.setdefault("promoted", False)
    output_prefs["collections_schedule"] = schedule_cfg
    return output_prefs


def _sync_job_schedule(db: WatchlistsDatabase, job_row, *, current_user: User) -> None:
    try:
        if job_row.wf_schedule_id:
            from tldw_Server_API.app.services.workflows_scheduler import get_workflows_scheduler
            svc = get_workflows_scheduler()
            svc.update(
                job_row.wf_schedule_id,
                {
                    "cron": job_row.schedule_expr or "* * * * *",
                    "timezone": _normalize_tz(job_row.schedule_timezone) or "UTC",
                    "enabled": bool(job_row.active),
                },
            )
            return
    except _COLLECTIONS_FEEDS_NONCRITICAL_EXCEPTIONS:
        logger.debug("Collections feeds schedule update failed")
    if not job_row.wf_schedule_id and job_row.schedule_expr:
        _register_schedule(db, job_row, current_user=current_user)
    try:
        return db.get_job(job_row.id)
    except _COLLECTIONS_FEEDS_NONCRITICAL_EXCEPTIONS:
        return None


def _collections_feed_event_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _feeds_pagination(*, total: int, page: int, size: int) -> PagePaginationMeta:
    total_pages = (total + size - 1) // size if total > 0 else 0
    return build_page_pagination_meta(
        page=page,
        per_page=size,
        total=total,
        total_pages=total_pages,
    )


@router.post("", response_model=CollectionsFeed, summary="Create a Collections feed subscription")
async def create_feed_subscription(
    payload: CollectionsFeedCreateRequest = Body(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: User = Depends(get_request_user),
    db: WatchlistsDatabase = Depends(get_watchlists_db_for_user),
) -> CollectionsFeed:
    settings = payload.settings if isinstance(payload.settings, dict) else {}
    settings = dict(settings)
    settings["collections_origin"] = FEED_ORIGIN
    name = payload.name.strip() if isinstance(payload.name, str) and payload.name.strip() else _default_name(str(payload.url))
    tags = [t for t in (payload.tags or []) if t]
    schedule_expr = payload.schedule_expr or _DEFAULT_HOURLY_CRON
    schedule_timezone = payload.timezone
    detected_type = _detect_source_type(str(payload.url))
    try:
        source = db.create_source(
            name=name,
            url=str(payload.url),
            source_type=detected_type,
            active=payload.active,
            settings_json=json.dumps(settings),
            tags=tags,
            group_ids=None,
        )
    except _COLLECTIONS_FEEDS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("collections_feeds_create_source_failed")
        raise HTTPException(status_code=400, detail="feed_create_failed") from exc

    try:
        output_prefs = {"collections_origin": FEED_ORIGIN}
        if payload.schedule_expr is None:
            output_prefs = _ensure_collections_schedule(
                output_prefs,
                mode="hourly_then_daily",
                daily_expr=_DEFAULT_DAILY_CRON,
                promote_after_hours=24,
            )
        job = db.create_job(
            name=f"Feed: {name}",
            description=f"Collections feed subscription for {source.url}",
            scope_json=json.dumps({"sources": [int(source.id)]}),
            schedule_expr=schedule_expr,
            schedule_timezone=schedule_timezone,
            active=payload.active,
            max_concurrency=None,
            per_host_delay_ms=None,
            retry_policy_json=json.dumps({}),
            output_prefs_json=json.dumps(output_prefs),
            job_filters_json=None,
        )
    except _COLLECTIONS_FEEDS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("collections_feeds_create_job_failed")
        with contextlib.suppress(_COLLECTIONS_FEEDS_NONCRITICAL_EXCEPTIONS):
            db.delete_source(int(source.id))
        raise HTTPException(status_code=400, detail="feed_create_failed") from exc

    settings["collections_feed_job_id"] = int(job.id)
    try:
        db.update_source(int(source.id), {"settings_json": json.dumps(settings)})
        source = db.get_source(int(source.id))
    except _COLLECTIONS_FEEDS_NONCRITICAL_EXCEPTIONS:
        pass
    next_run = _compute_next_run(job.schedule_expr, job.schedule_timezone)
    if next_run:
        db.set_job_history(job_id=int(job.id), next_run_at=next_run)
        job = db.get_job(int(job.id))
    _register_schedule(db, job, current_user=current_user)
    job = db.get_job(int(job.id))

    response = _to_feed_response(source, job_row=job, settings=settings)
    record_watchlist_source_created(
        user_id=current_user.id,
        source=response,
        route="/api/v1/collections/feeds",
        surface=_COLLECTIONS_COMPANION_SURFACE,
    )

    if payload.active:
        async def _run_first_job(user_id: int, job_id: int) -> None:
            try:
                from tldw_Server_API.app.core.Watchlists.pipeline import run_watchlist_job
                await run_watchlist_job(
                    user_id,
                    job_id,
                    capture_companion_activity=True,
                    companion_route="/api/v1/collections/feeds",
                )
            except _COLLECTIONS_FEEDS_NONCRITICAL_EXCEPTIONS:
                logger.debug("collections_feeds_first_run_failed")

        background_tasks.add_task(_run_first_job, int(current_user.id), int(job.id))

    return response


@router.get("", response_model=CollectionsFeedsListResponse, summary="List Collections feed subscriptions")
async def list_feed_subscriptions(
    q: str | None = Query(None),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=200),
    current_user: User = Depends(get_request_user),
    db: WatchlistsDatabase = Depends(get_watchlists_db_for_user),
) -> CollectionsFeedsListResponse:
    offset = max(0, (page - 1) * size)
    limit = max(size, 100)
    collected: list[tuple[Any, dict[str, Any]]] = []
    total_sources = 0
    fetch_offset = 0
    while True:
        rows, total = db.list_sources(q=q, tag_names=None, limit=limit, offset=fetch_offset)
        total_sources = total
        if not rows:
            break
        for row in rows:
            settings = _parse_settings(row.settings_json)
            if not _is_feed_source(settings):
                continue
            collected.append((row, settings))
        fetch_offset += limit
        if fetch_offset >= total_sources:
            break
    total = len(collected)
    paged = collected[offset: offset + size]
    items: list[CollectionsFeed] = []
    for row, settings in paged:
        job = _load_job(db, settings)
        items.append(_to_feed_response(row, job_row=job, settings=settings))
    return CollectionsFeedsListResponse(
        items=items,
        total=total,
        pagination=_feeds_pagination(total=total, page=page, size=size),
    )


@router.get("/{feed_id}", response_model=CollectionsFeed, summary="Get a Collections feed subscription")
async def get_feed_subscription(
    feed_id: int = Path(..., ge=1),
    current_user: User = Depends(get_request_user),
    db: WatchlistsDatabase = Depends(get_watchlists_db_for_user),
) -> CollectionsFeed:
    try:
        source = db.get_source(feed_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="feed_not_found") from None
    settings = _parse_settings(source.settings_json)
    if not _is_feed_source(settings):
        raise HTTPException(status_code=404, detail="feed_not_found")
    job = _load_job(db, settings)
    return _to_feed_response(source, job_row=job, settings=settings)


@router.patch("/{feed_id}", response_model=CollectionsFeed, summary="Update a Collections feed subscription")
async def update_feed_subscription(
    feed_id: int = Path(..., ge=1),
    payload: CollectionsFeedUpdateRequest = Body(...),
    current_user: User = Depends(get_request_user),
    db: WatchlistsDatabase = Depends(get_watchlists_db_for_user),
) -> CollectionsFeed:
    try:
        source = db.get_source(feed_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="feed_not_found") from None
    settings = _parse_settings(source.settings_json)
    if not _is_feed_source(settings):
        raise HTTPException(status_code=404, detail="feed_not_found")

    if payload.settings is not None and isinstance(payload.settings, dict):
        settings = _merge_settings(settings, payload.settings)
    settings["collections_origin"] = FEED_ORIGIN
    activity_patch = payload.model_dump(exclude_unset=True)
    patch: dict[str, Any] = {"settings_json": json.dumps(settings)}
    if payload.name is not None:
        name = payload.name.strip()
        if not name:
            raise HTTPException(status_code=400, detail="feed_name_required")
        patch["name"] = name
    if payload.url is not None:
        patch["url"] = str(payload.url)
    if payload.active is not None:
        patch["active"] = payload.active
    try:
        source = db.update_source(feed_id, patch)
    except _COLLECTIONS_FEEDS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("collections_feeds_update_source_failed")
        raise HTTPException(status_code=400, detail="feed_update_failed") from exc
    if payload.tags is not None:
        try:
            db.set_source_tags(feed_id, [t for t in payload.tags if t])
            source = db.get_source(feed_id)
        except _COLLECTIONS_FEEDS_NONCRITICAL_EXCEPTIONS:
            logger.error("collections_feeds_update_tags_failed")

    job = _load_job(db, settings)
    if job and any(value is not None for value in (payload.schedule_expr, payload.timezone, payload.active)):
        job_patch: dict[str, Any] = {}
        if payload.schedule_expr is not None:
            job_patch["schedule_expr"] = payload.schedule_expr
        if payload.timezone is not None:
            job_patch["schedule_timezone"] = payload.timezone
        if payload.active is not None:
            job_patch["active"] = payload.active

        if payload.schedule_expr is not None:
            try:
                output_prefs = json.loads(job.output_prefs_json or "{}")
                output_prefs = _ensure_collections_schedule(
                    output_prefs,
                    mode="manual",
                    daily_expr=_DEFAULT_DAILY_CRON,
                    promote_after_hours=24,
                )
                job_patch["output_prefs_json"] = json.dumps(output_prefs)
            except _COLLECTIONS_FEEDS_NONCRITICAL_EXCEPTIONS:
                pass

        try:
            job = db.update_job(int(job.id), job_patch)
            if any(k in job_patch for k in ("schedule_expr", "schedule_timezone")):
                next_run = _compute_next_run(job.schedule_expr, job.schedule_timezone)
                db.set_job_history(int(job.id), next_run_at=next_run)
                job = db.get_job(int(job.id))
            _sync_job_schedule(db, job, current_user=current_user)
        except _COLLECTIONS_FEEDS_NONCRITICAL_EXCEPTIONS:
            logger.error("collections_feeds_update_job_failed")

    response = _to_feed_response(source, job_row=job, settings=settings)
    if activity_patch:
        record_watchlist_source_updated(
            user_id=current_user.id,
            source=response,
            patch=activity_patch,
            event_timestamp=_collections_feed_event_timestamp(),
            route=f"/api/v1/collections/feeds/{feed_id}",
            surface=_COLLECTIONS_COMPANION_SURFACE,
        )
    return response


@router.delete("/{feed_id}", summary="Delete a Collections feed subscription")
async def delete_feed_subscription(
    feed_id: int = Path(..., ge=1),
    current_user: User = Depends(get_request_user),
    db: WatchlistsDatabase = Depends(get_watchlists_db_for_user),
):
    try:
        source = db.get_source(feed_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="feed_not_found") from None
    settings = _parse_settings(source.settings_json)
    if not _is_feed_source(settings):
        raise HTTPException(status_code=404, detail="feed_not_found")
    job_id = _extract_job_id(settings)
    if job_id:
        try:
            job = db.get_job(job_id)
            if getattr(job, "wf_schedule_id", None):
                from tldw_Server_API.app.services.workflows_scheduler import get_workflows_scheduler
                get_workflows_scheduler().delete(job.wf_schedule_id)  # type: ignore[arg-type]
        except _COLLECTIONS_FEEDS_NONCRITICAL_EXCEPTIONS:
            pass
        with contextlib.suppress(_COLLECTIONS_FEEDS_NONCRITICAL_EXCEPTIONS):
            db.delete_job(job_id)
    source_response = _to_feed_response(source, settings=settings)
    event_timestamp = _collections_feed_event_timestamp()
    deleted = db.delete_source(feed_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="feed_not_found")
    record_watchlist_source_deleted(
        user_id=current_user.id,
        source=source_response,
        event_timestamp=event_timestamp,
        route=f"/api/v1/collections/feeds/{feed_id}",
        surface=_COLLECTIONS_COMPANION_SURFACE,
        hard_delete=True,
    )
    return {"success": True}
