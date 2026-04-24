from __future__ import annotations

import contextlib
import inspect
from collections.abc import Awaitable, Callable
from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import rbac_rate_limit, require_permissions
from tldw_Server_API.app.api.v1.API_Deps.billing_deps import require_within_limit
from tldw_Server_API.app.core.Billing.enforcement import LimitCategory
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.personalization_deps import (
    UsageEventLogger,
    get_usage_event_logger,
)
from tldw_Server_API.app.api.v1.schemas.media_request_models import WebScrapingRequest
from tldw_Server_API.app.core.AuthNZ.permissions import MEDIA_CREATE
from tldw_Server_API.app.services.web_scraping_service import process_web_scraping_task

router = APIRouter()

WebScrapingTask = Callable[..., Awaitable[Any]]


def _resolve_process_web_scraping_task() -> WebScrapingTask:
    """Return the active web scraping task, honoring the media shim when patched."""
    # Compatibility shim: older integration tests monkeypatch
    # media.process_web_scraping_task at package scope.
    from tldw_Server_API.app.api.v1.endpoints import media as media_mod

    shim_task = getattr(media_mod, "process_web_scraping_task", None)
    if callable(shim_task) and (
        inspect.iscoroutinefunction(shim_task)
        or inspect.iscoroutinefunction(getattr(shim_task, "__call__", None))
    ):
        return cast(WebScrapingTask, shim_task)
    if shim_task is not None:
        logger.warning(
            "Ignoring non-async media.process_web_scraping_task shim: {}",
            shim_task,
        )
    return process_web_scraping_task


@router.post(
    "/process-web-scraping",
    dependencies=[
        Depends(require_permissions(MEDIA_CREATE)),
        Depends(rbac_rate_limit("media.create")),
        Depends(require_within_limit(LimitCategory.STORAGE_MB, 1)),
        Depends(require_within_limit(LimitCategory.API_CALLS_DAY, 1)),
    ],
)
async def process_web_scraping_endpoint(
    payload: WebScrapingRequest,
    db: Any = Depends(get_media_db_for_user),
    usage_log: UsageEventLogger = Depends(get_usage_event_logger),
):
    """
    Ingest / scrape data from websites or sitemaps, optionally summarize,
    then either store ephemeral or persist in DB.

    This is the modular implementation of `/process-web-scraping`, mirroring
    the legacy behavior while routing through a local resolver seam so tests
    can patch deterministic task resolution.
    """
    try:
        # Usage logging is best-effort; never fail the request.
        with contextlib.suppress(Exception):
            usage_log.log_event(
                "webscrape.process",
                tags=[str(payload.scrape_method or "")],
                metadata={
                    "mode": payload.mode,
                    "max_pages": payload.max_pages,
                    "max_depth": payload.max_depth,
                },
            )

        task = _resolve_process_web_scraping_task()

        result = await task(
            scrape_method=payload.scrape_method,
            url_input=payload.url_input,
            url_level=payload.url_level,
            max_pages=payload.max_pages,
            max_depth=payload.max_depth,
            summarize_checkbox=payload.summarize_checkbox,
            custom_prompt=payload.custom_prompt,
            api_name=payload.api_name,
            api_key=None,  # API key retrieved from server config
            keywords=payload.keywords or "",
            custom_titles=payload.custom_titles,
            system_prompt=payload.system_prompt,
            temperature=payload.temperature,
            custom_cookies=payload.custom_cookies,
            mode=payload.mode,
            user_id=(
                getattr(getattr(db, "user", None), "id", None)
                if db is not None
                else None
            ),
            user_agent=payload.user_agent,
            custom_headers=payload.custom_headers,
            crawl_strategy=payload.crawl_strategy,
            include_external=payload.include_external,
            score_threshold=payload.score_threshold,
        )
        return result
    except HTTPException:
        # Preserve downstream HTTP status codes (validation, upstream errors).
        raise
    except Exception:  # pragma: no cover - defensive path
        error_detail = "Web scraping failed due to an internal error."
        logger.exception("Web scraping endpoint error")
        with contextlib.suppress(Exception):
            logger.error(
                "Request details - scrape_method: {}, url_input: {}",
                payload.scrape_method,
                (payload.url_input[:100] if payload.url_input else "None"),
            )
        raise HTTPException(status_code=500, detail=error_detail) from None


__all__ = ["router"]
