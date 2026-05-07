from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import TokenScopeGuard
from tldw_Server_API.app.api.v1.API_Deps.backpressure import (
    guard_backpressure_and_quota,
)
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.personalization_deps import (
    UsageEventLogger,
    get_usage_event_logger,
)
from tldw_Server_API.app.api.v1.schemas.media_request_models import (
    IngestWebContentRequest,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.chunking_options import (
    async_resolve_chunking_options_and_plan,
    attach_chunking_plan_to_result,
)
from tldw_Server_API.app.services.web_scraping_service import (
    ingest_web_content_orchestrate,
)

router = APIRouter()


@router.post(
    "/ingest-web-content",
    dependencies=[
        Depends(guard_backpressure_and_quota),
        Depends(
            TokenScopeGuard(
                "any",
                require_if_present=True,
                endpoint_id="media.ingest",
                count_as="call",
            )
        ),
    ],
)
async def ingest_web_content(
    request: IngestWebContentRequest,
    background_tasks: BackgroundTasks,  # Parity with legacy signature
    token: str = Header(..., description="Authentication token"),
    db: Any = Depends(get_media_db_for_user),
    usage_log: UsageEventLogger = Depends(get_usage_event_logger),
) -> dict[str, Any]:
    """
    Ingest and process web content from various scraping strategies.

    Supports:
      - individual: each URL scraped independently
      - sitemap:   treat the first URL as a sitemap and scrape it
      - url_level: scrape pages up to a path depth from the base URL
      - recursive: crawl links up to max_depth / max_pages
    """

    # Basic validation: require at least one URL.
    if not request.urls:
        raise HTTPException(status_code=400, detail="At least one URL is required")

    # Shared usage logging, topic monitoring, and per-method scraping are
    # handled by the orchestration helper.
    raw_results: list[dict[str, Any]] = []
    try:
        helper_results = await ingest_web_content_orchestrate(
            request=request,
            db=db,
            usage_log=usage_log,
        )
    except HTTPException:
        # Preserve explicit HTTP errors from downstream helpers (e.g., cookie
        # parsing / validation) so client-facing 4xx semantics are not
        # converted into generic 500s.
        raise
    except Exception:  # noqa: BLE001
        logger.error("Web content ingestion failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to ingest web content",
        ) from None
    if helper_results:
        raw_results.extend(helper_results)

    # Scrape method validation / logging.
    scrape_method = request.scrape_method
    logger.info("Selected scrape method: {}", scrape_method)

    if not raw_results:
        return {
            "status": "warning",
            "message": "No articles were successfully scraped for this request.",
            "results": [],
        }

    # Optional translation stub (kept for behavioural parity with legacy).
    if request.perform_translation:
        logger.info(
            "Translating to {} (placeholder).",
            request.translation_language,
        )
        # Real translation logic would go here.

    # Optional chunking stub (kept for behavioural parity with legacy).
    if request.perform_chunking:
        logger.info("Performing chunking on each article (placeholder).")
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            _, chunking_plan = await async_resolve_chunking_options_and_plan(
                request,
                media_type="web",
                source_name=str(item.get("url") or ""),
                extracted_text=content if isinstance(content, str) else None,
            )
            attach_chunking_plan_to_result(item, chunking_plan)

    # Timestamp results when requested.
    if request.timestamp_option:
        timestamp_str = datetime.now().isoformat()
        for item in raw_results:
            item["ingested_at"] = timestamp_str

    return {
        "status": "success",
        "message": "Web content processed",
        "count": len(raw_results),
        "results": raw_results,
    }


__all__ = ["router"]
