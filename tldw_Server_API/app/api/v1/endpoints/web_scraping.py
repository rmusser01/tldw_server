# web_scraping.py - Web Scraping Management Endpoints
"""
Additional endpoints for managing the enhanced web scraping service.
Provides job management, status checking, and service control.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, User

from tldw_Server_API.app.core.Metrics import get_metrics_registry
from tldw_Server_API.app.core.Security.url_validation import assert_url_safe
from tldw_Server_API.app.services.enhanced_web_scraping_service import get_web_scraping_service

router = APIRouter(
    prefix="/web-scraping",
    tags=["web-scraping"],
)

_WEB_SCRAPING_ENDPOINT_EXCEPTIONS = (
    AttributeError,
    ConnectionError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


@router.get("/status")
async def get_scraping_service_status(
    current_user: User = Depends(get_request_user)
) -> dict[str, Any]:
    """
    Get the status of the web scraping service including queue statistics.

    Returns:
        Service status including:
        - Initialization status
        - Queue statistics (active, pending, completed jobs)
        - Rate limiting configuration
    """
    try:
        service = get_web_scraping_service()
        return service.get_service_status()
    except _WEB_SCRAPING_ENDPOINT_EXCEPTIONS as e:
        logger.error("Failed to get scraping service status")
        raise HTTPException(status_code=500, detail="Failed to get scraping service status") from e


@router.get("/job/{job_id}")
async def get_scraping_job_status(
    job_id: str,
    current_user: User = Depends(get_request_user)
) -> dict[str, Any]:
    """
    Get the status of a specific scraping job.

    Args:
        job_id: The ID of the scraping job

    Returns:
        Job details including status, progress, and results
    """
    try:
        service = get_web_scraping_service()
        return await service.get_job_status(job_id, current_user)
    except _WEB_SCRAPING_ENDPOINT_EXCEPTIONS as e:
        logger.error("Failed to get scraping job status")
        raise HTTPException(status_code=500, detail="Failed to get scraping job status") from e


@router.delete("/job/{job_id}")
async def cancel_scraping_job(
    job_id: str,
    current_user: User = Depends(get_request_user)
) -> dict[str, Any]:
    """
    Cancel a pending or active scraping job.

    Args:
        job_id: The ID of the scraping job to cancel

    Returns:
        Cancellation status
    """
    try:
        service = get_web_scraping_service()
        return await service.cancel_job(job_id, current_user)
    except _WEB_SCRAPING_ENDPOINT_EXCEPTIONS as e:
        logger.error("Failed to cancel scraping job")
        raise HTTPException(status_code=500, detail="Failed to cancel scraping job") from e


@router.post("/service/initialize")
async def initialize_scraping_service(
    current_user: User = Depends(get_request_user)
) -> dict[str, Any]:
    """
    Initialize the web scraping service if not already initialized.

    This starts the worker pool and prepares the service for scraping.
    """
    try:
        service = get_web_scraping_service()
        await service.initialize()
        return {
            "status": "success",
            "message": "Web scraping service initialized",
            "service_status": service.get_service_status()
        }
    except _WEB_SCRAPING_ENDPOINT_EXCEPTIONS as e:
        logger.error("Failed to initialize scraping service")
        raise HTTPException(status_code=500, detail="Failed to initialize scraping service") from e


@router.post("/service/shutdown")
async def shutdown_scraping_service(
    current_user: User = Depends(get_request_user)
) -> dict[str, Any]:
    """
    Shutdown the web scraping service gracefully.

    This stops all workers and cleans up resources.
    """
    try:
        service = get_web_scraping_service()
        await service.shutdown()
        return {
            "status": "success",
            "message": "Web scraping service shutdown completed"
        }
    except _WEB_SCRAPING_ENDPOINT_EXCEPTIONS as e:
        logger.error("Failed to shutdown scraping service")
        raise HTTPException(status_code=500, detail="Failed to shutdown scraping service") from e


@router.get("/progress/{task_id}")
async def get_scraping_progress(
    task_id: str,
    current_user: User = Depends(get_request_user)
) -> dict[str, Any]:
    """
    Get progress information for a scraping task.

    Useful for long-running recursive or sitemap scraping tasks.

    Args:
        task_id: The task identifier

    Returns:
        Progress information including pages scraped, remaining, current URL, etc.
    """
    try:
        service = get_web_scraping_service()
        if not service._initialized:
            raise HTTPException(status_code=503, detail="Service not initialized")

        progress = service.scraper.get_progress(task_id)
        if not progress:
            raise HTTPException(status_code=404, detail="Task not found or no progress available")

        return {
            "task_id": task_id,
            "progress": progress,
            "status": "in_progress"
        }
    except HTTPException:
        raise
    except _WEB_SCRAPING_ENDPOINT_EXCEPTIONS as e:
        logger.error("Failed to get scraping progress")
        raise HTTPException(status_code=500, detail="Failed to get scraping progress") from e


@router.get("/cookies/{domain}")
async def get_cookies_for_domain(
    domain: str,
    current_user: User = Depends(get_request_user)
) -> dict[str, Any]:
    """
    Get stored cookies for a specific domain.

    Args:
        domain: The domain to get cookies for

    Returns:
        List of cookies for the domain
    """
    try:
        service = get_web_scraping_service()
        if not service._initialized:
            await service.initialize()

        cookies = service.scraper.cookie_manager.get_cookies(f"https://{domain}")

        return {
            "domain": domain,
            "cookies": cookies or [],
            "cookie_count": len(cookies) if cookies else 0
        }
    except _WEB_SCRAPING_ENDPOINT_EXCEPTIONS as e:
        logger.error("Failed to get cookies for domain")
        raise HTTPException(status_code=500, detail="Failed to get cookies for domain") from e


@router.post("/cookies/{domain}")
async def set_cookies_for_domain(
    domain: str,
    cookies: list[dict[str, Any]],
    current_user: User = Depends(get_request_user)
) -> dict[str, Any]:
    """
    Set cookies for a specific domain.

    Useful for handling authentication or paywalled content.

    Args:
        domain: The domain to set cookies for
        cookies: List of cookie dictionaries

    Returns:
        Success status
    """
    try:
        service = get_web_scraping_service()
        if not service._initialized:
            await service.initialize()

        service.scraper.cookie_manager.add_cookies(domain, cookies)

        return {
            "status": "success",
            "message": f"Added {len(cookies)} cookies for {domain}",
            "domain": domain
        }
    except _WEB_SCRAPING_ENDPOINT_EXCEPTIONS as e:
        logger.error("Failed to set cookies for domain")
        raise HTTPException(status_code=500, detail="Failed to set cookies for domain") from e


@router.get("/duplicates/check")
async def check_url_duplicate(
    url: str = Query(..., description="URL to check for duplicate content"),
    current_user: User = Depends(get_request_user)
) -> dict[str, Any]:
    """
    Check if a URL's content has already been scraped (duplicate detection).

    Args:
        url: The URL to check

    Returns:
        Duplicate status and information about the original if found
    """
    try:
        # SSRF guard
        try:
            assert_url_safe(url)
        except HTTPException:
            get_metrics_registry().increment("security_ssrf_block_total", 1)
            raise

        service = get_web_scraping_service()
        if not service._initialized:
            await service.initialize()

        # For checking, we'd need to get the content first
        # This is a simplified check - in production you might want to
        # check against URL patterns or pre-fetch headers

        return {
            "url": url,
            "is_duplicate": False,  # Placeholder
            "message": "Duplicate checking requires content analysis"
        }
    except _WEB_SCRAPING_ENDPOINT_EXCEPTIONS as e:
        logger.error("Failed to check URL duplicate")
        raise HTTPException(status_code=500, detail="Failed to check URL duplicate") from e


# Include this router in your main app
# In main.py or wherever you configure routes:
# app.include_router(web_scraping_router, prefix="/api/v1")
