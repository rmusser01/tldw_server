# enhanced_web_scraping_service.py - Enhanced Web Scraping Service
"""
Enhanced web scraping service that integrates the production scraping pipeline
with the existing tldw_server API structure.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from fastapi import HTTPException
from loguru import logger

from tldw_Server_API.app.core.Chunking.chunker import Chunker
from tldw_Server_API.app.core.config import load_and_log_configs
from tldw_Server_API.app.core.DB_Management.db_path_utils import get_user_media_db_path
from tldw_Server_API.app.core.DB_Management.media_db.api import managed_media_database
from tldw_Server_API.app.core.Ingestion_Media_Processing.chunking_options import (
    async_resolve_chunking_options_and_plan,
    attach_chunking_plan_to_result,
)
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze
from tldw_Server_API.app.core.Metrics import get_metrics_registry
from tldw_Server_API.app.core.testing import is_truthy
from tldw_Server_API.app.core.Utils.prompt_loader import load_prompt
from tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib import ContentMetadataHandler, is_content_page

# Import the enhanced scraper
from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import (
    EnhancedWebScraper,
    JobPriority,
    JobStatus,
    ScrapingJob,
    create_enhanced_scraper,
)

# Import existing components
from tldw_Server_API.app.services.ephemeral_store import ephemeral_storage

_WEB_SCRAPE_CONFIG_PARSE_EXCEPTIONS = (TypeError, ValueError)
_WEB_SCRAPE_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    ConnectionError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    json.JSONDecodeError,
)
_PLATFORM_ADMIN_ROLES = frozenset({"admin"})
_ADMIN_CLAIM_PERMISSIONS = frozenset({"*", "system.configure"})


def _web_chunking_form(
    *,
    perform_chunking: bool,
    chunking_mode: str | None,
    auto_chunking_goal: str,
    auto_chunking_use_llm: bool,
    api_name: str | None = None,
    api_provider: str | None = None,
    model_name: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        media_type="web",
        perform_chunking=perform_chunking,
        chunking_mode=chunking_mode,
        auto_chunking_goal=auto_chunking_goal,
        auto_chunking_use_llm=auto_chunking_use_llm,
        api_name=api_name,
        api_provider=api_provider,
        model_name=model_name,
        chunk_method=None,
        chunk_size=500,
        chunk_overlap=200,
        chunk_language=None,
        use_adaptive_chunking=False,
        use_multi_level_chunking=False,
        hierarchical_chunking=False,
        hierarchical_template=None,
    )


def _normalized_claim_values(values: Any) -> set[str]:
    if not isinstance(values, (list, tuple, set)):
        return set()
    return {str(value).strip().lower() for value in values if str(value).strip()}


class WebScrapingService:
    """Enhanced web scraping service with production features"""

    def __init__(self):
        self.scraper: Optional[EnhancedWebScraper] = None
        self._initialized = False
        self._active_jobs: dict[str, ScrapingJob] = {}

    async def initialize(self):
        """Initialize the scraping service"""
        if not self._initialized:
            try:
                self.scraper = await create_enhanced_scraper()
                self._initialized = True
                logger.info("Web scraping service initialized with Playwright")
            except ImportError as e:
                logger.warning(f"Playwright not available: {e}. Service will use basic scraping only.")
                self.scraper = None  # Will use fallback methods
                self._initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize enhanced scraper: {e}")
                import traceback

                logger.error(f"Traceback: {traceback.format_exc()}")
                raise

    async def shutdown(self):
        """Shutdown the scraping service"""
        if self.scraper:
            await self.scraper.stop()
            self._initialized = False
            logger.info("Web scraping service shutdown")

    async def process_web_scraping_task(
        self,
        scrape_method: str,
        url_input: str,
        url_level: Optional[int] = None,
        max_pages: Optional[int] = None,
        max_depth: int = 3,
        summarize_checkbox: bool = False,
        custom_prompt: Optional[str] = None,
        api_name: Optional[str] = None,
        api_key: Optional[str] = None,
        keywords: str = "",
        custom_titles: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        custom_cookies: Optional[list[dict[str, Any]]] = None,
        mode: str = "persist",
        priority: str = "normal",
        user_id: Optional[int] = None,
        user_agent: Optional[str] = None,
        custom_headers: Optional[dict[str, str]] = None,
        # Crawl overrides from UI/clients
        crawl_strategy: Optional[str] = None,
        include_external: Optional[bool] = None,
        score_threshold: Optional[float] = None,
        perform_chunking: bool = True,
        chunking_mode: Optional[str] = None,
        auto_chunking_goal: str = "balanced",
        auto_chunking_use_llm: bool = False,
    ) -> dict[str, Any]:
        """
        Process web scraping task with enhanced features.

        This replaces the placeholder implementation with production features:
        - Concurrent scraping with rate limiting
        - Job queue management
        - Progress tracking
        - Content deduplication
        - Robust error handling
        """
        # Ensure service is initialized
        if not self._initialized:
            await self.initialize()

        try:
            # Read crawl feature flags (env > config.txt); keep behavior unchanged if params provided
            cfg = load_and_log_configs() or {}
            wc = cfg.get("web_scraper", {}) if isinstance(cfg, dict) else {}

            def _as_bool(v: Any, d: bool) -> bool:
                try:
                    s = str(v).strip().lower()
                    if is_truthy(s):
                        return True
                    if s in {"0", "false", "no", "off", "n"}:
                        return False
                except _WEB_SCRAPE_CONFIG_PARSE_EXCEPTIONS:
                    pass
                return d

            def _as_int(v: Any, d: int) -> int:
                try:
                    return int(v)
                except _WEB_SCRAPE_CONFIG_PARSE_EXCEPTIONS:
                    return d

            def _as_float(v: Any, d: float) -> float:
                try:
                    return float(v)
                except _WEB_SCRAPE_CONFIG_PARSE_EXCEPTIONS:
                    return d

            crawl_strategy_cfg: str = str(wc.get("web_crawl_strategy", "default"))
            include_external_cfg: bool = _as_bool(wc.get("web_crawl_include_external", False), False)
            score_threshold_cfg: float = _as_float(wc.get("web_crawl_score_threshold", 0.0), 0.0)
            default_max_pages: int = _as_int(wc.get("web_crawl_max_pages", 100), 100)

            # Effective max-pages precedence:
            # - explicit request value wins
            # - otherwise use config default
            requested_max_pages: Optional[int] = None
            if max_pages is not None:
                requested_max_pages = _as_int(max_pages, default_max_pages)
            effective_max_pages: int = requested_max_pages if requested_max_pages is not None else default_max_pages
            max_pages_source = "request" if requested_max_pages is not None else "config_default"

            # Effective overrides with explicit source tracking for observability.
            requested_strategy = (crawl_strategy or "").strip()
            if requested_strategy:
                eff_strategy = requested_strategy
                strategy_source = "request"
            else:
                eff_strategy = crawl_strategy_cfg
                strategy_source = "config_default"

            if include_external is not None:
                eff_include_external = include_external
                include_external_source = "request"
            else:
                eff_include_external = include_external_cfg
                include_external_source = "config_default"

            if score_threshold is not None:
                eff_score_threshold = score_threshold
                score_threshold_source = "request"
            else:
                eff_score_threshold = score_threshold_cfg
                score_threshold_source = "config_default"

            # Map priority string to enum
            priority_map = {
                "low": JobPriority.LOW,
                "normal": JobPriority.NORMAL,
                "high": JobPriority.HIGH,
                "critical": JobPriority.CRITICAL,
            }
            job_priority = priority_map.get(priority.lower(), JobPriority.NORMAL)

            # Create task ID
            task_id = f"scrape_{uuid.uuid4().hex[:8]}"

            # Process based on scraping method
            if scrape_method == "Individual URLs":
                result = await self._scrape_individual_urls(
                    url_input,
                    custom_titles,
                    summarize_checkbox,
                    custom_prompt,
                    api_name,
                    api_key,
                    keywords,
                    system_prompt,
                    temperature,
                    custom_cookies,
                    job_priority,
                    user_agent,
                    custom_headers,
                    user_id=user_id,
                )

            elif scrape_method == "Sitemap":
                result = await self._scrape_sitemap(
                    url_input,
                    effective_max_pages,
                    summarize_checkbox,
                    custom_prompt,
                    api_name,
                    api_key,
                    keywords,
                    system_prompt,
                    temperature,
                    job_priority,
                    custom_cookies,
                    user_agent,
                    custom_headers,
                    user_id=user_id,
                    task_id=task_id,
                )

            elif scrape_method == "URL Level":
                if url_level is None:
                    raise ValueError("url_level must be provided for URL Level scraping")
                result = await self._scrape_by_url_level(
                    url_input,
                    url_level,
                    effective_max_pages,
                    summarize_checkbox,
                    custom_prompt,
                    api_name,
                    api_key,
                    keywords,
                    system_prompt,
                    temperature,
                    job_priority,
                    custom_cookies,
                    user_agent,
                    custom_headers,
                    user_id=user_id,
                    include_external=eff_include_external,
                    score_threshold=eff_score_threshold,
                    crawl_strategy=eff_strategy,
                    task_id=task_id,
                )

            elif scrape_method == "Recursive Scraping":
                result = await self._scrape_recursive(
                    url_input,
                    effective_max_pages,
                    max_depth,
                    summarize_checkbox,
                    custom_prompt,
                    api_name,
                    api_key,
                    keywords,
                    system_prompt,
                    temperature,
                    custom_cookies,
                    job_priority,
                    user_agent,
                    custom_headers,
                    user_id=user_id,
                    include_external=eff_include_external,
                    score_threshold=eff_score_threshold,
                    crawl_strategy=eff_strategy,
                    task_id=task_id,
                )

            else:
                raise ValueError(f"Unknown scrape method: {scrape_method}")

            # Attach crawl configuration used (observable but non-breaking)
            if isinstance(result, dict):
                result.setdefault("crawl_config", {})
                result["crawl_config"].update(
                    {
                        "strategy": eff_strategy,
                        "strategy_source": strategy_source,
                        "include_external": eff_include_external,
                        "include_external_source": include_external_source,
                        "score_threshold": eff_score_threshold,
                        "score_threshold_source": score_threshold_source,
                        "default_max_pages": default_max_pages,
                        "requested_max_pages": requested_max_pages,
                        "effective_max_pages": effective_max_pages,
                        "max_pages_source": max_pages_source,
                        "enable_keyword_scorer": bool(wc.get("web_crawl_enable_keyword_scorer", False)),
                        "enable_domain_map": bool(wc.get("web_crawl_enable_domain_map", False)),
                    }
                )

            # Process results based on mode
            if mode == "ephemeral":
                stored = await self._store_ephemeral(result, task_id, user_id)
            else:
                stored = await self._store_persistent(
                    result,
                    keywords,
                    user_id,
                    perform_chunking=perform_chunking,
                    chunking_mode=chunking_mode,
                    auto_chunking_goal=auto_chunking_goal,
                    auto_chunking_use_llm=auto_chunking_use_llm,
                    api_name=api_name,
                )
                if isinstance(stored, dict):
                    stored["task_id"] = task_id
            return stored

        except _WEB_SCRAPE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Web scraping task failed: {e}")
            raise HTTPException(status_code=500, detail="Enhanced web scraping task failed") from e

    async def _scrape_individual_urls(
        self,
        url_input: str,
        custom_titles: Optional[str],
        summarize: bool,
        custom_prompt: Optional[str],
        api_name: Optional[str],
        api_key: Optional[str],
        keywords: str,
        system_prompt: Optional[str],
        temperature: float,
        custom_cookies: Optional[list[dict[str, Any]]],
        priority: JobPriority,
        user_agent: Optional[str],
        custom_headers: Optional[dict[str, str]],
        *,
        user_id: Optional[int] = None,
    ) -> dict[str, Any]:
        """Scrape individual URLs with enhanced features"""
        # Parse URLs and titles
        urls = [url.strip() for url in url_input.split("\n") if url.strip()]
        titles = custom_titles.split("\n") if custom_titles else []

        # Check if scraper is available
        if self.scraper is None:
            logger.warning("Enhanced scraper not available, falling back to basic scraping")
            # Return empty results or raise to trigger fallback
            raise RuntimeError("Enhanced scraper not initialized - Playwright may not be available")

        logger.info(f"Starting to scrape {len(urls)} URLs with enhanced scraper")
        logger.debug(f"URLs to scrape: {urls}")

        # Scrape with enhanced scraper
        results = await self.scraper.scrape_multiple(
            urls,
            method="trafilatura",  # Can be made configurable
            priority=priority,
            summarize=summarize,
            custom_prompt=custom_prompt,
            api_name=api_name,
            api_key=api_key,
            system_prompt=system_prompt,
            temperature=temperature,
            custom_cookies=custom_cookies,
            user_agent=user_agent,
            custom_headers=custom_headers,
            user_id=str(user_id) if user_id is not None else None,
        )

        logger.info(f"Scraping completed, got {len(results)} results")
        for i, result in enumerate(results):
            logger.debug(
                f"Result {i}: extraction_successful={result.get('extraction_successful')}, "
                f"has_content={bool(result.get('content'))}, "
                f"error={result.get('error')}"
            )

        # Apply custom titles if provided
        for i, result in enumerate(results):
            if i < len(titles) and titles[i]:
                result["title"] = titles[i]

        return {"method": "Individual URLs", "total_articles": len(results), "articles": results, "keywords": keywords}

    async def _scrape_sitemap(
        self,
        sitemap_url: str,
        max_pages: int,
        summarize: bool,
        custom_prompt: Optional[str],
        api_name: Optional[str],
        api_key: Optional[str],
        keywords: str,
        system_prompt: Optional[str],
        temperature: float,
        priority: JobPriority,
        custom_cookies: Optional[list[dict[str, Any]]],
        user_agent: Optional[str],
        custom_headers: Optional[dict[str, str]],
        *,
        user_id: Optional[int] = None,
        task_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Scrape from sitemap with filtering"""
        # Check if scraper is available
        if self.scraper is None:
            logger.warning("Enhanced scraper not available for sitemap scraping")
            raise RuntimeError("Enhanced scraper not initialized - Playwright may not be available")

        # Scrape sitemap with content page filter
        results = await self.scraper.scrape_sitemap(
            sitemap_url,
            filter_func=is_content_page,
            max_urls=max_pages,
            custom_cookies=custom_cookies,
            user_agent=user_agent,
            custom_headers=custom_headers,
            user_id=str(user_id) if user_id is not None else None,
            task_id=task_id,
        )

        # Add summarization if requested
        if summarize:
            for result in results:
                if result.get("extraction_successful") and result.get("content"):
                    summary = await self._summarize_content(
                        result["content"], custom_prompt, api_name, api_key, system_prompt, temperature
                    )
                    result["summary"] = summary

        return {
            "method": "Sitemap",
            "total_articles": len(results),
            "articles": results,
            "keywords": keywords,
            "sitemap_url": sitemap_url,
        }

    async def _scrape_by_url_level(
        self,
        base_url: str,
        url_level: int,
        max_pages: int,
        summarize: bool,
        custom_prompt: Optional[str],
        api_name: Optional[str],
        api_key: Optional[str],
        keywords: str,
        system_prompt: Optional[str],
        temperature: float,
        priority: JobPriority,
        custom_cookies: Optional[list[dict[str, Any]]],
        user_agent: Optional[str],
        custom_headers: Optional[dict[str, str]],
        *,
        user_id: Optional[int] = None,
        include_external: Optional[bool] = None,
        score_threshold: Optional[float] = None,
        crawl_strategy: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Scrape by URL level"""

        # Define URL level filter
        def url_level_filter(url: str) -> bool:
            from urllib.parse import urlparse

            path_parts = urlparse(url).path.strip("/").split("/")
            return len(path_parts) <= url_level and is_content_page(url)

        # Recursive scrape with level filter
        results = await self.scraper.recursive_scrape(
            base_url,
            max_pages=max_pages,
            max_depth=url_level,
            url_filter=url_level_filter,
            custom_cookies=custom_cookies,
            user_agent=user_agent,
            custom_headers=custom_headers,
            user_id=str(user_id) if user_id is not None else None,
            include_external_override=include_external,
            score_threshold_override=score_threshold,
            crawl_strategy=crawl_strategy,
            task_id=task_id,
        )

        # Add summarization if requested
        if summarize:
            for result in results:
                if result.get("extraction_successful") and result.get("content"):
                    summary = await self._summarize_content(
                        result["content"], custom_prompt, api_name, api_key, system_prompt, temperature
                    )
                    result["summary"] = summary

        return {
            "method": "URL Level",
            "total_articles": len(results),
            "articles": results,
            "keywords": keywords,
            "url_level": url_level,
        }

    async def _scrape_recursive(
        self,
        base_url: str,
        max_pages: int,
        max_depth: int,
        summarize: bool,
        custom_prompt: Optional[str],
        api_name: Optional[str],
        api_key: Optional[str],
        keywords: str,
        system_prompt: Optional[str],
        temperature: float,
        custom_cookies: Optional[list[dict[str, Any]]],
        priority: JobPriority,
        user_agent: Optional[str],
        custom_headers: Optional[dict[str, str]],
        *,
        user_id: Optional[int] = None,
        include_external: Optional[bool] = None,
        score_threshold: Optional[float] = None,
        crawl_strategy: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Recursive scraping with progress tracking"""
        # Create progress file for resumability
        progress_file = Path(f"./scrape_progress_{uuid.uuid4().hex[:8]}.json")
        progress_key = task_id or "recursive_scrape"

        try:
            # Perform recursive scrape
            results = await self.scraper.recursive_scrape(
                base_url,
                max_pages=max_pages,
                max_depth=max_depth,
                url_filter=is_content_page,
                custom_cookies=custom_cookies,
                user_agent=user_agent,
                custom_headers=custom_headers,
                user_id=str(user_id) if user_id is not None else None,
                include_external_override=include_external,
                score_threshold_override=score_threshold,
                crawl_strategy=crawl_strategy,
                task_id=task_id,
            )

            # Add summarization if requested
            if summarize:
                for result in results:
                    if result.get("extraction_successful") and result.get("content"):
                        summary = await self._summarize_content(
                            result["content"], custom_prompt, api_name, api_key, system_prompt, temperature
                        )
                        result["summary"] = summary

            # Save final progress
            await self.scraper.save_progress(progress_key, progress_file)

            return {
                "method": "Recursive Scraping",
                "total_articles": len(results),
                "articles": results,
                "keywords": keywords,
                "max_depth": max_depth,
                "progress_file": str(progress_file),
            }

        except Exception:
            # Save progress on error for resumability
            await self.scraper.save_progress(progress_key, progress_file)
            raise

    async def _summarize_content(
        self,
        content: str,
        custom_prompt: Optional[str],
        api_name: Optional[str],
        api_key: Optional[str],
        system_prompt: Optional[str],
        temperature: float,
    ) -> str:
        """Summarize content using LLM"""
        try:
            # Provide default prompts from Prompts/webscraping if not supplied
            custom_prompt = (
                custom_prompt
                or load_prompt("webscraping", "article_summary_user")
                or "Summarize this article concisely."
            )
            system_prompt = (
                system_prompt
                or load_prompt("webscraping", "article_summary_system")
                or "You are a professional summarizer."
            )
            summary = analyze(
                input_data=content,
                custom_prompt_arg=custom_prompt,
                api_name=api_name or "openai",
                api_key=api_key,
                temp=temperature,
                system_message=system_prompt,
            )
            return summary
        except _WEB_SCRAPE_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Summarization failed: {e}")
            return "Summary generation failed"

    async def _store_ephemeral(self, result: dict[str, Any], task_id: str, user_id: Optional[int]) -> dict[str, Any]:
        """Store results in ephemeral storage"""
        ephemeral_data = {
            "task_id": task_id,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "result": result,
        }

        ephemeral_id = ephemeral_storage.store_data(ephemeral_data)

        return {
            "status": "ephemeral-ok",
            "ephemeral_id": ephemeral_id,
            "task_id": task_id,
            "total_articles": result.get("total_articles", 0),
            "method": result.get("method"),
            "preview": {
                "articles": len(result.get("articles", [])),
                "first_article": (
                    result.get("articles", [{}])[0].get("title", "N/A") if result.get("articles") else None
                ),
            },
        }

    async def _store_persistent(
        self,
        result: dict[str, Any],
        keywords: str,
        user_id: Optional[int],
        *,
        perform_chunking: bool,
        chunking_mode: str | None,
        auto_chunking_goal: str,
        auto_chunking_use_llm: bool,
        api_name: str | None = None,
    ) -> dict[str, Any]:
        """Store results in database"""
        media_ids = []
        errors = []

        logger.info(f"Storing {len(result.get('articles', []))} articles to database")

        # Get the database path and create instance
        # Default to user_id 1 if not provided (single-user mode)
        effective_user_id = user_id if user_id is not None else 1
        db_path = get_user_media_db_path(effective_user_id)
        with managed_media_database(
            client_id=f"webscraping_service_{effective_user_id}",
            db_path=db_path,
            initialize=False,
        ) as db:

            # Metrics
            try:
                reg = get_metrics_registry()
                reg.set_gauge(
                    "webscraping.persist.last_batch_articles",
                    float(len(result.get("articles", []))),
                    {"method": str(result.get("method", "unknown"))},
                )
            except _WEB_SCRAPE_NONCRITICAL_EXCEPTIONS:
                reg = None
            chunking_form = _web_chunking_form(
                perform_chunking=perform_chunking,
                chunking_mode=chunking_mode,
                auto_chunking_goal=auto_chunking_goal,
                auto_chunking_use_llm=auto_chunking_use_llm,
                api_name=api_name,
            )
            _batch_t0 = time.perf_counter()
            for article in result.get("articles", []):
                logger.debug(
                    f"Processing article: url={article.get('url')}, "
                    f"extraction_successful={article.get('extraction_successful')}"
                )
                if not article.get("extraction_successful"):
                    error_msg = f"Failed to extract: {article.get('url', 'Unknown URL')}"
                    logger.warning(error_msg)
                    errors.append(error_msg)
                    continue

                try:
                    # Prepare data for database
                    {
                        "title": article.get("title", "Untitled"),
                        "author": article.get("author", "Unknown"),
                        "source": article.get("url", ""),
                        "scrape_method": result.get("method", "Unknown"),
                        "extraction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }

                    # Format content with metadata
                    # Include crawl metadata (depth, parent_url, score) if present
                    md = article.get("metadata") if isinstance(article.get("metadata"), dict) else {}
                    crawl_depth = md.get("depth") if md.get("depth") is not None else md.get("crawl_depth")
                    if crawl_depth is None:
                        crawl_depth = article.get("crawl_depth")
                    crawl_parent = (
                        md.get("parent_url") if md.get("parent_url") is not None else md.get("crawl_parent_url")
                    )
                    if crawl_parent is None:
                        crawl_parent = article.get("crawl_parent_url")
                    crawl_score = md.get("score") if md.get("score") is not None else md.get("crawl_score")
                    if crawl_score is None:
                        crawl_score = article.get("crawl_score")

                    # Normalize metadata types when possible so persistence remains stable.
                    try:
                        if crawl_depth is not None:
                            crawl_depth = int(crawl_depth)
                    except _WEB_SCRAPE_NONCRITICAL_EXCEPTIONS:
                        pass
                    try:
                        if crawl_score is not None:
                            crawl_score = float(crawl_score)
                    except _WEB_SCRAPE_NONCRITICAL_EXCEPTIONS:
                        pass

                    content_with_metadata = ContentMetadataHandler.format_content_with_metadata(
                        url=article.get("url", ""),
                        content=article.get("content", ""),
                        pipeline=article.get("method", "enhanced"),
                        additional_metadata={
                            "date": article.get("date", ""),
                            "author": article.get("author", "Unknown"),
                            # Propagate traversal metadata for context
                            "crawl_depth": crawl_depth,
                            "crawl_parent_url": crawl_parent,
                            "crawl_score": crawl_score,
                        },
                    )
                    chunk_options = None
                    chunking_plan = None
                    if perform_chunking:
                        chunk_options, chunking_plan = await async_resolve_chunking_options_and_plan(
                            chunking_form,
                            media_type="web",
                            source_name=str(article.get("url") or ""),
                            extracted_text=content_with_metadata,
                        )
                        attach_chunking_plan_to_result(article, chunking_plan)

                    # Prepare segments

                    # Use summary if available
                    article.get("summary", "No summary available")

                    # Add to database using the instance method
                    # Build safe metadata
                    safe_meta = {
                        "title": article.get("title"),
                        "author": article.get("author"),
                        "url": article.get("url"),
                        "date": article.get("date"),
                        "source": "web",
                        # Persist traversal metadata in safe metadata payload
                        "crawl_depth": crawl_depth,
                        "crawl_parent_url": crawl_parent,
                        "crawl_score": crawl_score,
                    }
                    if chunking_plan:
                        safe_meta["chunking_plan"] = chunking_plan
                    safe_metadata_json = json.dumps(
                        {k: v for k, v in safe_meta.items() if v is not None}, ensure_ascii=False
                    )

                    # Build plaintext chunks for chunk-level FTS
                    if not perform_chunking:
                        chunks_for_sql = []
                    else:
                        try:
                            # Chunk in a worker thread to avoid blocking the event loop for long documents
                            options = chunk_options or {
                                "method": "sentences",
                                "max_size": 500,
                                "overlap": 200,
                            }
                            flat = await asyncio.to_thread(
                                lambda _cwm=content_with_metadata, _opts=options: Chunker().chunk_text_hierarchical_flat(
                                    _cwm,
                                    method=_opts.get("method") or "sentences",
                                    max_size=_opts.get("max_size") or 500,
                                    overlap=_opts.get("overlap") or 200,
                                    language=_opts.get("language"),
                                )
                            )
                            kind_map = {
                                "paragraph": "text",
                                "list_unordered": "list",
                                "list_ordered": "list",
                                "code_fence": "code",
                                "table_md": "table",
                                "header_line": "heading",
                                "header_atx": "heading",
                            }
                            chunks_for_sql = []
                            for it in flat:
                                md = it.get("metadata") or {}
                                ctype = kind_map.get(str(md.get("paragraph_kind") or "").lower(), "text")
                                small = {}
                                if md.get("ancestry_titles"):
                                    small["ancestry_titles"] = md.get("ancestry_titles")
                                if md.get("section_path"):
                                    small["section_path"] = md.get("section_path")
                                chunks_for_sql.append(
                                    {
                                        "text": it.get("text", ""),
                                        "start_char": md.get("start_offset"),
                                        "end_char": md.get("end_offset"),
                                        "chunk_type": ctype,
                                        "metadata": small,
                                    }
                                )
                        except _WEB_SCRAPE_NONCRITICAL_EXCEPTIONS as e:
                            logger.debug(f"webscraping.persist: chunking failed; storing without chunks: {e}")
                            try:
                                if reg:
                                    reg.increment(
                                        "app_warning_events_total",
                                        1,
                                        {"component": "webscraping", "event": "chunking_failed"},
                                    )
                            except _WEB_SCRAPE_NONCRITICAL_EXCEPTIONS:
                                logger.debug("metrics increment failed for webscraping chunking_failed")
                            chunks_for_sql = []

                    # Run blocking DB write off the event loop and observe latency
                    _t0 = time.perf_counter()
                    media_id, media_uuid, message = await asyncio.to_thread(
                        db.add_media_with_keywords,
                        url=article.get("url", ""),
                        title=article.get("title", "Untitled"),
                        media_type="web_document",
                        content=content_with_metadata,  # The full content
                        keywords=keywords.split(",") if keywords else [],
                        prompt=None,  # Optional prompt parameter
                        analysis_content=article.get("summary", None),  # Store summary as analysis
                        safe_metadata=safe_metadata_json,
                        transcription_model="web-scraping-import",
                        author=article.get("author", None),
                        ingestion_date=None,  # Will use current time
                        overwrite=False,
                        chunks=chunks_for_sql,
                    )
                    _dt = max(0.0, time.perf_counter() - _t0)
                    try:
                        if reg:
                            reg.observe(
                                "webscraping.persist.article_duration_seconds",
                                _dt,
                                {"method": str(result.get("method", "unknown"))},
                            )
                    except _WEB_SCRAPE_NONCRITICAL_EXCEPTIONS as me:
                        logger.debug(f"webscraping.persist: metric observe failed: {me}")

                    if media_id:
                        media_ids.append(media_id)
                        logger.info(f"Stored article with media_id: {media_id}, uuid: {media_uuid}")
                        try:
                            if reg:
                                reg.increment(
                                    "webscraping.persist.stored_total",
                                    1,
                                    {"method": str(result.get("method", "unknown"))},
                                )
                        except _WEB_SCRAPE_NONCRITICAL_EXCEPTIONS as me:
                            logger.debug(f"webscraping.persist: metric increment failed: {me}")
                    else:
                        logger.warning(f"Failed to get media_id for article: {article.get('url')}")
                        try:
                            if reg:
                                reg.increment(
                                    "webscraping.persist.failed_total",
                                    1,
                                    {"method": str(result.get("method", "unknown"))},
                                )
                        except _WEB_SCRAPE_NONCRITICAL_EXCEPTIONS as me:
                            logger.debug(f"webscraping.persist: metric increment failed: {me}")

                except _WEB_SCRAPE_NONCRITICAL_EXCEPTIONS as e:
                    logger.error(f"Failed to store article: {e}")
                    errors.append("Storage failed for article")
                    try:
                        if reg:
                            reg.increment(
                                "webscraping.persist.failed_total", 1, {"method": str(result.get("method", "unknown"))}
                            )
                    except _WEB_SCRAPE_NONCRITICAL_EXCEPTIONS as me:
                        logger.debug(f"webscraping.persist: metric increment failed: {me}")

            try:
                if reg:
                    _batch_dt = max(0.0, time.perf_counter() - _batch_t0)
                    reg.observe(
                        "webscraping.persist.batch_duration_seconds",
                        _batch_dt,
                        {"method": str(result.get("method", "unknown"))},
                    )
            except _WEB_SCRAPE_NONCRITICAL_EXCEPTIONS as me:
                logger.debug(f"webscraping.persist: batch metric observe failed: {me}")

        return {
            "status": "persist-ok",
            "media_ids": media_ids,
            "total_articles": len(result.get("articles", [])),
            "stored_articles": len(media_ids),
            "method": result.get("method"),
            "errors": errors if errors else None,
        }

    def _is_admin_user(self, user: Any) -> bool:
        if user is None:
            return False
        role = str(user.get("role", "") if isinstance(user, dict) else getattr(user, "role", "")).strip().lower()
        if role in _PLATFORM_ADMIN_ROLES:
            return True
        roles = _normalized_claim_values(
            user.get("roles", []) if isinstance(user, dict) else getattr(user, "roles", [])
        )
        if roles & _PLATFORM_ADMIN_ROLES:
            return True
        permissions = _normalized_claim_values(
            user.get("permissions", []) if isinstance(user, dict) else getattr(user, "permissions", [])
        )
        return bool(permissions & _ADMIN_CLAIM_PERMISSIONS)

    def _can_access_job(self, job: ScrapingJob, user: Any) -> bool:
        if self._is_admin_user(user):
            return True
        user_id = getattr(user, "id", None)
        if user_id is None:
            return False
        return job.user_id is not None and str(job.user_id) == str(user_id)

    async def get_job_status(self, job_id: str, current_user: Any) -> dict[str, Any]:
        """Get status of a scraping job (scoped to the requesting user)."""
        if not self._initialized:
            raise HTTPException(status_code=503, detail="Service not initialized")
        if self.scraper is None:
            raise HTTPException(status_code=503, detail="Scraping service unavailable")
        if current_user is None:
            raise HTTPException(status_code=401, detail="Authentication required")

        job = self.scraper.job_queue.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if not self._can_access_job(job, current_user):
            raise HTTPException(status_code=403, detail="Not authorized to access this job")

        return job.to_dict()

    async def cancel_job(self, job_id: str, current_user: Any) -> dict[str, Any]:
        """Cancel a scraping job (best-effort, scoped to the requesting user)."""
        if not self._initialized:
            raise HTTPException(status_code=503, detail="Service not initialized")
        if self.scraper is None:
            raise HTTPException(status_code=503, detail="Scraping service unavailable")
        if current_user is None:
            raise HTTPException(status_code=401, detail="Authentication required")

        job = self.scraper.job_queue.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if not self._can_access_job(job, current_user):
            raise HTTPException(status_code=403, detail="Not authorized to cancel this job")

        cancelled = await self.scraper.job_queue.cancel_job(job_id)
        if not cancelled:
            raise HTTPException(status_code=404, detail="Job not found")

        return {
            "job_id": job_id,
            "status": cancelled.status.name.lower(),
            "cancel_requested": cancelled.cancel_requested,
            "message": "Job cancellation requested" if cancelled.status != JobStatus.CANCELLED else "Job cancelled",
        }

    def get_service_status(self) -> dict[str, Any]:
        """Get service status and statistics"""
        if not self._initialized:
            return {"status": "not_initialized", "initialized": False}

        if self.scraper is None:
            return {
                "status": "unavailable",
                "initialized": True,
                "scraper_available": False,
                "queue": {
                    "active_jobs": 0,
                    "completed_jobs": 0,
                    "pending_by_priority": {priority.name: 0 for priority in JobPriority},
                },
                "active_jobs": 0,
                "rate_limiter": None,
            }

        queue_status = self.scraper.job_queue.get_status()

        return {
            "status": "operational",
            "initialized": True,
            "queue": queue_status,
            "active_jobs": queue_status.get("active_jobs", 0),
            "rate_limiter": {
                "max_rps": self.scraper.rate_limiter.max_rps,
                "max_rpm": self.scraper.rate_limiter.max_rpm,
                "max_rph": self.scraper.rate_limiter.max_rph,
            },
        }


# Global service instance
_web_scraping_service: Optional[WebScrapingService] = None


def get_web_scraping_service() -> WebScrapingService:
    """Get or create web scraping service instance"""
    global _web_scraping_service
    if _web_scraping_service is None:
        _web_scraping_service = WebScrapingService()
    return _web_scraping_service


# Integration with existing API
async def process_web_scraping_task(**kwargs) -> dict[str, Any]:
    """
    Process web scraping task - drop-in replacement for the placeholder function.

    This function maintains the same interface as the original but uses
    the enhanced scraping service.
    """
    service = get_web_scraping_service()
    return await service.process_web_scraping_task(**kwargs)


# Cleanup on shutdown
async def shutdown_web_scraping_service():
    """Shutdown the web scraping service"""
    global _web_scraping_service
    if _web_scraping_service:
        await _web_scraping_service.shutdown()
        _web_scraping_service = None
