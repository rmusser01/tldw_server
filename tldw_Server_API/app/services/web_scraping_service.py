# /Server_API/app/services/web_scraping_service.py
#
# Enhanced Web Scraping Service
# This replaces the placeholder with a production-ready implementation
#
# Imports
import asyncio
import contextlib
import json
import logging
from types import SimpleNamespace
from typing import Any, Optional

#
# Third-party Libraries
from fastapi import HTTPException
from loguru import logger

from tldw_Server_API.app.api.v1.schemas.media_request_models import ScrapeMethod
from tldw_Server_API.app.core.Chunking.chunker import Chunker
from tldw_Server_API.app.core.DB_Management.db_path_utils import get_user_media_db_path
from tldw_Server_API.app.core.DB_Management.media_db.api import (
    get_media_repository,
    managed_media_database,
)
from tldw_Server_API.app.core.deprecations import log_runtime_deprecation
from tldw_Server_API.app.core.Ingestion_Media_Processing.chunking_options import (
    async_resolve_chunking_options_and_plan,
    attach_chunking_plan_to_result,
)
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze
from tldw_Server_API.app.core.testing import env_flag_enabled

# Keep fallback imports for compatibility mode
from tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib import (
    recursive_scrape,
    scrape_and_summarize_multiple,
    scrape_article,
    scrape_by_url_level,
    scrape_from_sitemap,
)

# Import the enhanced service
from tldw_Server_API.app.services.enhanced_web_scraping_service import (
    get_web_scraping_service,
)

#
# Local Imports
from tldw_Server_API.app.services.ephemeral_store import ephemeral_storage

#
########################################################################################################################
#
# Functions:

_FALLBACK_UNSUPPORTED_CONTROLS = (
    "custom_headers",
    "crawl_strategy",
    "include_external",
    "score_threshold",
)


def _normalize_strategy_value(crawl_strategy: Optional[str]) -> Optional[str]:
    if crawl_strategy is None:
        return None
    value = crawl_strategy.strip().lower()
    if not value:
        return None
    if value in {"best-first", "bestfirst"}:
        return "best_first"
    return value


def _collect_fallback_unsupported_controls(
    *,
    scrape_method: str,
    custom_headers: Optional[dict[str, str]],
    crawl_strategy: Optional[str],
    include_external: Optional[bool],
    score_threshold: Optional[float],
) -> list[str]:
    """
    Return controls that cannot be honored by the fallback implementation.

    Fallback behavior is intentionally conservative:
    - Recursive fallback only supports the default BFS-like traversal.
    - URL-level/sitemap/individual fallback do not support advanced crawl controls.
    - score_threshold is only meaningfully supported when > 0.0.
    """
    unsupported: list[str] = []
    normalized_strategy = _normalize_strategy_value(crawl_strategy)

    if custom_headers:
        unsupported.append("custom_headers")

    score_threshold_active = score_threshold is not None and float(score_threshold) > 0.0

    if scrape_method == "Recursive Scraping":
        if normalized_strategy and normalized_strategy != "default":
            unsupported.append("crawl_strategy")
        if include_external is True:
            unsupported.append("include_external")
        if score_threshold_active:
            unsupported.append("score_threshold")
    else:
        if normalized_strategy:
            unsupported.append("crawl_strategy")
        if include_external is True:
            unsupported.append("include_external")
        if score_threshold_active:
            unsupported.append("score_threshold")

    return sorted(set(unsupported))


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


def _build_fallback_context(
    *,
    fallback_error: Exception,
    scrape_method: str,
    degraded_controls: Optional[list[str]] = None,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "engine": "legacy_fallback",
        "scrape_method": scrape_method,
        "trigger_error_type": type(fallback_error).__name__,
        "degraded_controls_applied": degraded_controls or [],
        "unsupported_controls": list(_FALLBACK_UNSUPPORTED_CONTROLS),
    }


def _legacy_web_scraping_fallback_enabled() -> bool:
    return env_flag_enabled("TLDW_ENABLE_LEGACY_WEB_SCRAPING_FALLBACK")


async def process_web_scraping_task(
    scrape_method: str,
    url_input: str,
    url_level: Optional[int],
    max_pages: Optional[int],
    max_depth: int,
    summarize_checkbox: bool,
    custom_prompt: Optional[str],
    api_name: Optional[str],
    api_key: Optional[str],
    keywords: str,
    custom_titles: Optional[str],
    system_prompt: Optional[str],
    temperature: float,
    custom_cookies: Optional[list[dict[str, Any]]],
    mode: str = "persist",
    user_id: Optional[int] = None,
    user_agent: Optional[str] = None,
    custom_headers: Optional[dict[str, str]] = None,
    # Crawl overrides from UI / WebScrapingRequest
    crawl_strategy: Optional[str] = None,
    include_external: Optional[bool] = None,
    score_threshold: Optional[float] = None,
    perform_chunking: bool = True,
    chunking_mode: Optional[str] = None,
    auto_chunking_goal: str = "balanced",
    auto_chunking_use_llm: bool = False,
) -> dict[str, Any]:
    """
    Enhanced web scraping with production features:
    - Concurrent scraping with rate limiting
    - Job queue management with priority
    - Cookie/session management
    - Progress tracking and resumability
    - Content deduplication
    - Robust error handling and retries

    This function delegates to the enhanced service while maintaining
    backward compatibility with the existing API.

    Parameters:
    - crawl_strategy: Optional crawl strategy override for enhanced crawling.
      Normalized to lowercase and validated against: "default", "best_first",
      "best-first", "bestfirst".
    - include_external: Optional flag to allow following external links during crawl.
      Forwarded as-is to the enhanced service when provided.
    - score_threshold: Optional relevance threshold in [0.0, 1.0] for URL scoring.
      Coerced to float and validated to be within the closed interval [0.0, 1.0].
    - custom_headers: Optional HTTP headers to use for outbound scraping requests.
      Forwarded as-is to the enhanced service and used for session keying.

    Fallback behaviour:
    - When the enhanced service is unavailable, a compatibility implementation is used.
    - The fallback path validates advanced crawl controls and returns explicit
      `400` errors for unsupported options instead of silently ignoring them.
    - Fallback responses include `engine="legacy_fallback"` and
      `fallback_context` to make degradation observable for API clients.
    """
    # Normalize and validate crawl overrides before dispatch
    normalized_crawl_strategy: Optional[str] = None
    if crawl_strategy is not None:
        candidate_strategy = _normalize_strategy_value(crawl_strategy)
        if candidate_strategy is None:
            candidate_strategy = "default"
        allowed_strategies = {"default", "best_first"}
        if candidate_strategy not in allowed_strategies:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid crawl_strategy '{crawl_strategy}'. "
                    "Valid options are: 'default', 'best_first', 'best-first', 'bestfirst'."
                ),
            )
        normalized_crawl_strategy = candidate_strategy

    normalized_score_threshold: Optional[float] = None
    if score_threshold is not None:
        try:
            normalized_score_threshold = float(score_threshold)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=400,
                detail=(f"score_threshold must be a float between 0.0 and 1.0; " f"got {score_threshold!r}."),
            ) from None
        if not 0.0 <= normalized_score_threshold <= 1.0:
            raise HTTPException(
                status_code=400,
                detail=("score_threshold must be between 0.0 and 1.0 inclusive; " f"got {normalized_score_threshold}."),
            )

    if normalized_crawl_strategy is not None:
        crawl_strategy = normalized_crawl_strategy
    if normalized_score_threshold is not None:
        score_threshold = normalized_score_threshold

    # Try to use enhanced service
    try:
        service = get_web_scraping_service()

        # Determine priority based on number of URLs or max_pages
        priority = "normal"
        if scrape_method == "Individual URLs":
            url_count = len([u for u in url_input.split("\n") if u.strip()])
            if url_count > 10:
                priority = "high"
        elif (max_pages or 0) > 50:
            priority = "high"

        # Call enhanced service
        result = await service.process_web_scraping_task(
            scrape_method=scrape_method,
            url_input=url_input,
            url_level=url_level,
            max_pages=max_pages,
            max_depth=max_depth,
            summarize_checkbox=summarize_checkbox,
            custom_prompt=custom_prompt,
            api_name=api_name,
            api_key=api_key,
            keywords=keywords,
            custom_titles=custom_titles,
            system_prompt=system_prompt,
            temperature=temperature,
            custom_cookies=custom_cookies,
            mode=mode,
            priority=priority,
            user_id=user_id,
            user_agent=user_agent,
            custom_headers=custom_headers,
            crawl_strategy=crawl_strategy,
            include_external=include_external,
            score_threshold=score_threshold,
            perform_chunking=perform_chunking,
            chunking_mode=chunking_mode,
            auto_chunking_goal=auto_chunking_goal,
            auto_chunking_use_llm=auto_chunking_use_llm,
        )

        return result

    except Exception as e:
        logger.exception("Enhanced scraping service failed: {}", e)
        log_runtime_deprecation(
            "web_scraping_legacy_fallback",
            message=("Enhanced web scraping service failed; using deprecated " "runtime compatibility fallback."),
        )
        if not _legacy_web_scraping_fallback_enabled():
            raise HTTPException(
                status_code=400,
                detail=(
                    "Legacy web scraping fallback is deprecated and disabled. "
                    "Set TLDW_ENABLE_LEGACY_WEB_SCRAPING_FALLBACK=1 only as a temporary "
                    "compatibility override while restoring the enhanced scraping service."
                ),
            ) from e
        logging.warning("Falling back to compatibility implementation")
        fallback_context = _build_fallback_context(
            fallback_error=e,
            scrape_method=scrape_method,
        )

        # Fallback implementation path
        try:
            unsupported_controls = _collect_fallback_unsupported_controls(
                scrape_method=scrape_method,
                custom_headers=custom_headers,
                crawl_strategy=crawl_strategy,
                include_external=include_external,
                score_threshold=score_threshold,
            )
            if unsupported_controls:
                detail = (
                    "Enhanced web scraping options are only available when the enhanced "
                    f"scraping service is running. The legacy fallback for '{scrape_method}' "
                    "does not support the following parameters: "
                    f"{', '.join(sorted(unsupported_controls))}. "
                    "Retry when enhanced scraping is available, or remove unsupported fields."
                )
                raise HTTPException(status_code=400, detail=detail)

            degraded_controls: list[str] = []

            # 1) Perform scraping based on method
            if scrape_method == "Individual URLs":
                # For multi-line text input, your existing function supports that
                result_list = await scrape_and_summarize_multiple(
                    urls=url_input,
                    custom_prompt_arg=custom_prompt,
                    api_name=api_name,
                    api_key=api_key,
                    keywords=keywords,
                    custom_article_titles=custom_titles,
                    system_prompt=system_prompt,
                    summarize_checkbox=summarize_checkbox,
                    custom_cookies=custom_cookies,
                    temperature=temperature,
                )
            elif scrape_method == "Sitemap":
                # Synchronous approach in your code, might need `asyncio.to_thread`
                result_list = await asyncio.to_thread(scrape_from_sitemap, url_input)
            elif scrape_method == "URL Level":
                if url_level is None:
                    raise ValueError("`url_level` must be provided when scraping method is 'URL Level'")
                result_list = await asyncio.to_thread(scrape_by_url_level, url_input, url_level)
            elif scrape_method == "Recursive Scraping":
                # Call the existing async recursive_scrape implementation.
                # It returns a list of dicts:
                # { url, title, content, extraction_successful, ... }
                recursive_kwargs: dict[str, Any] = {
                    "base_url": url_input,
                    # Fallback path has no config-driven default resolution.
                    # Keep historical behavior when explicit value is unavailable.
                    "max_pages": max_pages if max_pages is not None else 10,
                    "max_depth": max_depth,
                    "progress_callback": (lambda x: None),  # no-op
                    "delay": 1.0,
                    "custom_cookies": custom_cookies,
                }
                # Only override user-agent if explicitly provided, otherwise keep
                # the default behavior inside recursive_scrape.
                if user_agent:
                    recursive_kwargs["user_agent"] = user_agent

                result_list = await recursive_scrape(**recursive_kwargs)
            else:
                raise ValueError(f"Unknown scrape method: {scrape_method}")

            # 1b) Apply predictable max_pages cap for fallback methods that do not
            # natively support page-count control.
            if max_pages is not None and scrape_method in {"Sitemap", "URL Level"} and isinstance(result_list, list):
                before_count = len(result_list)
                result_list = result_list[:max_pages]
                if before_count != len(result_list):
                    degraded_controls.append("max_pages")
                logging.info(
                    "Legacy fallback applied max_pages cap for %s: requested=%s, before=%s, after=%s",
                    scrape_method,
                    max_pages,
                    before_count,
                    len(result_list),
                )

            fallback_context["degraded_controls_applied"] = degraded_controls
            chunking_form = _web_chunking_form(
                perform_chunking=perform_chunking,
                chunking_mode=chunking_mode,
                auto_chunking_goal=auto_chunking_goal,
                auto_chunking_use_llm=auto_chunking_use_llm,
                api_name=api_name,
            )

            # 2) Summarize after the fact, if the method doesn't handle it
            #    (For "Individual URLs," you already did so inside scrape_and_summarize_multiple.)
            #    For the others, if summarize_checkbox is True:
            if summarize_checkbox and scrape_method != "Individual URLs":
                # ensure all results are a list of dicts with 'content'
                for article in result_list:
                    content = article.get("content", "")
                    if content:
                        summary = analyze(
                            input_data=content,
                            custom_prompt_arg=custom_prompt or "",
                            api_name=api_name,
                            api_key=api_key,
                            temp=temperature,
                            system_message=system_prompt or "",
                        )
                        article["summary"] = summary
                    else:
                        article["summary"] = "No content to summarize."

            # 3) If "persist" mode, insert into DB; if ephemeral, store ephemeral
            #    (We can store all articles in the DB or ephemeral. Typically you'd store each as a new "media" row.)
            if mode == "ephemeral":
                if perform_chunking:
                    for article in result_list:
                        if not isinstance(article, dict):
                            continue
                        content_text = article.get("content")
                        chunk_options, chunking_plan = await async_resolve_chunking_options_and_plan(
                            chunking_form,
                            media_type="web",
                            source_name=str(article.get("url") or ""),
                            extracted_text=content_text if isinstance(content_text, str) else None,
                        )
                        attach_chunking_plan_to_result(article, chunking_plan)
                # Just store the entire "result_list" in ephemeral, returning the ephemeral ID.
                # Or store each article individually. Up to you. We'll do one ephemeral object:
                ephemeral_id = ephemeral_storage.store_data({"articles": result_list})
                return {
                    "status": "ephemeral-ok",
                    "media_id": ephemeral_id,
                    "total_articles": len(result_list),
                    "results": result_list,
                    "engine": "legacy_fallback",
                    "fallback_context": fallback_context,
                }
            else:
                # Get the database path and create instance
                if user_id is None:
                    raise HTTPException(
                        status_code=400,
                        detail="user_id is required for legacy persistence in multi-user mode.",
                    )
                effective_user_id = user_id
                db_path = get_user_media_db_path(effective_user_id)
                with managed_media_database(
                    client_id="webscraping_legacy_service",
                    db_path=db_path,
                    initialize=False,
                ) as db:

                    # Persist each article in the DB
                    media_ids = []
                    for article in result_list:
                        # We'll treat article['content'] as the main text
                        # Combine content and metadata
                        content_text = article.get("content", "")
                        chunk_options = None
                        chunking_plan = None
                        if perform_chunking:
                            chunk_options, chunking_plan = await async_resolve_chunking_options_and_plan(
                                chunking_form,
                                media_type="web",
                                source_name=str(article.get("url") or ""),
                                extracted_text=content_text if isinstance(content_text, str) else None,
                            )
                            attach_chunking_plan_to_result(article, chunking_plan)

                        # Fix the function call to match the actual signature
                        # Build safe metadata
                        safe_meta = {
                            "title": article.get("title"),
                            "author": article.get("author"),
                            "url": article.get("url"),
                            "source": "web",
                        }
                        if chunking_plan:
                            safe_meta["chunking_plan"] = chunking_plan
                        safe_metadata_json = json.dumps(
                            {k: v for k, v in safe_meta.items() if v is not None}, ensure_ascii=False
                        )

                        # Build plaintext chunks for FTS-first retrieval
                        if not perform_chunking:
                            chunks_for_sql = []
                        else:
                            try:
                                ck = Chunker()
                                options = chunk_options or {
                                    "method": "sentences",
                                    "max_size": 500,
                                    "overlap": 200,
                                }
                                flat = ck.chunk_text_hierarchical_flat(
                                    content_text,
                                    method=options.get("method") or "sentences",
                                    max_size=options.get("max_size") or 500,
                                    overlap=options.get("overlap") or 200,
                                    language=options.get("language"),
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
                            except Exception as chunk_err:
                                logger.debug(
                                    "Chunking failed for scraped article {}; using empty chunks: {}",
                                    article.get("url", ""),
                                    chunk_err,
                                )
                                chunks_for_sql = []

                        media_id, media_uuid, message = get_media_repository(db).add_media_with_keywords(
                            url=article.get("url", ""),
                            title=article.get("title", "Untitled"),
                            media_type="web_document",
                            content=content_text,
                            keywords=keywords.split(",") if keywords else [],
                            prompt=(
                                (system_prompt or "") + "\n\n" + (custom_prompt or "")
                                if (system_prompt or custom_prompt)
                                else None
                            ),
                            analysis_content=article.get("summary", None),
                            safe_metadata=safe_metadata_json,
                            transcription_model="web-scraping-import",
                            author=article.get("author", None),
                            ingestion_date=None,
                            overwrite=False,
                            chunks=chunks_for_sql,
                        )
                        if media_id:
                            media_ids.append(media_id)

                return {
                    "status": "persist-ok",
                    "media_ids": media_ids,
                    "total_articles": len(result_list),
                    "engine": "legacy_fallback",
                    "fallback_context": fallback_context,
                }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail="Legacy web scraping fallback failed",
            ) from e


async def ingest_web_content_orchestrate(
    request: Any,
    db: Any,
    usage_log: Any,
) -> Optional[list[dict[str, Any]]]:
    """
    Shared helper for `/media/ingest-web-content` side effects and, for
    selected scrape methods, the scraping + summarization:
      - ScrapeMethod.INDIVIDUAL: per-URL scrape + summary
      - ScrapeMethod.SITEMAP: sitemap scrape + summary
    """

    # Log usage for web scraping ingest
    with contextlib.suppress(Exception):
        usage_log.log_event(
            "webscrape.ingest",
            tags=[str(getattr(request, "scrape_method", "") or "")],
            metadata={
                "url_count": len(getattr(request, "urls", []) or []),
                "perform_analysis": bool(getattr(request, "perform_analysis", False)),
            },
        )

    # Topic monitoring (non-blocking): URLs and provided titles
    try:
        from tldw_Server_API.app.core.Monitoring.topic_monitoring_service import (
            get_topic_monitoring_service,
        )

        mon = get_topic_monitoring_service()
        uid = getattr(db, "client_id", None) if hasattr(db, "client_id") else None
        for u in (getattr(request, "urls", []) or [])[:10]:
            if u:
                mon.schedule_evaluate_and_alert(
                    user_id=str(uid) if uid else None,
                    text=str(u),
                    source="ingestion.web",
                    scope_type="user",
                    scope_id=str(uid) if uid else None,
                )
        for t in (getattr(request, "titles", []) or [])[:10]:
            if t:
                mon.schedule_evaluate_and_alert(
                    user_id=str(uid) if uid else None,
                    text=str(t),
                    source="ingestion.web",
                    scope_type="user",
                    scope_id=str(uid) if uid else None,
                )
    except Exception as monitoring_error:
        # Do not let monitoring failures break ingestion.
        _ = monitoring_error

    scrape_method = getattr(request, "scrape_method", None)

    async def maybe_summarize_one(article: dict[str, Any]) -> dict[str, Any]:
        """
        Shared summarization helper for sitemap/individual scraping.
        Mirrors the previous ingest_web_content summarization behavior.
        """
        if not getattr(request, "perform_analysis", False):
            article["analysis"] = None
            return article

        content = article.get("content", "")
        if not content:
            article["analysis"] = "No content to analyze."
            return article

        analysis_results = analyze(
            input_data=content,
            custom_prompt_arg=getattr(request, "custom_prompt", None) or "Summarize this article.",
            api_name=getattr(request, "api_name", None),
            temp=0.7,
            system_message=getattr(request, "system_prompt", None) or "Act as a professional summarizer.",
        )
        article["analysis"] = analysis_results

        if getattr(request, "perform_rolling_summarization", False):
            logging.info("Performing rolling summarization (placeholder).")
        if getattr(request, "perform_confabulation_check_of_analysis", False):
            logging.info("Performing confabulation check of analysis (placeholder).")

        return article

    def parse_cookies() -> Optional[list[dict[str, Any]]]:
        """
        Parse cookies from the request when `use_cookies` is enabled.
        Mirrors prior JSON parsing + 400 semantics, but ensures that
        malformed or incorrectly-typed cookie payloads yield a 400 instead
        of bubbling up as a 500 error.
        """
        custom_cookies_list: Optional[list[dict[str, Any]]] = None
        if getattr(request, "use_cookies", False) and getattr(request, "cookies", None):
            raw_cookies = request.cookies
            if isinstance(raw_cookies, (bytes, bytearray)):
                try:
                    raw_cookies = raw_cookies.decode("utf-8")
                except UnicodeDecodeError:
                    raise HTTPException(status_code=400, detail="Invalid cookies format") from None

            if isinstance(raw_cookies, str):
                try:
                    parsed = json.loads(raw_cookies)
                except json.JSONDecodeError:
                    raise HTTPException(status_code=400, detail="Invalid JSON format for cookies") from None
            elif isinstance(raw_cookies, (dict, list)):
                parsed = raw_cookies
            else:
                raise HTTPException(status_code=400, detail="Invalid cookies format")

            if isinstance(parsed, dict):
                custom_cookies_list = [parsed]
            elif isinstance(parsed, list):
                if not all(isinstance(item, dict) for item in parsed):
                    raise HTTPException(status_code=400, detail="Invalid cookies format")
                custom_cookies_list = parsed
            else:
                raise HTTPException(status_code=400, detail="Invalid cookies format")

        return custom_cookies_list

    # INDIVIDUAL URLs: per-URL scrape + summarization
    if scrape_method == ScrapeMethod.INDIVIDUAL:
        urls = getattr(request, "urls", []) or []
        if not urls:
            return []

        titles = getattr(request, "titles", None) or []
        authors = getattr(request, "authors", None) or []
        keywords = getattr(request, "keywords", None) or []
        num_urls = len(urls)

        if len(titles) < num_urls:
            titles += ["Untitled"] * (num_urls - len(titles))
        if len(authors) < num_urls:
            authors += ["Unknown"] * (num_urls - len(authors))
        if len(keywords) < num_urls:
            keywords += ["no_keyword_set"] * (num_urls - len(keywords))

        custom_cookies_list = parse_cookies()

        results: list[dict[str, Any]] = []
        for i, url in enumerate(urls):
            title_ = titles[i]
            author_ = authors[i]
            kw_ = keywords[i]

            article_data = await scrape_article(url, custom_cookies=custom_cookies_list)
            if not article_data or not article_data.get("extraction_successful"):
                logging.warning(f"Failed to scrape: {url}")
                continue

            article_data["title"] = title_ or article_data.get("title")
            article_data["author"] = author_ or article_data.get("author")
            article_data["keywords"] = kw_

            article_data = await maybe_summarize_one(article_data)
            results.append(article_data)

        return results

    # SITEMAP: scrape sitemap URL, then summarize each article
    if scrape_method == ScrapeMethod.SITEMAP:
        urls = getattr(request, "urls", []) or []
        if not urls:
            return []

        sitemap_url = urls[0]

        def scrape_in_thread() -> list[dict[str, Any]]:
            return scrape_from_sitemap(sitemap_url)

        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, scrape_in_thread)

        if not results:
            logging.warning("No articles returned from sitemap scraping.")
            return []

        summarized: list[dict[str, Any]] = []
        for r in results:
            # Legacy path expects dict-like articles; skip anything else defensively.
            if not isinstance(r, dict):
                continue
            summarized_article = await maybe_summarize_one(r)
            summarized.append(summarized_article)

        return summarized

    # URL LEVEL: route to enhanced service (friendly ingest)
    if scrape_method == ScrapeMethod.URL_LEVEL:
        urls = getattr(request, "urls", []) or []
        if not urls:
            return []

        base_url = urls[0]
        level = getattr(request, "url_level", None) or 2
        requested_max_pages = getattr(request, "max_pages", None)

        custom_cookies_list = parse_cookies()

        try:
            from tldw_Server_API.app.api.v1.endpoints import media as media_mod

            scrape_task = getattr(media_mod, "process_web_scraping_task", process_web_scraping_task)
        except Exception:  # pragma: no cover - defensive fallback
            scrape_task = process_web_scraping_task

        try:
            service_result = await scrape_task(
                scrape_method="URL Level",
                url_input=base_url,
                url_level=level,
                max_pages=requested_max_pages,
                max_depth=level,
                summarize_checkbox=bool(getattr(request, "perform_analysis", False)),
                custom_prompt=getattr(request, "custom_prompt", None),
                api_name=getattr(request, "api_name", None),
                api_key=None,
                keywords=(
                    ",".join(request.keywords or [])
                    if isinstance(getattr(request, "keywords", None), list)
                    else (getattr(request, "keywords", None) or "")
                ),
                custom_titles=None,
                system_prompt=getattr(request, "system_prompt", None),
                temperature=0.7,
                custom_cookies=custom_cookies_list,
                mode="ephemeral",
                user_agent=getattr(request, "user_agent", None) if hasattr(request, "user_agent") else None,
                custom_headers=None,
                crawl_strategy=getattr(request, "crawl_strategy", None),
                include_external=getattr(request, "include_external", None),
                score_threshold=getattr(request, "score_threshold", None),
                perform_chunking=bool(getattr(request, "perform_chunking", True)),
                chunking_mode=getattr(request, "chunking_mode", None),
                auto_chunking_goal=getattr(request, "auto_chunking_goal", "balanced"),
                auto_chunking_use_llm=bool(getattr(request, "auto_chunking_use_llm", False)),
            )
            articles: list[dict[str, Any]] = []
            if isinstance(service_result, dict):
                if service_result.get("articles"):
                    articles = service_result["articles"]
                elif service_result.get("results"):
                    articles = service_result["results"]

            for r in articles:
                if isinstance(r, dict) and "summary" in r and "analysis" not in r:
                    r["analysis"] = r.get("summary")

            return articles
        except Exception as exc:  # pragma: no cover - propagate for fallback handler
            logging.exception(f"Enhanced URL Level crawl failed: {exc}")
            raise

    # RECURSIVE SCRAPING: route to enhanced service (friendly ingest)
    if scrape_method == ScrapeMethod.RECURSIVE:
        urls = getattr(request, "urls", []) or []
        if not urls:
            return []

        base_url = urls[0]
        max_pages = getattr(request, "max_pages", None)
        max_depth = getattr(request, "max_depth", None) or 3

        custom_cookies_list = parse_cookies()

        try:
            from tldw_Server_API.app.api.v1.endpoints import media as media_mod

            scrape_task = getattr(media_mod, "process_web_scraping_task", process_web_scraping_task)
        except Exception:  # pragma: no cover - defensive fallback
            scrape_task = process_web_scraping_task

        try:
            service_result = await scrape_task(
                scrape_method="Recursive Scraping",
                url_input=base_url,
                url_level=None,
                max_pages=max_pages,
                max_depth=max_depth,
                summarize_checkbox=bool(getattr(request, "perform_analysis", False)),
                custom_prompt=getattr(request, "custom_prompt", None),
                api_name=getattr(request, "api_name", None),
                api_key=None,
                keywords=(
                    ",".join(request.keywords or [])
                    if isinstance(getattr(request, "keywords", None), list)
                    else (getattr(request, "keywords", None) or "")
                ),
                custom_titles=None,
                system_prompt=getattr(request, "system_prompt", None),
                temperature=0.7,
                custom_cookies=custom_cookies_list,
                mode="ephemeral",
                user_agent=getattr(request, "user_agent", None) if hasattr(request, "user_agent") else None,
                custom_headers=None,
                crawl_strategy=getattr(request, "crawl_strategy", None),
                include_external=getattr(request, "include_external", None),
                score_threshold=getattr(request, "score_threshold", None),
                perform_chunking=bool(getattr(request, "perform_chunking", True)),
                chunking_mode=getattr(request, "chunking_mode", None),
                auto_chunking_goal=getattr(request, "auto_chunking_goal", "balanced"),
                auto_chunking_use_llm=bool(getattr(request, "auto_chunking_use_llm", False)),
            )
            articles: list[dict[str, Any]] = []
            if isinstance(service_result, list):
                articles = service_result
            elif isinstance(service_result, dict):
                if service_result.get("articles"):
                    articles = service_result.get("articles") or []
                elif service_result.get("results"):
                    articles = service_result.get("results") or []

            for r in articles:
                if isinstance(r, dict) and "summary" in r and "analysis" not in r:
                    r["analysis"] = r.get("summary")

            return articles
        except Exception as exc:  # pragma: no cover - propagate for fallback handler
            logging.exception(f"Enhanced recursive crawl failed: {exc}")
            raise

    # Other methods (or unrecognized) are handled by caller fallback logic.
    return None
