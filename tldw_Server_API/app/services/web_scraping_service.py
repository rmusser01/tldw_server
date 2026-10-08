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
from collections.abc import Mapping
from typing import Any, Optional

#
# Third-party Libraries
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.schemas.media_request_models import ScrapeMethod
from tldw_Server_API.app.core.exceptions import ResourceNotFoundError
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze
from tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib import (
    scrape_from_sitemap,
)
from tldw_Server_API.app.core.Web_Scraping.orchestration import scrape_article

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

_ANALYSIS_PROVIDER_REQUIRED_MESSAGE = "Choose an analysis provider before running ingest analysis."


def _has_analysis_provider(api_name: Optional[str]) -> bool:
    provider = str(api_name or "").strip()
    return bool(provider) and provider.lower() != "none"


def _analysis_error_detail(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text.lower().startswith("error:"):
        return None
    return text.split(":", 1)[1].strip() or "Analysis failed."


def _mark_analysis_state(
    article: dict[str, Any],
    field_name: str,
    status: str,
    message: str,
) -> dict[str, Any]:
    article[field_name] = None
    article["analysis_status"] = status
    article["analysis_error"] = message
    warnings = article.setdefault("warnings", [])
    if isinstance(warnings, list) and message not in warnings:
        warnings.append(message)
    return article


def _sanitize_analysis_result(
    article: dict[str, Any],
    field_name: str,
    value: Any,
) -> bool:
    detail = _analysis_error_detail(value)
    if not detail:
        return False
    normalized = detail.lower()
    status = (
        "skipped"
        if "provider is required" in normalized or "no api specified" in normalized
        else "failed"
    )
    _mark_analysis_state(article, field_name, status, detail)
    return True


def _normalize_strategy_value(crawl_strategy: Optional[str]) -> Optional[str]:
    if crawl_strategy is None:
        return None
    value = crawl_strategy.strip().lower()
    if not value:
        return None
    if value in {"best-first", "bestfirst"}:
        return "best_first"
    return value




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
    summary_prompt_overrides: Mapping[str, str] | None = None,
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

    # Delegate to the enhanced scraping service.
    service = get_web_scraping_service()

    # Determine priority based on number of URLs or max_pages
    priority = "normal"
    if scrape_method == "Individual URLs":
        url_count = len([u for u in url_input.split("\n") if u.strip()])
        if url_count > 10:
            priority = "high"
    elif (max_pages or 0) > 50:
        priority = "high"

    return await service.process_web_scraping_task(
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
        summary_prompt_overrides=summary_prompt_overrides,
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


def _ingest_crawl_articles(service_result: Any) -> list[dict[str, Any]]:
    """Recover articles and release temporary payloads not exposed by ingestion.

    Missing enhanced payloads raise ResourceNotFoundError with their result ID.
    Malformed stored envelopes still release their storage before propagating.
    """
    if isinstance(service_result, dict) and service_result.get("ephemeral_id"):
        result_id = service_result["ephemeral_id"]
        try:
            stored = ephemeral_storage.get_data(result_id)
            if stored is None:
                raise ResourceNotFoundError("Web crawl results", result_id, "expired before ingestion")
            service_result = stored["result"]
        finally:
            ephemeral_storage.remove_data(result_id)
    elif (
        isinstance(service_result, dict)
        and service_result.get("status") == "ephemeral-ok"
        and service_result.get("media_id")
    ):
        # Legacy crawls also store a copy, but already return the articles inline.
        ephemeral_storage.remove_data(service_result["media_id"])
    if isinstance(service_result, dict):
        service_result = service_result.get("articles") or service_result.get("results") or []
    articles = service_result if isinstance(service_result, list) else []
    for article in articles:
        if isinstance(article, dict) and "summary" in article and "analysis" not in article:
            article["analysis"] = article.get("summary")
    return articles


async def ingest_web_content_orchestrate(
    request: Any,
    db: Any,
    usage_log: Any,
    *,
    summary_prompt_overrides: Mapping[str, str] | None = None,
) -> Optional[list[dict[str, Any]]]:
    """
    Shared helper for `/media/ingest-web-content` side effects and summarization:
      - ScrapeMethod.INDIVIDUAL: per-URL scrape + summary
      - ScrapeMethod.SITEMAP: sitemap scrape + summary
      - URL_LEVEL / RECURSIVE: crawl task + ephemeral result retrieval

    The HTTP caller may supply one owner-bound prompt snapshot. Unscoped callers
    retain existing defaults without acquiring user prompt storage.
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

    credential_free = bool(getattr(request, "credential_free", False))

    if not credential_free:
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
            return _mark_analysis_state(article, "analysis", "skipped", "No content to analyze.")

        api_name = getattr(request, "api_name", None)
        if not _has_analysis_provider(api_name):
            return _mark_analysis_state(
                article,
                "analysis",
                "skipped",
                _ANALYSIS_PROVIDER_REQUIRED_MESSAGE,
            )

        analysis_results = analyze(
            input_data=content,
            custom_prompt_arg=(summary_prompt_overrides or {}).get(
                "user", getattr(request, "custom_prompt", None) or "Summarize this article."
            ),
            api_name=api_name,
            temp=0.7,
            system_message=(summary_prompt_overrides or {}).get(
                "system", getattr(request, "system_prompt", None) or "Act as a professional summarizer."
            ),
        )
        if not _sanitize_analysis_result(article, "analysis", analysis_results):
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

            try:
                article_data = await scrape_article(
                    url,
                    custom_cookies=custom_cookies_list,
                    allow_llm_extraction=bool(request.perform_analysis) and not credential_free,
                    **({"credential_free": True} if credential_free else {}),
                )
            except Exception:  # noqa: BLE001 - public preview never returns transport details
                if not credential_free:
                    raise
                article_data = {"extraction_successful": False, "error": "fetch_error"}
            if credential_free:
                from tldw_Server_API.app.core.Web_Scraping.orchestration.article_models import PUBLIC_FAILURE_CODES

                article_data = article_data if isinstance(article_data, dict) else {}
                content = article_data.get("content")
                content = content.strip() if isinstance(content, str) else ""
                error = None
                if not article_data.get("extraction_successful"):
                    code = article_data.get("error")
                    error = code if isinstance(code, str) and code in PUBLIC_FAILURE_CODES else "extraction_error"
                    if article_data.get("policy_reason"):
                        error = "policy_denied"
                elif not content:
                    error = "empty_content"
                elif len(content) > 1_000_000:
                    error = "content_too_large"
                if error:
                    results.append({"url": url, "content": "", "extraction_successful": False, "error": error})
                else:
                    results.append(
                        {
                            "url": url,
                            "title": str(article_data.get("title") or "Untitled")[:1000],
                            "content": content,
                            "extraction_successful": True,
                        }
                    )
                continue
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
            return scrape_from_sitemap(
                sitemap_url,
                allow_llm_extraction=bool(request.perform_analysis),
            )

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
                summary_prompt_overrides=summary_prompt_overrides,
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
            return _ingest_crawl_articles(service_result)
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
                summary_prompt_overrides=summary_prompt_overrides,
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
            return _ingest_crawl_articles(service_result)
        except Exception as exc:  # pragma: no cover - propagate for fallback handler
            logging.exception(f"Enhanced recursive crawl failed: {exc}")
            raise

    # Other methods (or unrecognized) are handled by caller fallback logic.
    return None
