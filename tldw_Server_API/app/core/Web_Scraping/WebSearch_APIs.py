# WebSearch_APIs.py
# Description: This file contains the functions that are used for performing queries against various Search Engine APIs
#
# Imports
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from functools import lru_cache
from html import unescape
from typing import Any, TypedDict
from urllib.parse import unquote, urlencode, urlparse

#
# 3rd-Party Imports
from lxml.etree import _Element
from lxml.html import document_fromstring

# Removed: HTTPAdapter/Retry (migrated to http_client)
#
# Local Imports
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
from tldw_Server_API.app.core.Chat.chat_helpers import extract_response_content
from tldw_Server_API.app.core.config import load_and_log_configs
from tldw_Server_API.app.core.exceptions import NetworkError, RetryExhaustedError
from tldw_Server_API.app.core.http_client import fetch
from tldw_Server_API.app.core.LLM_Calls.adapter_utils import (
    ensure_app_config,
    get_adapter_or_raise,
    normalize_provider,
    resolve_provider_api_key_from_config,
    resolve_provider_model,
    split_system_message,
)
from tldw_Server_API.app.core.Utils.Utils import logging
from tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib import scrape_article
from tldw_Server_API.app.core.Web_Scraping.outbound_policy import (
    WebOutboundPolicyDecision,
    decide_web_outbound_policy_sync,
)
from tldw_Server_API.app.core.Web_Scraping.ua_profiles import (
    build_browser_headers,
    pick_ua_profile,
)

_WEBSEARCH_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    asyncio.TimeoutError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    json.JSONDecodeError,
    ChatConfigurationError,
    NetworkError,
    RetryExhaustedError,
)


def _websearch_browser_headers(
    *, accept_lang: str = "en-US,en;q=0.5", referer: str = "https://www.google.com/", restrict_encodings_for_requests: bool = True
):
    """Build realistic browser headers for web search endpoints.

    Uses centralized UA profiles so providers do not hard-code User-Agent.
    """
    profile = pick_ua_profile("fixed")
    base = build_browser_headers(profile=profile, accept_lang=accept_lang)
    if restrict_encodings_for_requests:
        # requests doesn't decode br/zstd by default; restrict to gzip,deflate
        base["Accept-Encoding"] = "gzip, deflate"
    base.update({
        "Referer": referer,
        "Connection": "keep-alive",
    })
    return base


@lru_cache(maxsize=1)
def get_loaded_config() -> dict[str, Any]:
    """Lazy, cached config loader to avoid import-time I/O and duplicate logs."""
    return load_and_log_configs()


def _provider_outbound_policy_decision(
    url: str,
    *,
    source: str,
    stage: str = "provider_request",
) -> WebOutboundPolicyDecision:
    """Return the shared outbound-policy decision for a websearch provider URL."""
    try:
        return decide_web_outbound_policy_sync(
            url,
            respect_robots=False,
            source=source,
            stage=stage,
        )
    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as exc:
        raise ValueError(f"Outbound policy evaluation failed: {exc}") from exc


def _enforce_provider_outbound_policy(
    url: str,
    *,
    source: str,
    stage: str = "provider_request",
) -> None:
    decision = _provider_outbound_policy_decision(url, source=source, stage=stage)
    if not decision.allowed:
        raise ValueError(f"Blocked by outbound policy: {decision.reason}")


def _get_relevance_jitter_ms() -> int:
    """Optional jitter (ms) for LLM calls. Defaults to 0 (disabled)."""
    cfg = get_loaded_config()
    section = cfg.get('Web-Scraping', {}) or {}
    # Accept single value or min/max; if both given, use max
    val = section.get('relevance_jitter_ms', 0)
    try:
        return int(val)
    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS:
        try:
            # support min/max pair
            int(section.get('relevance_jitter_min_ms', 0) or 0)
            max_v = int(section.get('relevance_jitter_max_ms', 0) or 0)
            return max(0, max_v)
        except _WEBSEARCH_NONCRITICAL_EXCEPTIONS:
            return 0


def _get_llm_timeouts() -> dict[str, float]:
    """Timeouts (seconds) for relevance LLM calls and article fetches."""
    cfg = get_loaded_config()
    section = cfg.get('Web-Scraping', {}) or {}
    llm_to = float(section.get('relevance_llm_timeout_s', 30) or 30)
    scrape_to = float(section.get('relevance_scrape_timeout_s', 30) or 30)
    return {"llm": llm_to, "scrape": scrape_to}


def _get_websearch_circuit_breaker(
    fail_threshold: int = 3,
    reset_after_s: float = 30.0,
    provider_key: str | None = None,
):
    """Return the shared WebSearch circuit breaker (singleton via registry)."""
    from tldw_Server_API.app.core.Infrastructure.circuit_breaker import (
        CircuitBreakerConfig as _Cfg,
    )
    from tldw_Server_API.app.core.Infrastructure.circuit_breaker import (
        registry as _cb_registry,
    )
    normalized_provider = normalize_provider(provider_key)
    if not normalized_provider:
        normalized_provider = str(provider_key or "default").strip().lower() or "default"
    normalized_provider = re.sub(r"[^a-z0-9_.:-]+", "-", normalized_provider)

    return _cb_registry.get_or_create(
        f"websearch_llm:{normalized_provider}",
        config=_Cfg(
            failure_threshold=int(fail_threshold),
            recovery_timeout=float(reset_after_s),
            half_open_max_calls=1,
            success_threshold=1,
            category="websearch",
        ),
    )


def _close_response(resp: Any) -> None:
    close = getattr(resp, "close", None)
    if callable(close):
        close()


def _truncate_text(value: str | None, max_len: int = 600) -> str:
    if not value:
        return ""
    text = str(value).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "..."


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return default
        if normalized in {"1", "true", "yes", "on", "y"}:
            return True
        if normalized in {"0", "false", "no", "off", "n"}:
            return False
    return default


def _map_searx_safesearch(value: Any) -> int:
    """Normalize safesearch values to Searx levels (0=off, 1=moderate, 2=strict)."""
    if value is None:
        return 1
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        try:
            return max(0, min(2, int(value)))
        except _WEBSEARCH_NONCRITICAL_EXCEPTIONS:
            return 1

    normalized = str(value).strip().lower()
    mapping = {
        "off": 0,
        "false": 0,
        "none": 0,
        "disabled": 0,
        "0": 0,
        "moderate": 1,
        "medium": 1,
        "active": 1,
        "on": 1,
        "true": 1,
        "1": 1,
        "strict": 2,
        "high": 2,
        "2": 2,
    }
    return mapping.get(normalized, 1)


def _sanitize_sub_questions(raw_values: Any) -> list[str]:
    """Normalize model-generated sub-questions into a deduplicated list of non-empty strings."""
    if isinstance(raw_values, str):
        candidates = [raw_values]
    elif isinstance(raw_values, (list, tuple, set)):
        candidates = list(raw_values)
    else:
        return []

    sanitized: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        text = ""
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            query_value = item.get("query")
            if not isinstance(query_value, str):
                query_value = item.get("text")
            if isinstance(query_value, str):
                text = query_value.strip()
        else:
            continue

        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        sanitized.append(text)
    return sanitized


def _summarize_relevant_results_for_log(
    relevant_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return compact debug metadata without logging result bodies."""
    summaries: list[dict[str, Any]] = []
    for result_id, result in relevant_results.items():
        summaries.append(
            {
                "id": result_id,
                "reasoning": _truncate_text(result.get("reasoning"), max_len=160),
                "content_chars": len(str(result.get("content") or "")),
                "original_content_chars": len(str(result.get("original_content") or "")),
            }
        )
    return summaries


def _build_result_fallback_content(result: dict[str, Any]) -> str:
    """Construct a safe fallback text blob from the normalized result payload."""
    metadata = result.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else {}
    parts: list[str] = []
    seen: set[str] = set()

    def _append(label: str, value: Any) -> None:
        text = str(value or "").strip()
        if not text:
            return
        key = text.casefold()
        if key in seen:
            return
        seen.add(key)
        parts.append(f"{label}: {text}")

    _append("Title", result.get("title"))
    _append("Snippet", result.get("content"))
    _append("Snippet", metadata_dict.get("snippet"))
    _append("URL", result.get("url"))
    return "\n".join(parts).strip()
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze


#
#
def summarize(
    input_data: str,
    custom_prompt_arg: str | None = None,
    api_name: str | None = None,
    api_key: str | None = None,
    temp: float = 0.7,
    system_message: str | None = None,
    streaming: bool = False,
    **extra_kwargs: Any,
) -> str:
    """
    Backwards-compatible summarization helper to keep monkeypatch-based tests working.

    All parameters map directly onto :func:`analyze`.
    """
    return analyze(
        input_data=input_data,
        custom_prompt_arg=custom_prompt_arg,
        api_name=api_name,
        api_key=api_key,
        temp=temp,
        system_message=system_message,
        streaming=streaming,
        **extra_kwargs,
    )


def _build_messages(
    *,
    system_prompt: str | None,
    user_prompt: str | None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if user_prompt:
        messages.append({"role": "user", "content": user_prompt})
    return messages


def _call_adapter_text(
    *,
    api_endpoint: str,
    messages_payload: list[dict[str, Any]],
    temperature: float | None = None,
    api_key: str | None = None,
    model: str | None = None,
    app_config: dict[str, Any] | None = None,
    timeout: float | None = None,
    **extra_kwargs: Any,
) -> str:
    provider = normalize_provider(api_endpoint)
    if not provider:
        raise ChatConfigurationError(provider=api_endpoint, message="LLM provider is required.")
    cfg = ensure_app_config(app_config or get_loaded_config())
    resolved_model = model or resolve_provider_model(provider, cfg)
    if not resolved_model:
        raise ChatConfigurationError(provider=provider, message="Model is required for provider.")
    system_message, cleaned_messages = split_system_message(messages_payload or [])
    request: dict[str, Any] = {
        "messages": cleaned_messages,
        "system_message": system_message,
        "model": resolved_model,
        "api_key": api_key or resolve_provider_api_key_from_config(provider, cfg),
        "temperature": temperature,
        "app_config": cfg,
    }
    request.update(extra_kwargs)
    response = get_adapter_or_raise(provider).chat(request, timeout=timeout)
    return extract_response_content(response) or str(response)


def chat_api_call(
    *,
    api_endpoint: str,
    messages_payload: list[dict[str, Any]],
    temperature: float | None = None,
    api_key: str | None = None,
    model: str | None = None,
    app_config: dict[str, Any] | None = None,
    timeout: float | None = None,
    **extra_kwargs: Any,
) -> str:
    """Compatibility wrapper for tests and legacy call sites."""
    return _call_adapter_text(
        api_endpoint=api_endpoint,
        messages_payload=messages_payload,
        temperature=temperature,
        api_key=api_key,
        model=model,
        app_config=app_config,
        timeout=timeout,
        **extra_kwargs,
    )
#
#######################################################################################################################
#
# Functions:
# 1. analyze_question
#
#######################################################################################################################
#
# Functions:

######################### Main Orchestration Workflow #########################
#
# FIXME - Add Logging

def initialize_web_search_results_dict(search_params: dict) -> dict:
    """
    Initializes and returns a dictionary for storing web search results and metadata.

    Args:
        search_params (Dict): A dictionary containing search parameters.

    Returns:
        Dict: A dictionary initialized with search metadata.
    """
    return {
        "search_engine": search_params.get('engine', 'google'),
        "search_query": "",
        "content_country": search_params.get('content_country', 'US'),
        "search_lang": search_params.get('search_lang', 'en'),
        "output_lang": search_params.get('output_lang', 'en'),
        "result_count": 0,
        "date_range": search_params.get('date_range'),
        "safesearch": search_params.get('safesearch', 'active'),
        "site_whitelist": search_params.get('site_whitelist') or search_params.get('include_domains', []),
        "site_blacklist": search_params.get('site_blacklist', []),
        "exactTerms": search_params.get('exactTerms'),
        "excludeTerms": search_params.get('excludeTerms'),
        "filter": search_params.get('filter'),
        "geolocation": search_params.get('geolocation'),
        "search_result_language": search_params.get('search_result_language'),
        "sort_results_by": search_params.get('sort_results_by'),
        "google_domain": search_params.get('google_domain'),
        "results": [],
        "total_results_found": 0,
        "search_time": 0.0,
        "error": None,
        "warnings": [],
        "processing_error": None
    }


def generate_and_search(question: str, search_params: dict) -> dict:
    """
    Generates sub-queries (if enabled) and performs web searches for each query.

    Args:
        question (str): The user's original question or query.
        search_params (Dict): A dictionary containing parameters for performing web searches
                              and specifying LLM endpoints.

    Returns:
        Dict: A dictionary containing all search results and related metadata.

    Raises:
        ValueError: If the input parameters are invalid.
    """
    logging.info(f"Starting generate_and_search with query: {question}")

    # Validate input parameters
    if not question or not isinstance(question, str):
        raise ValueError("Invalid question parameter")
    if not search_params or not isinstance(search_params, dict):
        raise ValueError("Invalid search_params parameter")

    # Check for required keys in search_params
    required_keys = ["engine", "content_country", "search_lang", "output_lang", "result_count"]
    for key in required_keys:
        if key not in search_params:
            raise ValueError(f"Missing required key in search_params: {key}")

    # 1. Generate sub-queries if requested
    logging.info(f"Generating sub-queries for the query: {question}")
    sub_query_dict = {
        "main_goal": question,
        "sub_questions": [],
        "search_queries": [],
        "analysis_prompt": None
    }

    if search_params.get("subquery_generation", False):
        logging.info("Sub-query generation enabled")
        api_endpoint = search_params.get("subquery_generation_llm", "openai")
        sub_query_dict = analyze_question(question, api_endpoint)

    # Merge original question with sub-queries
    sub_queries = _sanitize_sub_questions(sub_query_dict.get("sub_questions", []))
    question_key = question.strip().casefold()
    sub_queries = [
        sub_query
        for sub_query in sub_queries
        if sub_query.strip().casefold() != question_key
    ]
    sub_query_dict["sub_questions"] = sub_queries
    sub_query_dict["search_queries"] = sub_queries
    logging.info(f"Sub-queries generated: {sub_queries}")
    all_queries = [question, *sub_queries]

    # 2. Initialize a single web_search_results_dict
    web_search_results_dict = initialize_web_search_results_dict(search_params)
    web_search_results_dict["search_query"] = question
    observed_provider_errors: list[str] = []
    deferred_provider_error_warnings: list[dict[str, str]] = []

    # 3. Perform searches and accumulate all raw results
    for idx, q in enumerate(all_queries):
        # Add a small random delay between sub-queries to avoid rate limiting
        # Skip delay for the very first query
        if idx > 0:
            sleep_time = random.uniform(1, 1.5)
            time.sleep(sleep_time)
        logging.info(f"Performing web search for query: {q}")
        raw_results = perform_websearch(
            search_engine=search_params.get('engine'),
            search_query=q,
            content_country=search_params.get('content_country', 'US'),
            search_lang=search_params.get('search_lang', 'en'),
            output_lang=search_params.get('output_lang', 'en'),
            result_count=search_params.get('result_count', 10),
            date_range=search_params.get('date_range'),
            safesearch=search_params.get('safesearch', 'active'),
            site_whitelist=search_params.get('site_whitelist') or search_params.get('include_domains'),
            site_blacklist=search_params.get('site_blacklist', []),
            exactTerms=search_params.get('exactTerms'),
            excludeTerms=search_params.get('excludeTerms'),
            filter=search_params.get('filter'),
            geolocation=search_params.get('geolocation'),
            search_result_language=search_params.get('search_result_language'),
            sort_results_by=search_params.get('sort_results_by'),
            google_domain=search_params.get('google_domain'),
            search_params=search_params,
        )

        # Debug: Inspect raw results
        logging.debug(f"Raw results for query '{q}': {raw_results}")

        # Check for errors or invalid data
        if not isinstance(raw_results, dict):
            logging.error(f"Error or invalid data returned for query '{q}': {raw_results}")
            continue
        if raw_results.get("processing_error"):
            processing_error = str(raw_results.get("processing_error")).strip()
            if processing_error:
                observed_provider_errors.append(processing_error)
                deferred_provider_error_warnings.append(
                    {
                        "query": q,
                        "phase": "provider",
                        "message": processing_error,
                    }
                )
            logging.error(f"Provider processing error for query '{q}': {raw_results}")
            continue

        raw_warnings = raw_results.get("warnings")
        if isinstance(raw_warnings, list) and raw_warnings:
            web_search_results_dict["warnings"].extend(raw_warnings)
        elif raw_warnings is not None:
            web_search_results_dict["warnings"].append(raw_warnings)

        raw_error = raw_results.get("error")
        if raw_error is not None:
            error_text = str(raw_error).strip()
            if error_text:
                observed_provider_errors.append(error_text)
                deferred_provider_error_warnings.append(
                    {
                        "query": q,
                        "phase": "provider",
                        "message": error_text,
                    }
                )

        raw_results_list = raw_results.get("results", [])
        if not isinstance(raw_results_list, list):
            raw_results_list = []

        logging.info(f"Search results found for query '{q}': {len(raw_results_list)}")

        # Append results to the single web_search_results_dict
        web_search_results_dict["results"].extend(raw_results_list)
        web_search_results_dict["total_results_found"] += raw_results.get("total_results_found", 0)
        web_search_results_dict["search_time"] += raw_results.get("search_time", 0.0)
        logging.info(f"Total results found so far: {len(web_search_results_dict['results'])}")

    if deferred_provider_error_warnings:
        web_search_results_dict["warnings"].extend(deferred_provider_error_warnings)

    if web_search_results_dict["results"]:
        web_search_results_dict["error"] = None
    elif observed_provider_errors:
        web_search_results_dict["error"] = observed_provider_errors[0]

    if not web_search_results_dict["warnings"]:
        web_search_results_dict.pop("warnings", None)

    return {
        "web_search_results_dict": web_search_results_dict,
        "sub_query_dict": sub_query_dict
    }


async def analyze_and_aggregate(web_search_results_dict: dict, sub_query_dict: dict, search_params: dict, cancel_event: asyncio.Event | None = None) -> dict:
    logging.info("Starting analyze_and_aggregate")

    # 4. Score/filter results
    logging.info("Scoring and filtering search results")
    sub_questions = sub_query_dict.get("sub_questions", [])
    relevant_results = await search_result_relevance(
        web_search_results_dict["results"],
        sub_query_dict["main_goal"],
        sub_questions,
        search_params.get('relevance_analysis_llm'),
        cancel_event=cancel_event,
    )
    logging.debug("Relevant results returned by search_result_relevance:")
    logging.debug(json.dumps(_summarize_relevant_results_for_log(relevant_results), indent=2))

    # 5. Allow user to review and select relevant results (if enabled)
    logging.info("Reviewing and selecting relevant results")
    if search_params.get("user_review", False):
        logging.info("User review enabled")
        relevant_results = review_and_select_results(relevant_results)

    # 6. Summarize/aggregate final answer
    final_answer = aggregate_results(
        relevant_results,
        sub_query_dict["main_goal"],
        sub_questions,
        search_params.get('final_answer_llm')
    )

    # 7. Return the final data
    logging.info("Returning final websearch results")
    return {
        "final_answer": final_answer,
        "relevant_results": relevant_results,
        "web_search_results_dict": web_search_results_dict
    }


# NOTE: module-level demos/tests moved into tests/WebScraping/ to avoid import-time side effects


######################### Question Analysis #########################
#
#
def analyze_question(question: str, api_endpoint) -> dict:
    logging.debug(f"Analyzing question: {question} with API endpoint: {api_endpoint}")
    """
    Analyzes the input question and generates sub-questions

    Returns:
        Dict containing:
        - main_goal: str
        - sub_questions: List[str]
        - search_queries: List[str]
        - analysis_prompt: str
    """
    original_query = question
    sub_question_generation_prompt = f"""
            You are an AI assistant that helps generate search queries. Given an original query, suggest alternative search queries that could help find relevant information. Your goal is to generate queries that are diverse, specific, and highly relevant to the original query, ensuring comprehensive coverage of the topic.

            Important instructions:
            1. Generate between 2 and 6 queries unless a fixed count is specified. Generate more queries for complex or multifaceted topics and fewer for simple or straightforward ones.
            2. Ensure the queries are diverse, covering different aspects or perspectives of the original query, while remaining highly relevant to its core intent.
            3. Prefer specific queries over general ones, as they are more likely to yield targeted and useful results.
            4. If the query involves comparing two topics, generate separate queries for each topic.
            5. If previous queries and an answer are provided, generate new queries that address the shortcomings of the previous answer and avoid repeating the previous queries.
            6. If the original query is broad or ambiguous, generate queries that explore specific subtopics or clarify the intent.
            7. If the query is too specific or unclear, generate queries that explore related or broader topics to ensure useful results.
            8. Return the queries as a JSON array in the format ["query_1", "query_2", ...].

            Examples:
            1. For the query "What are the benefits of exercise?", generate queries like:
               ["health benefits of physical activity", "mental health benefits of exercise", "long-term effects of regular exercise", "how exercise improves cardiovascular health", "role of exercise in weight management"]

            2. For the query "Compare Python and JavaScript", generate queries like:
               ["key features of Python programming language", "advantages of JavaScript for web development", "use cases for Python vs JavaScript", "performance comparison of Python and JavaScript", "ease of learning Python vs JavaScript"]

            3. For the query "How does climate change affect biodiversity?", generate queries like:
               ["impact of climate change on species extinction", "effects of global warming on ecosystems", "role of climate change in habitat loss", "how rising temperatures affect marine biodiversity", "climate change and its impact on migratory patterns"]

            4. For the query "Best practices for remote work", generate queries like:
               ["tips for staying productive while working from home", "how to maintain work-life balance in remote work", "tools for effective remote team collaboration", "managing communication in remote teams", "ergonomic setup for home offices"]

            5. For the query "What is quantum computing?", generate queries like:
               ["basic principles of quantum computing", "applications of quantum computing in real-world problems", "difference between classical and quantum computing", "key challenges in developing quantum computers", "future prospects of quantum computing"]

            Original query: {original_query}
            """

    input_data = "Follow the above instructions."

    sub_questions: list[str] = []
    for attempt in range(3):
        try:
            logging.info(f"Generating sub-questions (attempt {attempt + 1})")

            messages_payload = _build_messages(
                system_prompt=sub_question_generation_prompt,
                user_prompt=input_data,
            )
            response = _call_adapter_text(
                api_endpoint=api_endpoint,
                messages_payload=messages_payload,
                temperature=0.7,
                app_config=get_loaded_config(),
            )
            if response:
                try:
                    # Try to parse as JSON first
                    parsed_response = json.loads(response)
                    if isinstance(parsed_response, list):
                        sub_questions = _sanitize_sub_questions(parsed_response)
                    elif isinstance(parsed_response, dict):
                        sub_questions = _sanitize_sub_questions(
                            parsed_response.get("sub_questions", parsed_response.get("search_queries", []))
                        )
                    else:
                        sub_questions = []
                    if sub_questions:
                        logging.info("Successfully generated sub-questions from JSON")
                        break
                except json.JSONDecodeError:
                    # If JSON parsing fails, attempt a regex-based fallback
                    logging.warning("Failed to parse as JSON. Attempting regex extraction.")
                    matches = re.findall(r'"([^"]*)"', response)
                    sub_questions = _sanitize_sub_questions(matches)
                    if sub_questions:
                        logging.info("Successfully extracted sub-questions using regex")
                        break

        except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as e:
            logging.error(f"Error generating sub-questions: {str(e)}")

    if not sub_questions:
        logging.error("Failed to extract sub-questions from API response after all attempts.")
        sub_questions = []

    # Construct and return the result dictionary
    logging.info("Sub-questions generated successfully")
    return {
        "main_goal": original_query,
        "sub_questions": sub_questions,
        "search_queries": sub_questions,
        "analysis_prompt": sub_question_generation_prompt
    }


######################### Relevance Analysis #########################
#
# FIXME - Ensure edge cases are handled properly / Structured outputs?
async def search_result_relevance(
    search_results: list[dict],
    original_question: str,
    sub_questions: list[str],
    api_endpoint: str,
    cancel_event: asyncio.Event | None = None,
) -> dict[str, dict]:
    """
    Evaluate whether each search result is relevant to the original question and sub-questions.

    Args:
        search_results (List[Dict]): List of search results to evaluate.
        original_question (str): The original question posed by the user.
        sub_questions (List[str]): List of sub-questions generated from the original question.
        api_endpoint (str): The LLM or API endpoint to use for relevance analysis.

    Returns:
        Dict[str, Dict]: A dictionary of relevant results, keyed by a unique ID or index.
    """
    relevant_results = {}

    # Summarization prompt template
    summarization_prompt = """
    Summarize the following text in a concise way that captures the key information relevant to this question: "{question}"

    Text to summarize:
    {content}

    Instructions:
    1. Focus on information relevant to the question
    2. Keep the summary under 2000 characters
    3. Maintain factual accuracy
    4. Include key details and statistics if present
    """

    # Simple circuit breaker for LLM provider
    cfg = get_loaded_config()
    ws_section = cfg.get('Web-Scraping', {}) or {}
    breaker = _get_websearch_circuit_breaker(
        fail_threshold=int(ws_section.get('llm_cb_fail_threshold', 3) or 3),
        reset_after_s=float(ws_section.get('llm_cb_reset_after_s', 30) or 30.0),
        provider_key=api_endpoint,
    )

    timeouts = _get_llm_timeouts()
    jitter_ms = _get_relevance_jitter_ms()

    for idx, result in enumerate(search_results):
        if cancel_event and cancel_event.is_set():
            break
        content = _build_result_fallback_content(result)
        if not content:
            logging.error("No Content found in search results array!")
            continue

        # First, evaluate relevance
        eval_prompt = f"""
                Given the following search results for the user's question: "{original_question}" and the generated sub-questions: {sub_questions}, evaluate the relevance of the search result to the user's question.
                Explain your reasoning for selection.

                Search Results:
                {content}

                Instructions:
                1. You MUST only answer TRUE or False while providing your reasoning for your answer.
                2. A result is relevant if the result most likely contains comprehensive and relevant information to answer the user's question.
                3. Provide a brief reason for selection.

                You MUST respond using EXACTLY this format and nothing else:

                Selected Answer: [True or False]
                Reasoning: [Your reasoning for the selections]
                """
        input_data = "Evaluate the relevance of the search result."
        messages_payload = _build_messages(
            system_prompt=eval_prompt,
            user_prompt=input_data,
        )

        try:
            # Optional jitter
            if jitter_ms > 0:
                await asyncio.sleep(jitter_ms / 1000.0)

            # Evaluate relevance with timeout and circuit breaker
            if not breaker.can_attempt():
                logging.warning("LLM circuit breaker open; skipping relevance evaluation")
                continue

            async def _llm_call(_messages_payload=messages_payload):
                return await asyncio.to_thread(
                    lambda _mp=_messages_payload: chat_api_call(
                        api_endpoint=api_endpoint,
                        messages_payload=_mp,
                        temperature=0.7,
                        app_config=get_loaded_config(),
                        timeout=timeouts["llm"],
                    )
                )

            relevancy_result = await asyncio.wait_for(_llm_call(), timeout=timeouts["llm"])

            # Verbose debug for provider output
            logging.debug(f"[DEBUG] Relevancy LLM response for index {idx}:\n{relevancy_result}\n---")

            if relevancy_result:
                # Extract the selected answer and reasoning via regex
                logging.debug(f"LLM Relevancy Response for item: {relevancy_result}")
                selected_answer_match = re.search(
                    r"Selected Answer:\s*(True|False)",
                    relevancy_result,
                    re.IGNORECASE
                )
                reasoning_match = re.search(
                    r"Reasoning:\s*(.+)",
                    relevancy_result,
                    re.IGNORECASE
                )

                if selected_answer_match and reasoning_match:
                    is_relevant = selected_answer_match.group(1).strip().lower() == "true"
                    reasoning = reasoning_match.group(1).strip()

                    if is_relevant:
                        logging.debug("Relevant result found.")
                        # Use the 'id' from the result if available, otherwise use idx
                        result_id = result.get("id", str(idx))
                        source_content = content
                        try:
                            scraped_content = await asyncio.wait_for(
                                scrape_article(result['url']), timeout=timeouts["scrape"]
                            )
                            scraped_text = ""
                            if isinstance(scraped_content, dict):
                                scraped_text = str(scraped_content.get('content') or "").strip()
                            elif isinstance(scraped_content, str):
                                scraped_text = scraped_content.strip()
                            if scraped_text:
                                source_content = scraped_text
                        except asyncio.CancelledError:
                            raise
                        except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as scrape_error:
                            logging.warning(
                                f"Scrape failed for relevant result {result_id}; "
                                f"falling back to search snippet/title/url: {scrape_error}"
                            )

                        # Create Summarization prompt
                        logging.debug(f"Creating Summarization Prompt for result idx={idx}")
                        summary_prompt = summarization_prompt.format(
                            question=original_question,
                            content=source_content
                        )

                        # Generate summary using the summarize function with timeout
                        logging.info(f"Summarizing relevant result: ID={result_id}")
                        async def _summ_call(_source_content=source_content, _summary_prompt=summary_prompt):
                            return await asyncio.to_thread(
                                lambda _sc=_source_content, _sp=_summary_prompt: summarize(
                                    input_data=_sc,
                                    custom_prompt_arg=_sp,
                                    api_name=api_endpoint,
                                    api_key=None,
                                    temp=0.7,
                                    system_message=None,
                                    streaming=False,
                                )
                            )
                        try:
                            summary = await asyncio.wait_for(_summ_call(), timeout=timeouts["llm"])
                        except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as e:
                            logging.error(f"Summary generation failed: {e}")
                            summary = _truncate_text(source_content, max_len=2000) or "Summary generation failed"

                        relevant_results[result_id] = {
                            "content": summary,  # Store the summary instead of full content
                            "original_content": source_content,  # Keep original content if needed
                            "reasoning": reasoning
                        }
                        logging.info(f"Relevant result found and summarized: ID={result_id}; Reasoning={reasoning}")
                    else:
                        logging.info(f"Irrelevant result: {reasoning}")

                else:
                    logging.warning("Failed to parse the API response for relevance analysis.")
            breaker.record_success()
        except asyncio.TimeoutError:
            breaker.record_failure()
            logging.error(f"Timeout during LLM/scrape for result idx={idx}")
        except asyncio.CancelledError:
            logging.warning("Relevance evaluation cancelled")
            raise
        except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as e:
            breaker.record_failure()
            logging.error(f"Error during relevance evaluation/summarization for result idx={idx}: {e}")

    return relevant_results


def review_and_select_results(web_search_results_dict: dict, selector: callable | None = None) -> dict:
    """
    Allows the user to review and select relevant results from the search results.

    Args:
        web_search_results_dict (Dict): The dictionary containing all search results.

    Returns:
        Dict: A dictionary containing only the user-selected relevant results.
    """
    if not isinstance(web_search_results_dict, dict):
        return {}

    if "results" in web_search_results_dict and isinstance(web_search_results_dict.get("results"), list):
        candidates = [
            (
                str(result.get("id", idx)) if isinstance(result, dict) else str(idx),
                result,
            )
            for idx, result in enumerate(web_search_results_dict.get("results", []))
            if isinstance(result, dict)
        ]
    else:
        candidates = [
            (str(result_id), result)
            for result_id, result in web_search_results_dict.items()
            if isinstance(result, dict)
        ]

    # If no selector provided, default to keeping all results as relevant while preserving IDs
    if selector is None:
        return {result_id: result for result_id, result in candidates}

    relevant_results: dict[str, dict] = {}
    for result_id, result in candidates:
        try:
            if selector(result):
                relevant_results[result_id] = result
        except _WEBSEARCH_NONCRITICAL_EXCEPTIONS:
            # If selector throws, skip selection for this item
            continue
    return relevant_results


######################### Result Aggregation & Combination #########################
#
class FinalAnswerDict(TypedDict):
    """Structured payload returned by the aggregation phase."""
    text: str
    evidence: list[dict[str, Any]]
    confidence: float
    chunks: list[dict[str, Any]]


def aggregate_results(
    relevant_results: dict[str, dict],
    question: str,
    sub_questions: list[str],
    api_endpoint: str | None,
) -> FinalAnswerDict:
    """
    Combines and summarizes relevant results into a final answer.

    Args:
        relevant_results (Dict[str, Dict]): Dictionary of relevant articles/content.
        question (str): Original question.
        sub_questions (List[str]): List of sub-questions.
        api_endpoint (str): LLM or API endpoint for summarization.

    Returns:
        Dict containing:
        - summary (str): Final summarized answer.
        - evidence (List[Dict]): List of relevant content items included in the summary.
        - confidence (float): A rough confidence score (placeholder).
    """
    logging.info("Aggregating and summarizing relevant results")
    if not relevant_results:
        no_results: FinalAnswerDict = {
            "text": "No relevant results found. Unable to provide an answer.",
            "evidence": [],
            "confidence": 0.0,
            "chunks": [],
        }
        return no_results

    logging.info("Summarizing relevant results")

    def _build_chunk_infos(
        items: list[tuple[str, dict[str, Any]]],
        max_chars: int = 6000,
    ) -> list[dict[str, Any]]:
        chunk_infos: list[dict[str, Any]] = []
        current_entries: list[tuple[str, str]] = []
        current_length = 0

        def flush_entries() -> None:
            nonlocal current_entries, current_length
            if not current_entries:
                return
            text = "\n\n".join(entry for _, entry in current_entries)
            chunk_infos.append({
                "index": len(chunk_infos) + 1,
                "result_ids": [rid for rid, _ in current_entries],
                "text": text,
                "truncated": False,
            })
            current_entries = []
            current_length = 0

        for rid, res in items:
            entry = f"ID: {rid}\nContent: {res.get('content', '')}\nReasoning: {res.get('reasoning', '')}"
            entry_length = len(entry)
            if entry_length >= max_chars:
                flush_entries()
                chunk_infos.append({
                    "index": len(chunk_infos) + 1,
                    "result_ids": [rid],
                    "text": entry[:max_chars],
                    "truncated": True,
                })
                continue

            if current_length + entry_length > max_chars and current_entries:
                flush_entries()

            current_entries.append((rid, entry))
            current_length += entry_length

        flush_entries()
        return chunk_infos

    def _estimate_confidence(
        relevant_count: int,
        chunk_count: int,
        failed_chunks: int,
        has_llm: bool,
    ) -> float:
        if relevant_count <= 0:
            return 0.0
        coverage = min(relevant_count, 10) / 10.0
        chunk_success = 1.0 if chunk_count == 0 else (chunk_count - failed_chunks) / chunk_count
        base = 0.35 + 0.45 * coverage
        modifier = 0.6 + 0.4 * chunk_success
        llm_bonus = 0.1 if has_llm and failed_chunks == 0 else (0.05 if has_llm else 0.0)
        confidence = base * modifier + llm_bonus
        return max(0.1, min(0.99, round(confidence, 3)))

    result_items = list(relevant_results.items())
    chunk_infos = _build_chunk_infos(result_items)
    chunk_assignments: dict[str, int] = {}
    for info in chunk_infos:
        for rid in info["result_ids"]:
            chunk_assignments[rid] = info["index"]

    chunk_metadata: list[dict[str, Any]] = []
    evidence_payload: list[dict[str, Any]] = []

    for rid, res in relevant_results.items():
        evidence_payload.append({
            "id": rid,
            "content": res.get("content"),
            "original_content": res.get("original_content"),
            "reasoning": res.get("reasoning"),
            "chunk_index": chunk_assignments.get(rid),
        })

    if not api_endpoint:
        logging.warning("No final answer LLM configured; returning evidence summaries only.")
        for info in chunk_infos:
            preview = info["text"][:1500]
            chunk_metadata.append({
                "chunk_index": info["index"],
                "result_ids": info["result_ids"],
                "summary": preview,
                "generated": False,
                "source_characters": len(info["text"]),
                "truncated_source": info["truncated"],
            })
        combined_text = "\n\n".join(entry.get("content", "") or "" for entry in relevant_results.values())
        fallback_answer: FinalAnswerDict = {
            "text": combined_text or "Unable to generate a final answer without an LLM.",
            "evidence": evidence_payload,
            "confidence": _estimate_confidence(len(evidence_payload), len(chunk_infos), 0, has_llm=False),
            "chunks": chunk_metadata,
        }
        return fallback_answer

    summarized_chunks: list[str] = []
    failed_chunks = 0

    for info in chunk_infos:
        chunk_prompt = f"""
            Summarize the following set of relevant search snippets into a concise digest that preserves
            high-signal facts for answering the question: "{question}".

            Requirements:
            1. Keep the summary under 1500 characters.
            2. Focus on verifiable facts and key statistics.
            3. Mention the reasoning tags when helpful.

            <chunk id="{info['index']}">
            {info['text']}
            </chunk>
            """
        try:
            chunk_summary = summarize(
                input_data=info["text"],
                custom_prompt_arg=chunk_prompt,
                api_name=api_endpoint,
                api_key=None,
                temp=0.3,
                system_message=None,
                streaming=False,
            )
            generated = True
        except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as chunk_error:
            failed_chunks += 1
            logging.warning(f"Chunk summarization failed for chunk {info['index']}: {chunk_error}")
            chunk_summary = info["text"][:1500]
            generated = False

        chunk_metadata.append({
            "chunk_index": info["index"],
            "result_ids": info["result_ids"],
            "summary": chunk_summary,
            "generated": generated,
            "source_characters": len(info["text"]),
            "truncated_source": info["truncated"],
        })
        summarized_chunks.append(f"Chunk {info['index']} Summary:\n{chunk_summary}")

    context_payload = "\n\n".join(summarized_chunks)
    current_date = time.strftime("%Y-%m-%d")

    # Aggregation Prompt #1

    # Aggregation Prompt #2
    analyze_search_results_prompt_2 = (
        """INITIAL_QUERY: Here are some sources {context_payload}. Read these carefully, as you will be asked a Query about them.
        # General Instructions

        Write an accurate, detailed, and comprehensive response to the user's query located at INITIAL_QUERY. Additional context is provided as "USER_INPUT" after specific questions. Your answer should be informed by the provided "Search results". Your answer must be precise, of high-quality, and written by an expert using an unbiased and journalistic tone. Your answer must be written in the same language as the query, even if language preference is different.

        You MUST cite the most relevant search results that answer the query. Do not mention any irrelevant results. You MUST ADHERE to the following instructions for citing search results:
        - to cite a search result, enclose its index located above the summary with brackets at the end of the corresponding sentence, for example "Ice is less dense than water[1][2]." or "Paris is the capital of France[1][4][5]."
        - NO SPACE between the last word and the citation, and ALWAYS use brackets. Only use this format to cite search results. NEVER include a References section at the end of your answer.
        - If you don't know the answer or the premise is incorrect, explain why.
        If the search results are empty or unhelpful, answer the query as well as you can with existing knowledge.

        You MUST NEVER use moralization or hedging language. AVOID using the following phrases:
        - "It is important to ..."
        - "It is inappropriate ..."
        - "It is subjective ..."

        You MUST ADHERE to the following formatting instructions:
        - Use markdown to format paragraphs, lists, tables, and quotes whenever possible.
        - Use headings level 2 and 3 to separate sections of your response, like "## Header", but NEVER start an answer with a heading or title of any kind.
        - Use single new lines for lists and double new lines for paragraphs.
        - Use markdown to render images given in the search results.
        - NEVER write URLs or links.

        # Query type specifications

        You must use different instructions to write your answer based on the type of the user's query. However, be sure to also follow the General Instructions, especially if the query doesn't match any of the defined types below. Here are the supported types.

        ## Academic Research

        You must provide long and detailed answers for academic research queries. Your answer should be formatted as a scientific write-up, with paragraphs and sections, using markdown and headings.

        ## Recent News

        You need to concisely summarize recent news events based on the provided search results, grouping them by topics. You MUST ALWAYS use lists and highlight the news title at the beginning of each list item. You MUST choose news from diverse perspectives while also prioritizing trustworthy sources. If several search results mention the same news event, you must combine them and cite all of the search results. Prioritize more recent events, ensuring to compare timestamps. You MUST NEVER start your answer with a heading of any kind.

        ## Weather

        Your answer should be very short and only provide the weather forecast. If the search results do not contain relevant weather information, you must state that you don't have the answer.

        ## People

        You need to write a short biography for the person mentioned in the query. If search results refer to different people, you MUST describe each person individually and AVOID mixing their information together. NEVER start your answer with the person's name as a header.

        ## Coding

        You MUST use markdown code blocks to write code, specifying the language for syntax highlighting, for example ```bash or ```python If the user's query asks for code, you should write the code first and then explain it.

        ## Cooking Recipes

        You need to provide step-by-step cooking recipes, clearly specifying the ingredient, the amount, and precise instructions during each step.

        ## Translation

        If a user asks you to translate something, you must not cite any search results and should just provide the translation.

        ## Creative Writing

        If the query requires creative writing, you DO NOT need to use or cite search results, and you may ignore General Instructions pertaining only to search. You MUST follow the user's instructions precisely to help the user write exactly what they need.

        ## Science and Math

        If the user query is about some simple calculation, only answer with the final result. Follow these rules for writing formulas:
        - Always use \\( and\\) for inline formulas and\\[ and\\] for blocks, for example\\(x^4 = x - 3 \\)
        - To cite a formula add citations to the end, for example\\[ \\sin(x) \\] [1][2] or \\(x^2-2\\) [4].
        - Never use $ or $$ to render LaTeX, even if it is present in the user query.
        - Never use unicode to render math expressions, ALWAYS use LaTeX.
        - Never use the \\label instruction for LaTeX.

        ## URL Lookup

        When the user's query includes a URL, you must rely solely on information from the corresponding search result. DO NOT cite other search results, ALWAYS cite the first result, e.g. you need to end with [1]. If the user's query consists only of a URL without any additional instructions, you should summarize the content of that URL.

        ## Shopping

        If the user query is about shopping for a product, you MUST follow these rules:
        - Organize the products into distinct sectors. For example, you could group shoes by style (boots, sneakers, etc.)
        - Cite at most 9 search results using the format provided in General Instructions to avoid overwhelming the user with too many options.

        The current date is: {current_date}

        The user's query is: {question}
        """.format(
            context_payload=context_payload,
            current_date=current_date,
            question=question,
        )  # nosec B608
    )

    input_data = "Follow the above instructions."
    messages_payload = _build_messages(
        system_prompt=analyze_search_results_prompt_2,
        user_prompt=input_data,
    )

    try:
        logging.info("Generating the report")
        returned_response = chat_api_call(
            api_endpoint=api_endpoint,
            messages_payload=messages_payload,
            temperature=0.7,
            app_config=get_loaded_config(),
        )
        logging.debug("Returned response from LLM for aggregation: %s", returned_response)
        if returned_response:
            success_answer: FinalAnswerDict = {
                "text": returned_response,
                "evidence": evidence_payload,
                "confidence": _estimate_confidence(
                    len(evidence_payload),
                    len(chunk_infos),
                    failed_chunks,
                    has_llm=True,
                ),
                "chunks": chunk_metadata,
            }
            return success_answer
    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as e:
        logging.error(f"Error aggregating results: {e}")

    logging.error("Could not create the report due to an error.")
    failure_answer: FinalAnswerDict = {
        "text": "Could not create the report due to an error.",
        "evidence": evidence_payload,
        "confidence": _estimate_confidence(
            len(evidence_payload),
            len(chunk_infos),
            failed_chunks=len(chunk_infos),
            has_llm=False,
        ),
        "chunks": chunk_metadata,
    }
    return failure_answer

#
# End of Orchestration functions
#######################################################################################################################


#######################################################################################################################
#
# Discussion / Forum Search

# Platform → domain mapping for site-restricted searches
_DISCUSSION_PLATFORM_DOMAINS: dict[str, str] = {
    "reddit": "reddit.com",
    "stackoverflow": "stackoverflow.com",
    "hackernews": "news.ycombinator.com",
    "hn": "news.ycombinator.com",
    "stackexchange": "stackexchange.com",
    "quora": "quora.com",
    "4chan": "boards.4chan.org",
    "4channel": "boards.4channel.org",
}


async def search_discussions(
    query: str,
    platforms: list[str] | None = None,
    max_results: int = 10,
    search_engine: str = "duckduckgo",
) -> list[dict[str, Any]]:
    """Search discussion platforms for community knowledge and opinions.

    Appends ``site:<domain>`` filters to the query and dispatches to
    the existing ``perform_websearch`` / ``process_web_search_results``
    infrastructure.

    Args:
        query: The search query.
        platforms: Platform names to search (default: reddit, stackoverflow, hackernews).
        max_results: Maximum total results to return.
        search_engine: Web search engine to use (default: duckduckgo).

    Returns:
        List of result dicts with keys: title, url, content, source, platform.
    """
    if platforms is None:
        platforms = ["reddit", "stackoverflow", "hackernews"]

    # Build site: filter string
    site_filters: list[str] = []
    for platform in platforms:
        domain = _DISCUSSION_PLATFORM_DOMAINS.get(platform.lower())
        if domain:
            site_filters.append(f"site:{domain}")

    if not site_filters:
        return []

    # Combine into a single OR-joined filter
    site_clause = " OR ".join(site_filters)
    augmented_query = f"{query} ({site_clause})"

    # Dispatch to existing web search (sync function, run in thread)
    try:
        raw_results = await asyncio.to_thread(
            perform_websearch,
            search_engine=search_engine,
            search_query=augmented_query,
            content_country="US",
            search_lang="en",
            output_lang="en",
            result_count=max_results,
        )

        results_list: list[dict[str, Any]] = []
        if isinstance(raw_results, dict):
            existing_results = raw_results.get("results")
            if isinstance(existing_results, list):
                normalized_results = [item for item in existing_results if isinstance(item, dict)]
                if normalized_results and any(
                    ("url" in item) or ("content" in item) or ("snippet" in item)
                    for item in normalized_results
                ):
                    results_list = normalized_results

        if not results_list:
            processed = process_web_search_results(raw_results, search_engine)
            results_list = processed.get("results", [])

        docs: list[dict[str, Any]] = []
        for r in results_list[:max_results]:
            url = r.get("url", "")
            # Detect which platform the result came from
            detected_platform = "unknown"
            url_lower = url.lower()
            for plat, domain in _DISCUSSION_PLATFORM_DOMAINS.items():
                if domain in url_lower:
                    detected_platform = plat
                    break

            docs.append({
                "title": r.get("title", ""),
                "url": url,
                "content": str(r.get("content", r.get("snippet", "")))[:500],
                "source": "discussion",
                "platform": detected_platform,
            })

        return docs

    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as exc:
        logging.warning(f"Discussion search failed: {exc}")
        return []


#
# End of Discussion / Forum Search
#######################################################################################################################


#######################################################################################################################
#
# Search Engine Functions

# FIXME
def perform_websearch(search_engine, search_query, content_country, search_lang, output_lang, result_count, date_range=None,
                      safesearch=None, site_blacklist=None, exactTerms=None, excludeTerms=None, filter=None, geolocation=None, search_result_language=None, sort_results_by=None, search_params=None, site_whitelist=None, google_domain=None):
    try:
        if search_engine.lower() == "baidu":
            web_search_results = search_web_baidu(search_query, None, None)

        elif search_engine.lower() == "bing":
            # Provider deprecated
            raise ValueError("Bing provider is deprecated and not supported")

        elif search_engine.lower() == "brave":
            # Call using explicit keywords to avoid argument misalignment
            web_search_results = search_web_brave(
                search_term=search_query,
                country=content_country,
                search_lang=search_lang,
                ui_lang=output_lang,
                result_count=result_count,
                safesearch=safesearch or "moderate",
                date_range=date_range,
                site_blacklist=site_blacklist,
            )

        elif search_engine.lower() == "duckduckgo":
            # Prepare the arguments for search_web_duckduckgo
            ddg_args = {
                "keywords": search_query,
                "region": f"{content_country.lower()}-{search_lang.lower()}",  # Format: "us-en"
                "timelimit": date_range[0] if date_range else None,  # Use first character of date_range (e.g., "y" -> "y")
                "max_results": result_count,
            }

            # Call the search_web_duckduckgo function with the prepared arguments
            ddg_results = search_web_duckduckgo(**ddg_args)

            # Wrap the results in a dictionary to match the expected format
            web_search_results = {"results": ddg_results}

        elif search_engine.lower() == "google":
            site_whitelist_list = site_whitelist if isinstance(site_whitelist, list) else None
            site_blacklist_list = site_blacklist if isinstance(site_blacklist, list) else None
            site_whitelist_domains = (
                [domain.strip() for domain in site_whitelist.split(",") if domain.strip()]
                if isinstance(site_whitelist, str)
                else None
            )
            site_blacklist_domains = (
                [domain.strip() for domain in site_blacklist.split(",") if domain.strip()]
                if isinstance(site_blacklist, str)
                else None
            )
            site_blacklist_value: str | None
            if site_blacklist_list:
                site_blacklist_value = ",".join(site_blacklist_list)
            elif isinstance(site_blacklist, str):
                site_blacklist_value = site_blacklist
            else:
                site_blacklist_value = None

            # Prepare the arguments for search_web_google
            google_args = {
                "search_query": search_query,
                "google_search_api_key": get_loaded_config()['search_engines']['google_search_api_key'],
                "google_search_engine_id": get_loaded_config()['search_engines']['google_search_engine_id'],
                "result_count": result_count,
                "c2coff": "1",  # Default value
                "results_origin_country": content_country,
                "ui_language": output_lang,
                "search_result_language": search_result_language or "lang_en",  # Default value
                "geolocation": geolocation or "us",  # Default value
                "safesearch": safesearch or "off",  # Default value,
                "google_domain": google_domain,
            }

            # Prefer include-domain filter when present; otherwise apply exclude-domain filter.
            if site_whitelist_list and len(site_whitelist_list) == 1:
                google_args["siteSearch"] = site_whitelist_list[0]
                google_args["siteSearchFilter"] = "i"
            elif site_whitelist_domains and len(site_whitelist_domains) == 1:
                google_args["siteSearch"] = site_whitelist_domains[0]
                google_args["siteSearchFilter"] = "i"
            elif site_blacklist_list and len(site_blacklist_list) == 1:
                google_args["siteSearch"] = site_blacklist_list[0]
                google_args["siteSearchFilter"] = "e"
            elif site_blacklist_domains and len(site_blacklist_domains) == 1:
                google_args["siteSearch"] = site_blacklist_domains[0]
                google_args["siteSearchFilter"] = "e"

            # Add optional parameters only if they are provided
            if date_range:
                google_args["date_range"] = date_range
            if exactTerms:
                google_args["exactTerms"] = exactTerms
            if excludeTerms:
                google_args["excludeTerms"] = excludeTerms
            if filter:
                google_args["filter"] = filter
            if site_blacklist_value:
                google_args["site_blacklist"] = site_blacklist_value
            if sort_results_by:
                google_args["sort_results_by"] = sort_results_by

            # Call the search_web_google function with the prepared arguments
            web_search_results = search_web_google(**google_args)  # raw JSON
            web_search_results_dict = process_web_search_results(web_search_results, "google")
            return web_search_results_dict

        elif search_engine.lower() == "kagi":
            # Limit uses result_count; content_country is not applicable
            web_search_results = search_web_kagi(query=search_query, limit=result_count)

        elif search_engine.lower() == "serper":
            web_search_results = search_web_serper(
                search_query=search_query,
                result_count=result_count,
                content_country=content_country,
                search_lang=search_lang,
                output_lang=output_lang,
                date_range=date_range,
                safesearch=safesearch,
                site_whitelist=site_whitelist,
                site_blacklist=site_blacklist,
                exactTerms=exactTerms,
                excludeTerms=excludeTerms,
            )

        elif search_engine.lower() == "tavily":
            web_search_results = search_web_tavily(
                search_query=search_query,
                result_count=result_count,
                site_whitelist=site_whitelist,
                site_blacklist=site_blacklist,
            )

        elif search_engine.lower() == "exa":
            web_search_results = search_web_exa(
                search_query=search_query,
                result_count=result_count,
                content_country=content_country,
                site_whitelist=site_whitelist,
                site_blacklist=site_blacklist,
            )

        elif search_engine.lower() == "firecrawl":
            web_search_results = search_web_firecrawl(
                search_query=search_query,
                result_count=result_count,
                content_country=content_country,
                date_range=date_range,
            )

        elif search_engine.lower() == "4chan":
            web_search_results = search_web_4chan(
                search_query=search_query,
                result_count=result_count,
                search_params=search_params,
            )

        elif search_engine.lower() == "searx":
            web_search_results = search_web_searx(
                search_query,
                language='auto',
                time_range=date_range or '',
                safesearch=_map_searx_safesearch(safesearch),
                pageno=1,
                categories='general',
                searx_url=(search_params or {}).get('searx_url'),
                json_mode=(search_params or {}).get('searx_json_mode', False),
            )

        elif search_engine.lower() == "yandex":
            web_search_results = search_web_yandex()

        elif search_engine.lower() in {"sogou", "startpage", "stract"}:
            raise ValueError(f"{search_engine} provider not implemented")

        else:
            return {"processing_error": f"Error: Invalid Search Engine Name {search_engine}"}

        # Process the raw search results
        web_search_results_dict = process_web_search_results(web_search_results, search_engine)
        # FIXME
        #logging.debug("After process_web_search_results:")
        #logging.debug(json.dumps(web_search_results_dict, indent=2))
        return web_search_results_dict

    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as e:
        return {"processing_error": f"Error performing web search: {str(e)}"}


def test_perform_websearch_google():
    # Google Searches
    try:
        test_1 = perform_websearch("google", "What is the capital of France?", "US", "en", "en", 10)
        print(f"Test 1: {test_1}")
        # FIXME - Fails. Need to fix arg formatting
        test_2 = perform_websearch("google", "What is the capital of France?", "US", "en", "en", 10, date_range="y", safesearch="active", site_blacklist=["spam-site.com"])
        print(f"Test 2: {test_2}")
        test_3 = perform_websearch("google", "What is the capital of France?", "US", "en", "en", 10)
        print(f"Test 3: {test_3}")
    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as e:
        print(f"Error performing google searches: {str(e)}")
    pass


def test_perform_websearch_bing():
    # Deprecated provider; no-op test placeholder
    pass


def test_perform_websearch_brave():
    # Brave Searches
    try:
        test_7 = perform_websearch("brave", "What is the capital of France?", "US", "en", "en", 10)
        print(f"Test 7: {test_7}")
    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as e:
        print(f"Error performing brave searches: {str(e)}")


def test_perform_websearch_ddg():
    # DuckDuckGo Searches
    try:
        test_6 = perform_websearch("duckduckgo", "What is the capital of France?", "US", "en", "en", 10)
        print(f"Test 6: {test_6}")
        test_7 = perform_websearch("duckduckgo", "What is the capital of France?", "US", "en", "en", 10, date_range="y")
        print(f"Test 7: {test_7}")
    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as e:
        print(f"Error performing duckduckgo searches: {str(e)}")


# FIXME
def test_perform_websearch_kagi():
    # Kagi Searches
    try:
        test_8 = perform_websearch("kagi", "What is the capital of France?", "US", "en", "en", 10)
        print(f"Test 8: {test_8}")
    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as e:
        print(f"Error performing kagi searches: {str(e)}")

# FIXME
def test_perform_websearch_serper():
    # Serper Searches
    try:
        test_9 = perform_websearch("serper", "What is the capital of France?", "US", "en", "en", 10)
        print(f"Test 9: {test_9}")
    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as e:
        print(f"Error performing serper searches: {str(e)}")

# FIXME
def test_perform_websearch_tavily():
    # Tavily Searches
    try:
        test_10 = perform_websearch("tavily", "What is the capital of France?", "US", "en", "en", 10)
        print(f"Test 10: {test_10}")
    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as e:
        print(f"Error performing tavily searches: {str(e)}")


# FIXME
def test_perform_websearch_searx():
    # Searx Searches
    try:
        test_11 = perform_websearch("searx", "What is the capital of France?", "US", "en", "en", 10)
        print(f"Test 11: {test_11}")
    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as e:
        print(f"Error performing searx searches: {str(e)}")


# FIXME
def test_perform_websearch_yandex():
    #Yandex Searches
    try:
        test_12 = perform_websearch("yandex", "What is the capital of France?", "US", "en", "en", 10)
        print(f"Test 12: {test_12}")
    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as e:
        print(f"Error performing yandex searches: {str(e)}")
    pass

#
######################### Search Result Parsing ##################################################################
#

def process_web_search_results(search_results: dict, search_engine: str) -> dict:
    """
    Processes search results from a search engine and formats them into a standardized dictionary structure.

    Args:
        search_results (Dict): The raw search results from the search engine.
        search_engine (str): The name of the search engine (e.g., "Google", "Bing").

    Returns:
        Dict: A dictionary containing the processed search results in the specified structure.

    web_search_results_dict = {
        "search_engine": search_engine,
        "search_query": search_results.get("search_query", ""),
        "content_country": search_results.get("content_country", ""),
        "search_lang": search_results.get("search_lang", ""),
        "output_lang": search_results.get("output_lang", ""),
        "result_count": search_results.get("result_count", 0),
        "date_range": search_results.get("date_range", None),
        "safesearch": search_results.get("safesearch", None),
        "site_blacklist": search_results.get("site_blacklist", None),
        "exactTerms": search_results.get("exactTerms", None),
        "excludeTerms": search_results.get("excludeTerms", None),
        "filter": search_results.get("filter", None),
        "geolocation": search_results.get("geolocation", None),
        "search_result_language": search_results.get("search_result_language", None),
        "sort_results_by": search_results.get("sort_results_by", None),
        "results": [
            {
                "title": str,
                "url": str,
                "content": str,
                "metadata": {
                    "date_published": Optional[str],
                    "author": Optional[str],
                    "source": Optional[str],
                    "language": Optional[str],
                    "relevance_score": Optional[float],
                    "snippet": Optional[str]
                }
            },
        "total_results_found": search_results.get("total_results_found", 0),
        "search_time": search_results.get("search_time", 0.0),
        "error": search_results.get("error", None),
        "processing_error": None
    }
    """
    # Validate input parameters
    if not isinstance(search_results, dict):
        raise TypeError("search_results must be a dictionary")

    # Initialize the output dictionary with default values
    web_search_results_dict = {
        "search_engine": search_engine,
        "search_query": search_results.get("search_query", ""),
        "content_country": search_results.get("content_country", ""),
        "search_lang": search_results.get("search_lang", ""),
        "output_lang": search_results.get("output_lang", ""),
        "result_count": search_results.get("result_count", 0),
        "date_range": search_results.get("date_range"),
        "safesearch": search_results.get("safesearch"),
        "site_whitelist": search_results.get("site_whitelist"),
        "site_blacklist": search_results.get("site_blacklist"),
        "exactTerms": search_results.get("exactTerms"),
        "excludeTerms": search_results.get("excludeTerms"),
        "filter": search_results.get("filter"),
        "geolocation": search_results.get("geolocation"),
        "search_result_language": search_results.get("search_result_language"),
        "sort_results_by": search_results.get("sort_results_by"),
        "google_domain": search_results.get("google_domain"),
        "results": [],
        "total_results_found": search_results.get("total_results_found", 0),
        "search_time": search_results.get("search_time", 0.0),
        "error": search_results.get("error"),
        "processing_error": None
    }
    try:
        # Parse results based on the search engine
        if search_engine.lower() == "baidu":
            pass  # Placeholder for Baidu-specific parsing
        elif search_engine.lower() == "bing":
            parse_bing_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "brave":
            parse_brave_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "duckduckgo":
            parse_duckduckgo_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "google":
            parse_google_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "kagi":
            parse_kagi_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "serper":
            parse_serper_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "tavily":
            parse_tavily_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "exa":
            parse_exa_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "firecrawl":
            parse_firecrawl_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "4chan":
            parse_4chan_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "searx":
            parse_searx_results(search_results, web_search_results_dict)
        elif search_engine.lower() == "yandex":
            parse_yandex_results(search_results, web_search_results_dict)
        else:
            raise ValueError(f"Error: Invalid Search Engine Name {search_engine}")

    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as e:
        web_search_results_dict["processing_error"] = f"Error processing search results: {str(e)}"
        logging.error(f"Error in process_web_search_results: {str(e)}")

    return web_search_results_dict


def parse_html_search_results_generic(soup):
    results = []
    for result in soup.find_all('div', class_='result'):
        title = result.find('h3').text if result.find('h3') else ''
        url = result.find('a', class_='url')['href'] if result.find('a', class_='url') else ''
        content = result.find('p', class_='content').text if result.find('p', class_='content') else ''
        published_date = result.find('span', class_='published_date').text if result.find('span',
                                                                                          class_='published_date') else ''

        results.append({
            'title': title,
            'url': url,
            'content': content,
            'publishedDate': published_date
        })
    return results


######################### Baidu Search #########################
#
# https://cloud.baidu.com/doc/APIGUIDE/s/Xk1myz05f
# https://oxylabs.io/blog/how-to-scrape-baidu-search-results
def search_web_baidu(arg1, arg2, arg3):
    """Baidu provider stub with egress policy check."""
    _enforce_provider_outbound_policy(
        "https://www.baidu.com",
        source="websearch_baidu",
    )
    return {"error": "Baidu provider not implemented"}


def test_baidu_search(arg1, arg2, arg3):
    result = search_web_baidu(arg1, arg2, arg3)
    return result

def search_parse_baidu_results():
    pass


######################### Bing Search #########################
#
# https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/overview0
# https://learn.microsoft.com/en-us/bing/search-apis/bing-news-search/overview
# https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/quickstarts/rest/python
# https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/query-parameters
# Country/Language code: https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/market-codes#country-codes
# https://github.com/Azure-Samples/cognitive-services-REST-api-samples/tree/master/python/Search
def search_web_bing(*args, **kwargs):
    raise NotImplementedError("Bing provider is deprecated and has been removed")


def test_search_web_bing():
    pass


def parse_bing_results(raw_results: dict, output_dict: dict) -> None:
    # Deprecated
    output_dict.setdefault("processing_error", "Bing provider deprecated")


def brave_http_get(url: str, *, headers: dict[str, str], params: dict[str, Any]):
    return fetch(method="GET", url=url, headers=headers, params=params, timeout=15.0)


######################### Brave Search #########################
#
# https://brave.com/search/api/
# https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-brave-search/README.md
def search_web_brave(
    search_term,
    country,
    search_lang,
    ui_lang,
    result_count,
    safesearch="moderate",
    brave_api_key=None,
    result_filter=None,
    search_type="ai",
    date_range=None,
    site_blacklist: list[str] | None = None,
):
    search_url = "https://api.search.brave.com/res/v1/web/search"
    if search_type not in {"ai", "web"}:
        raise ValueError("Invalid search type. Please choose 'ai' or 'web'.")

    search_cfg = get_loaded_config().get("search_engines", {})
    if not brave_api_key:
        key_name = "brave_search_ai_api_key" if search_type == "ai" else "brave_search_api_key"
        brave_api_key = search_cfg.get(key_name)
    if not brave_api_key:
        raise ValueError("Please provide a valid Brave Search API subscription key")

    # Respect provided country; fallback to config default
    if not country:
        country = search_cfg.get("search_engine_country_code_brave", "US")
    if not search_lang:
        search_lang = "en"
    if not ui_lang:
        ui_lang = "en"
    if not result_count:
        result_count = search_cfg.get("search_result_max_per_query", 10)
    if not result_filter:
        result_filter = "webpages"
    safesearch = (safesearch or "moderate").capitalize()

    headers = {"Accept": "application/json", "Accept-Encoding": "gzip", "X-Subscription-Token": brave_api_key}

    # https://api.search.brave.com/app/documentation/web-search/query#WebSearchAPIQueryParameters
    params = {
        "q": search_term,
        "count": result_count,
        "freshness": date_range,
        "promote": result_filter,
        "safeSearch": safesearch,
        "source": search_type,
        "country": country,
        "search_lang": search_lang,
        "ui_lang": ui_lang,
    }

    if site_blacklist:
        params["exclude_sites"] = ",".join(site_blacklist)

    filtered_params = {key: value for key, value in params.items() if value is not None}

    _enforce_provider_outbound_policy(search_url, source="websearch_brave")

    # Response: https://api.search.brave.com/app/documentation/web-search/responses#WebSearchApiResponse
    response = brave_http_get(search_url, headers=headers, params=filtered_params)
    try:
        return response.json()
    finally:
        _close_response(response)


def test_search_brave():
    search_term = "How can I bake a cherry cake"
    country = "US"
    search_lang = "en"
    ui_lang = "en"
    result_count = 10
    safesearch = "moderate"
    date_range = None
    result_filter = None
    result = search_web_brave(search_term, country, search_lang, ui_lang, result_count, safesearch, date_range,
                             result_filter)
    print("Brave Search Results:")
    print(result)

    output_dict = {"results": []}
    parse_brave_results(result, output_dict)
    print("Parsed Brave Results:")
    print(json.dumps(output_dict, indent=2))


def parse_brave_results(raw_results: dict, output_dict: dict) -> None:
    """
    Parse Brave search results and update the output dictionary

    Args:
        raw_results (Dict): Raw Brave API response
        output_dict (Dict): Dictionary to store processed results
    """
    try:
        # Initialize results list if not present
        if "results" not in output_dict:
            output_dict["results"] = []

        # Extract query information
        if "query" in raw_results:
            query_info = raw_results["query"]
            output_dict.update({
                "search_query": query_info.get("original", ""),
                "content_country": query_info.get("country", ""),
                "city": query_info.get("city", ""),
                "state": query_info.get("state", ""),
                "more_results_available": query_info.get("more_results_available", False)
            })

        # Process web results
        if "web" in raw_results and "results" in raw_results["web"]:
            for result in raw_results["web"]["results"]:
                processed_result = {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("description", ""),
                    "metadata": {
                        "date_published": result.get("page_age", None),
                        "author": None,
                        "source": result.get("profile", {}).get("name", None),
                        "language": result.get("language", None),
                        "relevance_score": None,
                        "snippet": result.get("description", None),
                        "family_friendly": result.get("family_friendly", None),
                        "type": result.get("type", None),
                        "subtype": result.get("subtype", None),
                        "thumbnail": result.get("thumbnail", {}).get("src", None)
                    }
                }
                output_dict["results"].append(processed_result)

        # Update total results count
        if "mixed" in raw_results:
            output_dict["total_results_found"] = len(raw_results["mixed"].get("main", []))

        # Set family friendly status
        if "mixed" in raw_results:
            output_dict["family_friendly"] = raw_results.get("family_friendly", True)

    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as e:
        logging.error(f"Error processing Brave results: {str(e)}")
        output_dict["processing_error"] = f"Error processing Brave results: {str(e)}"

def test_parse_brave_results():
    pass


######################### DuckDuckGo Search #########################
#
# https://github.com/deedy5/duckduckgo_search
# Copied request format/structure from https://github.com/deedy5/duckduckgo_search/blob/main/duckduckgo_search/duckduckgo_search.py
def search_web_duckduckgo(
    keywords: str,
    region: str = "wt-wt",
    timelimit: str | None = None,
    max_results: int | None = None,
) -> list[dict[str, str]]:
    assert keywords, "keywords is mandatory"

    payload = {
        "q": keywords,
        "s": "0",
        "o": "json",
        "api": "d.js",
        "vqd": "",
        "kl": region,
        "bing_market": region,
    }

    def _normalize_url(url: str) -> str:
        """Unquote URL and replace spaces with '+'."""
        return unquote(url).replace(" ", "+") if url else ""

    def _normalize(raw_html: str) -> str:
        """Strip HTML tags from the raw_html string."""
        REGEX_STRIP_TAGS = re.compile("<.*?>")
        return unescape(REGEX_STRIP_TAGS.sub("", raw_html)) if raw_html else ""

    if timelimit:
        payload["df"] = timelimit

    cache = set()
    results: list[dict[str, str]] = []

    ddg_url = "https://html.duckduckgo.com/html"
    _enforce_provider_outbound_policy(
        ddg_url,
        source="websearch_duckduckgo_html",
    )

    headers = _websearch_browser_headers(restrict_encodings_for_requests=True)
    for _ in range(5):
        response = fetch(method="POST", url=ddg_url, data=payload, headers=headers, timeout=10.0)
        try:
            resp_content = response.content
        finally:
            _close_response(response)
        if b"No  results." in resp_content:
            return results

        tree = document_fromstring(resp_content)
        elements = tree.xpath("//div[h2]")
        if not isinstance(elements, list):
            return results

        for e in elements:
            if isinstance(e, _Element):
                hrefxpath = e.xpath("./a/@href")
                href = str(hrefxpath[0]) if hrefxpath and isinstance(hrefxpath, list) else None
                if (
                    href
                    and href not in cache
                    and not href.startswith(
                        ("http://www.google.com/search?q=", "https://duckduckgo.com/y.js?ad_domain")
                    )
                ):
                    cache.add(href)
                    titlexpath = e.xpath("./h2/a/text()")
                    title = str(titlexpath[0]) if titlexpath and isinstance(titlexpath, list) else ""
                    bodyxpath = e.xpath("./a//text()")
                    body = "".join(str(x) for x in bodyxpath) if bodyxpath and isinstance(bodyxpath, list) else ""
                    results.append(
                        {
                            "title": _normalize(title),
                            "href": _normalize_url(href),
                            "body": _normalize(body),
                        }
                    )
                    if max_results and len(results) >= max_results:
                        return results

        npx = tree.xpath('.//div[@class="nav-link"]')
        if not npx or not max_results:
            return results
        next_page = npx[-1] if isinstance(npx, list) else None
        if isinstance(next_page, _Element):
            names = next_page.xpath('.//input[@type="hidden"]/@name')
            values = next_page.xpath('.//input[@type="hidden"]/@value')
            if isinstance(names, list) and isinstance(values, list):
                payload = {str(n): str(v) for n, v in zip(names, values)}

    return results


def test_search_duckduckgo():
    try:
        results = search_web_duckduckgo(
            keywords="How can I bake a cherry cake?",
            region="us-en",
            timelimit="w",
            max_results=10
        )
        print(f"Number of results: {len(results)}")
        for result in results:
            print(f"Title: {result['title']}")
            print(f"URL: {result['href']}")
            print(f"Snippet: {result['body']}")
            print("---")

        # Parse the results
        output_dict = {"results": []}
        parse_duckduckgo_results({"results": results}, output_dict)
        print("Parsed DuckDuckGo Results:")
        print(json.dumps(output_dict, indent=2))

    except ValueError as e:
        print(f"Invalid input: {str(e)}")
    except (NetworkError, RetryExhaustedError) as e:
        print(f"Request error: {str(e)}")


def parse_duckduckgo_results(raw_results: dict, output_dict: dict) -> None:
    """
    Parse DuckDuckGo search results and update the output dictionary

    Args:
        raw_results (Dict): Raw DuckDuckGo response
        output_dict (Dict): Dictionary to store processed results
    """
    try:
        # Initialize results list if not present
        if "results" not in output_dict:
            output_dict["results"] = []

        # DuckDuckGo results are in a list of dictionaries
        results = raw_results.get("results", [])

        for result in results:
            # Extract information directly from the dictionary
            title = result.get("title", "")
            url = result.get("href", "")
            snippet = result.get("body", "")

            # Log warnings for missing data
            if not title:
                logging.warning("Missing title in result")
            if not url:
                logging.warning("Missing URL in result")
            if not snippet:
                logging.warning("Missing snippet in result")

            # Add the processed result to the output dictionary
            processed_result = {
                "title": title,
                "url": url,
                "content": snippet,
                "metadata": {
                    "date_published": None,  # DuckDuckGo doesn't typically provide this
                    "author": None,  # DuckDuckGo doesn't typically provide this
                    "source": extract_domain(url) if url else None,
                    "language": None,  # DuckDuckGo doesn't typically provide this
                    "relevance_score": None,  # DuckDuckGo doesn't typically provide this
                    "snippet": snippet
                }
            }

            output_dict["results"].append(processed_result)

        # Update total results count
        output_dict["total_results_found"] = len(output_dict["results"])

    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as e:
        logging.error(f"Error processing DuckDuckGo results: {str(e)}")
        output_dict["processing_error"] = f"Error processing DuckDuckGo results: {str(e)}"


def extract_domain(url: str) -> str:
    """
    Extract domain name from URL

    Args:
        url (str): Full URL

    Returns:
        str: Domain name
    """
    try:
        from urllib.parse import urlparse
        parsed_uri = urlparse(url)
        domain = parsed_uri.netloc
        return domain.replace('www.', '')
    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as e:
        logging.warning(f"Failed to extract domain from URL {url}: {str(e)}")
        return url


def test_parse_duckduckgo_results():
    pass



######################### Google Search #########################
#
# https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list
def search_web_google(
    search_query: str,
    google_search_api_key: str | None = None,
    google_search_engine_id: str | None = None,
    result_count: int | None = None,
    c2coff: str | None = None,
    results_origin_country: str | None = None,
    date_range: str | None = None,
    exactTerms: str | None = None,
    excludeTerms: str | None = None,
    filter: str | None = None,
    geolocation: str | None = None,
    ui_language: str | None = None,
    search_result_language: str | None = None,
    safesearch: str | None = None,
    google_domain: str | None = None,
    site_blacklist: str | None = None,
    siteSearch: str | None = None,
    siteSearchFilter: str | None = None,
    sort_results_by: str | None = None
) -> dict[str, Any]:
    """
    Perform a Google web search with the given parameters.

    :param search_query: The search query string
    :param google_search_api_key: Google Search API key
    :param google_search_engine_id: Google Search Engine ID
    :param result_count: Number of results to return
    :param c2coff: Enable/disable traditional Chinese search
    :param results_origin_country: Limit results to a specific country
    :param date_range: Limit results to a specific date range
    :param exactTerms: Exact terms that must appear in results
    :param excludeTerms: Terms that must not appear in results
    :param filter: Control duplicate content filter
    :param geolocation: Geolocation of the user
    :param ui_language: Language of the user interface
    :param search_result_language: Language of search results
    :param safesearch: Safe search setting
    :param google_domain: Google host/domain hint (e.g. "google.de")
    :param site_blacklist: Single Site to exclude from search
    :param siteSearch: Google CSE siteSearch parameter
    :param siteSearchFilter: Google CSE siteSearchFilter parameter (e=exclude, i=include)
    :param sort_results_by: Sorting criteria for results
    :return: JSON response from Google Search API
    """
    try:
        # Load Search API URL from config file
        search_url = get_loaded_config()['search_engines']['google_search_api_url']
        logging.info(f"Using search URL: {search_url}")

        # Initialize params dictionary
        query_value = search_query
        if site_blacklist:
            raw_domains = [domain.strip() for domain in str(site_blacklist).split(",") if domain.strip()]
            if len(raw_domains) > 1:
                cleaned_domains: list[str] = []
                for domain in raw_domains:
                    normalized_domain = domain
                    if normalized_domain.startswith("-site:"):
                        normalized_domain = normalized_domain[len("-site:"):]
                    elif normalized_domain.startswith("site:"):
                        normalized_domain = normalized_domain[len("site:"):]
                    normalized_domain = normalized_domain.strip()
                    if normalized_domain:
                        cleaned_domains.append(normalized_domain)
                if cleaned_domains:
                    query_value = " ".join(
                        [search_query, *[f"-site:{domain}" for domain in cleaned_domains]]
                    )
        params: dict[str, Any] = {"q": query_value}

        # Handle c2coff
        if c2coff is None:
            c2coff = get_loaded_config()['search_engines']['google_simp_trad_chinese']
        if c2coff is not None:
            params["c2coff"] = c2coff

        # Handle results_origin_country
        if results_origin_country is None:
            limit_country_search = get_loaded_config()['search_engines']['limit_google_search_to_country']
            if limit_country_search:
                results_origin_country = get_loaded_config()['search_engines']['google_search_country']
        if results_origin_country:
            params["cr"] = results_origin_country

        # Handle google_search_engine_id
        if google_search_engine_id is None:
            google_search_engine_id = get_loaded_config()['search_engines']['google_search_engine_id']
        if not google_search_engine_id:
            raise ValueError("Please set a valid Google Search Engine ID in the config file")
        params["cx"] = google_search_engine_id

        # Handle google_search_api_key
        if google_search_api_key is None:
            google_search_api_key = get_loaded_config()['search_engines']['google_search_api_key']
        if not google_search_api_key:
            raise ValueError("Please provide a valid Google Search API subscription key")
        params["key"] = google_search_api_key

        # Handle other parameters
        if result_count:
            params["num"] = result_count
        if date_range:
            params["dateRestrict"] = date_range
        if exactTerms:
            params["exactTerms"] = exactTerms
        if excludeTerms:
            params["excludeTerms"] = excludeTerms
        if filter:
            params["filter"] = filter
        if geolocation:
            params["gl"] = geolocation
        if ui_language:
            params["hl"] = ui_language
        if search_result_language:
            params["lr"] = search_result_language
        if safesearch is None:
            safesearch = get_loaded_config()['search_engines']['google_safe_search']
        if safesearch:
            params["safe"] = safesearch
        if google_domain:
            params["googlehost"] = google_domain
        if siteSearch:
            params["siteSearch"] = siteSearch
        if siteSearchFilter:
            params["siteSearchFilter"] = siteSearchFilter
        if sort_results_by:
            params["sort"] = sort_results_by

        logging.info(f"Prepared parameters for Google Search: {params}")

        _enforce_provider_outbound_policy(search_url, source="websearch_google")

        # Make the API call with centralized client
        from tldw_Server_API.app.core.http_client import fetch_json
        google_search_results = fetch_json(method="GET", url=search_url, params=params, timeout=15.0)

        logging.info(f"Successfully retrieved search results. Items found: {len(google_search_results.get('items', []))}")

        return google_search_results

    except ValueError as ve:
        logging.error(f"Configuration error: {str(ve)}")
        raise

    except (NetworkError, RetryExhaustedError) as re:
        logging.error(f"Error during API request: {str(re)}")
        raise

    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as e:
        logging.error(f"Unexpected error occurred: {str(e)}")
        raise


def test_search_google():
    search_query = "How can I bake a cherry cake?"
    google_search_api_key = get_loaded_config()['search_engines']['google_search_api_key']
    google_search_engine_id = get_loaded_config()['search_engines']['google_search_engine_id']
    result_count = 10
    c2coff = "1"
    results_origin_country = "countryUS"
    date_range = None
    exactTerms = None
    excludeTerms = None
    filter = None
    geolocation = "us"
    ui_language = "en"
    search_result_language = "lang_en"
    safesearch = "off"
    site_blacklist = None
    sort_results_by = None
    result = search_web_google(
        search_query=search_query,
        google_search_api_key=google_search_api_key,
        google_search_engine_id=google_search_engine_id,
        result_count=result_count,
        c2coff=c2coff,
        results_origin_country=results_origin_country,
        date_range=date_range,
        exactTerms=exactTerms,
        excludeTerms=excludeTerms,
        filter=filter,
        geolocation=geolocation,
        ui_language=ui_language,
        search_result_language=search_result_language,
        safesearch=safesearch,
        site_blacklist=site_blacklist,
        sort_results_by=sort_results_by,
    )
    print(result)
    return result


def parse_google_results(raw_results: dict, output_dict: dict) -> None:
    """
    Parse Google Custom Search API results and update the output dictionary.

    Args:
        raw_results (Dict): Raw Google API response.
        output_dict (Dict): Dictionary to store processed results.
    """
    # Lower verbosity: only log raw payload at debug level
    logging.debug(f"Raw results received: {json.dumps(raw_results, indent=2)}")
    try:
        # Initialize results list if not present
        if "results" not in output_dict:
            output_dict["results"] = []

        # Extract search information
        if "searchInformation" in raw_results:
            search_info = raw_results["searchInformation"]
            output_dict["total_results_found"] = int(search_info.get("totalResults", "0"))
            output_dict["search_time"] = float(search_info.get("searchTime", 0.0))

        # Extract spelling suggestions
        if "spelling" in raw_results:
            output_dict["spell_suggestions"] = raw_results["spelling"].get("correctedQuery")

        # Extract search parameters from queries
        if "queries" in raw_results and "request" in raw_results["queries"]:
            request = raw_results["queries"]["request"][0]
            output_dict.update({
                "search_query": request.get("searchTerms", ""),
                "search_lang": request.get("language", ""),
                "result_count": request.get("count", 0),
                "safesearch": request.get("safe", None),
                "exactTerms": request.get("exactTerms", None),
                "excludeTerms": request.get("excludeTerms", None),
                "filter": request.get("filter", None),
                "geolocation": request.get("gl", None),
                # Google CSE uses "lr" for result language; retain "hl" as fallback.
                "search_result_language": request.get("lr", request.get("hl", None)),
                "sort_results_by": request.get("sort", None)
            })
            google_host = request.get("googleHost") or request.get("googlehost")
            if google_host:
                output_dict["google_domain"] = google_host

        # Process search results
        if "items" in raw_results:
            for item in raw_results["items"]:
                processed_result = {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    # IMPORTANT: 'snippet' is used as 'content'
                    "content": item.get("snippet", ""),
                    "metadata": {
                        "date_published": item.get("pagemap", {})
                                             .get("metatags", [{}])[0]
                                             .get("article:published_time"),
                        "author": item.get("pagemap", {})
                                      .get("metatags", [{}])[0]
                                      .get("article:author"),
                        "source": item.get("displayLink", None),
                        "language": item.get("language", None),
                        "relevance_score": None,  # Google doesn't provide this directly
                        "snippet": item.get("snippet", None),
                        "file_format": item.get("fileFormat", None),
                        "mime_type": item.get("mime", None),
                        "cache_url": item.get("cacheId", None)
                    }
                }

                # Extract additional metadata if available
                if "pagemap" in item:
                    pagemap = item["pagemap"]
                    if "metatags" in pagemap and pagemap["metatags"]:
                        metatags = pagemap["metatags"][0]
                        processed_result["metadata"].update({
                            "description": metatags.get("og:description",
                                                        metatags.get("description")),
                            "keywords": metatags.get("keywords"),
                            "site_name": metatags.get("og:site_name")
                        })

                output_dict["results"].append(processed_result)

        # Add pagination information
        output_dict["pagination"] = {
            "has_next": "nextPage" in raw_results.get("queries", {}),
            "has_previous": "previousPage" in raw_results.get("queries", {}),
            "current_page": raw_results.get("queries", {})
                                   .get("request", [{}])[0]
                                   .get("startIndex", 1)
        }

    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as e:
        logging.error(f"Error processing Google results: {str(e)}")
        output_dict["processing_error"] = f"Error processing Google results: {str(e)}"


def test_parse_google_results():
    parsed_results = {}
    raw_results = {}
    raw_results = test_search_google()
    parse_google_results(raw_results, parsed_results)
    print(f"Parsed search results: {parsed_results}")
    pass



######################### Kagi Search #########################
#
# https://help.kagi.com/kagi/api/search.html
def search_web_kagi(query: str, limit: int = 10) -> dict:
    search_url = "https://kagi.com/api/v0/search"

    # load key from config file
    kagi_api_key = get_loaded_config()['search_engines']['kagi_search_api_key']
    if not kagi_api_key:
        raise ValueError("Please provide a valid Kagi Search API subscription key")

    """
    Queries the Kagi Search API with the given query and limit.
    """
    if kagi_api_key is None:
        raise ValueError("API key is required.")

    headers = {"Authorization": f"Bot {kagi_api_key}"}
    endpoint = search_url
    params = {"q": query, "limit": limit}

    _enforce_provider_outbound_policy(endpoint, source="websearch_kagi")

    from tldw_Server_API.app.core.http_client import fetch_json
    data = fetch_json(method="GET", url=endpoint, headers=headers, params=params, timeout=15.0)
    logging.debug(data)
    return data


def test_search_kagi():
    search_term = "How can I bake a cherry cake"
    result_count = 10
    result = search_web_kagi(search_term, result_count)
    print(result)


def parse_kagi_results(raw_results: dict, output_dict: dict) -> None:
    """
    Parse Kagi search results and update the output dictionary

    Args:
        raw_results (Dict): Raw Kagi API response
        output_dict (Dict): Dictionary to store processed results
    """
    try:
        # Extract metadata
        if "meta" in raw_results:
            meta = raw_results["meta"]
            output_dict["search_time"] = meta.get("ms", 0) / 1000.0  # Convert to seconds
            output_dict["api_balance"] = meta.get("api_balance")
            output_dict["search_id"] = meta.get("id")
            output_dict["node"] = meta.get("node")

        # Process search results
        if "data" in raw_results:
            for item in raw_results["data"]:
                # Skip related searches (type 1)
                if item.get("t") == 1:
                    output_dict["related_searches"] = item.get("list", [])
                    continue

                # Process regular search results (type 0)
                if item.get("t") == 0:
                    processed_result = {
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "content": item.get("snippet", ""),
                        "metadata": {
                            "date_published": item.get("published"),
                            "author": None,  # Kagi doesn't typically provide this
                            "source": None,  # Could be extracted from URL if needed
                            "language": None,  # Kagi doesn't typically provide this
                            "relevance_score": None,
                            "snippet": item.get("snippet"),
                            "thumbnail": item.get("thumbnail", {}).get("url") if "thumbnail" in item else None
                        }
                    }
                    output_dict["results"].append(processed_result)

            # Update total results count
            output_dict["total_results_found"] = len([
                item for item in raw_results["data"]
                if item.get("t") == 0
            ])

    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as e:
        output_dict["processing_error"] = f"Error processing Kagi results: {str(e)}"


def test_parse_kagi_results():
    pass



######################### SearX Search #########################
#
# https://searx.space
# https://searx.github.io/searx/dev/search_api.html
# (legacy session helper removed; using http_client directly)

def search_web_searx(
    search_query,
    language='auto',
    time_range='',
    safesearch=0,
    pageno=1,
    categories='general',
    searx_url=None,
    json_mode: bool = False,
):
    """
    Perform a search using a Searx instance.

    Args:
        search_query (str): The search query.
        language (str): Language for the search results.
        time_range (str): Time range for the search results.
        safesearch (int): Safe search level (0=off, 1=moderate, 2=strict).
        pageno (int): Page number of the results.
        categories (str): Categories to search in (e.g., 'general', 'news').
        searx_url (str): Custom Searx instance URL (optional).

    Returns:
        Dict: Dictionary containing the search results or an error message.
    """
    # Use the provided Searx URL or fall back to the configured one
    if not searx_url:
        searx_url = get_loaded_config()['search_engines']['searx_search_api_url']
    if not searx_url:
        return {"error": "SearX Search is disabled and no content was found. This functionality is disabled because the user has not set it up yet."}

    # Validate and construct URL
    try:
        parsed_url = urlparse(searx_url)
        params = {
            'q': search_query,
            'language': language,
            'time_range': time_range,
            'safesearch': safesearch,
            'pageno': pageno,
            'categories': categories
        }
        if json_mode:
            params['format'] = 'json'
        search_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}?{urlencode(params)}"
        logging.info(f"Search URL: {search_url}")
    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as e:
        return {"error": f"Invalid URL configuration: {str(e)}"}

    # Perform the search request
    try:
        _enforce_provider_outbound_policy(search_url, source="websearch_searx")

        # Mimic browser headers via centralized UA builder
        headers = _websearch_browser_headers(accept_lang="en-US,en;q=0.5", restrict_encodings_for_requests=True)

        # Add a random delay to mimic human behavior
        delay = random.uniform(2, 5)  # Random delay between 2 and 5 seconds
        time.sleep(delay)

        response = fetch(method="GET", url=search_url, headers=headers, timeout=15.0)
        try:
            # Check if the response is JSON
            content_type = response.headers.get('Content-Type', '')
            if 'application/json' in content_type:
                search_data = response.json()
            else:
                # If not JSON, assume it's HTML and parse it
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.text, 'html.parser')
                search_data = parse_html_search_results_generic(soup)
        finally:
            _close_response(response)

        # Process results
        if isinstance(search_data, dict):
            results_iter = search_data.get("results", [])
        elif isinstance(search_data, list):
            results_iter = search_data
        else:
            results_iter = []

        data = []
        for result in results_iter:
            if not isinstance(result, dict):
                continue
            data.append({
                'title': result.get('title'),
                'link': result.get('url') or result.get('link'),
                'snippet': result.get('content') or result.get('snippet'),
                'publishedDate': result.get('publishedDate') or result.get('published'),
            })

        if not data:
            return {"error": "No information was found online for the search query."}

        return {"results": data}

    except (NetworkError, RetryExhaustedError) as e:
        logging.error(f"Error searching for content: {str(e)}")
        return {"error": f"There was an error searching for content. {str(e)}"}

def test_search_searx():
    # Use a different Searx instance to avoid rate limiting
    searx_url = "https://searx.be"  # Example of a different Searx instance
    result = search_web_searx("What goes into making a cherry cake?", searx_url=searx_url)
    print(result)

def parse_searx_results(searx_search_results, web_search_results_dict):
    try:
        if "results" not in web_search_results_dict:
            web_search_results_dict["results"] = []

        items = searx_search_results.get("results", []) if isinstance(searx_search_results, dict) else []
        for item in items:
            title = item.get("title", "")
            url = item.get("link") or item.get("url") or ""
            snippet = item.get("snippet") or item.get("content") or ""
            published = item.get("publishedDate") or item.get("published") or None

            processed = {
                "title": title,
                "url": url,
                "content": snippet,
                "metadata": {
                    "date_published": published,
                    "author": None,
                    "source": extract_domain(url) if url else None,
                    "language": None,
                    "relevance_score": None,
                    "snippet": snippet,
                },
            }
            web_search_results_dict["results"].append(processed)

        web_search_results_dict["total_results_found"] = len(web_search_results_dict["results"])
    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as e:
        web_search_results_dict["processing_error"] = f"Error processing Searx results: {e}"

def test_parse_searx_results():
    pass




######################### Serper.dev Search #########################
#
# https://github.com/YassKhazzan/openperplex_backend_os/blob/main/sources_searcher.py
def _map_serper_date_range(date_range: str | None) -> str | None:
    if not date_range:
        return None
    value = str(date_range).strip().lower()
    if not value:
        return None
    if value.startswith("qdr:"):
        return value
    mapping = {
        "h": "qdr:h",
        "d": "qdr:d",
        "day": "qdr:d",
        "w": "qdr:w",
        "week": "qdr:w",
        "m": "qdr:m",
        "month": "qdr:m",
        "y": "qdr:y",
        "year": "qdr:y",
    }
    return mapping.get(value)


def _as_list_or_empty(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return []


def _build_serper_query(
    *,
    search_query: str,
    site_whitelist: list[str] | str | None,
    site_blacklist: list[str] | str | None,
    exact_terms: str | None,
    exclude_terms: str | None,
) -> str:
    query_parts = [str(search_query or "").strip()]

    if exact_terms and str(exact_terms).strip():
        query_parts.append(f"\"{str(exact_terms).strip()}\"")

    for domain in _as_list_or_empty(site_whitelist):
        query_parts.append(f"site:{domain}")
    for domain in _as_list_or_empty(site_blacklist):
        query_parts.append(f"-site:{domain}")

    if exclude_terms and str(exclude_terms).strip():
        # Apply explicit negative terms as expected by web search syntax.
        for term in str(exclude_terms).split():
            cleaned = term.strip()
            if cleaned:
                query_parts.append(f"-{cleaned}")

    return " ".join(part for part in query_parts if part).strip()


def search_web_serper(
    search_query: str,
    result_count: int = 10,
    content_country: str | None = None,
    search_lang: str | None = None,
    output_lang: str | None = None,
    date_range: str | None = None,
    safesearch: str | None = None,
    site_whitelist: list[str] | str | None = None,
    site_blacklist: list[str] | str | None = None,
    exactTerms: str | None = None,
    excludeTerms: str | None = None,
    serper_api_key: str | None = None,
    serper_api_url: str | None = None,
):
    """Query Serper.dev and return raw JSON results."""
    cfg = get_loaded_config().get("search_engines", {})
    if not serper_api_url:
        serper_api_url = (
            cfg.get("serper_search_api_url")
            or os.getenv("SERPER_API_URL")
            or os.getenv("SEARCH_ENGINE_API_URL_SERPER")
            or "https://google.serper.dev/search"
        )
    if not serper_api_key:
        serper_api_key = (
            cfg.get("serper_search_api_key")
            or os.getenv("SERPER_API_KEY")
            or os.getenv("SEARCH_ENGINE_API_KEY_SERPER")
        )
    if not serper_api_key:
        raise ValueError("Please provide a valid Serper API key")

    query = _build_serper_query(
        search_query=search_query,
        site_whitelist=site_whitelist,
        site_blacklist=site_blacklist,
        exact_terms=exactTerms,
        exclude_terms=excludeTerms,
    )
    if not query:
        raise ValueError("search_query is required")

    payload: dict[str, Any] = {
        "q": query,
        "num": int(result_count),
        "gl": (content_country or "us").lower(),
        "hl": (output_lang or search_lang or "en").lower(),
        "safe": (safesearch or "off").lower(),
    }
    tbs = _map_serper_date_range(date_range)
    if tbs:
        payload["tbs"] = tbs

    _enforce_provider_outbound_policy(serper_api_url, source="websearch_serper")

    headers = {
        "Content-Type": "application/json",
        "X-API-KEY": serper_api_key,
    }

    from tldw_Server_API.app.core.http_client import fetch_json
    return fetch_json(method="POST", url=serper_api_url, headers=headers, json=payload, timeout=20.0)


def test_search_serper():
    pass

def parse_serper_results(serper_search_results, web_search_results_dict):
    try:
        if "results" not in web_search_results_dict:
            web_search_results_dict["results"] = []

        if not isinstance(serper_search_results, dict):
            web_search_results_dict["processing_error"] = "Error processing Serper results: invalid payload"
            return

        def _append_item(item: dict[str, Any]) -> None:
            title = item.get("title", "")
            url = item.get("link") or item.get("url") or ""
            snippet = item.get("snippet") or item.get("description") or ""
            published = item.get("date") or item.get("publishedDate") or item.get("published") or None
            processed = {
                "title": title,
                "url": url,
                "content": snippet,
                "metadata": {
                    "date_published": published,
                    "author": item.get("author"),
                    "source": extract_domain(url) if url else item.get("source"),
                    "language": item.get("language"),
                    "relevance_score": item.get("position") or item.get("rank") or item.get("score"),
                    "snippet": snippet,
                },
            }
            web_search_results_dict["results"].append(processed)

        for item in serper_search_results.get("organic", []):
            if isinstance(item, dict):
                _append_item(item)

        for item in serper_search_results.get("news", []):
            if isinstance(item, dict):
                _append_item(item)

        answer_box = serper_search_results.get("answerBox")
        if isinstance(answer_box, dict):
            _append_item(answer_box)

        knowledge_graph = serper_search_results.get("knowledgeGraph")
        if isinstance(knowledge_graph, dict):
            _append_item(knowledge_graph)

        web_search_results_dict["total_results_found"] = len(web_search_results_dict["results"])
    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as e:
        web_search_results_dict["processing_error"] = f"Error processing Serper results: {e}"




######################### Tavily Search #########################
#
# https://github.com/YassKhazzan/openperplex_backend_os/blob/main/sources_searcher.py
def search_web_tavily(search_query, result_count=10, site_whitelist=None, site_blacklist=None):
    # Check if API URL is configured
    tavily_api_url = "https://api.tavily.com/search"

    tavily_api_key = get_loaded_config()['search_engines']['tavily_search_api_key']

    # Prepare the request payload
    payload = {
        "api_key": tavily_api_key,
        "query": search_query,
        "max_results": result_count
    }

    # Add optional parameters if provided
    if site_whitelist:
        payload["include_domains"] = site_whitelist
    if site_blacklist:
        payload["exclude_domains"] = site_blacklist

    _enforce_provider_outbound_policy(tavily_api_url, source="websearch_tavily")

    # Perform the search request
    try:
        headers = {"Content-Type": "application/json"}
        ua_only = build_browser_headers(pick_ua_profile("fixed"))
        if "User-Agent" in ua_only:
            headers["User-Agent"] = ua_only["User-Agent"]

        from tldw_Server_API.app.core.http_client import fetch_json
        data = fetch_json(method="POST", url=tavily_api_url, headers=headers, json=payload, timeout=15.0)
        return data
    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as e:
        return {"error": f"There was an error searching for content. {str(e)}"}


def test_search_tavily():
    result = search_web_tavily("How can I bake a cherry cake?")
    print(result)


def parse_tavily_results(tavily_search_results, web_search_results_dict):
    try:
        if "results" not in web_search_results_dict:
            web_search_results_dict["results"] = []

        # Tavily returns dict with key 'results': list of dicts
        items = tavily_search_results.get("results", []) if isinstance(tavily_search_results, dict) else []
        for item in items:
            title = item.get("title", "")
            url = item.get("url", "")
            content = item.get("content") or item.get("snippet") or ""
            published = item.get("publishedDate") or item.get("published_date") or None

            processed = {
                "title": title,
                "url": url,
                "content": content,
                "metadata": {
                    "date_published": published,
                    "author": item.get("author"),
                    "source": extract_domain(url) if url else None,
                    "language": item.get("language"),
                    "relevance_score": item.get("score"),
                    "snippet": content,
                },
            }
            web_search_results_dict["results"].append(processed)

        web_search_results_dict["total_results_found"] = len(web_search_results_dict["results"])
    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as e:
        web_search_results_dict["processing_error"] = f"Error processing Tavily results: {e}"


def test_parse_tavily_results():
    pass




######################### Exa Search #########################
#
# https://exa.ai/docs/reference/search
def search_web_exa(
    search_query: str,
    result_count: int = 10,
    content_country: str | None = None,
    site_blacklist: list[str] | None = None,
    site_whitelist: list[str] | None = None,
    exa_api_key: str | None = None,
    exa_api_url: str | None = None,
):
    exa_cfg = get_loaded_config().get("search_engines", {})
    if not exa_api_url:
        exa_api_url = exa_cfg.get("exa_search_api_url") or "https://api.exa.ai/search"
    if not exa_api_key:
        exa_api_key = exa_cfg.get("exa_search_api_key")
    if not exa_api_key:
        raise ValueError("Please provide a valid Exa API key")

    payload: dict[str, Any] = {
        "query": search_query,
        "numResults": result_count,
        "text": True,
    }
    if content_country:
        payload["userLocation"] = content_country
    if site_whitelist:
        payload["includeDomains"] = site_whitelist
    if site_blacklist:
        payload["excludeDomains"] = site_blacklist

    _enforce_provider_outbound_policy(exa_api_url, source="websearch_exa")

    headers = {"Content-Type": "application/json", "x-api-key": exa_api_key}
    from tldw_Server_API.app.core.http_client import fetch_json
    return fetch_json(method="POST", url=exa_api_url, headers=headers, json=payload, timeout=20.0)


def parse_exa_results(exa_search_results, web_search_results_dict):
    try:
        if "results" not in web_search_results_dict:
            web_search_results_dict["results"] = []

        items = exa_search_results.get("results", []) if isinstance(exa_search_results, dict) else []
        for item in items:
            title = item.get("title", "")
            url = item.get("url", "")
            highlights = item.get("highlights") if isinstance(item.get("highlights"), list) else []
            snippet = item.get("summary") or (highlights[0] if highlights else "") or item.get("text") or ""
            snippet = _truncate_text(snippet)

            processed = {
                "title": title,
                "url": url,
                "content": snippet,
                "metadata": {
                    "date_published": item.get("publishedDate"),
                    "author": item.get("author"),
                    "source": extract_domain(url) if url else None,
                    "language": item.get("language"),
                    "relevance_score": item.get("score"),
                    "snippet": snippet,
                },
            }
            web_search_results_dict["results"].append(processed)

        web_search_results_dict["total_results_found"] = len(web_search_results_dict["results"])
    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as e:
        web_search_results_dict["processing_error"] = f"Error processing Exa results: {e}"


def test_parse_exa_results():
    pass




######################### Firecrawl Search #########################
#
# https://docs.firecrawl.dev/api-reference/search
def search_web_firecrawl(
    search_query: str,
    result_count: int = 10,
    content_country: str | None = None,
    date_range: str | None = None,
    firecrawl_api_key: str | None = None,
    firecrawl_api_url: str | None = None,
):
    fc_cfg = get_loaded_config().get("search_engines", {})
    if not firecrawl_api_url:
        firecrawl_api_url = fc_cfg.get("firecrawl_search_api_url") or "https://api.firecrawl.dev/v2/search"
    if not firecrawl_api_key:
        firecrawl_api_key = fc_cfg.get("firecrawl_api_key")
    if not firecrawl_api_key:
        raise ValueError("Please provide a valid Firecrawl API key")

    payload: dict[str, Any] = {"query": search_query, "limit": result_count}
    if content_country:
        payload["country"] = content_country
    if date_range:
        payload["tbs"] = date_range

    _enforce_provider_outbound_policy(
        firecrawl_api_url,
        source="websearch_firecrawl",
    )

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {firecrawl_api_key}"}
    from tldw_Server_API.app.core.http_client import fetch_json
    return fetch_json(method="POST", url=firecrawl_api_url, headers=headers, json=payload, timeout=20.0)


def parse_firecrawl_results(firecrawl_search_results, web_search_results_dict):
    try:
        if "results" not in web_search_results_dict:
            web_search_results_dict["results"] = []

        data = firecrawl_search_results.get("data", []) if isinstance(firecrawl_search_results, dict) else []

        items: list[dict[str, Any]] = []
        if isinstance(data, dict):
            for key in ("web", "news", "images"):
                bucket = data.get(key)
                if isinstance(bucket, list):
                    items.extend(bucket)
        elif isinstance(data, list):
            items = data

        for item in items:
            title = item.get("title") or item.get("name") or ""
            url = item.get("url") or item.get("link") or ""
            snippet = item.get("markdown") or item.get("description") or item.get("content") or ""
            snippet = _truncate_text(snippet)
            published = item.get("publishedDate") or item.get("published") or None

            processed = {
                "title": title,
                "url": url,
                "content": snippet,
                "metadata": {
                    "date_published": published,
                    "author": item.get("author"),
                    "source": extract_domain(url) if url else None,
                    "language": item.get("language"),
                    "relevance_score": item.get("score"),
                    "snippet": snippet,
                },
            }
            web_search_results_dict["results"].append(processed)

        web_search_results_dict["total_results_found"] = len(web_search_results_dict["results"])
    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as e:
        web_search_results_dict["processing_error"] = f"Error processing Firecrawl results: {e}"


def test_parse_firecrawl_results():
    pass


######################### 4chan Search #########################
#
# https://github.com/4chan/4chan-API
def _normalize_4chan_boards(value: Any) -> list[str]:
    tokens: list[str] = []
    if isinstance(value, str):
        tokens = [part.strip().lower() for part in re.split(r"[,\s]+", value) if part.strip()]
    elif isinstance(value, list):
        tokens = [str(part).strip().lower() for part in value if str(part).strip()]
    else:
        return []

    board_pattern = re.compile(r"^[a-z0-9]{1,12}$")
    out: list[str] = []
    for token in tokens:
        if board_pattern.match(token) and token not in out:
            out.append(token)
    return out


def _clean_4chan_text(value: Any) -> str:
    if value is None:
        return ""
    text = unescape(str(value))
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _score_4chan_match(query_terms: list[str], haystack: str, full_query: str) -> float:
    text = haystack.lower()
    score = 0.0
    if not text:
        return score

    if full_query and full_query in text:
        score += 3.0

    for term in query_terms:
        if term in text:
            score += 1.0
            repeats = max(0, text.count(term) - 1)
            score += min(1.0, repeats * 0.2)

    return score


def _append_4chan_candidate(
    *,
    candidates: list[dict[str, Any]],
    board: str,
    thread: dict[str, Any],
    query_terms: list[str],
    query_text: str,
    archived: bool,
) -> None:
    thread_no = thread.get("no")
    if thread_no is None:
        return

    subject = _clean_4chan_text(thread.get("sub"))
    comment = _clean_4chan_text(thread.get("com"))
    semantic = _clean_4chan_text(thread.get("semantic_url"))
    haystack = " ".join(part for part in [subject, semantic, comment] if part)
    score = _score_4chan_match(query_terms, haystack, query_text)
    if query_terms and score <= 0:
        return

    published_epoch = thread.get("time")
    published_date = None
    if isinstance(published_epoch, (int, float)):
        try:
            published_date = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(published_epoch)))
        except _WEBSEARCH_NONCRITICAL_EXCEPTIONS:
            published_date = None

    thread_url = f"https://boards.4chan.org/{board}/thread/{thread_no}"
    snippet = _truncate_text(comment or semantic or subject)
    title = subject or f"/{board}/ Thread {thread_no}"

    candidates.append(
        {
            "title": title,
            "url": thread_url,
            "content": snippet,
            "publishedDate": published_date,
            "author": _clean_4chan_text(thread.get("name")) or None,
            "source": "4chan",
            "board": board,
            "thread_no": thread_no,
            "replies": thread.get("replies"),
            "images": thread.get("images"),
            "archived": archived,
            "score": round(score, 4),
            "time_epoch": int(published_epoch) if isinstance(published_epoch, (int, float)) else 0,
        }
    )


def _normalize_4chan_thread_no(value: Any) -> str:
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, float)):
        try:
            return str(int(value))
        except _WEBSEARCH_NONCRITICAL_EXCEPTIONS:
            return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.isdigit():
        try:
            return str(int(text))
        except _WEBSEARCH_NONCRITICAL_EXCEPTIONS:
            return ""
    return ""


def _4chan_candidate_key(candidate: dict[str, Any]) -> tuple[str, str] | None:
    board = str(candidate.get("board") or "").strip().lower()
    if not board:
        return None
    thread_no = _normalize_4chan_thread_no(candidate.get("thread_no"))
    if not thread_no:
        return None
    return board, thread_no


def _is_4chan_placeholder_title(title: str) -> bool:
    return bool(re.match(r"^/[a-z0-9]{1,12}/\s+thread\s+\d+$", title.strip().lower()))


def _select_4chan_text(preferred: Any, secondary: Any) -> str:
    preferred_text = str(preferred or "").strip()
    secondary_text = str(secondary or "").strip()
    if not preferred_text:
        return secondary_text
    if not secondary_text:
        return preferred_text
    if len(secondary_text) > len(preferred_text):
        return secondary_text
    return preferred_text


def _select_4chan_title(preferred: Any, secondary: Any) -> str:
    preferred_title = str(preferred or "").strip()
    secondary_title = str(secondary or "").strip()
    if not preferred_title:
        return secondary_title
    if not secondary_title:
        return preferred_title
    if _is_4chan_placeholder_title(preferred_title) and not _is_4chan_placeholder_title(secondary_title):
        return secondary_title
    if len(secondary_title) > len(preferred_title):
        return secondary_title
    return preferred_title


def _to_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except _WEBSEARCH_NONCRITICAL_EXCEPTIONS:
            return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS:
        return None


def _merge_4chan_numeric(preferred: Any, secondary: Any) -> Any:
    preferred_num = _to_int_or_none(preferred)
    secondary_num = _to_int_or_none(secondary)
    if preferred_num is None:
        return secondary
    if secondary_num is None:
        return preferred
    return max(preferred_num, secondary_num)


def _merge_4chan_candidates(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    existing_archived = _coerce_bool(existing.get("archived"), default=False)
    incoming_archived = _coerce_bool(incoming.get("archived"), default=False)

    if existing_archived and not incoming_archived:
        preferred = incoming
        secondary = existing
    else:
        preferred = existing
        secondary = incoming

    merged = dict(preferred)
    merged["title"] = _select_4chan_title(preferred.get("title"), secondary.get("title"))
    merged["content"] = _select_4chan_text(preferred.get("content"), secondary.get("content"))

    preferred_score = float(preferred.get("score") or 0.0)
    secondary_score = float(secondary.get("score") or 0.0)
    merged["score"] = round(max(preferred_score, secondary_score), 4)

    preferred_time = int(preferred.get("time_epoch") or 0)
    secondary_time = int(secondary.get("time_epoch") or 0)
    if secondary_time > preferred_time:
        merged["time_epoch"] = secondary_time
        merged["publishedDate"] = secondary.get("publishedDate") or preferred.get("publishedDate")
    else:
        merged["time_epoch"] = preferred_time
        merged["publishedDate"] = preferred.get("publishedDate") or secondary.get("publishedDate")

    merged["author"] = preferred.get("author") or secondary.get("author")
    merged["replies"] = _merge_4chan_numeric(preferred.get("replies"), secondary.get("replies"))
    merged["images"] = _merge_4chan_numeric(preferred.get("images"), secondary.get("images"))

    board = str(preferred.get("board") or secondary.get("board") or "").strip().lower()
    if board:
        merged["board"] = board

    thread_no = _normalize_4chan_thread_no(preferred.get("thread_no") or secondary.get("thread_no"))
    if thread_no:
        try:
            merged["thread_no"] = int(thread_no)
        except _WEBSEARCH_NONCRITICAL_EXCEPTIONS:
            merged["thread_no"] = thread_no

    if board and thread_no:
        merged["url"] = f"https://boards.4chan.org/{board}/thread/{thread_no}"

    merged["archived"] = existing_archived and incoming_archived
    return merged


def _dedupe_4chan_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    key_to_index: dict[tuple[str, str], int] = {}

    for candidate in candidates:
        key = _4chan_candidate_key(candidate)
        if key is None:
            deduped.append(candidate)
            continue

        existing_index = key_to_index.get(key)
        if existing_index is None:
            key_to_index[key] = len(deduped)
            deduped.append(candidate)
            continue

        deduped[existing_index] = _merge_4chan_candidates(deduped[existing_index], candidate)

    return deduped


def search_web_4chan(
    search_query: str,
    result_count: int = 10,
    search_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    query_text = str(search_query or "").strip().lower()
    if not query_text:
        raise ValueError("search_query is required")

    cfg = get_loaded_config().get("search_engines", {})
    boards_raw = (
        (search_params or {}).get("boards")
        or cfg.get("4chan_boards")
        or cfg.get("fourchan_boards")
        or os.getenv("FOURCHAN_BOARDS")
        or os.getenv("FOURCHAN_DEFAULT_BOARDS")
        or ["g", "tv", "pol"]
    )
    boards = _normalize_4chan_boards(boards_raw)
    if not boards:
        boards = ["g", "tv", "pol"]

    max_threads_raw = (
        (search_params or {}).get("max_threads_per_board")
        or cfg.get("4chan_max_threads_per_board")
        or cfg.get("fourchan_max_threads_per_board")
        or 250
    )
    try:
        max_threads_per_board = int(max_threads_raw)
    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS:
        max_threads_per_board = 250
    max_threads_per_board = max(1, min(max_threads_per_board, 1000))

    try:
        limit = int(result_count)
    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS:
        limit = 10
    limit = max(1, min(limit, 50))

    include_archived_raw = (
        (search_params or {}).get("include_archived")
        if isinstance(search_params, dict)
        else None
    )
    if include_archived_raw is None:
        include_archived_raw = (
            cfg.get("4chan_include_archived")
            or cfg.get("fourchan_include_archived")
            or os.getenv("FOURCHAN_INCLUDE_ARCHIVED")
        )
    include_archived = _coerce_bool(include_archived_raw, default=False)

    max_archived_raw = (
        (search_params or {}).get("max_archived_threads_per_board")
        if isinstance(search_params, dict)
        else None
    )
    if max_archived_raw is None:
        max_archived_raw = (
            cfg.get("4chan_max_archived_threads_per_board")
            or cfg.get("fourchan_max_archived_threads_per_board")
            or min(max_threads_per_board, 50)
        )
    try:
        max_archived_threads_per_board = int(max_archived_raw)
    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS:
        max_archived_threads_per_board = min(max_threads_per_board, 50)
    max_archived_threads_per_board = max(1, min(max_archived_threads_per_board, 500))

    query_terms = [t for t in re.findall(r"[a-z0-9]{2,}", query_text) if t]
    if not query_terms and query_text:
        query_terms = [query_text]

    headers = _websearch_browser_headers(accept_lang="en-US,en;q=0.5")
    from tldw_Server_API.app.core.http_client import fetch_json

    candidates: list[dict[str, Any]] = []
    board_warnings: list[dict[str, str]] = []
    successful_boards: set[str] = set()

    def _record_board_warning(board_name: str, phase: str, exc: Any) -> None:
        message = _truncate_text(str(exc), max_len=240)
        board_warnings.append(
            {
                "board": board_name,
                "phase": phase,
                "message": message,
            }
        )
        logging.warning(f"4chan board '{board_name}' {phase} failed: {message}")

    for board in boards:
        catalog_url = f"https://a.4cdn.org/{board}/catalog.json"

        try:
            _enforce_provider_outbound_policy(
                catalog_url,
                source="websearch_4chan_catalog",
                stage="board_catalog_request",
            )
            catalog_payload = fetch_json(method="GET", url=catalog_url, headers=headers, timeout=15.0)
            pages = catalog_payload if isinstance(catalog_payload, list) else []
            successful_boards.add(board)
            scanned_threads = 0

            for page in pages:
                if scanned_threads >= max_threads_per_board:
                    break
                if not isinstance(page, dict):
                    continue
                threads = page.get("threads", [])
                if not isinstance(threads, list):
                    continue

                for thread in threads:
                    if scanned_threads >= max_threads_per_board:
                        break
                    scanned_threads += 1

                    if not isinstance(thread, dict):
                        continue

                    _append_4chan_candidate(
                        candidates=candidates,
                        board=board,
                        thread=thread,
                        query_terms=query_terms,
                        query_text=query_text,
                        archived=False,
                    )
        except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as exc:
            _record_board_warning(board, "catalog", exc)

        if include_archived:
            archive_url = f"https://a.4cdn.org/{board}/archive.json"
            try:
                _enforce_provider_outbound_policy(
                    archive_url,
                    source="websearch_4chan_archive",
                    stage="board_archive_request",
                )
                archive_payload = fetch_json(method="GET", url=archive_url, headers=headers, timeout=15.0)
                archive_ids = archive_payload if isinstance(archive_payload, list) else []
                successful_boards.add(board)
                archived_thread_ids = list(reversed(archive_ids))[:max_archived_threads_per_board]

                for archived_thread_id in archived_thread_ids:
                    thread_id = str(archived_thread_id).strip()
                    if not thread_id.isdigit():
                        continue

                    thread_api_url = f"https://a.4cdn.org/{board}/thread/{thread_id}.json"
                    try:
                        decision = _provider_outbound_policy_decision(
                            thread_api_url,
                            source="websearch_4chan_thread",
                            stage="thread_request",
                        )
                        if not decision.allowed:
                            continue
                    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS:
                        continue

                    try:
                        thread_payload = fetch_json(method="GET", url=thread_api_url, headers=headers, timeout=15.0)
                    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS:
                        continue

                    posts = thread_payload.get("posts", []) if isinstance(thread_payload, dict) else []
                    if not isinstance(posts, list) or not posts:
                        continue

                    op = next((post for post in posts if isinstance(post, dict)), None)
                    if not op:
                        continue
                    op_data = dict(op)
                    op_data.setdefault("no", int(thread_id))
                    op_data.setdefault(
                        "replies",
                        max(0, len([post for post in posts if isinstance(post, dict)]) - 1),
                    )
                    op_data.setdefault(
                        "images",
                        sum(1 for post in posts if isinstance(post, dict) and post.get("tim") is not None),
                    )
                    op_data.setdefault("archived", 1)

                    _append_4chan_candidate(
                        candidates=candidates,
                        board=board,
                        thread=op_data,
                        query_terms=query_terms,
                        query_text=query_text,
                        archived=True,
                    )
            except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as exc:
                _record_board_warning(board, "archive", exc)

    candidates = _dedupe_4chan_candidates(candidates)
    candidates.sort(
        key=lambda item: (
            float(item.get("score") or 0.0),
            int(item.get("time_epoch") or 0),
        ),
        reverse=True,
    )

    result: dict[str, Any] = {
        "results": candidates[:limit],
        "total_results_found": len(candidates),
        "boards": boards,
        "include_archived": include_archived,
        "query": search_query,
    }
    if board_warnings:
        result["warnings"] = board_warnings
    if board_warnings and not successful_boards:
        result["error"] = "4chan search failed for all requested boards."
    return result


def parse_4chan_results(fourchan_search_results, web_search_results_dict):
    try:
        if "results" not in web_search_results_dict:
            web_search_results_dict["results"] = []

        items = fourchan_search_results.get("results", []) if isinstance(fourchan_search_results, dict) else []
        for item in items:
            if not isinstance(item, dict):
                continue

            title = item.get("title", "")
            url = item.get("url", "")
            content = item.get("content") or item.get("snippet") or ""
            published = item.get("publishedDate") or item.get("published") or item.get("date")

            processed = {
                "title": title,
                "url": url,
                "content": content,
                "metadata": {
                    "date_published": published,
                    "author": item.get("author"),
                    "source": item.get("source") or (extract_domain(url) if url else "4chan"),
                    "language": item.get("language"),
                    "relevance_score": item.get("score"),
                    "snippet": content,
                    "board": item.get("board"),
                    "thread_no": item.get("thread_no"),
                    "replies": item.get("replies"),
                    "images": item.get("images"),
                    "archived": bool(item.get("archived", False)),
                },
            }
            web_search_results_dict["results"].append(processed)

        warnings = (
            fourchan_search_results.get("warnings", [])
            if isinstance(fourchan_search_results, dict)
            else []
        )
        if isinstance(warnings, list) and warnings:
            web_search_results_dict["warnings"] = warnings

        web_search_results_dict["total_results_found"] = len(web_search_results_dict["results"])
    except _WEBSEARCH_NONCRITICAL_EXCEPTIONS as e:
        web_search_results_dict["processing_error"] = f"Error processing 4chan results: {e}"




######################### Yandex Search #########################
#
# https://yandex.cloud/en/docs/search-api/operations/web-search
# https://yandex.cloud/en/docs/search-api/quickstart/
# https://yandex.cloud/en/docs/search-api/concepts/response
# https://github.com/yandex-cloud/cloudapi/blob/master/yandex/cloud/searchapi/v2/search_query.proto
def search_web_yandex():
    """Yandex provider stub with egress policy check."""
    _enforce_provider_outbound_policy(
        "https://yandex.cloud",
        source="websearch_yandex",
    )
    return {"error": "Yandex provider not implemented"}


def test_search_yandex():
    pass

def parse_yandex_results(yandex_search_results, web_search_results_dict):
    pass


#
# End of WebSearch_APIs.py
#######################################################################################################################
