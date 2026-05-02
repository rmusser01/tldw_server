# Document Insights Endpoint
# Generate AI-powered insights from document content using LLM
#
from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Path, status
from loguru import logger
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, rbac_rate_limit, User

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import DEFAULT_LLM_PROVIDER
from tldw_Server_API.app.api.v1.schemas.document_insights import (
    DocumentInsightsResponse,
    GenerateInsightsRequest,
    InsightCategory,
    InsightItem,
)
from tldw_Server_API.app.api.v1.utils.cache import (
    cache_response,
    get_cached_response,
)
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
from tldw_Server_API.app.core.Chat.chat_helpers import extract_response_content
from tldw_Server_API.app.core.Chat.chat_service import resolve_provider_api_key
from tldw_Server_API.app.core.config import load_and_log_configs
from tldw_Server_API.app.core.DB_Management.media_db.api import (
    get_latest_transcription,
)
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import get_registry
from tldw_Server_API.app.core.LLM_Calls.provider_metadata import provider_requires_api_key
from tldw_Server_API.app.core.LLM_Calls.structured_output import (
    StructuredOutputOptions,
    parse_structured_output,
)

router = APIRouter(tags=["Document Workspace"])

# Default maximum content length for analysis
DEFAULT_MAX_CONTENT_LENGTH = 5000


def _get_db_scope(db: Any) -> str:
    """Return a stable scope identifier for the active MediaDatabase."""
    return getattr(db, "db_path_str", None) or str(getattr(db, "db_path", ""))


def _build_insights_cache_key(
    media_id: int,
    request: GenerateInsightsRequest,
    *,
    user_id: str,
    db_scope: str,
    max_content_length: int,
) -> str:
    """Build cache key for insights based on media, user scope, and request params."""
    categories_str = ",".join(sorted(c.value for c in request.categories)) if request.categories else "all"
    model_str = request.model or "default"
    scope_str = f"user:{user_id}:db:{db_scope}"
    return (
        f"cache:/api/v1/media/{media_id}/insights:{scope_str}:"
        f"{categories_str}:{model_str}:maxlen:{max_content_length}"
    )


# System prompt for generating insights
INSIGHTS_SYSTEM_PROMPT = """You are a research analyst. Analyze the following document and extract structured insights.
For each category, provide a concise title and detailed content.

Categories to analyze:
- research_gap: What problem or gap does this work address?
- research_question: What is the main research question?
- motivation: Why is this research important?
- methods: What methods or approaches were used?
- key_findings: What are the main results or findings?
- limitations: What are the limitations or caveats?
- future_work: What future work is suggested?
- summary: A brief 2-3 sentence summary

Return JSON with this structure:
{"insights": [{"category": "...", "title": "...", "content": "..."}]}

Important:
- Only include categories that are relevant to the document
- Keep titles short (5-10 words)
- Keep content concise but informative (1-3 sentences)
- If the document is not a research paper, adapt the categories as appropriate
- For non-academic documents, focus on: summary, key_findings, and any applicable categories
- Return ONLY valid JSON, no other text
"""


def _get_adapter(provider: str):
    """Get the LLM adapter for the specified provider."""
    registry = get_registry()
    adapter = registry.get_adapter(provider)
    if adapter is None:
        raise ChatConfigurationError(provider=provider, message="LLM adapter unavailable.")
    return adapter


def _resolve_model(provider: str, model: str | None, app_config: dict[str, Any]) -> str | None:
    """Resolve the model to use for the provider."""
    if model:
        return model
    key = f"{provider.replace('-', '_').replace('.', '_')}_api"
    return (app_config.get(key) or {}).get("model")


def _normalize_insights(raw_insights: list[Any]) -> list[InsightItem]:
    """Normalize raw LLM insights into InsightItem list."""
    normalized: list[InsightItem] = []
    valid_categories = {c.value for c in InsightCategory}

    for item in raw_insights:
        if not isinstance(item, dict):
            continue

        category_raw = str(item.get("category") or "").strip().lower()
        if category_raw not in valid_categories:
            continue

        title = str(item.get("title") or "").strip()
        content = str(item.get("content") or "").strip()

        if not title or not content:
            continue

        confidence = item.get("confidence")
        if confidence is not None:
            try:
                confidence = max(0.0, min(1.0, float(confidence)))
            except (TypeError, ValueError):
                confidence = None

        normalized.append(
            InsightItem(
                category=InsightCategory(category_raw),
                title=title,
                content=content,
                confidence=confidence,
            )
        )

    return normalized


@router.post(
    "/{media_id:int}/insights",
    status_code=status.HTTP_200_OK,
    summary="Generate Document Insights",
    response_model=DocumentInsightsResponse,
    dependencies=[Depends(rbac_rate_limit("media.insights"))],
    responses={
        200: {"description": "Insights generated successfully"},
        404: {"description": "Media item not found"},
        422: {"description": "No content available for analysis"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Server error (LLM call failed or response parsing error)"},
        503: {"description": "LLM service unavailable"},
    },
)
async def generate_document_insights(
    media_id: int = Path(..., description="The ID of the media item"),
    request: GenerateInsightsRequest | None = None,
    db: Any = Depends(get_media_db_for_user),
    current_user: User = Depends(get_request_user),
) -> DocumentInsightsResponse:
    """
    Generate AI-powered insights from a document using an LLM.

    This endpoint analyzes the document content and extracts structured insights
    including research gaps, questions, methods, findings, and more.

    ## Request Body (optional)

    - **categories**: Specific insight categories to generate (default: all)
    - **model**: LLM model to use (default: provider's default)
    - **max_content_length**: Maximum characters of document to analyze (default: 5000)
    - **force**: Bypass cache and force a fresh LLM call (default: false)

    ## Insight Categories

    - `research_gap`: What problem or gap the document addresses
    - `research_question`: The main research question
    - `motivation`: Why the research/content is important
    - `methods`: Methods or approaches used
    - `key_findings`: Main results or findings
    - `limitations`: Limitations or caveats
    - `future_work`: Suggested future work
    - `summary`: Brief 2-3 sentence summary

    ## Response

    Returns a list of insights with category, title, and content for each.
    """
    user_id = str(getattr(current_user, "id", "anonymous"))
    logger.debug(
        "Generating document insights for media_id={}, user_id={}",
        media_id,
        user_id,
    )

    # Use default request if none provided
    if request is None:
        request = GenerateInsightsRequest()

    max_content_length = request.max_content_length or DEFAULT_MAX_CONTENT_LENGTH
    db_scope = _get_db_scope(db)

    # Check cache first unless forced (cached responses don't count against rate limit)
    cache_key = _build_insights_cache_key(
        media_id,
        request,
        user_id=user_id,
        db_scope=db_scope,
        max_content_length=max_content_length,
    )
    if not request.force:
        cached = get_cached_response(cache_key)
        if cached is not None:
            _etag, payload = cached
            logger.debug("Returning cached insights for media_id={}", media_id)
            return DocumentInsightsResponse(
                media_id=payload["media_id"],
                insights=[InsightItem(**i) for i in payload["insights"]],
                model_used=payload["model_used"],
                cached=True,
            )

    # 1. Verify media item exists
    try:
        media = db.get_media_by_id(media_id, include_deleted=False, include_trash=False)
    except Exception as e:
        logger.error("Database error fetching media item")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database error while fetching media item",
        ) from e

    if not media:
        logger.warning(
            "Media not found for insights generation: {} (user: {})",
            media_id,
            getattr(current_user, "id", "?"),
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Media not found or is inactive/trashed",
        )

    # 2. Get document content
    content = str(media.get("content") or "").strip()
    if not content:
        content = (get_latest_transcription(db, media_id) or "").strip()

    if not content:
        logger.warning("No content available for media_id={}", media_id)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No content available for analysis. The document may not have been processed yet.",
        )

    # Truncate content if needed
    if len(content) > max_content_length:
        content = content[:max_content_length] + "\n\n[Content truncated for analysis...]"

    # 3. Build prompt
    category_instruction = ""
    if request.categories:
        category_list = ", ".join(c.value for c in request.categories)
        category_instruction = f"\n\nOnly generate insights for these categories: {category_list}"

    user_prompt = f"""Analyze this document and extract insights:

---
{content}
---
{category_instruction}"""

    # 4. Resolve provider and credentials
    provider = (DEFAULT_LLM_PROVIDER or "openai").strip().lower()
    api_key, _debug = resolve_provider_api_key(provider, prefer_module_keys_in_tests=True)
    if provider_requires_api_key(provider) and not api_key:
        logger.error("No API key available for configured provider")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM provider configuration error",
        )

    # 5. Call LLM
    messages_payload = [
        {"role": "system", "content": INSIGHTS_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    response_format = {"type": "json_object"}

    def _call_llm():
        adapter = _get_adapter(provider)
        app_config = load_and_log_configs() or {}
        model_to_use = _resolve_model(provider, request.model, app_config)
        if model_to_use is None:
            raise ChatConfigurationError(provider=provider, message="Model is required for provider.")
        return (
            adapter.chat(
                {
                    "messages": messages_payload,
                    "api_key": api_key,
                    "model": model_to_use,
                    "temperature": 0.3,
                    "max_tokens": 2000,
                    "response_format": response_format,
                    "app_config": app_config,
                }
            ),
            model_to_use,
        )

    try:
        start = time.time()
        loop = asyncio.get_running_loop()
        raw_response, model_used = await loop.run_in_executor(None, _call_llm)
        elapsed_ms = (time.time() - start) * 1000.0
        logger.info(
            "Document insights LLM call completed in {:.1f}ms (media_id={})",
            elapsed_ms,
            media_id,
        )
    except ChatConfigurationError as e:
        logger.error("LLM configuration error for insights")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM provider configuration error",
        ) from e
    except Exception as e:
        logger.error("LLM call failed for document insights")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate insights. LLM service error.",
        ) from e

    # 6. Parse response
    try:
        content_text = extract_response_content(raw_response)
        payload = parse_structured_output(
            content_text if content_text is not None else raw_response,
            options=StructuredOutputOptions(parse_mode="lenient", strip_think_tags=True),
        )

        raw_insights = payload.get("insights") if isinstance(payload, dict) else payload

        if not isinstance(raw_insights, list):
            logger.warning("LLM response did not include an insights list")
            raw_insights = []

        insights = _normalize_insights(raw_insights)

        logger.debug(
            "Generated {} insights for media_id={}",
            len(insights),
            media_id,
        )

        response = DocumentInsightsResponse(
            media_id=media_id,
            insights=insights,
            model_used=model_used or provider,
            cached=False,
        )

        # Cache the response for future requests
        cache_response(
            cache_key,
            response.model_dump(),
            media_id=media_id,
        )

        return response

    except Exception as e:
        logger.error("Failed to parse LLM response for document insights")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to parse insights from LLM response.",
        ) from e


__all__ = ["router"]
