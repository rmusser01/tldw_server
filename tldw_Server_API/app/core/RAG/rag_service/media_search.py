"""
Media Search — Image and Video Search.

Provides image and video search capabilities using existing web search
infrastructure. The LLM reformulates queries for optimal image/video
search before executing. Inspired by Perplexica's media search agents.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Query reformulation prompts
# ---------------------------------------------------------------------------

_IMAGE_QUERY_SYSTEM = """\
You are a search query optimizer for image search. Rewrite the user's query \
into an optimal image search query. Strip conversational language and focus \
on the visual subject.

Rules:
- Output ONLY the optimized query string, nothing else
- Remove filler words, questions, and conversational phrases
- Keep proper nouns and specific descriptors
- Add "photo", "image", or descriptive terms when helpful

Examples:
- "What does the KFC logo look like?" → "KFC logo"
- "Show me LeBron James playing basketball" → "LeBron James basketball action shot"
- "What are the different types of cloud formations?" → "cloud formation types diagram"
- "I want to see pictures of the Eiffel Tower at night" → "Eiffel Tower night illuminated"
- "How does a diesel engine work?" → "diesel engine diagram cutaway"
"""

_VIDEO_QUERY_SYSTEM = """\
You are a search query optimizer for video search. Rewrite the user's query \
into an optimal video search query. Focus on finding tutorial, explainer, \
or documentary content.

Rules:
- Output ONLY the optimized query string, nothing else
- Remove conversational language
- Add terms like "tutorial", "explained", "guide" when appropriate
- Keep specific technical terms and proper nouns

Examples:
- "How do I change a tire?" → "how to change a tire tutorial"
- "Tell me about the history of Rome" → "history of Rome documentary"
- "What's the best way to learn Python?" → "Python programming beginner tutorial"
"""


# ---------------------------------------------------------------------------
# Query reformulation helper
# ---------------------------------------------------------------------------

async def _reformulate_query(
    query: str,
    system_prompt: str,
    llm_provider: str,
    llm_model: str | None,
) -> str:
    """Use an LLM to reformulate a query for media search."""
    try:
        from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async

        provider = (llm_provider or "openai").strip().lower()
        model = (llm_model or "").strip() or None

        call_kwargs: dict[str, Any] = {
            "api_provider": provider,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            "temperature": 0.0,
            "max_tokens": 100,
            "stream": False,
        }
        if model:
            call_kwargs["model"] = model

        raw = await asyncio.wait_for(
            perform_chat_api_call_async(**call_kwargs),
            timeout=10.0,
        )

        text = ""
        if isinstance(raw, str):
            text = raw
        elif isinstance(raw, dict):
            choices = raw.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                text = msg.get("content", "")
            if not text:
                text = raw.get("content", "") or raw.get("text", "")
        elif hasattr(raw, "content"):
            text = str(raw.content)
        else:
            text = str(raw)

        reformulated = text.strip().strip('"\'')
        return reformulated if reformulated else query

    except Exception:
        logger.debug("Media query reformulation failed; using original query")
        return query


# ---------------------------------------------------------------------------
# Image search
# ---------------------------------------------------------------------------

async def search_images(
    query: str,
    llm_provider: str = "openai",
    llm_model: str | None = None,
    max_results: int = 10,
    search_engine: str = "duckduckgo",
) -> list[dict[str, Any]]:
    """Search for relevant images.

    The LLM first reformulates the query for optimal image search,
    then results are returned with titles, URLs, and thumbnail info.

    Args:
        query: User's search query.
        llm_provider: LLM provider for query reformulation.
        llm_model: Optional model override.
        max_results: Maximum number of image results.
        search_engine: Web search engine to use.

    Returns:
        List of image result dicts with keys: title, url, thumbnail_url, source, width, height.
    """
    # Reformulate query for image search
    image_query = await _reformulate_query(query, _IMAGE_QUERY_SYSTEM, llm_provider, llm_model)
    logger.debug(f"Image search query reformulated: '{query}' → '{image_query}'")

    try:
        from tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs import (
            perform_websearch,
        )

        # Use web search with image-focused query
        raw_results = await asyncio.to_thread(
            perform_websearch,
            search_engine=search_engine,
            search_query=f"{image_query} images",
            content_country="us",
            search_lang="en",
            output_lang="en",
            result_count=max_results,
        )

        if not isinstance(raw_results, dict):
            return []

        results = raw_results.get("results", [])
        if not isinstance(results, list):
            return []

        images = []
        for r in results[:max_results]:
            if not isinstance(r, dict):
                continue
            images.append({
                "title": r.get("title", ""),
                "url": r.get("url", "") or r.get("link", ""),
                "thumbnail_url": r.get("thumbnail", "") or r.get("image", "") or r.get("url", ""),
                "source": r.get("source", "") or r.get("url", ""),
                "description": r.get("snippet", "") or r.get("content", "") or r.get("description", ""),
            })

        return images

    except Exception:
        logger.warning("Image search failed; returning no image results")
        return []


# ---------------------------------------------------------------------------
# Video search
# ---------------------------------------------------------------------------

async def search_videos(
    query: str,
    llm_provider: str = "openai",
    llm_model: str | None = None,
    max_results: int = 10,
    search_engine: str = "duckduckgo",
) -> list[dict[str, Any]]:
    """Search for relevant videos (primarily YouTube).

    The LLM first reformulates the query for optimal video search,
    then results are filtered to video platforms.

    Args:
        query: User's search query.
        llm_provider: LLM provider for query reformulation.
        llm_model: Optional model override.
        max_results: Maximum number of video results.
        search_engine: Web search engine to use.

    Returns:
        List of video result dicts with keys: title, url, thumbnail_url, source, description.
    """
    # Reformulate query for video search
    video_query = await _reformulate_query(query, _VIDEO_QUERY_SYSTEM, llm_provider, llm_model)
    logger.debug(f"Video search query reformulated: '{query}' → '{video_query}'")

    try:
        from tldw_Server_API.app.core.Web_Scraping.WebSearch_APIs import (
            perform_websearch,
        )

        # Search YouTube specifically
        raw_results = await asyncio.to_thread(
            perform_websearch,
            search_engine=search_engine,
            search_query=f"site:youtube.com {video_query}",
            content_country="us",
            search_lang="en",
            output_lang="en",
            result_count=max_results,
        )

        if not isinstance(raw_results, dict):
            return []

        results = raw_results.get("results", [])
        if not isinstance(results, list):
            return []

        videos = []
        for r in results[:max_results]:
            if not isinstance(r, dict):
                continue
            url = r.get("url", "") or r.get("link", "")
            title = r.get("title", "")
            # Extract YouTube video ID for thumbnail
            thumbnail = ""
            yt_match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
            if yt_match:
                vid_id = yt_match.group(1)
                thumbnail = f"https://img.youtube.com/vi/{vid_id}/mqdefault.jpg"

            videos.append({
                "title": title,
                "url": url,
                "thumbnail_url": thumbnail,
                "source": "youtube" if "youtube.com" in url or "youtu.be" in url else "video",
                "description": r.get("snippet", "") or r.get("content", "") or r.get("description", ""),
            })

        return videos

    except Exception:
        logger.warning("Video search failed; returning no video results")
        return []
