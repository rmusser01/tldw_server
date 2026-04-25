"""Manuscript analysis service -- structured LLM analysis of writing content."""
from __future__ import annotations

import json
from typing import Any

from loguru import logger


PACING_PROMPT = """Analyze the pacing of the following text. Return a JSON object with:
- "pacing": float 0-1 (0=very slow, 1=very fast)
- "tension": float 0-1 (0=no tension, 1=maximum)
- "atmosphere": float 0-1 (0=flat, 1=rich)
- "engagement": float 0-1 (0=boring, 1=gripping)
- "assessment": string (1-2 sentence summary)
- "beats": list of strings (key story beats found)

Text:
{text}

Return ONLY valid JSON, no markdown fences."""


PLOT_HOLES_PROMPT = """Analyze the following manuscript for plot holes and inconsistencies.

Characters: {characters}
World Info: {world_info}

Manuscript text:
{text}

Return a JSON object with:
- "plot_holes": list of objects with "title" (string), "description" (string), "severity" ("low"/"medium"/"high"/"critical"), "location_hint" (string)
- "inconsistencies": list of strings describing each inconsistency

Return ONLY valid JSON, no markdown fences."""


CONSISTENCY_PROMPT = """Check the following manuscript for character and world-building consistency.

Characters: {characters}
World Info: {world_info}

Manuscript text:
{text}

Return a JSON object with:
- "character_issues": list of objects with "character_name", "issue", "severity" ("low"/"medium"/"high")
- "world_issues": list of objects with "entity_name", "issue", "severity"
- "timeline_issues": list of strings
- "overall_score": float 0-1 (1=perfectly consistent)

Return ONLY valid JSON, no markdown fences."""


_SYSTEM_MESSAGE = (
    "You are a literary analysis assistant. "
    "Respond ONLY with valid JSON. No explanatory text."
)


async def analyze_pacing(
    text: str,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Analyze pacing, tension, atmosphere of text."""
    return await _run_structured_analysis(
        PACING_PROMPT.format(text=text[:8000]),
        provider=provider,
        model=model,
    )


async def analyze_plot_holes(
    text: str,
    characters: str = "",
    world_info: str = "",
    *,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Detect plot holes and inconsistencies."""
    return await _run_structured_analysis(
        PLOT_HOLES_PROMPT.format(
            text=text[:12000],
            characters=characters[:2000],
            world_info=world_info[:2000],
        ),
        provider=provider,
        model=model,
    )


async def analyze_consistency(
    text: str,
    characters: str = "",
    world_info: str = "",
    *,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Check character/world consistency."""
    return await _run_structured_analysis(
        CONSISTENCY_PROMPT.format(
            text=text[:12000],
            characters=characters[:2000],
            world_info=world_info[:2000],
        ),
        provider=provider,
        model=model,
    )


async def _run_structured_analysis(
    prompt: str,
    *,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Send structured analysis prompt to LLM and parse JSON response."""
    try:
        from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async
    except ImportError:
        logger.warning("Chat service not available for manuscript analysis")
        return {"error": "LLM service unavailable"}

    kwargs: dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt}],
        "system_message": _SYSTEM_MESSAGE,
        "temp": 0.3,
    }
    if provider:
        kwargs["api_endpoint"] = provider
    if model:
        kwargs["model"] = model

    content: str = ""
    try:
        response = await perform_chat_api_call_async(**kwargs)
        content = _extract_content(response)
        content = _strip_markdown_fences(content)
        return json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Failed to parse analysis JSON response")
        return {"error": "Failed to parse LLM response", "raw": content[:500] if content else ""}
    except Exception:
        logger.exception("Analysis LLM call failed")
        return {
            "error": "analysis_failed",
            "message": "Analysis service unavailable",
        }


def _extract_content(response: Any) -> str:
    """Extract text content from various LLM response formats."""
    if isinstance(response, str):
        return response
    if isinstance(response, list):
        parts: list[str] = []
        for part in response:
            extracted = _extract_content_block(part)
            if extracted:
                parts.append(extracted)
        return "".join(parts)
    if isinstance(response, dict):
        choices = response.get("choices", [])
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                msg = choice.get("message", {})
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    extracted = _extract_content_block(content)
                    if extracted:
                        return extracted
        content = response.get("content", "")
        extracted = _extract_content_block(content)
        if extracted:
            return extracted
        return ""
    return _extract_content_block(response) or str(response)


def _extract_content_block(content: Any, *, _depth: int = 0, max_depth: int = 20) -> str:
    """Extract text from a content block, list of blocks, or nested payload.

    Parameters
    ----------
    content:
        The value to extract text from.
    _depth:
        Current recursion depth (internal use).
    max_depth:
        Maximum recursion depth to prevent stack overflow on malformed data.
    """
    if _depth >= max_depth:
        logger.warning(
            f"_extract_content_block: max recursion depth ({max_depth}) reached; "
            "returning str(content) to avoid stack overflow"
        )
        return str(content) if content is not None else ""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    next_depth = _depth + 1
    if isinstance(content, list):
        parts = [_extract_content_block(part, _depth=next_depth, max_depth=max_depth) for part in content]
        return "".join(part for part in parts if part)
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        nested_content = content.get("content")
        if nested_content is not None:
            extracted = _extract_content_block(nested_content, _depth=next_depth, max_depth=max_depth)
            if extracted:
                return extracted
        message = content.get("message")
        if message is not None:
            extracted = _extract_content_block(message, _depth=next_depth, max_depth=max_depth)
            if extracted:
                return extracted
        value = content.get("value")
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return _extract_content_block(value, _depth=next_depth, max_depth=max_depth)
        return ""
    text = getattr(content, "text", None)
    if isinstance(text, str):
        return text
    nested_content = getattr(content, "content", None)
    if nested_content is not None:
        return _extract_content_block(nested_content, _depth=next_depth, max_depth=max_depth)
    return ""


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` wrapping if present."""
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence
        first_newline = text.find("\n")
        text = text[first_newline + 1:] if first_newline != -1 else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    if text.startswith("json"):
        text = text[4:].strip()
    return text
