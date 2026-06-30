"""
Query Classification and Reformulation for Agentic RAG.

This module provides LLM-based query classification that determines:
- Whether search is needed at all (skip_search)
- Which search types to activate (local DB, web, academic, discussions)
- Standalone query reformulation for conversational follow-ups
- Detected intent for downstream routing

Inspired by Perplexica's search classifier pattern, adapted for tldw_server2's
existing query analysis infrastructure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.LLM_Calls.structured_output import (
    StructuredOutputOptions,
    StructuredOutputParseError,
    parse_structured_output,
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QueryClassification:
    """Result of LLM-based query classification."""

    skip_search: bool = False  # Can answer without search
    search_local_db: bool = True  # Search Media DB, Notes, etc.
    search_web: bool = False  # Run web search
    search_academic: bool = False  # Use academic APIs (arXiv, Semantic Scholar, etc.)
    search_discussions: bool = False  # Search forums/Reddit
    standalone_query: str = ""  # Context-independent reformulation
    detected_intent: str = "factual"  # factual, exploratory, comparative, etc.
    confidence: float = 0.5  # Classification confidence (0-1)
    reasoning: str = ""  # Brief explanation of classification decision


# ---------------------------------------------------------------------------
# Classification prompt
# ---------------------------------------------------------------------------

_CLASSIFIER_SYSTEM_PROMPT = """\
You are a search query classifier for a research assistant. Your job is to analyze \
a user query (and optional conversation history) to determine:

1. Whether the query requires a search at all, or can be answered directly.
2. Which search types are most appropriate.
3. A standalone reformulation of the query (important for follow-up questions in conversations).

Respond ONLY with a valid JSON object matching this exact schema:
{
  "skip_search": <bool>,
  "search_local_db": <bool>,
  "search_web": <bool>,
  "search_academic": <bool>,
  "search_discussions": <bool>,
  "standalone_query": "<string>",
  "detected_intent": "<string>",
  "confidence": <float 0-1>,
  "reasoning": "<brief string>"
}

Classification rules:
- skip_search=true for: greetings ("hi", "hello"), simple math ("2+2"), general knowledge \
that doesn't need sources, meta-questions about the system itself.
- search_local_db=true for: queries about previously ingested content, notes, media, \
specific documents the user has uploaded.
- search_web=true for: current events, recent information, general factual questions, \
product comparisons, how-to guides.
- search_academic=true for: scientific questions, research topics, citations needed, \
technical papers, medical/legal topics requiring authoritative sources.
- search_discussions=true for: opinion questions ("What do people think about X?"), \
troubleshooting ("common issues with Y"), recommendations, experience-based questions.
- standalone_query: ALWAYS reformulate the query to be self-contained. If chat history \
is provided and the query references prior context (pronouns like "it", "they", "this"), \
resolve those references. If no history, return the original query cleaned up.
- detected_intent: one of "factual", "exploratory", "comparative", "procedural", \
"definitional", "causal", "temporal", "analytical", "conversational", "creative".

Multiple search types CAN be true simultaneously for complex queries."""


def _build_classifier_user_prompt(
    query: str,
    chat_history: list[dict[str, str]] | None = None,
) -> str:
    """Build the user prompt for classification."""
    parts: list[str] = []

    if chat_history:
        parts.append("=== Conversation History ===")
        for msg in chat_history[-10:]:  # Last 10 messages max
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("=== End History ===\n")

    parts.append(f"Current query: {query}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Heuristic fallback classifier (no LLM required)
# ---------------------------------------------------------------------------

_GREETING_PATTERNS = re.compile(
    r"^(hi|hello|hey|howdy|good\s+(morning|afternoon|evening)|greetings|sup|yo)\b",
    re.IGNORECASE,
)
_MATH_PATTERNS = re.compile(
    r"^\s*\d+[\s+\-*/^%]+\d+[\s+\-*/^%\d]*\s*[=?]?\s*$",
)
_ACADEMIC_KEYWORDS = {
    "paper", "papers", "study", "studies", "research", "journal",
    "arxiv", "pubmed", "doi", "citation", "citations", "peer-reviewed",
    "scientific", "hypothesis", "methodology", "theorem", "proof",
    "clinical", "trial", "meta-analysis", "systematic review",
}
_DISCUSSION_KEYWORDS = {
    "reddit", "forum", "opinions", "experiences", "recommend",
    "recommendations", "what do people think", "common issues",
    "troubleshooting", "community", "discussion", "anyone else",
}


def _heuristic_classify(
    query: str,
    chat_history: list[dict[str, str]] | None = None,
) -> QueryClassification:
    """Fast heuristic classification without LLM.

    Used as fallback when LLM is unavailable or for speed mode.
    """
    q_lower = query.strip().lower()
    words = set(q_lower.split())

    # Greetings / trivial
    if _GREETING_PATTERNS.match(q_lower) or len(q_lower) < 4:
        return QueryClassification(
            skip_search=True,
            search_local_db=False,
            search_web=False,
            standalone_query=query,
            detected_intent="conversational",
            confidence=0.9,
            reasoning="Detected greeting or trivial input",
        )

    # Simple math
    if _MATH_PATTERNS.match(q_lower):
        return QueryClassification(
            skip_search=True,
            search_local_db=False,
            search_web=False,
            standalone_query=query,
            detected_intent="factual",
            confidence=0.85,
            reasoning="Detected simple arithmetic expression",
        )

    # Academic signals
    has_academic = bool(words & _ACADEMIC_KEYWORDS)

    # Discussion signals
    has_discussion = bool(words & _DISCUSSION_KEYWORDS) or any(
        phrase in q_lower for phrase in [
            "what do people think", "common issues", "anyone else",
            "in your experience",
        ]
    )

    # Web search heuristic: current events, "latest", "news", time-sensitive
    has_web = any(
        kw in q_lower for kw in [
            "latest", "news", "today", "current", "2024", "2025", "2026",
            "price", "stock", "weather", "score",
        ]
    )

    return QueryClassification(
        skip_search=False,
        search_local_db=True,
        search_web=has_web or (not has_academic),
        search_academic=has_academic,
        search_discussions=has_discussion,
        standalone_query=query,
        detected_intent=_detect_intent_heuristic(q_lower),
        confidence=0.5,
        reasoning="Heuristic classification (no LLM)",
    )


def _detect_intent_heuristic(q_lower: str) -> str:
    """Detect query intent using simple heuristics."""
    if q_lower.startswith(("what is", "what are", "define", "meaning of")):
        return "definitional"
    if q_lower.startswith(("how to", "how do", "how can", "steps to")):
        return "procedural"
    if q_lower.startswith(("why", "what causes", "reason for")):
        return "causal"
    if q_lower.startswith(("when", "what year", "what date", "timeline")):
        return "temporal"
    if any(kw in q_lower for kw in ["compare", "vs", "versus", "difference between", "better"]):
        return "comparative"
    if any(kw in q_lower for kw in ["analyze", "evaluate", "assess", "review"]):
        return "analytical"
    if q_lower.startswith(("tell me about", "explain", "describe", "overview")):
        return "exploratory"
    return "factual"


# ---------------------------------------------------------------------------
# LLM-based classifier
# ---------------------------------------------------------------------------

def _parse_classification_response(raw: str) -> dict[str, Any]:
    """Parse LLM JSON response, handling common formatting issues."""
    text = raw.strip()

    try:
        payload = parse_structured_output(
            text,
            options=StructuredOutputOptions(parse_mode="lenient", strip_think_tags=True),
        )
        if isinstance(payload, dict):
            return dict(payload)
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    return dict(item)
    except StructuredOutputParseError:
        pass

    raise ValueError(f"Could not parse classification JSON from: {text[:200]}")


async def classify_query(
    query: str,
    chat_history: list[dict[str, str]] | None = None,
    llm_provider: str = "openai",
    llm_model: str | None = None,
    timeout_sec: float = 10.0,
) -> QueryClassification:
    """Classify a query to determine search routing and reformulation.

    Uses an LLM to analyze the query and optional conversation history.
    Falls back to heuristic classification if LLM is unavailable.

    Args:
        query: The user's query string.
        chat_history: Optional list of prior messages [{"role": "user"|"assistant", "content": "..."}].
        llm_provider: LLM provider name (e.g., "openai", "anthropic").
        llm_model: Optional model override (defaults to provider's default).
        timeout_sec: Maximum seconds to wait for LLM response.

    Returns:
        QueryClassification with routing decisions and reformulated query.
    """
    # Fast path: heuristic for obviously classifiable queries
    q_stripped = query.strip()
    if _GREETING_PATTERNS.match(q_stripped.lower()) or _MATH_PATTERNS.match(q_stripped):
        return _heuristic_classify(query, chat_history)

    # Attempt LLM classification
    try:
        import asyncio

        from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async

        provider = (llm_provider or "openai").strip().lower()
        model = (llm_model or "").strip() or None

        user_prompt = _build_classifier_user_prompt(query, chat_history)

        call_kwargs: dict[str, Any] = {
            "api_provider": provider,
            "messages": [
                {"role": "system", "content": _CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 500,
            "stream": False,
        }
        if model:
            call_kwargs["model"] = model

        raw_response = await asyncio.wait_for(
            perform_chat_api_call_async(**call_kwargs),
            timeout=timeout_sec,
        )

        # Extract text content from response
        response_text = ""
        if isinstance(raw_response, str):
            response_text = raw_response
        elif isinstance(raw_response, dict):
            # OpenAI-style response
            choices = raw_response.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                response_text = msg.get("content", "")
            if not response_text:
                response_text = raw_response.get("content", "") or raw_response.get("text", "")
        elif hasattr(raw_response, "content"):
            response_text = str(raw_response.content)
        else:
            response_text = str(raw_response)

        if not response_text.strip():
            logger.warning("Empty LLM response for query classification, falling back to heuristic")
            return _heuristic_classify(query, chat_history)

        parsed = _parse_classification_response(response_text)

        return QueryClassification(
            skip_search=bool(parsed.get("skip_search", False)),
            search_local_db=bool(parsed.get("search_local_db", True)),
            search_web=bool(parsed.get("search_web", False)),
            search_academic=bool(parsed.get("search_academic", False)),
            search_discussions=bool(parsed.get("search_discussions", False)),
            standalone_query=str(parsed.get("standalone_query", query)).strip() or query,
            detected_intent=str(parsed.get("detected_intent", "factual")),
            confidence=float(parsed.get("confidence", 0.7)),
            reasoning=str(parsed.get("reasoning", "")),
        )

    except Exception as exc:
        logger.warning(
            "LLM query classification failed ({exc_type}), falling back to heuristic",
            exc_type=type(exc).__name__,
        )
        return _heuristic_classify(query, chat_history)


# ---------------------------------------------------------------------------
# Standalone query reformulation (can be used independently)
# ---------------------------------------------------------------------------

_REFORMULATION_SYSTEM_PROMPT = """\
You are a query reformulation assistant. Given a conversation history and a follow-up \
question, rewrite the follow-up as a completely standalone, self-contained query that \
can be understood without any prior context.

Rules:
- Resolve all pronouns and references (it, they, this, that, those, the above, etc.)
- Include all relevant context from the conversation
- Keep the reformulated query concise but complete
- If the query is already standalone, return it as-is with minimal changes
- Return ONLY the reformulated query text, nothing else"""


async def reformulate_query(
    query: str,
    chat_history: list[dict[str, str]],
    llm_provider: str = "openai",
    llm_model: str | None = None,
    timeout_sec: float = 8.0,
) -> str:
    """Reformulate a conversational follow-up into a standalone query.

    Args:
        query: The current user query (potentially a follow-up).
        chat_history: Prior conversation messages.
        llm_provider: LLM provider name.
        llm_model: Optional model override.
        timeout_sec: Maximum seconds to wait for LLM response.

    Returns:
        A standalone, self-contained query string.
    """
    if not chat_history:
        return query

    try:
        import asyncio

        from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async

        # Build context from history
        history_text = "\n".join(
            f"{msg.get('role', 'user')}: {msg.get('content', '')}"
            for msg in chat_history[-8:]  # Last 8 messages max
        )
        user_prompt = (
            f"Conversation history:\n{history_text}\n\n"
            f"Follow-up question: {query}\n\n"
            f"Standalone reformulation:"
        )

        provider = (llm_provider or "openai").strip().lower()
        model = (llm_model or "").strip() or None

        call_kwargs: dict[str, Any] = {
            "api_provider": provider,
            "messages": [
                {"role": "system", "content": _REFORMULATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 300,
            "stream": False,
        }
        if model:
            call_kwargs["model"] = model

        raw_response = await asyncio.wait_for(
            perform_chat_api_call_async(**call_kwargs),
            timeout=timeout_sec,
        )

        # Extract text content from response
        response_text = ""
        if isinstance(raw_response, str):
            response_text = raw_response
        elif isinstance(raw_response, dict):
            choices = raw_response.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                response_text = msg.get("content", "")
            if not response_text:
                response_text = raw_response.get("content", "") or raw_response.get("text", "")
        elif hasattr(raw_response, "content"):
            response_text = str(raw_response.content)
        else:
            response_text = str(raw_response)

        reformulated = response_text.strip()
        if reformulated and len(reformulated) >= 3:
            return reformulated

        logger.warning("Empty or too-short reformulation result, returning original query")
        return query

    except Exception as exc:
        logger.warning(
            "Query reformulation failed ({exc_type}), returning original query",
            exc_type=type(exc).__name__,
        )
        return query


# ---------------------------------------------------------------------------
# Combined helper: classify + reformulate in a single call
# ---------------------------------------------------------------------------

async def classify_and_reformulate(
    query: str,
    chat_history: list[dict[str, str]] | None = None,
    llm_provider: str = "openai",
    llm_model: str | None = None,
) -> QueryClassification:
    """Classify query and ensure standalone_query is properly reformulated.

    This is the recommended entry point. It runs classification first (which
    includes basic reformulation), and if chat_history is provided but the
    classification's standalone_query still looks like a follow-up, runs
    dedicated reformulation.

    Args:
        query: The user's query.
        chat_history: Optional conversation history.
        llm_provider: LLM provider name.
        llm_model: Optional model override.

    Returns:
        QueryClassification with fully resolved standalone_query.
    """
    classification = await classify_query(
        query=query,
        chat_history=chat_history,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    # If chat history exists and the standalone_query is still the same as
    # the original (classification might not have reformulated properly),
    # attempt dedicated reformulation
    if chat_history and classification.standalone_query.strip().lower() == query.strip().lower():
        # Check if query looks like a follow-up (contains pronouns/references)
        follow_up_indicators = {"it", "they", "this", "that", "those", "these", "its", "their"}
        query_words = set(query.lower().split())
        if query_words & follow_up_indicators:
            reformulated = await reformulate_query(
                query=query,
                chat_history=chat_history,
                llm_provider=llm_provider,
                llm_model=llm_model,
            )
            classification.standalone_query = reformulated

    return classification
