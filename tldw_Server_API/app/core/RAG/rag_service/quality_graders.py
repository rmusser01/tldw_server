"""
Quality Graders for Self-Correcting RAG

This module provides lightweight quality grading for RAG responses:
- Fast Groundedness Grader (Stage 5): Binary check if answer is grounded in sources
- Utility Grader (Stage 6): Rate response usefulness on a 1-5 scale

Part of the Self-Correcting RAG feature set (Stages 5-6).
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from loguru import logger

from tldw_Server_API.app.core.LLM_Calls.structured_output import (
    StructuredOutputOptions,
    StructuredOutputParseError,
    parse_structured_output,
)


@dataclass
class FastGroundednessResult:
    """Result of fast groundedness check."""

    is_grounded: bool
    confidence: float  # 0.0 to 1.0
    rationale: str
    latency_ms: int
    method: str  # "llm", "heuristic", "error_fallback"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UtilityResult:
    """Result of utility grading."""

    utility_score: int  # 1-5
    explanation: str
    latency_ms: int
    method: str  # "llm", "heuristic", "error_fallback"
    metadata: dict[str, Any] = field(default_factory=dict)


# Prompt for fast groundedness check
GROUNDEDNESS_PROMPT_TEMPLATE = """You are a groundedness checker. Your task is to quickly determine if an answer is supported by the provided source documents.

Query: {query}

Answer to check:
{answer}

Source Documents:
{sources}

Evaluate whether the key claims in the answer are supported by the source documents.
Focus on factual claims, not stylistic choices.

Respond with a JSON object containing:
- "is_grounded": true if the answer's main claims are supported, false otherwise
- "confidence": a number from 0.0 to 1.0 indicating how confident you are
- "rationale": a brief explanation (1-2 sentences)

JSON Response:"""


# Prompt for utility grading
UTILITY_PROMPT_TEMPLATE = """You are a response quality evaluator. Rate how useful this answer is for the given query.

Query: {query}

Answer:
{answer}

Rate the answer's usefulness on a scale of 1-5:
1 - Not useful: Doesn't address the query at all
2 - Slightly useful: Tangentially related but doesn't answer the question
3 - Moderately useful: Partially answers the question
4 - Very useful: Answers the question well
5 - Excellent: Comprehensive, well-structured answer that fully addresses the query

Respond with a JSON object containing:
- "utility_score": an integer from 1 to 5
- "explanation": a brief explanation (1-2 sentences)

JSON Response:"""


def _resolve_quality_config(config_prefix: str = "RAG_QUALITY") -> tuple[str, Optional[str], float]:
    """
    Resolve LLM provider, model, and temperature from config with fallbacks.

    Args:
        config_prefix: Prefix for config keys (e.g., "RAG_QUALITY" or "RAG_UTILITY")

    Returns:
        Tuple of (provider, model_override, temperature)
    """
    try:
        from tldw_Server_API.app.core.config import load_and_log_configs
        cfg = load_and_log_configs() or {}
    except (ImportError, AttributeError, OSError, TypeError, ValueError):
        cfg = {}

    provider = None
    model_override = None
    temperature = 0.1

    # Try quality-specific config first
    provider = str(cfg.get(f"{config_prefix}_PROVIDER", "")).strip() or None
    model_override = str(cfg.get(f"{config_prefix}_MODEL", "")).strip() or None

    # Fall back to general RAG config
    if provider is None:
        provider = str(cfg.get("RAG_DEFAULT_LLM_PROVIDER", "")).strip() or None

    # Final fallback to default_api
    if provider is None:
        provider = str(cfg.get("default_api", "openai")).strip() or "openai"

    return provider or "openai", model_override, temperature


class FastGroundednessGrader:
    """
    Fast binary groundedness check for RAG answers.

    This provides a lightweight check to determine if an answer is grounded
    in the source documents, without the full overhead of claims extraction
    and verification.
    """

    def __init__(
        self,
        analyze_fn: Optional[Callable] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        timeout_sec: float = 5.0,
    ):
        """
        Initialize the fast groundedness grader.

        Args:
            analyze_fn: Optional callback function for LLM calls.
            provider: LLM provider to use.
            model: Optional model override.
            timeout_sec: Timeout for the grading call.
        """
        self._analyze = analyze_fn
        self.provider = provider
        self.model = model
        self.timeout_sec = timeout_sec

        # Lazy load analyze function if not provided
        if self._analyze is None:
            try:
                import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl

                def _default_analyze(
                    api_name: str,
                    input_data: Any,
                    custom_prompt_arg: Optional[str] = None,
                    api_key: Optional[str] = None,
                    system_message: Optional[str] = None,
                    temp: Optional[float] = None,
                    **kwargs
                ):
                    return sgl.analyze(
                        api_name,
                        input_data,
                        custom_prompt_arg,
                        api_key,
                        system_message,
                        temp,
                        **kwargs
                    )

                self._analyze = _default_analyze
            except ImportError:
                logger.warning("Summarization_General_Lib not available for groundedness grading")
                self._analyze = None

    async def grade(
        self,
        query: str,
        answer: str,
        documents: list[Any],
    ) -> FastGroundednessResult:
        """
        Perform fast groundedness check on an answer.

        Args:
            query: The original query
            answer: The generated answer to check
            documents: List of source documents

        Returns:
            FastGroundednessResult with groundedness assessment
        """
        start_time = time.time()

        # If no analyze function available, use heuristic fallback
        if self._analyze is None:
            return self._heuristic_groundedness(query, answer, documents, start_time)

        # Build sources text from documents
        sources_text = self._build_sources_text(documents)

        # Build prompt
        prompt = GROUNDEDNESS_PROMPT_TEMPLATE.format(
            query=query,
            answer=answer[:2000] if len(answer) > 2000 else answer,  # Truncate long answers
            sources=sources_text,
        )

        try:
            # Resolve provider/model from config
            cfg_provider, cfg_model, cfg_temp = _resolve_quality_config("RAG_GROUNDEDNESS")
            use_provider = self.provider or cfg_provider
            use_model = self.model or cfg_model

            # Make LLM call with timeout
            raw_response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._analyze,
                    use_provider,
                    "",  # input_data (not used when custom_prompt_arg is set)
                    prompt,
                    None,  # api_key
                    "You are a groundedness checker. Output valid JSON only.",
                    cfg_temp,
                    model_override=use_model,
                ),
                timeout=self.timeout_sec,
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # Parse response
            return self._parse_groundedness_response(raw_response, latency_ms)

        except asyncio.TimeoutError:
            logger.warning(f"Fast groundedness check timed out after {self.timeout_sec}s")
            return FastGroundednessResult(
                is_grounded=True,  # Assume grounded on timeout (fail open)
                confidence=0.0,
                rationale="Timeout during groundedness check",
                latency_ms=int((time.time() - start_time) * 1000),
                method="error_fallback",
                metadata={"error": "timeout"},
            )

        except (AttributeError, ConnectionError, OSError, RuntimeError, TypeError, ValueError):
            logger.warning("Fast groundedness check failed; using heuristic fallback")
            return self._heuristic_groundedness(query, answer, documents, start_time)

    def _build_sources_text(self, documents: list[Any], max_chars: int = 3000) -> str:
        """Build a text representation of source documents."""
        sources = []
        total_chars = 0

        for idx, doc in enumerate(documents):
            content = getattr(doc, "content", str(doc))
            title = getattr(doc, "metadata", {}).get("title", f"Source {idx + 1}")

            # Truncate individual sources
            if len(content) > 500:
                content = content[:500] + "..."

            source_text = f"[{title}]\n{content}"
            if total_chars + len(source_text) > max_chars:
                break

            sources.append(source_text)
            total_chars += len(source_text)

        return "\n\n".join(sources) if sources else "No sources provided."

    def _parse_groundedness_response(self, raw_response: Any, latency_ms: int) -> FastGroundednessResult:
        """Parse LLM response into FastGroundednessResult."""
        try:
            response_str = str(raw_response)
            payload = parse_structured_output(
                response_str,
                options=StructuredOutputOptions(parse_mode="lenient", strip_think_tags=True),
            )
            parsed: dict[str, Any] | None = None
            if isinstance(payload, dict):
                parsed = payload
            elif isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict):
                        parsed = item
                        break
            if parsed is not None:
                is_grounded = bool(parsed.get("is_grounded", True))
                confidence = float(parsed.get("confidence", 0.5))
                rationale = str(parsed.get("rationale", ""))

                return FastGroundednessResult(
                    is_grounded=is_grounded,
                    confidence=min(1.0, max(0.0, confidence)),
                    rationale=rationale,
                    latency_ms=latency_ms,
                    method="llm",
                )

        except (StructuredOutputParseError, AttributeError, TypeError, ValueError):
            logger.debug("Failed to parse groundedness response; using parse fallback")

        # Default on parse failure
        return FastGroundednessResult(
            is_grounded=True,
            confidence=0.5,
            rationale="Could not parse LLM response",
            latency_ms=latency_ms,
            method="error_fallback",
            metadata={"parse_error": True},
        )

    def _heuristic_groundedness(
        self,
        query: str,
        answer: str,
        documents: list[Any],
        start_time: float,
    ) -> FastGroundednessResult:
        """
        Heuristic fallback for groundedness checking.

        Uses keyword overlap between answer and sources as a proxy.
        """
        # Extract significant words from answer
        answer_words = {
            w.lower() for w in re.findall(r'\b\w{4,}\b', answer)
        }

        # Extract words from all documents
        source_words: set[str] = set()
        for doc in documents:
            content = getattr(doc, "content", str(doc))
            source_words.update(
                w.lower() for w in re.findall(r'\b\w{4,}\b', content)
            )

        # Calculate overlap
        if not answer_words:
            overlap_ratio = 0.5  # Default
        else:
            overlap = answer_words & source_words
            overlap_ratio = len(overlap) / len(answer_words)

        is_grounded = overlap_ratio > 0.3
        confidence = overlap_ratio

        return FastGroundednessResult(
            is_grounded=is_grounded,
            confidence=confidence,
            rationale=f"Heuristic check: {len(answer_words & source_words)}/{len(answer_words)} answer terms found in sources",
            latency_ms=int((time.time() - start_time) * 1000),
            method="heuristic",
        )


class UtilityGrader:
    """
    Rate response usefulness on a 1-5 scale.

    This provides a quick assessment of how useful a response is,
    independent of factual grounding.
    """

    def __init__(
        self,
        analyze_fn: Optional[Callable] = None,
        provider: str = "openai",
        model: Optional[str] = None,
        timeout_sec: float = 5.0,
    ):
        """
        Initialize the utility grader.

        Args:
            analyze_fn: Optional callback function for LLM calls.
            provider: LLM provider to use.
            model: Optional model override.
            timeout_sec: Timeout for the grading call.
        """
        self._analyze = analyze_fn
        self.provider = provider
        self.model = model
        self.timeout_sec = timeout_sec

        # Lazy load analyze function if not provided
        if self._analyze is None:
            try:
                import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl

                def _default_analyze(
                    api_name: str,
                    input_data: Any,
                    custom_prompt_arg: Optional[str] = None,
                    api_key: Optional[str] = None,
                    system_message: Optional[str] = None,
                    temp: Optional[float] = None,
                    **kwargs
                ):
                    return sgl.analyze(
                        api_name,
                        input_data,
                        custom_prompt_arg,
                        api_key,
                        system_message,
                        temp,
                        **kwargs
                    )

                self._analyze = _default_analyze
            except ImportError:
                logger.warning("Summarization_General_Lib not available for utility grading")
                self._analyze = None

    async def grade(
        self,
        query: str,
        answer: str,
    ) -> UtilityResult:
        """
        Grade the utility of an answer.

        Args:
            query: The original query
            answer: The generated answer to grade

        Returns:
            UtilityResult with utility score
        """
        start_time = time.time()

        # If no analyze function available, use heuristic fallback
        if self._analyze is None:
            return self._heuristic_utility(query, answer, start_time)

        # Build prompt
        prompt = UTILITY_PROMPT_TEMPLATE.format(
            query=query,
            answer=answer[:2000] if len(answer) > 2000 else answer,
        )

        try:
            # Resolve provider/model from config
            cfg_provider, cfg_model, cfg_temp = _resolve_quality_config("RAG_UTILITY")
            use_provider = self.provider or cfg_provider
            use_model = self.model or cfg_model

            # Make LLM call with timeout
            raw_response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._analyze,
                    use_provider,
                    "",  # input_data
                    prompt,
                    None,  # api_key
                    "You are a response quality evaluator. Output valid JSON only.",
                    cfg_temp,
                    model_override=use_model,
                ),
                timeout=self.timeout_sec,
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # Parse response
            return self._parse_utility_response(raw_response, latency_ms)

        except asyncio.TimeoutError:
            logger.warning(f"Utility grading timed out after {self.timeout_sec}s")
            return UtilityResult(
                utility_score=3,  # Default to moderate on timeout
                explanation="Timeout during utility grading",
                latency_ms=int((time.time() - start_time) * 1000),
                method="error_fallback",
                metadata={"error": "timeout"},
            )

        except (AttributeError, ConnectionError, OSError, RuntimeError, TypeError, ValueError):
            logger.warning("Utility grading failed; using heuristic fallback")
            return self._heuristic_utility(query, answer, start_time)

    def _parse_utility_response(self, raw_response: Any, latency_ms: int) -> UtilityResult:
        """Parse LLM response into UtilityResult."""
        try:
            response_str = str(raw_response)
            payload = parse_structured_output(
                response_str,
                options=StructuredOutputOptions(parse_mode="lenient", strip_think_tags=True),
            )
            parsed: dict[str, Any] | None = None
            if isinstance(payload, dict):
                parsed = payload
            elif isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict):
                        parsed = item
                        break
            if parsed is not None:
                utility_score = int(parsed.get("utility_score", 3))
                utility_score = max(1, min(5, utility_score))  # Clamp to 1-5
                explanation = str(parsed.get("explanation", ""))

                return UtilityResult(
                    utility_score=utility_score,
                    explanation=explanation,
                    latency_ms=latency_ms,
                    method="llm",
                )

        except (StructuredOutputParseError, AttributeError, TypeError, ValueError):
            logger.debug("Failed to parse utility response; using parse fallback")

        # Default on parse failure
        return UtilityResult(
            utility_score=3,
            explanation="Could not parse LLM response",
            latency_ms=latency_ms,
            method="error_fallback",
            metadata={"parse_error": True},
        )

    def _heuristic_utility(
        self,
        query: str,
        answer: str,
        start_time: float,
    ) -> UtilityResult:
        """
        Heuristic fallback for utility grading.

        Uses simple signals like answer length, query term overlap, etc.
        """
        score = 3  # Start with moderate

        # Check answer length
        if len(answer) < 50:
            score -= 1  # Too short
        elif len(answer) > 200:
            score += 1  # Reasonable length

        # Check query term overlap
        query_words = {w.lower() for w in re.findall(r'\b\w{3,}\b', query)}
        answer_words = {w.lower() for w in re.findall(r'\b\w{3,}\b', answer)}

        if query_words:
            overlap = len(query_words & answer_words) / len(query_words)
            if overlap > 0.5:
                score += 1
            elif overlap < 0.2:
                score -= 1

        # Clamp to 1-5
        score = max(1, min(5, score))

        return UtilityResult(
            utility_score=score,
            explanation="Heuristic assessment based on length and relevance",
            latency_ms=int((time.time() - start_time) * 1000),
            method="heuristic",
        )


# Convenience functions for pipeline integration

async def check_fast_groundedness(
    query: str,
    answer: str,
    documents: list[Any],
    provider: Optional[str] = None,
    model: Optional[str] = None,
    timeout_sec: float = 5.0,
    analyze_fn: Optional[Callable] = None,
) -> tuple[FastGroundednessResult, dict[str, Any]]:
    """
    Convenience function to check answer groundedness.

    Args:
        query: The original query
        answer: The generated answer
        documents: Source documents
        provider: Optional LLM provider override
        model: Optional model override
        timeout_sec: Timeout for the check
        analyze_fn: Optional analyze function override

    Returns:
        Tuple of (FastGroundednessResult, metadata_dict)
    """
    grader = FastGroundednessGrader(
        analyze_fn=analyze_fn,
        provider=provider or "openai",
        model=model,
        timeout_sec=timeout_sec,
    )

    result = await grader.grade(query, answer, documents)

    metadata = {
        "fast_groundedness_enabled": True,
        "is_grounded": result.is_grounded,
        "confidence": result.confidence,
        "rationale": result.rationale,
        "latency_ms": result.latency_ms,
        "method": result.method,
    }

    return result, metadata


async def grade_utility(
    query: str,
    answer: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    timeout_sec: float = 5.0,
    analyze_fn: Optional[Callable] = None,
) -> tuple[UtilityResult, dict[str, Any]]:
    """
    Convenience function to grade answer utility.

    Args:
        query: The original query
        answer: The generated answer
        provider: Optional LLM provider override
        model: Optional model override
        timeout_sec: Timeout for grading
        analyze_fn: Optional analyze function override

    Returns:
        Tuple of (UtilityResult, metadata_dict)
    """
    grader = UtilityGrader(
        analyze_fn=analyze_fn,
        provider=provider or "openai",
        model=model,
        timeout_sec=timeout_sec,
    )

    result = await grader.grade(query, answer)

    metadata = {
        "utility_grading_enabled": True,
        "utility_score": result.utility_score,
        "explanation": result.explanation,
        "latency_ms": result.latency_ms,
        "method": result.method,
    }

    return result, metadata
