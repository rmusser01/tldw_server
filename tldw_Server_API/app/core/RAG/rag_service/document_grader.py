"""
Document Grader for Self-Correcting RAG

This module provides LLM-based document relevance grading for pre-generation filtering.
Documents are graded for relevance to the query before expensive reranking/generation.

Part of the Self-Correcting RAG feature set (Stage 1).
"""

import asyncio
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
class GradingConfig:
    """Configuration for document grading."""

    provider: Optional[str] = None
    model: Optional[str] = None  # None uses provider default
    batch_size: int = 5
    timeout_seconds: float = 30.0
    fallback_to_score: bool = True
    fallback_min_score: float = 0.3
    temperature: float = 0.1


@dataclass
class GradingResult:
    """Result of grading a single document."""

    document_id: str
    is_relevant: bool
    relevance_score: float  # 0.0 to 1.0
    reasoning: str
    latency_ms: int
    method: str  # "llm", "score_fallback", "error_fallback"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GradingBatchResult:
    """Result of grading a batch of documents."""

    results: list[GradingResult]
    relevant_count: int
    total_count: int
    avg_relevance: float
    total_latency_ms: int
    metadata: dict[str, Any] = field(default_factory=dict)


# Default grading prompt template
GRADING_PROMPT_TEMPLATE = """You are a document relevance grader. Your task is to assess whether a document is relevant to a given query.

Query: {query}

Document Content:
{document_content}

Evaluate the document's relevance to the query. Consider:
1. Does the document contain information that helps answer the query?
2. Is the document topically related to the query?
3. Would this document be useful as context for answering the query?

Respond with a JSON object containing:
- "is_relevant": true or false
- "relevance_score": a number from 0.0 (completely irrelevant) to 1.0 (highly relevant)
- "reasoning": a brief explanation (1-2 sentences)

JSON Response:"""


def _fallback_error_label(error: Optional[str]) -> Optional[str]:
    """Map fallback errors to fixed labels safe for result metadata."""
    if error is None:
        return None
    if error in {"timeout", "batch_timeout"}:
        return error
    return "grading_error"


def _resolve_grading_config() -> tuple[str, Optional[str], float]:
    """
    Resolve LLM provider, model, and temperature from config with fallbacks.

    Returns:
        Tuple of (provider, model_override, temperature)
    """
    try:
        from tldw_Server_API.app.core.config import load_and_log_configs
        cfg = load_and_log_configs() or {}
    except Exception:
        cfg = {}

    provider = None
    model_override = None
    temperature = 0.1

    # Try grading-specific config first
    provider = str(cfg.get("RAG_GRADING_PROVIDER", "")).strip() or None
    model_override = str(cfg.get("RAG_GRADING_MODEL", "")).strip() or None

    # Fall back to general RAG config
    if provider is None:
        provider = str(cfg.get("RAG_DEFAULT_LLM_PROVIDER", "")).strip() or None

    # Final fallback to default_api
    if provider is None:
        provider = str(cfg.get("default_api", "openai")).strip() or "openai"

    return provider or "openai", model_override, temperature


class DocumentGrader:
    """
    Grades documents for relevance to a query using LLM-based assessment.

    This is used in the Self-Correcting RAG pipeline to filter out irrelevant
    documents before expensive reranking or generation steps.
    """

    def __init__(
        self,
        analyze_fn: Optional[Callable] = None,
        config: Optional[GradingConfig] = None,
    ):
        """
        Initialize the document grader.

        Args:
            analyze_fn: Optional callback function for LLM calls.
                        If not provided, will use the default Summarization_General_Lib.analyze
            config: Optional grading configuration. Uses defaults if not provided.
        """
        self._analyze = analyze_fn
        self.config = config or GradingConfig()

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
                logger.warning("Summarization_General_Lib not available for document grading")
                self._analyze = None

    async def grade_document(
        self,
        query: str,
        document: Any,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> GradingResult:
        """
        Grade a single document for relevance to the query.

        Args:
            query: The user's query
            document: Document object with 'id', 'content', and optionally 'score' attributes
            provider: Optional LLM provider override
            model: Optional model override

        Returns:
            GradingResult with relevance assessment
        """
        doc_id = getattr(document, "id", str(id(document)))
        doc_content = getattr(document, "content", "")
        doc_score = getattr(document, "score", 0.0)

        # Truncate content if too long (avoid token limits)
        max_content_chars = 3000
        if len(doc_content) > max_content_chars:
            doc_content = doc_content[:max_content_chars] + "..."

        start_time = time.time()

        # If no analyze function available, fall back to score-based grading
        if self._analyze is None:
            return self._fallback_to_score(doc_id, doc_score, start_time)

        # Resolve provider/model
        cfg_provider, cfg_model, cfg_temp = _resolve_grading_config()
        use_provider = provider or self.config.provider or cfg_provider
        use_model = model or self.config.model or cfg_model

        # Build the prompt
        prompt = GRADING_PROMPT_TEMPLATE.format(
            query=query,
            document_content=doc_content,
        )

        try:
            # Call LLM via asyncio.to_thread to avoid blocking
            raw_response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._analyze,
                    use_provider,
                    "",  # input_data (not used when custom_prompt_arg is provided)
                    prompt,
                    None,  # api_key
                    "You are a document relevance grader. Output valid JSON only.",
                    self.config.temperature,
                    model_override=use_model,
                    streaming=False,
                ),
                timeout=self.config.timeout_seconds,
            )

            latency_ms = int((time.time() - start_time) * 1000)

            # Parse the response
            return self._parse_grading_response(
                doc_id, raw_response, latency_ms, doc_score
            )

        except asyncio.TimeoutError:
            logger.warning(f"Document grading timed out for doc {doc_id}")
            return self._fallback_to_score(doc_id, doc_score, start_time, error="timeout")

        except Exception as e:
            logger.warning(f"Document grading failed for doc {doc_id}; using fallback")
            return self._fallback_to_score(doc_id, doc_score, start_time, error="grading_error")

    def _parse_grading_response(
        self,
        doc_id: str,
        raw_response: str,
        latency_ms: int,
        fallback_score: float,
    ) -> GradingResult:
        """Parse LLM response into a GradingResult."""
        try:
            response_text = str(raw_response).strip()
            payload = parse_structured_output(
                response_text,
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
            if parsed is None:
                raise ValueError("grading response must be a JSON object")

            is_relevant = bool(parsed.get("is_relevant", False))
            relevance_score = float(parsed.get("relevance_score", 0.0))
            reasoning = str(parsed.get("reasoning", ""))

            # Clamp score to valid range
            relevance_score = max(0.0, min(1.0, relevance_score))

            return GradingResult(
                document_id=doc_id,
                is_relevant=is_relevant,
                relevance_score=relevance_score,
                reasoning=reasoning,
                latency_ms=latency_ms,
                method="llm",
            )

        except (StructuredOutputParseError, ValueError, KeyError):
            logger.warning(
                f"Failed to parse grading response for doc {doc_id}; using heuristic fallback"
            )
            # Fall back to heuristic parsing
            return self._heuristic_parse(doc_id, raw_response, latency_ms, fallback_score)

    def _heuristic_parse(
        self,
        doc_id: str,
        raw_response: str,
        latency_ms: int,
        fallback_score: float,
    ) -> GradingResult:
        """Heuristic parsing when JSON parsing fails."""
        response_lower = str(raw_response).lower()

        # Simple keyword detection
        positive_keywords = ["relevant", "useful", "helpful", "related", "yes", "true"]
        negative_keywords = ["irrelevant", "not relevant", "unrelated", "no", "false"]

        positive_count = sum(1 for kw in positive_keywords if kw in response_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in response_lower)

        if positive_count > negative_count:
            is_relevant = True
            relevance_score = min(0.8, 0.5 + (positive_count * 0.1))
        elif negative_count > positive_count:
            is_relevant = False
            relevance_score = max(0.2, 0.5 - (negative_count * 0.1))
        else:
            # Unclear, use fallback score
            is_relevant = fallback_score >= self.config.fallback_min_score
            relevance_score = fallback_score

        return GradingResult(
            document_id=doc_id,
            is_relevant=is_relevant,
            relevance_score=relevance_score,
            reasoning="Heuristic parsing from LLM response",
            latency_ms=latency_ms,
            method="llm_heuristic",
        )

    def _fallback_to_score(
        self,
        doc_id: str,
        doc_score: float,
        start_time: float,
        error: Optional[str] = None,
    ) -> GradingResult:
        """Fall back to using the document's existing score for grading."""
        latency_ms = int((time.time() - start_time) * 1000)
        error_label = _fallback_error_label(error)
        metadata = {"error": error_label} if error_label else {}

        if not self.config.fallback_to_score:
            # If fallback is disabled, mark as irrelevant
            return GradingResult(
                document_id=doc_id,
                is_relevant=False,
                relevance_score=0.0,
                reasoning="Grading failed and fallback disabled" if error else "Grading unavailable",
                latency_ms=latency_ms,
                method="error_fallback",
                metadata=metadata,
            )

        # Use existing score
        is_relevant = doc_score >= self.config.fallback_min_score
        return GradingResult(
            document_id=doc_id,
            is_relevant=is_relevant,
            relevance_score=doc_score,
            reasoning="Using retrieval score as fallback",
            latency_ms=latency_ms,
            method="score_fallback",
            metadata=metadata,
        )

    async def grade_documents(
        self,
        query: str,
        documents: list[Any],
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> GradingBatchResult:
        """
        Grade multiple documents for relevance to the query.

        Args:
            query: The user's query
            documents: List of Document objects
            provider: Optional LLM provider override
            model: Optional model override

        Returns:
            GradingBatchResult with all results and aggregated metrics
        """
        if not documents:
            return GradingBatchResult(
                results=[],
                relevant_count=0,
                total_count=0,
                avg_relevance=0.0,
                total_latency_ms=0,
            )

        start_time = time.time()
        results: list[GradingResult] = []

        # Process in batches
        for i in range(0, len(documents), self.config.batch_size):
            batch = documents[i:i + self.config.batch_size]

            # Grade batch concurrently
            batch_tasks = [
                self.grade_document(query, doc, provider, model)
                for doc in batch
            ]

            try:
                batch_results = await asyncio.wait_for(
                    asyncio.gather(*batch_tasks, return_exceptions=True),
                    timeout=self.config.timeout_seconds,
                )

                for j, result in enumerate(batch_results):
                    if isinstance(result, BaseException):
                        # Handle exception for this document
                        doc = batch[j]
                        doc_id = getattr(doc, "id", str(id(doc)))
                        doc_score = getattr(doc, "score", 0.0)
                        results.append(self._fallback_to_score(
                            doc_id, doc_score, start_time, error=str(result)
                        ))
                    else:
                        results.append(result)

            except asyncio.TimeoutError:
                logger.warning(f"Batch grading timed out at index {i}")
                # Fall back for remaining documents in batch
                for doc in batch[len(results) - i:]:
                    doc_id = getattr(doc, "id", str(id(doc)))
                    doc_score = getattr(doc, "score", 0.0)
                    results.append(self._fallback_to_score(
                        doc_id, doc_score, start_time, error="batch_timeout"
                    ))

        total_latency_ms = int((time.time() - start_time) * 1000)

        # Compute aggregates
        relevant_count = sum(1 for r in results if r.is_relevant)
        total_count = len(results)
        avg_relevance = (
            sum(r.relevance_score for r in results) / total_count
            if total_count > 0 else 0.0
        )

        return GradingBatchResult(
            results=results,
            relevant_count=relevant_count,
            total_count=total_count,
            avg_relevance=avg_relevance,
            total_latency_ms=total_latency_ms,
            metadata={
                "batch_size": self.config.batch_size,
                "provider": provider or self.config.provider,
                "model": model or self.config.model,
            },
        )

    async def filter_relevant(
        self,
        query: str,
        documents: list[Any],
        threshold: float = 0.5,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> tuple[list[Any], dict[str, Any]]:
        """
        Filter documents to keep only those above the relevance threshold.

        Args:
            query: The user's query
            documents: List of Document objects
            threshold: Minimum relevance score to keep (0.0 to 1.0)
            provider: Optional LLM provider override
            model: Optional model override

        Returns:
            Tuple of (filtered_documents, grading_metadata)
        """
        if not documents:
            return [], {"grading_skipped": True, "reason": "no_documents"}

        # Grade all documents
        batch_result = await self.grade_documents(query, documents, provider, model)

        # Build a mapping of doc_id to document
        doc_map = {getattr(d, "id", str(id(d))): d for d in documents}

        # Filter by threshold
        filtered_docs = []
        for result in batch_result.results:
            if result.relevance_score >= threshold:
                doc = doc_map.get(result.document_id)
                if doc is not None:
                    filtered_docs.append(doc)

        metadata = {
            "total_graded": batch_result.total_count,
            "relevant_count": batch_result.relevant_count,
            "filtered_count": len(filtered_docs),
            "removed_count": batch_result.total_count - len(filtered_docs),
            "avg_relevance": batch_result.avg_relevance,
            "threshold": threshold,
            "total_latency_ms": batch_result.total_latency_ms,
            "grading_results": [
                {
                    "doc_id": r.document_id,
                    "is_relevant": r.is_relevant,
                    "score": r.relevance_score,
                    "method": r.method,
                }
                for r in batch_result.results
            ],
        }

        return filtered_docs, metadata


# Convenience function for pipeline integration
async def grade_and_filter_documents(
    query: str,
    documents: list[Any],
    threshold: float = 0.5,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    batch_size: int = 5,
    timeout_seconds: float = 30.0,
    fallback_to_score: bool = True,
    fallback_min_score: float = 0.3,
) -> tuple[list[Any], dict[str, Any]]:
    """
    Convenience function to grade and filter documents in one call.

    Args:
        query: The user's query
        documents: List of Document objects
        threshold: Minimum relevance score to keep
        provider: LLM provider
        model: LLM model
        batch_size: Number of documents to grade concurrently
        timeout_seconds: Timeout for batch grading
        fallback_to_score: Whether to use retrieval score as fallback
        fallback_min_score: Minimum score for fallback relevance

    Returns:
        Tuple of (filtered_documents, grading_metadata)
    """
    config = GradingConfig(
        provider=provider,
        model=model,
        batch_size=batch_size,
        timeout_seconds=timeout_seconds,
        fallback_to_score=fallback_to_score,
        fallback_min_score=fallback_min_score,
    )

    grader = DocumentGrader(config=config)
    return await grader.filter_relevant(query, documents, threshold, provider, model)
