"""
Custom evaluation metrics for RAG system.

Provides domain-specific metrics beyond standard RAG evaluations,
tailored for production use cases and business requirements.
"""

import asyncio
import os
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import analyze

# Safe import of embeddings helpers to avoid heavy deps during app import
try:
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import (
        create_embedding,
        get_embedding_config,
    )
    _RAG_METRICS_EMBEDDINGS_AVAILABLE = True
except Exception:
    _RAG_METRICS_EMBEDDINGS_AVAILABLE = False
    def create_embedding(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("Embeddings backend unavailable; install required dependencies")
    def get_embedding_config():  # type: ignore[misc]
        return {"embedding_config": {"default_model_id": ""}}


class MetricType(Enum):
    """Types of custom metrics"""
    # Retrieval quality
    RETRIEVAL_COVERAGE = "retrieval_coverage"
    RETRIEVAL_DIVERSITY = "retrieval_diversity"
    SOURCE_ATTRIBUTION = "source_attribution"

    # Response quality
    RESPONSE_COMPLETENESS = "response_completeness"
    RESPONSE_COHERENCE = "response_coherence"
    RESPONSE_FACTUALITY = "response_factuality"

    # User experience
    RESPONSE_TIME = "response_time"
    RESULT_RELEVANCE = "result_relevance"
    ANSWER_CLARITY = "answer_clarity"

    # Business metrics
    QUERY_SUCCESS_RATE = "query_success_rate"
    USER_SATISFACTION = "user_satisfaction"
    COST_EFFICIENCY = "cost_efficiency"


@dataclass
class CustomMetricResult:
    """Result of a custom metric evaluation"""
    metric_type: MetricType
    score: float  # 0.0 to 1.0
    confidence: float  # Confidence in the score
    details: dict[str, Any] = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class RAGCustomMetrics:
    """Custom metrics evaluator for RAG systems"""

    def __init__(self, embedding_config: Optional[dict[str, Any]] = None):
        """
        Initialize custom metrics evaluator.

        Args:
            embedding_config: Configuration for embeddings
        """
        self.embedding_config = embedding_config or get_embedding_config()
        self.metrics_cache = {}

    async def evaluate_retrieval_coverage(
        self,
        query: str,
        retrieved_contexts: list[str],
        reference_answer: Optional[str] = None
    ) -> CustomMetricResult:
        """
        Evaluate how well retrieved contexts cover the query intent.

        Measures whether all aspects of the query are addressed by retrieved content.
        """
        # Extract key concepts from query
        query_concepts = await self._extract_concepts(query)

        # Check coverage in retrieved contexts
        covered_concepts = set()
        context_text = " ".join(retrieved_contexts)

        for concept in query_concepts:
            if concept.lower() in context_text.lower():
                covered_concepts.add(concept)

        coverage_score = len(covered_concepts) / len(query_concepts) if query_concepts else 0

        # Additional check with reference answer if provided
        confidence = 0.8
        if reference_answer:
            ref_concepts = await self._extract_concepts(reference_answer)
            ref_coverage = len(covered_concepts.intersection(ref_concepts)) / len(ref_concepts) if ref_concepts else 0
            coverage_score = (coverage_score + ref_coverage) / 2
            confidence = 0.9

        suggestions = []
        if coverage_score < 0.7:
            uncovered = set(query_concepts) - covered_concepts
            suggestions.append(f"Missing coverage for: {', '.join(uncovered)}")
            suggestions.append("Consider expanding search parameters or databases")

        return CustomMetricResult(
            metric_type=MetricType.RETRIEVAL_COVERAGE,
            score=coverage_score,
            confidence=confidence,
            details={
                "query_concepts": query_concepts,
                "covered_concepts": list(covered_concepts),
                "coverage_percentage": coverage_score * 100
            },
            suggestions=suggestions
        )

    async def evaluate_retrieval_diversity(
        self,
        retrieved_contexts: list[str],
        sources: Optional[list[str]] = None
    ) -> CustomMetricResult:
        """
        Evaluate diversity of retrieved content.

        Measures how diverse the retrieved contexts are to avoid redundancy.
        """
        if not retrieved_contexts:
            return CustomMetricResult(
                metric_type=MetricType.RETRIEVAL_DIVERSITY,
                score=0.0,
                confidence=1.0,
                details={"reason": "No contexts retrieved"}
            )

        # Calculate pairwise similarities
        embeddings = []
        # Offload embedding calls to a thread so callers can timebox with asyncio.wait_for
        for context in retrieved_contexts[:10]:  # Limit for performance
            embedding = await asyncio.to_thread(create_embedding, context, self.embedding_config)
            embeddings.append(embedding)

        # Convert to numpy array
        embeddings_array = np.array(embeddings)

        # Calculate diversity as inverse of average similarity
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity(
                    embeddings_array[i:i+1],
                    embeddings_array[j:j+1]
                )[0][0]
                similarities.append(sim)

        avg_similarity = np.mean(similarities) if similarities else 0
        diversity_score = 1 - avg_similarity

        # Source diversity bonus
        source_diversity_bonus = 0
        if sources:
            unique_sources = len(set(sources))
            total_sources = len(sources)
            source_diversity_bonus = (unique_sources / total_sources) * 0.2
            diversity_score = min(1.0, diversity_score + source_diversity_bonus)

        suggestions = []
        if diversity_score < 0.5:
            suggestions.append("Retrieved contexts are too similar")
            suggestions.append("Consider using different search strategies or sources")

        return CustomMetricResult(
            metric_type=MetricType.RETRIEVAL_DIVERSITY,
            score=diversity_score,
            confidence=0.85,
            details={
                "average_similarity": avg_similarity,
                "unique_sources": len(set(sources)) if sources else 0,
                "total_contexts": len(retrieved_contexts)
            },
            suggestions=suggestions
        )

    async def evaluate_response_completeness(
        self,
        query: str,
        response: str,
        expected_elements: Optional[list[str]] = None
    ) -> CustomMetricResult:
        """
        Evaluate completeness of the generated response.

        Checks if the response addresses all aspects of the query.
        """
        # Use LLM to evaluate completeness
        prompt = f"""
        Evaluate the completeness of this response to the given query.

        Query: {query}

        Response: {response}

        Rate the completeness on a scale of 0-10 where:
        - 0-3: Response is incomplete, missing major elements
        - 4-6: Response covers basics but lacks depth
        - 7-8: Response is mostly complete with minor gaps
        - 9-10: Response is comprehensive and complete

        Also identify any missing elements.

        Format your response as:
        Score: [0-10]
        Missing: [list any missing elements or "None"]
        """

        try:
            llm_response = await analyze(prompt, "You are an expert at evaluating response completeness")

            # Parse score
            score_line = [l for l in llm_response.split('\n') if 'Score:' in l]
            score = float(score_line[0].split(':')[1].strip()) / 10 if score_line else 0.5

            # Parse missing elements
            missing_line = [l for l in llm_response.split('\n') if 'Missing:' in l]
            missing = missing_line[0].split(':')[1].strip() if missing_line else ""

            # Check expected elements if provided
            if expected_elements:
                found_elements = [e for e in expected_elements if e.lower() in response.lower()]
                element_score = len(found_elements) / len(expected_elements)
                score = (score + element_score) / 2

            suggestions = []
            if score < 0.7:
                suggestions.append("Response lacks completeness")
                if missing and missing != "None":
                    suggestions.append(f"Consider adding: {missing}")

            return CustomMetricResult(
                metric_type=MetricType.RESPONSE_COMPLETENESS,
                score=score,
                confidence=0.75,
                details={
                    "missing_elements": missing,
                    "expected_elements_found": found_elements if expected_elements else None
                },
                suggestions=suggestions
            )

        except Exception:
            logger.error("Failed to evaluate completeness; using heuristic fallback")
            # Fallback to simple heuristic
            word_count = len(response.split())
            score = min(1.0, word_count / 200)  # Assume 200 words is complete

            return CustomMetricResult(
                metric_type=MetricType.RESPONSE_COMPLETENESS,
                score=score,
                confidence=0.5,
                details={"word_count": word_count, "method": "heuristic"},
                suggestions=["Could not perform deep evaluation"]
            )

    async def evaluate_source_attribution(
        self,
        response: str,
        sources: list[dict[str, Any]],
        check_citations: bool = True
    ) -> CustomMetricResult:
        """
        Evaluate quality of source attribution in the response.

        Checks if claims are properly attributed to sources.
        """
        # Check for citation markers in response
        citation_patterns = [
            r'\[\d+\]',  # [1], [2], etc.
            r'\(.*?, \d{4}\)',  # (Author, Year)
            r'According to.*',
            r'.*states that.*',
            r'Source:.*'
        ]

        import re
        citations_found = 0
        for pattern in citation_patterns:
            citations_found += len(re.findall(pattern, response))

        # Check if sources are referenced
        sources_referenced = 0
        for source in sources:
            source_title = source.get('title', '')
            source_id = source.get('id', '')
            if source_title and source_title in response or source_id and source_id in response:
                sources_referenced += 1

        # Calculate attribution score
        if not sources:
            attribution_score = 1.0 if citations_found == 0 else 0.5
        else:
            reference_ratio = sources_referenced / len(sources)
            citation_ratio = min(1.0, citations_found / 3)  # Expect at least 3 citations
            attribution_score = (reference_ratio + citation_ratio) / 2

        suggestions = []
        if attribution_score < 0.5:
            suggestions.append("Add citations to support claims")
            suggestions.append("Reference source materials explicitly")
        elif attribution_score < 0.8:
            suggestions.append("Improve source attribution clarity")

        return CustomMetricResult(
            metric_type=MetricType.SOURCE_ATTRIBUTION,
            score=attribution_score,
            confidence=0.9,
            details={
                "citations_found": citations_found,
                "sources_referenced": sources_referenced,
                "total_sources": len(sources)
            },
            suggestions=suggestions
        )

    async def evaluate_response_coherence(
        self,
        response: str,
        check_structure: bool = True
    ) -> CustomMetricResult:
        """
        Evaluate the coherence and logical flow of the response.
        """
        # Check structural elements
        sentences = response.split('.')
        paragraphs = response.split('\n\n')

        structure_score = 0
        if len(sentences) > 1:
            structure_score += 0.25
        if len(paragraphs) > 1:
            structure_score += 0.25
        if any(word in response.lower() for word in ['first', 'second', 'finally', 'moreover']):
            structure_score += 0.25
        if response.count('\n-') > 0 or response.count('\n*') > 0:  # Bullet points
            structure_score += 0.25

        # Check sentence connectivity (simple heuristic)
        transition_words = [
            'however', 'therefore', 'moreover', 'furthermore',
            'additionally', 'consequently', 'thus', 'hence'
        ]
        transitions_used = sum(1 for word in transition_words if word in response.lower())
        transition_score = min(1.0, transitions_used / 3)

        coherence_score = (structure_score + transition_score) / 2

        suggestions = []
        if coherence_score < 0.5:
            suggestions.append("Improve response structure with clear paragraphs")
            suggestions.append("Add transition words for better flow")

        return CustomMetricResult(
            metric_type=MetricType.RESPONSE_COHERENCE,
            score=coherence_score,
            confidence=0.7,
            details={
                "sentence_count": len(sentences),
                "paragraph_count": len(paragraphs),
                "transitions_used": transitions_used
            },
            suggestions=suggestions
        )

    async def evaluate_cost_efficiency(
        self,
        tokens_used: int,
        response_quality: float,
        estimated_cost: float,
        baseline_cost: float = 0.01
    ) -> CustomMetricResult:
        """
        Evaluate cost efficiency of the RAG operation.
        """
        # Calculate efficiency ratio
        cost_ratio = baseline_cost / estimated_cost if estimated_cost > 0 else 1.0
        quality_adjusted_efficiency = cost_ratio * response_quality

        # Normalize to 0-1 scale
        efficiency_score = min(1.0, quality_adjusted_efficiency)

        # Token efficiency
        expected_tokens = 1000  # Baseline expectation
        token_efficiency = min(1.0, expected_tokens / tokens_used) if tokens_used > 0 else 0

        final_score = (efficiency_score + token_efficiency) / 2

        suggestions = []
        if final_score < 0.5:
            if tokens_used > expected_tokens * 1.5:
                suggestions.append("Reduce context size or use more efficient models")
            if estimated_cost > baseline_cost * 2:
                suggestions.append("Consider using cheaper models for this query type")

        return CustomMetricResult(
            metric_type=MetricType.COST_EFFICIENCY,
            score=final_score,
            confidence=0.95,
            details={
                "tokens_used": tokens_used,
                "estimated_cost": estimated_cost,
                "cost_ratio": cost_ratio,
                "token_efficiency": token_efficiency
            },
            suggestions=suggestions
        )

    async def evaluate_all_metrics(
        self,
        query: str,
        retrieved_contexts: list[str],
        response: str,
        sources: Optional[list[dict[str, Any]]] = None,
        tokens_used: Optional[int] = None,
        estimated_cost: Optional[float] = None,
        response_time_ms: Optional[float] = None
    ) -> dict[str, CustomMetricResult]:
        """
        Evaluate all applicable custom metrics.
        """
        results = {}

        # Retrieval metrics
        coverage_task = self.evaluate_retrieval_coverage(query, retrieved_contexts)
        diversity_task = self.evaluate_retrieval_diversity(retrieved_contexts,
                                                          [s.get('source') for s in sources] if sources else None)

        # Response metrics
        completeness_task = self.evaluate_response_completeness(query, response)
        coherence_task = self.evaluate_response_coherence(response)

        # Attribution metric
        attribution_task = None
        if sources:
            attribution_task = self.evaluate_source_attribution(response, sources)

        # Cost metric
        cost_task = None
        if tokens_used and estimated_cost:
            # Estimate quality from other metrics (simplified)
            cost_task = self.evaluate_cost_efficiency(tokens_used, 0.75, estimated_cost)

        # Gather all results
        tasks = [
            ("coverage", coverage_task),
            ("diversity", diversity_task),
            ("completeness", completeness_task),
            ("coherence", coherence_task)
        ]

        if attribution_task:
            tasks.append(("attribution", attribution_task))
        if cost_task:
            tasks.append(("cost", cost_task))

        # Execute all evaluations in parallel
        for name, task in tasks:
            try:
                result = await task
                results[name] = result
            except Exception:
                logger.error(f"Failed to evaluate {name}; skipping metric")

        # Add response time metric if provided
        if response_time_ms:
            time_score = 1.0 if response_time_ms < 1000 else 0.5 if response_time_ms < 5000 else 0.2
            results["response_time"] = CustomMetricResult(
                metric_type=MetricType.RESPONSE_TIME,
                score=time_score,
                confidence=1.0,
                details={"response_time_ms": response_time_ms},
                suggestions=["Optimize query processing"] if time_score < 0.5 else []
            )

        return results

    async def _extract_concepts(self, text: str) -> list[str]:
        """Extract key concepts from text."""
        # Simple implementation - extract noun phrases and important words
        # In production, use NLP libraries like spaCy
        import re

        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were'}

        # Extract words
        words = re.findall(r'\b\w+\b', text.lower())
        concepts = [w for w in words if w not in stop_words and len(w) > 3]

        # Return unique concepts
        return list(set(concepts))[:10]  # Limit to top 10

    def aggregate_metrics(self, metrics: dict[str, CustomMetricResult]) -> dict[str, Any]:
        """
        Aggregate multiple metrics into summary statistics.
        """
        if not metrics:
            return {}

        scores = [m.score for m in metrics.values()]
        confidences = [m.confidence for m in metrics.values()]

        # Calculate weighted average using confidence
        weighted_sum = sum(m.score * m.confidence for m in metrics.values())
        total_confidence = sum(confidences)
        weighted_average = weighted_sum / total_confidence if total_confidence > 0 else 0

        # Identify weak areas
        weak_metrics = [name for name, m in metrics.items() if m.score < 0.5]
        strong_metrics = [name for name, m in metrics.items() if m.score >= 0.8]

        # Compile all suggestions
        all_suggestions = []
        for m in metrics.values():
            all_suggestions.extend(m.suggestions)

        return {
            "overall_score": weighted_average,
            "average_confidence": statistics.mean(confidences),
            "min_score": min(scores),
            "max_score": max(scores),
            "weak_areas": weak_metrics,
            "strong_areas": strong_metrics,
            "suggestions": list(set(all_suggestions)),  # Unique suggestions
            "metric_count": len(metrics)
        }


# Singleton instance
_custom_metrics = None


class _LightRAGCustomMetrics(RAGCustomMetrics):
    """Lightweight metrics implementation for tests or when embeddings backend is unavailable.

    Uses lexical heuristics and avoids heavy model loading to keep tests fast and deterministic.
    """

    async def evaluate_retrieval_diversity(
        self,
        retrieved_contexts: list[str],
        sources: Optional[list[str]] = None
    ) -> CustomMetricResult:
        if not retrieved_contexts:
            return CustomMetricResult(
                metric_type=MetricType.RETRIEVAL_DIVERSITY,
                score=0.0,
                confidence=1.0,
                details={"reason": "No contexts retrieved", "method": "lexical"},
            )

        # Simple lexical Jaccard similarity over tokens instead of embeddings
        def tokenize(s: str):
            return {w for w in s.lower().split() if w}

        toks = [tokenize(c) for c in retrieved_contexts[:10]]
        similarities = []
        for i in range(len(toks)):
            for j in range(i + 1, len(toks)):
                a, b = toks[i], toks[j]
                union = a | b
                inter = a & b
                sim = (len(inter) / len(union)) if union else 0.0
                similarities.append(sim)
        avg_similarity = float(np.mean(similarities)) if similarities else 0.0
        diversity_score = max(0.0, 1.0 - avg_similarity)

        if sources:
            unique_sources = len(set(sources))
            total_sources = len(sources)
            diversity_score = min(1.0, diversity_score + ((unique_sources / total_sources) * 0.2 if total_sources else 0.0))

        return CustomMetricResult(
            metric_type=MetricType.RETRIEVAL_DIVERSITY,
            score=diversity_score,
            confidence=0.8,
            details={
                "average_similarity": avg_similarity,
                "total_contexts": len(retrieved_contexts),
                "method": "lexical",
            },
            suggestions=["Retrieved contexts are too similar"] if diversity_score < 0.5 else [],
        )


def get_custom_metrics() -> RAGCustomMetrics:
    """Get the global custom metrics instance.

    Returns a lightweight implementation during tests or when embeddings backend is unavailable,
    to avoid heavy model initialization and keep evaluations responsive.
    """
    global _custom_metrics
    if _custom_metrics is None:
        try:
            is_testing = bool(os.getenv("PYTEST_CURRENT_TEST")) or os.getenv("TESTING", "").lower() == "true"
        except Exception:
            is_testing = False

        disable_custom = os.getenv("TLDW_DISABLE_CUSTOM_METRICS", "").lower() in {"1", "true", "yes"}

        if is_testing or disable_custom or not _RAG_METRICS_EMBEDDINGS_AVAILABLE:
            _custom_metrics = _LightRAGCustomMetrics()
        else:
            _custom_metrics = RAGCustomMetrics()
    return _custom_metrics
