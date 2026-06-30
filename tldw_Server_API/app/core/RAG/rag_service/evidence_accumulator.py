"""
Progressive Evidence Accumulator for RAG queries.

This module provides iterative evidence gathering with multiple retrieval
rounds until evidence is sufficient or budget is exhausted.

Design:
- Wraps existing retrieval with iterative refinement
- Max 3 rounds (configurable) to bound latency
- LLM assesses evidence gaps and generates follow-up queries
- Deduplicates and merges results across rounds
"""

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger

from .types import Document


@dataclass
class AccumulationRound:
    """Results from a single accumulation round."""
    round_number: int
    query: str
    documents: list[Document]
    duration_sec: float
    gap_queries: list[str] = field(default_factory=list)


@dataclass
class AccumulationResult:
    """Complete result from evidence accumulation."""
    query: str
    documents: list[Document]
    rounds: list[AccumulationRound]
    total_rounds: int
    is_sufficient: bool
    sufficiency_reason: str
    total_duration_sec: float
    metadata: dict[str, Any] = field(default_factory=dict)


# Prompts for gap assessment
GAP_ASSESSMENT_PROMPT = """You are assessing whether the retrieved evidence is sufficient to answer the query.

Query: {query}

Retrieved Evidence:
{evidence}

Instructions:
1. Determine if the evidence is SUFFICIENT or INSUFFICIENT to answer the query.
2. If INSUFFICIENT, identify 1-3 specific gaps in the evidence.
3. For each gap, generate a focused follow-up query to fill that gap.

Respond in this exact format:
STATUS: [SUFFICIENT or INSUFFICIENT]
REASON: [Brief explanation]
GAP_QUERIES:
- [Follow-up query 1]
- [Follow-up query 2]
- [Follow-up query 3]

If SUFFICIENT, omit the GAP_QUERIES section.
"""


def _compute_document_hash(doc: Document) -> str:
    """Compute a hash for document deduplication."""
    content = doc.content or ""
    doc_id = doc.id or ""
    return hashlib.sha256(f"{doc_id}:{content[:500]}".encode()).hexdigest()[:16]


def _parse_gap_assessment(response: str) -> tuple[bool, str, list[str]]:
    """
    Parse the LLM's gap assessment response.

    Returns:
        Tuple of (is_sufficient, reason, gap_queries)
    """
    is_sufficient = False
    reason = "Unable to parse response"
    gap_queries = []

    if not response:
        return False, "Empty response", []

    response = response.strip()

    # Extract status
    status_match = re.search(r"STATUS:\s*(SUFFICIENT|INSUFFICIENT)", response, re.IGNORECASE)
    if status_match:
        is_sufficient = status_match.group(1).upper() == "SUFFICIENT"

    # Extract reason
    reason_match = re.search(r"REASON:\s*(.+?)(?=GAP_QUERIES:|$)", response, re.IGNORECASE | re.DOTALL)
    if reason_match:
        reason = reason_match.group(1).strip()

    # Extract gap queries
    gap_section = re.search(r"GAP_QUERIES:\s*(.+)", response, re.IGNORECASE | re.DOTALL)
    if gap_section:
        gap_text = gap_section.group(1)
        # Parse bullet points
        queries = re.findall(r"[-*]\s*(.+?)(?=\n[-*]|\n\n|$)", gap_text, re.DOTALL)
        gap_queries = [q.strip() for q in queries if q.strip()]

    return is_sufficient, reason, gap_queries


class EvidenceAccumulator:
    """
    Iteratively accumulates evidence through multiple retrieval rounds.

    Uses LLM to assess evidence sufficiency and generate follow-up queries
    to fill gaps. Deduplicates results across rounds.
    """

    def __init__(
        self,
        max_rounds: int = 3,
        min_docs_per_round: int = 3,
        max_docs_total: int = 20,
        sufficiency_threshold: float = 0.8,
        enable_gap_assessment: bool = True,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
    ):
        """
        Initialize the evidence accumulator.

        Args:
            max_rounds: Maximum number of retrieval rounds
            min_docs_per_round: Minimum documents needed per round to continue
            max_docs_total: Maximum total documents to accumulate
            sufficiency_threshold: Score threshold for considering evidence sufficient
            enable_gap_assessment: Use LLM for gap assessment
            llm_provider: LLM provider for gap assessment
            llm_model: LLM model for gap assessment
        """
        self.max_rounds = max(1, min(max_rounds, 5))  # Cap at 5 rounds
        self.min_docs_per_round = min_docs_per_round
        self.max_docs_total = max_docs_total
        self.sufficiency_threshold = sufficiency_threshold
        self.enable_gap_assessment = enable_gap_assessment
        self.llm_provider = llm_provider
        self.llm_model = llm_model

    async def accumulate(
        self,
        query: str,
        initial_results: list[Document],
        retrieval_fn,
        time_budget_sec: Optional[float] = None,
    ) -> AccumulationResult:
        """
        Accumulate evidence through iterative retrieval.

        Args:
            query: The original search query
            initial_results: Documents from initial retrieval
            retrieval_fn: Async function to call for additional retrieval
                         Signature: async (query: str, exclude_ids: Set[str]) -> List[Document]
            time_budget_sec: Optional time budget for accumulation

        Returns:
            AccumulationResult with all accumulated evidence
        """
        start_time = time.time()
        deadline = start_time + time_budget_sec if time_budget_sec else None

        # Track all documents and their hashes
        all_documents: list[Document] = []
        seen_hashes: set[str] = set()
        rounds: list[AccumulationRound] = []

        # Add initial results
        for doc in initial_results:
            doc_hash = _compute_document_hash(doc)
            if doc_hash not in seen_hashes:
                seen_hashes.add(doc_hash)
                all_documents.append(doc)

        rounds.append(AccumulationRound(
            round_number=1,
            query=query,
            documents=initial_results,
            duration_sec=0.0,
            gap_queries=[],
        ))

        # Check if we already have sufficient evidence
        is_sufficient, reason, gap_queries = await self._assess_evidence(
            query, all_documents
        )

        if is_sufficient or len(all_documents) >= self.max_docs_total:
            return AccumulationResult(
                query=query,
                documents=all_documents,
                rounds=rounds,
                total_rounds=1,
                is_sufficient=is_sufficient,
                sufficiency_reason=reason,
                total_duration_sec=time.time() - start_time,
                metadata={"initial_docs": len(initial_results)},
            )

        # Iterative accumulation
        current_round = 1
        while current_round < self.max_rounds:
            # Check time budget
            if deadline and time.time() >= deadline:
                logger.debug("Evidence accumulation stopped: time budget exhausted")
                break

            # Check document budget
            if len(all_documents) >= self.max_docs_total:
                logger.debug("Evidence accumulation stopped: max docs reached")
                break

            current_round += 1
            round_start = time.time()

            # Use gap queries or variations of original query
            queries_to_try = gap_queries if gap_queries else [self._generate_variation(query, current_round)]

            round_documents: list[Document] = []
            exclude_ids = {doc.id for doc in all_documents}

            for gap_query in queries_to_try[:3]:  # Max 3 gap queries per round
                try:
                    new_docs = await retrieval_fn(gap_query, exclude_ids)
                    for doc in new_docs:
                        doc_hash = _compute_document_hash(doc)
                        if doc_hash not in seen_hashes:
                            seen_hashes.add(doc_hash)
                            all_documents.append(doc)
                            round_documents.append(doc)
                            exclude_ids.add(doc.id)

                        if len(all_documents) >= self.max_docs_total:
                            break
                except Exception:
                    logger.warning("Retrieval error during evidence accumulation round")
                    continue

                if len(all_documents) >= self.max_docs_total:
                    break

            round_duration = time.time() - round_start
            rounds.append(AccumulationRound(
                round_number=current_round,
                query=queries_to_try[0] if queries_to_try else query,
                documents=round_documents,
                duration_sec=round_duration,
                gap_queries=gap_queries,
            ))

            # Check if we found enough new documents
            if len(round_documents) < self.min_docs_per_round:
                logger.debug(f"Evidence accumulation stopped: insufficient new docs in round {current_round}")
                break

            # Re-assess evidence
            is_sufficient, reason, gap_queries = await self._assess_evidence(
                query, all_documents
            )

            if is_sufficient:
                break

        # Final result
        total_duration = time.time() - start_time
        return AccumulationResult(
            query=query,
            documents=all_documents,
            rounds=rounds,
            total_rounds=len(rounds),
            is_sufficient=is_sufficient,
            sufficiency_reason=reason,
            total_duration_sec=total_duration,
            metadata={
                "initial_docs": len(initial_results),
                "final_docs": len(all_documents),
                "docs_added": len(all_documents) - len(initial_results),
            },
        )

    async def _assess_evidence(
        self,
        query: str,
        documents: list[Document],
    ) -> tuple[bool, str, list[str]]:
        """
        Assess whether the accumulated evidence is sufficient.

        Returns:
            Tuple of (is_sufficient, reason, gap_queries)
        """
        if not documents:
            return False, "No documents retrieved", [query]

        # Heuristic assessment if LLM is disabled
        if not self.enable_gap_assessment:
            return self._heuristic_assessment(query, documents)

        # Build evidence summary for LLM
        evidence_summary = self._build_evidence_summary(documents)

        try:
            response = await self._call_llm_for_assessment(query, evidence_summary)
            return _parse_gap_assessment(response)
        except Exception:
            logger.warning("LLM gap assessment failed, using heuristic")
            return self._heuristic_assessment(query, documents)

    def _heuristic_assessment(
        self,
        query: str,
        documents: list[Document],
    ) -> tuple[bool, str, list[str]]:
        """
        Rule-based assessment of evidence sufficiency.

        Returns:
            Tuple of (is_sufficient, reason, gap_queries)
        """
        # Extract query terms
        query_terms = set(re.findall(r"\b\w{3,}\b", query.lower()))

        if not query_terms:
            return True, "No specific terms to match", []

        # Check coverage
        matched_terms = set()
        for doc in documents:
            content_lower = (doc.content or "").lower()
            for term in query_terms:
                if term in content_lower:
                    matched_terms.add(term)

        coverage = len(matched_terms) / len(query_terms) if query_terms else 1.0

        # Consider average document score
        avg_score = sum(doc.score for doc in documents) / len(documents) if documents else 0.0

        # Combine signals
        is_sufficient = (
            coverage >= self.sufficiency_threshold
            and avg_score >= 0.3
            and len(documents) >= 3
        )

        if is_sufficient:
            return True, f"Coverage: {coverage:.0%}, Avg score: {avg_score:.2f}", []

        # Generate gap queries for missing terms
        missing_terms = query_terms - matched_terms
        gap_queries = []
        if missing_terms:
            # Focus follow-up queries on missing terms
            gap_queries.append(f"{query} {' '.join(list(missing_terms)[:3])}")

        return False, f"Coverage: {coverage:.0%}, missing: {missing_terms}", gap_queries

    def _build_evidence_summary(self, documents: list[Document], max_chars: int = 2000) -> str:
        """Build a summary of the evidence for LLM assessment."""
        summary_parts = []
        remaining = max_chars

        for i, doc in enumerate(documents[:10]):  # Max 10 docs
            title = (doc.metadata or {}).get("title", f"Document {i+1}")
            snippet = (doc.content or "")[:200]
            entry = f"[{i+1}] {title}: {snippet}..."

            if len(entry) > remaining:
                break

            summary_parts.append(entry)
            remaining -= len(entry)

        return "\n\n".join(summary_parts)

    async def _call_llm_for_assessment(self, query: str, evidence: str) -> str:
        """Call LLM to assess evidence sufficiency."""
        try:
            from .generation import AnswerGenerator

            generator = AnswerGenerator(
                provider=self.llm_provider,
                model=self.llm_model,
            )

            prompt = GAP_ASSESSMENT_PROMPT.format(query=query, evidence=evidence)

            result = await generator.generate(
                query=query,
                context=evidence,
                prompt_template=prompt,
                max_tokens=300,
            )

            if isinstance(result, dict):
                return str(result.get("answer", ""))
            return str(result)

        except Exception:
            logger.warning("LLM assessment call failed")
            raise

    def _generate_variation(self, query: str, round_num: int) -> str:
        """Generate a query variation for additional retrieval."""
        variations = [
            f"{query} details",
            f"more about {query}",
            f"{query} examples",
            f"{query} explanation",
        ]
        idx = (round_num - 1) % len(variations)
        return variations[idx]

    def assess_evidence_gaps(
        self,
        query: str,
        evidence: list[Document],
    ) -> list[str]:
        """
        Synchronous method to identify gaps in evidence.

        Args:
            query: The original query
            evidence: Current evidence documents

        Returns:
            List of follow-up queries to fill gaps
        """
        _, _, gap_queries = self._heuristic_assessment(query, evidence)
        return gap_queries

    def merge_results(self, rounds: list[list[Document]]) -> list[Document]:
        """
        Merge and deduplicate results from multiple rounds.

        Args:
            rounds: List of document lists from each round

        Returns:
            Deduplicated list of documents, sorted by score
        """
        seen_hashes: set[str] = set()
        merged: list[Document] = []

        for round_docs in rounds:
            for doc in round_docs:
                doc_hash = _compute_document_hash(doc)
                if doc_hash not in seen_hashes:
                    seen_hashes.add(doc_hash)
                    merged.append(doc)

        # Sort by score descending
        merged.sort(key=lambda d: d.score, reverse=True)
        return merged


# Module-level convenience function
async def accumulate_evidence(
    query: str,
    initial_results: list[Document],
    retrieval_fn,
    max_rounds: int = 3,
    time_budget_sec: Optional[float] = None,
) -> AccumulationResult:
    """
    Convenience function for evidence accumulation.

    Args:
        query: The search query
        initial_results: Documents from initial retrieval
        retrieval_fn: Async function for additional retrieval
        max_rounds: Maximum number of rounds
        time_budget_sec: Optional time budget

    Returns:
        AccumulationResult with all evidence
    """
    accumulator = EvidenceAccumulator(max_rounds=max_rounds)
    return await accumulator.accumulate(
        query=query,
        initial_results=initial_results,
        retrieval_fn=retrieval_fn,
        time_budget_sec=time_budget_sec,
    )
