"""
anti_context_retriever.py - Anti-context retrieval for FVA pipeline.

This module implements the anti-context (counter-evidence) retrieval strategy
inspired by the FVA-RAG paper (arXiv:2512.07015). It generates negation and
contrary queries to actively seek evidence that might contradict a claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Claims_Extraction.claims_engine import Claim
    from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import (
        MultiDatabaseRetriever,
        RetrievalConfig,
    )

# Import Document and DataSource - matches existing pattern
try:
    from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document
except ImportError:
    from dataclasses import dataclass as _dc
    from enum import Enum as _Enum

    class DataSource(_Enum):  # type: ignore
        MEDIA_DB = "media_db"

    @_dc
    class Document:  # type: ignore
        id: str
        content: str
        metadata: dict[str, Any] = field(default_factory=dict)
        score: float = 0.0

# Import RetrievalConfig for type hints
try:
    from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import (
        RetrievalConfig,
    )
except ImportError:
    RetrievalConfig = None  # type: ignore


@dataclass
class AntiContextConfig:
    """Configuration for anti-context retrieval."""

    max_queries: int = 3
    max_docs_per_query: int = 5
    min_relevance_score: float = 0.3  # the retriever's min_score: trims each query's weakest full-text matches
    exclude_original_doc_ids: bool = True
    use_negation_templates: bool = True
    use_contrary_templates: bool = True
    max_docs_per_source: int = 2  # Diversity: limit docs from same source
    cache_ttl_seconds: int = 300  # Query result caching


@dataclass
class AntiContextResult:
    """Result from a single anti-context query."""

    query_used: str
    documents: list[Document]
    strategy: str  # negation | contrary


# Templates for generating negation queries
NEGATION_TEMPLATES = [
    "evidence against {claim}",
    "contradicts {claim}",
    "disproves {claim}",
    "counterarguments to {claim}",
    "studies refuting {claim}",
    "criticism of {claim}",
    "problems with {claim}",
    "exceptions to {claim}",
    "limitations of {claim}",
]

# Domain-specific contrary templates keyed by claim type value
CONTRARY_TEMPLATES: dict[str, list[str]] = {
    "statistic": [
        "different statistics for {topic}",
        "conflicting data on {topic}",
        "alternative measurements of {topic}",
    ],
    "causal": [
        "alternative causes of {effect}",
        "factors besides {cause} that affect {effect}",
        "no relationship between {cause} and {effect}",
    ],
    "comparative": [
        "{item_b} better than {item_a}",
        "advantages of {item_b} over {item_a}",
        "{item_a} and {item_b} are similar",
    ],
    "ranking": [
        "alternatives to {subject} ranking",
        "different rankings for {subject}",
        "{subject} is not the top",
    ],
    "temporal": [
        "different timeline for {event}",
        "alternative dates for {event}",
    ],
}


class AntiContextRetriever:
    """
    Retrieves documents that may contradict a given claim.

    Inspired by FVA-RAG's anti-context retrieval strategy, this class
    generates negation and contrary queries to find potential counter-evidence.
    """

    def __init__(
        self,
        retriever: MultiDatabaseRetriever,
        config: AntiContextConfig | None = None,
    ):
        """
        Initialize the anti-context retriever.

        Args:
            retriever: MultiDatabaseRetriever instance for searching
            config: Configuration options
        """
        self.retriever = retriever
        self.config = config or AntiContextConfig()
        self._query_cache: dict[str, list[Document]] = {}

    async def retrieve_anti_context(
        self,
        claim: Claim,
        original_doc_ids: set[str],
        user_id: str | None = None,
        sources: list[DataSource] | None = None,
    ) -> list[AntiContextResult]:
        """
        Generate negation/contrary queries and retrieve potential counter-evidence.

        Args:
            claim: The claim to find counter-evidence for
            original_doc_ids: Document IDs from original retrieval to exclude
            user_id: User ID for scoped retrieval (unused, for interface compat)
            sources: Data sources to search (defaults to all configured)

        Returns:
            List of AntiContextResult with retrieved documents
        """
        results: list[AntiContextResult] = []
        seen_doc_ids = set(original_doc_ids) if self.config.exclude_original_doc_ids else set()

        # Generate queries
        queries = self._generate_anti_queries(claim)

        for query, strategy in queries[: self.config.max_queries]:
            # Check cache
            cache_key = f"{query}:{sources}"
            if cache_key in self._query_cache:
                docs = self._query_cache[cache_key]
            else:
                # Use actual retriever interface (MultiDatabaseRetriever.retrieve)
                try:
                    # Build retrieval config
                    retrieval_config = self._build_retrieval_config()

                    docs = await self.retriever.retrieve(
                        query=query,
                        sources=sources,
                        config=retrieval_config,
                    )
                    self._query_cache[cache_key] = docs
                except Exception as e:
                    logger.warning(
                        f"Anti-context retrieval failed for query '{query[:50]}...': {e}"
                    )
                    continue

            # No score filter here. min_relevance_score goes to the retriever as min_score, which drops each query's
            # weakest full-text matches before hybrid fusion (scores are min-max normalised per query, so the best
            # match always passes). A fused (RRF) score is at most 1/61, so a floor on it dropped every document.
            # ponytail: no absolute relevance floor on either half; the adjudicator's stance check decides, and a
            # NEUTRAL document does not count. A similarity floor per vector store is the upgrade.

            # Filter already-seen documents
            new_docs = [d for d in docs if d.id not in seen_doc_ids]
            seen_doc_ids.update(d.id for d in new_docs)

            # Apply source diversity
            new_docs = self._diversify_by_source(new_docs)

            if new_docs:
                results.append(
                    AntiContextResult(
                        query_used=query,
                        documents=new_docs,
                        strategy=strategy,
                    )
                )

        return results

    def _build_retrieval_config(self) -> Any:
        """Build a RetrievalConfig for anti-context queries."""
        try:
            from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import (
                RetrievalConfig,
            )

            return RetrievalConfig(
                max_results=self.config.max_docs_per_query,
                min_score=self.config.min_relevance_score,
                use_fts=True,
                use_vector=True,
                include_metadata=True,
            )
        except ImportError:
            # Return None if RetrievalConfig not available
            return None

    def _generate_anti_queries(self, claim: Claim) -> list[tuple[str, str]]:
        """
        Generate queries designed to find contradicting evidence.

        Args:
            claim: The claim to generate anti-queries for

        Returns:
            List of (query, strategy) tuples
        """
        queries: list[tuple[str, str]] = []

        # Standard negation templates
        if self.config.use_negation_templates:
            for template in NEGATION_TEMPLATES[: 3]:
                query = template.format(claim=claim.text)
                queries.append((query, "negation"))

        # Claim-type specific contrary queries
        if self.config.use_contrary_templates:
            claim_type_key = claim.claim_type.value if claim.claim_type else None
            if claim_type_key and claim_type_key in CONTRARY_TEMPLATES:
                for template in CONTRARY_TEMPLATES[claim_type_key][: 2]:
                    filled = self._fill_contrary_template(template, claim)
                    if filled:
                        queries.append((filled, "contrary"))

        return queries

    def _fill_contrary_template(
        self, template: str, claim: Claim
    ) -> str | None:
        """
        Fill contrary template with extracted claim entities.

        Args:
            template: Template string with placeholders
            claim: The claim with extracted_values

        Returns:
            Filled template string or None if filling failed
        """
        # Use extracted_values from claim if available
        if claim.extracted_values:
            try:
                return template.format(
                    topic=claim.text[:50],
                    claim=claim.text,
                    subject=claim.text[:50],
                    event=claim.text[:50],
                    **claim.extracted_values,
                )
            except KeyError:
                pass

        # Fallback: use claim text for all placeholders
        try:
            return template.format(
                topic=claim.text[:50],
                claim=claim.text,
                subject=claim.text[:50],
                event=claim.text[:50],
                cause=claim.text[:30],
                effect=claim.text[:30],
                item_a=claim.text[:30],
                item_b="alternatives",
            )
        except KeyError:
            # Template has unknown placeholders
            return None

    def _diversify_by_source(self, docs: list[Document]) -> list[Document]:
        """
        Ensure diversity by limiting docs per source.

        This prevents a single document from dominating the anti-context,
        encouraging a broader view of potential counter-evidence.

        Args:
            docs: List of documents to diversify

        Returns:
            Diversified list with max docs per source
        """
        source_counts: dict[str, int] = {}
        diverse_docs: list[Document] = []

        # Sort by score to keep the best docs from each source
        for doc in sorted(docs, key=lambda d: d.score, reverse=True):
            # Use media_id from metadata if available, otherwise use doc id
            source_id = doc.metadata.get("media_id", doc.id)
            if isinstance(source_id, int):
                source_id = str(source_id)

            current_count = source_counts.get(source_id, 0)
            if current_count < self.config.max_docs_per_source:
                diverse_docs.append(doc)
                source_counts[source_id] = current_count + 1

        return diverse_docs

    def clear_cache(self) -> None:
        """Clear the query result cache."""
        self._query_cache.clear()

    def get_cache_size(self) -> int:
        """Get the number of cached query results."""
        return len(self._query_cache)
