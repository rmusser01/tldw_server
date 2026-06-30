"""
Query expansion strategies for improved retrieval.

This module provides various query expansion techniques to improve
search recall and precision, including:
- Synonym expansion
- Multi-query generation
- Acronym expansion and contraction
- Entity recognition and expansion
- Domain-specific term expansion
- Semantic similarity-based expansion
"""

import asyncio
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from loguru import logger

from .types import DataSource, RetrieverStrategy, SearchResult


class ExpansionStrategy(Enum):
    """Available query expansion strategies."""
    SYNONYM = "synonym"           # Basic synonym expansion
    MULTI_QUERY = "multi_query"   # Generate query variations
    SEMANTIC = "semantic"         # Use embeddings for semantic similarity
    LINGUISTIC = "linguistic"     # Advanced linguistic processing
    ENTITY = "entity"            # Named entity recognition
    ACRONYM = "acronym"          # Expand/contract acronyms
    DOMAIN = "domain"            # Domain-specific expansions
    HYBRID = "hybrid"            # Combine multiple strategies


@dataclass(frozen=True)
class ExpandedQuery:
    """Represents an expanded query with variations (immutable after creation)."""
    original_query: str
    variations: list[str]
    synonyms: dict[str, list[str]]
    keywords: list[str]
    entities: list[str]
    acronym_expansions: dict[str, list[str]] = field(default_factory=dict)
    domain_terms: dict[str, list[str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class QueryExpansionStrategy(ABC):
    """Base class for query expansion strategies."""

    @abstractmethod
    async def expand(self, query: str) -> ExpandedQuery:
        """
        Expand a query into variations.

        Args:
            query: Original query string

        Returns:
            ExpandedQuery with variations
        """
        pass


class SynonymExpansion(QueryExpansionStrategy):
    """
    Expands queries using synonyms and related terms.

    This implementation uses a simple dictionary for demonstration.
    In production, you'd use WordNet, ConceptNet, or a custom thesaurus.
    """

    def __init__(self, synonym_dict: Optional[dict[str, list[str]]] = None):
        """
        Initialize synonym expansion.

        Args:
            synonym_dict: Dictionary mapping words to synonyms
        """
        self.synonym_dict = synonym_dict or self._get_default_synonyms()

    def _get_default_synonyms(self) -> dict[str, list[str]]:
        """Get default synonym dictionary for common terms."""
        return {
            # Technical terms
            "search": ["find", "query", "lookup", "retrieve"],
            "document": ["file", "text", "content", "record"],
            "database": ["db", "storage", "repository", "datastore"],
            "create": ["make", "generate", "build", "construct"],
            "delete": ["remove", "erase", "clear", "purge"],
            "update": ["modify", "change", "edit", "revise"],
            "error": ["bug", "issue", "problem", "fault"],
            "fix": ["repair", "resolve", "patch", "correct"],

            # Machine learning terms
            "model": ["algorithm", "network", "system"],
            "train": ["fit", "learn", "optimize"],
            "predict": ["forecast", "estimate", "infer"],
            "accuracy": ["precision", "performance", "score"],

            # General terms
            "fast": ["quick", "rapid", "speedy", "swift"],
            "slow": ["sluggish", "delayed", "lagging"],
            "large": ["big", "huge", "massive", "extensive"],
            "small": ["tiny", "minor", "compact", "minimal"],
            "good": ["excellent", "great", "positive", "beneficial"],
            "bad": ["poor", "negative", "problematic", "inferior"],
        }

    async def expand(self, query: str) -> ExpandedQuery:
        """
        Expand query using synonyms.

        Args:
            query: Original query string

        Returns:
            ExpandedQuery with synonym variations
        """
        words = query.lower().split()
        found_synonyms = {}
        variations = set()

        # Find synonyms for each word
        for word in words:
            if word in self.synonym_dict:
                found_synonyms[word] = self.synonym_dict[word]

                # Create variations by replacing the word with each synonym
                for synonym in self.synonym_dict[word]:
                    variation = query.lower().replace(word, synonym)
                    if variation != query.lower():
                        variations.add(variation)

        # Also create a variation with all synonyms combined
        if found_synonyms:
            expanded_words = []
            for word in words:
                if word in found_synonyms:
                    # Add original and first synonym
                    expanded_words.append(f"({word} OR {found_synonyms[word][0]})")
                else:
                    expanded_words.append(word)
            variations.add(" ".join(expanded_words))

        return ExpandedQuery(
            original_query=query,
            variations=list(variations)[:5],  # Limit variations
            synonyms=found_synonyms,
            keywords=words,
            entities=[],
            metadata={"strategy": "synonym_expansion"}
        )


class MultiQueryGeneration(QueryExpansionStrategy):
    """
    Generates multiple query variations using different perspectives.

    This simulates what an LLM might do to rephrase queries.
    """

    def __init__(self, generation_templates: Optional[list[str]] = None):
        """
        Initialize multi-query generation.

        Args:
            generation_templates: Templates for query generation
        """
        self.templates = generation_templates or self._get_default_templates()

    def _get_default_templates(self) -> list[str]:
        """Get default query generation templates."""
        return [
            "What is {query}",
            "Explain {query}",
            "How to {query}",
            "Definition of {query}",
            "Examples of {query}",
            "{query} tutorial",
            "{query} documentation",
            "Best practices for {query}",
            "Common issues with {query}",
            "Troubleshooting {query}"
        ]

    async def expand(self, query: str) -> ExpandedQuery:
        """
        Generate multiple query variations.

        Args:
            query: Original query string

        Returns:
            ExpandedQuery with generated variations
        """
        variations = []

        # Apply templates that make sense for the query
        query_lower = query.lower()

        # Determine query type
        is_question = any(query_lower.startswith(q) for q in ["what", "how", "why", "when", "where", "who"])
        is_action = any(word in query_lower for word in ["create", "make", "build", "fix", "solve", "implement"])

        if is_question:
            # For questions, rephrase differently
            variations.append(query.replace("?", ""))
            if "what is" in query_lower:
                variations.append(query_lower.replace("what is", "define"))
            if "how to" in query_lower:
                variations.append(query_lower.replace("how to", "steps to"))
        elif is_action:
            # For action queries, add context
            variations.append(f"tutorial {query}")
            variations.append(f"guide to {query}")
            variations.append(f"best way to {query}")
        else:
            # For general queries, use templates
            for template in self.templates[:3]:
                if "{query}" in template:
                    variations.append(template.format(query=query))

        # Extract key phrases
        keywords = self._extract_key_phrases(query)

        return ExpandedQuery(
            original_query=query,
            variations=variations[:5],  # Limit variations
            synonyms={},
            keywords=keywords,
            entities=self._extract_entities(query),
            metadata={"strategy": "multi_query_generation"}
        )

    def _extract_key_phrases(self, query: str) -> list[str]:
        """Extract key phrases from query."""
        # Simple implementation - in production, use NLP
        phrases = []
        words = query.split()

        # Extract 2-grams and 3-grams
        for i in range(len(words) - 1):
            phrases.append(f"{words[i]} {words[i+1]}")
        for i in range(len(words) - 2):
            phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")

        return phrases[:5]

    def _extract_entities(self, query: str) -> list[str]:
        """Extract named entities from query."""
        # Simple pattern matching - in production, use NER
        entities = []

        # Look for capitalized words (potential entities)
        words = query.split()
        for word in words:
            if word[0].isupper() and word.lower() not in ["what", "how", "why", "when", "where", "who"]:
                entities.append(word)

        return entities


class AcronymExpansion(QueryExpansionStrategy):
    """
    Expands or contracts acronyms in queries.
    """

    # Common acronyms and their expansions
    ACRONYM_DB = {
        "ai": ["artificial intelligence"],
        "ml": ["machine learning"],
        "nlp": ["natural language processing"],
        "llm": ["large language model"],
        "rag": ["retrieval augmented generation"],
        "api": ["application programming interface"],
        "db": ["database"],
        "ui": ["user interface"],
        "ux": ["user experience"],
        "cli": ["command line interface"],
        "gpu": ["graphics processing unit"],
        "cpu": ["central processing unit"],
        "ram": ["random access memory"],
        "ssd": ["solid state drive"],
        "hdd": ["hard disk drive"],
        "os": ["operating system"],
        "vm": ["virtual machine"],
        "ci": ["continuous integration"],
        "cd": ["continuous deployment", "continuous delivery"],
        "sdk": ["software development kit"],
        "ide": ["integrated development environment"],
        "json": ["javascript object notation"],
        "xml": ["extensible markup language"],
        "html": ["hypertext markup language"],
        "css": ["cascading style sheets"],
        "sql": ["structured query language"],
        "http": ["hypertext transfer protocol"],
        "https": ["hypertext transfer protocol secure"],
        "rest": ["representational state transfer"],
        "crud": ["create read update delete"],
        "mvc": ["model view controller"],
        "mvp": ["model view presenter", "minimum viable product"],
        "tdd": ["test driven development"],
        "bdd": ["behavior driven development"],
        "poc": ["proof of concept"],
        "roi": ["return on investment"],
        "kpi": ["key performance indicator"],
        "saas": ["software as a service"],
        "paas": ["platform as a service"],
        "iaas": ["infrastructure as a service"],
    }

    def __init__(self, custom_acronyms: Optional[dict[str, list[str]]] = None):
        """
        Initialize acronym expansion.

        Args:
            custom_acronyms: Additional acronyms to add
        """
        self.acronyms = self.ACRONYM_DB.copy()
        if custom_acronyms:
            self.acronyms.update(custom_acronyms)

        # Build reverse mapping for contraction
        self.reverse_acronyms = {}
        for acronym, expansions in self.acronyms.items():
            for expansion in expansions:
                self.reverse_acronyms[expansion] = acronym

    async def expand(self, query: str) -> ExpandedQuery:
        """
        Expand or contract acronyms in the query.

        Args:
            query: Original query string

        Returns:
            ExpandedQuery with acronym variations
        """
        variations = []
        acronym_expansions = {}
        words = query.lower().split()

        # Check each word for acronym expansion
        for i, word in enumerate(words):
            if word in self.acronyms:
                # Expand acronym
                acronym_expansions[word] = self.acronyms[word]
                for expansion in self.acronyms[word]:
                    new_words = words.copy()
                    new_words[i] = expansion
                    variations.append(" ".join(new_words))

        # Check for contractable phrases
        query_lower = query.lower()
        for full_form, acronym in self.reverse_acronyms.items():
            if full_form in query_lower:
                contracted = query_lower.replace(full_form, acronym)
                if contracted != query_lower:
                    variations.append(contracted)
                    if acronym not in acronym_expansions:
                        acronym_expansions[acronym] = [full_form]

        return ExpandedQuery(
            original_query=query,
            variations=variations[:5],
            synonyms={},
            keywords=words,
            entities=[],
            acronym_expansions=acronym_expansions,
            metadata={"strategy": "acronym_expansion"}
        )


class DomainExpansion(QueryExpansionStrategy):
    """
    Expands queries using domain-specific vocabulary.
    """

    # Domain-specific term relationships
    DOMAIN_TERMS = {
        "transcription": ["speech to text", "audio to text", "voice recognition"],
        "summarization": ["summary", "abstract", "synopsis", "digest"],
        "embedding": ["vector", "representation", "encoding", "feature vector"],
        "chunking": ["segmentation", "splitting", "partitioning"],
        "retrieval": ["search", "lookup", "query", "fetch"],
        "generation": ["creation", "synthesis", "production"],
        "model": ["algorithm", "network", "architecture"],
        "training": ["learning", "fitting", "optimization"],
        "inference": ["prediction", "generation", "output"],
        "context": ["background", "setting", "environment"],
        "prompt": ["input", "query", "instruction"],
        "token": ["word", "subword", "piece"],
        "attention": ["focus", "weight", "importance"],
        "transformer": ["model", "architecture", "network"],
        "fine-tuning": ["adaptation", "customization", "specialization"],
    }

    def __init__(self, custom_terms: Optional[dict[str, list[str]]] = None):
        """
        Initialize domain expansion.

        Args:
            custom_terms: Additional domain-specific terms
        """
        self.domain_terms = self.DOMAIN_TERMS.copy()
        if custom_terms:
            self.domain_terms.update(custom_terms)

    async def expand(self, query: str) -> ExpandedQuery:
        """
        Expand query using domain-specific terms.

        Args:
            query: Original query string

        Returns:
            ExpandedQuery with domain variations
        """
        variations = []
        found_domain_terms = {}
        query_lower = query.lower()

        # Check for domain terms in query
        for term, related_terms in self.domain_terms.items():
            if term in query_lower:
                found_domain_terms[term] = related_terms
                for related in related_terms[:2]:  # Limit related terms
                    expanded = query_lower.replace(term, related)
                    if expanded != query_lower:
                        variations.append(expanded)

        return ExpandedQuery(
            original_query=query,
            variations=variations[:5],
            synonyms={},
            keywords=query_lower.split(),
            entities=[],
            domain_terms=found_domain_terms,
            metadata={"strategy": "domain_expansion"}
        )


class EntityExpansion(QueryExpansionStrategy):
    """
    Expands queries based on recognized entities.
    """

    def __init__(self):
        """Initialize entity expansion."""
        # Entity patterns for recognition
        self.patterns = {
            "date": r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
            "time": r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "url": r'https?://[^\s]+',
            "number": r'\b\d+\.?\d*\b',
            "version": r'\bv?\d+\.\d+(?:\.\d+)?\b',
        }

    async def expand(self, query: str) -> ExpandedQuery:
        """
        Expand query based on recognized entities.

        Args:
            query: Original query string

        Returns:
            ExpandedQuery with entity-based variations
        """
        variations = []
        entities = []

        # Check for entities and create targeted expansions
        for entity_type, pattern in self.patterns.items():
            matches = re.findall(pattern, query)
            if matches:
                entities.extend(matches)

                if entity_type == "date":
                    variations.append(query + " events")
                    variations.append(query + " timeline")
                elif entity_type == "version":
                    variations.append(query + " changelog")
                    variations.append(query + " release notes")
                elif entity_type == "email":
                    variations.append(query + " sender")
                    variations.append(query + " conversation")

        # Look for capitalized words (potential named entities)
        words = query.split()
        for word in words:
            if len(word) > 2 and word[0].isupper() and word not in ["What", "How", "Why", "When", "Where", "Who"]:
                entities.append(word)

        return ExpandedQuery(
            original_query=query,
            variations=variations[:5],
            synonyms={},
            keywords=query.split(),
            entities=entities,
            metadata={"strategy": "entity_expansion"}
        )


# --- Compatibility helper functions for tests ---
async def expand_acronyms(query: str) -> str:
    """
    Expand acronyms in a query and return a simple expanded string.

    Provided for test compatibility where a function-level helper
    `expand_acronyms` is patched. Uses AcronymExpansion under the hood.
    """
    expander = AcronymExpansion()
    expanded = await expander.expand(query)
    # Return a basic string: original + first variation (if any)
    if expanded.variations:
        return f"{query} {expanded.variations[0]}"
    return query


async def expand_synonyms(query: str) -> list[str]:
    """Return a small list of synonym-based variations for compatibility."""
    expander = SynonymExpansion()
    expanded = await expander.expand(query)
    return expanded.variations or []


async def domain_specific_expansion(query: str) -> list[str]:
    """Return domain-specific variations for compatibility."""
    expander = DomainExpansion()
    expanded = await expander.expand(query)
    return expanded.variations or []


async def entity_recognition_expansion(query: str) -> list[str]:
    """Return entity-based variations for compatibility."""
    expander = EntityExpansion()
    expanded = await expander.expand(query)
    return expanded.variations or []


async def multi_strategy_expansion(query: str, strategies: Optional[list[str]] = None, corpus: Optional[str] = None) -> str:
    """Combined quick expansion used by unified pipeline tests.

    Returns a simple string that concatenates the first available expansion.
    """
    strategies = strategies or ["acronym", "synonym"]
    first_variation: Optional[str] = None

    if "acronym" in strategies:
        try:
            ac = await expand_acronyms(query)
            if isinstance(ac, str) and ac.strip() and ac != query:
                first_variation = ac.replace(query, "").strip()
        except Exception:
            logger.debug("Acronym expansion failed; continuing to next strategy")
    if not first_variation and "synonym" in strategies:
        try:
            # Load corpus-level synonyms if provided
            if corpus:
                try:
                    from .synonyms_registry import get_corpus_synonyms  # lazy import
                    synmap = get_corpus_synonyms(corpus)
                except Exception:
                    synmap = {}
                syn_expander = SynonymExpansion(synmap if synmap else None)
                expanded = await syn_expander.expand(query)
                syns = expanded.variations or []
            else:
                syns = await expand_synonyms(query)
            if syns:
                first_variation = syns[0]
        except Exception:
            logger.debug("Synonym expansion failed; continuing to next strategy")
    if not first_variation and "domain" in strategies:
        try:
            if corpus:
                try:
                    from .synonyms_registry import get_corpus_synonyms
                    dom_terms = get_corpus_synonyms(corpus)
                except Exception:
                    dom_terms = {}
                dom_expander = DomainExpansion(custom_terms=dom_terms if dom_terms else None)
                expanded = await dom_expander.expand(query)
                dom = expanded.variations or []
            else:
                dom = await domain_specific_expansion(query)
            if dom:
                first_variation = dom[0]
        except Exception:
            logger.debug("Domain expansion failed; continuing")

    return f"{query} {first_variation}".strip() if first_variation else query


class HybridQueryExpansion(QueryExpansionStrategy):
    """
    Combines multiple expansion strategies for comprehensive query expansion.
    """

    def __init__(self, strategies: Optional[list[QueryExpansionStrategy]] = None):
        """
        Initialize hybrid expansion.

        Args:
            strategies: List of expansion strategies to combine
        """
        self.strategies = strategies or [
            SynonymExpansion(),
            MultiQueryGeneration(),
            AcronymExpansion(),
            DomainExpansion(),
            EntityExpansion()
        ]

    async def expand(self, query: str) -> ExpandedQuery:
        """
        Expand query using multiple strategies.

        Args:
            query: Original query string

        Returns:
            Combined ExpandedQuery from all strategies
        """
        # Run all strategies in parallel
        expansion_tasks = [strategy.expand(query) for strategy in self.strategies]
        expansions = await asyncio.gather(*expansion_tasks)

        # Combine results
        all_variations = []
        all_synonyms = {}
        all_keywords = []
        all_entities = []

        for expansion in expansions:
            all_variations.extend(expansion.variations)
            all_synonyms.update(expansion.synonyms)
            all_keywords.extend(expansion.keywords)
            all_entities.extend(expansion.entities)

        # Deduplicate while preserving order
        seen = set()
        unique_variations = []
        for v in all_variations:
            if v not in seen:
                seen.add(v)
                unique_variations.append(v)

        # Merge additional fields
        all_acronyms = {}
        all_domain_terms = {}

        for expansion in expansions:
            if hasattr(expansion, 'acronym_expansions'):
                all_acronyms.update(expansion.acronym_expansions)
            if hasattr(expansion, 'domain_terms'):
                all_domain_terms.update(expansion.domain_terms)

        return ExpandedQuery(
            original_query=query,
            variations=unique_variations[:10],  # Limit total variations
            synonyms=all_synonyms,
            keywords=list(set(all_keywords))[:20],
            entities=list(set(all_entities)),
            acronym_expansions=all_acronyms,
            domain_terms=all_domain_terms,
            metadata={
                "strategy": "hybrid",
                "strategies_used": [type(s).__name__ for s in self.strategies]
            }
        )


class QueryExpansionRetriever(RetrieverStrategy):
    """
    Retriever wrapper that applies query expansion before search.

    This wrapper:
    - Expands the query using configured strategies
    - Performs retrieval with original and expanded queries
    - Merges and deduplicates results
    - Preserves query variations in metadata
    """

    def __init__(
        self,
        base_retriever: RetrieverStrategy,
        expansion_strategy: Optional[QueryExpansionStrategy] = None,
        max_variations: int = 3,
        merge_strategy: str = "union"  # "union" or "intersection"
    ):
        """
        Initialize query expansion retriever.

        Args:
            base_retriever: The underlying retriever
            expansion_strategy: Strategy for query expansion
            max_variations: Maximum query variations to use
            merge_strategy: How to merge results from variations
        """
        self.base_retriever = base_retriever
        self.expansion_strategy = expansion_strategy or HybridQueryExpansion()
        self.max_variations = max_variations
        self.merge_strategy = merge_strategy

        logger.info(f"Initialized QueryExpansionRetriever with {type(expansion_strategy).__name__}")

    @property
    def source_type(self) -> DataSource:
        """Delegate to base retriever."""
        return self.base_retriever.source_type

    async def retrieve(
        self,
        query: str,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 10
    ) -> SearchResult:
        """
        Retrieve with query expansion.

        Args:
            query: The search query
            filters: Optional filters to apply
            top_k: Number of results to return

        Returns:
            SearchResult with expanded query results
        """
        # Expand the query
        expanded = await self.expansion_strategy.expand(query)

        # Select variations to use
        queries_to_search = [query] + expanded.variations[:self.max_variations]

        # Perform parallel retrieval for all query variations
        retrieval_tasks = [
            self.base_retriever.retrieve(q, filters, top_k)
            for q in queries_to_search
        ]
        results = await asyncio.gather(*retrieval_tasks)

        # Merge results
        merged_result = self._merge_results(results, query, expanded)

        # Add expansion metadata
        merged_result.query_variations = queries_to_search
        if not merged_result.metadata:
            merged_result.metadata = {}
        merged_result.metadata["expansion"] = {
            "variations_used": len(queries_to_search),
            "synonyms_found": len(expanded.synonyms),
            "keywords_extracted": len(expanded.keywords)
        }

        logger.debug(f"Query expansion: {len(queries_to_search)} variations, {len(merged_result.documents)} results")

        return merged_result

    def _merge_results(
        self,
        results: list[SearchResult],
        original_query: str,
        expanded: ExpandedQuery
    ) -> SearchResult:
        """
        Merge results from multiple query variations.

        Args:
            results: List of search results from variations
            original_query: The original query
            expanded: The expanded query information

        Returns:
            Merged SearchResult
        """
        if self.merge_strategy == "intersection":
            # Only keep documents that appear in multiple results
            doc_counts: dict[str, int] = {}
            for result in results:
                for doc in result.documents:
                    doc_counts[doc.id] = doc_counts.get(doc.id, 0) + 1

            # Filter documents that appear in at least 2 results
            merged_docs = []
            seen_ids = set()
            for result in results:
                for doc in result.documents:
                    if doc.id not in seen_ids and doc_counts[doc.id] >= 2:
                        merged_docs.append(doc)
                        seen_ids.add(doc.id)
        else:  # union
            # Keep all unique documents
            merged_docs = []
            seen_ids = set()

            for result in results:
                for doc in result.documents:
                    if doc.id not in seen_ids:
                        merged_docs.append(doc)
                        seen_ids.add(doc.id)

        # Re-score based on appearance frequency
        for doc in merged_docs:
            appearances = sum(1 for r in results if any(d.id == doc.id for d in r.documents))
            doc.score = doc.score * (1 + 0.1 * (appearances - 1))  # Boost score for multiple appearances

        # Sort by score
        merged_docs.sort(key=lambda d: d.score, reverse=True)

        return SearchResult(
            documents=merged_docs,
            query=original_query,
            search_type="expanded_" + results[0].search_type if results else "expanded"
        )
