"""
Type definitions and base interfaces for the RAG service.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, Optional, Protocol, TypeVar

import numpy as np


class DataSource(Enum):
    """Supported data sources for RAG."""
    MEDIA_DB = "media_db"
    CHAT_HISTORY = "chat_history"
    NOTES = "notes"
    CHARACTER_CARDS = "character_cards"
    WEB_CONTENT = "web_content"
    PROMPTS = "prompts"  # Add missing PROMPTS source
    WORLD_BOOKS = "world_books"
    DICTIONARIES = "dictionaries"
    SQL = "sql"
    KANBAN = "kanban"
    CLAIMS = "claims"  # Claims table/vector store


class QueryType(Enum):
    """Classification of query types for granularity routing."""
    BROAD = "broad"          # Overview/summary queries - use document-level
    SPECIFIC = "specific"    # Detail queries - use chunk-level
    FACTOID = "factoid"      # Precise fact queries - use passage-level


class Granularity(Enum):
    """Retrieval granularity levels."""
    DOCUMENT = "document"    # Full document retrieval
    CHUNK = "chunk"          # Chunk-level retrieval (default)
    PASSAGE = "passage"      # Fine-grained passage retrieval


class CitationType(Enum):
    """Type of citation match."""
    EXACT = "exact"          # Exact phrase match
    SEMANTIC = "semantic"    # Semantic similarity
    FUZZY = "fuzzy"         # Fuzzy/partial match
    KEYWORD = "keyword"      # Keyword/FTS5 match


class ClaimType(Enum):
    """Classification of claim types for structured verification."""
    STATISTIC = "statistic"      # "Revenue grew 15%"
    COMPARATIVE = "comparative"  # "X is larger than Y"
    TEMPORAL = "temporal"        # "In 2023, ..."
    ATTRIBUTION = "attribution"  # "According to Smith, ..."
    CAUSAL = "causal"           # "X caused Y"
    EXISTENCE = "existence"      # "There exists..."
    RANKING = "ranking"         # "X is the largest"
    QUOTE = "quote"             # Direct quotation
    GENERAL = "general"         # Unclassified


class VerificationStatus(Enum):
    """Rich verification status for deterministic claim checking."""
    VERIFIED = "verified"                    # Claim is supported by evidence
    CITATION_NOT_FOUND = "citation_not_found"  # Referenced source not in corpus
    MISQUOTED = "misquoted"                  # Quote doesn't match source
    MISLEADING = "misleading"                # Technically true but deceptive
    HALLUCINATION = "hallucination"          # No evidence supports claim
    UNVERIFIED = "unverified"                # Insufficient evidence (NEI)
    NUMERICAL_ERROR = "numerical_error"      # Numbers don't match source
    REFUTED = "refuted"                      # Evidence contradicts claim
    CONTESTED = "contested"                  # Evidence exists both for and against


class MatchLevel(Enum):
    """Classification of match confidence between claim and evidence."""
    EXACT = "exact"              # Verbatim match
    PARAPHRASE = "paraphrase"    # Same meaning, different words
    INTERPRETATION = "interpretation"  # Inference from source


class SourceAuthority(Enum):
    """Source authority ranking for evidence weighting."""
    PRIMARY = 5        # Original research, primary sources
    GOVERNMENT = 4     # .gov, official statistics
    PEER_REVIEWED = 3  # DOI, academic journals
    INDUSTRY = 2       # Whitepapers, industry reports
    SECONDARY = 1      # News, Wikipedia, blogs


@dataclass
class Citation:
    """
    Represents a citation to a source document.

    Attributes:
        document_id: Unique identifier of the source document
        document_title: Human-readable title of the document
        chunk_id: ID of the specific chunk within the document
        text: The actual text snippet being cited
        start_char: Character offset in the original document
        end_char: End character offset in the original document
        confidence: Confidence score (0-1) for this citation
        match_type: Type of match that produced this citation
        metadata: Additional metadata (author, date, URL, etc.)
        location: Human-readable location (page, section, etc.)
        formatted_citation: Pre-formatted citation string (MLA/APA/etc.)
    """
    document_id: str
    document_title: str
    chunk_id: str
    text: str
    start_char: int
    end_char: int
    confidence: float
    match_type: CitationType
    metadata: dict[str, Any] = field(default_factory=dict)
    location: Optional[str] = None  # "Chapter 3, Page 45" or "Section 2.1"
    formatted_citation: Optional[str] = None  # Pre-formatted academic citation

    def __post_init__(self):
        """Validate citation data."""
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        if self.start_char < 0:
            raise ValueError(f"start_char must be non-negative, got {self.start_char}")
        if self.end_char < self.start_char:
            raise ValueError(f"end_char must be >= start_char, got start={self.start_char}, end={self.end_char}")

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "document_id": self.document_id,
            "document_title": self.document_title,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "confidence": self.confidence,
            "match_type": self.match_type.value,
            "metadata": self.metadata
        }


@dataclass
class Document:
    """
    Represents a document in the RAG system.

    This is a unified representation for all types of content
    (media transcripts, chat messages, notes, etc.)
    """
    id: str  # Unique identifier
    content: str  # The actual text content
    metadata: dict[str, Any]  # Source-specific metadata
    source: DataSource = DataSource.MEDIA_DB  # Default for compatibility with tests
    score: float = 0.0  # Relevance score (set during retrieval)
    embedding: Optional[np.ndarray] = None  # Vector embedding if available

    # Citation support
    citations: list[Citation] = field(default_factory=list)

    # Enhanced chunk lineage tracking
    source_document_id: Optional[str] = None  # Original document this chunk came from
    source_document_metadata: dict[str, Any] = field(default_factory=dict)  # Title, author, date, URL, etc.

    # Parent document support (for hierarchical chunking)
    parent_id: Optional[str] = None  # ID of parent document if this is a chunk
    children_ids: list[str] = field(default_factory=list)  # IDs of child chunks
    chunk_index: Optional[int] = None  # Position in parent document (1-based)
    total_chunks: Optional[int] = None  # Total number of chunks in source document

    # Character positions for citation tracking
    start_char: Optional[int] = None  # Start position in original document
    end_char: Optional[int] = None  # End position in original document

    # Location information for academic citations
    page_number: Optional[int] = None  # Page number in source document
    section_title: Optional[str] = None  # Section/chapter title
    paragraph_number: Optional[int] = None  # Paragraph number within section

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Document):
            return False
        return self.id == other.id

    def add_citation(self, citation: Citation) -> None:
        """Add a citation to this document."""
        self.citations.append(citation)

    def get_citations_by_type(self, citation_type: CitationType) -> list[Citation]:
        """Get citations of a specific type."""
        return [c for c in self.citations if c.match_type == citation_type]

    def get_source_info(self) -> dict[str, Any]:
        """Get source document information for citation generation."""
        if self.source_document_metadata:
            return self.source_document_metadata
        # Fallback to regular metadata
        return self.metadata

    def get_location_string(self) -> str:
        """Get human-readable location within source document."""
        parts = []
        if self.section_title:
            parts.append(f"Section: {self.section_title}")
        if self.page_number:
            parts.append(f"Page {self.page_number}")
        if self.paragraph_number:
            parts.append(f"Paragraph {self.paragraph_number}")
        if self.chunk_index is not None and self.total_chunks:
            parts.append(f"Chunk {self.chunk_index}/{self.total_chunks}")
        return ", ".join(parts) if parts else "Unknown location"


@dataclass
class SearchResult:
    """Result from a search operation."""
    documents: list[Document]
    query: str
    search_type: str  # "vector", "fts", "hybrid"
    metadata: dict[str, Any] = field(default_factory=dict)  # Additional search metadata

    # Enhanced features
    citations: list[Citation] = field(default_factory=list)
    expanded_context: Optional[str] = None  # Context expanded with parent documents
    query_variations: list[str] = field(default_factory=list)  # Query expansion results


@dataclass
class EnhancedSearchResult(SearchResult):
    """Enhanced search result with additional features."""
    parent_documents: list[Document] = field(default_factory=list)
    reranked: bool = False
    diversity_score: float = 0.0


@dataclass
class RAGContext:
    """Context prepared for generation."""
    documents: list[Document]
    combined_text: str
    total_tokens: int
    metadata: dict[str, Any]

    # Enhanced features
    citations: list[Citation] = field(default_factory=list)
    parent_context: Optional[str] = None  # Expanded context from parent documents
    structured_sections: dict[str, str] = field(default_factory=dict)  # Structured content


@dataclass
class RAGPipelineContext:
    """
    Lightweight pipeline context used by functional RAG steps and property tests.

    This mirrors the minimal fields expected by tests and pipeline utilities and
    is intentionally simple: it captures the original and current query, any
    retrieved documents, configuration/metadata, a cache flag, timings, and a
    mutable errors list for error accumulation.
    """
    query: str
    original_query: str
    documents: list[Document] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    cache_hit: bool = False
    timings: dict[str, float] = field(default_factory=dict)
    errors: list[Any] = field(default_factory=list)


@dataclass
class RAGResponse:
    """Response from the RAG pipeline."""
    answer: str
    context: RAGContext
    sources: list[Document]
    metadata: dict[str, Any]  # Timing, model used, etc.

    # Enhanced features
    citations: list[Citation] = field(default_factory=list)
    confidence_score: float = 0.0  # Overall confidence in the response


# Protocol definitions for better type checking

class Embedder(Protocol):
    """Protocol for embedding models."""
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        ...

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts."""
        ...

# Provide a backwards-compatible global for tests referencing the name directly
try:  # pragma: no cover - defensive convenience for tests
    import builtins as _builtins
    if not hasattr(_builtins, "RAGPipelineContext"):
        _builtins.RAGPipelineContext = RAGPipelineContext  # type: ignore[attr-defined]
except Exception as builtins_patch_error:
    _ = builtins_patch_error


class Reranker(Protocol):
    """Protocol for reranking models."""
    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """Rerank documents based on relevance to query."""
        ...


T = TypeVar('T')


class Cache(Protocol, Generic[T]):
    """Protocol for cache implementations."""
    def get(self, key: str) -> Optional[T]:
        """Get item from cache."""
        ...

    def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Set item in cache with optional TTL."""
        ...

    def delete(self, key: str) -> None:
        """Delete item from cache."""
        ...

    def clear(self) -> None:
        """Clear all items from cache."""
        ...


# Abstract base classes for strategy pattern

class RetrieverStrategy(ABC):
    """Base class for retrieval strategies."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        filters: Optional[dict[str, Any]] = None,
        top_k: int = 10
    ) -> SearchResult:
        """
        Retrieve relevant documents for the query.

        Args:
            query: The search query
            filters: Optional filters to apply
            top_k: Number of results to return

        Returns:
            SearchResult containing relevant documents
        """
        pass

    @property
    @abstractmethod
    def source_type(self) -> DataSource:
        """The data source this retriever handles."""
        pass


class ProcessingStrategy(ABC):
    """Base class for document processing strategies."""

    @abstractmethod
    def process(
        self,
        search_results: list[SearchResult],
        query: str,
        max_context_length: int = 4096
    ) -> RAGContext:
        """
        Process search results into a context for generation.

        Args:
            search_results: Results from various retrievers
            query: The original query
            max_context_length: Maximum context length in tokens

        Returns:
            Processed context ready for generation
        """
        pass


class GenerationStrategy(ABC):
    """Base class for generation strategies."""

    @abstractmethod
    async def generate(
        self,
        context: RAGContext,
        query: str,
        **kwargs
    ) -> str:
        """
        Generate response using the context.

        Args:
            context: The prepared context
            query: The original query
            **kwargs: Additional generation parameters

        Returns:
            Generated response
        """
        pass


# Evidence Chain Types for Multi-Hop Reasoning

@dataclass
class EvidenceNode:
    """
    A node in an evidence chain representing a single fact from a source.

    Tracks the relationship between a source document, the fact extracted,
    and which claims this fact supports.
    """
    document_id: str
    chunk_id: str
    fact: str
    confidence: float
    supports: list[str] = field(default_factory=list)  # Claim IDs this fact supports
    source_text: str = ""  # Original text from which fact was extracted
    extraction_method: str = "llm"  # "llm", "pattern", "keyword"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate node data."""
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "fact": self.fact,
            "confidence": self.confidence,
            "supports": self.supports,
            "source_text": self.source_text[:200] if self.source_text else "",
            "extraction_method": self.extraction_method,
            "metadata": self.metadata,
        }


@dataclass
class EvidenceChain:
    """
    A chain of evidence nodes supporting claims in a response.

    Tracks dependencies between evidence nodes and computes aggregate
    confidence scores for the chain.
    """
    query: str
    nodes: list[EvidenceNode] = field(default_factory=list)
    root_claims: list[str] = field(default_factory=list)  # Top-level claims being supported
    chain_confidence: float = 0.0  # Aggregate confidence (product of node confidences)
    hop_count: int = 0  # Number of reasoning hops in the chain
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compute chain statistics after initialization."""
        self._compute_chain_confidence()
        self._compute_hop_count()

    def _compute_chain_confidence(self) -> None:
        """Compute aggregate chain confidence as product of node confidences."""
        if not self.nodes:
            self.chain_confidence = 0.0
            return

        # Use product of confidences (conservative estimate)
        confidence = 1.0
        for node in self.nodes:
            confidence *= node.confidence
        self.chain_confidence = confidence

    def _compute_hop_count(self) -> None:
        """Compute the number of reasoning hops in the chain."""
        if not self.nodes:
            self.hop_count = 0
            return

        # Count unique documents involved
        unique_docs = {node.document_id for node in self.nodes}
        self.hop_count = len(unique_docs)

    def add_node(self, node: EvidenceNode) -> None:
        """Add a node to the chain and recompute statistics."""
        self.nodes.append(node)
        self._compute_chain_confidence()
        self._compute_hop_count()

    def get_nodes_for_claim(self, claim_id: str) -> list[EvidenceNode]:
        """Get all nodes that support a specific claim."""
        return [n for n in self.nodes if claim_id in n.supports]

    def get_source_documents(self) -> list[str]:
        """Get unique list of source document IDs."""
        return list({node.document_id for node in self.nodes})

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "nodes": [n.to_dict() for n in self.nodes],
            "root_claims": self.root_claims,
            "chain_confidence": self.chain_confidence,
            "hop_count": self.hop_count,
            "source_documents": self.get_source_documents(),
            "metadata": self.metadata,
        }


# Exceptions

class RAGError(Exception):
    """Base exception for RAG service."""
    pass


class RetrievalError(RAGError):
    """Error during document retrieval."""
    pass


class ProcessingError(RAGError):
    """Error during document processing."""
    pass


class GenerationError(RAGError):
    """Error during response generation."""
    pass


class ConfigurationError(RAGError):
    """Error in configuration."""
    pass
