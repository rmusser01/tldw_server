"""
Type definitions and base interfaces for the RAG service.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Protocol, TypeVar, Generic
import numpy as np


class DataSource(Enum):
    """Supported data sources for RAG."""
    MEDIA_DB = auto()
    CHAT_HISTORY = auto()
    NOTES = auto()
    CHARACTER_CARDS = auto()
    WEB_CONTENT = auto()


@dataclass
class Document:
    """
    Represents a document in the RAG system.
    
    This is a unified representation for all types of content
    (media transcripts, chat messages, notes, etc.)
    """
    id: str  # Unique identifier
    content: str  # The actual text content
    metadata: Dict[str, Any]  # Source-specific metadata
    source: DataSource  # Where this document came from
    score: float = 0.0  # Relevance score (set during retrieval)
    embedding: Optional[np.ndarray] = None  # Vector embedding if available
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, Document):
            return False
        return self.id == other.id


@dataclass
class SearchResult:
    """Result from a search operation."""
    documents: List[Document]
    query: str
    search_type: str  # "vector", "fts", "hybrid"
    metadata: Dict[str, Any] = None  # Additional search metadata


@dataclass
class RAGContext:
    """Context prepared for generation."""
    documents: List[Document]
    combined_text: str
    total_tokens: int
    metadata: Dict[str, Any]


@dataclass
class RAGResponse:
    """Response from the RAG pipeline."""
    answer: str
    context: RAGContext
    sources: List[Document]
    metadata: Dict[str, Any]  # Timing, model used, etc.


# Protocol definitions for better type checking

class Embedder(Protocol):
    """Protocol for embedding models."""
    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text."""
        ...
    
    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        ...


class Reranker(Protocol):
    """Protocol for reranking models."""
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
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
        filters: Optional[Dict[str, Any]] = None,
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
        search_results: List[SearchResult],
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