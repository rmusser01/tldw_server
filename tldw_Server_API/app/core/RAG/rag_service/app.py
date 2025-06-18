"""
Main RAG Application class that orchestrates the service.
"""

import asyncio
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import time
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

from loguru import logger

from .config import RAGConfig
from .types import (
    DataSource, Document, SearchResult, RAGContext, RAGResponse,
    RetrieverStrategy, ProcessingStrategy, GenerationStrategy,
    RAGError, ConfigurationError
)
from .cache import LRUCache
from .metrics import MetricsCollector


class RAGApplication:
    """
    Main application class for the RAG service.
    
    This class orchestrates the retrieval, processing, and generation
    components to provide a complete RAG solution.
    """
    
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        config_path: Optional[Path] = None
    ):
        """
        Initialize the RAG application.
        
        Args:
            config: Pre-built configuration object
            config_path: Path to configuration file
        """
        # Load configuration
        if config is None:
            config = RAGConfig.from_toml(config_path)
        
        # Validate configuration
        validation_errors = config.validate()
        if validation_errors:
            raise ConfigurationError(f"Invalid configuration: {', '.join(validation_errors)}")
        
        self.config = config
        
        # Initialize components
        self._retrievers: Dict[DataSource, RetrieverStrategy] = {}
        self._processor: Optional[ProcessingStrategy] = None
        self._generator: Optional[GenerationStrategy] = None
        
        # Initialize cache if enabled
        self._cache = None
        if config.cache.enable_cache:
            self._cache = LRUCache(max_size=config.cache.max_cache_size)
        
        # Initialize metrics collector
        self._metrics = MetricsCollector() if config.log_performance_metrics else None
        
        # Thread pool for parallel operations (reduced for single user)
        self._executor = ThreadPoolExecutor(max_workers=min(config.num_workers, 4))
        
        # Setup logging (simplified for TUI - uses existing logger)
        logger.level(config.log_level)
        
        logger.info("RAG Application initialized")
    
    def register_retriever(self, retriever: RetrieverStrategy) -> None:
        """Register a retriever for a specific data source."""
        source = retriever.source_type
        if source in self._retrievers:
            logger.warning(f"Overwriting existing retriever for {source}")
        self._retrievers[source] = retriever
        logger.debug(f"Registered retriever for {source}")
    
    def register_processor(self, processor: ProcessingStrategy) -> None:
        """Register the document processor."""
        self._processor = processor
        logger.debug("Registered document processor")
    
    def register_generator(self, generator: GenerationStrategy) -> None:
        """Register the response generator."""
        self._generator = generator
        logger.debug("Registered response generator")
    
    async def search(
        self,
        query: str,
        sources: Optional[List[DataSource]] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        Perform search across specified data sources.
        
        Args:
            query: Search query
            sources: List of data sources to search (None = all registered)
            filters: Optional filters to apply
            **kwargs: Additional arguments passed to retrievers
            
        Returns:
            List of search results from all sources
        """
        start_time = time.time()
        
        # Determine which sources to search
        if sources is None:
            sources = list(self._retrievers.keys())
        else:
            # Validate requested sources
            available = set(self._retrievers.keys())
            requested = set(sources)
            missing = requested - available
            if missing:
                logger.warning(f"Requested sources not available: {missing}")
                sources = list(requested & available)
        
        if not sources:
            logger.warning("No valid sources to search")
            return []
        
        # Check cache if enabled
        cache_key = None
        if self._cache and self.config.cache.cache_search_results:
            cache_key = self._make_cache_key("search", query, sources, filters)
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for search: {cache_key}")
                return cached
        
        # Perform parallel searches
        logger.info(f"Searching {len(sources)} sources for query: {query[:50]}...")
        
        async def search_source(source: DataSource) -> Optional[SearchResult]:
            try:
                retriever = self._retrievers[source]
                result = await retriever.retrieve(
                    query=query,
                    filters=filters,
                    top_k=kwargs.get(f"{source.name.lower()}_top_k", self.config.retriever.vector_top_k)
                )
                logger.debug(f"Retrieved {len(result.documents)} documents from {source}")
                return result
            except Exception as e:
                logger.error(f"Error retrieving from {source}: {e}")
                if self._metrics:
                    self._metrics.increment_error("retrieval", source.name)
                return None
        
        # Execute searches concurrently
        tasks = [search_source(source) for source in sources]
        results = await asyncio.gather(*tasks)
        
        # Filter out None results
        search_results = [r for r in results if r is not None]
        
        # Cache results if enabled
        if cache_key and self._cache:
            self._cache.set(
                cache_key, 
                search_results, 
                ttl=self.config.cache.cache_ttl
            )
        
        # Record metrics
        if self._metrics:
            elapsed = time.time() - start_time
            self._metrics.record_latency("search", elapsed)
            self._metrics.increment_counter(
                "search_requests",
                {"sources": len(sources), "results": len(search_results)}
            )
        
        return search_results
    
    async def generate(
        self,
        query: str,
        sources: Optional[List[DataSource]] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RAGResponse:
        """
        Complete RAG pipeline: search, process, and generate response.
        
        Args:
            query: User query
            sources: Data sources to search
            filters: Optional filters
            **kwargs: Additional arguments
            
        Returns:
            Complete RAG response with answer and sources
        """
        start_time = time.time()
        
        # Validate components
        if not self._processor:
            raise RAGError("No processor registered")
        if not self._generator:
            raise RAGError("No generator registered")
        
        try:
            # Step 1: Search
            search_results = await self.search(query, sources, filters, **kwargs)
            
            if not search_results:
                logger.warning("No search results found")
                return RAGResponse(
                    answer="I couldn't find any relevant information to answer your question.",
                    context=RAGContext(documents=[], combined_text="", total_tokens=0, metadata={}),
                    sources=[],
                    metadata={"elapsed_time": time.time() - start_time}
                )
            
            # Step 2: Process
            context = self._processor.process(
                search_results=search_results,
                query=query,
                max_context_length=kwargs.get("max_context_length", self.config.processor.max_context_length)
            )
            
            logger.info(f"Processed context with {len(context.documents)} documents, {context.total_tokens} tokens")
            
            # Step 3: Generate
            answer = await self._generator.generate(
                context=context,
                query=query,
                **kwargs
            )
            
            # Prepare response
            response = RAGResponse(
                answer=answer,
                context=context,
                sources=context.documents,
                metadata={
                    "elapsed_time": time.time() - start_time,
                    "num_sources_searched": len(search_results),
                    "num_documents_used": len(context.documents),
                    "total_tokens": context.total_tokens
                }
            )
            
            # Record metrics
            if self._metrics:
                self._metrics.record_latency("rag_pipeline", time.time() - start_time)
                self._metrics.increment_counter("rag_requests")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            if self._metrics:
                self._metrics.increment_error("rag_pipeline", str(type(e).__name__))
            raise RAGError(f"RAG pipeline failed: {e}") from e
    
    async def embed_documents(
        self,
        documents: List[Document],
        source: DataSource,
        batch_size: Optional[int] = None
    ) -> None:
        """
        Embed and store documents for a specific data source.
        
        Args:
            documents: Documents to embed
            source: Data source type
            batch_size: Batch size for embedding (None = use config)
        """
        if source not in self._retrievers:
            raise ValueError(f"No retriever registered for {source}")
        
        retriever = self._retrievers[source]
        if not hasattr(retriever, 'embed_and_store'):
            raise ValueError(f"Retriever for {source} does not support embedding")
        
        batch_size = batch_size or self.config.batch_size
        
        logger.info(f"Embedding {len(documents)} documents for {source}")
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            await retriever.embed_and_store(batch)
            logger.debug(f"Embedded batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
    
    def _make_cache_key(self, *args) -> str:
        """Create a cache key from arguments."""
        import hashlib
        import json
        
        # Convert args to JSON-serializable format
        key_data = []
        for arg in args:
            if isinstance(arg, list):
                # Handle lists of enums or other objects
                list_items = []
                for item in arg:
                    if isinstance(item, Enum):
                        list_items.append(item.name)
                    else:
                        list_items.append(str(item))
                key_data.append(json.dumps(list_items, sort_keys=True))
            elif isinstance(arg, dict):
                key_data.append(json.dumps(arg, sort_keys=True))
            elif isinstance(arg, Enum):
                key_data.append(arg.name)
            else:
                key_data.append(str(arg))
        
        key_string = "|".join(key_data)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def clear_cache(self) -> None:
        """Clear all cached data."""
        if self._cache:
            self._cache.clear()
            logger.info("Cache cleared")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        if self._metrics:
            return self._metrics.get_summary()
        return {}
    
    @asynccontextmanager
    async def lifespan(self):
        """Context manager for application lifecycle."""
        logger.info("Starting RAG Application")
        yield
        logger.info("Shutting down RAG Application")
        
        # Cleanup
        self._executor.shutdown(wait=True)
        if self._cache:
            await self.clear_cache()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.lifespan().__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.lifespan().__aexit__(exc_type, exc_val, exc_tb)