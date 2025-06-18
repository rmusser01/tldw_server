"""
Integration module for connecting the RAG service with the existing tldw_chatbook app.

This module provides adapters and utilities to integrate the new RAG service
with the existing codebase.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import asyncio

from loguru import logger

from .app import RAGApplication
from .config import RAGConfig
from .types import DataSource
from .retrieval import (
    MediaDBRetriever, ChatHistoryRetriever, NotesRetriever,
    VectorRetriever, HybridRetriever
)
from .processing import DefaultProcessor, AdvancedProcessor
from .generation import LLMGenerator, StreamingGenerator, FallbackGenerator


class RAGService:
    """
    High-level service class for easy integration with the TUI app.
    
    This provides a simplified interface that matches the app's needs.
    """
    
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        config_path: Optional[Path] = None,
        media_db_path: Optional[Path] = None,
        chachanotes_db_path: Optional[Path] = None,
        chroma_path: Optional[Path] = None,
        llm_handler: Optional[Any] = None
    ):
        """
        Initialize the RAG service for multi-user server.
        
        Args:
            config: Pre-configured RAGConfig object
            config_path: Path to TOML config file (used if config not provided)
            media_db_path: Path to media database
            chachanotes_db_path: Path to ChaChaNotes database
            chroma_path: Path to ChromaDB storage
            llm_handler: The app's LLM handler for generation
        """
        # Load configuration
        if config is not None:
            self.config = config
        else:
            self.config = RAGConfig.from_toml(config_path)
        
        # Store paths
        self.media_db_path = media_db_path
        self.chachanotes_db_path = chachanotes_db_path
        self.chroma_path = chroma_path or Path.home() / ".tldw_chatbook" / "chroma"
        
        # Ensure chroma directory exists
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize application
        self.app = RAGApplication(self.config)
        
        # Store LLM handler
        self.llm_handler = llm_handler
        
        # Initialize components
        self._initialized = False
    
    async def initialize(self):
        """Initialize all components."""
        if self._initialized:
            return
        
        logger.info("Initializing RAG service components...")
        
        # Initialize retrievers
        await self._setup_retrievers()
        
        # Initialize processor
        self._setup_processor()
        
        # Initialize generator
        self._setup_generator()
        
        self._initialized = True
        logger.info("RAG service initialized successfully")
    
    async def _setup_retrievers(self):
        """Set up all retrievers for the single-user app."""
        retrievers_config = []
        
        # Media DB retriever
        if self.media_db_path and self.media_db_path.exists():
            media_retriever = MediaDBRetriever(self.media_db_path)
            self.app.register_retriever(media_retriever)
            retrievers_config.append(("MediaDB", "FTS"))
            
            # Vector retriever for media
            if self.config.retriever.vector_top_k > 0:
                media_vector = VectorRetriever(
                    DataSource.MEDIA_DB,
                    self.chroma_path,
                    self.config.retriever.media_collection
                )
                
                # Create hybrid retriever
                media_hybrid = HybridRetriever(
                    media_retriever,
                    media_vector,
                    alpha=self.config.retriever.hybrid_alpha
                )
                self.app.register_retriever(media_hybrid)
                retrievers_config.append(("MediaDB", "Hybrid"))
        
        # Chat history retriever
        if self.chachanotes_db_path and self.chachanotes_db_path.exists():
            chat_retriever = ChatHistoryRetriever(self.chachanotes_db_path)
            self.app.register_retriever(chat_retriever)
            retrievers_config.append(("ChatHistory", "Keyword"))
            
            # Vector retriever for chat
            if self.config.retriever.vector_top_k > 0:
                chat_vector = VectorRetriever(
                    DataSource.CHAT_HISTORY,
                    self.chroma_path,
                    self.config.retriever.chat_collection
                )
                
                # Create hybrid retriever
                chat_hybrid = HybridRetriever(
                    chat_retriever,
                    chat_vector,
                    alpha=self.config.retriever.hybrid_alpha
                )
                self.app.register_retriever(chat_hybrid)
                retrievers_config.append(("ChatHistory", "Hybrid"))
        
        # Notes retriever
        if self.chachanotes_db_path and self.chachanotes_db_path.exists():
            notes_retriever = NotesRetriever(self.chachanotes_db_path)
            self.app.register_retriever(notes_retriever)
            retrievers_config.append(("Notes", "Keyword"))
        
        logger.info(f"Configured retrievers: {retrievers_config}")
    
    def _setup_processor(self):
        """Set up the document processor."""
        if self.config.processor.enable_reranking:
            processor = AdvancedProcessor(self.config.processor.__dict__)
        else:
            processor = DefaultProcessor(self.config.processor.__dict__)
        
        self.app.register_processor(processor)
        logger.info(f"Configured processor: {type(processor).__name__}")
    
    def _setup_generator(self):
        """Set up the response generator."""
        if self.llm_handler:
            # Use the app's LLM handler
            if self.config.generator.enable_streaming:
                generator = StreamingGenerator(
                    self.llm_handler,
                    self.config.generator.__dict__
                )
            else:
                generator = LLMGenerator(
                    self.llm_handler,
                    self.config.generator.__dict__
                )
        else:
            # Fallback generator when no LLM available
            generator = FallbackGenerator(self.config.generator.__dict__)
            logger.warning("No LLM handler provided, using fallback generator")
        
        self.app.register_generator(generator)
        logger.info(f"Configured generator: {type(generator).__name__}")
    
    async def search(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Perform search across configured sources.
        
        Args:
            query: Search query
            sources: List of source names (None = all)
            filters: Filters to apply
            
        Returns:
            List of search results
        """
        if not self._initialized:
            await self.initialize()
        
        # Convert source names to DataSource enums
        if sources:
            source_enums = []
            for source in sources:
                try:
                    source_enums.append(DataSource[source.upper()])
                except KeyError:
                    logger.warning(f"Unknown source: {source}")
        else:
            source_enums = None
        
        # Perform search
        results = await self.app.search(query, source_enums, filters, **kwargs)
        
        # Convert to simple format
        all_docs = []
        for result in results:
            for doc in result.documents:
                all_docs.append({
                    "id": doc.id,
                    "content": doc.content,
                    "source": doc.source.name,
                    "score": doc.score,
                    "metadata": doc.metadata
                })
        
        return all_docs
    
    async def generate_answer(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate an answer using RAG.
        
        Args:
            query: User question
            sources: Sources to search
            filters: Search filters
            stream: Whether to stream the response
            
        Returns:
            Dictionary with answer and metadata
        """
        if not self._initialized:
            await self.initialize()
        
        # Convert sources
        source_enums = None
        if sources:
            source_enums = []
            for source in sources:
                try:
                    source_enums.append(DataSource[source.upper()])
                except KeyError:
                    logger.warning(f"Unknown source: {source}")
        
        # Generate response
        response = await self.app.generate(
            query=query,
            sources=source_enums,
            filters=filters,
            **kwargs
        )
        
        # Format response
        return {
            "answer": response.answer,
            "sources": [
                {
                    "id": doc.id,
                    "source": doc.source.name,
                    "title": doc.metadata.get("title", "Untitled"),
                    "score": doc.score,
                    "snippet": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "metadata": doc.metadata
                }
                for doc in response.sources
            ],
            "metadata": response.metadata,
            "context_preview": response.context.combined_text[:500] + "..."
            if len(response.context.combined_text) > 500 else response.context.combined_text,
            "context_size": len(response.context.combined_text)
        }
    
    async def generate_answer_stream(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Generate an answer using RAG with streaming support.
        
        Args:
            query: User question
            sources: Sources to search
            filters: Search filters
            **kwargs: Additional arguments passed to generation
            
        Yields:
            Dict chunks with type and content
        """
        if not self._initialized:
            await self.initialize()
        
        # Convert sources
        source_enums = None
        if sources:
            source_enums = []
            for source in sources:
                try:
                    source_enums.append(DataSource[source.upper()])
                except KeyError:
                    logger.warning(f"Unknown source: {source}")
        
        # Stream response
        async for chunk in self.app.generate_stream(
            query=query,
            sources=source_enums,
            filters=filters,
            **kwargs
        ):
            if hasattr(chunk, 'content'):
                yield {"type": "content", "content": chunk.content}
            elif hasattr(chunk, 'citation'):
                yield {
                    "type": "citation", 
                    "citation": {
                        "id": chunk.citation.id,
                        "source": chunk.citation.source,
                        "title": chunk.citation.title,
                        "snippet": chunk.citation.snippet
                    }
                }
            else:
                # Raw text chunk
                yield {"type": "content", "content": str(chunk)}
    
    async def embed_documents(
        self,
        source: str,
        documents: List[Dict[str, Any]]
    ) -> None:
        """
        Embed and store documents for vector search.
        
        Args:
            source: Source name (e.g., "MEDIA_DB")
            documents: List of documents to embed
        """
        if not self._initialized:
            await self.initialize()
        
        # Convert to Document objects
        from .types import Document
        
        doc_objects = []
        source_enum = DataSource[source.upper()]
        
        for doc in documents:
            doc_obj = Document(
                id=doc["id"],
                content=doc["content"],
                metadata=doc.get("metadata", {}),
                source=source_enum
            )
            doc_objects.append(doc_obj)
        
        # Embed
        await self.app.embed_documents(doc_objects, source_enum)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        stats = {
            "initialized": self._initialized,
            "config": {
                "cache_enabled": self.config.cache.enable_cache,
                "reranking_enabled": self.config.processor.enable_reranking,
                "hybrid_search_alpha": self.config.retriever.hybrid_alpha
            }
        }
        
        if self._initialized and self.app._cache:
            stats["cache"] = self.app._cache.get_stats()
        
        return stats
    
    async def clear_cache(self):
        """Clear all caches."""
        if self._initialized:
            await self.app.clear_cache()
    
    async def close(self):
        """Clean up resources."""
        if self._initialized:
            # Close database connections
            for retriever in self.app._retrievers.values():
                if hasattr(retriever, 'close'):
                    retriever.close()
            
            logger.info("RAG service closed")


# Compatibility functions for easier migration

async def enhanced_rag_pipeline(
    query: str,
    api_choice: str,
    media_db_path: Path,
    chachanotes_db_path: Path,
    keywords: Optional[str] = None,
    fts_top_k: int = 10,
    vector_top_k: int = 10,
    apply_re_ranking: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Compatibility wrapper for the old enhanced_rag_pipeline function.
    
    This allows gradual migration from the old API.
    """
    # Create temporary service
    service = RAGService(
        media_db_path=media_db_path,
        chachanotes_db_path=chachanotes_db_path
    )
    
    # Override config
    service.config.retriever.fts_top_k = fts_top_k
    service.config.retriever.vector_top_k = vector_top_k
    service.config.processor.enable_reranking = apply_re_ranking
    
    # Initialize
    await service.initialize()
    
    # Process keywords into filters
    filters = {}
    if keywords:
        filters["keywords"] = keywords.split(",")
    
    # Generate answer
    result = await service.generate_answer(
        query=query,
        sources=["MEDIA_DB", "NOTES"],
        filters=filters,
        **kwargs
    )
    
    # Format for compatibility
    return {
        "answer": result["answer"],
        "context": result["context_preview"],
        "source_documents": result["sources"]
    }