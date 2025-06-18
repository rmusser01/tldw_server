"""
Example usage of the RAG service.

This demonstrates how to use the new RAG service architecture.
"""

import asyncio
from pathlib import Path
from typing import List

from loguru import logger

from .app import RAGApplication
from .config import RAGConfig
from .types import DataSource, Document, SearchResult, RAGContext
from .retrieval import SimpleRetriever
from .processing import DefaultProcessor
from .generation import MockGenerator


async def basic_usage_example():
    """Basic example of using the RAG service."""
    
    # 1. Create configuration
    config = RAGConfig()
    config.retriever.vector_top_k = 5
    config.processor.enable_reranking = True
    config.cache.enable_cache = True
    
    # 2. Initialize application
    app = RAGApplication(config)
    
    # 3. Register components
    # In real usage, these would be actual implementations
    app.register_retriever(SimpleRetriever(DataSource.MEDIA_DB))
    app.register_retriever(SimpleRetriever(DataSource.NOTES))
    app.register_processor(DefaultProcessor())
    app.register_generator(MockGenerator())
    
    # 4. Perform RAG query
    response = await app.generate(
        query="What is the main topic discussed in the video?",
        sources=[DataSource.MEDIA_DB],
        filters={"media_type": "video"}
    )
    
    # 5. Use the response
    print(f"Answer: {response.answer}")
    print(f"Sources used: {len(response.sources)}")
    print(f"Time taken: {response.metadata['elapsed_time']:.2f}s")
    
    # 6. Get metrics
    metrics = await app.get_metrics()
    print(f"Cache hit rate: {metrics.get('cache', {}).get('hit_rate', 0):.2%}")


async def advanced_usage_example():
    """Advanced example with custom configuration and multiple sources."""
    
    # Load config from TOML file
    config = RAGConfig.from_toml(Path("config.toml"))
    
    async with RAGApplication(config) as app:
        # Register all retrievers
        retrievers = [
            MediaDBRetriever(media_db_instance),
            ChatHistoryRetriever(chat_db_instance),
            NotesRetriever(notes_db_instance),
            WebContentRetriever()
        ]
        
        for retriever in retrievers:
            app.register_retriever(retriever)
        
        # Register advanced processor with reranking
        processor = AdvancedProcessor(
            reranker_model="ms-marco-MiniLM-L-12-v2",
            deduplication_threshold=0.9
        )
        app.register_processor(processor)
        
        # Register LLM generator
        generator = LLMGenerator(
            model="gpt-3.5-turbo",
            temperature=0.7,
            streaming=True
        )
        app.register_generator(generator)
        
        # Perform complex query across multiple sources
        response = await app.generate(
            query="Compare the discussions about AI safety in my recent chat conversations with the notes I've taken",
            sources=[DataSource.CHAT_HISTORY, DataSource.NOTES],
            filters={
                "date_range": "last_7_days",
                "keywords": ["AI", "safety", "alignment"]
            },
            max_context_length=8192,
            stream=True
        )
        
        # Handle streaming response
        async for chunk in response:
            print(chunk, end="", flush=True)


async def embedding_example():
    """Example of embedding documents for later retrieval."""
    
    config = RAGConfig()
    app = RAGApplication(config)
    
    # Prepare documents
    documents = [
        Document(
            id="doc1",
            content="This is a document about machine learning basics.",
            metadata={"source_id": "media_123", "type": "transcript"},
            source=DataSource.MEDIA_DB
        ),
        Document(
            id="doc2",
            content="Advanced topics in deep learning and neural networks.",
            metadata={"source_id": "media_124", "type": "transcript"},
            source=DataSource.MEDIA_DB
        )
    ]
    
    # Embed and store
    await app.embed_documents(
        documents=documents,
        source=DataSource.MEDIA_DB,
        batch_size=32
    )
    
    print(f"Embedded {len(documents)} documents")


async def caching_example():
    """Example showing caching benefits."""
    
    config = RAGConfig()
    config.cache.enable_cache = True
    config.cache.cache_ttl = 300  # 5 minutes
    
    app = RAGApplication(config)
    
    # First query - will hit the sources
    response1 = await app.generate(query="What is RAG?")
    time1 = response1.metadata['elapsed_time']
    
    # Second identical query - should hit cache
    response2 = await app.generate(query="What is RAG?")
    time2 = response2.metadata['elapsed_time']
    
    print(f"First query: {time1:.2f}s")
    print(f"Second query (cached): {time2:.2f}s")
    print(f"Speed improvement: {time1/time2:.1f}x")
    
    # Get cache statistics
    cache_stats = app._cache.get_stats()
    print(f"Cache stats: {cache_stats}")


if __name__ == "__main__":
    # Run the basic example
    asyncio.run(basic_usage_example())