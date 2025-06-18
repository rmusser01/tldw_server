# RAG Service for tldw_chatbook

A modern, efficient RAG (Retrieval-Augmented Generation) implementation optimized for single-user TUI applications.

## Overview

This RAG service replaces the monolithic `Unified_RAG_v2.py` with a clean, modular architecture designed specifically for the tldw_chatbook TUI application. It provides efficient search across media transcripts, chat history, and notes to enhance LLM responses with relevant context.

## Features

- **Modular Architecture**: Clean separation between retrieval, processing, and generation
- **Single-User Optimized**: Designed for local TUI usage with persistent connections and caching
- **Multiple Data Sources**: Search across media files, chat history, and notes
- **Hybrid Search**: Combines keyword (FTS5) and vector search for best results
- **Smart Processing**: Document deduplication, reranking, and context optimization
- **Flexible Generation**: Supports multiple LLM providers with streaming
- **Performance**: Built-in caching, metrics, and efficient resource usage

## Architecture

```
rag_service/
├── app.py          # Main RAGApplication orchestrator
├── config.py       # Configuration management (TOML integration)
├── types.py        # Type definitions and interfaces
├── retrieval.py    # Document retrieval strategies
├── processing.py   # Document processing and ranking
├── generation.py   # LLM response generation
├── cache.py        # Caching implementations
├── metrics.py      # Performance monitoring
├── utils.py        # Helper utilities
└── integration.py  # TUI integration helpers
```

## Quick Start

### 1. Configuration

Add RAG configuration to your `config.toml`:

```toml
[rag]
batch_size = 32
num_workers = 4
use_gpu = false

[rag.retriever]
fts_top_k = 10
vector_top_k = 10
hybrid_alpha = 0.7  # Balance between keyword and vector search

[rag.processor]
enable_reranking = true
max_context_length = 4096

[rag.generator]
default_temperature = 0.7
enable_streaming = true

[rag.cache]
enable_cache = true
cache_ttl = 3600
```

### 2. Basic Usage

```python
from tldw_chatbook.Services.rag_service.integration import RAGService

# Initialize service
rag_service = RAGService(
    media_db_path=Path("path/to/media.db"),
    chachanotes_db_path=Path("path/to/chachanotes.db"),
    llm_handler=your_llm_handler
)

# Initialize components
await rag_service.initialize()

# Generate RAG response
result = await rag_service.generate_answer(
    query="What was discussed about AI safety?",
    sources=["MEDIA_DB", "CHAT_HISTORY", "NOTES"]
)

print(result["answer"])
```

### 3. TUI Integration

```python
# In your TUI event handler
async def handle_rag_search(self, query: str):
    result = await self.rag_service.generate_answer(query)
    
    # Display in TUI
    self.update_results_widget(result["answer"])
    self.show_sources(result["sources"])
```

## Key Components

### Retrievers

- **MediaDBRetriever**: Searches media transcripts using SQLite FTS5
- **ChatHistoryRetriever**: Searches conversation history
- **NotesRetriever**: Searches user notes
- **VectorRetriever**: ChromaDB-based semantic search
- **HybridRetriever**: Combines keyword and vector search

### Processors

- **DefaultProcessor**: Basic deduplication and ranking
- **AdvancedProcessor**: Enhanced with snippet extraction and diversity scoring
- **StreamingProcessor**: For progressive UI updates

### Generators

- **LLMGenerator**: Integrates with existing LLM infrastructure
- **StreamingGenerator**: Supports streaming responses
- **FallbackGenerator**: Structured responses without LLM

## Performance Optimizations

### For Single-User TUI

1. **Persistent Connections**: Database connections stay open
2. **In-Memory Caching**: LRU cache for frequent queries
3. **Lazy Initialization**: Components load on-demand
4. **Reduced Threading**: Optimized for single-user access

### Caching Strategy

- Search results cached with configurable TTL
- Embeddings cached persistently
- Cache statistics available for monitoring

### Metrics

- Query latency tracking
- Cache hit rates
- Error monitoring
- Source diversity metrics

## Migration from Unified_RAG_v2

### Compatibility Mode

```python
# Use compatibility wrapper
from tldw_chatbook.Services.rag_service.integration import enhanced_rag_pipeline

result = await enhanced_rag_pipeline(
    query="Your question",
    api_choice="openai",
    media_db_path=media_path,
    chachanotes_db_path=notes_path
)
```

### Direct Migration

Replace:
```python
# Old
from tldw_chatbook.RAG_Search.Unified_RAG_v2 import enhanced_rag_pipeline
```

With:
```python
# New
from tldw_chatbook.Services.rag_service.integration import RAGService
rag_service = RAGService(...)
```

## Advanced Features

### Custom Retrieval Strategy

```python
class CustomRetriever(BaseRetriever):
    async def retrieve(self, query, filters=None, top_k=10):
        # Your custom retrieval logic
        return SearchResult(documents=[...])

rag_app.register_retriever(CustomRetriever(DataSource.CUSTOM))
```

### Streaming Responses

```python
# For TUI progress updates
async for partial_context in processor.process_streaming(results, query):
    update_ui(partial_context)
```

### Embedding Management

```python
# Embed new documents
await rag_service.embed_documents(
    source="MEDIA_DB",
    documents=[
        {"id": "1", "content": "Document text", "metadata": {...}}
    ]
)
```

## Troubleshooting

### Common Issues

1. **"No retriever registered"**
   - Ensure databases exist at specified paths
   - Check initialization completed successfully

2. **Poor search results**
   - Adjust `hybrid_alpha` (0=keyword only, 1=vector only)
   - Increase `fts_top_k` and `vector_top_k`
   - Enable reranking

3. **Slow performance**
   - Enable caching
   - Reduce `max_context_length`
   - Check database indexes

### Debug Mode

```python
# Enable debug logging
rag_service.config.log_level = "DEBUG"

# Get performance stats
stats = rag_service.get_stats()
print(f"Cache hit rate: {stats['cache']['hit_rate']}")
```

## Requirements

- Python 3.11+
- SQLite with FTS5 support
- ChromaDB (for vector search)
- FlashRank (optional, for reranking)

## Future Enhancements

- [ ] Add more embedding models
- [ ] Support for multimodal search
- [ ] Advanced caching strategies
- [ ] Query expansion techniques
- [ ] Feedback loop for improving results

## License

Same as tldw_chatbook (AGPLv3+)