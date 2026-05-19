# Unified RAG Pipeline Examples

These examples use the active unified HTTP routes or the active `unified_rag_pipeline(...)` function. For extension work, see `Docs/Code_Documentation/RAG-Developer-Guide.md`.

## HTTP Search

```bash
curl -X POST http://localhost:8000/api/v1/rag/search \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is retrieval augmented generation?",
    "sources": ["media_db"],
    "search_mode": "hybrid",
    "top_k": 5
  }'
```

## Search With Query Expansion And Reranking

```bash
curl -X POST http://localhost:8000/api/v1/rag/search \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is ML?",
    "sources": ["media_db", "notes"],
    "search_mode": "hybrid",
    "expand_query": true,
    "expansion_strategies": ["acronym", "synonym"],
    "enable_reranking": true,
    "reranking_strategy": "flashrank",
    "top_k": 10
  }'
```

## Search With Generated Answer And Citations

```bash
curl -X POST http://localhost:8000/api/v1/rag/search \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize my notes on transformer attention",
    "sources": ["notes"],
    "search_mode": "hybrid",
    "enable_generation": true,
    "enable_citations": true,
    "citation_style": "apa"
  }'
```

## Agentic Strategy

```bash
curl -X POST http://localhost:8000/api/v1/rag/search \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare the main claims across these saved sources",
    "sources": ["media_db", "notes"],
    "strategy": "agentic",
    "enable_generation": true,
    "agentic_top_k_docs": 5,
    "agentic_quote_spans": true
  }'
```

## Streaming Generated Answer

```bash
curl -N -X POST http://localhost:8000/api/v1/rag/search/stream \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain vector search in this project",
    "sources": ["media_db"],
    "search_mode": "hybrid",
    "enable_generation": true
  }'
```

## Batch Search

```bash
curl -X POST http://localhost:8000/api/v1/rag/batch \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      "What is RAG?",
      "Explain hybrid search",
      "When should vector retrieval be used?"
    ],
    "sources": ["media_db"],
    "search_mode": "hybrid",
    "max_concurrent": 5,
    "enable_checkpoint": true
  }'
```

## Programmatic Use

```python
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import unified_rag_pipeline

result = await unified_rag_pipeline(
    query="What is retrieval augmented generation?",
    sources=["media_db", "notes"],
    search_mode="hybrid",
    top_k=10,
    expand_query=True,
    enable_reranking=True,
    reranking_strategy="flashrank",
    enable_generation=True,
    enable_citations=True,
)

print(result.generated_answer)
print(result.documents)
```

## Programmatic Batch

```python
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import unified_batch_pipeline

batch = await unified_batch_pipeline(
    queries=["What is RAG?", "Explain vector search"],
    sources=["media_db"],
    search_mode="hybrid",
    max_concurrent=3,
)

successful = sum(1 for result in batch if not result.errors)
failed = len(batch) - successful

print(successful, failed)
```

## Response Shape

```json
{
  "documents": [
    {
      "id": "doc_123",
      "content": "Matched content...",
      "metadata": {"title": "Example"},
      "score": 0.92
    }
  ],
  "query": "What is retrieval augmented generation?",
  "expanded_queries": [],
  "metadata": {"sources_searched": ["media_db"]},
  "timings": {"total": 0.25},
  "generated_answer": "Optional generated answer",
  "citations": []
}
```
