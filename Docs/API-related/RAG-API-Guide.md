# RAG API Consumer Guide

This guide shows client-facing examples for the active unified RAG API. For a compact endpoint list, see [RAG_API_Documentation.md](RAG_API_Documentation.md). For implementation internals, see [RAG-Developer-Guide.md](../Code_Documentation/RAG-Developer-Guide.md).

## Base URL

```text
http://localhost:8000/api/v1/rag
```

## Authentication

Single-user deployments use:

```http
X-API-KEY: your-api-key
```

Multi-user deployments use:

```http
Authorization: Bearer <JWT>
```

All JSON examples below also require:

```http
Content-Type: application/json
```

## Available RAG Routes

- `POST /ablate`
- `POST /search`
- `POST /search/stream`
- `POST /batch`
- `POST /batch/resume/{checkpoint_id}`
- `GET /simple`
- `GET /advanced`
- `GET /capabilities`
- `GET /vlm/backends`
- `GET /features`
- `POST /feedback/implicit`
- `GET /health/simple`
- `GET /health`
- `GET /health/live`
- `GET /health/ready`
- `GET /cache/stats`
- `POST /cache/clear`
- `GET /cache/warm`
- `GET /metrics/summary`
- `GET /costs/summary`
- `GET /batch/jobs`
- `POST /quality-gate`
- `POST /baseline/save`
- `GET /regression/check`
- `POST /regression/check`

## Core Request Options

`POST /search` accepts `UnifiedRAGRequest`.

Common fields:

```typescript
type SearchMode = "fts" | "vector" | "hybrid";
type PublicSource = "media_db" | "notes" | "characters" | "chats" | "kanban" | "sql";

interface UnifiedSearchRequest {
  query: string;
  sources?: PublicSource[];
  search_mode?: SearchMode;
  top_k?: number;
  min_score?: number;
  keyword_filter?: string[];
  expand_query?: boolean;
  enable_reranking?: boolean;
  reranking_strategy?: "flashrank" | "cross_encoder" | "hybrid" | "llama_cpp" | "llm_scoring" | "two_tier" | "none";
  enable_citations?: boolean;
  enable_generation?: boolean;
  strategy?: "standard" | "agentic";
  session_id?: string;
}
```

## Basic Search

```bash
curl -X POST http://localhost:8000/api/v1/rag/search \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning concepts",
    "sources": ["media_db"],
    "search_mode": "hybrid",
    "top_k": 5
  }'
```

## Search With Generated Answer

```bash
curl -X POST http://localhost:8000/api/v1/rag/search \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do transformer models use attention?",
    "sources": ["media_db", "notes"],
    "search_mode": "hybrid",
    "enable_generation": true,
    "enable_citations": true,
    "top_k": 10
  }'
```

## Agentic Strategy

Agentic mode is selected through the unified search route.

```bash
curl -X POST http://localhost:8000/api/v1/rag/search \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare the competing explanations in my saved material",
    "sources": ["media_db", "notes"],
    "strategy": "agentic",
    "enable_generation": true,
    "agentic_top_k_docs": 5,
    "agentic_quote_spans": true
  }'
```

## Streaming Answers

`POST /search/stream` streams NDJSON and requires `enable_generation: true`.

```javascript
const response = await fetch("http://localhost:8000/api/v1/rag/search/stream", {
  method: "POST",
  headers: {
    "X-API-KEY": "your-api-key",
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    query: "Explain retrieval augmented generation",
    sources: ["media_db"],
    search_mode: "hybrid",
    enable_generation: true
  })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = "";

while (true) {
  const { done, value } = await reader.read();
  if (done) break;

  buffer += decoder.decode(value, { stream: true });
  let newlineIndex;
  while ((newlineIndex = buffer.indexOf("\n")) >= 0) {
    const line = buffer.slice(0, newlineIndex);
    buffer = buffer.slice(newlineIndex + 1);
    if (!line.trim()) continue;

    const event = JSON.parse(line);
    if (event.type === "delta") {
      process.stdout.write(event.text);
    }
  }
}
```

## Batch Search

```bash
curl -X POST http://localhost:8000/api/v1/rag/batch \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      "What is RAG?",
      "Explain vector search",
      "When should I use FTS?"
    ],
    "sources": ["media_db"],
    "search_mode": "hybrid",
    "max_concurrent": 5,
    "enable_checkpoint": true
  }'
```

Resume a checkpointed batch:

```bash
curl -X POST http://localhost:8000/api/v1/rag/batch/resume/checkpoint-id \
  -H "X-API-KEY: your-api-key"
```

## Convenience Search

Simple:

```bash
curl -G http://localhost:8000/api/v1/rag/simple \
  -H "X-API-KEY: your-api-key" \
  --data-urlencode "query=machine learning" \
  --data-urlencode "top_k=5"
```

Advanced:

```bash
curl -G http://localhost:8000/api/v1/rag/advanced \
  -H "X-API-KEY: your-api-key" \
  --data-urlencode "query=quantum computing notes" \
  --data-urlencode "with_citations=true" \
  --data-urlencode "with_answer=true"
```

## Capabilities

Use runtime discovery rather than hard-coding optional feature support:

```bash
curl http://localhost:8000/api/v1/rag/capabilities \
  -H "X-API-KEY: your-api-key"
```

Related discovery endpoints:

```bash
curl http://localhost:8000/api/v1/rag/features \
  -H "X-API-KEY: your-api-key"

curl http://localhost:8000/api/v1/rag/vlm/backends \
  -H "X-API-KEY: your-api-key"
```

## Operational Checks

```bash
curl http://localhost:8000/api/v1/rag/health/live \
  -H "X-API-KEY: your-api-key"

curl http://localhost:8000/api/v1/rag/health/ready \
  -H "X-API-KEY: your-api-key"

curl http://localhost:8000/api/v1/rag/cache/stats \
  -H "X-API-KEY: your-api-key"
```

## Response Shape

`UnifiedRAGResponse` commonly includes:

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
  "query": "machine learning concepts",
  "expanded_queries": [],
  "metadata": {"sources_searched": ["media_db"]},
  "timings": {"total": 0.25},
  "generated_answer": "Optional generated answer",
  "citations": []
}
```

## Client Tips

- Prefer `POST /search` for most use cases.
- Use `search_mode: "hybrid"` unless you explicitly need exact text matching or vector-only behavior. Vector behavior is source-dependent; sources without vector indexing use their source-specific retrieval path.
- Use `strategy: "agentic"` through `POST /search` when you need the agentic retrieval strategy.
- Use `POST /search/stream` for generated answers that should render incrementally.
- Call `/capabilities` at startup if your client exposes optional controls.
