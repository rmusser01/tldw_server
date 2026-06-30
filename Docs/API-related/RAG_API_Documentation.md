# RAG API Documentation

This is the concise endpoint reference for the active unified RAG API. For implementation internals and contributor guidance, see [RAG-Developer-Guide.md](../Code_Documentation/RAG-Developer-Guide.md).

## Base URL

```text
/api/v1/rag
```

The unified router is implemented in `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`. Health and operations routes are implemented in `tldw_Server_API/app/api/v1/endpoints/rag_health.py`.

## Authentication

- Single-user mode: `X-API-KEY: <key>`
- Multi-user mode: `Authorization: Bearer <JWT>`

Deployments may also enforce RBAC scopes and rate limits.

## Unified Routes

| Method | Route | Purpose |
| --- | --- | --- |
| `POST` | `/ablate` | Compare configured retrieval/generation variants for one query. |
| `GET` | `/capabilities` | Return runtime-supported RAG capabilities, defaults, limits, and quick-start payloads. |
| `GET` | `/vlm/backends` | List available VLM/table-processing backends. |
| `POST` | `/search` | Run the primary unified RAG search pipeline. |
| `POST` | `/feedback/implicit` | Record implicit interaction events such as clicks, expansions, copies, dwell time, and citation use. |
| `POST` | `/batch` | Run multiple RAG queries concurrently. |
| `GET` | `/simple` | Convenience search endpoint using query parameters. |
| `POST` | `/batch/resume/{checkpoint_id}` | Resume a checkpointed batch run. |
| `POST` | `/search/stream` | Stream generated answer events as NDJSON. |
| `GET` | `/advanced` | Convenience endpoint enabling common advanced options. |
| `GET` | `/features` | Return feature groups and request parameter names. |
| `GET` | `/health/simple` | Lightweight unified-pipeline health check. |

## Health And Operations Routes

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Comprehensive RAG health report. |
| `GET` | `/health/live` | Liveness probe. |
| `GET` | `/health/ready` | Readiness probe. |
| `GET` | `/cache/stats` | Cache statistics and recommendations. |
| `POST` | `/cache/clear` | Clear RAG caches. |
| `GET` | `/cache/warm` | Cache warmer status. |
| `GET` | `/metrics/summary` | Recent RAG metrics summary. |
| `GET` | `/costs/summary` | Cost tracking summary, when configured. |
| `GET` | `/batch/jobs` | Batch job state summary. |
| `POST` | `/quality-gate` | Run a quality-gate check. |
| `POST` | `/baseline/save` | Save a RAG regression baseline. |
| `GET` | `/regression/check` | Load a stored regression baseline by `baseline_id`. |
| `POST` | `/regression/check` | Compare current metrics against a stored regression baseline. |

## Primary Request And Response Models

- `UnifiedRAGRequest`
- `UnifiedRAGResponse`
- `UnifiedBatchRequest`
- `UnifiedBatchResponse`

Public request `sources` are:

- `media_db`
- `notes`
- `characters`
- `chats`
- `kanban`
- `sql`

`characters` and `chats` are separate public source values, but they currently share the character/chat retrieval backend.

Public request `search_mode` values are:

- `fts`
- `vector`
- `hybrid`

`search_mode` is accepted for all searches, but vector behavior depends on the selected sources and configured vector adapters. Sources without vector indexing use their source-specific retrieval path.

## Examples

### Unified Search

```bash
curl -X POST http://localhost:8000/api/v1/rag/search \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning applications",
    "sources": ["media_db", "notes"],
    "search_mode": "hybrid",
    "top_k": 10,
    "enable_reranking": true,
    "enable_generation": false
  }'
```

### Generated Answer

```bash
curl -X POST http://localhost:8000/api/v1/rag/search \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Summarize the strongest evidence for this topic",
    "sources": ["media_db"],
    "search_mode": "hybrid",
    "enable_generation": true,
    "enable_citations": true
  }'
```

### Agentic Strategy Through Unified Search

```bash
curl -X POST http://localhost:8000/api/v1/rag/search \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare the main arguments across my notes",
    "sources": ["notes"],
    "strategy": "agentic",
    "enable_generation": true
  }'
```

### Streaming Search

`POST /search/stream` returns NDJSON. Streaming requires `enable_generation: true`.

```bash
curl -N -X POST http://localhost:8000/api/v1/rag/search/stream \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Explain transformer attention",
    "search_mode": "hybrid",
    "enable_generation": true
  }'
```

Common event types include `delta`, `claims_overlay`, and `final_claims`.

### Batch Search

```bash
curl -X POST http://localhost:8000/api/v1/rag/batch \
  -H "X-API-KEY: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["What is RAG?", "Explain vector search"],
    "sources": ["media_db"],
    "search_mode": "hybrid",
    "max_concurrent": 5,
    "enable_checkpoint": true
  }'
```

### Resume A Batch

```bash
curl -X POST http://localhost:8000/api/v1/rag/batch/resume/checkpoint-id \
  -H "X-API-KEY: your-api-key"
```

### Capabilities And Health

```bash
curl http://localhost:8000/api/v1/rag/capabilities \
  -H "X-API-KEY: your-api-key"

curl http://localhost:8000/api/v1/rag/health \
  -H "X-API-KEY: your-api-key"
```

## Error Handling

Typical responses:

- `400` for invalid request fields or unsupported combinations.
- `401` or `403` for missing credentials, invalid credentials, or missing permissions.
- `429` for rate-limit failures.
- `500` for unexpected service errors.
