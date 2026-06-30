# Embeddings

## 1. Descriptive of Current Feature Set

- Purpose: OpenAI-compatible embeddings API with production safeguards and a Redis Streams worker for media embeddings stages (Jobs remains the root status/billing record). Focus on reliability (circuit breaker, retries), performance (TTL cache, connection pooling, batching), and observability (metrics, health, DLQ tools).
- Covering ADR: `Docs/ADR/022-embeddings-api-and-media-pipeline.md`
- Capabilities:
  - OpenAI-compatible embeddings endpoint with provider auto-detect (OpenAI, HuggingFace/Transformers, ONNX, Local API)
  - TTL cache, request batching, connection pooling, per-provider circuit breakers
  - Policy/fallback chain, model warmup/download admin ops, rate limiting/backpressure
  - ChromaDB per-user vector storage; optional pgvector via RAG adapters
  - Re-embed/compaction services; DLQ visibility and requeue controls
- Inputs/Outputs:
  - Input: `CreateEmbeddingRequest` with `input` (str | list[str] | token arrays), `model`, optional `encoding_format`, `dimensions`
  - Output: `CreateEmbeddingResponse` (OpenAI-style) with `data[*].embedding` floats or base64
- Related Endpoints:
  - Create embeddings — tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py:1625
  - List models — tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py:2115
  - Providers config — tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py:1240
  - Cache clear — tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py:2318
  - Collections (create/list/delete/stats) — tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py:2350,2405,2420,2438
  - Health — tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py:2488
  - DLQ/state/control — tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py:2721,3071,3157
  - Media embeddings — tldw_Server_API/app/api/v1/endpoints/media_embeddings.py:427 (generate), 367 (status), 658 (delete)
- Related Schemas:
  - `CreateEmbeddingRequest` — tldw_Server_API/app/api/v1/schemas/embeddings_models.py:16
  - `CreateEmbeddingResponse` — tldw_Server_API/app/api/v1/schemas/embeddings_models.py:42

## 2. Technical Details of Features

- Architecture & Data Flow:
  - API: `embeddings_v5_production_enhanced.py` handles request validation, caching, policy, circuit breaker, metrics, and admin ops
  - Engine: `Embeddings_Server/Embeddings_Create.py` provides provider adapters, batching, warmup, and model storage resolution
  - Vector store: `ChromaDB_Library.py` manages per-user collections and safe pathing; pgvector via RAG vector store factory
  - Redis worker: `services/redis_worker.py` (media embeddings via Redis Streams; root Jobs status)
- Key Components:
  - Circuit breaker and registry — `tldw_Server_API/app/core/Infrastructure/circuit_breaker.py`
  - Connection pooling — `connection_pool.py` (aiohttp clients)
  - Request batching — `request_batching.py`
  - Cache — `multi_tier_cache.py` (TTL, size, hit/miss metrics)
  - Rate limiting/backpressure — `rate_limiter.py`, per-endpoint limiters
  - Services — `services/redis_worker.py` (media + content stages), `services/jobs_worker.py` (legacy Jobs worker), `services/vector_compactor.py`
- Data Models & DB:
  - ChromaDB per-user collections; helpers in `vector_store_meta_db.py`, `vector_store_batches_db.py`
- Configuration:
  - Throughput/limits: `EMBEDDINGS_MAX_BATCH_SIZE`, `EMBEDDINGS_CONNECTION_POOL_SIZE`, `EMBEDDINGS_REQUEST_TIMEOUT`, `EMBEDDINGS_MAX_RETRIES`
  - Cache: `EMBEDDINGS_CACHE_TTL_SECONDS`, `EMBEDDINGS_CACHE_MAX_SIZE`, `EMBEDDINGS_CACHE_CLEANUP_INTERVAL`, `EMBEDDINGS_TTLCACHE_DAEMON`
  - Policy/fallback: `EMBEDDINGS_ENFORCE_POLICY`, `EMBEDDINGS_ENFORCE_POLICY_STRICT`, `EMBEDDINGS_ALLOW_FALLBACK_WITH_HEADER`
  - Dimensions: `EMBEDDINGS_DIMENSION_POLICY` (reduce|pad|ignore)
  - Backpressure: `EMBEDDINGS_RATE_LIMIT`, `EMB_BACKPRESSURE_*`, Redis `REDIS_URL`
  - Model ops: `PRELOAD_EMBEDDING_MODELS`, `AUTO_DOWNLOAD_MODELS`, `TRUSTED_HF_REMOTE_CODE_MODELS`
  - Redis pipeline: `EMBEDDINGS_REDIS_STREAM_*`, `EMBEDDINGS_REDIS_GROUP_*`, `EMBEDDINGS_REDIS_WORKERS_*`, `EMBEDDINGS_REDIS_ALLOW_STUB`
  - Testing: `TESTING`, `USE_REAL_OPENAI_IN_TESTS`, `CHROMADB_FORCE_STUB`
- Concurrency & Performance:
  - Batching by `MAX_BATCH_SIZE`; connection reuse; async I/O; per-provider breakers
  - Cache avoids duplicate embeddings across requests; key includes text/provider/model/dimensions
- Error Handling:
  - 4xx for validation/policy; 429 rate limits; 5xx or 503 on provider failures (with breaker trip)
  - DLQ encryption optional via `dlq_crypto.py`
- Security:
  - AuthNZ on all endpoints; RBAC limiter on create; admin-only for cache/model admin ops
  - No logging of raw inputs or secrets; audit via adapters when present

## 3. Developer-Related/Relevant Information for Contributors

- Folder Structure:
  - API endpoint and admin ops in endpoints file; engine under `Embeddings_Server/`; infra in core files; workers/services/DB helpers organized by concern
- Extension Points:
  - Add a provider adapter in `Embeddings_Create.py`; register in provider factory; expose in models listing
  - Extend admin ops for model warmup/download; wire policy in endpoints
  - Add worker stages or DLQ processors by defining message schema and worker loop
- Coding Patterns:
  - Lazy-load heavy deps; avoid import-time failures; prefer DI for storage/DB
  - Emit Prometheus metrics via provided helpers; use circuit breaker wrappers for provider calls
  - Keep logs high-level; redact payloads; attach audit context where available
- Tests:
  - E2E — tldw_Server_API/tests/e2e/test_embeddings_e2e.py:1
  - Claims/Chroma helpers — tldw_Server_API/tests/Claims/test_claim_embeddings_chroma.py:1
  - RAG integration — tldw_Server_API/tests/RAG_NEW/integration/test_retriever_pgvector_multi_search.py:1
  - Usage reporting — tldw_Server_API/tests/Admin/test_llm_usage_endpoints.py:1
- Local Dev Tips:
  - Create embeddings: `POST /api/v1/embeddings` with `{ "model": "text-embedding-3-small", "input": "hello" }`
  - List models/providers: `GET /api/v1/embeddings/models` and `/embeddings/providers-config`
  - Use `x-provider: huggingface` and a HF model id to target Transformers backend
- Pitfalls & Gotchas:
  - `dimensions` applies only to models that support it (OpenAI t-e-3 family); HF/ONNX outputs are fixed
  - Policy enforcement may block unknown providers/models; set enforcement flags accordingly in dev
  - Backpressure may reject requests when queues are deep; check health and stage status endpoints
- Follow-up candidates:
  - Expand pgvector-first pathway and tests; unify Chroma/pgvector adapters
  - Stabilize re-embed scheduling; improve compactor heuristics
  - Extend providers list and add auto-tuning for batch sizes per model

---

Example Quick Start

```bash
curl -X POST http://127.0.0.1:8000/api/v1/embeddings \
  -H "Content-Type: application/json" -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  -d '{"model": "text-embedding-3-small", "input": "Embeddings test"}'
```
