# Embeddings ADR Candidate Confirmation Audit - 2026-06-04

**Related task:** TASK-2261
**Follow-up:** TASK-2262
**Inventory row:** INV-032 in `Docs/ADR/inventory/2026-06-03-decision-inventory.md`
**Source:** `tldw_Server_API/app/core/Embeddings/README.md`

## Candidate under review

INV-032 summarized the Embeddings convention as:

> Embeddings use OpenAI-compatible API safeguards, provider auto-detect/adapters, cache/batching/breakers, and Redis Streams workers while Jobs remains the root status/billing record.

## Confirmation result

Current governing, with bounded scope. Create one accepted ADR via TASK-2262, expected as `Docs/ADR/022-embeddings-api-and-media-pipeline.md`.

The future ADR should cover:

1. OpenAI-compatible embeddings request/response semantics and endpoint safeguards.
2. Provider resolution by explicit header, provider-qualified model id, or model-name heuristic, plus allowlist and unsupported-provider guards.
3. Optional LLM adapter-registry routing when enabled, with legacy provider-config/direct provider execution as the fallback path.
4. Endpoint reliability controls: keyed TTL cache, request batching, provider-scoped circuit breakers, connection reuse, provider fallback rules, and health/admin breaker visibility.
5. Media embeddings pipeline ownership: core Jobs creates and exposes the durable root `embeddings_pipeline` record, while Redis Streams carries chunking, embedding, storage, and content stage messages.

## Evidence

- `tldw_Server_API/app/api/v1/schemas/embeddings_models.py:18` defines an OpenAI-style `CreateEmbeddingRequest` with forbidden extra fields, string/list/token-array input, required model, `encoding_format`, `dimensions`, and `user`.
- `tldw_Server_API/app/api/v1/schemas/embeddings_models.py:64` defines an OpenAI-style list response with embedding data and usage.
- `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py:2241` wires the main create endpoint behind embeddings create rate limits and API-call billing limits.
- `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py:2272` resolves provider input, including explicit `x-provider`, provider-qualified model ids, and HuggingFace-style model-name heuristics.
- `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py:2323` validates requested dimensions before provider execution.
- `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py:2331` rejects empty inputs and enforces list/token-array shape limits before policy checks.
- `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py:2372` enforces per-model token limits with fail-fast `input_too_long` responses.
- `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py:2398` applies provider/model allowlists after input validation, and `:2419` rejects recognized but unimplemented providers with 501 instead of silently falling through.
- `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py:2552` optionally routes through the LLM embeddings adapter registry when `LLM_EMBEDDINGS_ADAPTERS_ENABLED` is truthy.
- `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py:2633` uses a fallback chain on provider failure, while explicit `x-provider` disables fallback by default unless `EMBEDDINGS_ALLOW_FALLBACK_WITH_HEADER` is enabled.
- `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py:1867` wraps provider execution in provider-scoped circuit breakers; `:3548` exposes breaker status in health output; `:3611` and `:3623` expose admin breaker status/reset endpoints.
- `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py:2057` performs cache lookup, uncached batching, provider execution, response-count validation, and cache writeback.
- `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py:1281` partitions local API cache identity by backend URL while stripping credentials and sensitive query parameters.
- `tldw_Server_API/app/core/Embeddings/request_batching.py:87` defines the request batcher; `:166` queues requests per provider/model/config; `:313` collects batches by size/timeout; `:364` processes and distributes batched results.
- `tldw_Server_API/app/api/v1/endpoints/media_embeddings.py:124` forces media embedding job backend ownership to core Jobs, ignoring other backend override values.
- `tldw_Server_API/app/api/v1/endpoints/media_embeddings.py:663` and `:749` enqueue single and batch media embedding work through `EmbeddingsJobsAdapter`.
- `tldw_Server_API/app/core/Embeddings/jobs_adapter.py:129` creates root Jobs records with `job_type="embeddings_pipeline"` and enqueues Redis stages instead of creating durable stage Jobs.
- `tldw_Server_API/app/core/Embeddings/services/redis_worker.py:86` handles Redis stage messages and `:131`, `:160`, `:187`, and `:202` run chunking, embedding, storage, and content stage handlers.
- `tldw_Server_API/app/core/Embeddings/services/redis_worker.py:172`, `:197`, and `:240` update the root Jobs result/status on stage progress, completion, or failure.
- `tldw_Server_API/app/core/Embeddings/services/jobs_worker.py:1` explicitly labels the Jobs worker as legacy while stating root Jobs remain the status/billing record.
- `tldw_Server_API/tests/Embeddings/test_embeddings_policy.py:19` and `:65` cover token-limit and allowlist rejection.
- `tldw_Server_API/tests/Embeddings/test_embeddings_fallback.py:28` and `:86` cover fallback behavior and explicit-header fallback suppression.
- `tldw_Server_API/tests/Embeddings/test_embeddings_endpoint_cache_identity.py:19` and `:129` cover cache backend identity sanitization and distinct local API backend cache keys.
- `tldw_Server_API/tests/Embeddings/test_request_batching.py:24` covers provider credentials through batched requests.
- `tldw_Server_API/tests/Embeddings/test_embeddings_jobs_adapter.py:21` and `:64` cover idempotent root job creation and status derivation from the root Job.
- `tldw_Server_API/tests/Embeddings/test_embeddings_redis_worker.py:8` and `:62` cover Redis stage handoff and root-job completion ordering.

## Caveats for the ADR

- Do not claim Redis Streams owns durable status or billing. Redis Streams is the stage-delivery mechanism; root Jobs records remain the durable status surface. The direct create endpoints also have API-call billing limits, Resource Governor token reservation, and best-effort usage logging, but the media pipeline billing behavior was not fully audited as a separate accounting decision.
- Do not claim all providers route through one adapter registry. The adapter registry is optional and currently gated by `LLM_EMBEDDINGS_ADAPTERS_ENABLED`; legacy provider config/direct execution remains the fallback path and is still current.
- Do not import INV-027's local provider URL policy. Embeddings `local_api` accepts configured/API URL inputs in provider config paths and partitions cache keys by sanitized backend identity.
- Do not overstate cache architecture. The main endpoint uses the local keyed `TTLCache`; broader multi-tier cache modules exist, but this confirmation only supports a bounded claim around endpoint cache identity and supporting cache behavior.
- Do not turn ChromaDB versus pgvector storage into this ADR. The README mentions Chroma per-user collections and optional pgvector via RAG adapters, but storage-backend evolution should be a separate ADR if needed.
- Do not make the legacy Jobs worker the primary pipeline path. The current module explicitly labels it legacy; the accepted decision should focus on root Jobs ownership plus Redis Streams stage delivery.

## Recommendation

Create TASK-2262 for one accepted Embeddings ADR. Scope it to the API/provider safeguards, reliability controls, and Jobs-root/Redis-stage media pipeline ownership confirmed above. Update INV-032 to reference TASK-2261 confirmation and TASK-2262 backfill.
