# Infrastructure

Centralized helpers for shared runtime infrastructure. Today this module focuses on Redis connectivity with a robust in‑memory fallback and metrics instrumentation. Other modules (Embeddings orchestrator, backpressure guards, privilege cache) import clients exclusively via this package.

## 1. Descriptive of Current Feature Set

- Purpose: Provide a single, dependable factory for Redis clients (async and sync) with seamless fallback to an in‑memory stub when Redis is unavailable. Emits standard metrics for observability.
- Capabilities:
  - Async and sync Redis clients: `create_async_redis_client`, `create_sync_redis_client`.
  - In‑memory stub with feature coverage for module needs: strings, hashes, sets, sorted sets, TTL, simple pipelines, streams (XADD/XRANGE/XLEN/XDEL), basic Lua script load/eval for rate‑limiter logic.
  - Metrics: records connection attempts, durations, errors, and fallbacks via the Metrics registry.
  - Graceful close helpers: `ensure_async_client_closed`, `ensure_sync_client_closed`.
- Inputs/Outputs:
  - Inputs: Optional `preferred_url`, config/env keys, `redis_kwargs` for SSL/password/etc.
  - Outputs: A client exposing the subset of redis‑py API used by the codebase; either real Redis or the in‑memory stub.
- Related Endpoints/Modules (usage examples):
  - Backpressure dependency (Embeddings orchestrator depth/age): tldw_Server_API/app/api/v1/API_Deps/backpressure.py:16, tldw_Server_API/app/api/v1/API_Deps/backpressure.py:24
  - Embeddings API (tenant RPS + DLQ admin): tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py:78, tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py:474
  - Privilege maps cache invalidation (sync client): tldw_Server_API/app/core/PrivilegeMaps/cache.py:171
- Related Schemas: N/A (this module exposes service clients, not Pydantic models).

## 2. Technical Details of Features

- Architecture & Data Flow:
  - Caller requests a Redis client with optional `preferred_url` and `context` label.
  - Factory resolves the URL (config/env precedence), attempts to connect/ping real Redis, and records metrics.
  - On failure (or if redis is not installed) and when `fallback_to_fake=True`, it returns an in‑memory stub implementing the subset of Redis features used by the application.
- Key Classes/Functions (entry points):
  - `create_async_redis_client(preferred_url=None, decode_responses=True, fallback_to_fake=True, context="default", redis_kwargs=None)`
  - `create_sync_redis_client(preferred_url=None, decode_responses=True, fallback_to_fake=True, context="default", redis_kwargs=None)`
  - `ensure_async_client_closed(client)`, `ensure_sync_client_closed(client)`
  - In‑memory clients: `InMemoryAsyncRedis`, `InMemorySyncRedis` (with thin pipeline helpers)
- Dependencies:
  - Optional runtime: `redis` / `redis.asyncio` (import‑guarded). When unavailable, factory can still return in‑memory clients.
  - Metrics (optional during early startup): hooks into `tldw_Server_API.app.core.Metrics.metrics_manager.get_metrics_registry`.
- Data Models & DB:
  - No relational tables. Interacts with Redis using well‑known keys used by other modules, e.g.:
    - Embeddings streams: `embeddings:chunking`, `embeddings:embedding`, `embeddings:storage`, `embeddings:content`, plus `:dlq` variants.
    - Rate limit counters: `rl:req:{user}:{window}`, `rl:tok:{user}:{YYYYMMDD}`.
    - Tenant RPS: `embeddings:tenant:rps:{user}`, `ingest:tenant:rps:{user}:{ts}`.
    - Privilege cache generation key/channel: e.g., `privilege:cache:generation`, `privilege:cache:invalidate`.
- Configuration (precedence and keys):
  - URL resolution: `preferred_url` arg → `settings.get('EMBEDDINGS_REDIS_URL')` → `settings.get('REDIS_URL')` → `ENV[EMBEDDINGS_REDIS_URL|REDIS_URL]` → default `redis://localhost:6379`.
  - Module‑specific consumers may reference additional env/keys (e.g., backpressure limits `EMB_BACKPRESSURE_MAX_DEPTH`, `EMB_BACKPRESSURE_MAX_AGE_SECONDS`; privilege cache `PRIVILEGE_CACHE_REDIS_URL`).
- Concurrency & Performance:
  - In‑memory stubs are protected by an `asyncio.Lock` (async) or `threading.Lock` (sync) to avoid races when used concurrently.
  - Streams and sorted‑set operations in the stub are implemented for the project’s use cases (XRANGE/XLEN/XADD; ZADD/ZCARD/ZREMRANGEBYSCORE), adequate for tests and single‑node dev.
- Error Handling:
  - Connection errors emit metrics and fall back to stub when allowed; callers can disable fallback via `fallback_to_fake=False` (raises the original error).
  - Close helpers are best‑effort and safe to call on both real and stub clients.
- Security:
  - Secrets come from env or config; do not log credentials. Logs include a `context` label only.
  - Fallback avoids external dependencies in CI/dev, keeping tests hermetic.
- Metrics emitted (names defined in Metrics module):
  - `infra_redis_connection_attempts_total{mode,context,outcome}`
  - `infra_redis_connection_duration_seconds{mode,context,outcome}`
  - `infra_redis_connection_errors_total{mode,context,error}`
  - `infra_redis_fallback_total{mode,context,reason}`

## 3. Developer-Related/Relevant Information for Contributors

- Folder Structure:
  - `redis_factory.py` — main factory and in‑memory client implementations.
  - `__init__.py` — module docstring and export surface (import helpers from here).
- Extension Points:
  - Add new infra factories using the same pattern: centralized URL/config resolution, optional metrics, and an in‑memory stub when feasible.
  - If extending the in‑memory Redis API surface, keep it minimal and driven by concrete caller needs; add tests.
- Coding Patterns:
  - Always import from `tldw_Server_API.app.core.Infrastructure.redis_factory`.
  - Pass a meaningful `context` string to aid debugging/metrics.
  - Use `ensure_async_client_closed`/`ensure_sync_client_closed` in `finally` blocks for real clients.
- Tests:
  - Metrics behavior tests: tldw_Server_API/tests/Infrastructure/test_redis_factory_metrics.py:17, tldw_Server_API/tests/Infrastructure/test_redis_factory_metrics.py:60, tldw_Server_API/tests/Infrastructure/test_redis_factory_metrics.py:108
  - Embeddings orchestrator tests exercise streams/queues against the stub (e.g., DLQ/orchestrator snapshot): see tests under `tldw_Server_API/tests/Embeddings/`.
- Local Dev Tips:
  - No Redis running? Do nothing — the factory falls back to an in‑memory client automatically.
  - To use a real Redis, set `REDIS_URL` (or `EMBEDDINGS_REDIS_URL`) and ensure `redis` extras are installed.
  - Example (async):

    ```python
    from tldw_Server_API.app.core.Infrastructure.redis_factory import create_async_redis_client, ensure_async_client_closed

    client = await create_async_redis_client(context="demo")
    try:
        await client.xadd("embeddings:embedding", {"doc_id": "123", "ev": "enqueue"})
        depth = await client.xlen("embeddings:embedding")
    finally:
        await ensure_async_client_closed(client)
    ```

  - Example (sync):

    ```python
    from tldw_Server_API.app.core.Infrastructure.redis_factory import create_sync_redis_client

    client = create_sync_redis_client(context="demo-sync")
    client.incr("rl:req:demo:0")
    ```
- Pitfalls & Gotchas:
  - In‑memory client does not implement full Redis feature parity. Pub/Sub is not available in the stub.
  - If you rely on strict Redis behavior (e.g., exact stream IDs, ordering guarantees, Lua semantics), add integration tests with a real Redis and gate them via env.
  - When `fallback_to_fake=False`, be prepared to handle connection exceptions during startup.
- Follow-up candidates:
  - Consider factories for additional infra (Postgres pool, object storage) following the same pattern.
  - Expand metrics (pool usage, cache hit/miss) if/when additional factories are introduced.
