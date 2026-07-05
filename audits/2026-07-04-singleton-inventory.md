# Process-Global Singleton Inventory

**Date:** 2026-07-04
**Context:** Task 3 of `Docs/superpowers/plans/2026-07-04-test-suite-improvement-implementation-plan.md`, targeting finding RA5 of `audits/2026-07-04-test-suite-audit-round2.md` — the top-severity (all-High) defect class of process-global state leaking across test modules (#2580 service/DB caches, #2581 drain/lifecycle singletons, #2585 app-module identity).

The `tests/_plugins/singleton_guard.py` plugin watches the highest-risk subset of these at test-module boundaries (opt-in via `TLDW_SINGLETON_GUARD=warn|error`).

## Watched by the guard (initial set)

Ranked by mutability × cross-test reach × adjacency to the three known bugs.

| # | Import path : global | Defect | Reader | Existing reset hook |
|---|---|---|---|---|
| 1 | `core/DB_Management/media_db/runtime/defaults:content_db_backend` | #2580 | identity | `reset_media_runtime_defaults()` |
| 2 | `api/v1/API_Deps/ChaCha_Notes_DB_Deps:_chacha_db_instances` | #2580 | len | `shutdown_chacha_resources()` + autouse `reset_chacha_shutdown_state` |
| 3 | `app.main` module identity | #2585 | `id(module)` | `restore_app_main()` (manual) |
| 4 | `core/RAG/rag_service/semantic_cache:_SHARED_CACHES` | #2580 | len | `clear_shared_caches(namespace=None)` |
| 5 | `core/Embeddings/connection_pool:_pool_manager` | #2581 | is-set | `close_all_pools()` |
| 6 | `core/Embeddings/async_embeddings:_async_service_fallback` | #2581 | is-set | atexit only |
| 7 | `core/Embeddings/request_batching:_batcher_fallback` | #2581 | is-set | `RequestBatcher.shutdown()` |

## Full inventory (not all watched yet)

The following are process-global mutable state that *could* leak; they are documented so the watchlist can be expanded as needed. Items marked **NO reset** are the highest-value future additions.

**DB / service caches (#2580 class):**
- `core/RAG/rag_service/advanced_cache:_shared_semantic_cache` — reset via `register_semantic_cache()` (no null-reset)
- `core/Evaluations/unified_evaluation_service:_service_instance`, `db_adapter:_global_adapter` — **NO reset**; constructed with a `db_path` at first call
- `core/Agent_Orchestration/db_factory` — `@lru_cache(maxsize=64)` DB-handle factory; `.cache_clear()` not wired to conftest
- `core/RAG/rag_service/analytics_db:_analytics_db`, `agentic_execution:_STRUCT_DB` — **NO reset**
- `core/Chat/provider_manager:_provider_manager`, `core/LLM_Calls/adapter_registry:_registry`, `embeddings_adapter_registry:_emb_registry` — **NO reset**
- `core/Storage/__init__:_storage_backend_instance` — helper sets to None

**Drain / lifecycle singletons (#2581 class):**
- `core/TTS/tts_resource_manager:_resource_manager` — `close_resource_manager()` intentionally does *not* close the global
- `core/TTS/tts_service_v2:_service_instance`, `voice_manager:_voice_manager`, `Scheduler/scheduler:_GLOBAL_SCHEDULER` — background tasks bound to first event loop
- Resource-governor singletons `_rg_*_governor` across `Chat/`, `Embeddings/`, `Evaluations/`, `MCP_unified/auth/`, `Web_Scraping/`, `Usage/` rate limiters — **NO reset**
- Daily-ledger singletons in `Usage/`, `Workflows/`, `Evaluations/`, `Resource_Governance/` — **NO reset**

**Already-covered infrastructure (autouse reset in conftest, low risk):**
- `core/config:_CONFIG_PARSER_CACHE` + `@lru_cache` fns — `clear_config_cache()` (autouse)
- `core/http_client:{_AIOHTTP_SESSION_CACHE,_HTTPX_ASYNC_CLIENT_CACHE,_HTTPX_CLIENT_CACHE}` — `shutdown_http_client()`
- App lifecycle `draining` flag lives on `app.state` (per-app-instance), reset by `reset_lifecycle_state()` — **not** a module global, so not the #2581 source

## Verification (this PR)

- **Mechanism:** 5 synthetic tests in `tests/Infrastructure/test_singleton_guard_mechanism.py` prove leak detection, false-positive avoidance (cleanup within a module), skipping unloaded modules, error-mode nonzero exit, and reader helpers.
- **No false positives:** Config (166), Embeddings_isolated (117), and RAG (653) — 936 tests across many module boundaries — produced **zero** guard warnings, confirming the existing autouse reset fixtures cover these lanes. Error mode over a real-app lane exits 0.
- The guard is opt-in and a no-op by default; it exists to catch *regressions* of the #2580/#2581/#2585 classes and to be pointed at broader lanes / more watched globals over time.
