# Services Module - Overview and Operational Guide

This document summarizes the background services in `tldw_server`, their responsibilities, configuration, and production hardening notes. It reflects the code in `tldw_Server_API/app/services` as of v0.1.0.

## At-a-Glance

- Core workers: Chatbooks Jobs worker, Jobs metrics gauges worker, Claims rebuild worker
- Aggregators: API request usage aggregator, LLM usage aggregator
- Quotas: User storage quota service (filesystem + DB)
- Lifecycle helpers: focused `startup_*.py`, `shutdown_*.py`, and `lifespan_*.py` modules extracted from the FastAPI lifespan
- Web scraping: Enhanced scraping service (Playwright-based) with compatibility fallback behavior
- Placeholders (non-production): document/ebook/podcast/XML processors
- Ephemeral storage: process-local in-memory store with TTL and capacity limits

## Startup/Shutdown

`tldw_Server_API/app/main.py` remains the FastAPI entry point, but startup and shutdown orchestration is now delegated to lifecycle service modules:

- `tldw_Server_API/app/services/lifespan_startup_sequence.py` runs the startup sequence.
- `tldw_Server_API/app/services/lifespan_shutdown_sequence.py` runs the shutdown sequence.
- `tldw_Server_API/app/services/lifespan_worker_runtime_state.py` carries task, stop-event, scheduler, and resource handles from startup into shutdown.
- Focused `startup_*.py` and `shutdown_*.py` helpers own individual lifecycle slices such as auth initialization, validation, worker bootstrap, worker late-stop handling, coordinated shutdown, and resource cleanup.

Background loops should support graceful stop via an `asyncio.Event` where possible. Startup helpers should return or register task/stop-event handles, and shutdown helpers should set stop events, wait with bounded timeouts, cancel as fallback, and clear stale handles. See `tldw_Server_API/app/services/README.md` for the package-level lifecycle service pattern and checklist.

## Services

### Storage Quota Service
- File: `tldw_Server_API/app/services/storage_quota_service.py`
- Purpose: Tracks per-user storage (`USER_DATA_BASE_PATH` and optionally per-user ChromaDB). Updates usage in `users` table. Provides quota checks, recalculation, temp cleanup, and breakdown.
- Key methods:
  - `calculate_user_storage(user_id)` - scans disk, updates DB (async + threadpool)
  - `check_quota(user_id, new_bytes)` - cached, emits gauges
  - `update_usage(user_id, bytes_delta, operation)` - safe increments with floor at 0
  - `cleanup_temp_files(user_id?, older_than_hours=24)`
- Caching: `TTLCache` for quota and storage results (5-10 minutes)
- Backend: Works with SQLite and PostgreSQL via `DatabasePool`
- Notes: All heavy filesystem ops run in a small threadpool. Methods self-init on first use.

### Jobs Metrics Gauges
- File: `tldw_Server_API/app/services/jobs_metrics_service.py`
- Purpose: Periodically emits metrics: stale processing counts, queue depths, and time to expiry for processing jobs.
- Env:
  - `JOBS_METRICS_INTERVAL_SEC` (default 30)
  - `JOBS_TTL_ENFORCE` with `JOBS_TTL_AGE_SECONDS`, `JOBS_TTL_RUNTIME_SECONDS`, `JOBS_TTL_ACTION` (cancel|fail)

### Chatbooks Core Jobs Worker
- File: `tldw_Server_API/app/services/core_jobs_worker.py`
- Purpose: Processes Chatbooks import/export jobs from the core jobs backend with lease renewal and cancellation checks; writes job result and updates per-user Chatbooks job records.
- Env:
  - `CHATBOOKS_JOBS_BACKEND` (core-only; overrides ignored)
  - `CHATBOOKS_CORE_WORKER_ENABLED` (true/false)
  - `JOBS_POLL_INTERVAL_SECONDS`, `JOBS_LEASE_SECONDS`, `JOBS_LEASE_RENEW_SECONDS`, `JOBS_LEASE_RENEW_JITTER_SECONDS`

### Claims Rebuild Worker
- File: `tldw_Server_API/app/core/Claims_Extraction/claims_rebuild_service.py`
- Purpose: Background thread pool to rebuild claims for media, using chunking and the claims extractor.
- Env (via app settings): `CLAIMS_REBUILD_ENABLED`, `CLAIMS_REBUILD_INTERVAL_SEC`, `CLAIMS_REBUILD_POLICY` (missing|all|stale), `CLAIMS_STALE_DAYS`

### Usage Aggregators
- Files: `usage_aggregator.py`, `llm_usage_aggregator.py`
- Purpose: Aggregate request/usage logs to daily summaries (per user; and per provider/model for LLMs).
- Env/Settings:
  - `USAGE_LOG_ENABLED`, `USAGE_AGGREGATOR_INTERVAL_MINUTES`
  - `LLM_USAGE_AGGREGATOR_ENABLED`, `LLM_USAGE_AGGREGATOR_INTERVAL_MINUTES`

### Web Scraping Service (Enhanced)
- File: `tldw_Server_API/app/services/enhanced_web_scraping_service.py`
- Purpose: Production scraping pipeline with queueing, rate limiting, cookie/session support, and rich progress.
- Notes:
  - Uses Playwright if available; otherwise raises to trigger compatibility fallback behavior (`services/web_scraping_service.py`).
  - Persists scraped content into Media DB with chunk-level FTS.
  - Ephemeral mode stores results in a bounded in-memory store (`ephemeral_store.py`) with TTL and cap controls.

## Known Gaps and Recommendations

- Placeholders not production-ready:
  - `document_processing_service.py`, `ebook_processing_service.py`, `podcast_processing_service.py`, `xml_processing_service.py` - keep out of critical paths until completed and tested.
  - Feature flag: set `PLACEHOLDER_SERVICES_ENABLED=1` (env) to enable these placeholders; otherwise they return 503 to prevent accidental use.

- Ephemeral store deployment notes:
  - `ephemeral_store.py` now enforces expiry and deterministic eviction in-process, but remains process-local (not shared across workers/instances).
  - Tune with `EPHEMERAL_STORE_TTL_SECONDS`, `EPHEMERAL_STORE_MAX_ENTRIES`, `EPHEMERAL_STORE_MAX_BYTES`.
  - For horizontally scaled deployments, prefer a shared backend (Redis/DB) if cross-process retrieval is required.

- Async/blocking mix:
  - Some persistence paths (e.g., storing scraped articles) call synchronous DB methods from async contexts. Wrap heavy sync calls with `asyncio.to_thread` to avoid event-loop blocking.

- Observability:
  - Jobs workers emit useful metrics; consider adding counters/timers to storage quota recalculation and temp cleanup.

- Tests:
  - Add integration tests for quota checks, breakdown, and cleanup. Mock filesystem with temporary dirs.
  - Add end-to-end tests for aggregators with seed rows in `usage_log` / `llm_usage_log`.
  - Add smoke tests for enhanced web scraping service initialization and fallback path.

## Quick Reference (Env Flags)

- Chatbooks jobs worker: `CHATBOOKS_CORE_WORKER_ENABLED` (`CHATBOOKS_JOBS_BACKEND` ignored)
- Jobs metrics gauges: `JOBS_METRICS_GAUGES_ENABLED`, interval and TTL flags above
- Aggregators: `DISABLE_USAGE_AGGREGATOR`, `DISABLE_LLM_USAGE_AGGREGATOR`
- Claims rebuild: `CLAIMS_REBUILD_ENABLED`, `CLAIMS_REBUILD_INTERVAL_SEC`

## Safety/Performance Notes

- Quota scanning runs in a small threadpool; avoid large max_workers to prevent I/O contention.
- Ensure user directories exist and have expected layout: `media`, `notes`, `embeddings`, `exports`, `temp`.
- PostgreSQL is recommended for multi-user deployments; SQLite supported for single-user.

---

For implementation details, see the service files referenced above, the package README at `tldw_Server_API/app/services/README.md`, and the startup/shutdown wiring in `tldw_Server_API/app/main.py`.
