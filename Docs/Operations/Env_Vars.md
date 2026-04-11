# Environment Variables - tldw_server (v0.1)

This reference lists environment variables recognized by the server. Environment variables take precedence over values from `tldw_Server_API/Config_Files/.env`, which in turn take precedence over `tldw_Server_API/Config_Files/config.txt` (where supported).

Precedence (highest → lowest):
- Process environment variables
- `.env` (Pydantic / dotenv; default `tldw_Server_API/Config_Files/.env`)
- `config.txt` (sections parsed by the app; not all settings support file overrides; default `tldw_Server_API/Config_Files/config.txt`)

Note: Secrets should be set via environment or `.env`. `config.txt` is supported for convenience in dev; prefer env in production.

## Core Server
- `tldw_production`: Enable production guards (`true|false`). Masks API key in logs and enforces DB/secret checks.
- `ENABLE_OPENAPI`: Show OpenAPI/Swagger UI when `true`. Defaults to hidden in production unless explicitly enabled.
- `ALLOWED_ORIGINS`: Optional browser-origin allowlist. Comma-separated or JSON array.
  - Local self-hosting already permits common `localhost` and `127.0.0.1` browser origins by default.
  - Set this only when your browser UI runs from another origin, such as a LAN IP, reverse proxy host, or custom port.
- `CORS_ALLOW_CREDENTIALS`: Enable credentialed CORS responses (`true|false`). Default `false`.
- `TLDW_CONFIG_PATH`: Absolute path to the primary `config.txt`. When set, the parent directory is treated as the config root for auxiliary assets (e.g., `Synonyms/`).
- `TLDW_CONFIG_DIR`: Explicit directory containing `config.txt` and related config assets. Checked after `TLDW_CONFIG_PATH`.
- `ENABLE_SECURITY_HEADERS`: Enable security headers middleware (defaults to true in production).
- `UVICORN_WORKERS`: Uvicorn worker count (default 4 in Docker).
- `LOG_LEVEL`: Application log level (`DEBUG|INFO|WARNING|ERROR`).
- `MAGIC_FILE_PATH`: Path to `magic.mgc` for `python-magic` if needed.

## Storage
- `USER_DB_BASE_DIR`: Base directory for per-user DBs and assets (defined in `tldw_Server_API.app.core.config`). Defaults to `Databases/user_databases` under the repo root; relative paths resolve from repo root and `~` expands. Override via environment variable or `Config_Files/config.txt` as needed.
- `USER_DB_BASE_DIR_ALLOWED_ROOTS` / `TLDW_USER_DB_BASE_DIR_ALLOWED_ROOTS`: Optional allowlist for setup-time changes to `USER_DB_BASE_DIR`. Comma- or colon-separated list of parent directories permitted for the new base.
- `USER_DB_BASE`: Deprecated alias for `USER_DB_BASE_DIR` (used only by rewrite cache resolution).

## Unified Circuit Breaker (Registry + Cross-Worker Semantics)
- `CIRCUIT_BREAKER_REGISTRY_MODE`: Registry storage mode (`auto|memory|persistent` plus synonyms). Default `auto`.
  - `auto`: Uses persistent shared state in normal runtime and in-memory state under explicit pytest runtime.
  - `memory`: Process-local registry only (no cross-worker synchronization).
  - `persistent`: Shared DB-backed registry with optimistic locking and lease coordination.
- `CIRCUIT_BREAKER_REGISTRY_DB_PATH`: Optional path override for shared registry DB. Relative paths resolve from project root.
- `CIRCUIT_BREAKER_PERSIST_MAX_RETRIES`: Max optimistic-lock merge/retry attempts per persist operation (default `4`, clamped to `>=1`).
- `CIRCUIT_BREAKER_HALF_OPEN_LEASE_TTL_SECONDS`: TTL for distributed HALF_OPEN probe leases (default `120.0`, clamped to `>=1.0`).

Operational notes:
- For multi-worker deployments, prefer `CIRCUIT_BREAKER_REGISTRY_MODE=persistent`.
- If `CIRCUIT_BREAKER_PERSIST_MAX_RETRIES` is too low under heavy write contention, `circuit_breaker_persist_conflicts_total` may spike and stale local mutations can be dropped after retry exhaustion.
- If `CIRCUIT_BREAKER_HALF_OPEN_LEASE_TTL_SECONDS` is too high, abandoned probe slots take longer to self-heal; if too low, long-running probes can lose their lease before completion.

## OCR (PDF pipeline)
- `OCR_PAGE_CONCURRENCY`: Global per-page OCR concurrency cap (default `1`). The PDF pipeline applies `min(OCR_PAGE_CONCURRENCY, backend profile max_page_concurrency)` before dispatching page OCR work.

### Llama.cpp OCR
- `LLAMACPP_OCR_MODE`: `auto|remote|managed|cli`.
- `LLAMACPP_OCR_ALLOW_MANAGED_START`: `true|false`. Managed mode is single-process only in v1.
- `LLAMACPP_OCR_AUTO_ELIGIBLE`: Enables participation in `auto` when the backend is configured and locally available.
- `LLAMACPP_OCR_AUTO_HIGH_QUALITY_ELIGIBLE`: Enables participation in `auto_high_quality` when the backend is configured and locally available.
- `LLAMACPP_OCR_MAX_PAGE_CONCURRENCY`: Backend-local per-page cap; defaults to `1` unless you raise it explicitly.
- Remote: `LLAMACPP_OCR_HOST`, `LLAMACPP_OCR_PORT`, `LLAMACPP_OCR_MODEL_PATH`, `LLAMACPP_OCR_USE_DATA_URL`.
- Managed: `LLAMACPP_OCR_ARGV`, `LLAMACPP_OCR_MODEL_PATH`, optional `LLAMACPP_OCR_HOST`, `LLAMACPP_OCR_PORT`, `LLAMACPP_OCR_STARTUP_TIMEOUT_SEC`. Managed startup in `auto` is preferred over `remote` when `LLAMACPP_OCR_ALLOW_MANAGED_START=true` and the managed profile is configured.
- CLI: `LLAMACPP_OCR_ARGV`, `LLAMACPP_OCR_MODEL_PATH`.

### ChatLLM OCR
- `CHATLLM_OCR_MODE`: `auto|remote|managed|cli`.
- `CHATLLM_OCR_ALLOW_MANAGED_START`: `true|false`. Managed mode is single-process only in v1.
- `CHATLLM_OCR_AUTO_ELIGIBLE`: Enables participation in `auto` when the backend is configured and locally available.
- `CHATLLM_OCR_AUTO_HIGH_QUALITY_ELIGIBLE`: Enables participation in `auto_high_quality` when the backend is configured and locally available.
- `CHATLLM_OCR_MAX_PAGE_CONCURRENCY`: Backend-local per-page cap; defaults to `1` unless you raise it explicitly.
- Remote: `CHATLLM_OCR_URL`, `CHATLLM_OCR_MODEL`, `CHATLLM_OCR_API_KEY`.
- Managed: `CHATLLM_OCR_SERVER_BINARY`, `CHATLLM_OCR_MODEL_PATH`, `CHATLLM_OCR_HOST`, `CHATLLM_OCR_PORT`, `CHATLLM_OCR_STARTUP_TIMEOUT_SEC`, `CHATLLM_OCR_SERVER_ARGS_JSON`, `CHATLLM_OCR_HEALTHCHECK_URL`.
- CLI: `CHATLLM_OCR_CLI_BINARY`, `CHATLLM_OCR_MODEL_PATH`, `CHATLLM_OCR_CLI_ARGS_JSON`.

### MinerU OCR
- `MINERU_CMD`: Command used to launch MinerU for document-level PDF OCR. Defaults to `mineru`. The command is tokenized safely and executed without a shell.
- `MINERU_TIMEOUT_SEC`: Whole-document MinerU timeout in seconds (default `120`).
- `MINERU_MAX_CONCURRENCY`: Max concurrent MinerU document runs (default `1`). Applies at the document level, not per page.
- `MINERU_TMP_ROOT`: Optional root directory for MinerU temporary working directories.
- `MINERU_DEBUG_SAVE_RAW`: When `true`, include full raw `content_list.json` and `middle.json` payloads in the normalized structured OCR artifact block. Off by default.

Notes
- MinerU is PDF-only in v1 and does not participate in `auto` backend selection.
- `ocr_lang` and `ocr_dpi` remain request parameters for API consistency, but MinerU currently treats them as advisory metadata only.

### Dolphin OCR
- `DOLPHIN_MODE`: `auto` | `transformers` | `remote`.
- `DOLPHIN_PROMPT`, `DOLPHIN_PROMPT_PRESET`: main prompt override/preset (`general|doc|table|json`).
- `DOLPHIN_JSON_PROMPT`: override JSON prompt (empty disables). `DOLPHIN_DISABLE_JSON=true` disables JSON pass.
- `DOLPHIN_URL`: remote server base URL.
- `DOLPHIN_REMOTE_MODE`: `dolphin_vllm` | `dolphin_trt` | `openai`.
- `DOLPHIN_ENCODER_PROMPT`, `DOLPHIN_DECODER_PROMPT`: remote prompt overrides.
- `DOLPHIN_REMOTE_MODEL`: model name for OpenAI-compatible mode.
- `DOLPHIN_TIMEOUT`: request timeout seconds (default `60`).
- `DOLPHIN_USE_DATA_URL`: `true` to send base64 image URLs (recommended for remote).
- `DOLPHIN_MODEL_PATH`: local model id/path (default `ByteDance/Dolphin-v2`).
- `DOLPHIN_DEVICE`: override device (`cuda`, `cpu`, etc.).
- Generation: `DOLPHIN_MAX_NEW_TOKENS`, `DOLPHIN_MAX_LENGTH`, `DOLPHIN_TEMPERATURE`, `DOLPHIN_TOP_P`,
  `DOLPHIN_TOP_K`, `DOLPHIN_REPETITION_PENALTY`, `DOLPHIN_DO_SAMPLE`, `DOLPHIN_NUM_BEAMS`.

Config file overrides (`Config_Files/config.txt`)
- `[OCR] backend_priority`: comma-separated list or JSON array of backends for auto selection.
  - Example: `backend_priority = ["dolphin", "hunyuan", "points", "dots", "tesseract"]`
  - When set, this list is used for both `auto` and `auto_high_quality` resolution.

## Testing & CI Controls
- `TEST_MODE`: Enables test-friendly behaviors (`true|1|yes`). Used across modules to:
  - Relax or bypass certain rate limiter keys (e.g., client IP) to avoid false positives in tests.
  - Prefer offline/test-safe code paths (e.g., RAG/Chunking avoid network downloads; health endpoints may expose additional diagnostics in tests).
- `DISABLE_NLTK_DOWNLOADS`: Prevent NLTK dataset downloads (`1|true|yes`).
  - RAG query features and Chunking modules will not attempt to download `punkt`, `wordnet`, or `stopwords` when this is set; they degrade gracefully to local fallbacks.
- `ALLOW_NLTK_DOWNLOADS`: Force-enable NLTK downloads even when running tests (`1|true|yes`).
  - Overrides `TEST_MODE`/`DISABLE_NLTK_DOWNLOADS`/pytest auto-detection to allow downloads for development scenarios that require full NLTK resources.

### Jobs Postgres (Test-only Helpers)
- `RUN_PG_JOBS_TESTS`: Enable Jobs outbox Postgres tests (`1|true|yes`). Disabled by default due to environment variability.
- `TLDW_TEST_NO_DOCKER`: When set (`1|true|yes`), disables auto-start of a local Postgres Docker container during Jobs tests.
- `TLDW_TEST_PG_IMAGE`: Docker image for the optional local Postgres used by Jobs tests (default `postgres:15`).
- `TLDW_TEST_PG_CONTAINER_NAME`: Container name for the optional local Postgres (default `tldw_jobs_postgres_test`).
  - The Jobs tests/fixtures first try a TCP probe to the configured DSN; when unreachable and the host is local, they attempt to start this container unless `TLDW_TEST_NO_DOCKER` is set.
  - You can also set `POSTGRES_TEST_*` vars or `JOBS_DB_URL` explicitly to point at an existing cluster.

## RAG Module
- `tldw_production`: When `true`, RAG retrievers disable raw SQL fallbacks and require adapters (MediaDatabase/ChaChaNotesDB). Unified endpoints already pass adapters; direct pipeline usage must supply them.
- `RAG_LLM_RERANK_TIMEOUT_SEC`: Per-document LLM rerank timeout (seconds). Default `10`.
- `RAG_LLM_RERANK_TOTAL_BUDGET_SEC`: Total time budget for LLM reranking per query (seconds). Default `20`.
- `RAG_LLM_RERANK_MAX_DOCS`: Cap on number of documents scored by LLM reranker per query. Default `20`.
- `RAG_TRANSFORMERS_RERANKER_MODEL`: Cross-encoder model id for fast reranking (stage 1). Default `BAAI/bge-reranker-v2-m3`.
- `RAG_FLASHRANK_CACHE_DIR`: Cache directory for FlashRank model bundles. Default resolves to repo-local `models/flashrank`.
- `RAG_FLASHRANK_MODEL_NAME`: FlashRank model directory name. Default `ms-marco-TinyBERT-L-2-v2`.
- `RAG_REWRITE_CACHE_PATH`: Optional override for query→rewrite cache JSONL. When unset, cache is per-user under `<USER_DB_BASE_DIR>/<user_id>/Rewrite_Cache/rewrite_cache.jsonl` (deprecated alias: `USER_DB_BASE`).
- `RAG_PRECOMPUTED_SPANS_MAX_VECTORS_PER_CORPUS`: Cap on stored span vectors per corpus (default `200000`). Config key: `[RAG] precomputed_spans_max_vectors_per_corpus`.
- `RAG_PRECOMPUTED_SPANS_MAX_MB_PER_CORPUS`: Cap on precomputed span storage per corpus in MB (default `512`). Config key: `[RAG] precomputed_spans_max_mb_per_corpus`.
- `RAG_PRECOMPUTED_SPANS_RETENTION_DAYS`: Retention window for precomputed spans before GC (default `30`). Config key: `[RAG] precomputed_spans_retention_days`.

### RAG Guardrails (Production Defaults)
- `RAG_GUARDRAILS_STRICT`: When `true`, enable strict guardrails in the unified pipeline (enables numeric fidelity and hard citations by default). Useful for non-prod environments where you still want strict behavior.
- `RAG_ENABLE_NUMERIC_FIDELITY`: Force-enable numeric fidelity verification of answers (overrides request default). Optional; typically implied by `RAG_GUARDRAILS_STRICT`.
- `RAG_REQUIRE_HARD_CITATIONS`: Force-enable per-sentence hard citations mapping (overrides request default). Optional; typically implied by `RAG_GUARDRAILS_STRICT`.
- `RAG_NUMERIC_FIDELITY_BEHAVIOR`: Default behavior when numeric values are not verified in sources: `continue` | `ask` | `decline` | `retry`. Default `ask` when strict mode is active.
- `RAG_PAYLOAD_EXEMPLAR_SAMPLING`: Sampling rate (0..1) to record redacted payload exemplars when adaptive check fails (default `0.05`).
- `RAG_PAYLOAD_EXEMPLAR_PATH`: Optional path for payload exemplars JSONL sink (default `Databases/observability/rag_payload_exemplars.jsonl`).
- `RAG_PERSONALIZATION_HALF_LIFE_DAYS`: Half-life for decay of per-user priors (default `7`).
- `RAG_PERSONALIZATION_WEIGHT`: Additive weight applied to prior during boosting (default `0.1`).

### RAG Quality Evaluations (Nightly)
- `RAG_QUALITY_EVAL_ENABLED`: Enable nightly eval scheduler in-process (`true|false`, default `false`).
- `RAG_QUALITY_EVAL_INTERVAL_SEC`: Interval between eval runs in seconds (default `86400`).
- `RAG_QUALITY_EVAL_DATASET`: Path to JSONL eval dataset (default `Docs/Deployment/Monitoring/Evals/nightly_rag_eval.jsonl`).

### Embeddings A/B Persistence
- `EVALS_ABTEST_PERSISTENCE`: Backend for embeddings A/B test storage. Defaults to `sqlalchemy` (or `repo`) which enables the SQLAlchemy repository with typed models. Set to any other value (for example `legacy`) to fall back to the previous SQLite helper implementation. Only the SQLite deployment path honors this toggle; Postgres deployments always use the legacy adapter.

Notes:
- In production (`tldw_production=true`) or when `RAG_GUARDRAILS_STRICT=true`, the unified pipeline will default to enabling numeric fidelity and strict citations unless explicitly configured otherwise by the request.

### Two-Tier Reranking Calibration & Gating
- `RAG_RERANK_CALIB_BIAS`: Logistic calibration bias. Default `-1.5`.
- `RAG_RERANK_CALIB_W_ORIG`: Weight for original retrieval score. Default `0.8`.
- `RAG_RERANK_CALIB_W_CE`: Weight for cross-encoder score. Default `2.5`.
- `RAG_RERANK_CALIB_W_LLM`: Weight for LLM reranker score. Default `3.0`.
- `RAG_MIN_RELEVANCE_PROB`: Minimum calibrated probability to allow generation. Default `0.35`.
- `RAG_SENTINEL_MARGIN`: Required margin of (top_prob - sentinel_prob) to consider evidence strong enough. Default `0.10`.

### RAG Rollout Toggles (Structure, Planner, Cache)
- `RAG_ENABLE_STRUCTURE_INDEX`: Enable persisted document structure index (sections/paragraphs with char offsets) and retrieval metadata enrichment. Defaults to `true`. Config file key: `[RAG] enable_structure_index`.
- `RAG_STRICT_EXTRACTIVE`: Use strict extractive answer path in the standard pipeline (assemble only from retrieved spans). Defaults to `false`. Config key: `[RAG] strict_extractive`.
- `RAG_LOW_CONFIDENCE_BEHAVIOR`: Behavior when evidence is insufficient after guardrails (`continue` | `ask` | `decline`). Defaults to `continue`. Config key: `[RAG] low_confidence_behavior`.
- `RAG_AGENTIC_CACHE_BACKEND`: Agentic ephemeral cache backend (`memory` | `sqlite`). Defaults to `memory`. Config key: `[RAG] agentic_cache_backend`.
- `RAG_AGENTIC_CACHE_TTL_SEC`: TTL for agentic cache entries in seconds. Defaults to `600`. Config key: `[RAG] agentic_cache_ttl_sec`.

Notes:
- These env vars take precedence over `.env`, which takes precedence over `config.txt`. The loader now propagates `config.txt` defaults into process env when unset, so modules reading `os.getenv` will honor file settings by default.

### Ingest & Chunking
- `INGEST_ENABLE_DEDUP`: Enable near-duplicate removal at ingestion time (`true|false`, default `true`).
- `INGEST_DEDUP_THRESHOLD`: Jaccard similarity threshold for shingle-based dedupe (0-1, default `0.9`).
- Chunker adaptive controls are primarily request-level, but ingestion defaults set `adaptive=true` and `adaptive_overlap=true`.

### Web Scraping Ephemeral Store
- `EPHEMERAL_STORE_TTL_SECONDS`: Default TTL for ephemeral web-scraping results (seconds). Default `900`.
- `EPHEMERAL_STORE_MAX_ENTRIES`: Maximum number of in-memory ephemeral entries retained before oldest-first eviction. Default `256`.
- `EPHEMERAL_STORE_MAX_BYTES`: Optional aggregate payload size cap (bytes) for the in-memory ephemeral store. Default `0` (disabled).
  - Applies to process-local storage (`app/services/ephemeral_store.py`) used by ephemeral scraping/result retrieval paths.

### RAG Adaptive Post-Verification
- `RAG_ADAPTIVE_TIME_BUDGET_SEC`: Optional hard cap (seconds) for post-generation verification and repair. When unset or `0`, no cap is applied. Other knobs are request-level (`enable_post_verification`, `adaptive_max_retries`, `adaptive_unsupported_threshold`, `adaptive_max_claims`).
- `RAG_ADAPTIVE_ADVANCED_REWRITES`: Toggle advanced rewrite strategy (HyDE + multi-strategy + diversity) for the adaptive pass. `true|false` (default `true`). When `false`, the adaptive pass uses a simple single-query retrieval.

### Chunking (regex safety and templates)
- `CHUNKING_REGEX_TIMEOUT`: Float seconds to cap regex execution for chapter/section detection and template boundaries. Default: `2`. Values <= 0 disable. On timeout, strategies fall back to safe paths.
- `CHUNKING_DISABLE_MP`: Disable process-based isolation for regex (default: disabled, i.e., no MP). Set `0|false|no` to enable optional MP fallback; note platform constraints.
- `CHUNKING_REGEX_SIMPLE_ONLY`: When `1|true|yes`, only a safe regex subset is allowed for custom boundary patterns. Unsafe constructs are rejected during validation.
- `CHUNKING_TEMPLATES_FALLBACK_ENABLED`: When `0|false|no`, disallow the in-process fallback store for chunking templates. Endpoints will return `500` with a hint if DB methods are missing. Default: enabled for dev/test.

### Security Health (Audit Thresholds)
- `AUDIT_SEC_CRITICAL_HIGH_RISK_MIN`: Minimum count of high-risk security events in the last 24h to mark status as `at_risk` and risk level `critical`. Default: `1`.
- `AUDIT_SEC_ELEVATED_FAILURE_MIN`: Minimum count of failure events in the last 24h to mark status `elevated` and risk level `high`. Default: `50`.

### Test Suite Toggles
- `TLDW_TEST_POSTGRES_REQUIRED`: Require Postgres-backed AuthNZ tests; when unset and Postgres is unavailable, tests auto-skip.
- `RUN_MCP_TESTS`: Enable MCP unified tests (defaults to skipped). Set to `1|true|yes` to run.
- `RUN_MOCK_OPENAI`: Enable Mock OpenAI server tests (defaults to skipped). Set to `1|true|yes` to run.

## User Profile
- `PROFILE_SLA_MS`: Base SLA threshold in milliseconds for profile reads (default `300`). Used when `PROFILE_SLA_MS_SELF` / `PROFILE_SLA_MS_ADMIN` are unset.
- `PROFILE_SLA_MS_SELF`: SLA threshold in milliseconds for `/users/me/profile` (default inherits `PROFILE_SLA_MS`).
- `PROFILE_SLA_MS_ADMIN`: SLA threshold in milliseconds for `/admin/users/{id}/profile` (default inherits `PROFILE_SLA_MS`).
- `PROFILE_BATCH_BASE_MS`: Base SLA threshold in milliseconds for `/admin/users/profile` (default `800`).
- `PROFILE_BATCH_BASE_SIZE`: Baseline page size for batch SLA scaling (default `50`). Threshold scales linearly with `page_size`.
- `PROFILE_BATCH_TIMEOUT_SECONDS`: Soft timeout threshold for batch profile reads (default `10`). Breaches emit metrics/log warnings.

## Jobs Backend / Worker
- `JOBS_DB_URL`: PostgreSQL DSN for the core Jobs backend (e.g., `postgresql://user:pass@host:5432/jobs`). When unset, SQLite is used (Databases/jobs.db).
- `JOBS_LEASE_SECONDS`: Default lease granted when acquiring a job (default `60`).
- `JOBS_LEASE_RENEW_SECONDS`: Renewal cadence while a worker processes a job (default `30`).
- `JOBS_LEASE_RENEW_JITTER_SECONDS`: Jitter (seconds) applied to renewals to avoid herd behavior (default `5`).
- `JOBS_LEASE_MAX_SECONDS`: Cap for acquire/renew lease seconds (default `3600`).
- `TLDW_WORKERS_SIDECAR_MODE`: When true, skip in-process Jobs workers so you can run them as sidecars (`true|false`, default `false`).
- `EVALUATIONS_ABTEST_JOBS_WORKER_ENABLED`: Enable the in-process Embeddings A/B Jobs worker (`true|false`, default `false`). Alias: `EVALS_ABTEST_JOBS_WORKER_ENABLED`.
- `EVALUATIONS_JOBS_QUEUE`: Queue name for evaluations jobs (default `default`). Alias: `EVALS_JOBS_QUEUE`.
- `FILES_JOBS_WORKER_ENABLED`: Enable the in-process file artifacts jobs worker (`true|false`, default follows route policy for `files`). When false, run `python -m tldw_Server_API.app.core.File_Artifacts.jobs_worker`.
- `PROMPT_STUDIO_JOBS_WORKER_ENABLED`: Enable the in-process Prompt Studio jobs worker (`true|false`, default follows route policy for `prompt-studio`).
- `PRIVILEGE_SNAPSHOT_WORKER_ENABLED`: Enable the in-process privilege snapshot jobs worker (`true|false`, default follows route policy for `privileges`).

## Embeddings Jobs
- `EMBEDDINGS_JOBS_BACKEND`: Backend is fixed to "core"; this environment variable exists for compatibility and is ignored.
- `EMBEDDINGS_JOBS_QUEUE`: Queue for embeddings stage jobs (default `default`).
- `EMBEDDINGS_ROOT_JOBS_QUEUE`: Queue for embeddings root jobs (default `low` when stage queue is not `low`).
- `EMBEDDINGS_JOBS_WORKER_ID`: Worker identifier for embeddings jobs (default `embeddings-jobs-<pid>`).
- `EMBEDDINGS_JOBS_LEASE_SECONDS`: Lease duration for embeddings stage jobs (default `60`).
- `EMBEDDINGS_JOBS_RENEW_JITTER_SECONDS`: Lease renew jitter in seconds (default `5`).
- `EMBEDDINGS_JOBS_RENEW_THRESHOLD_SECONDS`: Renew threshold in seconds (default `10`).
- `EMBEDDINGS_JOBS_BACKOFF_BASE_SECONDS`: Base retry backoff in seconds (default `2`).
- `EMBEDDINGS_JOBS_BACKOFF_MAX_SECONDS`: Max retry backoff in seconds (default `30`).
- `EMBEDDINGS_JOBS_RETRY_BACKOFF_SECONDS`: Backoff for retryable errors (default `10`).
- `EMBEDDINGS_JOBS_EXPOSE_PROGRESS`: Include `progress_percent`/`total_chunks` in public jobs responses (`true|false`, default `false`).

## Data Tables Jobs
- `DATA_TABLES_JOBS_WORKER_ENABLED`: Enable the in-process data tables jobs worker (`true|false`, default follows route policy for `data-tables`). When false, run `python -m tldw_Server_API.app.core.Data_Tables.jobs_worker`.
- `DATA_TABLES_JOBS_QUEUE`: Queue for data table generation jobs (default `default`).
- `DATA_TABLES_JOBS_WORKER_ID`: Worker identifier for data tables jobs (default `data-tables-jobs-<pid>`).
- `DATA_TABLES_JOBS_LEASE_SECONDS`: Lease duration for data tables jobs (default `60`).
- `DATA_TABLES_JOBS_RENEW_JITTER_SECONDS`: Lease renew jitter in seconds (default `5`).
- `DATA_TABLES_JOBS_RENEW_THRESHOLD_SECONDS`: Renew threshold in seconds (default `10`).
- `DATA_TABLES_JOBS_BACKOFF_BASE_SECONDS`: Base retry backoff in seconds (default `2`).
- `DATA_TABLES_JOBS_BACKOFF_MAX_SECONDS`: Max retry backoff in seconds (default `30`).
- `DATA_TABLES_JOBS_RETRY_BACKOFF_SECONDS`: Backoff for retryable errors (default `10`).

## Data Tables Generation Limits
- `DATA_TABLES_DEFAULT_MAX_ROWS`: Default max rows per table when request omits `max_rows` (default `200`).
- `DATA_TABLES_MAX_ROWS`: Hard cap on generated rows per table (default `2000`).
- `DATA_TABLES_MAX_SOURCE_CHARS`: Per-source character cap used when building prompts (default `12000`).
- `DATA_TABLES_MAX_TOTAL_SOURCE_CHARS`: Aggregate character cap across all sources (default `60000`).
- `DATA_TABLES_MAX_SNAPSHOT_CHARS`: Per-chunk snapshot text cap for rag_query sources (default `8000`).
- `DATA_TABLES_MAX_PROMPT_CHARS`: Total prompt size cap (default `24000`).
- `DATA_TABLES_CHAT_BATCH_SIZE`: Batch size when loading chat messages (default `250`).
- `DATA_TABLES_CHAT_MAX_MESSAGES`: Maximum chat messages loaded per source (default `1500`).
- `DATA_TABLES_LLM_MAX_TOKENS`: LLM response token budget for table generation (default `2000`).
- `DATA_TABLES_LLM_TEMPERATURE`: LLM temperature for table generation (default `0.2`).

## Chat Commands & Weather
- `CHAT_COMMANDS_ENABLED`: Enable slash-command preprocessing (`true|false`, default `false`).
- `CHAT_COMMAND_INJECTION_MODE`: Slash-command injection mode (`system|preface|replace`, default `system`).
- `CHAT_COMMANDS_REQUIRE_PERMISSIONS`: Require per-command RBAC permission checks (`true|false`, default `false`).
- `CHAT_COMMANDS_RATE_LIMIT_USER`: Per-user, per-command RPM limit (accepts `10` or `10/min`; default `10`).
- `CHAT_COMMANDS_RATE_LIMIT`: Backward-compatible alias for `CHAT_COMMANDS_RATE_LIMIT_USER`.
- `CHAT_COMMANDS_RATE_LIMIT_GLOBAL`: Global, per-command RPM limit (accepts `100` or `100/min`; default `100`).
- `CHAT_COMMANDS_MAX_CHARS`: Max characters injected from a slash-command result (default `300`).
- `DEFAULT_LOCATION`: Optional fallback location for `/weather` when no argument is supplied.
- `WEATHER_PROVIDER`: Weather backend (`openweather`, `noop`, `none`, `disabled`; default `openweather`).
- `OPENWEATHER_API_KEY`: API key for the `openweather` provider.
- `WEATHER_UNITS`: Unit system for weather summaries (`metric|imperial`, default `metric`).
- `WEATHER_LANG`: OpenWeather language code for descriptions (default `en`).
- `WEATHER_TIMEOUT_MS`: OpenWeather HTTP timeout in milliseconds (default `1500`).

## Chatbooks
- `CHATBOOKS_JOBS_BACKEND`: Backend is fixed to "core"; this environment variable exists for compatibility and is ignored.
- `CHATBOOKS_CORE_WORKER_ENABLED`: Enable shared Chatbooks worker when backend=core (default `true`).
- `CHATBOOKS_SIGNED_URLS`: Require HMAC-signed download URLs (`true|false`, default `false`).
- `CHATBOOKS_SIGNING_SECRET`: Secret key used for download URL signing (required when signed URLs are enabled).
- `CHATBOOKS_ENFORCE_EXPIRY`: Enforce job `expires_at` on download (`true|false`, default `true`).
- `CHATBOOKS_URL_TTL_SECONDS`: Default expiry TTL for generated download links (default `86400`).
- `CHATBOOKS_EXPORT_RETENTION_DEFAULT_HOURS`: Retention window for completed exports before expiry (default `24`).
- `CHATBOOKS_CLEANUP_INTERVAL_SEC`: Scheduled cleanup cadence in seconds (set `0` to disable scheduling).
- `CHATBOOKS_EVAL_EXPORT_MAX_ROWS`: Max rows exported per evaluation run (default `200`).
- `CHATBOOKS_BINARY_LIMITS_MB`: JSON map of content type to max bundled size in MB (for example, `{"media": 0, "conversations": 10, "generated_docs": 25}`).
- `CHATBOOKS_TEMPLATE_MODE`: Default Chatbooks template mode (`pass_through|render_on_export`; default `pass_through`).
- `CHATBOOKS_TEMPLATE_DEFAULTS_JSON`: JSON object merged into Chatbooks template defaults (optional).
- `CHATBOOKS_TEMPLATE_TIMEZONE`: Default timezone used for Chatbooks template rendering (default `UTC`).
- `CHATBOOKS_TEMPLATE_LOCALE`: Optional default locale used for Chatbooks template rendering.
- `CHATBOOKS_IMPORT_DICT_STRICT`: When true, skip dictionaries with fatal validation errors instead of importing with warnings.

## Audio Jobs
- `AUDIO_JOBS_WORKER_ENABLED`: Enable the in-process Audio Jobs worker (`true|false`, default follows route policy for `audio-jobs`). When true, the worker starts at app startup and polls the Jobs backend for the `audio` domain pipeline stages.
- `AUDIO_JOBS_OWNER_STRICT`: Enable owner-aware acquisition for fairness across users (`true|false`, default `false`). When enabled, the worker preferentially acquires jobs for owners under their concurrent-job caps.
- `REDIS_URL`: Optional Redis URL used by Resource Governor when `RG_BACKEND=redis`.

## Media Ingest Jobs
- `MEDIA_INGEST_JOBS_WORKER_ENABLED`: Enable the in-process media ingest jobs worker (`true|false`, default follows route policy for `media`). When true, the worker starts at app startup and polls the Jobs backend for the `media_ingest` domain.

## Email Search Rollout
- `EMAIL_NATIVE_PERSIST_ENABLED`: Enable normalized email persistence on ingest (`true|false`, default `true`).
- `EMAIL_OPERATOR_SEARCH_ENABLED`: Enable normalized email operator search APIs and bridge support (`true|false`, default `true`).
- `EMAIL_MEDIA_SEARCH_DELEGATION_MODE`: Default `/api/v1/media/search` delegation mode for email-only scope when `email_query_mode` is not explicitly set. Allowed: `opt_in` (default, delegate only when `email_query_mode=operators`) or `auto_email` (delegate automatically when `media_types=['email']` and operator search is enabled).
- `EMAIL_GMAIL_CONNECTOR_ENABLED`: Enable Gmail source/sync endpoints (`true|false`, default `false`).

## Email Connector Sync
- `EMAIL_SYNC_RETRY_MAX_ATTEMPTS`: Maximum retry budget for failed Gmail sync runs before a source is skipped until operator intervention/data correction (default `5`).
- `EMAIL_SYNC_RETRY_BASE_SECONDS`: Base retry backoff duration in seconds (default `60`).
- `EMAIL_SYNC_RETRY_MAX_BACKOFF_SECONDS`: Maximum exponential backoff cap in seconds (default `3600`).
- `EMAIL_SYNC_CURSOR_RECOVERY_WINDOW_DAYS`: Window (days) used for bounded replay when Gmail `startHistoryId` is invalid/expired (default `7`).
- `EMAIL_SYNC_CURSOR_RECOVERY_MAX_MESSAGES`: Max replay message cap for invalid-cursor recovery runs (default `2000`).

Pytest markers
- `-m jobs`: Run all core Jobs tests (SQLite + PG-gated).
- `-m pg_jobs`: Run Postgres-only Jobs tests (requires JOBS_DB_URL and psycopg).
- `-m pg_jobs_stress`: Run heavier multi-process concurrency tests for PG (opt-in only).
  - Also set `RUN_PG_JOBS_STRESS=1` to enable these tests during runs.

## Resource Governor (Unified Rate Limiting)

The Resource Governor (RG) is the **primary enforcement path** for all rate limiting. Some deprecated module-local compatibility knobs remain during cutover and will be removed once shadow-mode exit criteria are met (see `Docs/Product/Completed/AuthNZ-Refactor/Resource_Governor_PRD.md`). AuthNZ dependency shims (`check_rate_limit`, `check_auth_rate_limit`) are diagnostics-only and do not enforce fallback 429 behavior.

### Core Settings
- `RG_ENABLED`: Master toggle for Resource Governor enforcement (`true|1|false|0`). Resolution: env var > `config.txt` `[ResourceGovernor] enabled` > default `false`.
- `RG_BACKEND`: Backend type (`memory` | `redis`). Default `memory`. Redis requires `REDIS_URL`.
- `RG_POLICY_PATH`: Path to YAML policy file. Default `tldw_Server_API/Config_Files/resource_governor_policies.yaml`.
- `RG_POLICY_STORE`: Policy persistence backend (`yaml` | `db`). Default `yaml`.
- `RG_POLICY_RELOAD_ENABLED`: Enable hot-reload of policy changes (`true|false`). Default `true`.
- `RG_POLICY_RELOAD_INTERVAL_SEC`: Policy reload check interval in seconds. Default `30`.
- `RG_ROUTE_MAP_AUDIT`: When `true`, log warnings for HTTP routes not covered by the RG route map.
- `RG_REDIS_FAIL_MODE`: Behavior when Redis is unavailable (`fail_open` | `fail_closed` | `fallback_memory`). Default `fail_open`.

### Client Identity
- `RG_TRUSTED_PROXIES`: Comma-separated list of trusted proxy IPs for `X-Forwarded-For` resolution.
- `RG_CLIENT_IP_HEADER`: Custom header for client IP extraction (e.g., `CF-Connecting-IP`).

### Per-Module Policy Overrides
- `RG_CHAT_POLICY_ID`: Override chat policy ID (default `chat.default`).
- `RG_EMBEDDINGS_POLICY_ID`: Override embeddings policy ID (default `embeddings.default`).
- `RG_EMBEDDINGS_SERVER_POLICY_ID`: Override embeddings server policy ID.
- `RG_EMBEDDINGS_SERVER_SYNC_TIMEOUT_SEC`: Timeout for synchronous RG enforcement in the embeddings server thread.
- `RG_CHARACTER_CHAT_POLICY_ID`: Override character chat policy ID.
- `RG_CHARACTER_CHAT_ENFORCE_REQUESTS`: Enable RG request enforcement for character chat (`true|1`).
- `RG_EVALUATIONS_POLICY_ID`: Override evaluations policy ID (default `evals.free`).
- `RG_WEB_SCRAPING_POLICY_ID`: Override web scraping policy ID.

### Shadow / Migration
- `RG_SHADOW_CHAT`: Enable shadow-mode comparison for chat (emit mismatch metrics without enforcing). Used during migration validation.
- `RG_METRICS_ENTITY_LABEL`: Include entity labels in Prometheus metrics (`true|false`). Default `false` (high cardinality).

### Debug / Test-Only
- `RG_DEBUG`: Enable verbose RG decision logging.
- `RG_TEST_DISABLE_ACCEPT_WINDOW`: Test-only: disable acceptance window logic.
- `RG_TEST_FORCE_STUB_RATE`: Test-only: force stub rate values.
- `RG_TEST_PURGE_LEASES_BEFORE_RESERVE`: Test-only: purge concurrency leases before each reserve call.
- `RG_REAL_REDIS_URL`: Test-only: real Redis URL for integration tests.

### Legacy Rate Limit Knobs (Deprecated Compatibility)

The following env vars are retained as **deprecated compatibility knobs** during cutover. RG policy configuration is the authoritative enforcement path.

#### Chat (deprecated compatibility)
- `TEST_CHAT_PER_USER_RPM`: Per-user requests per minute (test override).
- `TEST_CHAT_PER_CONVERSATION_RPM`: Per-conversation requests per minute (test override).
- `TEST_CHAT_GLOBAL_RPM`: Global requests per minute (test override).
- `TEST_CHAT_TOKENS_PER_MINUTE`: Token limit per user per minute (test override).
- `TEST_CHAT_BURST_MULTIPLIER`: Burst multiplier (test override).

#### Embeddings (deprecated compatibility)
- `EMBEDDINGS_RATE_LIMIT_PER_MINUTE`: Per-user embeddings requests per minute.
- `EMBEDDINGS_RATE_LIMIT_MODE`: Rate limit mode (`tokens` or other). Controls whether token-based limiting is applied.

#### Character Chat (deprecated compatibility)
- `CHARACTER_RATE_LIMIT_OPS`: Per-window operation limit. Superseded by RG policy.
- `CHARACTER_RATE_LIMIT_WINDOW`: Window size in seconds. Superseded by RG policy.
- `CHARACTER_RATE_LIMIT_ENABLED`: Enable/disable character chat rate limiting. Superseded by `RG_CHARACTER_CHAT_ENFORCE_REQUESTS`.

## AuthNZ (Authentication)
- `AUTH_MODE`: `single_user` | `multi_user`.
- `DATABASE_URL`: AuthNZ database URL. For production multi-user, use Postgres (e.g., `postgresql://user:pass@host:5432/db`). SQLite supported for dev.
- `SINGLE_USER_API_KEY`: API key for single-user mode (>=24 chars recommended in production).
- `JWT_SECRET_KEY`: JWT signing secret (>=32 chars). Required for `multi_user` in production.
- `ACCESS_TOKEN_EXPIRE_MINUTES`: Access token lifetime (default 30).
- `REFRESH_TOKEN_EXPIRE_DAYS`: Refresh token lifetime (default 7).
- `PUBLIC_WEB_BASE_URL`: Public web app origin used for hosted auth links (for example `https://app.example.com`). When unset, auth emails fall back to `BASE_URL`.
- `PUBLIC_PASSWORD_RESET_PATH`: Public hosted path for password reset completion (default `/auth/reset-password`).
- `PUBLIC_EMAIL_VERIFICATION_PATH`: Public hosted path for email verification completion (default `/auth/verify-email`).
- `PUBLIC_MAGIC_LINK_PATH`: Public hosted path for magic-link sign-in completion (default `/auth/magic-link`).
- Hosted SaaS profile: expect `AUTH_MODE=multi_user`, PostgreSQL `DATABASE_URL`, `tldw_production=true`, and `PUBLIC_WEB_BASE_URL=https://<public-app-origin>`.
- `REDIS_URL`: Optional Redis URL for sessions (`redis://` or `rediss://`).
- `ENABLE_REGISTRATION`: Enable user registration (`true|false`).
- `REQUIRE_REGISTRATION_CODE`: Require code to register (`true|false`).
- `SECURITY_ALERTS_ENABLED`: Enable AuthNZ security alert dispatching (`true|false`, default `false`).
- `SECURITY_ALERT_MIN_SEVERITY`: Minimum severity to deliver (`low|medium|high|critical`, default `high`).
- `SECURITY_ALERT_FILE_PATH`: JSONL file sink for security alerts (default `Databases/security_alerts.log`).
- `SECURITY_ALERT_WEBHOOK_URL`: Optional webhook endpoint for security alerts (e.g., Slack/PagerDuty).
- `SECURITY_ALERT_WEBHOOK_HEADERS`: JSON object of extra headers for webhook calls (e.g., auth tokens).
- `SECURITY_ALERT_EMAIL_TO`: Comma-separated recipient list for email alerts.
- `SECURITY_ALERT_EMAIL_FROM`: From address for email alerts (required when using SMTP).
- `SECURITY_ALERT_EMAIL_SUBJECT_PREFIX`: Subject prefix for alert emails (default `[AuthNZ]`).
- `SECURITY_ALERT_SMTP_HOST`: SMTP host for email delivery.
- `SECURITY_ALERT_SMTP_PORT`: SMTP port (default `587`).
- `SECURITY_ALERT_SMTP_STARTTLS`: Enable STARTTLS negotiation (`true|false`, default `true`).
- `SECURITY_ALERT_SMTP_USERNAME`: SMTP username (if authentication required).
- `SECURITY_ALERT_SMTP_PASSWORD`: SMTP password/secret.
- `SECURITY_ALERT_SMTP_TIMEOUT`: SMTP connection timeout in seconds (default `10`).
- `SECURITY_ALERT_FILE_MIN_SEVERITY`: Override the global severity threshold for the file sink; choose from `low|medium|high|critical`.
- `SECURITY_ALERT_WEBHOOK_MIN_SEVERITY`: Override the global severity threshold for the webhook sink.
- `SECURITY_ALERT_EMAIL_MIN_SEVERITY`: Override the global severity threshold for email delivery.
- `SECURITY_ALERT_BACKOFF_SECONDS`: Cooldown applied after a sink fails before retrying (default `30`).
- `SHOW_API_KEY_ON_STARTUP`: In single-user mode, show API key once at startup (`true|false`). Avoid in production.
- `REDIS_ENABLED`: Boolean hint used in logs/metrics reporting.

## Billing
- `BILLING_ALLOWED_REDIRECT_HOSTS`: Comma-separated exact or wildcard host allowlist for checkout success, cancel, and billing portal return URLs.
- `BILLING_REDIRECT_ALLOWLIST_REQUIRED`: Require `BILLING_ALLOWED_REDIRECT_HOSTS` to be set before billing redirects are accepted (`true|false`).
- `BILLING_REDIRECT_REQUIRE_HTTPS`: Reject non-HTTPS billing redirect URLs when enabled (`true|false`).
- Hosted SaaS profile: set `BILLING_REDIRECT_ALLOWLIST_REQUIRED=true`, `BILLING_REDIRECT_REQUIRE_HTTPS=true`, and include the `PUBLIC_WEB_BASE_URL` host in `BILLING_ALLOWED_REDIRECT_HOSTS`.

Config file support (optional):
- Section `[AuthNZ]` in `Config_Files/config.txt` can define: `auth_mode`, `database_url`, `jwt_secret_key`, `single_user_api_key`, `enable_registration`, `require_registration_code`, `rate_limit_enabled`, `rate_limit_per_minute`, `rate_limit_burst`, `access_token_expire_minutes`, `refresh_token_expire_days`, `public_web_base_url`, `public_password_reset_path`, `public_email_verification_path`, `public_magic_link_path`, `redis_url`, plus security alert keys (`security_alerts_enabled`, `security_alert_min_severity`, `security_alert_file_path`, `security_alert_webhook_url`, `security_alert_webhook_headers`, `security_alert_email_to`, `security_alert_email_from`, `security_alert_email_subject_prefix`, `security_alert_smtp_host`, `security_alert_smtp_port`, `security_alert_smtp_starttls`, `security_alert_smtp_username`, `security_alert_smtp_password`, `security_alert_smtp_timeout`, `security_alert_file_min_severity`, `security_alert_webhook_min_severity`, `security_alert_email_min_severity`).
- Section `[Image-Generation]` in `Config_Files/config.txt` can define:
  - General: `default_backend`, `enabled_backends`
  - Limits: `max_width`, `max_height`, `max_pixels`, `max_steps`, `max_prompt_length`, `inline_max_bytes`
  - `sd_cpp_*`: `sd_cpp_binary_path`, `sd_cpp_diffusion_model_path`, `sd_cpp_model_path`, `sd_cpp_llm_path`, `sd_cpp_vae_path`, `sd_cpp_lora_paths`, `sd_cpp_allowed_extra_params`, `sd_cpp_default_steps`, `sd_cpp_default_cfg_scale`, `sd_cpp_default_sampler`, `sd_cpp_device`, `sd_cpp_timeout_seconds`
  - `swarmui_*`: `swarmui_base_url`, `swarmui_default_model`, `swarmui_swarm_token`, `swarmui_allowed_extra_params`, `swarmui_timeout_seconds`
  - `openrouter_*`: `openrouter_image_base_url`, `openrouter_image_api_key`, `openrouter_image_default_model`, `openrouter_image_allowed_extra_params`, `openrouter_image_timeout_seconds`
  - `novita_*`: `novita_image_base_url`, `novita_image_api_key`, `novita_image_default_model`, `novita_image_allowed_extra_params`, `novita_image_timeout_seconds`, `novita_image_poll_interval_seconds`
  - `together_*`: `together_image_base_url`, `together_image_api_key`, `together_image_default_model`, `together_image_allowed_extra_params`, `together_image_timeout_seconds`
  - `modelstudio_*`: `modelstudio_image_base_url`, `modelstudio_image_api_key`, `modelstudio_image_default_model`, `modelstudio_image_region`, `modelstudio_image_mode`, `modelstudio_image_poll_interval_seconds`, `modelstudio_image_timeout_seconds`, `modelstudio_image_allowed_extra_params`

## Chat / UI
- `CHAT_SAVE_DEFAULT`: Persist new chats by default (`true|false`).
- `DEFAULT_CHAT_SAVE`: Legacy alias; same as above.
- `CHAT_STREAM_INCLUDE_METADATA`: Include `tldw_*` IDs in chat SSE streaming chunks (`true|false`, default `true`). Set `false` for strict OpenAI streaming compatibility.
- `PERSONA_EXEMPLAR_DEFAULT_BUDGET_TOKENS`: Default persona exemplar budget for character chat when request override is omitted (default `600`, clamped to `1..20000`).
- `PERSONA_IOO_BUDGET_AUTO_ADJUST_ENABLED`: Auto-adjust persona exemplar budget after sustained IOO alerts (`true|false`, default `true`).
- `PERSONA_IOO_BUDGET_AUTO_REDUCTION_FACTOR`: Multiplicative downshift applied when auto-adjust triggers (default `0.75`, clamped to `0.10..0.95`).
- `PERSONA_IOO_BUDGET_AUTO_MIN_TOKENS`: Lower bound for auto-adjusted persona exemplar budget (default `240`, clamped to `1..20000`).

### Tokenizer (Chat Dictionaries & World Books)
- `TOKEN_ESTIMATOR_MODE`: `whitespace` (default) or `char_approx`
  - `whitespace` counts whitespace-separated tokens.
  - `char_approx` estimates by character length (≈ length/divisor).
- `TOKEN_CHAR_APPROX_DIVISOR`: Integer divisor for `char_approx` (default `4`).

Runtime overrides (non-persistent) are available via API:
- `GET /api/v1/config/tokenizer` → read current mode/divisor
- `PUT /api/v1/config/tokenizer` → update mode/divisor in memory

## Usage Logging & Aggregators
- `USAGE_LOG_ENABLED`: Enable lightweight HTTP usage logging middleware (`true|false`, default `false`).
- `USAGE_LOG_EXCLUDE_PREFIXES`: JSON array of path prefixes to skip (default includes `/docs`, `/metrics`, `/static`). Example: `USAGE_LOG_EXCLUDE_PREFIXES='["/docs","/metrics"]'`.
- `USAGE_AGGREGATOR_INTERVAL_MINUTES`: Background aggregation cadence for `usage_daily` (default `60`).
- `USAGE_LOG_RETENTION_DAYS`: Retain `usage_log` rows for this many days; daily job prunes older rows (default `180`).
- `USAGE_LOG_DISABLE_META`: When `true`, do not store IP/User-Agent in `usage_log.meta` (stores `{}`) regardless of `PII_REDACT_LOGS`.
- `DISABLE_USAGE_AGGREGATOR`: When `true`, skip starting the HTTP usage background aggregator at startup (env-only override).

- `LLM_USAGE_ENABLED`: Enable per-request LLM usage logging (`true|false`, default `true`). Can also be set via env and respected by the tracker.
- `LLM_USAGE_AGGREGATOR_ENABLED`: Enable background aggregation of `llm_usage_log` into `llm_usage_daily` (`true|false`, default `true`).
- `LLM_USAGE_AGGREGATOR_INTERVAL_MINUTES`: Background LLM aggregation cadence in minutes (default `60`).
- `LLM_USAGE_LOG_RETENTION_DAYS`: Retain `llm_usage_log` rows for this many days; daily job prunes older rows (default `180`).
- `PRIVILEGE_SNAPSHOT_RETENTION_DAYS`: Keep privilege snapshots at full granularity for this many days before weekly downsampling (default `90`).
- `PRIVILEGE_SNAPSHOT_WEEKLY_RETENTION_DAYS`: Retain the downsampled weekly snapshots for this many days before purging entirely (default `365`).
- `PRIVILEGE_MAP_CACHE_TTL_SECONDS`: TTL for cached privilege summaries in seconds (default `120`, floor `10`). Controls the in-process and distributed cache expiry.
- `PRIVILEGE_CACHE_BACKEND`: `memory` (default) keeps cache local to each worker; set to `redis` to enable distributed caching with pub/sub invalidation.
- `PRIVILEGE_CACHE_REDIS_URL`: Redis connection string used when `PRIVILEGE_CACHE_BACKEND=redis`. Falls back to `REDIS_URL` if unset.
- `PRIVILEGE_CACHE_NAMESPACE`: Optional namespace prefix for distributed cache keys/channels (default `privmap`).
- `PRIVILEGE_CACHE_SLIDING_TTL`: `1|true` (default) refreshes Redis TTL on reads; set to `0|false` to keep a fixed expiry.
- `PRIVILEGE_CACHE_GENERATION_SYNC_SECONDS`: Polling interval (seconds) for generation checks when Redis pub/sub is unavailable (default `2`).
- `DISABLE_LLM_USAGE_AGGREGATOR`: When `true`, skip starting the LLM usage background aggregator at startup (env-only override).

## LLM Pricing
- `PRICING_OVERRIDES`: JSON object to override model/provider pricing used to compute costs. Example:
  ``
  export PRICING_OVERRIDES='{"openai":{"gpt-4o":{"prompt":0.005,"completion":0.015}}}'
  ``
  File-based overrides are also supported at `tldw_Server_API/Config_Files/model_pricing.json`.
  In addition to cost tracking, this catalog now seeds the available models list for commercial providers
  surfaced by `GET /api/v1/llm/providers`. Add a model here to have it appear in the WebUI model selectors
  (you can still list models in `config.txt`; both sources are merged, with `model_pricing.json` acting as
  the primary reference).

## Embeddings
- `EMBEDDINGS_DEDUPE_TTL_SECONDS`: Dedupe window for worker replay suppression. Defaults to `3600` seconds. Workers compute a stage-specific dedupe key (or use `dedupe_key`/`idempotency_key` if provided) and suppress processing if the same key was seen within this TTL.
- `TRUSTED_HF_REMOTE_CODE_MODELS`: Comma-separated allowlist patterns for models that require `trust_remote_code=True` (e.g., `NovaSearch/stella_en_400M_v5,BAAI/*bge*`). This allowlist is also consulted by the Transformers reranker; `mxbai-rerank*` models are auto-enabled for reranking without extra config.
- `ALLOW_ZERO_EMBEDDINGS_MEDIA_TYPES`: Comma-separated media types that may legitimately yield zero embeddings (e.g., `audio,video`). When set, media-embeddings jobs for these types complete successfully even if no vectors are stored.

### Backpressure & Quotas
- `EMB_BACKPRESSURE_MAX_DEPTH`: Maximum depth across core embeddings queues (`embeddings:chunking`, `embeddings:embedding`, `embeddings:storage`, `embeddings:content`) before ingest/embeddings endpoints return HTTP 429 with `Retry-After`. Default: `25000`.
- `EMB_BACKPRESSURE_MAX_AGE_SECONDS`: Maximum age (seconds) of the oldest message across core embeddings queues before HTTP 429. Default: `300`.
- `EMBEDDINGS_TENANT_RPS`: Per-tenant requests per second limit for embeddings endpoints (multi-tenant mode only). `0` disables. Default: `0`.
- `INGEST_TENANT_RPS`: Per-tenant requests per second limit for ingestion endpoints (multi-tenant mode). Falls back to `EMBEDDINGS_TENANT_RPS` if unset. `0` disables. Default: `0`.
- `EMBEDDINGS_REDIS_URL`: Redis URL for embeddings Redis Streams (falls back to `REDIS_URL`).

### Redis Streams Worker (Embeddings)
- `EMBEDDINGS_REDIS_STREAM_{STAGE}`: Override stream names for `CHUNKING`, `EMBEDDING`, `STORAGE`, `CONTENT`.
- `EMBEDDINGS_REDIS_GROUP_{STAGE}`: Override consumer group names for `CHUNKING`, `EMBEDDING`, `STORAGE`, `CONTENT`.
- `EMBEDDINGS_REDIS_WORKERS_{STAGE}`: Worker count per stage when running `redis_worker` (default `1`).
- `EMBEDDINGS_REDIS_POLL_INTERVAL_MS`: XREADGROUP block interval in milliseconds (default `1000`).
- `EMBEDDINGS_REDIS_BATCH_SIZE`: XREADGROUP batch size (default `1`).
- `EMBEDDINGS_REDIS_MAX_RETRIES`: Retry count per message before DLQ (default `2`).
- `EMBEDDINGS_REDIS_RETRY_BACKOFF_BASE`: Base backoff seconds for retries (default `2`).
- `EMBEDDINGS_REDIS_RETRY_BACKOFF_MAX`: Max backoff seconds for retries (default `30`).
- `EMBEDDINGS_REDIS_IDEMPOTENCY_TTL`: Enqueue idempotency TTL seconds (default `86400`).
- `EMBEDDINGS_REDIS_DLQ_PREFIX`: DLQ stream prefix (default `embeddings:dlq`).
- `EMBEDDINGS_REDIS_ALLOW_STUB`: Allow in-memory stub Redis client for local runs/tests (`true|false`, default `false`).

### Priority Queues
- `EMBEDDINGS_PRIORITY_ENABLED`: Enable per-stage priority sub-queues with weighted fair consumption (`true|false`). Default: `false`.
- `EMBEDDINGS_PRIORITY_WEIGHTS`: Comma-separated weights for `high`, `normal`, `low` priority buckets used by workers when `EMBEDDINGS_PRIORITY_ENABLED=true`. Example: `high:5,normal:3,low:1` (default).

### Vector Store: pgvector
- `RAG.vector_store_type`: Set to `pgvector` to activate the pgvector adapter (default `chromadb`).
  - Important: For normal server operation, pgvector connection settings are sourced from `config.txt` and are not overridden by environment variables. Tests and helper scripts may still use env-only DSNs.
- Test/runtime variables (used by tests/scripts; not overriding server pgvector settings):
  - `PGVECTOR_HOST`: Postgres host (default `localhost`).
  - `PGVECTOR_PORT`: Postgres port (default `5432`).
  - `PGVECTOR_DATABASE`: Database name (default `postgres`).
  - `PGVECTOR_USER`: Username (default `postgres`).
  - `PGVECTOR_PASSWORD`: Password (no default).
  - `PGVECTOR_SSLMODE`: SSL mode (default `prefer`).
  - `PGVECTOR_DSN`: Optional DSN string.
  - `PGVECTOR_POOL_SIZE`: Optional connection pool size (default `5`).
  - `PGVECTOR_HNSW_EF_SEARCH`: Optional session `ef_search` for HNSW queries (default `64`).

Quick start (local dev):
- `docker-compose -f docker-compose.pg.yml up -d` to start Postgres with pgvector.
- Set `RAG.vector_store_type=pgvector` and either `PGVECTOR_DSN` or the discrete `PGVECTOR_*` vars.
- Vector Store API (`/api/v1/vector_stores`) and the embeddings storage worker will use pgvector when configured.

## LLM Provider Keys
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `COHERE_API_KEY`, `DEEPSEEK_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY`, `HUGGINGFACE_API_KEY`, `MISTRAL_API_KEY`, `OPENROUTER_API_KEY`, `QWEN_API_KEY`
- Additional provider-specific variables as required by their APIs.
- `OPENROUTER_MODEL_DISCOVERY_TTL_SECONDS`: TTL for cached OpenRouter `/models` discovery results used by `/api/v1/llm/models/metadata` (default `600`, minimum `30`). Use `refresh_openrouter=true` on the metadata endpoint to force an immediate refresh.

### Qwen / Model Studio Routing
- `QWEN_BASE_URL`: Overrides Qwen chat base URL directly (highest non-request precedence).
- `QWEN_REGION`: Region preset for Qwen chat when `QWEN_BASE_URL` and `qwen_api.api_base_url` are unset (`sg|cn|us`).
- `DASHSCOPE_API_KEY`: Optional fallback API key for `modelstudio` image backend when `modelstudio_image_api_key` is unset.
- `DASHSCOPE_BASE_URL`: Optional fallback base URL for `modelstudio` image backend.
- `MODELSTUDIO_IMAGE_BASE_URL`: Env override for Model Studio image base URL.
- `MODELSTUDIO_IMAGE_REGION`: Region preset for Model Studio image backend when no explicit base URL override is set (`sg|cn|us`).
- `MODELSTUDIO_IMAGE_MODEL`: Env override for Model Studio default image model.

## MCP Unified
- `MCP_JWT_SECRET`: Secret used by the MCP server for issuing/verifying tokens.
- `MCP_API_KEY_SALT`: Salt used for API key hashing/derivation.
- `MCP_LOG_LEVEL`: MCP module log level (`DEBUG|INFO|WARNING|ERROR`).

## OCR - POINTS Reader (optional)
- `POINTS_MODE`: `sglang` or `transformers` (default: auto).
- `POINTS_SGLANG_URL`: SGLang chat/completions endpoint (e.g., `http://127.0.0.1:8081/v1/chat/completions`).
- `POINTS_SGLANG_MODEL`: Model name in SGLang server (e.g., `WePoints`).

## OCR - HunyuanOCR (optional)
- `HUNYUAN_MODE`: `auto` | `vllm` | `transformers` (default: `auto`).
- `HUNYUAN_PROMPT`: Prompt override (free-form).
- `HUNYUAN_PROMPT_PRESET`: `general|doc|table|spotting|json` (used when `HUNYUAN_PROMPT` is unset).
- vLLM:
  - `HUNYUAN_VLLM_URL`: OpenAI-compatible `/v1/chat/completions` endpoint.
  - `HUNYUAN_VLLM_MODEL`: Model name (served-model-name).
  - `HUNYUAN_VLLM_TIMEOUT`: Request timeout seconds (default `60`).
  - `HUNYUAN_VLLM_USE_DATA_URL`: `true|false` (default `true`).
- Transformers:
  - `HUNYUAN_MODEL_PATH`: HF model id or local path (default: `tencent/HunyuanOCR`).
  - `HUNYUAN_DEVICE`: Optional device override (`cuda`, `cpu`, etc.).
- Generation:
  - `HUNYUAN_MAX_NEW_TOKENS`, `HUNYUAN_TEMPERATURE`, `HUNYUAN_DO_SAMPLE`.
- Post-processing:
  - `HUNYUAN_CLEAN_REPEATS`: `true|false` (default `true`).

## OCR - DeepSeek (optional)
- `DEEPSEEK_OCR_MODEL_ID`: HF model id or local path (default: `deepseek-ai/DeepSeek-OCR`).
- `DEEPSEEK_OCR_PROMPT`: Prompt override (default: layout-aware markdown conversion).
- `DEEPSEEK_OCR_BASE_SIZE`: Base resolution size (default: `1024`).
- `DEEPSEEK_OCR_IMAGE_SIZE`: Secondary resolution size (default: `640`).
- `DEEPSEEK_OCR_CROP_MODE`: `true|false` (default: `true`).
- `DEEPSEEK_OCR_SAVE_RESULTS`: `true|false` (default: `false`).
- `DEEPSEEK_OCR_TEST_COMPRESS`: `true|false` (default: `false`).
- `DEEPSEEK_OCR_DTYPE`: `bfloat16|float16|float32` (default: `bfloat16`).
- `DEEPSEEK_OCR_ATTN_IMPL`: Attention implementation (default: `flash_attention_2`).
- `DEEPSEEK_OCR_DEVICE`: `cuda|cpu` (default: `cuda`).
- `DEEPSEEK_OCR_OUTPUT_DIR`: Optional output directory used when `DEEPSEEK_OCR_SAVE_RESULTS=true`.

## Workflows (File Access)
- `WORKFLOWS_FILE_BASE_DIR`: Base directory for workflow `file://` access. Relative paths resolve from the project root; defaults to the per-user base dir under `USER_DB_BASE_DIR` (with a `Databases/` fallback).
- `WORKFLOWS_ALLOW_UNSAFE_FILE_ACCESS`: `true|false` - allow workflow file access outside the per-user base dir, but only under allowlisted base directories (default `false`).
- `WORKFLOWS_FILE_ALLOWLIST`: Comma- or newline-separated list of allowed base directories for unsafe file access; relative paths resolve from the project root.
- `WORKFLOWS_FILE_ALLOWLIST_<TENANT>`: Optional per-tenant override (uppercase, `-` replaced by `_`); when set, it replaces the global allowlist for that tenant (comma- or newline-separated).

## Scheduler
- `SCHEDULER_DATABASE_URL`: Database URL for the core task scheduler. Defaults to `sqlite:///PROJECT_ROOT/Databases/scheduler.db` (test mode uses a per-process temp file). Set this to place the scheduler DB alongside other DBs.
- `SCHEDULER_BASE_PATH`: Base path for the scheduler’s payload storage. Defaults to `PROJECT_ROOT/Databases/scheduler`.
- `WORKFLOWS_SCHEDULER_DATABASE_URL`: Optional override for the Workflows Scheduler (cron) persistence; if using SQLite and not set, it defaults to the per-user path under `USER_DB_BASE_DIR/<user_id>/workflows/workflows_scheduler.db`.
- `WORKFLOWS_SCHEDULER_RESCAN_SEC`: Interval (seconds) for the Workflows Scheduler to rescan all users for new/removed schedules. Default: `600`.
- `POINTS_MODEL_PATH`: HF model path when running locally (e.g., `tencent/POINTS-Reader`).
- `POINTS_PROMPT`: Optional prompt override.

## Notes
- Many subsystems also support file-based configuration under `Config_Files/` and module-specific YAML files (e.g., TTS provider config). Environment variables always take precedence when present.

## STT vNext Controls

These settings back the canonical `get_stt_config()` loader and apply to REST `/api/v1/audio/transcriptions`, WS `/api/v1/audio/stream/transcribe`, and STT persistence paths.

- `STT_WS_CONTROL_V2_ENABLED`: Enable explicit WS control v2 negotiation (`true|false`, default `false`). Config key: `[STT-Settings] ws_control_v2_enabled`.
- `STT_PAUSED_AUDIO_QUEUE_CAP_SECONDS`: Paused-audio queue cap in seconds for WS control v2 (default `2.0`). Config key: `[STT-Settings] paused_audio_queue_cap_seconds`.
- `STT_OVERFLOW_WARNING_INTERVAL_SECONDS`: Minimum interval between paused-queue overflow warnings (default `5.0`). Config key: `[STT-Settings] overflow_warning_interval_seconds`.
- `STT_TRANSCRIPT_DIAGNOSTICS_ENABLED`: Emit deterministic final/full transcript diagnostics (`true|false`, default `false`). Config key: `[STT-Settings] transcript_diagnostics_enabled`.
- `STT_DELETE_AUDIO_AFTER_SUCCESS`: Delete raw audio after successful transcription (`true|false`, default `true`). Legacy alias: `STT_DELETE_AUDIO_AFTER`. Config keys: `[STT-Settings] delete_audio_after_success` or `delete_audio_after`.
- `STT_AUDIO_RETENTION_HOURS`: Default retained-audio TTL in hours (default `0.0`). Config key: `[STT-Settings] audio_retention_hours`.
- `STT_REDACT_PII`: Enable transcript redaction (`true|false`, default `false`). Config key: `[STT-Settings] redact_pii`.
- `STT_ALLOW_UNREDACTED_PARTIALS`: Allow unredacted partial WS frames when policy permits it (`true|false`, default `false`). Config key: `[STT-Settings] allow_unredacted_partials`.
- `STT_REDACT_CATEGORIES`: Comma-separated or JSON list of redact categories. Config key: `[STT-Settings] redact_categories`.

Policy precedence:
- Multi-user mode: org STT settings override global defaults.
- Single-user mode: global defaults only.
- Request-level overrides may only be stricter than the effective policy.

Operator notes:
- Retention TTL is only meaningful when retained artifacts are indexed; otherwise `delete_audio_after_success=true` remains the safe default.
- Org admins manage per-org policy with `GET/PATCH /api/v1/admin/orgs/{org_id}/stt/settings`.

## TTS Placeholder Handling (2026-03-02)
- Legacy placeholder literals in `[TTS-Settings]` are now treated as unset during config load: empty string, `FIXME`, `TODO`, `TBD`, `CHANGE_ME`, `PLACEHOLDER`, `NONE`, `NULL`, `N/A`, `NA`.
- When placeholders are encountered, safe defaults are applied:
  - `default_google_tts_model`: `en-US`
  - `default_google_tts_voice`: `en-US-Neural2-A`
  - `default_eleven_tts_model`: `eleven_monolingual_v1`
  - `default_eleven_tts_voice`: `pNInz6obpgDQGcFmaJgB`
  - `default_eleven_tts_language_code`: `en`
  - `default_eleven_tts_voice_stability`: `0.5`
  - `default_eleven_tts_voice_similiarity_boost`: `0.75`
  - `default_eleven_tts_voice_style`: `0.0`
  - `default_eleven_tts_voice_use_speaker_boost`: `true`

## Telemetry & Observability

- OpenTelemetry service identity
  - `OTEL_SERVICE_NAME`: Logical service name (default `tldw_server`).
  - `OTEL_SERVICE_VERSION`: Service version string (default `1.0.0`).
  - `OTEL_SERVICE_NAMESPACE`: Namespace grouping (default `production`).
  - `DEPLOYMENT_ENV`: Deployment environment label (default `development`).

- Exporters and enablement
  - `ENABLE_METRICS`: Enable metrics pipeline (`true|false`, default `true`).
  - `ENABLE_TRACING`: Enable tracing pipeline (`true|false`, default `true`).
  - `ENABLE_OTEL_LOGGING`: Enable OTEL logging integration (`true|false`, default `false`).
  - `ENABLE_OTEL_CONSOLE_METRICS_EXPORTER`: Add the console metrics exporter (`true|false`, default `false`).
  - `METRICS_RING_BUFFER_MAXLEN_OR_UNBOUNDED`: Rolling metrics sample window size (default `10000`). Set `0` or a negative value for an unbounded buffer.
  - `METRICS_CUMULATIVE_SERIES_MAX_PER_METRIC`: Hard cap for in-memory cumulative label sets per metric (default `10000`). New STT metric families rely on bounded-label mapping and this cap together.
  - `OTEL_METRICS_EXPORTER`: Comma list of metrics exporters (`prometheus` by default).
  - `OTEL_TRACES_EXPORTER`: Comma list of traces exporters (`console` by default).

- Prometheus (pull/endpoint exporter)
  - `PROMETHEUS_HOST`: Bind host for Prometheus exporter (default `0.0.0.0`).
  - `PROMETHEUS_PORT`: Bind port for Prometheus exporter (default `9090`).

- OTLP (push exporters for traces/metrics)
  - `OTEL_EXPORTER_OTLP_ENDPOINT`: e.g., `http://otel-collector:4317`.
  - `OTEL_EXPORTER_OTLP_PROTOCOL`: `grpc` or `http/protobuf` (default `grpc`).
  - `OTEL_EXPORTER_OTLP_HEADERS`: Optional headers (e.g., `authorization=Bearer <token>`).
  - `OTEL_EXPORTER_OTLP_INSECURE`: Allow insecure transport (`true|false`, default `true`).

Notes
- Metrics/OTEL wiring is initialized in the server; see `tldw_Server_API/app/core/Metrics/telemetry.py` for defaults.
- When `OTEL_METRICS_EXPORTER` includes `prometheus`, the server exposes a scrape endpoint consumed by Prometheus; the port/host are controlled by `PROMETHEUS_*` above.

### Quick Defaults

| Variable                        | Default             | Notes |
|---------------------------------|---------------------|-------|
| `OTEL_SERVICE_NAME`             | `tldw_server`       | Logical service name |
| `OTEL_SERVICE_VERSION`          | `1.0.0`             | Freeform version string |
| `OTEL_SERVICE_NAMESPACE`        | `production`        | Logical namespace/group |
| `DEPLOYMENT_ENV`                | `development`       | Environment label |
| `ENABLE_METRICS`                | `true`              | Enable metrics pipeline |
| `ENABLE_TRACING`                | `true`              | Enable tracing pipeline |
| `ENABLE_OTEL_LOGGING`           | `false`             | Enable OTEL logging integration |
| `ENABLE_OTEL_CONSOLE_METRICS_EXPORTER` | `false`      | Add console metrics exporter |
| `METRICS_RING_BUFFER_MAXLEN_OR_UNBOUNDED` | `10000`    | Rolling metrics sample window size |
| `METRICS_CUMULATIVE_SERIES_MAX_PER_METRIC` | `10000`   | Hard cap for cumulative label sets per metric |
| `OTEL_METRICS_EXPORTER`         | `prometheus`        | Comma-separated exporters |
| `OTEL_TRACES_EXPORTER`          | `console`           | Comma-separated exporters |
| `PROMETHEUS_HOST`               | `0.0.0.0`           | Bind host for Prometheus exporter |
| `PROMETHEUS_PORT`               | `9090`              | Bind port for Prometheus exporter |
| `OTEL_EXPORTER_OTLP_ENDPOINT`   | (empty)             | e.g., `http://otel-collector:4317` |
| `OTEL_EXPORTER_OTLP_PROTOCOL`   | `grpc`              | `grpc` or `http/protobuf` |
| `OTEL_EXPORTER_OTLP_HEADERS`    | (empty)             | Optional headers string |
| `OTEL_EXPORTER_OTLP_INSECURE`   | `true`              | Allow insecure transport |
| `STREAMS_UNIFIED`               | `0`                 | Feature flag: unified SSE/WS streams for pilot endpoints. Recommended `1` in non‑prod (default via dev/test compose overlays). Use the dev overlay: `Dockerfiles/docker-compose.dev.yml`. |

Quick rollback

- To disable unified streaming quickly, set `STREAMS_UNIFIED=0` and restart the app (or `docker compose up -d` to re‑create with the new env). This reverts pilot endpoints to legacy streaming code paths.

Non‑prod defaults

- `Dockerfiles/docker-compose.dev.yml` exports `STREAMS_UNIFIED=1` for dev/staging overlays.
- `Dockerfiles/docker-compose.test.yml` also sets `STREAMS_UNIFIED=1` for test environments.
  In production, keep the flag unset or `0` until you explicitly opt into unified streams or are ready to flip them on by default.
| `STREAM_HEARTBEAT_INTERVAL_S`   | `10`                | Default heartbeat interval for streams (seconds) |
| `STREAM_HEARTBEAT_MODE`         | `comment`           | `comment` or `data` heartbeats (prefer `data` behind reverse proxies) |
| `STREAM_IDLE_TIMEOUT_S`         | (disabled)          | Idle timeout for SSE streams (seconds) |
| `AUDIO_WS_IDLE_TIMEOUT_S`       | (disabled)          | Optional idle timeout for Audio WebSocket (seconds); overrides `STREAM_IDLE_TIMEOUT_S` for audio handler |
| `AUDIO_WS_QUOTA_CLOSE_1008`     | `0`                 | When `1`, Audio WS closes with 1008 for quota/rate-limit instead of legacy 4003 |
| `AUDIO_WS_COMPAT_ERROR_TYPE`    | `1`                 | When `1`, Audio WS error payloads include legacy `error_type` alias in addition to canonical `code`; set `0` to disable alias during migration |
| `STREAM_MAX_DURATION_S`         | (disabled)          | Maximum duration for SSE streams (seconds) |
| `STREAM_QUEUE_MAXSIZE`          | `256`               | Default bounded queue size for SSE streams |
| `STREAM_PROVIDER_CONTROL_PASSTHRU` | `0`              | Preserve provider SSE control lines (`event/id/retry`) when `1` |

## Prometheus & Grafana (deployment)

- Grafana container (see compose service `grafana`)
  - `GF_SECURITY_ADMIN_USER`: Admin user (default `admin` in samples).
  - `GF_SECURITY_ADMIN_PASSWORD`: Admin password (default `admin` in samples; change in production).
  - `GF_AUTH_ANONYMOUS_ENABLED`: Enable anonymous access (`true|false`).
  - `GF_AUTH_ANONYMOUS_ORG_ROLE`: Role for anonymous users (e.g., `Viewer`).
  - `GF_PLUGINS_PREINSTALL`: Optional list of plugins to preinstall (preferred over deprecated `GF_INSTALL_PLUGINS`).

- Prometheus container
  - Configured via mounted `prometheus.yml`; see `tldw_Server_API/Config_Files/prometheus.yml` (sample provided).
  - No mandatory env vars by default; override scrape targets in the YAML.

Dashboards/Alerts/Annotations
- Dashboards are provisioned from `Docs/Deployment/Monitoring/` (mounted to `/var/lib/grafana/dashboards`).
- Alerts are provisioned from `Docs/Deployment/Monitoring/Alerts` (mounted to `/etc/grafana/provisioning/alerting`).
- A sample Prometheus-backed annotations source is provisioned from `Samples/Grafana/provisioning/annotations/deploys.yml`.
  - Push a metric like `tldw_deploy_info{version="vX.Y.Z",git_sha="..."} 1` at deploy time to see release markers.

## Monitoring & Alerts
- Topic Monitoring (watchlists and alerting):
  - `MONITORING_WATCHLISTS_FILE`: JSON file with watchlists (default `tldw_Server_API/Config_Files/monitoring_watchlists.json`).
  - `MONITORING_ALERTS_DB`: SQLite DB path for topic alerts (default `Databases/monitoring_alerts.db`).
  - `TOPIC_MONITOR_MAX_SCAN_CHARS`: Max characters scanned per text (default `200000`).
  - `TOPIC_MONITOR_DEDUP_SECONDS`: Deduplication window to avoid repeated alerts for same (user,watchlist,pattern,source) (default `300`).

- Notifications (scaffold; local-first with optional external hooks):
  - `MONITORING_NOTIFY_ENABLED`: Enable notification output (`true|false`, default `false`).
  - `MONITORING_NOTIFY_MIN_SEVERITY`: Minimum severity to notify (`info|warning|critical`, default `critical`).
  - `MONITORING_NOTIFY_FILE`: JSONL file sink for notifications (default `Databases/monitoring_notifications.log`).
  - `MONITORING_NOTIFY_WEBHOOK_URL`: Optional HTTP webhook URL (best-effort, async, retries).
  - `MONITORING_NOTIFY_EMAIL_TO`: Optional email recipient (comma not supported; one address).
  - `MONITORING_NOTIFY_EMAIL_FROM`: Sender email address (defaults to SMTP user if unset).
  - `MONITORING_NOTIFY_SMTP_HOST`: SMTP server host (required for email).
  - `MONITORING_NOTIFY_SMTP_PORT`: SMTP port (default `587`).
  - `MONITORING_NOTIFY_SMTP_STARTTLS`: Enable STARTTLS (`true|false`, default `true`).
  - `MONITORING_NOTIFY_SMTP_USER`: SMTP auth username (optional if server allows anon relay; not recommended).
- `MONITORING_NOTIFY_SMTP_PASSWORD`: SMTP auth password.

Notes:
- Monitoring alerts do not block or modify content; they create reviewable signals for admins.
- Webhook/email delivery is best-effort and runs in background threads with small timeouts and retries.

- Claims monitoring (alerts + digests):
  - `CLAIMS_MONITORING_ENABLED`: Master toggle for claims monitoring.
  - `CLAIMS_ALERT_THRESHOLD_DEFAULT`: Default unsupported ratio threshold.
  - `CLAIMS_ALERTS_SCHEDULER_ENABLED`: Enable periodic alert evaluation.
  - `CLAIMS_ALERTS_EVAL_INTERVAL_SEC`: Scheduler interval (seconds).
  - `CLAIMS_ALERTS_WINDOW_SEC`: Window for ratio calculations (seconds).
  - `CLAIMS_ALERTS_BASELINE_SEC`: Baseline window (seconds).
  - `CLAIMS_ALERT_EMAIL_DIGEST_ENABLED`: Enable email digest delivery for claims alerts.
  - `CLAIMS_ALERT_EMAIL_DIGEST_INTERVAL_SEC`: Minimum interval between digests per user.
  - `CLAIMS_ALERT_EMAIL_DIGEST_MAX_EVENTS`: Max events per digest batch.
  - `CLAIMS_REBUILD_MAX_QUEUE_ALERT`: Queue size threshold for rebuild alerts.
  - `CLAIMS_REBUILD_HEARTBEAT_WARN_SEC`: Heartbeat staleness threshold.
  - `CLAIMS_PROVIDER_COST_MULTIPLIERS`: Cost map for provider metrics.
  - `CLAIMS_REVIEW_METRICS_SCHEDULER_ENABLED`: Enable nightly review metrics aggregation.
  - `CLAIMS_REVIEW_METRICS_INTERVAL_SEC`: Review metrics scheduler interval (seconds).
  - `CLAIMS_REVIEW_METRICS_LOOKBACK_DAYS`: Days of review log to aggregate per run.
  - Email delivery uses `EMAIL_PROVIDER` (default `mock`) and SMTP settings when enabled.

## Watchlists Module
- `WATCHLIST_OUTPUT_DEFAULT_TTL_SECONDS`: Default retention (seconds) applied to persisted outputs. `0` keeps outputs indefinitely. Defaults to `0`.
- `WATCHLIST_OUTPUT_TEMP_TTL_SECONDS`: Retention (seconds) for temporary outputs (`temporary=true`). Defaults to `86400` (24h).
- `WATCHLIST_TEMPLATE_DIR`: Override directory for watchlist templates (defaults to `Config_Files/templates/watchlists`).
- `EMAIL_PROVIDER`: Delivery backend for NotificationsService (`mock`, `smtp`, ...). Defaults to `mock` for local setups.
- `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_USE_TLS`: SMTP settings consumed by NotificationsService when `EMAIL_PROVIDER=smtp`.


## Privilege Maps Snapshot Workflow
- `PRIVILEGE_METADATA_VALIDATE_ON_STARTUP`: Defaults to `1`. Set to `0` when running tests that inject a fake privilege service to bypass catalog validation.
- Snapshot guard: CI compares the live privilege route registry (collected at runtime) against `tldw_Server_API/tests/fixtures/privilege_route_registry_snapshot.json`. If the snapshot drifts, CI fails with guidance to rerun `python Helper_Scripts/update_privilege_registry_snapshot.py` and commit the refreshed file. Use this script whenever you intentionally add or modify FastAPI routes or privilege dependencies.
