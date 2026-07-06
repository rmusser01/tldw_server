# Workflows (v0.1)

This document captures the current state of the Workflows module, its APIs, data model, execution engine behavior, security model, and the near-term roadmap. It supersedes the placeholder links that were here previously. For the consolidated PRD, see `Docs/Product/Completed/Workflows_PRD.md` (the historical draft remains at `Workflows-PRD-1.md`).

## Status & Scope

- Implemented: Linear workflows with a small, composable step set and a robust runtime (retries, timeouts, pause/resume, cancel, heartbeats, orphan reaping). Streaming run events over WebSocket and HTTP polling. Artifacts persisted and downloadable with guardrails. SQLite default; PostgreSQL supported via shared content backend.
- Non-goals in v0.1: Full graph editor and distributed workers. Minimal branching (branch/map) has been added to enable early DAG-like flows; richer parallelism is planned for v0.2+.

## Module Layout

- Engine and registry: `tldw_Server_API/app/core/Workflows/`
  - `engine.py`: workflow runtime, scheduler, control actions
  - `adapters.py`: step implementations (prompt, rag_search, media_ingest, mcp_tool, webhook)
  - `subprocess_utils.py`: child process lifecycle helpers
  - `registry.py`: available step types
- Persistence: `tldw_Server_API/app/core/DB_Management/Workflows_DB.py` (SQLite and PostgreSQL via `DatabaseBackend`)
- API: `tldw_Server_API/app/api/v1/endpoints/workflows.py` and `.../schemas/workflows.py`
- WebUI: Next.js client in `apps/tldw-frontend`
- PRD: `Workflows-PRD-1.md`

## API Surface

Base prefix: `/api/v1/workflows`

- Definitions
  - `POST /` → Create definition (immutable versions)
  - `GET /` → List active definitions (tenant + owner scoped)
  - `POST /preflight` → Server-authoritative definition preflight. Reuses runtime validation and adds replay-safety warnings for unsafe step types. Supports `validation_mode=block|non-block`.
  - `GET /{workflow_id}` → Get definition (includes stored snapshot)
  - `POST /{workflow_id}/versions` → Create new version
  - `DELETE /{workflow_id}` → Soft delete
- Runs
  - `POST /{workflow_id}/run?mode=async|sync` → Run saved definition (idempotency supported)
  - `POST /run?mode=async|sync` → Run ad-hoc definition (configurable rate-limited)
  - `GET /runs?status=&owner=&workflow_id=&created_after=&created_before=&last_n_hours=&order_by=&order=&limit=&offset=&cursor=` → List runs with filters. Owner by default; admin may filter by `owner`. Returns `runs`, `next_offset` (legacy) and `next_cursor` (opaque continuation token). When `cursor` is present, `offset` is ignored.
  - `GET /runs/{run_id}` → Run status and final outputs
  - `GET /runs/{run_id}/investigation` → Derived diagnostics summary for failed runs. Includes `primary_failure`, `failed_step`, attempt timeline, evidence refs, and recommended actions.
  - `GET /runs/{run_id}/steps` → Step history for the run, including latest failure and attempt counts per step.
  - `GET /runs/{run_id}/steps/{step_id}/attempts` → Attempt-level retry timeline for a logical step execution.
  - `GET /runs/{run_id}/events?since=&types=&limit=&cursor=` → Ordered event stream (HTTP polling). Supports server-side filtering by `types` (comma-separated). Returns `Next-Cursor` header when a full page is returned. When `cursor` is present, `since` is ignored.
  - `WS /ws?run_id=...&token=...` → Live event stream (JWT required; run owner only)
  - `POST /runs/{run_id}/pause|resume|cancel|retry` → Control actions
- Human in the loop
  - `POST /runs/{run_id}/steps/{step_id}/approve|reject` → Approve/reject with optional edits
- Artifacts
  - `GET /runs/{run_id}/artifacts` → List run artifacts
  - `GET /artifacts/{artifact_id}/download` → Download a single artifact (file:// only)
  - `GET /runs/{run_id}/artifacts/download` → Zip and stream all eligible artifacts
  - `GET /runs/{run_id}/artifacts/manifest?verify=` → Per-run artifact manifest; optional checksum verification
- Options discovery
  - `GET /options/chunkers` → Chunker methods, defaults, and basic param schema
  - `GET /step-types` → List available step types for UI builders

- Templates
  - `GET /templates` → List available workflow templates (with `tags` and titles)
  - `GET /templates/{name}` → Fetch a specific template by name
  - `GET /templates/tags` → List aggregated tags across templates

## Control Semantics

- Pause sets run status to `paused` and emits `run_paused`; engine cooperatively idles and maintains step leases. Resume sets `running` and emits `run_resumed`.
- Cancel sets a cancel flag, attempts to terminate recorded subprocesses, updates status to `cancelled`, and emits `run_cancelled`. Adapters should consult `ctx.is_cancelled()`.
- Control endpoints are idempotent; repeated calls do not duplicate state transitions.

## Webhook Lifecycle & Signing

- Completion webhooks are configured via `on_completion_webhook` on the definition. Delivery outcomes are recorded as `webhook_delivery` events with status `delivered|failed|blocked`.
- Egress is policy-controlled (global + per-tenant allow/deny; private IPs blocked by default).
- Signing (v1): the server computes HMAC-SHA256 over `f"{ts}.{body}"` using `WORKFLOWS_WEBHOOK_SECRET` and sets headers:
  - `X-Workflows-Signature-Version: v1`, `X-Workflows-Signature`, `X-Hub-Signature-256`
  - `X-Signature-Timestamp`, `X-Webhook-ID`, `X-Workflow-Id`, `X-Run-Id`
- DLQ worker retries failed deliveries with exponential backoff when enabled.

## Definition Schema (v0.1)

Minimal JSON body for `WorkflowDefinitionCreate`:

```
{
  "name": "hello",
  "version": 1,
  "description": "optional",
  "tags": ["demo"],
  "inputs": {},
  "steps": [
    {"id": "s1", "name": "Greet", "type": "prompt", "config": {"template": "Hello {{ inputs.name }}"}}
  ],
  "metadata": {},
  "visibility": "private"
}
```

Server-enforced limits (config in code):
- Definition size ≤ 256 KB
- ≤ 50 steps per definition
- Step `config` size ≤ 32 KB
- Unknown step types rejected at create/run

## Step Types

Registered under `StepTypeRegistry`:

- `prompt`: Render a prompt template using sandboxed template engine; returns `{ text }`. Supports `simulate_delay_ms` for testing timeouts/retries and optional artifact persistence.
- `rag_search`: Execute RAG search via unified pipeline; returns `{ documents, metadata, timings, citations? }`. Passes through core pipeline options such as reranking, security filters, and generation.
- `media_ingest`: Ingest local files (`file://...`) or optional network sources via yt-dlp/ffmpeg (egress allowlisted). Supports text extraction, chunking strategies, optional indexing into the Media DB, and artifact persistence of downloaded files.
- `mcp_tool`: Execute MCP tools through the unified server registry. Test-friendly fallback for `tool_name=echo`.
- `webhook`: Send events to a URL (HMAC signing and SSRF/egress controls) or dispatch to registered webhooks.
- `wait_for_human`: Pause run with status `waiting_human` until `approve`/`reject`.
- `wait_for_approval`: Pause run with status `waiting_approval` until `approve`/`reject`. Semantically identical wait state, surfaced distinctly for UI/ops.
- `delay`: Pause the workflow for a fixed time (milliseconds). Useful for demos, backoffs or pacing.
- `log`: Log a templated message at the chosen level (`debug|info|warning|error`). Helps with debugging and audit trails.
- `branch`: Evaluate a boolean condition and optionally jump to a target step id (`true_next` / `false_next`).
- `map`: Fan-out over a list and apply a nested step with optional concurrency; returns a list of `results`.

See `adapters.py` for configuration keys and behavior of each step.

### Additional Step Types

The following additional step types are available and surfaced via `/step-types` for schema discovery:

- `tts`: Text-to-speech with optional transcript artifact save and download link attachment. Advanced options include `lang_code`, `normalization_options`, and provider-specific passthrough.
- `process_media`: Ephemeral fetch/process of media via internal services. No DB persistence.
  - Implemented workflow kinds currently include `web_scraping`, `pdf`, and `mediawiki_dump`.
  - Placeholder kinds `ebook`, `xml`, and `podcast` return explicit `{"error":"not_implemented", ...}` responses until real handlers are wired.
- `rss_fetch` / `atom_fetch`: Fetch and parse RSS/Atom feeds; returns `results[]` and concatenated `text` for downstream summarization.
- `embed`: Create vector embeddings for provided text and upsert to a per-user Chroma collection.
- `translate`: Provider-agnostic translation returning translated text and metadata (source/target languages).
- `stt_transcribe`: First-class speech-to-text step (local faster-whisper path supported), optional diarization/word timestamps.
- `notify`: Minimal notifier (Slack/webhook) respecting SSRF/egress policy.
- `diff_change_detector`: Compare previous vs current text and mark `changed` with diff metrics.

### Replay Safety & Preflight

- Each step type exposes replay metadata used by retries, diagnostics, and authoring warnings:
  - `replay_safe`
  - `idempotency_strategy`
  - `compensation_supported`
  - `requires_human_review_for_rerun`
- Unknown or side-effecting steps default to unsafe replay semantics until explicitly classified.
- `POST /preflight` uses the same server validation helpers as run-time definition checks and adds structured warnings such as `unsafe_replay_step`.
- Preflight is the supported way to catch definition-time issues before save/run; client-only checks should be treated as hints, not the final authority.

## Engine Behavior

- Modes: `async` (background via in-process scheduler) and `sync` (server-side synchronous; UI reattaches by `run_id`).
- Lifecycle: `queued → running → (waiting_human|cancelled|failed|succeeded)`.
- Retries/Timeouts: Per-step `retry` (exponential backoff + jitter) and `timeout_seconds` enforced. Cooperative cancellation via a cancel flag checked between sleeps/attempts.
- Execution entities:
  - `workflow_step_runs` remains the logical step execution record for a run.
  - `workflow_step_attempts` records each retry attempt for that logical step, including structured failure metadata.
- Failure taxonomy is layered and attempt-level:
  - `reason_code_core`
  - `reason_code_detail`
  - `category`
  - `blame_scope`
  - `retryable`
  - `retry_recommendation`
- Terminal run failures copy the primary `reason_code_core` into `workflow_runs.status_reason` so run listings and alert triage can pivot quickly without replaying the full event stream.
- Heartbeats/Leases: Step runs record `locked_by/locked_at/lock_expires_at/heartbeat_at`; orphan reaper marks stale step runs as failed and emits events.
- Subprocess Steps: Launched in new process group; engine records `pid/pgid/workdir/stdout/stderr` and escalates termination on cancel/timeout.
- Events: Strict `event_seq` ordering per run; WebSocket emits a `snapshot` and then ordered events with lightweight heartbeats.

## Persistence

- Default: SQLite (`Databases/workflows.db`) with WAL enabled.
- PostgreSQL: Supported when a content backend is configured; schema migrations applied automatically (see tests under `tests/Workflows/test_workflows_postgres_migrations.py`).
- Tables: `workflows`, `workflow_runs`, `workflow_step_runs`, `workflow_step_attempts`, `workflow_events`, `workflow_artifacts` (+ optional `workflow_event_counters`, `workflow_webhook_dlq`).
- Step run vs attempt model:
  - `workflow_step_runs` stores the logical step lifecycle and approval/branching anchor.
  - `workflow_step_attempts` stores retry order, structured reason codes, retryability decisions, and per-attempt metadata/evidence refs.
- Idempotency: `idempotency_key` prevents duplicate run creation per `(tenant_id, user_id, workflow)`.
- Indices/constraints and types:
  - Per-run event ordering: composite index and unique constraint on `(run_id, event_seq)` in `workflow_events`.
  - Cascading deletes: `workflow_events`, `workflow_step_runs`, and `workflow_artifacts` reference `workflow_runs(run_id)` with `ON DELETE CASCADE` (PostgreSQL). New SQLite DBs use the same FK with `PRAGMA foreign_keys=ON`.
  - Partial indexes on statuses: `status='running'`, `status='queued'`, `status='succeeded'`, and `status='failed'` for faster active/queue/history queries (PostgreSQL and modern SQLite).
  - JSON payloads: PostgreSQL stores `workflow_events.payload_json` as `JSONB` with a GIN index; SQLite stores JSON as `TEXT`.
- Pooling & contention:
  - PostgreSQL uses `psycopg_pool` when available (min/max size, timeouts, max lifetime/idle) with `dict_row` rows; falls back to a minimal pool if not installed.
  - SQLite applies `PRAGMA busy_timeout=5000`, `wal_autocheckpoint=1000`; hot write paths include exponential backoff on `database is locked`.

## Security Model

- AuthNZ: All HTTP endpoints use standard API auth; WS requires a JWT and enforces run-owner equality (subject must match `run.user_id`).
- Tenant Isolation: Read operations enforce tenant boundaries, and HTTP reads now enforce run-owner or admin (consistent with WS).
- Rate Limits: Ad-hoc runs and run-saved endpoints are rate-limited via RG policies.
  - When `TEST_MODE=true` is set, endpoint rate limits and restrictive RG policies are disabled. Stress tests (`TLDW_WORKFLOW_STRESS=1`) re-enable rate limits to validate throttling behavior.
- Egress Controls: Webhook step checks URL via `is_url_allowed` to block private IPs/SSRF; optional HMAC signature header.
- Artifact Downloads: Only `file://` URIs; size and MIME allowlists enforced; basic path containment checks.

### RBAC Claims Exposure

`get_request_user` now enriches the request user with AuthNZ-backed RBAC claims:
- `roles`: names from `user_roles`
- `permissions`: aggregated from role permissions and explicit `user_permissions` overrides
- `is_admin`: true if `admin` role or `is_superuser`

In single-user mode, the fixed user is exposed with admin-like claims for compatibility.

## Configuration

- Concurrency limits: `WORKFLOWS_TENANT_CONCURRENCY` (default 2), `WORKFLOWS_WORKFLOW_CONCURRENCY` (default 1)
- Ad-hoc runs: `WORKFLOWS_DISABLE_ADHOC=true` to disable
- Artifacts: `WORKFLOWS_ARTIFACT_MAX_DOWNLOAD_BYTES`, `WORKFLOWS_ARTIFACT_ALLOWED_MIME`, `WORKFLOWS_ARTIFACT_BULK_MAX_BYTES`
- Webhooks: `WORKFLOWS_WEBHOOK_SECRET`, `WORKFLOWS_WEBHOOK_TIMEOUT`
- Completion hooks: `WORKFLOWS_DISABLE_COMPLETION_WEBHOOKS=true` globally disables completion webhooks.
- Webhook egress policy: Per-tenant allow/deny lists are enforced for completion webhooks and webhook steps.
  - Global: `WORKFLOWS_WEBHOOK_ALLOWLIST`, `WORKFLOWS_WEBHOOK_DENYLIST`
  - Tenant overrides: `WORKFLOWS_WEBHOOK_ALLOWLIST_<TENANT>`, `WORKFLOWS_WEBHOOK_DENYLIST_<TENANT>` (tenant upper-cased, `-` → `_`)
  - Private IP blocking follows `WORKFLOWS_EGRESS_BLOCK_PRIVATE` (defaults true).
- General egress policy with per-tenant overrides:
  - Global: `WORKFLOWS_EGRESS_ALLOWLIST`, `WORKFLOWS_EGRESS_DENYLIST`
  - Tenant overrides: `WORKFLOWS_EGRESS_ALLOWLIST_<TENANT>`, `WORKFLOWS_EGRESS_DENYLIST_<TENANT>` (tenant upper-cased, `-` → `_`)
  - Profile: `WORKFLOWS_EGRESS_PROFILE=strict|permissive|custom` (default strict in prod)
  - Precedence: Any deny (global or tenant) blocks; allowlists are unioned (host allowed if present in either global or tenant allowlist). When allowlists are empty, permissive profile allows public hosts; strict requires an allowlist match.
- DB URI (SQLite): `DATABASE_URL_WORKFLOWS=sqlite:///path/to/workflows.db`
- Webhook global disable: `WORKFLOWS_DISABLE_COMPLETION_WEBHOOKS=true` disables completion hooks globally
- Artifact scope validation: `WORKFLOWS_ARTIFACT_VALIDATE_STRICT=true|false` (default true). When false, validation failures log a warning but do not block download.
- Artifact validation modes (per-run override):
  - Each run can opt into `validation_mode = 'non-block'` to allow artifact downloads even when scope validation fails. This is a per-run override intended for trusted/internal workflows.
  - Resolution order: per-run setting is evaluated first; if not set (or not `non-block`), the server falls back to `WORKFLOWS_ARTIFACT_VALIDATE_STRICT` (default true).
  - Example behavior:
    - `WORKFLOWS_ARTIFACT_VALIDATE_STRICT=false` → downloads proceed with a warning when path scope validation fails.
    - `WORKFLOWS_ARTIFACT_VALIDATE_STRICT=true` and run’s `validation_mode='non-block'` → downloads proceed for that run despite strict env setting.
  - Scope check uses `commonpath` between the resolved artifact path and recorded `workdir` (when available). Only `file://` artifacts are downloadable via the API.
- Artifact integrity: When `checksum_sha256` is recorded for an artifact, downloads verify the checksum. On mismatch, responses use `409 Conflict` in strict mode; non-strict modes warn and continue.
- Artifact manifest: `GET /runs/{run_id}/artifacts/manifest?verify=true` computes checksums and returns `integrity_summary`.
- Artifact retention and GC:
  - Worker: `WORKFLOWS_ARTIFACT_GC_ENABLED=true` to enable.
  - Settings: `WORKFLOWS_ARTIFACT_RETENTION_DAYS` (default 30), `WORKFLOWS_ARTIFACT_GC_INTERVAL_SEC` (default 3600).
  - Behavior: Deletes DB rows for artifacts older than retention. For `file://` URIs, also deletes files from disk.
- SQLite connection pool: `WORKFLOWS_SQLITE_POOL_SIZE` (default 0 disables) enables a lightweight pool for hot paths (events).

### Security & Governance

- Egress policy (centralized):
  - Profile: `WORKFLOWS_EGRESS_PROFILE=strict|permissive|custom` (default `strict` in `prod`, `permissive` otherwise based on `ENVIRONMENT`/`APP_ENV`).
  - Allowed schemes: `http, https` (fixed).
  - Allowed ports: `WORKFLOWS_EGRESS_ALLOWED_PORTS` (comma-separated; default `80,443,8080`).
  - Allowlist (host/domain): `WORKFLOWS_EGRESS_ALLOWLIST` (comma-separated; supports subdomains via `example.com`).
  - Block private/reserved IPs: `WORKFLOWS_EGRESS_BLOCK_PRIVATE=true|false` (default true). DNS is resolved and all target IPs must be public.
  - Webhook-specific allow/deny (global and per-tenant) remain available as documented above; they use the centralized evaluator.

- Audit logging (Unified Audit Service):
  - Workflows endpoints log key events: admin owner overrides, permission denials (tenant mismatch, not owner), and run creation (saved/adhoc). Logs include a request ID when provided (`X-Request-ID`), IP, user agent, endpoint and method where available.
  - Query audit logs via `GET /api/v1/admin/audit-log`.

- PII/Secrets:
  - Log redaction for `log` step: `WORKFLOWS_REDACT_LOGS=true|false` (default true) applies PII redaction to messages before emitting to logs.
  - Field-level encryption for artifact metadata: enable with `WORKFLOWS_ARTIFACT_ENCRYPTION=true` and provide `WORKFLOWS_ARTIFACT_ENC_KEY` (strict base64-encoded 32-byte AES-256 key). Encrypted metadata is transparently decrypted on read when the key is present; otherwise a placeholder is returned.
  - Scoped secrets injection: run requests accept `secrets` (map of strings). These are injected into the execution context as `context.secrets` and never persisted. Secrets are cleared from memory when the run reaches a terminal state.

### Webhooks: Delivery History, Replay, and Replay Protection

- Engine attaches headers on delivery:
  - `X-Webhook-ID`: unique ID per delivery
  - `X-Signature-Timestamp`: unix seconds
  - `X-Workflows-Signature`: HMAC-SHA256 of `"<timestamp>.<body>"` using `WORKFLOWS_WEBHOOK_SECRET`
  - Compatibility header: `X-Hub-Signature-256: sha256=<sig>`
  - Receivers can enforce a replay window by rejecting timestamps older than a configured threshold.

- Delivery history:
  - `GET /api/v1/workflows/runs/{run_id}/webhooks/deliveries` returns the sequence of webhook delivery events recorded for a run (status, HTTP code, timestamp).

- Dead-letter queue (DLQ):
  - Failures enqueue into `workflow_webhook_dlq`. The DLQ worker retries with exponential backoff.
  - Admin endpoints:
    - `GET /api/v1/workflows/webhooks/dlq?limit=&offset=`: list entries with payload excerpt.
    - `POST /api/v1/workflows/webhooks/dlq/{id}/replay`: attempt immediate replay (enforces egress policy and signing). In test mode (`TEST_MODE=true` and `WORKFLOWS_TEST_REPLAY_SUCCESS=true`) the replay is simulated.

### Artifacts: Range Requests and Batch Verification

- Single artifact downloads support HTTP Range requests (single range):
  - `206 Partial Content` with `Content-Range` and `Accept-Ranges: bytes`.
  - Full downloads include `Accept-Ranges: bytes` and `Content-Disposition` filename.

- Batch checksum verification:
  - `POST /api/v1/workflows/runs/{run_id}/artifacts/verify-batch` with `{items:[{artifact_id, expected_sha256?}]}` returns calculated hashes and mismatch status. If `expected_sha256` is not provided, the recorded checksum is used when present.

- Quotas and rate limits:
  - Endpoint rate-limits (RG): disabled when `TEST_MODE=true` is set. Stress tests (`TLDW_WORKFLOW_STRESS=1`) re-enable rate limits to validate throttling behavior.
  - Per-user quotas at run start (saved and ad-hoc):
    - Burst ingress control: RG policy `requests` settings (for example `rpm`/`burst`).
    - Daily: `workflows_runs.daily_cap` from the active RG policy.
    - On exceed: returns `429 Too Many Requests` with legacy headers `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`, and RFC headers `RateLimit-Limit`, `RateLimit-Remaining`, `RateLimit-Reset` plus `Retry-After`.
    - Disable (e.g., tests): `WORKFLOWS_DISABLE_QUOTAS=true`.

### Observability

- Metrics (Prometheus-compatible):
  - Counters: `workflows_runs_started{tenant,mode}`, `workflows_runs_completed{tenant}`, `workflows_runs_failed{tenant}`.
  - Histograms: `workflows_run_duration_ms{tenant}`, `workflows_step_duration_ms{tenant,type}`.
  - Step counters: `workflows_steps_started{type}`, `workflows_steps_succeeded{type}`, `workflows_steps_failed{type}`.
  - Webhooks: `workflows_webhook_deliveries_total{status,host}` with status in `delivered|failed|blocked`.
  - Engine gauges: `workflows_engine_queue_depth`.
  - Scrape: `/metrics` (root) or `/api/v1/metrics` (JSON).
- Investigation-first operations:
  - The current metrics are rate/latency oriented; they do not yet expose `reason_code_core` as a Prometheus label.
  - Operators should pivot from an alert into `GET /runs?status=failed` and then `GET /runs/{run_id}/investigation` / `GET /runs/{run_id}/steps/{step_id}/attempts` to classify failures by `reason_code_core`, `category`, `blame_scope`, and `retryable`.
  - Attempt rows preserve the structured taxonomy even when raw event payloads are noisy or incomplete.

- Tracing (OpenTelemetry):
  - Spans: `workflows.run` (per run), nested `workflows.step` (per step), and `workflows.webhook` (completion hook delivery).
  - W3C trace context injected into outbound webhook headers (`traceparent`, optional `baggage`).

- Health & Readiness:
  - Liveness: `GET /healthz` returns `{status, queue_depth, time}`.
  - Readiness: `GET /readyz` returns `{ready, engine, db, time}` with DB connectivity and backend schema version (when using PostgreSQL) checked against the expected version.

In production, when the content backend is configured for PostgreSQL (recommended), the Workflows DB will default to PostgreSQL automatically via the shared backend wiring. SQLite remains the default for development and tests.

### Storage & Migrations

- PostgreSQL schema stores `workflow_events.payload_json` as JSONB with a GIN index (`idx_events_payload_json_gin`) for efficient payload queries.
- Run status indexes: a general `idx_runs_status` plus partial indexes for common statuses `running`, `queued`, `succeeded`, and `failed` speed common filters.
- Per-run event sequence uniqueness is enforced via a unique index on `(run_id, event_seq)`.
- Versioned migrations: a `workflow_schema_version` table tracks the current schema version. Migrations are forward-only and idempotent, and are applied on startup as needed.
- Tests ensure both fresh initialization and upgrades from legacy schemas reach the current version and include the expected indexes (see `tests/Workflows/test_workflows_postgres_indexes.py`).

### DB Maintenance (opt-in)

- A lightweight maintenance worker can periodically checkpoint SQLite WAL files and optionally run VACUUM, or run `VACUUM (ANALYZE)` on PostgreSQL.
- Disabled by default; enable via:
  - `WORKFLOWS_DB_MAINTENANCE_ENABLED=true`
  - Interval: `WORKFLOWS_DB_MAINTENANCE_INTERVAL_SEC` (default 1800)
  - SQLite tunables:
    - `WORKFLOWS_SQLITE_CHECKPOINT=TRUNCATE|RESTART|PASSIVE` (default TRUNCATE)
    - `WORKFLOWS_SQLITE_VACUUM=true|false` (default false)
  - PostgreSQL tunable:
    - `WORKFLOWS_POSTGRES_VACUUM=true|false` (default false)

The worker runs only when enabled and is started/stopped with the app lifecycle.

### Pagination

- Runs listing supports offset and cursor pagination. Cursor is an opaque base64url token that encodes stable seek positions (timestamp and `run_id` tie-breaker). When a `cursor` is provided, `offset` is ignored. Responses include `next_cursor` when there is a subsequent page.
- Events listing supports `cursor` similarly and sets a `Next-Cursor` response header. When a `cursor` is provided, `since` is ignored.

Ordering is stable with a tie-breaker (`run_id` for runs; `event_id` for events) to avoid duplicates or gaps.

## Scheduler

- API: `/api/v1/scheduler/workflows` (create/list/get/update/delete), `/{id}/run-now`, and `/dry-run` to validate cron/timezone and preview next run.
- Cron/timezone validation occurs on create/update with clear 422 feedback.
- Per-job concurrency control:
  - `concurrency_mode`: `skip` (drop overlapping triggers) or `queue` (allow overlaps) maps to APScheduler `max_instances`/`coalesce` behavior.
  - `misfire_grace_sec`: grace window for late triggers.
  - `coalesce`: when true, combine multiple missed runs into a single execution.
- History fields tracked per schedule: `last_run_at`, `next_run_at`, `last_status`.

## Validation Modes

- Global strict validation for artifact download scope:
  - Env: `WORKFLOWS_ARTIFACT_VALIDATE_STRICT=true|false` (default: true)
  - Behavior: when true, file:// downloads must be under the recorded workdir and have an allowed MIME; otherwise 400.
  - When false, scope mismatches are logged as warnings and the download proceeds.
- Per-run override:
  - Run metadata may include `validation_mode: 'block'|'non-block'` (default: `block`).
  - Non-block allows artifact download even when global strict is enabled.

## Control: Cancel / Pause / Resume

- `POST /runs/{run_id}/cancel` sets a cancel flag and attempts to terminate any recorded subprocesses; the running step cooperatively checks `is_cancelled()`.
- `POST /runs/{run_id}/pause` transitions the run to `paused` and emits a `run_paused` event; `resume` returns the run to `running`.
- Steps receive periodic heartbeats (`heartbeat_at`) and may be locked for a short TTL to prevent concurrent work.

## Webhooks: Lifecycle and Delivery

- Completion webhooks can be configured on definitions (`on_completion_webhook: {url, include_outputs}`) and are dispatched on terminal states.
- Global disable: `WORKFLOWS_DISABLE_COMPLETION_WEBHOOKS=true` short-circuits dispatch.
- SSRF/egress policy:
  - Central evaluator enforces allowed schemes/ports, host allowlist, and private IP blocking.
  - Per-tenant allow/deny via `WORKFLOWS_WEBHOOK_ALLOWLIST(_<TENANT>)`, `WORKFLOWS_WEBHOOK_DENYLIST(_<TENANT>)`.
- Signing and tracing:
  - HMAC header when `WORKFLOWS_WEBHOOK_SECRET` is set: `X-Workflows-Signature` and `X-Hub-Signature-256`.
  - W3C `traceparent` is injected when tracing is enabled.
- Delivery outcomes are recorded as `webhook_delivery` run events with `status=delivered|failed|blocked` and masked host.
- Dead-letter queue (DLQ): failures enqueue into `workflow_webhook_dlq` for background retries with exponential backoff and jitter.

## Retention Policy (Artifacts)

- Optional background GC removes file:// artifacts and DB rows older than `WORKFLOWS_ARTIFACT_RETENTION_DAYS` (default 30).
- Enable worker via `WORKFLOWS_ARTIFACT_GC_ENABLED=true`; interval controlled by `WORKFLOWS_ARTIFACT_GC_INTERVAL_SEC`.

## Configuration Reference

- Backend selection
  - Uses shared content backend (see `Config_Files/config.txt` → `[Database] type=sqlite|postgres`); Workflows will default to Postgres when the content backend is Postgres.
  - SQLite DB path (fallback): `Databases/workflows.db` (config key `workflows_path`).

- Rate limits and quotas (disabled when `TEST_MODE=true` is set; stress tests with `TLDW_WORKFLOW_STRESS=1` re-enable rate limits)
  - Endpoint rate limits (RG): disabled when `TEST_MODE=true` is set.
  - Quotas at run start:
    - Burst: RG policy `requests` limits.
    - Daily: `workflows_runs.daily_cap` from policy.
    - Disable with `WORKFLOWS_DISABLE_QUOTAS=true`.

- Engine concurrency
  - `WORKFLOWS_TENANT_CONCURRENCY` (default 2), `WORKFLOWS_WORKFLOW_CONCURRENCY` (default 1).

- Secrets lifecycle
  - In-memory per-run secrets are stored with a TTL and purged automatically on start/resume and when expired.
  - `WORKFLOWS_SECRETS_TTL_SECONDS` (default 3600) controls the TTL window.

- Egress policy
  - Profile: `WORKFLOWS_EGRESS_PROFILE=strict|permissive|custom` (defaults to `strict` in prod, `permissive` elsewhere).
  - Allowed ports: `WORKFLOWS_EGRESS_ALLOWED_PORTS` (default `80,443,8080`).
  - Host allowlist: `WORKFLOWS_EGRESS_ALLOWLIST` (comma-separated, supports subdomains).
  - Private IP blocking: `WORKFLOWS_EGRESS_BLOCK_PRIVATE=true|false` (default true).
  - Webhook allow/deny: `WORKFLOWS_WEBHOOK_ALLOWLIST(_<TENANT>)`, `WORKFLOWS_WEBHOOK_DENYLIST(_<TENANT>)`.

- Webhooks
  - Global disable: `WORKFLOWS_DISABLE_COMPLETION_WEBHOOKS=true|false` (default false).
  - Signing secret: `WORKFLOWS_WEBHOOK_SECRET`.
  - DLQ worker: `WORKFLOWS_WEBHOOK_DLQ_ENABLED`, `WORKFLOWS_WEBHOOK_DLQ_INTERVAL_SEC`, `WORKFLOWS_WEBHOOK_DLQ_BATCH`, `WORKFLOWS_WEBHOOK_DLQ_TIMEOUT_SEC`, `WORKFLOWS_WEBHOOK_DLQ_BASE_SEC`, `WORKFLOWS_WEBHOOK_DLQ_MAX_BACKOFF_SEC`, `WORKFLOWS_WEBHOOK_DLQ_MAX_ATTEMPTS`.

- Artifacts
  - Validation strictness: `WORKFLOWS_ARTIFACT_VALIDATE_STRICT=true|false` (default true).
  - Metadata encryption: `WORKFLOWS_ARTIFACT_ENCRYPTION=true|false` (+ `WORKFLOWS_ARTIFACT_ENC_KEY`).
  - Retention worker: `WORKFLOWS_ARTIFACT_GC_ENABLED`, `WORKFLOWS_ARTIFACT_RETENTION_DAYS`, `WORKFLOWS_ARTIFACT_GC_INTERVAL_SEC`.

## WebUI

- Next.js UI for definition CRUD, run start (sync/async), run status, event stream, and artifact downloads (server and client-side zip).
- The Runs list displays status chips including a dedicated `waiting_approval` chip for human-in-the-loop runs awaiting approval.
- The Events viewer supports cursor pagination when enabled, and client/server-side filtering by event types.
- Builder UX (MVP)
  - Step inspector panels must expose `On success go to…` / `On failure go to…` selectors (populated from known step ids) so routing is discoverable without editing raw JSON.
  - Palette should include a “Branch (if/else)” tile that preloads the canonical config (`condition`, `true_next`, `false_next`) and pairs with an inline helper explaining when to use branch vs. success/failure wiring.
  - Adapter advanced settings should surface a read-only hint about runtime `__next__` overrides so power users know custom steps can redirect execution.

## Testing

- Unit/integration tests for DB and engine across SQLite and PostgreSQL backends: `tldw_Server_API/tests/Workflows/*`
- Stress tests (disabled by default): enable with `TLDW_WORKFLOW_STRESS=1`.
- E2E tests exercise workflows in user flows where relevant.

## Known Gaps (to address)

- Artifact path scope validation uses `commonpath` with a strict/relaxed toggle; consider moving to `Path.is_relative_to` when minimum Python supports it.
- Per-run event counters reduce races; assess using DB-side sequences if needed under extreme concurrency.
- SQLite still uses a lightweight pool; evaluate moving workflows to Postgres by default in production.
- DAG validation currently checks explicit routing (`on_success`, `on_failure`, and branch targets) for cycles with helpful diagnostics; richer implicit chaining rules may require extended checks in future.
- Per-run event sequences are maintained via a counters table; existing runs fall back to `MAX+1` behavior on older schemas.

## Roadmap (v0.2+)

- Branching/Conditionals/Parallelism: Expand minimal `branch/map` to full JSONLogic-like conditions and parallel fan-out with deterministic fan-in/merge semantics.
- Step Library Expansion: STT/TTS adapters, data transformation sandbox, export steps.
- Schedules/Triggers: Time-based triggers and inbound webhook triggers.
- Budgets/Quotas: Per-tenant/user budgets with enforcement and reporting.
- GUI Builder: Drag-and-drop graph editor in the WebUI backed by the APIs above, with success/failure routing selectors and a preconfigured branch template in the step palette.
- Rich trigger catalog: Webhooks, cron, polling, IMAP, queues, and event stream triggers plus sub-workflows, reusable workflow modules, and error hooks for structured retry/compensation flows.
  - Implement a trigger registry with shared schema and validation so new trigger types can be registered without engine changes.
  - Ship inbound webhook service with signature verification, replay protection, and trigger-level rate limiting.
  - Extend the scheduler to support long-polling connectors (IMAP, queues, event streams) using incremental checkpoint state.
  - Add sub-workflow invocation step with typed input/output contracts and version pinning.
  - Introduce error hooks that bind to failure events and dispatch compensating workflows or alerts.
- Advanced flow control: Enhanced branching, looping, merges, wait states, inline Python function nodes, and first-class data transformation helpers.
  - Extend definition schema and engine to support loop constructs (while, foreach) with deterministic exit conditions.
  - Add merge/join semantics so parallel branches can synchronize before continuing.
  - Provide wait-state primitives that emit human/system tasks with expiring tokens for resume.
  - Deliver sandboxed Python function nodes with resource limits and audit logging.
  - Bundle reusable data transformation helpers (templating, JSON logic, vector math) exposed via step configs.
- Execution management: Configurable retry policies, manual reruns, comprehensive execution logs, definition version history, and environment-scoped credential stores.
  - Allow per-step and per-workflow retry policies (max attempts, backoff curves, retryable errors) managed by the engine.
  - Add run replay endpoints and UI controls for manual reruns and partial restarts from step checkpoints.
  - Persist structured execution logs with search/filter APIs and retention policies.
  - Track definition versions with diff, promotion, and rollback workflows.
  - Build environment-scoped credential store with rotation APIs and usage auditing.
- Horizontal scaling & collaboration: Queue/worker modes, worker sharding, multi-tenant RBAC improvements, and hosted/cloud deployment options with shared workspace collaboration features.
  - Introduce a queue-backed execution mode with distributed workers and lease-based run ownership.
  - Support worker sharding by tenant, workflow tags, or resource class to isolate workloads.
  - Expand RBAC to cover shared spaces, workflow-level permissions, and audit trails for edits.
  - Provide IaC templates and observability packs for hosted/cloud deployment footprints.
  - Add collaboration features (shared drafts, reviewer workflows, comments) to the builder and API.

## Quick Start

1. Create a definition:
   `POST /api/v1/workflows` with a minimal body (see schema above)
2. Run it: `POST /api/v1/workflows/{id}/run?mode=async|sync`
3. Watch progress: `GET /api/v1/workflows/runs/{run_id}/events` or `WS /api/v1/workflows/ws?run_id=...`
4. Download outputs: `GET /api/v1/workflows/runs/{run_id}/artifacts` and `.../download`

For deeper goals and rationale, see `Workflows-PRD-1.md`.

### Example: Branch → Map → Log

A small pipeline that branches based on an input flag. If `inputs.enabled` is true, it maps over an `items` array and logs each item; otherwise it logs an error and exits.

```
{
  "name": "branch-map-log",
  "version": 1,
  "inputs": {"enabled": true, "items": ["alpha", "beta", "gamma"]},
  "steps": [
    {"id": "s_branch", "type": "branch", "config": {
      "condition": "{{ inputs.enabled }}",
      "true_next": "s_map",
      "false_next": "s_err"
    }},

    {"id": "s_map", "type": "map", "config": {
      "items": "{{ inputs.items }}",
      "concurrency": 2,
      "step": {"type": "log", "config": {"message": "Item {{ item }}", "level": "info"}}
    }, "on_success": "s_done", "on_failure": "s_err"},

    {"id": "s_err", "type": "log", "config": {"message": "Disabled or an error occurred", "level": "warning"}},
    {"id": "s_done", "type": "log", "config": {"message": "All done", "level": "info"}}
  ]
}
```

Notes:
- `branch` sets `__next__` internally when a target is provided; engine jumps by deterministic IDs.
- `map` runs the nested step for each element with limited concurrency and returns `{ results, count }` in `last` for downstream steps.
- `on_failure` on `s_map` routes to `s_err` if the map step cannot complete.
