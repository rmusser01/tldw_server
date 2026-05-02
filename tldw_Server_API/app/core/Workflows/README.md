# Workflows

Define, run, and monitor multi‑step workflows. Includes definition/versioning, ad‑hoc and saved runs, events/artifacts, minimal engine with step adapters, and integration with the core Scheduler + recurring scheduler.

## 1. Descriptive of Current Feature Set

- Definitions & versions
  - Create/list/delete workflow definitions; create new immutable versions under a definition.
- Runs
  - Start runs from a saved definition (`run_mode` async/sync) or ad‑hoc payload; inject per‑run secrets (never persisted) and idempotency keys.
  - Ad‑hoc runs require the same `workflows` scope as saved-definition runs.
  - Events and artifacts persisted; step run entries tracked with timestamps and status.
- Step types (initial set)
  - `media_ingest`, `prompt` (templating), `llm`, `rag_search`, `kanban`, `mcp_tool`, `tts`, `webhook`, `delay`, `log`, `wait_for_human`, `wait_for_approval`, `branch`, `map`, `process_media`, `policy_check`, `rss_fetch`, `atom_fetch`, `embed`, `translate`, `stt_transcribe`, `notify`, `diff_change_detector`.
  - Map sub-steps: `map` supports nested types `prompt`, `log`, `delay`, `rag_search`, `media_ingest`, `mcp_tool`, `webhook`, `kanban` only (others are rejected).
- Scheduling
  - Recurring schedules via Workflows Scheduler (cron/APS); presence gating, concurrency mode (skip/queue), coalesce/misfire behavior, jitter.
  - Scheduler bootstrap/rescan scans all tenant rows inside each per-user scheduler DB, rather than assuming `tenant_id="default"`.
  - Manual `run-now` uses the same target resolver as recurring execution, so watchlist-backed schedules route to `watchlist_run` on the `watchlists` queue.
- Governance
  - RBAC checks on run listing and control; optional virtual keys for scheduled runs; audit events on key lifecycle operations.
  - Webhook DLQ replay and artifact GC both append workflow evidence events (`webhook_delivery` and `artifact_gc`).

Related Endpoints (file:line)
- Router (prefix `/api/v1/workflows`): tldw_Server_API/app/api/v1/endpoints/workflows.py:46
  - Create definition: 104
  - List definitions: 140
  - Create new version: 165
  - Delete definition: 201
  - Run saved: search for `@router.post("/{workflow_id}/runs"` in file (near 560+) — returns `WorkflowRunResponse`
  - Run ad‑hoc: `@router.post("/runs")` (near 620+)
  - Get run: `@router.get("/runs/{run_id}")` (near 730+)
  - List runs: `@router.get("/runs")` (near 780+)
  - Stream events: `@router.get("/runs/{run_id}/events")` (SSE/streaming; near 820+)
  - Artifacts download/export: see same file around 880+
- Scheduler (recurring): tldw_Server_API/app/api/v1/endpoints/scheduler_workflows.py:18

Related Schemas
- tldw_Server_API/app/api/v1/schemas/workflows.py:1 (definitions, runs, events)

## 2. Technical Details of Features

- Engine & adapters
  - Minimal engine executes linear flows with pause/resume, cooperative cancel checks, and per‑run secrets cache (TTL in memory).
  - File: tldw_Server_API/app/core/Workflows/engine.py:1; adapters: tldw_Server_API/app/core/Workflows/adapters.py:1
  - Metrics integrated via `core/Metrics` where available.
- State & persistence
  - DB adapter: tldw_Server_API/app/core/DB_Management/Workflows_DB.py:1 (definitions, runs, step_runs, events, artifacts; SQLite and Postgres schemas provided).
  - Event sequence counters and partial indexes included in PG schema.
- Recurring schedules
  - Service: tldw_Server_API/app/services/workflows_scheduler.py:1; DB: tldw_Server_API/app/core/DB_Management/Workflows_Scheduler_DB.py:1
  - Endpoints: `/api/v1/scheduler/workflows` provide CRUD + dry-run + run-now.
  - Presence gating, concurrency mode (skip vs queue), jitter, misfire/coalesce, next‑run persistence handled in service.
- Security & RBAC
  - Endpoint gates: claim-first dependencies (`RequirePermission(...)`) plus token scope (`TokenScopeGuard("workflows", ...)`) on scheduler routes; per‑user scoping for definitions/runs.
  - Optional minting of short‑lived virtual keys for scheduled runs (env‑gated) in `workflows_scheduler`.
- Configuration
  - `WORKFLOWS_SCHEDULER_ENABLED`, `WORKFLOWS_SCHEDULER_TZ`, `WORKFLOWS_SCHEDULER_RESCAN_SEC`, `WORKFLOWS_SCHEDULER_DATABASE_URL`, `WORKFLOWS_SCHEDULER_SQLITE_PATH`.
  - Engine heartbeat and secrets TTL in `EngineConfig`; content backend selected via `DB_Management` helpers.
- Audit & metrics
  - API actions emit audit events; metrics counters and histograms instrument engine paths.

## 3. Developer‑Related/Relevant Information for Contributors

- Folder structure
  - `core/Workflows/engine.py` — run lifecycle and execution primitives.
  - `core/Workflows/adapters.py` — integrations for step types (prompt/llm, RAG, media, webhook, MCP tools).
  - `core/Workflows/metrics.py`, `core/Workflows/registry.py` — metrics wiring and step registry.
  - `api/v1/endpoints/workflows.py` — REST endpoints for definitions/runs/events.
  - `api/v1/endpoints/scheduler_workflows.py` — recurring schedules API.
  - `services/workflows_scheduler.py` — APScheduler bridge to core Scheduler; DB: `Workflows_Scheduler_DB`.
- Patterns & tips
  - Validate definitions with `_validate_definition_payload` (size, steps count, per‑type config schema) and `_validate_dag` (targets/cycles).
  - Prefer ad‑hoc runs for experimentation; persist as definitions once stable; use idempotency keys for retries.
  - For new step types, add schema to `_validate_definition_payload`, implement adapter, register in registry, and add tests.
- Tests
  - Scheduler API/unit: tldw_Server_API/tests/Workflows/test_workflows_scheduler.py:51, 62, 85, 107, 124
  - Health/queue stats: tldw_Server_API/app/api/v1/endpoints/health.py:64, 78 (queries `WorkflowScheduler`)
  - PG advisory metrics (dual‑backend): tldw_Server_API/tests/prompt_studio/integration/test_pg_advisory_lock_stress.py:27
- Pitfalls
  - Use IANA timezones in cron; invalid expressions return 422.
  - Streaming endpoints require proper client handling (SSE); in tests some streaming paths may be skipped.
  - Secrets are in‑memory only; never persisted to DB—expect `None` after engine cleanup.
  - `stt_transcribe` uses the same STT model parsing as the audio REST API (`parse_transcription_model`), so model strings like `parakeet-mlx`, `parakeet-tdt-0.6b-v3-onnx`, `parakeet-onnx`, or `qwen2audio-*` route to the expected providers. When `language` is omitted in the step config, the adapter passes `selected_source_lang=None` to `speech_to_text`, allowing the backend to auto-detect language (consistent with `/api/v1/audio/transcriptions`).
