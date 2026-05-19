# Chatbooks Code Guide (Developers)

This guide helps project developers get up to speed on the Chatbooks module: what it contains, how it works end‑to‑end, and how to work with it when building or extending the server.

See also:
- Code README: `tldw_Server_API/app/core/Chatbooks/README.md:1`
- API docs: `Docs/API-related/Chatbook_API_Documentation.md:1`, `Docs/API-related/Chatbook_Features_API_Documentation.md:1`

Navigation:
- Back to module overview: `tldw_Server_API/app/core/Chatbooks/README.md:1`
- Back to API reference: `Docs/API-related/Chatbook_API_Documentation.md:1`

**Scope & Goals**
- Export a user-selected subset of content into a portable “chatbook” ZIP with a JSON manifest.
- Import chatbooks back into a user’s workspace with conflict strategies.
- Preview chatbooks safely without importing.
- Support async jobs via the core Jobs backend; enforce quotas and security.

**Quick Map (Where Things Live)**
- Core service and models
  - `tldw_Server_API/app/core/Chatbooks/chatbook_service.py:1` — orchestrates export/import/preview, per‑user storage, job rows, signed URLs.
  - `tldw_Server_API/app/core/Chatbooks/chatbook_models.py:1` — dataclasses/enums: manifest, content items, relationships, ExportJob/ImportJob + statuses.
  - `tldw_Server_API/app/core/Chatbooks/chatbook_validators.py:1` — centralized filename/archive validation and sanitization.
  - `tldw_Server_API/app/core/Chatbooks/quota_manager.py:1` — per-user quotas for storage, file size, daily ops, concurrent jobs.
  - `tldw_Server_API/app/core/Chatbooks/exceptions.py:1` — domain exceptions.
- API surface
  - Router: `tldw_Server_API/app/api/v1/endpoints/chatbooks.py:1`
  - Schemas: `tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py:1`
- Jobs (core backend)
  - Worker: `tldw_Server_API/app/services/core_jobs_worker.py:1`
- Databases used
  - Primary DB: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:1`
  - Optional DBs: Prompts, Media v2, Evaluations (if installed) — resolved via `tldw_Server_API/app/core/DB_Management/db_path_utils.py:1`
- Tests (good references)
  - Unit/integration: `tldw_Server_API/tests/Chatbooks/`, `tldw_Server_API/tests/integration/test_chatbook_integration.py:1`, `tldw_Server_API/tests/Health/test_feature_health_endpoints.py:1`

**Key Endpoints**
- Base: `/api/v1/chatbooks`
- Export: `POST /export` — sync returns download URL; async returns job id.
- Import: `POST /import` — sync returns results; async returns job id.
- Preview: `POST /preview` — parse and validate `manifest.json` without importing.
- Jobs: `GET /export/jobs`, `GET /export/jobs/{job_id}`, `GET /import/jobs`, `GET /import/jobs/{job_id}`
- List jobs support filters/pagination: `status`, `limit`, `offset`, `order_by`, `order_desc`.
- Cancel: `DELETE /export/jobs/{job_id}`, `DELETE /import/jobs/{job_id}`
- Download: `GET /download/{job_id}` — optional signed URLs, ownership enforced (when enabled, token = HMAC-SHA256 of `"{job_id}:{exp}"` using `CHATBOOKS_SIGNING_SECRET`).
- Cleanup: `POST /cleanup` — remove expired export files.
- Health: `GET /health` — storage base existence/writability.

**Architecture & Data Flow**
- Per-user isolation
  - `ChatbookService(user_id, CharactersRAGDB, user_id_int)` creates a secure per‑user storage root under `USER_DB_BASE_DIR/<user_id>/chatbooks/{exports,imports,temp}`.
  - Base path selection: `USER_DB_BASE_DIR` (from `tldw_Server_API.app.core.config`) defaults to `Databases/user_databases/` under the project root. Override via environment variable or `Config_Files/config.txt` as needed.
- Export (sync)
  - Validate metadata and quotas → collect selected content → write `manifest.json` + content tree → zip to `exports/` → persist completed ExportJob with `download_url` + `expires_at`.
- Export (async)
  - Create ExportJob `pending` → enqueue via core Jobs; worker completes and fills `output_path`, `download_url`, `expires_at`.
- Import (sync)
  - Save uploaded ZIP to per‑user temp → validate archive → extract securely → import selected content with conflict strategy → return warnings and counts.
- Import (async)
  - Create ImportJob `pending` → enqueue/run in background → track `progress`, `conflicts`, `warnings`.
- Preview
  - Save uploaded ZIP to per‑user temp → validate → open `manifest.json` → return structured summary; delete temp file.

- Jobs storage separation
  - Chatbooks keeps user‑scoped job rows (`export_jobs`, `import_jobs`) inside the user’s ChaChaNotes DB to drive the Chatbooks UI/API.
  - The core Jobs backend is a separate, shared job queue (domain `chatbooks`) used to process async work across users.

**Jobs Backend**
- Core Jobs (only)
  - Enqueue via `tldw_Server_API.app.core.Jobs.manager.JobManager:1` with `domain="chatbooks"`, `queue="default"`.
  - Worker: `run_chatbooks_core_jobs_worker(...)` picks jobs across users and updates Chatbooks job rows.

  Cancellation semantics: the core worker checks for cancellation before starting a job (pre-flight) and applies best-effort in-flight cancellation via job lease/cancel flags.

**Data Model (Selected)**
- Enums: `ContentType`, `ExportStatus`, `ImportStatus`, `ConflictResolution`, `ChatbookVersion`.
- Manifest: `ChatbookManifest` includes metadata, configuration, statistics, and `content_items` of type `ContentItem`.
- Jobs:
  - `ExportJob(job_id, user_id, status, chatbook_name, output_path, ... , download_url, expires_at)`
  - `ImportJob(job_id, user_id, status, chatbook_path, ... , conflicts, warnings)`
- Tables (created in ChaChaNotes DB by service):
  - `export_jobs(job_id PK, user_id, status, chatbook_name, output_path, created_at, started_at, completed_at, error_message, progress_percentage, total_items, processed_items, file_size_bytes, download_url, expires_at)`
  - `import_jobs(job_id PK, user_id, status, chatbook_path, created_at, started_at, completed_at, error_message, progress_percentage, total_items, processed_items, successful_items, failed_items, skipped_items, conflicts JSON, warnings JSON)`

**Validation & Security**
- Validators: `ChatbookValidator` centralizes input checks for filenames, job ids, and ZIP archives (magic, integrity, traversal, per‑file size caps, unsafe extensions, required `manifest.json`).
- Download hardening:
  - Job ownership enforced; `output_path` must reside under the user’s export dir; optional HMAC‑signed URLs.
  - Expiry enforced when `CHATBOOKS_ENFORCE_EXPIRY=true`.
- Upload hardening:
  - Per‑user temp directory under `USER_DB_BASE_DIR/<user_id>/chatbooks/temp`; sanitizes `user.id` and filenames; rejects symlinks and traversal. Preview/import delete temp files on completion.
- Rate limits: RG ingress policies + per-user quotas. Defaults live in RG policy config (export/import 5/min, preview 10/min, download 20/min).

**Quotas**
- Managed by `QuotaManager` with tiered limits (free/premium/enterprise): storage (MB), daily exports/imports, max concurrent jobs, file size caps, chatbook count.
- Quota checks occur in the endpoints before dispatching to the service.

**Working With Chatbooks in Code**
- Get the service in endpoints via DI: `get_chatbook_service()` → `ChatbookService` with user‑scoped `CharactersRAGDB`.
- Export (sync)
  - Call `await ChatbookService.create_chatbook(..., async_mode=False)` with `content_selections: Dict[ContentType, List[str]]`.
  - Returns `(success, message, archive_path)`; the endpoint persists a completed `ExportJob` and returns `job_id` plus `download_url`.
- Export (async)
  - Call with `async_mode=True`; returns `(success, message, job_id)`. A core Jobs worker completes the job.
- Export continuation
  - `POST /api/v1/chatbooks/export/continue` is sync-only. Successful continuations are persisted as completed export jobs with job-backed download URLs.
- Import (sync/async)
  - `await ChatbookService.import_chatbook(file_path, content_selections?, conflict_resolution?, prefix_imported?, import_media?, import_embeddings?, async_mode?)`.
  - Current API-supported conflict strategies: `skip` and `rename`. `overwrite` and `merge` remain planned.
- Preview
  - `service.preview_chatbook(file_path)` parses and validates the manifest without DB writes. Invalid archives/manifests are surfaced as HTTP `400` by the API layer.
- Jobs & status
  - `service.list_export_jobs()`, `service.get_export_job(job_id)`, `service.cancel_export_job(job_id)`; analogous import methods.

**Storage & Paths**
- Base directory selection: `USER_DB_BASE_DIR` (from `tldw_Server_API.app.core.config`) defaults to `Databases/user_databases/` under the project root. Override via environment variable or `Config_Files/config.txt` as needed.
- Per user: `USER_DB_BASE_DIR/<user_id>/chatbooks/{exports,imports,temp}` with `0700` perms when possible.
- Archives: ZIP packages with `manifest.json` and content folders. Exports use `.zip` by default; imports accept `.zip` and `.chatbook`.

**Configuration (Selected)**
- Jobs backend:
  - Core Jobs only; any `CHATBOOKS_JOBS_BACKEND`/`TLDW_JOBS_BACKEND` overrides are ignored for Chatbooks.
  - `CHATBOOKS_CORE_WORKER_ENABLED`: `true|false` controls starting the core worker (default true).
- Downloads:
  - `CHATBOOKS_SIGNED_URLS=true` and `CHATBOOKS_SIGNING_SECRET=<secret>` to require `token` + `exp` on `/download/{job_id}`.
  - `CHATBOOKS_URL_TTL_SECONDS` (default 86400) controls `expires_at` and URL token expiry.
  - `CHATBOOKS_ENFORCE_EXPIRY=true` to enforce link expiration.
- Core Jobs worker tuning: `JOBS_POLL_INTERVAL_SECONDS`, `JOBS_LEASE_SECONDS`, `JOBS_LEASE_RENEW_SECONDS`, `JOBS_LEASE_RENEW_JITTER_SECONDS`.

**Audit Integration**
- Endpoints emit unified audit events for key actions:
  - Export start/completion (sync), Import start/completion (sync), Download, and Cleanup.
- Service code avoids logging secrets; download enforces ownership and logs security violations for traversal attempts via the unified audit service.
- Metrics: increments warning counters (e.g., `app_warning_events_total`) in endpoints and worker for observability; queue metrics are exposed via the core Jobs module.

**API Usage Examples (curl)**
- Setup
  - `API="http://127.0.0.1:8000/api/v1"`
  - `KEY="<API_KEY_OR_BEARER>"`  (use `X-API-KEY` for single-user; `Authorization: Bearer` for JWT)
- Export (sync)
  - `curl -sS -X POST "$API/chatbooks/export" -H 'Content-Type: application/json' -H "X-API-KEY: $KEY" -d '{"name":"My Chatbook","description":"Demo","content_selections":{"conversation":["<conv_id>"]},"async_mode":false}'`
- List export jobs
  - `curl -sS "$API/chatbooks/export/jobs" -H "X-API-KEY: $KEY"`
  - With filters/pagination: `curl -sS "$API/chatbooks/export/jobs?status=completed&limit=20&offset=0&order_by=created_at&order_desc=true" -H "X-API-KEY: $KEY"`
- Download
  - `curl -OJ "$API/chatbooks/download/<job_id>" -H "X-API-KEY: $KEY"`
- Remove export job (completed/cancelled)
  - Export remove supports completed, cancelled, failed, and expired jobs.
  - `curl -sS -X DELETE "$API/chatbooks/export/jobs/<job_id>/remove" -H "X-API-KEY: $KEY"`
- Import (sync)
  - `curl -sS -X POST "$API/chatbooks/import" -H "X-API-KEY: $KEY" -F "file=@/path/to/file.chatbook" -F 'conflict_resolution=skip' -F 'async_mode=false'`
  - Note: Import options are sent as multipart form fields alongside the uploaded file.
- Remove import job (completed/cancelled)
  - Import remove supports completed, cancelled, and failed jobs.
  - `curl -sS -X DELETE "$API/chatbooks/import/jobs/<job_id>/remove" -H "X-API-KEY: $KEY"`
- Preview
  - `curl -sS -X POST "$API/chatbooks/preview" -H "X-API-KEY: $KEY" -F "file=@/path/to/file.chatbook"`

**Common Gotchas & Tips**
- Always use `ContentType` enums for `content_selections`; endpoints coerce schema enums/strings to core enums.
- When adding new content types, update: `ContentType` enums, export collectors, import handlers, schemas, and tests.
- Don’t expose internal `output_path` in APIs; use `download_url` built from job id.
- For tests, set `USER_DB_BASE_DIR` (env or `Config_Files/config.txt`) to isolate chatbooks storage under a temp or test-specific directory.
- Embeddings export currently derives from media rows when `include_embeddings=true`. Explicit embedding exports beyond media vectors are pending in service TODOs.
- Prompts/Evaluations/Media DBs are optional in some builds; the service will skip those sections if their DBs aren’t available.

**Where To Extend**
- Add new content collectors/importers in `chatbook_service.py` alongside existing `_collect_*` helpers and manifest population.
- Augment validation in `chatbook_validators.py` and quotas in `quota_manager.py`.
- Expose new controls in `chatbook_schemas.py` and surface via `endpoints/chatbooks.py`.
- If adding new job workflows, ensure `core_jobs_worker.py` maps payloads and status transitions appropriately.

**Testing Pointers**
- Security: `tldw_Server_API/tests/Chatbooks/test_chatbook_security.py:1` (filenames, traversal, extensions)
- Signed URLs: `tldw_Server_API/tests/Chatbooks/test_chatbooks_signed_urls.py:1`
- Sync export path + download: `tldw_Server_API/tests/Chatbooks/test_chatbooks_export_sync.py:1`
- Service behaviors: `tldw_Server_API/tests/Chatbooks/test_chatbook_service.py:1`
- End‑to‑end quotas/limits: `tldw_Server_API/tests/e2e/test_background_jobs_and_rate_limits.py:1`
