**Chatbooks Module**

Note: This README is aligned to the project’s 3-section template. The original content is preserved below under section 3 to avoid any loss of information.

Developer Code Guide: `Docs/Code_Documentation/Guides/Chatbooks_Code_Guide.md:1`

## 1. Descriptive of Current Feature Set

- Purpose: Export, import, preview, and manage user content as portable chatbooks (ZIP + manifest), with multi-user isolation, quotas, and async job processing.
- Capabilities:
  - Sync/async export and import with robust validation and sanitization
  - Signed download URLs (optional), per-user storage roots, job tracking
  - Quotas (storage, daily ops, concurrency, file caps) and health checks
- Inputs/Outputs:
  - Input: JSON requests for export/import/preview; file upload for import
  - Output: Job metadata, manifest preview, and downloadable ZIPs
- Related Endpoints (selected):
  - Router: tldw_Server_API/app/api/v1/endpoints/chatbooks.py:52 (prefix `/api/v1/chatbooks`)
  - GET `/health`: tldw_Server_API/app/api/v1/endpoints/chatbooks.py:67
  - POST `/export`: tldw_Server_API/app/api/v1/endpoints/chatbooks.py:119
  - POST `/import`: tldw_Server_API/app/api/v1/endpoints/chatbooks.py:303
  - POST `/preview`: tldw_Server_API/app/api/v1/endpoints/chatbooks.py:513
  - GET `/export/jobs`: tldw_Server_API/app/api/v1/endpoints/chatbooks.py:686
  - GET `/import/jobs`: tldw_Server_API/app/api/v1/endpoints/chatbooks.py:794
  - GET `/download/{job_id}`: tldw_Server_API/app/api/v1/endpoints/chatbooks.py:896
  - POST `/cleanup`: tldw_Server_API/app/api/v1/endpoints/chatbooks.py:1067
  - DELETE `/export/jobs/{job_id}`: tldw_Server_API/app/api/v1/endpoints/chatbooks.py:1114
  - DELETE `/import/jobs/{job_id}`: tldw_Server_API/app/api/v1/endpoints/chatbooks.py:1156
- Related Schemas:
  - tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py:70 (`CreateChatbookRequest`), :242 (`CreateChatbookResponse`)
  - tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py:108 (`ImportChatbookRequest`), :251 (`ImportChatbookResponse`)
  - tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py:262 (`PreviewChatbookResponse`)
  - tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py:268 (`ListExportJobsResponse`), :274 (`ListImportJobsResponse`)
  - tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py:193 (`ExportJobResponse`), :211 (`ImportJobResponse`)

## 2. Technical Details of Features

- Architecture & Flow:
  - API → Service (`chatbook_service.py`) → Validators/Quota → ZIP/manifest I/O → Core Jobs backend
  - Per-user directories under `USER_DB_BASE_DIR/<user_id>/chatbooks` with strict path sanitization and safe file handling

  Request/Job Flow (ASCII):
  ```text
  Export (sync)
  -----------
  Client
    → POST /api/v1/chatbooks/export (async_mode=false)
      → Validate (ChatbookValidator) + Quotas (QuotaManager)
      → Service collects content → writes manifest + files → creates ZIP in exports/
      → Persist completed ExportJob (download_url + expires_at)
      ← 200 { job_id, download_url }

  Export (async)
  --------------
  Client
    → POST /api/v1/chatbooks/export (async_mode=true)
      → Create ExportJob (pending)
      → Enqueue core Jobs (domain=chatbooks)
      ← 200 { job_id }
      Worker (core)
        → process → write ZIP → update job (output_path, download_url, expires_at, status=completed)

  Import (sync/async)
  -------------------
  Client
    → POST /api/v1/chatbooks/import (multipart file)
      → Save to per-user temp → Validate ZIP → Secure extract → Import selections
      → Sync: return counts/warnings; Async: create ImportJob + process in background
  ```
- Key Components:
  - `chatbook_service.py` (export/import/preview, job state, signed URLs)
  - `chatbook_validators.py` (file/ZIP/manifest validation), `quota_manager.py` (tier limits)
  - `chatbook_models.py` (content types, job models)
- Configuration:
  - Core Jobs only; `CHATBOOKS_JOBS_BACKEND`/`TLDW_JOBS_BACKEND` overrides are ignored for Chatbooks.
  - `CHATBOOKS_CORE_WORKER_ENABLED`: `true|false` controls starting the core worker (default true).
  - `CHATBOOKS_SIGNED_URLS`, `CHATBOOKS_SIGNING_SECRET`, `CHATBOOKS_URL_TTL_SECONDS`, `CHATBOOKS_ENFORCE_EXPIRY`
  - Core jobs tuning: `JOBS_POLL_INTERVAL_SECONDS`, `JOBS_LEASE_SECONDS`, `JOBS_LEASE_RENEW_SECONDS`, `JOBS_LEASE_RENEW_JITTER_SECONDS`
- Concurrency & Performance:
  - BackgroundTasks for async paths; worker-based job execution; quotas prevent abuse
- Error Handling & Security:
  - Path traversal prevention, symlink rejection, per-file size caps, unsafe extension filters
  - Ownership checks on download; avoid logging secrets; structured errors
- Tests (examples):
  - tldw_Server_API/tests/Chatbooks/test_chatbooks_export_sync.py
  - tldw_Server_API/tests/Chatbooks/test_chatbooks_signed_urls.py
  - tldw_Server_API/tests/Chatbooks/test_chatbook_service.py
  - tldw_Server_API/tests/integration/test_chatbook_integration.py

## 3. Developer-Related/Relevant Information for Contributors

- Path: `tldw_Server_API/app/core/Chatbooks`
- Purpose: Backup, export, import, and preview of user content (conversations, notes, characters, world books, dictionaries, media, embeddings, generated docs) as a portable “chatbook” ZIP with a JSON manifest. Supports multi-user isolation, quotas, and async job processing.

**Overview**
- Produces ZIP archives containing a `manifest.json` plus referenced files (media, documents) based on user selections.
- Supports export/import in both sync and async modes; async jobs run via the core Jobs backend. Continuation exports are currently sync-only.
- Enforces strong validation and security: filename/path sanitization, ZIP checks, signed download URLs (optional), per-user storage dirs.

**Key Files**
- `chatbook_service.py`: Main service for export/import/preview and job tracking. Creates per-user storage, writes manifests/archives, manages job rows, optional signed URLs.
- `chatbook_models.py`: Data classes and enums for manifests, content items, relationships, and job records (export/import + statuses).
- `chatbook_validators.py`: Centralized validation and sanitization (filenames, ZIP integrity, traversal checks, ID formats, metadata limits).
- `quota_manager.py`: Per-user quotas (storage, per-day ops, concurrent jobs, file size). DB-backed when available with fallbacks.
- `exceptions.py`: Domain-specific exceptions and helpers.

**Content Types**
- Enum in `chatbook_models.py`: `conversation`, `note`, `character`, `world_book`, `dictionary`, `generated_document`, `media`, `embedding`, `prompt`, `evaluation`.
- API schema mirror in `app/api/v1/schemas/chatbook_schemas.py`.

**Storage Layout**
- Chatbooks storage lives under `USER_DB_BASE_DIR/<user_id>/chatbooks/{exports,imports,temp}`.
- File and directory creation uses 0700 perms where possible. Names are sanitized to avoid traversal or unsafe characters.

**Job Backends**
- Core Jobs backend: default and only supported. A shared worker can process Chatbooks jobs across users.
  - Worker entry: `tldw_Server_API/app/services/core_jobs_worker.py:run_chatbooks_core_jobs_worker`
  - App startup may launch the worker based on flags (see `app/main.py`).

**API Endpoints**
- File: `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`
- Base prefix: `/api/v1/chatbooks`
- `POST /export` → create chatbook (sync or async). Sync returns a completed job reference with `job_id` + `download_url`; async returns `job_id`.
- `POST /export/continue` → continue a truncated export in sync mode. `async_mode=true` is rejected.
- `POST /import` → import chatbook ZIP (sync or async) using direct multipart form fields. Current API support is limited to `skip` and `rename`; media/embedding imports remain disabled.
- `POST /preview` → preview manifest without importing. Invalid archives/manifests return `400`.
- `GET  /export/jobs` and `GET /import/jobs` → list jobs; `GET /export/jobs/{id}`/`GET /import/jobs/{id}` → job status.
- `DELETE /export/jobs/{id}` and `DELETE /import/jobs/{id}` → cancel in-flight jobs.
- `GET  /download/{job_id}` → download completed export; supports optional signed URLs.
- `POST /cleanup` → delete expired exports.
- `GET  /health` → lightweight service health for storage checks.

**Export Flow (Sync)**
- Validate request with `ChatbookValidator` and quota checks (`QuotaManager`).
- Build working dir, gather selected content, write `manifest.json`, include optional media/embeddings.
- Create a ZIP in the user’s `exports/` dir. Persist a completed `ExportJob` row to support download URL and expiry.

**Export Flow (Async)**
- Create `ExportJob` row (status `pending`), enqueue through core Jobs; worker updates job state and writes archive; job receives `download_url` and `expires_at`.
- Download is served from job metadata once `completed`.

**Import Flow**
- Save uploaded ZIP to secure per-user temp; validate ZIP thoroughly.
- If async, create `ImportJob` row and process in background via core Jobs. Track `progress`, `conflicts`, `warnings`.
- Apply conflict resolution: `skip` and `rename` today; broader strategies are planned.

**Security & Validation**
- Filename and path sanitization, symlink rejection, traversal prevention.
- ZIP validation: magic number, integrity, per-file size caps, suspicious compression ratio checks, unsafe extensions rejection, required `manifest.json`.
- Optional signed download URLs: HMAC(SHA256) over `{job_id}:{exp}` with `CHATBOOKS_SIGNING_SECRET`.
- Access control: jobs and files are scoped to the authenticated user. Download validates ownership and path containment.

**Quotas**
- Managed by `quota_manager.py` with tiered limits (free, premium, enterprise). Defaults:
  - Storage: 1GB free; 5GB premium
  - Daily ops: 10 exports/imports free; 50 premium
  - Concurrent jobs: 2 free; 5 premium
  - File size caps enforced per tier

**Database**
- The service initializes job tables in the per-user ChaChaNotes DB (`export_jobs`, `import_jobs`).
- Interacts via `CharactersRAGDB.execute_query(...)`; no raw SQL outside DB abstractions elsewhere in the project.

**Configuration**
- Core Jobs only; any `CHATBOOKS_JOBS_BACKEND`/`TLDW_JOBS_BACKEND` overrides are ignored for Chatbooks.
- `CHATBOOKS_CORE_WORKER_ENABLED`: `true|false` controls starting the core worker (default true).
- `USER_DB_BASE_DIR` (from `tldw_Server_API.app.core.config`): base path for per-user data (chatbooks live under `<USER_DB_BASE_DIR>/<user_id>/chatbooks`); defaults to `Databases/user_databases/` under the project root. Override via environment variable or `Config_Files/config.txt` as needed.
- `CHATBOOKS_URL_TTL_SECONDS`: download URL expiry TTL (default 86400).
- `CHATBOOKS_ENFORCE_EXPIRY`: `true|false` enforce expiry at download.
- `CHATBOOKS_SIGNED_URLS`: `true|false` enable HMAC signing of download URLs (token = HMAC-SHA256 of `"{job_id}:{exp}"`).
- `CHATBOOKS_SIGNING_SECRET`: secret key for HMAC token.
- Core Jobs worker tuning: `JOBS_POLL_INTERVAL_SECONDS`, `JOBS_LEASE_SECONDS`, `JOBS_LEASE_RENEW_SECONDS`, `JOBS_LEASE_RENEW_JITTER_SECONDS`.

**Local Development Tips**
- Start API: `python -m uvicorn tldw_Server_API.app.main:app --reload`
- Health check: `GET /api/v1/chatbooks/health`
- Set `USER_DB_BASE_DIR` (via environment variable or `Config_Files/config.txt`) to direct per-user storage somewhere writable in dev.
- Async exports: ensure core Jobs worker is enabled via app startup, or switch to sync for quick iteration.

**Testing**
- Run tests from repo root: `python -m pytest -v`
- Jobs metrics/health tests live under `tldw_Server_API/tests` and target domain `chatbooks`.
- When adding features, mirror existing test patterns and add:
  - Unit tests for validators and service behavior
  - Integration tests for endpoints (`export`, `import`, `download`, `preview`)
  - Mock external dependencies (for example, Jobs worker or storage IO) when applicable

**Adding or Changing Features**
- New content type:
  - Add enum to `chatbook_models.py` and `app/api/v1/schemas/chatbook_schemas.py`.
  - Implement selection/export/import handling in `chatbook_service.py`.
  - Extend validators and update manifest serialization if needed.
  - Add tests and update Docs references.
**Job backend behavior**
- Keep status transitions consistent: `pending → in_progress → completed|failed|cancelled`.
- Persist URLs/expiries for exports on terminal states.
- Security:
  - All file I/O goes through validators and sanitized paths.
  - Never trust client filenames or paths; never log secrets.

**Error Handling & Logging**
- Use `loguru` for structured logs.
- Raise HTTP 4xx for validation/quota errors; 5xx for unexpected failures (API layer handles mapping).
- Persist error messages to job rows and surface in job status endpoints.

**Cross-Module References**
- Endpoints: `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`
- Schemas: `tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py`
- Jobs core: `tldw_Server_API/app/core/Jobs/`
- DB: `tldw_Server_API/app/core/DB_Management/`

**Contributing**
- Follow project guidelines (PEP8, typing, docstrings). Keep changes minimal and consistent with existing patterns.
- Add or update tests for all behavioral changes.
- Document new env vars and flows here and in `Docs` when appropriate.
