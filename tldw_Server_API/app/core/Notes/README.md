# Notes Module

Notebook-style knowledge management for creating, searching, tagging, and organizing short-form notes. Backed by the per‑user ChaChaNotes database with FTS5 search, optimistic locking, soft delete, and keyword linking.

## 1. Descriptive of Current Feature Set

- Notes CRUD
  - Create, get, list, patch/update, and soft-delete notes (optimistic locking via `version`).
  - Export selected or filtered notes as JSON or CSV.
- Search & keywords
  - FTS5-based search with pagination; optional inline keyword expansion per note.
  - Keywords CRUD (create, get by id/text, list, search, soft-delete) and link/unlink notes ↔ keywords.
- Per-user scoping
- Each user has an isolated ChaChaNotes DB: `<USER_DB_BASE_DIR>/<user_id>/ChaChaNotes.db` (defaults to `Databases/user_databases` under repo root).
- Governance & safety
  - RBAC permission checks (`rbac_rate_limit`) and token-bucket rate limits per action.
  - Topic monitoring hooks evaluate note text (non-blocking alerts) on create/update/bulk create.
- API surface (mounted under `/api/v1/notes`)
  - Health: `/health`
  - Notes: `POST /`, `GET /{id}`, `GET /`, `PATCH /{id}`, `PUT /{id}`, `DELETE /{id}`, `GET /export`, `POST /export`, `GET /search/`
  - Keywords: `POST /keywords/`, `GET /keywords/{id}`, `GET /keywords/text/{text}`, `GET /keywords/`, `DELETE /keywords/{id}`, `GET /keywords/search/`
  - Linking: `POST /{note_id}/keywords/{keyword_id}`, `DELETE /{note_id}/keywords/{keyword_id}`, `GET /{note_id}/keywords/`, `GET /keywords/{keyword_id}/notes/`

Related Endpoints (file:line)
- tldw_Server_API/app/api/v1/endpoints/notes.py:97 — `GET /health`
- tldw_Server_API/app/api/v1/endpoints/notes.py:168 — `POST /` (create)
- tldw_Server_API/app/api/v1/endpoints/notes.py:242 — `GET /{note_id}` (get)
- tldw_Server_API/app/api/v1/endpoints/notes.py:275 — `GET /` (list)
- tldw_Server_API/app/api/v1/endpoints/notes.py:334 — `PUT /{note_id}` (update)
- tldw_Server_API/app/api/v1/endpoints/notes.py:402 — `PATCH /{note_id}` (partial update)
- tldw_Server_API/app/api/v1/endpoints/notes.py:466 — `DELETE /{note_id}` (soft delete; requires `expected-version` header)
- tldw_Server_API/app/api/v1/endpoints/notes.py:642 — `POST /export` (IDs)
- tldw_Server_API/app/api/v1/endpoints/notes.py:620 — `GET /export` (query filter)
- tldw_Server_API/app/api/v1/endpoints/notes.py:515 — `GET /search/`
- tldw_Server_API/app/api/v1/endpoints/notes.py:808 — `POST /keywords/`
- tldw_Server_API/app/api/v1/endpoints/notes.py:848 — `GET /keywords/{id}`
- tldw_Server_API/app/api/v1/endpoints/notes.py:872 — `GET /keywords/text/{text}`
- tldw_Server_API/app/api/v1/endpoints/notes.py:893 — `GET /keywords/`
- tldw_Server_API/app/api/v1/endpoints/notes.py:923 — `DELETE /keywords/{id}` (soft delete; requires `expected-version`)
- tldw_Server_API/app/api/v1/endpoints/notes.py:965 — `GET /keywords/search/`
- tldw_Server_API/app/api/v1/endpoints/notes.py:995 — `POST /{note_id}/keywords/{keyword_id}` (link)
- tldw_Server_API/app/api/v1/endpoints/notes.py:1039 — `DELETE /{note_id}/keywords/{keyword_id}` (unlink)
- tldw_Server_API/app/api/v1/endpoints/notes.py:1070 — `GET /{note_id}/keywords/`
- tldw_Server_API/app/api/v1/endpoints/notes.py:1095 — `GET /keywords/{keyword_id}/notes/`

Related Schemas
- tldw_Server_API/app/api/v1/schemas/notes_schemas.py:30 — Note and Keyword request/response models (e.g., `NoteCreate`, `NoteUpdate`, `NoteResponse`, `KeywordResponse`, `NotesListResponse`, `NotesExportResponse`).

## 2. Technical Details of Features

- Architecture & data flow
  - API router: `app/api/v1/endpoints/notes.py` (mounted via `main.py` with prefix `/api/v1/notes`, see `tldw_Server_API/app/main.py:2952`).
  - DB access via per-user `CharactersRAGDB` (ChaChaNotes) obtained from `get_chacha_db_for_user`.
  - Service wrapper for tests and utilities: `Notes_InteropService` in `tldw_Server_API/app/core/Notes/Notes_Library.py` provides a stable facade around `CharactersRAGDB`.
- Storage & schema
  - Tables: `notes`, `keywords`, linking table, soft-delete flags, `version` for optimistic locking, FTS5-backed search with triggers.
  - Optimistic locking: update/delete require an `expected-version` header; conflicts return 409 with a helpful message.
  - Sync logging: changes recorded for downstream exports (see `ChaChaNotes_DB.py`).
- Security, RBAC, and rate limits
  - Endpoints gated with `rbac_rate_limit("<scope>")` and a shared `RateLimiter` dependency.
  - Common scopes: `notes.create`, `notes.list`, `notes.update`, `notes.search`, `notes.export`, `notes.bulk_create`, `notes.link_keyword`, `notes.unlink_keyword`, `keywords.create`, `keywords.list`, `keywords.search`, `keywords.delete`.
  - Single-user and multi-user modes supported; per-request `User` injected by `get_request_user`.
- Topic monitoring (non-blocking)
  - On create/update/bulk create, text is passed to `Monitoring.topic_monitoring_service.evaluate_and_alert` to emit alerts to configured sinks.
  - Related: `tldw_Server_API/app/core/Monitoring/topic_monitoring_service.py`.
- Exports
  - JSON or CSV. CSV uses `StreamingResponse` to avoid large in-memory payloads. Optional `include_keywords` column lists comma-separated keywords.
- Configuration
- `USER_DB_BASE_DIR` (from `tldw_Server_API.app.core.config`): per-user DB root directory (`<USER_DB_BASE_DIR>/<user_id>` by default); defaults to `Databases/user_databases/` under the project root. Override via environment variable or `Config_Files/config.txt` as needed.
  - `SERVER_CLIENT_ID` tags DB writes with the service client identity.
- Error handling
  - `handle_db_errors()` maps `InputError`, `ConflictError`, and `CharactersRAGDBError` to appropriate HTTP status codes and messages.
  - 404 on missing resources; 400 on invalid inputs; 429 when rate limits exceeded.

## 3. Developer‑Related/Relevant Information for Contributors

- Folder structure
  - `tldw_Server_API/app/core/Notes/Notes_Library.py` — Notes service facade for unit tests and utilities.
  - `tldw_Server_API/app/api/v1/endpoints/notes.py` — FastAPI router for notes/keywords/linking endpoints.
  - `tldw_Server_API/app/api/v1/schemas/notes_schemas.py` — Pydantic models for requests/responses.
  - `tldw_Server_API/app/core/Character_Chat/chacha_db_provider.py` — Per-user DB provider: path resolution, default character ensure, and the accessors workers use.
    The API layer's ChaCha dependency module wraps it as `get_chacha_db_for_user` and maps its errors to HTTP.
- Patterns & tips
  - Always fetch the per-user DB via `get_chacha_db_for_user` (never instantiate `CharactersRAGDB` directly in endpoints).
  - Use `rbac_rate_limit("scope")` + `RateLimiter` for any new endpoints and maintain consistent scope naming.
  - For updates and deletes, require and honor `expected-version` to preserve optimistic locking semantics.
  - When adding list/search endpoints, keep pagination parameters (`limit`, `offset`) and consider an `include_keywords` toggle for performance.
- Cross‑module interactions
  - RAG retrieval can incorporate Notes via `NotesDBRetriever` (see `tldw_Server_API/app/core/RAG/rag_service/database_retrievers.py:1079`).
  - Chat/Characters and Chatbooks use the same ChaChaNotes DB for transcripts and exports.
- Tests
  - Integration: `tldw_Server_API/tests/Notes_NEW/integration/test_notes_api.py` covers CRUD, linking, search, export, and rate limits.
  - Unit: `tldw_Server_API/tests/Notes/test_notes_library_unit.py` exercises `NotesInteropService` behaviors.
  - Fixtures: integration tests override `get_chacha_db_for_user` and `get_request_user` to inject a temp DB and test user.
- Local dev quick checks (curl)
  - Create: `curl -X POST \
    http://127.0.0.1:8000/api/v1/notes/ \
    -H 'Content-Type: application/json' \
    -d '{"title":"T","content":"C"}'`
  - Update (requires version): `curl -X PATCH http://127.0.0.1:8000/api/v1/notes/<id> \
    -H 'Content-Type: application/json' \
    -H 'expected-version: <version>' \
    -d '{"title":"T2"}'`
  - Link keyword: `curl -X POST http://127.0.0.1:8000/api/v1/notes/<id>/keywords/<kw_id>`
- Pitfalls & gotchas
  - Missing `expected-version` on delete/update returns 400/409; clients must refetch and retry.
  - Large exports: prefer CSV streaming; consider filtering or pagination.
  - Rate limits apply per action; tests may bypass or lower limits depending on `TEST_MODE`.
