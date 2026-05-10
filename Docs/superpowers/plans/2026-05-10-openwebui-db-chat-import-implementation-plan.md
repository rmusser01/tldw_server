# OpenWebUI webui.db Chat Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add uploaded OpenWebUI `webui.db` chat import through the existing Chatbooks preview/import workflow.

**Architecture:** Add a focused SQLite adapter that normalizes selected OpenWebUI DB rows into the existing OpenWebUI conversation/message plan model. Extend Chatbooks schemas, endpoints, service dispatch, and Jobs handling with `source_format=openwebui_db`, while keeping upload storage, quota, cleanup, duplicate handling, and import result patterns aligned with `openwebui_json`. Folder mirroring is implemented as a small service helper over existing keyword collection and keyword link APIs, not as a new folder schema.

**Tech Stack:** FastAPI, Pydantic, SQLite via Python `sqlite3`, existing ChaCha DB keyword collection APIs, pytest, React/Ant Design, Vitest, Playwright only if manual browser verification is needed.

---

## Source Inputs

- Design spec: `Docs/superpowers/specs/2026-05-10-openwebui-db-chat-import-design.md`
- OpenWebUI schema reference: `https://docs.openwebui.com/reference/database-schema/`
- Existing JSON adapter: `tldw_Server_API/app/core/Chatbooks/import_adapters/openwebui.py`
- Existing Chatbooks service: `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
- Existing endpoints: `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`
- Existing schemas: `tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py`
- Existing frontend import UI: `apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx`
- Existing frontend client: `apps/packages/ui/src/services/tldw/TldwApiClient.ts` and `apps/packages/ui/src/services/tldw/domains/chat-rag.ts`

## Constraints

- Accept uploaded `.db` and `.sqlite` files only.
- Do not support server-local paths, live OpenWebUI connections, admin exports, or binary attachment hydration.
- Preview lists detected OpenWebUI users; import requires an explicit `selected_openwebui_user_id`.
- Import only chats where `chat.user_id` matches the selected OpenWebUI user.
- Preserve OpenWebUI source user, folder, project-like, attachment, file, image, and artifact references as metadata and warnings.
- Keep duplicate detection compatible with JSON imports by using `source="openwebui"` and the same OpenWebUI chat id when present.
- Do not log raw chat/message content, full uploaded paths, source emails outside authenticated preview responses, or attachment storage paths.
- Run Python commands from the repo root after `source .venv/bin/activate`.

## File Structure

Create:

- `tldw_Server_API/app/core/Chatbooks/import_adapters/openwebui_db.py`
  - Read-only SQLite validation and extraction.
  - DB preview dataclasses and `to_dict()` helpers.
  - Selected-user extraction into `OpenWebUIConversationPlan`.
- `tldw_Server_API/app/core/Chatbooks/openwebui_folders.py`
  - Folder namespace sanitization and mirroring helper over existing ChaCha keyword collection APIs.
  - No endpoint or adapter concerns.
- `tldw_Server_API/tests/Chatbooks/test_openwebui_db_import_adapter.py`
  - SQLite adapter unit tests with temporary DB fixtures.
- `tldw_Server_API/tests/Chatbooks/test_openwebui_db_import_service.py`
  - Service import, duplicate, metadata, and folder mirroring tests.
- `tldw_Server_API/tests/Chatbooks/test_chatbooks_openwebui_db_api.py`
  - Endpoint-level preview/import request contract tests.
- `apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx`
  - Frontend OpenWebUI JSON and DB-mode source selector, preview user selection, and import request tests.

Modify:

- `tldw_Server_API/app/core/Chatbooks/import_adapters/openwebui.py`
  - Expose or reuse normalization helpers if needed to avoid duplicating message-tree parsing.
- `tldw_Server_API/app/core/Chatbooks/chatbook_validators.py`
  - Add safe DB filename validation for `.db` and `.sqlite`.
- `tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py`
  - Add `OPENWEBUI_DB`, selected-user request field, DB preview/result schemas.
- `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`
  - Add DB source branches for preview/import and `selected_openwebui_user_id` form field.
- `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
  - Add preview/import service methods and dispatch from `import_chatbook`.
- `tldw_Server_API/app/core/Chatbooks/services/jobs_worker.py`
  - Add async import dispatch for DB source and selected user payload.
- `apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx`
  - Add OpenWebUI database mode and selected-user preview UI.
- `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
  - Include optional `selected_openwebui_user_id` in upload fields.
- `apps/packages/ui/src/services/tldw/domains/chat-rag.ts`
  - Mirror the optional selected user field in the domain client.
- `apps/packages/ui/src/services/__tests__/tldw-api-client.chatbooks-openwebui.test.ts`
  - Add upload field coverage for DB source and selected user.
- User/API docs touched by TASK-233.3:
  - `README.md`
  - `Docs/User_Guides/WebUI_Extension/Chatbook_User_Guide.md`
  - `Docs/API-related/Chatbook_API_Documentation.md`
  - `Docs/API-related/chatbook_openapi.yaml`
  - `Docs/API-related/API_README.md`
  - `Docs/API-related/API_Tags_Index.md`
  - published mirrors under `Docs/Published/...`
- `tldw_Server_API/tests/Docs/test_chatbook_openwebui_import_docs.py`
  - Extend docs regression coverage from JSON-only to JSON plus DB import.

## Stage 1: SQLite Adapter And Preview

**Goal:** Build and test a DB adapter that safely validates an uploaded OpenWebUI SQLite database and returns per-user preview data without importing.

**Success Criteria:** Invalid files are rejected before unsafe reads; required schema is validated; multiple users return isolated counts; preview never exposes raw chat/message content.

**Tests:** `tldw_Server_API/tests/Chatbooks/test_openwebui_db_import_adapter.py`

**Status:** Complete

### Task 1.1: Add failing adapter tests

**Files:**
- Create: `tldw_Server_API/tests/Chatbooks/test_openwebui_db_import_adapter.py`

- [x] Add a fixture helper that creates temporary OpenWebUI-shaped SQLite DBs:

```python
def write_openwebui_db(path: Path, *, users: list[dict], chats: list[dict], folders: list[dict] | None = None) -> Path:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE user (id TEXT PRIMARY KEY, name TEXT, email TEXT, role TEXT, created_at INTEGER, updated_at INTEGER)")
    conn.execute("CREATE TABLE folder (id TEXT PRIMARY KEY, parent_id TEXT, user_id TEXT, name TEXT, items TEXT, meta TEXT, is_expanded INTEGER, created_at INTEGER, updated_at INTEGER)")
    conn.execute("CREATE TABLE chat (id TEXT PRIMARY KEY, user_id TEXT, title TEXT, chat TEXT, created_at INTEGER, updated_at INTEGER, share_id TEXT, archived INTEGER, pinned INTEGER, meta TEXT, folder_id TEXT)")
    ...
```

- [x] Test cases:
  - non-SQLite bytes are rejected with `"Invalid OpenWebUI SQLite database"`.
  - DB missing `user`, `chat`, or required columns is rejected with a schema-focused error.
  - preview lists two users with separate chat, folder, message, branched, archived/pinned, duplicate, and attachment counts.
  - extraction requires a selected user id and extracts only that user's chats.
  - `chat.folder_id` drives folder membership; inconsistent `folder.items` only creates a warning.
  - folder cycle or missing parent yields `Unfiled` warning plan data.

- [x] Run the focused tests and verify they fail for missing module/types:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_openwebui_db_import_adapter.py -q
```

Expected: FAIL because `openwebui_db.py` does not exist yet.

### Task 1.2: Implement the adapter

**Files:**
- Create: `tldw_Server_API/app/core/Chatbooks/import_adapters/openwebui_db.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/import_adapters/openwebui.py`

- [x] Add dataclasses:
  - `OpenWebUIDatabaseUserPreview`
  - `OpenWebUIDatabasePreview`
  - `OpenWebUIDatabaseFolderPlan`
  - `OpenWebUIDatabaseExtractionResult`
- [x] Add constants for required table/column allowlists.
- [x] Add SQLite safety helpers:
  - check first 16 bytes equal `b"SQLite format 3\x00"`.
  - open with `sqlite3.connect(..., uri=True)` using read-only URI mode.
  - call `conn.enable_load_extension(False)` when available.
  - set `conn.row_factory = sqlite3.Row`.
  - use parameterized queries for selected user id.
- [x] Reuse the existing OpenWebUI message-tree normalization shape:
  - Prefer adding a small public helper in `openwebui.py` to convert a wrapper/chat payload into `OpenWebUIConversationPlan`.
  - Keep existing JSON behavior unchanged.
- [x] Preserve DB-specific metadata in `source_metadata`, including `source_kind`, `source_user_id`, row timestamps, `folder_id`, `pinned`, `archived`, `share_id`, `meta`, and project-like metadata if present.
- [x] Attach folder plans to extraction results rather than writing folders from the adapter.
- [x] Return DB preview via `.to_dict()` with users and aggregate warnings.

- [x] Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_openwebui_db_import_adapter.py -q
```

Expected: PASS.

### Task 1.3: Add DB filename validation

**Files:**
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_validators.py`
- Test: `tldw_Server_API/tests/Chatbooks/test_chatbook_security.py`

- [x] Add `VALID_SQLITE_EXTENSIONS = {".db", ".sqlite"}`.
- [x] Add `validate_sqlite_filename()` that delegates to `_validate_filename_for_extensions`.
- [x] Add tests for accepted `.db`/`.sqlite`, rejected double extensions, traversal, and unsupported extensions.

- [x] Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbook_security.py -k "sqlite or filename" -q
```

Expected: PASS.

### Task 1.4: Commit adapter slice

- [x] Review staged diff for raw content/path logging.
- [x] Run `git diff --check`.
- [x] Commit:

```bash
git add tldw_Server_API/app/core/Chatbooks/import_adapters/openwebui.py \
        tldw_Server_API/app/core/Chatbooks/import_adapters/openwebui_db.py \
        tldw_Server_API/app/core/Chatbooks/chatbook_validators.py \
        tldw_Server_API/tests/Chatbooks/test_openwebui_db_import_adapter.py \
        tldw_Server_API/tests/Chatbooks/test_chatbook_security.py
git commit -m "Add OpenWebUI database import adapter"
```

## Stage 2: Chatbooks API And Service Dispatch

**Goal:** Thread `source_format=openwebui_db` and `selected_openwebui_user_id` through Pydantic schemas, preview/import endpoints, and `ChatbookService.import_chatbook`.

**Success Criteria:** Preview accepts DB uploads and returns DB preview data; sync import rejects missing selected user id before service import; unsupported source formats remain rejected.

**Tests:** `tldw_Server_API/tests/Chatbooks/test_chatbooks_openwebui_db_api.py`, existing Chatbooks API tests.

**Status:** Complete

### Task 2.1: Add failing schema/API tests

**Files:**
- Create: `tldw_Server_API/tests/Chatbooks/test_chatbooks_openwebui_db_api.py`
- Modify: `tldw_Server_API/tests/Chatbooks/test_chatbooks_import_validation.py` if existing validation coverage is a better fit.

- [x] Add endpoint tests:
  - `/api/v1/chatbooks/preview` with `source_format=openwebui_db` accepts `.db` and does not call ZIP validation.
  - preview response includes `source_format="openwebui_db"` and `openwebui_db_preview`.
  - `/api/v1/chatbooks/import` with DB source and no `selected_openwebui_user_id` returns 400.
  - import passes selected user id, conflict strategy, prefix flag, and async flag to the service.
  - `.json` still maps only to `openwebui_json`; `.db` is rejected for `openwebui_json`.

- [x] Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_openwebui_db_api.py -q
```

Expected: FAIL because schemas/endpoints do not know `openwebui_db`.

### Task 2.2: Extend schemas and endpoint form contract

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`

- [x] Add enum value `OPENWEBUI_DB = "openwebui_db"`.
- [x] Add `selected_openwebui_user_id: Optional[str]` to `ImportChatbookRequest`.
- [x] Add DB preview models:
  - `OpenWebUIDatabaseUserPreview`
  - `OpenWebUIDatabasePreview`
  - `OpenWebUIDatabaseImportResult`
- [x] Add `openwebui_db_preview` to `PreviewChatbookResponse`.
- [x] Add `openwebui_db_result` to `ImportChatbookResponse`.
- [x] Add `selected_openwebui_user_id: str | None = Form(None)` to `import_chatbook`.
- [x] Branch filename validation:
  - chatbook -> `validate_filename`
  - JSON -> `validate_json_filename`
  - DB -> `validate_sqlite_filename`
- [x] For DB preview, save to temp, call `service.preview_openwebui_db`, always cleanup preview temp file.
- [x] For DB import, require selected user id before calling `service.import_chatbook`.

### Task 2.3: Extend service dispatch

**Files:**
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`

- [x] Add `selected_openwebui_user_id: str | None = None` parameter to `import_chatbook`.
- [x] Accept source formats `{"chatbook", "openwebui_json", "openwebui_db"}`.
- [x] For DB source:
  - reject content selections.
  - reject media/embedding import.
  - require selected user id.
  - resolve file using `_resolve_import_upload_path`, not archive path.
  - sync dispatch to `import_openwebui_db`.
  - async payload includes selected user id.
- [x] Add `preview_openwebui_db(file_path)` service method that resolves upload path and calls the adapter.
- [x] Add `import_openwebui_db(...)` service method for selected-user DB imports; folder mirroring remains Stage 3.

- [x] Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_openwebui_db_api.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_import_validation.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_api_preview.py -q
```

Expected: PASS.

### Task 2.4: Commit API dispatch slice

- [x] Run `git diff --check`.
- [x] Commit:

```bash
git add tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py \
        tldw_Server_API/app/api/v1/endpoints/chatbooks.py \
        tldw_Server_API/app/core/Chatbooks/chatbook_service.py \
        tldw_Server_API/tests/Chatbooks/test_chatbooks_openwebui_db_api.py \
        tldw_Server_API/tests/Chatbooks/test_chatbooks_import_validation.py \
        tldw_Server_API/tests/Chatbooks/test_chatbooks_api_preview.py
git commit -m "Wire OpenWebUI database import API"
```

## Stage 3: Import Service And Folder Mirroring

**Goal:** Import selected-user DB chats into ChaCha conversations and mirror OpenWebUI folders into existing visible tldw folder support.

**Success Criteria:** Selected user chats import with existing JSON duplicate/message behavior; folder links are created idempotently; folder collisions are deterministic and warning-backed.

**Tests:** `tldw_Server_API/tests/Chatbooks/test_openwebui_import_service.py`, `tldw_Server_API/tests/Chatbooks/test_openwebui_folder_mirroring.py`.

**Status:** Complete

### Task 3.1: Add failing service and folder tests

**Files:**
- Create: `tldw_Server_API/tests/Chatbooks/test_openwebui_db_import_service.py`
- Optionally create: `tldw_Server_API/tests/Chatbooks/test_openwebui_folder_mirroring.py`

- [x] Service tests:
  - import selected user only.
  - conversation settings include `source_kind="openwebui_db"`, selected source user metadata, and folder metadata.
  - message metadata preserves attachment refs and unsupported source keys.
  - import result reports mirrored folder and folder link counts.
- [x] Folder tests:
  - namespace root/user/folder path is created under existing keyword collections.
  - conversation is linked to the folder via collection keyword and conversation keyword links.
  - repeated imports reuse existing folder links.
  - duplicate collection names outside the namespace are disambiguated or warned without merging.
  - invalid/empty path segments are sanitized, and original names remain in metadata.

- [x] Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_openwebui_folder_mirroring.py \
  tldw_Server_API/tests/Chatbooks/test_openwebui_import_service.py -q
```

Result: RED failed during collection before `openwebui_folders.py`; GREEN passed 16 tests after implementation.

### Task 3.2: Implement folder mirroring helper

**Files:**
- Create: `tldw_Server_API/app/core/Chatbooks/openwebui_folders.py`

- [x] Add helpers:
  - `sanitize_openwebui_folder_segment(value: str) -> str`
  - `build_openwebui_namespace_segments(source_user_label: str, source_user_id: str) -> list[str]`
  - `mirror_openwebui_folder_for_conversation(db, conversation_id, namespace_segments, source_path_segments, metadata) -> OpenWebUIFolderMirrorResult`
- [x] Use existing DB methods only:
  - `get_keyword_collection_by_name`
  - `add_keyword_collection`
  - `get_keyword_by_text`
  - `add_keyword`
  - `link_collection_to_keyword`
  - `link_conversation_to_keyword`
- [x] Account for `keyword_collections.name` being globally unique:
  - Use normal display segments when available.
  - If a name collides under a different parent, append a stable short OpenWebUI source hash.
  - Warn when a source path had to be disambiguated.
- [x] Keep helper idempotent. Re-running with the same source path should not create duplicate collection-keyword or conversation-keyword links.

### Task 3.3: Implement DB import service

**Files:**
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/openwebui_folders.py`

- [x] Extend `import_openwebui_db(file_path, selected_user_id, conflict_resolution, prefix_imported)` with the same result counters as JSON plus:
  - `selected_user_id`
  - `selected_user_label`
  - `mirrored_folders`
  - `folder_links`
  - `warnings`
- [x] Reuse `_ordered_openwebui_messages`, `_openwebui_timestamp_to_iso`, `_openwebui_message_metadata`, duplicate checks, rollback, and title copy helpers.
- [x] Extend `_store_openwebui_conversation_settings` to accept DB-specific source metadata without breaking JSON imports. Keep JSON result shape stable.
- [x] After each successful conversation creation and message import, call folder mirroring. If mirroring fails, keep the conversation and add a warning.
- [x] Preserve `Unfiled` for chats without usable folder path.
- [x] Avoid raw SQL and raw source content in warnings/logs.

- [x] Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_openwebui_folder_mirroring.py \
  tldw_Server_API/tests/Chatbooks/test_openwebui_import_service.py -q
```

Result: PASS, 16 tests.

### Task 3.4: Commit import service slice

- [x] Run `git diff --check`.
- [x] Run overlapping Chatbooks regression tests:
  - `python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_api_error_and_preview_mapping.py tldw_Server_API/tests/Chatbooks/test_chatbooks_openwebui_db_api.py tldw_Server_API/tests/Chatbooks/test_openwebui_import_service.py tldw_Server_API/tests/Chatbooks/test_openwebui_db_import_adapter.py tldw_Server_API/tests/Chatbooks/test_chatbook_security.py tldw_Server_API/tests/Chatbooks/test_openwebui_folder_mirroring.py -q`
  - Result: PASS, 59 tests.
- [x] Run Bandit on touched backend production files:
  - `python -m bandit -r tldw_Server_API/app/core/Chatbooks/openwebui_folders.py tldw_Server_API/app/core/Chatbooks/chatbook_service.py tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py -f json -o /private/tmp/bandit_openwebui_folders.json`
  - Result: 0 findings.
- [x] Commit:

```bash
git add tldw_Server_API/app/core/Chatbooks/chatbook_service.py \
        tldw_Server_API/app/core/Chatbooks/openwebui_folders.py \
        tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py \
        tldw_Server_API/tests/Chatbooks/test_openwebui_folder_mirroring.py \
        tldw_Server_API/tests/Chatbooks/test_openwebui_import_service.py
git commit -m "Mirror OpenWebUI database import folders"
```

## Stage 4: Async Jobs And Cleanup

**Goal:** Make DB imports work through Chatbooks async Jobs with correct payload, dispatch, status, and cleanup behavior.

**Success Criteria:** Async DB import jobs do not use archive extraction or JSON handlers; temp files are cleaned on success/failure; selected user id survives enqueue and worker execution.

**Tests:** `tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_worker_import_defaults.py`, `tldw_Server_API/tests/Chatbooks/test_chatbooks_import_cleanup.py`

**Status:** Complete

### Task 4.1: Add failing worker tests

**Files:**
- Modify: `tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_worker_import_defaults.py`
- Existing cleanup regression: `tldw_Server_API/tests/Chatbooks/test_chatbooks_import_cleanup.py`

- [x] Add test that `_handle_import` dispatches `source_format=openwebui_db` to `service.import_openwebui_db`.
- [x] Assert `_resolve_import_upload_path` is used and `_resolve_import_archive_path` is not used.
- [x] Assert selected user id is required in worker payload.
- [x] Assert worker result wraps DB result under `openwebui_db_result`.
- [x] Assert temp uploaded DB is removed after worker completion on success and failure.
- [x] Verify RED: focused worker tests failed because `openwebui_db` was still rejected as an unsupported source format.

### Task 4.2: Implement worker support

**Files:**
- Modify: `tldw_Server_API/app/core/Chatbooks/services/jobs_worker.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`

- [x] Branch `source_format == "openwebui_db"` before archive handling.
- [x] Resolve upload path.
- [x] Validate selected user id in payload.
- [x] Call `service.import_openwebui_db`.
- [x] Return `{"openwebui_db_result": result or {}}`.
- [x] Preserve the existing cleanup `finally` behavior for resolved upload paths.
- [x] Include selected user id in core Jobs and Prompt Studio fallback payloads. This was already present from Stage 2 and remains covered by existing API/service tests.

- [x] Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_worker_import_defaults.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_import_cleanup.py -q
```

Result: PASS in the broader 35-test Chatbooks regression command.

### Task 4.3: Commit Jobs slice

- [x] Run `git diff --check`.
- [x] Run overlapping Chatbooks regression tests:
  - `python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_worker_import_defaults.py tldw_Server_API/tests/Chatbooks/test_chatbooks_import_cleanup.py tldw_Server_API/tests/Chatbooks/test_chatbooks_openwebui_db_api.py tldw_Server_API/tests/Chatbooks/test_openwebui_import_service.py tldw_Server_API/tests/Chatbooks/test_openwebui_db_import_adapter.py tldw_Server_API/tests/Chatbooks/test_openwebui_folder_mirroring.py -q`
  - Result: PASS, 35 tests.
- [x] Run Bandit on touched backend production file:
  - `python -m bandit -r tldw_Server_API/app/core/Chatbooks/services/jobs_worker.py -f json -o /private/tmp/bandit_openwebui_db_jobs.json`
  - Result: 0 findings.
- [x] Commit:

```bash
git add tldw_Server_API/app/core/Chatbooks/services/jobs_worker.py \
        tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_worker_import_defaults.py \
        backlog/tasks/task-233.9\ -\ Support-OpenWebUI-DB-async-Chatbooks-import-jobs.md \
        Docs/superpowers/plans/2026-05-10-openwebui-db-chat-import-implementation-plan.md
git commit -m "Support async OpenWebUI database imports"
```

## Stage 5: Frontend Import UI

**Goal:** Add OpenWebUI database mode to the Chatbooks import tab with preview-time user selection and selected user submission.

**Success Criteria:** Users can choose OpenWebUI database, upload `.db`/`.sqlite`, preview users, select exactly one user, see destination namespace/attachment warnings, and import with `selected_openwebui_user_id`.

**Tests:** `apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx`, existing OpenWebUI UI and client tests.

**Status:** Complete

### Task 5.1: Extend frontend client tests first

**Files:**
- Modify: `apps/packages/ui/src/services/__tests__/tldw-api-client.chatbooks-openwebui.test.ts`

- [x] Add preview test for `source_format="openwebui_db"`.
- [x] Add import test that includes `selected_openwebui_user_id`.
- [x] Confirm boolean options are still stringified as multipart fields.

- [x] Run:

```bash
cd apps/packages/ui
./node_modules/.bin/vitest run src/services/__tests__/tldw-api-client.chatbooks-openwebui.test.ts src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx
```

Verified: PASS after implementation, 2 files and 7 tests.

### Task 5.2: Update API client upload options

**Files:**
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/chat-rag.ts`

- [x] Add `selected_openwebui_user_id?: string` to `importChatbook` options.
- [x] No custom serialization is needed beyond the existing normalized multipart field loop.
- [x] Keep `previewChatbook` unchanged except tests should prove it passes `source_format=openwebui_db`.

### Task 5.3: Add failing UI tests

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx`
- DB-mode coverage was folded into the existing OpenWebUI import test file to reuse the existing setup and stale-preview coverage.

- [x] Test source selector includes `OpenWebUI database`.
- [x] Switching to DB mode changes dropzone accept to `.db,.sqlite`.
- [x] DB mode hides archive content selection, media, and embeddings controls.
- [x] Preview response with two users renders selectable user rows.
- [x] Import button is disabled or errors until a user is selected.
- [x] Import sends `source_format=openwebui_db` and `selected_openwebui_user_id`.
- [x] Existing stale preview guard remains covered by the shared OpenWebUI preview test.

### Task 5.4: Implement UI state and rendering

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx`

- [x] Extend type:

```ts
type ImportSourceFormat = "chatbook" | "openwebui_json" | "openwebui_db"
```

- [x] Split source helpers:
  - `isOpenWebUIJsonImport`
  - `isOpenWebUIDatabaseImport`
  - `isOpenWebUIImport = isOpenWebUIJsonImport || isOpenWebUIDatabaseImport`
- [x] Add state:
  - `openwebuiDbPreview`
  - `selectedOpenWebUIUserId`
- [x] Reset selected user on source/file changes.
- [x] Source options:
  - Chatbook archive
  - OpenWebUI JSON
  - OpenWebUI database
- [x] Dropzone accept:
  - chatbook -> `.zip,.chatbook`
  - JSON -> `.json`
  - DB -> `.db,.sqlite`
- [x] In preview handler:
  - JSON reads `res.openwebui_preview`
  - DB reads `res.openwebui_db_preview`
- [x] Render DB preview card with user table:
  - label/name/email-safe value
  - chat count
  - folder count
  - message count
  - warning count
  - attachment refs
  - destination namespace `OpenWebUI / <label>`
- [x] In import handler:
  - reject missing selected user id for DB mode before calling client.
  - send selected user id.
  - force media/embeddings false for both OpenWebUI modes.
  - limit conflicts to skip/rename for both OpenWebUI modes.

- [x] Run:

```bash
cd apps/packages/ui
./node_modules/.bin/vitest run src/services/__tests__/tldw-api-client.chatbooks-openwebui.test.ts src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx
```

Verified: PASS, 2 files and 7 tests.

### Task 5.5: Commit frontend slice

- [x] Run `git diff --check`.
- [x] Commit:

```bash
git add apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx \
        apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx \
        apps/packages/ui/src/services/tldw/TldwApiClient.ts \
        apps/packages/ui/src/services/tldw/domains/chat-rag.ts \
        apps/packages/ui/src/services/__tests__/tldw-api-client.chatbooks-openwebui.test.ts
git commit -m "Add OpenWebUI database import UI"
```

## Stage 6: Documentation, OpenAPI, And Final Verification

**Goal:** Make the DB import feature discoverable and verify the full touched surface.

**Success Criteria:** User docs and API docs describe DB import separately from JSON import; OpenAPI docs include new source format and selected-user field; focused backend/frontend tests and Bandit pass.

**Tests:** Docs regression test plus focused backend/frontend suites.

**Status:** Not Started

### Task 6.1: Update docs and OpenAPI docs

**Files:**
- Modify: `README.md`
- Modify: `Docs/User_Guides/WebUI_Extension/Chatbook_User_Guide.md`
- Modify: `Docs/User_Guides/index.md`
- Modify: `Docs/API-related/Chatbook_API_Documentation.md`
- Modify: `Docs/API-related/chatbook_openapi.yaml`
- Modify: `Docs/API-related/API_README.md`
- Modify: `Docs/API-related/API_Tags_Index.md`
- Modify: published mirrors under `Docs/Published/...`
- Modify: `tldw_Server_API/tests/Docs/test_chatbook_openwebui_import_docs.py`

- [ ] Document the two OpenWebUI sources distinctly:
  - OpenWebUI JSON from `Export Chats`.
  - OpenWebUI database from uploaded `webui.db`.
- [ ] Explain selected-user requirement.
- [ ] Explain folder namespace mirroring.
- [ ] Explain duplicate skip/rename behavior.
- [ ] State that attachment binaries are not imported.
- [ ] Include multipart form fields:
  - `source_format=openwebui_db`
  - `selected_openwebui_user_id`
  - `conflict_resolution`
  - `prefix_imported`
  - `async_mode`
- [ ] Extend docs tests to assert DB import is discoverable.

### Task 6.2: Run final verification

- [ ] Backend focused tests:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_openwebui_db_import_adapter.py \
  tldw_Server_API/tests/Chatbooks/test_openwebui_db_import_service.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_openwebui_db_api.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_worker_import_defaults.py \
  tldw_Server_API/tests/Chatbooks/test_openwebui_import_adapter.py \
  tldw_Server_API/tests/Chatbooks/test_openwebui_import_service.py \
  tldw_Server_API/tests/Docs/test_chatbook_openwebui_import_docs.py -q
```

- [ ] Frontend focused tests:

```bash
cd apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx \
  ../packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-db-import.test.tsx \
  ../packages/ui/src/services/__tests__/tldw-api-client.chatbooks-openwebui.test.ts
```

- [ ] Security scan:

```bash
source .venv/bin/activate
python -m bandit -r tldw_Server_API/app/core/Chatbooks \
  tldw_Server_API/app/api/v1/endpoints/chatbooks.py \
  tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py \
  -f json -o /tmp/bandit_openwebui_db_import.json
```

Expected: 0 new findings in touched code. If Bandit flags URI construction or SQL strings, inspect and either fix or document why an existing static SQL allowlist is safe.

- [ ] Diff hygiene:

```bash
git diff --check
```

Expected: no output.

### Task 6.3: Commit docs and final verification notes

- [ ] Update the Backlog task for implementation with final commands and results.
- [ ] Commit:

```bash
git add README.md Docs/User_Guides/WebUI_Extension/Chatbook_User_Guide.md \
        Docs/User_Guides/index.md Docs/API-related/Chatbook_API_Documentation.md \
        Docs/API-related/chatbook_openapi.yaml Docs/API-related/API_README.md \
        Docs/API-related/API_Tags_Index.md Docs/Published \
        tldw_Server_API/tests/Docs/test_chatbook_openwebui_import_docs.py
git commit -m "Document OpenWebUI database imports"
```

## Manual Review Checklist

- [ ] No code path imports all OpenWebUI users by default.
- [ ] No endpoint accepts a server-local DB path.
- [ ] Preview does not return raw chat/message content.
- [ ] Errors and logs do not echo raw chat content, message content, full upload paths, private source emails, or attachment storage paths.
- [ ] DB imports use read-only SQLite and parameterized selected-user queries.
- [ ] Repeated JSON and DB imports of the same OpenWebUI chat id use the same duplicate surface.
- [ ] Folder mirroring does not create a new folder schema and does not merge with non-OpenWebUI folders.
- [ ] Attachment/file/artifact references are metadata-only.
- [ ] Async and sync import behavior return comparable structured result counts.
- [ ] Docs clearly distinguish JSON export import from `webui.db` import.
