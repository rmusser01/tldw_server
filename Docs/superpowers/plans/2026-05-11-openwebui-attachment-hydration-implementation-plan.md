# OpenWebUI Attachment Hydration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a post-import OpenWebUI attachment hydration workflow that restores referenced images to imported chat messages and registers referenced non-image files in Media DB from a trusted local OpenWebUI data root.

**Architecture:** Keep chat import unchanged and add a separate Chatbooks hydration surface. A focused backend service reads imported OpenWebUI metadata, validates a server-local data root under configured ingestion-source allowed roots, resolves `file` and `chat_file` rows from `webui.db`, copies bytes into tldw-owned storage, updates message hydration metadata with deep-merge semantics, and runs through a dedicated `openwebui_attachment_hydration` Jobs contract for long-running work.

**Tech Stack:** FastAPI, Pydantic, SQLite via `sqlite3`, existing ChaChaNotesDB message metadata/image APIs, Media DB `Media`/`MediaFiles`, core Jobs, pytest, Bandit, React/Ant Design, Vitest.

---

## Source Inputs

- Design spec: `Docs/superpowers/specs/2026-05-11-openwebui-attachment-hydration-design.md`
- Prior OpenWebUI JSON plan: `Docs/superpowers/plans/2026-05-10-openwebui-chat-import-implementation-plan.md`
- Prior OpenWebUI DB plan: `Docs/superpowers/plans/2026-05-10-openwebui-db-chat-import-implementation-plan.md`
- OpenWebUI DB helper: `tldw_Server_API/app/core/DB_Management/OpenWebUI_DB.py`
- OpenWebUI import adapter: `tldw_Server_API/app/core/Chatbooks/import_adapters/openwebui.py`
- OpenWebUI DB adapter: `tldw_Server_API/app/core/Chatbooks/import_adapters/openwebui_db.py`
- Chatbooks service: `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
- Chatbooks Jobs worker: `tldw_Server_API/app/core/Chatbooks/services/jobs_worker.py`
- Chatbooks API: `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`
- Chatbooks API schemas: `tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py`
- ChaCha message store: `tldw_Server_API/app/core/DB_Management/chacha/message_store.py`
- Media DB runtime helpers: `tldw_Server_API/app/core/DB_Management/media_db/runtime/media_entrypoint_ops.py`
- MediaFiles runtime helpers: `tldw_Server_API/app/core/DB_Management/media_db/runtime/media_file_ops.py`
- Allowed local path helper: `tldw_Server_API/app/core/Ingestion_Media_Processing/path_utils.py`
- Ingestion allowed-root config: `tldw_Server_API/app/core/config.py`
- Chatbooks WebUI: `apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx`
- WebUI client methods: `apps/packages/ui/src/services/tldw/TldwApiClient.ts` and `apps/packages/ui/src/services/tldw/domains/chat-rag.ts`

## Constraints

- V1 accepts a server-local OpenWebUI data root only. It does not accept uploaded ZIP bundles and does not call live OpenWebUI APIs.
- Hydrate only attachments referenced by imported OpenWebUI chats plus DB `chat_file` fallback for those imported chats.
- Do not alter baseline OpenWebUI JSON or DB chat import semantics.
- Do not make `file` or `chat_file` tables mandatory for text-only `openwebui_db` import.
- Do not use global content-hash dedupe for binary file registration.
- Do not reuse media rows across tldw users.
- Do not store durable Media DB records pointing at the OpenWebUI source path.
- Do not overwrite or duplicate existing non-OpenWebUI message images.
- Do not rely on `set_message_metadata_extra(..., merge=True)` for nested `openwebui_import` updates.
- Do not trust file extension or OpenWebUI MIME metadata for image embedding; sniff bytes conservatively.
- Do not log raw chat content, message content, arbitrary absolute source paths, or file bytes.
- Run Python commands from repo root after `source .venv/bin/activate`.
- Run Bandit on touched backend code before claiming completion.

## File Structure

Create:

- `tldw_Server_API/app/core/Chatbooks/openwebui_hydration.py`
  - Hydration dataclasses, preview/result summaries, reference extraction, local data-root validation, file resolution, file classification, metadata merging, image hydration, and non-image registration orchestration.
  - No FastAPI request parsing and no raw SQL beyond calling DB_Management helpers.
- `tldw_Server_API/app/core/Chatbooks/openwebui_hydration_jobs.py`
  - Constants and small helpers for core Jobs payload creation: domain, queue, job type, payload normalization.
- `tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_db_helpers.py`
  - Hydration-specific OpenWebUI DB file/chat_file schema and row helper tests.
- `tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_paths.py`
  - Data-root validation and path traversal/symlink/file-candidate tests.
- `tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_service.py`
  - Reference extraction, preview summaries, metadata merge, image hydration, non-image registration, dedupe, and partial failure tests.
- `tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_api.py`
  - Preview/job endpoint contract and authorization tests.
- `tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_jobs_worker.py`
  - Dedicated Jobs routing and worker revalidation tests.

Modify:

- `tldw_Server_API/app/core/DB_Management/OpenWebUI_DB.py`
  - Add hydration-specific `file` and `chat_file` schema validation and read-only row helpers.
- `tldw_Server_API/app/core/DB_Management/chacha/message_store.py`
  - Add post-insert message-image append helper that can append after current max position inside a caller-managed transaction.
- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
  - Expose the new message image helper through the legacy DB facade if required by the existing delegation pattern.
- `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
  - Wire hydration preview and job creation helper methods while keeping existing import/export paths untouched.
- `tldw_Server_API/app/core/Chatbooks/services/jobs_worker.py`
  - Route `job_type=openwebui_attachment_hydration` to the hydration service.
- `tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py`
  - Add hydration request/response schemas and result item schemas.
- `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`
  - Add `/openwebui/hydration/preview`, `/openwebui/hydration/jobs`, and `/openwebui/hydration/jobs/{job_id}`.
- `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
  - Add hydration preview/job/status client methods.
- `apps/packages/ui/src/services/tldw/domains/chat-rag.ts`
  - Mirror hydration client methods in the domain client.
- `apps/packages/ui/src/services/__tests__/tldw-api-client.chatbooks-openwebui.test.ts`
  - Add API-client contract coverage for hydration endpoints.
- `apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx`
  - Add authorized hydration UI in the Chatbooks import area.
- `apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx`
  - Add hydration UI tests.
- User/API docs:
  - `Docs/User_Guides/WebUI_Extension/Chatbook_User_Guide.md`
  - `Docs/API-related/Chatbook_API_Documentation.md`
  - `Docs/API-related/chatbook_openapi.yaml`
  - `Docs/API-related/API_README.md`
  - `Docs/API-related/API_Tags_Index.md`
  - `Docs/Published/User_Guides/WebUI_Extension/Chatbook_User_Guide.md`
  - `Docs/Published/API-related/Chatbook_API_Documentation.md`
  - `Docs/Published/API-related/chatbook_openapi.yaml`
  - `Docs/Published/API-related/API_README.md`
  - `Docs/Published/API-related/API_Tags_Index.md`
- `tldw_Server_API/tests/Docs/test_chatbook_openwebui_import_docs.py`
  - Add docs regression coverage for hydration discoverability.

## Stage 1: OpenWebUI File DB Helpers

**Goal:** Add read-only helpers for OpenWebUI `file` and `chat_file` rows without changing chat-import validation.

**Success Criteria:** Hydration can validate file tables and load rows by file/chat ids; existing `open_validated_openwebui_db()` still accepts DBs with only chat-import tables.

**Tests:** `tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_db_helpers.py`

**Status:** Complete

### Task 1.1: Write failing DB helper tests

**Files:**
- Create: `tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_db_helpers.py`
- Modify: `tldw_Server_API/app/core/DB_Management/OpenWebUI_DB.py`

- [x] Add a temporary SQLite fixture with `user`, `chat`, `folder`, `file`, and `chat_file` tables:

```python
def write_openwebui_hydration_db(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE user (id TEXT PRIMARY KEY, name TEXT, email TEXT, created_at INTEGER, updated_at INTEGER)")
    conn.execute("CREATE TABLE folder (id TEXT PRIMARY KEY, parent_id TEXT, user_id TEXT, name TEXT, items TEXT, meta TEXT, is_expanded INTEGER, created_at INTEGER, updated_at INTEGER)")
    conn.execute("CREATE TABLE chat (id TEXT PRIMARY KEY, user_id TEXT, title TEXT, chat TEXT, created_at INTEGER, updated_at INTEGER, share_id TEXT, archived INTEGER, pinned INTEGER, meta TEXT, folder_id TEXT)")
    conn.execute("CREATE TABLE file (id TEXT PRIMARY KEY, user_id TEXT, hash TEXT, filename TEXT, path TEXT, data TEXT, meta TEXT, created_at INTEGER, updated_at INTEGER)")
    conn.execute("CREATE TABLE chat_file (id TEXT PRIMARY KEY, chat_id TEXT, file_id TEXT, message_id TEXT, user_id TEXT, created_at INTEGER, updated_at INTEGER)")
    ...
```

- [x] Test `validate_openwebui_file_schema(conn)` fails when `file` is missing.
- [x] Test `validate_openwebui_file_schema(conn)` fails when `file.id`, `file.user_id`, or `file.filename` is missing.
- [x] Test baseline `validate_openwebui_schema(conn)` still passes without `file` and `chat_file`.
- [x] Test `load_openwebui_file_rows_for_ids(conn, ["file-a"], user_id="owui-user")` returns only that user's row.
- [x] Test `load_openwebui_chat_file_rows_for_chats(conn, ["chat-a"], user_id="owui-user")` ignores unrelated chats/users.
- [x] Test helpers use bound parameters by checking ids containing quotes are treated as literal values.

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_db_helpers.py -q
```

Expected: FAIL because the helper functions do not exist.

### Task 1.2: Implement hydration-specific OpenWebUI DB helpers

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/OpenWebUI_DB.py`

- [x] Add `HYDRATION_FILE_SCHEMA` and `HYDRATION_CHAT_FILE_SCHEMA` constants separate from `REQUIRED_SCHEMA`.
- [x] Add `validate_openwebui_file_schema(conn: sqlite3.Connection) -> None`.
- [x] Add `load_openwebui_file_rows_for_ids(conn, file_ids, user_id=None)`.
- [x] Add `load_openwebui_chat_file_rows_for_chats(conn, chat_ids, user_id=None)`.
- [x] Add `iter_openwebui_files_for_user(conn, user_id)` only for future/full-library support; do not use it for v1 default hydration scope.
- [x] Keep all SQL parameterized.
- [x] Return `sqlite3.Row` objects for consistency with the existing OpenWebUI DB helpers.

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_db_helpers.py -q
python -m pytest tldw_Server_API/tests/Chatbooks/test_openwebui_db_import_adapter.py -q
```

Expected: PASS. The second command proves text-only DB import validation was not broadened.

### Task 1.3: Commit DB helper slice

- [x] Run `git diff --check`.
- [x] Commit:

```bash
git add tldw_Server_API/app/core/DB_Management/OpenWebUI_DB.py \
        tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_db_helpers.py
git commit -m "Add OpenWebUI hydration DB helpers"
```

## Stage 2: Hydration Service Core And Path Safety

**Goal:** Build preview-safe path validation, reference extraction, source row lookup, and file resolution without writing images/media yet.

**Success Criteria:** Preview can identify referenced files, resolve safe source paths under the OpenWebUI data root/uploads root, classify basic file kinds, and report per-reference warnings without copying bytes.

**Tests:** `tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_paths.py`, first service tests in `test_openwebui_hydration_service.py`

**Status:** Complete

### Task 2.1: Write failing path and preview tests

**Files:**
- Create: `tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_paths.py`
- Create: `tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_service.py`
- Create: `tldw_Server_API/app/core/Chatbooks/openwebui_hydration.py`

- [x] Test data root outside `INGESTION_SOURCE_ALLOWED_ROOTS` is rejected.
- [x] Test missing `webui.db` is a fatal preview error.
- [x] Test missing `uploads/` is a fatal preview error only when referenced file bytes are needed.
- [x] Test `file.path="../secret.txt"` is rejected as `path_rejected`.
- [x] Test symlink escape is rejected by canonical target checks.
- [x] Test fallback `uploads/{file.id}_{file.filename}` resolves when safe.
- [x] Test imported message metadata refs are extracted from `extra.openwebui_import.attachment_refs`.
- [x] Test unsupported ref shapes produce `unsupported_reference_shape`.
- [x] Test DB `chat_file` fallback uses preserved `openwebui_import.metadata.row_id` and skips fallback when absent.

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_paths.py \
  tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_service.py \
  -q
```

Expected: FAIL because `openwebui_hydration.py` does not exist.

### Task 2.2: Implement preview data models and path resolution

**Files:**
- Create: `tldw_Server_API/app/core/Chatbooks/openwebui_hydration.py`

- [x] Add dataclasses:
  - `OpenWebUIHydrationScope`
  - `OpenWebUIHydrationReference`
  - `OpenWebUIHydrationResolvedFile`
  - `OpenWebUIHydrationPreviewItem`
  - `OpenWebUIHydrationPreview`
  - `OpenWebUIHydrationResult`
- [x] Add `validate_openwebui_data_root(root: str | Path) -> OpenWebUIDataRoot`.
- [x] Use `get_ingestion_source_allowed_roots(reload=True)` or the same reload behavior established by ingestion-source tests.
- [x] Use `resolve_safe_local_path()` to confirm data root is under one allowed root.
- [x] Require canonical `webui.db` under data root.
- [x] Resolve canonical `uploads/` under data root and require it when file bytes are needed.
- [x] Add `resolve_openwebui_file_path(file_row, data_root)`.
- [x] Try candidates in this order:
  1. relative `file.path` under data root/uploads root
  2. absolute `file.path` only if it canonicalizes under data root/uploads root
  3. `uploads/{file.id}_{file.filename}`
- [x] Return structured warnings, not raw absolute paths, for user-facing preview.

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_paths.py -q
```

Expected: PASS.

### Task 2.3: Implement reference extraction and preview assembly

**Files:**
- Modify: `tldw_Server_API/app/core/Chatbooks/openwebui_hydration.py`

- [x] Add `extract_openwebui_hydration_references(chacha_db, scope)`.
- [x] Query messages for the active tldw user/conversation scope using existing DB helpers where available; if no focused helper exists, add the smallest ChaCha helper needed rather than scanning raw DB files outside DB_Management.
- [x] Read current `message_metadata.extra.openwebui_import`.
- [x] Preserve the raw reference index and shape in the preview item.
- [x] Recognize dict ids from `id`, `file_id`, and `fileId`.
- [x] Recognize string refs only when non-empty.
- [x] Add DB fallback from `chat_file` rows only for imported conversation source chat ids.
- [x] Bound preview warning arrays to avoid huge responses.

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_service.py -k "preview or reference or fallback" -q
```

Expected: PASS.

### Task 2.4: Commit preview/path slice

- [x] Run `git diff --check`.
- [x] Commit:

```bash
git add tldw_Server_API/app/core/Chatbooks/openwebui_hydration.py \
        tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_paths.py \
        tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_service.py
git commit -m "Add OpenWebUI hydration preview service"
```

## Stage 3: ChaCha Image Hydration And Metadata Merge

**Goal:** Restore safe image refs as message images after import, without losing original OpenWebUI metadata and without duplicating images on retry.

**Success Criteria:** Image hydration appends to message images idempotently, records source-key to image-position mapping in metadata, and preserves original `openwebui_import` fields.

**Tests:** `tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_service.py`

**Status:** Complete

### Task 3.1: Write failing image and metadata merge tests

**Files:**
- Modify: `tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_service.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/message_store.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`

- [x] Test a message with existing metadata:

```python
original = {
    "openwebui_import": {
        "source_message_id": "msg-a",
        "source_parent_id": None,
        "source_children_ids": [],
        "role": "user",
        "model": "model-a",
        "attachment_refs": [{"id": "file-image"}],
        "metadata": {"done": True},
    }
}
```

- [x] Hydrate one PNG ref and assert all original keys remain.
- [x] Assert `hydration.items[0].status == "hydrated_image"`.
- [x] Assert image position is appended after any existing `message_images` positions.
- [x] Run hydration twice and assert one image row, with second item status `already_hydrated` or unchanged existing item.
- [x] Test oversized image returns `oversized`.
- [x] Test extension says `.png` but bytes are not image-like and returns `unsupported_file_type`.
- [x] Test metadata update failure rolls back the appended image row.

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_service.py -k "image or metadata" -q
```

Expected: FAIL because append/idempotency helpers are missing.

### Task 3.2: Add post-insert message image append helper

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/message_store.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`

- [x] Add `append_message_image(message_id: str, image_bytes: bytes, mime_type: str) -> int`.
- [x] Validate max image size using the same `MAX_MESSAGE_IMAGE_BYTES` setting used by `add_message()`.
- [x] Select `COALESCE(MAX(position), -1) + 1` for the message.
- [x] Insert a new row at that position.
- [x] Return the inserted position.
- [x] Keep transaction boundaries compatible with service-level calls; if service manages a larger transaction, expose a helper that accepts `commit=False` or uses existing DB transaction conventions.
- [x] Do not change `_insert_message_images()` behavior for normal chatbook import.

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_service.py -k "image" -q
```

Expected: still FAIL until service code calls the helper.

### Task 3.3: Implement image hydration and deep metadata merge

**Files:**
- Modify: `tldw_Server_API/app/core/Chatbooks/openwebui_hydration.py`

- [x] Add `merge_openwebui_message_hydration_metadata(chacha_db, message_id, item_update)`.
- [x] Read `get_message_metadata(message_id)`.
- [x] Copy the full `extra.openwebui_import` object.
- [x] Add or update `hydration.version`, `last_job_id`, and `items`.
- [x] Preserve original `attachment_refs`, source ids, role, model, and metadata.
- [x] Add `hydrate_image_reference(...)`.
- [x] Derive `source_key = f"openwebui:file:{source_file_id}"` when file id exists, else `openwebui:hash:{sha256}`.
- [x] If `source_key` already appears in hydration items with `message_image_position`, return `already_hydrated` without inserting.
- [x] Sniff image bytes with a small allowlist for PNG, JPEG, GIF, and WebP signatures.
- [x] Append the image and record position.
- [x] Wrap image insert plus metadata update in one transaction when the DB exposes transaction support.

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_service.py -k "image or metadata" -q
```

Expected: PASS.

### Task 3.4: Commit image hydration slice

- [x] Run `git diff --check`.
- [x] Commit:

```bash
git add tldw_Server_API/app/core/Chatbooks/openwebui_hydration.py \
        tldw_Server_API/app/core/DB_Management/chacha/message_store.py \
        tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
        tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_service.py
git commit -m "Hydrate OpenWebUI image attachments"
```

## Stage 4: Non-Image Media Registration

**Goal:** Register referenced non-image files as durable tldw Media DB entries without automatic processing.

**Success Criteria:** Files are copied into tldw-owned storage, parent Media rows and MediaFiles rows are created with owner-aware source metadata, and dedupe does not cross tldw users or unrelated imports.

**Tests:** `tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_service.py`

**Status:** Complete

### Task 4.1: Write failing media registration tests

**Files:**
- Modify: `tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_service.py`

- [x] Test a PDF ref creates a Media row and a MediaFiles original row.
- [x] Assert the MediaFiles storage path is not under the OpenWebUI source data root.
- [x] Assert `source_hash` or `checksum` equals byte SHA-256.
- [x] Assert `owner_user_id` is the active tldw user and `visibility == "personal"`.
- [x] Assert a same `source_file_id` in the same user scope reuses the existing Media link.
- [x] Assert the same bytes for a different tldw user does not reuse another user's Media row.
- [x] Assert two source-id-less files with different filenames but empty extracted text do not collapse to the same placeholder content hash.
- [x] Assert `process_supported_files=false` leaves chunking/processing pending and does not enqueue processing.
- [x] Assert `process_supported_files=true` calls a mocked processing hook after registration and processing failure does not remove the MediaFiles row.

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_service.py -k "media or non_image or processing" -q
```

Expected: FAIL because non-image registration is missing.

### Task 4.2: Implement durable copy and owner-aware media registration

**Files:**
- Modify: `tldw_Server_API/app/core/Chatbooks/openwebui_hydration.py`
- Modify: Media DB helper files only if an owner-aware lookup helper is missing:
  - `tldw_Server_API/app/core/DB_Management/media_db/runtime/query_ops.py`
  - `tldw_Server_API/app/core/DB_Management/media_db/api.py`
  - `tldw_Server_API/app/core/DB_Management/media_db/repositories/media_repository.py`

- [x] Add a storage helper that copies source bytes into a per-user tldw-owned import directory. Prefer an existing media storage convention if one exists; otherwise use a new Chatbooks-owned subdirectory under the configured user database/media area.
- [x] Use `open_safe_local_path()` for source file reads when practical.
- [x] Compute `sha256` from source bytes while streaming/copying.
- [x] Add owner-aware lookup by OpenWebUI source id in safe metadata/source URL.
- [x] Add owner-aware lookup by byte hash only inside current hydration run when no source id exists.
- [x] Register a parent Media row with:

```python
placeholder_content = json.dumps(
    {
        "source": "openwebui",
        "source_file_id": source_file_id,
        "filename": filename,
        "mime_type": mime_type,
        "sha256": byte_sha256,
    },
    sort_keys=True,
)
```

- [x] Use an OpenWebUI-specific URL:
  - `openwebui://user/{owner_user_id}/file/{source_file_id}` when source id exists
  - `openwebui://user/{owner_user_id}/run/{job_id}/{byte_sha256}` when source id is missing
  - Chosen instead of the earlier non-owner URL shape because `Media.url` is globally unique in the current schema.
- [x] Pass `source_hash=byte_sha256`, `owner_user_id=<tldw user id>`, and `visibility="personal"`.
- [x] Insert `MediaFiles` row with `file_type="original"`, copied storage path, original filename, file size, MIME type, and checksum.
- [x] Record `status="registered_media"` and `media_id`/`media_file_id` in message hydration metadata.
- [x] Keep optional processing hook separated behind `process_supported_files`.

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_service.py -k "media or non_image or processing" -q
```

Expected: PASS.

### Task 4.3: Commit media registration slice

- [x] Run `git diff --check`.
- [x] Commit:

```bash
git add tldw_Server_API/app/core/Chatbooks/openwebui_hydration.py \
        tldw_Server_API/app/core/DB_Management/media_db/runtime/query_ops.py \
        tldw_Server_API/app/core/DB_Management/media_db/api.py \
        tldw_Server_API/app/core/DB_Management/media_db/repositories/media_repository.py \
        tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_service.py
git commit -m "Register OpenWebUI attachments in Media DB"
```

If no Media DB helper files changed, omit them from `git add`.

## Stage 5: API Schemas And Endpoints

**Goal:** Expose hydration preview, job creation, and job status endpoints with stronger authorization than normal import.

**Success Criteria:** Authorized single-user owner and multi-user admins can preview/enqueue; multi-user non-admins are rejected; request/response schemas match the service; endpoints do not leak raw source paths.

**Tests:** `tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_api.py`

**Status:** Complete

### Task 5.1: Write failing API contract and auth tests

**Files:**
- Create: `tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_api.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`

- [x] Test `POST /api/v1/chatbooks/openwebui/hydration/preview` validates `openwebui_data_root`.
- [x] Test preview request passes `source_user_id`, `conversation_ids`, and `process_supported_files`.
- [x] Test preview response includes counts and warning totals.
- [x] Test multi-user non-admin receives 403.
- [x] Test single-user principal is allowed.
- [x] Test admin role/claim in multi-user mode is allowed.
- [x] Test response warnings redact full absolute paths.
- [x] Test `POST /api/v1/chatbooks/openwebui/hydration/jobs` enqueues a core Jobs row with job type `openwebui_attachment_hydration`.
- [x] Test `GET /api/v1/chatbooks/openwebui/hydration/jobs/{job_id}` rejects jobs not owned by the current user unless admin access is explicitly intended.

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_api.py -q
```

Expected: FAIL because schemas/endpoints do not exist.

### Task 5.2: Add Pydantic hydration schemas

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py`

- [x] Add `OpenWebUIHydrationScopeRequest`.
- [x] Add `OpenWebUIHydrationPreviewRequest`.
- [x] Add `OpenWebUIHydrationJobRequest`.
- [x] Add `OpenWebUIHydrationItemResponse`.
- [x] Add `OpenWebUIHydrationSummaryResponse`.
- [x] Add `OpenWebUIHydrationPreviewResponse`.
- [x] Add `OpenWebUIHydrationJobResponse`.
- [x] Constrain `conversation_ids` to a bounded list of non-empty strings.
- [x] Keep `process_supported_files` default `False`.
- [x] Do not include raw source path fields in response schemas.

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_api.py -k "schema or preview" -q
```

Expected: schema-specific tests progress, endpoint tests still fail.

### Task 5.3: Add endpoint authorization helper and preview route

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`

- [x] Add `_require_openwebui_hydration_access(user)`:
  - allow single-user principal/current user as configured by existing auth helpers
  - require admin-style claims/role in multi-user mode
- [x] Prefer existing auth helper patterns in `auth_deps.py`, `setup_deps.py`, or `chat_workflows_deps.py`.
- [x] Add service method `preview_openwebui_attachment_hydration(...)`.
- [x] Add endpoint `POST /api/v1/chatbooks/openwebui/hydration/preview`.
- [x] Convert service preview dataclasses to Pydantic response models.
- [x] Map validation/security errors to 400/403 without stack traces.

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_api.py -k "preview or auth" -q
```

Expected: PASS for preview/auth subset.

### Task 5.4: Add job creation and status routes

**Files:**
- Create: `tldw_Server_API/app/core/Chatbooks/openwebui_hydration_jobs.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`

- [x] Add constants:

```python
CHATBOOKS_DOMAIN = "chatbooks"
OPENWEBUI_ATTACHMENT_HYDRATION_JOB_TYPE = "openwebui_attachment_hydration"
```

- [x] Add `create_openwebui_hydration_job(jobs_manager, payload, owner_user_id)`.
- [x] Service job payload includes:
  - `user_id`
  - `openwebui_data_root`
  - `scope`
  - `process_supported_files`
  - preview confirmation token or preview summary hash if implemented
- [ ] Revalidate authorization and roots in the worker, not only in endpoint. Deferred to Stage 6 worker execution.
- [x] Add endpoint `POST /api/v1/chatbooks/openwebui/hydration/jobs`.
- [x] Add endpoint `GET /api/v1/chatbooks/openwebui/hydration/jobs/{job_id}`.
- [x] Status endpoint reads core Jobs by uuid/id and filters by domain/job_type/owner.

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_api.py -q
```

Expected: PASS.

### Task 5.5: Commit API slice

- [x] Run `git diff --check`.
- [x] Commit:

```bash
git add tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py \
        tldw_Server_API/app/api/v1/endpoints/chatbooks.py \
        tldw_Server_API/app/core/Chatbooks/chatbook_service.py \
        tldw_Server_API/app/core/Chatbooks/openwebui_hydration_jobs.py \
        tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_api.py
git commit -m "Expose OpenWebUI attachment hydration API"
```

Verification:
- Red run before implementation: focused API tests failed because schemas/routes did not exist.
- `python -m pytest tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_api.py -q` -> 7 passed.
- `python -m pytest tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_api.py tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_paths.py tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_service.py tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_db_helpers.py tldw_Server_API/tests/Chatbooks/test_openwebui_db_import_adapter.py -q` -> 47 passed.
- `python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_path_traversal.py -q` -> 3 passed.
- `git diff --check` -> clean.
- `python -m bandit -r tldw_Server_API/app/api/v1/endpoints/chatbooks.py tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py tldw_Server_API/app/core/Chatbooks/chatbook_service.py tldw_Server_API/app/core/Chatbooks/openwebui_hydration_jobs.py -f json -o /tmp/bandit_openwebui_hydration_api.json` -> 0 findings, 0 errors.
- Known non-gating check: `test_chatbooks_api_path_guard.py` with the path-traversal suite timed out during full `app.main` `TestClient` teardown after unrelated lifespan workers started.

## Stage 6: Jobs Worker Execution

**Goal:** Execute hydration jobs asynchronously through a dedicated Chatbooks core Jobs type.

**Success Criteria:** Worker routes only `openwebui_attachment_hydration`, revalidates permissions and paths, updates core Jobs result summary, and leaves existing import/export behavior unchanged.

**Tests:** `tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_jobs_worker.py`, existing Chatbooks worker tests.

**Status:** Complete

### Task 6.1: Write failing worker routing tests

**Files:**
- Create: `tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_jobs_worker.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/services/jobs_worker.py`

- [x] Test `_handle_job()` dispatches `job_type=openwebui_attachment_hydration`.
- [x] Test missing `openwebui_data_root` fails non-retryably.
- [x] Test worker calls hydration service and returns summary.
- [x] Test worker revalidates data root even when endpoint accepted the job.
- [x] Test existing `job_type=import` and `job_type=export` tests still pass.
- [x] Test unsupported job type still errors.

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_jobs_worker.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_worker_import_defaults.py \
  -q
```

Expected: FAIL for missing hydration routing.

### Task 6.2: Implement worker routing

**Files:**
- Modify: `tldw_Server_API/app/core/Chatbooks/services/jobs_worker.py`
- Modify: `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
- Modify: `tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_service.py`

- [x] Import `OPENWEBUI_ATTACHMENT_HYDRATION_JOB_TYPE`.
- [x] Add `_handle_openwebui_attachment_hydration(service, payload, job)` or equivalent.
- [x] Normalize user id through existing `_normalize_user_id()`.
- [x] Validate required payload keys.
- [x] Call service method `run_openwebui_attachment_hydration(...)`.
- [x] Return JSON-safe summary:

```python
{
    "referenced_files": result.referenced_files,
    "resolved_files": result.resolved_files,
    "hydrated_images": result.hydrated_images,
    "registered_media_files": result.registered_media_files,
    "already_hydrated": result.already_hydrated,
    "missing_files": result.missing_files,
    "unsupported_files": result.unsupported_files,
    "failed_files": result.failed_files,
    "processed_files": result.processed_files,
    "warnings": result.warnings[:100],
}
```

- [x] Keep import/export routing unchanged.

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_jobs_worker.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_worker_import_defaults.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_adapter.py \
  -q
```

Expected: PASS.

### Task 6.3: Commit worker slice

- [x] Run `git diff --check`.
- [x] Commit:

```bash
git add tldw_Server_API/app/core/Chatbooks/services/jobs_worker.py \
        tldw_Server_API/app/core/Chatbooks/chatbook_service.py \
        tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_service.py \
        tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_jobs_worker.py
git commit -m "Run OpenWebUI hydration jobs"
```

Verification:
- Red run before implementation: `test_openwebui_hydration_jobs_worker.py` failed because hydration jobs still required `chatbooks_job_id`.
- `python -m pytest tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_jobs_worker.py tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_worker_import_defaults.py -q` -> 11 passed.
- `python -m pytest tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_jobs_worker.py tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_worker_import_defaults.py tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_adapter.py -q` -> 13 passed.
- `python -m pytest tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_jobs_worker.py tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_worker_import_defaults.py tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_adapter.py tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_service.py tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_api.py -q` -> 36 passed.
- `python -m pytest tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_api.py tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_jobs_worker.py tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_paths.py tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_service.py tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_db_helpers.py tldw_Server_API/tests/Chatbooks/test_openwebui_db_import_adapter.py tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_worker_import_defaults.py tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_adapter.py -q` -> 61 passed.
- `git diff --check` -> clean.
- `python -m bandit -r tldw_Server_API/app/core/Chatbooks/services/jobs_worker.py tldw_Server_API/app/core/Chatbooks/chatbook_service.py -f json -o /tmp/bandit_openwebui_hydration_worker.json` -> 0 findings, 0 errors.

## Stage 7: Frontend Hydration Workflow

**Goal:** Make the feature discoverable in the existing Chatbooks import area with preview-first controls and opt-in processing.

**Success Criteria:** Authorized users can enter an OpenWebUI data root, preview counts/warnings, enqueue hydration, and see status/result summary. Processing remains off by default.

**Tests:** Vitest client and Chatbooks page tests.

**Status:** Not Started

### Task 7.1: Write failing API client tests

**Files:**
- Modify: `apps/packages/ui/src/services/__tests__/tldw-api-client.chatbooks-openwebui.test.ts`
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/chat-rag.ts`

- [ ] Test `previewOpenWebUIHydration({ openwebui_data_root, scope, process_supported_files })` POSTs JSON to `/chatbooks/openwebui/hydration/preview`.
- [ ] Test `createOpenWebUIHydrationJob(...)` POSTs JSON to `/chatbooks/openwebui/hydration/jobs`.
- [ ] Test `getOpenWebUIHydrationJob(jobId)` GETs `/chatbooks/openwebui/hydration/jobs/{jobId}`.

Run:

```bash
bunx vitest run apps/packages/ui/src/services/__tests__/tldw-api-client.chatbooks-openwebui.test.ts
```

Expected: FAIL because client methods do not exist.

### Task 7.2: Implement frontend client methods

**Files:**
- Modify: `apps/packages/ui/src/services/tldw/TldwApiClient.ts`
- Modify: `apps/packages/ui/src/services/tldw/domains/chat-rag.ts`

- [ ] Add method names:
  - `previewOpenWebUIHydration`
  - `createOpenWebUIHydrationJob`
  - `getOpenWebUIHydrationJob`
- [ ] Use existing `request()`/`bgRequest()` conventions for JSON calls.
- [ ] Do not use upload helpers; V1 sends server-local path text, not a file.

Run:

```bash
bunx vitest run apps/packages/ui/src/services/__tests__/tldw-api-client.chatbooks-openwebui.test.ts
```

Expected: PASS.

### Task 7.3: Write failing Chatbooks UI tests

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx`

- [ ] Test hydration controls are visible near OpenWebUI import when capability/auth allows.
- [ ] Test the data root input is required before preview.
- [ ] Test preview sends `process_supported_files: false` by default.
- [ ] Test toggling processing sends `process_supported_files: true`.
- [ ] Test preview counts render `referenced_files`, `hydrated_images`, `registered_media_files`, and warnings.
- [ ] Test enqueue calls `createOpenWebUIHydrationJob` only after a successful preview.
- [ ] Test non-authorized/capability-missing state hides or disables hydration controls.

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx
```

Expected: FAIL until UI is implemented.

### Task 7.4: Implement Chatbooks hydration UI

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx`

- [ ] Add local state:
  - `hydrationDataRoot`
  - `hydrationProcessingEnabled`
  - `hydrationPreview`
  - `hydrationJob`
  - `hydrationLoading`
  - `hydrationError`
- [ ] Place controls in the existing import panel near OpenWebUI JSON/DB preview areas.
- [ ] Use an input for server-local data root.
- [ ] Use a checkbox/switch for opt-in processing.
- [ ] Show preview counts and warnings in compact rows.
- [ ] Disable job creation until preview succeeds.
- [ ] Surface API errors with existing Ant Design alert/message patterns.
- [ ] Add job id/status link or summary near existing job tracker.

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx
```

Expected: PASS.

### Task 7.5: Commit frontend slice

- [ ] Run `git diff --check`.
- [ ] Commit:

```bash
git add apps/packages/ui/src/services/tldw/TldwApiClient.ts \
        apps/packages/ui/src/services/tldw/domains/chat-rag.ts \
        apps/packages/ui/src/services/__tests__/tldw-api-client.chatbooks-openwebui.test.ts \
        apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx \
        apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx
git commit -m "Add OpenWebUI hydration UI"
```

## Stage 8: Documentation And Final Verification

**Goal:** Document the user workflow and API contract, then run focused verification and security checks.

**Success Criteria:** Users can discover the feature, know the required local data-root shape, understand permissions/allowed roots, and see that processing is opt-in. Backend/frontend targeted tests and Bandit pass.

**Tests:** Docs test plus focused backend/frontend suites.

**Status:** Not Started

### Task 8.1: Update user and API docs

**Files:**
- Modify: `Docs/User_Guides/WebUI_Extension/Chatbook_User_Guide.md`
- Modify: `Docs/API-related/Chatbook_API_Documentation.md`
- Modify: `Docs/API-related/chatbook_openapi.yaml`
- Modify: `Docs/API-related/API_README.md`
- Modify: `Docs/API-related/API_Tags_Index.md`
- Modify: `Docs/Published/User_Guides/WebUI_Extension/Chatbook_User_Guide.md`
- Modify: `Docs/Published/API-related/Chatbook_API_Documentation.md`
- Modify: `Docs/Published/API-related/chatbook_openapi.yaml`
- Modify: `Docs/Published/API-related/API_README.md`
- Modify: `Docs/Published/API-related/API_Tags_Index.md`
- Modify: `tldw_Server_API/tests/Docs/test_chatbook_openwebui_import_docs.py`

- [ ] Document that v1 hydrates referenced files only.
- [ ] Document required local root shape: `webui.db` plus `uploads/`.
- [ ] Document `Files.ingestion_source_allowed_roots` and env var configuration.
- [ ] Document image vs non-image behavior.
- [ ] Document processing opt-in default.
- [ ] Document single-user owner/admin permission requirement.
- [ ] Document common warnings: missing files, unsupported type, oversized, path rejected.
- [ ] Update docs tests to assert hydration feature text is present.

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Docs/test_chatbook_openwebui_import_docs.py -q
```

Expected: PASS.

### Task 8.2: Run focused backend verification

**Files:** no edits unless failures identify necessary fixes.

- [ ] Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_db_helpers.py \
  tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_paths.py \
  tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_service.py \
  tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_api.py \
  tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_jobs_worker.py \
  tldw_Server_API/tests/Chatbooks/test_openwebui_db_import_adapter.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_openwebui_db_api.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_worker_import_defaults.py \
  -q
```

Expected: PASS.

### Task 8.3: Run focused frontend verification

**Files:** no edits unless failures identify necessary fixes.

- [ ] Run:

```bash
bunx vitest run \
  apps/packages/ui/src/services/__tests__/tldw-api-client.chatbooks-openwebui.test.ts \
  apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx
```

Expected: PASS.

### Task 8.4: Run Bandit on touched backend scope

**Files:** no edits unless findings identify new issues.

- [ ] Run:

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/Chatbooks/openwebui_hydration.py \
  tldw_Server_API/app/core/Chatbooks/openwebui_hydration_jobs.py \
  tldw_Server_API/app/core/DB_Management/OpenWebUI_DB.py \
  tldw_Server_API/app/core/DB_Management/chacha/message_store.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/api/v1/endpoints/chatbooks.py \
  tldw_Server_API/app/core/Chatbooks/services/jobs_worker.py \
  -f json -o /tmp/bandit_openwebui_hydration.json
```

Expected: PASS or no new findings in touched hydration code. Fix new findings before continuing.

### Task 8.5: Commit docs and verification slice

- [ ] Run `git diff --check`.
- [ ] Commit:

```bash
git add Docs/User_Guides/WebUI_Extension/Chatbook_User_Guide.md \
        Docs/API-related/Chatbook_API_Documentation.md \
        Docs/API-related/chatbook_openapi.yaml \
        Docs/API-related/API_README.md \
        Docs/API-related/API_Tags_Index.md \
        Docs/Published/User_Guides/WebUI_Extension/Chatbook_User_Guide.md \
        Docs/Published/API-related/Chatbook_API_Documentation.md \
        Docs/Published/API-related/chatbook_openapi.yaml \
        Docs/Published/API-related/API_README.md \
        Docs/Published/API-related/API_Tags_Index.md \
        tldw_Server_API/tests/Docs/test_chatbook_openwebui_import_docs.py
git commit -m "Document OpenWebUI attachment hydration"
```

Omit any published mirror path that does not change in the implementation.

## Final Branch Closeout

- [ ] Run final status:

```bash
git status --short --branch
```

- [ ] Run final focused verification if any implementation task changed after Stage 8:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_db_helpers.py \
  tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_paths.py \
  tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_service.py \
  tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_api.py \
  tldw_Server_API/tests/Chatbooks/test_openwebui_hydration_jobs_worker.py \
  -q
bunx vitest run \
  apps/packages/ui/src/services/__tests__/tldw-api-client.chatbooks-openwebui.test.ts \
  apps/packages/ui/src/components/Option/Chatbooks/__tests__/ChatbooksPlaygroundPage.openwebui-import.test.tsx
```

- [ ] Run final Bandit command from Task 8.4 if backend code changed after the previous Bandit run.
- [ ] Update the Backlog implementation task with verification evidence and final summary.
- [ ] Open or update a PR against `dev`.

## Plan Review Notes

- This plan intentionally keeps hydration separate from import and uses a dedicated Jobs type.
- The largest implementation risk is Media DB registration. Do not start Stage 4 until the exact owner-aware Media DB helper approach is clear from current code.
- The second largest risk is image idempotency. Metadata-only source-key tracking is acceptable for v1 only if tests prove retries do not duplicate rows. Add a source-mapping table if concurrent jobs make that unreliable.
- A plan-review subagent was not dispatched while writing this document because this session requires explicit user permission before spawning delegated agents. Manual review should be treated as the current review pass unless the user explicitly approves subagent review.
