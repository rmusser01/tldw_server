# OpenWebUI Attachment Hydration Design

Date: 2026-05-11
Owner: Codex collaboration session
Status: User-approved design, pending implementation planning
Backlog: TASK-233.14

## Summary

Add a post-import hydration workflow for OpenWebUI attachment and file bytes after OpenWebUI JSON or `webui.db` chat import.

The existing OpenWebUI import paths preserve attachment, file, image, and artifact references as metadata only. This follow-up lets an operator point tldw at a trusted local copy of the OpenWebUI data directory and hydrate only the files referenced by already-imported OpenWebUI chats. Hydration is deliberately separate from chat import so chat migration remains fast, retryable, and safe even when an OpenWebUI `uploads/` tree is large or partially missing.

V1 targets a local server bundle first: an OpenWebUI data root containing `webui.db` and `uploads/`, both under configured allowed roots. Uploaded ZIP bundles and live OpenWebUI API hydration are planned extensions, not v1 behavior.

## Source Context

OpenWebUI documents its SQLite database under the OpenWebUI data directory beside `uploads/` and `vector_db`. Its schema includes `chat`, `chat_file`, and `file` tables. The `file` table stores file identifiers, user ownership, hashes, filenames, paths, data, metadata, and timestamps; the `chat_file` table links files to chats and optionally messages. OpenWebUI plugin docs also describe the common uploaded-file path convention under `/app/backend/data/uploads/{file_id}_{filename}`.

References:

- OpenWebUI database schema: https://docs.openwebui.com/reference/database-schema/
- OpenWebUI file management: https://docs.openwebui.com/features/chat-conversations/data-controls/files/
- OpenWebUI reserved file arguments: https://docs.openwebui.com/features/plugin/development/reserved-args/

The schema and path convention should be treated as versioned source data, not as a permanent API. The hydrator must preview compatibility before copying bytes and must report missing or unsupported file references without failing the entire hydration run.

## Goals

1. Hydrate OpenWebUI file bytes for chats already imported through `source_format=openwebui_json` or `source_format=openwebui_db`.
2. Keep hydration as a separate post-import job with progress, retry, and partial-failure reporting.
3. Support a trusted server-local OpenWebUI data root first.
4. Hydrate only files referenced by imported OpenWebUI chats, not the entire OpenWebUI file library.
5. Restore supported images onto imported chat messages where possible.
6. Register non-image files as tldw Media DB entries and link them from imported message metadata.
7. Make document/media processing opt-in instead of automatic.
8. Deduplicate by OpenWebUI source file id first, then by run-local content hash when the source id is absent.
9. Restrict server-local hydration to the single-user owner in single-user mode and admins in multi-user mode.
10. Constrain all server-local paths to configured allowed roots.

## Non-Goals

1. Hydrate attachments during the existing chat import transaction.
2. Import every OpenWebUI file owned by a selected source user.
3. Accept uploaded ZIP bundles containing `webui.db` and `uploads/` in v1.
4. Connect to a live OpenWebUI server or fetch files through OpenWebUI APIs in v1.
5. Recreate OpenWebUI knowledge bases, channels, permissions, or sharing state.
6. Process, chunk, transcribe, OCR, embed, or analyze files by default.
7. Globally deduplicate user files by content hash across unrelated imports.
8. Read arbitrary server-local paths outside configured allowed roots.

## Requirements Confirmed With User

1. Design both local-bundle and live-API concepts, but implement the local bundle first.
2. Use a hybrid target model: inline images become chat-message attachments; non-image files become Media DB items linked from message metadata.
3. Run hydration as a separate post-import job.
4. Support server-local paths first, with uploaded ZIP bundles as a later extension.
5. Hydrate only files referenced by imported chats.
6. Register files by default; expose processing as an explicit option.
7. Deduplicate by source file id first, then run-local content hash fallback.
8. Allow server-local hydration for the single-user owner or multi-user admins only.

## Current tldw Context

### OpenWebUI Import Metadata

The merged OpenWebUI import paths already normalize message references into `OpenWebUIMessagePlan.attachment_refs`. During import, `ChatbookService._openwebui_message_metadata()` stores them under:

```text
message_metadata.extra.openwebui_import.attachment_refs
```

The same metadata block stores source message ids, parent ids, child ids, role, model, and source-specific metadata. The hydrator should treat this metadata as the source-of-truth for "referenced by imported chats" and should not scan the entire OpenWebUI file table as the default scope.

### Existing Message Image Storage

Chatbook archive import already restores embedded images by passing `images` into `ChaChaNotesDB.add_message()`, which persists image bytes into the `message_images` table. Hydration happens after messages already exist, so implementation will need a focused ChaCha helper for post-insert image attachment, such as appending or replacing hydrated OpenWebUI image slots idempotently.

The helper should preserve existing non-OpenWebUI images and should avoid duplicate image rows on retries.

### Existing Media Storage

The Media DB has `add_media_with_keywords`, file-record helpers such as `insert_media_file`, and lookup helpers such as `get_media_by_hash` and `get_media_by_uuid`. The hydrator should register non-image source files through the Media DB rather than inventing a Chatbooks-private file store.

For v1, "register" means create a durable tldw-side record and source metadata link without running expensive processing. If the user enables processing, hydration can enqueue or call the existing media ingestion path for supported file types as a later implementation stage.

### Local Allowed Roots

The repository already has a configured local-source allowlist via:

```text
Files.ingestion_source_allowed_roots
INGESTION_SOURCE_ALLOWED_ROOTS
TLDW_INGESTION_SOURCE_ALLOWED_ROOTS
```

Hydration should reuse the same allowed-root resolver and path checks. The OpenWebUI data root, `webui.db`, `uploads/`, and every resolved source file path must remain under an allowed root after canonicalization.

## Approaches Considered

### Approach 1: Inline hydration during chat import

OpenWebUI chat import reads source files and writes image/media records before returning success.

Pros:

- one user action
- import result immediately includes hydrated files

Cons:

- slows and destabilizes the already-working chat import path
- makes retrying attachment failures require reimporting chats
- couples file processing and SQLite/file-tree access to conversation creation
- large `uploads/` trees can turn a chat import into a long-running storage job

### Approach 2: Separate post-import hydration job

Chat import remains text/tree/metadata migration. A second job reads imported OpenWebUI metadata, resolves source file rows and bytes, writes image/media records, and updates message metadata with hydration status.

Pros:

- retryable without duplicating conversations
- natural place for progress, partial failures, and per-file warnings
- easier to gate behind admin/single-user-owner permissions
- keeps chat migration fast and predictable

Cons:

- introduces another job/action surface
- users need to provide the OpenWebUI data root after or alongside import

### Approach 3: Full OpenWebUI file library migration

Import every file row for the selected OpenWebUI user, whether referenced by imported chats or not.

Pros:

- broader backup coverage
- can eventually support knowledge base migration

Cons:

- imports unrelated and stale files
- makes dedupe and destination semantics unclear
- expands scope beyond chat migration

## Recommendation

Use Approach 2.

Implement a separate OpenWebUI attachment hydration job that targets only imported-chat references. V1 uses a trusted server-local OpenWebUI data root, restores images as message attachments, registers non-image files in Media DB, and records per-reference hydration status in message metadata. Processing remains opt-in.

## User Workflow

1. User imports OpenWebUI chats through JSON or uploaded `webui.db` using the existing Chatbooks import UI.
2. User chooses "Hydrate OpenWebUI attachments" from the Chatbooks import area or an imported OpenWebUI result/job detail.
3. UI asks for:
   - OpenWebUI data root path, such as `/path/to/openwebui/data`
   - selected source user when the imported scope is DB-backed or ambiguous
   - optional processing flag for supported files
   - optional dry-run/preview first
4. Backend validates permissions and allowed roots.
5. Backend previews hydration:
   - referenced file count
   - resolvable file count
   - image count
   - non-image count
   - missing/ambiguous/oversized/unsupported count
   - estimated bytes
6. User starts the hydration job.
7. Job updates imported message metadata as each reference is hydrated, skipped, missing, or failed.
8. UI shows job status and the final summary. Imported chat messages show restored images where supported, and file references link to registered Media DB entries.

## Backend Design

### New Domain Service

Add a focused service module under Chatbooks, for example:

```text
tldw_Server_API/app/core/Chatbooks/openwebui_hydration.py
```

Responsibilities:

- validate OpenWebUI data root and expected children
- build a referenced-file index from imported message metadata
- read OpenWebUI `file` and `chat_file` rows through DB_Management helpers
- resolve file bytes under `uploads/`
- classify image versus non-image targets
- enforce file size and MIME/type policy
- compute content hashes
- deduplicate within the selected tldw user scope
- write message image rows or Media DB records through existing DB abstractions
- update message metadata with hydration results
- return preview and job result summaries

It should not own auth, FastAPI request parsing, UI state, or low-level raw SQL.

### DB_Management Helpers

Extend `tldw_Server_API/app/core/DB_Management/OpenWebUI_DB.py` so SQL remains in DB_Management. New helpers should be read-only:

```text
load_openwebui_file_rows_for_ids(conn, file_ids, user_id=None)
load_openwebui_chat_file_rows_for_chats(conn, chat_ids, user_id=None)
iter_openwebui_files_for_user(conn, user_id)
```

The hydrator should use file-id lookups derived from imported metadata first. `chat_file` rows can fill gaps when message-level attachment refs are incomplete, but v1 should still keep the scope constrained to imported chat ids.

### Path Resolution

Input is an OpenWebUI data root, not arbitrary paths to individual files.

Resolution rules:

1. Resolve the data root under configured allowed roots.
2. Require a readable `webui.db` under the data root.
3. Require an `uploads/` directory under the data root for actual bytes.
4. For each file row, try safe path candidates in order:
   - `file.path` if it is relative under the data root or uploads root
   - `file.path` if absolute and still under the data root or uploads root
   - `uploads/{file.id}_{file.filename}`
   - future-compatible variants only if covered by tests
5. Reject any candidate that escapes allowed roots or the OpenWebUI data root after canonicalization.
6. Treat symlinks conservatively: canonical target must stay under the data root and allowed roots.
7. Missing files become per-reference warnings, not job-fatal errors.

The preview should expose counts and warnings without returning raw absolute paths to non-admin clients. Admin-facing diagnostics may include sanitized path suffixes when useful.

### Reference Extraction

The hydrator should scan tldw messages with `message_metadata.extra.openwebui_import` for the active tldw user. It should collect:

- tldw conversation id
- tldw message id
- OpenWebUI source chat external ref
- OpenWebUI source message id
- source user id when present
- attachment refs

Supported reference forms in v1:

- dicts with `id`, `file_id`, or `fileId`
- dicts with `name`, `filename`, `mime_type`, `type`, or `content_type` as enrichment
- string ids when the value is clearly a file id

Unsupported refs should remain in metadata with a hydration status of `unsupported_reference_shape`.

For `openwebui_db` imports, source chat ids and selected user metadata are available. For `openwebui_json` imports, file IDs may exist in the JSON refs but source user may be absent; the job can resolve by file id and warn when user ownership cannot be verified.

### Hydration Status Metadata

Update message metadata without replacing the existing import record:

```json
{
  "openwebui_import": {
    "attachment_refs": [{ "...": "..." }],
    "hydration": {
      "version": 1,
      "last_job_id": "job_...",
      "items": [
        {
          "source_file_id": "file-1",
          "source_hash": "sha256:...",
          "source_filename": "notes.pdf",
          "status": "registered_media",
          "kind": "document",
          "media_id": 123,
          "media_file_id": "file-row-or-uuid",
          "message_image_position": null,
          "warnings": []
        }
      ]
    }
  }
}
```

Allowed statuses:

- `hydrated_image`
- `registered_media`
- `already_hydrated`
- `missing_source_file`
- `unsupported_reference_shape`
- `unsupported_file_type`
- `oversized`
- `path_rejected`
- `failed`

The implementation may store a job-level summary separately, but message metadata should be sufficient for chat rendering and future retries.

### Image Hydration

Images should be restored as chat-message images when:

- file MIME/type is image-like or safe sniffing identifies a supported image
- file size is within the existing message-image size limit or a new OpenWebUI hydration-specific image cap
- file path passes all root checks

Implementation needs a ChaCha DB helper for post-insert images, for example:

```text
append_message_image_if_absent(message_id, source_key, image_bytes, mime_type)
```

`source_key` should be derived from OpenWebUI source file id or run-local hash so retries are idempotent. If the current message image table cannot store source keys, add a small metadata-side guard and use deterministic positions carefully. Do not duplicate images on rerun.

### Non-Image File Registration

Non-image files should become Media DB entries by default, without automatic processing.

Recommended source metadata:

```json
{
  "source": "openwebui",
  "source_kind": "attachment",
  "source_file_id": "file-1",
  "source_chat_ref": "chat-1",
  "source_message_id": "msg-1",
  "source_user_id": "user-a",
  "content_hash": "sha256:...",
  "openwebui_filename": "notes.pdf"
}
```

The Media DB record should be discoverable as an imported OpenWebUI attachment and should link back from the message hydration metadata. The exact storage/copy location should follow existing Media DB file storage conventions; do not leave records pointing at the OpenWebUI source path as the durable copy.

If `process_supported_files=true`, the implementation should enqueue or invoke existing ingestion/reprocessing flows for supported file types after registration. Processing failures should not undo registration.

### Deduplication

Deduplication should be conservative:

1. For the same tldw user, if an OpenWebUI `source_file_id` has already been hydrated, reuse the existing image/media link where compatible.
2. If no source file id exists, dedupe by content hash only within the current hydration run.
3. Do not globally merge unrelated files solely because bytes match.
4. Do not reuse media across tldw users in multi-user mode.
5. Preserve each original OpenWebUI reference in message metadata even when the underlying tldw media item is reused.

This avoids storage bloat during retries without collapsing unrelated source records across imports.

## API And Jobs Design

Use Jobs for hydration because this is user-visible, may be long-running, needs progress, and should be retryable.

Suggested endpoint shape:

```text
POST /api/v1/chatbooks/openwebui/hydration/preview
POST /api/v1/chatbooks/openwebui/hydration/jobs
GET  /api/v1/chatbooks/openwebui/hydration/jobs/{job_id}
```

Preview request:

```json
{
  "openwebui_data_root": "/srv/migrations/openwebui/data",
  "scope": {
    "source_user_id": "user-a",
    "conversation_ids": ["optional", "tldw", "conversation", "ids"]
  },
  "process_supported_files": false
}
```

Job request uses the same fields plus explicit confirmation of the preview if the implementation wants preview-first safety.

Result summary:

```json
{
  "referenced_files": 42,
  "resolved_files": 39,
  "hydrated_images": 12,
  "registered_media_files": 24,
  "already_hydrated": 3,
  "missing_files": 2,
  "unsupported_files": 1,
  "failed_files": 0,
  "processed_files": 0,
  "warnings": []
}
```

Job payload should include normalized root tokens or absolute paths only after server-side validation. Job workers must revalidate roots and permissions at execution time, not just at enqueue time.

## Authorization And Security

Server-local hydration reads local filesystem paths, so it requires stronger gates than normal upload import.

Rules:

1. Single-user mode: allow the single-user owner.
2. Multi-user mode: require admin.
3. Always require every path to resolve under `ingestion_source_allowed_roots`.
4. Revalidate paths in the worker.
5. Do not log raw chat content, message content, or full arbitrary source paths.
6. Do not follow symlinks outside the allowed root/data root.
7. Enforce per-file and total-run byte caps.
8. Enforce MIME and file-extension policy before image embedding or Media DB registration.
9. Keep failures per-file unless the data root or DB itself is invalid.
10. Redact source paths from non-admin job status surfaces where practical.

## Frontend Design

Place the workflow in the existing Chatbooks import area, because OpenWebUI chat import already lives there.

Suggested UI behavior:

- Show "Hydrate OpenWebUI attachments" only when the current user is authorized.
- Let the user enter a server-local OpenWebUI data root.
- Explain that the root should contain `webui.db` and `uploads/`.
- Offer a preview action before enqueueing the job.
- Show counts for images, files, missing refs, unsupported refs, and estimated bytes.
- Default `process_supported_files` to off.
- Allow scope narrowing to imported OpenWebUI conversations or selected source user.
- Surface final job results near import history.

Do not add a new global OpenWebUI migration page in v1.

## Error Handling

Fatal preview/job errors:

- unauthorized user
- no allowed roots configured
- data root outside allowed roots
- missing or unreadable `webui.db`
- unsupported OpenWebUI DB schema for file hydration
- missing `uploads/` directory when referenced files need byte hydration

Per-reference warnings:

- missing file row
- missing source bytes
- ambiguous source ref
- unsupported reference shape
- unsupported file type
- oversized file
- rejected path escape
- failed image insert
- failed Media DB registration
- optional processing failed

Hydration should continue after per-reference warnings and summarize them.

## Testing Strategy

Backend tests:

1. Local data root outside allowed roots is rejected.
2. Worker revalidates data root even if enqueue accepted it.
3. OpenWebUI DB file rows resolve only under the data root/uploads root.
4. Path traversal and symlink escape candidates are rejected.
5. Referenced-only extraction ignores unrelated file rows for the same user.
6. Image refs create idempotent message image attachments.
7. Non-image refs create Media DB records and metadata links without processing by default.
8. `process_supported_files=true` triggers the chosen ingestion/processing hook and does not undo registration on processing failure.
9. Source-file-id dedupe reuses existing hydrated records.
10. Run-local hash dedupe works only inside one hydration run when source IDs are absent.
11. Missing source files produce warnings and do not fail the whole job.
12. Multi-user non-admin requests are rejected.

Frontend tests:

1. Authorized users see the hydration action; unauthorized users do not.
2. Preview sends the data root and processing flag.
3. Preview result renders counts and warnings.
4. Job creation is disabled until preview succeeds or required fields are valid.
5. Processing is opt-in and defaults off.

Security checks:

- Bandit on touched backend hydration and DB helper files.
- `git diff --check`.
- Focused pytest for Chatbooks, OpenWebUI DB helpers, ChaCha message images, Media DB registration, and Jobs worker dispatch.

## Documentation Plan

Update user and API docs after implementation:

- explain local OpenWebUI data-root requirements
- explain that v1 hydrates only referenced files
- explain image versus non-image behavior
- document opt-in processing
- document permissions and allowed-root setup
- document missing-file and unsupported-file warnings
- keep live API and ZIP bundle hydration listed as future work

## Follow-Up Work

1. Uploaded ZIP bundle containing `webui.db` and `uploads/`, with strict ZIP bomb/path traversal protection.
2. Live OpenWebUI API hydration with URL/token auth, pagination, and version compatibility checks.
3. Full OpenWebUI file library migration for selected users.
4. Knowledge base migration using OpenWebUI `knowledge` and `knowledge_file` tables.
5. Source-format auto-detection for local bundles after explicit workflows are stable.
6. More granular user controls for which file types to hydrate or process.
