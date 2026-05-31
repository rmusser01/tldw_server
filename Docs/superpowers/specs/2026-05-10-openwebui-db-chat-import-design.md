# OpenWebUI webui.db Chat Import Design

Date: 2026-05-10
Owner: Codex collaboration session
Status: User-approved design, pending implementation planning
Backlog: TASK-233.4

## Summary

Add v2 OpenWebUI import support for an uploaded OpenWebUI `webui.db` SQLite database file through the existing Chatbooks import workflow.

The first OpenWebUI import slice already supports normal `Export Chats` JSON via `source_format=openwebui_json`. This v2 slice extends the same upload, preview, import, quota, and Jobs surface with `source_format=openwebui_db`. Users upload a `webui.db`, preview detected OpenWebUI users, choose one user, and import only that user's chats.

The importer should preserve the same conversation/message guarantees as the JSON path: imported OpenWebUI chats become normal tldw ChaCha conversations, message trees are preserved through `parent_message_id`, duplicates are detected through `source=openwebui` plus stable external references, and source metadata is preserved without logging or duplicating raw chat content.

## Source Format Context

OpenWebUI documents database tables such as `user`, `chat`, and `folder`. Chats include fields such as `id`, `user_id`, `title`, `chat`, `created_at`, `updated_at`, `share_id`, `archived`, `pinned`, `meta`, and `folder_id`; folders include `id`, `parent_id`, `user_id`, `name`, `items`, `meta`, `is_expanded`, `created_at`, and `updated_at`.

Reference: https://docs.openwebui.com/reference/database-schema/

The implementation should treat that schema as the supported starting point, not as an immutable contract. Preview should validate required tables and columns and return actionable shape errors when an uploaded DB comes from an unsupported OpenWebUI version.

If an OpenWebUI version exposes project-like organization outside the documented folder fields, v2 should preserve that organization as namespaced source metadata and warnings unless it has a clear, folder-like destination in existing tldw folder support. Do not invent a parallel tldw project model in this slice.

## Goals

1. Let users upload an OpenWebUI `webui.db` file and import chats from it.
2. Reuse the existing Chatbooks import tab, preview endpoint, import endpoint, async Jobs flow, job history, quotas, and cleanup behavior.
3. Require users to select one detected OpenWebUI user before import.
4. Import only the selected user's chats.
5. Mirror OpenWebUI folder organization into existing visible tldw folder support under an import namespace.
6. Preserve OpenWebUI chat/message/folder/user metadata under namespaced import metadata.
7. Preserve attachment, file, image, and artifact references as metadata and warnings only.
8. Avoid server-local path import, live OpenWebUI connections, and binary attachment hydration in this slice.

## Non-Goals

1. Import from a server-local filesystem path to `webui.db`.
2. Import all OpenWebUI users in one operation.
3. Live migration from a running OpenWebUI server.
4. Import from an OpenWebUI admin export format unless it is the same uploaded SQLite database shape.
5. Hydrate OpenWebUI attachment/file/artifact binaries.
6. Migrate OpenWebUI users, permissions, sharing state, or provider settings as first-class tldw objects.
7. Add a separate OpenWebUI import endpoint or a parallel folder schema.
8. Auto-detect source format in v2; users should explicitly choose `OpenWebUI database`.

## Requirements Confirmed With User

1. v2 accepts an uploaded OpenWebUI `webui.db` SQLite file only.
2. v2 does not support server-local path import.
3. Preview should require selecting one OpenWebUI user before import.
4. Import should mirror OpenWebUI folder/project organization into tldw-side organization when a natural destination exists.
5. tldw's existing visible folder support is the target organization surface.
6. OpenWebUI folders should be namespaced under `OpenWebUI / <selected user label> / ...`.
7. Attachment/file/artifact references should be preserved as metadata only; binary hydration is future work.

## Current tldw Context

### OpenWebUI JSON Import

The merged v1 importer already provides:

- `source_format=chatbook|openwebui_json`
- `OpenWebUIConversationPlan` and `OpenWebUIMessagePlan`
- JSON preview counts and warnings
- sync import through `ChatbookService.import_openwebui_json`
- async dispatch through `tldw_Server_API/app/core/Chatbooks/services/jobs_worker.py`
- duplicate detection through `source=openwebui` and `external_ref`
- message parent preservation with deterministic UUIDv5 IDs
- namespaced conversation and message metadata
- WebUI source selector and OpenWebUI preview summary

This v2 design should extend that importer family rather than fork a new import system.

### Chatbooks

Chatbooks already owns:

- per-user temp/import/export storage
- upload validation and preview
- sync and async import
- import jobs stored in the user's ChaCha DB
- core Jobs integration
- conflict strategy support for `skip` and `rename`
- the existing `/chatbooks` WebUI surface

The OpenWebUI DB importer should use this surface with a new explicit source format.

### tldw Folders

tldw already exposes chat folders in the WebUI. The current WebUI folder store models folders and conversation membership through existing keywords and keyword collections:

- folder records map to `keyword_collections`
- folder-keyword links map to `collection_keywords`
- conversation-folder membership is represented by `conversation_keywords` through a keyword associated with the folder

The importer should target that existing visible folder system through DB/service helpers. It should not create a new conversation-folder schema.

## Approaches Considered

### Approach 1: Extend Chatbooks with `source_format=openwebui_db`

Upload `webui.db` through the existing Chatbooks preview/import endpoints. Preview validates the DB and returns detected users and counts. Import requires `selected_openwebui_user_id`, extracts that user's chats/folders, normalizes them into the existing OpenWebUI conversation/message plans, and imports through the same service family.

Pros:

- reuses existing upload, preview, quotas, Jobs, cleanup, and WebUI discovery
- keeps chat migration sources in one import surface
- allows direct reuse of v1 OpenWebUI message-tree import logic
- keeps `webui.db` import explicit without pretending it is just JSON import

Cons:

- Chatbooks gains another external source-format branch
- preview and import need DB-specific response/request fields

### Approach 2: Dedicated OpenWebUI DB import endpoint

Add a separate endpoint such as `/chat/import/openwebui-db`.

Pros:

- one endpoint can be tailored to the database source

Cons:

- duplicates upload, preview, conflict, async job, cleanup, and status behavior
- gives users another import page to discover
- increases long-term maintenance for future import sources

### Approach 3: Convert DB rows to temporary OpenWebUI JSON

Extract selected DB rows into a temporary OpenWebUI JSON export and hand that to the v1 importer.

Pros:

- quickest reuse of the current JSON parser

Cons:

- hides DB-specific validation and user/folder selection behind a fake intermediate format
- makes preview semantics awkward
- loses a clean place to report DB table/column compatibility errors
- makes folder mirroring feel bolted on

## Recommendation

Use Approach 1.

Add a focused OpenWebUI DB adapter beside the existing OpenWebUI JSON adapter. The DB adapter owns uploaded SQLite inspection and extraction. It should produce the same normalized conversation/message plan type used by the JSON importer so import behavior stays consistent.

## Backend Design

### Modules

Extend the import adapter package:

```text
tldw_Server_API/app/core/Chatbooks/import_adapters/
  openwebui.py
  openwebui_db.py
```

`openwebui_db.py` should own:

- read-only SQLite opening
- OpenWebUI schema detection
- user listing and aggregate preview
- selected-user chat/folder extraction
- conversion into normalized OpenWebUI conversation/message plans
- DB-specific warning generation

It should not own endpoint concerns, auth, quota enforcement, temp-file placement, Jobs state, or final ChaCha writes.

Recommended internal types:

- `OpenWebUIDatabasePreview`
- `OpenWebUIDatabaseUserPreview`
- `OpenWebUIDatabaseFolderPlan`
- `OpenWebUIDatabaseExtractionResult`

The existing `OpenWebUIConversationPlan`, `OpenWebUIMessagePlan`, and import result shapes should be reused where practical.

### API Shape

Extend source format values:

```text
source_format = chatbook | openwebui_json | openwebui_db
```

Preview:

```text
POST /api/v1/chatbooks/preview
multipart/form-data:
  file: uploaded .db file
  source_format: openwebui_db
```

Import:

```text
POST /api/v1/chatbooks/import
multipart/form-data:
  file: uploaded .db file
  source_format: openwebui_db
  selected_openwebui_user_id: <OpenWebUI user id from preview>
  conflict_resolution: skip | rename
  prefix_imported: bool
  async_mode: bool
```

`selected_openwebui_user_id` is required for `source_format=openwebui_db` import and ignored for Chatbook archive and JSON import. Preview must not import implicitly when only one user exists; the UI may preselect the only user, but the import request should still send the selected user explicitly.

Response extensions:

```text
PreviewChatbookResponse
  source_format: chatbook | openwebui_json | openwebui_db
  openwebui_db_preview: OpenWebUIDatabasePreview | null

ImportChatbookResponse
  source_format: chatbook | openwebui_json | openwebui_db
  openwebui_result: OpenWebUIImportResult | null
  openwebui_db_result: OpenWebUIDatabaseImportResult | null
```

The import result can wrap the existing `openwebui_result` plus DB-specific metadata such as selected user id, mirrored folder count, folder-link count, and DB compatibility warnings.

### Preview Flow

For `source_format=openwebui_db`:

1. Save the uploaded DB to the per-user Chatbooks temp area.
2. Enforce upload size, filename, and path traversal checks.
3. Require a safe `.db` or `.sqlite` extension.
4. Validate SQLite magic bytes before opening.
5. Open the DB read-only.
6. Disable extension loading.
7. Validate required OpenWebUI tables and columns.
8. List users from the OpenWebUI `user` table.
9. Aggregate chat, message, folder, and attachment-reference counts per user.
10. Return preview rows for user selection.
11. Clean up the temp file according to existing preview cleanup behavior.

Preview user rows should include:

- OpenWebUI user id
- user name/email display label where available
- chat count
- folder count
- message count
- branched chat count when feasible
- attachment/file/artifact reference count
- archived/pinned counts when feasible
- warning count

Do not return raw chat or message content in preview.

### Import Flow

For `source_format=openwebui_db`:

1. Validate `selected_openwebui_user_id`.
2. Resolve and open the uploaded DB read-only from the allowed import/temp roots.
3. Validate the DB shape again; do not trust a previous preview response.
4. Extract only chats where `chat.user_id` equals the selected OpenWebUI user id.
5. Extract that user's folder tree and per-chat `folder_id`.
6. Convert selected chats into normalized `OpenWebUIConversationPlan` objects.
7. Reuse the existing OpenWebUI import path for conversation creation, message ordering, duplicate handling, timestamp normalization, and metadata writes.
8. Mirror folder paths under the import namespace after each conversation is successfully created.
9. Return aggregate counts and warnings.

If the selected user id is missing or no longer present in the uploaded DB, return a user-facing validation error. If a selected user's DB contains malformed chats, skip at chat granularity and continue where possible, matching v1 behavior.

### Async Jobs

Async import jobs should include:

```json
{
  "source_format": "openwebui_db",
  "selected_openwebui_user_id": "...",
  "conflict_resolution": "skip"
}
```

The Jobs worker must branch on `source_format=openwebui_db` before archive resolution and before JSON-specific handling. It should resolve the uploaded file from the same allowed roots, open it read-only, and dispatch to the DB import service path.

Temp-file ownership should follow the existing Chatbooks import model: preview cleans up preview uploads, while async import leaves the uploaded DB for the worker to clean up on success, failure, enqueue failure, or cancellation paths where the existing worker owns cleanup.

## Data Mapping

### User Selection

The selected OpenWebUI user is not migrated as a tldw user. It is an import source selector and metadata source.

Store selected source user metadata under conversation settings:

```text
conversation_settings.settings_json.openwebui_import = {
  "source": "openwebui",
  "source_kind": "openwebui_db",
  "source_user_id": "...",
  "source_user_label": "...",
  ...
}
```

The UI should label the selected user with the safest available display value:

1. name
2. email local part or email if existing UI patterns allow it
3. OpenWebUI user id

Logs and warnings should avoid user emails unless they are already visible in the authenticated preview response.

Before using the selected user label in a tldw folder namespace, normalize it to a safe folder segment. If two source users would produce the same label, or if the label is empty after normalization, append a stable short source-user-id suffix so repeated imports remain deterministic and do not collide.

### Conversation Mapping

Each selected OpenWebUI chat becomes one tldw conversation.

Use the existing v1 mapping where possible:

| OpenWebUI DB field | tldw field |
| --- | --- |
| `chat.title` | `conversations.title` |
| `chat.id` | `conversations.external_ref` |
| source format | `conversations.source = "openwebui"` |
| selected tldw user | `conversations.client_id` |
| fallback assistant | existing fallback character/assistant identity |
| `created_at` / `updated_at` | `openwebui_import` metadata |
| `folder_id`, `pinned`, `archived`, `share_id`, `meta` | `openwebui_import` metadata and warnings where relevant |

External refs should remain compatible with v1 JSON import when the same OpenWebUI chat id is available. That means a DB import of a chat already imported from JSON should be recognized as duplicate when both sources expose the same chat id.

If the DB lacks a usable chat id, derive a stable ref with a DB-specific canonical hash:

```text
openwebui_db:<source_user_id>:<row-index-or-rowid>:<sha256(canonical_chat_payload)[0:16]>
```

### Message Mapping

OpenWebUI DB chat content should be normalized to the same message plan shape used by JSON import.

The DB adapter should parse the `chat.chat` payload defensively:

- support the same `history.messages` tree as JSON export
- preserve all valid branches, not just the current path
- normalize `user` and `assistant` messages
- skip unsupported roles with warnings
- use existing deterministic UUIDv5 message ID generation
- preserve source message IDs, parent IDs, children IDs, model/context/done fields, and unsupported keys in message metadata

### Folder Mapping

Mirror OpenWebUI folders into tldw visible folders.

Import namespace:

```text
OpenWebUI / <selected user label> / <OpenWebUI folder path>
```

Chats without a source folder should be linked under:

```text
OpenWebUI / <selected user label> / Unfiled
```

The backend representation should use existing tldw folder support, which currently presents chat folders through keyword collections and conversation-keyword links. The implementation should add or reuse a small service helper that can:

1. ensure a folder path exists under `keyword_collections`
2. ensure a folder keyword exists and is linked to that collection
3. link the imported conversation to that folder keyword
4. avoid duplicate links on repeated imports

Do not write ad hoc SQL in the adapter. Route through ChaCha DB abstractions or a focused helper on the same store layer.

Use `chat.folder_id` as the authoritative chat-to-folder membership source. Treat `folder.items` as secondary, denormalized evidence for preview counts or compatibility warnings, because it can drift from the normalized chat rows.

Folder path collisions:

- Reuse an existing folder path inside the same `OpenWebUI / <user>` namespace.
- Do not merge with same-named folders outside that namespace.
- If the OpenWebUI folder tree contains cycles, missing parents, duplicate sibling names, or invalid names, import valid folders where possible and place affected chats under `Unfiled` with warnings.

Folder segment names should be sanitized for tldw folder creation while preserving the original OpenWebUI names in metadata.

Preserve original folder metadata in conversation settings:

```text
openwebui_import.folder = {
  "source_folder_id": "...",
  "source_parent_id": "...",
  "source_path": ["Parent", "Child"],
  "source_meta": {...}
}
```

If project-like grouping metadata exists in the source chat or folder metadata and cannot be represented as a tldw folder path, preserve it under `openwebui_import.project` or a similarly namespaced metadata key and report a non-blocking warning.

### Attachments And Unsupported Data

V2 preserves attachment, image, file, and artifact references as metadata only.

The importer should:

- count references in preview
- preserve reference IDs/names/types/URLs where present and safe
- store references under existing message metadata
- warn that binaries were not hydrated
- avoid reading OpenWebUI storage paths or external object storage
- avoid creating broken tldw attachments

Binary hydration needs a future design that includes source bundle availability, file-root trust boundaries, storage quota handling, and UI expectations.

## Frontend Design

Update the existing Chatbooks import tab.

The source selector becomes:

- `Chatbook archive`
- `OpenWebUI JSON`
- `OpenWebUI database`

When `OpenWebUI database` is selected:

- accept `.db` and `.sqlite`
- send `source_format=openwebui_db`
- hide archive content selection
- hide or disable media and embedding import controls
- limit conflict choices to `skip` and `rename`
- preview users and counts
- require a selected OpenWebUI user before import
- show folder mirroring destination, e.g. `OpenWebUI / Alice`
- warn that attachment binaries are not imported

The preview panel should be optimized for selection:

- user list/table
- chat/folder/message counts
- warning count
- selected-user import summary

The import result should show imported chats/messages, skipped duplicates, failed chats, mirrored folders, folder links, and warnings.

## Error Handling

Hard failures:

- unsafe filename or path traversal
- unsupported extension for `openwebui_db`
- file is not SQLite
- SQLite DB cannot be opened read-only
- required OpenWebUI tables/columns are missing
- no OpenWebUI users found
- import request lacks `selected_openwebui_user_id`
- selected user does not exist in the uploaded DB
- no chats found for selected user
- no fallback tldw assistant/character identity can be resolved

Recoverable warnings:

- individual malformed chat skipped
- malformed message skipped
- unsupported role skipped
- folder cycle or missing parent recovered to `Unfiled`
- duplicate chat skipped
- attachment/file/artifact reference preserved without binary hydration
- unsupported OpenWebUI table/column ignored
- archived/shared/pinned state preserved as metadata only

Errors should identify the source format and the validation category without echoing chat content, message content, full file paths, or private source data.

## Security And Privacy

The uploaded `webui.db` is a private chat database and must be handled more conservatively than a normal JSON export.

Required controls:

- upload-only input; no server-local path support
- per-user temp/import storage only
- existing quota and upload limits
- safe filename validation
- SQLite magic-byte check before opening
- read-only SQLite URI mode
- no extension loading
- no SQL generated from uploaded DB values
- parameterized queries for selected user id and row lookups
- required table/column allowlist validation
- sanitized errors and logs
- preview/import cleanup on success, failure, and enqueue failure
- no raw chat/message content in logs, audit entries, warnings, or job errors
- no attachment binary hydration

If available in the local SQLite helper stack, implementation should set conservative read limits, busy timeouts, and row/page-count guardrails. If those are not available, tests should still cover large-count preview behavior with bounded queries.

## Testing Plan

Backend adapter tests:

1. Preview rejects non-SQLite files.
2. Preview rejects missing required OpenWebUI tables/columns.
3. Preview lists multiple users with per-user counts.
4. Preview counts folders, chats, messages, branched chats, archived/pinned chats, and attachment references.
5. Extraction imports only the selected user.
6. Extraction converts DB chats into the existing OpenWebUI conversation/message plan.
7. Folder cycles and missing parents produce warnings and route affected chats to `Unfiled`.

Chatbooks endpoint/service tests:

1. `source_format=openwebui_db` accepts `.db` uploads without archive validation.
2. Import requires `selected_openwebui_user_id`.
3. Sync import returns DB-specific preview/import result fields.
4. Async job payload includes `source_format=openwebui_db` and selected user id.
5. Worker dispatches DB imports without calling archive or JSON handlers.
6. Cleanup behavior covers invalid DB, failed enqueue, sync import, and async worker completion.

ChaCha folder tests:

1. Helper creates nested folder namespace.
2. Helper reuses existing namespace paths on repeated imports.
3. Helper links conversations to folders through existing folder support.
4. Helper does not merge with same-named folders outside `OpenWebUI / <user>`.

Frontend tests:

1. Source selector includes OpenWebUI database mode.
2. DB mode accepts `.db`/`.sqlite`.
3. Preview sends `source_format=openwebui_db`.
4. Preview renders user selection rows.
5. Import requires a selected user and sends `selected_openwebui_user_id`.
6. Unsupported controls are hidden or disabled.
7. Result renders folder and attachment-reference warnings.

Verification:

- focused backend tests for DB adapter and service dispatch
- focused ChaCha folder helper tests
- focused frontend component/service tests
- Bandit on touched Python source
- `git diff --check`

## Documentation Plan

Update user and API docs after implementation:

- document OpenWebUI database upload as v2 import source
- explain selected-user import behavior
- explain folder namespace mirroring
- explain duplicate skip/rename behavior
- state that attachment binaries are not imported
- keep JSON export and DB import behavior distinct

## Follow-Up Work

1. Attachment/file hydration when source file bundles or trusted storage roots are available.
2. Direct admin export support if OpenWebUI ships a documented admin export shape distinct from `webui.db`.
3. Live OpenWebUI server import after auth, pagination, and API stability are understood.
4. Optional import of OpenWebUI folder sharing/permissions if tldw gains a matching concept.
5. Optional source-format auto-detection after explicit format support is stable.
