# OpenWebUI Chat Import Design

Date: 2026-05-10
Owner: Codex collaboration session
Status: User-approved design, pending implementation planning
Backlog: TASK-233

## Summary

Add first-class import support for OpenWebUI chat JSON exports through the existing Chatbooks import workflow.

The first version accepts the normal OpenWebUI `Export Chats` JSON file, previews the contents, and imports each OpenWebUI chat into tldw as a normal ChaCha conversation. It preserves the OpenWebUI message tree by mapping message parent links onto tldw `parent_message_id` values.

Direct import from an OpenWebUI `webui.db`, admin database export, or live OpenWebUI instance is a planned follow-up and is explicitly out of scope for v1.

## Source Format Context

OpenWebUI's current import/export documentation describes chat exports as JSON arrays of chat objects. The standard object shape contains a `chat` object with `title`, `models`, and `history`; `history.messages` is a map of message ID to message object, and messages form a tree through `parentId` and `childrenIds`. OpenWebUI also documents a legacy format where each array item is treated directly as the chat data.

Reference: https://docs.openwebui.com/features/chat-conversations/data-controls/import-export/

## Goals

1. Let users import all chats from an OpenWebUI exported JSON file.
2. Reuse the existing Chatbooks import tab, preview behavior, quotas, async jobs, and job history.
3. Preserve OpenWebUI branch structure using tldw message `parent_message_id`.
4. Make repeated imports safe by detecting duplicates through `source=openwebui` and `external_ref=<openwebui_chat_id>`.
5. Preserve useful OpenWebUI metadata where tldw has an existing safe place for it.
6. Surface unsupported data, especially attachments/artifacts, as preview/import warnings instead of silently pretending it was imported.
7. Keep the implementation bounded to JSON export import only.

## Non-Goals

1. Direct import from OpenWebUI `webui.db`.
2. Direct import from an OpenWebUI admin database export.
3. Live migration from a running OpenWebUI server.
4. Hydrating OpenWebUI attachments, files, artifacts, or storage-provider objects.
5. Migrating OpenWebUI users, permissions, sharing state, or folders as first-class tldw objects.
6. Recreating OpenWebUI model/provider configuration in tldw provider settings.
7. Importing ChatGPT exports directly unless they already conform to the OpenWebUI JSON shape accepted by the v1 parser.

## Requirements Confirmed With User

1. Support OpenWebUI chat JSON export first.
2. Treat direct database/admin export import as planned future work.
3. Preserve the full branched message tree.
4. Start with message text, tree structure, and metadata; do not hydrate attachments in v1.
5. Preserve unsupported attachment/artifact references as metadata or warnings where possible.
6. Default duplicate behavior is skip by original OpenWebUI chat identity.
7. Allow rename/import-copy behavior for users who intentionally want another imported copy.
8. Put the feature in the existing Chatbooks import surface.

## Current tldw Context

### Chatbooks

Chatbooks already owns portable import/export workflows:

- `tldw_Server_API/app/api/v1/endpoints/chatbooks.py`
- `tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py`
- `tldw_Server_API/app/core/Chatbooks/chatbook_service.py`
- `tldw_Server_API/app/core/Chatbooks/chatbook_validators.py`
- `tldw_Server_API/app/core/Chatbooks/services/jobs_worker.py`

The module already provides:

- per-user temp/import/export storage
- upload validation and preview
- sync and async import
- import jobs stored in the user's ChaCha DB
- core Jobs integration
- conflict strategy support for `skip` and `rename`
- existing WebUI route at `/chatbooks`

The OpenWebUI importer should extend this portability surface instead of creating another import page.

### ChaCha Conversations

tldw conversations and messages already support the key target fields:

- conversations have `source` and `external_ref`
- messages have stable IDs and `parent_message_id`
- conversation tree APIs already reconstruct trees from parent links
- message metadata storage exists for additional details such as model, source IDs, and unsupported references

This makes full OpenWebUI branch preservation feasible without changing the core chat model.

## Approaches Considered

### Approach 1: Chatbooks import adapter

Add an OpenWebUI JSON adapter under the Chatbooks import workflow.

Pros:

- reuses the current user-facing import page
- reuses preview, quotas, async jobs, and job history
- keeps all chat portability workflows together
- avoids a second import framework

Cons:

- Chatbooks service gains an external-format adapter path

### Approach 2: Dedicated chat import endpoint

Create a separate `/chat/import/openwebui` endpoint and a dedicated UI surface.

Pros:

- clear API boundary for one external source format

Cons:

- duplicates preview, conflict, upload, async job, and status behavior
- creates another surface users must discover
- makes follow-up external chat import formats harder to organize consistently

### Approach 3: Convert JSON into a temporary Chatbook archive

Convert OpenWebUI JSON into a generated Chatbook ZIP and then run the existing Chatbook importer.

Pros:

- maximum reuse of the existing Chatbook archive importer

Cons:

- adds an internal archive generation step
- makes error reporting indirect
- fights current Chatbook archive assumptions where imported conversations are native Chatbook content rather than source-specific migration records

## Recommendation

Use Approach 1.

Add a focused OpenWebUI adapter to Chatbooks import. The adapter validates the JSON export, builds a preview summary, normalizes each OpenWebUI chat into an import plan, and writes tldw conversations/messages through existing ChaCha DB methods.

## Backend Design

### Modules

Add a small import adapter package:

```text
tldw_Server_API/app/core/Chatbooks/import_adapters/
  __init__.py
  openwebui.py
```

The adapter owns source-format parsing and normalization only. It should not own endpoint concerns, quota logic, temp-file storage, or Jobs integration.

Recommended internal types:

- `OpenWebUIImportPreview`
- `OpenWebUIPreviewChatItem`
- `OpenWebUIConversationPlan`
- `OpenWebUIMessagePlan`
- `OpenWebUIImportResult`
- `OpenWebUIImportWarning`

These can be dataclasses or Pydantic models depending on local service style. They should be easy to unit test without FastAPI.

### API Shape

Extend Chatbooks preview/import requests with a source format selector:

```text
source_format = chatbook | openwebui_json
```

The default stays `chatbook` to preserve existing behavior. The UI should send `openwebui_json` explicitly when the user selects OpenWebUI JSON.

Because the existing Chatbooks preview/import routes are multipart upload endpoints, `source_format` should be passed as a multipart form field alongside `file` and the existing import options.

The backend may later add `auto`, but v1 should not depend on fragile inference. A clear source selector gives better errors and avoids accidental JSON uploads being interpreted as another format.

OpenWebUI preview/import responses should be explicit extensions of the existing Chatbooks response models rather than overloading the Chatbook manifest. Add optional source-specific fields while preserving the current Chatbook fields for archive imports:

```text
PreviewChatbookResponse
  source_format: chatbook | openwebui_json
  manifest: existing Chatbook manifest summary, null for OpenWebUI JSON
  openwebui_preview: OpenWebUIImportPreview | null

ImportChatbookResponse
  source_format: chatbook | openwebui_json
  success/message/job_id: existing fields
  openwebui_result: OpenWebUIImportResult | null for sync imports and completed job details when available
```

`OpenWebUIImportPreview` should include:

- `chat_count`
- `message_count`
- `branched_chat_count`
- `duplicate_chat_count`
- `attachment_reference_count`
- `malformed_chat_count`
- `warnings`
- `items`: lightweight per-chat summaries with source chat ref, title, message count, branch flag, duplicate flag, and warning count

`OpenWebUIImportResult` should include:

- `imported_chats`
- `skipped_chats`
- `failed_chats`
- `imported_messages`
- `skipped_messages`
- `duplicate_chats`
- `warnings`

### Preview Flow

For `source_format=chatbook`, keep the existing preview path unchanged.

For `source_format=openwebui_json`:

1. Save the uploaded `.json` to the same per-user temp area used by Chatbooks.
2. Enforce the existing upload size and filename safety rules adapted for JSON.
3. Parse JSON as UTF-8.
4. Require a top-level array.
5. Accept both OpenWebUI standard wrapper objects and documented legacy chat objects.
6. Validate enough of each chat to count and preview it.
7. Query existing conversations for duplicate `source=openwebui` and `external_ref`.
8. Return a preview summary with counts and warnings.
9. Clean up temp files according to the existing preview cleanup behavior.

Preview should report:

- source format
- chat count
- message count
- branched chat count
- duplicate chat count
- attachment/artifact reference count
- malformed chat count
- skipped/unsupported field warnings
- per-chat item summaries when practical

### Import Flow

For `source_format=chatbook`, keep the existing import path unchanged.

For `source_format=openwebui_json`:

1. Validate and parse the file with the OpenWebUI adapter.
2. Resolve the default/fallback assistant identity required for current tldw conversations.
3. For each valid OpenWebUI chat:
   - derive an original chat external reference
   - check for an existing non-deleted tldw conversation with `source=openwebui` and matching `external_ref`
   - apply conflict strategy
   - create one tldw conversation
   - insert all valid message nodes with deterministic mapped IDs
   - preserve parent links after mapping OpenWebUI IDs to tldw IDs
   - record warnings for unsupported or malformed data
4. Update import job counts and warnings.

Partial import is allowed at the chat level. One malformed chat should not fail an entire large export. A malformed message inside an otherwise valid chat should be skipped with a warning unless it prevents safe tree preservation for dependent child messages.

### Async Jobs

OpenWebUI imports should use the same Chatbooks import job infrastructure:

- user-scoped import job row in ChaCha
- core Jobs payload for async mode
- status transitions matching existing Chatbooks import jobs
- cancellation before job start and best-effort in-flight behavior consistent with existing workers

The Jobs payload should include `source_format=openwebui_json` so the worker dispatches to the correct import path.

## Data Mapping

### Conversation Mapping

Each OpenWebUI chat becomes one tldw conversation.

Map fields as follows:

| OpenWebUI field | tldw field |
| --- | --- |
| chat title | `conversations.title` |
| original chat ID | `conversations.external_ref` |
| source format | `conversations.source = "openwebui"` |
| current user | `conversations.client_id` |
| fallback assistant | `conversations.character_id` or assistant identity required by current DB rules |
| created timestamp | conversation metadata if native created timestamp cannot be preserved by current API |
| updated timestamp | conversation metadata if native last-modified timestamp cannot be preserved by current API |
| models/options/meta/pinned/folder_id | message/conversation metadata or import warnings |

If the OpenWebUI export lacks a clear chat ID, derive a stable external ref from the chat index plus a hash of the normalized chat object. This keeps duplicate detection deterministic for the same file.

Derived refs should be built from a canonical JSON representation:

```text
openwebui:<index>:<sha256(canonical_chat_json)[0:16]>
```

Canonical JSON means sorted keys, compact separators, and omission of import-only transient fields. This is enough for deterministic duplicate detection across repeated imports of the same exported file without claiming cross-file identity for chats that have no source ID.

Conversation titles should use this fallback order:

1. OpenWebUI chat title
2. first user message excerpt
3. `OpenWebUI Import <YYYY-MM-DD>`

### Message Mapping

Each OpenWebUI message object becomes one tldw message.

Map fields as follows:

| OpenWebUI field | tldw field |
| --- | --- |
| message ID | deterministic tldw message ID derived from the import namespace plus source message ID |
| `parentId` | `messages.parent_message_id` after deterministic ID mapping |
| `role` | `messages.sender` after role normalization |
| `content` | `messages.content` |
| `timestamp` | `messages.timestamp` |
| `model`, `done`, `context`, source IDs, raw unsupported refs | message metadata |

Role handling:

- `user` -> `user`
- `assistant` -> `assistant`
- `system` and `tool` are not part of the documented OpenWebUI import roles for this JSON path and should be treated as unsupported in v1 unless implementation finds real OpenWebUI exports that require them and verifies downstream support
- unknown or unsupported roles should be skipped with a warning

Message ordering should be deterministic. The importer should insert parent messages before child messages. If the export contains cycles, missing parents, or invalid child links, the affected messages should be skipped with warnings rather than creating corrupted parent references.

The import namespace prevents message ID collisions:

- For the default `skip` path, namespace message IDs with the canonical OpenWebUI external ref.
- For `rename`, generate a new import-copy namespace after the new tldw conversation ID or a generated import-copy UUID is known.
- Store the original OpenWebUI message ID in metadata for both cases.
- Never reuse the same deterministic tldw message IDs for a duplicate-copy import.

### Branch Preservation

OpenWebUI branches must be preserved by importing every valid message node, not just the active `history.currentId` path.

The importer should:

1. Read all messages from `history.messages`.
2. Build an ID map from OpenWebUI message IDs to deterministic tldw message IDs.
3. Validate parent references.
4. Topologically insert root messages before descendants.
5. Store every valid branch through `parent_message_id`.

`history.currentId` should be preserved as metadata for future UI highlighting, but it should not cause non-current branches to be dropped.

## Duplicate And Conflict Handling

Default duplicate key:

```text
source = openwebui
external_ref = <original OpenWebUI chat ID or derived stable ref>
```

Conflict strategies:

- `skip`: default; skip the duplicate chat and count it as skipped.
- `rename`: import another copy with a unique title and a modified external ref suffix so it does not collide with the original import identity.

Existing unsupported strategies such as `overwrite` and `merge` should not be exposed for OpenWebUI v1 unless they are already fully supported by the endpoint and service. The UI should avoid offering unsupported choices for this source format.

For `rename`, the imported conversation should receive:

- a unique title
- an external ref such as `<original-ref>#copy:<new-conversation-id-or-import-copy-id>`
- message IDs namespaced by that copy ref

This preserves duplicate detection for the original while allowing intentional copies without primary-key collisions.

## Attachments And Unsupported Data

V1 does not hydrate OpenWebUI attachments or artifact files.

If message objects contain file, image, artifact, or attachment references:

- preserve the original references in metadata where existing storage makes that safe
- increment unsupported attachment/artifact counts
- show warnings in preview and import results
- do not create broken tldw attachments

This is intentionally conservative because OpenWebUI deployments may store files differently depending on storage backend and export path.

## Frontend Design

Update the existing Chatbooks page at `/chatbooks`.

The Import tab gets a source selector:

- `Chatbook archive`
- `OpenWebUI JSON`

When `Chatbook archive` is selected:

- preserve current behavior
- accept `.zip` and `.chatbook` as supported by backend behavior
- show current Chatbook manifest preview

When `OpenWebUI JSON` is selected:

- accept `.json`
- call Chatbooks preview with `source_format=openwebui_json`
- render an OpenWebUI-specific preview summary
- disable unsupported options such as media/embedding import
- show duplicate and unsupported attachment warnings
- send `source_format=openwebui_json` on import

The existing Jobs tab should continue to list import jobs. OpenWebUI jobs can be distinguished through job metadata if the API exposes it; otherwise they may appear as normal import jobs in v1.

## Error Handling

Hard failures:

- invalid filename or unsafe path
- non-JSON file for `openwebui_json`
- malformed JSON
- top-level value is not an array
- file exceeds configured upload limits
- no valid OpenWebUI chat objects found
- no fallback assistant identity can be resolved for conversation creation

Recoverable warnings:

- individual malformed chat skipped
- malformed message skipped
- missing title fallback used
- duplicate chat skipped
- unsupported attachment/artifact reference preserved only as metadata or warning
- unknown optional OpenWebUI field ignored
- parent references invalid for a subset of messages

Error messages should identify the source format and avoid logging raw message content.

## Security And Privacy

The importer handles private chat history, so it must follow the same posture as Chatbooks:

- use per-user temp storage
- keep import jobs scoped to the authenticated user
- validate filenames and paths
- reject path traversal
- avoid raw chat content in logs
- apply existing quotas and rate limits
- avoid writing source JSON outside the per-user import/temp area

No external network calls are required for v1.

## Testing Plan

Backend tests:

1. Parser accepts standard OpenWebUI wrapper exports.
2. Parser accepts documented legacy chat objects.
3. Parser rejects malformed JSON and non-array roots.
4. Preview counts chats, messages, branches, duplicates, and unsupported attachments.
5. Import preserves message trees through `parent_message_id`.
6. Import inserts parent messages before children.
7. Import handles missing titles with deterministic fallbacks.
8. Import skips duplicate conversations by `source=openwebui` and `external_ref`.
9. Import supports rename/import-copy behavior for duplicates.
10. Import records warnings for unsupported attachments/artifacts.
11. Async job payload dispatches to the OpenWebUI importer.
12. Existing Chatbook archive import/preview tests keep passing unchanged.

Frontend tests:

1. Import source selector switches between Chatbook archive and OpenWebUI JSON.
2. OpenWebUI mode accepts `.json` and sends `source_format=openwebui_json`.
3. OpenWebUI preview summary renders chat/message/branch/duplicate/warning counts.
4. Unsupported Chatbook-only options are hidden or disabled in OpenWebUI mode.
5. Chatbook archive mode behavior is unchanged.

Verification:

- focused backend unit tests for the adapter
- focused endpoint tests for preview/import
- focused frontend component tests for the import tab
- no Bandit-specific code path is expected for the design doc; implementation work must run Bandit on touched Python scope

## Follow-Up Work

1. Direct OpenWebUI `webui.db` import.
2. OpenWebUI admin database export import.
3. Attachment/file hydration when the source export includes accessible file payloads.
4. Folder/tag mapping into tldw-native organization surfaces.
5. Import source auto-detection after explicit source-format behavior is stable.
6. Optional UI highlighting of the imported active branch from `history.currentId`.
