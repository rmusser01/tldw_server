# OpenWebUI Chat Import Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship v1 OpenWebUI `Export Chats` JSON import through the existing Chatbooks import workflow, preserving all valid OpenWebUI message branches as tldw conversation/message trees while making repeated imports safe by `source=openwebui` and `external_ref`.

**Architecture:** Keep Chatbooks as the upload, preview, quota, sync/async import, and jobs surface. Add a focused OpenWebUI JSON import adapter under `tldw_Server_API/app/core/Chatbooks/import_adapters/`, branch Chatbooks preview/import by a multipart `source_format` field before ZIP validation, persist imported conversations through existing ChaCha stores, and extend the current `/chatbooks` WebUI import tab with a source selector instead of creating a new page.

**Tech Stack:** FastAPI, Pydantic, ChaCha SQLite/PostgreSQL store abstractions, core Jobs worker, pytest, Next.js/React, Ant Design, Vitest/Testing Library.

---

## Stage 1: Source Format Schemas And Validation Boundaries

**Goal:** Add explicit source-format contracts and JSON-safe upload validation without changing the existing Chatbook archive behavior.

**Success Criteria:**
- `source_format` defaults to `chatbook` for existing clients.
- `openwebui_json` uploads accept safe `.json` filenames and never run ZIP-only filename rewriting or archive validation.
- Existing `.zip` and `.chatbook` preview/import behavior and tests remain unchanged.
- API responses can carry OpenWebUI preview/import details without overloading the Chatbook manifest fields.

**Implementation Tasks:**
- [ ] Add a source-format type in `tldw_Server_API/app/api/v1/schemas/chatbook_schemas.py`, for example `ChatbookImportSourceFormat` with `chatbook` and `openwebui_json`.
- [ ] Add `source_format: ChatbookImportSourceFormat = chatbook` to `ImportChatbookRequest`.
- [ ] Add optional OpenWebUI response schemas in `chatbook_schemas.py`: `OpenWebUIPreviewChatItem`, `OpenWebUIImportPreview`, and `OpenWebUIImportResult`.
- [ ] Extend `PreviewChatbookResponse` with `source_format` and `openwebui_preview`.
- [ ] Extend `ImportChatbookResponse` with `source_format` and `openwebui_result`.
- [ ] Add `source_format` to `preview_chatbook()` as a multipart form field, with the same default `chatbook` behavior as existing preview callers.
- [ ] Verify `ImportChatbookRequest` can parse `source_format` from the actual upload path used by the WebUI; if `Depends()` only handles query parameters in this route, add explicit `Form` parsing or an `as_form` dependency while preserving existing query-param compatibility tested by current import tests.
- [ ] Add a JSON-specific filename validator in `tldw_Server_API/app/core/Chatbooks/chatbook_validators.py`, or make the existing filename validator source-format aware, so `.json` is accepted only for `openwebui_json` and archive extensions remain archive-only.
- [ ] Keep the current path traversal checks in `tldw_Server_API/app/api/v1/endpoints/chatbooks.py` before both archive and JSON handling.
- [ ] Branch `preview_chatbook()` by `source_format` before `ChatbookValidator.validate_filename()` and `validate_zip_file()`.
- [ ] Branch `import_chatbook()` by `source_format` before `ChatbookValidator.validate_filename()` and `validate_zip_file()`.
- [ ] Keep quota checks, per-user temp directory handling, audit logging, and cleanup behavior shared between source formats.

**Tests:**
- [ ] Extend `tldw_Server_API/tests/Chatbooks/test_chatbooks_api_preview.py` to prove default preview still accepts Chatbook archives without sending `source_format`.
- [ ] Add endpoint tests proving `source_format=openwebui_json` accepts a `.json` upload and does not call ZIP validation.
- [ ] Add endpoint tests proving JSON mode rejects unsafe filenames, path traversal names, non-JSON extensions, malformed JSON, and top-level non-array JSON.
- [ ] Add regression tests proving archive mode still rejects `.json` and still validates archive integrity.
- [ ] Add a multipart test proving `source_format` is honored when sent as a form field, because the current WebUI upload helper sends upload options through FormData fields.

**Status:** Not Started

---

## Stage 2: OpenWebUI Parser And Preview Adapter

**Goal:** Parse normal OpenWebUI chat export JSON into a source-specific preview model with deterministic warnings, duplicate checks, and no database writes.

**Success Criteria:**
- Standard OpenWebUI wrapper objects and documented legacy chat objects are accepted.
- Preview counts chats, messages, branched chats, duplicate chats, malformed chats, and attachment/artifact references.
- Preview includes lightweight per-chat items when practical.
- Parser and preview code are unit-testable without FastAPI.
- Raw chat content is not logged or returned in warnings.

**Implementation Tasks:**
- [ ] Create `tldw_Server_API/app/core/Chatbooks/import_adapters/__init__.py`.
- [ ] Create `tldw_Server_API/app/core/Chatbooks/import_adapters/openwebui.py`.
- [ ] Add adapter dataclasses or Pydantic models for normalized chat plans, message plans, preview items, warnings, and import results.
- [ ] Implement UTF-8 JSON loading with explicit errors for malformed JSON and non-array roots.
- [ ] Implement standard wrapper detection where each item contains a `chat` object with `history.messages`.
- [ ] Implement documented legacy chat-object handling where each array item is treated as the chat payload.
- [ ] Derive stable external refs from OpenWebUI IDs when available, otherwise `openwebui:<index>:<sha256(canonical_chat_json)[0:16]>`.
- [ ] Normalize title fallback order: OpenWebUI title, first user-message excerpt, then `OpenWebUI Import <YYYY-MM-DD>`.
- [ ] Detect branched chats from multiple children, non-current branches, or non-linear parent relationships.
- [ ] Count attachment, file, image, and artifact-like references without hydrating files.
- [ ] Add preview service entry point on `ChatbookService`, for example `preview_openwebui_json(file_path: str)`.
- [ ] During preview, call a new ChaCha duplicate helper from Stage 3 to mark duplicate items by `source=openwebui` and `external_ref`.
- [ ] If parser work lands before the ChaCha helper work, keep duplicate detection behind an injected callback/service method and complete the duplicate-flag wiring after Stage 3.

**Tests:**
- [ ] Add `tldw_Server_API/tests/Chatbooks/test_openwebui_import_adapter.py`.
- [ ] Test standard wrapper parsing with `chat.title`, `chat.models`, `history.currentId`, and `history.messages`.
- [ ] Test legacy object parsing.
- [ ] Test deterministic derived external refs for source objects without IDs.
- [ ] Test branch detection and message counting across sibling branches.
- [ ] Test attachment/artifact reference detection without file hydration.
- [ ] Test malformed chats are counted and warned while valid chats remain previewable.
- [ ] Test preview duplicate flags using a fake duplicate lookup callback or service stub.

**Status:** Not Started

---

## Stage 3: ChaCha Duplicate And Metadata Helpers

**Goal:** Add small store-level helpers needed by the importer while preserving the repository rule that database access goes through DB abstractions.

**Success Criteria:**
- Duplicate lookup is a reusable ChaCha helper, not ad hoc SQL in the adapter or endpoint.
- Conversation import metadata is persisted under `conversation_settings.settings_json.openwebui_import`.
- Message import metadata is persisted under `message_metadata.extra.openwebui_import`.
- Metadata persistence failure produces warnings without rolling back an otherwise valid imported chat.

**Implementation Tasks:**
- [ ] Add `get_conversation_by_source_ref(source, external_ref, client_id=None, include_deleted=False)` to `tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py`.
- [ ] Expose the helper through `CharactersRAGDB` delegation if needed by existing store extraction patterns.
- [ ] Use parameterized queries and honor `client_id`, defaulting to the DB instance client id when omitted.
- [ ] Add a helper such as `merge_conversation_settings(conversation_id, patch)` if existing `upsert_conversation_settings()` cannot safely preserve unrelated settings keys.
- [ ] Reuse existing `set_message_metadata_extra()` for message metadata; do not add a parallel metadata table.
- [ ] Add tests for SQLite helper behavior and, if nearby patterns exist, PostgreSQL SQL generation/compatibility.

**Tests:**
- [ ] Add or extend `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_conversation_store.py` or adjacent ChaCha tests for source/external-ref lookup.
- [ ] Test helper excludes deleted conversations by default.
- [ ] Test helper scopes results by client id.
- [ ] Test conversation settings merge preserves existing settings keys while adding `openwebui_import`.
- [ ] Test message metadata merge preserves existing `extra` keys while adding `openwebui_import`.

**Status:** Not Started

---

## Stage 4: OpenWebUI Import Service Path

**Goal:** Import valid OpenWebUI chats as normal tldw conversations/messages with preserved parent links, duplicate handling, and structured results.

**Success Criteria:**
- One OpenWebUI chat imports to one tldw conversation with `source=openwebui` and a deterministic `external_ref`.
- All valid OpenWebUI message nodes are imported, not just the `history.currentId` active path.
- Parent messages are inserted before descendants and `parent_message_id` points at mapped tldw message IDs.
- Message IDs are UUID-shaped deterministic IDs, not raw OpenWebUI IDs.
- Duplicate `skip` and `rename` behaviors match the approved design.
- Partial import continues at chat granularity and returns useful counts/warnings.

**Implementation Tasks:**
- [ ] Add `ChatbookService.import_openwebui_json(...)` for sync import results.
- [ ] Add a private helper to resolve the fallback assistant identity for OpenWebUI imports by reusing `_resolve_import_character_id(None)` or a narrower wrapper around `_get_fallback_character_id()`.
- [ ] Validate that no valid chat is imported when no fallback character is available, and surface a clear source-format-specific error.
- [ ] For each valid chat, check duplicate conversations with `get_conversation_by_source_ref("openwebui", external_ref, client_id)`.
- [ ] For `skip`, count duplicate chats as skipped and do not insert messages.
- [ ] For `rename`, generate a unique title and copy external ref suffix before deriving message IDs.
- [ ] Create conversations with `source`, `external_ref`, current `client_id`, fallback `character_id`, and normalized title.
- [ ] Store OpenWebUI conversation metadata in `conversation_settings.settings_json.openwebui_import`, including source timestamps, models, options, meta, pinned, folder id, and `history.currentId`.
- [ ] Normalize roles: `user` and `assistant` are imported; unsupported roles are skipped with warnings for v1.
- [ ] Convert valid Unix-second timestamps to UTC ISO strings before `db.add_message()`, using import time plus warnings for missing or invalid timestamps.
- [ ] Generate UUIDv5 message IDs from an import namespace and source message ID.
- [ ] Validate parent references, detect cycles, and topologically order messages before insertion.
- [ ] Insert messages with `id`, `conversation_id`, `sender`, `content`, `timestamp`, and mapped `parent_message_id`.
- [ ] Store per-message OpenWebUI metadata with source message id, source parent id, source children ids, model, done/context fields, attachment refs, and raw unsupported key names.
- [ ] Avoid logging raw chat/message content in exceptions or warnings.

**Tests:**
- [ ] Test simple two-message import creates one conversation and two messages.
- [ ] Test branched import creates all branch messages with correct `parent_message_id`.
- [ ] Test parent-before-child ordering for unordered OpenWebUI message maps.
- [ ] Test cycles, missing parents, malformed messages, and unsupported roles produce warnings and skip affected messages.
- [ ] Test Unix timestamp conversion to UTC ISO strings.
- [ ] Test duplicate `skip` prevents reimport.
- [ ] Test duplicate `rename` creates a second conversation with unique external ref and non-colliding message IDs.
- [ ] Test conversation and message metadata persistence paths.
- [ ] Test metadata write failures become warnings without failing the chat import.

**Status:** Not Started

---

## Stage 5: Async Jobs And Endpoint Dispatch

**Goal:** Make OpenWebUI imports work in both sync and async Chatbooks flows while preserving existing import job lifecycle semantics.

**Success Criteria:**
- Sync OpenWebUI import returns `ImportChatbookResponse` with `source_format=openwebui_json` and `openwebui_result`.
- Async OpenWebUI import creates an existing-style Chatbooks import job and enqueues a core Jobs payload containing `source_format=openwebui_json`.
- The core Jobs worker dispatches JSON imports to the OpenWebUI service path and never treats JSON tokens as archives.
- Temp files are cleaned up on success, failure, and enqueue failure according to the existing ownership model.
- Existing archive import job tests keep passing.

**Implementation Tasks:**
- [ ] Extend `ChatbookService.import_chatbook()` with `source_format`, defaulting to `chatbook`.
- [ ] Keep archive imports routed to the existing `_import_chatbook_sync()` path.
- [ ] Route sync OpenWebUI imports to `import_openwebui_json()`.
- [ ] Include `source_format` in core Jobs import payloads.
- [ ] Include `source_format` in any Prompt Studio adapter payload branches if those branches still exist, even if they currently fall back to core Jobs.
- [ ] Update `tldw_Server_API/app/core/Chatbooks/services/jobs_worker.py` to parse `source_format` and branch before calling `_resolve_import_archive_path()` or `_import_chatbook_sync()`.
- [ ] Add or reuse a general uploaded-import-file path resolver for JSON files that allows the same temp/import roots but does not imply archive-only semantics in error messages.
- [ ] Keep archive workers on `_resolve_import_archive_path()` and route JSON workers through the general resolver plus `import_openwebui_json()`.
- [ ] Update import job completion result mapping to preserve `openwebui_result` when available.
- [ ] Audit preview/import cleanup paths so JSON preview always removes temp files and async import lets the worker remove the uploaded JSON.

**Tests:**
- [ ] Extend `tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_worker_import_defaults.py` for `source_format=openwebui_json` dispatch.
- [ ] Add endpoint sync import test for OpenWebUI JSON returning `openwebui_result`.
- [ ] Add endpoint async import test verifying job payload includes `source_format`.
- [ ] Add worker test proving JSON dispatch does not call `_import_chatbook_sync()`.
- [ ] Add cleanup tests for invalid JSON preview/import paths where practical.

**Status:** Not Started

---

## Stage 6: WebUI Import Experience

**Goal:** Extend the existing `/chatbooks` import tab so users can choose between Chatbook archives and OpenWebUI JSON without adding another import page.

**Success Criteria:**
- Import tab has a source selector: `Chatbook archive` and `OpenWebUI JSON`.
- Chatbook archive mode keeps current behavior.
- OpenWebUI mode accepts `.json`, sends `source_format=openwebui_json` to preview/import, and renders OpenWebUI preview counts.
- Unsupported archive-only options are hidden or disabled for OpenWebUI mode.
- Unsupported conflict choices are not offered for OpenWebUI v1.

**Implementation Tasks:**
- [ ] Add source-format state to `apps/packages/ui/src/components/Option/Chatbooks/ChatbooksPlaygroundPage.tsx`.
- [ ] Change the upload `accept` value to `.zip,.chatbook` for archive mode and `.json` for OpenWebUI mode.
- [ ] Update drop-zone copy to match the selected source format.
- [ ] Update `handlePreview()` to call `tldwClient.previewChatbook(file, { source_format })`.
- [ ] Update `handleImport()` to call `tldwClient.importChatbook(file, { source_format, ...existingOptions })`.
- [ ] Render OpenWebUI preview summary from `openwebui_preview` when selected, including chat count, message count, branch count, duplicate count, attachment reference count, malformed count, and warnings.
- [ ] In OpenWebUI mode, disable or hide media/embedding import controls.
- [ ] In OpenWebUI mode, force `import_media=false`, `import_embeddings=false`, and omit Chatbook content selections from the request, even if prior archive-mode state had those options enabled.
- [ ] In OpenWebUI mode, limit conflict choices to `skip` and `rename`.
- [ ] Do not show the existing Chatbook content-type picker for OpenWebUI v1; v1 imports all valid chats from the selected JSON export.
- [ ] Ensure switching source formats clears stale preview state and file selection.
- [ ] Update `apps/packages/ui/src/services/tldw/domains/chat-rag.ts` and `apps/packages/ui/src/services/tldw/TldwApiClient.ts` method signatures to support preview/import option fields.

**Tests:**
- [ ] Add component tests under `apps/packages/ui/src/components/Option/Chatbooks/__tests__/` for source selector behavior.
- [ ] Test OpenWebUI mode sends `source_format=openwebui_json` for preview and import.
- [ ] Test archive mode keeps existing preview/import calls compatible.
- [ ] Test unsupported controls are hidden or disabled in OpenWebUI mode.
- [ ] Test OpenWebUI preview summary renders counts and warnings.

**Status:** Not Started

---

## Stage 7: Documentation And Verification

**Goal:** Document the supported v1 workflow and run focused verification across backend, frontend, and security checks.

**Success Criteria:**
- User-facing docs explain that v1 supports normal OpenWebUI `Export Chats` JSON only.
- Docs explicitly list out-of-scope direct `webui.db`, admin export, live server, and attachment hydration support.
- Backend and frontend focused tests pass.
- Bandit is run on touched Python scope.
- The implementation task and plan statuses are updated before final handoff.

**Implementation Tasks:**
- [ ] Update a Chatbooks or import-related doc under `Docs/` with the OpenWebUI JSON workflow.
- [ ] Update the static API contract documentation under `Docs/API-related/` if the project keeps Chatbooks multipart schemas there in addition to generated OpenAPI output.
- [ ] Mention duplicate behavior, branch preservation, warning semantics, and unsupported attachments/artifacts.
- [ ] Keep `Docs/superpowers/specs/2026-05-10-openwebui-chat-import-design.md` as the design reference; do not duplicate the full design in user docs.
- [ ] Update this plan file's stage statuses as implementation progresses.
- [ ] Update the Backlog task with touched files, verification commands, skips, and final summary.

**Verification Commands:**
- [ ] `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chatbooks/test_openwebui_import_adapter.py -v`
- [ ] `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chatbooks/test_chatbooks_api_preview.py tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_worker_import_defaults.py -v`
- [ ] `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/ChaChaNotesDB -k "conversation or metadata" -v`
- [ ] `bunx vitest run apps/packages/ui/src/components/Option/Chatbooks/__tests__`
- [ ] `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/api/v1/endpoints/chatbooks.py tldw_Server_API/app/core/Chatbooks tldw_Server_API/app/core/DB_Management/chacha -f json -o /tmp/bandit_openwebui_chat_import.json`
- [ ] Run a manual browser smoke test of `/chatbooks` import mode if a frontend dev server is available in the implementation session.

**Status:** Not Started

---

## Implementation Notes

- The approved design reference is `Docs/superpowers/specs/2026-05-10-openwebui-chat-import-design.md`; keep v1 bounded to OpenWebUI chat JSON exports.
- Do not implement direct `webui.db`, admin database export, live OpenWebUI import, or attachment hydration in this plan.
- The current endpoint code validates Chatbook uploads as ZIP archives before service dispatch; OpenWebUI JSON must branch before that validation.
- `ChatbookValidator.validate_filename()` is currently archive-only and rewrites unsupported extensions to `.zip`; avoid using it for JSON unless it is made source-format aware.
- `ChatbookService._resolve_import_archive_path()` currently has archive wording but already constrains paths to per-user import/temp roots; either rename/generalize carefully or add a JSON-specific wrapper to avoid misleading errors.
- Existing `ChatbookService._get_fallback_character_id()` and `_resolve_import_character_id()` can likely satisfy OpenWebUI's required conversation assistant identity without adding new persona/character concepts.
- `ConversationStore.add_conversation()` already accepts `source` and `external_ref`.
- `MessageStore.add_message()` already accepts caller-provided UUID string IDs and `parent_message_id`.
- Use existing `upsert_conversation_settings()` / `get_conversation_settings()` and `set_message_metadata_extra()` rather than new metadata storage.
- Avoid raw chat/message content in logs, warnings, audit metadata, and job errors.
