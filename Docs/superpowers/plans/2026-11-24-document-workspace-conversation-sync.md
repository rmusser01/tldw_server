# Document Workspace Conversation Sync Implementation Plan

> **For agentic workers** Use superpowers:executing-plans or superpowers:subagent-driven-development. Work in order and mark each completed step.

**Goal** Let a signed-in user resume and continue the same saved document chat from a second browser.

**Architecture** Media verifies access to the document. ChaCha stores the link to the original native chat record and supplies its transcript. The sidebar combines those server chats with its existing browser histories, and the user chooses **Save to server** for each browser-only chat.

**Tech stack** FastAPI, ChaCha SQLite and PostgreSQL, shared WebUI, Dexie, Vitest, Playwright.

**Spec** [Cross-device conversations in Document Workspace](../specs/2026-11-23-document-workspace-conversation-sync-design.md)

**ADR check** ADR required. [Proposed ADR-098](../../ADR/098-document-workspace-chat-association.md) records the Media–Chat link and access boundary. [ADR-049](../../ADR/049-chat-history-selection-ownership.md) governs continuation. [TASK-15586](../../../backlog/tasks/task-15586%20-%20Sync-Document-Workspace-conversations-across-devices.md) tracks delivery.

## Global constraints

- First release covers browsers on the same tldw server and account. Empty tabs, drafts, selection, and streams remain browser state. Respect the current persistence mode.
- Server-side authorization gates every document-scoped open and new send of a linked chat. Keep existing Chat ownership, admission, quotas, and Sync v2 policy in force.
- Use one canonical chat and one link per conversation. Persisted text and visible document citations must round-trip. Unsupported history stays readable in its original verified browser scope.
- Complete the PDF and EPUB two-browser send with **Use document content** before enabling discovery and continuation together.

## Review focus

- A chat ID reused after hard delete fails an old attachment attempt. Stage 1.
- Access revoked after selection stops the next send before appending a question. Stage 2.
- A different account inherits neither pending responses nor cached server tabs. Stage 3.
- An import accepted before a lost response publishes one conversation even when two browser tabs retry. Stage 4.
- An idle tab discovers the new conversation on foreground refresh without replacing its draft. Stage 5.

## Stage 1 — Integrate the tabs and add authorized chat links

**Status** Complete.

**Goal** Resolve the tabs prerequisite and make owned, nonempty linked conversations discoverable through a paginated Media route.

- [x] Integrate [PR #3534](https://github.com/rmusser01/tldw_server/pull/3534) with Dev in an isolated branch. The actual merge with Dev `6d31f3d1a7` resolved the single Translate conflict in `76cc496428`. It retained both the provider lookup and the expected-account check. GitHub reports the PR mergeable with auto-merge enabled.
- [x] Verify the integrated baseline. The Translation and expected-actor suites passed 16 tests, and the Document Workspace chat and text selection suites passed 35 tests.

**Files and interfaces**

- `tldw_Server_API/app/core/DB_Management/chacha/conversation_store.py` owns `attach_document_chat(conversation_id: str, *, owner_id: str, media_uuid: str, expected_record: str) -> None`, `list_document_chats(*, owner_id: str, media_uuid: str, limit: int, offset: int) -> list[dict[str, Any]]`, and `get_document_chat_link(conversation_id: str, *, owner_id: str, media_uuid: str) -> dict[str, Any] | None`. List requires a saved message. A live empty link is available to first-turn admission.
- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` and its `chacha/migrations/{sqlite_steps,postgres_steps}.py` register the next unused migration. Store `conversation_id` as the unique native foreign key, the original `source_incarnation_id`, `owner_id`, and `media_uuid`. Index `(owner_id, media_uuid)`. Use the existing PostgreSQL private-owner policy pattern.
- Add `tldw_Server_API/app/api/v1/API_Deps/document_chat_scope.py`. It defines `DocumentChatTarget(media_id: int, media_uuid: UUID)` and `resolve_document_chat_target(request: Request, current_user: User, *, media_id: int, media_uuid: UUID) -> DocumentChatTarget` as an async function. Look up the live Media row with the request principal and require the current UUID.
- Add `tldw_Server_API/app/api/v1/endpoints/media/document_chats.py` to the Media router modules. `GET /api/v1/media/{media_id}/chat-conversations` requires `media_uuid` and returns `{items: [{id, title, last_modified}], has_more}`. Default to `limit=50` and `offset=0` with maximum limit 100. `PUT /api/v1/media/{media_id}/chat-conversations/{chat_id}` takes `{media_uuid, document_chat_record}` and returns 204 for initial attach or exact replay. Return 404 for unavailable or replaced targets and a generic 409 for an owned live chat already linked to another document. Use the existing expected-user request dependency.
- In `character_chat_sessions.py` accept paired `document_media_id` and `document_media_uuid` query parameters on the owner-authorized Chat detail, create, and import routes. Use the same access check when present. Add optional `document_chat_record` to `ChatSessionResponse` in `chat_session_schemas.py`. Compute it server-side from the native chat ID and stored incarnation. Return it for a verified document request and compare it inside the attach transaction.

- [x] Cover attach and replay, empty and paginated discovery, mismatched actors, two owners of a shared document, conflicting links, trash and restore, and a deleted chat ID reused by a new record.
- [x] Reproduce the missing list route and record token in the SQLite test run.
- [x] Verify the live Media row and persist one link per owned Chat record in SQLite schema 86 and PostgreSQL schema 92.
- [x] Run the authenticated SQLite and PostgreSQL route tests and the migration registry tests. All 39 passed with PostgreSQL required.

## Stage 2 — Admit and settle document-grounded turns through Chat

**Status** Complete.

**Goal** Prove a saved chat can ask the next question about the linked document and reopen with its cited reply.

**Files and interfaces**

- Add `require_linked_document_chat(request: Request, current_user: User, db: CharactersRAGDB, *, target: DocumentChatTarget, chat_id: str) -> DocumentChatTarget` as an async helper in Stage 1’s `document_chat_scope.py`. It calls the Media access check and uses `get_document_chat_link`. Open the Media session only when the document query parameters are supplied.
- Wire that scope into `capture_conversation_history` in `endpoints/chat.py` and selected admission and settlement in `endpoints/character_messages.py`. Validate live access on the open and admission requests. Native admission uses `tldw_history_selection_v1` and `tldw_history_branch: false` for a regular send. Settlement uses `tldw_history_admission_v1` and the original input under its verified chat and document binding.
- Add `DocumentChatCitation` in `tldw_Server_API/app/api/v1/schemas/document_chat_schemas.py` with `media_uuid`, `title`, `excerpt`, and optional one-based `page` or `epub_cfi`. Store assistant sources under `document_chat_sources_v1` through existing message metadata and restore them through native readback. Reuse the bounds of the existing workspace reference validation and require the verified Media UUID on every citation. Expose source links through authorized Media reads.
- Export `DocumentChatTarget = { mediaId: number; mediaUuid: string }` from `apps/packages/ui/src/services/tldw/domains/media.ts`. Carry `documentTarget?: DocumentChatTarget` on the native owner in `services/chat-history-selection.ts`. Thread it through `captureHistorySnapshot`, `appendSelectedUser`, and `settleAcceptedAssistant` using the existing API scope headers. Apply the same query options to `services/tldw/domains/chat-rag.ts` and the corresponding `TldwApiClient.ts` methods. Use `runChatModeWithHistory` in `hooks/chat-modes/normalChatMode.ts` and `ragMode.ts` retrieval with only the verified Media item. Allow native grounded mode only for this server-validated document path. Use the client-managed native turn so admission and settlement go through the existing message API with document context and citations. Reuse account-specific capability and Sync v2 errors and retain the draft.

- [x] Add `test_document_admission_refuses_revoked_media` and `test_document_reply_round_trips_citation` to `test_document_chat_turns.py`. Cover revoked access, stale concurrent input, citation scope and canonical readback.
- [x] Cover the native grounded send in the mounted history-selection and RAG pipeline test.
- [x] Reproduce the unguarded admission and unsupported native RAG mode before implementing.
- [x] Run both storage backends for the new turns (4 passing) and the focused frontend suites (341 passing).

## Stage 3 — Discover and resume the same tab on another browser

**Status** Not Started.

**Goal** Server history and local tabs work together without losing drafts or carrying state across accounts.

- [ ] Cover `apps/packages/ui/src/hooks/document-workspace/__tests__/useDocumentChat.ownership.test.tsx` and `apps/packages/ui/src/components/DocumentWorkspace/RightPanel/__tests__/DocumentChat.test.tsx`. Add `resolves one tab per verified server chat` with `expect(tabs.filter(tab => tab.serverChatId === id)).toHaveLength(1)` and `keeps the draft and drops retired results on account change` with `expect(queryByText(previousTitle)).toBeNull()` after the old read resolves. Cover an unavailable server retaining an already-owned local transcript and a denial clearing the affected server transcript.
- [ ] Run `bun run test:run -- useDocumentChat.ownership DocumentChat.test` from `apps/tldw-frontend` and confirm the new server discovery case fails.
- [ ] Add `hasDocumentChatSync` to `services/tldw/server-capabilities.ts` using authoritative discovery of the Stage 1 routes. Add `listDocumentChats(target: DocumentChatTarget, options: DocumentChatRequestOptions): Promise<DocumentChatPage>` and `attachDocumentChat(target: DocumentChatTarget, chatId: string, record: string, options: DocumentChatRequestOptions): Promise<void>` to `services/tldw/domains/media.ts`. `DocumentChatPage` mirrors Stage 1’s list response. `DocumentChatRequestOptions` contains `requestScope: ServicePromptRequestScope`, optional `signal: AbortSignal`, `limit` and `offset`.
- [ ] Update `hooks/document-workspace/useDocumentChat.ts` and `components/DocumentWorkspace/RightPanel/DocumentChat.tsx`. Hydrate the selected linked tab through the authorized native capture even on a browser without a mirror. Bind the native history controller from `hooks/chat/useHistorySelection.ts` to that chat with the Stage 2 `documentTarget`. Enable sends after history is ready. Keep private browser scenes for local and temporary tabs. Store `document_media_uuid?: string` on `HistoryInfo` in `db/dexie/types.ts` and populate it from authorized document metadata through the existing history save path.
- [ ] Refresh the list on open, foreground return with the sidebar open, and acknowledged save. In server-save mode, bind the creation recovery handle to the new tab’s own draft key and attach its acknowledged Chat ID and record token before the first admission. After a transient failure show the last verified list for that session and preserve inputs. A supported server’s access denial clears the affected list or chat. Paginated refresh preserves the selected tab and drafts.
- [ ] Re-run the hook and sidebar tests, `bun run typecheck`, and the request-scope tests in `services/tldw/__tests__/TldwApiClient.request-scope.test.ts`. Commit after passing.

## Stage 4 — Save existing chats one at a time

**Status** Not Started.

**Goal** Link verified server-backed histories and promote eligible browser-only histories without data loss or duplicate copies.

- [ ] Add `test_document_chat_import_replays_citations` to `tldw_Server_API/tests/Chat_NEW/integration/test_chat_import_api.py`. Assert `first.status_code == 201`, `second.status_code == 200`, `second.json()["id"] == first.json()["id"]`, and `stored_sources == [citation]`. Assert a citation for a different Media UUID fails before writing and that the current `sync_chat_import_unsupported` refusal leaves the chat absent.
- [ ] Add `reuses a browser chat import after a lost response` to the Stage 3 hook tests. Simulate two tabs reserving the same import. Assert `expect(secondPayload).toEqual(firstPayload)`, `expect(linkedServerIds).toEqual([firstPayload.id])`, and `expect(getByText(lateTurn)).toBeVisible()`. Assert zero writes before numeric-only document confirmation and the original transcript remains visible after refusal.
- [ ] Run `python -m pytest -q tldw_Server_API/tests/Chat_NEW/integration/test_chat_import_api.py tldw_Server_API/tests/ChaChaNotesDB/test_conversation_import_store.py` and `bun run test:run -- useDocumentChat.ownership DocumentChat.test`. Expected red is citation rejection and missing per-chat save.
- [ ] Extend `ChatImportMessage` in `schemas/chat_import_schemas.py` with optional `document_chat_sources_v1` using Stage 2’s citation type. In `endpoints/character_chat_sessions.py` verify the same live document target before importing. In `core/Chat/conversation_import.py` include the normalized sources in the native replay fingerprint and pass them as validated extra metadata to the existing `chat_import_store.py` transaction.
- [ ] Add `reserveDocumentChatImport(historyId: string, target: DocumentChatTarget, requestScope: ServicePromptRequestScope): Promise<DocumentChatImportReservation>` in `apps/packages/ui/src/db/dexie/document-chat-import.ts`. The reservation stores `chatId`, `mediaUuid`, the exact frozen native import body, and optional acknowledged `documentChatRecord`. Reserve it in one owner-scoped Dexie transaction. Preserve message IDs, timestamps, validated reply links, text, supported assets, and citations. Use existing legacy history review for ambiguous ancestry. Retain the browser copy when any of those fields cannot be preserved.
- [ ] Wire per-chat **Save to server** into `DocumentChat.tsx` through `useDocumentChat.ts`. Reuse the receipt after reload. After the import acknowledges, retry just the same attach through Stage 1 upon link failure. Reconcile a verified server copy automatically when its saved Media UUID matches. For numeric-only history, display the current document title and require per-chat confirmation before linking or uploading. Keep new local turns readable until their save is verified.
- [ ] Re-run the native import and hook suites. Pass includes the existing plain Chat import contract and the same server ID on replay. Commit after passing.

## Stage 5 — Verify the release path

**Status** Not Started.

**Goal** Confirm the actual two-device path, including document mode, native retention, and account isolation.

- [ ] Add `apps/tldw-frontend/e2e/workflows/tier-2-features/document-workspace-chat-sync.spec.ts` using `DocumentWorkspacePage` and the existing auth fixtures. Use `tldw_Server_API/tests/Media_Ingestion_Modification/test_media/sample.pdf` and `sample.epub` against a real test backend with a deterministic model response. Create two independent browser contexts for one account and a third for a different account. Assert the same saved tab and citation open after foreground refresh, the existing draft survives, a follow-up question uses document content, and the other account sees no private chat titles. Test a per-chat legacy save and an unconfirmed first message.
- [ ] Run `bun run e2e:pw e2e/workflows/tier-2-features/document-workspace-chat-sync.spec.ts --project=chromium` from `apps/tldw-frontend` against the test backend. Pass means the PDF and EPUB cases execute with zero skipped tests. Verify native, active Sync v2, and older-server behavior according to their capabilities.
- [ ] Generate the API contract with `bun run generate` in `apps/packages/api-types` with the Python venv active. Commit `paths.d.ts` and `openapi.fingerprint.json`. Run `make openapi-drift-check`, the focused Python and frontend suites above, and Bandit on the touched Python modules in that venv. Record CI and two-browser evidence in TASK-15586 using backlog-py before marking its acceptance criteria complete.
