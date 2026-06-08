# Notes UX Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the `/notes` WebUI and directly connected browser-extension capture workflows into a reliable, understandable, and efficient note-taking experience.

**Architecture:** Keep storage contracts stable unless a slice explicitly introduces a backend change. Use the UI term "tags" while preserving the existing `keywords` API/DB model, and treat captured-note Inbox behavior as a durable view over capture provenance and/or reserved tags rather than client-only state. Each PR must be small enough to review independently and must include tests for the workflow it changes.

**Tech Stack:** Next.js/React WebUI under `apps/packages/ui`, FastAPI endpoints under `tldw_Server_API/app/api/v1`, ChaChaNotes persistence under `tldw_Server_API/app/core/DB_Management`, Vitest/React Testing Library for UI tests, pytest for backend tests, Playwright/browser checks for end-to-end UX verification.

---

## Backlog And Scope

**Tracking task:** `TASK-481`

**Primary surface:** `/notes`

**Direct handoffs in scope:**
- Navigation into `/notes`
- Creating, editing, saving, deleting, restoring, searching, filtering, and tagging notes
- Existing note graph/backlink/connection affordances
- Existing chat conversation/message backlinks
- Existing reading-item/source/clipper note links where route support exists
- Exposed import, export, and offline draft/sync workflows
- Browser-extension sidepanel quick-save and Web Clipper save-to-notes flows

**Explicit non-goals:**
- Do not change sidebar defaults or setup-time sidebar customization.
- Do not rename backend/API fields from `keywords` to `tags`.
- Do not add first-class media, research-run, prompt, or arbitrary object links in this plan.
- Do not introduce a new hierarchy model unless a specific slice verifies the API/storage contract.
- Do not silently migrate existing notes into an Inbox/captured tag without an explicit migration decision.

## Evidence To Preserve

Implementation should stay grounded in these current code facts:

- Latest code review update:
  - `GET /api/v1/notes/search` currently returns `list[NoteResponse]`, while `/api/v1/notes/` and `/api/v1/notes/trash` return `NotesListResponse` with canonical offset pagination. The frontend already contains fallback parsing for both shapes, but search totals degrade to the current page length.
  - `/api/v1/notes/keywords/search/` is registered after `/api/v1/notes/keywords/{keyword_id}`. Move both trailing and non-trailing search aliases before the integer ID route so `search` cannot bind as `keyword_id`.
  - Notes FTS already searches note title and content. Do not add a parallel body-search feature; fix API shape, totals, and UX state around existing search.
  - The notes list hook uses React Query `placeholderData: keepPreviousData` and stores `total` separately, but does not expose query error/stale state to list rendering. This explains stale list counts after backend failure.
  - Offline save already queues drafts, but sets the top-level save indicator to `saved`. Preserve the queue while distinguishing server-saved from locally queued/syncing/conflicted states.
  - Extension context-menu quick-save already opens the sidepanel and pre-fills `NoteQuickSaveModal`; hardening should verify delivery, source metadata persistence, and failure recovery rather than invent the flow.
- `NoteCreate` and `NoteUpdate` persist `keywords`, `conversation_id`, and `message_id`, but do not define a general durable note `metadata` field.
- `note_store.add_note` stores title, content, and chat backlink fields; arbitrary metadata sent from clients is not part of the core insert contract.
- The notes editor currently sends both a `metadata` object and top-level `keywords`/chat link fields; the top-level fields are the durable part.
- Sidepanel quick-save currently sends `metadata.source_url` and `metadata.origin`; this is likely not durable through the generic notes create endpoint.
- Web Clipper save has a stronger provenance path via `WebClipperService`, `source_url`, `capture_metadata`, and clipper document storage.
- Web Clipper destination fields currently expose raw Folder ID and Workspace ID inputs.
- Notes graph/backlink APIs and UI already exist; this plan should improve clarity and reliability before expanding link targets.

## Finding Coverage Map

Use this map to keep implementation PRs tied to the UX review findings. If a PR discovers that a finding has already been fixed or was misdiagnosed, update the relevant Backlog subtask with that evidence before changing scope.

- N-01 stale/disconnected list state and persistence confidence: PR 2, PR 3, PR 10.
- N-02 beginner empty/no-results confusion: PR 2, PR 4.
- N-03 mobile horizontal overflow: PR 9.
- N-04 small mobile touch targets and dense controls: PR 9.
- N-05 tag/keyword suggestion failure and tag filtering reliability: PR 1, PR 5.
- N-06 first-note creation, save confirmation, and recovery guidance: PR 3, PR 4.
- N-07 saved-versus-queued ambiguity: PR 3, PR 10.
- N-08 search/filter pagination and result-count mismatch: PR 1, PR 2.
- N-09 modal/focus/accessibility leakage: PR 9.
- N-10 organization, linking, and relationship clarity: PR 5, PR 8.
- N-11 backend/offline error recovery and reliability signaling: PR 2, PR 3, PR 10.
- N-12 responsive QA and mobile workflow confidence: PR 9.
- Browser-extension note capture needs verification: PR 6, PR 7.

## File Map

Likely frontend files:
- `apps/packages/ui/src/components/Notes/NotesManagerPage.tsx` - main `/notes` orchestration, list/editor state, graph handoffs.
- `apps/packages/ui/src/components/Notes/NotesSidebar.tsx` - notes list/sidebar controls, responsive widths, search/filter UI, pagination summary.
- `apps/packages/ui/src/components/Notes/NotesListPanel.tsx` - result list rendering, pagination copy, empty-state selection.
- `apps/packages/ui/src/components/Notes/NotesListPanelEmptyStates.tsx` - empty, offline, unsupported, and no-results state copy/actions.
- `apps/packages/ui/src/components/Notes/NotesEditorPane.tsx` - editor controls, save state, connection panels.
- `apps/packages/ui/src/components/Notes/NotesEditorHeader.tsx` - header-level actions, save-status display, mobile touch target sizing.
- `apps/packages/ui/src/components/Notes/NotesSaveStatus.tsx` - compact dirty/saving/saved/error state indicator.
- `apps/packages/ui/src/components/Notes/NotesGraphModal.tsx` - graph/edge labeling and navigation.
- `apps/packages/ui/src/components/Notes/hooks/useNotesEditorState.tsx` - editor state, dirty/save behavior, note payload construction.
- `apps/packages/ui/src/components/Notes/hooks/useNotesListManagement.tsx` - note list fetching, search/filter params, totals, stale/error state.
- `apps/packages/ui/src/services/note-keywords.ts` - keyword/tag suggestion and stats client calls.
- `apps/packages/ui/src/components/Sidepanel/Clipper/WebClipperPanel.tsx` - extension clipping workflow and save payload.
- `apps/packages/ui/src/components/Sidepanel/Clipper/ClipDestinationFields.tsx` - capture destination controls.
- `apps/packages/ui/src/routes/sidepanel-chat.tsx` - sidepanel quick-save path.
- `apps/tldw-frontend/extension/routes/sidepanel-chat.tsx` - frontend extension route wrapper for sidepanel quick-save path.
- `apps/packages/ui/src/components/Sidepanel/Notes/NoteQuickSaveModal.tsx` - quick-save title/content/source modal.
- `apps/packages/ui/src/entries/background.ts` - extension context-menu note handoff.
- `apps/packages/ui/src/services/tldw/domains/collections.ts` - notes, reading-item links, and related client calls.
- `apps/packages/ui/src/services/tldw/domains/workspace-api.ts` - workspace destination lookup/placement support.
- `apps/packages/ui/src/routes/option-notes.tsx` - `/notes` route wrapper and route error boundary.
- `apps/packages/ui/src/routes/route-registry.tsx` and `apps/packages/ui/src/routes/route-metadata.ts` - route registration and discoverability metadata.
- `apps/packages/ui/src/public/_locales/en/option.json` and `apps/packages/ui/src/public/_locales/en/sidepanel.json` - directly connected copy changes.

Likely backend files:
- `tldw_Server_API/app/api/v1/schemas/notes_schemas.py` - notes API contracts.
- `tldw_Server_API/app/api/v1/endpoints/notes.py` - notes create/update/list/search/delete/restore behavior.
- `tldw_Server_API/app/api/v1/endpoints/notes_graph.py` - graph and neighbor endpoints.
- `tldw_Server_API/app/api/v1/schemas/web_clipper_schemas.py` - Web Clipper payload contracts.
- `tldw_Server_API/app/api/v1/endpoints/web_clipper.py` - Web Clipper API route.
- `tldw_Server_API/app/core/WebClipper/service.py` - durable capture provenance and destination sync.
- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` and `tldw_Server_API/app/core/DB_Management/chacha/note_store.py` - persistence helpers if a slice requires backend changes.

Likely tests:
- `apps/packages/ui/src/components/Notes/**/__tests__/*`
- `apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage11.search-filtering.test.tsx`
- `apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage14.sorting-pagination.test.tsx`
- `apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage20.accessibility-shortcuts.test.tsx`
- `apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage21.accessibility-modal-focus.test.tsx`
- `apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage23.responsive-layout.test.tsx`
- `apps/packages/ui/src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx`
- `apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage13.navigation-filter-summary.test.tsx`
- `apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage36.import-workflow.test.tsx`
- `apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage30.export-progress.test.tsx`
- `apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage41.offline-drafting-sync.test.tsx`
- `apps/packages/ui/src/routes/__tests__/sidepanel-chat.note-quick-save-lazy-mount.guard.test.ts`
- `apps/packages/ui/src/entries/__tests__/background.web-clipper.test.ts`
- `apps/packages/ui/src/routes/**/__tests__/*sidepanel*`
- `apps/packages/ui/src/routes/__tests__/route-metadata.coverage.test.ts`
- `apps/packages/ui/src/services/**/__tests__/*`
- `tldw_Server_API/tests/Notes_NEW/integration/test_notes_api.py`
- `tldw_Server_API/tests/ChaChaNotesDB/test_chachanotes_db.py`
- `tldw_Server_API/tests/**/test_*notes*.py`
- `tldw_Server_API/tests/**/test_*web_clipper*.py`
- `tests/e2e/**` or existing WebUI Playwright test locations if present.

## Cross-Cutting Rules For Every PR

- [ ] Start by updating or creating the Backlog.md subtask for that PR.
- [ ] Write the failing test first when behavior changes are testable.
- [ ] Keep API/storage migrations out unless the PR explicitly owns them.
- [ ] Preserve unrelated dirty worktree changes.
- [ ] Include accessibility acceptance checks in every UI PR.
- [ ] Run focused tests for touched code.
- [ ] Run Bandit for touched Python scope when backend Python changes are made; document skip for frontend-only PRs.
- [ ] Perform a browser check for user-visible flow changes.
- [ ] Record verification results in the Backlog task before finalizing.

---

## PR 1: Backend Search And Keyword API Contracts

**Goal:** Make notes search and keyword suggestions reliable before further UX work depends on them.

**Likely files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/notes.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/notes_schemas.py` only if `NotesListResponse` needs aliases or schema clarification
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` and `tldw_Server_API/app/core/DB_Management/chacha/note_store.py` only if keyword-filtered search counts are implemented in persistence
- Modify: `apps/packages/ui/src/components/Notes/hooks/useNotesListManagement.tsx`
- Modify: `apps/packages/ui/src/services/note-keywords.ts` only if the final keyword-search URL contract changes
- Test: `tldw_Server_API/tests/Notes_NEW/integration/test_notes_api.py`
- Test: `tldw_Server_API/tests/ChaChaNotesDB/test_chachanotes_db.py`
- Test: `apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage11.search-filtering.test.tsx`

**Acceptance criteria:**
- `/api/v1/notes/search` and `/api/v1/notes/search/` return the same list envelope shape as `/api/v1/notes/`: `notes`, `items`, `results`, `count`, `limit`, `offset`, `total`, `pagination`.
- Query-only search uses the existing title/content FTS path and includes an accurate `pagination.total`/`total` when available.
- Keyword-token search returns a stable total. If exact counts require new DB helpers, add focused helpers and tests instead of estimating from the current page.
- `sort_by` and `sort_order` either work consistently for search or are explicitly normalized/ignored with a documented backend decision and frontend alignment.
- `/api/v1/notes/keywords/search` and `/api/v1/notes/keywords/search/` both return keyword suggestions and cannot route through `/keywords/{keyword_id}`.
- Existing list, trash, export, graph, and keyword CRUD behavior remains unchanged.

**Test plan:**
- Backend integration test: keyword search without trailing slash returns 200 for `query=fru`.
- Backend integration test: keyword search with trailing slash returns 200 and does not bind `search` as `keyword_id`.
- Backend integration test: note search returns `NotesListResponse` envelope and canonical pagination metadata.
- Backend integration test: search `total` remains larger than the current page when more matching notes exist.
- Backend DB test: FTS still finds content and title matches.
- Frontend component test: filtered notes consume paginated search response and render correct total/range.
- Frontend regression test: legacy array response fallback still works if backward compatibility is intentionally preserved.

**Implementation steps:**
- [x] Create or update the Backlog subtask for PR 1.
- [x] Add failing backend tests for keyword-search route order and search response pagination.
- [x] Run the focused backend test command and confirm the intended failures.
- [x] Move keyword search route declarations before all `/keywords/{keyword_id}` routes and add a non-trailing alias.
- [x] Change `search_notes_endpoint` to return `NotesListResponse` using `build_offset_pagination_meta`.
- [x] Add exact count support for keyword-filtered search if missing.
- [x] Align frontend parsing and any `sort_by`/`sort_order` assumptions with the backend decision.
- [x] Run focused backend and frontend tests.
- [x] Run Bandit for touched backend scope.
- [x] Update Backlog notes with changed files and verification.

---

## PR 2: Notes List Reliability And Empty States

**Goal:** Make the `/notes` list trustworthy after create, delete, restore, filter, and search changes.

**Likely files:**
- Modify: `apps/packages/ui/src/components/Notes/NotesManagerPage.tsx`
- Modify: `apps/packages/ui/src/components/Notes/hooks/useNotesListManagement.tsx`
- Modify: `apps/packages/ui/src/components/Notes/NotesListPanel.tsx`
- Modify: `apps/packages/ui/src/components/Notes/NotesListPanelEmptyStates.tsx`
- Modify: `apps/packages/ui/src/components/Notes/hooks/useNotesEditorState.tsx` only if list/editor coupling causes stale state
- Test: `apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage11.search-filtering.test.tsx`
- Test: `apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage14.sorting-pagination.test.tsx`
- Test: notes component tests under `apps/packages/ui/src/components/Notes/**/__tests__/`

**Acceptance criteria:**
- Deleted notes disappear from active/recent lists without requiring a page refresh.
- Restored notes reappear in the expected active list state.
- Creating a note updates the visible list immediately.
- Loading, empty library, empty search result, and error states are visually and semantically distinct.
- Search/filter state does not display stale notes after deletion or restore.
- React Query placeholder/stale data is represented honestly: stale content may remain visible only with a stale/error banner, retry action, and non-authoritative count treatment.
- Active filters with zero matches render no-results guidance and a clear-filters action, not first-time "No notes yet" copy.
- Backend offline/unreachable state does not leave stale pagination such as "Showing 1-12 of 12" without an error/stale marker.
- Sidebar behavior is untouched.

**Test plan:**
- Component test: creating a note prepends or inserts it according to the current sort.
- Component test: deleting a note removes it from the active list and selected state.
- Component test: restoring a note returns it to active list state.
- Component test: search with no matches renders no-results state, not first-time empty state.
- Component test: failed search/list request renders error or stale-results state with retry.
- Component test: disconnected state does not present stale totals as fresh.
- Browser check: `/notes` create -> search -> delete -> restore -> search again.

**Implementation steps:**
- [x] Create or update the Backlog subtask for PR 2.
- [x] Add failing tests for stale delete/restore/list-refresh behavior.
- [x] Run the focused test command and confirm the intended failures.
- [x] Fix list state invalidation/refetch/update logic with the smallest local change.
- [x] Add or adjust empty/loading/error state rendering.
- [x] Run focused Notes tests.
- [x] Run browser verification on desktop and a narrow mobile viewport.
- [x] Update Backlog notes with changed files and verification.

---

## PR 3: Save State And Error Recovery

**Goal:** Users always know whether a note is clean, dirty, saving, saved, failed, or conflicted, and can recover without losing edits.

**Likely files:**
- Modify: `apps/packages/ui/src/components/Notes/NotesEditorPane.tsx`
- Modify: `apps/packages/ui/src/components/Notes/hooks/useNotesEditorState.tsx`
- Modify: notes API client/domain tests if payload behavior is involved
- Test: notes editor state/component tests

**Acceptance criteria:**
- Dirty editor state enables save and is represented by visible, accessible status text.
- Saving state prevents duplicate saves and announces progress.
- Failed save keeps unsaved edits visible and recoverable.
- Successful save clears dirty state and updates visible modified/version state.
- Conflict/version errors produce a clear next action.
- Offline local save is not labeled as server-saved. It uses a distinct queued/syncing/conflict/error/synced state in the header or immediately adjacent status area.
- `NotesSaveStatus` and `offlineStatusText` do not contradict each other.
- Navigation away from dirty notes follows one consistent policy: existing autosave/prompt behavior should be clarified, not replaced wholesale.

**Test plan:**
- Unit/component test: editing title/content/tags marks the note dirty.
- Component test: save success clears dirty state and updates status.
- Component test: save failure displays an error and preserves draft content.
- Component test: duplicate save is blocked while saving.
- Component test: conflict response renders recovery affordance.
- Browser check: edit -> save -> reload -> verify content; force/mock failure where feasible.

**Implementation steps:**
- [x] Create or update the Backlog subtask for PR 3.
- [x] Add failing tests for dirty/saving/failed/conflict states.
- [x] Run the focused test command and confirm failures.
- [x] Normalize save-state transitions in `useNotesEditorState.tsx`.
- [x] Update `NotesEditorPane.tsx` controls and status copy.
- [x] Verify keyboard and screen-reader access to save/error status.
- [x] Run focused tests and browser check.
- [x] Update Backlog notes with verification.

---

## PR 4: Navigation And First-Time Notes UX

**Goal:** Users can discover and enter `/notes`, understand the initial screen, and create their first useful note without hunting.

**Likely files:**
- Modify: `apps/packages/ui/src/routes/option-notes.tsx` only if the route wrapper/error boundary blocks comprehension
- Modify: `apps/packages/ui/src/routes/route-registry.tsx` and `apps/packages/ui/src/routes/route-metadata.ts` only if route discoverability metadata is missing or misleading
- Modify: `apps/packages/ui/src/components/Notes/NotesManagerPage.tsx`
- Modify: `apps/packages/ui/src/components/Notes/NotesEditorPane.tsx`
- Modify: locale files if copy is externalized
- Test: route metadata, navigation summary, empty-state, and create-flow tests

**Acceptance criteria:**
- `/notes` route metadata labels the destination clearly as Notes.
- Route wrapper/error boundary uses user-facing Notes language and does not obscure recovery actions.
- Any existing in-page route/filter summary accurately reflects the current `/notes` state.
- First-time empty state has one obvious primary action: create note.
- Create action opens a writable editor and places focus in the most useful field.
- Blank title/content behavior is deterministic and understandable.
- First successful save gives visible confirmation.
- Empty-state layout works on desktop and mobile without overlap.

**Test plan:**
- Route metadata test: `/notes` remains registered and labeled as Notes.
- Component test: navigation/filter summary reflects the current notes view.
- Component test: empty state renders create action.
- Component test: create action opens editor and focuses title or content.
- Component test: blank note handling follows the chosen rule.
- Browser screenshots/checks for desktop and mobile empty state.

**Implementation steps:**
- [x] Create or update the Backlog subtask for PR 4.
- [x] Add failing tests for route metadata, navigation/filter summary, empty state, and first-create focus behavior.
- [x] Run focused tests and confirm failures.
- [x] Implement focused empty-state and focus-management improvements.
- [x] Verify mobile layout.
- [x] Run tests and browser checks.
- [x] Update Backlog notes with verification.

---

## PR 5: Tags Terminology And Organization Semantics

**Goal:** Present one understandable user concept, "tags", while preserving the existing `keywords` implementation contract.

**Likely files:**
- Modify: `apps/packages/ui/src/components/Notes/*`
- Modify: `apps/packages/ui/src/components/Sidepanel/Clipper/*`
- Modify: `apps/packages/ui/src/public/_locales/en/option.json`
- Modify: `apps/packages/ui/src/public/_locales/en/sidepanel.json`
- Test: affected UI copy/component tests

**Acceptance criteria:**
- User-facing labels in `/notes` and directly connected capture flows say "Tags".
- API payloads, TypeScript client fields, and backend schemas continue to use `keywords`.
- Tests assert user-facing tag labels while preserving `keywords` payload assertions.
- Filter/search UI makes clear whether a control filters by text, tag, folder, or captured state.
- No database/API rename or migration is introduced.

**Test plan:**
- Component tests query "Tags" labels in notes and clipper flows.
- Client/service tests continue to assert `keywords` payloads.
- Locale snapshot or targeted assertions cover changed strings.

**Implementation steps:**
- [x] Create or update the Backlog subtask for PR 5.
- [x] Add or update tests that currently expose "Keywords" to users.
- [x] Change user-facing copy to "Tags" only.
- [x] Confirm API/client payloads remain `keywords`.
- [x] Run focused UI tests.
- [x] Update Backlog notes with verification.

---

## PR 6: Capture Provenance And Inbox View

**Goal:** Captured extension notes reliably land in All Notes and are discoverable through an Inbox/captured view backed by durable data.

**Likely files:**
- Modify: `apps/packages/ui/src/routes/sidepanel-chat.tsx`
- Modify: `apps/packages/ui/src/services/tldw/domains/collections.ts`
- Modify: `apps/packages/ui/src/components/Sidepanel/Clipper/WebClipperPanel.tsx`
- Modify: `tldw_Server_API/app/core/WebClipper/service.py` if clipper provenance/tag handling changes
- Modify: `tldw_Server_API/app/api/v1/endpoints/notes.py` only if a supported notes provenance contract is added
- Test: sidepanel quick-save tests, Web Clipper save-flow tests, backend clipper/notes tests as needed

**Acceptance criteria:**
- Web Clipper saves continue to persist `source_url`, `capture_metadata`, title/comment, and tags.
- Sidepanel quick-save no longer relies on arbitrary ignored note metadata for source URL/origin.
- Captured notes appear in All Notes.
- Inbox/captured view is backed by durable clipper provenance and/or a reserved capture tag.
- The reserved capture marker, if used, is documented in code comments/tests and does not silently rewrite existing notes.
- Capture failure clearly tells the user whether note creation, provenance storage, or destination placement failed.

**Product decision before implementation:**
- Decide whether Inbox is implemented as:
  - a saved view over clipper provenance,
  - a reserved tag such as `captured` or `inbox`,
  - or both.

**Decision:** Use a normal reserved tag, `captured`, as the first Inbox/captured view backing model. Web Clipper continues to persist first-class `source_url` and `capture_metadata`; both Web Clipper and sidepanel quick-save add `captured` only to newly captured notes. Sidepanel quick-save persists the page URL in note content because the generic notes create endpoint does not support arbitrary `metadata.source_url`.

**Test plan:**
- Backend test: Web Clipper save creates/updates note and clipper provenance.
- Backend or client test: sidepanel quick-save persists source URL/origin through supported storage.
- Component test: captured/inbox filter shows captured note and excludes ordinary note.
- Existing `WebClipperPanel.save-flow.test.tsx` remains green after payload changes.
- Browser/sidepanel check where feasible.

**Implementation steps:**
- [x] Create or update the Backlog subtask for PR 6.
- [x] Record the Inbox backing decision in the task.
- [x] Add failing tests for sidepanel quick-save provenance persistence.
- [x] Add failing tests for captured/inbox list filtering.
- [x] Implement the smallest durable provenance path.
- [x] Update notes list filter/view model.
- [x] Run frontend and backend focused tests.
- [x] Run Bandit for touched Python scope if backend code changes.
- [x] Run browser/sidepanel verification.
- [x] Update Backlog notes with verification.

**Verification notes:** Focused frontend tests passed. Browser smoke reached `/notes` on the local Next dev server and confirmed the Captured filter renders in the Notes sidebar. Pointer activation could not be completed because the broader page layout placed the notes list partly under the app/sidebar region (`Captured` button left edge at x=-13), causing surrounding elements to intercept clicks; this is a separate layout/accessibility issue for PR 9 rather than the PR 6 data-model change. No Python files were touched, so Bandit was not applicable.

---

## PR 7: Destination Pickers For Capture

**Goal:** Replace raw destination IDs in capture flows where the app can discover valid destinations.

**Likely files:**
- Modify: `apps/packages/ui/src/components/Sidepanel/Clipper/ClipDestinationFields.tsx`
- Modify: `apps/packages/ui/src/components/Sidepanel/Clipper/WebClipperPanel.tsx`
- Modify: `apps/packages/ui/src/services/tldw/domains/workspace-api.ts`
- Modify/add: notes folder API client only if a public notes-folder API exists or is added
- Backend optional: notes-folder listing/create endpoint if missing and approved
- Test: `apps/packages/ui/src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx`

**Acceptance criteria:**
- Workspace destination uses a picker/search when workspace list data is available.
- Folder destination uses a picker only after verifying or adding a notes-folder list/create API.
- Invalid destination is prevented before submit where possible.
- Raw ID fallback, if retained, is clearly secondary/advanced.
- Existing save-to-note-only flow remains fast.

**Pre-implementation gate:**
- Verify whether a public notes-folder list/create endpoint exists. If not, split this into PR 7A workspace picker and PR 7B notes-folder API plus picker.

**Decision:** Split PR 7. PR 7A implements the workspace destination picker because `GET /api/v1/workspaces` exists and the frontend workspace client only needed a list method. Notes folders are still exposed to the Web Clipper service through `note_folders`/`note_folder_memberships`, but no public `/api/v1/notes/folders` list/create endpoint was found; keep the folder raw-ID fallback for now and handle folder API plus picker in PR 7B.

**Test plan:**
- Component test: workspace picker loads, selects, and submits a workspace ID.
- Component test: invalid/missing required destination blocks save with accessible error text.
- Component test: note-only save does not require workspace/folder selection.
- Backend tests only if adding folder API.
- Browser check: capture panel destination selection.

**Implementation steps:**
- [x] Create or update the Backlog subtask for PR 7.
- [x] Verify destination APIs and record the split/no-split decision.
- [x] Add failing picker tests for available destination type.
- [x] Implement picker UI and payload mapping.
- [x] Preserve or remove raw ID tests according to the approved UI.
- [x] Run focused tests and assess browser check feasibility.
- [x] Run Bandit if backend folder API is added.
- [x] Update Backlog notes with verification.

**Verification notes:** Focused frontend tests passed for `tldw-api-client.workspace-api.test.ts`, `WebClipperPanel.save-flow.test.tsx`, and `sidepanel-clipper.test.tsx` with 31 tests total. Browser verification is marked needs-verification for PR 7A because the changed surface is the browser-extension sidepanel/Web Clipper component rather than a routable `/notes` WebUI page in this worktree. No Python files were touched, so Bandit was not applicable.

---

## PR 8: Connections And Cross-Surface Link Clarity

**Goal:** Make existing note relationships understandable and navigable without expanding the object model.

**Likely files:**
- Modify: `apps/packages/ui/src/components/Notes/NotesEditorPane.tsx`
- Modify: `apps/packages/ui/src/components/Notes/NotesGraphModal.tsx`
- Modify: `apps/packages/ui/src/components/Notes/NotesManagerPage.tsx`
- Modify: `apps/packages/ui/src/services/tldw/domains/collections.ts`
- Backend optional: `tldw_Server_API/app/api/v1/endpoints/notes_graph.py` if labels/data are missing
- Test: notes graph/link tests and API tests if backend changes

**Acceptance criteria:**
- Manual note links, backlinks, graph edges, chat/message backlinks, source/clipper links, and reading-item links use understandable labels.
- Users can navigate from a note to a linked target where route support exists.
- Missing, deleted, or inaccessible linked targets show a clear unavailable state.
- Edge type labels distinguish manual links, backlinks, tags, and source membership.
- No media/research/prompt first-class link support is added.

**Test plan:**
- Component test: manual note link appears and opens linked note.
- Component test: missing linked target renders unavailable state.
- Component test: graph edge labels are user-readable.
- API test: neighbors endpoint returns expected edge types if backend changes.
- Browser smoke: create link -> reload note -> navigate via connection.

**Implementation steps:**
- [x] Create or update the Backlog subtask for PR 8.
- [x] Add failing tests for unclear/missing link labels or broken target states.
- [x] Improve connection labels and route affordances.
- [x] Add unavailable/deleted target UI state.
- [x] Run focused tests and assess browser smoke feasibility.
- [x] Run Bandit if backend code changes.
- [x] Update Backlog notes with verification.

**Verification notes:** Focused frontend connection tests passed for `NotesGraphModal.stage2.graph-view.test.tsx`, `NotesManagerPage.stage6.manual-links.test.tsx`, `NotesManagerPage.stage7.wikilinks.test.tsx`, `NotesManagerPage.stage26.backlink-labels.test.tsx`, and `NotesManagerPage.stage27.source-links.test.tsx` with 22 tests total. A full Notes component sweep was also run: 66 of 67 files passed and 203 of 204 tests passed; the remaining deterministic failure is the unrelated `NotesManagerPage.stage10.ai-title.test.tsx` strategy dropdown option lookup (`LLM (quality)`) and is recorded as an existing/non-PR8 issue. Browser smoke is marked needs-verification for PR 8 because this slice used mocked graph/link payloads and did not start a live API/WebUI stack. No Python files were touched, so Bandit was not applicable.

---

## PR 9: Responsive And Accessibility Hardening

**Goal:** Make the completed notes workflow usable by keyboard, screen reader, and mobile users after the concrete workflow slices have landed.

**Likely files:**
- Modify: affected Notes components
- Modify: affected Sidepanel Clipper components
- Test: accessibility and keyboard-flow tests

**Acceptance criteria:**
- Keyboard-only user can create, edit, save, search/filter, tag, and recover from save error.
- Focus is managed after create, delete, modal open/close, save success, and save failure.
- Form controls have labels, error associations, and accessible names.
- Mobile layout keeps list, editor, and primary actions usable without overlap.
- Loading and reduced-motion states do not block task completion.
- This PR is a regression-hardening pass over known gaps from PRs 1-8, not a catch-all for unrelated redesign.

**Test plan:**
- React Testing Library keyboard tests for create/edit/save/search/tag.
- Axe or existing accessibility utility checks for notes page and clipper panel.
- Playwright/browser desktop and mobile screenshot checks.
- Manual keyboard pass recorded in Backlog notes.

**Implementation steps:**
- [x] Create or update the Backlog subtask for PR 9.
- [x] Add failing accessibility/keyboard tests for known gaps.
- [x] Fix focus, labels, error associations, and responsive constraints.
- [x] Run focused UI tests.
- [x] Assess browser desktop/mobile checks.
- [x] Update Backlog notes with verification.

**Verification notes:** Focused PR 9 tests passed for `NotesListPanel.stage18.accessibility-selected-state.test.tsx`, `NotesManagerPage.stage19.accessibility-skip-links.test.tsx`, `NotesManagerPage.stage20.accessibility-shortcuts.test.tsx`, `NotesManagerPage.stage21.accessibility-modal-focus.test.tsx`, `NotesManagerPage.stage22.accessibility-regression.test.tsx`, `NotesManagerPage.stage23.responsive-layout.test.tsx`, `NotesManagerPage.stage26.backlink-labels.test.tsx`, and `NotesManagerPage.stage27.source-links.test.tsx` with 29 tests total. A full Notes component sweep was also run: 66 of 67 files passed and 204 of 205 tests passed; the remaining deterministic failure is the unrelated `NotesManagerPage.stage10.ai-title.test.tsx` strategy dropdown option lookup (`LLM (quality)`). Browser desktop/mobile screenshot smoke is marked needs-verification because no live API/WebUI stack was started for this component-level accessibility hardening slice. No Python files were touched, so Bandit was not applicable.

---

## PR 10: Import, Export, And Offline Draft Sync

**Goal:** Ensure exposed import, export, and offline draft/sync workflows are understandable, reliable, and recoverable.

**Likely files:**
- Modify: `apps/packages/ui/src/components/Notes/hooks/useNotesImport.tsx`
- Modify: `apps/packages/ui/src/components/Notes/hooks/useNotesExport.tsx`
- Modify: `apps/packages/ui/src/components/Notes/hooks/useNotesEditorState.tsx`
- Modify: `apps/packages/ui/src/components/Notes/NotesSidebar.tsx`
- Modify: `apps/packages/ui/src/components/Notes/NotesEditorPane.tsx`
- Backend optional: `tldw_Server_API/app/api/v1/endpoints/notes.py` if import/export API behavior is unclear or broken
- Test: import, export progress, and offline sync tests

**Acceptance criteria:**
- Import flow explains accepted file types, duplicate handling, success, partial success, and failure.
- Export flow communicates scope, format, progress, partial failure, and download/print result.
- Offline draft status is visible without being noisy and distinguishes queued, syncing, synced, error, and conflict states.
- Offline drafts recover after reload and sync when online without overwriting newer remote content silently.
- Import/export/offline errors preserve user data and provide a clear next action.

**Test plan:**
- Component test: import submits `/api/v1/notes/import` with selected duplicate strategy and surfaces result summary.
- Component test: export progress updates across batches and clears when complete.
- Component test: export partial failure displays warning and leaves successful export available.
- Component test: offline save queues locally without hitting create/update endpoints.
- Component test: queued offline draft recovers after reload and syncs when online.
- Backend tests for `/api/v1/notes/import`, `/api/v1/notes/export`, and `/api/v1/notes/export.csv` only if backend behavior changes.

**Implementation steps:**
- [x] Create or update the Backlog subtask for PR 10.
- [x] Add failing tests for any missing import/export/offline feedback or recovery behavior.
- [x] Clarify import/export/offline copy and status presentation.
- [x] Fix data-loss or overwrite-risk behavior before visual polish.
- [x] Run focused import/export/offline tests.
- [x] Run backend tests and Bandit if `notes.py` changes.
- [x] Assess browser smoke for import/export/offline status.
- [x] Update Backlog notes with verification.

**Verification notes:** Focused PR 10 tests passed for `NotesManagerPage.stage30.export-progress.test.tsx`, `NotesManagerPage.stage31.single-note-export-copy.test.tsx`, `NotesManagerPage.stage35.export-preflight.test.tsx`, `NotesManagerPage.stage36.import-workflow.test.tsx`, `NotesManagerPage.stage37.print-export.test.tsx`, and `NotesManagerPage.stage41.offline-drafting-sync.test.tsx` with 13 tests total. A full Notes component sweep was also run: 66 of 67 files passed and 205 of 206 tests passed; the remaining deterministic failure is the unrelated `NotesManagerPage.stage10.ai-title.test.tsx` strategy dropdown option lookup (`LLM (quality)`). Browser smoke remains needs-verification because no live API/WebUI stack was started for this component-level import guard slice. No Python files were touched, so backend tests and Bandit were not applicable.

---

## PR 11: Power-User Workflow Polish

**Goal:** Speed up repeated note workflows after reliability, saving, capture, and accessibility are stable.

**Likely files:**
- Modify: `apps/packages/ui/src/components/Notes/NotesManagerPage.tsx`
- Modify: `apps/packages/ui/src/components/Notes/NotesEditorPane.tsx`
- Modify: relevant hooks/client files
- Test: shortcut, repeated-create, and bulk-flow tests

**Acceptance criteria:**
- Fast create is available from `/notes` without losing current context.
- Search/filter state is predictable and restorable if existing routing/state patterns support it.
- Repeated create/save/tag flows do not require unnecessary pointer movement.
- Bulk tag or repeated tag workflow is available only if selection state is reliable.
- Shortcuts do not fire while typing in text inputs/editors.

**Test plan:**
- Component test: shortcut opens quick-create only outside text entry.
- Component test: repeated create/save preserves list state.
- Component test: bulk tag action applies to selected notes if implemented.
- E2E/browser smoke: create -> tag -> search/filter -> reopen/edit.

**Implementation steps:**
- [x] Create or update the Backlog subtask for PR 11.
- [x] Add failing tests for chosen power-user affordance.
- [x] Implement one focused speed improvement at a time.
- [x] Verify shortcut conflicts and keyboard accessibility.
- [x] Run focused tests; browser smoke needs verification in an app/backend environment.
- [x] Update Backlog notes with verification.

**PR 11 implementation notes:**
- Added `Alt + Shift + D` to duplicate the current note or unsaved draft into a new dirty draft, reusing the existing duplicate action and preserving the global shortcut guard for text inputs/editors.
- Added shortcut-help copy for the duplicate shortcut.
- Verification:
  - Red: `bunx vitest run src/components/Notes/__tests__/NotesManagerPage.stage38.productivity-extensions.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage20.accessibility-shortcuts.test.tsx` failed for missing duplicate shortcut behavior and missing help-modal copy.
  - Green: same command passed 2 files / 9 tests.
  - Focused regression: `bunx vitest run src/components/Notes/__tests__/NotesManagerPage.stage1.editor-reliability.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage16.bulk-actions.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage20.accessibility-shortcuts.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage38.productivity-extensions.test.tsx` passed 4 files / 18 tests.
  - Full Notes sweep: `bunx vitest run src/components/Notes/__tests__` reported 66 files passed and 1 known unrelated failure in `NotesManagerPage.stage10.ai-title.test.tsx` (`LLM (quality)` selector not found), with 206 / 207 tests passing.
  - Browser smoke: Needs verification; this frontend-only worktree did not have a running backend/app browser-smoke harness for `/notes`.
  - Bandit: Not applicable; no Python/backend files changed.

---

## Priority And Dependencies

**Fix first:**
1. PR 1: Backend Search And Keyword API Contracts
2. PR 2: Notes List Reliability And Empty States
3. PR 3: Save State And Error Recovery
4. PR 6: Capture Provenance And Inbox View

These are trust and data-integrity slices. Do them before spending time on polish.

**Then:**
5. PR 4: First-Time Notes UX
6. PR 5: Tags Terminology And Organization Semantics
7. PR 10: Import, Export, And Offline Draft Sync
8. PR 7: Destination Pickers For Capture

These improve comprehension and remove raw-ID friction once core behavior is reliable.

**Can wait:**
9. PR 8: Connections And Cross-Surface Link Clarity
10. PR 9: Responsive And Accessibility Hardening
11. PR 11: Power-User Workflow Polish

These are valuable, but they depend on stable save/list/capture behavior.

## Product Clarifications To Resolve Before Related PRs

- Inbox backing model: provenance view, reserved tag, or both.
- Notes-folder destination API: existing public API, new endpoint, or defer folder picker.
- Structured media/research/prompt links: separate discovery/design task if desired.
- Existing captured-note backfill: no migration by default; decide separately if users need old captures surfaced in Inbox.

## Verification Commands

Use the project-specific package/test commands discovered in the target branch. Likely focused commands include:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Notes_NEW/integration/test_notes_api.py tldw_Server_API/tests/ChaChaNotesDB/test_chachanotes_db.py -k "search or keyword" -v
```

```bash
bunx vitest run apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage11.search-filtering.test.tsx apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage14.sorting-pagination.test.tsx
```

```bash
bunx vitest run apps/packages/ui/src/components/Sidepanel/Clipper/__tests__/WebClipperPanel.save-flow.test.tsx
```

```bash
bunx vitest run apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage20.accessibility-shortcuts.test.tsx apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage21.accessibility-modal-focus.test.tsx apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage23.responsive-layout.test.tsx
```

```bash
bunx vitest run apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage36.import-workflow.test.tsx apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage30.export-progress.test.tsx apps/packages/ui/src/components/Notes/__tests__/NotesManagerPage.stage41.offline-drafting-sync.test.tsx
```

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests -k "notes or web_clipper" -v
```

```bash
source .venv/bin/activate
python -m bandit -r tldw_Server_API/app/api/v1/endpoints tldw_Server_API/app/core/WebClipper tldw_Server_API/app/core/DB_Management -f json -o /tmp/bandit_notes_ux.json
```

Adjust paths to the actual touched files for each PR. Document skipped commands and baseline failures in the Backlog task.
