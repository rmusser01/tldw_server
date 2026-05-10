# Persona Visual Personal Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement GitHub issue #1468 by adding a reference-backed personal Persona/Buddy visual-pack library.

**Architecture:** Add a user-scoped `persona_visual_library_items` metadata table in the ChaChaNotes persona store, then layer a small `PersonaVisualLibraryService` over existing pack persistence and `PersonaVisualService.duplicate_pack_to_persona(...)`. Expose REST endpoints under the existing persona API and add a compact Personal Library panel to `VisualPackEditor`; using a library item always creates a reviewed draft and never activates the target persona.

**Tech Stack:** FastAPI, Pydantic, ChaChaNotes SQLite/PostgreSQL migrations, `PersonaStateStore`, `PersonaVisualService`, pytest, React/TypeScript, Vitest, Ant Design, lucide-react.

---

## Stage 1: Persistence Foundation

**Goal:** Add durable library-entry storage and row helpers.

**Success Criteria:** Schema v46 creates `persona_visual_library_items`; DB helpers create, list, update, and soft-delete user-scoped entries; stale source refs are preserved as removable unavailable rows.

**Tests:** `tldw_Server_API/tests/ChaChaNotesDB/test_persona_visual_library_db.py`

**Status:** Complete

### Task 1: Add failing DB migration and persistence tests

**Files:**
- Create: `tldw_Server_API/tests/ChaChaNotesDB/test_persona_visual_library_db.py`
- Modify later: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify later: `tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py`

- [x] **Step 1: Write migration test**

  Add a test that seeds schema v45, drops `persona_visual_library_items`, opens `CharactersRAGDB`, and asserts:
  - schema version is current
  - table exists
  - index `idx_persona_visual_library_items_user_time` exists
  - unique live-source index exists

- [x] **Step 2: Write create/list/upsert test**

  Create a source persona and pack, save a library entry twice for `user-1`, and assert:
  - only one active entry exists
  - title/notes/tags are updated on the second save
  - source persona/pack display metadata and snapshots are present
  - `source_available` is true and `source_changed` reflects version drift

- [x] **Step 3: Write stale-source and ownership tests**

  Add tests proving:
  - soft-deleted source packs list with `source_available: false`
  - stale entries can be soft-deleted
  - another user cannot list or update the entry

- [x] **Step 4: Run tests and verify RED**

  Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_persona_visual_library_db.py -q
  ```

  Expected: fail because schema/helpers do not exist.

### Task 2: Implement schema v46 and delegated DB helpers

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py`

- [x] **Step 1: Add SQLite and Postgres migrations**

  Bump `_CURRENT_SCHEMA_VERSION` to 46. Add `_MIGRATION_SQL_V45_TO_V46` and `_MIGRATION_SQL_V45_TO_V46_POSTGRES` with:
  - nullable `source_persona_id`
  - nullable `source_pack_id`
  - `source_persona_name_snapshot`
  - `source_pack_title_snapshot`
  - `ON DELETE SET NULL` FKs where supported
  - `(user_id, deleted, last_modified)` listing index
  - partial unique active live-source index for `(user_id, source_persona_id, source_pack_id)` where not deleted and refs are not null

- [x] **Step 2: Wire migration execution**

  Add v45-to-v46 execution to SQLite/Postgres migration paths.

- [x] **Step 3: Add row normalization helpers**

  In `PersonaStateStore`, add:
  - `_persona_visual_library_item_row_to_dict`
  - `_normalize_persona_visual_library_tags`
  - `_require_persona_visual_library_item_owner`

- [x] **Step 4: Add CRUD helpers**

  Add delegated methods:
  - `upsert_persona_visual_library_item(...)`
  - `list_persona_visual_library_items(...)`
  - `get_persona_visual_library_item(...)`
  - `update_persona_visual_library_item(...)`
  - `soft_delete_persona_visual_library_item(...)`

  Listing should left join live source persona/pack rows and derive:
  - `source_available`
  - `source_current_version`
  - `source_changed`
  - live display fields with snapshot fallbacks

- [x] **Step 5: Run DB test and verify GREEN**

  Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_persona_visual_library_db.py -q
  ```

  Expected: pass.

- [x] **Step 6: Commit persistence foundation**

  ```bash
  git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py tldw_Server_API/tests/ChaChaNotesDB/test_persona_visual_library_db.py
  git commit -m "Add persona visual library persistence"
  ```

## Stage 2: Service And API

**Goal:** Add library business logic and REST endpoints that reuse duplicate-to-persona draft semantics.

**Success Criteria:** The API can list, save, edit, delete, and use library entries; cross-user access is rejected; stale entries cannot be used.

**Tests:** `tldw_Server_API/tests/Persona/test_persona_visual_library_service.py`, `tldw_Server_API/tests/Persona/test_persona_visuals_api.py`

**Status:** Complete

### Task 3: Add failing service tests

**Files:**
- Create: `tldw_Server_API/tests/Persona/test_persona_visual_library_service.py`
- Create later: `tldw_Server_API/app/core/Persona/visual_library_service.py`

- [x] **Step 1: Write save/list metadata test**

  Assert `PersonaVisualLibraryService.save_pack(...)` saves a source pack with normalized metadata and does not mutate source pack status or assets.

- [x] **Step 2: Write use-library-item test**

  Create source and target personas with a valid source pack and asset. Save source pack to library, call `use_for_persona(...)`, and assert:
  - returned pack is a draft on the target persona
  - `parent_pack_id` points at source pack
  - active source/target packs are unchanged

- [x] **Step 3: Write stale/cross-user service tests**

  Assert stale source entries raise `source_pack_unavailable` on use and cross-user source/target attempts raise not-found style service errors.

- [x] **Step 4: Run tests and verify RED**

  Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_library_service.py -q
  ```

  Expected: fail because service module does not exist.

### Task 4: Implement service and API endpoints

**Files:**
- Create: `tldw_Server_API/app/core/Persona/visual_library_service.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/persona.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/persona.py`
- Modify: `tldw_Server_API/tests/Persona/test_persona_visuals_api.py`

- [x] **Step 1: Implement `PersonaVisualLibraryService`**

  Add service methods:
  - `save_pack(...)`
  - `list_items(...)`
  - `update_item(...)`
  - `delete_item(...)`
  - `use_item_for_persona(...)`

  The service should normalize title, notes, and tags; enforce item/source/target ownership through DB helpers; and call `PersonaVisualService.duplicate_pack_to_persona(...)` for use.

- [x] **Step 2: Add schemas**

  Add Pydantic models:
  - `PersonaVisualLibraryItemResponse`
  - `PersonaVisualLibrarySaveRequest`
  - `PersonaVisualLibraryUpdateRequest`
  - `PersonaVisualLibraryUseRequest`
  - `PersonaVisualLibraryListResponse`

  Validators should trim text, cap title/notes/tags, and reject empty titles/tags.

- [x] **Step 3: Add dependency and error mapping**

  In `persona.py`, add `get_persona_visual_library_service(...)` and map service codes:
  - `library_item_not_found` -> 404
  - `source_pack_not_found` -> 404
  - `source_pack_unavailable` -> 409
  - `target_persona_not_found` -> 404
  - `same_persona_target_unsupported` -> 400
  - `invalid_library_metadata` -> 422
  - `library_item_conflict` -> 409

- [x] **Step 4: Add endpoints**

  Add:
  - `GET /api/v1/persona/visual-library`
  - `POST /api/v1/persona/profiles/{persona_id}/visual-packs/{pack_id}/library`
  - `PATCH /api/v1/persona/visual-library/{item_id}`
  - `DELETE /api/v1/persona/visual-library/{item_id}`
  - `POST /api/v1/persona/visual-library/{item_id}/use`

- [x] **Step 5: Add API tests**

  Extend `test_persona_visuals_api.py` for:
  - save/list/update/delete happy path
  - duplicate use creates draft
  - stale source returns 409 on use but delete succeeds
  - cross-user item/source/target access returns 404

- [x] **Step 6: Run service/API tests**

  Run:

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_library_service.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q
  ```

  Expected: pass.

- [x] **Step 7: Commit service/API**

  ```bash
  git add tldw_Server_API/app/core/Persona/visual_library_service.py tldw_Server_API/app/api/v1/schemas/persona.py tldw_Server_API/app/api/v1/endpoints/persona.py tldw_Server_API/tests/Persona/test_persona_visual_library_service.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py
  git commit -m "Add persona visual library API"
  ```

## Stage 3: WebUI Library Panel

**Goal:** Add visible library affordances in the existing Persona Garden Visuals editor.

**Success Criteria:** Users can save the selected pack, view their personal library, edit/remove entries, and use an entry for another persona as a draft without leaving the Visuals tab.

**Tests:** `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`

**Status:** Complete

### Task 5: Add failing frontend service and editor tests

**Files:**
- Modify: `apps/packages/ui/src/types/persona-visuals.ts`
- Modify: `apps/packages/ui/src/services/persona-visuals.ts`
- Modify: `apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx`
- Modify: `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`

- [x] **Step 1: Add TypeScript types**

  Add `PersonaVisualLibraryItem`, save/update/use request types, and `PersonaVisualLibraryListResponse`.

- [x] **Step 2: Add service function tests through component mocks**

  Extend component tests to expect calls to:
  - `GET /api/v1/persona/visual-library`
  - `POST /api/v1/persona/profiles/{persona_id}/visual-packs/{pack_id}/library`
  - `PATCH /api/v1/persona/visual-library/{item_id}`
  - `DELETE /api/v1/persona/visual-library/{item_id}`
  - `POST /api/v1/persona/visual-library/{item_id}/use`

- [x] **Step 3: Add UI behavior tests**

  Add tests that verify:
  - selected pack can be saved to library
  - saved/source-changed/source-unavailable states are displayed
  - use-for-persona sends target payload and shows target handoff
  - remove entry updates the panel without deleting the pack

- [x] **Step 4: Run frontend test and verify RED**

  Run:

  ```bash
  bunx vitest run apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx
  ```

  Expected: fail because library UI/service functions do not exist.

### Task 6: Implement frontend service functions and panel

**Files:**
- Modify: `apps/packages/ui/src/types/persona-visuals.ts`
- Modify: `apps/packages/ui/src/services/persona-visuals.ts`
- Modify: `apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx`
- Modify: `apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`

- [x] **Step 1: Add service functions**

  Implement:
  - `listPersonaVisualLibraryItems`
  - `savePersonaVisualPackToLibrary`
  - `updatePersonaVisualLibraryItem`
  - `deletePersonaVisualLibraryItem`
  - `usePersonaVisualLibraryItem`

- [x] **Step 2: Add editor state and loaders**

  Add state for library items, loading/mutating item IDs, selected library target, and edit metadata.

- [x] **Step 3: Add save-to-library controls**

  In the selected-pack header area, add an icon+text button `Save to library`, saved/source-changed tags, and failure copy. Keep active pack copy separate.

- [x] **Step 4: Add Personal Library panel**

  Add a compact panel under Portability or beside it with:
  - item title
  - source persona/pack display fields
  - tags
  - unavailable/source-changed badges
  - target persona select
  - Use for persona
  - Edit details
  - Remove

  Use existing `duplicateTargets` for target persona options.

- [x] **Step 5: Wire use action to draft handoff**

  On success, set `lastDuplicatedPersonaId`, reuse the existing "Open target Visuals" affordance/copy, and do not merge the created pack into the current persona's pack list unless the target is the current persona, which V1 rejects.

- [x] **Step 6: Run frontend test and verify GREEN**

  Run:

  ```bash
  bunx vitest run apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx
  ```

  Expected: pass.

  Verification note: in the isolated worktree, the root `bunx vitest ...`
  command uses a transient runner that misses the UI package alias config.
  Green verification was run from `apps/packages/ui` with
  `./node_modules/.bin/vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`.

- [ ] **Step 7: Commit WebUI panel**

  ```bash
  git add apps/packages/ui/src/types/persona-visuals.ts apps/packages/ui/src/services/persona-visuals.ts apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx
  git commit -m "Add persona visual library editor"
  ```

## Stage 4: Documentation And Tracker Closeout

**Goal:** Update docs/tracker text and run focused verification.

**Success Criteria:** Docs describe reference-backed V1 behavior, GitHub issue #1468 has implementation evidence, and the Backlog task records verification.

**Tests:** Focused backend, frontend, Bandit, and diff checks.

**Status:** In Progress

### Task 7: Update docs and trackers

**Files:**
- Modify: `Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md`
- Modify: `Docs/Code_Documentation/Persona_Visual_Packs.md`
- Modify: `backlog/tasks/task-203 - Implement-personal-Persona-Visual-pack-library-foundation.md` via Backlog CLI
- GitHub issue comments: #1449, #1468

- [x] **Step 1: Update product PRD**

  Add a Phase 3 personal library note describing reference-backed, user-owned V1 library entries and non-goals.

- [x] **Step 2: Update code documentation**

  Add endpoint/service/schema notes under a Personal Library section.

- [x] **Step 3: Update Backlog task notes**

  Use `backlog task edit TASK-203 --append-notes ... --plain` with implementation and verification notes.

- [ ] **Step 4: Update GitHub issues**

  Comment on #1468 with summary and PR link. Update #1449 checklist only after PR is merged, not before.

### Task 8: Final verification and PR

**Files:** All touched files.

- [x] **Step 1: Run focused backend tests**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_persona_visual_library_db.py tldw_Server_API/tests/Persona/test_persona_visual_library_service.py tldw_Server_API/tests/Persona/test_persona_visuals_api.py -q
  ```

- [x] **Step 2: Run existing persona visual regression tests**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_persona_visuals_db.py tldw_Server_API/tests/Persona/test_persona_visual_service.py -q
  ```

- [x] **Step 3: Run frontend focused tests**

  ```bash
  bunx vitest run apps/packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx apps/packages/ui/src/services/__tests__/persona-visuals.test.ts
  ```

  If there is no service unit-test file, omit it and record why.

- [x] **Step 4: Run Bandit on touched backend code**

  ```bash
  /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py tldw_Server_API/app/core/Persona/visual_library_service.py tldw_Server_API/app/api/v1/endpoints/persona.py tldw_Server_API/app/api/v1/schemas/persona.py -f json -o /tmp/bandit_persona_visual_library.json
  ```

- [x] **Step 5: Run diff checks**

  ```bash
  git diff --check
  git status --short --branch
  ```

- [x] **Step 6: Update Backlog final summary**

  Check acceptance criteria and Definition of Done items via Backlog CLI, then add final summary.

- [ ] **Step 7: Commit docs/verification updates**

  ```bash
  git add Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md Docs/Code_Documentation/Persona_Visual_Packs.md "backlog/tasks/task-203 - Implement-personal-Persona-Visual-pack-library-foundation.md"
  git commit -m "Document persona visual library foundation"
  ```

- [ ] **Step 8: Push and open PR**

  ```bash
  git push -u origin codex/persona-visual-library-foundation
  gh pr create --repo rmusser01/tldw_server --base dev --head codex/persona-visual-library-foundation --title "Add personal Persona Visual pack library foundation" --body-file /tmp/persona_visual_library_pr.md --draft
  ```
