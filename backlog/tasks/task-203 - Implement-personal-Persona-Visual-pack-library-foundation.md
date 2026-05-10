---
id: TASK-203
title: Implement personal Persona Visual pack library foundation
status: In Progress
assignee: []
created_date: '2026-05-09 23:41'
updated_date: '2026-05-10 01:05'
labels:
  - persona
  - buddy
  - webui
dependencies:
  - TASK-201
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1468'
  - 'https://github.com/rmusser01/tldw_server/issues/1449'
documentation:
  - Docs/superpowers/specs/2026-05-09-persona-visual-personal-library-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1468: a reference-backed, user-scoped personal library for Persona/Buddy visual packs. The first implementation should add persistence, service/API behavior, WebUI affordances in the Visuals editor, and docs while preserving duplicate-to-persona draft semantics and explicit activation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Library entries are user-scoped metadata references to existing same-user persona visual packs.
- [x] #2 Saving a pack to the library is idempotent for the same user/source persona/source pack and does not mutate source assets or active pack state.
- [x] #3 Using a library item for another persona creates a draft through duplicate-to-persona semantics and does not activate it.
- [x] #4 Stale source entries list as unavailable, can be removed, and cannot be used.
- [x] #5 The WebUI exposes save/list/edit/remove/use affordances in the existing Persona Visuals flow.
- [x] #6 Docs and tracker notes explain reference-backed V1 behavior and non-goals.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation plan: Docs/superpowers/plans/2026-05-09-persona-visual-personal-library-implementation-plan.md

Review-fix pass for PR #1482: add regressions for metadata-preserving re-save, library visibility/use when a persona has no local packs, forbidden error mapping, and duplicate-key recovery in library upsert; then patch the UI/backend/docs-tracking accordingly. Keep source display snapshots because they are stale-entry display metadata required for removability, not copied asset snapshots; verify and respond to that false-positive review thread. Verify ChaChaNotes_DB method delegation and respond to that false-positive review thread instead of duplicating delegated implementations.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created implementation plan for the reference-backed personal Persona Visual pack library. Focused baseline backend tests passed before implementation: 50 passed across persona visual DB/service/API suites.

Stage 1 persistence foundation implemented. Schema v46 creates persona_visual_library_items and DB helpers now upsert list get update and soft-delete user-scoped entries while preserving stale source rows as unavailable. Verification: DB library test 4 passed. Persona visual DB service API suite 54 passed. git diff --check clean. Bandit on touched production files reported 0 results.

Stage 2 service and API implemented. Added PersonaVisualLibraryService and REST endpoints to save list update delete and use personal visual library items. Using a library item duplicates the source pack to the target persona as a draft and keeps activation explicit. Verification: persona visual focused backend suite 62 passed. git diff --check clean. Bandit on touched Stage 2 production files reported 0 results.

Stage 3 WebUI library panel implemented. Added TypeScript types and service functions for list save update delete and use. VisualPackEditor now loads the personal library, saves the selected pack, shows available source-changed and unavailable states, edits and removes entries, and uses entries as draft copies for target personas. Verification: RED and GREEN component tests run in apps/packages/ui with ./node_modules/.bin/vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx; final result 19 passed. Root bunx vitest form still uses a transient runner and misses the UI package alias config in this isolated worktree. git diff --check clean.

Final verification for TASK-203: backend library/API suite 41 passed; existing persona visual DB/service regression suite 21 passed; VisualPackEditor focused Vitest 19 passed; Bandit on touched backend production files wrote /tmp/bandit_persona_visual_library.json with results []; git diff --check clean. No service-specific frontend unit test file exists for persona-visuals, so frontend service coverage is through VisualPackEditor authenticated-client mocks.

PR tracking: draft PR #1482 opened at https://github.com/rmusser01/tldw_server/pull/1482. Issue #1468 was updated with PR and verification link. The overarching #1449 tracker was updated with the active PR/workstream link in https://github.com/rmusser01/tldw_server/issues/1449#issuecomment-4414095201; its completion checklist should move only after PR #1482 merges.

PR #1482 review-fix pass completed. Fixed metadata-preserving re-save in both WebUI and API by preserving omitted library metadata for existing source entries while still allowing explicit clears. Moved the Personal library panel outside selected-pack-only rendering and loaded duplicate/library targets without requiring a local pack, so empty target personas can use saved library items. Added forbidden-to-403 mapping, a module docstring, deterministic duplicate-key race recovery for library upsert, and a source lookup helper used by the save endpoint. Updated #1449 with active PR tracking. Verification after fixes: backend focused suite 43 passed; VisualPackEditor Vitest 20 passed; Bandit touched backend production files results []; git diff --check clean.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the personal Persona Visual pack library foundation for issue #1468. The backend adds a user-scoped reference-backed library table, DB helpers, service layer, and REST endpoints for save/list/update/delete/use. The WebUI adds typed service helpers plus a Personal library panel in VisualPackEditor for saving the selected pack, showing available/source-changed/unavailable states, editing/removing entries, and using entries as draft copies on target personas. Docs now describe V1 reference-backed behavior and non-goals. Verification covered focused backend suites, visual regressions, focused frontend Vitest, Bandit, and diff checks.

Review follow-up addressed PR comments: metadata-preserving re-save now works in WebUI and API, empty target personas can use library items, upsert handles duplicate-key races idempotently, forbidden library errors map to 403, #1449 has active tracker linkage, and false-positive snapshot/delegation review threads were documented for response.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
