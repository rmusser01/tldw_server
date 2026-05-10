---
id: TASK-203
title: Implement personal Persona Visual pack library foundation
status: In Progress
assignee: []
created_date: '2026-05-09 23:41'
updated_date: '2026-05-10 00:24'
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
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created implementation plan for the reference-backed personal Persona Visual pack library. Focused baseline backend tests passed before implementation: 50 passed across persona visual DB/service/API suites.

Stage 1 persistence foundation implemented. Schema v46 creates persona_visual_library_items and DB helpers now upsert list get update and soft-delete user-scoped entries while preserving stale source rows as unavailable. Verification: DB library test 4 passed. Persona visual DB service API suite 54 passed. git diff --check clean. Bandit on touched production files reported 0 results.

Stage 2 service and API implemented. Added PersonaVisualLibraryService and REST endpoints to save list update delete and use personal visual library items. Using a library item duplicates the source pack to the target persona as a draft and keeps activation explicit. Verification: persona visual focused backend suite 62 passed. git diff --check clean. Bandit on touched Stage 2 production files reported 0 results.

Stage 3 WebUI library panel implemented. Added TypeScript types and service functions for list save update delete and use. VisualPackEditor now loads the personal library, saves the selected pack, shows available source-changed and unavailable states, edits and removes entries, and uses entries as draft copies for target personas. Verification: RED and GREEN component tests run in apps/packages/ui with ./node_modules/.bin/vitest run src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx; final result 19 passed. Root bunx vitest form still uses a transient runner and misses the UI package alias config in this isolated worktree. git diff --check clean.

Final verification for TASK-203: backend library/API suite 41 passed; existing persona visual DB/service regression suite 21 passed; VisualPackEditor focused Vitest 19 passed; Bandit on touched backend production files wrote /tmp/bandit_persona_visual_library.json with results []; git diff --check clean. No service-specific frontend unit test file exists for persona-visuals, so frontend service coverage is through VisualPackEditor authenticated-client mocks.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the personal Persona Visual pack library foundation for issue #1468. The backend adds a user-scoped reference-backed library table, DB helpers, service layer, and REST endpoints for save/list/update/delete/use. The WebUI adds typed service helpers plus a Personal library panel in VisualPackEditor for saving the selected pack, showing available/source-changed/unavailable states, editing/removing entries, and using entries as draft copies on target personas. Docs now describe V1 reference-backed behavior and non-goals. Verification covered focused backend suites, visual regressions, focused frontend Vitest, Bandit, and diff checks.
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
