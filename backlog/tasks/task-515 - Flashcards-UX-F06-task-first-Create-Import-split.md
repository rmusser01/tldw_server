---
id: TASK-515
title: Flashcards UX F06 task-first Create & Import split
status: Done
labels:
- ux
- flashcards
- webui
priority: medium
modified_files:
- apps/packages/ui/src/components/Flashcards/tabs/ImportExportTab.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.decomposition.test.tsx
- Flashcards-UX-Fix-List.md
- backlog/tasks/task-515 - Flashcards-UX-F06-task-first-Create-Import-split.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the deferred F06 WebUI-only flashcards improvement: make the Create & Import tab task-first so users can choose between creating/generating cards, importing content, and exporting/backup without unrelated controls competing at once. Preserve existing direct flashcards workflows and avoid extension-native F12 scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Create & Import presents task-specific navigation for Create cards, Import file, and Export backup.
- [x] #2 Existing generate and export handoffs open the relevant task without changing route keys or backend APIs.
- [x] #3 Transfer summary remains visible and reflects actions from all task panels.
- [x] #4 Focused component tests cover task switching and handoff defaults.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 1: add red component tests for task-first grouping and handoff defaults. Stage 2: update ImportExportTab only, preserving panel APIs. Stage 3: update master checklist and run focused verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the F06 WebUI task-first split for Create & Import. The tab now has Create cards, Import file, and Export backup task controls, keeps the existing transfer summary visible, preserves existing Generate/Import/Export panel APIs and handoffs, and opens export automatically for deck export handoffs. Added focused decomposition tests for default task state, switching, export handoff, and generated-card handoff. Updated Flashcards-UX-Fix-List.md to mark F06 complete while keeping native extension deck-picker/save deferred. PR review follow-up replaced root `space-y-4` spacing with `flex flex-col gap-4` and switched task-panel visibility to Tailwind `hidden` classes instead of HTML `hidden` attributes. Verification: red decomposition tests failed before implementation; passing checks include bunx vitest run src/components/Flashcards/tabs/__tests__/ImportExportTab.decomposition.test.tsx and the adjacent ImportExport suite of 40 tests across decomposition/import-results/deck-creation/llm-gating. Browser verification on http://127.0.0.1:3001/flashcards confirmed the Create & Import task selector and that Export backup hides the Create and Import task panels; review follow-up browser verification confirmed create/import panels render with `class="hidden"` and `display: none` while export remains visible. git diff --check passes. Non-blocking baseline notes: bun run verify:design-system-state still fails on unrelated product-state baseline entries outside touched flashcards files; NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false still fails on pre-existing CharacterListContent and sidepanel-flashcards test errors, with no touched-file diagnostics. Bandit is not applicable because this slice changed TSX/Markdown/Backlog files only.
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
