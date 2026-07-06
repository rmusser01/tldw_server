---
id: TASK-12903
title: Full notes page UAT and root-cause fixes
status: Done
labels:
- uat
- webui
- notes
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run full UAT on the WebUI notes page against the live backend, identify user-facing functional/visual issues, and patch root causes with focused regression coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-06-notes-page-uat-fixes-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created isolated worktree .worktrees/codex-notes-page-uat-fixes from origin/main for UAT and patches.

Live UAT setup:
- Backend: FastAPI on `127.0.0.1:8000` using temporary UAT profile and live llama.cpp-compatible server at `127.0.0.1:9099`.
- WebUI: Next dev server on `http://localhost:8080`, advanced mode with live API URL/key.
- No mock backend used.

Confirmed root cause:
- Notes first-visit Joyride tour auto-started while users/tests were already in the delete flow. The Joyride overlay sat above Ant Design dropdown/menu/modal surfaces and intercepted clicks on More Actions -> Delete and the delete confirmation.
- The desktop `Create study pack` action was absolutely positioned at the notes page level, so it could float over the editor header instead of participating in the toolbar layout.

Patch:
- The first-visit notes tutorial no longer auto-starts after the user has already interacted with the page; the page marks the prompt as seen and cancels the pending timer.
- Notes confirmation calls now close any active tutorial before showing blocking dialogs.
- Notes editor overflow trigger closes any active tutorial before opening the menu.
- Shared confirm-danger wrapper now uses AntD 6 `focusable.autoFocusButton`, removing the console warning from notes confirm dialogs.
- The `Create study pack` action now lives inside the editor header action row using the existing header props, eliminating the desktop overlap while preserving mobile behavior.

Verification:
- `bunx vitest run src/components/Notes/__tests__/NotesManagerPage.stage1.editor-reliability-followup.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage20.accessibility-shortcuts.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage23.responsive-layout.test.tsx src/components/Notes/__tests__/NotesManagerPage.stage33.link-aware-delete-warning.test.tsx src/components/Notes/__tests__/NotesEditorHeader.stage2.touch-layout.test.tsx` passed 22/22.
- Live `npx playwright test e2e/workflows/tier-1-critical/notes.spec.ts --reporter=line` passed 4/4 against `127.0.0.1:8000` and `http://localhost:8080`.
- Live `/private/tmp/notes-uat-sweep.mjs` passed 16/16 against `http://localhost:8080` with no failing API events or console events.
- Screenshot evidence saved outside the repo at `/private/tmp/notes-uat-sweep-shots-final-4`.
- Bandit skipped because only TypeScript/TSX frontend files were touched.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the notes-page guided-tour overlay so automatic first-visit tutorial startup cannot interrupt active work, active tours are closed before overflow/destructive confirmation flows, and the desktop study-pack action no longer overlaps the editor header. Verified the notes page against the live backend with create/save/search/select, editor modes, formatting, keyboard shortcut save, shortcut help, pinning, view switches, tag filter, trash delete/restore, export menu, AI title suggestion endpoint, enabled assist controls, and mobile browse drawer.
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
