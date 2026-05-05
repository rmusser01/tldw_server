---
id: TASK-78
title: Stabilize Flashcards ManageTab Vitest mocks
status: Done
assignee: []
created_date: '2026-05-05 17:11'
updated_date: '2026-05-05 17:34'
labels:
  - frontend
  - tests
  - tldw-frontend
  - vitest
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the local tldw-frontend test stabilization series after PR #1311 merged. Use a fresh dev-based worktree and keep this slice narrow: reproduce the next repeated Vitest failure cluster, expected to be stale Flashcards ManageTab mocks around deck mutation hooks, fix the smallest test-harness surface, and leave unrelated broad-suite failures for later slices.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fresh dev-based worktree records current tldw-frontend TypeScript baseline before edits
- [x] #2 Flashcards ManageTab Vitest failure is reproduced with a focused command before edits
- [x] #3 Root cause is documented from the failing mock/export path
- [x] #4 Smallest scoped mock or test-harness fix is implemented
- [x] #5 Focused Flashcards Vitest command passes after the fix
- [x] #6 tldw-frontend TypeScript and git diff checks pass after the fix
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Run tldw-frontend TypeScript on the fresh origin/dev worktree to confirm the baseline remains green. 2. Run the focused Flashcards ManageTab Vitest file(s) to reproduce the stale-mock failure and capture the exact missing export or runtime error. 3. Inspect the production hook/store exports and nearby working tests to identify the correct mock contract. 4. Patch only the affected Flashcards test mock or setup surface. 5. Re-run focused Vitest, tldw-frontend TypeScript, and git diff --check; record any unrelated broad-suite failures as notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Baseline after dependency setup: tldw-frontend TypeScript passed with node node_modules/typescript/bin/tsc --noEmit --pretty false -p tsconfig.json. Focused ManageTab Vitest reproduced the next repeated cluster: document-editing, document-mode, and undo-stage3 failed before assertions because their full ../../hooks mocks omitted useUpdateDeckMutation. ManageTab imports and calls useUpdateDeckMutation unconditionally at render; scheduling-metadata already mocks it and passed, confirming the intended test-harness contract.

Fix implemented: added useUpdateDeckMutation to the three stale ManageTab hook mocks, added missing FlashcardMarkdownSnippet component mocks for document-mode/document-editing, added useGlobalFlashcardTagSuggestionsQuery to undo-stage3 because it intentionally renders the real edit drawer, and reset service mutation/query mocks between undo-stage3 tests to prevent queued mock version leakage. Verification: focused ManageTab Vitest passed 4 files / 23 tests; tldw-frontend TypeScript passed; git diff --check passed. Bandit skipped because touched source files are TypeScript test files only.

Opened PR #1316: https://github.com/rmusser01/tldw_server/pull/1316.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stabilized the next tldw-frontend Vitest slice after PR #1311 merged. The affected Flashcards ManageTab tests were failing because their full mocks had drifted behind the current ManageTab/rendered child component contract: useUpdateDeckMutation was missing, document-mode mocks omitted FlashcardMarkdownSnippet, and undo-stage3 rendered the real edit drawer without the global tag-suggestions hook mock. The undo-stage3 service mocks now reset between tests so one-shot getFlashcard versions cannot leak across assertions.

Verification: focused ManageTab Vitest passed 4 files / 23 tests; tldw-frontend TypeScript passed; git diff --check passed. Bandit was skipped because this slice only touched frontend TypeScript tests. PR: https://github.com/rmusser01/tldw_server/pull/1316.
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
