---
id: TASK-523
title: Fix sidepanel chat narrow-width overflow
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-27 18:09'
labels:
  - chat
  - extension
  - ux
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent horizontal overflow in the browser-extension sidepanel chat workflow at typical 390px sidepanel width, especially around the /chat handoff surface and composer controls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At 390px sidepanel width, sidepanel /chat does not create document-level horizontal overflow.
- [x] #2 Primary sidepanel chat header, message rail, and composer controls remain reachable without clipped controls.
- [x] #3 Regression coverage exercises the narrow sidepanel width.
- [x] #4 Focused sidepanel chat tests pass.
- [x] #5 Verification and known skips are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Root cause: the sidepanel chat shell and composer used nested flex children without min-w-0 and the control/action rows did not consistently wrap, so min-content width could exceed the 390px sidepanel viewport.

Implementation: constrained the shell, main region, dropzone, message rail, sticky composer, form card, textarea shell, and action groups with min-w-0 and overflow-safe flex wrapping while preserving existing controls and labels.

Verification: RED contract failed before layout fixes, then passed after implementation. Focused sidepanel Vitest suite passed with 8 files and 25 tests. Targeted extension Playwright 390px layout test passed. git diff --check passed. TypeScript still fails on the pre-existing CharacterListContent GalleryCardDensity baseline outside this slice. Bandit skipped because no Python files changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the extension sidepanel /chat narrow-width overflow by constraining the shell, message rail, sticky composer, form, and control rows with min-width and overflow-safe flex behavior. Added a source contract test for the narrow layout and an extension Playwright regression that validates a 390px viewport has no document-level horizontal overflow. Verification: focused sidepanel Vitest suite passed (8 files, 25 tests); targeted extension Playwright 390px case passed; git diff --check passed. TypeScript still stops on the pre-existing CharacterListContent GalleryCardDensity baseline outside this slice. Bandit skipped because only TypeScript/Playwright/Backlog files changed.
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
