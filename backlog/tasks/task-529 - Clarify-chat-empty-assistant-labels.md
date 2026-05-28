---
id: TASK-529
title: Clarify chat empty assistant labels
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-27 19:53'
labels:
  - chat
  - ux
  - a11y
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address /chat UX rebaseline F9 by qualifying empty assistant labels by region so desktop cockpit scanning and screen-reader output do not repeat the same generic No assistant selected phrase across composition and runtime surfaces. Keep behavior limited to copy/label clarity, not assistant state ownership.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The next-message composition preview uses a region-specific empty assistant title instead of the generic No assistant selected copy.
- [x] #2 The runtime assistant rail uses a runtime-specific empty assistant title while preserving the existing detail and Select character or persona action.
- [x] #3 Regression coverage proves the generic No assistant selected phrase is no longer repeated across composition and runtime in the empty state.
- [x] #4 Existing selected character/persona behavior remains unchanged.
- [x] #5 Focused /chat a11y/composition/runtime tests pass and verification/known skips are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Starting focused /chat F9 copy/a11y slice. Investigation found the empty assistant phrase No assistant selected still appeared in composition preview and runtime rail paths.

Implemented region-specific empty assistant labels: composition uses No assistant attached to next message, runtime uses No runtime assistant selected, and old generic detail text is suppressed or replaced with the existing explanatory detail.

Verification: RED focused run failed as expected on the generic labels. GREEN focused run passed 65 tests across playground-composition-preview, PlaygroundCompositionPreview, PlaygroundRuntimeInspector.first-slice, Playground.cockpit-a11y, and Playground.cockpit-controls. Final diff check passed. UI tsc remains blocked by the known unrelated CharacterListContent.design-system.test.tsx GalleryCardDensity baseline mismatch. Bandit skipped because touched code is TS/TSX UI only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Clarified empty assistant state copy in /chat cockpit rails. The next-message composition preview and runtime rail now use region-specific empty labels while preserving selected assistant behavior and the existing explanatory detail.
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
