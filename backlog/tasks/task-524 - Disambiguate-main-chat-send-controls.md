---
id: TASK-524
title: Disambiguate main chat send controls
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-27 18:28'
labels:
  - chat
  - ux
  - accessibility
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the main /chat composer send control naming ambiguity from the UX rebaseline so primary send and the adjacent options trigger have distinct accessible names and fuzzy automation does not match both as Send.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Primary /chat submit action exposes a stable Send message accessible name when ready to send.
- [x] #2 Adjacent send/options trigger uses a distinct accessible name that does not also match Send.
- [x] #3 Regression coverage proves fuzzy Send role lookup only finds the primary submit action in the ready state.
- [x] #4 Focused chat send-control tests pass.
- [x] #5 Verification and known skips are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Root cause: the primary submit action was correctly named Send message, but the adjacent split-menu trigger used Open send options, so broad/fuzzy role queries for Send matched both controls and made automation and screen-reader scan output less distinct.

Implementation: kept the primary ready action as Send message and renamed the adjacent split trigger to Open message delivery options in WebUI /chat and sidepanel chat. This preserves the same visual affordance and menu behavior while separating the accessible names.

Verification: RED PlaygroundSendControl accessibility test and sidepanel compact-toolbar contract failed on the old Open send options label before implementation. Focused Vitest passed 5 files and 16 tests. git diff --check passed. Full UI tsc still fails on the pre-existing CharacterListContent GalleryCardDensity baseline outside this slice. Bandit skipped because no Python files changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Disambiguated the /chat send split control by keeping the primary ready action named Send message and renaming the adjacent options trigger to Open message delivery options in both main WebUI chat and directly connected sidepanel chat. Added a main PlaygroundSendControl accessibility regression proving fuzzy Send role lookup only finds the primary submit action, and extended the sidepanel compact-toolbar contract to forbid the old Open send options label. Verification: red tests failed on the old label before implementation; focused Vitest passed 5 files and 16 tests; git diff --check passed. TypeScript still stops on the pre-existing CharacterListContent GalleryCardDensity baseline outside this slice. Bandit skipped because only TypeScript/TSX and Backlog files changed.
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
