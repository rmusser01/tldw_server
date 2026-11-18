---
id: TASK-15569
title: Restore media selection in the Study Pack drawer
status: In Progress
labels:
- flashcards
- webui
references:
- https://github.com/rmusser01/tldw_server/pull/3504
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Flashcards Study Pack drawer only accepts manual source IDs or handoffs. A user creating a pack from Flashcards should be able to browse and search their media library and submit selected items by ID.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The picker lists recent media and searches the current media library.
- [x] #2 Selecting a result submits its actual media ID and title. Manual source IDs and handoffs continue to work.
- [x] #3 Regressions cover browse, search, selection-to-submit, and request failure with retry.
- [x] #4 Document the new picker and link the dev PR.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. The picker uses existing media APIs, Study Pack source references, and the Flashcards account lifecycle. Both new tests reproduced the missing picker on public dev b7d0b2d and passed after the fix. The 62 focused tests, all 857 Flashcards and Media search tests, and the WebUI typecheck passed on the frozen dev lockfile. The guide documents creating a Study Pack from Media. Bandit has no applicable code because the implementation and tests are TypeScript. Live browser smoke is unverified. The draft PR requires a human-written Change summary before merge.
Owner requested independent review and follow-up corrections on PR #3504.
Addressed the independent review. The drawer now clears on extension account changes even when initial identity lookup failed. A failed keyword search keeps its term and can retry the same request. Duplicate media titles show their IDs, and a result-limit message shows how many items are available. Updated the guide to include Create cards. The new failure cases reproduced before the fixes and the 65 focused tests passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a searchable media picker and regression coverage for browsing, searching, selecting a source, and retrying after a library load failure. Preserved manual IDs and prefilled handoffs. Opened the dev pull request.
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
