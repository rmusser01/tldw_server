---
id: TASK-15569
title: Restore media selection in the Study Pack drawer
status: In Progress
labels:
- flashcards
- webui
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
- [ ] #4 Document the new picker and link the dev PR.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. Use existing media APIs, source references, and Flashcards account lifecycle without changing any durable rule. Plan: reproduce the missing picker with UI regressions, add the picker and update the guide, verify the affected suites, and open a PR to dev.
Both new tests reproduced the missing picker on public dev b7d0b2d and passed after the fix. All 62 focused tests and the WebUI TypeScript check passed on the frozen dev lockfile. The broader Flashcards suite is running.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
