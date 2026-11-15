---
id: TASK-15501
title: Stop retrying unavailable saved chat history
status: In Progress
labels:
- chat
- webui
references:
- https://github.com/rmusser01/tldw_server/pull/3474
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Opening a deleted chat by its saved URL repeats the failed history capture, leaving the center of the chat page refreshing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Missing saved chat history settles after the first failed capture.
- [x] #2 Deliberate Retry or selecting another chat still works.
- [x] #3 A failed versioned chat DELETE does not read back or retry the chat unless the server confirms a conflict.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no; this fixes retry behavior at the existing chat loader and preserves the history authorization contract.
Red: opening a missing chat through the real history controller retried the rejected capture on rerender (2 requests instead of 1). Green: 111 focused WebUI tests passed, including the new regression. The loader suppresses only ambient retries of a failed capture for the same conversation; explicit selection and existing authorization checks still apply.
The shared client mistook the expected_version query parameter in a network error for a conflict and hid the original DELETE failure behind a follow-up GET. The added transport control accepts the same bodyless 204 in the open-source client. Final focused matrix: 184 WebUI tests; eight backend Trash and lifecycle controls.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stops repeated requests for a failed saved chat history capture and prevents network failures on a versioned DELETE from being retried as version conflicts. A real conflict still refreshes the version and retries once.
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
