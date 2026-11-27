---
id: TASK-15688
title: Wait for validated draft in Presentation Studio overwrite test
status: In Progress
labels:
- ci
- webui
- testing
dependencies:
- TASK-12115
references:
- https://github.com/rmusser01/tldw_server/pull/3625
- https://github.com/rmusser01/tldw_server/actions/runs/46017936921/job/140613876861
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the shared UI coverage failure observed on Server PR #3625 when the overwrite test treats a displayed draft as ready before asynchronous validation settles.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The overwrite test waits for the mounted editor and an accepted draft before releasing the fresh server response.
- [x] #2 The existing overwrite assertions still require the new ETag, the accepted second edit, and preservation of the third edit.
- [x] #3 The focused test and shared UI coverage shard pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Reproduce the failure by holding the second edit digest, then synchronize the existing journey with the live conflict action. Run the isolated test, the Presentation Studio test file, and the affected coverage shard. ADR required no. This test-only repair preserves the existing standalone HTML overwrite and validation contract.
A held second-candidate digest reproduced the hosted missing Confirm overwrite button with the old wait. The same control passed after the test observed the enabled live conflict action before releasing the fresh response. The test also waits for mounted Monaco before editing. The ETag, submitted draft and later unsaved edit assertions are intact. The isolated test and all 1,319 tests in shared UI coverage shard 28/32 passed locally. Hosted verification of the replacement head is pending.
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
