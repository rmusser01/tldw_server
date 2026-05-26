---
id: TASK-515
title: Enable server deletion eligibility for Research Workspace migrations
status: To Do
labels:
- research-workspace
- migration
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the backend follow-up that can safely emit client_delete_eligible=true for Research Workspace migration sessions after server-side read-back/integrity verification proves all declared payloads were accepted and recoverable. Until this exists, WebUI migration must retain local legacy data and show recovery copy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Migration sessions expose client_delete_eligible=true only after explicit server-side verification of all declared chunks and manifest hash.
- [ ] #2 Client delete acknowledgement succeeds only for eligible finalized sessions with matching manifest hash.
- [ ] #3 Conflicts and failed verification remain recoverable and do not mark sessions delete-eligible.
- [ ] #4 Focused backend/API tests cover eligible, ineligible, mismatch, and idempotent retry paths.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

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
