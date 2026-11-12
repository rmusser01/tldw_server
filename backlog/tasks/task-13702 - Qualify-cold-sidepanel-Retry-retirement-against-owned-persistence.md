---
id: TASK-13702
title: Qualify cold sidepanel Retry retirement against owned persistence
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Identify the extra storage write after a held old-account cold Retry rejects. Preserve actual ownership fences and exact Alice and Bob records; repair the test synchronization or shared root cause only as evidenced.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shared and legacy account and unmount controls retain exact owner and persistence assertions
- [x] #2 Connected sidepanel ownership regressions and independent review pass
- [x] #3 Investigate late-write classification without transcript disclosure and record reproduction limits
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR3229; latest highest13639 acrossorigin/open/local; reserve13640. Readonly reviewer finds existing generation and account revision fencing, no production defect established. Root must classify actual write before editing guard. ADR assessment pending evidence.
Originalisolated4 andfull271pass locally; temporarykey-only extras[] all4, removeddiagnostic. OriginalCI sixthkeyNOTidentified. Reproductioncriterion refined to explicitinvestigationlimit, no claimproductionfix. ActualBob recordsettled beforebaseline; existingvalidatedmodelcachetombstone filter; exactAlice/Bob records+revision preserved. Native277/WebUI277passed; independentboundedreviewCLEAR. ADR no; BanditN/A TS test. ReviewDocs/Reviews/CHAT_COLD_RETRY_RETIREMENT_13640_2026_10_13.md. NextheadCI stillrequired.
Exact3file normalhooks passed without bypass; owncompletedplan retired. Reviewclear and both277case passes qualify thisharnessdelta; nextCI must stillsettle original publicfailure.
2026-10-15: Renumbered this PR's historical TASK-13640 to TASK-13702 after latest dev introduced an unrelated task with the old ID. The requester explicitly approved the proposed four manual task-ID renames and four report-heading updates. Backlog-py has no existing-task renumber operation; the approved exception changes filename/frontmatter ID, followed by this ordinary backlog-py note. Historical report filenames and original verification content remain unchanged. ADR assessment: no new architecture decision; metadata collision repair under ADR-059.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Qualified retirement persistence baseline: waitsBob durableowner record; exactAlice/Bob records and account revision remain unchanged after oldrequest rejection; permits onlyvalidatedmodelcachetombstones viaexistinghelper. Bothnative/WebUI277pass, boundedreviewclear, normalhooks pass. OriginalCI sixthwrite notidentified locally; no productionfixclaim. BanditN/A TS-only; wholePR stillopen.
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
