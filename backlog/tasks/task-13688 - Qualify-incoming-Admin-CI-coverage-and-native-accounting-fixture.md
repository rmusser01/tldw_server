---
id: TASK-13688
title: Qualify incoming Admin CI coverage and native accounting fixture
status: Done
labels:
- ci
- chat
- integration
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Current-head CI38640596000 fails shard coverage because merged PR3218 added three Admin regression files without full-suite assignments. Add each file to its existing related shard in every repeated matrix; retain the guard, ignore file and baseline unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current coverage guard reproduces the three missing files before the edit and passes without any ignore or baseline expansion afterward.
- [x] #2 All full-suite workflow contracts and the three newly assigned Admin suites pass; normal workflow hooks pass.
- [x] #3 Independent reviewers verify every repeated matrix assigns the files exactly once and all existing checks remain enforced.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ParentTASK13458/PR3229. Existing implementation plan stage4 integration; no newarchitecture rule orADR needed. CIjob116125963148/currenta34log records Admin backups cache, event-loop offload and LLM usage reads unassigned. No application change is planned.
Full workflow contracts plus assigned Admin suites:70pass/1failure. The incoming _PgTokensDb test double only accepted legacy SUM(storage_used_mb), while real production now requires combined retained native/legacy aggregate. Fix the strict declared aggregate branch; retain all LLMUsageLog tokens/query/index assertions and fail-closed production behavior. No application change.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added only three exact incoming Admin paths to all five CI shard matrices; existing coverage guard and contracts pass (847 patterns,5103 files,44 unchanged baseline,0 newly uncovered). All71 workflow/Admin tests pass. PG native-storage aggregate mock now follows actual fail-closed query, retains real SQLite/index/token assertions. Fixture Bandit42 identical HEAD B101s,0 new; normal hooks pass. PR3229 exact-head CI and merge remain parent landing work.
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
