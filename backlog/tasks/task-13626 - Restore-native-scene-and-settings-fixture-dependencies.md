---
id: TASK-13626
title: Restore native scene and settings fixture dependencies
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three remaining scene/settings suites omit actual browser storage and retain WebUI-only prototype read controls. Reuse the qualified shared physical I/O controls, restore the declared API per case after owned global cleanup, and retain all source/account/model refusal assertions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Scene image, service prompt and settings ownership suites pass in both native UI and WebUI configurations
- [x] #2 Previous native history and physical storage suites still pass after per-case runtime restoration
- [x] #3 Both client types, independent review and normal hooks pass with no production guard changes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no; restore declared test dependencies using existing fixture and safe storage boundary. Native RED69failed23passed across3suites. Plan Docs/Plans/IMPLEMENTATION_PLAN_chat_scene_storage_13626.md. PR https://github.com/rmusser01/tldw_server/pull/3229.
Initial repair100passed1failed. Isolated Compare copy diagnostics confirm account/destination/preparation leases true and original scene retained, but local copy refuses. The suite had only spied userSettings.get: shared UI still constructed actual Dexie without IndexedDB, unlike WebUI callback-table setup. Declared existing local-history memory DB including its real callback transaction to qualify original profile rereads. Temporary primitive copy diagnostics removed; production bind guards unchanged.
Final native and WebUI7suite runs930/930 each, both normal client type checks exit0, independent bounded review clear. Review Docs/Reviews/CHAT_SCENE_STORAGE_13626_2026_10_13.md. Bandit N/A TS test-only. Normal hooks next.
Normal exact-file hooks pass. Both930-test runners and client type checks pass; bounded review clear. Bandit N/A TS test-only. Other PR CI/UX/H2, final independent review and merge remain open.
Completed owned plan retired to /private/tmp/IMPLEMENTATION_PLAN_chat_scene_storage_13626_completed.md; durable evidence remains in the review.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Restored native browser storage dependencies and actual physical read fault controls for scene/settings suites. Shared setup rebinds the API per case without weakening explicit retirement. Compare fixture now declares existing memory DB transaction/profile I/O. Each runner passes930 tests, both types and normal hooks pass; bounded review clear.
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
