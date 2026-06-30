---
id: TASK-2241
title: Revise Workspace file inventory Jobs design after review
status: Done
priority: high
references:
- TASK-2240
documentation:
- Docs/superpowers/specs/2026-06-03-workspace-file-inventory-jobs-design.md
- Docs/superpowers/plans/2026-06-03-workspace-file-inventory-jobs-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address pre-implementation review issues in the Workspace file inventory Jobs design spec and implementation plan. Clarify stale-state semantics, partial-scan item deletion behavior, FK/cascade cleanup, Jobs enqueue failure recovery, parallelization dependencies, startup worker registration details, root resolution helper contract, and item cursor ordering before implementation starts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design spec clarifies stale as read-computed and defines root-version mismatch scan behavior.
- [x] #2 Design spec prevents partial scans from deleting unseen previous inventory rows.
- [x] #3 Design spec/plan define FK/cascade cleanup, enqueue failure recovery, item ordering/cursor behavior, and reusable root resolution contract.
- [x] #4 Implementation plan updates task sequencing, startup registration details, and tests for all review findings.
- [x] #5 Backlog record documents verification and completion state.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Review fixes applied:
- Clarified that `stale` is read-computed projected status, not a terminal scan-row state.
- Defined active root-version mismatch as a failed scan attempt with `root_version_mismatch`.
- Added `coverage_state` and full-vs-partial replacement semantics so partial scans do not delete unseen previous rows.
- Added FK/cascade and hard-delete expectations for workspace/root cleanup.
- Added `job_enqueue_failed` recovery and active-scan reuse rules for queued rows without Jobs ids.
- Added a named `resolve_workspace_root_for_inventory_scan(...)` helper contract.
- Added stable `relative_path ASC` item ordering and opaque cursor behavior.
- Updated the implementation plan dependency map, DB/worker/API tests, startup worker handle requirements, and final review checklist.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the pre-implementation design review fixes for Workspace file inventory Jobs. The spec and plan now cover stale-state projection, partial-scan preservation, FK/cascade cleanup, enqueue failure recovery, cursor ordering, a reusable scan root resolver, startup registration details, and corrected task sequencing. Verification: `git diff --check` passed, ASCII scan passed, and targeted search found no stale colliding task references or unsafe stale/partial wording beyond the intended full-coverage delete rule. Bandit/Python tests were not run because this task changed only docs and Backlog records.
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
