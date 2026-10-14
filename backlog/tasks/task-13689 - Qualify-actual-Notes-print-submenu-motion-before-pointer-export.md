---
id: TASK-13689
title: Qualify actual Notes print submenu motion before pointer export
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Current a34 UX CI reaches visible Print flyout but the ordinary click sees entering motion instability then a hidden retained leaf; 19 other UX cases pass. One unchanged local NE04 passes, so there is no established production print closure fix. Qualify actual flyout motion and ordinary pointer entry in the shared NotesPage driver, preserving known-defect print oracle and evidence. Both callers must continue to work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Wait on observable flyout motion using bounded actual browser state; no force, retries, fixed sleeps, disabled motion or oracle relaxation
- [x] #2 Qualify NE04 and derived note shared caller using actual server and record limitations
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no new ADR; test driver readiness contract only, existing Notes production print defect #3117 and oracle remain unchanged. Evidence: /private/tmp/chat-pr3229-a34-ux-regression-job.log and private CI video; actual flyout visible then unstable then hidden. Source @rc-component/menu uses delayed entering motion. Driver has two callers: UX notes-p0 and workflows/notes-studio-derived-note.
Driver NE04 now passes3 repeatEach zero-retry cases and second derived-note caller; original oracle remains. Broadened full UX20 on reused runtime is invalid qualification: empty-library precondition got4 notes and native extension fell back to an existing unqualified build. Preserve this invalid report, do not claim full pass or infer production roots. Re-run using supported isolated live-tier runner with fresh backend and newly built native extension. All contexts headless.
Supported isolated runner first local launch failed Turbopack node_modules symlink outside filesystem root after current native extension production build35.1s. Second launch uses already-qualified private artifact with owned Bun dependencies: all97 production hashes match;314 current tracked E2E/config files copied without transforms; new native extension build supplied explicitly; fresh backend remains managed-worktree current source. Preserve first failure artifact; no product-source change for local environment.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Shared NotesPage print driver waits for actual visible flyout ancestor motion (two RAF preparation, bounded10s poll), hovers Print and performs one ordinary click. No force, interaction retries, fixed sleeps, disabled motion, menu reopening or oracle changes.3 NE04 repeats and actual derived-note caller pass. Supported isolated fullUX20 passes2.4m with fresh backend/current35.1s production extension and97-hash-matching owned-dependency artifact;314 current tracked tests/config copied. Invalid reused-runtime full run and local Turbopack symlink failure retained as excluded diagnostics. Known Notes print defect #3117 remains represented by original oracle; no production print fix claimed. All browsers headless. Root exact-head CI/merge remain parent work.
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
