---
id: TASK-13935
title: Stabilize backup-restore confirmation coverage
status: In Progress
labels:
- frontend
- testing
- ci
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and fix the repeated advisory shared-ui coverage failure in DataOpsPage.backup-restore.test.tsx without weakening restore confirmation, scope, visibility or cancellation assertions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Reproduce the recorded hidden confirmation failure with evidence identifying its cause.
- [x] #2 Repair the cause while preserving exact artifact/dataset/user context, explicit confirmation and cancellation never posting.
- [ ] #3 Verify focused and relevant coverage tests, lint/types as applicable, and obtain fresh independent review.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Start from current dev e992d32c77144a1b6ed4851609783fede5eb6534 after PR3316 merged. Latest repeated failure: shared-ui-coverage11/16 run40218094826/job121246303547 on7f43108249,1505pass/1fail; real AntD Restore confirmation button exists but remains not visible within3000ms. Logs /private/tmp/pr3316-coverage11-current{-clean,}.log. Same failure repeated on earlier composer/header heads; one sanctioned historical retry also failed. Local unchanged22/22 normal/V8/Node20macOS passes previously did not establish hosted root. No blind retries or timeout/assertion weakening. Preserve prior branches/worktrees/.impeccable evidence; never stage node_modules symlink. ADR assessment: no new ADR anticipated for a test/fixture correction; reassess if evidence requires durable behavior change.
Fresh isolated Debian Linux arm64 source-only copy installed the frozen Bun lock with Node 20.20.2 and Bun 1.3.2. The unchanged focused V8 suite passed 22/22; this does not reproduce hosted Ubuntu x64 failure. A CPU-constrained temporary diagnostic measured whole-page getAllByRole at 3442ms then 3220ms while the first popup began in ant-zoom-big-appear-prepare; helper timeout remains 3000ms. This supports synchronous query starvation of popup preparation. The two diagnostic cases passed; its coverage report is invalid because it overlapped the next run in the same temporary coverage output directory. Subsequent coverage runs are sequential. A full shard replay with unchanged assertions and popup diagnostics is in progress. No product/test repair, timeout increase, assertion weakening or CI rerun has been made.
Candidate repair scopes getByRole to the real popup found by its existing overwrite warning. Same3000ms timeout and all22 artifact/dataset/owner/explicit-confirmation/cancel/no-POST/refusal/disabled assertions remain. Native candidate22/22 passes; installed ESLint actually checked the edited test with0errors/0warnings; normal frontend TypeScript noEmit8GB passed. Bandit ran on the touched TS path,0PythonLOC, no TS security clearance. Independent backup_restore_coverage_review found no coverage loss or actionable code defect but correctly requires distinguishing mechanism evidence from hosted cause; softened the explanatory comment. Full original isolated Linux shard11/16 (twoCPU) passed1506tests/166files, including22/22backup, so no hosted reproduction claimed. A new controlled full-page-ten-backup fixture probe tests the measured whole-page role cost; no blind unchanged rerun.
Controlled reproduction now matches the hosted symptom: with a realistic ten-backup first page (one chacha target plus nine distinct media rows), the ORIGINAL helper fails the SAME cancellation case at toBeVisible within unchanged3000ms. Diagnostic: first whole-page query3423ms; popup remained ant-zoom-big-appear-prepare with computed opacity0 when timeout expired. Source /private/tmp/backup-restore-linux-20261025; log /private/tmp/backup-restore-linux-inventory-red.log. This proves a query-starvation failure mechanism under a controlled larger inventory, not the exact original hosted timing. Retain the ten-row fixture in the existing case to prevent regression; no assertion removed. Correction to prior Bandit note: explicit TSX path produced a Python AST syntax-error entry, so Bandit did NOT validate this TypeScript scope (not zero parsed findings/clearance). No Python changed.
Identical-condition controlled red/green succeeded: ten-row first-page fixture, real AntD popup, sameNode20.20.2/Bun1.3.2/V8/twoCPU and unchanged3000ms timeout/assertions. ORIGINAL helper fails at hidden Restore; candidate passes. First query falls3423ms ->191ms; candidate continues polling the real button through prepare and reaches visible stable popup. Logs /private/tmp/backup-restore-linux-inventory-{red,green}.log; only helper query scope differs between these instrumented copies. Final production/test diff contains no diagnostic logging or dependency/config changes. Retained ten-row fixture is regression context for existing cancellation case; all22 safety assertions remain. Hosted exact-head qualification still pending publication.
Final regression fixture additionally asserts the ninth neighboring artifact is visibly present in the target backup tbody; this closes review concern that a changed request order could silently bypass ten-row coverage. Final native verification:23passes in2files (all22backup cases plus selected real DSR erasure-confirm/cancel case;11unrelatedDSR cases not selected), /private/tmp/backup-restore-final-native-guarded.log. Final test ESLint0errors/0warnings, noneignored, /private/tmp/backup-restore-eslint-final.json. Independent backup_restore_coverage_review approves final diff, no remaining findings. Unique own plan IMPLEMENTATION_PLAN_backup_restore_coverage_20261025.md stages1/2Complete,3InProgress; final exact-source Linux166-file V8 shard and hosted qualification pending. Production files, transport/domain implementation, dependencies, config, workflows/protection and earlier evidence unchanged.
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
