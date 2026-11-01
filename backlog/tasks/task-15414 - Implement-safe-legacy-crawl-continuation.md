---
id: TASK-15414
title: Implement safe legacy crawl continuation
status: In Progress
labels:
- web-scraping
- phase5
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Task 7 of the approved Phase 5 plan as an independently reviewable unit. Parent TASK-15407. Scope, interfaces, exact defaults, named tests and commits are defined by that plan task; preserve all global constraints.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Task 7 behavior and interfaces match the approved spec and plan.
- [ ] #2 Focused red-green tests, relevant regressions, lint and scoped Bandit evidence are recorded.
- [ ] #3 Fresh task reviewer approves spec compliance and code quality; findings are addressed before the next task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Parent TASK-15407. Plan Docs/superpowers/plans/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs.md, Task 7. ADR assessment: no new ADR; existing ADR-085 governs. Use backlog-py; no product edits outside this task scope without controller ruling. Implementation and review evidence will be linked from the plan-scoped SDD ledger.
Tasks1-6 independently accepted. Begin Task7 safe local continuation using actual current CheckpointPort/State/Scope and new URL byte ledger, trusted boundscope, actual per-crawl owned_handles/PhysicalWork drain. Read baseline-notes.md existing lock candidate/tests, sync I/O ownership and trusted-root gap. Service root must be trusted composition, never stored file/HTTP options; legacy path mode remains explicit. Controller Task7 execution rulings will define root/offload contract before edits. No wrapper cutover or remote resume endpoint, no new dependencies.
Task 7 started at BASE 7a1fac8ea2 in authorized worktree. Read task-7 brief, global constraints and baseline notes including execution rulings. Implement store, bounded codec, trusted-root/native lock boundary and physically owned sync I/O; TDD suites plus final scoped covering verification. Child remains In Progress for independent gate. Controller owns plan/ledger and parent ID integration.
Task 7 implemented at BASE 7a1fac8ea2: bounded versioned/legacy checkpoint store; hidden trusted root, descriptor/no-follow native writer locking and PID/fork ownership; credential-free atomic writes preserving prior state; per-crawl owned sync I/O and current profile/egress/robots resume admission. Added exact Task 7 suites with deterministic syscall barriers and cancel-then-held-write evidence. Final scoped covering: 245 passed; one failing size-guard test contains only six controller-carried out-of-scope findings. run_crawl AST size 265 <= 300, all new helpers <= 300. Broad Web_Scraping/native-lock/size run: 3181 passed, 26 skipped, three carried failures (two arXiv assertions and size guard). Late cancellation hardening and additional cleanup tests verified in final scoped run with visible warnings summary; no new warning filters. Ruff/format/compile and production Bandit clean; test Bandit clean excluding assertion-only B101. Full report: .superpowers/sdd/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs/task-7-report.md. No wrapper cutover or unrelated fixes. Keep In Progress until fresh independent controller gate; parent ID integration remains controller-owned.
Review fix round 1 started from reviewed HEAD a73cad8ad7. Read exact Important I1/I2 in task-7-review.md. TDD scope: strict version-1 wire shape before defaults and legacy integer page count; one aggregate monotonic cleanup deadline, preserving loading=True active deadline and actual background ownership. Will run Task7/engine/composition/controls/native lock scope and static/security checks, append fix report, and commit explicit paths. Keep In Progress; M1 and platform/composition follow-ups are not in this fix loop.
Controller added concrete packaging correction in round 1: setuptools namespaces=false omits crawl without __init__.py. Include minimal crawl/__init__.py without legacy re-exports and configured package-discovery/import regression in this same fix scope/commit, with RED/GREEN and quality evidence. No dependency changes.
Round 1 fixes verified from a73cad8ad7: I1 explicit version-1 wire field sets before defaults and bounded integer legacy page counts (RED 6 failed/19 passed -> GREEN 25 passed); I2 one monotonic finalization deadline shared across drain/result/save/clear/close, no logical per-stage renewal, background physical ownership retained (RED 4 failed/1 passed -> GREEN 5 passed with virtual-clock/syscall barriers and canceled/budget/deadline parent cases). Load remains under active _bounded context. Packaging correction: minimal crawl/__init__.py and configured setuptools discovery/import test (RED 1 failed -> GREEN 1 passed). Final requested Task7 + packaging + engine/composition/controls/native-lock scope: 346 passed, 705 warnings, exit 0, 9.21s; warning summary visible, no new filters. Scoped Ruff/format/compile and production Bandit clean; test-only Bandit clean with B101 assertion exclusion. AST run_crawl 265, all touched functions <=300; no ceiling edits or fullrepo/arXiv/other guard fixes. Appended round-1 evidence to task-7-report.md. Keep In Progress for fresh independent full Task7 re-review; M1 and approved Windows/composition follow-ups unchanged.
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
