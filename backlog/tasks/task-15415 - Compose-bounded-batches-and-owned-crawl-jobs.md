---
id: TASK-15415
title: Compose bounded batches and owned crawl jobs
status: In Progress
labels:
- web-scraping
- phase5
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Task 8 of the approved Phase 5 plan as an independently reviewable unit. Parent TASK-15407. Scope, interfaces, exact defaults, named tests and commits are defined by that plan task; preserve all global constraints.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Task 8 behavior and interfaces match the approved spec and plan.
- [x] #2 Focused red-green tests, relevant regressions, lint and scoped Bandit evidence are recorded.
- [ ] #3 Fresh task reviewer approves spec compliance and code quality; findings are addressed before the next task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Parent TASK-15407. Plan Docs/superpowers/plans/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs.md, Task 8. ADR assessment: no new ADR; existing ADR-085 governs. Use backlog-py; no product edits outside this task scope without controller ruling. Implementation and review evidence will be linked from the plan-scoped SDD ledger.
Tasks1-7 independently accepted. Start Task8 bounded producer and queue-owned pages; read baseline-notes.md Task8 private execute-binding preflight and concrete concurrency/window test lens. Callback/control/plan state must not enter serialized metadata or hidden global registry; per-entry trusted private execution override will be documented in taskbrief. Actual available-capacity filling test required in addition to upperbound; sequential direct-crawl latency implication remains explicit. Preserve one terminal publication/actual-data marker timing, failed per-input payload/order, private callbacks through real drain, immediate no-owned-progress overload and owned-only cancellation.
Task8 implementation starting at be363284df on codex/web-scraping-phase-5. Binding task-8-brief.md/global-constraints.md/baseline-notes.md read. Planned scoped changes: private queue entry execute capability, lazy bounded batch producer, optional queue crawl composition reusing canonical acquisition. TDD and verification evidence will be in .superpowers/sdd/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs/task-8-report.md. Status remains In Progress for controller independent gate.
Task8 implementation verified: lazy producer fills five owned slots before first release; 23 reverse-completing inputs retain ordered output. Per-entry private execute capability is captured at admission, reused for retries, and released only after physical drain; no callbacks/plans/controls in JSON or public history. Explicit child cancel_job remains a per-item failure; batch task/context cancellation propagates CancelledError and cancels owned work only. Own logical-complete drains pace admission under parent deadline; final drain uses one separate 2s cleanup allowance. Optional crawl queue composition reuses profile conversion and single terminal publication; direct crawl remains sequential/default. Final named suites: 31 passed, 74 warnings; all test_phase5*.py: 753 passed, 1519 warnings. Architecture/inventory run: 855 passed, 1 known inventory drift failure. Scoped Ruff/format/compile/Bandit clean. Size ratchet: 12 passed/1 failed on same six branch debts outside Task8; all touched production functions <=300 AST lines. No fullrepo rerun, dependencies or network/LLM calls. Self-review only per controller; independent gate still pending. Full commands, RED/GREEN evidence, remaining arXiv2/inventory/size/Workspaces-httpx2 gates and latency caveat: .superpowers/sdd/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs/task-8-report.md.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Task8 implementation and scoped verification complete; TASK-15415 remains In Progress pending independent controller gate. See task-8-report.md in the plan-scoped SDD directory for evidence and whole-branch concerns.
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
