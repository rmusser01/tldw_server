---
id: TASK-15417
title: Preserve crawl outcomes through services and ingestion
status: In Progress
labels:
- web-scraping
- phase5
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Task 10 of the approved Phase 5 plan as an independently reviewable unit. Parent TASK-15407. Scope, interfaces, exact defaults, named tests and commits are defined by that plan task; preserve all global constraints.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Task 10 behavior and interfaces match the approved spec and plan.
- [ ] #2 Focused red-green tests, relevant regressions, lint and scoped Bandit evidence are recorded.
- [ ] #3 Fresh task reviewer approves spec compliance and code quality; findings are addressed before the next task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Parent TASK-15407. Plan Docs/superpowers/plans/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs.md, Task 10. ADR assessment: no new ADR; existing ADR-085 governs. Use backlog-py; no product edits outside this task scope without controller ruling. Implementation and review evidence will be linked from the plan-scoped SDD ledger.
Task9 independently accepted at82ed7a3544 with all fix findings closed; begin Task10 report-first service/ingestion outcome cutover. Requirements task-10-brief.md plus task1 actual consumer inventory/globalconstraints, plan Task10 and existing ADR085. Preserve every summary/prompt/cookie/header/config/persistence/receipt/progress behavior; no new runner/persistence engine. Include deferred Task4 entered-shutdown/startup cancellation ownership race and decompose _store_persistent below original ceiling. New report tests first; relevant service/security/persistence/media regressions, Ruff/format/compile/Bandit/hooks and fresh scoped review required. Current task/ADR namespace collisions handled later by controller before integration; do not renumber active records. No push/merge.
Task10 fresh implementer BASE 9feeaee6c73d02f7f5db6ac1fe533c0f8d41e282; brief/global constraints/baseline/ADR085 read. In Progress. TDD service report-first cutover, ingestion receipt propagation and deferred entered-shutdown cancellation race. No Task9 engine/browser changes, metadata renumbering, rebase, push or merge. Checkpoints/review evidence: .superpowers/sdd/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs/task-10-interim.md and task-10-report.md.
Task10 implementation ready for controller fresh review. ADR085 retained; report-first service/ingestion receipts, exact public three-field outcome, HTTP mappings, summary/prompt/cookie/metadata/dedup preservation, cancellation/physical DB drain, entered-shutdown startup ownership and failed-startup retry. Narrow adapter progress attachment only; no accepted Task9 engine/browser/queue rewrite. 62 new tests; TDD RED logs /tmp/task10-*-red.log. Final source affected service/ingestion/HTTP gate270passed0skipped (/tmp/task10-final-service-source.log); amended endpoint/progress+adapter/live-import+mandatory3browser/28native gate355passed0skipped (/tmp/task10-final-amended-covering.log), before ONLY final one-line resource-free shutdown retry guard subsequently covered by270gate. Ruff/format/compile/diff12files PASS; Bandit5production files0findings; explicit-path normal hooks PASS. _store_persistent307<=352, ingestion364<=367; ceilings unchanged. Historical broad3588passed28optional skips5fail:2owned live-import assertions fixed/rerun; frozen generated inventory and2oldarXiv assertions remain certification/integration concerns. Final ratchet12passed1failure ONLY4Task11sharedHTTPentries. Bare repo collection oldhttpx2 blocker; controller confirms fetched dev576feef820 already fixes httpx2/arXiv, no duplicate fixes/dependency install/rebase here. Full report .superpowers/sdd/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs/task-10-report.md. Asked controller to confirm no invented aggregate association for per-page queue polling; aggregate in task progress/receipts, existing optional public_job projection reused. Status remains In Progress pending controller acceptance; no push/merge/renumber.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Task10 within existing architecture; ready for independent controller review. Final affected service-source270pass/0skip; amended service+adapter+browser gate355pass/0skip; static and normal hooks clean, Bandit0findings, original storage ceiling preserved. Full report carries historical pre-integration certification concerns and exact logs. Controller review/acceptance pending.
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
