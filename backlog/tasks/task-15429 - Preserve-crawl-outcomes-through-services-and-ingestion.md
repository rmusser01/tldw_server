---
id: TASK-15429
title: Preserve crawl outcomes through services and ingestion
status: Done
labels:
- web-scraping
- phase5
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Task 10 of the approved Phase 5 plan as an independently reviewable unit. Parent TASK-15421. Scope, interfaces, exact defaults, named tests and commits are defined by that plan task; preserve all global constraints.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Task 10 behavior and interfaces match the approved spec and plan.
- [x] #2 Focused red-green tests, relevant regressions, lint and scoped Bandit evidence are recorded.
- [x] #3 Fresh task reviewer approves spec compliance and code quality; findings are addressed before the next task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Parent TASK-15421. Plan Docs/superpowers/plans/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs.md, Task 10. ADR assessment: no new ADR; existing ADR-093 governs. Use backlog-py; no product edits outside this task scope without controller ruling. Implementation and review evidence will be linked from the plan-scoped SDD ledger.
Task9 independently accepted at82ed7a3544 with all fix findings closed; begin Task10 report-first service/ingestion outcome cutover. Requirements task-10-brief.md plus task1 actual consumer inventory/globalconstraints, plan Task10 and existing ADR093. Preserve every summary/prompt/cookie/header/config/persistence/receipt/progress behavior; no new runner/persistence engine. Include deferred Task4 entered-shutdown/startup cancellation ownership race and decompose _store_persistent below original ceiling. New report tests first; relevant service/security/persistence/media regressions, Ruff/format/compile/Bandit/hooks and fresh scoped review required. Current task/ADR namespace collisions handled later by controller before integration; do not renumber active records. No push/merge.
Task10 fresh implementer BASE 9feeaee6c73d02f7f5db6ac1fe533c0f8d41e282; brief/global constraints/baseline/ADR093 read. In Progress. TDD service report-first cutover, ingestion receipt propagation and deferred entered-shutdown cancellation race. No Task9 engine/browser changes, metadata renumbering, rebase, push or merge. Checkpoints/review evidence: .superpowers/sdd/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs/task-10-interim.md and task-10-report.md.
Task10 implementation ready for controller fresh review. ADR093 retained; report-first service/ingestion receipts, exact public three-field outcome, HTTP mappings, summary/prompt/cookie/metadata/dedup preservation, cancellation/physical DB drain, entered-shutdown startup ownership and failed-startup retry. Narrow adapter progress attachment only; no accepted Task9 engine/browser/queue rewrite. 62 new tests; TDD RED logs /tmp/task10-*-red.log. Final source affected service/ingestion/HTTP gate270passed0skipped (/tmp/task10-final-service-source.log); amended endpoint/progress+adapter/live-import+mandatory3browser/28native gate355passed0skipped (/tmp/task10-final-amended-covering.log), before ONLY final one-line resource-free shutdown retry guard subsequently covered by270gate. Ruff/format/compile/diff12files PASS; Bandit5production files0findings; explicit-path normal hooks PASS. _store_persistent307<=352, ingestion364<=367; ceilings unchanged. Historical broad3588passed28optional skips5fail:2owned live-import assertions fixed/rerun; frozen generated inventory and2oldarXiv assertions remain certification/integration concerns. Final ratchet12passed1failure ONLY4Task11sharedHTTPentries. Bare repo collection oldhttpx2 blocker; controller confirms fetched dev576feef820 already fixes httpx2/arXiv, no duplicate fixes/dependency install/rebase here. Full report .superpowers/sdd/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs/task-10-report.md. Asked controller to confirm no invented aggregate association for per-page queue polling; aggregate in task progress/receipts, existing optional public_job projection reused. Status remains In Progress pending controller acceptance; no push/merge/renumber.
Author committed8307be613b; final-source270pass/0skip, amended-boundary355pass/0skip, postcommit87pass/0skip; scopedquality/hooks/Bandit0. Independent review pending, not accepted. Controller checked page-job binding question: queue _Entry controls private, cleared queue.py333 after physical drain; safe get_job362 returns copies/history, service get_job_status returns standalone page public_job. No existing aggregate service job binding. Plan clarifies aggregate outcome on actual owning progress/envelopes/receipts; optional typed public_job projection shared when real owner supplies outcome. Do not infer parent outcome from page status, retain controls for polling, or add registry. This preserves existing API/no new subsystem; reviewer must assess interpretation, not waive true lost data. ADR093 ownership unchanged.
Fresh task10 reviewer Sagan NEEDSFIXES one Important I1: malformed nested stored articles [None] or [] after a valid partial receipt raise AttributeError/ValueError outside outcome attachment; storage released but HTTP500 loses known terminal outcome. Preserve validated exact3key outcome across all normalization/consistency failures, exception types/cancellation/release unchanged; endpoint-level RED/GREEN required. All other reviewed ownership/compatibility/architecture coherent, aggregate page-job clarification accepted on merits. Original author Lagrange resumed fixround1/3 BASE2987eec27f; Task10 remains In Progress. No freshdev/metadata integration yet.
I1 repair started at reviewed HEAD2987eec27f, original Task10 BASE9feeaee6c7 retained. Read task-10-review.md verbatim and verified lookup-only exception handler leaves reconstruction outside the boundary. TDD endpoint payloads [None] and [] with valid partial receipt, plus exact exception/cleanup checks; extend only existing ingestion failure boundary. Preserve controller aggregate ownership clarification, no new registry or architecture. Existing controller I1 task note retained. Report/interim will carry exact amended-source verification and scoped fix commit; fresh narrow re-review required.
I1 repair complete for narrow re-review, not accepted. BASE2987eec27f; original Task10 BASE9feeaee6c7 preserved. Two real HTTP malformed nested payloads [None]/[] plus AttributeError/ValueError+cleanup checks observed RED4fail (/tmp/task10-i1-red.log), GREEN4pass (/tmp/task10-i1-green.log); supplementary cancellation/cleanup check added. Existing ingestion helper now retains validated known receipt outcome through full unwrap/normalization/consistency boundary and releases enhanced/legacy storage in one finally; original exceptions and CancelledError remain unchanged, no manufactured completion/empty-success. Final amended-source covering195passed0skipped3682warnings141.33s (/tmp/task10-i1-covering.log): ingestion15, crawl outcomes52, Task4security25, legacy ingest16, friendly flags5, summaryHTTP82. Changed-scope Ruff/format/compile/diff and normal explicit hooks PASS; Bandit changed production file0findings0errors0nosec0skipped. Helper43ASTlines, existing ingestion364 unchanged; no ceiling changes. Only service web_scraping_service.py, existing phase5_ingestion_outcomes tests, and this child record in fix commit. Report I1 append contains exact commands/output and original failure history. Accepted aggregate ownership clarification unchanged; no plan/ledger/controller source edits, no new subsystem/dependency/rebase/push/merge. Fresh narrow re-review required.
Task10 independently accepted: fullreview9feeaee6c7..2987eec27f coherent spec/quality exceptI1; freshI1review2987eec27f..a864ccb315 ADDRESSED/no new findings. Final-source270pass,amended355pass,postcommit87pass beforefix; finalfixcovering195pass3682warnings141.33s andpostcommit15pass85warnings10.84s,0skips. AllscopedRuff/format/compile/diff/hooks/Bandit0. All owned service size ceilings restored without raising. Current sourcea864ccb315; page-job ownership clarification accepted, no invented registry. Stage4remainsTask11 integration/certification; oldarxiv/httpx2 failures alreadyfixedonfreshdev not duplicated. No publish/merge.
Namespace migration before fresh-dev integration: original identifier TASK-15417; current identifier TASK-15429. Parent and task/ADR references updated with the official backlog-py API after the 45-PR namespace scan; governing web-scraping ADR-085 renamed ADR-093 only to avoid unrelated accepted ADR-085. Work history, status and criteria retained.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Services consume canonical reports before summaries/storage and preserve immutable outcomes independently of operation/storage status through all receipts and owned progress. Retained existing prompts, metadata, cookies, deduplication, public lists and failure shapes. Managed DB writes drain after cancellation without starting later writes; entered shutdown owns startup-through-cleanup. Malformed nested receipts retain known validated outcomes while preserving original exceptions and releasing storage. Independently reviewed; final integration and certification remain Task11.
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
