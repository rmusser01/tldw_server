---
id: TASK-15430
title: Certify Phase 5 architecture consumers and lifecycle
status: In Progress
labels:
- web-scraping
- phase5
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Task 11 of the approved Phase 5 plan as an independently reviewable unit. Parent TASK-15421. Scope, interfaces, exact defaults, named tests and commits are defined by that plan task; preserve all global constraints.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Task 11 behavior and interfaces match the approved spec and plan.
- [x] #2 Focused red-green tests, relevant regressions, lint and scoped Bandit evidence are recorded.
- [ ] #3 Fresh task reviewer approves spec compliance and code quality; findings are addressed before the next task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Parent TASK-15407. Plan Docs/superpowers/plans/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs.md, Task 11. ADR assessment: no new ADR; existing ADR-085 governs. Use backlog-py; no product edits outside this task scope without controller ruling. Implementation and review evidence will be linked from the plan-scoped SDD ledger.
Task10 accepteda864ccb315. Begin Task11 with controller-owned metadata namespace repair and fresh-dev rebase BEFORE certification:45openPR/4568fullypaginatedpaths, maxperPR1168; dev576feef820, externalTASKmax15420/ADR092; mainlocalmax12027/feature15418. Existingcollisions15407/15410..15418 andADR085 require officialbacklog-py renumber+scopedreference updates, namespace-onlyADRrename preservingacceptedrationale, backupanchor, thenrebase ontofreshdev. No sharedbranchpush/merge. LatestdevalreadyfixesarxivNoneassertions/httpx2import, sharedHTTPNativeReady+boundedstream changesmustpreserveandcertifycombinedcode. Single-threadnativeforkbootstrapread-onlyproof passesexisting4assertions withwarningaserror; implementisolationinTask11 withoutdroppingcoverage. Final architecture/lifecycle/consumers/docs/inventory/security/quality/wholebranchreview gates unchanged.
Namespace migration before fresh-dev integration: original identifier TASK-15418; current identifier TASK-15430. Parent and task/ADR references updated with the official backlog-py API after the 45-PR namespace scan; governing web-scraping ADR-085 renamed ADR-093 only to avoid unrelated accepted ADR-085. Work history, status and criteria retained.
Metadata namespace correction completed through officialbacklog-pyMutableRepository renumber/editAPI, no hand-written task files: collision-only15407->15421,15410..15418->15422..15430;15408/15409unchanged. All12recordstatus/title/checklistsassertionsunchanged, normalizationcheckNoncanonical[], duplicate-IDCIexit0. ADR085web renamed093; original acceptedbody preservedbyte-for-byte afterheading, appended namespacehistoryonly. Plan/spec/indexrefsupdated,108MarkdownItparsedlinksvalidated0missing. Fresh45PR/4568filepaths/maxPR1168,dev576feef820,externalmax15420/ADR092 checked. Backuprefcodex/web-scraping-phase-5-pre-integration-e7dddc2b52 preservesacceptedhistory. Commitmetadata before localrebase; do not overwrite unrelateddevtask/ADRrecords. No product behavior changes or push/merge.
Historical implementation notes retain the original identifiers and observations verbatim; current descriptions/dependencies and the explicit namespace migration notes define today's identifiers. No past numeric limits or unrelated-record observations were relabeled.
Fresh-dev integration completed: all 67 local commits rebased onto origin/dev 576feef820e6521f021e6f57c82c24df2429b901. Only ADR index conflicts, resolved preserving upstream rows and scraping ADR-093; no production code conflicts. Selected scraping package and service code remains byte-identical to pre-rebase feature code; shared HTTP includes expected upstream bounded-streaming changes. Duplicate-ID certification exposed two older copied design/planning collisions, corrected path-specifically through backlog-py to TASK-15431 and TASK-15432. All fourteen task records retain original historical notes verbatim, criteria and status unchanged; current references and migration notes define action identifiers. Task11 review baseline after rebase: a16565207c9f2f51692142e29c79a0618187a4a3. No startup hook, new dependency/connection/subsystem or functionality removal. Actual post-rebase certification remains in progress.
Task11 certification started at clean a4a58a80e9a5b9c7460c6dc7d0b1a40897b0a179, review BASE a16565207c9f2f51692142e29c79a0618187a4a3. Brief and current instructions read; interim evidence: .superpowers/sdd/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs/task-11-interim.md. Controller retains progress ledger, bare whole-repository pytest and metadata audit ownership.
Interim: native mandatory Phase3+Phase4 smokes plus Phase5 native/combined lifecycle passed36,0skips,121warnings; log /tmp/web-phase5-task11-native-final.log. Corrected Watchlists fixture proves unchanged consumer propagates CrawlInterrupted/partial identity and cancellation, preserves ordinary fallback; no consumer code change warranted. Native fork isolation4assertions retained in fresh bounded single-thread subprocesses with fork warnings as errors. Four authorized sharedHTTP AST corrections measured exactly; ratchet green13. Architecture guard negative fixtures fail31 when disabled; real graph catches deferred jobs.admission<->jobs.projections cycle; requesting minimal existing-helper relocation ruling, no production edits pending approval. Harness correction failures are NOT product RED. AC3 remains unchecked pending independent controller review.
Task11 final implementation evidence: sanctioned minimal existing redaction-helper relocation removes actual admission/projections cycle; all5definitions AST-identical to clean executionHEAD; projection-module helper imports preserved. Architecture negative fixtures fail31when disabled; realcycle RED and missing canonical reexport RED ->325passed focused architecture/projection/admission/history/checkpoint/package/inventory. Final full scraping/HTTP4106passed28optional skips13010warnings252.34s; final actualconsumers770passed12optional skips6914warnings189.57s; final service115passed0skips394warnings7.70s; mandatory native36passed0skips121warnings58.06s and rerun in final full gate. Earlier2Postgres skips were sandbox reachability, resolved by official-fixture sanctioned2case recheck2passed then finalservice0skips. Watchlists unchanged; fixture corrections are NOT product RED. Finalinventory regenerated after approved source move and artifact equality passes. Four authorized HTTP ratchet entries only; size+taskformat17passed64warnings. Scoped Ruff10files/compile/normalization/difffcheck/explicitnormalhooks pass. Final endpoint-complete Bandit includes media/ingest_web_content.py:8knownuntouchedLOW,0MED/HIGH,0scanerrors, no new findings. Detailed warnings/skips/exact commands/limitations/self-review in .superpowers/sdd/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs/task-11-report.md. Controller owns bare whole-repo pytest baseline Meetings basename collision and metadata audit (task-11-controller-evidence.md). No nested agents/ledger edits/push/PR/merge. AC3 and DOD1 remain unchecked pending independent controller Task11 review and later wholebranch review.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Task11 certification with architecture negative fixtures, preserved true-fork safety assertions in bounded fresh single-thread subprocesses, real combined browser/crawl/queue/executor/checkpoint/service lifecycle, actual consumer certification, exact four HTTP ratchet corrections, final inventory/docs and scoped quality/security gates. Only product change is controller-approved AST-identical existing sanitizer relocation with compatibility reexports to remove the guard-proven dependency cycle. Final scraping/HTTP4106pass, consumers770pass, services115pass, native36pass0skip; optional skips and known LOW Bandit findings fully disclosed in task-11-report.md. Ready for independent controller review; AC3 is NOT self-approved and task remains In Progress.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
