---
id: TASK-15413
title: Extract bounded profile-aware crawl traversal
status: In Progress
labels:
- web-scraping
- phase5
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute Task 6 of the approved Phase 5 plan as an independently reviewable unit. Parent TASK-15407. Scope, interfaces, exact defaults, named tests and commits are defined by that plan task; preserve all global constraints.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Task 6 behavior and interfaces match the approved spec and plan.
- [ ] #2 Focused red-green tests, relevant regressions, lint and scoped Bandit evidence are recorded.
- [ ] #3 Fresh task reviewer approves spec compliance and code quality; findings are addressed before the next task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Parent TASK-15407. Plan Docs/superpowers/plans/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs.md, Task 6. ADR assessment: no new ADR; existing ADR-085 governs. Use backlog-py; no product edits outside this task scope without controller ruling. Implementation and review evidence will be linked from the plan-scoped SDD ledger.
Tasks1-5 independently gated. Begin bounded discovery/profile traversal through injected Task5 start_page handle. Read baseline-notes.md Task6 derived-memory risk and live compatibility hook gaps before implementation; request controller rulings if extra trusted limit or private hook needed. Preserve actual enhanced success-only appended result predicate and expansion timing; queue/frontier never duplicate page_attempt charge. note_successful_result must commit without await before article/frontier transition; generic stop cleanup cancels owned handles directly, not context.cancel. Verify prompt release of drained context-held credential/callback references. No wrapper cutover before Task9.
Task6 preflight at BASE/HEAD 92bc73fe37: NEEDS_CONTEXT; no production/test implementation begun. Controller rulings required before implementation: (1) approved trusted derived-URL retained-byte bounds, accounting scope and overflow behavior (small read-only urljoin witness: 16 distinct relative links, 38 relative bytes, 1046-byte base -> 16774 derived URL bytes; 4096 is illustrative only, not a chosen default); (2) explicit private nonserialized live hook path for enhanced recursive url_filter, enhanced sitemap filter_func before max_urls, and legacy progress callback; listed execute_crawl/discover signatures have no such path; (3) Task5 acquired callback currently invokes discovery before extraction (orchestration/article.py:595-600,1242-1246), conflicting with enhanced success-only discovery; authorize a narrow private defer/gating path without refetch. Report: .superpowers/sdd/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs/task-6-report.md. Child remains In Progress pending controller rulings and independent gate. TDD/Hypothesis/regression/Ruff/compile/Bandit not run: no production/test changes, do not infer completion. ADR assessment unchanged: ADR-085 governs; no new durable decision made.
Controller resolved all three preflight requests in Task6 execution-rulings block and regenerated exact brief. Trusted max_url_bytes65536 and cumulative discovered_url_bytes67108864 (separate ledger/snapshot; UTF8 locations/candidate-parent pairs; no frontier/transfer double charge; explicit overflow/checkpoint ceilings). Private nonserialized CrawlHooks supports url_filter and legacy/snapshot progress with existing invocation timing. Private start_page discovery_on_success defaultsFalse, enhanced recursion selectsTrue with exact truthy extraction_successful predicate; same document, no duplicatefetch; interrupted acquisition stays unfinished, only prior committed articles count as retained partials. These are controller builder decisions closing approved boundedness/compatibility gaps, not new client knobs/human-selected defaults. Implement TDD and request further concrete ambiguity rather than guess.
Task6 partial implementation checkpoint: validated URL-byte limits/counter/snapshots, private hooks/checkpoint models, bounded frontier/discovery, Task5 truthy-success discovery timing, prompt physical-drain callback release, initial injected-handle/continuation lifecycle. Focused TDD latest146passed; Ruff/compile/scoped production Bandit/diff checks passed. NOT full Task6 or all-profile parity; orchestration/crawl.py not created pending concrete private composition rulings: ordered enhanced_batch multiple-input seeds cannot travel through scalar CrawlPlan/CrawlRequest; async robots/policy/scorer composition cannot travel through synchronous hooks or existing engine arguments. Requested private seeds Iterable[str] and structural TraversalPolicy port (or explicit methods on CrawlControls), no public knobs or wrapper cutover. Exact truthy profile counting remains distinct from strict actual retained-success marker. Detailed commands/output/limitations: .superpowers/sdd/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs/task-6-report.md. Original reviewBASE92bc73fe37 retained, controller plan/ledger not staged. Child remains InProgress pending completion and independent gate; final covering regression not yet run; known Workspaces/httpx2 collection, arXiv assertions and inventory gates remain explicit/unchanged.
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
