---
id: TASK-15407
title: Implement Web Scraping Phase 5 crawl and process-local jobs
status: In Progress
labels:
- web-scraping
- refactor
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved Phase 5 implementation plan using fresh implementers and task reviewers. Track eleven reviewable child units, preserve analyzer and all compatibility profiles, certify the whole branch before integration. No merge or push authorized by this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All eleven plan tasks pass task-level spec and quality review with recorded TDD evidence.
- [ ] #2 Combined compatibility, security, browser, lifecycle and consumer certification passes with exact results and justified skips.
- [ ] #3 Whole-branch review is addressed and the human change-summary merge gate is preserved.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Human selected option 1 (subagent-driven) on 2026-11-01, approving plan execution. Governing plan Docs/superpowers/plans/2026-11-01-web-scraping-phase-5-crawl-process-local-jobs.md; approved spec and ADR-085 unchanged. Existing isolated worktree reused on new codex/web-scraping-phase-5 branch from origin/dev 54b321dcec8cdbb6358581f42c1548696260570b. Design branch preserved; only final approved docs/task records copied forward, no dirty-main changes. ADR assessment: no new ADR; implementing ADR-085. IDs reserved above local 15406 and refreshed dev/all paginated open PR maximum 15281 (ADR max 084).
Task 1 implemented at 2606d6569c and independently approved (spec compliant, quality Approved); 62 added cases pass, final distinct scoped suites 149 pass plus pre-existing inventory line-drift gate. Existing Phase3/4/HTTP baseline 2366 pass/1 skip; whole-repo collection blocked by unchanged Workspaces httpx2 import. Ruling: enhanced recursion appends successes only and expands links after success; preserve actual appended-result predicate/timing, not an assumption failed attempts consume cap. Planned Task 11 owns inventory refresh. Two minor review observations retained in SDD ledger for Task 6/final review. Task 2 next; no product cutover yet.
Stage 1 complete: Task 2 contracts/controls plus publication marker and five review fixes independently approved at cc15ee1d10; 233 covering tests pass, explicit lint/compile/pre-commit/Bandit clean. Task 3 queue ownership next, no legacy traversal/queue cutover yet. Real Chromium baseline 3 pass. Existing admin/persistence service baseline 32 pass and 2 PostgreSQL cases fail due missing declared psycopg-pool dependency; user asked asynchronously before any venv installation. Whole-repo httpx2 and import-inventory line-drift baseline gates remain disclosed. Shared origin/dev advanced externally by14 unrelated commits; planned integration check before PR readiness remains.
User explicitly approved installing the declared psycopg-pool>=3.3 for verification. Installed psycopg-pool3.3.3 into the existing parent project venv, no dependency declaration or other package upgrade. Reran exactly Services/test_enhanced_webscraping_persist.py::test_persisted_web_article_is_searchable_only_by_its_effective_owner[2-2-postgres] and [None-1-postgres]: 2 passed,16 warnings,exit0 under official project fixture. Earlier PostgreSQL dependency blocker is resolved; unrelated httpx2 collection and inventory drift gates remain.
Task3 completed and independently approved after fix round1 at b8224e38ba. 123 dedicated queue and343 focused tests pass; scoped Ruff/format/compile/pre-commit/Bandit clean. Original raised live exception and legacy retry/lifecycle contracts retained; public diagnostic projections independently safe; actual attempt accounting remains canonical callback-owned. Exception cause/context/group frames release private bindings after drain. Warning provenance reconciled, no new source. Task4 lifecycle authorization/projections next. Legacy runtime has not switched to new queue.
Stage2 complete: Task3 queue and Task4 lifecycle/public projections independently reviewed with important findings fixed and scoped re-reviewed. Task4 cc4db584dc + afe62367b6; 471 pre-review /197 post-fix covering tests; zero touched-production Bandit findings. Task4 warning sources categorized to existing baseline, legacy UP015 and full-suite httpx2 blocker remain visible. Deferred startup/shutdown cancellation transition minor recorded for final review. Enter Stage3 with Task5 parent-controlled canonical acquisition.
Task5 complete after fresh review and fix re-review: e3b4515f54 + db27425016 + f2abedc4e1; 773 final covering pass, no skips; actual Chromium3pass from initial implementation; zero touched Bandit findings. Corrected branch-introduced Task2 AST/Task4 management test integration before review rather than calling them clean-baseline failures. Historical broad run3288pass26skip5fail retained; two existing arXiv assertion failures and inventory drift remain explicit Task11 certification obligations. Task6 now starts profile engine/discovery with known derived-byte and private callback interface risks, no silent functionality loss.
Task6 gate complete after 3 Important findings fixed/re-reviewed:87c794be0f,0702dc3ee2,55033a803b + documented controller rulings. Post-fix636 covering pass, warnings categorized; quality/Banditclean. Prior broad3427pass26skip2knownarXivfail retained; inventory and fullrepo collection gates still explicit. Begin Task7 checkpointfile/storage/security and actual I/O ownership. Remote dev advanced to6bf581da089a7aa6d90440a8801de07f8fdd238b (50commits afterbaseline), no changedselectedproductionfiles. Parent15407 nowcollideswith unrelated mergedModeration task: coordinated backlog-py renumber/reference refresh required before integration/PR, never overwriteeitherrecord. No current PR/merge-readiness claim.
Task7 complete: efbd0e70fc +70e58f4a86 (controllerplatformdoca73cad8ad7), fresh task and fix review accepted3findings inclmissingpackageinitializer. Final346coveringpass, scopedquality/Banditclean. Newengine265lines fixedoneof7CI size regressions; 6 remain OURBRANCH integration debts forTask9/10/11, not baseline failures. M1forktestnewwarning trackedforfinalreview; nativeWindowsunverified/service-rootfailsclosedwithoutcapability, futuremodecompositionexplicit. BeginTask8 boundedbatchproduction/per-entryprivateexecutionbinding. ParenttaskIDcollision15407stillmustrenumberwithreferencesbeforeintegration/PR.
Execution checkpoint: Tasks1-8 independently accepted. Task9 adapter cutover plus three reviewed fixes committed, but browser request-setting/redirect-child/proxy parity findingI2 remains open; Tasks10-11 not begun. Three native probes show public Playwright cannot address flattened browser child sessions. Human permission requested to design/review an owned same-browser loopback CDP control connection, no endpoint opened/unsafe workaround/fourth same-issue attempt. Do not mark whole stream done or PR ready. Reports/plan-scoped ledger retained; final namespace collisions, integration, size, fork, inventory, consumer and environment gates remain. Latest fresh controller adapter check51passed115warnings2.20s. Psycopg-pool3.3.3 installed only after human approval; both earlier PostgreSQL cases passed as recorded. No push/merge/rebase.
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
