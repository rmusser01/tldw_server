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
