---
id: TASK-15413
title: Close Jobs resource lifecycle and attempt authority
status: In Progress
labels:
- claims
- jobs
- implementation
dependencies:
- TASK-15412
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Head-before-row lifecycle closure, UUID-fenced worker failure finalization, contract-preserving payload replacement, retention and exact attempt authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The approved task's interfaces, safety invariants and scope are implemented without Claims-owned lifecycle state.
- [ ] #2 Test-first RED/GREEN, appropriate characterization and official backend fixture evidence are recorded with exact results and skips.
- [ ] #3 Focused lint, compilation, touched-production Bandit, normal hooks, local commit and independent spec/quality review are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved plan task 3: Docs/superpowers/plans/2026-11-01-claims-jobs-stage2c-cluster-rebuilds.md. ADR required: no additional ADR; accepted Docs/ADR/085-jobs-resource-admission-and-domain-publication.md governs the approved resource/publication boundary with ADR-003 and ADR-058. Execute only in the existing linked worktree/branch, test first, use shared Jobs lifecycle APIs, official backend fixtures, focused verification and touched-code Bandit, normal hooks and scoped commits. Task-ID audit explicitly targeted rmusser01/tldw_server after discovering gh defaults to upstream: fetched dev 61d02ed484a3c80f5423676b959cf14e7b572a62, checked all 37 open PR paginated file lists (223 selected Backlog/ADR records, no truncation) and current branch; highest observed id 15410. Reserved TASK-15411 through TASK-15421 for the eleven reviewable plan tasks. No collision with planning records TASK-15282/TASK-15283 or ADR-085. Preserve unrelated main-checkout changes; no push/PR/merge authorized.
Task 2 prerequisite independently approved in a205a21934 and 996a70c7a5 after 460 passing affected tests/zero skips; Task 1 prerequisite also reviewed clean. Begin approved Task 3 using task-3-brief.md, constraints.md, task-1-interfaces.md, task-2-interfaces.md and task-3-integration-notes.md in this plans isolated SDD workspace. Cover all applicable lifecycle/admin/batch/retention paths under resource-head-before-row order, exact UUID and actual lease attempt authority, full validated semantic replacement on live/archive, and retained request targets. Older reactivated Jobs retain their generation <= head high-water counter. Existing specialized Slides/admin-webhook guarded APIs remain domain-specialized, not new Claims protocols. Test first with official SQLite/PostgreSQL fixtures, keep this task In Progress until independent review, no producer enablement/push/PR.
Task 3 sole implementation started at d880dd0eedb135ad2fb8c92d11b67d2ad12a1201 in assigned worktree. Brief/interfaces/constraints read. ADR required: no new ADR; Docs/ADR/085-jobs-resource-admission-and-domain-publication.md governs shared Jobs lifecycle and domain publication; implementation does not change the architecture decision. Report: .superpowers/sdd/2026-11-01-claims-jobs-stage2c-cluster-rebuilds/task-3-report.md. Status retained In Progress for controller review.
Task 3 implementation evidence: final core 202 passed / 0 skips / 440 warnings (/tmp/task3-final-core-green.log); required characterization 189 passed / 0 skips / 3222 warnings (/tmp/task3-characterization-required-final.log); other affected characterization 237 passed / 0 skips / 1461 warnings (/tmp/task3-characterization-affected.log). Earlier real TTL PostgreSQL barrier regressions (187 passed, 2 failed) were fixed by preserving opted-out UPDATE/CTE lock points, not dismissed as intermittent failures; exact RED and corrective evidence is in task-3-report.md. Touched compilation, focused Ruff and scoped Black checks pass. Full touched-production Bandit exits 1 with only the byte-equivalent BASE manager B608 (BASE line10058, current line10239), zero new findings/errors; baseline comparison exits0 (/tmp/task3-bandit-baseline-diff.log). Existing test observer initialized ignored worktree-local Databases/jobs.db on early runs; final fixtures/commands use temporary observer storage, and no runtime DB was deleted/reset or staged. No operator/live migrations, nested reviewers, producer rollout or Task4+ work. Report: .superpowers/sdd/2026-11-01-claims-jobs-stage2c-cluster-rebuilds/task-3-report.md. Keep In Progress for controller review; normal scoped hooks and local commit follow.
Final same-code combined qualification after the shared ordinary-mutation guard fix: 584 passed, 0 skipped, 5118 warnings, 201.18s, exit0 (/tmp/task3-final-combined.log), union of core/required/affected groups over 22 modules. Source/test staged diff SHA256 remained 1d2863186cb88d8f47365bdc7a7fced70a5fd39d73a8dd70a0c8e177797d9671 before/after qualification. Report maps the guard coverage and explicitly documents broad maintenance discovery/head/active-row lock breadth and throughput review concern; no new load architecture. Focused checks and scoped pre-commit passed; full Bandit remains exit1 only for the unchanged BASE B608. No independent reviewer dispatched; retain In Progress for controller review. Normal scoped commit hooks will run on this checkpoint.
Hook evidence clarification: the normal configured pre-commit checks were explicitly run over all ten scoped paths before local commit and passed, including canonical Backlog and Python syntax guards. Git has no automatic pre-commit hook installed and no core.hooksPath configured; the commit log therefore has no automatic hook output. No hook bypass, SKIP, or hook-configuration change was used. This corrects the earlier prospective statement about commit-time hooks; explicit staged hook qualification is the evidence. Final local commit and exact commands/results are recorded in task-3-report.md; task remains In Progress.
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
