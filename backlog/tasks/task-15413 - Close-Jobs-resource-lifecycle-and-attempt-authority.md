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
Independent Task3 review at e52cd86a674f82715f065e2c15ae657bbb8dd3fc returned Issues found / Needs fixes. Four Important findings validated by controller source/evidence checks: (1) ordinary discovery-window numeric-ID replacement bypasses frozen UUID in single-row cancellation (reviewer official PG probe cancelled resource replacement and left active head), analogous unfrozen reads in completion/failure/release require audit; (2) TTL discovers/locks noneligible resource heads and recovery fetches unbounded populations before batch selection, causing unrelated-owner blocking/materialization; (3) SQLite correlated JSON target membership is quadratic (64/128 targets 11247/42956 VM steps in focused probe); (4) ordinary configured touched-file Ruff confirmed all 21 new findings, narrow selectors missed the gate. No Critical finding; no dependent Task4 work until corrected/re-reviewed. Fix round1/3 resumes original implementer with test-first regressions, bounded eligibility/selection, efficient exact-ID+UUID matching and full configured lint; no gate weakening or new lifecycle system. Full report: .superpowers/sdd/2026-11-01-claims-jobs-stage2c-cluster-rebuilds/task-3-review.md.
Task3 fix round1 against e52cd86a674f82715f065e2c15ae657bbb8dd3fc addresses all four independently reviewed Important findings: frozen ID+UUID authority at PG single-row rereads/cancellation writes; actual eligible TTL/reschedule discovery and bounded metadata-only oldest-lease recovery before head locks, with head/row SKIP LOCKED; indexed target-driven null-safe SQLite live/archive matching; and all21 ordinary configured Ruff findings with explicit named local pytest fixture registration, no gates/selects/ignores/noqa weakening. New deterministic mixed-owner/corrupt-or-locked-head/row, bounded materialization, VM/query-plan and ordinary-to-resource ID-replacement regressions pass. Final same-code22-module qualification: 605 passed, 0 skipped, 5160 warnings, 208.09s, exit0 (/tmp/task3-fix1-combined-final-qualified.log). Initial602pass/3fail characterization and one interrupted run remain explicitly recorded; strict SQL hook detection was corrected without changing race assertions, final eight race checks pass. Full configured Ruff and compilation over all nine original Python files plus release race-hook test pass; scoped Black passes. Full touched-production Bandit remains exit1 only for byte-equivalent BASE B608 at manager10244, zero new findings/errors. Final source/test diff SHA256 c622d20be1765a33e37a94ffcc25644ed341074cc98eecfa8db36c1d600b9c2b. Report appended at .superpowers/sdd/2026-11-01-claims-jobs-stage2c-cluster-rebuilds/task-3-report.md, including precise commands, warnings triage and bounded contended-window semantics (may return less than budget; no additional heads acquired after row locking). Keep In Progress for independent controller review. No ADR/public API/schema change or Task4+ work; normal scoped hooks/local commit follow.
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
