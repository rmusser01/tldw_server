---
id: TASK-15417
title: Implement atomic cluster publication and replay receipts
status: In Progress
labels:
- claims
- jobs
- implementation
dependencies:
- TASK-15416
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Atomic owner-scoped publication, source recheck, replay receipts and monotonic authority/generation fencing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The approved task's interfaces, safety invariants and scope are implemented without Claims-owned lifecycle state.
- [ ] #2 Test-first RED/GREEN, appropriate characterization and official backend fixture evidence are recorded with exact results and skips.
- [ ] #3 Focused lint, compilation, touched-production Bandit, normal hooks, local commit and independent spec/quality review are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved plan task 7: Docs/superpowers/plans/2026-11-01-claims-jobs-stage2c-cluster-rebuilds.md. ADR required: no additional ADR; accepted Docs/ADR/085-jobs-resource-admission-and-domain-publication.md governs the approved resource/publication boundary with ADR-003 and ADR-058. Execute only in the existing linked worktree/branch, test first, use shared Jobs lifecycle APIs, official backend fixtures, focused verification and touched-code Bandit, normal hooks and scoped commits. Task-ID audit explicitly targeted rmusser01/tldw_server after discovering gh defaults to upstream: fetched dev 61d02ed484a3c80f5423676b959cf14e7b572a62, checked all 37 open PR paginated file lists (223 selected Backlog/ADR records, no truncation) and current branch; highest observed id 15410. Reserved TASK-15411 through TASK-15421 for the eleven reviewable plan tasks. No collision with planning records TASK-15282/TASK-15283 or ADR-085. Preserve unrelated main-checkout changes; no push/PR/merge authorized.
Begin approved Task7 atomic publication after Task6 fix1 independent re-review passed in e3ba2dd187. Use task-7-brief.md, constraints.md, task-5-interfaces.md and task-7-integration-notes.md. One Media DB transaction, head-before-cleanupcluster/Media/Claims locks, immutable receipt replay and generation fencing; no Jobs lifecycle in Claims. Close carried Task5 Mapping annotation minor while preserving defensive copy/explicit JSON conversion. Official restricted PostgreSQL fixture and SQLite tests, faults/barriers/properties test-first, configured quality/Bandit/hooks/scoped commit, independent task review before completion. ADR assessment: existing accepted Jobs/domain ADR085 governs; metadata renumber of private Claims ADR deferred to Task11 collision cleanup, no change to accepted rationale or new publication architecture. No push/PR/merge.
Task7 implementation started from clean ebc5cd0f0095ac15936c8fa42fc77c2bffe4ec1e in assigned worktree/branch. Binding brief, constraints and Task5/6 handoffs read. RED publication/parity/barrier tests added before production. Evidence and final handoff: .superpowers/sdd/2026-11-01-claims-jobs-stage2c-cluster-rebuilds/task-7-report.md. Keep In Progress pending controller independent review.
Task7 implementation ready for independent review; keep In Progress. Review BASE now 20b97fad578cb5943248d7f6324230e6fdb55185 (controller namespace-only checkpoint); current governing accepted ADR086, rationale unchanged. Atomic publication/load APIs, lifetime immutable receipts, authority/historical equal-generation UUID fencing, caller-connection cleanup/source locks and owner-bounded writes, skip/exact-empty semantics; Task5 result Mapping annotation minor closed. RED 98 missing APIs then GREEN98; historical RED2/GREEN104; head consistency RED8/covering GREEN14. Final affected scope 209 passed, 0 skips, 432 existing warnings in 30.40s, including 86 real official restricted-role PostgreSQL cases and 118 new tests; barriers/faults/mixed publication-skip ordering properties, actual Jobs prune survival and real calculator handoff covered. All six touched Python paths compile and configured whole-file Ruff is zero (existing-file baseline zero); scoped format passes; all touched production Bandit zero findings/errors, no new suppressions. Exact commands/logs/interfaces/limitations and local commit/hooks recorded in .superpowers/sdd/2026-11-01-claims-jobs-stage2c-cluster-rebuilds/task-7-report.md. No later-task, Task6, parent governance or ledger edits; no push/PR/merge. Controller to dispatch independent review before finalization.
Final self-review caught and test-first closed concurrent cleanup cluster-set drift before destructive writes (restricted PG RED1 then GREEN1). Superseding earlier 209-case evidence: FINAL code qualified once with preserved configured --import-mode=importlib/--strict-markers: 210 passed, 0 skipped/failed/xfail/xpass, 434 existing warnings, 30.48s; 119 new tests, 87 actual official restricted-role PostgreSQL cases. Final log /tmp/task7-final-qualification.log. An earlier warning-display run cleared addopts; it is superseded, not final gate qualification. Refreshed whole-file configured Ruff0, six-file compilation/scoped format pass, touched-production Bandit0 findings/0 errors with no new suppressions. Report contains exact commands, all RED/GREEN evidence and limitations. Keep In Progress for controller independent review; no finalization authorized.
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
