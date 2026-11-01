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
