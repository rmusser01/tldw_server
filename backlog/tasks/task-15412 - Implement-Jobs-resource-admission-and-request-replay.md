---
id: TASK-15412
title: Implement Jobs resource admission and request replay
status: In Progress
labels:
- claims
- jobs
- implementation
dependencies:
- TASK-15411
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Atomic resource admission/replay, independent receipt-facade guards and scoped Jobs RLS context.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The approved task's interfaces, safety invariants and scope are implemented without Claims-owned lifecycle state.
- [ ] #2 Test-first RED/GREEN, appropriate characterization and official backend fixture evidence are recorded with exact results and skips.
- [ ] #3 Focused lint, compilation, touched-production Bandit, normal hooks, local commit and independent spec/quality review are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved plan task 2: Docs/superpowers/plans/2026-11-01-claims-jobs-stage2c-cluster-rebuilds.md. ADR required: no additional ADR; accepted Docs/ADR/085-jobs-resource-admission-and-domain-publication.md governs the approved resource/publication boundary with ADR-003 and ADR-058. Execute only in the existing linked worktree/branch, test first, use shared Jobs lifecycle APIs, official backend fixtures, focused verification and touched-code Bandit, normal hooks and scoped commits. Task-ID audit explicitly targeted rmusser01/tldw_server after discovering gh defaults to upstream: fetched dev 61d02ed484a3c80f5423676b959cf14e7b572a62, checked all 37 open PR paginated file lists (223 selected Backlog/ADR records, no truncation) and current branch; highest observed id 15410. Reserved TASK-15411 through TASK-15421 for the eleven reviewable plan tasks. No collision with planning records TASK-15282/TASK-15283 or ADR-085. Preserve unrelated main-checkout changes; no push/PR/merge authorized.
Task 1 prerequisites completed in commits 5b12a5f3fa and 3a3c0991ce; independent spec/quality review and fix re-review are clean. Begin approved plan Task 2 using shared contracts/schema, task-1-interfaces.md and task-2-integration-notes.md in this plans isolated SDD workspace. Resource bindings are immutable on UPDATE and therefore must be inserted atomically as complete metadata; schema permits payload representation rewrites while Task 3 later guards semantic replacement. Queue/fair-share/payload preparation must remain shared with ordinary admission while retained replay runs first under the resource lock. This task also closes independent receipt-facade bypasses and defines token-restoring Jobs-specific RLS context. Test first with official PostgreSQL fixtures; no Claims lifecycle, producer enablement, push or PR.
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
