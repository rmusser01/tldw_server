---
id: TASK-15415
title: Implement Claims cluster source snapshot and cleanup locks
status: In Progress
labels:
- claims
- jobs
- implementation
dependencies:
- TASK-15414
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Bounded owner snapshots, publication schema and fail-closed reference/ownership integrity using existing DB_Management.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The approved task's interfaces, safety invariants and scope are implemented without Claims-owned lifecycle state.
- [ ] #2 Test-first RED/GREEN, appropriate characterization and official backend fixture evidence are recorded with exact results and skips.
- [ ] #3 Focused lint, compilation, touched-production Bandit, normal hooks, local commit and independent spec/quality review are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved plan task 5: Docs/superpowers/plans/2026-11-01-claims-jobs-stage2c-cluster-rebuilds.md. ADR required: no additional ADR; accepted Docs/ADR/085-jobs-resource-admission-and-domain-publication.md governs the approved resource/publication boundary with ADR-003 and ADR-058. Execute only in the existing linked worktree/branch, test first, use shared Jobs lifecycle APIs, official backend fixtures, focused verification and touched-code Bandit, normal hooks and scoped commits. Task-ID audit explicitly targeted rmusser01/tldw_server after discovering gh defaults to upstream: fetched dev 61d02ed484a3c80f5423676b959cf14e7b572a62, checked all 37 open PR paginated file lists (223 selected Backlog/ADR records, no truncation) and current branch; highest observed id 15410. Reserved TASK-15411 through TASK-15421 for the eleven reviewable plan tasks. No collision with planning records TASK-15282/TASK-15283 or ADR-085. Preserve unrelated main-checkout changes; no push/PR/merge authorized.
Tasks1-4 prerequisites independently reviewed complete; begin approved Task5 using task-5-brief.md, constraints.md and task-5-integration-notes.md in this plans SDDworkspace. Produce strict owner-consistent bounded source snapshots/digests, publicationhead/receipt schema in fresh and upgradepaths with real restricted-role policies, transactionconnection helpers and bounded foreign-reference failclosed cleanup. Coordinate add_claim_to_cluster destination-first cluster->Media->Claims order; do not rely on hiddenMedia innerjoin or absentFK as proof. Testfirst official Media PostgreSQL fixtures; no syntheticadmin, missingstorageprovisioning, globalrolewidening, transferrepair or legacyalgorithmrewrite. Keep task InProgress for independentreview; configuredlint/Bandit/normalhooks/scopedlocalcommit required, no push/PR/merge.
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
