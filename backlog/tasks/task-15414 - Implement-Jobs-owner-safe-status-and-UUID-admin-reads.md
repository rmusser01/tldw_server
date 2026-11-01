---
id: TASK-15414
title: Implement Jobs owner-safe status and UUID admin reads
status: In Progress
labels:
- claims
- jobs
- implementation
dependencies:
- TASK-15413
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Allowlisted owner read route with scoped Jobs RLS and unchanged administration guarded by exact UUID identity.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The approved task's interfaces, safety invariants and scope are implemented without Claims-owned lifecycle state.
- [ ] #2 Test-first RED/GREEN, appropriate characterization and official backend fixture evidence are recorded with exact results and skips.
- [ ] #3 Focused lint, compilation, touched-production Bandit, normal hooks, local commit and independent spec/quality review are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved plan task 4: Docs/superpowers/plans/2026-11-01-claims-jobs-stage2c-cluster-rebuilds.md. ADR required: no additional ADR; accepted Docs/ADR/085-jobs-resource-admission-and-domain-publication.md governs the approved resource/publication boundary with ADR-003 and ADR-058. Execute only in the existing linked worktree/branch, test first, use shared Jobs lifecycle APIs, official backend fixtures, focused verification and touched-code Bandit, normal hooks and scoped commits. Task-ID audit explicitly targeted rmusser01/tldw_server after discovering gh defaults to upstream: fetched dev 61d02ed484a3c80f5423676b959cf14e7b572a62, checked all 37 open PR paginated file lists (223 selected Backlog/ADR records, no truncation) and current branch; highest observed id 15410. Reserved TASK-15411 through TASK-15421 for the eleven reviewable plan tasks. No collision with planning records TASK-15282/TASK-15283 or ADR-085. Preserve unrelated main-checkout changes; no push/PR/merge authorized.
Tasks1-3 shared Jobs prerequisites completed and independently reviewed; Task3 final correction9dbe881ab1 has605 passing affected tests/zero skips and configured Ruff/hooks pass. Begin approved Task4 Jobs-owned safe owner projection and UUID-bound admin detail. Read task-4-brief.md, constraints.md, Task1/2 interfaces and task-4-integration-notes.md in this plans isolated SDD workspace. New owner route is authenticated but separate from global-admin controls; use actual executing Jobs owner/domain RLS context with token restoration. Foreign/disallowed UUIDs return404; allowlisted data only. Keep admin role/domain checks and UUIDprecondition exact; no Claims status proxy or privilege widening. Test first with official restricted-role fixtures, configured lint/Bandit/normalhooks, retain In Progress for independent review. No producer/push/PR/merge.
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
