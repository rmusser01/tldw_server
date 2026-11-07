---
id: TASK-15465
title: Rebase and publish Claims cluster rebuild Jobs integration
status: In Progress
labels:
- claims
- jobs
- publication
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase the completed Claims Stage2C branch onto current dev, preserve shared Jobs workflow integration, repair only private identifier collisions, reverify and open a PR against dev as explicitly requested on 2026-11-07. No merge authorization.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Current dev integration and namespace repairs preserve both resource and enrolled-workflow behavior.
- [ ] #2 Fresh scoped backend, static, security and hook results plus repository-wide attempt limitations are recorded.
- [ ] #3 Verified branch is pushed and PR against dev is created without fabricating a human Change summary or bypassing gates.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
User explicitly authorized rebase, reverify and PR. Fresh audit dfac3ca7f34a296a2dc3a3fbc340efdd1a0bae5c/all43openPR complete paginatedfiles: max task15451/maxADR093; all13privateClaims tasks15414..15426 now collide; privateADR086/087 collide. Allocate OUR task records15452..15464 in ascending old-ID order, this publication15465; move OUR acceptedADR086->094 and Proposed087->095, preserve accepted rationale. No otherowners records changed. ADR assessment: no new decision, existing resource/publication boundary and ADR003/058/082 govern; identifier-only moves do not accept SDK ADR. Work only isolatedworktree, rootvenv/backlog-py, normalhooks, noCIcancel/noadminmerge/no directdevpush. Pre-rebaseHEAD0f370e5e235f74bee76fe8c0c8ff55ca2114620d, originbranchabsent, noarmedPR. Old /tmp evidence expired; fresh evidence required.
Rebased40commits onto devdfac3ca7f34a296a2dc3a3fbc340efdd1a0bae5c, retained upstreamworkflow capability/enrollment/DBtime/recovery and resourcehead/exactUUIDguards. Conflictfiles ADRindex, Jobscontracts reexports, manager preparation/recovery, bothlifecyclerelease and recoverycounterfake; no side discarded. FrozenUUIDPGrelease query retained with optionalworkflowprojection. All13private task IDs renumbered15452..15464 and actualdependencies repaired viaCLI; OUR acceptedADR094/Proposed095 activefiles/references, body rationale preserved. Freshcompile100Pythonpass; configuredfullscopeRuff114vs115dev with1newimportsort in resolvedcontractsexports, correctedbyconfiguredformatter and narrowcompile/Ruffpass. Bandit58prod1baseline/currentfinding0errors. FreshClaimsJobs/storage/rootattempt and independentreadonlyconflictintegrationreview running. Plan Docs/Plans/IMPLEMENTATION_PLAN_claims_stage2c_publication_20261107.md. No remote branch yet/noPR/merge.
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
