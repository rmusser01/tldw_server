---
id: TASK-15283
title: Plan Claims Jobs Stage 2C cluster rebuilds
status: In Progress
labels:
- claims
- jobs
- planning
dependencies:
- TASK-15282
- TASK-9935
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Translate the approved Stage 2C cluster-rebuild design and ADR-085 into a test-first implementation plan. Keep exclusive admission and execution lifecycle in shared Jobs; preserve synchronous defaults, strict owner scope, replay-safe publication, all three review corrections, and controlled rollout. Planning only; implementation requires plan review and explicit handoff.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The approved design and ADR status are recorded and linked without starting implementation.
- [x] #2 A five-stage plan defines exact files, shared interfaces, red-green test cycles and independent commit boundaries starting with Jobs.
- [x] #3 The plan covers all approved invariants and the ownership-transfer, existing-only Watchlists and UUID-bound status-link corrections.
- [ ] #4 Self-review, documentation-only verification and local commit evidence are recorded; execution waits for plan review.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Requester approved the revised written design after local commit 92ac607403. No implementation authorized. Use only the existing linked worktree on codex/claims-jobs-stage2c-cluster-rebuild-design. Task ID audit fetched origin/dev, checked all current open PR file lists (one open PR, none truncated) and the local branch; highest id 15282, so reserved 15283. ADR check: no additional ADR required; approved ADR-085 governs resource admission/publication and composes with ADR-003 and ADR-058. Record its current owner approval before planning. Bandit is not applicable while changes remain documentation/task records only.
Drafted Docs/superpowers/plans/2026-11-01-claims-jobs-stage2c-cluster-rebuilds.md: five stages, eleven reviewable test-first tasks, explicit file map and shared interfaces, config names/ceilings, lifecycle mutation inventory, official PostgreSQL fixtures, fault/property tests, rollout/rollback and merge gates. Recorded written-design approval, completed TASK-15282 and accepted ADR-085 with its index entry. Self-review is checking full spec coverage, interface spelling/types, five Review Focus risks and plan proportion; tightened async admin guard reuse, single-request captured routing, source transaction connection reuse and constructor-free async storage. No implementation started; Bandit not applicable to documentation-only scope.
Self-review completed against every spec section and the five Review Focus risks; all implementation stages remain Not Started. Clarified exact immutable limit fields, payload/settings equality, live active-head authority, asynchronous reuse of the existing Jobs role/domain guard in API dependencies, source snapshot transaction-connection reuse, non-creating Chroma helper signatures, route-key availability and one captured route context per HTTP request. Removed an unintended embedding-ID coverage change to keep algorithm behavior within the approved migration. Technical inability to provide existing-only SDK access or prove RLS-visible cleanup remains an explicit stop-and-review gate, not permission to weaken safety. Fresh verification: 165 Backlog/format/duplicate-ID tests passed (32 warnings, 2.12s), plus 46 unchanged clustering/admission/owner-read baseline tests passed (105 warnings, 1.91s), all zero skips. Logs /tmp/claims-stage2c-plan-backlog-20261101.log and /tmp/claims-stage2c-plan-baseline-20261101.log. Markdown parser checked 90 local links and Python AST parsed all 11 plan test examples; five stages and eleven sequential tasks verified. Bandit not applicable: only documentation/task records changed.
Applicable staged hooks passed with only normal no-code skips; log /tmp/claims-stage2c-plan-hooks-20261101.log. Official duplicate-task-ID check, exact canonical checks for TASK-15282/TASK-15283 and staged whitespace checks passed. The reviewed commit scope is six documentation/task files, including the new plan; no production/test code, push, PR or implementation has occurred. Preparing the verified local planning commit; this task remains In Progress for plan handoff/review. All five implementation stages remain Not Started. Preserve the previously requested subagent-driven execution preference, but do not execute before the requester reviews the plan.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
