---
id: TASK-15283
title: Plan Claims Jobs Stage 2C cluster rebuilds
status: Done
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
- [x] #4 Self-review, documentation-only verification and local commit evidence are recorded; execution waits for plan review.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Requester approved the revised written design after local commit 92ac607403. No implementation authorized. Use only the existing linked worktree on codex/claims-jobs-stage2c-cluster-rebuild-design. Task ID audit fetched origin/dev, checked all current open PR file lists (one open PR, none truncated) and the local branch; highest id 15282, so reserved 15283. ADR check: no additional ADR required; approved ADR-085 governs resource admission/publication and composes with ADR-003 and ADR-058. Record its current owner approval before planning. Bandit is not applicable while changes remain documentation/task records only.
Drafted Docs/superpowers/plans/2026-11-01-claims-jobs-stage2c-cluster-rebuilds.md: five stages, eleven reviewable test-first tasks, explicit file map and shared interfaces, config names/ceilings, lifecycle mutation inventory, official PostgreSQL fixtures, fault/property tests, rollout/rollback and merge gates. Recorded written-design approval, completed TASK-15282 and accepted ADR-085 with its index entry. Self-review is checking full spec coverage, interface spelling/types, five Review Focus risks and plan proportion; tightened async admin guard reuse, single-request captured routing, source transaction connection reuse and constructor-free async storage. No implementation started; Bandit not applicable to documentation-only scope.
Self-review completed against every spec section and the five Review Focus risks; all implementation stages remain Not Started. Clarified exact immutable limit fields, payload/settings equality, live active-head authority, asynchronous reuse of the existing Jobs role/domain guard in API dependencies, source snapshot transaction-connection reuse, non-creating Chroma helper signatures, route-key availability and one captured route context per HTTP request. Removed an unintended embedding-ID coverage change to keep algorithm behavior within the approved migration. Technical inability to provide existing-only SDK access or prove RLS-visible cleanup remains an explicit stop-and-review gate, not permission to weaken safety. Fresh verification: 165 Backlog/format/duplicate-ID tests passed (32 warnings, 2.12s), plus 46 unchanged clustering/admission/owner-read baseline tests passed (105 warnings, 1.91s), all zero skips. Logs /tmp/claims-stage2c-plan-backlog-20261101.log and /tmp/claims-stage2c-plan-baseline-20261101.log. Markdown parser checked 90 local links and Python AST parsed all 11 plan test examples; five stages and eleven sequential tasks verified. Bandit not applicable: only documentation/task records changed.
Applicable staged hooks passed with only normal no-code skips; log /tmp/claims-stage2c-plan-hooks-20261101.log. Official duplicate-task-ID check, exact canonical checks for TASK-15282/TASK-15283 and staged whitespace checks passed. The reviewed commit scope is six documentation/task files, including the new plan; no production/test code, push, PR or implementation has occurred. Preparing the verified local planning commit; this task remains In Progress for plan handoff/review. All five implementation stages remain Not Started. Preserve the previously requested subagent-driven execution preference, but do not execute before the requester reviews the plan.
Requester authorized correction of all five validated implementation-plan review findings on 2026-11-01. Scope: plan and this task only; implementation, rebase, push and PR creation remain unauthorized. Tighten shared Jobs failure finalization with acquired UUID fencing, reject resource-required types through independent idempotent-operation admission/replay, protect admitted semantics in live/archive payload replacement, establish and restore Jobs RLS context for owned status/admission, and scope Watchlists cluster reads/count writes/notifications to the cluster owner. Each correction will have explicit owning tasks and regression assertions. The prior review used a real WorkerSDK stale-failure reproduction, temporary SQLite admission/payload probes, six RLS unit tests and source tracing; this is review evidence, not passing future implementation tests. Preserve the approved design and shared Jobs lifecycle ownership.
Revised the implementation plan for all five authorized corrections. Task 2 now closes independent receipt admission/replay bypasses and defines token-restoring scoped Jobs RLS context; Task 3 includes WorkerSDK acquired-UUID failure fencing regardless of optional lease enforcement plus live/archive contract-preserving payload replacement; Task 4 scopes owned reads; Tasks 9-10 scope actual threaded admission without synthetic admin; Task 8 adds owner-predicated aggregate operations and foreign-subscription disclosure/mutation regression coverage. Added exact helper parameters, bounded conflict errors, named regression cases, backend/side-effect assertions and coverage rows. Self-review preserved all approved spec/ADR boundaries, five Not Started stages, eleven tasks and 55 pending steps. Clarified keyless prune setup and explicit PostgreSQL numeric-ID reuse in the isolated fixture so the stale-worker regression does not assume SQLite sequence behavior. Fresh verification: 165 Backlog/format/duplicate-ID tests passed, zero skips, 32 warnings, 1.88s (/tmp/claims-stage2c-plan-review-fixes-backlog-20261101.log); Markdown parser validated both local links, all eleven Python examples parsed with AST, and all five findings mapped to their defining/consuming tasks. Whitespace check passed. Bandit not applicable: only the plan and this task record changed; no production/test code or PostgreSQL runtime qualification was performed. Preparing normal scoped hooks and a local correction commit; execution remains paused pending revised-plan review.
Fresh full-branch documentation/task-scoped pre-commit passed all applicable hooks with only normal no-code skips; log /tmp/claims-stage2c-plan-review-fixes-hooks-20261101.log. The official duplicate-task-ID checker, canonical TASK-15283 format and staged whitespace checks also passed. Only the plan and TASK-15283 are staged for the normal local correction commit, docs(claims): close stage 2c plan review gaps (TASK-15283). No implementation gate or execution checkbox has been marked complete. Task remains In Progress for requester review; preserve the selected subagent-driven method for a separately approved execution handoff.
Execution handoff on 2026-11-01: user approved the corrected plan, paused to ask about the existing shared WorkerSDK, then said ok continue. Verified clean linked worktree and 44 baseline clustering/admission/owner-read tests passed with zero skips (101 warnings, 1.45s; /tmp/claims-stage2c-execution-baseline-20261101.log). Fresh task-ID/ADR inventory explicitly targeted origin rmusser01/tldw_server: 37 open PRs with all paginated file lists, 223 selected Backlog/ADR paths, max task ID 15410 and no collision with this plan/design or ADR-085. This supersedes the earlier planning inventory that accidentally used gh default upstream. Created TASK-15411..TASK-15421 for the eleven review units and initialized this plans isolated SDD ledger. Latest dev 61d02ed484a3c80f5423676b959cf14e7b572a62 differs only in disjoint Moderation work; no rebase needed for this local handoff. No additional ADR needed; accepted ADR-085 governs. No push/PR/merge authorized.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Requester approved revision 05f3cf32be and explicitly authorized continuing after WorkerSDK clarification. Completed the five-stage, eleven-task test-first plan with all five review corrections; planning verification passed 165 Backlog/format tests, eleven Python examples, local links and scoped hooks. Execution is tracked separately in TASK-15411 through TASK-15421. Planning is complete; implementation is not claimed complete.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
