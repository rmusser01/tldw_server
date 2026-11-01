---
id: TASK-15411
title: Implement Claims Stage 2C shared contracts and schema
status: In Progress
labels:
- claims
- jobs
- implementation
dependencies:
- TASK-15283
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Contracts, resource-required registration, additive SQLite/PostgreSQL resource schema and RLS; no producer or lifecycle implementation yet.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The approved task's interfaces, safety invariants and scope are implemented without Claims-owned lifecycle state.
- [ ] #2 Test-first RED/GREEN, appropriate characterization and official backend fixture evidence are recorded with exact results and skips.
- [ ] #3 Focused lint, compilation, touched-production Bandit, normal hooks, local commit and independent spec/quality review are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved plan task 1: Docs/superpowers/plans/2026-11-01-claims-jobs-stage2c-cluster-rebuilds.md. ADR required: no additional ADR; accepted Docs/ADR/085-jobs-resource-admission-and-domain-publication.md governs the approved resource/publication boundary with ADR-003 and ADR-058. Execute only in the existing linked worktree/branch, test first, use shared Jobs lifecycle APIs, official backend fixtures, focused verification and touched-code Bandit, normal hooks and scoped commits. Task-ID audit explicitly targeted rmusser01/tldw_server after discovering gh defaults to upstream: fetched dev 61d02ed484a3c80f5423676b959cf14e7b572a62, checked all 37 open PR paginated file lists (223 selected Backlog/ADR records, no truncation) and current branch; highest observed id 15410. Reserved TASK-15411 through TASK-15421 for the eleven reviewable plan tasks. No collision with planning records TASK-15282/TASK-15283 or ADR-085. Preserve unrelated main-checkout changes; no push/PR/merge authorized.
Task 1 execution started at BASE 54ab18f94838e40a406b0dfb5a3b409b30e5491d in assigned worktree/branch. Read task-1 brief, binding constraints, worktree AGENTS and TDD/execution/verification skills; scoped to neutral contracts, resource policy, durable schema/RLS and tests. ADR assessment unchanged: ADR-085 governs; no new durable rule. Generic resource owners preserve Jobs string compatibility; Claims payload owners use existing strict routable-owner helper. Controller retains review/finalization; task remains In Progress.
Task 1 implementation evidence (controller review pending; keep In Progress): exact contracts/resource policy, neutral strict Claims settings/payload validation, additive SQLite/PostgreSQL resource tables, immutable nullable jobs/archive binding metadata, and owner/domain RLS implemented. Controller clarified singular backend paths operations/sqlite/resource.py and operations/postgres/resource.py; used those paths. Initial RED: missing two contract modules; executable schema RED 3 failed on absent resource columns before production changes. Initial GREEN 136 passed. Enabled characterization exposed two legacy PG archive owner-column-order failures, fixed by explicit prerequisite; additional lone-surrogate canonical UTF-8 test failed then passed after encoding validation. Final combined new+named characterization+existing operation/Claims contracts RUN_JOBS=1: 494 passed, 0 skipped (1001 warnings), actual official jobs_pg_dsn/restricted-role PostgreSQL exercised. After comment-only cleanup: 150 passed, 0 skipped; RLS ratchet 21 passed. 14-file compilation and focused Ruff passed; 11-production-file Bandit 0 findings/0 scanner errors (5616 LOC). Initial exact characterization had 44 RUN_JOBS-gated skips; corrected by enabling RUN_JOBS, not reported as parity. Scoped pre-commit 10 passed/4 normal no-file skips; diff whitespace clean. Report with exact storage/API fields, commands/logs and full path list: .superpowers/sdd/2026-11-01-claims-jobs-stage2c-cluster-rebuilds/task-1-report.md. ADR assessment unchanged: ADR-085; no new rule. Local commit follows. No admission/producer/lifecycle implementation or other tasks/ledger/constraint edits; controller owns independent review and whole-branch qualification.
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
