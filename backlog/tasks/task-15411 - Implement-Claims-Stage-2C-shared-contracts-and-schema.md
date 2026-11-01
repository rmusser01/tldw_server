---
id: TASK-15411
title: Implement Claims Stage 2C shared contracts and schema
status: Done
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
- [x] #1 The approved task's interfaces, safety invariants and scope are implemented without Claims-owned lifecycle state.
- [x] #2 Test-first RED/GREEN, appropriate characterization and official backend fixture evidence are recorded with exact results and skips.
- [x] #3 Focused lint, compilation, touched-production Bandit, normal hooks, local commit and independent spec/quality review are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Approved plan task 1: Docs/superpowers/plans/2026-11-01-claims-jobs-stage2c-cluster-rebuilds.md. ADR required: no additional ADR; accepted Docs/ADR/085-jobs-resource-admission-and-domain-publication.md governs the approved resource/publication boundary with ADR-003 and ADR-058. Execute only in the existing linked worktree/branch, test first, use shared Jobs lifecycle APIs, official backend fixtures, focused verification and touched-code Bandit, normal hooks and scoped commits. Task-ID audit explicitly targeted rmusser01/tldw_server after discovering gh defaults to upstream: fetched dev 61d02ed484a3c80f5423676b959cf14e7b572a62, checked all 37 open PR paginated file lists (223 selected Backlog/ADR records, no truncation) and current branch; highest observed id 15410. Reserved TASK-15411 through TASK-15421 for the eleven reviewable plan tasks. No collision with planning records TASK-15282/TASK-15283 or ADR-085. Preserve unrelated main-checkout changes; no push/PR/merge authorized.
Task 1 execution started at BASE 54ab18f94838e40a406b0dfb5a3b409b30e5491d in assigned worktree/branch. Read task-1 brief, binding constraints, worktree AGENTS and TDD/execution/verification skills; scoped to neutral contracts, resource policy, durable schema/RLS and tests. ADR assessment unchanged: ADR-085 governs; no new durable rule. Generic resource owners preserve Jobs string compatibility; Claims payload owners use existing strict routable-owner helper. Controller retains review/finalization; task remains In Progress.
Task 1 implementation evidence (controller review pending; keep In Progress): exact contracts/resource policy, neutral strict Claims settings/payload validation, additive SQLite/PostgreSQL resource tables, immutable nullable jobs/archive binding metadata, and owner/domain RLS implemented. Controller clarified singular backend paths operations/sqlite/resource.py and operations/postgres/resource.py; used those paths. Initial RED: missing two contract modules; executable schema RED 3 failed on absent resource columns before production changes. Initial GREEN 136 passed. Enabled characterization exposed two legacy PG archive owner-column-order failures, fixed by explicit prerequisite; additional lone-surrogate canonical UTF-8 test failed then passed after encoding validation. Final combined new+named characterization+existing operation/Claims contracts RUN_JOBS=1: 494 passed, 0 skipped (1001 warnings), actual official jobs_pg_dsn/restricted-role PostgreSQL exercised. After comment-only cleanup: 150 passed, 0 skipped; RLS ratchet 21 passed. 14-file compilation and focused Ruff passed; 11-production-file Bandit 0 findings/0 scanner errors (5616 LOC). Initial exact characterization had 44 RUN_JOBS-gated skips; corrected by enabling RUN_JOBS, not reported as parity. Scoped pre-commit 10 passed/4 normal no-file skips; diff whitespace clean. Report with exact storage/API fields, commands/logs and full path list: .superpowers/sdd/2026-11-01-claims-jobs-stage2c-cluster-rebuilds/task-1-report.md. ADR assessment unchanged: ADR-085; no new rule. Local commit follows. No admission/producer/lifecycle implementation or other tasks/ledger/constraint edits; controller owns independent review and whole-branch qualification.
Fix round 1 started from FIX_BASE 5b12a5f3fa0f31f7d77ce4bf5c3e3b91407657c2 after reading task-1-review.md. Implement controller ruling: resource metadata and bound domain/owner/UUID/type remain immutable, but payload physical representation is not frozen by schema; full validated/decrypting semantic replacement boundary stays Task 3. Write failing live/archive representation and installed-SQLite-trigger refresh tests first; strengthen both-table restricted-role INSERT/UPDATE/DELETE matrix. No new architecture rule or expansion into admission/producer/Task 3; keep In Progress for re-review.
Fix round 1 TDD: before production edits, RUN_JOBS=1 python -m pytest -q -ra tldw_Server_API/tests/Jobs/test_jobs_resource_migrations.py produced 8 failed/74 passed/0 skipped (176 warnings): SQLite live/archive serialization and AES-GCM rotation, PostgreSQL live/archive AES-GCM rotation, and both installed-SQLite-trigger refresh tests failed at the existing immutability guard. Minimal correction excludes payload from bound identity comparison while retaining all five metadata fields and domain/owner/UUID/type; SQLite drops/recreates only its owned immutable trigger on every migration. GREEN same command: 82 passed/0 skipped (176 warnings). Both-table restricted-role matrix now proves successful same-scope INSERT/UPDATE/DELETE with rowcounts, rejects missing-context and foreign owner/domain INSERT and owner/domain-changing UPDATE, and proves zero-row foreign-scope updates/deletes plus unchanged foreign rows. No Task 3 API enforcement, producer, admission, lifecycle or unrelated changes; prerequisite cannot enable producers alone. Compilation/Ruff/Black check passed on 3 touched Python files; Bandit on 2 touched production modules: 0 findings/0 scanner errors, 174 LOC. Full fix evidence and regression/hooks/commit results are appended to the assigned report; keep In Progress for re-review.
Controller completion gate: task-1-fix1-review.md from independent reviewer 01a198a5-7d7a-7561-8288-e2ec8fbb99ef verdicts both findings ADDRESSED, no new breakage, Task 1 spec PASS and quality APPROVE. Controller checked saved 494/21 test summaries and zero-finding/error Bandit JSON; read appended 8-failure RED, 82-test GREEN and 195-test final regression evidence before re-review. No unresolved Task 1 finding or cannot-verify requirement remains; future semantic replacement/admission/lifecycle behavior and whole-branch qualification are explicitly assigned to later tasks, not claimed complete. Lean interface handoff is preserved in the plan-scoped ignored SDD workspace. Controller will commit plan/checkpoint tracking only before dispatching Task 2.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed approved plan Task 1: neutral Claims contract, shared Jobs resource dataclasses/registration, additive SQLite/PostgreSQL live/archive metadata, heads/request receipts and owner/domain RLS. Commits 5b12a5f3fa and 3a3c0991ce. Independent review found an overstrict physical payload guard and an RLS INSERT-test gap; both were corrected and scoped re-review approved spec compliance and quality. Final focused evidence: 494 initial combined tests plus 21 RLS ratchet tests, then 82 correction tests and 195 affected migration/RLS regressions; all final runs zero skips. Compilation/lint/Bandit/hooks passed; no new findings/errors. Resource metadata/identity remains immutable; Task 3 still owns semantic payload replacement enforcement. No producer, admission or worker integration enabled yet.
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
