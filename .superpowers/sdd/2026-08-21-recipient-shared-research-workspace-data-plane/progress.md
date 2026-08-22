# SDD ledger - plan: Docs/superpowers/plans/2026-08-21-recipient-shared-research-workspace-data-plane.md

## Execution preflight

- 2026-08-21: Rebased eight feature commits onto `origin/dev` at `2e0815c1e4` without conflicts. New HEAD: `0e006b5388`.
- 2026-08-21: Confirmed this is an isolated linked worktree on `codex/research-workspace-power-user-uat`.
- 2026-08-21: Confirmed ChaChaNotes current schema remains V60; Task 2 may use V61.
- 2026-08-21: Two unrelated untracked watchlist templates remain excluded from every task and commit.

## Preflight consistency scan

| Tasks/interfaces | Producer -> consumer | Finding |
| --- | --- | --- |
| 1 -> 5/7 | Task 1 removes unsafe recipient routes; Tasks 5 and 7 install typed bounded replacements | Intentional fail-closed interval; no alias or redirect allowed. |
| 1 -> 8/9 | Task 1 creates route parser/gate and temporary shared shell; Tasks 8/9 connect and render it | Interfaces and file lifecycle agree. |
| 1 -> 10 | Task 1 route-absence assertions become Task 10 request-ledger/security invariants | Tests and final contract agree. |
| 2 -> 3 | Task 2 creates V61 tables/RLS; Task 3 wires the modular store in `ChaChaNotes_DB.py` | Shared file and schema interfaces agree; Task 3 must preserve Task 2 policy wiring. |
| 2 -> 10 | Task 2 migration/policy tests feed the integrated matrix | Final command includes the policy contract and both backends. |
| 3 -> 5 | Recipient store supplies bounded history without creating a thread on read | API/history contract agrees. |
| 3 -> 7 | Store claims, source freezing, leases, replay, and atomic completion back chat orchestration | Lifecycle and failure semantics agree. |
| 4 -> 5 | Access service and read helpers feed bootstrap/source/preview/history endpoints | Authorization-before-owner-DB rule is consistent. |
| 4 -> 6 | Access context supplies authoritative owner/share/scope data to retrieval | Scope and ownership boundaries agree. |
| 4 -> 10 | Authoritative membership repository is exercised by security/UAT matrices | Stale token/request claims remain non-authoritative. |
| 5 -> 7 | Task 5 creates recipient schemas/routes; Task 7 extends chat and generation default | Task 5 is deliberately fail closed until Task 7 wires provider readiness. |
| 5/7 -> 8 | Approved API schemas feed typed frontend client/controller | Task 8 may implement from the spec but cannot be considered integrated before Tasks 5/7. |
| 5 -> 10 | Typed bounded envelopes feed OpenAPI and cross-user tests | Contract and generation steps agree. |
| 6 -> 7 | Retrieval creates verified evidence; generation consumes only budgeted evidence | Dropped/trimmed evidence cannot become citations. |
| 6 -> 10 | Locked RAG policy and provenance postconditions feed sentinel security tests | Security checks agree. |
| 7 -> 9 | Generation default, model target, errors, and citations feed shared chat UI | Generic model catalog remains discovery-only. |
| 7 -> 10 | BYOK scope, context budget, and no-fallback behavior feed integrated tests | Final matrix covers exact share scope and local-only token counting. |
| 8 -> 9 | Typed client, reducer, and controller feed dedicated panes | File lifecycle and state ownership agree. |
| 8 -> 10 | Abort/stale-response/request-ID behavior feeds E2E request ledger | Tests and implementation agree. |
| 9 -> 10/11 | Shared UI feeds Playwright and live CDP acceptance | No banner stack, local controls, or computer-control fallback. |
| 10 -> 11 | Integrated contracts/docs establish the live runner acceptance matrix | Task 11 remains real-process truth; Task 10 stubs only deterministic UI behavior. |

| Task | Internal consistency | Finding |
| --- | --- | --- |
| 1 | Files, route tests, removal assertions, and fail-closed behavior | Consistent. |
| 2 | V61 migration, constraints, PostgreSQL forced RLS, and parity tests | Consistent after post-rebase V60 confirmation. |
| 3 | DB-derived text tenant key, fenced receipts, cleanup, and two-backend tests | Consistent. |
| 4 | Authoritative membership, access ordering, Jobs helper, and preview helper | Consistent. |
| 5 | Typed envelopes, route-scoped errors, bounds, and neutral failures | Consistent; provider readiness remains fail closed until Task 7. |
| 6 | Frozen source scope, locked RAG policy, and runtime provenance checks | Consistent. |
| 7 | Target resolution, exact-scope BYOK, local context budget, generation, and endpoint | Consistent. |
| 8 | Typed client, reducer, abort behavior, idempotent retry, and route reset | Consistent. |
| 9 | Compact Sources/Chat UI, evidence, accessibility, responsive layout, and copy | Consistent. |
| 10 | Cross-layer tests, docs, OpenAPI, Bandit, and browser request ledger | Consistent. |
| 11 | Real backend/WebUI/CDP, ingestion wait, personas, race probe, evidence, closeout | Consistent. |

No preflight ruling was required; the apparent Task 5/7 and Task 8 dependency overlaps are explicit fail-closed staging, not contradictory contracts.

## Baseline verification

- Frontend route/layout baseline: 2 files passed, 17 tests passed.
- Focused sharing, migration, and RLS baseline: 91 passed, 1 failed. The pre-existing failure is `test_sqlite_migration_v38_to_v39_reopens_legacy_database`, which reaches the current V59 migration and fails closed on `Notes attachment v59 registry collision requires explicit repair`. No Task 1 changes existed when this was recorded.
- PostgreSQL AuthNZ sharing integration baseline: 5 tests skipped through the repository fixture because PostgreSQL remained unavailable after its Docker-start attempt. This is an explicit environment limitation, not a passing PostgreSQL result.
- Task implementations must preserve the 91 passing backend tests and may not attribute the V39 baseline failure or PostgreSQL skip to their own changes.

## Task 1 - Fail-closed recipient route gate

- Base: `0e006b5388b48c40a59e6326b68df9934fe75089`
- Implementer: `01a02663-595c-79e2-bfe2-189b54709736` (`Cicero`), `gpt-5.6-terra`, high reasoning.
- Brief: `task-1-brief.md`
- Commit: `1c5ad69f58 fix(workspaces): fail closed for recipient shares`.
- Implementer verification: frontend 23/23; backend 53/53; Ruff, Bandit, and `git diff --check` clean. Full UI typecheck remains non-green on reported pre-existing errors.
- Reviewer: `01a0267e-3fe5-7423-a021-084c1bbe4eae` (`Locke`), `gpt-5.6-terra`, high reasoning.
- Review package: `review-task-1-0e006b5..1c5ad69.diff`.
- Reviewer verdict: spec compliant; approved; no Critical, Important, or Minor findings.
- Controller-resolved review cautions: `rg` found no remaining active UI consumer of the deleted context/banner; `git status --short --branch` confirmed only the two unrelated watchlist templates remain untracked and unstaged.
- Status: complete.

## Task 5 - Typed bounded recipient read APIs

- Base: `52e95b5bb4bae2efefea1acb5b5147e759cfd776`.
- Implementer: `01a0270a-769c-72f3-a7cf-0dadde73d69e` (`Russell`), `gpt-5.6-sol`, high reasoning.
- Brief: `task-5-brief.md`.
- Staging guard: any interim chat route may validate/fail closed only; the removed unconstrained chat contract must not return.
- Commit: `6be3d619b8 feat(sharing): add bounded recipient workspace reads`.
- Reviewer findings accepted for Fix Round 1: raw URL search oracle, non-aggregate preview text bound, malformed canonical cursor misclassification, and OpenAPI/runtime response mismatch.
- Fix Round 1: source search now uses only projected safe fields; preview uses one focus-first aggregate text budget; canonical cursor `InputError` maps to exact typed 422 while operational failures remain 503; every recipient operation declares the strict error wrapper and interim chat advertises only its fail-closed 503 path while retaining its request schema.
- Fix verification: exact Task 5 matrix 102 passed; focused auth/introspection/OpenAPI/old-route target 11 passed; all-new-fix target 7 passed; Ruff clean; Bandit zero findings across 2,022 touched production LOC; `git diff --check` clean.
- PostgreSQL state: untouched. Residual staging condition: Task 7 still owns canonical safe chat generation.
- Fix Round 2 findings: broad API handling of all store `InputError` and representation-count-based preview truncation semantics.
- Fix Round 2: exported a canonical cursor-input-specific subtype and normalized every decoder rejection to it; the route now maps only that subtype to 422 while operational/corrupt stored-data errors remain 503. Preview truncation now follows the canonical source-content flag plus primary-preview shortening, not omitted supplemental chunks.
- Fix Round 2 verification: focused cursor/preview target 8 passed; full canonical SQLite store 33 passed; Ruff clean; Bandit zero findings across 3,112 production LOC with one existing fixed-SQL `nosec B608` skip; `git diff --check` clean. Serial aggregate attempt 1 and xdist/loadfile aggregate attempt 2 exceeded established cleanup-latency profiles; no assertion failed, no third aggregate was run, and no Task 5 pytest worker remained. Fix Round 1 accepted matrix evidence remains 102 passed plus 11 focused auth/introspection/OpenAPI/old-route passes.
- Final re-reviewer: `01a02773-878e-7051-b938-84ca5b4bb887` (`Leibniz`), `gpt-5.6-sol`, high reasoning.
- Final fix review package: `review-task-5-fix2-d800b952da..e09af9a232.diff`.
- Final re-review verdict: approved with no actionable findings. Cursor decoding is isolated to the canonical cursor subtype, operational stored-data failures remain 503, and aggregate preview allocation preserves source-content truncation semantics.
- Residual verification gap: aggregate tests were not rerun by the reviewer; the implementer recorded `104 passed` from the over-profile xdist diagnostic plus focused/store coverage.
- Status: complete.

## Task 6 - Frozen source scope and fail-closed retrieval provenance

- Base: `e09af9a232a3a606cf2117f684bff73e9c8d0d60`.
- Implementer: `01a02779-731d-7d02-bdb6-3000973551d5` (`Wegener`), `gpt-5.6-sol`, high reasoning.
- Brief: `task-6-brief.md`.
- Security gate: owner media is retrieved only through the frozen canonical source snapshot and a fully pinned retrieval-only RAG policy; any missing or out-of-scope provenance fails the whole result before generation.
- Implementation: added immutable canonical source snapshots, exact frozen-scope revalidation, duplicate-media canonical mapping, a signature-sentinel guarded retrieval-only RAG policy, complete-result provenance validation, and bounded deterministic evidence.
- TDD RED: focused Task 6 collection failed because `shared_workspace_chat_service` did not exist. GREEN: 35 focused tests passed.
- Regression verification: bounded Task 4/5/6 sharing/access matrix passed 199 tests; Ruff passed; Bandit reported zero findings across 917 production LOC; `git diff --check` passed.
- PostgreSQL state: untouched and not started. Task 6 adds no schema, policy, migration, fixture, or PostgreSQL query.
- Report: `task-6-implementer-report.md`.
- Residual boundary: Task 7 still owns API orchestration, receipt integration, generation, and citation serialization.
- Status: complete.

## Task 4 - Authoritative access and reusable read helpers

- Base: `ad1fbe49e79424fc2bbcd2353b6c201d7ae37c54`.
- Implementer: `01a026ea-8a7c-7601-a0b3-153f6042c40d` (`Turing`), `gpt-5.6-sol`, high reasoning.
- Brief: `task-4-brief.md`.
- Reconciliation: existing `test_authnz_sharing_postgres.py` must be extended rather than recreated.
- Commit: `52e95b5bb4 feat(sharing): authorize canonical shared workspace reads`.
- Implementer verification: final combined 81 passed/6 standard PostgreSQL skips; Ruff clean for Task 4 scope; Bandit zero findings; diff check clean.
- Reviewer: `01a02706-66f5-7891-a026-907f5e46fe33` (`Carver`), `gpt-5.6-sol`, high reasoning.
- Review package: `review-task-4-ad1fbe4..52e95b5.diff`.
- Reviewer verdict: spec compliant and approved; no Critical, Important, or Minor findings.
- Residual risk: live PostgreSQL execution standard-skipped; internal scope serialization remains a Task 5 gate.
- Status: complete.

## Task 3 - Fenced recipient chat store

- Base: `6378eb63e4c5ffef7856ebd212ee60cfeed882c2`.
- Implementer: `01a026ac-32d1-7403-8a5b-5595069ae815` (`Aquinas`), `gpt-5.6-sol`, high reasoning.
- Brief: `task-3-brief.md`.
- Commit: `7ba1ed7caf feat(sharing): persist fenced recipient chat turns`.
- Implementer verification: SQLite store 27 passed; combined store/RLS 46 passed/2 standard skips; regressions 35 passed; Task-scope Ruff clean; Bandit zero findings; diff checks clean.
- Residual environment limitation: live PostgreSQL store tests standard-skipped.
- Reviewer: `01a026ce-0f45-73e0-bb0c-a18c43a55061` (`Kuhn`), `gpt-5.6-sol`, high reasoning.
- Review package: `review-task-3-6378eb6..7ba1ed7.diff`.
- Reviewer verdict: cohesive design, but four Important findings: SQLite conflict-retention timestamp encoding, reclaim-race winner misclassification, non-discriminating PostgreSQL RLS store test, and role-cleanup ordering after pool closure.
- Controller ruling: all four findings are confirmed against the implementation and accepted. No schema change is required.
- Fix round 1 commit: `8d56a2f4c0 fix(sharing): harden recipient chat store races`.
- Fix verification: SQLite 32 passed; combined 51 passed/2 standard skips; RLS contracts 19 passed; Ruff clean; Bandit zero findings; diff check clean.
- Scoped re-reviewer: `01a026df-981f-71c3-bf67-b152a5b2fd2a` (`Descartes`), `gpt-5.6-terra`, high reasoning.
- Fix review package: `review-task-3-fix1-7ba1ed7..8d56a2f.diff`.
- Task 3 fix round 1/5: 3 addressed, 1 open; commits `7ba1ed7..8d56a2f`.
- Open finding: cleanup still lacks executable handling for pre-fix/native SQLite `CURRENT_TIMESTAMP` rows; canonicalizing only new transitions does not protect existing conflicted receipts.
- Fix round 2 commit: `ad1fbe49e7 fix(sharing): support legacy chat receipt timestamps`.
- Fix verification: focused legacy timestamp RED then GREEN; SQLite 33 passed; combined 52 passed/2 standard skips; Ruff clean; Bandit zero findings; diff checks clean.
- Scoped re-reviewer: `01a026e8-01df-76b1-aa4c-484b3c7e6805` (`Arendt`), `gpt-5.6-terra`, high reasoning.
- Fix review package: `review-task-3-fix2-8d56a2f..ad1fbe4.diff`.
- Task 3 fix round 2/5: 1 addressed, 0 open; commits `8d56a2f..ad1fbe4`.
- Scoped re-review verdict: all findings addressed; no new Critical/Important breakage.
- Residual risk: live PostgreSQL remains unavailable locally; typed PostgreSQL paths and integration collection are preserved with standard fixture skips.
- Status: complete.

## Task 2 - V61 recipient chat schema and forced RLS

- Base: `1c5ad69f58e34db13f486ccf175fb8a9bdfdcf2b`.
- Implementer: `01a02680-fc23-7f33-bd6b-147922fa9e6a` (`Hume`), `gpt-5.6-sol`, high reasoning.
- Brief: `task-2-brief.md`.
- Commit: `8cd3f1d900 feat(sharing): add recipient chat receipt schema`.
- Implementer verification: GREEN 41 passed/2 standard skips; related regressions 69 passed/1 standard skip; Ruff task scope clean; Bandit zero findings; `git diff --check` clean.
- Residual environment limitation: live PostgreSQL fixture unavailable; deterministic PostgreSQL DDL/RLS contracts passed.
- V39 treatment: historical V38->V39 step is tested directly; unrelated V59 repair behavior is unchanged.
- Reviewer: `01a0269b-e840-7761-beb0-bf3e1097752b` (`Faraday`), `gpt-5.6-sol`, high reasoning.
- Review package: `review-task-2-1c5ad69..8cd3f1d.diff`.
- Reviewer verdict: production schema/RLS coherent, but task needs fixes for executable PostgreSQL RLS isolation and constraint coverage.
- Controller ruling: both Important findings are accepted. Task 3's store-level isolation does not replace a restricted-role database-policy test, and Task 2's brief requires PostgreSQL constraint parity rather than source-string checks alone.
- Fix round 1 commit: `6378eb63e4 test(sharing): exercise recipient chat postgres contracts`.
- Fix verification: deterministic PostgreSQL contracts 3 passed; live integration 3 standard fixture skips; full Task 2 matrix 41 passed/4 standard skips; Ruff and diff checks clean.
- Scoped re-reviewer: `01a026a9-9c2c-7862-a253-e3501c5a609b` (`Newton`), `gpt-5.6-terra`, high reasoning.
- Fix review package: `review-task-2-fix1-8cd3f1d..6378eb6.diff`.
- Task 2 fix round 1/5: 2 addressed, 0 open; commits `8cd3f1d..6378eb6`.
- Scoped re-review verdict: all findings addressed; no new Critical/Important breakage.
- Residual risk: live PostgreSQL execution remains unavailable locally, but the new restricted-role and constraint tests standard-skip without custom suppression.
- Status: complete.
