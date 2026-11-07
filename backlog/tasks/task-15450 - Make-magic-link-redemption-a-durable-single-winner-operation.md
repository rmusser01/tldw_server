---
id: TASK-15450
title: Make magic-link redemption a durable single-winner operation
status: In Progress
references:
- https://github.com/rmusser01/tldw_server/pull/3430
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Consume a verified one-time sign-in token before account or session effects, using the existing blacklist unique token identifier.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Concurrent authorized submissions with the same token yield at most one successful grant.
- [x] #2 A failed durable write or invalid redemption metadata cannot return credentials or mutate account state.
- [x] #3 Existing one-time, active-account and registration behavior remains covered.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. Enforces the existing one-time sign-in contract using the existing blacklist table. Plan: reproduce duplicate and failed claims, implement a committed conditional insert before grants, verify the handler and both SQL dialects.
Review follow-up: accept only explicit insert outcomes. Unknown PostgreSQL command tags and SQLite row counts return 503. Two regressions returned 400 before this change. Final verification after review: 58 SQLite/adjacent tests and 7 PostgreSQL 18 integration tests passed. Ruff, compile, whitespace and changed-code Bandit checks passed.
CI follow-up: compare changed-file mypy diagnostics with the base, fix only introduced type errors, and reconcile the OpenAPI description fingerprint. Existing task remains the scope; no required gate changes or unrelated typing cleanup.
Connected CI cleanup: the changed-file mypy gate also exposes 33 inherited errors in auth.py and the touched integration test. Resolve these mechanically without ignores: unpack existing exception tuples, preserve validated dictionary reads, separate raw refresh IDs, represent optional claims at the call, supply existing details=None defaults, correct Response annotations/injection, and assert fetched test rows exist. Verify focused auth behaviors, exact CI mypy, and unchanged OpenAPI except the intended magic-link description.
CI repair verified: mypy 2.4.1 now passes all four changed Python files (35 diagnostics before, including 33 proven inherited). Refreshed OpenAPI fingerprint and generated ignored rich types; fingerprint and committed path union checks pass, and substituting only the prior magic-link description reproduces the base hash exactly. Current focused auth verification: 91 SQLite/unit tests, one legacy /auth/me HTTP check, and 10 local PostgreSQL tests passed. Ruff, diff checks, canonical task format, exact expiry-guard equivalence, and Bandit comparison passed (base 11 findings, final 10; no new findings). Required hosted CI will rerun after the reviewed push.
The human requester explicitly waived the human-written Change summary requirement for this public-dev PR together with the other three new PRs. Root and independent reviewer found no actionable issue in the CI repair. No merge or auto-merge is authorized by this follow-up.
November 7 requester follow-up explicitly expands CI repair scope to inherited failing tests and fixtures. Plan: repair the missing MCP plugin, cold/warm storage-resolver setup, stale import inventory and reproducible logging/fixture timing faults; preserve all assertions and runtime safety controls; validate each correction, obtain independent review, and port shared corrections to affected public PRs. No merge, auto-merge, production recovery/replay, or private code is authorized. ADR required: no for harness alignment with existing contracts. Root owns task edits and commits; independent agents may edit only assigned files.
Reviewed CI repairs: storage-resolver fixture now supplies the existing BOOTSTRAP trusted context to both membership APIs; all 73 file cases pass with SQLite and isolated PostgreSQL 18. The MCP accounting module explicitly registers its already-declared pytest-mock plugin; 63 tests pass both with disabled autoload and normal autoload. Regenerating the existing import inventory changes only three source line references in each artifact; all 10 inventory tests pass. Logging reload now drains queued writes before resetting module state, with an event-controlled regression that fails without the drain; 22 logging tests and 229 smoke tests pass (final regression-only timer removal rechecked in the 22-test suite). PostgreSQL return now synchronously rolls back unfinished work and scrubs scope before returning ownership to psycopg_pool, so shutdown cannot abandon a queued return. A deterministic live regression failed before the fix; all 29 focused pool/reset/live tests pass, including rollback, scope reuse, immediate closure and the unchanged offline guard. No restore or recovery operation was run. ADR required: no; this restores the existing return/close ownership and isolation contract under ADR-020, without changing recovery policy. Independent cross-review found no actionable issues; reviewers also reran 23 PostgreSQL tests and 22 logging tests. Exact mypy 2.4.1 passes all 10 changed Python files; Ruff and diff checks pass. No new runtime Bandit findings; existing test assertions remain and new regression assertions are documented B101 test-only findings. Original AuthNZ bootstrap case plus predecessor pass 2/2; the historical separate teardown/GC failure has not been reproduced and is not claimed fixed. Shared resolver/MCP/inventory corrections are also being copied to PRs 3428 and 3429. Fresh hosted CI is required after push.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Magic-link redemption commits its unique claim before account/session effects, with the original 102 auth checks and API fingerprint validation retained. The authorized CI follow-up repairs inherited resolver setup, MCP fixture registration and stale import inventory, plus proven logging reload and PostgreSQL return/shutdown races. Focused suites pass (73 resolver, 63 MCP, 10 inventory, 22 logging, 29 PostgreSQL), full smoke passed 229, all 10 changed Python files pass exact mypy 2.4.1 and Ruff, and independent review found no actionable issue. No new runtime Bandit findings. Hosted CI on the new head remains pending; the separate historical AuthNZ teardown/GC error remains unproven. Requester Change summary waiver remains in force; no merge, beta change or restore.
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
