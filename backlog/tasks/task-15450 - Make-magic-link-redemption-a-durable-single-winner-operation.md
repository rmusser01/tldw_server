---
id: TASK-15450
title: Make magic-link redemption a durable single-winner operation
status: Done
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
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Magic-link redemption commits its unique claim before account/session effects. Confirmed replay returns 400; unavailable or uncertain persistence returns 503. Follow-up CI repair resolves introduced and mechanically inherited typing errors in the touched auth files and refreshes the fingerprint for the documented 400/503 behavior. Final local validation: 91 SQLite/unit auth tests, one legacy HTTP check, 10 PostgreSQL tests, exact CI mypy 2.4.1, Ruff, OpenAPI fingerprint/path checks, generated rich TypeScript types, task format, and no new Bandit findings. Remote protected checks must pass on the new head before merge.
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
