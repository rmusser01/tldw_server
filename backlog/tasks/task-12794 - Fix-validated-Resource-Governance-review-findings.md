---
id: TASK-12794
title: Fix validated Resource Governance review findings
status: Done
assignee: []
created_date: 2026-06-24 04:36
updated_date: 2026-06-25 02:17
labels:
- backend
- resource-governance
- review-fix
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify the Resource_Governance module review findings against current code and address every validated correctness, safety, and maintainability issue in a focused backend patch. The final PR branch is based on origin/dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Redis token caps reject oversized single reservations and account full reserved units consistently.
- [x] #2 Redis stream/job concurrency reservations are atomic under concurrent reserve calls.
- [x] #3 Daily caps are enforced through atomic/idempotent reservation or consume semantics where Resource Governance makes decisions.
- [x] #4 Policy admin optimistic concurrency cannot lose writes under concurrent expected-version updates.
- [x] #5 Middleware honors configured fail mode instead of silently passing governed traffic through when enforcement is unavailable.
- [x] #6 Coverage audit reports actual configured governance coverage instead of blanket route protection.
- [x] #7 Tenant scoped policies can be enforced from trusted request/auth context.
- [x] #8 Reserve/commit idempotency records do not collide across operation phases.
- [x] #9 Validated cleanup candidates are removed or isolated without weakening tests.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-23-resource-governance-review-fixes.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Moved the reviewed patch onto clean worktree .worktrees/resource-governance-review-pr from origin/dev on branch codex/resource-governance-review-pr. During clean-worktree verification, updated an existing Resource Governance eval cutover test to patch the current _reserve_request_usage path instead of stale _check_cost_limits internals.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Validated and fixed the Resource Governance review findings, moved the work into a clean PR worktree, opened PR #2497 against `dev`, then rebased it cleanly onto latest `origin/dev` (`7ab6ae8c`). Addressed the 7 actionable Qodo comments on the PR and added coverage for the validated fixes. Fresh verification after the final rebase passed: targeted focused tests (`11 passed`), full Resource Governance pytest suite (`201 passed, 2 xfailed`), `py_compile` on touched modules, `git diff --check`, and Bandit on touched Resource Governance/DB scope with 0 findings.
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-06-23/24 rebase follow-up: rebased PR branch `codex/resource-governance-review-pr` cleanly onto `origin/dev` at `ab4ebc51e`. Rechecked PR #2497 issue comments, review comments, and reviews: no actionable code-review comments or review threads were present. Bot comments were Gemini quota warning, CodeRabbit skipped review, and Qodo summary/high-level assessment recommending the existing approach. Fresh verification after rebase: `python -m py_compile` on touched Resource Governance modules passed; Bandit on `tldw_Server_API/app/core/Resource_Governance` reported 0 findings; `LOGURU_LEVEL=ERROR JWT_SECRET_KEY=test-jwt-secret-for-resource-governance-direct-0001 SINGLE_USER_API_KEY=sk-test-single-user python -m pytest -q tldw_Server_API/tests/Resource_Governance` passed with 197 passed, 2 xfailed in 330.39s.
2026-06-24 PR review follow-up: rebased `codex/resource-governance-review-pr` cleanly onto latest `origin/dev` (`18c03de57`). Verified and addressed 7 Qodo review comments: moved SQL-backed policy admin implementation under `DB_Management` with compatibility imports, added missing coverage-audit docstrings, replaced silent tenant claim/config exceptions with debug logging, made daily-cap consume fail-open for AuthNZ/asyncpg DB errors, added database-serialized daily-cap consume semantics, used policy snapshot tenant config for endpoint-level entity derivation, and corrected Redis concurrency retry-after for multi-unit deficit waits. Also fixed the Resource Governance test cleanup helper to walk all Redis SCAN cursors after local verification exposed stale token windows on later scan pages. Verification: targeted focused tests passed (`11 passed`), isolated token refund property test passed, full `tldw_Server_API/tests/Resource_Governance` passed (`201 passed, 2 xfailed`), `py_compile` passed on touched modules, `git diff --check` passed, and Bandit on the touched Resource Governance/DB scope reported 0 findings.
2026-06-24 final PR rebase: `dev` advanced again during the follow-up, so the PR branch was rebased a second time onto `origin/dev` at `7ab6ae8c`. The rebase was clean. Final verification on this base passed: `py_compile` on touched modules, `git diff --check`, Bandit on touched Resource Governance/DB scope with 0 findings, and `LOGURU_LEVEL=ERROR JWT_SECRET_KEY=... SINGLE_USER_API_KEY=... python -m pytest -q tldw_Server_API/tests/Resource_Governance` with `201 passed, 2 xfailed`.
2026-06-24/25 PR comment follow-up: verified the 7 Qodo review threads were addressed and replied to each thread on PR #2497. Fetched latest `dev` again after comments were handled; `dev` had advanced to `e664332b682e9be4e1f89d05a262de155cebfa6e`, so the PR branch was rebased cleanly onto that tip. Re-verified on the rebased commit: `python -m compileall -q` on touched Resource Governance/DB modules passed, `git diff --check FETCH_HEAD..HEAD` passed, Bandit on the touched Resource Governance/DB scope reported 0 results and 0 errors, and `LOGURU_LEVEL=ERROR JWT_SECRET_KEY=... SINGLE_USER_API_KEY=... python -m pytest -q tldw_Server_API/tests/Resource_Governance` passed with `201 passed, 2 xfailed`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
