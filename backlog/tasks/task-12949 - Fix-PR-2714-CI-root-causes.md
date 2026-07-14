---
id: TASK-12949
title: Fix PR 2714 CI root causes
status: Done
labels:
- ci
- authnz
- chatbooks
- wizard
priority: high
references:
- https://github.com/rmusser01/tldw_server/pull/2714
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the PR-scoped CI regressions on rmusser01/tldw_server#2714: canonical AuthNZ role membership persistence, stale canonical-role contract assertions, Chatbook full-account fixture state isolation and plural character assertion, wizard dry-run filesystem purity, and OpenAPI fingerprint/frontend type drift.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Users created or role-updated through supported AuthNZ database paths have canonical user_roles membership.
- [x] #2 AuthNZ, MCP catalog, and resource-governance regression tests pass without restoring legacy role fallback.
- [x] #3 Chatbook full-account E2E verification uses the plural characters contract and does not leak auth/database environment or singleton state.
- [x] #4 The in-process single-user E2E lane excludes multi-user-only tests deterministically.
- [x] #5 Wizard --dry-run does not create USER_DB_BASE_DIR or database files.
- [x] #6 OpenAPI fingerprint and generated frontend API types match the current backend contract.
- [x] #7 Targeted tests, Bandit on touched Python scope, and git diff checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Completed from IMPLEMENTATION_PLAN_pr2714_ci_root_causes.md; temporary plan file removed before commit per repository policy.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented canonical role assignment in post-migration AuthNZ/MCP/resource-governance fixtures without restoring users.role authorization fallback. Updated canonical principal expectations, made in-process E2E auth-mode selection honor explicit AUTH_MODE, contained Chatbook fixture environment/AuthNZ singleton state, corrected the plural characters verification contract, moved DatabasePaths imports out of wizard dry-run, and refreshed the OpenAPI fingerprint after generating frontend types.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved the five PR-scoped CI root causes for PR #2714. Verification: 96 passed/1 skipped across all affected files; exact in-process critical E2E lane 15 passed/278 skipped; OpenAPI drift check passed; frontend API type generation completed; Bandit reported 0 findings; production/helper Ruff scope and git diff checks passed. The whole-file Ruff scan still reports pre-existing lint debt in legacy test modules; no new production/helper lint finding was introduced.
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
