---
id: TASK-5
title: Address PR 1237 OpenAPI tag declaration review comment
status: Done
assignee:
  - Codex
created_date: '2026-05-03 18:32'
updated_date: '2026-05-03 18:32'
labels:
  - pr-review
  - openapi
  - phase4
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1237'
  - 'https://github.com/rmusser01/tldw_server/pull/1237#discussion_r3178558700'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve the live review thread on PR #1237 by verifying the OpenAPI tag declaration helper is not repeated after schema caching and making that behavior explicit without broadening the Phase 4 OpenAPI contract scope.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The unresolved review comment on tldw_Server_API/app/main.py is addressed with a narrowly scoped change or documented technical response.
- [x] #2 Relevant OpenAPI contract tests verify that schema generation caching prevents repeated tag declaration work.
- [x] #3 Focused backend tests and security checks for the touched scope are run and recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused OpenAPI contract test that resets the app OpenAPI cache, wraps _ensure_openapi_operation_tags_declared, calls app.openapi() twice, and verifies the helper only runs during the first schema build.
2. Add a concise comment in custom_openapi() documenting that tag declaration work is covered by app.openapi_schema caching.
3. Run the focused OpenAPI contract test, git diff --check, and Bandit on the touched backend files. Record verification in the task.
4. If verification passes, commit the narrow PR follow-up and push to codex/phase4-openapi-contract-testing; then re-check the review thread/status.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a focused OpenAPI schema-cache regression test and added a cache-boundary comment in custom_openapi(). Verification: targeted pytest for the new regression plus existing tag-declaration contract passed (2 passed, 5 warnings); git diff --check passed; Bandit on main.py passed with zero findings; Bandit on touched app/test files passed with B101 skipped because pytest asserts are expected in tests. A full test_openapi_contracts.py run timed out in existing TestClient startup/teardown after one test, so the focused contract verification is the reliable signal for this review comment.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #1237 review feedback by documenting that OpenAPI schema normalization is covered by app.openapi_schema caching and adding a regression test proving tag declaration normalization runs only during the first schema build. No production behavior change beyond the clarifying comment.
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
