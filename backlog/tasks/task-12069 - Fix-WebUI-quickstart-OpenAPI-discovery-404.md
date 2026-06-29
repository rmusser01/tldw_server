---
id: TASK-12069
title: Fix WebUI quickstart OpenAPI discovery 404
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-29 18:05'
labels:
  - uat
  - release
  - pr-1982
  - frontend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1982'
  - >-
    Docs/Product/WebUI/evidence/pre_main_uat/pre-main-uat-20260629054510/findings.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
CDP UAT verified that quickstart/same-origin WebUI pages render but repeatedly request `http://127.0.0.1:8080/openapi.json`, which returns 404 because the backend OpenAPI route is not proxied at the WebUI origin. Fix the current-code issue with a minimal same-origin proxy route and regression coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed WebUI same-origin OpenAPI discovery by adding apps/tldw-frontend/pages/openapi.json.ts as a GET-only backend proxy with no-store caching, plus apps/tldw-frontend/__tests__/pages/openapi-json-proxy.test.ts regression coverage. Verification passed: bunx eslint pages/openapi.json.ts __tests__/pages/openapi-json-proxy.test.ts; bunx vitest run __tests__/pages/api/runtime-config.test.ts __tests__/pages/openapi-json-proxy.test.ts __tests__/extension/runtime-bootstrap.test.ts (68 tests); git diff --check; live GET http://127.0.0.1:8080/openapi.json returned 200; CDP UAT returned pageIdentityOk=true, notBlank=true, noFrameworkOverlay=true, interactionCount=3, relevantConsoleEvents=[]. Bandit skipped because this task touched TypeScript/docs only.
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
