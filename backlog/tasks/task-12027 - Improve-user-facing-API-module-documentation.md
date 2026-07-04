---
id: TASK-12027
title: Improve user-facing API module documentation
status: Done
assignee: []
created_date: '2026-07-04 22:25'
updated_date: '2026-07-04 22:37'
labels:
  - docs
  - api
  - openapi
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Improve the user-facing documentation that explains what each API module can do. Scope covers a practical OpenAPI tag/module capability guide and aligned OpenAPI tag metadata so /docs and /redoc are easier to browse. Keep endpoint behavior unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 API tag/module guide explains major API capabilities in user-facing language and links to existing detailed docs where available.
- [x] #2 OpenAPI tag descriptions and ReDoc tag groups align with the module guide for stable/common API surfaces.
- [x] #3 Experimental, admin-only, or low-level/internal surfaces are clearly labeled instead of presented as primary user workflows.
- [x] #4 Published docs mirror is updated when the source API tag index changes.
- [x] #5 Verification records OpenAPI/schema build checks, markdown/link sanity checks, and Bandit for touched Python files if Python is changed.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write a concise design spec capturing the reviewed scope and risk controls. 2. Inventory router tags against existing OpenAPI metadata and API docs. 3. Replace the short API_Tags_Index with a grouped capability guide and mirror it to Published. 4. Update OpenAPI tag metadata/grouping in main.py without changing routes or endpoint behavior. 5. Verify OpenAPI generation, markdown/link sanity, and security scan for touched Python.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Design review before implementation identified two scope controls: keep the guide comprehensive at the module/tag level rather than documenting every endpoint, and avoid letting main.py become a full API manual. OpenAPI metadata should summarize discoverability, while detailed usage stays in markdown docs.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Verification: cmp mirror passed; local markdown links resolve; py_compile for tldw_Server_API/app/main.py passed; OpenAPI schema smoke passed with dummy SINGLE_USER_API_KEY and reported openapi tags=170 groups=10; Bandit reported 0 findings in /tmp/bandit_api_module_docs.json.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Expanded the API tag index into a grouped module capability guide and mirrored it to Published. Aligned OpenAPI tag descriptions and ReDoc tag groups with the same user-facing capability categories. Endpoint behavior, routes, schemas, and security settings were unchanged.
<!-- SECTION:FINAL_SUMMARY:END -->

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
