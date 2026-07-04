---
id: TASK-12027
title: Improve user-facing API module documentation
status: Done
assignee: []
created_date: '2026-07-04 22:25'
updated_date: '2026-07-04 23:25'
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
- [x] #4 Verification records OpenAPI/schema build checks, markdown/link sanity checks, and Bandit for touched Python files if Python is changed.
- [x] #5 Generated Published docs are left unchanged; source docs are ready for the docs publishing process to regenerate Published.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write a concise design spec capturing the reviewed scope and risk controls. 2. Inventory router tags against existing OpenAPI metadata and API docs. 3. Replace the short source API_Tags_Index with a grouped capability guide while leaving generated Published docs unchanged. 4. Update OpenAPI tag metadata/grouping in main.py without changing routes or endpoint behavior. 5. Verify OpenAPI generation, markdown/link sanity, generated Published no-diff status, and security scan for touched Python.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Design review before implementation identified two scope controls: keep the guide comprehensive at the module/tag level rather than documenting every endpoint, and avoid letting main.py become a full API manual. OpenAPI metadata should summarize discoverability, while detailed usage stays in markdown docs. Verification update: Docs/Published/API-related/API_Tags_Index.md is generated output and is intentionally left unchanged; fresh verification confirmed no Published branch diff, source markdown links resolve, main.py py_compile passes, OpenAPI schema smoke reports openapi tags=170 groups=10, and Bandit reports errors=0 results=0 in /tmp/bandit_api_module_docs.json.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Expanded the source API tag index into a grouped module capability guide and aligned OpenAPI tag descriptions/ReDoc groups so users can browse by goal. Generated Published docs were left unchanged for the publishing process. Endpoint behavior, routes, schemas, and security settings were unchanged.
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
