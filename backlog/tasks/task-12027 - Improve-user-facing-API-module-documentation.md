---
id: TASK-12027
title: Improve user-facing API module documentation
status: In Progress
created_date: 2026-07-04 22:25
labels:
- docs
- api
- openapi
priority: Medium
modified_files:
- Docs/API-related/API_Tags_Index.md
- Docs/Published/API-related/API_Tags_Index.md
- tldw_Server_API/app/main.py
- Docs/superpowers/specs/2026-07-04-api-module-documentation-design.md
- Docs/superpowers/plans/2026-07-04-api-module-documentation.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Improve the user-facing documentation that explains what each API module can do. Scope covers a practical OpenAPI tag/module capability guide and aligned OpenAPI tag metadata so /docs and /redoc are easier to browse. Keep endpoint behavior unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 API tag/module guide explains major API capabilities in user-facing language and links to existing detailed docs where available.
- [ ] #2 OpenAPI tag descriptions and ReDoc tag groups align with the module guide for stable/common API surfaces.
- [ ] #3 Experimental, admin-only, or low-level/internal surfaces are clearly labeled instead of presented as primary user workflows.
- [ ] #4 Published docs mirror is updated when the source API tag index changes.
- [ ] #5 Verification records OpenAPI/schema build checks, markdown/link sanity checks, and Bandit for touched Python files if Python is changed.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write a concise design spec capturing the reviewed scope and risk controls. 2. Inventory router tags against existing OpenAPI metadata and API docs. 3. Replace the short API_Tags_Index with a grouped capability guide and mirror it to Published. 4. Update OpenAPI tag metadata/grouping in main.py without changing routes or endpoint behavior. 5. Verify OpenAPI generation, markdown/link sanity, and security scan for touched Python.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Design review before implementation identified two scope controls: keep the guide comprehensive at the module/tag level rather than documenting every endpoint, and avoid letting main.py become a full API manual. OpenAPI metadata should summarize discoverability, while detailed usage stays in markdown docs.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
