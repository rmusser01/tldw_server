---
id: TASK-520
title: Expose local calendar API and reminder handoff
status: Done
assignee: []
created_date: ''
updated_date: 2026-06-05 23:03
labels:
- implementation
- calendar
- backend
- api
dependencies: []
documentation:
- Docs/superpowers/specs/2026-06-05-calendar-module-prd-design.md
- Docs/superpowers/plans/2026-06-05-calendar-module-implementation-plan.md
modified_files:
- tldw_Server_API/app/api/v1/schemas/calendar_schemas.py
- tldw_Server_API/app/api/v1/endpoints/calendar.py
- tldw_Server_API/app/api/v1/router_groups/content.py
- tldw_Server_API/app/api/v1/router_groups/minimal.py
- tldw_Server_API/app/core/Calendar/recurrence.py
- tldw_Server_API/tests/Calendar/integration/test_calendar_api.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 4 from the Calendar module implementation plan: Pydantic calendar schemas, thin FastAPI calendar router, router group registration, reminder handoff endpoint, and integration tests for local calendars/items/views/annotations/links/copy/privacy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Task 4 execution notes: implemented Calendar schemas/router, registered content/minimal route specs, hardened raw RRULE parsing, and added API integration coverage. Spec-compliance follow-up expanded membership API coverage to successful owner add/list/remove for viewer, editor, commenter and non-owner denial for add/list/remove. Code-quality follow-up made CalendarService request-tenant aware and mapped invalid agenda date parsing to stable calendar validation responses.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Task 4: added Calendar API schemas and thin FastAPI router for local calendars, memberships, items, agenda views, annotations, links, provider-item copy, reminder handoff, and external account/binding/sync placeholders. Registered the calendar route in content and minimal router groups. Hardened raw RRULE parsing so unsupported keys and invalid numeric/UNTIL values are surfaced as CalendarValidationError/client errors instead of raw ValueError/500. Added permission-aware owner checks for external account/binding/sync placeholder routes. Review fixes: expanded membership API integration coverage for viewer/editor/commenter roles and non-owner management denials; made API CalendarService construction tenant-aware; mapped invalid agenda date params to stable calendar_validation_error client responses. Verification: API integration suite passed with 18 tests; OpenAPI config/jobs regression passed with 4 tests; Calendar unit/property bundle passed with 56 tests; Bandit on touched backend Calendar scope exited 0 with 0 findings. Spec re-review passed. Code-quality re-review found no Critical or Important issues; minor follow-up only notes broad TypeError/ValueError catch around agenda handling.
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
