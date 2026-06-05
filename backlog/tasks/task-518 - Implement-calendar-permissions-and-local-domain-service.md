---
id: TASK-518
title: Implement calendar permissions and local domain service
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-05 21:19'
labels:
  - implementation
  - calendar
  - backend
  - permissions
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-06-05-calendar-module-prd-design.md
  - Docs/superpowers/plans/2026-06-05-calendar-module-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 2 from the Calendar module implementation plan: calendar permission helpers, local CalendarService operations, AuthNZ calendar permission constants, DB support changes as needed, and unit tests for role enforcement, provider-owned read-only behavior, annotations, links, and copy-to-local behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Permission helpers expose CalendarRole and read/write/comment/manage checks.
- [x] #2 CalendarService enforces calendar membership roles for calendars, memberships, local items, annotations, local tag overlays, links, and provider item copy-to-local.
- [x] #3 Org-role membership grants access only through injected resolver approval.
- [x] #4 Provider-owned items are read-only through local service update/delete flows; copied provider items become tldw-owned local rows.
- [x] #5 Focused Calendar permission/service tests and Calendar DB regression tests pass; Bandit reports zero findings.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Follow-up review fix completed. Critical finding addressed: provider-owned personal imports are no longer authorized by generic calendar membership in CalendarService.get_item() or list_items_window(); item visibility now requires the calendar/external-account owner for provider-owned imports. Normal sharing remains available for tldw-owned local items and copied provider items. No Task 3+ recurrence, view, API, frontend, or CalDAV sync behavior was implemented.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Calendar Task 2 local permissions and service layer, then applied follow-up privacy hardening for the spec review critical finding. Added AuthNZ calendar permission constants; added calendar role evaluation helpers with injectable org-role resolution; added CalendarService for calendar CRUD, membership management, local item CRUD, provider-owned read-only guards, annotations, local tag overlays, links, and provider item copy-to-local. Follow-up fix adds item-level provider import visibility so shared viewers can read/list tldw-owned local items and copied provider items, but cannot get/list provider-owned personal imports; calendar/external-account owners can still read/list provider-owned imports. Added regression tests for provider import privacy, owner visibility, copied item sharing, and existing local-item edit denial for viewers. Verification: focused permission/service tests passed (18 passed); Calendar DB regression plus permission/service tests passed (40 passed); Bandit JSON report at /tmp/bandit_calendar_task518.json had 0 findings.
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
