---
id: TASK-516
title: Plan first-class calendar module implementation
status: Done
labels:
- planning
- calendar
- implementation-plan
documentation:
- Docs/superpowers/specs/2026-06-05-calendar-module-prd-design.md
- Docs/superpowers/plans/2026-06-05-calendar-module-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-06-05-calendar-module-implementation-plan.md
- backlog/tasks/task-516 - Plan-first-class-calendar-module-implementation.md
- Docs/Design/Calendar_Module.md
- Docs/Development/Calendar_CalDAV_Smoke_Test.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a task-by-task implementation plan for the approved first-class Calendar module PRD, covering MVP local calendar foundation, practical frontend UI, read-only CalDAV import, provider-owned context, hardening, tests, and verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-05-calendar-module-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implementation plan written at Docs/superpowers/plans/2026-06-05-calendar-module-implementation-plan.md for the approved Calendar module PRD. The plan covers staged backend Calendar DB/service/API work, recurrence and scheduled-task projections, frontend /calendar route and practical UI, read-only CalDAV account/discovery/sync, provider-owned item UX, documentation, Bandit, and focused backend/frontend verification. Plan review loop found two blockers around missing persistence methods and under-specified external account/binding lifecycle; both were fixed, and the reviewer approved the revised plan. git diff --check passed. No code tests or Bandit were run for the original planning-only task.

Implementation completion update, 2026-06-05: the Calendar module implementation plan has been executed through Task 10. Completed scope includes local calendars/items/todos, recurrence-backed agenda/week views, scheduled-task projections, annotations/links/local tags, reminder handoff, /calendar frontend route and practical UI, provider-owned read-only treatment, generic read-only CalDAV VEVENT import, Jobs-backed sync queue/worker/scheduler hooks, CalDAV sync settings UI, Calendar module docs, and Fastmail smoke-test documentation. Final verification recorded in TASK-526: Calendar backend tests passed 99 tests; focused frontend Calendar tests passed 4 files / 25 tests; OpenAPI config jobs tests passed 4 tests; Bandit report /tmp/bandit_calendar_module.json had 0 high, 0 medium, and 0 low findings. Known remaining verification gaps: the route parity guard was blocked by Vitest config/dependency state, and the optional browser route check was blocked by a local Next/Turbopack /calendar compile hang. Full Fastmail provider smoke remains manual because it requires real provider credentials.
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
