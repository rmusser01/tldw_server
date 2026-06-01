---
id: TASK-498
title: Implement Scheduled Tasks Automation Workbench Phase 1
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-01 22:05'
labels:
  - scheduled-tasks
  - webui
  - ux
  - phase-1
dependencies: []
references:
  - TASK-496
  - TASK-494
  - >-
    Docs/superpowers/specs/2026-06-01-scheduled-tasks-automation-workbench-prd-design.md
  - >-
    Docs/superpowers/plans/2026-06-01-scheduled-tasks-automation-workbench-phase1-implementation-plan.md
documentation:
  - >-
    Docs/superpowers/specs/2026-06-01-scheduled-tasks-automation-workbench-prd-design.md
  - >-
    Docs/superpowers/plans/2026-06-01-scheduled-tasks-automation-workbench-phase1-implementation-plan.md
  - >-
    Task 2 follow-up: stable status keys; disabled next-run filtering; empty
    state table suppression.
  - 'Task 5: safer reminder scheduling controls implemented in WebUI.'
  - >-
    Task 5 spec-review follow-up: recurring preview copy is next-run oriented
    for presets.
  - >-
    Task 5 code-quality follow-up: fixed recurring edit hydration cron
    preservation and APScheduler-aligned cron token validation.
  - >-
    Task 6: extension scheduled-tasks E2E copy assertion updated to the new
    workbench copy; route/component Vitest passed; extension E2E not run locally
    because prerequisites are optional for this slice.
  - >-
    Final review follow-up: strengthened recurring reminder validation for
    invalid cron ranges/words and invalid IANA timezones; focused utility/page
    Vitest passed 33 tests.
  - >-
    Final review follow-up: aligned numeric weekday validation with APScheduler
    where 0 is Monday and 6 is Sunday and 7 is invalid; focused utility/page
    Vitest passed 35 tests.
  - >-
    Final review follow-up: rejected APScheduler-invalid reversed named weekday
    and month ranges; focused utility/page Vitest passed 37 tests.
  - >-
    Final review follow-up: rejected APScheduler-invalid numeric-to-name month
    and weekday ranges; focused utility/page Vitest passed 38 tests.
  - >-
    Final review follow-up: allowed APScheduler-valid name-to-number and open
    named ranges plus nth weekday bounds; focused utility/page Vitest passed 40
    tests.
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Phase 1 of the Scheduled Tasks Automation Workbench UX from the approved PRD and implementation plan. Scope stays at the product/UI layer: unified /scheduled-tasks visibility, IA shell, status model, table/detail inspection, existing reminder plus Watchlists job run/result links, and safer reminder scheduling controls. Watchlists remains a separate full workspace and is not replaced or limited.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Scheduled task rows use user-facing statuses instead of raw API status strings.
- [x] #2 /scheduled-tasks shows overview metrics, loading, empty, partial, unsupported, and error states with recovery actions.
- [x] #3 Reminder tasks can be created and edited with safe one-time and recurring schedule controls instead of raw-first run_at/cron/timezone fields.
- [x] #4 Watchlist jobs are visible as externally managed monitors with deep links to Watchlists settings, activity, latest run, reports, and latest output when identifiers exist.
- [x] #5 Task detail inspection shows current state, schedule, last/next run, source metadata, and available actions without editing Watchlist jobs.
- [x] #6 Existing Watchlists functionality and UX remain intact and separate.
- [x] #7 Focused WebUI tests and relevant backend contract tests pass, or any skipped checks are documented with cause.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Task 5 code-quality follow-up strict TDD: added failing utility tests for APScheduler cron token alignment (# valid, ? invalid) and a page-level edit regression proving an existing recurring custom cron must be preserved when editing unrelated fields. Then removed the broad generated-cron effect from ReminderScheduleControls and moved cron writes to explicit user-change/defaulting paths only.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

Task 1 review clarification: failure-like status tokens (fail, error, missed) take precedence over result-like tokens when both appear in a single backend status string. This keeps states such as output_error in Needs attention rather than Found results. Helper tests now cover output ID aliases, token-boundary matching, and unsafe ID rejection.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Scheduled Tasks Automation Workbench Phase 1 implemented. User-facing statuses, overview metrics and states, searchable/filterable table, detail drawer, Watchlists deep links without Watchlists edit ownership, safer reminder schedule controls, and extension route copy parity. Verification: frontend focused Vitest passed 6 files / 61 tests; backend scheduled-tasks control-plane pytest passed 4 tests; git diff --check passed; final code review approved. Bandit skipped because no backend Python changed. Manual live-data browser verification skipped because no seeded backend/dev data environment is available; component tests cover empty, partial, loaded reminder, loaded Watchlists, filters/search, detail drawer, reminder scheduling, and Watchlists links. Follow-up phases remain templates for GitHub/YouTube/RAG/agents plus Home/results surfacing.
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
