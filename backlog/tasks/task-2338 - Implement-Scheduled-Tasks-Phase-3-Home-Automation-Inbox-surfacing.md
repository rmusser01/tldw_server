---
id: TASK-2338
title: Implement Scheduled Tasks Phase 3 Home Automation Inbox surfacing
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-09 07:32'
labels:
  - scheduled-tasks
  - webui
  - ux
  - implementation
  - companion-home
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 6 of the Scheduled Tasks Phase 3 plan: surface scheduled-task result and attention signals on Companion Home through a dedicated Automation Inbox module that loads independently of Companion personalization and deep-links to Scheduled Tasks results.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Home renders a dedicated Automation Inbox module with scheduled-task result and failure items.
- [x] #2 Automation Inbox shows visible status, owner, timestamp, summary, and exact Scheduled Tasks result/run/task deep links.
- [x] #3 Scheduled-task Home signals load independently of Companion personalization and still render when personalization is unavailable.
- [x] #4 Scheduled-task loading or notification loading failure does not block existing Companion Home cards or items.
- [x] #5 Existing Companion Inbox Preview, Needs Attention, layout customization, and Watchlists ownership behavior remain unchanged.
- [x] #6 Focused CompanionHome and scheduled-task helper tests pass; verification and Bandit/frontend-only rationale are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-09-scheduled-tasks-phase3-results-inbox-home-surfacing-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a dedicated AutomationInboxCard for Home with empty, loading, partial, result, and failure states.
- Added useScheduledTaskHomeSignals so scheduled-task Home signals load independently from Companion personalization using listScheduledTasks and non-blocking listNotifications({ limit: 50 }).
- Added scheduled-task automation Home adapters for projected task results, notification-derived result targets, CompanionHomeItem mapping, and dedupe across identical run/result signals.
- Extended CompanionHomeSource and CompanionHomeEntityType with scheduled_task and scheduled_task_result.
- Rendered Automation Inbox after WhatsNextCard and before the existing Companion Inbox Preview and Needs Attention cards; Customize Home layout data and existing Companion cards are unchanged.
- Verification: ./node_modules/.bin/vitest run src/components/Option/CompanionHome/__tests__/AutomationInboxCard.test.tsx src/components/Option/CompanionHome/__tests__/CompanionHomePage.test.tsx src/components/Option/ScheduledTasks/__tests__/scheduled-task-results.test.ts passed 29 tests.
- Verification: ./node_modules/.bin/vitest run src/components/Option/CompanionHome/__tests__ passed 34 tests.
- Verification: ./node_modules/.bin/vitest run src/components/Option/ScheduledTasks/__tests__ passed 138 tests.
- Bandit: skipped because this slice only changes frontend TypeScript/React and Backlog/plan text, not Python executable code.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Scheduled Tasks Phase 3 Home surfacing. Home now includes a fixed Automation Inbox module that can show scheduled-task result and failure signals without Companion personalization, merge projected task signals with notification-derived signals, handle partial failures without blocking existing Home cards, and deep-link to exact Scheduled Tasks result/run/task targets.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or documented frontend-only skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
