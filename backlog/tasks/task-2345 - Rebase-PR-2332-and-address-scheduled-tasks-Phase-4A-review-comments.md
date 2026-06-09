---
id: TASK-2345
title: Rebase PR 2332 and address scheduled tasks Phase 4A review comments
status: Done
assignee: []
created_date: ''
updated_date: 2026-06-09 18:37
labels:
- scheduled-tasks
- pr-review
- frontend
dependencies: []
references:
- https://github.com/rmusser01/tldw_server/pull/2332
priority: high
modified_files:
- apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskCreatePanel.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx
- backlog/tasks/task-2342 - Design-Scheduled-Tasks-Phase-4-API-first-recurring-question-and-agent-task-contract.md
- backlog/tasks/task-2343 - Plan-Scheduled-Tasks-Phase-4A-API-first-planned-shell-implementation.md
- backlog/tasks/task-2345 - Rebase-PR-2332-and-address-scheduled-tasks-Phase-4A-review-comments.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase the Scheduled Tasks Phase 4A planned-shell PR onto latest dev, verify all current PR review comments, address still-valid findings, run focused verification, and push the updated branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Rebases branch `codex/scheduled-tasks-phase4-api-contract` onto `origin/dev`.
- Verified current PR review threads and addressed the still-valid findings: conditionally hide the empty Safety section for planned create-panel templates; mark requested DoD items in TASK-2342 and TASK-2343.
- Added a create-panel regression assertion that Recurring Question planned templates do not render the Safety heading when no safety lines exist.
- Stabilized two long scheduled-task page drawer/form tests by applying the file's existing slow scheduled-form timeout budget after the rebased focused bundle exposed timeout-only flakiness.
- Verification: `bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx`; `bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-planned-template-copy.test.ts src/components/Option/ScheduledTasks/__tests__/scheduled-task-template-capabilities.test.ts src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx src/components/Option/ScheduledTasks/__tests__/ScheduledTaskResultsPanel.test.tsx src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx src/components/Option/CompanionHome/__tests__/AutomationInboxCard.test.tsx`; `git diff --check origin/dev..HEAD`.
- Bandit: not run because this review follow-up touched frontend tests/components and Backlog task docs only; no Python code was changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR 2332 onto latest `origin/dev`, addressed all current review comments, added/updated regression coverage, stabilized the focused scheduled-tasks Vitest bundle, pushed the rebased branch, and resolved the three PR review threads.
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
