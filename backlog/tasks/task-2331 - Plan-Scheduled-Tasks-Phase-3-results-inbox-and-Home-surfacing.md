---
id: TASK-2331
title: Plan Scheduled Tasks Phase 3 results inbox and Home surfacing
status: Done
labels:
- scheduled-tasks
- webui
- home
- ux
- implementation-plan
priority: high
references:
- TASK-494
- TASK-496
- TASK-498
- Docs/superpowers/specs/2026-06-01-scheduled-tasks-automation-workbench-prd-design.md
- Docs/superpowers/plans/2026-06-09-scheduled-tasks-phase3-results-inbox-home-surfacing-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create an implementation-ready plan for Scheduled Tasks Automation Workbench Phase 3: results inbox, Home automation surfacing, exact task/run/result deep links, failure triage, and notification/Home dedupe.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Plan is saved under Docs/superpowers/plans with exact files, tests, commands, and staged implementation tasks.
- [x] #2 Plan scopes Phase 3 to results discovery and Home surfacing, leaving recurring RAG and ACP/agent schedule creation to later Phase 4 work.
- [x] #3 Plan preserves Watchlists, Home, Notifications, and Scheduled Tasks ownership boundaries with deep links instead of moving domain configuration.
- [x] #4 Plan covers first-time user and power-user workflows, empty/loading/error/success states, accessibility, responsive extension-sized layouts, and copy recommendations.
- [x] #5 Plan review is completed or findings are addressed/documented, and Backlog records verification and known skips.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Cleanup: verified the stale Scheduled Tasks Phase 2B planning item is already Done on latest `origin/dev`; no cleanup edit was required for that task.
- Added implementation plan: `Docs/superpowers/plans/2026-06-09-scheduled-tasks-phase3-results-inbox-home-surfacing-implementation-plan.md`.
- Plan review findings addressed:
  - Clarified that Phase 3 ships as `/scheduled-tasks?tab=results` rather than assuming a new nested route.
  - Made Home integration concrete by extending the Companion Home item source/entity types for scheduled-task result items instead of leaving the type strategy open.
  - Added notification target normalization and dedupe requirements so Home and notification-derived result items resolve to the same exact task/run/result targets.
  - Kept Watchlists as the owner of monitor setup, source tuning, activity, and reports; Scheduled Tasks only links into those surfaces.
- Verification:
  - Unresolved-marker scan passed for the plan and Backlog task.
  - `git diff --cached --check` passed.
  - Bandit skipped because this is docs/Backlog-only planning work with no Python/backend file changes.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and reviewed the Scheduled Tasks Phase 3 implementation plan for results inbox and Home surfacing. The plan is scoped to product/UX and frontend implementation work, with backend APIs listed as dependencies. It covers `/scheduled-tasks` Results IA, result projection, exact deep links, Home inbox/attention surfacing, notification dedupe, empty/error/loading/success states, accessibility, responsive checks, tests, and explicit Watchlists preservation boundaries.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed.
- [x] #2 Plan document added and referenced from this task.
- [x] #3 Verification recorded, including docs scans and Bandit skip rationale for docs-only work.
- [x] #4 Final summary added.
- [x] #5 Changes committed on the planning branch.
<!-- DOD:END -->
