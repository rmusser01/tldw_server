---
id: TASK-2332
title: Address Scheduled Tasks Phase 3 plan review findings
status: Done
labels:
- scheduled-tasks
- webui
- home
- ux
- implementation-plan
priority: high
references:
- TASK-2331
- Docs/superpowers/plans/2026-06-09-scheduled-tasks-phase3-results-inbox-home-surfacing-implementation-plan.md
- Docs/superpowers/specs/2026-06-01-scheduled-tasks-automation-workbench-prd-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Patch the Phase 3 results inbox and Home surfacing implementation plan to resolve identified UX/product risks before implementation: review-state capability modes, mixed task/result states, route contract, Home rendering strategy, notification dedupe source, structured provenance, and unsupported action visibility.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Plan defines capability modes so projected task-list signals do not pretend to support durable review state or mutation actions.
- [x] #2 Plan defines mixed-state rules for failed runs with outputs and other task/result status conflicts.
- [x] #3 Plan reconciles the PRD /scheduled-tasks/results route with the current query-tab implementation strategy.
- [x] #4 Plan clarifies Home rendering strategy, notification dedupe source, and structured provenance fields.
- [x] #5 Plan updates tests/tasks/copy guidance for unsupported retry/review actions and avoids action graveyards.
- [x] #6 Backlog records verification and final summary.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Updated the Phase 3 plan to add explicit capability modes:
  - `projected_signals`
  - `normalized_results_read`
  - `normalized_results_mutation`
- Added rules that hide durable review filters/actions and retry/review mutation buttons in projected mode.
- Added mixed-state projection rules so failed tasks with produced output keep both the failure signal and the result signal.
- Reconciled routing by keeping query-tab state internally while adding `/scheduled-tasks/results` as a route alias for PRD/Home/notification stability.
- Changed Home surfacing from generic Companion card merging to a dedicated `Automation Inbox` module with status, owner, and exact deep links.
- Clarified notification dedupe source as a non-blocking `listNotifications({ limit: 50 })` load.
- Added structured provenance fields: `resultKind`, `matchReason`, `matchedRuleLabel`, `outputLabel`, and `domainHref`.
- Updated stages, file structure, tests, copy guidance, and PR checklist to reflect the above.
- Verification:
  - Consistency scan passed for unresolved markers and stale old-plan phrases.
  - `git diff --check` passed.
  - Bandit skipped because this amendment touched only Markdown planning/Backlog files.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed all identified Phase 3 plan review issues before implementation. The plan now prevents fake review-state UX in projected mode, preserves mixed result/failure states, ships a stable `/scheduled-tasks/results` alias, uses a dedicated Home Automation Inbox, defines the notification dedupe source, structures provenance fields, and hides unsupported mutation actions instead of presenting disabled action clutter.
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
