---
id: TASK-2333
title: Implement Scheduled Tasks Phase 3 result projection foundation
status: Done
labels:
- scheduled-tasks
- webui
- home
- implementation
priority: high
modified_files:
- apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-result-links.ts
- apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-results.ts
- apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-results.test.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 1 of the amended Scheduled Tasks Phase 3 plan: capability modes, result/signal projection from scheduled task list data, result deep links, notification/home dedupe helpers, mixed failure-plus-output rules, and pure unit tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pure result projection model and helpers are implemented for scheduled task list data.
- [x] #2 Capability modes distinguish projected signals from normalized results and hide durable review semantics in projected mode.
- [x] #3 Mixed failure-plus-output task states produce separate failure and result signals with separate dedupe keys.
- [x] #4 Result URL helpers build task, run, result, Home, and notification targets safely.
- [x] #5 Unit tests cover projection, capability modes, mixed states, dedupe keys, and unsafe source reference sanitization.
- [x] #6 Verification and Bandit/frontend-only rationale are recorded in Backlog.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the Stage 1 result projection foundation as pure UI helpers. Added route/dedupe helpers for scheduled task result, run, and task targets; notification target normalization that preserves notification behavior while creating scheduled-task result deep links; capability mode detection for projected/read/mutation result support; and a projection model that turns scheduled task list data into result, failure, running, and completed-without-results signals.

The projected mode intentionally keeps reviewAvailable and retryAvailable false so the UI does not imply durable result-review or retry semantics until normalized backend result mutation APIs exist. Mixed failed-with-output states now create separate failure and result signals with separate dedupe keys. Provenance labels are sanitized and private-looking source references such as token, api_key, Authorization/Bearer, secret, and password values are not surfaced.

Verification: `./node_modules/.bin/vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-results.test.ts` passed (8 tests). `./node_modules/.bin/vitest run src/components/Option/ScheduledTasks/__tests__` passed (9 files, 119 tests). Bandit skipped because this task touched only frontend TypeScript test/helper files and Backlog task metadata; no Python code was changed.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Scheduled Tasks Phase 3 Stage 1 result projection helpers and tests. The UI can now derive safe result/failure/running/completed-no-result signals from existing scheduled-task list data, route Home/notification clicks into `/scheduled-tasks?tab=results`, hide review/retry affordances in projected mode, separate mixed failure/output states, and scrub private-looking provenance values. Focused and full Scheduled Tasks component Vitest suites passed; Bandit was not applicable because no Python files were touched.
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
