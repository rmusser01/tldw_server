---
id: TASK-2352
title: Address PR 2343 review feedback and rebase Scheduled Tasks Phase 4B
status: Done
labels:
- scheduled-tasks
- api
- frontend
- review
priority: high
references:
- TASK-2351
- https://github.com/rmusser01/tldw_server/pull/2343
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase PR #2343 onto latest dev and address all actionable PR review comments/check failures for Scheduled Tasks Phase 4B API foundation. Track verification and final PR update evidence here.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PR #2343 rebased onto latest origin/dev without dropping intended work.
- [x] #2 All actionable PR review comments are evaluated and either fixed or documented with technical rationale.
- [x] #3 Focused backend/frontend tests and Bandit are rerun for touched scope.
- [x] #4 PR branch is pushed after fixes.
- [x] #5 Known unrelated local files remain untouched.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Rebased branch `codex/scheduled-tasks-phase4b-api-foundation-spec` onto `origin/dev`; final fetch showed `HEAD...origin/dev` as `28 0`, so `origin/dev` was not ahead.
- Addressed Gemini comments by closing per-call SQLite connections, replacing API-boundary catches of generic automation `KeyError`/`ValueError` with `ScheduledTaskAutomationError`, defensively validating non-object schedules, and converting strict TypeScript record reads to index access.
- Addressed Qodo comments by splitting preview read-path not-found errors to 404 while keeping missing preview preconditions as 400, converting synchronous automation handlers from `async def` to `def`, bulk-loading previews for definition list responses, and caching schema setup per service/repository key.
- Addressed CodeRabbit comments by guarding missing editor mutation handlers, closing the task detail drawer before opening the automation editor, using canonical definition IDs for lifecycle actions, disabling optional automation action buttons, rejecting empty projected automation IDs, constraining automation filter query params, deriving effective expired preview status from `expires_at`, enforcing preview create/update linkage invariants, using runtime-future test expirations, and exposing the Archived task filter.
- Left unrelated untracked watchlist template files unstaged and untouched:
  - `tldw_Server_API/Config_Files/templates/watchlists/cti_osint_report_markdown.md`
  - `tldw_Server_API/Config_Files/templates/watchlists/news_briefing_markdown.md`
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Rebased PR #2343 onto latest `origin/dev`, resolved actionable review feedback across backend API/service/DB code and WebUI/client scheduled task flows, reran focused verification, and prepared the branch for push.
- Verification:
  - `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Notifications/test_scheduled_task_automation_db.py tldw_Server_API/tests/Notifications/test_scheduled_task_automation_service.py tldw_Server_API/tests/Notifications/test_scheduled_task_automation_api.py -q` -> `90 passed, 14 warnings`
  - `bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-automation-status.test.ts src/components/Option/ScheduledTasks/__tests__/ScheduledTaskAutomationDefinitionEditor.test.tsx src/components/Option/ScheduledTasks/__tests__/ScheduledTaskDetailDrawer.test.tsx src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx src/services/__tests__/scheduled-tasks-control-plane.test.ts --maxWorkers=1 --no-file-parallelism` -> `5 passed`, `82 passed`
  - `bun run compile` in `apps/extension` -> `tsc --noEmit -p tsconfig.compile.json`
  - `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit tldw_Server_API/app/core/DB_Management/Scheduled_Tasks_DB.py tldw_Server_API/app/services/scheduled_task_automation_service.py tldw_Server_API/app/api/v1/endpoints/scheduled_tasks_control_plane.py -f json -o /tmp/bandit_scheduled_tasks_phase4b_review.json` -> zero findings
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
