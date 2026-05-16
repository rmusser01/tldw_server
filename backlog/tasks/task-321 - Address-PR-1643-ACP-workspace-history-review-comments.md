---
id: TASK-321
title: Address PR 1643 ACP workspace history review comments
status: In Progress
assignee: []
created_date: '2026-05-14 00:29'
updated_date: '2026-05-14 00:58'
labels:
  - review-fix
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1643'
documentation:
  - Docs/Design/ACP_Workspace_Integration_Decision_2026_05.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the live PR #1643 review findings for ACP workspace history: request cancellation, localized unsupported state, recent-task detail selection, scoped 404 handling, and ACP Playground query-param deep links.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 WorkspaceACPHistoryModal aborts in-flight fetches when closed or workspace changes
- [x] #2 Unsupported orchestration 404 handling does not mask missing project/task errors
- [x] #3 Recent run detail fetches prioritize newest tasks before applying the cap
- [x] #4 ACP Playground honors session and view query params from history links
- [x] #5 Focused tests and TypeScript verification cover the review fixes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Live PR sweep found unresolved review threads from Gemini, CodeRabbit, and Qodo. Valid fixes: abortable fetches, localized unsupported error rendering, task recency sort before detail cap, scoped 404 handling, and ACP Playground query-param session/view handling.

Implemented review fixes for PR #1643: abortable WorkspaceACPHistoryModal fetches, scoped unsupported 404 mapping, localized unsupported error rendering, task recency sorting before MAX_TASK_DETAILS, and ACPPlayground session/view query-param handling.

Verification: bunx vitest run src/components/Option/WorkspacePlayground/__tests__/WorkspaceHeader.test.tsx --maxWorkers=1 --no-file-parallelism passed 37 tests; bunx vitest run src/components/Option/ACPPlayground/__tests__/ACPPlayground.connection.test.tsx --maxWorkers=1 --no-file-parallelism passed 2 tests; bunx tsc --noEmit -p /private/tmp/acp-workspace-history-tsconfig.json --pretty false exited 0; git diff --check exited 0.

Bandit: skipped because the review-fix changes touch TypeScript UI/tests and Backlog metadata only; no Python backend files changed.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
