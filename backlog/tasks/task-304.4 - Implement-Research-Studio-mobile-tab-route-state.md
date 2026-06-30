---
id: TASK-304.4
title: Implement Research Studio mobile tab route state
status: Done
assignee:
  - Codex
created_date: '2026-05-12 17:55'
updated_date: '2026-05-12 18:21'
labels:
  - implementation
  - research-studio
  - webui
  - mobile
  - routing
dependencies:
  - TASK-304.3
documentation:
  - >-
    Docs/superpowers/plans/2026-05-12-research-studio-ux-remediation-implementation-plan.md
parent_task_id: TASK-304
priority: medium
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Allowed tab values are sources, chat, and studio
- [x] #2 Missing or invalid tab params fall back to Chat
- [x] #3 URL ?tab state wins over component defaults or future persisted tab state
- [x] #4 Mobile initial active tab comes from ?tab without requiring internal tab activation
- [x] #5 Desktop route state can focus the requested pane without hiding the normal multi-pane layout
- [x] #6 Focused helper and responsive tests cover valid, invalid, and shared-param combinations
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect current WorkspacePlayground tab initialization, pane focus helpers, and responsive tests.
2. Add failing pure helper tests for tab parsing and search param handling.
3. Add failing responsive tests for mobile /research-studio?tab=studio and invalid tab fallback.
4. Implement a focused route-state helper and wire WorkspacePlayground initialization to it.
5. Run focused UI tests, CDP smoke where practical, diff hygiene, and update this task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a focused Research Studio route-state helper for tab parsing and fallback behavior, then wired WorkspacePlayground initial activeTab to ?tab route state. The component now initializes mobile tabs from ?tab=sources|chat|studio and applies first-load pane focus on desktop for sources or studio without changing the desktop multi-pane layout.

TDD notes: helper tests first caught accepted and invalid tab parsing. Responsive tests failed before implementation because ?tab=studio still opened Chat and desktop ?tab=studio did not uncollapse/focus Studio. After wiring the helper into WorkspacePlayground, the same tests passed.

Browser/CDP verification: started Next dev server on 127.0.0.1:3002 and ran a temporary mobile Playwright smoke against /research-studio?tab=studio. The first smoke used the project default http://localhost:8080 and failed with connection refused; rerunning with TLDW_WEB_URL=http://localhost:3002 passed. The smoke verified the mobile route compiles without route/module errors. It did not assert authenticated panel content beyond compile/query behavior.

Verification run:
- bunx vitest run src/components/Option/WorkspacePlayground/__tests__/research-studio-route-state.test.ts src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.stage2.responsive.test.tsx src/components/Option/WorkspacePlayground/__tests__/WorkspacePlayground.stage3.test.tsx -> 3 files passed, 26 tests passed.
- TLDW_WEB_AUTOSTART=false TLDW_WEB_URL=http://localhost:3002 bunx playwright test e2e/research-studio-mobile-tab.codex-temp.spec.ts --reporter=line --workers=1 -> 1 passed.
- git diff --check -> clean.

Bandit was not run because this slice touched only frontend TypeScript, frontend tests, and Backlog metadata; no Python/backend code changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added a Research Studio route-state helper and wired WorkspacePlayground so /research-studio?tab=sources|chat|studio controls the initial mobile tab and first-load desktop pane focus. Invalid or missing tab state falls back to Chat, and existing shared query params are preserved.
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
