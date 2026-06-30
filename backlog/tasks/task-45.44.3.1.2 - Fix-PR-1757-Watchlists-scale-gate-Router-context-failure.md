---
id: TASK-45.44.3.1.2
title: Fix PR 1757 Watchlists scale gate Router context failure
status: Done
labels:
- design-system
- webui
- watchlists
- ci
- review
priority: medium
parent_task_id: TASK-45.44.3.1
references:
- https://github.com/rmusser01/tldw_server/pull/1757
- https://github.com/rmusser01/tldw_server/actions/runs/25955005494/job/76299980088
- apps/packages/ui/src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.run-notifications.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address the actionable PR #1757 CI failure where Watchlists scale gate tests render WorkspaceConnectionGate without Router context after useLocation was introduced.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The WatchlistsPlaygroundPage run-notifications test harness provides Router context for WorkspaceConnectionGate.
- [x] #2 The focused failing run-notifications test file passes locally.
- [x] #3 The CI-equivalent Watchlists static typecheck and scale gate scripts pass locally.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the WatchlistsPlaygroundPage run-notifications failure locally with the focused failing test.
2. Update the test harness to render WatchlistsPlaygroundPage under a Router-compatible provider while preserving the existing mocked `useNavigate`.
3. Rerun the focused failing test, the Watchlists gate scripts, and relevant design-system checks before committing and pushing.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
CI failure evidence:
- Watchlists Scale Gate failed in `WatchlistsPlaygroundPage.run-notifications.test.tsx` with `useLocation() may be used only in the context of a <Router> component.`
- The failure came from `WorkspaceConnectionGate`, which uses both `useNavigate()` and `useLocation()`.
- The focused local red run reproduced the same 4 failures before the fix.

Implementation:
- Added `MemoryRouter` to the run-notifications test harness.
- Introduced `renderWatchlistsPlaygroundPage()` so every test renders `WatchlistsPlaygroundPage` with Router context.
- Kept the existing mocked `useNavigate` behavior intact for notification deep-link assertions.

Verification:
- `bunx vitest run src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.run-notifications.test.tsx --reporter=dot --testTimeout=20000` passed 1 file / 4 tests.
- `bun run test:watchlists:typecheck` passed 1 file / 3 tests.
- `bun run test:watchlists:scale` passed 7 files / 49 tests.
- `bunx vitest run src/components/Option/Admin/__tests__/LlamacppRuntimePanel.test.tsx src/components/Option/Watchlists/shared/__tests__/WatchlistsHealthBar.test.tsx src/design-system/__tests__/product-state-guard.test.ts --reporter=dot --testTimeout=20000` passed 3 files / 56 tests.
- `bun run verify:design-system-state` passed with the expected existing baseline summary and stale-baseline reporting.
- `git diff --check` passed.
- Bandit is not applicable because the touched implementation is frontend test code plus this Backlog task record.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the PR #1757 Watchlists Scale Gate failure by wrapping the WatchlistsPlaygroundPage run-notifications test harness in `MemoryRouter`, giving WorkspaceConnectionGate the Router context required for `useLocation()` while preserving existing notification navigation assertions.
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
