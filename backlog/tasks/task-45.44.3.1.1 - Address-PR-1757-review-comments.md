---
id: TASK-45.44.3.1.1
title: Address PR 1757 review comments
status: Done
assignee:
- Codex
labels:
- design-system
- webui
- product-state
- review
priority: medium
parent_task_id: TASK-45.44.3.1
references:
- https://github.com/rmusser01/tldw_server/pull/1757
- apps/packages/ui/src/components/Option/Watchlists/shared/__tests__/WatchlistsHealthBar.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address actionable PR 1757 review feedback by preventing the WatchlistsHealthBar AntD Button test mock from forwarding AntD-only type props into native button semantics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The WatchlistsHealthBar AntD Button mock strips AntD's stylistic type prop and keeps native buttons type="button".
- [x] #2 Focused coverage asserts the refresh control uses safe native button semantics.
- [x] #3 Focused WatchlistsHealthBar verification passes after the review fix.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused assertion that the refresh button renders with native type="button", confirming the mock no longer forwards AntD stylistic type props into DOM semantics.
2. Update the AntD Button mock to destructure and ignore the AntD `type` prop before spreading DOM props, while preserving `htmlType` when a test needs a different native type.
3. Rerun the focused WatchlistsHealthBar test and relevant design-system guard checks, then push and resolve the review thread.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Qodo flagged that the AntD Button test mock forwarded `type="text"` into a native `<button>`, creating invalid/default submit semantics in the jsdom test surface. Added a regression assertion for the refresh control's native `type="button"`; the red run failed with received `type="text"`.

Updated the mock to strip AntD's stylistic `type` prop and map `htmlType` to the native button type, defaulting to `button`.

Verification:
- `bunx vitest run src/components/Option/Watchlists/shared/__tests__/WatchlistsHealthBar.test.tsx --reporter=dot --testTimeout=20000` passed 1 file / 1 test.
- `bunx vitest run src/components/Option/Admin/__tests__/LlamacppRuntimePanel.test.tsx src/components/Option/Watchlists/shared/__tests__/WatchlistsHealthBar.test.tsx src/design-system/__tests__/product-state-guard.test.ts --reporter=dot --testTimeout=20000` passed 3 files / 56 tests.
- `bun run verify:design-system-state` passed with the expected existing baseline summary and stale-baseline reporting.
- `git diff --check` passed.
- Backlog section marker sanity check passed.
- Bandit is not applicable because the touched implementation is frontend test code plus this Backlog task record.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed PR #1757 review feedback by stripping AntD's stylistic `type` prop from the WatchlistsHealthBar Button mock, preserving native button semantics, and adding regression coverage for `type="button"`. Verification passed for the focused WatchlistsHealthBar test, the combined runtime/watchlists/guard suite, and `bun run verify:design-system-state`. Bandit is not applicable because the touched implementation is frontend test code plus this Backlog task record.
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
