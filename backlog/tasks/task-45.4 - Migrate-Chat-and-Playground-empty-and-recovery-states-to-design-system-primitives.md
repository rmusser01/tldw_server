---
id: TASK-45.4
title: >-
  Migrate Chat and Playground empty and recovery states to design-system
  primitives
status: Done
assignee:
  - '@Codex'
created_date: '2026-05-05 03:42'
labels:
  - design-system
  - frontend
  - chat
  - playground
dependencies: []
documentation:
  - Docs/Design/tldw_web_design_system_contract.md
  - Docs/Design/tldw_web_design_system_inventory.md
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first bounded Chat/Playground migration slice from the design-system inventory. Scope is limited to empty, disconnected, loading/retrying, and recovery/error state surfaces in Playground and Sidepanel Chat. Do not migrate generic Button ownership, page shells, broad AntD usage, status chips, or modal footers in this slice unless required by the target components.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Playground empty/disconnected state uses shared design-system primitives or a thin adapter while preserving current user-facing actions and accessible labels.
- [x] #2 Sidepanel Chat empty state uses shared design-system primitives or a thin adapter while preserving current first-use and reconnect guidance.
- [x] #3 Sidepanel Chat connection/recovery banner uses canonical design-system state language for unavailable/degraded/retrying/error states without broad chat composer refactors.
- [x] #4 Focused tests cover the migrated empty and connection states and continue to assert accessible labels/actions.
- [x] #5 Verification includes the focused Chat/Playground Vitest slice from the inventory plus git diff checks; Bandit is run for touched Python only or explicitly skipped for frontend-only changes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect current Playground and Sidepanel Chat empty/recovery components and their focused tests.
2. Add or update failing tests for the expected design-system state primitives and accessible state labels/actions.
3. Migrate the narrow target components to shared `EmptyState`, `LoadingState`, `RecoveryCallout`, `StatePanel` or thin adapters while preserving behavior.
4. Run focused Vitest coverage from the inventory, update task notes, and commit the slice.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started after PR #1286 merged into dev at merge commit be9c41fa4f9a54e33ff84790125bb0ee083f7eaf. Work is isolated in .worktrees/tldw-chat-playground-design-system on branch codex/tldw-chat-playground-design-system from origin/dev.

Migrated PlaygroundEmpty to the shared EmptyState primitive while preserving Start chatting, Quick Ingest, Open Settings, mode launcher, and tour actions.

Migrated Sidepanel Chat setup/auth/unavailable empty state and ConnectionBanner to RecoveryCallout/StatePanel state language while preserving first-run guidance, retry/settings actions, and inline single-user API key repair.

Verification:
- PASS: bunx vitest run ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.disconnected.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundChatErrorBanner.test.tsx ../packages/ui/src/components/Sidepanel/Chat/__tests__/ConnectionBanner.test.tsx ../packages/ui/src/components/Sidepanel/Chat/__tests__/empty.test.tsx --reporter=dot
- PASS: apps/tldw-frontend/node_modules/.bin/eslint -c apps/tldw-frontend/eslint.config.mjs <touched UI files> (the Next plugin still reports the existing pages-directory notice, but exits 0 with no lint warnings/errors)
- PASS: git diff --check
- PASS: full frontend tsc output filtered for this slice's touched files shows no local type errors
- KNOWN BASELINE: bunx tsc --noEmit --pretty false -p tsconfig.json still exits 2 on unrelated pre-existing frontend type errors across packages/ui, e2e, generated client, and Vite/Vitest config typing.
- SKIP: Bandit is not applicable; this slice touched frontend TypeScript/TSX and Backlog docs only.

PR review pass:
- Address Gemini/Qodo feedback on first-run focus management by moving focus back to a component-scoped ref exposed through design-system action props.
- Address Gemini feedback on disconnected Playground visual hierarchy by keeping Open Settings adjacent to the disconnected message.
- Address Gemini feedback on ConnectionBanner secondary action readability.
- Address Gemini feedback on Sidepanel step summary styling by using StatePanel diagnostics instead of a custom child block.

PR review verification:
- PASS: bunx vitest run ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.disconnected.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundChatErrorBanner.test.tsx ../packages/ui/src/components/Sidepanel/Chat/__tests__/ConnectionBanner.test.tsx ../packages/ui/src/components/Sidepanel/Chat/__tests__/empty.test.tsx --reporter=dot
- PASS: apps/tldw-frontend/node_modules/.bin/eslint -c apps/tldw-frontend/eslint.config.mjs <touched UI files> (the Next plugin still reports the existing pages-directory notice, but exits 0 with no lint warnings/errors)
- PASS: bunx prettier --check <touched UI files>
- PASS: git diff --check
- PASS: full frontend tsc output filtered for this slice's touched files shows no local type errors
- KNOWN BASELINE: bunx tsc --noEmit --pretty false -p tsconfig.json still exits 2 on unrelated pre-existing frontend type errors.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the first Chat/Playground state slice to the design-system primitives. Playground empty now renders through EmptyState, Sidepanel Chat connection empty states render through RecoveryCallout, and ConnectionBanner now uses canonical recovery states while preserving the existing user actions and inline API-key repair path. PR review fixes keep disconnected Settings guidance adjacent to the message, restore scoped focus management through design-system action refs, simplify secondary-action logic, and render setup progress via diagnostics.
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
