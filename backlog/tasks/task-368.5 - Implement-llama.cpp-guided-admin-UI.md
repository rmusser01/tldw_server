---
id: TASK-368.5
title: Implement llama.cpp guided admin UI
status: Done
assignee:
  - Codex
created_date: '2026-05-15 03:45'
updated_date: '2026-05-29 05:11'
labels:
  - implementation
  - frontend
  - llamacpp
dependencies:
  - TASK-368.4
documentation:
  - Docs/superpowers/specs/2026-05-15-llamacpp-server-management-webui-design.md
  - >-
    Docs/superpowers/plans/2026-05-15-llamacpp-server-management-webui-implementation-plan.md
references:
  - https://github.com/rmusser01/tldw_server/pull/1727
  - https://github.com/rmusser01/tldw_server/pull/1764
  - https://github.com/rmusser01/tldw_server/pull/2121
modified_files:
  - apps/packages/ui/src/components/Option/Admin/LlamacppAdminPage.tsx
  - apps/packages/ui/src/components/Option/Admin/LlamacppReadinessPanel.tsx
  - apps/packages/ui/src/components/Option/Admin/LlamacppInventoryPanel.tsx
  - apps/packages/ui/src/components/Option/Admin/LlamacppLaunchPanel.tsx
  - apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx
  - apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppReadinessPanel.test.tsx
  - apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppInventoryPanel.test.tsx
  - apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppLaunchPanel.test.tsx
parent_task_id: TASK-368
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the guided WebUI slice from the implementation plan. Reshape the llama.cpp admin page into readiness inventory and launch panels using the new client methods while preserving existing advanced launch controls and admin guard behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The page renders readiness state with saved active and restart required messaging.
- [x] #2 The page renders model inventory and starts models by stable model ID.
- [x] #3 Hardware warnings are shown without disabling start solely because hardware data is unknown or risky.
- [x] #4 The chat wiring action appears after a running managed server is available and is never called automatically.
- [x] #5 Focused frontend component tests pass.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
This is a stale tracker closeout after PR #2121 merged. The guided llama.cpp Admin UI is already present on current `origin/dev`; this branch updates the Backlog record to match the shipped frontend.

Implementation provenance:
- PR #1727 (`726958be39 Improve llama.cpp WebUI server management`) reshaped the llama.cpp admin page into readiness, inventory, and launch panels, preserved advanced launch controls, and added focused component coverage.
- PR #1764 (`560c8e17b3 Implement llama.cpp asset inventory v2`) extended the guided UI and tests for stable model inventory selection and `start-by-model` behavior.
- PR #2121 closed the frontend API client prerequisite and refreshed ownership metadata required by this UI surface.

Verified behavior:
- The page renders saved/active readiness state and restart-required messaging.
- Model inventory renders display names, warnings, and stable model selection.
- Launch uses the selected stable `model_id`.
- Hardware warnings remain advisory and do not disable launch solely because hardware probing is unavailable.
- Chat wiring is shown only after a running managed server is available and is never invoked automatically.

Verification commands:
- Initial `bun run test ...` failed because the fresh worktree had incomplete frontend dependency symlinks and `vitest` was unavailable.
- `bun install` from `apps/` repaired the frontend dependency symlinks without changing tracked files.
- `bun run test src/components/Option/Admin/__tests__/LlamacppAdminPage.test.tsx src/components/Option/Admin/__tests__/LlamacppReadinessPanel.test.tsx src/components/Option/Admin/__tests__/LlamacppInventoryPanel.test.tsx src/components/Option/Admin/__tests__/LlamacppLaunchPanel.test.tsx`
- Result: 4 files passed, 26 tests passed in 12.33s. Node emitted expected localStorage experimental warnings.

Known skips:
- Bandit was not run because this closeout branch changes only Backlog metadata and no Python runtime code.
<!-- SECTION:NOTES:END -->

## Final Summary
<!-- SECTION:SUMMARY:BEGIN -->
Closed `TASK-368.5` against the guided llama.cpp Admin UI already merged into `dev`. The shipped UI includes readiness, inventory, and launch panels with stable model-id launch, advisory hardware warnings, preserved advanced controls, admin guard behavior, and manual-only chat wiring.
<!-- SECTION:SUMMARY:END -->
