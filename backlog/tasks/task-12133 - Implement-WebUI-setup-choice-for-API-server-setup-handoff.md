---
id: TASK-12133
title: Implement WebUI setup choice for API server setup handoff
status: In Progress
assignee: []
created_date: '2026-07-03 22:59'
updated_date: '2026-07-03 23:08'
labels:
  - webui
  - setup
  - onboarding
dependencies:
  - TASK-12123
references:
  - Docs/superpowers/specs/2026-07-03-webui-setup-choice-design.md
  - Docs/superpowers/plans/2026-07-03-webui-setup-choice-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved WebUI /setup pre-wizard choice screen that explains WebUI setup versus API server setup, resolves a browser-openable API setup URL when possible, handles blocked/recovery state safely, and preserves existing manual recovery UI when setup state or metadata cannot be loaded.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 SetupEntryChoice renders before the existing WebUI wizard on /setup when backend setup is incomplete.
- [ ] #2 API server setup link/fallback behavior follows the approved URL-resolution and local/remote copy rules.
- [ ] #3 Blocked first-run state cannot enter the normal WebUI wizard until refresh returns a mutable state.
- [ ] #4 Manual connection and recovery UI remain available when first-run state or metadata cannot be loaded.
- [ ] #5 Focused Vitest, Playwright, typecheck, and applicable security verification are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-03-webui-setup-choice-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation started in isolated worktree .worktrees/codex-webui-setup-choice-impl on branch codex/webui-setup-choice-impl. Baseline focused Vitest passed: setup-status and option-setup-readiness, 19 tests.

Task 1 complete: helper and resolver committed in f8adf59740. Spec-compliance and code-quality reviews approved. Focused helper Vitest passed: 12 tests.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
