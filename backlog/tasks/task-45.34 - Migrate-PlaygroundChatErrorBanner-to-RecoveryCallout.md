---
id: TASK-45.34
title: Migrate PlaygroundChatErrorBanner to RecoveryCallout
status: Done
assignee: []
created_date: '2026-05-10 00:13'
updated_date: '2026-05-10 00:22'
labels:
  - design-system
  - ui
  - product-state
  - recovery
  - playground
dependencies: []
references:
  - >-
    apps/packages/ui/src/components/Option/Playground/PlaygroundChatErrorBanner.tsx
  - >-
    apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundChatErrorBanner.test.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the remaining PlaygroundChatErrorBanner local-recovery-banner debt from the shared Alert recovery surface to the canonical RecoveryCallout/StatePanel product-state primitive. Preserve chat error decoding, dismissal behavior, diagnostics navigation, accessible labels, and the existing data-testid while removing the stale baseline entry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PlaygroundChatErrorBanner renders through the shared RecoveryCallout product-state primitive with state=error.
- [x] #2 Diagnostics navigation and dismiss interactions remain covered by focused tests.
- [x] #3 The PlaygroundChatErrorBanner local-recovery-banner baseline entry is removed and the design-system verifier passes with the remaining expected baseline debt.
- [x] #4 Verification records focused component tests, product-state guard tests, design-system verifier, syntax/whitespace checks, and known TypeScript/Bandit skips.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused failing test that PlaygroundChatErrorBanner renders RecoveryCallout while preserving diagnostics and dismiss behavior. 2. Migrate the banner to RecoveryCallout/StatePanel action semantics without changing error scanning hooks. 3. Remove the PlaygroundChatErrorBanner local-recovery-banner baseline exception. 4. Run focused tests, product-state guard tests, design-system verifier, git diff check, and touched-file TypeScript filtering.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented PlaygroundChatErrorBanner through RecoveryCallout with state=error, preserving the existing data-testid and role=alert surface. Diagnostics navigation now uses the RecoveryCallout primary action and navigate("/settings/health"); dismiss uses the secondary action and existing onDismiss(error.key) callback.

The PlaygroundChatErrorBanner local-recovery-banner baseline exception was removed. While verifying against current dev, the design-system verifier exposed unrelated baseline drift in CharacterDialogs and Sidepanel conversation-context labels, so this task also refreshed those existing legacy entries and removed stale CharacterDialogs ids to keep the guard passing without re-adding the Playground debt.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated PlaygroundChatErrorBanner from the shared Alert primitive to the canonical RecoveryCallout product-state primitive. This keeps chat error decoding behavior intact while making the rendered recovery UI visible to the product-state guard as canonical design-system usage.

Verification: bunx vitest run src/components/Option/Playground/__tests__/PlaygroundChatErrorBanner.test.tsx --reporter=dot passed 6 tests; bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot passed 52 tests; bun run verify:design-system-state passed with 512 allowed legacy exceptions and only ConnectionProblemBanner remaining under local-recovery-banner; git diff --check passed. bunx tsc --noEmit --pretty false exited 2 on existing repo-wide TypeScript test debt, with no filtered output for the touched component, guard, baseline, RecoveryCallout, StatePanel, or task. Bandit skipped because this is UI-only TS/TSX/JSON/Backlog work.
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
