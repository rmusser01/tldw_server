---
id: TASK-45.30
title: Adapt Prompt SyncStatusBadge to design-system state registry
status: Done
assignee:
  - Codex
created_date: '2026-05-09 18:58'
updated_date: '2026-05-09 19:56'
labels:
  - design-system
  - ui
  - product-state
dependencies: []
references:
  - apps/packages/ui/src/components/Option/Prompt/SyncStatusBadge.tsx
  - >-
    apps/packages/ui/src/components/Option/Prompt/__tests__/SyncStatusBadge.test.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
documentation:
  - Docs/Design/tldw_web_design_system_contract.md
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the Prompt workspace sync status badge away from local AntD Tag/color status UI and through the canonical design-system state registry while preserving existing retry, compact, tooltip, server ID, and conflict resolution behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sync statuses resolve through getDesignSystemState before choosing shared Badge severity styling.
- [x] #2 The full-size sync badge renders the shared Badge primitive instead of AntD Tag while preserving labels, icons, interactive click behavior, retry behavior, and server ID display.
- [x] #3 Compact mode preserves the existing icon-only affordance and retry behavior without introducing a visible text badge.
- [x] #4 Focused tests cover status-to-state/variant behavior, compact behavior, retry behavior, interactive behavior, and null/default local fallback.
- [x] #5 The design-system product-state baseline no longer contains the SyncStatusBadge local-status-badge exception.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused SyncStatusBadge tests that prove each sync status resolves through getDesignSystemState, renders the shared Badge variant, and no longer renders AntD Tag in full-size mode.
2. Migrate SyncStatusBadge full-size rendering from AntD Tag colors to the design-system state registry plus shared Badge severity variants, while preserving compact icon-only behavior, retry, interaction, tooltips, and server ID display.
3. Remove the SyncStatusBadge local-status-badge baseline exception and verify the design-system product-state guard accepts the migration.
4. Add guard coverage for compound status badge adapters so returned JSX trees that include shared Badge plus same-owner state registry mapping are accepted, while keeping LoadingState direct-return validation strict.
5. Run focused component tests, guard tests, design-system verifier, whitespace diff check, and a UI typecheck attempt; document unrelated typecheck failures and Bandit skip for UI-only TS/JS changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the Prompt SyncStatusBadge migration via TDD. The new component tests first failed because getDesignSystemState was not called, then passed after the full-size AntD Tag path was replaced with shared Badge severity variants.

The design-system verifier initially blocked local-status-badge for SyncStatusBadge because the guard only accepted status adapters whose direct return expression was Badge. Added a failing guard regression for compound returned JSX containing Badge plus same-owner state registry mapping, then changed only the status-badge detector to inspect returned JSX trees. LoadingState still uses the stricter direct-return detector.

Verification: bunx vitest run src/components/Option/Prompt/__tests__/SyncStatusBadge.test.tsx --reporter=dot passed 11 tests; bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot passed 47 tests; bun run verify:design-system-state exited 0 with SyncStatusBadge removed from blocked/baselined findings; git diff --check exited 0. bunx tsc --noEmit --pretty false exited 2 with unrelated existing UI/test type errors and no touched-file matches in /tmp/tldw_ui_tsc_prompt_sync_status.txt. Bandit skipped because this slice only touches UI TypeScript/JavaScript and JSON baseline data.

PR review pass for #1437: Qodo requested avoiding unused state/variant derivation in compact mode. Gemini noted the status-badge guard does not detect Badge usage inside nested render callbacks such as map(), and suggested restoring inline-flex/items-center/gap-1 classes. Verified Badge already provides inline-flex items-center gap-1 globally, so that layout comment will be addressed with evidence rather than redundant classes. Plan: add failing tests for compact no registry call and map-rendered Badge detection, implement the minimal component/guard fixes, rerun focused verification, update PR, and resolve/reply to review threads.

PR review fixes implemented. Added compact-mode coverage proving compact icon-only badges do not call getDesignSystemState or render Badge. Added guard coverage for returned map() render callbacks and an event-handler false-positive case. Moved state/variant derivation below the compact return and taught returnedExpressionContainsJsxTag to traverse map callbacks only.

Review-fix verification: bunx vitest run src/components/Option/Prompt/__tests__/SyncStatusBadge.test.tsx --reporter=dot passed 12 tests; bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot passed 49 tests; bun run verify:design-system-state exited 0; git diff --check exited 0; bunx tsc --noEmit --pretty false exited 2 with the existing 236-line unrelated UI/test type debt and no matches for SyncStatusBadge, design-system-product-state-rules, or product-state-guard.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated Prompt SyncStatusBadge full-size rendering from AntD Tag color props to the design-system state registry and shared Badge severity variants. Added focused component coverage for every sync status, default local fallback, compact behavior, retry, and interaction. Updated the product-state guard so status-badge adapters that return compound JSX are accepted when their returned UI contains the canonical Badge and the same owner calls getDesignSystemState, including returned map() render callbacks, while retaining strict direct-return validation for LoadingState and guarding against event-handler-only false positives. Removed the SyncStatusBadge baseline exception. PR review follow-up moved non-compact state/variant derivation below the compact return.
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
