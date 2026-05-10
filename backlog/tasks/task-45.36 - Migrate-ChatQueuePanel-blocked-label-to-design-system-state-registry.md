---
id: TASK-45.36
title: Migrate ChatQueuePanel blocked label to design-system state registry
status: Done
assignee:
  - Codex
created_date: '2026-05-10 01:22'
updated_date: '2026-05-10 01:24'
labels:
  - design-system
  - ui
  - product-state
  - chat
  - queue
dependencies: []
references:
  - apps/packages/ui/src/components/Common/ChatQueuePanel.tsx
  - apps/packages/ui/src/components/Common/__tests__/ChatQueuePanel.test.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Route the ChatQueuePanel generic blocked fallback label through the canonical design-system state registry instead of leaving the hardcoded product-state label as a baseline exception. This continues the design-system product-state migration under TASK-45 and keeps ChatQueuePanel translation override behavior intact while removing the matching guard baseline entry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ChatQueuePanel uses the design-system state registry for the generic blocked fallback label while preserving explicit blocked reasons and i18n fallback behavior.
- [x] #2 Focused ChatQueuePanel tests cover the registry fallback path and a custom blocked reason path.
- [x] #3 The ChatQueuePanel canonical-state-label baseline entry is removed and the design-system product-state verifier passes.
- [x] #4 Verification notes include focused Vitest coverage, product-state guard coverage, product-state verifier, diff check, TypeScript touched-file status, and Bandit applicability.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused ChatQueuePanel regression test that proves the generic blocked fallback can come from the design-system state registry, while a custom blocked reason still renders unchanged. 2. Verify the new test fails before production changes. 3. Update ChatQueuePanel to use getDesignSystemState("blocked").label as the translation fallback for the generic blocked state only. 4. Remove the matching ChatQueuePanel canonical-state-label baseline exception. 5. Run focused ChatQueuePanel tests, product-state guard tests, bun run verify:design-system-state, git diff --check, and a touched-file TypeScript check; record Bandit as skipped if the touched scope remains UI-only TypeScript/JSON/Backlog.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation notes: Added RED coverage for the generic blocked fallback by mocking getDesignSystemState("blocked") to return a distinct label; the first focused run failed because ChatQueuePanel still rendered the hardcoded Blocked fallback. Updated ChatQueuePanel to use getDesignSystemState("blocked").label as the i18n fallback only when blockedReason is empty, preserving explicit blocked reasons. Removed canonical-state-label:src/components/Common/ChatQueuePanel.tsx:Blocked from the product-state baseline. Verification: focused ChatQueuePanel Vitest passed 5 tests; product-state guard Vitest passed 52 tests; bun run verify:design-system-state exited 0 with 510 baseline exceptions; git diff --check passed; repo-wide bunx tsc --noEmit --pretty false exited 2 on existing unrelated UI TypeScript debt, and rg found no touched-file/design-system matches in the tsc output. Bandit skipped because the touched scope is UI TypeScript, JSON, and Backlog markdown only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated ChatQueuePanel's generic blocked fallback label to the design-system state registry and removed the corresponding baseline exception. Added focused tests proving the registry fallback is used when no blocked reason is present and custom blocked reasons still render unchanged. Verification passed for focused component coverage, product-state guard coverage, the design-system state verifier, and diff whitespace; repo-wide TypeScript still has unrelated existing failures with no touched-file matches. Bandit was not applicable to this UI-only TS/JSON/markdown slice.
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
