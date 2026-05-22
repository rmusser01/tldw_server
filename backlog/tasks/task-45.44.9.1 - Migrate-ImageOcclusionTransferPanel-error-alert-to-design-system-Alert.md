---
id: TASK-45.44.9.1
title: Migrate ImageOcclusionTransferPanel error alert to design-system Alert
status: Done
labels:
- design-system
- webui
- extension
- product-state
- flashcards
priority: medium
parent_task_id: TASK-45.44.9
references:
- https://github.com/rmusser01/tldw_server/issues/1666
- apps/packages/ui/src/components/Flashcards/tabs/ImageOcclusionTransferPanel.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ImageOcclusionTransferPanel.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- https://github.com/rmusser01/tldw_server/pull/1924
modified_files:
- apps/packages/ui/src/components/Flashcards/tabs/ImageOcclusionTransferPanel.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ImageOcclusionTransferPanel.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace the ImageOcclusionTransferPanel generation/save error banner from AntD Alert to the canonical design-system Alert, keep the existing user-visible error behavior, remove the matching product-state baseline exception, and record before/after verification evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ImageOcclusionTransferPanel error output renders through the canonical design-system Alert wrapper while preserving the existing error text and notification behavior.
- [x] #2 The AntD Alert import is removed from ImageOcclusionTransferPanel without broadening unrelated Flashcards UI changes.
- [x] #3 The matching ImageOcclusionTransferPanel product-state baseline entry is removed and verify:design-system-state reports one fewer exception.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
ImageOcclusionTransferPanel generation/save error output now uses the canonical design-system Alert while preserving the existing error text and message.error behavior. Review follow-up addressed Gemini's alert API feedback: full message text now renders as Alert children, and validation warning paths render with warning severity while caught generation/save failures render with error severity. Removed the matching product-state baseline exception, reducing verify:design-system-state from 323 to 322 allowed legacy exceptions. Draft PR: https://github.com/rmusser01/tldw_server/pull/1924. Verification: focused ImageOcclusionTransferPanel Vitest passed (4 tests), product-state guard Vitest passed (52 tests), bun run verify:design-system-state passed with 322 exceptions, git diff --check passed. NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false remains red on inherited repo-wide TypeScript debt; a targeted diagnostic filter for ImageOcclusionTransferPanel, design-system-product-state-baseline, and TASK-45.44.9.1 produced no matches. Bandit not applicable because this slice touched TS/TSX/JSON/Backlog markdown only.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Focused ImageOcclusionTransferPanel test records the canonical alert behavior.
- [x] #8 Product-state guard test and package verifier pass from apps/packages/ui.
- [x] #9 Bandit is run for touched Python paths or recorded as not applicable for TS/JSON-only changes.
<!-- DOD:END -->
