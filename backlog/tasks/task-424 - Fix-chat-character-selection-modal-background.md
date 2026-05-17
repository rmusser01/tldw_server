---
id: TASK-424
title: Fix chat character selection modal background
status: Done
labels:
- webui
- chat
- ux
priority: High
modified_files:
- apps/packages/ui/src/components/Common/AssistantSelect.tsx
- apps/packages/ui/src/components/Common/__tests__/AssistantSelect.behavior.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the /chat character/persona selection modal panel use a solid opaque background instead of a translucent/clear backing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Character/persona selection modal on /chat uses a solid opaque panel/background color.
- [ ] #2 Fix is scoped to the modal styling and does not alter unrelated chat cockpit/sidebar behavior.
- [ ] #3 Focused verification is run for the changed frontend scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
['User reported the character selection modal should not have a clear/opaque backing and should be a solid color.']
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated the main /chat assistant/character selector panel to use solid `bg-surface` and inactive option rows to use solid `bg-bg` instead of the unconfigured `bg-background` class. Added a focused regression test asserting the selector panel uses the solid surface token and does not regress to `bg-background`. Verification: `bunx vitest run src/components/Common/__tests__/AssistantSelect.behavior.test.tsx --maxWorkers=1 --no-file-parallelism` passed 16 tests; `git diff --check` passed. Bandit is not applicable because the touched files are TSX/Backlog task metadata only.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
