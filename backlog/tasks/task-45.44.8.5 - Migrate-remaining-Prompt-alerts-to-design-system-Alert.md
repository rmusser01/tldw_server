---
id: TASK-45.44.8.5
title: Migrate remaining Prompt alerts to design-system Alert
status: Done
labels:
- design-system
- webui
- product-state
priority: medium
parent_task_id: TASK-45.44.8
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the remaining Prompt and Prompt Studio AntD Alert product-state exceptions from the design-system baseline by migrating the owned alert surfaces to the shared design-system Alert primitive.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Prompt and Prompt Studio remaining AntD Alert product-state surfaces render the design-system Alert primitive instead of AntD Alert.
- [x] #2 The product-state baseline contains no Prompt or Prompt Studio Alert exceptions touched by this slice.
- [x] #3 Focused regression coverage verifies the migrated alert surfaces expose the design-system Alert marker and preserve user-facing copy.
- [x] #4 Relevant verification commands are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Migrated remaining Prompt and Prompt Studio Alert surfaces from AntD Alert to the shared design-system Alert primitive in ConflictResolutionModal, PromptBody, PromptDrawer, PromptFullPageEditor, ExecutePlayground, and PromptEditorDrawer.

Removed 11 Prompt entries from the product-state baseline, reducing live baseline exceptions from 97 to 86 and leaving no Prompt/Prompt Studio baseline entries.

Review follow-up: rebased on latest dev and wrapped the PromptFullPageEditor structured prompt Alert title/description in t(...) with defaultValue copy preserved.

Verification recorded:
- `bunx vitest run src/components/Option/Prompt/__tests__/ConflictResolutionModal.test.tsx src/components/Option/Prompt/__tests__/PromptDrawer.structured-prompts.test.tsx src/components/Option/Prompt/__tests__/PromptFullPageEditor.structured-prompts.test.tsx src/components/Option/Prompt/Studio/Prompts/__tests__/PromptEditorDrawer.structured.test.tsx src/components/Option/Prompt/Studio/Prompts/__tests__/ExecutePlayground.design-system.test.tsx --reporter=dot`
- `bun run verify:design-system-state`
- `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false`

Bandit skipped because this is TypeScript/UI-only work.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Remaining Prompt and Prompt Studio AntD Alert product-state exceptions are migrated to the design-system Alert primitive, with focused regression coverage and baseline cleanup recorded. Verification: focused Prompt Vitest suite passed, design-system product-state guard passed with 86 remaining exceptions and zero Prompt entries, and TypeScript passed with the larger Node heap. Bandit skipped because this is TypeScript/UI-only work.
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
