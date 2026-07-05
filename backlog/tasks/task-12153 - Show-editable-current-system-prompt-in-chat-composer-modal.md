---
id: TASK-12153
title: Show editable current system prompt in chat composer modal
status: Done
labels:
- ui
- chat
- prompt-select
modified_files:
- apps/packages/ui/src/components/Option/Playground/SystemPromptTemplates.tsx
- apps/packages/ui/src/components/Option/Playground/ComposerToolbar.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/SystemPromptTemplates.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update the chat composer System Prompts modal used by the WebUI and browser extension so the current system prompt is visible in an editable text box above the prompt search/select controls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The chat composer System Prompts modal now shows the current system prompt in an editable textarea above template search/select when opened from composer controls. Editing the textbox updates the existing systemPrompt state; saved template selection remains unchanged.

Verification: focused Vitest coverage passed for SystemPromptTemplates, ComposerToolbar desktop/mobile, and PromptSelect modal tests; extension compile passed; git diff whitespace check passed. Full WebUI/shared UI typechecks remain blocked by pre-existing unrelated TypeScript errors. Rendered verification passed on an isolated WebUI stack after avoiding the existing occupied frontend/backend ports.

Bandit: not applicable; touched files are TypeScript/TSX UI files only.
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
