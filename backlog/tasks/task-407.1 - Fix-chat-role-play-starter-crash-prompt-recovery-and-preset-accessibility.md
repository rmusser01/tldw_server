---
id: TASK-407.1
title: 'Fix chat role-play starter crash, prompt recovery, and preset accessibility'
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-17 07:53'
labels:
  - chat
  - ux
  - roleplay
  - stage-1
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-17-main-chat-role-play-preset-remediation-implementation-plan.md
parent_task_id: TASK-407
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Stage 1 implementation for the main /chat role-play preset plan: reproduce or retire the observed starter crash, harden prompt edit/clear recovery, and make compact generation presets accessible.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Chat as a character starter and default/equivalent assistant selection do not crash /chat.
- [x] #2 Custom system prompt edit and clear remain reachable even when the prompt library is empty.
- [x] #3 Compact generation preset controls expose understandable accessible names and keyboard behavior.
- [x] #4 Focused Stage 1 tests and browser verification are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Setup before implementation:
- Dedicated branch/worktree: codex/chat-role-play-preset-remediation at .worktrees/chat-role-play-preset-remediation, based on dev plus role-play spec/plan commits.
- Dependency setup: `bun install` hung after partial resolution and was stopped; `bun install --ignore-scripts` completed and left no tracked file changes.
- Baseline focused tests passed: `bunx vitest run ../packages/ui/src/components/Common/__tests__/AssistantSelect.behavior.test.tsx ../packages/ui/src/components/Common/__tests__/PromptSelect.system-prompt-modal.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/ParameterPresets.guard.test.ts` (13 tests passed). PromptSelect test emitted a pre-existing React key warning from the test Dropdown mock.

Stage 1 implementation notes:
- Added focus return for dropdown `AssistantSelect` selection and regression coverage for the `Chat as a character` starter opening character selection and selecting the default assistant.
- Fixed the reproduced crash chain by lazy-mounting `TtsClipsDrawer`, preventing unconditional recommendation dismissal cleanup in `useContextWindow`, and making `useCharacterGreeting` idempotent when the same rendered greeting is already injected.
- Added prompt recovery actions so a non-empty current system prompt can be edited or cleared even when the saved prompt library is empty.
- Added compact generation preset accessible names while preserving tooltip setting details.

Verification:
- `bunx vitest run ../packages/ui/src/components/Layouts/__tests__/Header.tts-clips-lazy-mount.test.ts ../packages/ui/src/components/Option/Playground/__tests__/useContextWindow.recommendations.test.tsx ../packages/ui/src/hooks/__tests__/useCharacterGreeting.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.role-play-starter.integration.test.tsx ../packages/ui/src/components/Common/__tests__/AssistantSelect.behavior.test.tsx ../packages/ui/src/components/Common/__tests__/PromptSelect.system-prompt-modal.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/ParameterPresets.guard.test.ts --reporter=verbose` passed: 7 files, 28 tests.
- Browser `/chat` verification on `http://127.0.0.1:3001/chat`: clicking `Chat as a character` opened the selector; selecting `Default Assistant` did not crash; picker closed; selected assistant was visible; focus returned to the `character-select` button. Captured serious error filter was empty for `Maximum update depth`, `RouteErrorBoundary`, `Application error`, and `Unhandled Runtime Error`.
- Browser still logs pre-existing/nonfatal chat settings 404 warnings for generated chat ids and an Ant Design `dropdownRender` deprecation warning; no crash or route error remained.
- Compact generation preset accessibility and empty prompt-library recovery are covered by component tests; the compact generation control was not present in the default desktop `/chat` browser surface inspected.
- Bandit not run for Stage 1 because touched implementation files are TypeScript/TSX frontend files, not Python.

Quality review follow-up:
- Addressed P2 by including character name and avatar URL in the `useCharacterGreeting` idempotency key so identical greeting text still refreshes hydrated character metadata. Added regression coverage for same-id/same-greeting name-avatar refresh.
- Addressed P3 by adding a ComposerToolbar guard that the real toolbar owns the dropdown `AssistantSelect` used by starter events, complementing the larger PlaygroundForm starter integration test.
- Re-ran focused Stage 1 suite with added coverage: 8 files, 52 tests passed.
- Re-ran browser `/chat` starter verification after review fixes: no crash text, selected assistant visible, picker closed, serious error filter empty.

Review status:
- Spec compliance review: PASS_WITH_NOTES; non-blocking plan Step 10 tracking note was addressed.
- Code-quality review: initial FAIL for stale greeting metadata and brittle starter coverage; fixed both. Re-review: PASS_WITH_NOTES with no blocking findings. The remaining note is that the toolbar ownership guard is source-text based and could be strengthened later by rendering the real toolbar path.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 1 stabilizes the current /chat role-play starter path and the adjacent recovery/accessibility affordances. The role-play starter opens character selection, selecting the default assistant no longer crashes, focus returns to the trigger, current custom system prompts can be edited or cleared with an empty prompt library, and compact generation presets expose accessible names in component coverage. Remaining observed browser noise is limited to pre-existing chat-settings 404 warnings and an Ant Design deprecation warning.
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
