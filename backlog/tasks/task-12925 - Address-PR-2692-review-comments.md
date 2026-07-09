---
id: TASK-12925
title: Address PR 2692 review comments
status: In Progress
priority: High
references:
- https://github.com/rmusser01/tldw_server/pull/2692
modified_files:
- IMPLEMENTATION_PLAN_pr2692_review_fixes.md
- apps/packages/ui/src/assets/locale/*/review.json
- apps/packages/ui/src/components/Common/Playground/Message.tsx
- apps/packages/ui/src/components/Common/Playground/PlaygroundUserMessage.tsx
- apps/packages/ui/src/components/Common/RecipeCard/RecipeCard.tsx
- apps/packages/ui/src/components/Common/RecipeCard/__tests__/RecipeCard.test.tsx
- apps/packages/ui/src/components/Layouts/SettingsOptionLayout.tsx
- apps/packages/ui/src/components/Option/Characters/CharacterEditorForm.tsx
- apps/packages/ui/src/components/Option/Characters/CharacterExpressionImagesSection.tsx
- apps/packages/ui/src/components/Option/Characters/Manager.tsx
- apps/packages/ui/src/components/Option/Characters/__tests__/CharacterEditorForm.expression-validation.test.tsx
- apps/packages/ui/src/components/Option/Characters/utils.ts
- apps/packages/ui/src/components/Option/Playground/PlaygroundComposerNotices.tsx
- apps/packages/ui/src/components/Option/Playground/hooks/usePlaygroundVoiceChat.ts
- apps/packages/ui/src/components/Option/Settings/__tests__/SearchModeSettings.test.tsx
- apps/packages/ui/src/components/Option/Settings/preferences-settings.tsx
- apps/packages/ui/src/components/Option/Settings/setup-recovery-settings.tsx
- apps/packages/ui/src/components/Option/Settings/ui-customization.tsx
- apps/packages/ui/src/components/Quiz/tabs/GenerateTab.tsx
- apps/packages/ui/src/components/Sidepanel/Chat/CharacterControlsSheet.tsx
- apps/packages/ui/src/services/settings/chat-opacity-css-vars.ts
- tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/audio_stream_protocol.py
- tldw_Server_API/app/core/exceptions.py
- tldw_Server_API/tests/Audio/test_ws_audio_chat_stream.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve verified CodeRabbit/Qodo/Gemini review findings on PR #2692 (dev to main release prep), update PR metadata where needed, run targeted verification, and push fixes to dev.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
IMPLEMENTATION_PLAN_pr2692_review_fixes.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created implementation plan and triaged unresolved PR #2692 review threads. Valid fixes are grouped into backend protocol/tests, frontend correctness/accessibility/i18n nits, and PR metadata. Cockpit remount/static-id suggestions conflict with recent CI flake fixes and will be handled with rationale rather than code churn.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Local verification recorded: Python audio websocket tests passed (25 passed); Bandit production scope passed and touched test scope passed with B101 excluded for pytest asserts; frontend typecheck passed; focused Vitest passed (RecipeCard, SearchModeSettings, SetupRecoverySettings, CharacterEditorForm expression validation; 13 tests); git diff --check passed. PR metadata/comments still need update after commit/push.
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
