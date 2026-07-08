---
id: TASK-12908
title: Add adjustable chat transparency theming controls
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-08 02:10'
labels:
  - frontend
  - theming
  - chat
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add user-adjustable transparency controls for chat text/windows and character images so chat theming can reveal configured backgrounds while preserving readability.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Chat window/surface transparency can be adjusted from existing chat/background settings.
- [x] #2 Chat text/message surface transparency can be adjusted without making text itself unreadable by default.
- [x] #3 Character image transparency can be adjusted where character images are rendered in chat.
- [x] #4 A targeted regression check covers the new transparency settings wiring.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Implemented adjustable chat transparency controls in the existing Chat Appearance settings. Added clamped percent settings for chatWindowOpacity, chatMessageOpacity, and chatCharacterImageOpacity.

Wired chatWindowOpacity into the full Playground background wash and cockpit shell plus both sidepanel chat routes. Wired chatMessageOpacity into shared assistant/user message cards and compact user bubbles using background-color alpha so text itself remains opaque. Wired chatCharacterImageOpacity into character portrait images and assistant avatars.

Touched files: apps/packages/ui/src/services/settings/ui-settings.ts, apps/packages/ui/src/components/Option/Settings/ChatSettings.tsx, apps/packages/ui/src/components/Option/Playground/Playground.tsx, apps/packages/ui/src/components/Option/Playground/PlaygroundCockpitShell.tsx, apps/packages/ui/src/routes/sidepanel-chat.tsx, apps/tldw-frontend/extension/routes/sidepanel-chat.tsx, apps/packages/ui/src/components/Common/Playground/Message.tsx, apps/packages/ui/src/components/Common/Playground/PlaygroundUserMessage.tsx, apps/packages/ui/src/routes/__tests__/chat-background-translucency.guard.test.ts.

Verification: bunx vitest run src/routes/__tests__/chat-background-translucency.guard.test.ts src/components/Option/Settings/__tests__/ChatSettings.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-rail-restore.test.tsx passed from apps/packages/ui with 4 files and 26 tests. git diff --check scoped to touched files passed.

Additional verification: ./node_modules/.bin/tsc -p tsconfig.json --noEmit initially hit Node heap limit. Retried with NODE_OPTIONS=--max-old-space-size=8192 and it completed with existing unrelated baseline errors in ChatGreetingPicker, Notes, AudioStudio, ScheduledTasks, background.ts, and other files, with no diagnostics in the touched chat transparency files.

Bandit skipped because the touched implementation is TypeScript/TSX frontend code only.

Code review found an Important risk: broader Playground tests may have stale @/services/settings/ui-settings mocks missing CHAT_WINDOW_OPACITY_SETTING. Reopened task to update mocks and rerun affected tests.

Code review follow-up completed: added CHAT_WINDOW_OPACITY_SETTING and setting-aware useSetting defaults to six Playground test mocks, added an opacity clamping behavior assertion to the transparency guard, and removed duplicate final-summary markers left by the CLI artifact.

Review follow-up verification: bunx vitest run src/routes/__tests__/chat-background-translucency.guard.test.ts src/components/Option/Playground/__tests__/Playground.research-context.integration.test.tsx src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx src/components/Option/Playground/__tests__/Playground.sticky-composer-layout.integration.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx src/components/Option/Playground/__tests__/Playground.search.integration.test.tsx passed from apps/packages/ui with 7 files and 102 tests. git diff --check scoped to touched files passed after review follow-up.

Clean PR worktree verification: bunx vitest run src/routes/__tests__/chat-background-translucency.guard.test.ts src/components/Option/Settings/__tests__/ChatSettings.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-rail-restore.test.tsx src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx src/components/Option/Playground/__tests__/Playground.sticky-composer-layout.integration.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx src/components/Option/Playground/__tests__/Playground.search.integration.test.tsx --reporter=dot passed from apps/packages/ui with 9 files and 108 tests. git diff --check passed. A temporary origin/dev baseline confirmed Playground.research-context.integration.test.tsx is already broken on dev before this PR because its ui-settings mock lacks HEADER_SHORTCUT_IDS; this PR updates related mocks, but that baseline-broken suite was not used as a completion gate.

TypeScript follow-up: VisualIdentityImage now passes an optional style prop through to its img element so character-image opacity can be applied to animated visual identity portraits without violating the component prop contract.

Final clean-worktree verification after VisualIdentityImage style passthrough: bunx vitest run src/components/Common/VisualIdentity/__tests__/VisualIdentityImage.test.tsx src/routes/__tests__/chat-background-translucency.guard.test.ts src/components/Option/Settings/__tests__/ChatSettings.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-rail-restore.test.tsx src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx src/components/Option/Playground/__tests__/Playground.sticky-composer-layout.integration.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx src/components/Option/Playground/__tests__/Playground.search.integration.test.tsx --reporter=dot passed from apps/packages/ui with 10 files and 110 tests. NODE_OPTIONS=--max-old-space-size=8192 ./node_modules/.bin/tsc -p tsconfig.json --noEmit still fails on existing baseline test typing errors outside the touched implementation; the prior touched-file VisualIdentityImage/Message.tsx style-prop error is resolved.

Code review follow-up for PR #2685: removed themedBackdropOpacity from PlaygroundCockpitShell so full Playground applies chatWindowOpacityAlpha only once via the root background wash. Added a cockpit-shell render assertion and source guard to prevent reintroducing a second wash. Verification: bunx vitest run src/routes/__tests__/chat-background-translucency.guard.test.ts src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx --reporter=dot passed with 2 files and 49 tests.

PR #2685 review follow-up completed after rebasing on origin/dev: fixed the invalid --color-surface-2 CSS variable, added opacity alpha fallbacks, moved message/character opacity setting subscriptions to the chat roots via CSS variables, kept message text opaque, and changed the guard test to async source reads with a clear monorepo-checkout error.

Review follow-up verification: bunx vitest run src/components/Common/VisualIdentity/__tests__/VisualIdentityImage.test.tsx src/routes/__tests__/chat-background-translucency.guard.test.ts src/components/Option/Settings/__tests__/ChatSettings.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-rail-restore.test.tsx src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx src/components/Option/Playground/__tests__/Playground.sticky-composer-layout.integration.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx src/components/Option/Playground/__tests__/Playground.search.integration.test.tsx --reporter=dot passed from apps/packages/ui with 10 files and 112 tests. git diff --check passed. NODE_OPTIONS=--max-old-space-size=8192 ./node_modules/.bin/tsc -p tsconfig.json --noEmit still fails on existing baseline type errors outside the touched transparency files.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added three Chat Appearance opacity controls for chat window wash, message window backgrounds, and character images. The settings are clamped percentages, apply across WebUI chat and both sidepanel routes, and preserve message text opacity/readability while allowing themed backgrounds to show through.
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
