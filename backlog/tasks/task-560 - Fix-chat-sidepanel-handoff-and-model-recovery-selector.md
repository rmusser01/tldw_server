---
id: TASK-560
title: Fix /chat sidepanel handoff and model recovery selector
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-30 16:34'
labels:
  - chat
  - ux
  - sidepanel
  - playground
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track the focused /chat UX follow-up: route extension sidepanel full-screen and continue handoffs to /chat, preserve rail regression coverage, and reuse the existing compact model selector for locally recoverable chat error recovery instead of opening the full model settings modal when the action is simple model switching.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Focused sidepanel route tests and cockpit rail guard pass after implementation.
- [x] #2 Chat error banner model recovery can open the existing compact model selector without routing users through the full model settings modal for simple switching.
- [x] #3 Model recovery selector wiring has failing-then-passing tests.
- [x] #4 Verification results and known skips are recorded before final handoff.
- [x] #5 Packaged extension sidepanel chat full-screen and continue handoff tests assert the options.html hash carrier for the WebUI /chat route.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation completed and rebased onto origin/dev during the work. RED/GREEN evidence: initial focused test run failed exactly on missing openModelSelector export, banner still falling back to Health diagnostics for open-model-selector, PlaygroundForm lacking OPEN_MODEL_SELECTOR_EVENT listener, and chat-error-message lacking selector recovery metadata. GREEN verification passed: bun run test src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.fullscreen-route.test.tsx src/components/Sidepanel/Chat/__tests__/ControlRow.chat-handoff.test.tsx src/components/Sidepanel/Chat/__tests__/ControlRow.role-play-handoff.test.tsx src/components/Option/Playground/__tests__/Playground.cockpit-regression.guard.test.ts src/components/Option/Playground/__tests__/playground-cockpit-actions.test.ts src/components/Option/Playground/__tests__/PlaygroundChatErrorBanner.test.tsx src/utils/__tests__/chat-error-message.test.ts (7 files, 42 tests). Browser smoke with local Next dev server on http://127.0.0.1:18001 verified /chat loads with cockpit rails and dispatching tldw:open-model-selector opens and focuses the existing composer model selector. Extension package inspection showed WXT builds options.html and sidepanel.html, not a bare /chat entrypoint; sidepanel full-app actions therefore use options.html#/chat as the packaged carrier for the WebUI /chat route. Packaged extension smoke target was attempted with escalation; the local mock server could bind only outside the sandbox, but the headed extension launch still skipped with browserType.launchPersistentContext timeout after Chromium launched, so packaged runtime remains an environment skip rather than passing evidence. Bandit skipped: frontend TypeScript/TSX and Backlog-only changes; no Python touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Preserved the packaged extension options.html hash carrier for /chat handoffs, added dedicated cockpit wiring so model-unavailable and empty-response recovery opens the existing compact ChatModelSelectorDropdown, preserved full model settings for settings/configuration recovery, and kept focused sidepanel/model recovery/rail regression coverage in place. A true chrome-extension://.../chat URL would require a dedicated WXT entrypoint or an external WebUI URL contract.
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
