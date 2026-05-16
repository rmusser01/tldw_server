---
id: TASK-402
title: Polish main /chat mobile cockpit experience
status: Done
labels:
- chat
- cockpit
- webui
- ux
- mobile
priority: HIGH
references:
- Docs/superpowers/specs/2026-05-15-main-chat-cockpit-maturity-roadmap-design.md
modified_files:
- Docs/superpowers/plans/2026-05-16-chat-cockpit-mobile-cockpit-polish.md
- apps/packages/ui/src/components/Option/Playground/PlaygroundCockpitShell.tsx
- apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-maturity.test.tsx
- apps/packages/ui/src/components/Common/PromptSelect.tsx
- apps/packages/ui/src/components/Common/__tests__/PromptSelect.system-prompt-modal.test.tsx
- apps/packages/ui/src/components/Common/Settings/CurrentChatModelSettings.tsx
- apps/packages/ui/src/components/Common/Settings/current-chat-model-settings-values.ts
- apps/packages/ui/src/components/Common/Settings/__tests__/current-chat-model-settings-values.test.ts
- apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts
documentation:
- 'Verification: Vitest focused cockpit/prompt/settings suite passed 23 tests. Real-server
  Playwright chat-cockpit.real-server.spec.ts passed 9/9 against localhost with no
  mocked routes. Targeted ESLint exited 0 with warnings only. git diff --check passed.
  Bandit skipped because touched code is frontend TypeScript/docs only.'
- 'Implementation notes: Mobile shell exposes active context/runtime panel metadata
  and summary copy; mobile tabs use 44px targets. Mobile composer gives the draft
  usable width by moving send controls to a second row. PromptSelect closes on Escape
  in capture phase to avoid mobile dropdown occlusion. Current Chat Model Settings
  normalizes numeric form strings and reads full form state so provider:model scoped
  settings appear in the cockpit runtime rail.'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR6 slice from the main /chat cockpit maturity roadmap. Keep scope limited to the main WebUI /chat page and improve/prove mobile cockpit behavior without touching extension/sidebar surfaces.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Mobile /chat users can switch context/runtime/focus cockpit surfaces without losing the composer draft.
- [ ] #2 Mobile /chat users can reach prompt, persona/model, context, and MCP/tool state controls from the cockpit flow.
- [ ] #3 Mobile layout preserves composer reachability, focus return, sticky status visibility, and usable tap targets.
- [ ] #4 Real-server/browser proof covers mobile context, runtime, focus/active conversation, and relevant error/degraded or blocked state handling where observable.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-16-chat-cockpit-mobile-cockpit-polish.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Mobile /chat cockpit now preserves the draft across context/runtime/focus transitions, exposes the active mobile panel with explicit copy, keeps prompt/persona/model/MCP controls reachable, avoids prompt dropdown occlusion, and has real-server Playwright proof plus focused unit/a11y coverage. Bandit is not applicable because the slice touched frontend TypeScript, tests, and documentation only.
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
