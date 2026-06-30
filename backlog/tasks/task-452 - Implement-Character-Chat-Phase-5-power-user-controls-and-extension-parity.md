---
id: TASK-452
title: Implement Character Chat Phase 5 power-user controls and extension parity
status: Done
references:
- TASK-426
- TASK-431
- TASK-438
- TASK-447
- TASK-449
documentation:
- Docs/Product/WebUI/Character_Chat_Roleplay_First_Class_PRD_2026_05_18.md
modified_files:
- Docs/superpowers/plans/2026-05-19-character-chat-phase5-power-user-extension-parity-plan.md
- apps/packages/ui/src/utils/sidepanel-full-app-route.ts
- apps/packages/ui/src/utils/__tests__/sidepanel-full-app-route.test.ts
- apps/packages/ui/src/components/Sidepanel/Chat/ControlRow.tsx
- apps/packages/ui/src/components/Sidepanel/Chat/CharacterSelect.tsx
- apps/packages/ui/src/components/Sidepanel/Chat/ConversationContextPopover.tsx
- apps/packages/ui/src/components/Sidepanel/Chat/__tests__/ControlRow.role-play-handoff.contract.test.ts
- apps/packages/ui/src/components/Sidepanel/Chat/__tests__/ConversationContextPopover.role-play-open.contract.test.ts
- apps/packages/ui/src/routes/sidepanel-chat.tsx
- apps/packages/ui/src/routes/__tests__/sidepanel-chat.character-chat-command.guard.test.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next first-class Character Chat PRD slice from Docs/Product/WebUI/Character_Chat_Roleplay_First_Class_PRD_2026_05_18.md. Scope starts with power-user and extension-parity improvements that keep character role-play fast and visible across WebUI /chat and the browser extension sidepanel.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Power user can switch character, apply or resume a role-play setup, and recover last Character Chat context without opening generic settings.
- [x] #2 Extension sidepanel and WebUI use the same visible labels for Character, Persona, Scene, and Character Chat where the sidepanel exposes role-play state.
- [x] #3 Sidepanel tab/open-full-app handoff preserves selected character/persona intent into /chat.
- [x] #4 Focused frontend tests cover role-play state handoff and sidepanel visible state; browser verification uses the real backend when possible.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Phase 5 sidepanel role-play parity: added shared sidepanel-to-/chat handoff routing, visible Character Chat chip with Character/Persona labeling, switch and clear actions, context-popover-assisted assistant picker opening, robust modern+legacy role-play storage clearing, and a command-palette full-app handoff action. Verified with focused frontend tests plus browser checks against the real backend: sidepanel chip renders, switch opens the mounted picker, clear removes selectedAssistant and selectedCharacter, menu and command-palette handoffs open /chat?mode=character&characterId=default-assistant in WebUI while preserving options.html# behavior for extension runtimes. Bandit not run because touched code is frontend TypeScript/docs only.
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
