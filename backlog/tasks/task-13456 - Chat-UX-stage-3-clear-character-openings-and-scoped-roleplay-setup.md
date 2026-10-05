---
id: TASK-13456
title: 'Chat UX stage 3: clear character openings and scoped roleplay setup'
status: To Do
labels:
- chat
- ux
- webui
- extension
- nng
dependencies:
- TASK-13454
- TASK-13455
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve findings 4, 6, and 7 and approved roleplay enhancements: prominent greeting, familiar identity/context terminology, predictable editing scope, reusable setups, and truthful next-turn context preview. Keep unsupported native context extensions tracked against the existing server contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A selected card replaces generic onboarding with its character introduction and visible greeting; Start with this greeting persists at most once through the valid owner-bound path, and users can instead type the first turn.
- [ ] #2 AI character, Your role, Scene, Inherited lore, and Additional pinned context are distinguished with concise contextual help; zero pinned sources never means no inherited character context.
- [ ] #3 Every editor identifies This conversation, Character library, or Provider-model default; staged editors support Apply/Cancel and dirty state, immediate controls state their timing, and library/default writes require their explicit existing action.
- [ ] #4 Existing saved roleplay setups can be previewed, applied, modified, and reused without changing library cards or global model defaults implicitly.
- [ ] #5 Next-turn summary distinguishes selected, supported/planned, server-acknowledged, excluded, and unknown context; it does not claim lore injection or custom prompt/scene support from UI selection or route presence.
- [ ] #6 Imported PNG/JSON card, inherited worldbook/dictionary, custom scene/prompt/context, cancel/retry, and reopened-session restoration checks pass for supported contracts; unmet native resume support remains open under TASK-13160 and its consuming enhancement cannot be marked complete.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Program: TASK-13453. Design: Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md. Base: origin/dev 75ab224081bf140ef52017c1a9b0a04f6878d488, verified 2026-10-04. User approved equal audience priority and all ten findings plus enhancements in five stages. Implementation waits for review of the written design. ADR required: no for the client remediation; ADR-049 and ADR-050 govern owner-bound history and native fork boundaries, ADR-059 governs canonical task editing. Reassess if implementation changes public APIs or persistence authority. Reuse existing shared UI, loaders, presets, and test harnesses; preserve temporary-chat privacy and account/workspace fences. Findings 4, 6, 7. Reuse RolePlaySetupDrawer/Preview, ChatGreetingPicker, useChatSettingsRecord, saved-role-play-setups and compatibility helpers. TASK-13160 is the existing conditional prerequisite for snapshot-based native resume; this task consumes it instead of inventing a competing prompt protocol. Existing native character path rejects client scene/actor/custom context; expose that honestly until authoritative support is delivered.
ADR detail: backlog/decisions/002-character-conversation-behavior-snapshot-and-fenced-completion.md governs snapshot-authoritative native resume and supported versioned settings mutations. TASK-13160 is a prerequisite, not a promise that arbitrary client scene/prompt overlays become supported. Never inject client overrides or fall back to less guarded completion; assess any missing durable API before extension and retain its consuming enhancement as open.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
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
