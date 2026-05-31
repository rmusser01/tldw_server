---
id: TASK-445
title: Implement chat character overlay and tracked identity
status: Done
labels:
- implementation
- chat
- webui
- extension
- characters
- personas
priority: high
documentation:
- Docs/superpowers/specs/2026-05-22-chat-character-overlay-and-tracked-identity-design.md
- Docs/superpowers/plans/2026-05-22-chat-character-overlay-and-tracked-identity-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the /chat tracked-vs-overlay assistant identity model from the approved design spec and implementation plan, preserving tracked character/persona chats while adding snapshot-based personality overlays for normal conversations and a side-rail control surface.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the approved tracked-vs-overlay assistant identity model across `/chat` and sidepanel surfaces. The completed slices now cover: assistant overlay settings contract and normalization, chat-scoped effective assistant state resolution, non-destructive overlay send behavior in normal chat mode, character control rail UI for the main chat surface, and sidepanel/mobile parity including scratch-tab overlay resume markers and the sidepanel character-controls sheet. The post-review hardening pass then blocked overlay writes in tracked chats, cleared overlay state before tracked-start actions, extracted a runtime-tested sidepanel character-controls sheet, and shared the sidepanel overlay-resume key helper between the form and resume detector. Final verification across the implementation stack included focused frontend vitest coverage, backend overlay settings pytest coverage, the sidepanel nextgen composer smoke spec, a direct Chromium check of the live sidepanel debug route, and Bandit reporting no findings while being unable to parse the touched TypeScript files.
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
