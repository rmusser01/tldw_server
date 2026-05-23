---
id: TASK-444
title: Design /chat character overlay and tracked identity rail model
status: Done
labels:
- design
- chat
- webui
- extension
- characters
- personas
priority: high
documentation:
- Docs/superpowers/specs/2026-05-22-chat-character-overlay-and-tracked-identity-design.md
- Docs/superpowers/plans/2026-05-22-chat-character-overlay-and-tracked-identity-implementation-plan.md
modified_files:
- Docs/superpowers/specs/2026-05-22-chat-character-overlay-and-tracked-identity-design.md
- Docs/superpowers/plans/2026-05-22-chat-character-overlay-and-tracked-identity-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write a design spec for /chat that preserves tracked character/persona chats while adding non-destructive character/persona personality overlays for normal conversations, with a side-rail control surface and no thread resets.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the /chat character overlay and tracked identity design slice. The approved spec and implementation plan define tracked character/persona chat as durable identity state, normal-chat overlays as non-destructive snapshot state, and the side-rail/sidepanel control model used by the merged implementation PR #1956. Bandit was not applicable to the design slice because it only produced Markdown planning artifacts.
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
