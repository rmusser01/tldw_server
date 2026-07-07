---
id: TASK-12164
title: 'Follow up: unify character emotes with Persona Visual runtime states'
status: To Do
labels:
- frontend
- persona-visuals
- follow-up
- emotes
priority: Medium
references:
- TASK-12163
documentation:
- Docs/superpowers/specs/2026-07-06-character-chat-streaming-emote-directives-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up after v1 character emote directives: evaluate and design a shared visual-state integration so character portraits, PersonaBuddy, and future agentic UIs can use a common set_emote/set_visual_state style runtime path instead of character-chat-only directive handling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Review existing persona visual runtime state handling and character mood image resolution for a shared boundary.
- [ ] #2 Define how character emote states map to Persona Visual runtime/custom states without breaking existing character chat behavior.
- [ ] #3 Define whether a future agentic UI tool should be set_emote or set_visual_state and how it coexists with text directives.
- [ ] #4 Produce a focused design/plan before implementation.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

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
