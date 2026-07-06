---
id: TASK-12163
title: Add explicit streaming emote directives for character chat portraits
status: To Do
assignee: []
created_date: ''
updated_date: 2026-07-06 16:36
labels:
- frontend
- character-chat
- emotes
dependencies: []
documentation:
- Docs/superpowers/specs/2026-07-06-character-chat-streaming-emote-directives-design.md
- Docs/superpowers/plans/2026-07-06-character-chat-streaming-emote-directives-implementation-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement v1 character chat emote control: parse standalone Emote: <state> directives from assistant responses, strip them from visible/stored text, update character portraits live during streaming, persist final mood_label plus emote_events metadata, and demote heuristic mood detection to fallback when explicit directives exist.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Streaming character chat can change the character portrait multiple times within one assistant response when valid Emote directives arrive.
- [ ] #2 Raw Emote directive lines never appear in rendered chat or persisted assistant content, including partial/chunked streaming cases.
- [ ] #3 Explicit emote directives override heuristic mood detection; detectCharacterMood only runs when no valid directive is present.
- [ ] #4 Non-streaming character responses are also parsed and stripped before display/persist.
- [ ] #5 Invalid, unsafe, duplicate consecutive, or over-cap directives are stripped but do not fire/store emote events.
- [ ] #6 Missing emote image assets do not break rendering; the UI keeps the current/base portrait.
- [ ] #7 Final emote, defined as the last accepted event, persists as mood_label and optional emote_events are stored in metadata_extra.
- [ ] #8 History reload restores the final emote and does not replay beat events in v1.
- [ ] #9 Parser, streaming-buffer, integration, and minimal UI behavior tests cover the directive flow.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-06-character-chat-streaming-emote-directives-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->

<!-- SECTION:NOTES:END -->

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
