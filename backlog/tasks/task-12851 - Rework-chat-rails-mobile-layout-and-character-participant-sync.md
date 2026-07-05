---
id: TASK-12851
title: Rework chat rails mobile layout and character participant sync
status: Done
labels:
- bug
- webui
- extension
- layout
- character-chat
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up to the insufficient rail spacing patch. Rework the chat page desktop rail tabs so they remain attached to the left screen edge, sit lower, and use shorter controls. Redesign mobile chat chrome to preserve message/composer space and hide nonessential status/context details. Investigate and fix the character chat persist error where speaker_character_name can reference a character that is not selected as a participant.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Desktop collapsed chat/context rails are attached to the left screen edge, positioned lower, shorter than the previous tall tabs, visually separated, and clickable
- [x] #2 Mobile chat page uses available vertical/horizontal space efficiently, avoids always-visible nonessential metadata, and keeps composer/message areas usable
- [x] #3 Character chat persistence cannot send speaker_character_name for a character that is not an active selected participant
- [x] #4 Regression coverage records rail layout/mobile behavior and character participant sync behavior
- [x] #5 Rendered browser QA verifies desktop and mobile chat layouts plus the character-switch send path
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce current desktop/mobile layout and capture measurements/screenshots.
2. Trace character-chat participant state, selected assistant state, and persist request payload to identify why speaker_character_name can become stale.
3. Write failing regression tests for the participant mismatch and rail/mobile layout contracts.
4. Implement scoped layout and state fixes following existing shared component boundaries.
5. Verify with focused unit/integration tests, browser QA on desktop/mobile, and update the PR branch.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Replaced the rejected horizontal context-rail offset with fixed viewport-left positioning, a lower top clamp, and a shorter context restore tab. The chat rail trigger was also shortened and moved lower.
- Compacted the mobile chat surface by hiding the Composer and Conversation timeline chips on small screens, suppressing disabled artifact chrome, tightening transcript/composer/status padding, and keeping the mobile send control on the same row as the textarea.
- Fixed stale tracked-character submission by resolving the active server chat character before stale draft/global character selections. Persist now sends `speaker_character_id` when available, but omits `speaker_character_name` unless the client can prove the name belongs to the active chat character.
- Browser QA artifacts:
  - `/private/tmp/tldw-chat-layout-desktop-qa-fixed2.png`
  - `/private/tmp/tldw-chat-layout-mobile-qa-fixed2.png`
  - Desktop measurement: context restore tab `left=0`, `top=345.59375`, `width=32`, `height=96`.
  - Mobile measurement: Composer chip hidden, Conversation timeline chip hidden.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reworked chat rail positioning, mobile chat density, and character-chat persist identity handling. Regression tests cover the rail contracts, mobile compact chrome, and stale selected-assistant participant mismatch.
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
