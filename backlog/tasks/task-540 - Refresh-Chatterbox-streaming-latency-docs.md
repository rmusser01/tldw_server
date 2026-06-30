---
id: TASK-540
title: Refresh Chatterbox streaming latency docs
status: Done
labels:
- tts
- chatterbox
- docs
- streaming
references:
- https://github.com/devnen/Chatterbox-TTS-Server
- Docs/Plans/2026-03-19-chatterbox-upstream-parity-implementation-plan.md
modified_files:
- Docs/STT-TTS/CHATTERBOX_SETUP.md
- Docs/Plans/2026-03-19-chatterbox-upstream-parity-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update Chatterbox setup documentation so streaming chunk guidance reflects configurable target_latency_ms / chatterbox_target_latency_ms instead of fixed 200 ms / 0.2 second chunks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Replace stale fixed 200 ms Chatterbox streaming wording in the setup runbook with configurable target_latency_ms guidance, record the docs slice in the parity plan, then verify with rg and git diff --check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated the Chatterbox setup runbook so streaming chunk guidance reflects configurable target_latency_ms / chatterbox_target_latency_ms rather than fixed 200 ms / 0.2 second chunks. Clarified that both TTS and voice-conversion streaming derive chunk duration from that config. Recorded the slice in the parity plan. Verification: rg found no stale fixed 200ms/0.2s wording in CHATTERBOX_SETUP.md; git diff --check passed. Bandit was not run because this slice only changed Markdown documentation.
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
