---
id: TASK-13195
title: Qualify an actual voice-capable Persona Buddy session
status: To Do
created_date: 2026-09-05 21:29
labels:
- persona
- buddy
- voice
- uat
priority: high
references:
- Docs/Reviews/MIGU_BUDDY_MERGED_LIVE_UAT_2026_09_05.md
- TASK-12419
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Post-merge Migu UAT on dev 220bf544b7 confirms the real live-control session advertises voice=false. TASK-12419 correctly hides Buddy voice controls for that capability; this is an unimplemented acceptance path, not permission to flip the flag. Define and verify the supported microphone/STT/provider/TTS lifecycle and truthful readiness before enabling Buddy voice.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Voice capability becomes available only when the supported end-to-end voice runtime is ready; absent credentials or unsupported backends retain actionable unavailable state.
- [ ] #2 Microphone startup is explicitly user initiated and Stop/session changes release recording and playback ownership.
- [ ] #3 An intentional human speech test produces a provider reply and actual audio output with request-correlated Buddy listening/thinking/speaking/idle states.
- [ ] #4 Targeted lifecycle regressions and sanitized live evidence verify no audio capture before the explicit start and no residual capture after Stop.
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
