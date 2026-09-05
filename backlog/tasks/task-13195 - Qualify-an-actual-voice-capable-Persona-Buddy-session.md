---
id: TASK-13195
title: Qualify an actual voice-capable Persona Buddy session
status: In Progress
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
updated_date: 2026-09-05 22:51
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: Docs/ADR/046-persona-live-conversation-and-voice-runtime.md
Reason: Voice readiness and microphone/playback lifecycle form a frontend/backend contract.
1. Specify explicit voice preparation and scoped readiness using real STT/TTS and the conversational provider.
2. Add regressions for model/language normalization, missing runtime, no placeholder transcription and Stop/late-event ownership.
3. Implement backend preparation and frontend explicitly initiated capture with correlated audio turns.
4. Run targeted checks and coordinate human speech/playback UAT; record exact revision and outstanding credential-specific acceptance.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Real isolated backend probe (2026-09-05): Whisper tiny.en and local Kokoro initialized successfully in 7.69s. A second prepared session showed capabilities.voice false→true; REST Stop returned 200, sent SESSION_TERMINAL for the exact connected session, and changed voice capability back to false. No microphone was opened and no conversational provider request was sent. Initial unready result was traced to scratch config default_api=openai taking precedence over DEFAULT_LLM_PROVIDER; corrected only the isolated config. Receipts are in /private/tmp/migu-server-voice-runset. Authentication token-shadowing found during review is being corrected before final-source UAT. Human voice and integrated conversational provider acceptance remain open.
Review fixes now preserve the original WebSocket credential during preparation, make direct voice_stop cancel active/queued owned turns, and bound preparation to one active initialization per connection. Pending-token checks fence expensive STT/TTS/VAD stages. STT initialization reuses Chat.streaming_utils.await_bounded_owned_operation with a 30-second deadline and reserved process-wide work/cleanup capacity. Cancellation or timeout transfers exact transcriber cleanup ownership; socket teardown does not wait for the thread, and the connection remains busy until detached cleanup finishes. Four new socket regressions failed before and passed after; voice plus existing bounded-owner scope: 36 passed, 13 deselected. ADR046 and plan describe the ownership decision. Frontend recovery/playback/cancellation/wake correlation regressions: 71 passed, ESLint zero errors, zero owned TypeScript diagnostics (74 existing dependency diagnostics). Real final-source human voice acceptance remains pending.
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
