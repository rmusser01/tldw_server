---
id: TASK-13202
title: Qualify an actual voice-capable Persona Buddy session
status: In Progress
created_date: 2026-09-06 02:07
updated_date: 2026-09-06 02:07
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
- [x] #1 Voice capability becomes available only when the supported end-to-end voice runtime is ready; absent credentials or unsupported backends retain actionable unavailable state.
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


Final rebased implementation 2270153980 passed 204 targeted Python and 198 frontend tests, including preparation-before-capture, stale event/session ownership and playback recovery. Real final-source preparation loaded Whisper tiny.en and Kokoro; a supplied synthetic voice_commit transcript produced a DeepSeek reply and 25388 audio bytes. No microphone opened and no playback occurred in that probe. The fresh harness initially omitted the separate TTS config, correctly yielding VOICE_TTS_UNAVAILABLE; restoring the established complete isolated config made preparation pass. Human speech, visible Buddy state and audible output acceptance remain open; this task remains In Progress and the PR remains draft.

Human browser UAT at c60f59a95e produced the expected DeepSeek reply and four Kokoro audio chunk notices; user confirmed hearing the reply. User also confirmed not saying the Thank you prefix, so exact transcript accuracy failed. The mic was already active at inspection, and speaking state was not sampled before playback finished; explicit-start/no-residual full lifecycle acceptance remains open. Source-bound local receipt: /private/tmp/migu-server-rate-retest-zd65lg4t/latest-browser-assessment.json. Non-speech filtering is investigated separately; do not mark full usability complete.

At f71d593e67 human UAT observed idle before explicit Start, preparation, listening, Send-now thinking and return to idle, followed by disconnect. User heard playback and said Reply with once, but transcript duplicated/corrupted the prefix. TASK-13201 repairs the reproduced overlapping-fragment failure with bounded whole-turn Whisper snapshots; fresh human transcript acceptance remains pending.

Manual human microphone → Whisper → DeepSeek → Kokoro path passed on 31046b8937: user clicked Send now, correct nonduplicated notebook transcript was committed once, provider reply matched, and user confirmed clear playback. Source-bound receipt is Docs/Reviews/assets/migu-buddy-browser-voice-2026-09-05/whole-turn-human-acceptance.json. Observed idle/preparing/listening/thinking/idle and disconnect; the short speaking badge was not sampled and raw MediaStream track state was not inspected. Keep broader lifecycle/state acceptance open rather than claiming those observations.

Renumbered from TASK-13195 after latest dev allocated that ID to an unrelated task; existing evidence is preserved.
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
