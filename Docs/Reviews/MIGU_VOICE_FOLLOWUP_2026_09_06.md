# Migu voice follow-up — 2026-09-06

TASK-13208 repairs the synchronous Whisper decode deferred from merged PR2908.
The branch starts at dev `3b82e4c8d6`. TASK-13202 still owns physical microphone,
audible reply and floating-Buddy state acceptance.

## Repair and verification

Whisper uses one bounded background decoder and retains the model until native
work exits. Incoming audio is coalesced within the existing 30-second bound;
the update interval begins after completed inference. Reset/Stop invalidate late
results, and same-connection preparation waits for retired recognition cleanup.
The original blocked-ingestion regression failed before the repair.

Independent review caught a VAD finalization race: an automatic turn could commit
an old partial while its decoder was running. The correction freezes the exact
audio boundary, waits for that snapshot and carries later audio into the next
turn and detector once. A real TestClient WebSocket regression covers the blocked
decoder, original correlation ID, full first transcript and next-turn audio.
Manual Send now retains its displayed-transcript semantics. Follow-up independent
review reported no actionable findings.

- 165 targeted Python tests passed across Whisper, voice runtime, Persona
  WebSocket, WebSocket authentication and conversation WebSocket modules.
- Changed transcriber/tests pass Ruff and Black. Endpoint Ruff reports only the
  unchanged SIM114 at its preexisting line 2472.
- Bandit reports zero findings on the changed production Python scope.
- No full repository test sweep was run.

The [final local-model receipt](assets/migu-voice-followup-2026-09-06/responsiveness-result.json)
binds to the transcriber's SHA-256. Real local Kokoro speech with 3.8 seconds of
leading silence was streamed to Whisper tiny.en CPU/int8. The final text was
“Reply with the Blue Notebook is ready.” Across 32 callbacks and 14 decodes,
maximum ingestion time was 0.154 ms and the maximum 10 ms heartbeat gap was
18.294 ms. Every subsequent decode began at least 350 ms after prior inference
finished. This is a synthetic responsiveness probe, not physical voice acceptance.
Its [harness](assets/migu-voice-followup-2026-09-06/probe-responsiveness.txt) uses
the actual transcriber and local speech route, with no microphone or external
conversation-provider request.

## Normal configuration selected for physical UAT

The requester chose normal server speech settings. The normal server config and
isolated UAT config both select Parakeet ONNX on CPU with
`istupakov/parakeet-tdt-0.6b-v3-onnx`. The synthetic Migu UAT Persona's saved STT
selection is now `parakeet-onnx`; the browser Live Session confirms that value.
Normal user configuration files were not edited.

The [server preparation receipt](assets/migu-voice-followup-2026-09-06/parakeet-readiness-result.json)
shows ready, Stop, terminal notice and revoked readiness. Parakeet initialization
selects a lazy decoder, so readiness alone does not establish actual model loading.
A separate [real recognition probe](assets/migu-voice-followup-2026-09-06/parakeet-speech-result.json)
loaded the cached ONNX model and recognized the complete synthetic notebook
phrase. Its first decoding callback took 1832 ms; later decoding callbacks were
approximately 57–169 ms. This does not qualify Parakeet Stop responsiveness under
slow native inference. These Parakeet probes preceded the final Whisper VAD
correction and exercised unchanged Parakeet code.

No human microphone was opened in this follow-up. Final physical acceptance is
pending; earlier Whisper/Kokoro human confirmations remain historical evidence
on their recorded source versions.

Architecture: [ADR046](../ADR/046-persona-live-conversation-and-voice-runtime.md).
Plan: [TASK-13208 plan](../superpowers/plans/2026-09-06-persona-whisper-responsiveness.md).

## Physical attempt and silence repair — TASK-13209

On `f696121698`, the first Parakeet recording captured unrelated narration and
was stopped without a provider submission. The next recording contained silence;
the requester confirmed they missed the recording window. The floating 96-pixel
asset was loaded and visibly followed listening → idle after Stop in both
attempts. Thinking, speaking and audible reply were not tested.

The second attempt rendered forty `[No speech detected]` markers as recognized
words. The ONNX file backend uses that exact status for empty recognition; its
streaming adapter now converts the status to an empty result before partial,
final or flush history handling. Ordinary spoken words such as “no speech
detected” are preserved. No phrase deduplication or general text blacklist was
added, and the file API remains unchanged.

Three new sentinel regressions failed before the repair. All 57 focused
Parakeet/core-streaming/Persona voice tests passed afterward; Ruff and Black pass
on the changed files, Bandit found no production-scope issues, and independent
review reported no actionable findings. The real local Parakeet ONNX probe
emitted no transcript frames or history for digital silence, then recognized the
complete synthetic notebook phrase. No microphone was used in that repair probe.

Sanitized [human-attempt receipt](assets/migu-parakeet-silence-2026-09-06/human-attempts.json),
[local-model result](assets/migu-parakeet-silence-2026-09-06/result.json), and
[probe harness](assets/migu-parakeet-silence-2026-09-06/probe.txt).
The next physical attempt will let the requester start recording directly to
avoid another missed timed window. TASK-13202 remains open.
