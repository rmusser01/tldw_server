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

No human microphone was opened during the TASK-13208 synthetic qualification; later physical attempts are recorded below. Final physical acceptance is
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

## Parakeet chunk artifact repair — TASK-13210

On `0837a8df23`, the requester reported saying “Say the blue notebook is ready”
once. The browser's Last heard and Last sent both contained “Right. The blue
notebook is ready. Ready. Yeah.” The same live session recorded a committed user
turn, an assistant reply, and four TTS chunks before returning to idle. This
establishes submission and generated reply, not transcription accuracy or a new
human confirmation of audible playback. Earlier cancelled attempts were explained
by clicking Stop voice before Send now; Stop cancels rather than submitting.

A local paced ONNX probe reproduced the underlying assembly failure: an early
final “Reps.” remained in history even after a later decode recognized the whole
phrase correctly. The production-factory regression failed with “Right. Say the
blue notebook is ready.” instead of the corrected whole phrase. The repair shares
Persona's existing bounded whole-turn scheduler with an ONNX backend adapter.
Late revisions replace earlier text; legitimate repetition and empty corrections
remain intact. Provider/authentication settings are unchanged.

The final focused checks passed 72 voice tests, covering both model adapters and
real TestClient sockets for Stop/disconnect and exact VAD finalization. The prior
regression run passed 168 tests, including 128 other Persona/WebSocket/Parakeet
cases; its overlapping 40 voice tests were rerun with three additional ONNX cases.
Ruff/Black passed seven changed Python files and Bandit found zero production
issues. Independent review found no actionable regression.

The real ONNX test used local Kokoro speech, 3 seconds of leading silence,
3 seconds of trailing silence, and browser-paced 4096-sample callbacks at 16 kHz.
It ended with exactly the words “Say the Blue Notebook is ready.” The maximum
observed ingestion call took 0.076 ms. Digital silence generated no words. A
second sample preserved the deliberate words “Say ready ready,” allowing normal
punctuation. These probes used no microphone or external chat provider.

Intermediate ONNX hypotheses still included wrong words before later revisions
corrected them. This fix removes stale chunk accumulation; it does not certify
perfect speech recognition. Physical acceptance, including floating Buddy states
and audible completion, remains open under TASK-13202.

Evidence: [local phrase result](assets/migu-parakeet-turn-2026-09-06/phrase-result.json),
[repetition result](assets/migu-parakeet-turn-2026-09-06/repetition-result.json),
[phrase harness](assets/migu-parakeet-turn-2026-09-06/phrase-probe.txt), and
[human receipt](assets/migu-parakeet-turn-2026-09-06/human-attempt.json).

### Human retry on cdf099fff7

The requester confirmed saying “Say the blue notebook is ready.” Browser Last
heard and Last sent contained “See the blue notebook is ready.” once. Persona
replied “I see the blue notebook is ready.” Four TTS chunks arrived and Live
returned to idle. The requester confirmed clear audible playback and that
recording/playback stopped afterward. The running backend's source hashes matched
the tested code.

Repeated fragments did not recur in this attempt. Exact transcription remains
unqualified because “Say” was recognized as “See.” No raw human audio was retained,
so this receipt alone cannot distinguish an acoustic/model error from submission
timing. Floating Buddy intermediate states were not observed. TASK-13202 remains
open. [Source-bound human receipt](assets/migu-parakeet-turn-2026-09-06/human-retry.json).

## Visual follow-up and PR verification

A later user-controlled turn on backend `cdf099fff7` recognized “Say the blue
notebook is ready.” once, received “The blue notebook is ready.” and four TTS
chunks, then returned to idle. The requester reported “Done; the reply finished.”
This successful phrase does not erase the preceding Say/See accuracy failure.

The floating visual check did **not** pass. Between 16:47:07 and 16:47:15 UTC,
the loaded idle image disappeared with “Visual pack did not load — rate_limited.”
Backend access metadata shows visual-pack list/detail and live-session list
requests roughly every 250 ms before HTTP 429. Changing blob URLs alone is not
evidence of a refetch: normal animation can cycle cached frames. The repeated
API calls establish a separate load problem, but its initiating trigger remains
unproven. Source inspection and independent review did not justify a lifecycle
repair. TASK-13211 records the investigation; TASK-13202 AC3 remains open.

After rebasing onto `dev` at `c5b777e9ba`, HMR reloaded the frontend. Reconnecting
restored the loaded idle image. Repeated screenshot sampling and one real text
provider reply did not reproduce the request loop. No new human voice sequence
was observed during that follow-up, so listening/thinking/speaking/idle transitions
remain unqualified. The running backend retained its recorded pre-rebase revision;
rebasing did not alter the voice implementation.

Fresh post-rebase verification passed **189 targeted Python tests** and **129
frontend tests** (BuddyShellHost and Persona route). Ruff passed the nine scoped
transcriber/runtime test files; Bandit found zero issues across the six changed
production Python files. No full test sweep was run. Previously linked sanitized
JSON receipts are explicitly staged despite the repository-wide JSON ignore rule.

[Sanitized visual observations and request metadata](assets/migu-parakeet-turn-2026-09-06/visual-follow-up.json).
