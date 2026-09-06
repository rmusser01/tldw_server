# Migu Buddy live acceptance after merge — 2026-09-05

PR2902 merged into dev as `84b6928dcfc48fdd7b424939a9ba52a82c37612c`. This real REST/WebSocket probe ran the full FastAPI app from later merged dev `220bf544b7110b0ee37fdf99a046e8a03d7ba868`, using the existing isolated synthetic UAT account and a new session. No browser mock, provider reply stub, tool approval, or microphone input was used.

## Results

- Create session200; WebSocket authenticated and connected.
- Input: `Hello Migu. Reply with: Migu buddy UAT ready.`
- Received a correlated `tool_plan` for `rag_search`, not a conversational response. The input did not request a knowledge-base search. No tool was approved or executed.
- The actual session capability is `text=true`, `voice=false`, `browser_microphone_required=false`.
- Stop session200. The probe closed its WebSocket, then the owned server received SIGTERM and its wrapper reaped it with returncode-15. This is a terminated-process receipt, not a claim of exit0.

[Assessment](assets/migu-buddy-merged-live-2026-09-05/assessment.json), [raw frames](assets/migu-buddy-merged-live-2026-09-05/direct-stream-frames.json), [session capability](assets/migu-buddy-merged-live-2026-09-05/live-session.json), [launch](assets/migu-buddy-merged-live-2026-09-05/backend-current-identity.json), [exit](assets/migu-buddy-merged-live-2026-09-05/backend-current-exit.json), [source/cleanup verification](assets/migu-buddy-merged-live-2026-09-05/verification.json).

The probe's first summary read a `type` field and top-level capabilities; inspection of the raw response showed the server uses `event` and `session.capabilities`. The published assessment is derived from those actual envelopes. Raw frames were preserved. This parser correction does not alter the observed tool plan or unavailable voice capability.

## Acceptance gaps

TASK13197 tracks conversational responses through the Buddy live session. The current `_handle_persona_live_turn` always proposes a plan; ordinary text falls through to a RAG search. A provider credential alone cannot make that path return the requested conversation. Its provider/Persona Chat integration needs an explicit design and real response/Stop/recovery acceptance.

TASK13195 tracks an actual voice-capable Buddy session. `core/Persona/live_control.py` deliberately advertises voicefalse. TASK12419's frontend gate correctly respects that capability; changing the flag would not establish microphone/STT/provider/TTS readiness. Full Persona Live and separate audio endpoints are distinct surfaces and were not certified by this Buddy probe.

No production code changed. The findings are recorded as To Do tasks rather than reopening the successfully verified cookie/feedback repairs. Existing architecture must be reviewed and any required ADR written before implementing new provider/runtime boundaries. JSON, source hashes, process identity and whitespace checks validate this evidence; Bandit is not applicable to documentation-only changes.

## Follow-up implementation: real readiness probe

The later implementation under TASK13197/13195 follows ADR046. On an uncommitted
working tree over `c1467cddd8`, a real server initialized local Whisper `tiny.en`
and Kokoro in 7.69 seconds. A subsequent preparation reused the loaded models.
The [readiness and Stop receipt](assets/migu-buddy-voice-readiness-2026-09-05/readiness-result.json)
shows session voice capability changing false → true, REST Stop returning 200,
an exact-session `SESSION_TERMINAL` event, and capability returning to false.
No microphone was opened and no provider message was sent.

The first successful [preparation-only probe](assets/migu-buddy-voice-readiness-2026-09-05/prepare-only-result.json)
queried an unsupported individual Live detail URL and recorded 404. The corrected
[harness](assets/migu-buddy-voice-readiness-2026-09-05/prepare-probe.txt) uses the
existing session list and matches the exact session identity. This was a probe
error, not a readiness failure.

The scratch config's `default_api=openai` initially took precedence over the
DeepSeek environment default. Only that isolated config was corrected to select
DeepSeek; the normal Chatbook config stayed unchanged. The owned server process
25966 was stopped and reaped with return code -15, as recorded in its
[exit receipt](assets/migu-buddy-voice-readiness-2026-09-05/backend-exit.json).

This is intermediate runtime evidence, not final-source acceptance. The launch
receipt hashes the tracked diff but does not inventory then-untracked helpers.
Subsequent review found authentication and preparation ownership issues that need
fresh verification. Integrated provider responses and intentional human voice
acceptance remain open; the server conversation callsite also awaits the user's
answer about forwarding bounded Persona context to the configured provider.

## Approved conversational integration

The user explicitly authorized the Persona profile/system prompt, enabled memory,
state, companion/exemplar context and session history to the configured Chat
provider, including DeepSeek for this UAT. The callsite is now implemented through
the full authenticated Chat HTTP boundary. It retains explicit Live tool review,
rejects slash commands before Chat's command preprocessor, and rejects empty sends
that could otherwise expose an older command after history filtering.

The [real provider assessment](assets/migu-buddy-conversation-2026-09-05/assessment.json)
records the exact expected reply in 1.11 seconds, cancellation with no cancelled
answer published, and a successful same-session recovery reply. A prepared
`voice_commit` with an explicitly supplied synthetic transcript returned 25388
bytes of real Kokoro audio. An explicit search still produced `rag_search` for
review; no tool was approved. Session Stop returned 200. The
[harness](assets/migu-buddy-conversation-2026-09-05/probe.txt) and
[source identity](assets/migu-buddy-conversation-2026-09-05/source-identity.json)
identify the tested working tree. No microphone or speaker playback was used.

The first probe expected TTS after `user_message`; the established protocol emits
speech for `voice_commit`. Its timeout was corrected in the harness. The successful
run proves provider and TTS integration, not transcription or human playback.

Real browser UAT at `/persona` completed synthetic Migu setup using the live reply
“Migu browser UAT ready.” The browser saved local Whisper `tiny.en`, language `en`,
Kokoro via `tldw`, voice `af_heart`, manual commit, no auto-resume and approval for
every tool action. No microphone was started while waiting for human readiness.
Some optional visual capability requests reported `Failed to fetch` despite
backend 200 responses; shared URL/auth and request-dedup review did not establish
an application cause. This browser observation remains an acceptance limitation.

Validation: 201 integrated Persona tests passed after updating obsolete greeting
fixtures to test actual conversational prompts or explicit planning intent.
The subsequent empty-send command regression failed before the guard and passed
after it; conversation/helper scope totals 23 passing tests. Frontend terminal
failure and Buddy diagnostics scope passed 101 tests. Human microphone/listening
state and playback acceptance remain open.

## Saved setup regression and browser verification

The first reload/reselection overwrote completed Migu setup and forced the Voice
stage again. TASK13196 fixes the selection handler to read and preserve saved
progress, including failing without writes when the profile cannot be read.
After the fix, a real DeepSeek reply “Migu setup preserved.” completed setup.
Reloading `/persona` and choosing Migu UAT opened the normal Live Session workspace
without another setup flow. Whisper `en`/`tiny.en`, Kokoro `tldw`/`af_heart`,
manual commit and auto-resume off were preserved. All 90 targeted route tests passed.
This verifies setup persistence; it does not claim microphone acceptance.

## Rebased verification

Implementation revision `2270153980c95f17af0b3bc85eb041ef62750387` is rebased on
`dev` `f6d6a673b628c77a7e262d7638c658782906aef0`. The six targeted Persona Python
modules passed all 204 tests (94.39 seconds). The nine frontend voice, microphone,
Buddy, diagnostics and route modules passed all 198 tests (28.35 seconds).
OpenAPI fingerprint verification passed. Touched production Bandit found zero
issues; new Python helpers/tests passed Ruff. Frontend lint has zero errors,
with existing warnings; scoped TypeScript has zero owned errors and 27 dependency
diagnostics. No full-suite result is claimed.

The fresh final UAT harness initially copied only `config.txt`, omitting the
separate TTS configuration. DeepSeek reply/Stop/retry passed but voice preparation
correctly returned `VOICE_TTS_UNAVAILABLE`. The harness was corrected to use the
same complete isolated configuration as the previous successful run; no product
code or normal user configuration was changed to address that harness failure.

The corrected [rebased probe](assets/migu-buddy-conversation-2026-09-05/rebased-assessment.json)
returned the exact expected DeepSeek answer in 0.77 seconds, suppressed the cancelled
answer, and recovered in the same session. Real Whisper/Kokoro preparation passed;
a synthetic `voice_commit` transcript produced 25388 speech bytes. Explicit search
still produced an unapproved `rag_search` plan, and final Stop returned 200.
The [rebased source receipt](assets/migu-buddy-conversation-2026-09-05/rebased-source-identity.json)
identifies revision and source hashes. Microphone and playback were not used.

## Human browser voice UAT and transcript repair

On revision `9ffa9a272e`, the user intentionally spoke once through Persona Live.
Whisper recognized speech, DeepSeek replied, and Kokoro played audio that the user
confirmed hearing clearly. The browser showed preparing, listening, thinking,
speaking and idle. Auto-resume stayed off; Live disconnected after playback.
The [assessment](assets/migu-buddy-browser-voice-2026-09-05/assessment.json) and
[backend identity](assets/migu-buddy-browser-voice-2026-09-05/backend-identity.json)
record sanitized observations and unchanged normal configuration. Server shutdown
waited for a remaining connection after SIGTERM; SIGINT ended the owned process.

Full acceptance failed: one phrase became repeated transcription fragments and
affected the provider reply; deliberate manual mode was also reported as a VAD
failure. TASK13198 repairs snapshot rollback between recognizer updates, browser
append-only handling of revisions, and the false warning. Last heard uses
replacement snapshots and the conversation log records one committed utterance.
Actual repeated words remain intact. Two backend and three frontend regressions
failed before the fix. Final scopes passed 125 Python and 165 frontend tests.
Touched Bandit found zero issues, Ruff added no findings, ESLint had zero errors
and 12 existing warnings, and scoped TypeScript had no owned diagnostics
(27 dependency diagnostics remain). The first full UI typecheck exceeded Node's
heap; the scoped check completed. Post-fix human acceptance remains pending.

### Corrected browser retest: transcript observed, recording canceled

On revision `71843993589539f8dcf0bc7e95175c2c5274c1f9` (see source identity receipt for authoritative revision), manual listening showed the corrected manual-mode status without a VAD failure. Last heard contained the notebook phrase once, with a misrecognized prefix and repeated punctuation. This is provisional STT evidence, not exact transcript acceptance. The operator allowed capture to outlast the intended 20-second window during context recovery, then canceled via Stop voice and disconnected. No provider turn or audible reply was qualified by this run.

The UI accumulated audio-rate warnings: browser capture emits 4096 samples at 16 kHz (about 234 chunks per minute), exceeding the old default of 120. TASK-13199 raises the bounded default to 300, preserves explicit lower settings, and stops browser capture with a retry message when throttled. Corrected short-window human acceptance remains pending; do not interpret the operator overrun as an intentional long-recording UAT.

TASK-13199 verification: 126 backend and 72 frontend tests passed; Bandit zero findings; Ruff four unchanged baseline findings; ESLint no rule findings; scoped TypeScript zero owned diagnostics with 27 existing dependency diagnostics.

### Audible reply confirmed; unspoken prefix isolated (TASK-13200)

On c60f59a95e, the browser submitted the notebook test phrase once and received the expected DeepSeek reply plus four Kokoro audio chunks. The user confirmed audible playback and denied speaking the preexisting gratitude prefix. The microphone was already active when the operator inspected this turn, so this run does not qualify explicit-start ownership. After Send now the UI moved through thinking to idle; the short speaking interval was not sampled. The operator disconnected after playback.

A local real-model comparison reproduced hallucinated text from five seconds of digital silence with Whisper filtering off. The existing faster-whisper audio filter suppressed the silence result and preserved synthetic Kokoro speech. Persona now enables that filter for Whisper independently of automatic turn commitment; there is no phrase blacklist. A production-config probe confirmed empty silence output and correct speech with the filter enabled. These synthetic probes use no microphone or external provider; fresh human transcript acceptance remains pending. See the focused JSON receipts in the browser-voice assets directory.

TASK-13200 validation: 129 focused Persona/Whisper tests passed; Bandit zero findings; Ruff no new findings (one unchanged endpoint SIM114). The two new regressions first failed with filtering disabled.

### Whole-turn recognition repair (TASK-13201)

Human UAT on f71d593e67 confirmed audible playback, no gratitude prefix, and only one spoken Reply with. The recognized transcript nevertheless repeated/corrupted that prefix. This run observed idle before explicit Start, preparing, listening, thinking and idle; the short speaking phase was not sampled. Send now ended capture and the operator disconnected after playback. Transcript accuracy remained failed.

Real local Kokoro/Whisper boundary probes reproduced the failure at 3.8 and 4.3 seconds of leading silence: generic Whisper finalizes five-second fragments and independently decodes the overlap. Removing overlap still corrupted words. Persona now uses a specialized local Whisper transcriber that replaces one whole-turn snapshot, including empty revisions. Its existing 30-second audio bound rejects overflow explicitly, and reset/cleanup clears prior audio. Other streaming endpoints and STT backends retain their behavior. All four synthetic boundary cases retained the complete phrase after the repair. The corpus uses simulated callback timing with real model inference; it is not a human microphone acceptance run. Fresh human validation remains pending.

TASK-13201 validation: 134 focused tests passed; Bandit zero findings; Ruff one unchanged endpoint SIM114; new-file Black checks passed. Three regressions failed before the change, and all four real local-model boundary cases passed after it.
