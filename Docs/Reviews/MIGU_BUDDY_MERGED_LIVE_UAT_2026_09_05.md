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

TASK13194 tracks conversational responses through the Buddy live session. The current `_handle_persona_live_turn` always proposes a plan; ordinary text falls through to a RAG search. A provider credential alone cannot make that path return the requested conversation. Its provider/Persona Chat integration needs an explicit design and real response/Stop/recovery acceptance.

TASK13195 tracks an actual voice-capable Buddy session. `core/Persona/live_control.py` deliberately advertises voicefalse. TASK12419's frontend gate correctly respects that capability; changing the flag would not establish microphone/STT/provider/TTS readiness. Full Persona Live and separate audio endpoints are distinct surfaces and were not certified by this Buddy probe.

No production code changed. The findings are recorded as To Do tasks rather than reopening the successfully verified cookie/feedback repairs. Existing architecture must be reviewed and any required ADR written before implementing new provider/runtime boundaries. JSON, source hashes, process identity and whitespace checks validate this evidence; Bandit is not applicable to documentation-only changes.
