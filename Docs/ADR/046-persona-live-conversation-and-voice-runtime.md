# ADR-046: Persona Live conversation and voice runtime

**Status:** Accepted
**Date:** 2026-09-05
**Backfilled from:** not backfilled
**Decision owner:** Migu Buddy UAT implementation session
**Related task:** TASK-13197, TASK-13202
**Related spec/plan:** Docs/superpowers/plans/2026-09-05-persona-live-conversation-voice.md

## Decision

Use the authenticated Chat completion boundary for ordinary Live conversation, and enable voice only after explicitly preparing the selected real audio runtime for the owned connection and session.

## Context

Live currently proposes a RAG tool for ordinary greetings, advertises voice=false, and can substitute placeholder transcription after an STT failure. Successful standalone audio tests do not qualify this session path. Provider replies must retain Chat admission, budgets, moderation, provider routing and usage accounting; tool approval remains in full Live.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Call a provider adapter directly from Live | Bypasses Chat admission, moderation and accounting. |
| Invoke the FastAPI Chat handler as a function | Skips its dependency and middleware guards. |
| Duplicate Chat orchestration in Live | Creates a second policy implementation. |
| Advertise audio route presence as voice readiness | Does not establish that the selected models initialize. |
| Substitute placeholder text or silently choose another audio provider | Misrepresents failures and the selected processing destination. |

## Consequences

- The Live adapter sends a bounded non-streaming completion through the existing application's ASGI HTTP Chat route, retaining the socket's authenticated credentials and trusted client context in memory. It uses a fixed internal route, never a client-supplied URL. The ordinary HTTP middleware and dependencies revalidate every generation; no server master credential or bypass marker is introduced.
- Persona context comes from the owned server profile and bounded runtime transcript. Chat persistence is disabled for this adapter; Live retains its existing transcript and memory policy. The request cannot grant tools. Explicit search, ingestion and skill requests continue through retained plans and explicit approval.
- Slash commands are rejected before conversation admission because Chat may execute them before inference even when no tools are supplied. They must not bypass Live's retained-plan review. Text credential resolution remains at the authenticated Chat boundary; voice preflight currently qualifies server-configured credentials only, not user/team/organization BYOK credentials.
- A connection owns cancellable turn tasks. Typed turns for the same connection and session execute in FIFO order; a new send does not implicitly cancel earlier requests. Explicit Stop atomically invalidates all active and queued turn publication authority for the owned session across connections. Session closure and disconnect cancel owned work. After Stop, a subsequent send uses a fresh queue so delayed cancellation of dispatched work cannot block it. Provider cancellation is best effort when a synchronous transport has already dispatched, but late output cannot publish or restart playback.
- Voice preparation is explicitly requested before microphone capture. It initializes the selected real STT/TTS services and checks the configured conversation target, returning bounded ready/unavailable feedback tied to a client request identifier. Readiness is ephemeral and invalidates on configuration changes, Stop, failures and disconnect. It is not an approval or permission grant.
- A connection admits one voice preparation at a time. STT initialization reuses Chat's `await_bounded_owned_operation` with a 30-second deadline and its existing process-wide work and cleanup capacity. Cancellation or timeout transfers the exact transcriber to retained cleanup after the worker really exits; socket teardown does not wait for the noninterruptible model load. Retries stay busy until that cleanup completes, and stale attempts cannot advance through TTS/VAD or publish readiness. This shares the existing ownership primitive rather than introducing another worker pool or abandoning model cleanup.
- Full Live owns microphone capture and playback. Buddy exposes the exact-session navigation and truthful availability. Capture starts only after the user's Start action and successful preparation. Stop and session changes invalidate pending preparation, capture, playback and automatic resume. Voice transcript and audio envelopes retain turn correlation; unowned or late audio is ignored.
- Unsupported STT identifiers, missing models and failed transcription return actionable errors and never synthetic transcripts. Local Whisper/Kokoro with a configured conversational provider is independent of the optional OpenAI realtime path.

## Follow-up

Focused ownership, admission, readiness and cancellation regressions, then real provider and human microphone/playback UAT under TASK-13197 and TASK-13202.

## Transcript revision semantics (TASK13198)

Persona `partial_transcript` events include a current `transcript` snapshot in
addition to the legacy append-only `text_delta`. Recognizers may revise
provisional words; the browser replaces its heard text with the snapshot instead
of treating a revision as more speech. An audio chunk without a new recognition
result retains the previous snapshot. The conversation log records the committed
transcript once, while Last heard displays provisional text. Intentional manual
auto-commit settings do not signal unavailable VAD. No text deduplication is used,
so actual repeated speech remains valid.

### Browser capture rate (TASK-13199)

The default audio admission limit is 300 chunks per rolling minute, covering the shipped 4096-sample / 16 kHz browser capture cadence (about 235 callbacks including a boundary callback) with scheduling headroom. Explicit operator limits and existing chunk-size bounds remain enforced. On an owned rate-limit response the browser stops capture and playback, clears voice authority, and requires an explicit retry after the window expires. Batching was rejected for this repair because it adds transcription latency and residual-buffer lifecycle complexity.

### Whisper speech filtering (TASK-13200)

Persona enables the existing local Whisper `vad_filter` independently of its turn detector. Manual commitment disables automatic turn finalization, not recognition filtering. Silence must be filtered at the audio boundary; no phrase blacklist or textual deduplication is introduced. Local silence generated hallucinated text with filtering off, while the existing filter suppressed it and preserved the known speech sample.

### Whole-turn Whisper snapshots (TASK-13201)

Persona Whisper keeps one bounded turn and revises its full transcript. It does not concatenate five-second finalized text with a decoded overlap. The existing Whisper loader, speech filter and model selection are retained; the generic streaming endpoints and other Persona STT backends are unchanged. Reset/Stop clears the turn. The existing 30-second audio buffer bound becomes an explicit rejection before overflow, rather than silently dropping earlier audio. The browser receives an actionable shorter-turn retry message through the existing owned STT failure path. Full-buffer decoding trades additional work on long turns for coherent revisions within that bound.

Real local-model probes rejected zero overlap: it stopped some duplicated words but corrupted boundary words. Textual suffix/prefix deduplication was rejected because repeated speech is valid. Timestamp-based fragment reconciliation remains an alternative for future long-form streaming, which is outside this bounded Persona conversation path.

### Responsive Whisper ownership (TASK-13208)

Whole-turn Whisper decoding runs outside the socket event loop. Audio ingestion
returns promptly and coalesces incoming samples in the existing bounded turn
buffer; each transcriber admits one decode at a time. The partial interval starts
when inference completes, and unchanged audio is not decoded again. Completed
snapshots are delivered on subsequent audio frames, retaining the existing manual
commit contract: Send now commits the transcript currently shown.

Automatic VAD commitment freezes the exact audio boundary and waits for its
recognition snapshot, including utterances shorter than the partial-update
minimum. Later audio stays within the same bounded buffer and is replayed once
to the fresh transcriber turn and detector after automatic commitment. Manual
commitment, Stop and session changes discard that carry. A VAD event cannot
commit an older partial or combine speech from a later turn.

Recognition reuses Chat's bounded task and owned-operation helpers, including a
30-second deadline and reserved cleanup capacity. Reset invalidates publication
authority without cancelling native inference. Stop/disconnect retire the
transcriber immediately, but release its model only after its worker exits. A
timeout or cancelled supervisor retains that ownership through the existing late
cleanup path. Capacity rejection starts no decoder and reaches the existing STT
failure response. No additional executor or unbounded per-frame task queue is
introduced.

Awaiting a thread directly in audio ingestion was rejected: it frees the event
loop but still holds the socket receive loop, delaying Stop. Cancelling a thread
future and immediately clearing its model was rejected because native inference
may still be using it. Other speech backends and model selection are unchanged.
