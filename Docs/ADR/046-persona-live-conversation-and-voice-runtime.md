# ADR-046: Persona Live conversation and voice runtime

**Status:** Accepted
**Date:** 2026-09-05
**Backfilled from:** not backfilled
**Decision owner:** Migu Buddy UAT implementation session
**Related task:** TASK-13197, TASK-13195
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

Focused ownership, admission, readiness and cancellation regressions, then real provider and human microphone/playback UAT under TASK-13197 and TASK-13195.

## Transcript revision semantics (TASK13198)

Persona `partial_transcript` events include a current `transcript` snapshot in
addition to the legacy append-only `text_delta`. Recognizers may revise
provisional words; the browser replaces its heard text with the snapshot instead
of treating a revision as more speech. An audio chunk without a new recognition
result retains the previous snapshot. The conversation log records the committed
transcript once, while Last heard displays provisional text. Intentional manual
auto-commit settings do not signal unavailable VAD. No text deduplication is used,
so actual repeated speech remains valid.
