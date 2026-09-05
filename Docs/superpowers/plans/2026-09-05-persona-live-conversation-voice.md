# Persona Live conversation and voice implementation

Tasks: TASK-13197 and TASK-13195.

ADR required: yes
ADR path: Docs/ADR/046-persona-live-conversation-and-voice-runtime.md
Reason: Authenticated provider reuse and live audio readiness/ownership cross transport and runtime boundaries.

## Intended behavior

An ordinary Buddy message obtains a correlated conversational answer using the configured Chat provider. Requests to search, ingest or invoke a skill retain explicit plan review. Same-connection, same-session typed turns retain FIFO order and separate request correlation. The receive loop stays responsive: explicit Stop invalidates active and queued turns, and a later send starts a fresh queue even if dispatched work delays cancellation. Full Live's explicit Start prepares the selected real speech services before requesting microphone access; failures identify the unavailable service, and Stop releases recording and playback. The local Whisper/Kokoro path does not require OpenAI credentials.

## Implementation sequence

1. Test and implement an internal authenticated Chat adapter with bounded history, fixed route, safe errors and cancellation; integrate it with owned FIFO Live turns, explicit Stop cancellation of active and queued tasks, and tool intent routing.
2. Test and implement real STT identifier/language normalization, preparation, failure handling and correlated voice envelopes. Remove scaffold fallback from the microphone path. Bound each connection to one preparation and reuse Chat's `await_bounded_owned_operation` for a 30-second STT initialization deadline, retaining late cleanup ownership after timeout/disconnect; verify exact-once cleanup, busy retries, capacity rejection before work starts, and no late readiness.
3. Test and implement frontend preparation/capture/playback generations, authoritative Stop and actionable readiness/setup navigation.
4. Verify focused Python and frontend suites, touched-scope lint/Bandit and exact-session provider, stop/retry and human voice UAT. Record exact source hashes and sanitized receipts; keep credential-specific unverified claims open.
5. Review changes, rebase onto latest dev, and create the requested PR. PR creation does not authorize merging before the repository's human-summary gate is satisfied.
