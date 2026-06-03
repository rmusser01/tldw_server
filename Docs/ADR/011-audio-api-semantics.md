# ADR-011: Audio API Semantics

**Status:** Accepted
**Date:** 2026-06-03
**Backfilled from:** `Docs/Design/STT_TTS_Audio_API_Design.md`
**Decision owner:** Human requester approval of ADR backfill continuation
**Related task:** TASK-516
**Related spec/plan:** `Docs/superpowers/plans/2026-06-03-adr-follow-up-sprint-implementation-plan.md`

## Decision

Use centralized Audio API auth, model-first TTS routing with configured fallback behavior, structured streaming errors by default, and non-streaming-only `return_download_link` semantics.

## Context

The Audio API exposes OpenAI-compatible speech, transcription, and streaming transcription surfaces:

- `POST /api/v1/audio/speech`
- `POST /api/v1/audio/transcriptions`
- `WS /api/v1/audio/stream/transcribe`

The implemented design centralizes HTTP auth through `get_request_user` and WebSocket auth through `_audio_ws_authenticate`, keeping single-user API key mode and multi-user JWT/API-key mode aligned across audio endpoints.

TTS provider selection is model-first, using explicit model-to-provider routing before provider aliases and capability/fallback search. Adapter initialization failures are tracked in the provider registry and can be retried after a configured cooldown. Streaming TTS failures default to structured errors rather than embedding error text as audio bytes, while a compatibility switch can still opt into error-as-audio behavior.

For generated speech storage, `return_download_link` is intentionally limited to non-streaming responses. The server can persist the generated audio and return storage headers while still returning the audio bytes in the response body.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Keep separate ad hoc auth paths per audio endpoint | Would diverge single-user and multi-user behavior across HTTP speech, HTTP transcription, and streaming transcription surfaces. |
| Route TTS by provider first instead of model first | Makes OpenAI-compatible model names less predictable and can pick the wrong provider when a model has a known canonical adapter. |
| Treat adapter initialization failures as permanent until process restart | Makes transient provider startup failures harder to recover from when a bounded retry cooldown can safely recheck availability. |
| Emit TTS streaming errors as audio by default | Preserves compatibility for some callers but hides failures from clients that expect structured HTTP or generator errors. |
| Allow `return_download_link` on streaming responses | Streaming and storage registration have different completion semantics; returning a durable download link before the full payload is buffered would be misleading. |
| Return only a storage link for non-streaming generated speech | Breaks OpenAI-compatible callers that expect audio bytes in the response body. |

## Consequences

Audio endpoint auth should continue to route through the shared HTTP and WebSocket auth helpers instead of custom endpoint-local credential parsing.

TTS adapters should preserve model-first selection and only fall back through configured provider priority and capability matching. Adapter failure retry behavior should remain explicit and cooldown-driven.

Structured streaming failures are the default contract. Compatibility-mode error-as-audio behavior remains a configuration escape hatch, not the normal behavior.

`return_download_link=true` requires `stream=false`. Non-streaming responses can include `X-Download-Path` and `X-Generated-File-Id` while still returning audio bytes.

This ADR does not decide TTS/STT preset storage ownership. That remains inventory-only under `INV-022` until a separate owner-reviewed decision is made.

## Follow-up

- Use this ADR as the covering record for `INV-021`.
- Create a separate ADR before changing Audio API auth ownership, TTS routing priority semantics, default streaming error behavior, or `return_download_link` streaming restrictions.
- Keep `INV-022` unresolved until preset storage ownership is reviewed.
