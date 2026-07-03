# ADR-022: Embeddings API And Media Pipeline

**Status:** Accepted
**Date:** 2026-06-05
**Backfilled from:** `tldw_Server_API/app/core/Embeddings/README.md`
**Decision owner:** TASK-2261 confirmation and TASK-2262 backfill scope
**Related task:** TASK-2262
**Related spec/plan:** `Docs/ADR/inventory/2026-06-04-embeddings-confirmation-audit.md`

## Decision

Embeddings use OpenAI-compatible request/response semantics with provider resolution safeguards, optional adapter-registry routing with legacy provider-config fallback, endpoint cache/batching/circuit-breaker reliability controls, and media embedding pipeline ownership where core Jobs owns the durable root `embeddings_pipeline` record while Redis Streams owns stage delivery.

## Context

Embeddings sit at the boundary between OpenAI-compatible public API behavior, multiple local and remote provider implementations, vector storage, reliability controls, and background media processing. The module needs predictable API semantics for clients, bounded provider selection rules for operators, and a clear ownership split between durable job status and asynchronous stage execution.

TASK-2261 confirmed the current Embeddings behavior that bounds this ADR:

- The API schemas use OpenAI-style embeddings request and response models, including required `model`, string/list/token-array input support, `encoding_format`, `dimensions`, `user`, forbidden extra fields, list response data, and usage fields.
- The create endpoint enforces request safeguards before provider execution, including empty-input rejection, list/token-array shape limits, per-model token limits, dimensions validation, create rate limits, and API-call billing limits.
- Provider resolution accepts explicit `x-provider`, provider-qualified model IDs, and model-name heuristics, then applies provider/model allowlists and rejects recognized but unsupported providers instead of silently falling through.
- The LLM embeddings adapter registry is an optional routing path gated by `LLM_EMBEDDINGS_ADAPTERS_ENABLED`; legacy provider configuration and direct provider execution remain the current fallback path.
- Reliability controls include endpoint TTL cache lookup/writeback, uncached request batching, provider-scoped circuit breakers, connection reuse, provider fallback behavior, health breaker visibility, and admin breaker reset/status endpoints.
- Explicit `x-provider` requests suppress provider fallback by default unless `EMBEDDINGS_ALLOW_FALLBACK_WITH_HEADER` enables it.
- Media embedding endpoints force backend ownership to core Jobs, create root Jobs records with `job_type="embeddings_pipeline"`, and enqueue Redis Streams stage messages for chunking, embedding, storage, and content work.
- Redis stage workers update the root Jobs result/status on stage progress, completion, and failure. The older Jobs worker path is explicitly labeled legacy.

This ADR is intentionally bounded. It does not decide billing/accounting semantics beyond the confirmed create-endpoint limits and root Jobs status surface, does not decide local provider URL policy, does not decide ChromaDB versus pgvector storage evolution, and does not turn the legacy Jobs worker into the primary media pipeline path.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Let each provider expose provider-specific embedding request and response shapes | Breaks the OpenAI-compatible API contract that clients and tests rely on, and would push provider differences into callers. |
| Resolve providers only from server defaults | Current behavior intentionally supports explicit provider selection, provider-qualified model IDs, and model-name heuristics while still applying allowlist and unsupported-provider guards. |
| Force all embeddings through the LLM adapter registry immediately | The adapter registry is optional and feature-gated today; legacy provider-config and direct provider execution remain active compatibility paths. |
| Treat Redis Streams as the durable source of truth for media embedding status | Redis Streams is the stage-delivery mechanism. Current durable status and result exposure come from the root Jobs record. |
| Create durable Jobs records for every media embedding stage | The confirmed pipeline uses one durable root `embeddings_pipeline` job and Redis stage messages, avoiding a durable Jobs fan-out for every internal stage. |
| Include vector-store backend policy in this ADR | ChromaDB and optional pgvector behavior are related but separate storage-backend evolution decisions that need their own confirmation if they become governing policy. |
| Include local provider URL override policy in this ADR | Embeddings local API configuration has separate behavior from LLM provider URL policy and should not inherit INV-027 without a focused review. |

## Consequences

Embeddings API changes should preserve OpenAI-compatible request and response semantics unless a later ADR supersedes this one. New validation or policy checks should keep failure modes explicit and client-facing rather than silently falling through to unrelated providers or model defaults.

Provider integrations should maintain the current resolution order and guardrails: explicit provider requests, provider-qualified model IDs, and model-name heuristics can choose a provider, but allowlists and unsupported-provider checks remain authoritative. Explicit provider requests should continue to suppress fallback by default.

The optional LLM embeddings adapter registry can expand, but code and docs must continue to account for the legacy provider-config/direct execution fallback path while it remains supported.

Endpoint reliability controls are part of the accepted production shape. Changes to cache identity, request batching, circuit-breaker behavior, connection reuse, fallback behavior, or health/admin breaker surfaces should be reviewed as API reliability changes, not incidental refactors.

Media embedding pipeline work should keep durable user/admin-visible status on the root Jobs record and use Redis Streams for internal stage delivery. Stage workers should report progress, completion, and failure back to the root job rather than creating competing durable status surfaces.

Billing/accounting behavior, local provider URL policy, vector-store backend evolution, broader multi-tier cache architecture, and legacy Jobs worker removal remain separate decisions.

## Follow-up

- Use this ADR as the covering record for the Embeddings portion of INV-032.
- Keep `Docs/ADR/inventory/2026-06-04-embeddings-confirmation-audit.md` as the evidence record and caveat boundary for this backfill.
- Create separate ADRs if billing/accounting semantics, local provider URL policy, ChromaDB versus pgvector storage ownership, or legacy Jobs worker removal becomes a durable architecture decision.
