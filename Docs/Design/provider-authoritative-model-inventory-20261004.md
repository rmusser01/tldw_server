# Provider-authoritative model inventory

Task: TASK-13460. Baseline: dev `75ab224081`.

The commercial catalog currently combines config and historical pricing IDs;
OpenRouter adds live IDs without removing stale ones. Chat validation also accepts
these IDs and fuzzy date-stripped aliases. This can display and call retired models.

## Contract

- Commercial selectable IDs originate from the provider's authenticated model
  list, never pricing, defaults or override allowlists. Pricing remains historical
  enrichment. Local discovery and native authentication remain unchanged.
- A short-lived, bounded credential/endpoint-scoped cache avoids repeated calls.
  Refresh failures and expired entries fail closed, with no stale fallback.
  Concurrent misses share one fetch within each caller's deadline. Forced
  refreshes supersede waiters and cache publication, not a leader's already
  authenticated response; this is request-snapshot behavior, not revocation.
- Discovery uses checked HTTP egress, no redirects, bounded pagination/timeouts,
  no secret logging and no generation probes. Unsupported APIs are explicitly
  unavailable rather than inventing an inventory.
- Shared Chat dispatch checks the exact selected ID with the effective credential
  and endpoint before generation, including saved selections and defaults.
  BYOK and audited transport restrictions must not be bypassed by discovery.
  Anthropic aliases require an authenticated single-model lookup resolving to a
  current inventory ID. Router selectors validate their current catalog entry
  while preserving the request string; catalog variants cannot be dropped.
- The shared adapter credential-binding boundary also checks direct generation
  callers, including speech, summaries, prompts and workflows. Discovery time
  consumes audited call deadlines; it does not reset the generation budget.
- Native Anthropic Messages (including token counting) and standalone Slides
  validate the same effective credential and endpoint before direct requests.
  Slides rechecks its closed target and tightened limits after discovery yields.
- Discovery's absolute network deadline cancels slow-trickle reads and closes
  owned clients; response-size caps are enforced before buffering the full body.
  Retry waits consume that same deadline. Async ordinary and character auto
  routing and prompt improvement offload discovery rather than blocking the
  event loop. DNS and certificate-pin connect/handshake checks share the deadline.
- OpenRouter uses its account-filtered `/models/user` endpoint and validates
  every explicit fallback model against the same credential-scoped inventory.
- Override model lists restrict current IDs; they cannot restore absent IDs.
- Browser catalogs use a five-minute TTL and a new persisted-cache version;
  expired cached getters and failed fetches cannot return old availability.

Discovery currently covers OpenAI, Anthropic, Cohere, DeepSeek, Google, Groq,
Mistral, Moonshot, OpenRouter, Qwen, Novita, Poe, Together and Hugging Face's
global chat router. Bedrock control-plane discovery is not implemented;
Z.AI and MiniMax listing contracts are not verified. These providers and legacy
Hugging Face model-specific routes are unavailable, not seeded with static IDs.

## Implementation Plan

### Stage 1: Discovery

Goal: bounded provider inventories using existing HTTP and readiness helpers.
Success criteria: no pricing/config resurrection, credential-scoped cache and
sanitized failures. Tests: stale IDs, pagination, egress and refresh concurrency.
Status: Complete.

### Stage 2: Generation Boundaries

Goal: provider-authoritative catalog and shared availability/dispatch checks.
Success criteria: retired selections cannot generate; provider-supported aliases
and routing modifiers remain compatible without fuzzy availability inference.
Tests: ordinary/direct/native/Slides calls, overrides and routing selections.
Status: Complete (provider-confirmed compatibility and dispatch/offload tests).

### Stage 3: Verification And Publication

Goal: Python 3.12 regressions, lint, Bandit and independent review; upstream dev
PR and separate private backport. Success criteria: verified exact source and
normal review gates, with source merge distinguished from live deployment.
Tests: integrated regressions and strict frozen-pin private patch replay.
Status: In Progress (PR #3191; final followup verification pending).

ADR check: no new ADR required. This repairs availability within the adapter and
trusted endpoint boundary governed by ADR-025; no new transport, authentication,
storage or dependency architecture is introduced.

Official API references: [Anthropic](https://platform.claude.com/docs/en/api/models),
[DeepSeek](https://api-docs.deepseek.com/api/list-models/),
[Gemini](https://ai.google.dev/api/models),
[Cohere](https://docs.cohere.com/reference/list-models).
Compatibility references: [Anthropic alias lookup](https://platform.claude.com/docs/en/api/models/retrieve),
[OpenRouter routing/catalog variants](https://openrouter.ai/docs/guides/routing/model-variants/overview),
[Hugging Face router selectors](https://huggingface.co/docs/inference-providers/main/en/index).
