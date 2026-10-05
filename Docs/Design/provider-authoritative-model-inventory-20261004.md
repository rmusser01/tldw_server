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
- Discovery uses checked HTTP egress, no redirects, bounded pagination/timeouts,
  no secret logging and no generation probes. Unsupported APIs are explicitly
  unavailable rather than inventing an inventory.
- Shared Chat dispatch checks the exact selected ID with the effective credential
  and endpoint before generation, including saved selections and defaults.
  BYOK and audited transport restrictions must not be bypassed by discovery.
- The shared adapter credential-binding boundary also checks direct generation
  callers, including speech, summaries, prompts and workflows. Discovery time
  consumes audited call deadlines; it does not reset the generation budget.
- Native Anthropic Messages (including token counting) and standalone Slides
  validate the same effective credential and endpoint before direct requests.
  Slides rechecks its closed target and tightened limits after discovery yields.
- Discovery's absolute network deadline cancels slow-trickle reads and closes
  owned clients; response-size caps are enforced before buffering the full body.
- OpenRouter uses its account-filtered `/models/user` endpoint and validates
  every explicit fallback model against the same credential-scoped inventory.
- Override model lists restrict current IDs; they cannot restore absent IDs.

Discovery currently covers OpenAI, Anthropic, Cohere, DeepSeek, Google, Groq,
Mistral, Moonshot, OpenRouter, Qwen, Novita, Poe, Together and Hugging Face's
global chat router. Bedrock control-plane discovery is not implemented;
Z.AI and MiniMax listing contracts are not verified. These providers and legacy
Hugging Face model-specific routes are unavailable, not seeded with static IDs.

## Implementation Plan

1. Add focused failing tests for stale catalog IDs, override resurrection and
   pre-generation validation. Implement bounded provider list discovery using
   existing HTTP and readiness helpers.
2. Replace commercial catalog sources and Chat's shared availability/dispatch
   checks. Keep response shapes and local-provider paths compatible.
3. Run Python 3.12 focused regressions, lint and Bandit; independent review;
   upstream dev PR. Track hosted backport and deployment separately, without
   claiming source merge is deployment or successful live Chat.

ADR check: no new ADR required. This repairs availability within the adapter and
trusted endpoint boundary governed by ADR-025; no new transport, authentication,
storage or dependency architecture is introduced.

Official API references: [Anthropic](https://platform.claude.com/docs/en/api/models),
[DeepSeek](https://api-docs.deepseek.com/api/list-models/),
[Gemini](https://ai.google.dev/api/models),
[Cohere](https://docs.cohere.com/reference/list-models).
