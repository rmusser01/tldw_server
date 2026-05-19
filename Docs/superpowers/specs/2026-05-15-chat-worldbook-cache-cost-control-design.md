# Chat And World-Book Cache Cost-Control Design

Date: 2026-05-15
Status: Approved brainstorming design, revised after design review
Backlog: TASK-377
Scope: Chat prompt assembly, world-book prompt injection, provider prompt-cache accounting, and local inference cache diagnostics

## Purpose

Design a cache-aware cost-control layer around the existing chat pipeline and
world-book functionality. The goal is to prevent cache-busting prompt assembly
from creating surprise provider costs, unexpected token bursts, or local
inference prefill spikes.

This is a design artifact only. Implementation sequencing, file ownership, API
diffs, migrations, and exact test file placement belong in a later
implementation plan.

## Evidence Base

The design is grounded in the current local code paths:

- `tldw_Server_API/app/core/Chat/chat_service.py` builds generic chat provider
  call parameters and already passes through `extra_body` and `extra_headers`.
- `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py` injects
  world-book context into character-session prompt previews and completion
  requests.
- `tldw_Server_API/app/core/Character_Chat/world_book_manager.py` selects and
  budgets world-book entries before prompt injection.
- `tldw_Server_API/app/core/Usage/usage_tracker.py` currently calculates cost
  from prompt and completion token totals.
- `tldw_Server_API/app/core/AuthNZ/repos/usage_repo.py` persists
  `llm_usage_log` rows with prompt, completion, total token, and USD fields.
- `tldw_Server_API/app/core/Local_LLM/LlamaCpp_Handler.py` already exposes
  llama.cpp runtime cache flags such as `prompt_cache`, `cache_prompt`, and
  `cache_reuse`.
- `tldw_Server_API/app/core/LLM_Calls/providers/local_adapters.py` sends local
  OpenAI-compatible payloads and merges `extra_body`, but strict compatibility
  filtering can strip non-standard fields.

Provider references:

- OpenAI prompt caching: https://platform.openai.com/docs/guides/prompt-caching
- Anthropic prompt caching: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- Gemini context caching: https://ai.google.dev/gemini-api/docs/caching
- OpenRouter prompt caching: https://openrouter.ai/docs/features/prompt-caching
- vLLM automatic prefix caching: https://docs.vllm.ai/en/stable/examples/features/automatic_prefix_caching/

## Design Review Corrections

The initial design had one important risk: it could be read as treating every
cache as a billing prompt cache. That is incorrect.

This design separates two cache classes:

- Billing prompt caches: OpenAI, Anthropic, Gemini, and OpenRouter. These can
  affect external provider billing, usage metadata, and cache-hit reporting.
- Local inference caches: vLLM and llama.cpp. These primarily affect prefill
  latency, throughput, memory pressure, and runtime stability. They generally
  do not reduce an external API bill because the server is self-hosted.

The shared concern is prompt stability. The accounting, diagnostics, and user
impact differ by cache class.

## Goals

- Make the final prompt shape observable before and after provider dispatch.
- Distinguish stable prompt prefix, volatile prompt tail, world-book
  contribution, history contribution, RAG contribution, and tool/schema
  contribution.
- Keep prompt preview and provider-send accounting on the same assembly model so
  diagnostics cannot describe a different prompt from the one sent to the
  provider.
- Normalize provider usage fields into a cache-aware internal usage shape.
- Preserve conservative cost estimation when provider pricing, cache discounts,
  or usage fields are missing.
- Detect prompt growth, world-book token bursts, output-token bursts, and
  reasoning-token risk before they become expensive.
- Support local engine cache diagnostics for vLLM and llama.cpp without
  pretending they are paid provider cache discounts.
- Keep provider adapters thin: translate supported cache knobs and normalize
  usage, but do not own prompt layout policy.

## Non-Goals

- No full implementation in this design phase.
- No automatic prompt restructuring that changes model-visible behavior without
  tests and opt-in rollout.
- No claim that local vLLM or llama.cpp cache eligibility proves a cache hit.
- No provider-specific cache-control injection unless outbound shape support is
  explicit and tested.
- No attempt to make cache pricing exact for unknown or unpriced models.
- No unrestricted persistence of raw prompts, raw provider usage payloads, or
  provider-specific request bodies.
- No broad rewrite of chat, character chat, or world-book storage.

## Conceptual Model

The design introduces one shared prompt diagnostic model and two cache lanes.

### Shared Prompt Diagnostics

Every LLM call should be able to produce a `PromptCostEnvelope` after final
prompt assembly and before provider dispatch.

The envelope records:

- provider and model
- endpoint and operation
- conversation id, when available
- final ordered messages or a redacted/hash-only representation
- estimated prompt tokens
- estimated completion cap
- requested choice count and output-token multiplier risk
- reasoning-token controls or defaults, where the provider exposes them
- stable prefix fingerprint
- dynamic tail fingerprint
- full prompt fingerprint
- prompt fingerprint version and canonicalization method
- history token estimate
- world-book token estimate
- RAG token estimate, if applicable
- tool/schema token estimate, if applicable
- prompt growth warnings
- selected guardrail decisions
- cache lane intent

The envelope should avoid storing raw sensitive content by default. It can store
hashes, counts, ids, and bounded diagnostics, with raw prompt capture reserved
for explicit local debugging modes.

Prompt fingerprints should be versioned. The implementation should canonicalize
message role, order, and content in a stable way before hashing so that
insignificant JSON key ordering does not create false cache-busting signals. The
fingerprint version must change if canonicalization rules change.

### Billing Prompt Cache Lane

`BillingPromptCacheIntent` describes paid-provider cache behavior:

- `none`: no cache behavior requested or expected
- `automatic_prefix`: provider may automatically cache stable prefixes
- `explicit_breakpoint`: provider needs a cache marker or cache-control field
- `cached_content_reference`: provider uses a separately created cached object
- `provider_passthrough`: caller supplied provider-specific controls through
  `extra_body` or `extra_headers`
- `unsupported_or_unknown`: provider is paid or remote, but cache semantics are
  unknown or unsupported in the app

This lane applies to OpenAI, Anthropic, Gemini, OpenRouter, and future paid
providers with billable cache semantics.

### Local Inference Cache Lane

`InferencePrefixCacheIntent` describes local runtime cache behavior:

- `unknown`: runtime support is not known
- `not_applicable`: provider is not a local inference engine
- `eligible_stable_prefix`: prompt shape is likely eligible for reuse if the
  engine is configured for it
- `runtime_enabled`: deployment reports prefix or prompt cache support enabled
- `runtime_disabled`: deployment reports support disabled
- `passthrough_requested`: caller or config requested engine-specific cache
  behavior
- `remote_compat_unknown`: provider uses an OpenAI-compatible local/custom
  surface, but runtime cache capabilities cannot be discovered

This lane applies to vLLM, llama.cpp, and similar local backends. It is a
performance and capacity signal, not a billing discount signal.

`extra_body` and `extra_headers` remain raw escape hatches, not the cache
contract. First-class cache controls should be represented in the internal
intent model, audited, redacted, and translated by provider-aware code. Raw
passthrough values may be recorded only as bounded key names or redacted
summaries.

## Prompt Layout Policy

The default prompt layout should favor stable prefixes without hiding dynamic
context:

1. App/system policy and stable global behavior.
2. Static persona or character base instructions.
3. Static tool/schema definitions, when required by the provider.
4. Pinned or static lore that the user explicitly wants always included.
5. Conversation history in chronological order where possible.
6. Dynamic world-book entries triggered by the current context.
7. RAG snippets, slash-command output, and other volatile context.
8. Current user turn.

This is a policy target, not an immediate mandate. Existing behavior should be
measured first, then changed in staged implementation only where tests prove the
model-visible outcome is preserved or the behavior change is explicitly
intended.

Prompt preview and provider-send paths must share the same prompt assembly
primitive or the same envelope builder. If preview remains separate from send,
the design must record preview/send fingerprints and warn when they diverge.

## World-Book Diagnostics

World-book prompt injection is a likely cost and cache-stability hotspot because
small user turns can activate large dynamic context blocks.

Extend world-book diagnostics with:

- world-book ids considered
- selected entry ids
- selected entry priorities and deterministic order
- estimated tokens by entry
- total world-book token estimate
- configured token budget
- entries dropped due to budget
- recursive scan depth used
- recursive expansion count
- static or pinned entry classification
- dynamic triggered entry classification
- world-book block fingerprint
- changed-materially-from-previous-turn flag, when prior data is available

Static or pinned entries may be cache-friendly if intentionally placed near the
stable prefix. Dynamic triggered entries should normally remain near the prompt
tail because they are likely to change turn by turn.

Pinned/static classification can be derived from existing fields in the first
implementation. Adding a new world-book schema field is optional and should not
block the measurement stage. Deterministic ordering should include explicit
priority, existing ordering fields, and a stable id tie-breaker.

## Provider Handling

### OpenAI

OpenAI prompt caching is prefix-oriented. The app should:

- keep stable prompt content early where possible
- support a stable provider cache key when configured
- capture `usage.prompt_tokens_details.cached_tokens` when present
- avoid claiming a cache hit if cached-token usage is missing

### Anthropic

Anthropic prompt caching requires explicit cache-control placement. The app
should:

- only emit cache-control fields through tested adapter translation
- avoid marking volatile world-book entries as cacheable by default
- capture cache creation and cache read token fields when present
- account for cache write/read costs separately when pricing is known

### Gemini

Gemini context caching uses cached-content concepts rather than simple prefix
discounts. The app should:

- model cached content as `cached_content_reference`
- separate cached object lifecycle from one-off chat dispatch
- capture provider usage metadata related to cached content when present
- avoid creating cached content automatically without a policy gate

### OpenRouter

OpenRouter sits between the app and upstream providers. The app should:

- preserve provider routing information where available
- capture cache-related usage fields exposed by OpenRouter
- avoid assuming cache support unless the routed model/provider combination
  reports or documents support
- treat provider switching as a possible cache-busting event

### vLLM

vLLM automatic prefix caching is a local inference optimization that reuses KV
cache for prompts sharing the same prefix when APC is enabled. The app should:

- record stable prefix fingerprints for vLLM calls
- expose whether the configured vLLM deployment is expected to have prefix
  caching enabled if that can be discovered
- classify calls as eligible or not eligible for prefix reuse
- avoid recording billed cache discounts for vLLM
- optionally consume vLLM metrics later if they can prove actual cache hits

### llama.cpp

llama.cpp support must distinguish server startup flags from request payload
fields.

The app should:

- record configured runtime cache flags from managed llama.cpp where available
- treat request-level cache fields such as `cache_prompt` as capability-gated
  passthrough, not universal OpenAI-compatible behavior
- respect strict OpenAI-compatible filtering, which may strip local extension
  fields
- capture prompt fingerprints and token estimates to diagnose full re-prefill
  risk
- treat cache-related failures as local runtime or compute-risk issues, not
  provider billing issues

## Usage Accounting Shape

Introduce a normalized usage shape before persistence:

```text
provider
model
endpoint
operation
prompt_tokens_total
prompt_tokens_billed_full
prompt_tokens_cached_read
prompt_tokens_cache_write
completion_tokens
total_tokens
prompt_cost_usd
cached_prompt_cost_usd
cache_write_cost_usd
completion_cost_usd
total_cost_usd
estimated
cache_lane
cache_hit_ratio
raw_usage_metadata
prompt_envelope_id
usage_normalizer_version
estimated_reason
```

For local engines, cost fields can remain zero or estimated according to the
existing pricing policy, while compute-risk fields and prompt diagnostics remain
useful.

Persisted schemas should add nullable fields so older rows and fallback inserts
remain compatible. Admin summaries can roll up new fields only when present.
`raw_usage_metadata` must be size-bounded and redacted. It should preserve
provider token/cost fields and omit prompt text, response text, request headers,
API keys, user identifiers, and arbitrary provider payload bodies.

## Guardrails

Pre-dispatch guardrails should include:

- total prompt token warning and hard cap
- estimated completion token warning and hard cap
- `n`/choice-count cost multiplier warning
- reasoning-token budget warning where the provider supports hidden or separate
  reasoning tokens
- world-book token warning and hard cap
- per-conversation prompt growth anomaly detection
- provider/model context window safety checks when known
- warning when a stable prefix fingerprint changes unexpectedly
- warning when provider routing changes in a cache-sensitive conversation
- local compute-risk warning for vLLM and llama.cpp when a large prefix is not
  cache-eligible or runtime cache support is disabled
- conservative cost estimate when exact provider cache fields are unavailable
- stream-disconnect accounting fallback so a partially streamed request is not
  silently counted as zero usage

Hard blocks should be rare and reserved for configured safety limits. Most first
pass behavior should warn, log, and expose diagnostics rather than silently
changing prompt content.

## Data Flow

1. Endpoint receives a chat request.
2. Existing validation and provider/model resolution run.
3. Prompt assembly builds final messages, including history, persona,
   world-book, RAG, tools, and current turn.
4. `PromptCostEnvelope` is created from the final prompt.
5. Guardrails inspect the envelope and either allow, warn, or block.
6. Provider adapter receives the prompt and supported cache intent.
7. Provider response returns normal content plus provider usage metadata where
   available.
8. `ProviderUsageNormalizer` converts provider usage into the normalized usage
   shape.
9. `CostCalculator` computes conservative costs.
10. Usage persistence writes the existing token totals plus cache-aware fields.
11. Admin/reporting surfaces aggregate cache and prompt-growth data.

For streaming calls, the envelope should be recorded at dispatch time. Final
provider usage should update the same record when available. If the stream
disconnects, errors, or never returns final usage, accounting should fall back to
the preflight estimate and mark the row as estimated with a specific reason.

## Staged Implementation Boundary

### Stage 1: Measurement Only

Add prompt envelopes, prompt fingerprints, world-book diagnostics, and
cache-aware usage normalization without changing outbound prompt layout or
injecting provider-specific cache controls.

This stage should also add preview/send fingerprint comparison for character
chat prompt preview and completion paths.

### Stage 2: Guardrails

Add warning and hard-cap policy for total prompt tokens, world-book tokens,
context-window risk, prompt growth, provider routing changes, and local
compute-risk cases. Include output-token and choice-count multipliers in the
same guardrail pass.

### Stage 3: Provider Billing Cache Controls

Add provider-specific cache intent translation for OpenAI, Anthropic, Gemini,
and OpenRouter only where outbound payload shape and response usage fields are
tested.

### Stage 4: Local Inference Cache Diagnostics

Add vLLM and llama.cpp runtime capability reporting, eligibility diagnostics,
and optional metrics ingestion if the engine exposes reliable cache-hit
telemetry.

### Stage 5: Admin Reporting

Expose cached-token ratio, prompt growth, world-book contribution, estimated
cost confidence, local compute-risk events, and cache-sensitive routing changes
in admin usage views.

## Testing Strategy

Tests should cover:

- deterministic prompt assembly for identical inputs
- prompt preview and provider-send fingerprint parity
- stable prefix, dynamic tail, and full prompt fingerprints
- fingerprint versioning and canonicalization stability
- world-book ordering, token budget enforcement, and dropped-entry diagnostics
- pinned/static lore versus dynamic triggered lore classification
- generic chat and character chat prompt envelope creation
- streaming and non-streaming usage accounting parity
- stream disconnect/error fallback accounting
- provider usage normalization for OpenAI, Anthropic, Gemini, OpenRouter, vLLM,
  llama.cpp, and unknown providers
- conservative cost fallback when cache pricing is unknown
- redaction and size limits for raw usage metadata
- output-token and `n`/choice multiplier guardrails
- strict OpenAI-compatible filtering for llama.cpp/local extension fields
- local compute-risk warnings independent from billing cost warnings

## Risks And Mitigations

- Risk: Prompt layout changes alter model behavior.
  Mitigation: Stage 1 measures only; layout changes require snapshot tests and
  explicit rollout.
- Risk: Provider cache pricing changes.
  Mitigation: pricing stays override-driven and conservative when unknown.
- Risk: Raw prompt diagnostics leak sensitive content.
  Mitigation: store hashes, ids, and counts by default; raw capture is opt-in
  local debugging only.
- Risk: Raw provider usage metadata becomes an unbounded or sensitive data sink.
  Mitigation: persist only bounded, redacted usage fields and version the
  normalizer.
- Risk: Prompt preview and provider-send prompts drift.
  Mitigation: share assembly primitives or compare fingerprints and surface a
  warning when preview and send diverge.
- Risk: Streaming requests are undercounted when final provider usage is absent.
  Mitigation: log a dispatch-time envelope and fall back to marked estimates on
  disconnects, errors, or missing final usage.
- Risk: Local cache telemetry is unavailable or misleading.
  Mitigation: record eligibility and runtime capability separately from proven
  hits.
- Risk: Schema migration adds admin/reporting complexity.
  Mitigation: nullable columns and backward-compatible fallback inserts.

## Resolved V2 Direction

The initial implementation answered the measurement, accounting, guardrail,
provider-cache, local-diagnostic, and usage-reporting questions. The remaining
product questions are resolved for a follow-up v2 implementation:

- Raw prompt envelopes may be persisted only behind an explicit debug mode.
  Debug persistence must be disabled by default, clearly marked as sensitive,
  bounded by size and retention limits, excluded from normal exports, and
  subject to the same redaction/canonicalization constraints as the request-local
  envelope diagnostics. Normal usage rows should continue to store hashes,
  counts, ids, and bounded metadata by default.
- Users should be able to pin specific world-book entries into the stable
  prefix. The UI and API must make this explicit, not infer it silently from
  activation metadata. Pinned entries should be ordered deterministically ahead
  of dynamic triggered entries, surfaced in diagnostics, and bounded by existing
  world-book prompt budgets so pinning cannot create invisible token bursts.
- Cache-aware reporting should stay in the existing usage analytics surface.
  A dedicated cost-control page is not needed for the next iteration. Existing
  usage list, summary, and CSV/reporting paths should gain the new debug-envelope,
  pinned-prefix, and local-latency fields where they are useful and safe.
- Local engines should report prefill latency when reliable signal exists. The
  product must distinguish:
  - `observed_exact`: response-level timing returned by the engine, such as
    llama.cpp prompt timing fields when present.
  - `observed_metrics`: server metrics or tracing correlated to the request,
    such as vLLM metrics where correlation is possible.
  - `app_observed_ttft`: application-measured time-to-first-token for streaming
    calls, which can include queueing and scheduling and is not pure prefill.
  - `estimated`: token-count or throughput-derived projections.

Estimated local latency must never be presented as authoritative. It is a
capacity and UX signal, not billing evidence.

## Approval State

The user approved the revised design direction before this spec was written.
The revision explicitly includes vLLM and llama.cpp and separates provider
billing cache concerns from local inference cache concerns.
