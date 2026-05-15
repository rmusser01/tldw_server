# Chat World-Book Cache Cost-Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement cache-aware prompt diagnostics, usage accounting, and guardrails for chat plus character world-book prompts so users can predict and contain provider/API-side cost bursts without accidentally defeating provider prompt caches.

**Architecture:** Add a small prompt-cost envelope layer between prompt assembly and provider dispatch, normalize provider usage metadata into bounded internal fields, persist cache-related accounting separately from existing token totals, and add guardrails/reporting on top. Paid provider billing-cache controls are opt-in and provider-specific. Local vLLM/llama.cpp work is diagnostic and runtime-oriented, because prefix/prompt cache reuse affects latency and throughput more than third-party billing.

**Tech Stack:** FastAPI, Pydantic/Python 3.11+, existing Chat/Character_Chat/LLM_Calls/Usage modules, SQLite/Postgres AuthNZ migrations, Loguru, pytest.

---

## Source Inputs

- Backlog design task: `TASK-377`
- Planning task: `TASK-378`
- Approved spec: `Docs/superpowers/specs/2026-05-15-chat-worldbook-cache-cost-control-design.md`
- Primary runtime paths:
  - `tldw_Server_API/app/core/Chat/chat_service.py`
  - `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
  - `tldw_Server_API/app/core/Character_Chat/world_book_manager.py`
  - `tldw_Server_API/app/core/LLM_Calls/`
  - `tldw_Server_API/app/core/Usage/`
  - `tldw_Server_API/app/core/AuthNZ/`

## PR And Task Slicing

Implement this as multiple small PRs/tasks. Do not combine provider behavior changes with the measurement foundation.

1. Measurement foundation: prompt-cost envelope primitives and world-book diagnostics. No provider behavior changes.
2. Usage accounting: provider usage normalization, cache-token persistence, and cache-aware cost fields. No prompt-layout or provider behavior changes.
3. Guardrails: warning/blocking policy and preview/send parity checks. Default to warn-only unless a caller configures hard caps.
4. Paid provider cache controls: OpenAI, Anthropic, Gemini, and OpenRouter opt-in billing-cache intent translation.
5. Local inference diagnostics: vLLM and llama.cpp prefix/prompt cache diagnostics, runtime hints, and non-billing reporting.
6. Reporting/admin surface: aggregate cost/cache metrics after raw data is persisted and validated.

Each slice should have its own Backlog task before edits begin. `TASK-378` only covers this plan artifact.

---

## Stage 1: Prompt-Cost Envelope Primitives

**Goal:** Create deterministic prompt diagnostics that can be computed after final prompt assembly and before provider dispatch.

**Success Criteria:**
- Final outbound chat messages can be converted into a `PromptCostEnvelope`.
- Stable, versioned fingerprints are produced for system/static, world-book, history, retrieval/tool, and user-turn prompt segments.
- Fingerprints use canonicalized provider-bound content, not pre-template source objects.
- No provider request payload changes are introduced in this stage.

**Files:**
- Add `tldw_Server_API/app/core/Chat/prompt_cost_envelope.py`
- Add `tldw_Server_API/tests/Chat/unit/test_prompt_cost_envelope.py`
- Modify `tldw_Server_API/app/core/Chat/chat_service.py` only enough to expose a safe hook point if tests need it.

**Implementation Steps:**
- [x] Write failing unit tests for message canonicalization:
  - same logical prompt produces same fingerprint despite dict key order;
  - changed message order changes the aggregate fingerprint;
  - large text is hashed rather than copied into diagnostics;
  - unsupported message parts are represented by bounded type markers.
- [x] Write failing unit tests for segment accounting:
  - static/system messages are counted separately from user/history;
  - world-book and retrieval segments can be passed explicitly;
  - token estimates are present and never negative;
  - fingerprint version is included in every envelope.
- [x] Implement `PromptSegment`, `PromptCostEnvelope`, and helpers:
  - `canonicalize_messages(messages: Sequence[Mapping[str, Any]]) -> str`
  - `fingerprint_text(text: str, *, version: str = "prompt-v1") -> str`
  - `estimate_segment_tokens(text: str) -> int`
  - `build_prompt_cost_envelope(...) -> PromptCostEnvelope`
- [x] Keep the token estimator conservative and local. Reuse existing chat token-estimate helpers if available; do not add a tokenizer dependency in this slice.
- [x] Add redaction limits for diagnostic payloads: IDs, counts, hashes, and bounded numeric estimates only.
- [x] Run focused tests.

**Tests:**
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Chat/unit/test_prompt_cost_envelope.py -v
```

**Commit Message:**
```text
feat(chat): add prompt cost envelope primitives
```

---

## Stage 2: World-Book Diagnostics And Preview Parity

**Goal:** Replace duplicated world-book prompt assembly in character preview and completion paths with a shared helper that returns the inserted prompt text plus cache/cost diagnostics.

**Success Criteria:**
- Preview and provider-send paths use the same world-book assembly helper.
- The helper exposes trigger counts, matched book/entry IDs, included/dropped entry counts, estimated token cost, and a stable world-book segment fingerprint.
- Diagnostics distinguish static/pinned world-book entries from dynamically triggered entries using existing world-book metadata where possible.
- No world-book schema migration is required in this stage.

**Files:**
- Add `tldw_Server_API/app/core/Character_Chat/world_book_prompt_context.py`
- Add `tldw_Server_API/tests/Character_Chat/test_world_book_prompt_context.py`
- Modify `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
- Modify `tldw_Server_API/tests/Character_Chat/test_world_book_and_limits.py` if existing expectations need to assert diagnostics.

**Implementation Steps:**
- [x] Write failing tests proving preview and completion code paths receive identical world-book text/fingerprint for the same inputs.
- [x] Write failing tests for bounded diagnostics:
  - matched book IDs and entry IDs are present;
  - trigger text is not persisted verbatim;
  - dropped entries and token-budget truncation are visible;
  - static/pinned entries are flagged when current metadata supports it.
- [x] Implement `WorldBookPromptContext`:
  - `text: str`
  - `system_message: Mapping[str, str] | None`
  - `diagnostics: Mapping[str, Any]`
  - `fingerprint: str`
  - `estimated_tokens: int`
- [x] Move duplicated `WorldBookService.process_context(... include_diagnostics=True)` orchestration behind `build_world_book_prompt_context(...)`.
- [x] Preserve existing insertion order: world-book system message after existing system messages and before conversational turns.
- [x] Wire preview and completion-v2 endpoints to the helper.
- [x] Add a guard in tests that preview envelope fingerprint equals provider-send fingerprint when no final prompt mutation occurs.

**Tests:**
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Character_Chat/test_world_book_prompt_context.py \
  tldw_Server_API/tests/Character_Chat/test_world_book_and_limits.py \
  -v
```

**Commit Message:**
```text
feat(character-chat): add world-book prompt diagnostics
```

---

## Stage 3: Provider Usage Normalization

**Goal:** Normalize provider usage metadata into bounded, provider-agnostic cache/cost fields before persistence.

**Success Criteria:**
- OpenAI, Anthropic, Gemini, OpenRouter, vLLM, llama.cpp, and unknown local responses can be normalized.
- Cached/billed/uncached prompt-token fields are represented without changing existing `prompt_tokens`, `completion_tokens`, or `total_tokens` semantics.
- Raw provider usage metadata is bounded and redacted.
- Streaming fallback accounting records an estimate reason when authoritative provider usage is unavailable.

**Files:**
- Add `tldw_Server_API/app/core/Usage/llm_usage_normalizer.py`
- Add `tldw_Server_API/tests/Usage/test_llm_usage_normalizer.py`
- Modify `tldw_Server_API/app/core/Usage/usage_tracker.py`
- Modify `tldw_Server_API/app/core/Chat/chat_service.py` only at usage-log call sites.

**Implementation Steps:**
- [ ] Write failing tests for usage metadata shapes:
  - OpenAI-style nested prompt token details;
  - Anthropic-style cache creation/read fields;
  - Gemini-style cache metadata when present;
  - OpenRouter provider-routing metadata when present;
  - OpenAI-compatible local server with no cache fields;
  - malformed or oversized raw metadata.
- [ ] Implement `NormalizedLLMUsage` with fields:
  - `input_tokens`
  - `output_tokens`
  - `total_tokens`
  - `cached_input_tokens`
  - `cache_write_input_tokens`
  - `cache_read_input_tokens`
  - `billable_input_tokens`
  - `reasoning_tokens`
  - `choice_count`
  - `estimate_source`
  - `raw_usage_metadata`
- [ ] Implement provider-specific extractors that fail closed to standard token fields.
- [ ] Bound `raw_usage_metadata` by size, key allowlist/denylist, and nesting depth.
- [ ] Add `estimate_source` values such as `provider_usage`, `stream_estimate`, `disconnect_estimate`, and `missing_usage`.
- [ ] Wire usage normalization into non-stream and stream completion logging without changing persisted schema yet. Store only in process-local structured data until Stage 4.

**Tests:**
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Usage/test_llm_usage_normalizer.py -v
```

**Commit Message:**
```text
feat(usage): normalize llm cache usage metadata
```

---

## Stage 4: Persistence And Cache-Aware Cost Calculation

**Goal:** Persist normalized cache usage fields and compute cost with cache-read/write discounts where provider pricing supports them.

**Success Criteria:**
- Existing usage inserts continue working against old and migrated schemas.
- New nullable columns store cache usage without breaking old rows.
- Pricing can represent input, output, cache-read input, cache-write input, and reasoning-token components.
- Cost computation preserves current behavior when cache pricing is unknown.

**Files:**
- Modify `tldw_Server_API/app/core/AuthNZ/migrations.py`
- Modify `tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py` if Postgres migration helpers are used for this table.
- Modify `tldw_Server_API/app/core/AuthNZ/repos/usage_repo.py`
- Modify `tldw_Server_API/app/core/Usage/usage_tracker.py`
- Modify `tldw_Server_API/app/core/Usage/pricing_catalog.py`
- Add or modify `tldw_Server_API/tests/Usage/test_usage_tracker_sqlite.py`
- Add or modify `tldw_Server_API/tests/Usage/test_pricing_catalog.py`
- Add or modify `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_llm_usage_log_router_columns_sqlite.py`
- Add or modify `tldw_Server_API/tests/AuthNZ_Postgres/test_authnz_llm_usage_log_router_columns_pg.py`

**Implementation Steps:**
- [ ] Write failing SQLite migration tests for new nullable `llm_usage_log` columns:
  - `cached_input_tokens`
  - `cache_write_input_tokens`
  - `cache_read_input_tokens`
  - `billable_input_tokens`
  - `reasoning_tokens`
  - `choice_count`
  - `estimate_source`
  - `prompt_fingerprint`
  - `prompt_fingerprint_version`
  - `world_book_fingerprint`
  - `raw_usage_metadata_json`
- [ ] Write/update Postgres migration tests for the same logical fields when the fixture is available.
- [ ] Extend repository insert logic with compatibility fallback for pre-migration schemas.
- [ ] Add pricing catalog fields for cache-read/cache-write rates without requiring every model entry to provide them.
- [ ] Update `compute_costs(...)` to:
  - use cache-specific rates when present;
  - fall back to current prompt-token pricing when missing;
  - never make cached token cost negative;
  - keep old return keys stable.
- [ ] Wire `NormalizedLLMUsage` into `log_llm_usage(...)` and repository insert.
- [ ] Run focused migration/usage tests.

**Tests:**
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Usage/test_usage_tracker_sqlite.py \
  tldw_Server_API/tests/Usage/test_pricing_catalog.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_llm_usage_log_router_columns_sqlite.py \
  -v
```

Optional Postgres verification when local fixture is available:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/AuthNZ_Postgres/test_authnz_llm_usage_log_router_columns_pg.py -v
```

**Commit Message:**
```text
feat(usage): persist cache-aware llm usage fields
```

---

## Stage 5: Guardrails And Preflight Warnings

**Goal:** Add pre-dispatch prompt/cost guardrails that can warn or block surprising prompt growth before the provider call.

**Success Criteria:**
- Guardrails can detect large static/world-book segments, cache-busting fingerprint churn, high output-token caps, high `n`/choice counts, and reasoning-token risk.
- Default behavior is non-breaking and warn-only unless configured otherwise.
- Guardrail decisions are visible in API response metadata where existing schemas allow it, or in logs when response schemas cannot change safely.
- Streaming and non-streaming paths share the same guardrail decision.

**Files:**
- Add `tldw_Server_API/app/core/Chat/prompt_cost_guardrails.py`
- Add `tldw_Server_API/tests/Chat/unit/test_prompt_cost_guardrails.py`
- Modify `tldw_Server_API/app/core/Chat/chat_service.py`
- Modify `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
- Modify config documentation if new settings are exposed.

**Implementation Steps:**
- [ ] Write failing guardrail tests for:
  - warn-only over threshold;
  - hard block over configured max;
  - fingerprint churn across adjacent turns;
  - high `max_tokens`/`max_completion_tokens`;
  - high `n`;
  - reasoning-effort risk when provider parameters indicate it.
- [ ] Implement `PromptCostGuardrailConfig` and `PromptCostGuardrailDecision`.
- [ ] Load config from existing config/env helpers, with disabled/warn-only defaults.
- [ ] Evaluate guardrails after final prompt templating and world-book insertion, before provider dispatch.
- [ ] Attach guardrail warnings to existing metadata/diagnostics surfaces without exposing full prompt text.
- [ ] Ensure a blocked request logs a structured reason and returns a descriptive 4xx response.
- [ ] Run focused tests plus existing chat token-estimate tests.

**Tests:**
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Chat/unit/test_prompt_cost_guardrails.py \
  tldw_Server_API/tests/Chat/unit/test_chat_service_token_estimates.py \
  -v
```

**Commit Message:**
```text
feat(chat): add prompt cost guardrails
```

---

## Stage 6: Paid Provider Billing-Cache Controls

**Goal:** Translate explicit cache intent into provider-specific request parameters only for providers with known billing-cache semantics.

**Success Criteria:**
- Cache behavior remains unchanged unless callers/config opt in.
- Provider-specific cache controls are isolated behind adapter-level helpers.
- OpenRouter pass-through is constrained to provider-supported metadata and does not leak arbitrary `extra_body` fields into accounting assumptions.
- Provider request tests verify exact outbound payloads.

**Files:**
- Add `tldw_Server_API/app/core/LLM_Calls/cache_intents.py`
- Add `tldw_Server_API/tests/LLM_Calls/test_cache_intents.py`
- Modify relevant provider adapter files under `tldw_Server_API/app/core/LLM_Calls/`
- Modify `tldw_Server_API/app/core/Chat/chat_service.py` only to pass explicit intent metadata, not provider-specific payload details.

**Implementation Steps:**
- [ ] Verify current official provider documentation on the implementation date before changing adapter payloads. Record the checked URLs and date in the Backlog task or PR notes.
- [ ] Write failing tests for provider-neutral `BillingPromptCacheIntent`.
- [ ] Write failing adapter tests for outbound request shapes:
  - OpenAI cache-capable prompt sections when supported;
  - Anthropic cache-control blocks at approved message/content boundaries;
  - Gemini cached-content references only when explicitly provided;
  - OpenRouter provider routing/metadata only through allowed keys.
- [ ] Implement provider-neutral cache intent types:
  - `enabled`
  - `scope`
  - `ttl_seconds`
  - `static_segment_fingerprint`
  - `provider_hint`
  - `fail_open`
- [ ] Add provider support declarations so unsupported providers ignore cache intent with a diagnostic warning.
- [ ] Ensure `extra_body` remains a caller escape hatch but is not treated as confirmed cache activation unless the adapter reports it.
- [ ] Add accounting metadata for "cache intent requested" versus "provider usage proved cache hit/read/write".
- [ ] Run focused LLM adapter tests.

**Tests:**
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/LLM_Calls/test_cache_intents.py \
  tldw_Server_API/tests/LLM_Adapters/unit \
  -v
```

**Commit Message:**
```text
feat(llm): add opt-in provider prompt cache intents
```

---

## Stage 7: Local vLLM And llama.cpp Diagnostics

**Goal:** Add local inference diagnostics and runtime guidance without conflating local prefix/prompt cache reuse with paid provider billing cache discounts.

**Success Criteria:**
- vLLM diagnostics describe expected prefix-cache compatibility and request-shape stability.
- llama.cpp diagnostics surface configured prompt-cache flags and request-level cache extension use.
- Local providers do not report cache-billing savings unless a provider explicitly returns usage metadata proving it.
- Strict-filter tests protect local OpenAI-compatible payload compatibility.

**Files:**
- Add `tldw_Server_API/app/core/LLM_Calls/local_cache_diagnostics.py`
- Add `tldw_Server_API/tests/LLM_Calls/test_local_cache_diagnostics.py`
- Modify `tldw_Server_API/app/core/LLM_Calls/providers/local_adapters.py`
- Modify `tldw_Server_API/app/core/Local_LLM/LlamaCpp_Handler.py` only if runtime config reporting needs a small helper.
- Modify existing tests:
  - `tldw_Server_API/tests/LLM_Calls/test_llamacpp_strict_filter.py`
  - `tldw_Server_API/tests/LLM_Calls/test_vllm_strict_filter.py`
  - `tldw_Server_API/tests/LLM_Calls/test_llamacpp_request_extensions.py`

**Implementation Steps:**
- [ ] Write failing diagnostics tests for vLLM:
  - prefix-cache hint is diagnostic-only;
  - no billing-cache fields are invented;
  - request shape instability produces a warning.
- [ ] Write failing diagnostics tests for llama.cpp:
  - startup flags such as `prompt_cache`, `prompt_cache_all`, `prompt_cache_ro`, `cache_prompt`, and `cache_reuse` are surfaced when known;
  - request extensions remain behind existing allow/strict-filter rules;
  - prompt-cache read-only mode is reported distinctly from writable cache mode.
- [ ] Implement `InferencePrefixCacheIntent` and local diagnostics separately from `BillingPromptCacheIntent`.
- [ ] Attach diagnostics to internal usage metadata or response diagnostics where available.
- [ ] Keep all local cache usage cost-neutral in `compute_costs(...)`.
- [ ] Run focused local adapter tests.

**Tests:**
```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/LLM_Calls/test_local_cache_diagnostics.py \
  tldw_Server_API/tests/LLM_Calls/test_llamacpp_strict_filter.py \
  tldw_Server_API/tests/LLM_Calls/test_vllm_strict_filter.py \
  tldw_Server_API/tests/LLM_Calls/test_llamacpp_request_extensions.py \
  -v
```

**Commit Message:**
```text
feat(llm): add local prompt cache diagnostics
```

---

## Stage 8: Reporting And Admin Visibility

**Goal:** Add aggregate visibility after cache-aware usage fields are populated.

**Success Criteria:**
- Existing usage reporting continues working.
- New reports can show prompt tokens, cached input tokens, cache write/read tokens, billable input tokens, output tokens, estimated cost, and estimate source.
- Reports distinguish paid-provider billing cache data from local inference diagnostics.
- No raw prompts or unbounded provider metadata are exposed.

**Files:**
- Modify existing usage/admin/reporting endpoints after locating the current usage reporting owner.
- Add tests under the existing usage or API test directory that owns those endpoints.
- Update relevant documentation under `Docs/` if the API surface changes.

**Implementation Steps:**
- [ ] Locate the current usage summary/reporting endpoints and services.
- [ ] Write failing tests for aggregate cache metrics and redaction behavior.
- [ ] Extend query layer to include cache-aware columns with backward-compatible defaults.
- [ ] Add API response fields only where schema versioning/backward compatibility is clear.
- [ ] Add docs for interpreting:
  - `cached_input_tokens`
  - `cache_write_input_tokens`
  - `cache_read_input_tokens`
  - `billable_input_tokens`
  - `estimate_source`
  - local diagnostic-only cache fields.
- [ ] Run endpoint/reporting tests.

**Tests:**
```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Usage -v
```

**Commit Message:**
```text
feat(usage): report cache-aware llm metrics
```

---

## Cross-Cutting Verification

Run these before considering an implementation PR complete. Narrow the test list to touched slices when working incrementally.

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Chat/unit/test_prompt_cost_envelope.py \
  tldw_Server_API/tests/Chat/unit/test_prompt_cost_guardrails.py \
  tldw_Server_API/tests/Character_Chat/test_world_book_prompt_context.py \
  tldw_Server_API/tests/Usage/test_llm_usage_normalizer.py \
  tldw_Server_API/tests/Usage/test_usage_tracker_sqlite.py \
  tldw_Server_API/tests/LLM_Calls/test_cache_intents.py \
  tldw_Server_API/tests/LLM_Calls/test_local_cache_diagnostics.py \
  -v
```

Run Bandit on touched Python scopes before finalizing each implementation PR:

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/Chat \
  tldw_Server_API/app/core/Character_Chat \
  tldw_Server_API/app/core/Usage \
  tldw_Server_API/app/core/LLM_Calls \
  tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py \
  -f json -o /tmp/bandit_chat_cache_cost_controls.json
```

For docs/planning-only changes, run:

```bash
git diff --check
```

---

## Design Risks To Recheck During Implementation

- Provider docs drift: verify current OpenAI, Anthropic, Gemini, and OpenRouter cache-control semantics before Stage 6.
- Preview/send drift: compute final prompt fingerprints as late as possible and test both paths.
- Cache savings overclaim: only persisted provider usage can prove billing cache hits. Local diagnostics are not billing savings.
- Streaming usage gaps: disconnections and provider streams without final usage must record `estimate_source`.
- Schema compatibility: usage logging must tolerate older local databases until migrations run.
- World-book growth: recursive scanning and static/pinned entries can create invisible prompt expansion; diagnostics must make dropped/included counts visible.
- Raw metadata privacy: keep raw provider usage bounded and redacted; never store prompt text in usage logs.

## Rollout Order

Recommended initial implementation sequence:

1. Stage 1 and Stage 2 in one small measurement PR if the diff stays focused.
2. Stage 3 and Stage 4 in one usage-accounting PR.
3. Stage 5 guardrails as a separate PR so default behavior can be reviewed independently.
4. Stage 6 paid provider cache controls as one provider at a time if adapter changes get large.
5. Stage 7 local diagnostics separately from Stage 6.
6. Stage 8 reporting after real cache-aware usage rows exist in development data.
