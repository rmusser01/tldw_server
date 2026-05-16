# Chat Cache Cost Controls V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the approved v2 cache-cost controls: debug-gated raw prompt-envelope persistence, explicit world-book stable-prefix pinning, usage-analytics reporting extensions, and local prefill-latency telemetry with source/confidence labels.

**Architecture:** Keep v1 cache-cost accounting as the stable foundation. Add v2 as opt-in, bounded extensions around existing prompt envelope, world-book, usage-log, local-cache diagnostic, and admin usage analytics modules. Raw prompt material remains disabled by default and isolated from normal usage exports.

**Tech Stack:** FastAPI, Pydantic/Python 3.11+, existing Chat/Character_Chat/LLM_Calls/Usage/AuthNZ modules, SQLite/Postgres migrations, Next.js shared UI under `apps/packages/ui`, pytest, Vitest.

---

## Source Inputs

- Backlog tracking task: `TASK-398`
- V1 design spec: `Docs/superpowers/specs/2026-05-15-chat-worldbook-cache-cost-control-design.md`
- V1 implementation plan: `Docs/superpowers/plans/2026-05-15-chat-worldbook-cache-cost-control-implementation-plan.md`
- Existing v1 implementation paths:
  - `tldw_Server_API/app/core/Chat/prompt_cost_envelope.py`
  - `tldw_Server_API/app/core/Chat/chat_service.py`
  - `tldw_Server_API/app/core/Character_Chat/world_book_prompt_context.py`
  - `tldw_Server_API/app/core/Character_Chat/world_book_manager.py`
  - `tldw_Server_API/app/core/LLM_Calls/local_cache_diagnostics.py`
  - `tldw_Server_API/app/core/Usage/usage_tracker.py`
  - `tldw_Server_API/app/core/AuthNZ/repos/usage_repo.py`
  - `tldw_Server_API/app/services/admin_usage_service.py`
  - `apps/packages/ui/src/components/Option/Admin/UsageAnalyticsPage.tsx`
  - `apps/packages/ui/src/components/Option/WorldBooks/WorldBookEntryManager.tsx`

## Scope Boundaries

- Do not enable raw prompt persistence by default.
- Do not put raw prompt text into normal `llm_usage_log` exports.
- Do not create a new dedicated LLM cost-control page in v2; extend existing usage analytics.
- Do not treat local prefill latency as billing evidence.
- Do not present token-count projections as observed latency.
- Do not change world-book prompt order except for explicitly pinned stable-prefix entries.

## Task 1: Debug-Gated Prompt Envelope Persistence

**Goal:** Persist raw prompt envelopes only when an explicit debug mode is enabled, with retention, size limits, redaction metadata, and admin-only access.

**Files:**
- Modify: `tldw_Server_API/app/core/Chat/prompt_cost_envelope.py`
- Create: `tldw_Server_API/app/core/Chat/prompt_envelope_debug_store.py`
- Modify: `tldw_Server_API/app/core/Chat/chat_service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/migrations.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/repos/usage_repo.py`
- Modify: `tldw_Server_API/Config_Files/config.txt`
- Modify: `tldw_Server_API/Config_Files/README.md`
- Test: `tldw_Server_API/tests/Chat/unit/test_prompt_envelope_debug_store.py`
- Test: `tldw_Server_API/tests/Usage/test_usage_tracker_sqlite.py`
- Test: `tldw_Server_API/tests/Admin/test_llm_usage_endpoints.py`

**Implementation Steps:**
- [ ] **Step 1: Create the Backlog implementation task**

  Create a child or follow-up task referencing `TASK-398` and this plan. Include the debug-mode privacy constraints in the description.

- [ ] **Step 2: Write failing unit tests for debug gating**

  Add tests proving:
  - debug persistence is disabled by default;
  - no raw prompt text is retained when disabled;
  - enabled persistence stores bounded envelope JSON plus redaction metadata;
  - oversized raw envelopes are rejected or truncated according to config;
  - prompt debug rows have an expiry timestamp.

  Run:

  ```bash
  source .venv/bin/activate
  python -m pytest tldw_Server_API/tests/Chat/unit/test_prompt_envelope_debug_store.py -q
  ```

  Expected: fail because `prompt_envelope_debug_store.py` does not exist.

- [ ] **Step 3: Add a separate debug table**

  Add migration `089` unless a newer migration number exists by implementation time. Prefer a separate table instead of expanding normal usage rows:

  ```text
  llm_prompt_envelope_debug
  - id
  - request_id
  - user_id
  - created_at
  - expires_at
  - provider
  - model
  - prompt_fingerprint
  - prompt_fingerprint_version
  - raw_envelope_json
  - redaction_version
  - capture_reason
  ```

  Add SQLite and Postgres migrations plus indexes on `request_id`, `user_id`, and `expires_at`.

- [ ] **Step 4: Implement the debug store**

  Implement a helper that accepts a `PromptCostEnvelope`, raw provider-bound messages, and request metadata. It should:
  - check config/env gates before doing any write;
  - reuse prompt canonicalization and data URI sanitization;
  - apply max-byte limits before insert;
  - store only when `prompt_envelope_debug_persistence_enabled` is true;
  - return a debug envelope id or `None`.

- [ ] **Step 5: Wire chat and character-chat dispatch**

  After final prompt assembly and before provider dispatch, call the debug store from:
  - `tldw_Server_API/app/core/Chat/chat_service.py`
  - `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`

  Store a debug-envelope id in usage metadata when a row is captured. Do not expose raw prompt text in normal chat responses.

- [ ] **Step 6: Add admin retrieval with explicit sensitive-data boundary**

  Add an admin-only fetch path if needed, but keep it separate from normal list/CSV endpoints. The response should be disabled unless debug persistence is enabled and should return `404` for expired/missing rows.

- [ ] **Step 7: Verify Task 1**

  Run:

  ```bash
  source .venv/bin/activate
  python -m pytest \
    tldw_Server_API/tests/Chat/unit/test_prompt_envelope_debug_store.py \
    tldw_Server_API/tests/Usage/test_usage_tracker_sqlite.py \
    tldw_Server_API/tests/Admin/test_llm_usage_endpoints.py \
    -q
  python -m bandit -r \
    tldw_Server_API/app/core/Chat \
    tldw_Server_API/app/core/AuthNZ/repos/usage_repo.py \
    tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py \
    -f json -o /tmp/bandit_chat_cache_v2_debug_envelopes.json
  git diff --check
  ```

## Task 2: User-Pinned World-Book Stable Prefix Entries

**Goal:** Let users explicitly mark world-book entries as stable-prefix pinned context and make that choice visible in diagnostics and prompt assembly.

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/world_book_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py`
- Modify: `tldw_Server_API/app/core/Character_Chat/world_book_manager.py`
- Modify: `tldw_Server_API/app/core/Character_Chat/world_book_prompt_context.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/character_chat_sessions.py`
- Modify: `apps/packages/ui/src/components/Option/WorldBooks/WorldBookEntryManager.tsx`
- Modify: `apps/packages/ui/src/components/Option/WorldBooks/hooks/useWorldBookImportExport.tsx`
- Modify: `apps/packages/ui/src/services/tldw/client-ownership.ts`
- Test: `tldw_Server_API/tests/Character_Chat/test_world_book_manager_legacy.py`
- Test: `tldw_Server_API/tests/Character_Chat/test_world_book_prompt_context.py`
- Test: `tldw_Server_API/tests/Character_Chat/test_world_book_and_limits.py`
- Test: `apps/packages/ui/src/components/Option/WorldBooks/__tests__/WorldBookEntryManager.budget.test.tsx`

**Implementation Steps:**
- [ ] **Step 1: Create the Backlog implementation task**

  State that pinning is user-controlled stable-prefix behavior, not a provider cache-hit guarantee.

- [ ] **Step 2: Write failing backend tests**

  Cover:
  - `stable_prefix_pinned` or equivalent request/response field round-trips for create/update/list/export/import;
  - pinned entries are selected deterministically before dynamic triggered entries;
  - pinned entries are still bounded by world-book token budgets;
  - diagnostics include pinned entry ids, token estimates, and fingerprints.

  Prefer storing the first implementation in entry `metadata` if that avoids a migration, but expose a typed schema field so users do not need to know metadata internals.

- [ ] **Step 3: Implement typed API support**

  Add a schema field such as:

  ```python
  stable_prefix_pinned: bool = Field(False, description="Always include this entry in the stable prompt prefix when its world book is active.")
  ```

  Normalize it into existing entry metadata keys (`stable_prefix_pinned` and/or `cache_pinned`) and preserve backward compatibility with existing metadata-driven `static_or_pinned` diagnostics.

- [ ] **Step 4: Update world-book prompt assembly**

  Ensure pinned entries:
  - are included before dynamic triggered entries;
  - are ordered by explicit priority, existing ordering fields, and id tie-breaker;
  - contribute to the stable-prefix/world-book diagnostics;
  - cannot bypass total prompt or world-book guardrails.

- [ ] **Step 5: Add UI controls**

  Add a compact pin toggle in `WorldBookEntryManager.tsx` entry create/edit flow. The UI should make the behavior explicit without using explanatory wall text. Use existing control patterns and preserve dense world-book workflows.

- [ ] **Step 6: Verify Task 2**

  Run:

  ```bash
  source .venv/bin/activate
  python -m pytest \
    tldw_Server_API/tests/Character_Chat/test_world_book_prompt_context.py \
    tldw_Server_API/tests/Character_Chat/test_world_book_manager_legacy.py \
    tldw_Server_API/tests/Character_Chat/test_world_book_and_limits.py \
    -q
  bunx vitest run apps/packages/ui/src/components/Option/WorldBooks/__tests__/WorldBookEntryManager.budget.test.tsx
  git diff --check
  ```

## Task 3: Usage Analytics V2 Reporting

**Goal:** Keep cache-cost reporting in existing usage analytics and add v2 fields without exposing raw prompt envelopes in normal list or CSV exports.

**Files:**
- Modify: `tldw_Server_API/app/services/admin_usage_service.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/admin/admin_usage.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/admin_schemas.py`
- Modify: `tldw_Server_API/app/services/llm_usage_aggregator.py`
- Modify: `apps/packages/ui/src/components/Option/Admin/UsageAnalyticsPage.tsx`
- Test: `tldw_Server_API/tests/Admin/test_admin_usage_service.py`
- Test: `tldw_Server_API/tests/Admin/test_llm_usage_endpoints.py`

**Implementation Steps:**
- [ ] **Step 1: Create the Backlog implementation task**

  Specify that usage analytics remains the reporting home for v2.

- [ ] **Step 2: Write failing admin service/API tests**

  Cover:
  - debug envelope availability is surfaced as a boolean/id reference, not raw text;
  - pinned world-book prefix token totals and counts appear in summaries when available;
  - local latency fields include source labels and confidence;
  - CSV exports omit raw prompt envelopes;
  - old schemas still return zero/null defaults.

- [ ] **Step 3: Extend admin usage schemas**

  Add optional fields for:
  - `debug_envelope_available`
  - `debug_envelope_id`
  - `pinned_world_book_tokens`
  - `pinned_world_book_entry_count`
  - `local_prefill_latency_ms`
  - `local_prefill_latency_source`
  - `local_prefill_latency_confidence`

- [ ] **Step 4: Extend usage analytics UI**

  Update `UsageAnalyticsPage.tsx` to show v2 fields in the existing LLM usage sections. Keep raw prompt retrieval as an explicit admin/debug action, not a default table expansion.

- [ ] **Step 5: Verify Task 3**

  Run:

  ```bash
  source .venv/bin/activate
  python -m pytest \
    tldw_Server_API/tests/Admin/test_admin_usage_service.py \
    tldw_Server_API/tests/Admin/test_llm_usage_endpoints.py \
    -q
  bunx vitest run apps/packages/ui/src/components/Option/Admin
  git diff --check
  ```

## Task 4: Local Prefill-Latency Telemetry

**Goal:** Report local prefill latency when reliable signal exists and label all values by source and confidence.

**Files:**
- Modify: `tldw_Server_API/app/core/LLM_Calls/local_cache_diagnostics.py`
- Modify: `tldw_Server_API/app/core/LLM_Calls/providers/local_adapters.py`
- Modify: `tldw_Server_API/app/core/Usage/llm_usage_normalizer.py`
- Modify: `tldw_Server_API/app/core/Usage/usage_tracker.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/migrations.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/pg_migrations_extra.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/repos/usage_repo.py`
- Test: `tldw_Server_API/tests/LLM_Calls/test_local_cache_diagnostics.py`
- Test: `tldw_Server_API/tests/LLM_Calls/test_llamacpp_strict_filter.py`
- Test: `tldw_Server_API/tests/Usage/test_llm_usage_normalizer.py`
- Test: `tldw_Server_API/tests/Usage/test_usage_tracker_sqlite.py`

**Implementation Steps:**
- [ ] **Step 1: Create the Backlog implementation task**

  State that latency is a local compute/UX signal, not billing evidence.

- [ ] **Step 2: Verify current local-engine telemetry docs**

  Before implementation, recheck current vLLM and llama.cpp documentation. Record URLs and date in the Backlog task. The implementation must not assume response timing fields that are not present in the configured runtime.

- [ ] **Step 3: Write failing diagnostics tests**

  Cover source classes:
  - `observed_exact`: engine response includes prompt/prefill timing fields.
  - `observed_metrics`: server metrics/tracing can be correlated to request id.
  - `app_observed_ttft`: streaming time-to-first-token measured by this app.
  - `estimated`: token-count or throughput-derived estimate.

  Tests must assert that estimated values are labeled as estimates and that local diagnostics remain cost-neutral.

- [ ] **Step 4: Extract llama.cpp response timings**

  If the llama.cpp response includes prompt timing fields, normalize them into `local_prefill_latency_ms` and `local_prefill_latency_source="observed_exact"`. Preserve strict OpenAI-compatible outbound filtering.

- [ ] **Step 5: Add vLLM metrics/tracing ingestion only when correlation is reliable**

  Use `observed_metrics` only if the runtime can correlate the metric to the request or a bounded time window with acceptable confidence. Otherwise fall back to `app_observed_ttft` or `estimated`.

- [ ] **Step 6: Persist source-labeled latency fields**

  Add nullable usage columns for local latency fields. Suggested fields:

  ```text
  local_prefill_latency_ms REAL
  local_prefill_latency_source TEXT
  local_prefill_latency_confidence TEXT
  local_prefill_latency_detail_json TEXT
  ```

  Keep details bounded and redacted.

- [ ] **Step 7: Verify Task 4**

  Run:

  ```bash
  source .venv/bin/activate
  python -m pytest \
    tldw_Server_API/tests/LLM_Calls/test_local_cache_diagnostics.py \
    tldw_Server_API/tests/LLM_Calls/test_llamacpp_strict_filter.py \
    tldw_Server_API/tests/Usage/test_llm_usage_normalizer.py \
    tldw_Server_API/tests/Usage/test_usage_tracker_sqlite.py \
    -q
  python -m bandit -r \
    tldw_Server_API/app/core/LLM_Calls \
    tldw_Server_API/app/core/Usage \
    tldw_Server_API/app/core/AuthNZ/repos/usage_repo.py \
    -f json -o /tmp/bandit_chat_cache_v2_local_latency.json
  git diff --check
  ```

## Task 5: End-To-End Closeout

**Goal:** Verify v2 behavior across chat, character chat, usage analytics, and local diagnostics.

**Files:**
- Modify docs as needed:
  - `Docs/API-related/Chat_API_Documentation.md`
  - `Docs/User_Guides/Server/Usage_Module.md`
  - `Docs/Published/User_Guides/Server/Usage_Module.md`
  - `tldw_Server_API/Config_Files/README.md`
- Test:
  - Focused Python suites from Tasks 1-4
  - Focused Vitest suites from Tasks 2-3
  - Browser or Playwright verification for usage analytics and world-book pinning if UI changes ship

**Implementation Steps:**
- [ ] **Step 1: Create the Backlog closeout task**

  Use one closeout task only after implementation tasks exist.

- [ ] **Step 2: Run all focused backend tests**

  Run:

  ```bash
  source .venv/bin/activate
  python -m pytest \
    tldw_Server_API/tests/Chat/unit/test_prompt_cost_envelope.py \
    tldw_Server_API/tests/Chat/unit/test_prompt_envelope_debug_store.py \
    tldw_Server_API/tests/Character_Chat/test_world_book_prompt_context.py \
    tldw_Server_API/tests/LLM_Calls/test_local_cache_diagnostics.py \
    tldw_Server_API/tests/Usage/test_llm_usage_normalizer.py \
    tldw_Server_API/tests/Usage/test_usage_tracker_sqlite.py \
    tldw_Server_API/tests/Admin/test_llm_usage_endpoints.py \
    -q
  ```

- [ ] **Step 3: Run focused frontend tests**

  Run:

  ```bash
  bunx vitest run \
    apps/packages/ui/src/components/Option/WorldBooks/__tests__/WorldBookEntryManager.budget.test.tsx \
    apps/packages/ui/src/components/Option/Admin
  ```

- [ ] **Step 4: Run security and formatting checks**

  Run:

  ```bash
  source .venv/bin/activate
  python -m bandit -r \
    tldw_Server_API/app/core/Chat \
    tldw_Server_API/app/core/Character_Chat \
    tldw_Server_API/app/core/LLM_Calls \
    tldw_Server_API/app/core/Usage \
    tldw_Server_API/app/core/AuthNZ/repos/usage_repo.py \
    tldw_Server_API/app/services/admin_usage_service.py \
    -f json -o /tmp/bandit_chat_cache_cost_v2.json
  git diff --check
  ```

- [ ] **Step 5: Confirm documentation**

  Confirm docs state:
  - raw prompt-envelope persistence is debug-only and sensitive;
  - pinned world-book entries affect stable-prefix assembly and token budgets;
  - usage analytics is the reporting home;
  - local prefill latency is source-labeled and may be estimated.

## Review Notes

- Each implementation task should be its own PR unless the diff remains small.
- Debug prompt persistence and local latency persistence both touch usage schema. If implemented together, merge migrations carefully and keep one compatibility fallback path.
- If vLLM request-level metric correlation cannot be made reliable, ship `app_observed_ttft` and `estimated` first, then create a follow-up task for metrics correlation.
- Never resolve a latency source to `observed_exact` unless the value comes from the engine response for the same request.
