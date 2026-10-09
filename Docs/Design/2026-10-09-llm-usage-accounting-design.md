# LLM usage accounting: record and gate every LLM call at one point

- **Date:** 2026-10-09
- **Status:** Draft, for owner review
- **Backlog:** TASK-13437 (gate LLM tokens/month at every LLM entry point) and TASK-13436 (evaluation cost caps per user, re-scoped below). Both are follow-ups of spec 2, TASK-13434.
- **Builds on:** `Docs/Design/2026-10-02-usage-quota-posture-design.md` (spec 2) and ADR-064. Quotas are per-user `limits.*` values, resolved by `quota_resolver.user_quota`. Nothing is enforced unless `USAGE_QUOTAS_ENABLED` is on, `0` blocks, and errors fail open.

## Owner decisions (2026-10-09)

1. **A general LLM cost quota replaces the evaluation-only cost caps.** The new keys are `limits.llm_cost_per_day_usd` and `limits.llm_cost_per_month_usd`. They cover every LLM path, evaluations included. The dead evaluation cost caps are deleted.
2. **Embeddings do not count against the LLM quotas.** One document ingest must not use up a chat budget. If an embeddings quota is needed later, it gets its own key.
3. **BYOK calls are gated too, for now.** This matches `/chat/completions` today. Most providers don't mark BYOK credentials on the request, so exempting them needs new plumbing. That is a follow-up.
4. **The design below is approved** as the basis for this spec.

## Problem

TASK-13437 says the other LLM paths are "counted in `llm_usage_log` but not gated". The verification on 2026-10-09 found the gap is larger: most paths are not counted at all.

- **Six recorders, few paths.** `llm_usage_log` is written by `log_llm_usage` (`core/Usage/usage_tracker.py:278`) and by nothing else. Its callers are:
  - `chat_service.py:6133`, `:6739` and `:6798`, inside `execute_streaming_call` and `_execute_non_stream_call_impl`. Only the `/chat/completions` pipeline reaches these (`endpoints/chat.py:5480`, `:5782`; `chat_service.py:7670`).
  - The claims engine (`claims_engine.py:712`).
  - Router accounting (`LLM_Calls/routing/accounting.py`), which records only the router model's own call.
  - The embeddings endpoint (`embeddings_v5_production_enhanced.py:4117`, `:4799`).
- **The shared call layer records nothing.** `perform_chat_api_call` and `perform_chat_api_call_async` (`chat_service.py:2624`, `:2652`) go straight from the request to the adapter. 38 files call them, including:
  - RAG generation (`RAG/rag_service/generation.py:527`);
  - workflow LLM steps (`Workflows/adapters/llm/llm.py:212`);
  - character-chat replies (`character_chat_sessions.py:7177`).
- **None of these record usage.** RAG, Workflows, character chat, Evaluations, Prompt Management and Persona contain no `log_llm_usage` call.
- **23 more files bypass the shared layer.** They get an adapter from the registry and call `chat`, `achat`, `stream` or `astream` directly: evaluations, summarization, web search, Data Tables, chunking, notes graph, speech chat, the realtime pipeline, BYOK key testing and others.
  - Nothing outside `core/LLM_Calls` calls the legacy `chat_with_*` functions, so the registry is the only way to reach a chat provider.
- **One gate.** `limits.llm_tokens_per_month` is enforced inline in `create_chat_completion` (`endpoints/chat.py:4294`) and nowhere else.
- **Embeddings count against chat.** The counter is `SELECT COALESCE(SUM(total_tokens), 0) FROM llm_usage_log WHERE user_id = ? AND ts >= ?` (`AuthNZ/repos/usage_repo.py:259`), with no `operation` filter, so embedding tokens already count against the monthly LLM token quota.
- **No identity at the call layer.** The only user-scoped context variable is the Jobs row-level-security owner (`Jobs/manager.py:589`). The recorders read the user from their caller (`request.state.user_id`, `current_user.id`, `job_context.user_id`). The synchronous adapter pool (`Chat/bounded_daemon.py`) starts raw `threading.Thread`s, so a context variable would not reach it today.
- **Evaluation cost caps are dead (TASK-13436).**
  - Every `check_rate_limit` caller passes `estimated_cost=0.0`, and the actual cost recorded is always 0.0.
  - The monthly cap is never compared. `max_cost_per_month` is stored and reported, but no check reads it.
  - `GET /evaluations/rate-limits` reports the tier's cost numbers truncated with `int()`.
  - `Production_Hardening_Checklist.md:48` tells operators the cost caps stay on. They have never fired.

## Goals

1. Every LLM generation call made on behalf of a user writes one `llm_usage_log` row attributed to that user, whichever module makes it.
2. With quotas on, every such call is checked against the user's resolved `limits.llm_tokens_per_month`, `limits.llm_cost_per_day_usd` and `limits.llm_cost_per_month_usd` before it starts.
3. No call is counted twice.
4. Embeddings stay recorded, but are excluded from the LLM counters.

## Non-goals

- **Embeddings quotas.** They go through a separate registry and stay as they are.
- **BYOK exemption.** Owner decision 3. Filed as a follow-up.
- **Per-request cost estimation.** The cost check compares spend so far against the cap. The call that crosses the cap completes, and the next call is refused.
- **Atomic budgets.** Counters are sums over `llm_usage_log` with no reservation, so concurrent calls can each pass a check that their sum exceeds. This is the same as the existing token check.
- **Request-rate limits.** Those belong to spec 1, the Resource Governor ingress limits.
- **Moving callers off direct adapter use.** The registry wrapper covers them where they are.

## Design

### 1. The LLM actor

A new module, `core/Usage/llm_actor.py`, holds one `ContextVar` with the identity of whoever an LLM call is made for. It holds `user_id`, `api_key_id`, `endpoint`, and an `exempt` flag. It provides:
- `set_llm_actor(...)`, which returns a reset token;
- `current_llm_actor()`;
- a context manager.

Where the actor is set:
- **HTTP requests:** where the request's user is resolved, in `get_request_user` (`AuthNZ/User_DB_Handling.py:1551`) and the `get_auth_principal` resolvers (`API_Deps/auth_deps.py:1556`, `AuthNZ/auth_principal_resolver.py:316`). These already populate `request.state.user_id`. Async dependencies run in the request's task, so the endpoint and any streaming response see the value.
- **WebSockets:** where the socket authenticates its user.
- **Jobs workers:** in `WorkerSDK` (`Jobs/worker_sdk.py`), for the duration of each handler call. The job row already carries `owner_user_id` (`worker_sdk.py:880`).
- **The synchronous adapter pool:** `bounded_daemon` starts threads with `contextvars.copy_context().run`, so the actor reaches synchronous adapter calls. `asyncio.to_thread` already copies context, and `asyncio.create_task` copies it at creation.
- **Exempt calls:** calls that must never be gated or attributed, such as provider key validation in `AuthNZ/byok_testing.py`, run inside `llm_actor(exempt=True)`.

With no actor set, a call is not gated. It is recorded with `user_id` null, as background calls are today.

### 2. One choke point: an accounting wrapper at the chat adapter registry

`ChatProviderRegistry.get_adapter` (`LLM_Calls/adapter_registry.py:181`) returns the adapter wrapped in an `AccountedChatProvider`.

What the wrapper does:
- **`chat` and `achat`:** check, call the adapter, then record from the response's `usage` block (`prompt_tokens`, `completion_tokens`, `total_tokens`). With no usage block, it records the existing estimate and marks the row `estimated`.
- **`stream` and `astream`:** check before the stream starts. The returned iterator records once, when the stream ends. That covers completion, a provider error, and the consumer closing the stream early (`GeneratorExit` or `aclose`). It uses a usage chunk when the provider sends one. Otherwise it records the estimate `chat_service` uses today (`estimate_source="stream_estimate"`).
- **Everything else** (attributes, `async_chat_is_native`, capabilities) is forwarded to the wrapped adapter, both reads and writes, so `_attach_internal_http_hooks` and capability checks behave as before.
- **One identity check changes.** `get_audited_call_policy_transport` (`adapter_registry.py:203`) checks `type(adapter) is OpenAIAdapter`. It reads the unwrapped adapter from the base registry, so strict transport is still recognized.

Both `perform_chat_api_call*` and the 23 direct callers get their adapters from the registry, so this covers every chat path with no caller changes.

The rejected alternative was to check and record inside `perform_chat_api_call*` and move the 23 direct callers onto it. That is more change, and a new direct caller would silently escape.

### 3. One recorder

The wrapper becomes the only writer of LLM generation rows. The two `chat_service` recorders are removed:
- `execute_streaming_call`'s `save_callback` call;
- `_record_dict_response_usage`.

Before deleting them, their extra fields move into the actor or onto the wrapper's row: conversation id, prompt fingerprints, `choice_count`, `estimate_source`, `remote_ip`, `user_agent` and `token_name`.

The router-accounting and claims-engine calls are removed too, because their model calls now go through the wrapper. Router rows keep their `operation`, set through the actor. Removing the old recorders is what keeps `/chat/completions` from being counted twice.

The embeddings recorders stay, with `operation="embeddings"`.

### 4. The check

Before each call with a non-exempt actor, the wrapper:
1. resolves the three keys with `user_quota(user_id, key)` (`None` means unlimited);
2. loads the counters (§5);
3. refuses the call when:
   - `tokens_this_month + prompt_estimate > llm_tokens_per_month`, where the prompt estimate is the same one `chat.py` computes today;
   - or `cost_today >= llm_cost_per_day_usd`;
   - or `cost_this_month >= llm_cost_per_month_usd`.

A refusal raises `LLMQuotaExceededError`, carrying the category, the current figure, the limit and `retry_after`.
- **The error shape.** A global exception handler in `main.py` maps it to the 402 that `/chat/completions` returns today: `{"error": "limit_exceeded", "category": ..., "current": ..., "limit": ..., "message": ...}` with a `Retry-After` header.
- **Categories:** `llm_tokens_month`, `llm_cost_day` and `llm_cost_month`.
- **Messages:** "Monthly LLM token limit reached", "Daily LLM cost limit reached" and "Monthly LLM cost limit reached".
- **`Retry-After`:** the next UTC midnight for the day cap, and the start of the next UTC month for the monthly caps.
- **Streaming:** for streaming endpoints the check runs before any byte is sent, so the client gets a normal 402.

The inline check in `create_chat_completion` (`chat.py:4290-4310`) is deleted, so a chat completion isn't checked twice.

Lookup and counter failures fail open, with a warning, as in spec 2.

**Known limit.** Some modules catch every exception around their LLM call. They turn a refusal into their own error response, so the client sees that module's error instead of the 402. The tests pin the 402 on the main entry points (§Testing). Other paths are listed in the operator page as "refused, but may report a module-specific error".

### 5. Counters

These are new `AuthnzUsageRepo` methods. Each excludes embeddings with `operation <> 'embeddings'`, and each has a SQLite and a Postgres form:
- `sum_user_llm_tokens_since(user_id, since)`, the existing method, which gains the filter;
- `sum_user_llm_cost_since(user_id, since)`, which sums `total_cost_usd`.

The quota code reads them through `core/Usage/quota_checks.py` helpers:
- `llm_tokens_this_month`, which exists and keeps its name;
- `llm_cost_today`;
- `llm_cost_this_month`.

Costs come from the existing pricing catalog when the row is written (`Usage/pricing_catalog.py`, via `compute_costs`). Unknown models are billed at the catalog's flagged estimate rate, as today.

### 6. Keys

Two keys are added to `Config_Files/user_profile_catalog.yaml`: `limits.llm_cost_per_day_usd` and `limits.llm_cost_per_month_usd`.
- Each is a non-negative number, default `null`, editable by `platform_admin`, sensitivity `internal`.
- They follow the same precedence as every `limits.*` key: the user's own value, then the most generous team value, then the most generous org value.
- They are listed in the operator page's key table.

### 7. TASK-13436 removals

- **The evaluations rate limiter** (`core/Evaluations/user_rate_limiter.py`): delete `max_cost_per_day`, `max_cost_per_month`, the daily cost branch in `_reserve_request_usage_sync`, `_check_cost_limits`, the cost parameters of `check_rate_limit` and `record_actual_usage`, and the cost columns' reads.
  - The tables keep their columns; nothing new writes them.
  - The tier YAML and `for_tier` lose their cost entries.
- **`GET /evaluations/rate-limits`:** the cost fields are removed from `limits`, `usage` and `remaining`.
- **Frontend:** the `cost_per_day`, `cost_per_month`, `cost_today` and `cost_month` fields are removed from `apps/packages/ui/src/services/evaluations.ts`, along with the test fixture that uses them.
- **Docs:** the cost-cap line in `Production_Hardening_Checklist.md` is removed. Evaluation LLM spend is now capped by the general LLM quotas.

### 8. Reporting

- The profile `quotas` section gains an `llm` block: tokens used this month, cost today, cost this month, and each resolved limit, with `null` meaning unlimited.
- `Docs/Operations/Usage_Quotas.md` replaces "enforced on `/chat/completions` only" with the new behaviour, and adds the two cost keys.

## Testing

Each test fails before its change.

1. **Recording, by path.** A fake adapter registered in the registry reports a fixed `usage` block. Calls through each of these write exactly one row each, with the actor's user id:
   - `perform_chat_api_call_async`;
   - `perform_chat_api_call` (synchronous, through the bounded pool);
   - a direct `get_adapter(...).achat` call;
   - a streamed call consumed fully;
   - a streamed call closed early.
2. **No double count.** One `/chat/completions` request writes one generation row, both streaming and not streaming.
3. **The actor.**
   - An HTTP request sets it.
   - A `WorkerSDK` handler sees the job owner.
   - A synchronous adapter call in the bounded pool sees it.
   - `byok_testing` runs as exempt.
   - With no actor, the call runs and the row has a null user.
4. **The check.** With quotas on, a user at the token cap gets a 402 (`llm_tokens_month`) from each of these:
   - `/chat/completions`, both streaming and not streaming;
   - a character-chat reply;
   - RAG generation;
   - a workflow LLM step.

   The cost caps refuse once the counted cost reaches the cap, with categories `llm_cost_day` and `llm_cost_month`.
   - A cap of `0` blocks.
   - `null` is unlimited.
   - With quotas off, nothing is refused.
   - A counter error lets the call through.
5. **Embeddings excluded.** Embedding rows don't move the LLM token or cost counters.
6. **Precedence.** A team value applies when the user has none, and the most generous of two teams wins. This reuses the spec 2 resolver tests' pattern.
7. **Strict transport.** `get_audited_call_policy_transport("openai")` still returns the strict transport.
8. **Removals.**
   - `/evaluations/rate-limits` has no cost fields.
   - The evaluations limiter has no cost parameters.
   - The frontend typecheck passes.
9. **Postgres.** The counters and the recorder run against a real Postgres through the existing fixture.

## Delivery

Two PRs, so recording can be checked before anything is refused.

- **PR 1: identity and recording.** §1 to §3 and §5. No enforcement changes.
  - After this PR, every path writes rows, and `/chat/completions` keeps its existing inline check.
  - Includes ADR-065, "LLM calls are accounted at the chat adapter registry", if the number is free when the PR opens.
- **PR 2: the check, the keys and the removals.** §4 and §6 to §8. It deletes the inline `chat.py` check and closes TASK-13437 and TASK-13436.

## Risks

- **Attribute forwarding.** Code that sets attributes on an adapter and expects to read them back on the same object must see the same behaviour through the wrapper. The wrapper forwards both reads and writes, and tests cover `_attach_internal_http_hooks`.
- **Lost rows on abandoned streams.** The stream recorder runs in `finally`. A process kill mid-stream loses the row, as today.
- **Wider attribution than before.** Calls that recorded nothing now write rows. On an install with quotas on and token limits set, users reach their limits sooner than before, because their usage is now counted. The operator page says so.
