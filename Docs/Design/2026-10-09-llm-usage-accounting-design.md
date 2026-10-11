# LLM usage accounting: record and gate every LLM call at one point

- **Date:** 2026-10-09 (revised 2026-10-10 after an independent review against the code; see [Review changes](#review-changes))
- **Status:** Draft, for owner review
- **Backlog:** TASK-13437 (gate LLM tokens/month at every LLM entry point) and TASK-13436 (evaluation cost caps per user, re-scoped below). Both are follow-ups of spec 2, TASK-13434.
- **Builds on:** `Docs/Design/2026-10-02-usage-quota-posture-design.md` (spec 2) and ADR-064. Quotas are per-user `limits.*` values, resolved by `quota_resolver.user_quota`. Nothing is enforced unless `USAGE_QUOTAS_ENABLED` is on, `0` blocks, and errors fail open.

## Owner decisions (2026-10-09)

1. **A general LLM cost quota replaces the evaluation-only cost caps.** The new keys are `limits.llm_cost_per_day_usd` and `limits.llm_cost_per_month_usd`. They cover every LLM path, evaluations included. The dead evaluation cost caps are deleted.
2. **Embeddings do not count against the LLM quotas.** One document ingest must not use up a chat budget. If an embeddings quota is needed later, it gets its own key.
3. **BYOK calls are gated too, for now.** This matches `/chat/completions` today. Most providers don't mark BYOK credentials on the request, so exempting them needs new plumbing. That is a follow-up.

## Problem

TASK-13437 says the other LLM paths are "counted in `llm_usage_log` but not gated". Verification on 2026-10-09 found the gap is larger: most paths are not counted at all.

- **One writer, few callers.** `llm_usage_log` is written only by `log_llm_usage` (`core/Usage/usage_tracker.py:278`). Its callers are:
  - `chat_service.py:6133`, `:6739` and `:6798`, inside `execute_streaming_call` and `_execute_non_stream_call_impl`. Only the `/chat/completions` pipeline reaches these.
  - The claims engine (`claims_engine.py:712`), with operations `claims_extract` and `claims_verify`.
  - Router accounting (`LLM_Calls/routing/accounting.py:71`), which records the router model's own call as `<surface>_router`.
  - The embeddings endpoint (`embeddings_v5_production_enhanced.py:4117`, `:4799`), with `operation="embeddings"`.
- **The shared call layer records nothing.** `perform_chat_api_call` and `perform_chat_api_call_async` (`chat_service.py:2624`, `:2652`) go straight from the request to the adapter. About 39 files call them, including:
  - RAG generation (`RAG/rag_service/generation.py:527`);
  - workflow LLM steps (`Workflows/adapters/llm/llm.py:212`);
  - character-chat replies (`character_chat_sessions.py:7177`).
- **About 23 more files bypass the shared layer.** They get an adapter from the registry (`get_adapter` or `adapter_utils.get_adapter_or_raise`) and call `chat`, `achat`, `stream` or `astream` directly: evaluations, summarization, web search, Data Tables, chunking, notes graph, speech chat, the realtime pipeline, BYOK key testing and others.
  - No code outside `core/LLM_Calls` constructs an adapter class or calls the legacy `chat_with_*` functions, so every chat call ends in a `ChatProvider` method.
- **One gate.** `limits.llm_tokens_per_month` is enforced inline in `create_chat_completion` (`endpoints/chat.py:4290-4309`) and nowhere else.
- **Embeddings count against chat.** The counter is `SELECT COALESCE(SUM(total_tokens), 0) FROM llm_usage_log WHERE user_id = ? AND ts >= ?` (`AuthNZ/repos/usage_repo.py:259`). It has no `operation` filter.
- **Evaluation cost caps are dead (TASK-13436).**
  - Every `check_rate_limit` caller passes `estimated_cost=0.0`, and the cost recorded afterwards is always 0.0.
  - The monthly cap is never compared. `max_cost_per_month` is stored and reported, but no check reads it.
  - `GET /evaluations/rate-limits` reports the tier's cost numbers truncated with `int()`.
  - `Docs/User_Guides/Server/Production_Hardening_Checklist.md:48` tells operators the cost caps stay on. They have never fired.

Four facts shape the design:

- **Identity already exists for most roots.** `core/DB_Management/scope_context.py:101` holds a `ContextVar`, `_SCOPE_CTX`, that carries `user_id`. `activate_authenticated_content_scope` sets it on every HTTP authentication (`User_DB_Handling.py:1548`; called from `auth_principal_resolver.py:328,376,572`, `auth_deps.py:388` and `content_org_deps.py:30`). Several workers set it with the job owner (`media_ingest_jobs_worker.py:482`, `claims_job_handlers.py:406`, others).
- **Adapters are mostly synchronous.** `ChatProvider.chat` and `.stream` are synchronous and abstract (`LLM_Calls/providers/base.py:133,137`). `achat` and `astream` raise `NotImplementedError` unless an adapter overrides them. Sync calls run in three places:
  - on raw daemon threads from the bounded pool (`chat_service.py:2690-2695`, `Chat/bounded_daemon.py:142`);
  - on a daemon thread that iterates a sync stream (`LLM_Calls/streaming.py:265-296`);
  - on the caller's own thread, for `perform_chat_api_call` and direct callers.
- **The AuthNZ pool belongs to one event loop.** On Postgres, `get_db_pool()` closes the live pool and recreates it when called from a different loop (`AuthNZ/database.py:1816-1829`). The quota resolver, the counters and `log_llm_usage` are all async and use that pool. The workflow engine runs every run on its own thread with its own loop (`Workflows/engine.py:2338-2347`).
- **Modules catch errors around their LLM call.** In `/chat/completions`, a refusal raised inside the provider call would not reach the client as a 402.
  - The streaming path turns an `HTTPException` into an HTTP 200 carrying an SSE error (`chat_service.py:5510-5553`).
  - The non-streaming path rewrites unknown errors into a generic provider failure (`:3503-3525`, `:6978-6981`).
  - Today's 402 comes from the check that runs before dispatch.

## Goals

1. Every LLM generation call made on behalf of a known user writes one `llm_usage_log` row attributed to that user, whichever module makes it.
2. With quotas on, every such call is checked against the user's resolved `limits.llm_tokens_per_month`, `limits.llm_cost_per_day_usd` and `limits.llm_cost_per_month_usd` before it starts.
3. No call is counted twice, and the rows the existing recorders write keep their shape.
4. Embeddings stay recorded, but are excluded from the LLM quota counters.
5. `/chat/completions` behaves as today, apart from also checking the two cost keys.

## Non-goals

- **Embeddings quotas.** Embeddings go through a separate registry and stay as they are.
- **BYOK exemption.** Owner decision 3. Filed as a follow-up.
- **Per-request cost estimation.** The cost check compares spend so far against the cap. The call that crosses the cap completes, and the next call is refused.
- **Atomic budgets.** Counters are sums over `llm_usage_log` with no reservation, so concurrent calls can each pass a check that their sum exceeds. The existing token check has the same limit.
- **Recording calls with no known user.** They are not recorded today and stay unrecorded.
- **Virtual-key budgets for the new paths.** New rows carry no `key_id`, so per-key budgets keep counting what they count today.
- **Changing what the hosted billing plan or the Resource Governor counts.** See [Readers of `llm_usage_log`](#6-readers-of-llm_usage_log).
- **Removing the existing recorders.** They stay, and the new recorder skips the calls they cover.
- **Request-rate limits.** Those belong to spec 1.

## Design

### 1. Who a call is for

A new module, `core/Usage/llm_call_context.py`, answers "which user is this LLM call for, and how should it be accounted?".

`current_llm_user_id()` returns, in order:
1. the user set explicitly with `llm_call_context(user_id=...)`;
2. otherwise `get_scope().user_id` from the existing content scope;
3. otherwise `None`.

`llm_call_context(...)` is a context manager over one `ContextVar`. Its fields overlay the current value:

| Field | Meaning |
|---|---|
| `user_id` | The user the call is for, when the content scope doesn't already say. |
| `operation` | The row's `operation`. Default `llm_call`. |
| `endpoint` | The row's `endpoint`. Default `internal`. |
| `exempt` | Don't check and don't record. For provider key validation and setup verification. |
| `records_itself` | The caller writes its own row, so don't record. |
| `prechecked` | The caller already checked the quota, so don't check again. |

With no user, a call is not checked and not recorded.

**Roots that need an explicit set point,** because the content scope doesn't reach them:
- **Jobs workers on `WorkerSDK`:** both handler call sites (`Jobs/worker_sdk.py:719`, `:955`) run the handler inside `llm_call_context(user_id=as_quota_user_id(owner))`. The owner is the job row's `owner_user_id`, a string that can be empty. This covers scheduled automations, which run under a `WorkerSDK` worker.
- **The workflow engine:** `start_run` sets the context from `run.user_id` (`Workflows/engine.py:530`, `:596`).
  - It has to be set there because `_spawn` runs each run on a new thread with its own loop (`:2337-2347`), and `threading.Thread` doesn't copy context.
  - `start_run` is also awaited directly for child runs (`Workflows/adapters/control/orchestration.py:74`).
- **The chat request queue:** `QueuedRequest` captures the LLM call state when the request is enqueued, and the worker runs the processor with it (`Chat/request_queue.py:366`, `:672`). The queue's worker tasks otherwise carry the context of whoever started the queue. Without this, a queued `/chat/completions` (`CHAT_QUEUED_EXECUTION`, off by default) would lose its `prechecked` and `records_itself` flags and be counted twice.
- **WebSockets:** no socket authentication sets the content scope. Each sets the context after it authenticates:
  - the four audio routes, through their shared helper `_audio_ws_authenticate` (`Audio/streaming_service.py:382`);
  - `voice_assistant.py`;
  - `agent_client_protocol.py` (two routes);
  - `acp_multiplex.py`;
  - `prompt_studio/prompt_studio_websocket.py` (two routes; its user id is a string).

**Places that lose context today, and their fix:**
- **The bounded daemon pools.** `BoundedDaemonPool.start` (`Chat/bounded_daemon.py:121-150`) starts raw threads. It is the one entry point for the sync adapter pool, the stream pool and the cleanup pool. It carries the LLM call state into the thread: the resolved user and the context fields. It does not copy the whole context, because other context variables, such as the provider-error privacy scope, are not passed to these threads today, and copying everything would change that.
- **`loop.run_in_executor`** doesn't copy context. The sites that lead to LLM calls carry the LLM call state the same way: `Evaluations/eval_runner.py:1398-1409` (G-Eval) and `Scheduler/base/registry.py:84-92,248-258`.
- `asyncio.to_thread` and `asyncio.create_task` already copy context. Async FastAPI dependencies run in the request's task, so an endpoint and its streaming response body see what authentication set.

**Exempt calls:** `AuthNZ/byok_testing.py:563,649` and `Setup/first_chat_verifier.py` run inside `llm_call_context(exempt=True)`.

### 2. One recording point: the provider base class

`ChatProvider.__init_subclass__` wraps each subclass's own `chat`, `stream`, `achat` and `astream` methods with accounting. Every chat call ends in one of those four methods, so this covers the shared call layer and the direct callers with no caller changes.

Why the base class and not a wrapper object returned by the registry:
- **Identity and type stay as they are.** `registry.get_adapter(x)` returns the same cached instance as before. Seven existing assertions depend on that:
  - `is canonical_adapter` in three Chat tests and `test_adapter_registry_wrapper_migration.py:73`;
  - `isinstance(adapter, OpenAIAdapter)` in `test_llm_streaming_and_security.py:722` and `test_speech_chat_service.py:1119,1441`.
- **Strict transport is untouched.** `get_audited_call_policy_transport` checks `type(adapter) is OpenAIAdapter` (`adapter_registry.py:203`), which keeps working.
- **No cache to maintain.** A wrapper would need its own cache, invalidated with the registry's, and write-forwarding for tests that monkeypatch an adapter's methods.
- **It covers adapters however they are obtained.**

Rules for the wrapping:
- **Only methods a subclass defines itself are wrapped.** The base class's `achat` and `astream`, which raise `NotImplementedError`, stay unwrapped, so the existing fallback to the sync methods is not accounted as a call.
- **A re-entrancy guard** (a `ContextVar`) makes only the outermost call account. An adapter's `achat` that calls its own `chat`, or a subclass that calls `super().chat()`, is counted once.
- **A test that monkeypatches a method** on an adapter instance or class replaces the wrapped method, so that call is not accounted. Accounting tests use a fake `ChatProvider` subclass.

What the wrapped methods do, when `current_llm_user_id()` is set and the context is not `exempt`:
- **`chat` and `achat`:** check (§4) unless `prechecked`; call the adapter; record (§3) unless `records_itself`.
- **`stream` and `astream`:** return an iterator that checks before it yields the first item (unless `prechecked`) and records once when it ends (unless `records_itself`). "Ends" covers completion, a provider error, and the consumer closing it early.
  - The wrapped methods stay iterator-returning functions, so `for x in adapter.stream(...)` and `async for x in adapter.astream(...)` work as today.
  - Because the check is lazy for direct stream callers, the shared call layer checks eagerly (§4).

### 3. Recording

The wrapper records with `log_llm_usage`, passing the context's user, `operation` and `endpoint`, the provider and model from the request, the status, and the latency.

- **Non-streaming:** tokens come from the response's `usage` block, through the existing `normalize_llm_usage`. With no usage block, it records the existing estimate and marks the row `estimated`.
- **Streaming:** tokens come from a usage chunk when the provider sends one. Otherwise it records the estimate `chat_service` uses today (`estimate_source="stream_estimate"`).

**The existing recorders stay.** They pass fields the base class can't know: the conversation id, the request's IP and user agent, `choice_count`, the router's scope. Each one's model call runs inside `llm_call_context(records_itself=True)`, so the wrapper skips it:
- the `/chat/completions` pipeline (`execute_streaming_call` and `_execute_non_stream_call_impl`);
- the three router call sites (`chat.py:1617`, `chat_route_resolver.py:187`, `character_chat_sessions.py:1046`);
- the claims engine's two analyze calls, extract and verify (`claims_engine.py:837`, `:1202`), which its four recorder sites account for. Its third LLM call, the APS proposition extractor (`:1486`), has no recorder, so it is not marked and the base class records it.

This keeps the claims budget guard working, which filters on the claims operations (`claims_service.py:1335-1380`). Router analytics, which break down by `<surface>_router`, keep working too.

### 4. The check

One function, `enforce_llm_quota(user_id, request)` in `core/Usage/quota_checks.py`:
1. resolves the three keys with `user_quota(user_id, key)` (`None` means unlimited);
2. loads only the counters whose limit is set (§5);
3. raises when:
   - `tokens_this_month + prompt_estimate > llm_tokens_per_month`, where the prompt estimate is the one `chat.py` computes today;
   - or `cost_today >= llm_cost_per_day_usd`;
   - or `cost_this_month >= llm_cost_per_month_usd`.

Lookup and counter failures fail open, with a warning, as in spec 2.

It raises `LLMQuotaExceededError`, a `ChatAPIError` subclass with `status_code=402`. The error carries the category, the current figure, the limit and `retry_after`.
- **Categories:** `llm_tokens_month`, `llm_cost_day` and `llm_cost_month`.
- **Messages:** "Monthly LLM token limit reached", "Daily LLM cost limit reached" and "Monthly LLM cost limit reached".
- **`retry_after`:** the next UTC midnight for the day cap, and the start of the next UTC month for the monthly caps.

It is called from three places:

| Where | Why | Result for the caller |
|---|---|---|
| `create_chat_completion`, before dispatch (`chat.py:4290-4309`) | The existing check, extended to the two cost keys. | The same 402 body and `Retry-After` as today. The pipeline then runs inside `llm_call_context(prechecked=True, records_itself=True)`. |
| The top of `perform_chat_api_call` and `perform_chat_api_call_async` | An eager check before any provider call, including a stream. | `LLMQuotaExceededError`. The nested adapter call runs as `prechecked`. |
| The wrapped provider methods (§2) | Direct adapter callers. | `LLMQuotaExceededError`, raised at the first item for streams. |

**What the client sees.** Because the error is a `ChatAPIError` with status 402, modules that pass a `ChatAPIError`'s status through return a 402: character chat (`character_chat_sessions.py:7249-7278`) and RAG (`rag_unified.py:2171`). A workflow step has no HTTP response; the step fails with the error and the run records it. Modules that catch every exception report their own error. The operator page says which entry points return a 402.

### 5. Accounting runs on one event loop

The check and the recorder are async and use the AuthNZ pool, which is bound to one loop. They must never run on another loop, and never through `asyncio.run`.

- **The accounting loop** is the server's loop, captured once at application startup. A captured loop that has closed, or has stopped running, is treated as missing: a hand-off to a stopped loop would be accepted and never run.
- **From an async method on the accounting loop:** await directly.
- **From an async method on another loop** (a workflow engine thread): hand off with `asyncio.run_coroutine_threadsafe` and await the wrapped future.
- **From a sync method on a thread that is not the accounting loop's thread:** hand off with `run_coroutine_threadsafe`.
  - The check waits with a short timeout and fails open when it expires.
  - The record is fire-and-forget, with a callback that logs a failure.
- **From a sync method on the accounting loop's own thread** (a caller that blocks the loop): the check is skipped, because waiting would deadlock, and the record is scheduled with `loop.create_task`. This is a documented fail-open case.
- **Streams** capture the loop when `stream()` is called. The iterator may run and close on a daemon thread.
- **With no accounting loop captured** (unit tests without the application lifespan): use the running loop if there is one, otherwise skip.

This follows the hand-off pattern already used at `LLM_Calls/streaming.py:233` and `chat_service.py:548-556`.

### 6. Counters and other readers

Two counters back the cost keys, next to the existing token counter. All three exclude embeddings with one shared SQL predicate, `COALESCE(operation, '') <> 'embeddings'`. `operation` is nullable, so a plain `<>` would drop rows with no operation.

- `sum_user_llm_tokens_since(user_id, since)`: the existing `AuthnzUsageRepo` method gains the predicate.
- `sum_user_llm_cost_since(user_id, since)`: new; sums `total_cost_usd`.
- One SQL string serves both backends, as the existing counter does. Only the bound timestamp differs.
- The quota code reads them through `quota_checks` helpers: `llm_tokens_this_month` (existing), `llm_cost_today` and `llm_cost_this_month`.
- The `(user_id, ts)` index that makes these sums cheap already exists on both backends.

Costs come from the pricing catalog when the row is written (`Usage/pricing_catalog.py`, via `compute_costs`). Unknown models are billed at the catalog's flagged estimate rate, as today.

#### Readers of `llm_usage_log`

More rows are written after this change. Nothing below changes its own query.

| Reader | Counts embeddings? | What changes |
|---|---|---|
| `limits.llm_tokens_per_month` and the new cost keys | No (new) | Counts every user-attributed LLM call. |
| Hosted org billing plan (`Billing/enforcement.py:406-446`) | Yes, as today | Counts RAG, workflow, evaluation and other calls it could not see before. Runs only on the hosted product. |
| Virtual-key budgets (`virtual_keys.py:100`, `usage_repo.py:265-335`) | Yes, as today | Unchanged: new rows have no `key_id`. |
| Daily ledger shadow write, read by Resource Governor daily token caps (`usage_tracker.py:470-501`) | Yes, as today | More token rows reach the ledger. RG daily token caps are off by default. |
| Claims budget guard | n/a (claims operations only) | Unchanged: the claims recorder stays. |
| Router analytics, admin usage reports, `llm_usage_daily` aggregation | Yes, as today | Reports show the calls that were invisible before, under `operation="llm_call"`. |

#### Two settings that can silently disable the quotas

- **`LLM_USAGE_ENABLED`** (default on): when it is off, `log_llm_usage` writes nothing, the counters read zero, and no LLM quota ever refuses.
- **`LLM_USAGE_LOG_RETENTION_DAYS`** (default 180, accepts 1): under 31 days, the prune job removes rows the monthly counters need.

At startup, when quotas are on and either setting defeats them, the server logs a warning in the same style as spec 2's startup warning. The operator page documents both.

### 7. Keys

Two keys are added to `Config_Files/user_profile_catalog.yaml`: `limits.llm_cost_per_day_usd` and `limits.llm_cost_per_month_usd`.
- Each is `type: number`, minimum 0, default `null`, editable by `platform_admin`, sensitivity `internal`.
- They follow the same precedence as every `limits.*` key.
- The description of `limits.llm_tokens_per_month` (`user_profile_catalog.yaml:280`) stops saying it is enforced on `/chat/completions` only.

### 8. TASK-13436 removals

These change in one commit, because `for_tier`, the YAML and the dataclass must agree:
- **`core/Evaluations/user_rate_limiter.py`:** `max_cost_per_day`, `max_cost_per_month`, the daily cost branch in `_reserve_request_usage_sync`, `_check_cost_limits`, the cost parameters of `check_rate_limit` and `record_actual_usage`, and the reads of the cost columns. The tables keep their columns; nothing new writes them.
- **`core/Evaluations/config_manager.py:50-65,489`:** the `RateLimitConfig` cost fields and the YAML validation entries.
- **`Config_Files/evaluations_config.yaml`:** the tier cost entries.
- **`GET /evaluations/rate-limits`** (`evaluations_unified.py:838-870`): the cost fields leave `limits`, `usage` and `remaining`.
- **Callers:** the seven `check_rate_limit` and five `record_actual_usage` call sites in `evaluations_unified.py` drop their cost arguments.
- **Frontend:** the `cost_per_day`, `cost_per_month`, `cost_today` and `cost_month` fields in `apps/packages/ui/src/services/evaluations.ts:78-85`, and the fixture in `EvaluationComponentAlerts.design-system.test.tsx:124-131`.
- **Tests that reference the cost fields:**
  - `tests/Evaluations/integration/test_rate_limits_endpoint.py`;
  - `tests/Config/test_module_yaml_integration.py`;
  - `tests/Evaluations/test_production_features_integration.py`;
  - `tests/Evaluations/unit/test_user_rate_limiter_minute_exact_and_reset.py`;
  - `tests/Resource_Governance/test_rg_cutover_evals_authnz_character_web.py`;
  - `tests/Evaluations/test_evaluations_core_hardening.py`;
  - `tests/Evaluations/test_evaluations_stage2_user_isolation_and_usage_accounting.py`;
  - `tests/Evaluations/test_evaluations_stage3_batch_failfast_and_metrics_none.py`.
- **Docs:** the cost-cap clause in `Docs/User_Guides/Server/Production_Hardening_Checklist.md:48`, with its `Docs/Published` mirror.

Evaluation LLM spend is capped by the general LLM quotas from then on.

### 9. Reporting

- The profile `quotas` section gains an `llm` block: tokens used this month, cost today, cost this month, and each resolved limit, with `null` meaning unlimited.
- `Docs/Operations/Usage_Quotas.md:68` replaces "enforced on `/chat/completions` only" with the new behaviour. It adds:
  - the two cost keys;
  - the entry points that return a 402;
  - the two settings in §6;
  - the fail-open cases in §5.

## Testing

Each test fails before its change. A fake `ChatProvider` subclass reports a fixed `usage` block.

1. **Recording, by path.** With a user in context, each of these writes exactly one row with that user:
   - `perform_chat_api_call_async`, through the bounded pool;
   - `perform_chat_api_call` on a plain thread with no loop;
   - a direct `adapter.chat` call on a thread;
   - a direct `await adapter.achat` call;
   - a stream consumed fully;
   - a stream closed early.
2. **No double count.** One `/chat/completions` request writes one generation row, both streaming and not streaming, and also with `CHAT_QUEUED_EXECUTION` on. A router call writes one `<surface>_router` row. A claims call writes one claims row, and the claims budget guard still sees it.
3. **Re-entrancy.** An adapter whose `achat` calls its own `chat` writes one row.
4. **Who the call is for.**
   - An HTTP request's content scope is used.
   - A `WorkerSDK` handler sees the job owner.
   - A workflow LLM step sees the run owner.
   - A G-Eval evaluation sees the user.
   - A call in the bounded pool sees the context.
   - `byok_testing` and `first_chat_verifier` neither check nor record.
   - With no user, the call runs and writes nothing.
5. **One loop.** A recorder call from a thread running its own event loop leaves `DatabasePool._loop` unchanged. A sync call on the accounting loop's thread skips the check and still records.
6. **The check.** With quotas on and the user at the token cap:
   - `/chat/completions` returns the same 402 body as today, both streaming and not streaming;
   - a character-chat reply and RAG generation return a 402;
   - a workflow LLM step fails with `LLMQuotaExceededError`, and the run records it;
   - a direct adapter call raises `LLMQuotaExceededError`.

   For the cost caps:
   - they refuse once the counted cost reaches the cap, with categories `llm_cost_day` and `llm_cost_month`;
   - a cap of `0` blocks, and `null` is unlimited;
   - with quotas off, nothing is refused;
   - a counter error lets the call through.
7. **Embeddings excluded.** Embedding rows, and rows with a NULL `operation`, are handled correctly: the first don't move the counters and the second do.
8. **Precedence.** A team value applies when the user has none, and the most generous of two teams wins.
9. **Unchanged behaviour.** `get_audited_call_policy_transport("openai")` still returns the strict transport. `registry.get_adapter(x) is registry.get_adapter(x)`.
10. **Startup warnings.** Quotas on with `LLM_USAGE_ENABLED` off, or with retention under 31 days, logs the warning.
11. **Removals.** `/evaluations/rate-limits` has no cost fields, the evaluations limiter has no cost parameters, and the frontend typecheck passes.
12. **Postgres.** The counters and the recorder run against a real Postgres through the existing fixture.

## Delivery

Four PRs, so recording can be checked before anything new is refused.

- **PR 1a: the mechanism and recording.** No new refusals. After it, every LLM call made in an HTTP request, or in a worker that already sets the content scope, is recorded. It contains:
  - the call context;
  - the single-loop hand-off (§5);
  - recording in the provider base class (§2, §3);
  - the state carried through the bounded pools and the chat request queue;
  - the `records_itself` and `exempt` markings;
  - the embeddings predicate in §6;
  - ADR-067, "LLM calls are accounted in the chat provider base class", if the number is free when the PR opens.
- **PR 1b: the remaining roots in §1.** `WorkerSDK`, the workflow engine, the WebSocket sites and the two executor sites.
- **PR 2: the check and the keys.** §4, the cost counters and startup warnings in §6, §7 and §9. Closes TASK-13437.
- **PR 3: the evaluation cost-cap removals.** §8. It has no dependency on the other two beyond the cost quota existing. Closes TASK-13436.

## Risks

- **Fail-open cases.** A sync adapter call made on the event loop's own thread is recorded but not checked. A timed-out check lets the call through. Both are logged.
- **Lost rows on abandoned streams.** The stream recorder runs when the iterator closes. A process kill mid-stream loses the row, as today.
- **Wider attribution than before.** Calls that recorded nothing now write rows. On an install with quotas on and token limits set, users reach their limits sooner. The hosted billing plan and admin reports also see more usage. The operator page says so.
- **Metaprogramming in the base class.** `__init_subclass__` wrapping is less obvious than an explicit call. It is confined to one function in `providers/base.py`, with the re-entrancy guard and the tests in §Testing 3 and 9.

## Review changes

An independent review against the code on 2026-10-10 found 3 blockers and 9 important issues in the first draft. This revision changes:

- **Where accounting lives.** The provider base class, not a wrapper object returned by the registry. The wrapper broke seven identity and type assertions and the strict-transport check.
- **Identity.** The existing content scope supplies the user. The first draft said no user identity existed at the call layer; that was wrong.
- **Sync and async.** §5 is new. The first draft did not say how a synchronous adapter method runs an async check and recorder against a pool bound to one loop.
- **Where the 402 comes from.** The pre-dispatch check in `/chat/completions` stays, and the shared call layer checks eagerly. The first draft deleted the pre-dispatch check, which would have turned a streaming refusal into an HTTP 200 with an SSE error.
- **The existing recorders stay.** The first draft removed them, which needed their fields moved and would have broken the claims budget guard and router analytics.
- **Roots.** The workflow engine, the chat request queue, two `run_in_executor` sites and five WebSocket sites are named.
- **Other readers.** §6 lists them, and the two settings that can silently disable the quotas get a startup warning.
- **The NULL trap.** The embeddings predicate uses `COALESCE`.
- **The removal list** in §8 is complete, and is its own PR.
