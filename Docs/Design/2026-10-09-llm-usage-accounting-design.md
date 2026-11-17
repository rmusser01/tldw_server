# LLM usage accounting: record and gate every LLM call at one point

- **Date:** 2026-10-09 (revised 2026-10-10 after an independent review against the code; see [Review changes](#review-changes))
- **Status:** Approved; PR 1a (#3222) and PR 1b (#3224) merged; PR 2a (#3230) merged; PR 2b (#3234) merged; PR 3 (#3236) merged.
- **Backlog:** TASK-13437 (gate LLM tokens/month at every LLM entry point) and TASK-13436 (evaluation cost caps per user, re-scoped below). Both are follow-ups of spec 2, TASK-13434.
- **Builds on:** `Docs/Design/2026-10-02-usage-quota-posture-design.md` (spec 2) and ADR-064. Quotas are per-user `limits.*` values, resolved by `quota_resolver.user_quota`. Nothing is enforced unless `USAGE_QUOTAS_ENABLED` is on, `0` blocks, and errors fail open.

## Owner decisions (2026-10-09)

1. **A general LLM cost quota replaces the evaluation-only cost caps.** The new keys are `limits.llm_cost_per_day_usd` and `limits.llm_cost_per_month_usd`. They cover every LLM path, evaluations included. The dead evaluation cost caps are deleted.
2. **Embeddings do not count against the LLM quotas.** One document ingest must not use up a chat budget. If an embeddings quota is needed later, it gets its own key.
3. **BYOK calls are gated too, for now.** This matches `/chat/completions` today. Most providers don't mark BYOK credentials on the request, so exempting them needs new plumbing. That is a follow-up. *Superseded on 2026-10-26 by ADR-083: a call on the user's own key is recorded but neither counted nor checked.*

## Owner decisions (2026-10-12)

4. **The revised §4 is approved:** one `HTTPException` 402, checked at the request boundary for the main entry points and at the call everywhere else, delivered as PR 2a then PR 2b.
5. **Local providers cost nothing by default.** The pricing catalog gives each one a zero default price that an operator can change (§6). Every reader of stored cost, not only the new limits, stops counting local calls.

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
  - The other entry points behave the same way, each with its own error mapping (§4a).

## Goals

1. Every LLM generation call made on behalf of a known user writes one `llm_usage_log` row attributed to that user, whichever module makes it.
2. With quotas on, LLM work for such a user is checked against the user's resolved `limits.llm_tokens_per_month`, `limits.llm_cost_per_day_usd` and `limits.llm_cost_per_month_usd` before it starts: once at the request boundary for the main HTTP entry points, and at each call everywhere else.
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
| `prechecked` | The quota was already checked for this work, so don't check again. Set by the boundary check for the rest of the request's task (§4a), and by the shared call layer for its nested adapter call (§4b). |

With no user, a call is not checked and not recorded.

**Roots that need an explicit set point,** because the content scope doesn't reach them:
- **Jobs workers on `WorkerSDK`:** both handler call sites (`Jobs/worker_sdk.py`) run each handler under `llm_call_root(owner)`. The owner is the job row's `owner_user_id`, a string that can be empty.
  - Scheduled automations do not run under `WorkerSDK`. `services/agent_task_jobs_worker.py` has its own loop and sets the root itself. So does `services/audio_jobs_worker.py`.
  - The `watchlists_enrich_output` Scheduler task sets the root from its payload's user.
- **The workflow engine:** `start_run` and `continue_run` both set the root from `run.user_id` (`Workflows/engine.py`).
  - It has to be set there because `_spawn` runs each run on a new thread with its own loop, and `threading.Thread` doesn't copy context.
  - `start_run` is also awaited directly for child runs (`Workflows/adapters/control/orchestration.py:74`).
- **The chat request queue:** `QueuedRequest` captures the LLM call state when the request is enqueued, and the worker runs the processor with it (`Chat/request_queue.py:366`, `:672`). The queue's worker tasks otherwise carry the context of whoever started the queue. Without this, a queued `/chat/completions` (`CHAT_QUEUED_EXECUTION`, off by default) would lose its `prechecked` and `records_itself` flags and be counted twice.
- **WebSockets:** no socket authentication sets the content scope, so a socket that makes chat LLM calls binds its user after it authenticates.
  - The four audio routes and the two realtime routes do, in their shared helper `_audio_ws_authenticate` (`Audio/streaming_service.py`). The realtime routes reach it through `Audio/Realtime/auth.py`.
  - The voice assistant socket reaches an LLM only through a workflow run, which the workflow engine covers.
  - The other sockets (ACP, prompt studio, persona, meetings, sandbox, watchlists, workflows) make no chat LLM call and are left alone.
- **MCP tool calls:** MCP authentication sets no content scope, so the tool runtime sets the user from the call's `RequestContext` around `module.execute_tool` (`MCP_unified/tool_execution/runtime.py`). It uses `llm_call_context`, not `llm_call_root`, so a tool that an admitted `/chat/completions` request runs keeps the request's `prechecked` mark. It also clears `records_itself`: the chat pipeline's own row covers only its model call, so the tool's calls are recorded (ADR-075).
- **The in-process claims rebuild:** each `ClaimsRebuildTask` carries its owner, and the worker runs it under `llm_call_root(owner)`. A quota refusal fails that task and the worker goes on.

**Places that lose context today, and their fix:**
- **The bounded daemon pools.** `BoundedDaemonPool.start` (`Chat/bounded_daemon.py:121-150`) starts raw threads. It is the one entry point for the sync adapter pool, the stream pool and the cleanup pool. It carries the LLM call state into the thread: the resolved user and the context fields. It does not copy the whole context, because other context variables, such as the provider-error privacy scope, are not passed to these threads today, and copying everything would change that.
- **`loop.run_in_executor`** doesn't copy context. The first draft named two sites. The survey found the pattern at many more: document insights, quiz generation, reading summaries, character-memory extraction, watchlist output summaries, the media processors, live meeting insights, Data Tables and G-Eval. Nearly all of them pass `None` as the executor, so one default executor that carries the LLM call state covers them. One request path does not: `/research/websearch` runs sub-query generation on a pool the endpoint module creates.
  - It is installed on the server loop at startup and by `llm_call_root` on any other loop.
  - The server runs on uvloop (`uvicorn[standard]`), whose loop does not expose its default executor, so the helper remembers the loops it has installed on.
  - The Scheduler registry's `run_in_executor` path is used only by tests.
  - Three pools the code creates itself carry the LLM call state at their submit: claims extraction (`Claims_Extraction/ingestion_claims.py`), the article extractor (`Web_Scraping/Article_Extractor_Lib.py`) and web search (`api/v1/endpoints/research.py`).
  - Two other pools already copy the whole context and needed no change: the web-extraction pool (`Web_Scraping/orchestration/executor.py`) and the email-archive pool in `Ingestion_Media_Processing/persistence.py`.
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
- **`chat` and `achat`:** check (§4b) unless `prechecked`; call the adapter; record (§3) unless `records_itself`.
- **`stream` and `astream`:** check before the provider method runs (unless `prechecked`), then return an iterator that records once when it ends (unless `records_itself`). "Ends" covers completion, a provider error, and the consumer closing it early.
  - The check cannot wait for the first item: three adapters send the request inside `stream()` (§4b).
  - The wrapped methods stay iterator-returning functions, so `for x in adapter.stream(...)` and `async for x in adapter.astream(...)` work as today.

### 3. Recording

The wrapper records with `log_llm_usage`, passing the context's user, `operation` and `endpoint`, the provider and model from the request, the status, and the latency.

- **Non-streaming:** tokens come from the response's `usage` block, through the existing `normalize_llm_usage`. With no usage block, it records the existing estimate and marks the row `estimated`.
- **Streaming:** tokens come from a usage chunk when the provider sends one. Otherwise it records the estimate `chat_service` uses today (`estimate_source="stream_estimate"`).
- **No adapter:** the Messages API native path (Anthropic, llama.cpp) posts to the provider itself, so no provider method runs. It records through the same functions, `record_direct_response` and `accounted_direct_stream` (ADR-075). An Anthropic stream's input tokens come from `message_start` and its output count from the last `message_delta`. An Anthropic-shaped usage block counts `cache_creation_input_tokens` and `cache_read_input_tokens` as prompt tokens, because its `input_tokens` is only the uncached remainder.

**The existing recorders stay.** They pass fields the base class can't know: the conversation id, the request's IP and user agent, `choice_count`, the router's scope. Each one's model call runs inside `llm_call_context(records_itself=True)`, so the wrapper skips it:
- the `/chat/completions` pipeline (`execute_streaming_call` and `_execute_non_stream_call_impl`);
- the three router call sites (`chat.py:1617`, `chat_route_resolver.py:187`, `character_chat_sessions.py:1046`);
- the claims engine's two analyze calls, extract and verify (`claims_engine.py:837`, `:1202`), which its four recorder sites account for. Its third LLM call, the APS proposition extractor (`:1486`), has no recorder, so it is not marked and the base class records it.

This keeps the claims budget guard working, which filters on the claims operations (`claims_service.py:1335-1380`). Router analytics, which break down by `<surface>_router`, keep working too.

### 4. The check

A refusal is made in two places. An endpoint that is about to do LLM work for a user checks at the request boundary (§4a), before it starts. Everything else is checked at the call (§4b). Both use one decision function and raise one error.

**The decision.** `llm_quota_refusal(user_id, requested_tokens)` in `core/Usage/quota_checks.py`:
1. resolves the three keys with `user_quota(user_id, key)` (`None` means unlimited);
2. loads only the counters whose limit is set (§6);
3. returns a refusal when:
   - `tokens_this_month + requested_tokens > llm_tokens_per_month`;
   - or `cost_today >= llm_cost_per_day_usd`;
   - or `cost_this_month >= llm_cost_per_month_usd`.

Otherwise it returns nothing. Lookup and counter failures fail open, with a warning, as in spec 2.

`/chat/completions` passes the prompt estimate it computes today. Every other caller passes 1, which refuses a user who is at or over the token limit and admits a user who is one token under it.

A refusal carries the category, the current figure, the limit, the message and `retry_after`.
- **Categories:** `llm_tokens_month`, `llm_cost_day` and `llm_cost_month`.
- **Messages:** "Monthly LLM token limit reached", "Daily LLM cost limit reached" and "Monthly LLM cost limit reached".
- **`retry_after`:** the next UTC midnight for the day cap, and the start of the next UTC month for the monthly caps.

**The error.** `LLMQuotaExceededError` is an `HTTPException` with status 402. Its `detail` is the body `/chat/completions` returns today, `{"error": "limit_exceeded", "category": ..., "current": ..., "limit": ..., "message": ...}`. It carries a `Retry-After` header, and the refusal's fields as attributes. A counter, `llm_quota_refusals_total{category}`, counts refusals.

It is an `HTTPException`, and not a `ChatAPIError`, because of how the endpoints handle each:
- **A `ChatAPIError` is reported as a provider failure.** RAG maps it to a 502 `provider_unavailable` (`RAG/rag_service/streaming_executor.py:114`). Document generation re-wraps it (`Chat/document_generator.py:721`) and the endpoint returns a 502. The study assistants return a 502. Summarization turns it into the text `"Error: Provider request failed for '<provider>'."`, which media processing stores as the summary. Sixteen endpoint handlers catch `ChatAPIError`, and each would need its own branch to return a 402.
- **An `HTTPException` reaches the client unchanged.** About 580 endpoint handlers begin with `except HTTPException: raise`, and the handlers above are among them. Summarization does not catch it either.
- **Core code already defines `HTTPException` subclasses.** `core/exceptions.py` has two.
- **It is not both.** In those handlers the `ChatAPIError` clause comes before the `HTTPException` one, so a class that is both would be reported as a provider failure.

#### 4a. At the request boundary

An endpoint asks before it starts LLM work, and answers 402 before it does anything else. Spec 2 does the same for RAG queries (`API_Deps/usage_quota_deps.py`), and the new functions live beside it.

- **`enforce_llm_quota(user_id, estimated_tokens=1)`** raises `LLMQuotaExceededError` when the decision refuses.
- **`require_llm_quota`** is the dependency form, for endpoints that always call an LLM.
- **An endpoint where LLM work is optional** calls `enforce_llm_quota` in its handler, once it knows the request asks for it.
- **Streaming requests are refused with a real 402.** The check runs before the streaming response is created.
- **Nothing has been done yet.** A media batch is refused before it is downloaded and transcribed, and a RAG search before it retrieves.

**An admitted request runs to completion,** even if it takes the user past a limit. After a passing check, the request's task is marked `prechecked`, so §4b does not refuse it part-way through.
- **The mark is a plain `ContextVar.set`,** like `bind_llm_call_user`. A `with llm_call_context(...)` block would not work: it resets when the function that opened it returns, which is before the endpoint runs after a dependency, and before a streaming response body runs.
- **It reaches** the endpoint body, a streaming response body, `BackgroundTasks`, `asyncio.to_thread`, the bounded pools, the default executor, the chat request queue, and the scraping queue jobs the request submits (revision of 2026-10-20). Work the request schedules with `BackgroundTasks` is admitted with the request.
- **It does not reach** a task created before the mark, the next request on the same connection, or a job the request enqueues in the Jobs queue. A job runs under `llm_call_root(owner)` and is checked by §4b when it runs; a media ingest job that asks for analysis checks itself once, when it starts (revision of 2026-10-20).
- **Why it matters for `/chat/completions`.** Without the mark, §4b could refuse an admitted request inside the chat pipeline. The pipeline would treat that as a provider outage: it records a provider failure, marks the provider unhealthy for every user after three of them, and tries another provider (`chat_service.py`, `provider_manager.record_failure`).

| Entry point | Checks |
|---|---|
| `/chat/completions` | Always. This is the existing check, extended to the two cost keys. |
| The Anthropic-compatible Messages API (`/messages`, `/v1/messages`) | Always. Its native path for Anthropic and llama.cpp never reaches a provider adapter, so only this check gates it. |
| Character chat reply (`character_chat_sessions.py`) | Always, at the top of the handler. The provider call is made before the streaming response is built. |
| RAG search, its batch and resume forms, its streaming form, advanced search and ablation (`rag_unified.py`) | When the effective request asks for answer generation. Advanced search generates by default; ablation does with answers on. |
| Document generation from a chat (`chat_documents.py`) | Always. |
| Flashcards and quizzes study assistants | Always, as a dependency. |
| `/evaluations/geval`, `/evaluations/rag` and `/evaluations/response-quality` | Always. |
| `/evaluations/batch` | When the evaluation type is `geval`, `rag` or `response_quality`. |
| Evaluation runs (`POST /evaluations/{eval_id}/runs`) | Always. The run is a task the request starts, so it is admitted with the request; checked at the call, a refused sample would be scored 0.0 and the run would complete. |
| Benchmark runs (`/evaluations/benchmarks/{name}/run`, `/benchmarks/{name}/run`) | Always. Checked only at the call, every item would be refused and counted as failed, and the run would still complete. |
| `/media/ingest-web-content` | When `perform_analysis` is on, for every scrape method, with or without a provider. Without one it makes no summary call, but it may extract with an LLM, and checked only at the call a refusal discarded the pages already fetched. |
| `/media/process-web-scraping` and `POST /watchlists/outputs` | When the request asks for summaries: `summarize_checkbox`, or `summarize`. Checked only at the call, a refusal discarded the pages or items the request had already fetched. |
| The six `/media/process-*` endpoints and `/media/add` | When `perform_analysis` is set and an `api_name` is given. |

Simple search (`/rag/simple`) no longer generates an answer it then discarded, so it makes no LLM call and needs no check.

- **A RAG search with answer generation off is not checked here and not marked.** The pipeline's optional LLM stages (query rewriting, expansion, HyDE, LLM reranking, gap analysis, critique, suggestions) are governed by §4b. For a user over quota they degrade as they do for any provider failure, and the response's stage metadata says so. Gap analysis and the agentic planner were the two stages that let a refusal escape; PR 2b made them degrade, and the three convenience handlers (`/rag/simple`, `/rag/advanced`, `/rag/ablate`) pass a 402 through.
- **An endpoint that enqueues a job has no boundary check:** `/media/ingest/jobs` and evaluation recipe runs. The job is checked when it runs: a media ingest job that asks for analysis once, when it starts; everything else by §4b.
- **Left to §4b:** these HTTP endpoints are not in the table: `/media/{id}/insights`, the writing and manuscript analysis endpoints, prompt improvement and Prompt Studio, reading, study suggestions, paper search, chunking, sharing, and `/translate`. So are background jobs (evaluation recipe runs, audio analysis jobs, and media ingest jobs that do not ask for analysis), workflow runs, WebSocket sessions (audio, realtime) and MCP tools. The operator page names both groups.

**Why these entry points need a boundary check.** With the error raised only at the call, each would do worse than a 402 before any work:
- **Streaming RAG** has already sent a 200 when the pipeline runs, so the client would get an error frame.
- **Character chat** catches `HTTPException` in a broad tuple and returns a 502 "Chat provider error" (`character_chat_sessions.py:304-323`).
- **`/evaluations/geval`** retries a failed call up to 10 times, with waits of up to 120 seconds (`Evaluations/ms_g_eval.py:477-483`).
- **`/evaluations/rag` and `/evaluations/response-quality`** count the failure against a circuit breaker that all users share.
- **Media processing** catches it for each item, after the item was downloaded and transcribed, and reports a failed item.

#### 4b. At the call

Background jobs, workflow runs, WebSocket sessions and any HTTP endpoint that is not in the table have no boundary check (a media ingest job that asks for analysis checks itself when it starts, revision of 2026-10-20). For those the call itself is refused with `LLMQuotaExceededError`. An MCP tool call is checked for the user its call names (§1). A refusal becomes the tool error `llm_quota_exceeded`, which the MCP module circuit breaker does not count.

| Where | Why |
|---|---|
| The top of `perform_chat_api_call` and `perform_chat_api_call_async` | To raise before the function's `try` block. Inside it, `privacy_safe_chat_error` rebuilds any exception as a generic provider error. When this check ran, the nested adapter call runs as `prechecked`; when it was skipped (§5), the nested call is not marked. |
| The wrapped provider methods (§2) | Direct adapter callers. The check runs before the provider method does. |

Rules:
- **Nothing is checked** when there is no user, or the context is `exempt` or `prechecked`.
- **A stream is checked when `stream()` or `astream()` is called, before the provider method runs.** Three adapters send the request inside `stream()`, before any item is read: moonshot, zai and cohere. Where the call is made on the server loop's own thread and cannot wait (§5), a sync stream is checked at its first item, on the thread that reads it.
- **What the caller sees.**
  - An HTTP endpoint that lets `HTTPException` through, as most do, returns the 402.
  - One that catches it in a broad tuple reports its own error.
  - A workflow step fails, and the run records the error.
  - A job fails as its handler decides. A media ingestion job that asks for analysis is checked when it starts instead; refused, it stores the item without analysis.
  - A WebSocket session reports it as it reports any failed reply: `/audio/chat/stream` sends a `provider_unavailable` error for that turn, and the realtime sockets send an `internal_error` and a failed `response.done`. Both stay open.
- **It must not look like a provider outage to shared machinery:**
  - the Evaluations circuit breaker (`Evaluations/circuit_breaker.py`) does not count it. The breaker is shared by all users per provider and opens after 3 to 5 failures, so one user's refusals would otherwise stop evaluations for everyone for 30 to 60 seconds;
  - retry loops do not retry it: G-Eval's, and the LLM extraction strategy's (`Web_Scraping/extraction/strategies/llm.py:164-198`).
  - a job worker does not retry it: the error carries `retryable = False` and `failure_code = "llm_quota_exceeded"`;
  - the chat pipeline neither rewrites it as a provider error nor records a provider failure for it, and neither does the `/chat/completions` auto-router, whose LLM call runs before that endpoint's boundary check;
  - the RAG answer generator does not replace it with a fallback answer.
- **What it costs.**
  - With quotas off, nothing: `usage_quotas_enabled()` is read in the calling thread and the check returns.
  - With quotas on and no LLM limit for the user: nothing, once the user's limits are cached. `quota_resolver` gains a synchronous read of its cache for this. A cold or expired cache entry costs one hand-off to the accounting loop (§5) and the resolver's queries, once per user per minute.
  - With a limit set: one hand-off and the counter queries, for each call. A sync caller waits at most 2 seconds for the answer, then fails open with a warning.
- **Sync LLM calls made on the server loop's own thread cannot be checked** (§5). PR 2b moved seven code paths off the loop with `asyncio.to_thread`: `/translate`, web search aggregation (`WebSearch_APIs.aggregate_results`, the live `/research/websearch` path; the legacy `WebSearch/Web_Search.py` has no caller in the app), the article extractor, enhanced web scraping, web content ingestion, the web scraping service, and the Explainer job. The RAG custom completeness metric named in an earlier draft is deleted: nothing called it, and its `analyze` call could not reach a provider (TASK-13635).

### 5. Accounting runs on one event loop

The check and the recorder are async and use the AuthNZ pool, which is bound to one loop. They must never run on another loop, and never through `asyncio.run`. The boundary check (§4a) runs in the request's task on the server's loop and needs none of this; the rules below for the check apply to §4b.

- **The accounting loop** is the server's loop, captured once at application startup. A captured loop that has closed, or has stopped running, is treated as missing: a hand-off to a stopped loop would be accepted and never run.
- **From an async method on the accounting loop:** await directly.
- **From an async method on another loop** (a workflow engine thread): hand off with `asyncio.run_coroutine_threadsafe` and await the wrapped future.
- **From a sync method on a thread that is not the accounting loop's thread:** hand off with `run_coroutine_threadsafe`.
  - The check waits with a short timeout and fails open when it expires.
  - The record is fire-and-forget, with a callback that logs a failure.
- **From a sync method on the accounting loop's own thread** (a caller that blocks the loop): the check is skipped, because waiting would deadlock, and the record is scheduled with `loop.create_task`. This is a fail-open case, and it is not rare: seven code paths call the sync `analyze` straight from a coroutine. PR 2b moves them off the loop (§4b).
- **Streams** capture the loop when `stream()` is called. The iterator may run and close on a daemon thread.
- **With no accounting loop captured** (unit tests without the application lifespan): the recorder uses the running loop if there is one, and otherwise skips. The check does the same from an async method, and is skipped from a sync one.

This follows the hand-off pattern already used at `LLM_Calls/streaming.py:233` and `chat_service.py:548-556`.

### 6. Counters and other readers

Two counters back the cost keys, next to the existing token counter. All three exclude embeddings with one shared SQL predicate, `COALESCE(operation, '') <> 'embeddings'`. `operation` is nullable, so a plain `<>` would drop rows with no operation.

- `sum_user_llm_tokens_since(user_id, since)`: the existing `AuthnzUsageRepo` method gains the predicate.
- `sum_user_llm_cost_since(user_id, since)`: new; sums `total_cost_usd`. One query can return both the day's and the month's sum.
- One SQL string serves both backends, as the existing counter does. Only the bound timestamp differs.
- The quota code reads them through `quota_checks` helpers: `llm_tokens_this_month` (existing), `llm_cost_today` and `llm_cost_this_month`.
- The `(user_id, ts)` index that makes these sums cheap already exists on both backends.

Costs come from the pricing catalog when the row is written (`Usage/pricing_catalog.py`, via `compute_costs`).

**Local providers cost nothing by default (owner decision 5).** No local provider is in the pricing catalog, so today every call to one is costed at the catalog's unknown-model rate: $0.01 per 1K prompt tokens and $0.03 per 1K completion tokens. A cost limit would be used up by calls that cost nothing. From PR 2a:
- **The catalog resolves a provider-wide `default` entry** when no model of that provider matches. The file already has one, `huggingface.default`, which nothing can reach today (`pricing_catalog.py:264-275`).
- **Each local provider ships with a zero `default`:** llama.cpp, kobold, ooba, tabbyapi, vllm, local-llm, ollama, aphrodite and mlx, the set in `DEFAULT_LOCAL_PROVIDERS` (`LLM_Calls/adapter_registry.py`). The entries are in the built-in defaults and in `Config_Files/model_pricing.json`.
- **The catalog lookup resolves the provider's canonical name first,** so an alias such as `llamacpp` is priced the same. The writers pass different spellings today. The resolution lives in `get_rate_details`, which `compute_costs` and the other pricing helpers all call.
- **It is data, so an operator can change it.** A local provider that points at a paid service can be priced, and a custom OpenAI-compatible endpoint that runs locally can be set to zero. That endpoint is not zero by default, because it may point at a paid service.
- **This changes what is stored,** so every reader of stored cost agrees: the new cost limits, admin usage reports, the Prometheus cost metric and virtual-key USD budgets. Virtual-key budgets have the same problem today. An operator who used a virtual-key USD budget to bound local usage loses that bound; the operator page says so.
- **Rows already written keep their cost.**
- **A paid model that is missing from the catalog** is still costed at the unknown-model rate.

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

Found while building PR 3, and removed with the rest: the `X-RateLimit-Daily-Cost-Remaining` and `X-RateLimit-Monthly-Cost-Remaining` response headers (`evaluations_auth.py`) and the dead `X-RateLimit-Cost-Remaining` line; the `rate_limit_source` value `legacy_cost_only`, now `legacy_local`; the positional `cost` of `_reserve_request_usage` and `record_actual_usage`; `config_manager`'s required `max_cost_per_day`; the CLI default; the frontend's `daily_cost` and `monthly_cost`; two more mirrored docs. `upgrade_user_tier` ignores `max_cost_*` in `custom_limits` with a warning, and an old `evaluations_config.yaml` that still sets them loads with a warning per tier.

### 9. Reporting

- The profile `quotas` section gains an `llm` block: tokens used this month, cost today, cost this month, and each resolved limit.
  - `UserProfileQuotas` (`schemas/user_profile_schemas.py`) gains an `llm` field. Without it the response model drops the block.
  - Both profile endpoints leave out null values, so a limit that is not set is absent from the block, as in the other quota blocks. Absent means unlimited.
- `Docs/Operations/Usage_Quotas.md:68` replaces "enforced on `/chat/completions` only" with the new behaviour. It adds:
  - the two cost keys;
  - the entry points that return a 402 before they start work (§4a), and the paths that are refused at the call and report a provider error (§4b);
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
   - each entry point in the §4a table returns the 402 body and `Retry-After`, and makes no LLM call;
   - `/chat/completions` returns the same 402 body as today, both streaming and not streaming;
   - a streaming RAG search and a streaming character chat reply return a 402, not a 200 with an error frame;
   - a RAG search with answer generation off, and media processing with analysis off, are not refused;
   - the mark set by the boundary check is in effect inside a streaming response body and a `BackgroundTask`, and is not in effect for the next request on the same connection;
   - a request that was admitted is not refused part-way through by §4b, and a job it enqueued is still refused when it runs;
   - an endpoint that is not in the §4a table and re-raises `HTTPException` returns the 402 from §4b;
   - a workflow LLM step fails with `LLMQuotaExceededError`, and the run records it;
   - a direct adapter call raises `LLMQuotaExceededError`, and a stream raises it from `stream()` before the provider method runs;
   - a refusal does not open the Evaluations circuit breaker, is not retried by G-Eval, and does not mark a chat provider unhealthy;
   - with quotas off, a provider call makes no hand-off to the accounting loop and no query; with quotas on and a cached user who has no LLM limit, it makes no hand-off.

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

Five PRs, so recording can be checked before anything new is refused.

- **PR 1a: the mechanism and recording.** No new refusals. After it, every LLM call made in an HTTP request, or in a worker that already sets the content scope, is recorded. It contains:
  - the call context;
  - the single-loop hand-off (§5);
  - recording in the provider base class (§2, §3);
  - the state carried through the bounded pools and the chat request queue;
  - the `records_itself` and `exempt` markings;
  - the embeddings predicate in §6;
  - ADR-067, "LLM calls are accounted in the chat provider base class", if the number is free when the PR opens.
- **PR 1b: the remaining roots in §1.** `WorkerSDK`, the job workers with their own loop, the workflow engine, the audio and realtime WebSockets, and the default executor that carries the LLM call state.
- **PR 2a: the keys, the counters and the boundary check.** The decision, the error and §4a; the cost counters, local-provider pricing and startup warnings in §6; §7 and §9. After it, the entry points in the §4a table refuse with a 402 and everything else behaves as today. The `prechecked` mark is set and nothing reads it yet.
- **PR 2b: the check at the call.** §4b and the hand-off rules for the check in §5, with the seven sync calls moved off the loop. After it, background jobs, workflow runs and every other path are gated too. Closes TASK-13437.
- **Why 2a first.** Each PR leaves the system coherent. After 2a alone, no path behaves worse than today. With 2b first, the entry points in the table would refuse part-way through a request until 2a landed.
- **PR 3: the evaluation cost-cap removals.** §8. It has no dependency on the other two beyond the cost quota existing. Closes TASK-13436.

## Risks

- **An admitted request can pass a limit.** The boundary check admits a request when the user is under the limit, and the request then runs to completion. Some requests fan out into many calls: a media batch with chunked analysis, an evaluation batch, a RAG request with every optional stage on. The overshoot is bounded by one request, not one call, and by one media ingest job, which is admitted when it starts. An operator who needs a hard ceiling sets the limit with that headroom.
- **Each checked call costs a query.** With a limit set, §4b reads the counters for every call that was not admitted at a boundary. A job that makes hundreds of calls makes hundreds of small indexed reads.
- **Fail-open cases.** A sync adapter call made on the event loop's own thread is recorded but not checked; PR 2b moved the known ones off the loop, and the first such call logs a warning. A timed-out check lets the call through. Both are logged.
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

Revision of 2026-10-11, while building PR 1b:

- **Scheduled automations.** They do not run under `WorkerSDK`; the agent-task and audio jobs workers have their own loops and set the root themselves.
- **Executor sites.** One default executor that carries the LLM call state replaces the two named sites, because the pattern is in many more places.
- **uvloop.** The default executor cannot be read back from the loop, so the helper tracks the loops it has installed on.
- **WebSockets.** Only the audio and realtime sockets make chat LLM calls; the rest are left alone.
- **Private pools.** Three pools the code creates itself carry the LLM call state at their submit: claims extraction, the article extractor and web search.
- **Still unrecorded.** MCP tool calls, the in-process claims rebuild threads, and system jobs with no user.

Revision of 2026-10-12, before planning PR 2. A survey of the code, and then an independent review of this revision, found that §4 and §6 did not hold:

- **What the client sees.** §4 said character chat and RAG pass a provider error's status code through, so a refusal raised at the call would reach the client as a 402. It would not: endpoints report a `ChatAPIError` as a provider failure, mostly a 502. Two changes follow.
  - The error is an `HTTPException`, which most endpoints pass through unchanged.
  - §4 has two parts. §4a checks at the request boundary, as spec 2 does for RAG queries. §4b keeps the check at the call for paths with no boundary.
- **The `prechecked` mark** is set for the rest of the request's task. A `with` block would be reset before a streaming response body runs.
- **Streams are checked before the provider method runs.** Three adapters send the request inside `stream()`.
- **Shared machinery.** A refusal must not open the Evaluations circuit breaker, be retried, or mark a chat provider unhealthy. The shared call layer raises it before its `try` block.
- **Sync calls on the server loop** cannot be checked. Seven code paths make them; PR 2b moves them off the loop.
- **Local providers.** They are not in the pricing catalog, so their calls were costed at the unknown-model rate. §6 gives each a zero default price in the catalog. The owner approved this and the revision on 2026-10-12.
- **Reporting.** The profile response model needs an `llm` field, and unset limits are absent, not `null`.
- **Delivery.** PR 2 is split into PR 2a (the boundary check, keys and counters) and PR 2b (the check at the call).

Revision of 2026-10-13, while planning PR 2b:

- **The check at the call** lives in `core/Usage/llm_quota_gate.py`. It runs the same decision as §4a on the accounting loop, and a sync caller waits at most two seconds for it.
- **The seven sync calls** are the ones in §4b's last bullet. Web search aggregation is the live module, not the legacy one, and the Explainer job replaces the RAG custom metric, which never reaches a provider.
- **The `/chat/completions` auto-router** makes its LLM call before the endpoint's boundary check, so its refusal must not count as a failure of the router's provider.
- **Evaluation runs** join the §4a table: their samples run in a task the request starts.
- **Jobs** do not retry a refusal; a workflow `llm` step is retried once by the engine's default policy and refused again without a provider call.
- **Roots** are never admitted by the request that started them.

Revision of 2026-10-18, TASK-13585 and TASK-13610:

- **MCP tool calls and the in-process claims rebuild** are recorded and checked (§1, §4b). The Messages API native path is recorded (§3). It needs no check at the call: `require_llm_quota` admits it first, so its calls run `prechecked`.

Revision of 2026-10-18, TASK-13638:

- **Callers that turned a refusal into degraded output** let it through, or keep their artifact and record the refusal. Benchmark runs join the §4a table.
- **The workflow engine does not retry a step refused with a 402.** This replaces the 2026-10-13 note that an `llm` step is retried once.

Revision of 2026-10-20, TASK-13862:

- **The FVA adjudicator's LLM call** runs on the bounded adapter pool, as the claims engine's own calls do, so it is checked at the call (§5). A refusal there fails the FVA batch instead of becoming a neutral stance. Its only caller, `POST /claims/verify/fva`, answers 500 to every request (TASK-13881).

Revision of 2026-10-20, TASK-13864:

- **A media ingest job that asks for analysis is checked once, when it starts,** and then runs `prechecked`, as an admitted request does; the admission ends with the job. Refused, it stores the item without analysis and records `analysis_error.code = "llm_quota_exceeded"` in its result, instead of failing the item; a reused transcript still records `video_lite.summary_error`. Other jobs are still checked at the call (§4b).
- **The two web scraping endpoints and `POST /watchlists/outputs` join the §4a table.**
- **Scraping queue jobs run as the request that submitted them.** The web scraping service starts its queue workers lazily, inside a request, and they ran every later job in that request's context. Each job now runs in a copy of its submitter's context, so its LLM calls are recorded and checked for that user and no request's admission reaches another's jobs (§1).

Revision of 2026-10-23, TASK-13881:

- **`POST /claims/verify/fva` runs,** and a refusal at its verify or adjudication call answers the 402. Its `user_id` query parameter counts only for a platform admin. The request then runs in that user's content scope, so its calls are checked and recorded for that user (§1). Anyone else's `user_id` is ignored.
