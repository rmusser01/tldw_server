# ADR-067: LLM calls are accounted in the chat provider base class

**Status:** Accepted
**Date:** 2026-10-11
**Backfilled from:** not backfilled
**Decision owner:** Repository owner (@rmusser01)
**Related task:** TASK-13437
**Related spec/plan:** `Docs/Design/2026-10-09-llm-usage-accounting-design.md`

## Decision

LLM chat calls are accounted in `ChatProvider.__init_subclass__`: each provider's own `chat`, `achat`, `stream` and `astream` record one `llm_usage_log` row for the user the call is for, taken from an explicit `llm_call_context` or the content scope, with the write run on the server's event loop, the loop set at startup.

## Context

Only `/chat/completions`, the router, the claims engine and embeddings recorded LLM usage. By the count in the design spec (`Docs/Design/2026-10-09-llm-usage-accounting-design.md`, Problem), about 39 files call the shared layer (`perform_chat_api_call` and `perform_chat_api_call_async`) and about 23 call adapters directly. None of them recorded, so RAG generation, character chat, summarization, evaluations and similar calls were missing from `llm_usage_log` and from the token counter that `limits.llm_tokens_per_month` reads.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| A wrapper object returned by the registry | It breaks adapter identity and type checks that existing tests assert, and the strict-transport check, which tests `type(adapter) is OpenAIAdapter`. |
| Recording in `perform_chat_api_call*` | It misses the callers that get an adapter from the registry and call it directly. |
| Removing the existing recorders | They pass fields the base class cannot know, and the claims budget guard and router analytics read their rows. |

## Consequences

- **More rows.** Anything that sums `llm_usage_log` sees more usage: the monthly token counter, the hosted billing plan, admin usage reports and the daily ledger. Users reach a token limit sooner.
- **No known user, no row.** A call with neither an explicit `llm_call_context` user nor a content scope user is not recorded.
- **Callers that record themselves are marked.** They run their call inside `llm_call_context(records_itself=True)`: the two chat pipeline calls, the three router calls, and the claims extract and verify calls. The mark on the pipeline calls covers everything the pipeline calls. Calls nested inside it, such as the tool-continuation call, are recorded by nobody, as before; a tool that itself calls an LLM is recorded since ADR-075. The claims APS proposition extractor has no recorder of its own, so it is not marked and the base class records it.
- **Exempt calls.** Provider key validation and first-chat verification run inside `llm_call_context(exempt=True)` and are never recorded.
- **Streams.** A stream is recorded once, however it ends: completion, a provider error or an early close. A stream that is opened and never read writes no row, because nothing was received.
- **Wrapped `stream` and `astream` are plain functions that return generators**, so `inspect.isgeneratorfunction` and `inspect.isasyncgenfunction` are False for them. The accounting decision is fixed when the method is called, not at first iteration.
- **Stream prompt tokens are estimated** unless the provider sends a usage chunk, and no adapter asks for one today.
- **Non-streaming calls** are recorded only when they succeed.
- **Failures stay out of the call.** An error in the recorder is logged and never reaches the caller. Non-finite usage values count as zero.
- **Threads.** `BoundedDaemonPool.start` and the chat request queue carry the call state into their threads and workers.
- **Tests that replace a provider method** on an instance or class replace the wrapped method, so that call is not accounted. Accounting tests use a fake `ChatProvider` subclass.
- **Embeddings** are excluded from the LLM token counter. They are still recorded.
- **Background roots.** Background work gets its user from `llm_call_root`. The loop's default executor carries the LLM call state, so `loop.run_in_executor(None, ...)` keeps the user. A thread pool created at a call site must carry the state itself. Three do: claims extraction, the article extractor and web search.
- **MCP tool calls** get their user from the tool call's context, which MCP authentication fills in, because it sets no content scope. A tool run by an admitted request keeps the admission (ADR-075).
- **Still unrecorded:** system jobs with no user, such as the scheduled quality evaluation.

## Follow-up

- PR 1b of the spec added the remaining roots: `WorkerSDK`, the job workers with their own loop, the workflow engine, the audio and realtime WebSockets, and a default executor that carries the LLM call state.
- TASK-13585 recorded MCP tool calls and the in-process claims rebuild. TASK-13610 recorded the Messages API native path (ADR-075).
- PR 2: the quota check and the cost keys. No call is refused by this ADR's change.
