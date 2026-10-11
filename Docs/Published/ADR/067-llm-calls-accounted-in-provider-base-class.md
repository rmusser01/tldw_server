# ADR-067: LLM calls are accounted in the chat provider base class

**Status:** Accepted
**Date:** 2026-10-11
**Backfilled from:** not backfilled
**Decision owner:** Repository owner (@rmusser01)
**Related task:** TASK-13437
**Related spec/plan:** `Docs/Design/2026-10-09-llm-usage-accounting-design.md`

## Decision

LLM chat calls are accounted in `ChatProvider.__init_subclass__`: each provider's own `chat`, `achat`, `stream` and `astream` record one `llm_usage_log` row for the user the call is for, taken from an explicit `llm_call_context` or the content scope, with the write run on the server's event loop.

## Context

Only `/chat/completions`, the router, the claims engine and embeddings recorded LLM usage. About 39 files call the shared layer (`perform_chat_api_call` and `perform_chat_api_call_async`) and about 23 call adapters directly. None of them recorded, so RAG generation, character chat, summarization, evaluations and similar calls were missing from `llm_usage_log` and from the token counter that `limits.llm_tokens_per_month` reads.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| A wrapper object returned by the registry | It breaks adapter identity and type checks that existing tests assert, and the strict-transport check, which tests `type(adapter) is OpenAIAdapter`. |
| Recording in `perform_chat_api_call*` | It misses the callers that get an adapter from the registry and call it directly. |
| Removing the existing recorders | They pass fields the base class cannot know, and the claims budget guard and router analytics read their rows. |

## Consequences

- **More rows.** Anything that sums `llm_usage_log` sees more usage: the monthly token counter, the hosted billing plan, admin usage reports and the daily ledger. Users reach a token limit sooner.
- **No known user, no row.** A call with neither an explicit `llm_call_context` user nor a content scope user is not recorded.
- **Callers that record themselves are marked.** They run their call inside `llm_call_context(records_itself=True)`: the two chat pipeline calls, the three router calls, and the claims extract and verify calls. The claims APS proposition extractor has no recorder of its own, so it is not marked and the base class records it.
- **Exempt calls.** Provider key validation and first-chat verification run inside `llm_call_context(exempt=True)` and are never recorded.
- **Streams.** A stream is recorded once, however it ends: completion, a provider error or an early close. A stream that is opened and never read writes no row, because nothing was received.
- **Non-streaming calls** are recorded only when they succeed.
- **Failures stay out of the call.** An error in the recorder is logged and never reaches the caller. Non-finite usage values count as zero.
- **Threads.** `BoundedDaemonPool.start` and the chat request queue carry the call state into their threads and workers.
- **Tests that replace a provider method** on an instance or class replace the wrapped method, so that call is not accounted. Accounting tests use a fake `ChatProvider` subclass.
- **Embeddings** are excluded from the LLM token counter. They are still recorded.
- **Not yet covered.** `WorkerSDK` jobs, workflow runs and WebSocket sessions set no user at the call point; they are added in a later change.

## Follow-up

- PR 1b of the spec: the remaining roots (`WorkerSDK`, the workflow engine, WebSockets, two executor sites).
- PR 2: the quota check and the cost keys. No call is refused by this ADR's change.
