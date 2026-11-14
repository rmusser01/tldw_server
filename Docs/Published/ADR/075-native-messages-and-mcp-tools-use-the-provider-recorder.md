# ADR-075: The Messages native path and MCP tools record through the provider recorder

**Status:** Accepted
**Date:** 2026-10-18
**Backfilled from:** not backfilled
**Decision owner:** Repository owner (@rmusser01)
**Related task:** TASK-13585, TASK-13610
**Related spec/plan:** `Docs/Design/2026-10-09-llm-usage-accounting-design.md` (§1, §3)
**Extends:** ADR-067 (does not supersede it)

## Decision

A provider call that no `ChatProvider` method makes, of which the Messages API native path is the one recorded today, records its one `llm_usage_log` row through the provider recorder's public wrappers `record_direct_response` and `accounted_direct_stream` (`LLM_Calls/usage_accounting.py`), and an MCP tool's LLM calls are recorded for the tool's caller even when a caller that records itself runs the tool.

## Context

ADR-067 records chat calls in `ChatProvider.__init_subclass__`. The Messages API sends Anthropic and llama.cpp requests over HTTP itself (`_native_post_json`, `_prepare_native_stream_iterator` in `endpoints/messages.py`), so no provider method runs and those calls were never recorded (TASK-13610). ADR-067 also marks the `/chat/completions` pipeline `records_itself`, and the mark covers everything the pipeline calls. So an MCP tool that chat tool auto-execution runs was recorded by nobody, although the pipeline's own row covers only its model call (TASK-13585).

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| A fake `ChatProvider` subclass for the native path | Its `achat` would record before the endpoint rejects an invalid 200, its stream wrapper would see raw byte chunks split mid-frame, and it would be a provider class that is not a provider. |
| A recorder of its own in `messages.py` | Token reading, estimates, the accounting loop and "never raise" would be written twice and drift. The wrappers reuse `_plan_call`, `_record_response` and `_accounted_async_stream`. |
| Recording in `perform_chat_api_call*` | ADR-067 rejected it because it misses callers, and the native path does not call it. |
| Leave MCP tools under chat tool auto-execution unrecorded | The chat row does not include the tool's tokens, so a user's tool LLM calls would count toward no limit. |

## Consequences

- **Two public recorder entry points.** `record_direct_response` records a successful non-streaming reply. `accounted_direct_stream` wraps a stream whose items hold whole SSE events and records once, however it ends. Both follow ADR-067's user rules and never raise. A new adapter-less provider call uses them, not a recorder of its own.
- **Anthropic-shaped usage counts cached input.** For a usage block with `input_tokens` and no `prompt_tokens`, prompt tokens are `input_tokens + cache_creation_input_tokens + cache_read_input_tokens`, OpenAI's meaning of `prompt_tokens`, which the usage normalizer and `compute_costs` already assume. llama.cpp's Messages API reports this way too.
- **Anthropic stream events.** Input tokens come from `message_start`; `message_delta` usage is merged over them, and a null field never erases a count.
- **Estimates stay OpenAI-shaped.** A native reply or stream without usable usage records an estimate that counts no Anthropic content blocks and no `system` prompt.
- **MCP tool calls inside a self-recording caller.** The MCP tool runtime sets the caller's user and clears `records_itself` around `module.execute_tool`, and keeps `prechecked`: a tool that an admitted `/chat/completions` request runs is recorded but not refused. This changes ADR-067's consequence that a tool calling an LLM inside the chat pipeline is "recorded by nobody". The tool-continuation call stays unrecorded.

## Follow-up

- The pricing catalog has no Anthropic cache read or write rates, so cache reads are billed at the full prompt rate. A follow-up task, TASK-13858, adds them.
- Other adapter-less chat-completions calls stay unrecorded: the local OCR backend (`OCR/backends/llamacpp_ocr.py`) and the Local_LLM handlers (`Llamafile_Handler.py`, `LlamaCpp_Handler.py`). They cost $0 by default, but their tokens should count toward `llm_tokens_per_month`. They can adopt the wrappers later.
