# Record a streamed /chat/completions reply from the provider's usage chunk (TASK-13951) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Review folded in (r2, 2026-10-26).** The independent review (`review_plan_13951.md`: 1 Blocker, 1 Important, 8 Minor; its dry run on `b94318cce9` reproduced every `Expected:` line) is folded in as follows, with the controller's rulings. OpenRouter's streaming reference and OpenAI's data-residency guide were fetched with WebFetch on 2026-10-26.
> - **B1 (fixed; Decision 2 re-presented):** OpenRouter's usage chunk carries one content-free choice, so today no `/chat/completions` client receives an empty-`choices` chunk it did not ask for. Decision 2 now recommends (c'): the handler reads the usage chunk for the record, and forwards an empty-`choices` usage chunk only when the client asked (`stream_options.include_usage`, at the top level where the OpenAI SDKs send it, or in `extra_body`). (a) always forward and (b) never forward stay as alternatives, with their costs corrected. Task 1 gains Step 3 (an endpoint test: not asked, asked at the top level, asked in `extra_body`) and Step 8 (the flag through `chat.py`, `execute_streaming_call`, `StreamingPipelineRequest` and `create_streaming_response_with_timeout`). Task 1 lands before Task 2 asks OpenAI, so no commit sends an unrequested chunk. The AC #1 test now asserts OpenAI's chunk is not forwarded and OpenRouter's is. `Chat_API_Documentation.md`, Review Focus 1, the close-out notes and the PR body say who receives the chunk. Each OpenAI chunk's `"usage": null` still reaches every client; Decision 2 says why that is safe and that stripping it was skipped.
> - **I1 (fixed):** the AC #1 OpenRouter case uses OpenRouter's documented chunk (one choice with `finish_reason` and `native_finish_reason` `"stop"`). The doc paragraph says OpenAI's chunk has empty `choices` and OpenRouter's carries a choice. "Vendor docs" cites the streaming page.
> - **M1 (fixed):** the adapter asks `api.openai.com` and any `*.api.openai.com` host. OpenAI's data-residency guide lists ten regional hosts, each a prefix on `api.openai.com`, and only OpenAI can serve such a name, so the suffix test is simple and safe. Task 2's tests add `eu.api.openai.com` (asked) and the lookalike `api.openai.com.example.net` (not asked).
> - **M2 (fixed):** a usage block that counts only one side takes the other from the stream estimate, keeps the block as `usage_metadata`, and is marked `estimated=True`. New test: `test_a_usage_block_without_completion_tokens_counts_the_reply_from_the_estimate`. The provider-level recorder has the same gap; it is in F1, with its own AC.
> - **M3 (fixed):** Task 3 Step 3 also amends ADR-067's "Streams" line (line 32): the chat pipeline's stream writes no row when dropped or failed. It names no follow-up id, because Task 3 commits before Task 4 files F1.
> - **M4 (fixed):** Task 3 Step 4 updates `CHARACTER_CHAT_API_DOCUMENTATION.md`. Its `/chat/completions` line gets the forwarding rule; `complete-v2` forwards OpenAI's usage chunk raw, and the doc says so. The mirror is now 3 files.
> - **M5 (fixed):** `chat.py` is touched anyway (B1), so Task 1 Step 8 also fixes `_build_streaming_commit_cb`'s docstring.
> - **M6 (fixed):** F1's AC #2 drops "or Usage_Quotas.md says why not". Its description names the bypass: read the reply, close on `finish_reason` before the save. F1 stays `high`. Decision 4 and Review Focus 3 say so.
> - **M7 (fixed):** Task 1 Step 2 anchors the `rg_commit_cb=` insertion on the `refresh_provider_params=` and `moderation_getter=` lines together.
> - **M8 (fixed):** the I001 is cited at `streaming_utils.py:5:1`, the start of the import block.
> - **Re-measured** in a worktree detached at `b94318cce9`, with the revised plan applied from its own code blocks:
>   - Task 1 RED: `4 failed, 3 passed` → `6 failed, 5 passed`. GREEN: `7 passed` → `11 passed`. Task 1 files: `190 passed, 1 skipped` → `200 passed, 1 skipped` (the endpoint test file joins the set).
>   - Task 2 RED: `1 failed, 2 passed` → `2 failed, 3 passed`. GREEN: `3 passed` → `5 passed`. The whole file: `7 passed` → `9 passed`.
>   - The touched test files: `84 passed` (four files) → `96 passed` (five).
>   - The focused set after the change: `2354 passed, 2 skipped` → `2357 passed, 2 skipped` (13 new tests; its baseline `2344 passed, 2 skipped` is unchanged).
>   - The wider set after the change: `1401 passed, 2 skipped` → `1404 passed, 2 skipped` (the endpoint test's 3 cases; the same two standing skips).
>   - The mirror: 2 files, 4 insertions, 1 deletion → 3 files, 7 insertions, 3 deletions. `tests/Docs`: `212 passed`, unchanged.
>   - Bandit: 0 results, `loc` 10263 → 17868 (`chat.py` and `streaming_pipeline.py` joined the list).
>   - New check, because Decision 2 changes what `/chat/completions` forwards by default: every test file that streams an empty-`choices` chunk (Task 4 Step 1), `5 failed, 2032 passed, 1 skipped` on base → `5 failed, 2043 passed, 1 skipped`. The 5 failures are the same order-dependent kobold summarization cases on both.
>   - Every `python`/`markdown` block in this plan was checked verbatim against the applied files (27 blocks, 0 missing). The shard and test-quality guards pass, ruff adds nothing, and `git diff --check` is clean.
>
> **Scoped re-review (r3, 2026-10-26).** Result: 0 Blocker, 0 Important, 4 Minor (`rereview_plan_13951.md`). The re-review dry-ran Tasks 1-3 and Task 4's checks on `b94318cce9`, and every `Expected:` line reproduced. It also ran 243 extra OpenAI host and adapter tests, which passed. All 10 r1 findings are fixed. Decision 2 (c') does what it says. The kobold failures are pre-existing: they reproduce on the base with only `test_roleplay_resume_completion.py` followed by `test_summarization_adapter.py`. The four minors go to the executors:
> - (1) A one-sided usage block is stored as `estimated=True` with `estimate_source` resolving to `provider_usage`, so the admin summary counts it as provider usage. Leave it for the final review's triage.
> - (2) The `Usage_Quotas.md` line "a tool-call-only reply counts no completion tokens" holds only on `/chat/completions`. Say so.
> - (3) "Fixes every OpenAI stream" excludes `/v1/messages` with an OpenAI provider, which stops reading at the finish chunk and stays estimated. Reword the text; this is not a regression.
> - (4) Two wording and anchor nits, applied as the executor reaches them.
>
> PR #3346 merged on 2026-10-26 (`35e711080a`), so Task 0 branches from `origin/dev`.

**Goal:** A streamed `/chat/completions` reply is recorded from the usage chunk its provider sends, so the service tier, cache rates and reported cost that PR #3346 (TASK-13936) prices apply to streams too. The server asks OpenAI for that chunk, and forwards it to a client only when the client asked for it. A stream without one is still recorded from the estimate.

**Architecture:** Today `chat_service.execute_streaming_call`'s `save_callback` records every streamed reply from an estimate (`estimate_source="stream_estimate"`). The stream handler (`streaming_utils.StreamingResponseHandler`) already parses every provider chunk, including the usage chunk, but keeps only three token counts for message metadata. The change:
- The handler feeds each parsed payload to the provider-level recorder's own parser, `usage_accounting._StreamUsage`, split so it takes a parsed payload (`feed_event`).
- It passes the whole usage block to a save callback that accepts a `provider_usage` keyword.
- `save_callback` records from that block (tokens via `usage_accounting._usage_tokens`, the block as `usage_metadata`), and from the estimate when there is none. A side the block does not count (a prompt-only block) comes from the estimate.
- The handler forwards a usage chunk with empty `choices` only when the client asked for usage (`stream_options.include_usage`, at the top level of the request or in `extra_body`). `chat.py` works that out and passes it down (Decision 2).
- `OpenAIAdapter.stream` asks OpenAI's own hosts (`api.openai.com` and its data-residency hosts, such as `eu.api.openai.com`) for the usage chunk (`stream_options.include_usage`).

There is one parser for both recorders, no schema change and no second recorder.

**Tech Stack:** Python, FastAPI SSE, pytest, loguru.

**Spec:** `backlog/tasks/task-13951 - A-streamed-chat-completions-reply-is-costed-from-an-estimate-whatever-the-provider-reports.md` (description, AC #1 and AC #2). Predecessor: `Docs/superpowers/plans/2026-10-26-pricing-tiers.md` (TASK-13936, PR #3346; this is its follow-up F1). Related: TASK-13916 (Anthropic adapter stream usage), TASK-13952 item 5 (Google adapter stream usage).

**Measured on:** branch `fix/pricing-service-tiers-13936`, head `b94318cce9` (PR #3346, not merged at planning), in a detached worktree on 2026-10-26.
- The prototype is `<scratchpad>/proto_13951_full.patch` in the planning session's scratchpad; the review fold's is `<scratchpad>/fold13951_proto.patch`. The worktree was reverted afterwards.
- Every count below was measured with the code and test text in this plan.
- Cited `file:line` references are for `b94318cce9`. Edits are anchored on text, not line numbers, because the branch is cut from `origin/dev` after #3346 merges (Task 0).

---

## What was verified (how a streamed reply's usage reaches the recorder)

### The recorders

| Path | Who records it | Usage it records today |
|---|---|---|
| `/chat/completions` stream (the WebUI's default) | `execute_streaming_call`'s `save_callback` (`chat_service.py:6191-6364`), after the stream completes | estimate only: `_estimate_tokens_from_messages` + `len(full_reply) // 4`, `estimated=True`, `estimate_source="stream_estimate"`, no `usage_metadata` (`chat_service.py:6317-6355`) |
| `/chat/completions` non-stream | `_record_dict_response_usage` (`chat_service.py:6906-6974`) | `usage_with_service_tier(response)` |
| Any other adapter stream (RAG, workflows, character chat, ...) | `usage_accounting._accounted_sync_stream` / `_accounted_async_stream`, via `_StreamUsage.feed` and `_record_stream` (`usage_accounting.py:112-164`, `230-246`) | the last usage chunk (tier folded in), else `stream_estimate` |
| Native `/v1/messages` stream | `accounted_direct_stream` (same `_StreamUsage`) | `message_start` usage, `message_delta` merged over it |

**No double recording.**
- `chat.py` runs `execute_streaming_call` inside `with_llm_call_context(..., records_itself=True)` (`chat.py:5574-5633`), its only caller.
- The call state is carried into the stream-factory worker and the request queue (`bounded_daemon.py:140-145`, `request_queue.py:1105`).
- So `_plan_call()` returns `None` (`usage_accounting.py:47-61`) and `_wrap_stream` returns the adapter's generator unwrapped (`usage_accounting.py:389-394`). The chat pipeline is the stream's only recorder, before and after this change.

**Character chat** streams (`character_chat_sessions.py:7702-7758`) are not marked `records_itself`. The provider-level recorder already records them from a usage chunk. They need no code change, and they gain OpenAI usage from Task 2. Its `records_itself` call at `:1172` is the non-stream router call.

### Where the usage chunk travels in the `/chat/completions` stream

1. The adapter's `stream()` yields SSE lines.
2. `chat.py`'s `_sanitize_provider_stream_call` passes every chunk through unchanged (`chat.py:1275-1359`).
3. `StreamingResponseHandler.process_line` parses each `data:` payload (`streaming_utils.py:1952-1960`), then:
   - keeps `sanitize_usage(data["usage"])` (`:1965-1966`), which keeps only `prompt_tokens`, `completion_tokens` and `total_tokens` (`generation_metadata.py:62`, `107-141`): tier, cache counts and `cost` are dropped;
   - forwards a chunk with choices through the choices branch (`:2074-2075`), and a chunk with empty or no `choices` unchanged (`:2078-2085`).
4. The handler calls `save_callback(full_text, tool_calls, function_call, generation=generation_snapshot())` only when the stream was not cancelled and had no error (`:2432-2462`).
5. `generation["usage"]` (the three counts) goes into message metadata only (`chat_service.py:6145-6154`).

So the usage chunk is visible to the handler. The only thing missing is the whole block reaching the recorder.

- **`generation_snapshot()` cannot carry it.** `test_streaming_generation_settlement.py:85-107` asserts the snapshot dict exactly, and the snapshot is message metadata.
- **So the block goes as its own keyword.** `_accepts_keyword` (`streaming_utils.py:1310-1324`) already gates `generation` the same way. No test save callback captures `**kwargs` (`git grep` over the three files that define save callbacks).

What the client receives today depends on the chunk's shape:
- **OpenRouter** puts its usage on a last chunk with one content-free choice (vendor doc below). It goes through the choices branch and reaches every client.
- **OpenAI** sends a chunk with empty `choices` only when asked. Nothing in app code asks, so a `/chat/completions` client receives one only if it asked OpenAI itself through `extra_body.stream_options`. A top-level `stream_options` is kept in the request model (`ChatCompletionRequest` allows extra fields; it lands in `model_extra`) but is not passed to the provider.
- So today no `/chat/completions` client receives an empty-`choices` chunk it did not ask for. Decision 2 keeps it that way once Task 2 asks OpenAI on every stream.

### Each provider's stream

| Provider (adapter) | What the stream yields | Usage chunk today | After this plan |
|---|---|---|---|
| OpenAI (`openai_adapter.py:347-375`) | `iter_sse_lines_raising` (`streaming.py:129-157`); `normalize_provider_line` passes `data:` lines unchanged (`sse.py:133-134`) | only with `stream_options.include_usage`, which nothing in app code sets (`git grep` finds none). `merge_extra_body` lets a caller set it (`payload_utils.py:371-378`); a top-level `stream_options` in the request is dropped (call params are built from declared fields, `chat_service.py:2900-2914`) | asked on OpenAI's own hosts (Task 2); recorded; forwarded only to a client that asked |
| OpenRouter (`openrouter_adapter.py:203-220`) | same pass-through | always, in the last SSE message, on a chunk with one content-free choice (vendor doc below) | recorded, `usage.cost` included; forwarded as today |
| Groq, DeepSeek, Mistral, Qwen, HuggingFace, Bedrock, custom-openai (`iter_sse_lines_raising`); Moonshot, Z.AI (`iter_sse_lines_requests`) | same pass-through | only if the provider sends one unasked or `extra_body` asks (not checked per provider) | recorded when sent; asking is follow-up F2 |
| Local OpenAI-compatible servers (`local_adapters.py:470-538`, `normalize_provider_line` at `510-512`) | same pass-through | server-dependent | recorded when sent |
| Anthropic adapter (`anthropic_adapter.py:453-581`) | its own loop: `message_start` is not handled; `message_delta` is yielded only as a finish chunk with the usage nested under `provider_response` (`566-574`) | no top-level usage | still the estimate (TASK-13916) |
| Google adapter (`google_adapter.py:601-681`) | its own loop; `_stream_event_deltas` (`461+`) drops `usageMetadata` | none | still the estimate (TASK-13952 item 5) |
| Cohere (`cohere_adapter.py:276-321`) | its own loop; the `stream-end` event is turned into DONE (`318-321`) | none | still the estimate (follow-up F2) |

### Who consumes the extra OpenAI usage chunk (Task 2)

With `include_usage`, every OpenAI chunk carries `"usage": null`, and the last one has `choices: []` and the usage (vendor doc below).
- `/chat/completions`: the handler records non-null usage only (`:1965`). After Task 1 it forwards the empty-`choices` chunk only to a client that asked (Decision 2). The `"usage": null` field on the other chunks reaches every client: OpenAI's chunk object defines `usage` as nullable, so the OpenAI SDKs parse it, and OpenAI's own streams carry it whenever usage is asked for.
- Character chat (`/api/v1/chats/{chat_id}/complete-v2`) forwards raw provider lines (`character_chat_sessions.py:7738-7741`), so its clients receive both. Its semantic-output check iterates `choices` (`character_chat_sessions.py:446+`, `7702-7742`). Task 3 documents it in the character chat API doc.
- The WebUI: `extractTokenFromChunk` returns "" for empty `choices` (`apps/packages/ui/src/utils/extract-token-from-chunk.ts:41-73`); `TldwChat.ts:65-77` uses optional chaining.
- Workflows guard empty `choices` (`Workflows/adapters/llm/llm.py:196-198`).
- `_StreamUsage.feed` skips `usage: null`.
- No app code indexes `choices[0]` of a streamed chunk without a guard (`git grep` for `choices[0]` with `delta`, and for `choices", [{}])[0]`: the hits are non-stream response parsers). The independent review re-checked this across `tldw_Server_API/app` and `apps/` (the WebUI, the extension and the Next.js frontend).

### What a dropped stream records today (premise check)

The task brief assumed "a partial stream still records the estimate, as today". It does not: today a `/chat/completions` stream that the client drops, or that fails after it starts, writes **no** usage row.
- `save_callback` runs only when the stream was not cancelled and had no error (`streaming_utils.py:2432-2437`).
- `partial_save_callback` saves text only (`chat_service.py:6162-6189`).
- `_finalize_stream` refunds the Resource Governor reservation (`chat_service.py:6684-6712`; `rg_refund_cb` commits 0 tokens, `chat.py:5612-5622`).
- **Measured:** a throwaway test (deleted) dropped the client after the first content chunk. The result was no row and one refund call, `refund(cancelled=True, error=False)`.
- This plan does not change that path (Decision 4). Follow-up F1 owns it.

### CI guards (checked on `b94318cce9`)

- None of the touched files is listed in `Helper_Scripts/ci/scope_predicate_baseline.txt` or the other `Helper_Scripts/ci/*baseline.txt` files. The exception is `test_quality_baseline.txt:235`: `test_streaming_generation_settlement.py::tautology_suspect=1`.
  - A test whose only assert compares its own collector list counts as a new offence (`test_quality_triage.py:310-360`).
  - The Task 1 handler test therefore unpacks `(saved,) = saves`. Measured: the `assert saves == [...]` form fails `--enforce` with `tautology_suspect=2 (baseline 1)`; the unpacked form passes.
- `check_shard_coverage.py`: `new_uncovered=0` (no new test files).
- Mirrors: `Docs/ADR/067-...`, `Docs/API-related/Chat_API_Documentation.md` and `Docs/API-related/CHARACTER_CHAT_API_DOCUMENTATION.md` are mirrored under `Docs/Published` (the refresh script copies the whole `API-related` tree); `Docs/Operations/Usage_Quotas.md` is not (`git ls-files Docs/Published | grep -c Usage_Quotas` prints 0).
- Ruff (`.venv/bin/ruff`): no new finding.
  - `streaming_utils.py` has one finding on base and after: I001 at `streaming_utils.py:5:1`, the start of the import block.
  - `test_chat_endpoint_streaming_normalization.py` has one on base and after: I001 at `1:1`.
  - `chat.py` and `streaming_pipeline.py` pass on base and after.
  - `test_openai_native_http.py` has 16 on base and 16 after (its old `Dict`/`List` style). The new helper uses `dict`/`list` and imports only the module, so it adds none.
- No schema change, so no OpenAPI drift and no migration.

## Vendor docs (fetched 2026-10-26; the executor re-checks them in Task 0)

**OpenAI.** Fetched with curl as raw Markdown from `https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create/index.md` and `https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events.md`. Quoted:
- `stream_options.include_usage`: "If set, an additional chunk will be streamed before the `data: [DONE]` message. The `usage` field on this chunk shows the token usage statistics for the entire request, and the `choices` field will always be an empty array. All other chunks will also include a `usage` field, but with a null value. **NOTE:** If the stream is interrupted, you may not receive the final usage chunk which contains the total token usage for the request."
- `stream_options.include_obfuscation` is a sibling option, so a caller's `stream_options` may hold other keys.
- The chunk object's `choices`: "Can also be empty for the last chunk if you set `stream_options: {"include_usage": true}`."
- The chunk object has `service_tier` ("Specifies the processing type used for serving the request"), so the usage chunk names its tier and `usage_with_service_tier` folds it in. Fast: "The response will show `service_tier=priority` regardless of if you specify `service_tier=fast` or `priority`".
- The chunk's `usage`: "An optional field that will only be present when you set `stream_options: {"include_usage": true}` in your request. When present, it contains a null value **except for the last chunk** ... If the stream is interrupted or cancelled, you may not receive the final usage chunk".

**OpenAI data residency.** `https://developers.openai.com/api/docs/guides/your-data` (WebFetch, 2026-10-26) lists the regional hosts as a domain prefix on `api.openai.com`: `us.`, `eu.`, `au.`, `ca.`, `jp.`, `in.`, `sg.`, `kr.`, `gb.` and `ae.api.openai.com`. A client selects one through its base URL ("add the domain prefix as defined in the table below to each request"; the Python example is `base_url='https://us.api.openai.com/v1'`). So `host == "api.openai.com" or host.endswith(".api.openai.com")` covers every one of them, and only OpenAI can serve a name under `api.openai.com`.

**OpenRouter.** `https://openrouter.ai/docs/use-cases/usage-accounting` (WebFetch):
- Usage is included in "the last SSE message for streaming responses".
- "`stream_options: { include_usage: true }` parameters are deprecated and have no effect. Full usage details are now always included automatically in every response."

`https://openrouter.ai/docs/api-reference/streaming` (WebFetch, 2026-10-26) gives that message's shape: "OpenAI's spec emits this chunk with an empty `choices` array, but many clients crash when accessing `choices[0].delta` on it, so OpenRouter intentionally deviates: the usage chunk contains one choice with a content-free `delta`". Its example is `data: {"id":"gen-abc123",...,"choices":[{"index":0,"delta":{"content":"","role":"assistant"},"finish_reason":"stop","native_finish_reason":"stop"}],"usage":{...}}`, and "the terminal `finish_reason` appears twice: once on the last content-bearing chunk and again on the usage chunk". The AC #1 test uses this shape.

**Not verified** (left to follow-up F2): whether DeepSeek, Groq, Mistral, Qwen, Moonshot, Z.AI, HuggingFace, Bedrock or Cohere send a stream usage chunk unasked, or which flag asks for one.

---

## Global Constraints

- **Git and PR.** Cut the branch from `origin/dev` after PR #3346 has merged (Task 0). The PR targets `dev`. Never `--no-verify`, never force-push, never `git stash`. One PR.
- **TDD.** Fix test-first: write the test, run it, see it fail for the stated reason, then fix. The counts below are measured; record the real ones.
- **No network in tests.** Tests make no real network calls:
  - the adapter tests use the file's `_FakeClient`;
  - the chat tests use in-memory provider generators and a fake `AuthnzUsageRepo`.
  - No PostgreSQL is needed. Never FLUSHALL Redis.
- **WebFetch** only for vendor API, pricing and usage docs.
- **Backlog.** Edit backlog tasks only with backlog-py: `PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task ...`.
  - Use `--append-notes`, never `--notes`.
  - Stage with `git add -A backlog/tasks`.
  - Never hand-edit task files.
  - New tasks get an explicit `--id`, allocated at execution under the id lock (Task 4 Step 4).
- **Python.** The worktree has no `.venv`. Use `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`, written out in each command (shell variables do not survive between agent calls). Tests run with `TLDW_TEST_NO_DOCKER=1` and `-p no:cacheprovider`.
- **Recorder contract.** Nothing the stream handler or `save_callback` adds may raise into the stream: `feed_event` catches everything, and the recording block keeps its `except _CHAT_NONCRITICAL_EXCEPTIONS`. Do not log usage payloads.
- **Signatures.** Every new parameter is keyword-only with a default, so no caller changes: `save_callback` gains `provider_usage`, `execute_streaming_call` gains `forward_usage_chunk: bool = False`, `create_streaming_response_with_timeout` gains `forward_usage_chunk: bool = True` (today's behaviour for any other caller), and `StreamingPipelineRequest` gains `forward_usage_chunk: bool | None = None`. `_StreamUsage.feed` keeps its signature. The private imports (`_StreamUsage`, `_usage_tokens`) follow existing practice: 47 import lines in app code import a private name from another module (`git grep -nE "^from tldw_Server_API\.app\.[A-Za-z_.]+ import \(?_[a-z]" -- tldw_Server_API/app`).
- **CI traps:**
  - `Helper_Scripts/ci/scope_predicate_baseline.txt`: none of the touched files is listed. Re-check with `git grep -n "chat_service.py\|streaming_utils.py\|streaming_pipeline.py\|endpoints/chat.py\|usage_accounting.py\|openai_adapter.py" Helper_Scripts/ci/scope_predicate_baseline.txt` (it must print nothing). If one is listed and its lines shift, regenerate with `--write-baseline`.
  - This plan is committed at `Docs/superpowers/plans/2026-10-26-stream-usage.md`, so it must have no trailing whitespace, including inside embedded code. Before each commit, `git add` the commit's files, then run `git diff --cached --check` (it must print nothing).
  - Docs mirrored under `Docs/Published` (ADR-067, `Chat_API_Documentation.md` and `CHARACTER_CHAT_API_DOCUMENTATION.md`) need `bash Helper_Scripts/refresh_docs_published.sh` and the mirror diff in the same commit (Task 3).
  - `Helper_Scripts/ci/test_quality_triage.py --enforce` and `Helper_Scripts/ci/check_shard_coverage.py` must stay green. Keep the handler test's `(saved,) = saves` shape.
  - OpenAPI drift: no schema change, so none.
- **Close-out.** The last task closes TASK-13951 with backlog-py (notes, ACs, DoD, final summary) and files the follow-ups.
- **Commits** end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## Decisions

Recommendation first. The plan implements each recommendation; each item gives the cost if it is wrong and the alternatives.

1. **Ask for the usage chunk per provider: OpenAI only, and only on OpenAI's own hosts, with the caller's `stream_options` winning (recommended).**
   - `OpenAIAdapter.stream` sets `stream_options = {"include_usage": True, **caller_stream_options}` when the URL's host is `api.openai.com` or ends with `.api.openai.com` (OpenAI's data-residency hosts, such as `eu.api.openai.com`; vendor doc above). A caller's `include_usage: false` wins.
   - The suffix test is simple and safe: only OpenAI can serve a name under `api.openai.com`, and a lookalike such as `api.openai.com.example.net` does not match (pinned).
   - OpenAI is where tiers (Flex, Fast), cached-input rates and reasoning tokens matter most. The estimate counts only the visible reply text, so a reasoning model's streamed reply is badly under-counted today.
   - Requesting it in the adapter fixes every OpenAI stream at once (`/chat/completions`, character chat, RAG, workflows); the provider-level recorder already reads the chunk.
   - *If wrong:*
     - **A consumer misses the empty `choices`.** Every OpenAI stream to OpenAI's hosts now carries `usage: null` on each chunk and ends with a `choices: []` chunk. `/chat/completions` clients receive that last chunk only when they asked (Decision 2). Character chat's `complete-v2` stream forwards raw lines, so its clients receive it; the WebUI tolerates it, and Task 3 documents it. An internal consumer that was missed and indexes `choices[0]` would raise on the last chunk; the review grepped `tldw_Server_API/app` and `apps/` and found none. Reverting is deleting the 11-line block.
     - **An `openai` provider pointed elsewhere is not asked** (Azure, LiteLLM or a local server, set via `app_config.openai_api.api_base_url`, `OPENAI_API_BASE_URL`, `OPENAI_API_BASE` or `OPENAI_BASE_URL`; `openai_adapter.py:231-257`). Its streams stay estimated unless `extra_body` asks.
   - *Alternatives:*
     - (b) Ask regardless of base URL. Proxies that accept the field would gain usage, but a server behind `openai` that rejects unknown fields would fail every stream.
     - (c) Ask on every OpenAI-compatible adapter, custom and local included. Some local servers may reject `stream_options`, and no vendor doc was checked per provider.
     - (d) Ask only on `/chat/completions`, by adding `extra_body.stream_options` in `chat_service.build_call_params_from_request`. That puts OpenAI wire knowledge in the chat service and leaves other callers' OpenAI streams estimated.
     - (e) Never ask. AC #1 then holds only for providers that send usage unasked (OpenRouter), and OpenAI streams keep the estimate: no tier, no cache rate, no reasoning tokens.
2. **(c') Read the usage chunk for the record; forward an empty-`choices` usage chunk to a `/chat/completions` client only when the client itself asked for usage (recommended).**
   - "Asked" means `"stream_options": {"include_usage": true}` in the client's request, at the top level (where the OpenAI SDKs send it, `extra_body` included, since the SDKs merge `extra_body` into the top level) or in tldw's own `extra_body` field. `chat.py`'s `_client_asked_for_stream_usage(request_data)` checks both and passes `forward_usage_chunk` to `execute_streaming_call`, through `StreamingPipelineRequest` and `create_streaming_response_with_timeout`, to the handler.
   - In `process_line`'s no-choices branch, the handler skips a payload with `choices == []` and a non-null `usage` unless `forward_usage_chunk` is set. `feed_event` has already read it, so the record is unaffected.
   - *Why:* clients see what they see today. Today no `/chat/completions` client receives an empty-`choices` chunk unless it asked OpenAI for one through `extra_body` ("Where the usage chunk travels"), and under (c') that client still receives it. OpenRouter's usage chunk carries one content-free choice, goes through the choices branch, and is forwarded unchanged. So a third-party OpenAI-SDK client that indexes `choices[0]` on every chunk (`for chunk in stream: print(chunk.choices[0].delta.content or "")`, OpenAI's own quickstart loop) does not break.
   - The one visible difference: each OpenAI chunk now carries `"usage": null`. OpenAI's chunk object defines `usage` as nullable and OpenAI's own streams carry it whenever usage is asked for, so the OpenAI SDKs parse it. Stripping it was skipped; add that if a client breaks on it.
   - Pinned: the AC #1 test (OpenAI's chunk is not forwarded to a client that did not ask; OpenRouter's is) and the endpoint test `test_endpoint_forwards_an_empty_choices_usage_chunk_only_to_a_client_that_asked` (not asked, asked at the top level, asked in `extra_body`). Documented in `Chat_API_Documentation.md` (Task 3).
   - *If wrong:*
     - **A provider sends an empty-`choices` usage chunk unasked.** A client that did not ask stops receiving it. OpenAI's contract never sends one unasked, and no provider in "Each provider's stream" is known to.
     - **The plumbing** is about 30 lines across `chat.py`, `chat_service.py`, `streaming_pipeline.py` and `streaming_utils.py`. Reverting it gives alternative (a).
   - *Alternatives:*
     - (a) Always forward. Every OpenAI stream then ends with an empty-`choices` chunk sent to every client, asked or not. OpenAI sends that chunk only to a client that asks, so this breaks the "OpenAI-compatible" contract the server advertises: the OpenAI SDK quickstart loop above raises `IndexError` on it. The docs would have to say OpenAI streams carry an unrequested empty-`choices` chunk. OpenRouter's chunk is forwarded either way.
     - (b) Never forward an empty-`choices` usage chunk. A client that asks OpenAI through `extra_body`, and receives the chunk today, loses it, and a client asking at the top level never gets what it asked for. OpenRouter's chunk carries a choice, so it is still forwarded, unchanged.
3. **Scope: the recorder reads any provider's usage chunk; only OpenAI is asked (recommended).**
   - The chat pipeline's change is provider-agnostic: any top-level `usage` on a stream chunk, read by `_StreamUsage`.
   - Out of scope, each owned elsewhere:
     - the Anthropic adapter's stream (TASK-13916);
     - the Google adapter's stream (TASK-13952 item 5);
     - asking the other OpenAI-compatible providers, and Cohere's dropped `stream-end` usage (follow-up F2).
   - *If wrong:* those providers' streams stay estimated, as today. Adding each is one block in its adapter plus a test.
   - *Alternative:* fold F2 in. That needs a vendor-doc check for eight providers, which this plan did not do.
4. **A dropped or failed `/chat/completions` stream: change nothing here; file F1, labelled `high` (recommended).**
   - Today such a stream writes no row and the RG reservation is refunded ("What a dropped stream records today").
   - ADR-067 says a stream is "recorded once, however it ends". The provider-level recorder does that; the chat pipeline, which records itself, does not. Task 3 Step 3 narrows ADR-067's sentence to the provider-level recorder and states the chat pipeline's gap (review M3).
   - Fixing it changes the Resource Governor refund (it commits 0 tokens on cancel) and needs its own tests. The ACs do not ask for it.
   - *If wrong (deferring):* until F1, a client can read a whole reply and close the connection on `finish_reason`, before the save, and the reply never counts toward per-user LLM cost limits. That is a per-user cost-limit bypass, not just an accounting gap. It exists today; this plan neither opens nor closes it. F1 is filed `high`, and its AC requires the Resource Governor and billing to commit such a stream's tokens, with no "or document why not" branch.
   - *Alternative:* fix it here. Record from `_finalize_stream` (the usage block if seen, else the prompt estimate plus the received text) and commit instead of refund. That is about 30 lines plus tests, and the Resource Governor change is an owner call.

Settled by the code, the ACs and the vendor docs (not decisions):
- **One parser.** The handler calls `_StreamUsage.feed_event`, split out of `feed`, so the tier folding (`usage_with_service_tier`), null filtering and Anthropic `message_start`/`message_delta` merge are the provider-level recorder's own. TASK-13916's fix reaches both recorders.
- **Tokens from the block** come via `_usage_tokens`, the provider-level recorder's own, so an Anthropic-shaped block's prompt includes its cache tokens (ADR-075). A block with zero prompt and completion tokens is recorded as an estimate, as `_record_stream` does.
- **A block that counts only one side** (prompt tokens only, say) takes the other side from the stream estimate, keeps the block as `usage_metadata` (so its tier and cache counts still apply), and is marked `estimated=True`. Without this, the review measured a prompt-only block recorded as 50/0: the reply received counted as nothing. The provider-level recorder (`_record_stream`, `_record_response`) still records such a block as given; F1's description says so.
- **The Resource Governor and billing commit get the recorded total.** `rg_commit_cb` receives the same total as the row, as the non-stream path commits the provider's `total_tokens` (`chat.py:6016-6039`).
- **The block reaches `save_callback` as `provider_usage`**, not in `generation`: the snapshot is message metadata, pinned exactly by an existing test.
- **The row** is `estimated=None` (resolved by `log_llm_usage` from the catalog match) with `estimate_source=None`, so it resolves to `provider_usage`, as `_submit_record` does for the provider-level recorder. A one-sided block is `estimated=True` with `estimate_source=None`.
- **Tool-call and auto-exec streams.** Legacy auto-exec runs tools after the stream inside `save_callback` and makes no second LLM call there, so the stream writes one row from its usage chunk (pinned). Loop mode is a separate endpoint (`chat_loop.py`); `is_chat_loop_mode_enabled` only turns legacy auto-exec off.

## Review Focus

Inputs and conditions the spec implies but the tests do not fully cover, most likely first:

1. **A third-party OpenAI-compatible client that indexes `choices[0]` on every chunk.** OpenAI streams now end with a `choices: []` chunk upstream (Decision 1); `/chat/completions` forwards it only to a client that asked (Decision 2).
   - Pinned: the AC #1 test (OpenAI's chunk is not forwarded to a client that did not ask; OpenRouter's, which carries a choice, is), and the endpoint test's three cases (not asked; asked at the top level; asked in `extra_body`).
   - Not pinned: each OpenAI chunk's `"usage": null` reaches every client (Decision 2 says why that is safe). Character chat's `complete-v2` forwards raw lines, so its clients receive the empty-`choices` chunk; documented in the character chat API doc (Task 3).
   - The WebUI's tolerance is from code reading (`extract-token-from-chunk.ts:46-53`).
   - Documented in `Chat_API_Documentation.md` (Task 3).
2. **An `openai` provider whose base URL is not OpenAI's** (Azure, LiteLLM, vLLM, Ollama). It is not asked and stays estimated, which is today's behaviour.
   - Pinned by `test_an_openai_provider_stream_to_another_server_is_not_asked_for_usage` (another server, and a lookalike host `api.openai.com.example.net`). A data-residency host (`eu.api.openai.com`) is asked: pinned by `test_an_openai_stream_asks_openai_for_its_usage_chunk`.
   - A caller's own `stream_options` wins: pinned by `test_the_callers_stream_options_win_over_the_usage_request`.
3. **A client that drops the stream, or a provider error after output starts.** No row and the RG reservation is refunded, unchanged.
   - Not pinned: a throwaway test confirmed it during planning, and F1 adds the real test when it changes the behaviour.
   - Documented in `Usage_Quotas.md` and ADR-067's "Streams" consequence (Task 3). It is a per-user cost-limit bypass; F1 is `high` and must close it.
4. **A stream whose usage chunk is missing, `null` or all zeros.** It is recorded from the estimate. Pinned by the three cases of `test_a_streamed_reply_without_usage_is_recorded_from_the_estimate` (AC #2).
   - A block with prompt tokens only takes the completion side from the estimate and is marked estimated: pinned by `test_a_usage_block_without_completion_tokens_counts_the_reply_from_the_estimate`. The completion-only case (the prompt side estimated) is not pinned separately.
5. **A tool-call reply.**
   - With a usage chunk it is recorded once, from the chunk: pinned by `test_a_streamed_tool_call_reply_is_recorded_once_from_its_usage_chunk`.
   - Without one, its estimate has 0 completion tokens. The measured RED row is `(9, 0)`: the estimate counts reply text only.
   - Documented in `Usage_Quotas.md`; F1 asks for the estimate to count tool-call arguments and reasoning, as `_StreamUsage.chars` does.

Also checked, not pinned beyond what is stated:
- **The RG and billing commit** now get the provider's total, which can exceed the reservation on a reasoning model. The non-stream path already commits the provider's total. The AC #1 and AC #2 tests pin the committed value.
- **Provider-reported counts are trusted**, as on the non-stream path. A user-controlled base URL exists only for providers in `BYOK_ALLOWED_BASE_URL_PROVIDERS` (empty by default; `AuthNZ/settings.py:482`).
- **A provider that sends usage on several chunks:** the last block wins (an Anthropic `message_delta` merges over the start). That is right for cumulative counts (Gemini, vLLM's continuous usage). `_StreamUsage` is covered by `tests/LLM_Calls/test_usage_accounting.py`.

## PR and commits

One PR, branch `fix/stream-usage-13951`, cut from `origin/dev` after #3346 merges. The tasks build on each other, and Task 3's docs describe Tasks 1-2. Commits:

0. `docs(plan): add the TASK-13951 stream usage plan`
1. `fix(usage): record a streamed chat reply from the provider's usage chunk (TASK-13951)` (the body says the chunk reaches a client only when it asked)
2. `fix(usage): ask OpenAI for a stream's usage chunk (TASK-13951)`
3. `docs(usage): say which streamed replies are costed from provider usage (TASK-13951)`
4. `docs(backlog): close TASK-13951 and file follow-ups`

## Files

- Modify: `tldw_Server_API/app/core/LLM_Calls/usage_accounting.py` (Task 1: split `_StreamUsage.feed`)
- Modify: `tldw_Server_API/app/core/Chat/streaming_utils.py` (Task 1)
- Modify: `tldw_Server_API/app/core/Chat/chat_service.py` (Task 1)
- Modify: `tldw_Server_API/app/core/Chat/streaming_pipeline.py` (Task 1: carry `forward_usage_chunk`)
- Modify: `tldw_Server_API/app/api/v1/endpoints/chat.py` (Task 1: `_client_asked_for_stream_usage`, and `_build_streaming_commit_cb`'s stale docstring)
- Modify: `tldw_Server_API/app/core/LLM_Calls/providers/openai_adapter.py` (Task 2)
- Modify tests:
  - Task 1: `tldw_Server_API/tests/Chat/unit/test_streaming_generation_settlement.py`, `tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py` and `tldw_Server_API/tests/Chat/integration/test_chat_endpoint_streaming_normalization.py`.
  - Task 2: `tldw_Server_API/tests/LLM_Adapters/unit/test_openai_native_http.py`.
- Modify docs (Task 3):
  - `Docs/Operations/Usage_Quotas.md`, `Docs/API-related/Chat_API_Documentation.md`, `Docs/API-related/CHARACTER_CHAT_API_DOCUMENTATION.md` and `Docs/ADR/067-llm-calls-accounted-in-provider-base-class.md`;
  - the mirrors `Docs/Published/API-related/Chat_API_Documentation.md`, `Docs/Published/API-related/CHARACTER_CHAT_API_DOCUMENTATION.md` and `Docs/Published/ADR/067-llm-calls-accounted-in-provider-base-class.md`, written by the refresh script.
- Create: `Docs/superpowers/plans/2026-10-26-stream-usage.md` (this plan, Task 0)

## Test commands

Each command is written out in full where a step uses it. The measured counts:

| Command | Before (`b94318cce9`) | After |
|---|---|---|
| Task 1 `-k` (three Chat files) | `6 failed, 5 passed` | `11 passed` |
| Task 1 files: `test_streaming_generation_settlement.py`, `test_chat_service_streaming_tool_autoexec.py`, `tests/LLM_Calls/test_usage_accounting.py`, `test_streaming_utils.py`, `tests/Chat_NEW/integration/test_server_settled_generation_metadata.py`, `tests/Chat/integration/test_chat_endpoint_streaming_normalization.py` | n/a | `200 passed, 1 skipped` (Task 1 code only) |
| Task 2 `-k` (`test_openai_native_http.py`) | `2 failed, 3 passed` | `5 passed` |
| `test_openai_native_http.py` whole | n/a | `9 passed` |
| The five touched test files together | n/a | `96 passed` |
| Focused regression set (below) | `2344 passed, 2 skipped` | `2357 passed, 2 skipped` (13 new tests) |
| Wider set (below) | `1401 passed, 2 skipped` | `1404 passed, 2 skipped` (the endpoint test's 3 new cases) |
| Every test file that streams an empty-`choices` chunk (Task 4 Step 1) | `5 failed, 2032 passed, 1 skipped` | `5 failed, 2043 passed, 1 skipped` (11 new tests; the 5 failures are the same, see Task 4) |
| `tldw_Server_API/tests/Docs` | n/a | `212 passed` |

The focused regression set (about 85 s):

```bash
TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/LLM_Adapters/unit tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py tldw_Server_API/tests/Usage tldw_Server_API/tests/Chat/unit/test_streaming_utils.py tldw_Server_API/tests/Chat/unit/test_streaming_generation_settlement.py tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py tldw_Server_API/tests/Chat/unit/test_chat_service_fallback.py tldw_Server_API/tests/Chat/unit/test_chat_service_fallback_credential_taxonomy.py tldw_Server_API/tests/Chat/unit/test_chat_history_and_streaming.py tldw_Server_API/tests/Chat/unit/test_streaming_structured_events.py tldw_Server_API/tests/Chat/unit/test_prompt_cost_guardrails.py tldw_Server_API/tests/Chat/unit/test_chat_service_content.py tldw_Server_API/tests/Chat/unit/test_async_stream_task_capacity.py tldw_Server_API/tests/Chat/unit/test_done_sentinel_detection.py tldw_Server_API/tests/Chat/unit/test_chat_service_tool_autoexec.py tldw_Server_API/tests/Chat/unit/test_messages_endpoint_usage.py tldw_Server_API/tests/Chat/unit/test_messages_native_usage_recording.py tldw_Server_API/tests/Chat/unit/test_chat_service_base_url_override.py tldw_Server_API/tests/Chat/unit/test_failed_retry_provider_order.py tldw_Server_API/tests/Chat/unit/test_chat_token_estimators_agree.py tldw_Server_API/tests/Chat_NEW/integration/test_server_settled_generation_metadata.py tldw_Server_API/app/core/MCP_unified/tests/test_tool_llm_usage_attribution.py
```

The wider set: every other test file that names `_StreamUsage`, `StreamingResponseHandler`, `create_streaming_response_with_timeout`, `execute_streaming_call`, `safe_stream_generator`, `usage_accounting`, `OpenAIAdapter` or `openai_adapter` (found by `git grep -l` over `tldw_Server_API/tests`), plus the chat tests that patch `log_llm_usage`. Left out: the TTS hits (`tests/TTS`, `tests/TTS_NEW`, `tests/Audio/test_tts_*`, `tests/Persona/test_persona_live_tts_providers.py`), which name the TTS module's own `OpenAIAdapter`, and `tests/AuthNZ_Postgres/test_byok_oauth_endpoints_pg.py` (PostgreSQL; see below). About 8-9 minutes:

```bash
TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q -rs tldw_Server_API/tests/Chat/integration/test_chat_endpoint_simplified.py tldw_Server_API/tests/Chat/integration/test_chat_endpoint_streaming_normalization.py tldw_Server_API/tests/LLM_Adapters/integration tldw_Server_API/tests/LLM_Calls/test_adapter_sse_loop_pinning.py tldw_Server_API/tests/LLM_Calls/test_adapter_validation.py tldw_Server_API/tests/LLM_Calls/test_async_streaming_dedup.py tldw_Server_API/tests/LLM_Calls/test_custom_openai_top_p.py tldw_Server_API/tests/LLM_Calls/test_http_status_extraction_parity.py tldw_Server_API/tests/LLM_Calls/test_llm_providers.py tldw_Server_API/tests/LLM_Calls/test_llm_streaming_and_security.py tldw_Server_API/tests/LLM_Calls/test_openai_compatible_provider_adapters.py tldw_Server_API/tests/LLM_Calls/test_provider_adapter_runtime_boundary.py tldw_Server_API/tests/LLM_Calls/test_provider_config_resolution.py tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py tldw_Server_API/tests/LLM_Calls/test_tool_choice_providers.py tldw_Server_API/tests/Streaming/test_provider_streaming_smoke.py tldw_Server_API/tests/Setup/test_setup_first_chat_completion.py tldw_Server_API/tests/RAG_NEW/unit/test_rag_provider_credentials.py tldw_Server_API/tests/Media/test_document_insights_service_prompt.py tldw_Server_API/tests/Media/test_video_summary_service_prompt.py tldw_Server_API/tests/Media/test_web_summary_service_prompt.py tldw_Server_API/tests/prompt_studio/unit/test_prompt_generator.py tldw_Server_API/tests/prompt_studio/unit/test_prompt_improver.py tldw_Server_API/tests/AuthNZ_Unit/test_byok_runtime.py tldw_Server_API/tests/AuthNZ_Unit/test_byok_validation_boundary.py tldw_Server_API/tests/AuthNZ_SQLite/test_byok_runtime_sqlite.py tldw_Server_API/tests/Config/test_config_providers_endpoints.py tldw_Server_API/tests/Audio/test_speech_chat_service.py tldw_Server_API/tests/Chat/unit/test_model_router_accounting.py tldw_Server_API/tests/Chat/unit/test_llm_call_state_propagation.py tldw_Server_API/tests/Chat/unit/test_chat_service_queue_estimate.py
```

The two skips in the wider set are standing:
- `test_chat_endpoint_simplified.py:766`, "Streaming tests hang with TestClient";
- `test_byok_runtime.py:2706`, "installed aiosqlite has no InterfaceError".

The focused set's two skips are the same before and after. `tests/AuthNZ_Postgres/test_byok_oauth_endpoints_pg.py` also names `OpenAIAdapter`. It needs PostgreSQL, was not run, and touches no streaming code.

One planning run of the touched test files aborted at interpreter exit with `libc++abi: ... recursive_mutex lock failed`, after the tests. The rerun exited 0. If it recurs, rerun before investigating; do not dismiss a test failure as this.

The empty-`choices` set (Task 4 Step 1) covers the default Decision 2 changes: what `/chat/completions` forwards. It is every test file that writes an empty `choices` list: `git grep -lE "\"choices\": ?\[\]|'choices': ?\[\]|choices\\\\\": ?\[\]" -- tldw_Server_API/tests` finds 30 on `b94318cce9`, and this plan's tests add two (`test_chat_endpoint_streaming_normalization.py`, `test_chat_service_streaming_tool_autoexec.py`). The 32 files are listed in Task 4 Step 1, and the same list runs before and after. About 9 minutes.

---

## Task 0: Branch, plan file, baseline

- [ ] **Step 1: Confirm #3346 has merged.** Run `gh pr view 3346 --json state --jq .state`; it must print `MERGED`.
  - If it does not, stop and tell the controller. This plan depends on `usage_with_service_tier`, `service_tiers` pricing and OpenRouter's reported cost from #3346.
  - Then run `git fetch origin dev --quiet && git switch -c fix/stream-usage-13951 origin/dev`.
  - Check: `git grep -n "def usage_with_service_tier" -- tldw_Server_API/app/core/Usage/llm_usage_normalizer.py` prints one line.
- [ ] **Step 2: Mark the task started:** `PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-13951 -s "In Progress" --append-notes "Plan: Docs/superpowers/plans/2026-10-26-stream-usage.md (owner-approved). ADR assessment: no new ADR; ADR-067's stream consequence is updated (the OpenAI adapter asks for a usage chunk; the chat pipeline shares the recorder's parser); ADR-075 governs Anthropic-shaped prompt tokens; ADR-059 the backlog tool."`
- [ ] **Step 3: Commit the plan.** Copy the approved plan to `Docs/superpowers/plans/2026-10-26-stream-usage.md`. Then run `git add Docs/superpowers/plans/2026-10-26-stream-usage.md && git add -A backlog/tasks && git diff --cached --check` (it must print nothing), and commit (message 0).
- [ ] **Step 4: Baseline.** Run the focused set, the wider set and the empty-`choices` set (its command is in Task 4 Step 1) on the unchanged branch, and record the three counts.
  - Measured on `b94318cce9`: focused `2344 passed, 2 skipped`; wider `1401 passed, 2 skipped`; empty-`choices` `5 failed, 2032 passed, 1 skipped` (the 5 order-dependent kobold failures named in Task 4 Step 1).
  - Dev after #3346 merges may differ. Compare later runs with this branch's own baseline, never with the numbers here alone, and never dismiss a failure as environment-only without it.
- [ ] **Step 5: Re-check the vendor docs.**
  - Fetch with curl: `curl -sL https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events.md | grep -n -A4 "include_usage"` and `curl -sL https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create/index.md | grep -n -A10 "include_usage: optional"`.
  - Fetch with WebFetch: `https://openrouter.ai/docs/use-cases/usage-accounting`, `https://openrouter.ai/docs/api-reference/streaming` and `https://developers.openai.com/api/docs/guides/your-data`.
  - The quotes under "Vendor docs" must still hold. Stop and report if OpenAI's usage chunk no longer has empty `choices`, `include_usage` changed, OpenRouter's usage chunk no longer carries a choice, or an OpenAI data-residency host is not a prefix on `api.openai.com`.

## Task 1: Record a streamed reply from the provider's usage chunk, and forward that chunk only to a client that asked (AC #1, AC #2, Decision 2)

The forwarding change lands in this task, before Task 2 asks OpenAI on every stream, so no commit sends a client an empty-`choices` chunk it did not ask for.

**Files:**
- Modify: `tldw_Server_API/app/core/LLM_Calls/usage_accounting.py` (`_StreamUsage.feed`, `usage_accounting.py:119-164` on `b94318cce9`)
- Modify: `tldw_Server_API/app/core/Chat/streaming_utils.py` (imports `:42-44`, `__init__` `:1432-1433`, docstring `:1824-1826`, `process_line` `:1965-1966` and `:2078-2079`, the save call `:2445-2449`, `create_streaming_response_with_timeout` `:2574-2622`)
- Modify: `tldw_Server_API/app/core/Chat/streaming_pipeline.py` (`StreamingPipelineRequest` `:17-34`, `create_chat_streaming_response` `:48-62`)
- Modify: `tldw_Server_API/app/core/Chat/chat_service.py` (imports `:198-208`, `execute_streaming_call`'s signature `:5296-5335`, `save_callback` `:6191-6197`, the recording block `:6317-6364`, `StreamingPipelineRequest(...)` `:6715-6730`)
- Modify: `tldw_Server_API/app/api/v1/endpoints/chat.py` (the `execute_streaming_call(...)` call `:5574-5633`; `_build_streaming_commit_cb` `:6432-6446`)
- Test: `tldw_Server_API/tests/Chat/unit/test_streaming_generation_settlement.py`, `tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py`, `tldw_Server_API/tests/Chat/integration/test_chat_endpoint_streaming_normalization.py`

**Interfaces:**
- Produces:
  - `_StreamUsage.feed_event(self, data: Any) -> None`, which reads one parsed SSE payload and never raises. `feed` calls it for each payload it parses.
  - `StreamingResponseHandler.stream_usage: _StreamUsage` and `StreamingResponseHandler.forward_usage_chunk: bool` (default `True`).
  - `save_callback(..., *, generation=None, provider_usage: dict[str, Any] | None = None)`: the handler passes `provider_usage` (a copy of `stream_usage.usage`) only when the callback accepts the keyword and a usage block was seen.
  - `create_streaming_response_with_timeout(..., forward_usage_chunk: bool = True)`, `StreamingPipelineRequest.forward_usage_chunk: bool | None = None` and `execute_streaming_call(..., forward_usage_chunk: bool = False)`.
  - `chat._client_asked_for_stream_usage(request_data: Any) -> bool`: true when `stream_options.include_usage` is `true` at the top level of the request (`model_extra`) or in `extra_body`.
  - The test fixture `stream_usage_rows` and helper `_reply_then(*frames)` in `test_chat_service_streaming_tool_autoexec.py`. TASK-13916 and TASK-13952 reuse them.
- Consumes: `usage_accounting._usage_tokens(usage: dict[str, Any]) -> tuple[int, int]` (unchanged) and `usage_tracker.log_llm_usage` (unchanged).

**RED reason (measured):** `6 failed, 5 passed`.
- The handler test fails with `AssertionError: assert None == {'prompt_tokens': 5, 'completion_tokens': 2, 'total_tokens': 7, 'prompt_tokens_details': {'cached_tokens': 4}, ...}`.
- The two AC #1 cases fail with `AssertionError: assert 'stream_estimate' == 'provider_usage'`.
- The one-sided block test fails with `assert (9, 3) == (50, 3)`.
- The tool-call test fails with `assert (9, 0) == (300, 40)`.
- The endpoint test's `not-asked` case fails with `AssertionError: assert [{'prompt_tok...l_tokens': 7}] == []`: today the chunk is forwarded.
- The three AC #2 cases and the endpoint test's two `asked` cases pass: that is today's behaviour, and they guard it.

- [ ] **Step 1: Add the handler test** to `tldw_Server_API/tests/Chat/unit/test_streaming_generation_settlement.py`, directly after `test_complete_reply_passes_finish_reason_and_provider_usage_to_save` (before `async def test_length_finish_reason_settles_as_length():`). Keep the `(saved,) = saves` unpacking: `assert saves == [...]` trips `test_quality_triage.py --enforce` (`tautology_suspect`).

```python
async def test_complete_reply_passes_the_whole_provider_usage_block_to_a_save_that_takes_it():
    """The usage record needs the tier, cache counts and cost; generation metadata keeps the counts (TASK-13951)."""
    handler = StreamingResponseHandler("conv", "model")
    saves: list[dict[str, Any]] = []

    async def save(text, tool_calls=None, function_call=None, *, generation=None, provider_usage=None):
        saves.append({"generation": generation, "provider_usage": provider_usage})
        return "saved-complete"

    usage = {
        "prompt_tokens": 5,
        "completion_tokens": 2,
        "total_tokens": 7,
        "prompt_tokens_details": {"cached_tokens": 4},
        "cost": 0.0001,
    }

    async def provider() -> AsyncIterator[str]:
        # OpenAI with stream_options.include_usage: usage is null until a last chunk with empty choices.
        yield _sse({"service_tier": "flex", "usage": None, "choices": [{"index": 0, "delta": {"content": "Hi"}}]})
        stop = {"index": 0, "delta": {}, "finish_reason": "stop"}
        yield _sse({"service_tier": "flex", "usage": None, "choices": [stop]})
        yield _sse({"service_tier": "flex", "choices": [], "usage": usage})
        yield "data: [DONE]\n\n"

    async for _frame in handler.safe_stream_generator(provider(), save_callback=save):
        pass

    (saved,) = saves
    assert saved["provider_usage"] == {**usage, "service_tier": "flex"}
    assert saved["generation"]["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
```

- [ ] **Step 2: Add the chat-pipeline tests** to `tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py`.
  - After the `tool_auto_exec` import block (the one ending `ToolExecutionRecord,\n)`), add:

```python
from tldw_Server_API.app.core.Usage import usage_tracker
from tldw_Server_API.app.core.Usage.pricing_catalog import reset_pricing_catalog
```

  - In `_run_execute_streaming_call`, add the parameter `rg_commit_cb=None,` after `refresh_provider_params=None,`. In its `execute_streaming_call(...)` call, add `rg_commit_cb=rg_commit_cb,` after the two lines `refresh_provider_params=refresh_provider_params or (lambda *_args, **_kwargs: ({}, None)),` / `moderation_getter=lambda: _NoModeration(),`. Anchor on both lines: `moderation_getter=lambda: _NoModeration(),` alone appears five times in the file.
  - Append to the end of the file (after `test_chat_sse_forbids_proxy_compression`, two blank lines before):

```python
# --- A streamed reply is recorded from the provider's usage chunk (TASK-13951) ---


@pytest.fixture
def stream_usage_rows(monkeypatch: pytest.MonkeyPatch) -> Any:
    """The llm_usage_log rows the stream's real log_llm_usage inserts, costed by the shipped catalog."""
    rows: list[dict[str, Any]] = []

    class _Repo:
        """Stands in for AuthnzUsageRepo and keeps the inserted rows."""

        def __init__(self, _pool: Any) -> None:
            """No database."""

        async def insert_llm_usage_log(self, **kwargs: Any) -> None:
            """Record the row instead of writing it."""
            rows.append(kwargs)

    async def _none() -> None:
        """No pool and no daily ledger."""

    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)
    monkeypatch.setenv("LLM_USAGE_ENABLED", "true")
    monkeypatch.setattr(usage_tracker, "get_db_pool", _none)
    monkeypatch.setattr(usage_tracker, "_get_tokens_daily_ledger", _none)
    monkeypatch.setattr(usage_tracker, "AuthnzUsageRepo", _Repo)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)
    reset_pricing_catalog()
    yield rows
    reset_pricing_catalog()


def _reply_then(*frames: dict[str, Any]) -> Any:
    """A provider call that streams "hello world!", stops, then sends ``frames`` before DONE."""

    def _call() -> Any:
        def _stream() -> Any:
            yield 'data: {"choices": [{"delta": {"content": "hello world!"}}]}\n\n'
            yield 'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n'
            for frame in frames:
                yield f"data: {json.dumps(frame)}\n\n"
            yield "data: [DONE]\n\n"

        return _stream()

    return _call


async def _save_nothing(*_args: Any, **_kwargs: Any) -> str:
    return "message-1"


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize(
    ("provider", "model", "usage_chunk", "tier", "total_cost", "forwarded"),
    [
        # OpenAI Fast, reported as priority: 2,000 uncached and 8,000 cached input tokens and 1,000 output tokens
        # at gpt-4o-mini's priority rates ($0.25, $0.125 and $1.00 per million).
        (
            "openai",
            "gpt-4o-mini",
            {
                "service_tier": "priority",
                "choices": [],
                "usage": {
                    "prompt_tokens": 10_000,
                    "completion_tokens": 1_000,
                    "total_tokens": 11_000,
                    "prompt_tokens_details": {"cached_tokens": 8_000},
                },
            },
            "priority",
            (2_000 * 0.25 + 8_000 * 0.125 + 1_000 * 1.00) / 1_000_000,
            False,
        ),
        # OpenRouter reports the call's cost; an uncatalogued model is not costed at the unknown-model rate.
        # Its usage chunk carries one content-free choice, not empty choices (OpenRouter's streaming reference).
        (
            "openrouter",
            "acme/brand-new-model",
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "", "role": "assistant"},
                        "finish_reason": "stop",
                        "native_finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10_000, "completion_tokens": 1_000, "total_tokens": 11_000, "cost": 0.0042},
            },
            None,
            0.0042,
            True,
        ),
    ],
    ids=["openai-fast-cached", "openrouter-reported-cost"],
)
async def test_a_streamed_reply_is_recorded_from_the_usage_chunk_its_provider_sends(
    stream_usage_rows, provider, model, usage_chunk, tier, total_cost, forwarded
) -> None:
    """Its service tier, cache rate and reported cost apply, as for a reply that is not streamed (AC #1)."""
    committed: list[int] = []

    response = await _run_execute_streaming_call(
        llm_call_func=_reply_then(usage_chunk),
        save_message_fn=_save_nothing,
        selected_provider=provider,
        provider=provider,
        model=model,
        rg_commit_cb=committed.append,
    )
    chunks = await _collect_sse_chunks(response)

    (row,) = stream_usage_rows
    assert row["estimate_source"] == "provider_usage"
    assert row["estimated"] is False
    assert (row["prompt_tokens"], row["completion_tokens"]) == (10_000, 1_000)
    assert row["total_cost_usd"] == pytest.approx(total_cost)
    assert json.loads(row["raw_usage_metadata_json"]).get("service_tier") == tier
    assert committed == [11_000]
    # This client did not ask for usage: OpenAI's empty-choices chunk is not forwarded, and OpenRouter's,
    # which carries a choice, reaches it as today (Decision 2).
    assert [payload["usage"] for _, payload in _json_payload_chunks(chunks) if payload.get("usage")] == (
        [usage_chunk["usage"]] if forwarded else []
    )


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.parametrize(
    "frames",
    [
        (),
        ({"choices": [], "usage": None},),
        ({"choices": [], "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},),
    ],
    ids=["no-usage-chunk", "null-usage", "zero-usage"],
)
async def test_a_streamed_reply_without_usage_is_recorded_from_the_estimate(stream_usage_rows, frames) -> None:
    """No usable usage chunk: the row is the stream estimate, as before (AC #2)."""
    committed: list[int] = []

    response = await _run_execute_streaming_call(
        llm_call_func=_reply_then(*frames),
        save_message_fn=_save_nothing,
        rg_commit_cb=committed.append,
    )
    await _collect_sse_chunks(response)

    (row,) = stream_usage_rows
    assert row["estimate_source"] == "stream_estimate"
    assert row["estimated"] is True
    assert row["completion_tokens"] == len("hello world!") // 4
    assert row["raw_usage_metadata_json"] is None
    assert committed == [row["total_tokens"]]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_a_usage_block_without_completion_tokens_counts_the_reply_from_the_estimate(stream_usage_rows) -> None:
    """A block with prompt tokens only does not count the reply received as nothing (TASK-13951)."""
    committed: list[int] = []

    response = await _run_execute_streaming_call(
        llm_call_func=_reply_then({"choices": [], "usage": {"prompt_tokens": 50}}),
        save_message_fn=_save_nothing,
        rg_commit_cb=committed.append,
    )
    await _collect_sse_chunks(response)

    (row,) = stream_usage_rows
    assert (row["prompt_tokens"], row["completion_tokens"]) == (50, len("hello world!") // 4)
    assert row["estimated"] is True
    assert committed == [row["total_tokens"]]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_a_streamed_tool_call_reply_is_recorded_once_from_its_usage_chunk(
    monkeypatch: pytest.MonkeyPatch, stream_usage_rows
) -> None:
    """Auto-executed tools add no row of their own; the reply's usage chunk is recorded (TASK-13951)."""
    executed: list[str] = []

    async def fake_autoexec(**kwargs):
        executed.extend(call["id"] for call in kwargs["tool_calls"])
        return ToolExecutionBatchResult(0, 0, 0, 0, False, [])

    monkeypatch.setattr(chat_service, "should_auto_execute_tools", lambda: True)
    monkeypatch.setattr(chat_service, "execute_assistant_tool_calls", fake_autoexec)
    usage_chunk = {"choices": [], "usage": {"prompt_tokens": 300, "completion_tokens": 40, "total_tokens": 340}}

    def llm_call_func() -> Any:
        def _stream() -> Any:
            *frames, done = _tool_call_stream()
            yield from frames
            yield f"data: {json.dumps(usage_chunk)}\n\n"
            yield done

        return _stream()

    async def save_message_fn(_db, _conv_id, payload, use_transaction=True):
        return "m-1"

    response = await _run_execute_streaming_call(llm_call_func=llm_call_func, save_message_fn=save_message_fn)
    await _collect_sse_chunks(response)

    assert executed == ["c1"]
    (row,) = stream_usage_rows
    assert (row["prompt_tokens"], row["completion_tokens"]) == (300, 40)
    assert row["estimate_source"] == "provider_usage"
```

  The rates come from the shipped catalog: `gpt-4o-mini`'s `service_tiers.priority` is `prompt 0.00025`, `completion 0.001` and `cache_read 0.000125` per 1K (#3346's table). Its entry is not `estimated`, which is why the row is `estimated=False`.

  `_run_execute_streaming_call` does not pass `forward_usage_chunk`, so these tests run as a client that did not ask for usage (the `execute_streaming_call` default).

- [ ] **Step 3: Add the endpoint test** to the end of `tldw_Server_API/tests/Chat/integration/test_chat_endpoint_streaming_normalization.py` (after `test_endpoint_non_stream_auto_continue_returns_followup_assistant_content`, two blank lines before). It goes through `/chat/completions`, so it pins `chat.py`'s reading of the request as well as the handler. `perform_chat_api_call` is patched, so no provider is called.

```python
@pytest.mark.unit
@pytest.mark.parametrize(
    ("request_fields", "forwarded"),
    [
        ({}, False),
        ({"stream_options": {"include_usage": True}}, True),
        ({"extra_body": {"stream_options": {"include_usage": True}}}, True),
    ],
    ids=["not-asked", "asked-top-level", "asked-in-extra-body"],
)
def test_endpoint_forwards_an_empty_choices_usage_chunk_only_to_a_client_that_asked(request_fields, forwarded):
    """OpenAI sends its usage chunk only to a client that asks, so the server forwards it only then (TASK-13951)."""
    db, db_path = _make_test_db()
    usage = {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}

    def upstream_stream():
        yield 'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
        yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        yield f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n"
        yield "data: [DONE]\n\n"

    _app: Any = app
    _app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    try:
        with (
            _make_test_client(db) as client,
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=False),
            patch("tldw_Server_API.app.api.v1.endpoints.chat.API_KEYS", {"openai": "sk-test"}),
            patch("tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call", return_value=upstream_stream()),
        ):
            body = {
                "api_provider": "openai",
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                **request_fields,
            }
            r = _post_with_csrf(client, "/api/v1/chat/completions", json=body, headers=_auth_headers(client))
        assert r.status_code == 200, r.text
        usage_chunks = [payload for payload in _json_sse_payloads(r.text) if payload.get("choices") == []]
        assert [payload.get("usage") for payload in usage_chunks] == ([usage] if forwarded else [])
    finally:
        _app.dependency_overrides.pop(get_chacha_db_for_user, None)
        for path in (db_path, db_path + "-wal", db_path + "-shm"):
            if os.path.exists(path):
                os.unlink(path)
```

- [ ] **Step 4: Run; see RED.**

```bash
TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Chat/unit/test_streaming_generation_settlement.py tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py tldw_Server_API/tests/Chat/integration/test_chat_endpoint_streaming_normalization.py -k "whole_provider_usage_block or usage_chunk_its_provider or without_usage or without_completion_tokens or tool_call_reply_is_recorded_once or only_to_a_client_that_asked"
```

Expected: `6 failed, 5 passed`, with the reasons above.

- [ ] **Step 5: Split `_StreamUsage.feed`** in `tldw_Server_API/app/core/LLM_Calls/usage_accounting.py`. Replace the whole `feed` method, from `    def feed(self, item: Any) -> None:` through its closing `        except Exception:  # noqa: BLE001 - a malformed chunk must never fail a stream` / `            return`, with:

```python
    def feed(self, item: Any) -> None:
        """Read one yielded item, which may hold several SSE lines."""
        try:
            text = item.decode("utf-8", "ignore") if isinstance(item, (bytes, bytearray)) else item
            if not isinstance(text, str):
                return
            for part in text.splitlines():
                line = part.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                # "reasoning_content" does not contain the substring '"content"', so it is listed too.
                if not any(key in payload for key in ('"usage"', '"content"', '"reasoning_content"', '"arguments"')):
                    continue
                try:
                    data = json.loads(payload)
                except ValueError:
                    continue
                self.feed_event(data)
        except Exception:  # noqa: BLE001 - a malformed chunk must never fail a stream
            return

    def feed_event(self, data: Any) -> None:
        """Read one parsed SSE payload; never raises. The chat stream handler calls it with the payloads it parses."""
        try:
            if not isinstance(data, dict):
                return
            usage = usage_with_service_tier(data)  # an OpenAI chunk names its service tier beside the usage
            event = data.get("type")
            if event == "message_start":  # Anthropic Messages: input tokens arrive here, output in message_delta
                message = data.get("message")
                usage = message.get("usage") if isinstance(message, dict) else None
            if isinstance(usage, dict):
                usage = {key: value for key, value in usage.items() if value is not None}  # null never erases a count
                if usage:
                    self.usage = {**(self.usage or {}), **usage} if event == "message_delta" else usage
            for choice in data.get("choices") or []:
                delta = choice.get("delta") if isinstance(choice, dict) else None
                if not isinstance(delta, dict):
                    continue
                for text_field in (delta.get("content"), delta.get("reasoning_content")):
                    if isinstance(text_field, str):
                        self.chars += len(text_field)
                tool_calls = delta.get("tool_calls")
                for tool_call in tool_calls if isinstance(tool_calls, list) else []:
                    function = tool_call.get("function") if isinstance(tool_call, dict) else None
                    arguments = function.get("arguments") if isinstance(function, dict) else None
                    if isinstance(arguments, str):
                        self.chars += len(arguments)
        except Exception:  # noqa: BLE001 - a malformed payload must never fail a stream
            return
```

  The body of `feed_event` is the old loop body unchanged. One difference: an error in one payload no longer stops the item's remaining lines from being read.

- [ ] **Step 6: Feed the handler's payloads to it** in `tldw_Server_API/app/core/Chat/streaming_utils.py`.
  - After `from tldw_Server_API.app.core.LLM_Calls.sse import SSE_CONTROL_FIELD_PREFIXES, is_done_line, sse_data, sse_done, sse_event`, add:

```python
from tldw_Server_API.app.core.LLM_Calls.usage_accounting import _StreamUsage
```

  - In `StreamingResponseHandler.__init__`, after `self.usage: Optional[dict[str, int]] = None`, add:

```python
        # The provider's whole usage block (tier, cache counts, cost), read as the
        # provider-level recorder reads it; it goes to the usage record, not to metadata.
        self.stream_usage = _StreamUsage()
```

  - In `safe_stream_generator`'s docstring, replace `accepts a ``generation`` keyword it also receives` / `                ``generation_snapshot()``.` with:

```python
                accepts a ``generation`` keyword it also receives
                ``generation_snapshot()``. When it accepts ``provider_usage`` and
                the provider sent a usage block, it receives that whole block.
```

  - In `process_line`, directly after the two lines `if isinstance(data, dict) and data.get("usage") is not None:` / `self._record_usage(data.get("usage"))`, add at the same indentation as that `if`:

```python
                    self.stream_usage.feed_event(data)
```

  - Where the complete reply is saved, directly after the `generation_kwargs: dict[str, Any] = (...)` statement (it ends `else {}` / `)`), add:

```python
                    if self.stream_usage.usage and _accepts_keyword(save_callback, "provider_usage"):
                        generation_kwargs["provider_usage"] = dict(self.stream_usage.usage)
```

- [ ] **Step 7: Record from the block** in `tldw_Server_API/app/core/Chat/chat_service.py`.
  - After the `from tldw_Server_API.app.core.LLM_Calls.structured_generation import (...)` block (it ends `parse_and_validate_structured_output,` / `)`), add:

```python
from tldw_Server_API.app.core.LLM_Calls.usage_accounting import _usage_tokens
```

  - In `execute_streaming_call`'s `save_callback`, after `generation: dict[str, Any] | None = None,`, add `provider_usage: dict[str, Any] | None = None,`.
  - Replace the block from `        # Usage logging (estimated) after stream completes` through `                res = rg_commit_cb(total_est)` with:

```python
        # Usage logging after the stream completes: from the provider's usage chunk, else an estimate
        total_tokens = 0
        try:
            reported_prompt, reported_completion = _usage_tokens(provider_usage) if provider_usage else (0, 0)
            reported = (reported_prompt, reported_completion) != (0, 0)
            # A side the block does not count (a prompt-only block, say) comes from the estimate.
            prompt_tokens = reported_prompt
            if not prompt_tokens:
                try:
                    prompt_tokens = _estimate_tokens_from_messages(templated_llm_payload)
                except _CHAT_NONCRITICAL_EXCEPTIONS:
                    prompt_tokens = 0
            completion_tokens = reported_completion or max(0, len(full_reply or "") // 4)
            user_id = None
            api_key_id = None
            try:
                if request is not None and hasattr(request, "state"):
                    user_id = getattr(request.state, "user_id", None)
                    api_key_id = getattr(request.state, "api_key_id", None)
            except _CHAT_NONCRITICAL_EXCEPTIONS:
                pass
            latency_ms = int((time.time() - llm_start_time) * 1000)
            total_tokens = int(prompt_tokens + completion_tokens)
            await log_llm_usage(
                user_id=user_id,
                key_id=api_key_id,
                request=request,
                endpoint=(f"{request.method}:{request.url.path}" if request else "POST:/api/v1/chat/completions"),
                operation="chat",
                provider=selected_provider,
                model=model,
                status=200,
                latency_ms=latency_ms,
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                total_tokens=total_tokens,
                request_id=(request.headers.get("X-Request-ID") if request else None) or (get_request_id() or None),
                conversation_id=(str(final_conversation_id) if final_conversation_id is not None else None),
                usage_metadata=provider_usage if reported else None,
                estimated=None if reported_prompt and reported_completion else True,
                estimate_source=None if reported else "stream_estimate",
            )
        except _CHAT_NONCRITICAL_EXCEPTIONS:
            pass
        # Commit the recorded tokens to the Resource Governor, if provided
        try:
            if callable(rg_commit_cb):
                # rg_commit_cb may be async or sync; call accordingly
                res = rg_commit_cb(total_tokens)
```

  Check: `git grep -n "total_est\|pt_est\|ct_est" -- tldw_Server_API/app/core/Chat/chat_service.py` prints nothing.

- [ ] **Step 8: Forward the usage chunk only to a client that asked (Decision 2).**
  - In `tldw_Server_API/app/core/Chat/streaming_utils.py`, `StreamingResponseHandler.__init__`, directly after the `self.stream_usage = _StreamUsage()` line from Step 6, add:

```python
        # OpenAI sends a usage chunk with empty choices only to a client that asks
        # (stream_options.include_usage); the chat pipeline sets False when its client did not.
        self.forward_usage_chunk = True
```

  - In `process_line`, replace the three lines `if isinstance(data, dict):` / `self._attach_stream_metadata(data)` / `elif data not in (None, "", [], {}):` (the no-choices branch, after the choices branch's `return outputs, False`; this three-line sequence is unique) with:

```python
                    if isinstance(data, dict):
                        if not self.forward_usage_chunk and data.get("choices") == [] and data.get("usage") is not None:
                            # A usage chunk the client did not ask for: read above for the usage record, not sent.
                            self.update_activity()
                            return outputs, False
                        self._attach_stream_metadata(data)
                    elif data not in (None, "", [], {}):
```

  - In `create_streaming_response_with_timeout`'s signature, after `partial_save_callback: Optional[Callable[..., Any]] = None,`, add `forward_usage_chunk: bool = True,`. In its docstring, after the two-line `partial_save_callback` entry (its second line starts `unfinished reply; see` and names `StreamingResponseHandler.safe_stream_generator`) and before the blank line above `Yields:`, add:

```python
        forward_usage_chunk: False keeps a usage chunk with empty ``choices`` from the client
            (it is still read for the usage record)
```

  - In its body, after `handler.user_message_id = user_message_id`, add `handler.forward_usage_chunk = forward_usage_chunk`.
  - In `tldw_Server_API/app/core/Chat/streaming_pipeline.py`, add the field `forward_usage_chunk: bool | None = None` after `partial_save_callback: Callable[..., Any] | None = None` in `StreamingPipelineRequest`, and the entry `"forward_usage_chunk": request.forward_usage_chunk,` after `"partial_save_callback": request.partial_save_callback,` in `create_chat_streaming_response`'s `optional_kwargs`. A `False` is not `None`, so it reaches the factory.
  - In `tldw_Server_API/app/core/Chat/chat_service.py`, add the parameter `forward_usage_chunk: bool = False,` after `history_persistence_ack: bool = False,` in `execute_streaming_call`'s signature, and `forward_usage_chunk=forward_usage_chunk,` after `history_persistence_ack=history_persistence_ack or None,` in its `StreamingPipelineRequest(...)` call.
  - In `tldw_Server_API/app/api/v1/endpoints/chat.py`, directly before `def _build_streaming_commit_cb(`, add (two blank lines either side):

```python
def _client_asked_for_stream_usage(request_data: Any) -> bool:
    """Whether the client asked for a stream's usage chunk (OpenAI's ``stream_options.include_usage``).

    The OpenAI SDKs send ``stream_options`` at the top level of the body; other clients may put it in ``extra_body``.
    """
    extra_body = getattr(request_data, "extra_body", None)
    for options in (
        (getattr(request_data, "model_extra", None) or {}).get("stream_options"),
        extra_body.get("stream_options") if isinstance(extra_body, dict) else None,
    ):
        if isinstance(options, dict) and options.get("include_usage") is True:
            return True
    return False
```

  - In the same file, in the `execute_streaming_call(...)` call inside `_execute_stream_response`, after the `provider_factory_timeout=max(` argument (it ends `_provider_stream_prime_remaining(error_state),` / `),`), add `forward_usage_chunk=_client_asked_for_stream_usage(request_data),` before `), records_itself=True)`.
  - In `_build_streaming_commit_cb`'s docstring (review M5), replace `The callback is invoked by chat_service when the stream finishes with the` / `total estimated token count.  It must:` with:

```python
    The callback is invoked by chat_service when the stream finishes with the
    recorded token count: the provider's usage chunk when it sent one, else the
    estimate.  It must:
```

- [ ] **Step 9: Run; see GREEN.**
  - The Step 4 command. Expected: `11 passed`.
  - Then the files the change touches:

```bash
TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Chat/unit/test_streaming_generation_settlement.py tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py tldw_Server_API/tests/Chat/unit/test_streaming_utils.py tldw_Server_API/tests/Chat_NEW/integration/test_server_settled_generation_metadata.py tldw_Server_API/tests/Chat/integration/test_chat_endpoint_streaming_normalization.py
```

  Expected: `200 passed, 1 skipped`.
  - Then `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python Helper_Scripts/ci/test_quality_triage.py --enforce`. Expected: exit 0, "OK — no offenses beyond the baseline."
  - Then `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/ruff check tldw_Server_API/app/core/LLM_Calls/usage_accounting.py tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/app/core/Chat/streaming_pipeline.py tldw_Server_API/app/api/v1/endpoints/chat.py tldw_Server_API/tests/Chat/unit/test_streaming_generation_settlement.py tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py`. Expected: "All checks passed!".
  - Two files keep one pre-existing I001 each, on base and after; do not reformat them: `streaming_utils.py` (at `5:1`, the start of the import block) and `test_chat_endpoint_streaming_normalization.py` (at `1:1`).

- [ ] **Step 10: Commit** (message 1). `git add` the eight files: `tldw_Server_API/app/core/LLM_Calls/usage_accounting.py`, `tldw_Server_API/app/core/Chat/streaming_utils.py`, `tldw_Server_API/app/core/Chat/streaming_pipeline.py`, `tldw_Server_API/app/core/Chat/chat_service.py`, `tldw_Server_API/app/api/v1/endpoints/chat.py`, `tldw_Server_API/tests/Chat/unit/test_streaming_generation_settlement.py`, `tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py` and `tldw_Server_API/tests/Chat/integration/test_chat_endpoint_streaming_normalization.py`. Then run `git diff --cached --check`, then commit.

## Task 2: Ask OpenAI for a stream's usage chunk (Decision 1)

**Files:**
- Modify: `tldw_Server_API/app/core/LLM_Calls/providers/openai_adapter.py` (imports `:3-6`, `stream` `:347-375`)
- Test: `tldw_Server_API/tests/LLM_Adapters/unit/test_openai_native_http.py`

**Interfaces:**
- Consumes: Task 1 (the chat pipeline records the chunk this asks for).
- Produces: `OpenAIAdapter.stream` sends `stream_options` with `include_usage: True` when the request URL's host is `api.openai.com` or ends with `.api.openai.com` (OpenAI's data-residency hosts), merged under the caller's `stream_options`.

**RED reason (measured):** `2 failed, 3 passed`. Both cases of `test_an_openai_stream_asks_openai_for_its_usage_chunk` (`global`, `data-residency-region`) fail with `KeyError: 'stream_options'`. The caller-wins test and the two other-server cases (`other-server`, `lookalike-host`) pass: they pin what must not change.

- [ ] **Step 1: Make the base URL deterministic.** In the autouse fixture `_enable_native_http`, after `monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")`, add:

```python
    # These take precedence over OPENAI_BASE_URL (OpenAIAdapter._openai_base_url).
    monkeypatch.delenv("OPENAI_API_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
```

- [ ] **Step 2: Append the tests** to the end of `tldw_Server_API/tests/LLM_Adapters/unit/test_openai_native_http.py` (two blank lines before):

```python
def _streamed_payload(monkeypatch, **request_fields: Any) -> dict[str, Any]:
    """Stream one request through a fake client and return the JSON body the adapter sent."""
    import tldw_Server_API.app.core.LLM_Calls.providers.openai_adapter as openai_mod

    clients: list[_FakeClient] = []

    def _factory(*args, **kwargs):
        client = _FakeClient(*args, **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(openai_mod, "http_client_factory", _factory, raising=True)
    request = {"messages": [{"role": "user", "content": "hello"}], "model": "gpt-4o-mini", "api_key": "sk-test"}
    list(openai_mod.OpenAIAdapter().stream({**request, **request_fields}))
    return clients[0].last_stream["json"]


@pytest.mark.parametrize(
    "base_url",
    ["https://api.openai.com/v1", "https://eu.api.openai.com/v1"],
    ids=["global", "data-residency-region"],
)
def test_an_openai_stream_asks_openai_for_its_usage_chunk(monkeypatch, base_url):
    """OpenAI sends a stream's usage only when asked, so a stream to OpenAI's own hosts asks (TASK-13951)."""
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    payload = _streamed_payload(monkeypatch)

    assert payload["stream_options"] == {"include_usage": True}


def test_the_callers_stream_options_win_over_the_usage_request(monkeypatch):
    payload = _streamed_payload(
        monkeypatch, extra_body={"stream_options": {"include_usage": False, "include_obfuscation": False}}
    )

    assert payload["stream_options"] == {"include_usage": False, "include_obfuscation": False}


@pytest.mark.parametrize(
    "base_url",
    ["https://llm.example.internal/v1", "https://api.openai.com.example.net/v1"],
    ids=["other-server", "lookalike-host"],
)
def test_an_openai_provider_stream_to_another_server_is_not_asked_for_usage(monkeypatch, base_url):
    """Another server behind the openai provider may reject stream_options (TASK-13951)."""
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    payload = _streamed_payload(monkeypatch)

    assert "stream_options" not in payload
```

- [ ] **Step 3: Run; see RED.**

```bash
TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/LLM_Adapters/unit/test_openai_native_http.py -k "usage_chunk or stream_options_win or another_server"
```

Expected: `2 failed, 3 passed` (`KeyError: 'stream_options'`).

- [ ] **Step 4: Implement** in `tldw_Server_API/app/core/LLM_Calls/providers/openai_adapter.py`.
  - After `from typing import Any`, add `from urllib.parse import urlparse`.
  - In `stream`, directly after `payload = merge_extra_body(payload, request)` (the one followed by `headers = merge_extra_headers(self._openai_headers(api_key, request), request)`; `chat` has its own `merge_extra_body` and is not changed), add:

```python
            host = urlparse(url).hostname or ""
            if host == "api.openai.com" or host.endswith(".api.openai.com"):
                # OpenAI sends a stream's usage only when asked: a last chunk with empty choices, which the
                # usage record reads (TASK-13951). OpenAI's data-residency hosts (eu.api.openai.com, ...) are
                # asked too. The caller's own stream_options win. Another server behind this provider is not
                # asked, as it may reject the field.
                caller_options = payload.get("stream_options")
                payload["stream_options"] = {
                    "include_usage": True,
                    **(caller_options if isinstance(caller_options, dict) else {}),
                }
```

- [ ] **Step 5: Run; see GREEN.**
  - The Step 3 command. Expected: `5 passed`.
  - The whole file: `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/LLM_Adapters/unit/test_openai_native_http.py`. Expected: `9 passed`.
  - The five touched test files together: `... -q tldw_Server_API/tests/LLM_Adapters/unit/test_openai_native_http.py tldw_Server_API/tests/Chat/unit/test_streaming_generation_settlement.py tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py tldw_Server_API/tests/Chat/integration/test_chat_endpoint_streaming_normalization.py`. Expected: `96 passed`.
  - Ruff adds no finding. `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/ruff check tldw_Server_API/app/core/LLM_Calls/providers/openai_adapter.py` prints "All checks passed!". The test file keeps its 16 pre-existing findings: `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/ruff check tldw_Server_API/tests/LLM_Adapters/unit/test_openai_native_http.py` and `git show HEAD:tldw_Server_API/tests/LLM_Adapters/unit/test_openai_native_http.py | /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/ruff check --stdin-filename tldw_Server_API/tests/LLM_Adapters/unit/test_openai_native_http.py -` both print "Found 16 errors."
- [ ] **Step 6: Commit** (message 2): `git add` the two files, then `git diff --cached --check`, then commit.

## Task 3: Say which streamed replies are costed from provider usage

Behaviour is final after Task 2.

- [ ] **Step 1: `Docs/Operations/Usage_Quotas.md`.** Replace the whole bullet that starts `- **Streamed replies and router calls use the standard rate.**` (one line; it ends "so it too costs the standard rate.") with:

```markdown
- **Streamed replies and router calls.** A streamed reply is costed from the provider's last usage chunk when the provider sends one, on `/chat/completions` as on every other caller, so its service tier, cache rates and reported cost apply as they do to a reply that is not streamed. OpenRouter always sends one. OpenAI sends one because the server asks for it (`stream_options.include_usage`) on every stream to `api.openai.com` and OpenAI's data-residency hosts (such as `eu.api.openai.com`); an `openai` provider whose base URL points at another server is not asked, since that server may reject the field, so a caller that wants it there sets `stream_options` in `extra_body`. A stream without a usage chunk, or with one that counts no tokens, is costed from an estimate at the standard rate: the prompt estimate plus a quarter of the reply's characters (a tool-call-only reply counts no completion tokens). That includes every stream through the Anthropic and Google adapters, which pass no usage chunk on, and other OpenAI-compatible providers unless they send one unasked or `extra_body` asks for it. On `/chat/completions`, a usage chunk that counts only one side (prompt tokens only, say) takes the other side from that estimate, and the row is marked estimated. A `/chat/completions` stream that the client drops, or that fails after it starts, writes no usage row, and its Resource Governor reservation is refunded. An auto-router call (the model choice made before a `/chat/completions` or character chat reply) is recorded from the provider's token counts without its usage block, so it costs the standard rate.
```

  The "**Prompt-cache tokens.**" bullet's last sentence (a streamed reply through the Anthropic adapter is costed at the input rate) stays true until TASK-13916 and is not changed.

- [ ] **Step 2: `Docs/API-related/Chat_API_Documentation.md`.**
  - After the line `Note: Stream chunks follow OpenAI-style `choices[].delta.content` for maximum client compatibility.`, add a blank line and:

```markdown
A stream's usage (the provider's token counts for the whole request) arrives on a usage chunk, and the usage record is costed from it whether or not the client receives it. OpenAI sends it only when asked, as a last chunk whose `choices` array is empty. The server asks OpenAI for it on every stream to `api.openai.com` and OpenAI's data-residency hosts (such as `eu.api.openai.com`), but forwards it only to a client that asked for it itself with `"stream_options": {"include_usage": true}`, at the top level of the request (as the OpenAI SDKs send it) or in `extra_body`. A client that asks must not assume `choices[0]` exists on that chunk. Every other OpenAI chunk carries `"usage": null`, as in OpenAI's own streams. OpenRouter always sends usage, on a last chunk with one content-free choice, and that chunk is forwarded unchanged. Other providers send usage only if they do so unasked or `extra_body` asks for it.
```

  - In the "Streaming Event Format" table, after the `| Delta chunk ...` row, add:

```markdown
| Usage chunk    | `data: {"choices":[],"usage":{"prompt_tokens":...,"completion_tokens":...}}` | Last provider chunk; sent only to a client that set `stream_options.include_usage` |
```

- [ ] **Step 3: `Docs/ADR/067-llm-calls-accounted-in-provider-base-class.md`** (ADR-067 is updated by later tasks as practice: TASK-13437, TASK-13585, TASK-13610).
  - Replace the line `- **Stream prompt tokens are estimated** unless the provider sends a usage chunk, and no adapter asks for one today.` with:

```markdown
- **Stream prompt tokens are estimated** unless the provider sends a usage chunk. The OpenAI adapter asks `api.openai.com` and OpenAI's data-residency hosts for one (`stream_options.include_usage`, since TASK-13951); no other adapter asks. The chat pipeline reads a streamed reply's usage chunk with the same parser as this recorder.
```

  - Replace the line that starts `- **Streams.** A stream is recorded once, however it ends:` (review M3; it ends "because nothing was received.") with the line below. It names no follow-up id, because this commit lands before Task 4 files F1.

```markdown
- **Streams.** A stream is recorded once by this recorder, however it ends: completion, a provider error or an early close. A stream that is opened and never read writes no row, because nothing was received. The chat pipeline's `/chat/completions` stream, which records itself, writes no row when the client drops it or it fails after it starts (found by TASK-13951; a follow-up owns it).
```

- [ ] **Step 4: `Docs/API-related/CHARACTER_CHAT_API_DOCUMENTATION.md`** (review M4).
  - Replace the `/chat/completions` section's line that starts `Streaming behavior follows the core Chat API:` (it ends "Duplicate terminal markers are suppressed.") with the same line plus one sentence:

```markdown
Streaming behavior follows the core Chat API: the server sends an initial `event: stream_start`, emits delta chunks as OpenAI-style `choices[].delta.content`, and terminates with a single `data: [DONE]` (heartbeat comments are sent periodically). Duplicate terminal markers are suppressed. A usage chunk with empty `choices` is sent only to a client that asks for one, as the Chat API documentation describes.
```

  - In `complete-v2`'s "Streaming Behavior" list, after the bullet that starts `- If the upstream provider already emits SSE-formatted lines` (it ends "they are forwarded as-is."), add:

```markdown
- Those lines include any usage chunk the provider sends. The server asks OpenAI (`api.openai.com` and its data-residency hosts) for one, so an OpenAI stream's chunks carry `"usage": null` and its last chunk before `[DONE]` has an empty `choices` array and the request's token counts: do not assume `choices[0]` exists.
```

- [ ] **Step 5: Refresh the mirror.** Run `bash Helper_Scripts/refresh_docs_published.sh`.
  - Expected: `git status --short Docs/Published` lists exactly `Docs/Published/ADR/067-llm-calls-accounted-in-provider-base-class.md`, `Docs/Published/API-related/CHARACTER_CHAT_API_DOCUMENTATION.md` and `Docs/Published/API-related/Chat_API_Documentation.md` (measured: 3 files, 7 insertions, 3 deletions).
  - `Usage_Quotas.md` is not mirrored.
- [ ] **Step 6: Run the docs tests:** `TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Docs`. Expected: `212 passed`.
- [ ] **Step 7: Commit** (message 3): `git add` the seven doc files (four sources, three mirrors), then `git diff --cached --check`, then commit.

## Task 4: Verify, close TASK-13951, file follow-ups, open the PR

- [ ] **Step 1: Verify.** Run each of these:
  - The focused regression set. Measured: `2357 passed, 2 skipped` (Task 0's baseline plus 13).
  - The wider set. Measured: `1404 passed, 2 skipped` (the baseline plus the endpoint test's 3 cases; the same two standing skips).
  - The empty-`choices` set (command below; compare with this branch's Task 0 baseline). Measured on `b94318cce9`: `5 failed, 2032 passed, 1 skipped` before, `5 failed, 2043 passed, 1 skipped` after (11 new tests). The 5 failures are the same on both: `test_summarization_adapter.py::test_a_local_provider_reply_reaches_the_summary_code` for kobold (`unauthorized`, `forbidden`, `rate-limit`, `unpulled-model`, `bad-request`), which fail only in this file order; the file alone gives `127 passed`, and it passes in the wider set. Any other failure, or a different count of these, is a regression until shown otherwise.

```bash
TLDW_TEST_NO_DOCKER=1 /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest -p no:cacheprovider -q -rs tldw_Server_API/tests/Audio/test_speech_chat_service.py tldw_Server_API/tests/AuthNZ_Unit/test_byok_validation_boundary.py tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_completion_precheck.py tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_session_error_mapping.py tldw_Server_API/tests/Chat_NEW/integration/test_roleplay_resume_completion.py tldw_Server_API/tests/Chat_NEW/integration/test_server_settled_generation_metadata.py tldw_Server_API/tests/Chat/integration/test_chat_endpoint_simplified.py tldw_Server_API/tests/Chat/integration/test_chat_endpoint_streaming_normalization.py tldw_Server_API/tests/Chat/unit/test_chat_helpers.py tldw_Server_API/tests/Chat/unit/test_chat_service_call_params.py tldw_Server_API/tests/Chat/unit/test_chat_service_content.py tldw_Server_API/tests/Chat/unit/test_chat_service_fallback.py tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py tldw_Server_API/tests/Chat/unit/test_chat_service_tool_autoexec.py tldw_Server_API/tests/Chat/unit/test_messages_endpoint_usage.py tldw_Server_API/tests/Chat/unit/test_streaming_generation_settlement.py tldw_Server_API/tests/LLM_Adapters/unit/test_notes_graph_suggestion_call_policy.py tldw_Server_API/tests/LLM_Adapters/unit/test_prompt_improvement_call_policy.py tldw_Server_API/tests/LLM_Calls/test_adapter_validation.py tldw_Server_API/tests/LLM_Calls/test_custom_openai_top_p.py tldw_Server_API/tests/LLM_Calls/test_llm_providers.py tldw_Server_API/tests/LLM_Calls/test_local_llm_param_forwarding.py tldw_Server_API/tests/LLM_Calls/test_openai_compatible_provider_adapters.py tldw_Server_API/tests/LLM_Calls/test_provider_timeout_and_role_regressions.py tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py tldw_Server_API/tests/Media_Ingestion_Modification/test_chatllm_ocr_backend.py tldw_Server_API/tests/Media_Ingestion_Modification/test_llamacpp_ocr_backend.py tldw_Server_API/tests/prompt_studio/unit/test_prompt_executor_structured.py tldw_Server_API/tests/RAG_NEW/unit/test_runtime_provider_call.py tldw_Server_API/tests/VN_Play/test_vn_play_db.py tldw_Server_API/tests/VN_Play/test_vn_play_turns.py
```

  - `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python Helper_Scripts/ci/check_shard_coverage.py`. Expected: `new_uncovered=0`.
  - `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python Helper_Scripts/ci/test_quality_triage.py --enforce`. Expected: exit 0.
  - `git diff origin/dev --check`: it must print nothing.
  - The `scope_predicate_baseline.txt` grep from Global Constraints.
- [ ] **Step 2: Bandit** on the touched source, through `uvx`: `/opt/homebrew/bin/uvx bandit -q -r tldw_Server_API/app/core/Chat/streaming_utils.py tldw_Server_API/app/core/Chat/streaming_pipeline.py tldw_Server_API/app/core/LLM_Calls/usage_accounting.py tldw_Server_API/app/core/LLM_Calls/providers/openai_adapter.py tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/app/api/v1/endpoints/chat.py -f json -o <scratchpad>/bandit_13951.json`.
  - Measured: 0 results (`loc` 17868; `chat.py` and `streaming_pipeline.py` joined the list).
  - Never install bandit into the shared venv. If `uvx` cannot fetch it, record the environment skip under DoD #4.
- [ ] **Step 3: Close TASK-13951** (backlog-py, written out). Fill in `<bandit result>` from Step 2 and the real counts before running.

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-13951 \
  --append-notes "RED/GREEN (record the real counts): Task 1 6 failed, 5 passed -> 11 passed (Task 1 files 200 passed, 1 skipped); Task 2 2 failed, 3 passed -> 5 passed (test_openai_native_http.py 9 passed); the five touched test files 96 passed; focused set 2344 passed, 2 skipped -> 2357 passed, 2 skipped; wider set 1401 passed, 2 skipped -> 1404 passed, 2 skipped (2 standing skips); empty-choices set 5 failed, 2032 passed, 1 skipped -> 5 failed, 2043 passed, 1 skipped (the same 5 order-dependent kobold summarization failures before and after); tests/Docs 212 passed; test_quality_triage --enforce OK; shard coverage new_uncovered=0." \
  --append-notes "How: StreamingResponseHandler feeds every parsed stream payload to usage_accounting._StreamUsage.feed_event (split out of feed, so both recorders share one parser) and passes the whole usage block to a save_callback that accepts provider_usage; chat_service's save_callback records from it (tokens via _usage_tokens, the block as usage_metadata), takes a side the block does not count from the stream estimate (marked estimated), and records the stream estimate when there is no block or it counts no tokens; the Resource Governor commit gets the recorded total. The handler forwards an empty-choices usage chunk only when the client asked (stream_options.include_usage at the top level or in extra_body; chat._client_asked_for_stream_usage, carried as forward_usage_chunk through execute_streaming_call and StreamingPipelineRequest); OpenRouter's usage chunk carries a choice and is forwarded as before. OpenAIAdapter.stream asks api.openai.com and OpenAI's data-residency hosts (*.api.openai.com) for the usage chunk (stream_options.include_usage, merged under the caller's stream_options). No double recording: /chat/completions runs with records_itself=True." \
  --append-notes "Sources, fetched 2026-10-26: developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create and .../streaming-events (include_usage adds a last chunk with empty choices; every other chunk has usage null; the chunk carries service_tier; an interrupted stream may lack the usage chunk); developers.openai.com/api/docs/guides/your-data (data-residency hosts are a prefix on api.openai.com: us., eu., au., ca., jp., in., sg., kr., gb., ae.); openrouter.ai/docs/use-cases/usage-accounting (usage in the last SSE message, always; include_usage deprecated, no effect); openrouter.ai/docs/api-reference/streaming (OpenRouter's usage chunk carries one content-free choice, not empty choices)." \
  --append-notes "Decisions (owner-approved plan): ask for usage only on OpenAI's own hosts, caller's stream_options win; read the usage chunk for the record and forward an empty-choices usage chunk only to a client that asked (c'), so clients receive what they did before; the recorder reads any provider's usage chunk, other providers are asked in a follow-up; dropped or failed streams unchanged (no row), follow-up filed high. Premise corrections: a dropped or failed /chat/completions stream wrote no usage row before this change either (save_callback runs only on a complete stream; rg_refund_cb refunds), verified with a throwaway test; and OpenRouter's usage chunk carries one choice, so no /chat/completions client received an unrequested empty-choices chunk before this change (the plan's first draft assumed otherwise; the independent review caught it)." \
  --append-notes "Out of scope, unchanged: the Anthropic adapter's stream passes no usage chunk on (TASK-13916), nor does the Google adapter's (TASK-13952 item 5), so those streams are still costed from the estimate; both now reach /chat/completions with no further change once their adapter emits the chunk." \
  --append-notes "Docs: Docs/Operations/Usage_Quotas.md (the streamed replies bullet), Docs/API-related/Chat_API_Documentation.md (the usage chunk and who receives it), Docs/API-related/CHARACTER_CHAT_API_DOCUMENTATION.md (complete-v2 forwards OpenAI's usage chunk raw) and ADR-067's two stream consequences, with the Docs/Published mirror refreshed. Bandit: <bandit result>." \
  --check-ac 1 --check-ac 2 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "A streamed /chat/completions reply is recorded from the usage chunk its provider sends, so its service tier, cache rates and OpenRouter's reported cost apply as for a reply that is not streamed; a stream without one is still recorded from the estimate. The stream handler reads usage with the provider-level recorder's own parser (_StreamUsage.feed_event) and hands the whole block to save_callback; the Resource Governor commit gets the recorded total. The OpenAI adapter asks api.openai.com and OpenAI's data-residency hosts for the usage chunk (stream_options.include_usage), which also gives real usage to every other caller's OpenAI stream; /chat/completions forwards that empty-choices chunk only to a client that asked for usage, so clients receive what they did before, and OpenRouter's usage chunk is forwarded as it always was. Anthropic and Gemini streams stay estimated until TASK-13916 and TASK-13952; dropped or failed streams still write no row (follow-up filed)." \
  -s Done
```

  `--check-dod 4` is for Bandit, run or skipped with a recorded reason.

- [ ] **Step 4: File the follow-ups under the id lock.**
  - Take the lock with `mkdir <scratchpad>/id_lock`. If it exists, wait; never break another session's lock.
  - Create `<scratchpad>/task_next.sh` with exactly this content (from `Docs/superpowers/plans/2026-10-22-chacha-off-loop-and-transaction-guard.md`, Task 5 Step 4):

```bash
#!/bin/zsh
# Highest backlog task id on origin/dev and in every open PR (run from the repository root).
{ git ls-tree -r --name-only origin/dev backlog/
  gh pr list --state open --limit 500 --json number,changedFiles,files \
    --jq '.[] | if .changedFiles > 100 then "PR#\(.number)" else .files[].path end' \
    | while IFS= read -r line; do
        if [[ $line == PR#* ]]; then
          gh api --paginate "repos/rmusser01/tldw_server/pulls/${line#PR#}/files" --jq '.[].filename'
        else
          printf '%s\n' "$line"
        fi
      done
} | sed -nE 's|^backlog/.*task-([0-9]+)[ .].*|\1|p' | sort -n | tail -1
```

  - Run `git fetch origin dev --quiet && zsh <scratchpad>/task_next.sh`. Take `<N1>` and `<N2>` as the printed value + 1 and + 2, check that no `task-<N1>` or `task-<N2>` file exists locally, and write each id literally in the commands below.
  - Release the lock after `gh pr view --json files` lists the new task files (Step 6).

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task create "A dropped or failed /chat/completions stream writes no usage row" \
  --id TASK-<N1> -l quotas,backend,high \
  -d "Found by TASK-13951 (verified 2026-10-26 with a throwaway test: a client dropped after the first content chunk left no llm_usage_log row and one rg_refund_cb(cancelled=True) call). execute_streaming_call's save_callback in chat_service.py is the only recorder of a /chat/completions stream (the call runs with records_itself=True), and StreamingResponseHandler calls it only when the stream completes without cancellation or error (streaming_utils.py, the save gate before 'Saved streaming response'). A stream the client drops, stops, or that fails after output starts therefore writes no usage row, and _finalize_stream refunds the Resource Governor reservation (rg_refund_cb commits 0 tokens). ADR-067 says the provider-level recorder records a stream once however it ends; the chat pipeline does not, and ADR-067 says so since TASK-13951. This is a per-user cost-limit bypass, not just an accounting gap: a client can read a whole reply, close the connection on finish_reason before the save, and the reply never counts toward the per-user LLM cost limits. Also, the stream estimate counts only the reply text: tool-call arguments and reasoning are not counted (a tool-call-only reply estimates 0 completion tokens), while the provider-level recorder's _StreamUsage.chars counts them. OpenAI notes an interrupted stream may lack its usage chunk, so an early end needs the estimate. Related: the provider-level recorder (usage_accounting._record_stream and _record_response) records a usage block that counts only one side (prompt tokens only, say) as given, so the reply counts 0 completion tokens; the chat pipeline takes the missing side from its estimate since TASK-13951." \
  --ac "A /chat/completions stream that ends early (client drop, stop, provider error after output) writes one usage row: from the provider's usage chunk if it arrived, else from the prompt estimate and the output received, with a test" \
  --ac "The Resource Governor and billing commit the recorded tokens for such a stream instead of refunding them, with a test" \
  --ac "The stream estimate counts tool-call arguments and reasoning text as the provider-level recorder does, with a test" \
  --ac "The provider-level recorder takes a side a usage block does not count from its estimate, as the chat pipeline does, with a test"
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task create "Ask the other OpenAI-compatible providers for a stream's usage chunk" \
  --id TASK-<N2> -l quotas,backend \
  -d "Found by TASK-13951, which asks only api.openai.com for a stream's usage chunk (stream_options.include_usage in OpenAIAdapter.stream). Streams from DeepSeek, Groq, Mistral, Qwen, Moonshot, Z.AI, HuggingFace and Bedrock reach both recorders with usage only if the provider sends a usage chunk unasked or the caller sets stream_options in extra_body; otherwise they are costed from an estimate that misses cache rates and reasoning tokens (for example deepseek-reasoner). Cohere's adapter stream turns the stream-end event, which carries Cohere's usage, into DONE (cohere_adapter.py). For each provider, read its API docs: if it documents a usage flag, ask for it in its adapter, only on the provider's own host as the OpenAI adapter does; if it sends usage unasked, pin that with a test; otherwise list it as estimated in Usage_Quotas.md. Local and custom OpenAI-compatible servers are left alone, since they may reject the field." \
  --ac "Each listed provider's stream is asked for usage (with a test), or is shown to send it unasked (with a test), or is listed as estimated in Usage_Quotas.md" \
  --ac "Cohere's stream passes its usage on, with a test, or Usage_Quotas.md says it is estimated"
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-13916 \
  --append-notes "From TASK-13951 (2026-10-26): /chat/completions now reads a streamed reply's usage with the provider-level recorder's own parser (usage_accounting._StreamUsage.feed_event, which StreamingResponseHandler calls for each parsed payload) and counts its tokens with _usage_tokens. So fixing the Anthropic adapter's stream usage reaches both recorders at once, whether the adapter puts a top-level usage on its OpenAI-shaped chunks or feed_event learns to read provider_response. Mind the merge: feed_event merges a later usage block over an earlier one only when type == 'message_delta'; an OpenAI-shaped chunk has no type, so emit one block per stream (message_start's usage with message_delta's merged over it) or keep the type. The selected provider 'anthropic' sends an Anthropic-shaped block through the normalizer's Anthropic branch, and _usage_tokens adds its cache tokens to the prompt (ADR-075). Add an Anthropic case to test_a_streamed_reply_is_recorded_from_the_usage_chunk_its_provider_sends (tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py, fixture stream_usage_rows)."
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-13952 \
  --append-notes "From TASK-13951 (2026-10-26), on item 5: until it lands, a streamed Gemini /chat/completions reply is still costed from the stream estimate (TASK-13951's AC #2 path), because GoogleAdapter.stream drops usageMetadata. When the stream ends with one chunk {\"choices\": [], \"usage\": <the block _normalize_to_openai_shape builds, cachedContentTokenCount included>} (Gemini's usageMetadata is cumulative, so the last one is the total), both recorders read it with no other change: StreamingResponseHandler feeds it to _StreamUsage, and the selected provider 'google' takes the normalizer's Gemini branch. Test it with a Gemini case in test_a_streamed_reply_is_recorded_from_the_usage_chunk_its_provider_sends (tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py)."
```

  - Then `git add -A backlog/tasks && git diff --cached --check`, and commit (message 4).

- [ ] **Step 5: Open the PR.**
  - `git push -u origin fix/stream-usage-13951`, then `gh pr create --base dev --title "fix(usage): record a streamed chat reply from the provider's usage chunk (TASK-13951)"`.
  - The body states:
    - what changes: a streamed `/chat/completions` reply is recorded from the provider's usage chunk (tier, cache, reported cost), and from the estimate without one; the OpenAI adapter asks OpenAI's own hosts for the chunk; the Resource Governor commit gets the recorded total;
    - `chat.py`'s `_build_streaming_commit_cb` docstring now says the callback gets the recorded total, not an estimate;
    - what clients see: on `/chat/completions`, what they saw before. The empty-`choices` usage chunk OpenAI now sends reaches a client only when it asked (`stream_options.include_usage`, top level or `extra_body`); OpenRouter's usage chunk carries a choice and is forwarded as before. Each OpenAI chunk now carries `"usage": null`. Character chat's `complete-v2` forwards raw lines, so its clients also receive OpenAI's empty-`choices` chunk (documented);
    - what is unchanged: Anthropic and Gemini streams (TASK-13916, TASK-13952), and dropped or failed streams, which write no row (TASK-<N1>);
    - the four decisions, and that no schema changed.
  - The body ends with the attribution line from the session's reminder.
  - **ADR-004 Change summary:** this PR is not in the owner's recorded waivers. Ask the owner for the "## Change summary" section before merge.
  - Editing the PR body re-triggers the required license status, so wait for it before merging.
- [ ] **Step 6: Review and merge.**
  - Run `/superpowers:requesting-code-review` in a clean subagent (the owner's rule while Qodo credits are out) and hold until findings are addressed.
  - Check `gh variable get MERGE_QUEUE`. With `on`, arm auto-merge and leave the PR alone. Otherwise rebase only when it is next, and merge after the seven statuses are green.
  - Confirm MERGED before deleting the branch.
  - Release `<scratchpad>/id_lock` once the PR's file list shows the new task files.

## Follow-ups to file (Task 4 Step 4)

- **F1 (label `high`):** A dropped or failed `/chat/completions` stream writes no usage row, a per-user cost-limit bypass; the stream estimate counts no tool-call arguments or reasoning; and the provider-level recorder records a one-sided usage block as given. New task TASK-<N1> (Decision 4). Its AC requires the commit, with no "or document why not" branch.
- **F2:** Ask the other OpenAI-compatible providers (DeepSeek, Groq, Mistral, Qwen, Moonshot, Z.AI, HuggingFace, Bedrock) for a stream's usage chunk, and pass Cohere's on. New task TASK-<N2> (Decision 3).
- **Notes on existing tasks:** TASK-13916 (how the Anthropic fix reaches both recorders, and the merge rule) and TASK-13952 item 5 (the chunk shape that both recorders read).

## How this interacts with TASK-13916 and TASK-13952

- **TASK-13916 (Anthropic adapter).**
  - Not duplicated: this plan does not touch `AnthropicAdapter.stream`.
  - Both recorders now share `_StreamUsage.feed_event`, so 13916's fix needs no `/chat/completions` change, whether it puts a top-level `usage` on the adapter's chunks or teaches `feed_event` to read `provider_response`.
  - The one trap is the merge rule (`message_delta` only). The note in Task 4 Step 4 says how to avoid it.
  - Until 13916 lands, Anthropic streams on `/chat/completions` keep the estimate (AC #2 path), as today.
- **TASK-13952 item 5 (Google adapter).**
  - Until it lands, a streamed Gemini reply is costed from the stream estimate, exactly as today: no cache-read rate and no thinking tokens.
  - This change makes the fix one-sided: once `GoogleAdapter.stream` emits one OpenAI-shaped usage chunk at the end, both recorders read it. The selected provider `google` routes it through the normalizer's Gemini branch, which reads `cachedContentTokenCount`.

## ADR assessment

No new ADR. The change adds a method (`feed_event`), one keyword to the stream save callback, a request field for one provider, a forwarding flag on the chat stream, and a different input to an existing recorder. The forwarding flag changes one default, what `/chat/completions` forwards; it keeps what clients receive, and Task 4 runs every test file that streams an empty-`choices` chunk. The wider set covers every test file that names the touched symbols.
- ADR-067's consequence "no adapter asks for one today" becomes false, and its "Streams" consequence does not hold for the chat pipeline (review M3), so Task 3 updates both and refreshes the mirror. ADR-067 has been updated this way by TASK-13437, TASK-13585 and TASK-13610.
- Governing decisions: ADR-067 (provider-level accounting, `records_itself`), ADR-075 (Anthropic-shaped prompt tokens include cache tokens; `_usage_tokens` applies it) and ADR-059 (backlog-py).

## Self-review against the spec

- **AC #1** (a stream whose provider sends a usage chunk is recorded with that usage: service tier, cache tokens, reported cost; with a test):
  - Task 1's `test_a_streamed_reply_is_recorded_from_the_usage_chunk_its_provider_sends` goes through the real `log_llm_usage` and the shipped catalog. The OpenAI Fast tier with cached input and OpenRouter's reported cost are each costed exactly, and the row keeps the tier. Its OpenRouter case uses OpenRouter's documented chunk shape (one content-free choice).
  - The handler test pins the whole block reaching the recorder. The tool-call test pins a tool-call stream. The one-sided block test pins a prompt-only block.
  - Task 2 makes OpenAI send the chunk, on its data-residency hosts too.
- **AC #2** (no usage chunk, still the estimate): Task 1's `test_a_streamed_reply_without_usage_is_recorded_from_the_estimate`, with three cases (none, `null`, zero).
- **Decision 2 (c'):** the AC #1 test (a client that did not ask gets OpenAI's chunk consumed and OpenRouter's forwarded) and the endpoint test (not asked; asked at the top level; asked in `extra_body`).
- **The description's "TASK-13916 ... that alone does not change /chat/completions":** after this plan it does, through the shared parser (TASK-13916 note).
- **The brief's checklist:**
  - per-provider chunk visibility, swallowing and filtering: "Each provider's stream";
  - double recording: "No double recording";
  - `include_usage`: Decision 1; forwarding: Decision 2; providers: Decision 3;
  - cancellation: "What a dropped stream records today" and Decision 4 (premise corrected);
  - tool-call and auto-exec streams: the tool-call test and "Settled by the code";
  - character chat and other streaming endpoints: "No double recording", "Character chat" and the character chat API doc (Task 3 Step 4).
