# Gemini pricing gaps: thinking tokens, the stream's usage chunk, Flex and Priority tiers, the newer models (TASK-13952) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Gemini reply is costed in full: its thinking tokens as output, its tool-use prompt tokens as input, at the Flex or Priority rates when `usageMetadata.serviceTier` says the call ran there; a streamed Gemini reply is recorded from the usage chunk the adapter now ends every stream with, also when the stream fails mid-way, instead of the estimate; the catalog carries Google's current rates, tiers and the four newer models; what is still not priced (audio input, Batch, Contributor, the 2.0 Flash entries, the dated price changes, the thinking tokens of a stopped or dropped stream) is recorded in `Usage_Quotas.md`, with a follow-up task for audio, the dated changes and the stopped or dropped stream.

**Architecture:** One function builds the OpenAI-shaped usage block from a Gemini `usageMetadata`: the new `GoogleAdapter._usage_block`, used by `_normalize_to_openai_shape` (the reply) and by `_stream_event_deltas`, which keeps the last block seen in the stream's state so `stream()` can yield it as one `{"choices": [], "usage": ...}` chunk before `[DONE]`, or before the error when the stream fails mid-way (`_usage_chunk`; `pop` makes it once, and a block with no count sends none). A proxy's `[DONE]` line ends the stream. Both recorders already read such a chunk: the provider-level `_StreamUsage.feed` and the chat handler's `feed_event`, through `usage_with_service_tier` and `_usage_tokens`. The block puts `thoughtsTokenCount` into `completion_tokens` and `toolUsePromptTokenCount` into `prompt_tokens`, keeps both under their Gemini names (the row's `reasoning_tokens`; the audit JSON), keeps `cachedContentTokenCount` and adds `service_tier`, which `normalize_llm_usage` already reads for any provider and `compute_costs` prices through the entry's `service_tiers`. `normalize_llm_usage`'s Gemini branch adds the thoughts to `output_tokens` only when it read raw `usageMetadata` keys (no `completion_tokens`), so an adapter block is never double counted. `model_pricing.json`'s Gemini rows gain `service_tiers` (flex, priority) and four rows (gemini-4-argon, gemini-3.8-flash, gemini-3.6-flash, gemini-3.5-flash-lite), written by a script from the page tables that live in the test module, as TASK-13936 did.

**Tech Stack:** Python 3.12, pytest (asyncio auto mode, xdist), httpx `MockTransport` for the adapter's transport, the repo's `llm_accounting.drain_accounting`.

**Spec:**
- `backlog/tasks/task-13952 - Gemini-pricing-gaps-thinking-tokens-are-not-costed-Batch-Flex-Priority-tiers-audio-rates-stream-usage.md` (description items 1-6, ACs #1-#3, Implementation Notes);
- the planner brief `brief_plan_13952.md` (planning session's plans dir);
- the executed plans `Docs/superpowers/plans/2026-10-26-pricing-tiers.md` (TASK-13936: `service_tiers`, Gemini `cache_read`, the JSON edit script) and `Docs/superpowers/plans/2026-11-03-stream-usage-on-early-end.md` (TASK-13953: how a stream's usage chunk is recorded now: `_StreamUsage`, the one-sided fill, `estimated`);
- ADR-067 (`Docs/ADR/067-llm-calls-accounted-in-provider-base-class.md`).

**Measured on:** `origin/fix/stream-usage-early-end-13953` at `6332d72b09` (PR #3413's head, TASK-13953, in the merge queue on 2026-11-03). It is `origin/dev` plus #3413, which touches `Docs/Operations/Usage_Quotas.md`, `tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py`, `chat_service.py`, `streaming_utils.py` and `usage_accounting.py`, all of which this plan reads or edits; so this plan branches from `origin/dev` only after #3413 has merged (Task 0). Everything was measured in a detached worktree, which was reverted afterwards.

The prototype is `proto_13952.patch` in the plans dir: 11 files, +489/−40. It holds every code, test, catalog and doc change in this plan, but not the plan copy or the task files. Every RED/GREEN count below was measured on that patch's text: first the tests on unchanged code, then the code. The patch was rebuilt from this plan's steps on a clean `6332d72b09` and compared byte for byte (`cmp`) after the plan review was folded in (the last section).

`file:line` citations are for `6332d72b09`.

---

## What was verified (on `6332d72b09`)

| # | Claim | Evidence | Verdict |
|---|---|---|---|
| V1 | Thinking tokens are not costed | **Yes.** `GoogleAdapter._normalize_to_openai_shape` (`google_adapter.py:441-449`) sets `completion_tokens` from `candidatesTokenCount` alone and drops `thoughtsTokenCount`. Both recorders read that block's `completion_tokens`: the provider-level `_record_response`/`_record_stream` through `_usage_tokens` (`usage_accounting.py:100-115`, `:223-262`), and the chat pipeline's `_record_dict_response_usage` through `normalize_llm_usage` (`chat_service.py:5798-5806`; `llm_usage_normalizer.py:124-133` reads `completion_tokens` first). `log_llm_usage` prices the caller's `completion_tokens` (`usage_tracker.py:405-418`). Google: "When thinking is turned on, response pricing is the sum of output tokens and thinking tokens" (thinking guide); the pricing page's output label is "Output (including thinking tokens)" on every text model; `UsageMetadata.totalTokenCount` = "prompt + tool-use prompt + thoughts + response candidates" (API reference, all fetched 2026-11-03, see Sources). Reproduced: the repo fixture (80 prompt, 30 candidates, 5 thoughts, total 115) records `completion_tokens == 30`. | **fix** |
| V2 | `total_tokens` after the fix | `total_tokens` stays Gemini's `totalTokenCount`, which already summed the thoughts. On `/chat/completions` the row's `total_tokens`, the Resource Governor commit and the daily token ledger take it (`chat_service.py:5838`, `_record_stream_usage`), so the thoughts were counted toward those limits there but not costed. On the provider-level recorder (RAG, workflows, every other caller) `_submit_record` passes no total and `log_llm_usage` sums the two sides (`usage_tracker.py:381`), so the thoughts were missing from its rows, costs and limits alike. With `toolUsePromptTokenCount` folded into `prompt_tokens`, `prompt + completion == total` for every Gemini block (pinned), so both paths agree after the fix. | **fix (prompt side too)** |
| V3 | No other consumer double counts | `reasoning_tokens` (`llm_usage_normalizer.py:134-141`, from `thoughtsTokenCount`) only reaches the row's `reasoning_tokens` column (`usage_tracker.py:530`; `usage_repo.py:851-919`); `compute_costs` (`usage_tracker.py:257-335`) and the limits never read it. The chat pipeline's estimate fill (`_record_stream_usage`, `chat_service.py:4031-4040`) uses `_usage_tokens`. So keeping `thoughtsTokenCount` in the block beside the folded `completion_tokens` double counts nothing. `normalize_llm_usage`'s raw-key fallback (`candidatesTokenCount`, `:130`) is reached only when a block has no `completion_tokens`/`output_tokens`: no production path passes raw `usageMetadata` (`git grep usageMetadata` over `app/` hits only the adapter), so the two tests that do (`test_llm_usage_normalizer.py:53-70`, `test_chat_service_content.py:485-550`) are updated to count the thoughts there too (fail closed), guarded against the adapter block. | **fix (2 lines)** |
| V4 | The stream emits no usage | **Yes.** `GoogleAdapter.stream` (`google_adapter.py:601-681`) yields deltas from `_stream_event_deltas` (`:460-523`), which never reads `usageMetadata`, then `finalize_stream` (`:678`). Gemini sends no `[DONE]`; the `is_done_line` branch (`:644-648`) serves proxies. `_StreamUsage.feed` (`usage_accounting.py:125-147`) reads any `data:` line with `"usage"`; the chat handler feeds every parsed payload to `feed_event` (TASK-13953's V2). Reproduced: a stubbed two-event stream yields two content deltas and `[DONE]`, nothing else; the provider-level row is the estimate. A stream that fails mid-way raises from the `except` at `:680-681` (a transport error) or from `_raise_if_in_band_provider_error` (`:640-643`, an in-band error event: a `ChatAPIError`, which that `except` does not catch); both recorders then estimate, as they do for a stream the client drops or stops (TASK-13953). The adapter cannot yield after a drop or stop (its generator is closed), so those endings stay estimates here (follow-up TASK-15424, Decision 6). | **fix (end, `[DONE]`, failure); follow-up (drop, stop)** |
| V5 | Tiers: the response field | `UsageMetadata.serviceTier`, "Output only. Service tier of the request.", enum `ServiceTier`: `unspecified` ("Default service tier, which is standard"), `standard`, `flex`, `priority` (API reference, two fetches on 2026-11-03 print the values in lower case). No request-side field on `generateContent` (`GenerationConfig` has none; the raw API reference has `serviceTier` only as `UsageMetadata`'s output field and the `ServiceTier` enum). The Flex and Priority docs (`/docs/flex-inference`, `/docs/priority-inference`, fetched 2026-11-03) request a tier with `service_tier` on the Interactions API (`/v1beta/interactions`), which no server code calls (`git grep -n "interactions" -- tldw_Server_API/app/core/LLM_Calls`: nothing); Priority reports the tier a request actually ran on in the **`x-gemini-service-tier` response header**, and "Downgraded requests are billed at the standard rate". A caller's `extra_body.service_tier` is forwarded by `merge_extra_body`; if Google ever honoured it on `generateContent` and reported the tier only in that header, a Priority call would cost the standard rate (1.8x under). No header fallback is added: no server path requests a tier. No `batch` value: the Batch API is `POST .../models/{model}:batchGenerateContent`, a separate endpoint "at 50% of the standard cost" that no server code calls (`git grep batchGenerateContent` over `app/`: nothing). The page's fifth tab, Contributor (private preview, invite-only, half the standard price on gemini-4-argon, 3.8-flash, 3.5-flash-lite and 3.1-pro-preview), has no `ServiceTier` value either. So Flex and Priority are priced from the usage block; Batch and Contributor are recorded as not priced. | **fix (flex, priority); not priced (batch, contributor)** |
| V6 | Audio rates | The page gives separate audio input and cached-audio prices on gemini-3.1-flash-lite, gemini-3-flash-preview, gemini-2.5-flash and gemini-2.5-flash-lite (2x to 3.3x the text rate), on every tier; the modality split is in `promptTokensDetails`/`cacheTokensDetails` (`ModalityTokenCount`: `TEXT`, `IMAGE`, `VIDEO`, `AUDIO`, `DOCUMENT`), which the adapter drops. Pricing it needs two normalized fields, two catalog rates, a `compute_costs` split and the JSON rows (about 40 lines in the money path). | **not priced; follow-up (Decision 3)** |
| V7 | 2.0 Flash | `gemini-2.0-flash`, `-001`, `-lite` are on no page section (checked both fetches). Already recorded in `Usage_Quotas.md:112`; the rows keep their rates. | **recorded** |
| V8 | Newer models and dated changes | The page (2026-11-03) lists gemini-4-argon ($2.00 / $10.00 / $0.10, doubling on 2026-12-01), gemini-3.8-flash and gemini-3.6-flash ($0.75 / $3.75 / $0.075, doubling on 2027-01-01) and gemini-3.5-flash-lite ($0.30 / $2.50 / $0.03), none catalogued; the six catalogued models' standard rates are unchanged since 2026-10-26; gemini-2.5-pro and gemini-2.5-flash are deprecated, and from 2026-12-01 their requests route to, and are billed as, gemini-3.8-flash and gemini-3.5-flash-lite. So gemini-2.5-pro calls are over-charged until 2027-01-01 (3.8 Flash costs less) and under-charged on input and cache after it (3.8 Flash's doubled $1.50 / $0.15 against the catalogued $1.25 / $0.125); gemini-2.5-flash calls cost the same on standard text ($0.30 / $2.50 / $0.03 on both), and its audio is billed at the text rate. A catalog cannot switch on a date (`pricing_catalog.py` has no effective-date field). The page also lists specialised models (Live, TTS, Image, Transcribe, Embedding, Omni Flash, Robotics ER), none catalogued: a call to one costs the unknown-model rate. | **add the four text models at today's rates; follow-up (Decision 4)** |
| V9 | Pinned counts | `test_chat_service_content.py:538` (`completion_tokens == 30`), `:550` (metric 30) and `test_llm_usage_normalizer.py:66` (`output_tokens == 30`) pin the thinking-free count through the raw-key fallback (V3). No adapter test pins a full usage dict. | **update (3 asserts)** |
| V10 | Ratchets | Neither `google_adapter.py` nor any function this plan touches is in `tests/lint/function_size_baseline.json`; none of the touched files is in `Helper_Scripts/ci/scope_predicate_baseline.txt` (both grepped). `stream` is 88 lines after the change. The new test file is covered by a shard (`check_shard_coverage.py`: `new_uncovered=0`). | **constraint met** |

### Sources (all fetched 2026-11-03; `ai.google.dev/...` now redirects to `aistudio.google.com/...`)

- https://aistudio.google.com/docs/pricing (two fetches agreed). Standard tier, per million tokens, text/image/video, prompts up to 200k: gemini-4-argon 2.00 / 10.00 / 0.10 (through 2026-11-30; 4.00 / 20.00 / 0.20 from 2026-12-01); gemini-3.8-flash and gemini-3.6-flash 0.75 / 3.75 / 0.075 (through 2026-12-31; 1.50 / 7.50 / 0.15 from 2027-01-01); gemini-3.5-flash-lite 0.30 / 2.50 / 0.03; gemini-3.1-flash-lite 0.25 / 1.50 / 0.025 (audio 0.50 / 0.05); gemini-3.1-pro-preview 2.00 / 12.00 / 0.20 (over 200k: 4.00 / 18.00 / 0.40); gemini-3-flash-preview 0.50 / 3.00 / 0.05 (audio 1.00 / 0.10); gemini-2.5-pro 1.25 / 10.00 / 0.125 (over 200k: 2.50 / 15.00 / 0.25; deprecated, routes to gemini-3.8-flash from 2026-12-01); gemini-2.5-flash 0.30 / 2.50 / 0.03 (audio 1.00 / 0.10; deprecated, routes to gemini-3.5-flash-lite from 2026-12-01); gemini-2.5-flash-lite 0.10 / 0.40 / 0.01 (audio 0.30 / 0.03). Every output label reads "Output (including thinking tokens)". The Flex and Priority tables are transcribed in `GEMINI_TIER_PRICES` / `GEMINI_LONG_TIER_PRICES` (Task 3): Flex input and output are half the standard price, Priority 1.8x; cache rates vary per model (see the tables: Flex cache equals the standard cache on 3.1-pro-preview, 3-flash-preview, 2.5-pro, 2.5-flash and 2.5-flash-lite, and 3.5-flash-lite's Flex and Priority cache are $0.02 and $0.05). Batch input and output equal Flex's, but Batch is not a reportable tier (V5). The Contributor tab (invite-only) is not priced (V5). Neither 2.0 Flash model appears.
- https://aistudio.google.com/docs/thinking: "When thinking is turned on, response pricing is the sum of output tokens and thinking tokens." Thinking on by default: gemini-4-argon, gemini-3.8-flash, gemini-3.6-flash, gemini-3.5-flash-lite, gemini-3.1-pro-preview, gemini-3-flash-preview, gemini-2.5-pro, gemini-2.5-flash; off on gemini-2.5-flash-lite.
- https://aistudio.google.com/api-reference/generate-content, `UsageMetadata`: `promptTokenCount` ("When cached_content is set, this is still the total effective prompt size"), `cachedContentTokenCount`, `candidatesTokenCount` ("Total number of tokens across all the generated response candidates"), `toolUsePromptTokenCount`, `thoughtsTokenCount` ("Number of tokens of thoughts for thinking models"), `totalTokenCount` ("prompt + tool-use prompt + thoughts + response candidates"), `serviceTier` (V5), the `*TokensDetails` modality lists.
- https://aistudio.google.com/docs/batch-api: `batchGenerateContent`, "50% of the standard cost"; no tier parameter.
- https://aistudio.google.com/docs/flex-inference and https://aistudio.google.com/docs/priority-inference: a tier is requested with `service_tier` on the Interactions API (`/v1beta/interactions`); Priority reports the tier used in the `x-gemini-service-tier` response header; "Downgraded requests are billed at the standard rate" (V5).
- https://aistudio.google.com/docs/models: the model IDs above.

## Decisions

1. **ADR-004 Change summary.** This PR is AI-authored. The recorded waivers (memory: "ADR-004 waiver for spec 2") name many TASK-139xx PRs but not TASK-13952.
   - **Recommendation:** ask the owner to extend the waiver to this PR, as for each earlier follow-up. Otherwise the owner writes the `## Change summary` before merge.
2. **Batch is not priced** (item 2, part): a `generateContent` response cannot report it (V5) and no server code calls the Batch API. Unlike OpenAI's `batch` rows (TASK-13936 kept them for a proxy that reports the tier), a Gemini `batch` row would be dead: Gemini's enum has no such value. Recorded in `Usage_Quotas.md`.
   - **Recommendation:** keep it out; add rows if Google ever adds `batch` to `ServiceTier`. The Interactions API, where Flex and Priority are requested with `service_tier`, is not called either (V5), so no server path requests a tier; Contributor is not priced for the same reason Batch is (no `ServiceTier` value). The reviewer agreed.
3. **Audio input is not priced here** (item 3). The under-billing is bounded (up to 3.3x on the input side of a reply that carries audio, on four models; output is the same) and the fix is a money-path change in four files (V6). It gets its own task, filed in Task 0 as TASK-15422 (provisional id; Task 0 takes the real one), and `Usage_Quotas.md` sizes the gap with the page's rates.
   - **Owner question:** price it in this PR instead? **Recommendation:** no; the thinking fix is the high-priority item and should merge without the extra review surface. The follow-up is well specified (the adapter keeps `promptTokensDetails`/`cacheTokensDetails`, the normalizer sums the `AUDIO` entries, the catalog gets `prompt_audio`/`cache_read_audio` rates, `compute_costs` splits the uncached and cached audio shares).
4. **Dated price changes** (item 6). The catalog holds the rates in force on 2026-11-03. On 2026-12-01 gemini-4-argon doubles and the 2.5 Pro/Flash routing starts: gemini-2.5-pro calls are over-charged until 2027-01-01 (3.8 Flash costs less) and under-charged on input and cache after it (V8); gemini-2.5-flash calls cost the same on standard text. On 2027-01-01 gemini-3.8-flash and gemini-3.6-flash double. Filed in Task 0 as TASK-15423 (provisional id), labelled `high`: re-fetch and bump on each date.
   - **Owner question:** catalogue the post-change rates now instead (fail closed on cost: a 2x over-charge on gemini-4-argon for four weeks)? **Recommendation:** no; the catalog is a statement of Google's price list, and over-charging a paying user is a complaint, not a safety margin. The follow-up carries both dates.
   - **Owner (2026-11-03): today's rates, with a trigger.** If TASK-15423 slipped past 2026-12-01, gemini-4-argon would be under-charged by half with no end date. So TASK-15423 is labelled `high`, its description and the PR body name both dates (2026-12-01: gemini-4-argon 2x; 2027-01-01: gemini-3.8-flash and gemini-3.6-flash 2x, and gemini-2.5-pro's input and cache become under-charged), and the controller (not the executor) schedules one-time remote agents for 2026-11-30 and 2026-12-31 once TASK-15423's id exists (Task 0 Step 4).
5. **`estimated: true` leaves `gemini-3.1-flash-lite`.** The flag (from the 2026-05-10 catalog refresh, "approximate or source-dependent") is stale: the page has priced the model exactly since at least 2026-10-26, and the script rewrites the row from the page. Pinned by `test_a_gemini_model_carries_its_published_flex_and_priority_rates` (`estimated is False`).
6. **A stopped or dropped Gemini stream is costed without its thinking tokens** (review finding I1). Gemini thinks before it answers, so by the time a user sees content and presses Stop, Google has billed the whole thinking budget; each event's cumulative `usageMetadata` already carries `thoughtsTokenCount`, but the adapter cannot yield from a closed generator, so the recorders estimate from the prompt and the characters received, which leave the thinking out. Not a regression: every Gemini stream is an estimate on the base. The same holds for OpenAI reasoning tokens, which arrive only in a stream's final chunk.
   - **Owner (2026-11-03): gap as follow-up.** In this PR, a stream that fails mid-way after usage arrived sends the chunk before the error (Task 2), and `Usage_Quotas.md`'s Gemini bullet says a stopped, dropped or failed-before-usage stream is costed from the estimate without its thinking tokens. Task 0 files TASK-15424 (provisional id), labelled `high`: `_StreamUsage` keeps the last `GoogleAdapter._usage_block` of a delta's `provider_response` as a fallback when no usage chunk arrives. Never a usage chunk per event: clients that sum usage chunks (LangChain aggregates `usage_metadata`) would double count.

Not owner questions (rulings for the reviewer, each pinned by a test):
- **`toolUsePromptTokenCount` counts as input.** It is outside `promptTokenCount` (V1's `totalTokenCount` definition), Google bills tool-use prompts, and without it the row's `prompt + completion` is short of `total_tokens`. Fail closed: count it. Pinned by `test_tool_use_prompt_tokens_are_counted_as_input`.
- **`thoughtsTokenCount` and `toolUsePromptTokenCount` stay in the block** under their Gemini names, so the row's `reasoning_tokens` keeps the thinking share and the audit JSON keeps the split. Nothing prices them twice (V3). Pinned by `test_the_normalized_block_is_not_double_counted`.
- **The normalizer's raw-key fallback counts the thoughts** (V3), guarded by "no `completion_tokens`/`output_tokens` key and no explicit count". Pinned by the updated `test_normalize_gemini_usage_extracts_cached_content_and_thought_tokens` and `test_execute_non_stream_call_normalizes_gemini_usage_for_logging` (35), and against double counting by `test_the_normalized_block_is_not_double_counted` (an adapter block stays 35).
- **`service_tier` is passed on as reported**, lowercased, an enum prefix (`SERVICE_TIER_`) stripped in case the API ever prints the upper-snake form; `unspecified` and `standard` are passed on too and cost the standard rates (the catalog has no row for them, `_entry_rate_details`, `pricing_catalog.py:391`). Pinned by `test_the_service_tier_is_passed_on`.
- **The usage chunk goes out once, before `[DONE]`, in every ending the adapter sees**: Gemini's (the stream closes; the chunk goes before `finalize_stream`), a proxy's `data: [DONE]` line (the chunk goes before the adapter's own `sse_done()`, and the stream ends there: an event after it is not read), the non-SSE JSON-array body, and a failure mid-way (the chunk goes before the error: a transport error in the `_GOOGLE_ADAPTER_RUNTIME_EXCEPTIONS` `except`, an in-band error event in a new `except ChatAPIError` that re-raises it unchanged). `state.pop` makes it once. A stream with no `usageMetadata`, or an empty one, sends no chunk and is estimated as before. Pinned by `test_the_stream_ends_with_one_usage_chunk_from_the_last_usage_metadata[gemini-end|proxy-done|json-array]`, `test_a_stream_that_fails_after_usage_arrived_is_recorded_from_it[transport-error|in-band-error]` and `test_a_stream_without_usage_metadata_sends_no_usage_chunk[absent|empty]`.
- **The chunk is not forwarded to a `/chat/completions` client that did not ask for usage**, like OpenAI's empty-choices chunk (TASK-13951's Decision 2). Pinned by the Gemini cases of `test_a_streamed_reply_is_recorded_from_the_usage_chunk_its_provider_sends` (`forwarded=False`).
- **A missing count stays `None`** in the block (as on the base; `_StreamUsage.feed_event` drops nulls, `_usage_tokens` and `_usage_int` read them as 0). Pinned by `test_a_reply_without_thinking_or_cache_is_unchanged`.
- **The 2.5 -preview aliases carry their GA model's tiers**, as they carry its `cache_read` (TASK-13936's rule). Pinned by `test_a_gemini_2_5_preview_entry_carries_its_ga_model_tiers`.

---

## Global Constraints

Carried verbatim from the brief:

- Tests make no real outbound network requests; stub the provider adapter or its transport (a Gemini adapter test must not reach Google).
- Function-size ratchet `tldw_Server_API/tests/lint/function_size_baseline.json`: frozen functions must not grow.
- Backlog: edit TASK-13952 with backlog-py only (`PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task edit ...`); `--append-notes`, never `--notes`; stage with `git add -A backlog/tasks`; never hand-edit task files. If a new task must be filed, its explicit id comes from `zsh <plans dir>/task_next.sh` (prints the HIGHEST id in use; new id = that + 1) after `mkdir <scratchpad>/id_lock`; release it once the PR's file list shows the file.
- Branch from `origin/dev`; the PR targets `dev`. Commit trailer: `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. PR body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- Never `--no-verify`, never force-push, never `git stash`.
- Redis: never FLUSHALL or SCRIPT FLUSH. PostgreSQL only through the repo fixtures (`tldw_postgres_test`), serially, with a session-unique `TEST_DB_NAME`; drop only your own objects.
- Mirrored docs (`Docs/Published`) need `bash Helper_Scripts/refresh_docs_published.sh` in the same PR.
- CI checks before the PR, in the last task: `tests/Docs`, `tests/lint`, `tests/CI`, backlog-py's tests; shard coverage and `test_quality_triage.py --enforce`; the route-auth ratchet and the OpenAPI export check; Bandit on touched source; every test file that names a touched symbol (git grep), with a baseline. Commands: see `Docs/superpowers/plans/2026-11-03-stream-usage-on-early-end.md`'s Task 3. A file listed in `Helper_Scripts/ci/scope_predicate_baseline.txt` whose lines shift fails backend-required: check yours.
- Agent shells don't keep the venv `PATH`: prefix commands that spawn `python` with `PATH="$(dirname $PY):$PATH"`.

Plan-specific:

- **`$PY`** is `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`. Shell variables do not survive between agent calls, so write it out or set it in each call. `<scratchpad>` is the executing session's scratchpad directory, and `<plans dir>` is the planning session's plans dir: `/Users/macbook-dev/.claude/projects/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/plans`.
- **Base.** Branch from `origin/dev` after PR #3413 has merged (Task 0 Step 1 checks). If dev has moved further, Task 0 Step 5's baseline is the check: the focused set must still be `1473 passed, 1 skipped` plus whatever new tests landed in those files, with no failure; re-cite `chat_service.py` lines from `git grep -n "def _record_dict_response_usage\|async def _record_stream_usage"`.
- **Three new tasks** (Decisions 3, 4 and 6): TASK-15422, TASK-15423 and TASK-15424 (provisional ids when this plan was written, one lower each; the highest id in use then was 15420). Task 0 takes the real ids under the id lock and substitutes them in the plan copy, `Usage_Quotas.md`, `model_pricing.json`'s `_catalog_note` and the test comment that name them.
- **Frozen functions.** Nothing touched is in the baseline (V10). Do not edit `function_size_baseline.json`.
- **TDD.** Write each task's tests first, run them, and see them fail for the stated reason. Record the real counts. Some of the new tests pin behaviour the base already has and pass before the code step (`test_a_reply_without_thinking_or_cache_is_unchanged`, `test_a_malformed_usage_metadata_never_raises[missing|wrong-types]`, `test_a_stream_without_usage_metadata_sends_no_usage_chunk[absent|empty]`, the `gemini-cached-thinking` pipeline case); the plan says so where they are run.
- **Rates.** Rates are per 1K tokens in the catalog, per 1M on the page (`MTOK` in the test module). Compare floats with `pytest.approx`; never hand `pytest.approx` a nested dict. JSON edits keep one entry per line in the existing `{ "k": v }` style, through the script in Task 3; do not hand-type the 14 rows.
- **The script imports the test module**, which initialises the app config and may create gitignored runtime files (`Databases/user_databases/1`, `Logs/`); `git status` stays clean, do not add them.
- **Git.** Stage named files. `git diff --cached --check` must print nothing before each commit.
- **Test runs** can leave a read-only Go module cache under `tldw_Server_API/Config_Files/acp_runner_home`. Never commit it; never `rm -rf` the directory (it holds a tracked file). If it appears: `chmod -R u+w tldw_Server_API/Config_Files/acp_runner_home && git clean -ffdx -- tldw_Server_API/Config_Files/acp_runner_home` before the last `git status`, which must be empty.
- **ruff.** `google_adapter.py` carries a pre-existing `I001` and `llm_usage_normalizer.py` a pre-existing `UP035` and `SIM114` on the base (checked with `ruff` on `git show HEAD:<file>`); every touched test file is clean before and after. Do not fix them here.
- **Docs.** `Docs/Operations/Usage_Quotas.md` and `Docs/Providers/Model_Pricing_Catalog.md` are not mirrored (`git ls-files Docs/Published` has neither; measured: after `bash Helper_Scripts/refresh_docs_published.sh`, `git status --short Docs/Published` prints nothing).

## Review Focus

1. **No double count.** The thinking share reaches the row twice by name (`completion_tokens` includes it; `reasoning_tokens` reports it) and must be priced once. `compute_costs` prices `completion_tokens` only; nothing reads `reasoning_tokens` but the insert. The normalizer's raw-key add is guarded by the absence of `completion_tokens`/`output_tokens` and of an explicit count. Pinned: `test_the_normalized_block_is_not_double_counted` (35, not 40), `test_normalize_openai_usage_extracts_cache_read_and_reasoning_tokens` (existing: an OpenAI block with `reasoning_tokens` 6 keeps `output_tokens` 20), the `_log` cost test (`completion_cost_usd == 120 x rate`).
2. **Exactly one chunk, before `[DONE]` or the error, in every ending.** `_usage_chunk` pops the state and sends nothing for a block with no count; the done branch, the post-loop site and both `except` clauses call it; the done branch returns, so `[DONE]` and the chunk go out once and a later event is not read; the `except ChatAPIError` clause re-raises the in-band error unchanged (it is already sanitized). Pinned: `test_the_stream_ends_with_one_usage_chunk_from_the_last_usage_metadata[gemini-end|proxy-done|json-array]` (`payloads[-2:] == [chunk, "[DONE]"]`, exactly one chunk; the `proxy-done` body has an event after `[DONE]`), `test_a_stream_that_fails_after_usage_arrived_is_recorded_from_it[transport-error|in-band-error]` and `test_a_stream_without_usage_metadata_sends_no_usage_chunk[empty]`.
3. **The last `usageMetadata` wins.** Gemini's counts are cumulative; an early event's block must not be the one recorded. Pinned: the same test (the first event's 2 candidates, the second's 30 + 5; the chunk says 35) and `test_a_gemini_stream_is_recorded_from_the_usage_chunk_the_adapter_emits` (the provider-level row says `(80, 35)`, `estimated None`).
4. **A malformed `usageMetadata` never turns the reply raw.** `_normalize_to_openai_shape`'s `except` returns the raw Gemini JSON on any error, which loses the OpenAI shape for the whole reply; on the base a string `usageMetadata` does exactly that (measured: `KeyError: 'object'`). `_usage_block` must not raise on a string, a null, a bool or a string count. Pinned: `test_a_malformed_usage_metadata_never_raises[string|missing|wrong-types]`.
5. **An unpriced tier costs standard.** `unspecified`, `standard`, or any value the entry has no row for (a future `batch`) must not cost $0 or raise. Pinned: `test_the_service_tier_is_passed_on[unspecified-unspecified]` plus the existing catalog rule `test_a_tier_a_model_has_no_row_for_costs_the_standard_rate` (TASK-13936, `test_pricing_service_tiers.py:198`; `_entry_rate_details` is unchanged here).

## Files

| File | Task | Change |
|---|---|---|
| `tldw_Server_API/app/core/LLM_Calls/providers/google_adapter.py:441-456` | 1 | `_usage_block` (new static method) replaces the inline usage dict in `_normalize_to_openai_shape` |
| `tldw_Server_API/app/core/Usage/llm_usage_normalizer.py:163-165` | 1 | the Gemini branch adds the thoughts to raw-key output |
| `tldw_Server_API/tests/LLM_Adapters/unit/test_google_usage_metadata.py` | 1, 2 | new file: 11 non-stream ids (Task 1), 10 stream ids (Task 2) |
| `tldw_Server_API/tests/Usage/test_llm_usage_normalizer.py:66`, `tldw_Server_API/tests/Chat/unit/test_chat_service_content.py:538, 550` | 1 | 30 becomes 35 |
| `tldw_Server_API/tests/Usage/test_pricing_service_tiers.py` | 1, 3 | the cost test (Task 1: standard; Task 3: flex, priority); the tables, four new models, three tier tests (Task 3) |
| `tldw_Server_API/app/core/LLM_Calls/providers/google_adapter.py:11, 460-470, 619-628, 644-648, 675-681` | 2 | `_usage_chunk` (new); `_stream_event_deltas` keeps the last block; `stream` yields the chunk at every ending, `[DONE]` returns, `tool_state` moves before the `try`, `except ChatAPIError` (new) |
| `tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py` | 2 | 1 new test: the real adapter stream through a stubbed transport, recorded by the provider-level recorder |
| `tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py:1097-1141` | 2, 3 | two Gemini cases in the parametrize (Task 2: standard; Task 3: flex) |
| `tldw_Server_API/Config_Files/model_pricing.json:168-177` (+ `:2`) | 3 | 14 Gemini rows rewritten/inserted by the script; `_catalog_note` |
| `Docs/Operations/Usage_Quotas.md:111-112`, `Docs/Providers/Model_Pricing_Catalog.md:33` | 4 | the Gemini bullet; the source line |
| `Docs/superpowers/plans/2026-11-03-gemini-pricing-gaps.md`, `backlog/tasks/task-13952 - ....md`, the three new task files | 0, 4 | this plan; the tasks |

- **Scope baseline.** `git grep -n -E "google_adapter.py|llm_usage_normalizer.py|pricing_catalog.py|usage_tracker.py|model_pricing.json|test_pricing_service_tiers.py|test_chat_service_content.py|test_usage_accounting.py|test_chat_service_streaming_tool_autoexec.py" Helper_Scripts/ci/scope_predicate_baseline.txt` prints nothing (measured).
- **Open PRs.** #3413 is the base (merged first, Task 0). No other open PR touches `google_adapter.py`, `llm_usage_normalizer.py`, the Gemini rows of `model_pricing.json` or the Gemini bullet of `Usage_Quotas.md` as of 2026-11-03; re-check with `gh pr list --state open --limit 200 --json number,files --jq '.[] | select(.files[].path | test("google_adapter|llm_usage_normalizer|model_pricing|Usage_Quotas|test_pricing_service_tiers")) | .number' | sort -u` in Task 0.

## Test commands

- **Focused set** (every test file this plan edits plus the adapter and usage suites, SQLite). Save as `<scratchpad>/focused.sh`, run `zsh <scratchpad>/focused.sh <label>` from anywhere:

```zsh
#!/bin/zsh
S=<scratchpad>
PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python
L=${1:-run}
cd <worktree root> || exit 1
PATH="/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin:$PATH" TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q --no-header -n 6 --dist loadfile -rfE \
  tldw_Server_API/tests/LLM_Adapters/unit tldw_Server_API/tests/Usage \
  tldw_Server_API/tests/Chat/unit/test_chat_service_content.py tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py \
  tldw_Server_API/tests/LLM_Calls/test_llm_providers.py tldw_Server_API/tests/LLM_Calls/test_provider_response_preservation.py \
  tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py tldw_Server_API/tests/LLM_Calls/test_gemini_and_hf_regressions.py \
  tldw_Server_API/tests/LLM_Calls/test_cache_intents.py > $S/focused_$L.out 2>&1
echo "focused $L: $(tail -1 $S/focused_$L.out)"
grep '^FAILED\|^ERROR' $S/focused_$L.out | cut -c1-200
```

  Measured: `1473 passed, 1 skipped` on the base (69 s); `1536 passed, 1 skipped` at the end (92 s; the 63 new ids: 21 in the adapter file, 1 in `test_usage_accounting.py`, 39 in `test_pricing_service_tiers.py`, 2 in the chat-pipeline parametrize).
- **Wider set**: every test file that names a touched symbol. Save as `<scratchpad>/wider.sh` and run `zsh <scratchpad>/wider.sh <label>`:

```zsh
#!/bin/zsh
S=<scratchpad>
PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python
L=${1:-run}
cd <worktree root> || exit 1
git grep --untracked -l -E "_normalize_to_openai_shape|GoogleAdapter|google_adapter|_stream_event_deltas|normalize_llm_usage|NormalizedLLMUsage|model_pricing|PricingCatalog|get_rate_details|compute_costs|log_llm_usage|_StreamUsage|usage_accounting|cachedContentTokenCount|thoughtsTokenCount|list_provider_models|compute_token_cost" -- 'tldw_Server_API/tests/*.py' 'tldw_Server_API/app/*/tests/*.py' | grep '/test_[^/]*\.py$' | sort -u > $S/wider_all.txt
SERIAL='AuthNZ_Unit/test_provider_usage_reservations_repo\.py$'
grep -vE '/AuthNZ_Postgres/|_pg\.py$|_postgres\.py$' $S/wider_all.txt | grep -vE "$SERIAL" > $S/wider_sqlite.txt
grep -E "$SERIAL" $S/wider_all.txt > $S/wider_serial.txt
grep -E '/AuthNZ_Postgres/|_pg\.py$|_postgres\.py$' $S/wider_all.txt > $S/wider_pg.txt
echo "files: sqlite $(wc -l < $S/wider_sqlite.txt) + serial $(wc -l < $S/wider_serial.txt), pg $(wc -l < $S/wider_pg.txt)"
PATH="/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin:$PATH" TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q --no-header -n 8 --dist loadfile -rfE $(cat $S/wider_sqlite.txt) > $S/wider_sqlite_$L.out 2>&1
echo "sqlite $L: $(tail -1 $S/wider_sqlite_$L.out)"
grep '^FAILED\|^ERROR' $S/wider_sqlite_$L.out | cut -c1-220
if [[ -s $S/wider_serial.txt ]]; then
  PATH="/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin:$PATH" TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q --no-header -rfE $(cat $S/wider_serial.txt) > $S/wider_serial_$L.out 2>&1
  echo "serial $L: $(tail -1 $S/wider_serial_$L.out)"
fi
if [[ -s $S/wider_pg.txt ]]; then
  PATH="/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin:$PATH" TEST_DB_NAME=tldw_test_13952 $PY -m pytest -p no:cacheprovider -q --no-header -rfE $(cat $S/wider_pg.txt) > $S/wider_pg_$L.out 2>&1
  echo "pg $L: $(tail -1 $S/wider_pg_$L.out)"
fi
```

  Measured: 68 SQLite files on the base and 69 after this plan (the new adapter test file), no serial file, no PostgreSQL file (the grep finds none; if dev adds one, the script runs it with the Docker fixture). On `6332d72b09`, clean: `2546 passed, 13 skipped` (25 minutes). After this plan: `2609 passed, 13 skipped` (24 minutes; the 63 new ids). Any other failure is this plan's.
- **Per-task selections** (the ids a task adds or changes), each `PATH="$(dirname $PY):$PATH" TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q --no-header -rfE <files> -k "<expr>"`:
  - Task 1: files `tldw_Server_API/tests/LLM_Adapters/unit/test_google_usage_metadata.py tldw_Server_API/tests/Usage/test_llm_usage_normalizer.py tldw_Server_API/tests/Chat/unit/test_chat_service_content.py tldw_Server_API/tests/Usage/test_pricing_service_tiers.py`, `-k "thinking_tokens_are_counted or not_double_counted or tool_use_prompt or without_thinking or malformed_usage_metadata or service_tier_is_passed_on or normalize_gemini_usage or normalizes_gemini_usage or (costs_its_thinking_tokens and standard)"` (14 ids).
  - Task 2: files `tldw_Server_API/tests/LLM_Adapters/unit/test_google_usage_metadata.py tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py`, `-k "(usage_chunk or fails_after_usage or gemini-cached-thinking) and not gemini-flex"` (17 ids; `usage_chunk` also selects 5 pre-existing TASK-13951/13953 ids in the chat-pipeline file, which pass throughout; pytest's `-k` takes the hyphenated ids; `gemini-flex` is Task 3's).
  - Task 3: files `tldw_Server_API/tests/Usage/test_pricing_service_tiers.py tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py`, `-k "flex_and_priority or on_a_tier_pays or ga_model_tiers or context_caching_rate or (costs_its_thinking_tokens and not standard) or gemini-flex"` (49 ids; `context_caching_rate` also selects the 10 TASK-13936 cache-rate ids, 4 of which are the new models).

---

## Task 0: Confirm the base, branch, start the task, file the three follow-ups, plan file, draft PR, baseline

- [ ] **Step 1: Confirm #3413 merged and branch.** `gh pr view 3413 --json state,mergeCommit --jq '.state + " " + .mergeCommit.oid'` must print `MERGED <sha>`. Then `git fetch origin dev --quiet && git switch -c fix/gemini-pricing-gaps-13952 origin/dev`, and `git log --oneline -1 origin/dev` for the ledger. `git grep -n "stream_usage: _StreamUsage | None" tldw_Server_API/app/core/Chat/chat_service.py` must hit (TASK-13953 is in).
- [ ] **Step 2: Take the follow-up ids.** `mkdir <scratchpad>/id_lock` (if it exists, wait; never break another session's lock). From the repository root, `zsh <plans dir>/task_next.sh` prints the highest id in use; `<A>` is that + 1 (audio), `<D>` that + 2 (dated changes) and `<F>` that + 3 (a stopped or dropped stream). `ls backlog/tasks backlog/completed backlog/archive/tasks | grep -E "^task-(<A>|<D>|<F>) "` must print nothing. Everywhere below, `TASK-15422` means `TASK-<A>`, `TASK-15423` means `TASK-<D>` and `TASK-15424` means `TASK-<F>`.
- [ ] **Step 3: Start the task and file the follow-ups** (backlog-py only; `<answer>` is the owner's recorded answer to Decisions 1 and 3, passed in the dispatch; never write it literally; Decisions 4 and 6 were answered on 2026-11-03):

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-13952 -s "In Progress" --append-notes "Plan: Docs/superpowers/plans/2026-11-03-gemini-pricing-gaps.md. Owner answers: ADR-004 Change summary <answer>; Decision 3 (audio not priced here, follow-up TASK-<A>) <answer>; Decision 4 (today's rates, dated follow-up TASK-<D> labelled high, reminders on 2026-11-30 and 2026-12-31) agreed 2026-11-03; Decision 6 (a stopped or dropped stream's thinking tokens, follow-up TASK-<F>) agreed 2026-11-03."

PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task create "Price Gemini audio input and cached audio at Google's audio rates" \
  --id TASK-<A> -l quotas,backend --dep TASK-13952 \
  -d "Google charges audio input and cached audio more than text on gemini-3.1-flash-lite (\$0.50 and \$0.05 per million tokens against \$0.25 and \$0.025), gemini-3-flash-preview (\$1.00 / \$0.10 against \$0.50 / \$0.05), gemini-2.5-flash (\$1.00 / \$0.10 against \$0.30 / \$0.03) and gemini-2.5-flash-lite (\$0.30 / \$0.03 against \$0.10 / \$0.01), on the standard, flex and priority tiers (aistudio.google.com/docs/pricing, fetched 2026-11-03). The usage record counts audio tokens inside promptTokenCount at the text rate, so a reply with audio input is under-billed by up to 3.3x on its input. The modality split is in usageMetadata.promptTokensDetails and cacheTokensDetails (ModalityTokenCount: modality TEXT/IMAGE/VIDEO/AUDIO/DOCUMENT, tokenCount). Fix: GoogleAdapter._usage_block keeps both lists (teach _sanitize_raw_usage_metadata that a list of counter mappings is not prompt text, or fold them to flat counts); normalize_llm_usage's Gemini branch sums the AUDIO entries into two new NormalizedLLMUsage fields (audio_input_tokens, cache_read_audio_tokens); the catalog accepts prompt_audio and cache_read_audio rates (in _parse_rates and _entry_rate_details, on a tier too); compute_costs prices the uncached audio share at prompt_audio and the cached audio share at cache_read_audio, falling back to the text rates; the rows above (and the four 2.5 -preview aliases) get the rates via the script in Docs/superpowers/plans/2026-11-03-gemini-pricing-gaps.md. Recorded as not priced in Docs/Operations/Usage_Quotas.md (Gemini bullet) until then." \
  --ac "A Gemini reply whose promptTokensDetails and cacheTokensDetails carry AUDIO entries is costed at the model's audio input and cached-audio rates on the tier it reports, with a test per rate" \
  --ac "A reply with no modality details or with TEXT only is costed as today, with a test" \
  --ac "Docs/Operations/Usage_Quotas.md and the _catalog_note say audio is priced"

PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task create "Gemini price changes of 2026-12-01 and 2027-01-01: re-fetch the page and bump the catalog" \
  --id TASK-<D> -l quotas,backend,high --dep TASK-13952 \
  -d "model_pricing.json holds Google's rates in force on 2026-11-03 (aistudio.google.com/docs/pricing). Two dates. 2026-12-01: gemini-4-argon doubles (standard \$2.00 / \$10.00 / \$0.10 becomes \$4.00 / \$20.00 / \$0.20 per million tokens, the flex, priority and batch tables likewise), so it is under-charged by half until the catalog follows; and all requests to gemini-2.5-pro route to gemini-3.8-flash and all requests to gemini-2.5-flash route to gemini-3.5-flash-lite, billed at those models' rates (the response's modelVersion says what ran): gemini-2.5-pro calls become over-charged (3.8 Flash costs less), and gemini-2.5-flash calls cost the same on standard text (\$0.30 / \$2.50 / \$0.03 on both; its audio is then billed at the text rate). 2027-01-01: gemini-3.8-flash and gemini-3.6-flash double (\$0.75 / \$3.75 / \$0.075 becomes \$1.50 / \$7.50 / \$0.15; tiers likewise), which also makes gemini-2.5-pro's catalogued input and cache rates (\$1.25, \$0.125) lower than what Google bills for it (\$1.50, \$0.15): under-charged on input and cache, over-charged on output. On each date: re-fetch the page, update GEMINI_PRICES, GEMINI_TIER_PRICES (and the long tables) in tests/Usage/test_pricing_service_tiers.py, re-run the Gemini script from Docs/superpowers/plans/2026-11-03-gemini-pricing-gaps.md (Task 3) and update the _catalog_note and Usage_Quotas.md. Decide whether gemini-2.5-pro and gemini-2.5-flash should be priced as the model they route to, or kept at their own rates. Not catalogued: TASK-13952 added the four text models; the page's specialised models (Live, TTS, Image, Transcribe, Embedding, Omni Flash, Robotics ER) are not, so a call to one costs the unknown-model rate." \
  --ac "On 2026-12-01: gemini-4-argon's standard, flex and priority rates are the page's new ones, with the tests in test_pricing_service_tiers.py updated; the 2.5 Pro/Flash routing is priced one way or the other, with the decision in Usage_Quotas.md" \
  --ac "On 2027-01-01: gemini-3.8-flash's and gemini-3.6-flash's rates are the page's new ones, with the tests updated, and gemini-2.5-pro's input and cache are no longer under-charged" \
  --ac "Each bump cites the page and its fetch date in the _catalog_note"

PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task create "Cost a stopped or dropped Gemini stream with its thinking tokens" \
  --id TASK-<F> -l quotas,backend,high --dep TASK-13952 \
  -d "Since TASK-13952, GoogleAdapter.stream ends with one usage chunk built from the last usageMetadata (Gemini's counts are cumulative): at the stream's end, before a proxy's [DONE], and before the error when the stream fails mid-way. A stream that the client drops, or that the WebUI's Stop ends (TASK-13953: the handler stops reading and closes the stream), gets no chunk: the adapter cannot yield from a closed generator. Both recorders then cost it from the estimate (the prompt plus a quarter of the characters received), which leaves out the thinking tokens. Gemini thinks before it answers, so by the time content is shown Google has billed the whole thinking budget, often most of the output cost. Each event's usageMetadata already carries thoughtsTokenCount, and every delta carries its raw event as provider_response. Fix: _StreamUsage (LLM_Calls/usage_accounting.py) keeps GoogleAdapter._usage_block of the last delta's provider_response usageMetadata as a fallback, used only when no usage chunk arrived; both recorders (the provider-level _record_stream and the chat handler, which feeds feed_event) then cost from it. Never emit a usage chunk per event: clients that sum usage chunks (LangChain aggregates usage_metadata across chunks) would double count. Related gap: OpenAI reasoning tokens arrive only in a stream's final usage chunk, so a stopped or dropped OpenAI stream is costed without them too. Recorded in Docs/Operations/Usage_Quotas.md (Gemini bullet)." \
  --ac "A Gemini stream that the client drops after usage arrived is costed with its thinking tokens, from the last usageMetadata, on /chat/completions and through the provider-level recorder, with a test" \
  --ac "A Gemini stream stopped from the WebUI after usage arrived is costed with its thinking tokens, with a test" \
  --ac "No usage chunk is emitted per event, and a completed stream still records its one final chunk, with a test" \
  --ac "Usage_Quotas.md is updated, and the OpenAI reasoning-token gap is noted there as related"
```

  Measured: run in the fold dry run with placeholder ids TASK-99997/99998/99999 and reverted: the edit and the three creates exit 0; TASK-99998 and TASK-99999 carry the `high` label, all three `--dep TASK-13952`.
- [ ] **Step 4: Plan file, commit, draft PR, release the lock.** Copy this plan to `Docs/superpowers/plans/2026-11-03-gemini-pricing-gaps.md` with `TASK-15422`/`TASK-15423`/`TASK-15424` replaced by the real ids (the `<A>`/`<D>`/`<F>`/`<answer>`/`<n>` placeholders stay: they are filled at the steps that use them). Then:
  - `git add Docs/superpowers/plans/2026-11-03-gemini-pricing-gaps.md && git add -A backlog/tasks && git diff --cached --check` (it must print nothing);
  - commit (message 0);
  - `git push -u origin fix/gemini-pricing-gaps-13952`, then `gh pr create --draft --base dev --title "fix(usage): cost Gemini thinking tokens, the stream's usage chunk, Flex and Priority tiers (TASK-13952)"` with a one-line body naming the task and ending with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`;
  - when `gh pr view <n> --json files --jq '.files[].path'` lists `backlog/tasks/task-<A> - ...md`, `backlog/tasks/task-<D> - ...md` and `backlog/tasks/task-<F> - ...md`, `rmdir <scratchpad>/id_lock`;
  - report `TASK-<D>`'s id to the controller. The controller (not the executor) then schedules two one-time remote agents, on 2026-11-30 and 2026-12-31, to start TASK-<D>'s two bumps (Decision 4).
- [ ] **Step 5: Baseline.** On the unchanged branch, run the focused set with label `base` (expected `1473 passed, 1 skipped` plus any tests dev added to those files since 2026-11-03; no failure) and the wider set with label `base` (expected `2546 passed, 13 skipped` on `6332d72b09`, 68 files; more if dev added tests to those files; no failure). Record both in the ledger. Any later failure is compared with this baseline. Never dismiss one as environment-only without it.

## Task 1: The usage block counts thinking tokens as output, tool-use prompt tokens as input, and carries the service tier (AC #1, AC #3 part)

**Files:** `tldw_Server_API/app/core/LLM_Calls/providers/google_adapter.py:441-456` (+ a new static method after `:458`); `tldw_Server_API/app/core/Usage/llm_usage_normalizer.py:163-165`; tests: `tldw_Server_API/tests/LLM_Adapters/unit/test_google_usage_metadata.py` (new), `tldw_Server_API/tests/Usage/test_llm_usage_normalizer.py:66`, `tldw_Server_API/tests/Chat/unit/test_chat_service_content.py:538, 550`, `tldw_Server_API/tests/Usage/test_pricing_service_tiers.py` (append).

**Interfaces:**
- Produces: `GoogleAdapter._usage_block(usage_src: Any) -> dict[str, Any]` (static): `{"prompt_tokens", "completion_tokens", "total_tokens"}` always (a missing count is `None`), plus `thoughtsTokenCount`, `toolUsePromptTokenCount`, `cachedContentTokenCount` when they are ints, and `service_tier` (lowercased, `service_tier_` prefix stripped) when `serviceTier` is a non-empty string. Task 2 calls it from `_stream_event_deltas`.
- `normalize_llm_usage` keeps its signature; its Gemini branch adds `reasoning_tokens` to `output_tokens` only when `completion_tokens is None` and the block has no positive `completion_tokens`/`output_tokens`.

- [ ] **Step 1: Write the failing tests.** Create `tldw_Server_API/tests/LLM_Adapters/unit/test_google_usage_metadata.py`:

```python
"""Gemini usageMetadata is costed in full: thinking tokens, tool-use prompt tokens, the service tier, and the
stream's usage chunk (TASK-13952)."""

from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.LLM_Calls.providers.google_adapter import GoogleAdapter
from tldw_Server_API.app.core.Usage.llm_usage_normalizer import normalize_llm_usage

pytestmark = pytest.mark.unit

# The repo fixture (tests/Chat/unit/test_chat_service_content.py): totalTokenCount 115 = 80 prompt + 30 candidates
# + 5 thoughts; 50 of the prompt tokens were cached.
USAGE_METADATA = {
    "promptTokenCount": 80,
    "candidatesTokenCount": 30,
    "totalTokenCount": 115,
    "cachedContentTokenCount": 50,
    "thoughtsTokenCount": 5,
}
EXPECTED_USAGE = {
    "prompt_tokens": 80,
    "completion_tokens": 35,
    "total_tokens": 115,
    "thoughtsTokenCount": 5,
    "cachedContentTokenCount": 50,
}


def _reply(usage_metadata: dict[str, Any]) -> dict[str, Any]:
    """A generateContent response with one text candidate and ``usage_metadata``."""
    return {
        "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
        "usageMetadata": usage_metadata,
    }


def test_thinking_tokens_are_counted_as_output() -> None:
    """Google bills thoughtsTokenCount as output, apart from candidatesTokenCount (AC #1); the block keeps the
    thinking share under its own name for the row's reasoning_tokens, and total_tokens is Gemini's total."""
    usage = GoogleAdapter._normalize_to_openai_shape(_reply(USAGE_METADATA))["usage"]

    assert usage == EXPECTED_USAGE


def test_the_normalized_block_is_not_double_counted() -> None:
    """The usage normalizer reads completion_tokens (thinking included) and reports the thinking share apart."""
    usage = GoogleAdapter._normalize_to_openai_shape(_reply(USAGE_METADATA))["usage"]

    normalized = normalize_llm_usage(provider="google", usage=usage)

    assert (normalized.input_tokens, normalized.output_tokens, normalized.total_tokens) == (80, 35, 115)
    assert (normalized.reasoning_tokens, normalized.cache_read_input_tokens) == (5, 50)


def test_tool_use_prompt_tokens_are_counted_as_input() -> None:
    """toolUsePromptTokenCount is outside promptTokenCount (Gemini's total = prompt + tool-use prompt + thoughts +
    candidates), so it joins prompt_tokens and the block's sides add up to its total."""
    usage = GoogleAdapter._normalize_to_openai_shape(
        _reply({**USAGE_METADATA, "toolUsePromptTokenCount": 7, "totalTokenCount": 122})
    )["usage"]

    assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (87, 35, 122)
    assert usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"]


def test_a_reply_without_thinking_or_cache_is_unchanged() -> None:
    """A plain usageMetadata gives the plain block, a missing count staying absent."""
    usage = GoogleAdapter._normalize_to_openai_shape(
        _reply({"promptTokenCount": 10, "candidatesTokenCount": 5})
    )["usage"]

    assert usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": None}


@pytest.mark.parametrize(
    "usage_metadata",
    ["garbage", None, {"candidatesTokenCount": "30", "thoughtsTokenCount": True, "serviceTier": 3}],
    ids=["string", "missing", "wrong-types"],
)
def test_a_malformed_usage_metadata_never_raises(usage_metadata: Any) -> None:
    """A count that is not an int is left out of the sums; the reply is still normalized (not returned raw)."""
    shaped = GoogleAdapter._normalize_to_openai_shape({**_reply({}), "usageMetadata": usage_metadata})

    assert shaped["object"] == "chat.completion"
    assert "service_tier" not in shaped["usage"] and "thoughtsTokenCount" not in shaped["usage"]


@pytest.mark.parametrize(
    ("reported", "expected"),
    [("flex", "flex"), ("priority", "priority"), ("SERVICE_TIER_PRIORITY", "priority"), ("unspecified", "unspecified")],
)
def test_the_service_tier_is_passed_on(reported: str, expected: str) -> None:
    """UsageMetadata.serviceTier (unspecified, standard, flex, priority) is kept as service_tier, lowercased and
    without an enum prefix, so the catalog's flex and priority rates apply (AC #3)."""
    usage = GoogleAdapter._normalize_to_openai_shape(_reply({**USAGE_METADATA, "serviceTier": reported}))["usage"]

    assert usage["service_tier"] == expected
```

  In `tldw_Server_API/tests/Usage/test_llm_usage_normalizer.py` (`:65-66`), replace `    assert normalized.output_tokens == 30` with:

```python
    # Raw usageMetadata keys: candidatesTokenCount leaves out the thoughts, which Google bills as output (TASK-13952).
    assert normalized.output_tokens == 35
```

  In `tldw_Server_API/tests/Chat/unit/test_chat_service_content.py`, replace `    assert logged_usage["completion_tokens"] == 30` (`:538`) with:

```python
    # candidatesTokenCount 30 + thoughtsTokenCount 5: thinking tokens are billed as output (TASK-13952).
    assert logged_usage["completion_tokens"] == 35
```

  and `    assert metrics.tokens["completion_tokens"] == 30` (`:550`) with `    assert metrics.tokens["completion_tokens"] == 35`.

  Append to `tldw_Server_API/tests/Usage/test_pricing_service_tiers.py` (after the last test, two blank lines before):

```python
# --- Gemini thinking tokens, service tiers and the newer models (TASK-13952) ---


@pytest.mark.parametrize(
    ("tier", "prices"),
    [
        (None, GEMINI_PRICES["gemini-2.5-flash"]),
    ],
    ids=["standard"],
)
async def test_a_gemini_reply_costs_its_thinking_tokens_as_output_at_the_tier_it_reports(
    costed_rows: list[dict[str, Any]], tier: str | None, prices: tuple[float, float, float | None]
) -> None:
    """thoughtsTokenCount is billed as output, apart from candidatesTokenCount, at the rates of the tier
    usageMetadata.serviceTier names; the cached prompt share keeps the tier's cache rate (AC #1, AC #3)."""
    from tldw_Server_API.app.core.LLM_Calls.providers.google_adapter import GoogleAdapter

    usage_metadata: dict[str, Any] = {
        "promptTokenCount": 10_000,
        "candidatesTokenCount": 100,
        "thoughtsTokenCount": 20,
        "totalTokenCount": 10_120,
        "cachedContentTokenCount": 8_000,
    }
    if tier is not None:
        usage_metadata["serviceTier"] = tier
    usage = GoogleAdapter._normalize_to_openai_shape(
        {"candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}], "usageMetadata": usage_metadata}
    )["usage"]

    await _log("google", "gemini-2.5-flash", usage, usage["prompt_tokens"], usage["completion_tokens"])

    (row,) = costed_rows
    prompt, completion, cached = prices
    assert (row["completion_tokens"], row["reasoning_tokens"], row["total_tokens"]) == (120, 20, 10_120)
    assert row["completion_cost_usd"] == pytest.approx(120 * completion / 1_000_000)
    assert row["prompt_cost_usd"] == pytest.approx((2_000 * prompt + 8_000 * cached) / 1_000_000)
    assert row["estimated"] is False
```

- [ ] **Step 2: Run them to verify they fail.** The Task 1 selection (Test commands).
  Expected: `11 failed, 3 passed` (measured): `test_thinking_tokens_are_counted_as_output` fails with `{'completion_tokens': 30} != {'completion_tokens': 35}` and `Right contains 1 more item: {'thoughtsTokenCount': 5}`; `..._not_double_counted` with `(80, 30, 115) == (80, 35, 115)`; `..._tool_use_...` with `(80, 30, 122) == (87, 35, 122)`; `..._malformed_...[string]` with `KeyError: 'object'` (on the base a string `usageMetadata` raises inside `_normalize_to_openai_shape`, which then returns the raw Gemini JSON: a real fix, not a pin); the four `..._service_tier_is_passed_on` with `KeyError: 'service_tier'`; the normalizer test with `30 == 35`; the chat test with `30 == 35`; the cost test with `(100, 0, 10100) == (120, 20, 10120)`. `test_a_reply_without_thinking_or_cache_is_unchanged`, `..._malformed_...[missing]` and `[wrong-types]` pass on the base (they pin behaviour the base has: a missing count stays `None`; a null or wrong-typed block does not turn the reply raw) and must still pass after Step 3.
- [ ] **Step 3: Implement.** In `google_adapter.py`, replace `:441-456` (from `usage_src = data.get("usageMetadata") or {}` through the `return {...}` of `_normalize_to_openai_shape`) with:

```python
            return {
                "id": data.get("id") or data.get("responseId"),
                "object": "chat.completion",
                "choices": choices or [{"index": 0, "message": {"role": "assistant", "content": None}, "finish_reason": None}],
                "usage": GoogleAdapter._usage_block(data.get("usageMetadata")),
                "provider_response": data,
            }
```

  and insert, directly after the `except _GOOGLE_ADAPTER_RUNTIME_EXCEPTIONS: return data` that ends `_normalize_to_openai_shape` (`:457-458`) and before `@staticmethod def _stream_event_deltas`:

```python
    @staticmethod
    def _usage_block(usage_src: Any) -> dict[str, Any]:
        """The OpenAI-shaped usage block of one Gemini ``usageMetadata``; the reply's and the stream's (TASK-13952).

        Thinking tokens (``thoughtsTokenCount``) are billed as output and counted apart from the candidates, so they
        are in ``completion_tokens``; tool-use prompt tokens are outside ``promptTokenCount`` and join ``prompt_tokens``.
        Both keep their Gemini name too (the row's ``reasoning_tokens``; the audit JSON). ``total_tokens`` is Gemini's
        ``totalTokenCount``, which sums all four. ``cachedContentTokenCount`` is the cached share of ``prompt_tokens``,
        costed at the cache-read rate (TASK-13936), and ``service_tier`` the tier the call ran on (``flex``, ``priority``).
        """
        src = usage_src if isinstance(usage_src, dict) else {}

        def _count(key: str) -> int | None:
            value = src.get(key)
            return value if isinstance(value, int) and not isinstance(value, bool) else None

        usage: dict[str, Any] = {
            "prompt_tokens": src.get("promptTokenCount"),
            "completion_tokens": src.get("candidatesTokenCount"),
            "total_tokens": src.get("totalTokenCount"),
        }
        for extra, side in (("thoughtsTokenCount", "completion_tokens"), ("toolUsePromptTokenCount", "prompt_tokens")):
            count = _count(extra)
            if count is not None:
                usage[side] = (_count("promptTokenCount" if side == "prompt_tokens" else "candidatesTokenCount") or 0) + count
                usage[extra] = count
        if _count("cachedContentTokenCount") is not None:
            usage["cachedContentTokenCount"] = src["cachedContentTokenCount"]
        tier = src.get("serviceTier")
        if isinstance(tier, str) and tier.strip():
            usage["service_tier"] = tier.strip().lower().removeprefix("service_tier_")
        return usage

```

  In `llm_usage_normalizer.py`, replace the Gemini branch (`:163-165`) with:

```python
    elif "google" in provider_key or "gemini" in provider_key:
        cache_read_input_tokens = _usage_int(usage_mapping, "cachedContentTokenCount", "cached_content_token_count")
        cached_input_tokens = cache_read_input_tokens
        if completion_tokens is None and _usage_int(usage_mapping, "completion_tokens", "output_tokens") == 0:
            # Raw usageMetadata keys: candidatesTokenCount leaves out the thoughts, which Google bills as output.
            # The Google adapter's block already counts them in completion_tokens (TASK-13952).
            output_tokens += reasoning_tokens
```

- [ ] **Step 4: Run; see GREEN.** The Task 1 selection: expected `14 passed` (measured). Then the three files the step edited, whole: `PATH="$(dirname $PY):$PATH" TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/LLM_Adapters/unit/test_google_usage_metadata.py tldw_Server_API/tests/Usage/test_llm_usage_normalizer.py tldw_Server_API/tests/Chat/unit/test_chat_service_content.py`: no failure. `$(dirname $PY)/ruff check` on the new test file and the three edited test files: "All checks passed!" (the two source files keep their pre-existing findings, Global Constraints).
- [ ] **Step 5: Commit** (message 1): `git add tldw_Server_API/app/core/LLM_Calls/providers/google_adapter.py tldw_Server_API/app/core/Usage/llm_usage_normalizer.py tldw_Server_API/tests/LLM_Adapters/unit/test_google_usage_metadata.py tldw_Server_API/tests/Usage/test_llm_usage_normalizer.py tldw_Server_API/tests/Chat/unit/test_chat_service_content.py tldw_Server_API/tests/Usage/test_pricing_service_tiers.py && git diff --cached --check`.

## Task 2: The stream ends with one usage chunk, also when it fails mid-way (AC #2)

**Files:** `tldw_Server_API/app/core/LLM_Calls/providers/google_adapter.py` (`:11`; after `_usage_block`; `_stream_event_deltas` at `:460-470`; `stream` at `:619-628`, `:644-648` and `:675-681`); tests: `tldw_Server_API/tests/LLM_Adapters/unit/test_google_usage_metadata.py` (imports; append), `tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py` (append), `tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py:1097-1141`.

**Interfaces:**
- Consumes (Task 1): `GoogleAdapter._usage_block`.
- Produces: `GoogleAdapter._usage_chunk(state: dict[str, Any]) -> Iterable[str]` (static; yields `sse_data({"choices": [], "usage": state.pop("usage")})` once, nothing when no block was seen or every value of the block is `None`); `_stream_event_deltas(event, state: dict[str, Any] | None)` stores `state["usage"]` from each event's `usageMetadata`. `stream` yields the chunk when the stream closes, before a proxy's `[DONE]` (then returns), and before re-raising a failure mid-way. The chunk's `usage` is a Task 1 block, so `_StreamUsage.feed`/`feed_event` (unchanged) record it and `normalize_llm_usage` reads its `service_tier` and `cachedContentTokenCount`.

- [ ] **Step 1: Write the failing tests.** In `test_google_usage_metadata.py`, replace the imports from `from typing import Any` through the `normalize_llm_usage` import with (sorted: `ruff check` passes):

```python
import json
from collections.abc import Iterable, Iterator
from typing import Any

import httpx
import pytest

from tldw_Server_API.app.core.exceptions import ChatAPIError
from tldw_Server_API.app.core.LLM_Calls.providers import google_adapter as gmod
from tldw_Server_API.app.core.LLM_Calls.providers.google_adapter import GoogleAdapter
from tldw_Server_API.app.core.LLM_Calls.usage_accounting import _StreamUsage
from tldw_Server_API.app.core.Usage.llm_usage_normalizer import normalize_llm_usage
```

  and append at the end of the file, two blank lines after the last test:

```python
# --- The stream ends with one usage chunk (AC #2) ---

# Two events; Gemini's usageMetadata is cumulative, so the last one is the total.
STREAM_EVENTS = [
    {
        "candidates": [{"content": {"parts": [{"text": "Hel"}]}, "index": 0}],
        "usageMetadata": {"promptTokenCount": 80, "candidatesTokenCount": 2, "totalTokenCount": 82},
    },
    {
        "candidates": [{"content": {"parts": [{"text": "lo"}]}, "finishReason": "STOP", "index": 0}],
        "usageMetadata": {**USAGE_METADATA, "serviceTier": "flex"},
    },
]
STREAM_BODY = "".join(f"data: {json.dumps(event)}\n\n" for event in STREAM_EVENTS).encode()
STREAM_BODIES = {
    "gemini-end": STREAM_BODY,  # Gemini ends a stream by closing it
    # A proxy may add a [DONE] line, which ends the stream: a later event is not read.
    "proxy-done": STREAM_BODY + b'data: [DONE]\n\ndata: {"usageMetadata": {"promptTokenCount": 1}}\n\n',
    "json-array": json.dumps(STREAM_EVENTS).encode(),  # the body without alt=sse
}
STREAM_REQUEST = {"model": "gemini-2.5-flash", "api_key": "sk-test", "messages": [{"role": "user", "content": "hi"}]}


def _stream(monkeypatch: pytest.MonkeyPatch, body: bytes | Iterator[bytes]) -> Iterable[str]:
    """GoogleAdapter.stream over ``body``, through a stubbed transport (no network)."""
    monkeypatch.setenv("LLM_ADAPTERS_NATIVE_HTTP_GOOGLE", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "1")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith(":streamGenerateContent")
        return httpx.Response(200, content=body)

    def fake_client(*_args: Any, **_kwargs: Any) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(gmod, "http_client_factory", fake_client)
    return GoogleAdapter().stream(dict(STREAM_REQUEST))


@pytest.fixture(params=list(STREAM_BODIES))
def gemini_stream(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> list[str]:
    """The lines GoogleAdapter.stream yields for each body of STREAM_EVENTS."""
    return list(_stream(monkeypatch, STREAM_BODIES[request.param]))


def _payloads(lines: list[str]) -> list[Any]:
    """The JSON payload of every data line, "[DONE]" as a string."""
    out: list[Any] = []
    for line in lines:
        for part in line.splitlines():
            if part.startswith("data:"):
                payload = part[5:].strip()
                out.append(payload if payload == "[DONE]" else json.loads(payload))
    return out


def test_the_stream_ends_with_one_usage_chunk_from_the_last_usage_metadata(gemini_stream: list[str]) -> None:
    """The chunk is OpenAI's shape ({"choices": [], "usage": ...}) with the block the non-streaming reply gets,
    thinking tokens and the service tier included; it comes after the last delta and before [DONE]."""
    payloads = _payloads(gemini_stream)

    usage_chunks = [p for p in payloads if isinstance(p, dict) and "usage" in p]
    assert usage_chunks == [{"choices": [], "usage": {**EXPECTED_USAGE, "service_tier": "flex"}}]
    assert payloads[-2:] == [usage_chunks[0], "[DONE]"]
    assert [p["choices"][0]["delta"]["content"] for p in payloads[:-2]] == ["Hel", "lo"]


def test_the_provider_level_recorder_reads_the_usage_chunk(gemini_stream: list[str]) -> None:
    """_StreamUsage, which the provider-level recorder and the chat stream handler feed, takes the chunk as the
    stream's usage, so a streamed Gemini reply is no longer costed from the estimate."""
    seen = _StreamUsage()
    for line in gemini_stream:
        seen.feed(line)

    assert seen.usage == {**EXPECTED_USAGE, "service_tier": "flex"}
    assert seen.chars == len("Hello")


@pytest.mark.parametrize("failure", ["transport-error", "in-band-error"])
def test_a_stream_that_fails_after_usage_arrived_is_recorded_from_it(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """A stream that fails mid-way sends the usage seen so far before the error, so the recorders cost it from
    Gemini's counts, thinking tokens included, not from the estimate."""

    def body() -> Iterator[bytes]:
        yield STREAM_BODY
        if failure == "transport-error":
            raise httpx.ReadError("connection reset")
        yield b'data: {"error": {"code": 500, "message": "internal", "status": "INTERNAL"}}\n\n'

    seen = _StreamUsage()
    with pytest.raises(ChatAPIError):
        for line in _stream(monkeypatch, body()):
            seen.feed(line)

    assert seen.usage == {**EXPECTED_USAGE, "service_tier": "flex"}


@pytest.mark.parametrize("extra", [{}, {"usageMetadata": {}}], ids=["absent", "empty"])
def test_a_stream_without_usage_metadata_sends_no_usage_chunk(
    monkeypatch: pytest.MonkeyPatch, extra: dict[str, Any]
) -> None:
    """No usageMetadata (a proxy that strips it, say), or an empty one: no chunk, and the recorders estimate as
    before."""
    event = {"candidates": [{"content": {"parts": [{"text": "Hi"}]}, "finishReason": "STOP"}], **extra}

    payloads = _payloads(list(_stream(monkeypatch, f"data: {json.dumps(event)}\n\n".encode())))

    assert [p for p in payloads if isinstance(p, dict) and "usage" in p] == []
    assert payloads[-1] == "[DONE]"
```

  The `proxy-done` body has an event after `[DONE]` (which must not be read), the `json-array` body is the non-SSE form a proxy may send (the adapter requests `alt=sse`), and the failure test stubs the transport with a body that raises `httpx.ReadError` or carries an in-band error event after the two events.

  Append to `tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py` (after `test_the_service_tier_a_response_reports_is_recorded_with_its_usage`, the file's last test):

```python
async def test_a_gemini_stream_is_recorded_from_the_usage_chunk_the_adapter_emits(rows, monkeypatch) -> None:
    """GoogleAdapter.stream ends with one usage chunk built from Gemini's last (cumulative) usageMetadata, thinking
    tokens counted as output; this recorder reads it like any provider's, so the row is not an estimate (TASK-13952)."""
    import httpx

    from tldw_Server_API.app.core.LLM_Calls.providers import google_adapter as gmod

    body = (
        b'data: {"candidates":[{"content":{"parts":[{"text":"Hel"}]}}],'
        b'"usageMetadata":{"promptTokenCount":80,"candidatesTokenCount":2,"totalTokenCount":82}}\n\n'
        b'data: {"candidates":[{"content":{"parts":[{"text":"lo"}]},"finishReason":"STOP"}],'
        b'"usageMetadata":{"promptTokenCount":80,"candidatesTokenCount":30,"totalTokenCount":115,'
        b'"cachedContentTokenCount":50,"thoughtsTokenCount":5}}\n\n'
    )
    monkeypatch.setenv("LLM_ADAPTERS_NATIVE_HTTP_GOOGLE", "1")
    monkeypatch.setattr(
        gmod,
        "http_client_factory",
        lambda *_a, **_k: httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=body))),
    )

    with llm_call_context(user_id=7):
        list(gmod.GoogleAdapter().stream({**REQUEST, "model": "gemini-2.5-flash", "api_key": "sk-test"}))

    (row,) = await _settled(rows)
    assert (row["prompt_tokens"], row["completion_tokens"]) == (80, 35)
    assert row["usage_metadata"] == {
        "prompt_tokens": 80,
        "completion_tokens": 35,
        "total_tokens": 115,
        "thoughtsTokenCount": 5,
        "cachedContentTokenCount": 50,
    }
    assert (row["estimated"], row["estimate_source"]) == (None, None)
```

  In `tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py`, in the parametrize of `test_a_streamed_reply_is_recorded_from_the_usage_chunk_its_provider_sends` (`:1097-1141`), add a third case after the OpenRouter one (before `    ],`) and extend `ids`:

```python
        # Gemini: the chunk GoogleAdapter.stream ends with (TASK-13952), 8,000 of 10,000 prompt tokens cached and
        # the thinking share inside completion_tokens, at gemini-2.5-flash's standard rates ($0.30, $0.03, $2.50
        # per million), then at its flex rates ($0.15, $0.03, $1.25) when usageMetadata.serviceTier says so.
        (
            "google",
            "gemini-2.5-flash",
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 10_000,
                    "completion_tokens": 1_000,
                    "total_tokens": 11_000,
                    "thoughtsTokenCount": 400,
                    "cachedContentTokenCount": 8_000,
                },
            },
            None,
            (2_000 * 0.30 + 8_000 * 0.03 + 1_000 * 2.50) / 1_000_000,
            False,
        ),
    ],
    ids=["openai-fast-cached", "openrouter-reported-cost", "gemini-cached-thinking"],
)
```

- [ ] **Step 2: Run them to verify they fail.** The Task 2 selection. Expected: `9 failed, 8 passed` (measured): the three `..._one_usage_chunk_...` ids with `assert [] == [{'choices': ...nt': 5, ...}}]`; the three `..._recorder_reads_...` ids and the two `..._fails_after_usage_arrived_...` ids with `assert None == {'prompt_tokens': 80, 'completion_tokens': 35, ...}` (no chunk, so no usage; the failure test's `pytest.raises(ChatAPIError)` already holds: the transport error is raised sanitized as a `ChatAPIError`, and the in-band error is one); the accounting test with `assert (15, 1) == (80, 35)` (the row is the stream estimate). `test_a_stream_without_usage_metadata_sends_no_usage_chunk[absent|empty]` and `gemini-cached-thinking` pass before the code, with the five pre-existing `usage_chunk` ids: the first pins that a stream with no or an empty `usageMetadata` keeps sending no chunk (`[empty]` fails against an `if usage:` guard; see the last section, M6); the second pins the chat pipeline's reading of a Gemini-shaped chunk (the pipeline already reads any provider's; what Task 2 adds is the adapter sending one).
- [ ] **Step 3: Implement.** In `google_adapter.py`, insert after `_usage_block` (before `@staticmethod def _stream_event_deltas`):

```python
    @staticmethod
    def _usage_chunk(state: dict[str, Any]) -> Iterable[str]:
        """The stream's one usage chunk, from the last ``usageMetadata`` seen: Gemini's counts are cumulative.
        An empty ``usageMetadata`` (every count ``None``) sends none."""
        usage = state.pop("usage", None) or {}
        if any(value is not None for value in usage.values()):
            yield sse_data({"choices": [], "usage": usage})

```

  Change `_stream_event_deltas`'s signature (`:461`) to `def _stream_event_deltas(event: Any, state: dict[str, Any] | None = None) -> Iterable[str]:` and insert, after `if not isinstance(event, dict): return` (`:468-469`) and before `cands = event.get("candidates") or []`:

```python
        if isinstance(event.get("usageMetadata"), dict):
            state["usage"] = GoogleAdapter._usage_block(event["usageMetadata"])
```

  The import at `:11` becomes `from tldw_Server_API.app.core.exceptions import ChatAPIError, ChatBadRequestError`: an in-band error event raises a `ChatAPIError` (`_raise_if_in_band_provider_error`, `:640-643`), which the `except` at `:680` does not catch. In `stream`, `tool_state` moves before the `try`, so both `except` clauses can read it: insert, after `headers = merge_extra_headers(headers, request)` (`:619`) and before `try:`,

```python
        tool_state: dict[str, Any] = {"tool_index": 0}  # the next tool-call index and the last usage block
```

  and delete `seen_done = False` (`:626`) and `tool_state = {"tool_index": 0}` (`:628`). The done branch (`:644-648`) becomes:

```python
                        if is_done_line(line):
                            # A proxy's [DONE] ends the stream: the usage chunk, then [DONE], once (TASK-13952).
                            yield from self._usage_chunk(tool_state)
                            yield sse_done()
                            return
```

  and the end of the loop through the end of `stream` (`:675-681`, from the last `normalized = normalize_provider_line(line)` through `self._raise_sanitized_provider_failure(e, phase="stream")`) becomes:

```python
                        normalized = normalize_provider_line(line)
                        if normalized is not None:
                            yield normalized
                    # Gemini sends no [DONE]: the usage chunk goes out here, before finalize_stream's (TASK-13952).
                    yield from self._usage_chunk(tool_state)
                    yield from finalize_stream(response=resp)
            return
        except ChatAPIError:
            # An in-band error event mid-way: the usage seen so far goes out first, so the stream is recorded from
            # it, thinking tokens included (TASK-13952).
            yield from self._usage_chunk(tool_state)
            raise
        except _GOOGLE_ADAPTER_RUNTIME_EXCEPTIONS as e:
            yield from self._usage_chunk(tool_state)  # a transport error mid-way: likewise
            self._raise_sanitized_provider_failure(e, phase="stream")
```

  `seen_done` is gone: the done branch returns, so `finalize_stream` is reached only when no `[DONE]` came. The `except ChatAPIError` clause re-raises the in-band error as it is (it is already sanitized). A client drop or a Stop closes the generator (`GeneratorExit`, not caught), so no chunk can follow those (Decision 6).
- [ ] **Step 4: Run; see GREEN.** The Task 2 selection: expected `17 passed` (measured). Then the whole new test file: ``21 passed`` (measured). Then `tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py` and `tldw_Server_API/tests/LLM_Adapters/unit`: no failure (`770 passed`, measured). `$(dirname $PY)/ruff check` on the two test files: "All checks passed!".
- [ ] **Step 5: Commit** (message 2): `git add tldw_Server_API/app/core/LLM_Calls/providers/google_adapter.py tldw_Server_API/tests/LLM_Adapters/unit/test_google_usage_metadata.py tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py && git diff --cached --check`.

## Task 3: The catalog: Flex and Priority rates, the four newer models (AC #3)

**Files:** `tldw_Server_API/Config_Files/model_pricing.json:2, 168-177` (script); tests: `tldw_Server_API/tests/Usage/test_pricing_service_tiers.py:344-353` and the TASK-13952 section, `tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py` (the parametrize).

**Interfaces:**
- Consumes: Task 1's `service_tier` in the block (the `_log` cost test); the catalog's `service_tiers` support (TASK-13936: `_merge_overrides`, `pricing_catalog.py:316-333`; `_entry_rate_details`, `:383-410`), unchanged.
- Produces: in the test module, `GEMINI_PRICES` (10 models), `GEMINI_TIER_PRICES`, `GEMINI_LONG_TIER_PRICES` (read by the script); 14 catalog rows with `service_tiers.flex` and `.priority`.

- [ ] **Step 1: Write the failing tests.** In `test_pricing_service_tiers.py`, replace the comment and the first line of `GEMINI_PRICES` (`:344-347`):

```python
# aistudio.google.com/docs/pricing (ai.google.dev/gemini-api/docs/pricing redirects there), Standard tier, paid,
# text/image/video, fetched 2026-10-26 and again 2026-11-03: (input, output including thinking tokens, context
# caching) per million tokens, prompts up to 200k tokens. The rates in force on 2026-11-03: gemini-4-argon's double
# on 2026-12-01, gemini-3.8-flash's and gemini-3.6-flash's on 2027-01-01 (TASK-15423).
GEMINI_PRICES: dict[str, tuple[float, float, float]] = {
    "gemini-4-argon": (2.00, 10.00, 0.10),
    "gemini-3.8-flash": (0.75, 3.75, 0.075),
    "gemini-3.6-flash": (0.75, 3.75, 0.075),
    "gemini-3.5-flash-lite": (0.30, 2.50, 0.03),
    "gemini-3.1-pro-preview": (2.00, 12.00, 0.20),
```

  (the six existing entries follow unchanged). In the TASK-13952 section, insert the tables and the three tier tests before the cost test, and extend the cost test's parametrize:

```python
# --- Gemini thinking tokens, service tiers and the newer models (TASK-13952) ---

# aistudio.google.com/docs/pricing, fetched 2026-11-03. The Flex and Priority tables: (input, output including
# thinking tokens, context caching) per million tokens, text/image/video, prompts up to 200k tokens. A
# generateContent response reports the tier it ran on in usageMetadata.serviceTier (unspecified, standard, flex,
# priority). Batch is not a reportable tier: the Batch API is a separate endpoint (batchGenerateContent) that no
# server code calls.
GEMINI_TIER_PRICES: dict[str, dict[str, tuple[float, float, float | None]]] = {
    "gemini-4-argon": {"flex": (1.00, 5.00, 0.05), "priority": (3.60, 18.00, 0.18)},
    "gemini-3.8-flash": {"flex": (0.375, 1.875, 0.0375), "priority": (1.35, 6.75, 0.135)},
    "gemini-3.6-flash": {"flex": (0.375, 1.875, 0.0375), "priority": (1.35, 6.75, 0.135)},
    "gemini-3.5-flash-lite": {"flex": (0.15, 1.25, 0.02), "priority": (0.54, 4.50, 0.05)},
    "gemini-3.1-flash-lite": {"flex": (0.125, 0.75, 0.0125), "priority": (0.45, 2.70, 0.045)},
    "gemini-3.1-pro-preview": {"flex": (1.00, 6.00, 0.20), "priority": (3.60, 21.60, 0.36)},
    "gemini-3-flash-preview": {"flex": (0.25, 1.50, 0.05), "priority": (0.90, 5.40, 0.09)},
    "gemini-2.5-pro": {"flex": (0.625, 5.00, 0.125), "priority": (2.25, 18.00, 0.225)},
    "gemini-2.5-flash": {"flex": (0.15, 1.25, 0.03), "priority": (0.54, 4.50, 0.054)},
    "gemini-2.5-flash-lite": {"flex": (0.05, 0.20, 0.01), "priority": (0.18, 0.72, 0.018)},
}
# The same tables' columns for prompts over 200k tokens (the two Pro models only).
GEMINI_LONG_TIER_PRICES: dict[str, dict[str, tuple[float, float, float | None]]] = {
    "gemini-3.1-pro-preview": {"flex": (2.00, 9.00, 0.40), "priority": (7.20, 32.40, 0.72)},
    "gemini-2.5-pro": {"flex": (1.25, 7.50, 0.25), "priority": (4.50, 27.00, 0.45)},
}


@pytest.mark.parametrize(
    ("model", "tier", "prices"),
    sorted((model, tier, prices) for model, tiers in GEMINI_TIER_PRICES.items() for tier, prices in tiers.items()),
)
def test_a_gemini_model_carries_its_published_flex_and_priority_rates(
    model: str, tier: str, prices: tuple[float, float, float | None]
) -> None:
    got, estimated = PricingCatalog().get_rate_details("google", model, service_tier=tier)

    assert got == pytest.approx(_rates(prices))
    assert estimated is False


@pytest.mark.parametrize(
    ("model", "tier", "prices"),
    sorted((model, tier, prices) for model, tiers in GEMINI_LONG_TIER_PRICES.items() for tier, prices in tiers.items()),
)
def test_a_gemini_pro_prompt_over_200k_tokens_on_a_tier_pays_the_tier_long_rates(
    model: str, tier: str, prices: tuple[float, float, float | None]
) -> None:
    got, _estimated = PricingCatalog().get_rate_details("google", model, prompt_tokens=200_001, service_tier=tier)

    assert got == pytest.approx(_rates(prices))


@pytest.mark.parametrize("tier", ["flex", "priority"])
@pytest.mark.parametrize(("model", "ga_model"), sorted(GEMINI_PREVIEW_ALIASES.items()))
def test_a_gemini_2_5_preview_entry_carries_its_ga_model_tiers(model: str, ga_model: str, tier: str) -> None:
    got, _estimated = PricingCatalog().get_rate_details("google", model, service_tier=tier)

    assert got == pytest.approx(_rates(GEMINI_TIER_PRICES[ga_model][tier]))


@pytest.mark.parametrize(
    ("tier", "prices"),
    [
        (None, GEMINI_PRICES["gemini-2.5-flash"]),
        ("flex", GEMINI_TIER_PRICES["gemini-2.5-flash"]["flex"]),
        ("priority", GEMINI_TIER_PRICES["gemini-2.5-flash"]["priority"]),
    ],
    ids=["standard", "flex", "priority"],
)
```

  (the cost test's body is Task 1's, unchanged). In `test_chat_service_streaming_tool_autoexec.py`, add a fourth case after `gemini-cached-thinking` and extend `ids`:

```python
        (
            "google",
            "gemini-2.5-flash",
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 10_000,
                    "completion_tokens": 1_000,
                    "total_tokens": 11_000,
                    "thoughtsTokenCount": 400,
                    "cachedContentTokenCount": 8_000,
                    "service_tier": "flex",
                },
            },
            "flex",
            (2_000 * 0.15 + 8_000 * 0.03 + 1_000 * 1.25) / 1_000_000,
            False,
        ),
    ],
    ids=["openai-fast-cached", "openrouter-reported-cost", "gemini-cached-thinking", "gemini-flex"],
)
```

- [ ] **Step 2: Run them to verify they fail.** The Task 3 selection. Expected: `39 failed, 10 passed` (measured): the four new models in `..._context_caching_rate` with the unknown-model rate `{'prompt': 0.01, 'completion': 0.03}` (`Impossible to compare mappings with different sizes`); every `..._flex_and_priority_rates` id with the standard rates (`Mismatched elements: 2 / 3` or `3 / 3`; `estimated` True on `gemini-3.1-flash-lite`, Decision 5); the long-tier and alias ids likewise; the `flex` and `priority` cost cases with the standard cost; `gemini-flex` with `total_cost_usd` `0.00334` instead of `0.00179`. The six existing models' and four aliases' `..._context_caching_rate` ids pass (TASK-13936).
- [ ] **Step 3: Write the rows with the script.** Save as `<scratchpad>/edit_gemini_pricing.py`, then run from the repository root: `PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python <scratchpad>/edit_gemini_pricing.py`. Expected last line: `google: 14 rows` (measured; the ten page models and the four -preview aliases).

```python
"""Write the Gemini rows of model_pricing.json from the page tables in tests/Usage/test_pricing_service_tiers.py
(TASK-13952): standard rates, cache_read, long_prompt and the flex and priority service_tiers; one entry per line.

Run from the repo root: PYTHONPATH=. .venv/bin/python <this file>
Every model in GEMINI_PRICES gets a row (a missing one is inserted at the top of the "google" block); the four 2.5
-preview aliases get their GA model's rates and tiers. Other members of an existing row (note) are kept; a stale
"estimated" flag is dropped, since the row is now the page's.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PATH = Path("tldw_Server_API/Config_Files/model_pricing.json")
ROW = re.compile(r'^    "(?P<key>[^"]+)": (?P<obj>\{ .* \})(?P<comma>,?)$')
BLOCK = re.compile(r'^  "(?P<provider>[^"]+)": \{$')
TESTS = "tldw_Server_API.tests.Usage.test_pricing_service_tiers"


def num(value: object) -> str:
    """A value in the file's style: plain decimals (no exponent), JSON for everything else."""
    if isinstance(value, float):
        return format(value, ".10f").rstrip("0").rstrip(".")
    return json.dumps(value)


def fmt(obj: dict) -> str:
    """An inline object in the file's '{ "k": v }' style."""
    return "{ " + ", ".join(f'"{k}": {fmt(v) if isinstance(v, dict) else num(v)}' for k, v in obj.items()) + " }"


def rate_set(prices: tuple) -> dict:
    """Catalog rates per 1K tokens from a page row per 1M tokens."""
    prompt, completion, cached = prices
    out = {"prompt": round(prompt / 1000, 10), "completion": round(completion / 1000, 10)}
    if cached is not None:
        out["cache_read"] = round(cached / 1000, 10)
    return out


def gemini_row(key: str, existing: dict | None) -> dict:
    """The row for ``key``: its existing members, with the rates, long_prompt and service_tiers from the tables."""
    tables = __import__(TESTS, fromlist=["GEMINI_PRICES"])
    page_key = tables.GEMINI_PREVIEW_ALIASES.get(key, key)
    row = dict(existing or {})
    row.pop("estimated", None)  # the rates are the page's now, not approximate
    row.update(rate_set(tables.GEMINI_PRICES[page_key]))
    if page_key in tables.GEMINI_LONG_PRICES:
        row["long_prompt"] = {"above_tokens": 200000, **rate_set(tables.GEMINI_LONG_PRICES[page_key])}
    tiers = {}
    for tier, prices in tables.GEMINI_TIER_PRICES[page_key].items():
        tiers[tier] = rate_set(prices)
        long_prices = tables.GEMINI_LONG_TIER_PRICES.get(page_key, {}).get(tier)
        if long_prices is not None:
            tiers[tier]["long_prompt"] = {"above_tokens": 200000, **rate_set(long_prices)}
    row["service_tiers"] = tiers
    return row


def main() -> None:
    tables = __import__(TESTS, fromlist=["GEMINI_PRICES"])
    wanted = list(tables.GEMINI_PRICES) + list(tables.GEMINI_PREVIEW_ALIASES)
    text = PATH.read_text()
    provider = None
    present: list[str] = []
    out: list[str] = []
    for line in text.split("\n"):
        block = BLOCK.match(line)
        if block:
            provider = block.group("provider")
            out.append(line)
            if provider == "google":
                data = json.loads(text)["google"]
                for key in wanted:
                    if key not in data:  # a new model: inserted at the top of the block
                        out.append(f'    "{key}": {fmt(gemini_row(key, None))},')
                        present.append(key)
            continue
        row = ROW.match(line)
        if row and provider == "google" and row.group("key") in wanted:
            line = f'    "{row.group("key")}": {fmt(gemini_row(row.group("key"), json.loads(row.group("obj"))))}{row.group("comma")}'
            present.append(row.group("key"))
        out.append(line)
    new_text = "\n".join(out)
    json.loads(new_text)  # still valid JSON
    assert sorted(present) == sorted(wanted), set(wanted) ^ set(present)
    PATH.write_text(new_text)
    print(f"google: {len(present)} rows")


if __name__ == "__main__":
    main()
```

  Check two rows by eye with `git diff -U0 -- tldw_Server_API/Config_Files/model_pricing.json | grep -E '"gemini-2.5-flash":|"gemini-4-argon":'`. The prototype produced:

```json
    "gemini-4-argon": { "prompt": 0.002, "completion": 0.01, "cache_read": 0.0001, "service_tiers": { "flex": { "prompt": 0.001, "completion": 0.005, "cache_read": 0.00005 }, "priority": { "prompt": 0.0036, "completion": 0.018, "cache_read": 0.00018 } } },
    "gemini-2.5-flash": { "prompt": 0.0003, "completion": 0.0025, "cache_read": 0.00003, "service_tiers": { "flex": { "prompt": 0.00015, "completion": 0.00125, "cache_read": 0.00003 }, "priority": { "prompt": 0.00054, "completion": 0.0045, "cache_read": 0.000054 } } },
```

  and for `gemini-2.5-pro`, `"service_tiers": { "flex": { "prompt": 0.000625, "completion": 0.005, "cache_read": 0.000125, "long_prompt": { "above_tokens": 200000, "prompt": 0.00125, "completion": 0.0075, "cache_read": 0.00025 } }, "priority": { ... } }`. `git diff --stat -- tldw_Server_API/Config_Files/model_pricing.json`: `14 insertions(+), 10 deletions(-)` before Step 4 (the four inserted rows, ten rewritten; the 2.0 Flash rows and the placeholders untouched).
- [ ] **Step 4: `_catalog_note`.** In `model_pricing.json:2`, replace the sentence `Gemini cache_read (the context caching price) and the 200,000-token long_prompt tiers are from ai.google.dev/gemini-api/docs/pricing, fetched 2026-10-26; the 2.5 -preview entries take their GA model's cache_read, and the 2.0 entries have none (no published price).` with:

```text
Gemini rates, cache_read (the context caching price), the 200,000-token long_prompt tiers and the flex and priority service_tiers (the tier a generateContent response reports in usageMetadata.serviceTier; Batch is a separate endpoint no server code calls) are from aistudio.google.com/docs/pricing (ai.google.dev/gemini-api/docs/pricing redirects there), fetched 2026-11-03, at the rates in force that day: gemini-4-argon's double on 2026-12-01, gemini-3.8-flash's and gemini-3.6-flash's on 2027-01-01, and Google routes gemini-2.5-pro to gemini-3.8-flash and gemini-2.5-flash to gemini-3.5-flash-lite from 2026-12-01 (TASK-15423); the 2.5 -preview entries take their GA model's rates and tiers, and the 2.0 entries are off the page (no cache rate, no tiers). Gemini audio input is costed at the text rate (TASK-15422).
```

  `PATH="$(dirname $PY):$PATH" $PY -c "import json; json.load(open('tldw_Server_API/Config_Files/model_pricing.json'))"` must print nothing.
- [ ] **Step 5: Run; see GREEN.** The Task 3 selection: expected `49 passed` (measured). Then `tldw_Server_API/tests/Usage` whole: no failure (`test_pricing_catalog.py` compares OpenAI rate dicts and the model lists; `service_tiers` never appears in a details dict, and the four new models join `list_provider_models("google")`, which no test pins to a fixed list; measured in the focused set). `$(dirname $PY)/ruff check` on the two test files: "All checks passed!".
- [ ] **Step 6: Commit** (message 3): `git add tldw_Server_API/Config_Files/model_pricing.json tldw_Server_API/tests/Usage/test_pricing_service_tiers.py tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py && git diff --cached --check`.

## Task 4: Docs, verify, close, review, PR

- [ ] **Step 1: `Docs/Operations/Usage_Quotas.md`.**
  - In the `- **Service tiers.**` bullet (`:111`), replace `and nothing else selects a tier. OpenAI reports it beside the usage block as `service_tier`:` with `and nothing else selects a tier. Gemini reports it in `usageMetadata.serviceTier` (the **Gemini** bullet below). OpenAI reports it beside the usage block as `service_tier`:`.
  - Replace the whole `- **Gemini cache reads.**` bullet (`:112`) with (one line; `TASK-15422`/`TASK-15423`/`TASK-15424` are the real ids):

```markdown
- **Gemini.** Thinking tokens are costed as output: Google bills them with the candidates (the pricing page's "Output price (including thinking tokens)"), and the Google adapter counts `thoughtsTokenCount` in `completion_tokens`; the row's `reasoning_tokens` keeps the thinking share, and `total_tokens` is Gemini's `totalTokenCount`. Tool-use prompt tokens (`toolUsePromptTokenCount`, outside `promptTokenCount`) count as input. A streamed Gemini reply is recorded from the provider's usage: the adapter ends every stream with one usage chunk built from the last `usageMetadata` (Gemini's counts are cumulative), also when the stream fails after usage arrived. A Gemini stream that is stopped or dropped, or that fails before Gemini sends any usage, is costed from the estimate, which leaves out its thinking tokens (TASK-15424); the same holds for OpenAI reasoning tokens, which arrive only in a stream's final chunk. Gemini reports the service tier a call ran on in `usageMetadata.serviceTier` (`unspecified`, `standard`, `flex` or `priority`; the adapter passes it on as `service_tier`): `flex` and `priority` cost the Flex and Priority rates of the model's entry, copied from Google's pricing page (fetched 2026-11-03) for gemini-4-argon, gemini-3.8-flash, gemini-3.6-flash, gemini-3.5-flash-lite, gemini-3.1-pro-preview, gemini-3.1-flash-lite, gemini-3-flash-preview, gemini-2.5-pro, gemini-2.5-flash and gemini-2.5-flash-lite, and for the four 2.5 -preview entries, priced like their GA model; any other tier costs the standard rates. Contributor (invite-only) is not priced: the response cannot report it, so such a call costs the standard rates. Batch is not priced: the Batch API is a separate endpoint (`batchGenerateContent`) that no server code calls, and a `generateContent` response cannot report it. Gemini cached input (`cachedContentTokenCount`, from implicit caching or a `cachedContent` reference) costs the model's context-caching rate on gemini-4-argon, gemini-3.8-flash, gemini-3.6-flash, gemini-3.5-flash-lite, gemini-3.1-pro-preview, gemini-3.1-flash-lite, gemini-3-flash-preview, gemini-2.5-pro, gemini-2.5-flash and gemini-2.5-flash-lite, and on the four -preview entries of 2.5 Flash and Flash-Lite, which are priced like their GA model. On gemini-3.1-pro-preview and gemini-2.5-pro, a prompt over 200,000 tokens pays the higher `long_prompt` rates. The 2.0 Flash entries are not on Google's pricing page and have no cache rate, so their cached tokens cost the input rate. The per-hour storage charge of an explicit cache is not costed: the cache is created outside the server and billed by the hour, not per call. Audio input is not priced: Google charges audio input and cached audio more than text on gemini-3.1-flash-lite ($0.50 and $0.05 per million tokens against $0.25 and $0.025), gemini-3-flash-preview ($1.00 / $0.10 against $0.50 / $0.05), gemini-2.5-flash ($1.00 / $0.10 against $0.30 / $0.03) and gemini-2.5-flash-lite ($0.30 / $0.03 against $0.10 / $0.01), on every tier, but the usage record counts audio tokens inside `promptTokenCount` at the text rate (the modality split is in `promptTokensDetails` and `cacheTokensDetails`, which the adapter drops), so a reply with audio input is under-billed by up to 3.3x on its input (TASK-15422). The catalog holds the rates in force on 2026-11-03 until it follows Google's dated changes (TASK-15423): gemini-4-argon's double on 2026-12-01, and gemini-3.8-flash's and gemini-3.6-flash's on 2027-01-01. From 2026-12-01 Google routes gemini-2.5-pro requests to gemini-3.8-flash and bills them at its rates, so a gemini-2.5-pro call is over-charged until 2027-01-01 and under-charged on input and cache after it (3.8 Flash's $1.50 and $0.15 per million tokens against $1.25 and $0.125); gemini-2.5-flash requests route to gemini-3.5-flash-lite, which costs the same on standard text and bills audio at the text rate.
```

  - In the `- **Streams without a usage chunk, and router calls.**` bullet (`:115`), change `That includes every stream through the Anthropic and Google adapters, which pass no usage chunk on,` to `That includes every stream through the Anthropic adapter, which passes no usage chunk on (the Google adapter sends one since TASK-13952),`, and `even with an OpenAI provider:` to `even with an OpenAI or Google provider:` (that path stops reading at the finish chunk, before the Gemini chunk too).
- [ ] **Step 2: `Docs/Providers/Model_Pricing_Catalog.md:33`.** Replace `- Google Gemini: https://ai.google.dev/gemini-api/docs/pricing (`cache_read` is the "Context caching price")` with:

```markdown
- Google Gemini: https://aistudio.google.com/docs/pricing, where https://ai.google.dev/gemini-api/docs/pricing redirects (`cache_read` is the "Context caching price"; the Flex and Priority tables fill `service_tiers`, keyed by the `usageMetadata.serviceTier` a response reports; Batch is a separate endpoint and is not catalogued)
```

- [ ] **Step 3: Mirror.** `bash Helper_Scripts/refresh_docs_published.sh`; `git status --short Docs/Published` must print nothing (measured: neither doc is mirrored). If it prints a file, stage it with the docs.
- [ ] **Step 4: Commit** (message 4): `git add Docs/Operations/Usage_Quotas.md Docs/Providers/Model_Pricing_Catalog.md && git diff --cached --check`.
- [ ] **Step 5: Verify.** Run each of these. The expected results were measured on the prototype unless marked:
  - The focused set with label `final`: `1536 passed, 1 skipped` (measured); the new test file alone `21 passed`.
  - The wider set with label `final`, compared with Task 0's baseline: `2609 passed, 13 skipped` (measured; 63 more than the base). Any other failure is this plan's.
  - `PATH="$(dirname $PY):$PATH" $PY -m pytest -p no:cacheprovider -q -n 8 tldw_Server_API/tests/Docs tldw_Server_API/tests/lint`: `501 passed` (measured; it includes the function-size ratchet and the `Docs/Published` mirror check). Then `PATH="$(dirname $PY):$PATH" $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/CI tools/backlog-py/tests`: `1114 passed, 5 skipped` (measured).
  - `PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/check_shard_coverage.py`: `new_uncovered=0` and "OK — no newly-unshared test files." (measured).
  - `PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/test_quality_triage.py --enforce`: exit 0, "OK — no offenses beyond the baseline." (measured).
  - `PATH="$(dirname $PY):$PATH" PYTHONPATH=. $PY Helper_Scripts/ci/route_auth_ratchet.py`: "208 unauthenticated routes; baseline exact." (measured).
  - `PATH="$(dirname $PY):$PATH" PYTHONPATH=. $PY Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json --check-paths apps/packages/api-types/paths.d.ts`: "OK — OpenAPI fingerprint matches the checked-in snapshot." and "OK — committed path union matches the canonical contract." (measured).
  - The scope-baseline grep (Files): nothing (measured).
  - `git diff origin/dev --stat -- tldw_Server_API/tests/lint/function_size_baseline.json`: nothing. `git diff origin/dev --check`: nothing.
- [ ] **Step 6: Bandit**, on the touched source:

```bash
/opt/homebrew/bin/uvx bandit -ll tldw_Server_API/app/core/LLM_Calls/providers/google_adapter.py tldw_Server_API/app/core/Usage/llm_usage_normalizer.py
```

  Measured: "No issues identified.", Medium 0, High 0.
- [ ] **Step 7: Close TASK-13952** (backlog-py; fill every `<...>` first; `TASK-<A>`/`TASK-<D>`/`TASK-<F>` are the real ids):

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-13952 \
  --append-notes "How: GoogleAdapter._usage_block builds the OpenAI-shaped usage block from a usageMetadata for the reply and the stream: thoughtsTokenCount joins completion_tokens (Google bills thinking as output), toolUsePromptTokenCount joins prompt_tokens, both kept under their Gemini names (reasoning_tokens; the audit JSON), total_tokens stays totalTokenCount (= prompt + tool-use + thoughts + candidates), cachedContentTokenCount kept, serviceTier passed on as service_tier (lowercased). _stream_event_deltas keeps the last block (Gemini's counts are cumulative) and stream() yields it once as {choices: [], usage}: before [DONE] at Gemini's end of stream, before a proxy's [DONE] line (which now ends the stream), and before the error when the stream fails mid-way (a transport error, or an in-band error event, now caught as ChatAPIError and re-raised unchanged); an empty usageMetadata sends none. _StreamUsage.feed/feed_event read it unchanged, so both recorders record the stream from it (verified: the provider-level row from the real adapter through a stubbed transport; the chat pipeline from the chunk shape). normalize_llm_usage's Gemini branch adds the thoughts to output_tokens only for raw usageMetadata keys (no completion_tokens), so an adapter block is never double counted. model_pricing.json: service_tiers (flex, priority) on the 14 Gemini rows and four new rows (gemini-4-argon, gemini-3.8-flash, gemini-3.6-flash, gemini-3.5-flash-lite), written by the script from the page tables in test_pricing_service_tiers.py; gemini-3.1-flash-lite's stale estimated flag dropped." \
  --append-notes "RED/GREEN (record the real counts): Task 1 <T1-RED> -> <T1-GREEN>; Task 2 <T2-RED> -> <T2-GREEN> (the new adapter file 21 passed); Task 3 <T3-RED> -> <T3-GREEN>. Base verification of item 1: the repo fixture recorded completion_tokens 30 of 35; of item 5: a stubbed two-event stream yielded two deltas and [DONE], no usage. Focused set <base> -> <final>; wider set <base> -> <final>; tests/Docs+lint <docs-lint>; tests/CI+backlog-py <ci>; shard coverage new_uncovered=0; test-quality ratchet OK; route-auth exact; OpenAPI unchanged; Bandit <result>. Sources (fetched 2026-11-03): aistudio.google.com/docs/pricing (ai.google.dev redirects there), /docs/thinking, /api-reference/generate-content (UsageMetadata.serviceTier: unspecified, standard, flex, priority; no request field), /docs/batch-api (batchGenerateContent, a separate endpoint), /docs/flex-inference and /docs/priority-inference (a tier is requested on the Interactions API, which no server code calls; Priority reports it in the x-gemini-service-tier header)." \
  --append-notes "Not priced, recorded in Usage_Quotas.md: Batch (not a reportable tier; no server code calls batchGenerateContent); Contributor (invite-only; no ServiceTier value); audio input and cached audio (2x-3.3x the text rate on four models; the modality split in promptTokensDetails/cacheTokensDetails is dropped) -> TASK-<A>; the 2.0 Flash entries (off the page; no cache rate, no tiers; rates kept); the dated changes (gemini-4-argon doubles 2026-12-01; 3.8/3.6 Flash double 2027-01-01; from 2026-12-01 2.5 Pro is billed as 3.8 Flash, over-charged until 2027-01-01 and under-charged on input and cache after it, and 2.5 Flash as 3.5 Flash-Lite, the same on standard text; the catalog holds the 2026-11-03 rates) -> TASK-<D>, labelled high, reminders on 2026-11-30 and 2026-12-31; a stream that is stopped or dropped, or fails before any usage, is costed from the estimate without its thinking tokens -> TASK-<F>. Not catalogued: the page's specialised models (Live, TTS, Image, Transcribe, Embedding, Omni Flash, Robotics ER). Docs: Usage_Quotas.md (Service tiers, Gemini, Streams without a usage chunk), Model_Pricing_Catalog.md (source line), the _catalog_note." \
  --check-ac 1 --check-ac 2 --check-ac 3 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "Gemini thinking tokens are costed as output and tool-use prompt tokens as input: GoogleAdapter._usage_block folds thoughtsTokenCount into completion_tokens and toolUsePromptTokenCount into prompt_tokens, keeps both by name for the row's reasoning_tokens and the audit JSON, keeps cachedContentTokenCount and passes usageMetadata.serviceTier on as service_tier. The Gemini stream ends with one usage chunk from the last usageMetadata, read by the provider-level recorder and the chat pipeline without any change to them, so a streamed Gemini reply is no longer costed from the estimate. The catalog prices Flex and Priority on every Gemini text model from Google's tables and adds gemini-4-argon, gemini-3.8-flash, gemini-3.6-flash and gemini-3.5-flash-lite at the 2026-11-03 rates. A stream that fails mid-way sends the chunk before the error. Batch, Contributor, audio input, the 2.0 Flash entries, the dated price changes and the thinking tokens of a stopped or dropped stream are recorded as not priced in Usage_Quotas.md; TASK-<A> (audio), TASK-<D> (dated changes) and TASK-<F> (a stopped or dropped stream) follow up." \
  -s Done
```

  Then `git grep -n -E "<(answer|base|final|result|ci|docs-lint|T[123]-(RED|GREEN)|A|D|F)>" -- "backlog/tasks/task-13952 - *.md"` must print nothing (the grep is scoped to this task's file); `git add -A backlog/tasks && git diff --cached --check`; commit (message 5).
- [ ] **Step 8: Review before the PR.** Run `/superpowers:requesting-code-review` in a clean subagent on `origin/dev..HEAD`. Give it this plan's Review Focus and Decisions. Address the findings before opening the PR, and record each ruling in the ledger.
- [ ] **Step 9: Finish the PR and merge.**
  - `git push` (the branch and the draft PR exist since Task 0). Replace the draft's body with `gh pr edit <n> --body-file <file>`, then `gh pr ready <n>`.
  - The body states: the six items of the task and what each became (fixed with which test; not priced and why, with the follow-up id); the sources and their fetch date, including the Flex and Priority docs (a tier is requested on the Interactions API, which no server code calls, and Priority reports the tier used in the `x-gemini-service-tier` response header: no header fallback, since no server path requests a tier); the design (one `_usage_block` for reply and stream; the chunk once, before `[DONE]` or the error, in every ending; `[DONE]` ends the stream; the guarded raw-key add); the Decisions (1-6) with the owner's answers; both dated changes beside Decision 4 (2026-12-01: gemini-4-argon 2x; 2027-01-01: gemini-3.8-flash and gemini-3.6-flash 2x, and gemini-2.5-pro's input and cache become under-charged once it bills at 3.8 Flash rates) and that TASK-<D> is labelled `high` with reminders on 2026-11-30 and 2026-12-31; the gap of Decision 6 and TASK-<F>; that the four text models are added and the page's specialised models (Live, TTS, Image, Transcribe, Embedding, Omni Flash, Robotics ER) are not; the Review Focus items; the measured counts; the `## Change summary` (or the waiver; Decision 1).
  - It ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
  - Merging: `gh variable get MERGE_QUEUE` (unset when this plan was written). When reviews are settled and the Change summary is in, run `QUEUE="<n>" sh <plans dir>/pr_queue3.sh` in the background. It reads the required checks from dev's ruleset (8 today), rebases server-side when BEHIND, and merges when all pass. Wait for each body edit's re-triggered license status before merging. Confirm MERGED before deleting the branch.

## Commit messages

Each ends with a blank line and `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

- 0: `chore(backlog): start TASK-13952, file its three follow-ups, add its plan`
- 1: `fix(usage): count Gemini thinking tokens as output and pass the service tier on (TASK-13952)`. Body: `_usage_block`, the tool-use fold, the guarded raw-key add in the normalizer, the three updated asserts and why.
- 2: `fix(usage): the Gemini stream ends with one usage chunk, also when it fails (TASK-13952)`. Body: the chunk once, before `[DONE]` or the error, in every ending (`pop`; `[DONE]` returns; the `except ChatAPIError` clause re-raises an in-band error unchanged); an empty `usageMetadata` sends none; what each recorder test shows; a drop or Stop stays an estimate (TASK-<F>).
- 3: `fix(usage): price Gemini Flex and Priority and add the four newer models (TASK-13952)`. Body: the page and date, the script, the dropped `estimated` flag, Batch not catalogued and why.
- 4: `docs(usage): Gemini thinking, stream usage, tiers, and what is still not priced (TASK-13952)`
- 5: `chore(backlog): close TASK-13952`

## Not measured

- The Gemini stream through the chat pipeline with the real adapter: `test_a_streamed_reply_is_recorded_from_the_usage_chunk_its_provider_sends[gemini-*]` fakes the provider's stream with the chunk's shape (as the OpenAI and OpenRouter cases do); the real adapter's stream is measured through the provider-level recorder (`test_a_gemini_stream_is_recorded_from_the_usage_chunk_the_adapter_emits`) and `_StreamUsage.feed` (`test_the_provider_level_recorder_reads_the_usage_chunk`), which is the same reader the chat handler feeds (`feed_event`).
- A real Gemini response's `serviceTier` casing: the API reference prints `unspecified`, `standard`, `flex`, `priority` (two fetches); the adapter also strips an upper-snake `SERVICE_TIER_` prefix (pinned) in case the wire form differs.
- Gemini's in-band error event mid-stream: pinned with `{"error": {"code": 500, ...}}`, a shape `normalize_provider_stream_error` reads; a real one was not observed.
- Google's audio rates, the dated changes and a stopped or dropped stream's thinking tokens: recorded, not priced (Decisions 3, 4 and 6).

## Folded after the plan review

The independent review and dry run (`review_plan_13952.md`, 2026-11-03: 0 Blocker, 2 Important, 10 Minor; every rate matched Google's page; the dry run reproduced the prototype) was folded in on 2026-11-03 with the owner's decisions. Everything was re-measured on `6332d72b09` (#3413 still open, `origin/dev` unmoved at `576feef820`). The rebuilt prototype matches `proto_13952.patch` byte for byte.

- **I1 (Important; owner: gap as follow-up).** In this PR, a stream that fails mid-way after usage arrived sends the usage chunk before the error: `tool_state` moves before the `try`, and the transport-error `except` yields it first. The `except` the review named does not catch an in-band error event: `_raise_if_in_band_provider_error` raises a `ChatAPIError`, outside `_GOOGLE_ADAPTER_RUNTIME_EXCEPTIONS`, and that was the review's own measured probe. So a second clause, `except ChatAPIError:`, yields the chunk and re-raises unchanged. Pinned RED-first by `test_a_stream_that_fails_after_usage_arrived_is_recorded_from_it[transport-error|in-band-error]` (stubbed transport). The Usage_Quotas Gemini bullet now says a stream that is stopped or dropped, or fails before any usage, is costed from the estimate without its thinking tokens (TASK-15424), and that OpenAI reasoning tokens have the same gap. Task 0 files TASK-15424 (`high`): the `_StreamUsage` fallback, never a per-event chunk. Decision 6; V4.
- **I2 (Important; owner: scheduled reminder agent).** TASK-15423 is labelled `high`. Its description and the PR body give both dates, and the 2027-01-01 under-charge of gemini-2.5-pro's input and cache. Task 0 Step 4 has the controller schedule one-time remote agents for 2026-11-30 and 2026-12-31 once the id exists. Decision 4.
- **M1.** "Both cheaper" is reworded in V8, Decision 4, the Gemini bullet and TASK-15423. gemini-2.5-pro is over-charged until 2027-01-01, then under-charged on input and cache. gemini-2.5-flash costs the same on standard text, and its audio is billed at the text rate.
- **M2.** V5, Sources, the close-out note and the PR body cite `/docs/flex-inference`, `/docs/priority-inference`, the Interactions API and the `x-gemini-service-tier` header. No header fallback: no server path requests a tier.
- **M3.** The Sources' tier prose now says: Flex and Priority input and output are 0.5x and 1.8x, cache rates vary per model, and Batch input and output equal Flex's.
- **M4.** Contributor (invite-only) is recorded as not priced in V5, Decision 2, the Gemini bullet and the close-out note.
- **M5.** V8, the PR body, the close-out note and TASK-15423 say the four text models are added and the specialised models are not.
- **M6.** `_usage_chunk` yields only when some value of the block is not `None`. It is pinned by `test_a_stream_without_usage_metadata_sends_no_usage_chunk[empty]`, which passes before Task 2's code (no chunk at all) and fails against the pre-review prototype's `if usage:` (measured: one chunk of nulls, `assert [{'choices': ...kens': None}}] == []`).
- **M7.** The done branch returns after `[DONE]`, and `seen_done` is gone. The `proxy-done` body now has an event after `[DONE]`; against the pre-review prototype its two ids fail (measured: two usage chunks, the second after `[DONE]`, and the recorder test reads `{'prompt_tokens': 1}`).
- **M8.** The Streams bullet says "even with an OpenAI or Google provider". The stream fixture gains the one-line JSON-array body (`json-array`, 2 ids), and its "Not measured" bullet is gone.
- **M9.** The open-PR check has `--limit 200`.
- **M10.** The wider set has 68 files on the base and 69 after. `tool_state` is annotated `dict[str, Any]`. Also fixed: the Files table's split of the new file's ids (11 + 10; it said 10 + 6), and "Task 0 Step 4's baseline" (it is Step 5).
- **Counts (re-measured).**
  - Task 1: `11 failed, 3 passed` → `14 passed`, unchanged.
  - Task 2: `9 failed, 8 passed` → `17 passed`; it was `5 failed, 7 passed` → `12 passed`. The new file has `21 passed` (was 16).
  - Task 3: `39 failed, 10 passed` → `49 passed`, unchanged.
  - Focused set: `1473 passed, 1 skipped` → `1536 passed, 1 skipped` (was 1531 at the end).
  - Wider set: `2546 passed, 13 skipped` → `2609 passed, 13 skipped` (was 2604).
  - tests/Docs + tests/lint: `501 passed`. tests/CI + backlog-py: `1114 passed, 5 skipped`.
  - Shard coverage, test-quality triage and Bandit: unchanged results.
  - The prototype is 11 files, +489/−40 (was +450/−32).

## Owner answers (2026-11-03, recorded by the controller)

- **Decision 1:** the ADR-004 Change-summary waiver covers this PR ("Yes, waiver covers it").
- **Decision 3:** audio input stays out of this PR; TASK-15422 carries it (the owner approved the plan with its decisions as recommended).
- **Decision 4 / review I2:** today's rates plus TASK-15423 (`high`); the controller schedules one-time remote agents for 2026-11-30 and 2026-12-31 to open the rate-bump PR ("Scheduled reminder agent").
- **Decision 6 / review I1:** a stopped or dropped Gemini stream stays an estimate; TASK-15424 (`high`) carries the fix ("Approve, gap as follow-up").
- **Controller ruling on the fold:** keep the `except ChatAPIError` clause in `stream` — the review measured the in-band error case, and recording a failed stream from the usage that arrived is the decision's intent; it re-raises the same error.

## Added on 2026-11-05 (owner): TASK-15419 AC 3 rides in this PR

#3323 merged on 2026-11-04 without changing task-13421's reference, so TASK-15419's AC 3 falls to us. The owner answered "yes" to folding it into this PR, under this PR's waiver. In Task 0, after the follow-ups are filed and before the commit, run (backlog-py only):

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-13421 \
  --ref TASK-15149 --ref "https://github.com/rmusser01/tldw_server/pull/3092#discussion_r4171067327" \
  --append-notes "References retargeted on <date> (TASK-15419 AC 3): TASK-13215 now names the keeper; the renumbered copy this task means is TASK-15149. #3323 merged without the change."
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-15419 \
  --check-ac 3 --append-notes "AC 3 done on <date> in the TASK-13952 PR: #3323 merged without the change."
```

- First re-read task-13421's `references:` on the branch: `--ref` replaces the whole list. If it no longer reads `[TASK-13215, <the discussion URL>]`, keep every other entry and replace only `TASK-13215`.
- Expected: `git diff -- backlog/tasks` shows only task-13421's `references:` (+ `updated_date`, one note) and TASK-15419's AC 3 checkbox and note; `check_backlog_task_ids.py` exits 0.
- The provisional follow-up ids shifted: TASK-15421 was taken on 2026-11-05 (the RG retry task, PR #3420), so Task 0 filed the three follow-ups as TASK-15422 (audio), TASK-15423 (dated changes) and TASK-15424 (a stopped or dropped stream); this copy names the real ids throughout.
- The PR body names this edit under its own heading.
