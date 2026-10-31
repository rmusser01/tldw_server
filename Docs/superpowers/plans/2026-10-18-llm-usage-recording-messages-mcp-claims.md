# LLM usage recording for MCP tools, the claims rebuild and the Messages API native path: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write one `llm_usage_log` row for the right user for three kinds of LLM call that write none today: MCP tool calls, the in-process claims rebuild, and the Messages API native path (Anthropic, llama.cpp).

**Architecture:** Reuse the PR 1a/1b/2a/2b machinery. MCP tool execution names the caller with `llm_call_context(user_id=..., records_itself=False)` around `module.execute_tool`, so the provider base class (ADR-067) records the tool's calls, including tools that `/chat/completions` tool auto-execution runs (Review I3), and the PR 2b check at the call gates them. A quota refusal becomes the catalogued tool error `llm_quota_exceeded`, which the MCP module circuit breaker ignores. Each claims rebuild task carries its owner and runs under `llm_call_root(owner)`. The rebuild worker treats a quota refusal as one failed task instead of dying. The Messages native path never reaches a `ChatProvider`, so it calls the base class's own recorder through two small public wrappers in `LLM_Calls/usage_accounting.py`; that new recording entry point gets its own ADR, ADR-075 (Review I1). The stream usage reader learns the Anthropic event shape, and a null usage field never erases a count (Review I2). An Anthropic-shaped usage block counts its cached input as prompt tokens (Review B1).

**Tech Stack:** Python 3.12, FastAPI/Starlette, pytest (asyncio_mode=auto), loguru, backlog-py.

**Spec:**
- `backlog/tasks/task-13585 - Record-LLM-usage-for-MCP-tool-calls-and-the-in-process-claims-rebuild.md` (on origin/dev)
- `backlog/tasks/task-13610 - Record-LLM-usage-for-the-Messages-API-native-path.md` (on origin/dev)
- Design: `Docs/Design/2026-10-09-llm-usage-accounting-design.md` (§1 roots, §2/§3 recording, §4a/§4b check). Decision: `Docs/ADR/067-llm-calls-accounted-in-provider-base-class.md`; new `Docs/ADR/075-native-messages-and-mcp-tools-use-the-provider-recorder.md` (Proposed, Task 5).

Independent review and dry run (2026-10-18): 1 Blocker, 3 Important, 7 Minor; all folded in.

## Measured on origin/dev

Measured on origin/dev `e1f7770953`, in a scratch worktree. All line numbers below are on that commit. The whole plan was prototyped in a second scratch worktree: every RED and GREEN result quoted below was observed there.

Re-verified for the review fixes on origin/dev `32eb11ae03` (2026-10-18): none of the touched source, test or doc files changed since `e1f7770953`, so every line number holds. The revised plan was prototyped again there, and every count marked as re-measured comes from that run.

- **Recorder.** `ChatProvider.__init_subclass__` (`LLM_Calls/providers/base.py:64-69`) wraps `chat/achat/stream/astream` through `instrument_provider_class` (`LLM_Calls/usage_accounting.py:444-454`). `_plan_call()` (`:45-59`) picks the user from `current_llm_user_id()`: the explicit context first, then the content scope. `_record_response` (`:197-215`) and `_accounted_async_stream` (`:300-320`) write through `submit_accounting` → `log_llm_usage` (`Usage/usage_tracker.py:278`).
- **Stream usage reader.** `_StreamUsage.feed` (`usage_accounting.py:113-152`) reads only a top-level `usage` key and replaces it on each chunk (`:135-137`). On an Anthropic stream, `message_start` carries `usage` inside `message`, and `message_delta` carries only the cumulative `output_tokens`. Today the reader would therefore record `prompt_tokens=0`.
- **Null usage fields in `message_delta` (Review I2).** The endpoint's validator admits `null` for `input_tokens`, both cache fields and even `output_tokens` in a `message_delta` (`_native_usage_is_valid`, `messages.py:384-396`: a field fails only if it `is not None and not _is_nonnegative_int(...)`). The Anthropic SDK also types `MessageDeltaUsage.input_tokens` as optional. A plain dict merge copies such a `None` over the `message_start` count: the review's probe recorded `(0, 9)` instead of `(17, 9)`.
- **Prompt caching (Review B1).** Anthropic's `input_tokens` is only the uncached remainder; the whole prompt is `input_tokens + cache_creation_input_tokens + cache_read_input_tokens` (Anthropic API docs; `claude-api` skill, `shared/prompt-caching.md:192`).
  - `_usage_tokens` (`usage_accounting.py:92-103`) takes `input_tokens` alone as `prompt_tokens`.
  - The rest of the recorder assumes OpenAI's meaning, where `prompt_tokens` includes cached tokens. `normalize_llm_usage` computes `billable_input_tokens = input - cache_read - cache_write` (`Usage/llm_usage_normalizer.py:136-172`), and `compute_costs` caps `cache_read` at the prompt total (`Usage/usage_tracker.py:255-256`).
  - The native path is where caching happens: `AnthropicMessagesRequest` passes `cache_control` through (`extra="allow"`, `schemas/anthropic_messages.py:9`), and the validator admits both cache fields (`messages.py:242-264`).
  - The review's probe: a reply with `input_tokens: 3`, `cache_creation_input_tokens: 2000`, `cache_read_input_tokens: 60000`, `output_tokens: 100` on `claude-sonnet-4-5` records 3 prompt tokens and $0.0015, against 62,003 tokens and $0.19 at catalog rates.
  - Pricing: no provider in `Config_Files/model_pricing.json` has a `cache_read` or `cache_write` rate (0 matches), so `compute_costs` bills cached tokens at the full prompt rate until rates are added.
  - Sibling, out of scope: the Anthropic adapter also shapes `prompt_tokens` from `input_tokens` alone and drops the cache fields (`providers/anthropic_adapter.py:373-378`). The follow-up task in Task 5 covers it.
- **llama.cpp semantics (Review B1).** Checked in upstream `ggml-org/llama.cpp` `tools/server/server-task.cpp` (latest commit touching it: `8b54361025`, 2026-10-09). Its `/v1/messages` reply and its `message_start` event both report `"cache_read_input_tokens": n_prompt_tokens_cache` and `"input_tokens": n_prompt_tokens - n_prompt_tokens_cache` (lines 779-783 and 1335-1339). It sends no `cache_creation_input_tokens`, and its `message_delta` carries only `output_tokens` (line 961). So llama.cpp follows Anthropic's meaning, and the same sum applies: `input_tokens + cache_read_input_tokens` is its whole prompt. Its OpenAI-compatible endpoint instead reports `prompt_tokens` with `prompt_tokens_details.cached_tokens` (line 362), which the sum does not touch.
- **Roots.** `llm_call_root(user_id, *, operation=None)` (`Usage/llm_roots.py:54-70`) sets the user and forces `prechecked=False` ("a root is never admitted by the request that started it").
- **MCP.** The only direct LLM call in `core/MCP_unified` is `quizzes_module.py:1097` (`perform_chat_api_call_async`). Every `tools/call` runs `module.execute_tool(tool_name, execution_args, context)` at one point: `tool_execution/runtime.py:515-519`, inside `module.execute_with_circuit_breaker` (`modules/base.py:535-613`). The breaker counts any exception except an `ExpectedToolFailure` whose reason says `BreakerAction.IGNORE` (`base.py:562-568`). It opens after `circuit_breaker_threshold=5` failures for 60 s (`base.py:150-151`), for every user of that module. The MCP JWT, API-key and single-user auth all put the numeric user id in `TokenData.sub` (`endpoints/mcp_unified_endpoint.py:387-388`, `:438-441`, `:532`, `:563`). It reaches `RequestContext.user_id` as a string (`MCP_unified/server.py:1951-1958`, `:2090-2097`, `:2190-2197`).
- **MCP inside an admitted request.** `/chat/completions` tool auto-execution runs MCP tools in-process (`Chat/tool_auto_exec.py:230` → `Tools/tool_executor.py:62` → the same runtime). Those calls carry `prechecked=True` from the boundary check. `llm_call_root` would clear that mark, so the MCP change uses `llm_call_context` and keeps `prechecked`.
- **MCP inside a caller that records itself (Review I3).**
  - `chat.py` runs `execute_streaming_call` and `_execute_non_stream_call_impl` with `records_itself=True` (`chat.py:5549`, `:5822`), and tool auto-execution runs inside them (`chat_service.py:6236`, `:7477`).
  - The chat recorder writes one row for the chat model call only (`chat_service.py:6279`, `operation="chat"`), so it never covers a tool's own LLM call.
  - An overlay that sets only the user inherits `records_itself=True`, so `_plan_call()` returns `None`. The review's probe made 1 provider call and wrote 0 rows. ADR-067 states this ("a tool that itself calls an LLM, are recorded by nobody").
  - Clearing `records_itself` in the MCP overlay double-counts nothing. The only `records_itself=True` callers are the chat pipeline, the three router calls and the claims extract and verify calls (`git grep records_itself`). None of them records a tool's calls. The only MCP module that calls an LLM, quizzes (`quizzes_module.py:1091-1097`), goes through `perform_chat_api_call_async`, which records nothing itself. A self-recording function a tool calls sets its own `records_itself=True` inside the overlay.
- **A defect that already exists.** `/api/v1/tools/execute` (`endpoints/tools.py:73-75`) authenticates with `get_request_user`, which sets the content scope. So since PR 2b, an over-quota MCP tool call on that path raises a raw `LLMQuotaExceededError` that the module breaker counts. This was found by reading the code; Task 2's translation fixes it for every path.
- **Claims rebuild.** `ClaimsRebuildTask` has only `media_id` and `db_path` (`Claims_Extraction/claims_rebuild_service.py:48-51`). `_process_task` (`:274-275`) runs on raw daemon threads (`:153`), and the singleton has one worker (`:309`). `_worker_loop` catches only `_CLAIMS_REBUILD_NONCRITICAL_EXCEPTIONS` (`:256`), which has no `HTTPException`. Once a task has an owner, a quota refusal would therefore end the only worker thread. The prototype measured this: `assert (0, 0, [1]) == (1, 1, [1, 2])`, with the second task stranded. LLM extraction (`ingestion_claims.py:262-470`, `run_sync_claims_strategy` with `catch_exceptions=_CLAIMS_NONCRITICAL_EXCEPTIONS`) lets the refusal through. It is raised before the old claims are soft-deleted (`claims_rebuild_service.py:89-113`).
- **Who submits rebuilds, and the owner each one knows.**
  - `claims_service._enqueue_claim_rebuild_if_needed` (`:634`) has `owner_user_id`.
  - `rebuild_claims` (`:4599`) and `rebuild_all_media` (`:4649`) each compute `owner_user_id`.
  - `services/startup_claims_rebuild.py:146,155` receives `user_id` from `_claims_rebuild_db_session` and discards it as `_`.
- **Messages native path.** `_handle_messages` (`endpoints/messages.py:2017-2154`) sends `MESSAGES_NATIVE_PROVIDERS = {"anthropic", "llama.cpp"}` (`:67`) through `_native_post_json` (`:2072`) or `_prepare_native_stream_iterator` (`:2054`). It reaches no adapter, and only `perform_chat_api_call_async` on the converted path (`:2104`, `:2131`) is recorded today. A valid native message always carries `usage` with `input_tokens` and `output_tokens`. The validators guarantee this: `_native_message_payload_is_semantic`; `_native_message_start_is_valid` requires `usage` (`:1341-1376`); `message_delta` requires `output_tokens` (`:1379-1415`); the case `missing-usage` is rejected in `test_messages_endpoint_usage.py`.
- **Messages check at the call (the brief's question).** Both routes into `_handle_messages` declare `Depends(require_llm_quota)` (`messages.py:2221`, `:2252`). That dependency calls `mark_llm_quota_prechecked()` (`API_Deps/usage_quota_deps.py:47-57`), and `user_to_check()` returns `None` when `prechecked` is set (`Usage/llm_quota_gate.py:60-61`). A PR 2b check on the native path would therefore always be skipped. PR 2a already pins the 402 for the native path, streaming and not: `tests/Chat/integration/test_messages_endpoint.py::test_messages_refuses_an_over_quota_user_before_any_provider_call` (its `llama.cpp/test-model` cases, with `stream` False and True). The plan adds no call-site check. `count_tokens` (`:2280-2321`) generates nothing and stays unrecorded and ungated.
- **Estimates ignore Anthropic shapes (Review M4).** `_response_chars` and `_StreamUsage.chars` count only OpenAI `choices`, and `_prompt_estimate` reads `system_message`, not Anthropic's `system` (`usage_accounting.py:67-89`, `:138-150`). So a native reply whose usage values are all null (the validator accepts it), or a stream that fails before `message_start`, records an estimate with `completion_tokens=0` and no system prompt. The review measured `(9, 0, estimated=True)` for both. An invalid native stream, which the sanitizer ends with an error frame, still writes that estimate row; that is ADR-067's "a stream is recorded once, however it ends". This plan leaves the estimators as they are and says so in ADR-075: `plan_chat_token_estimators.md` rewrites `_prompt_estimate`, so teaching it Anthropic shapes here would collide (see M6 below).
- **`/api/v1/tools/execute` and a tool error (Review M7).** The endpoint returns the MCP result inside `ExecuteToolResult(ok=True, result=...)` (`endpoints/tools.py:90-97`), and `ToolExecutor.execute` returns `resp.result` (`Tools/tool_executor.py:119-131`). So an over-quota tool call there is HTTP 200 with `ok: true` and `result.isError: true`, not a 402. That is the MCP tool-error contract, so `Usage_Quotas.md` names it in the MCP bullet.
- **ADR number (Review I1).** The highest ADR on origin/dev is 071. Open PR #3226 adds ADR-072 to ADR-074 (checked with the paginated files API, because `gh pr list --json files` stops at 100 files per PR; no other open PR adds an ADR above 071). `plan_concurrency_quotas.md` proposes ADR-075. So this plan's ADR is ADR-075; Task 1 Step 0 recomputes it.
- **Other plans touch the same files (Review M6).**
  - `plan_chat_token_estimators.md` edits `usage_accounting.py` (imports `:20-30`, `_prompt_estimate` `:67-78`).
  - `plan_audio_analysis.md` edits `Usage_Quotas.md:95`, where this plan inserts after `:95`.
  - `plan_quota_callers_rag_metric.md` edits `Usage_Quotas.md:93`, and `plan_concurrency_quotas.md` edits `Usage_Quotas.md` and adds a `Docs/ADR/README.md` row after ADR-071, as this plan does.
  - This plan's edits are anchored on text, which survives a rebase; its line numbers do not. If one of those plans lands first, find each edit by its quoted text, and recompute the ADR number (Task 1 Step 0).
- **Line-number baselines.** None of the touched files is listed in `Helper_Scripts/ci/scope_predicate_baseline.txt`.
- **Mirrors.** `Helper_Scripts/refresh_docs_published.sh` mirrors all of `Docs/ADR` (line 141): ADR-067, the new ADR-075 and `Docs/ADR/README.md` change under `Docs/Published/ADR/`. `Docs/Operations/Usage_Quotas.md` and the design doc are not mirrored (the script copies only `Operations/Env_Vars.md` from Operations). `tests/Docs/test_docs_published_refresh.py::test_refresh_output_matches_tracked_published_files` compares the refreshed tree with `git ls-files Docs/Published`, so a new mirrored file fails it until it is staged (measured: `1 failed, 211 passed` unstaged, `212 passed` staged).
- **Lint and security.**
  - `ruff check` on the touched files reports one error, which is already on dev: unsorted imports in `claims_service.py:62-75`. Leave it.
  - `bandit` is not installed in `.venv`. `uvx bandit` on the touched source reports 0 findings.
- **Flaky test.** `tests/Claims/test_claims_jobs_contracts.py::test_review_metrics_payload_inclusive_window_property` failed once in one full `tests/Claims` run in the prototype. It passed alone, in its file, on a rerun of the full suite with the change, and on unchanged dev (1077 passed). Treat a repeat as a flake: rerun it before you investigate.

## Global Constraints

- Every PR's final sweep includes `tldw_Server_API/tests/CI/test_test_quality_ratchet.py`: since #3243, dev enforces it, so new tests must not be status-only or tautological.
- PR targets `dev`. Branch from `origin/dev`. One PR for both tasks: they share `usage_accounting.py` and the same doc pages (ADR-067, the new ADR-075, `Usage_Quotas.md`, the design doc).
- Never `--no-verify`. Every commit ends with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- Backlog changes only through backlog-py (ADR-059): `PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit ...`. Use `--append-notes`, never `--notes`. Stage with `git add -A backlog/tasks`. One new task is created, the Anthropic cache-pricing follow-up (Review B1). Compute its id with the AGENTS.md command, which scans origin/dev and open PRs (Task 5 Step 8).
- After editing ADR-067 and adding ADR-075, run `bash Helper_Scripts/refresh_docs_published.sh`, stage the new files, and commit the `Docs/Published` diff in the same PR (Review I1).
- No report or scratch path hard-codes `/tmp` (ADR-006, Accepted): the Bandit report and the PR body go under `$($PY -c 'import tempfile; print(tempfile.gettempdir())')` (Review M1).
- Other plans edit `usage_accounting.py`, `Usage_Quotas.md` and `Docs/ADR/README.md` (see Measured, Review M6). After a rebase, find each edit by its quoted text, not its line number.
- Use `$PY` for `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`. Run tests from the repo root as `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q <paths>`.
- "No known user, no row" (spec §1, non-goal): a caller whose id is not a number is neither recorded nor checked.
- An accounting or check failure must never break the call (ADR-067). The only error the new code lets out is `LLMQuotaExceededError`.

## ADR assessment

- **Governing:** ADR-067 (Accepted). It is binding: chat calls are accounted by the provider base class's recorder, and the user comes from an explicit `llm_call_context` or the content scope. ADR-006 (Accepted) governs report paths (Review M1).
- **ADR required: yes. ADR path: `Docs/ADR/075-native-messages-and-mcp-tools-use-the-provider-recorder.md`, `Status: Proposed`. It extends ADR-067 and does not supersede it (Review I1).**
  - **The native path is a new decision, not a status update.** ADR-067 decides that chat calls are accounted in `ChatProvider.__init_subclass__`, and the design calls it "One recording point". Two public entry points outside the base class, `record_direct_response` and `accounted_direct_stream`, open that rule to any adapter-less provider call. That is a durable provider-integration rule (`Docs/ADR/README.md`, "ADR Required When"), and README rule 8 and AGENTS.md ("a changed decision needs a new ADR") require a new ADR for it.
  - **The MCP overlay changes an ADR-067 consequence (Review I3).** ADR-067 says a tool that itself calls an LLM inside the chat pipeline is "recorded by nobody". Clearing `records_itself` records it, so ADR-075 records that change.
  - **The cached-token sum (Review B1)** is part of how ADR-075's recorder reads an Anthropic-shaped block, so ADR-075 states it.
  - Rejected alternatives, kept in ADR-075: a fake `ChatProvider` subclass for the native path (its `achat` would record before `_native_message_payload_is_semantic` rejects a bad 200; its stream wrapper would see raw byte chunks split mid-frame; it would be a provider class that is not a provider); a recorder of its own in `messages.py`; recording in `perform_chat_api_call*`.
- **No new ADR for the roots.** The MCP user and the claims-rebuild owner are new roots under ADR-067's existing mechanism (`llm_call_context`, `llm_call_root`). This is the same kind of change PR 1b made, and PR #3224 recorded it as status updates to ADR-067's Consequences and Follow-up.
- **Update ADR-067's status lines only** (Task 5): MCP's user source, the shorter "Still unrecorded" list, and the Follow-up line. The native-path bullet the first draft added to ADR-067 is dropped; it lives in ADR-075 (Review I1). Refresh the mirror afterwards.
- **The ADR check** (required yes/no, path, reason) goes into the backlog notes and the PR body, as AGENTS.md and `.agents/skills/adr-assessment` require (Review M2).

## Decisions for the owner

Recommendations first. The plan implements each recommendation; the alternative says what to change if the owner declines.

1. **D1: Count cached input as prompt tokens on the Messages native path (Review B1).**
   - **Recommendation:** count Anthropic's `input_tokens + cache_creation_input_tokens + cache_read_input_tokens` as `prompt_tokens`, for any usage block that has `input_tokens` and no `prompt_tokens`. That is OpenAI's meaning of `prompt_tokens`, which `normalize_llm_usage` and `compute_costs` already assume. Without it, prompt-cached `/v1/messages` traffic is recorded at a tiny fraction: the review's probe recorded 3 tokens and $0.0015 for a call that used 62,003 tokens and $0.19 at catalog rates. Then `limits.llm_tokens_per_month` and both cost limits are barely enforced on the path TASK-13610 exists to account. llama.cpp reports the same way (Measured), so the sum applies to it too.
   - **Implemented in:** Task 1 Step 3 (`_usage_tokens`), with a non-streaming test for Anthropic and llama.cpp and a streaming test.
   - **Until cache rates exist,** cache reads are billed at the full prompt rate, which overstates cost. Task 5 Step 8 files the follow-up task that adds Anthropic cache rates.
   - **If the owner declines:** delete the `if "prompt_tokens" not in usage:` lines from Task 1 Step 3, the two `..._counts_its_cached_input_as_prompt_tokens` tests and `CACHED_USAGE` from Task 1 Step 1, the "Anthropic-shaped usage counts cached input" consequence from ADR-075, the last sentence of the design doc's "No adapter" bullet, and the follow-up task. Task 1's RED becomes `6 failed, 2 passed` and its GREEN `8 passed`.
2. **D2: Record MCP tools that `/chat/completions` tool auto-execution runs (Review I3).**
   - **Recommendation:** record them. The MCP overlay sets `records_itself=False`. TASK-13585's AC #1 covers MCP tool calls, and the chat request's own row covers only its model call, so nothing is counted twice (Measured). The tool keeps the request's `prechecked` mark, so it is recorded but never refused part-way (spec §4a). This changes ADR-067's consequence that such a tool is "recorded by nobody"; ADR-075 records the change.
   - **Implemented in:** Task 2 Step 5, with `test_a_tool_run_by_chat_tool_auto_exec_is_recorded_for_the_user`.
   - **If the owner declines:** apply "If the owner declines D2" in Task 2 Step 5 and in Task 5 Steps 1 to 3 and 9. Those tools then stay unrecorded and unchecked, as ADR-067 says today.
3. **D3: Accept ADR-075 (Review I1).** It is written as `Status: Proposed`. If the owner accepts it before merge, set `**Status:** Accepted` in the ADR and its index row, rerun the refresh script, and commit both (Task 5 Step 11). If D1 or D2 is declined, edit ADR-075 first as those alternatives say.

Not a decision: the brief's open question has a clear answer (see "Messages check at the call" above). The native path is always admitted by `require_llm_quota` before it runs, so a PR 2b check there would never run. The plan relies on the existing boundary gate and its existing test.

## Review Focus

These are the input classes the specs imply that could break for a user. Each has a test in the task that owns the code:

1. **A native stream whose first SSE frame arrives split across two socket reads.** Its tokens must still be read. Task 1: every stream test splits frame 1 at byte 30.
2. **A client that disconnects after the first native frame.** It must still cost the input tokens it was sent: one row, `prompt_tokens` from `message_start`. Task 1: `test_a_native_stream_closed_early_still_records_one_row`.
3. **A prompt-cached native call (Review B1).** Its cache reads and writes must count as prompt tokens, streaming and not, for Anthropic and for llama.cpp. Task 1: `test_a_cached_native_reply_counts_its_cached_input_as_prompt_tokens` (both providers) and `test_a_cached_native_stream_counts_its_cached_input_as_prompt_tokens`.
4. **A `message_delta` with null usage fields (Review I2).** It must not erase the `message_start` input count. Task 1: `test_message_delta_usage_keeps_the_input_count_from_message_start` (null fields, and a cumulative restatement).
5. **Two users' MCP tool calls in flight at once.** Each row must carry its own user. Task 2: `test_concurrent_tool_calls_are_recorded_for_their_own_users`.
6. **An MCP tool run by an admitted `/chat/completions` request (tool auto-execution).** It must not be refused part-way (spec §4a), which `llm_call_root` would allow. Task 2: `test_a_tool_call_inside_an_admitted_request_is_not_refused`, a guard that passes before and after. Under D2 it must also be recorded for the user, although the chat pipeline is marked `records_itself` (Review I3). Task 2: `test_a_tool_run_by_chat_tool_auto_exec_is_recorded_for_the_user`.
7. **A claims rebuild refused part-way.** It must delete none of the media's existing claims. Task 3: `test_a_refusal_during_extraction_leaves_the_media_claims_in_place`. It is a guard and passes before and after.

Not exercised by these tests (Review M4):

- **Estimate rows on the native path.** A native reply whose usage values are all null, or a stream that fails before `message_start`, records an estimate that counts no Anthropic content and no `system` prompt (Measured). ADR-075 states this; nothing tests it.
- **An invalid native stream** (the sanitizer sends an error frame) still writes that estimate row, by ADR-067's "recorded once, however it ends". Not tested.
- **The Anthropic adapter path** drops the cache fields (Measured). The follow-up task covers it.

---

### Task 1: The Messages API native path records one row per call (TASK-13610)

**Files:**
- Modify: `tldw_Server_API/app/core/LLM_Calls/usage_accounting.py:17` (import), `:103` (`_usage_tokens`, Review B1), `:135-137` (`_StreamUsage.feed`, Review I2), and before `:436` (two new functions)
- Modify: `tldw_Server_API/app/api/v1/endpoints/messages.py:9`, `:58`, `:2052-2069`, `:2080-2082`
- Create: `tldw_Server_API/tests/Chat/unit/test_messages_native_usage_recording.py`

**Interfaces:**
- Consumes: `_plan_call() -> _CallPlan | None`, `_record_response(provider, plan, request, response, started)` and `_accounted_async_stream(provider, plan, request, inner, started)` from `usage_accounting.py`. They read `provider.name` only.
- Changes: `_usage_tokens(usage) -> tuple[int, int]`. For a block with no `prompt_tokens` key, the prompt count adds `cache_creation_input_tokens` and `cache_read_input_tokens` to `input_tokens`, each read like the other fields (a missing, null or non-finite value counts 0). Every recorded provider call goes through it; only an Anthropic-shaped block (`input_tokens` with those cache fields and no `prompt_tokens`) changes, and today only the native path sends one (the adapters convert to `prompt_tokens`).
- Produces:
  - `record_direct_response(provider: str, request: dict[str, Any], response: Any, started: float) -> None`. It never raises.
  - `accounted_direct_stream(provider: str, request: dict[str, Any], stream: AsyncIterator[Any], started: float) -> AsyncIterator[Any]`. It returns `stream` unchanged when nothing is recorded.

- [ ] **Step 0: Branch, set the interpreter, and confirm the ADR number**

```bash
git fetch origin dev --quiet
git switch -c feat/llm-usage-mcp-claims-messages origin/dev
export PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python
{ git ls-tree -r --name-only origin/dev Docs/ADR
  for n in $(gh pr list --state open --limit 500 --json number --jq '.[].number'); do
    gh api "repos/rmusser01/tldw_server/pulls/$n/files" --paginate --jq '.[].filename'
  done; } | sed -nE 's|^Docs/ADR/([0-9]{3})-.*|\1|p' | sort -n | tail -1
```

Expected: `074` (2026-10-18; open PR #3226 adds ADR-072 to ADR-074). The loop reads each PR's files through the paginated API because `gh pr list --json files` stops at 100 files per PR. ADR-075 is reserved by `plan_concurrency_quotas.md`, so this plan's ADR is ADR-075 (Review I1, M6). If the output is 075 or higher, use one more than the output, and replace `076`/`ADR-075` everywhere this plan writes it: the code comments in Task 1 Step 5 and Task 2 Step 5, the ADR file name, its index row, ADR-067's status lines, the design doc, and the backlog notes.

- [ ] **Step 1: Write the failing tests**

Create `tldw_Server_API/tests/Chat/unit/test_messages_native_usage_recording.py`. Beyond the first draft it adds the cached-input tests for D1 (Review B1) and the null-field `message_delta` test (Review I2); `_frames`, `_native_reply`, `_native_stream` and `_stream_row` build the stream and reply variants those need:

```python
"""The Messages API native path records one llm_usage_log row per call (TASK-13610, LLM usage accounting spec §3)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.endpoints import messages as messages_endpoint
from tldw_Server_API.app.api.v1.schemas.anthropic_messages import AnthropicMessagesRequest
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.app.core.Usage import llm_accounting

pytestmark = pytest.mark.unit

NATIVE_RESPONSE = {
    "id": "msg-native-1",
    "type": "message",
    "role": "assistant",
    "model": "claude-native",
    "content": [{"type": "text", "text": "valid native output"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 21, "output_tokens": 8},
}
# Anthropic's input_tokens is only the uncached remainder; the prompt is the sum of the three input fields.
CACHED_USAGE = {"input_tokens": 3, "cache_creation_input_tokens": 2000, "cache_read_input_tokens": 60000}


def _frames(start_usage: dict[str, Any], delta_usage: dict[str, Any]) -> list[str]:
    """An Anthropic Messages SSE reply whose message_start and message_delta carry these usage blocks."""
    start = {
        "type": "message_start",
        "message": {
            "id": "msg-s",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": start_usage,
        },
    }
    delta = {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": delta_usage}
    return [
        f"event: message_start\ndata: {json.dumps(start, separators=(',', ':'))}\n\n",
        'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
        '"content_block":{"type":"text","text":""}}\n\n',
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"text_delta","text":"ok"}}\n\n',
        'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
        f"event: message_delta\ndata: {json.dumps(delta, separators=(',', ':'))}\n\n",
        'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ]


STREAM_FRAMES = _frames({"input_tokens": 17, "output_tokens": 1}, {"output_tokens": 9})


class _Runtime:
    """A credential runtime that resolves a key and records nothing."""

    def __init__(self, **_kwargs: Any) -> None:
        """Accept the endpoint's scope arguments."""

    async def resolve(self, provider: str, *, model: str | None = None) -> Any:
        """Return resolved credentials for the provider."""
        del model
        return SimpleNamespace(provider=provider, api_key="k", app_config={}, credentials_resolved=True)

    async def mark_used(self, _handle: Any) -> None:
        """Credential touches are not under test."""

    async def close(self) -> None:
        """Nothing to close."""


@pytest.fixture
async def rows(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture what would be written, and make the test's loop the accounting loop."""
    captured: list[dict[str, Any]] = []

    async def _fake_log(**kwargs: Any) -> None:
        """Record the row instead of writing it."""
        captured.append(kwargs)

    monkeypatch.setattr("tldw_Server_API.app.core.Usage.usage_tracker.log_llm_usage", _fake_log)
    monkeypatch.setattr(llm_accounting, "_accounting_loop", asyncio.get_running_loop())
    monkeypatch.setattr(messages_endpoint, "derive_trusted_credential_scope", lambda _r, user: (user.id_int, [], [], False))
    monkeypatch.setattr(messages_endpoint, "ProviderCredentialRuntime", _Runtime)
    monkeypatch.setattr(messages_endpoint, "_resolve_messages_base_url", lambda _p, _c: "http://provider.test/v1")
    return captured


async def _call(model: str, *, stream: bool) -> Any:
    """Run the Messages handler as an authenticated request for user 5 would."""
    return await messages_endpoint._handle_messages(
        AnthropicMessagesRequest(model=model, messages=[{"role": "user", "content": "hello"}], stream=stream),
        current_user=SimpleNamespace(id_int=5),
        request=SimpleNamespace(),
        anthropic_version=None,
        anthropic_beta=None,
    )


def _native_reply(usage: dict[str, Any]) -> Any:
    """A stand-in for _native_post_json whose JSON reply carries ``usage``."""

    async def _post(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        """The provider's JSON reply."""
        return {**NATIVE_RESPONSE, "usage": usage}

    return _post


def _native_stream(frames: list[str]) -> Any:
    """A stand-in for _prepare_native_stream_iterator whose SSE reply is ``frames``."""

    async def _prepare(*_args: Any, **_kwargs: Any) -> Any:
        """The provider's SSE reply, split mid-frame the way a socket splits it."""

        async def _source() -> Any:
            """Yield the frames as raw bytes, the first one in two pieces."""
            first = frames[0].encode()
            yield first[:30]
            yield first[30:]
            for frame in frames[1:]:
                yield frame.encode()

        return _source()

    return _prepare


async def _stream_row(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, Any]], frames: list[str]) -> dict[str, Any]:
    """Stream ``frames`` to the client as user 5 and return the one row written."""
    monkeypatch.setattr(messages_endpoint, "_prepare_native_stream_iterator", _native_stream(frames))
    with scoped_context(user_id=5):
        response = await _call("anthropic/claude-3-5-haiku-latest", stream=True)
        body = "".join([item async for item in response.body_iterator])
    assert body == "".join(frames)
    await llm_accounting.drain_accounting()
    (row,) = rows
    return row


@pytest.mark.parametrize("model", ["anthropic/claude-3-5-haiku-latest", "llama.cpp/local-model"])
async def test_a_native_reply_records_one_row_from_its_usage_block(rows, monkeypatch, model) -> None:
    """Non-streaming: one row for the scope user, with the provider's own token counts."""
    monkeypatch.setattr(messages_endpoint, "_native_post_json", _native_reply(NATIVE_RESPONSE["usage"]))
    with scoped_context(user_id=5):
        response = await _call(model, stream=False)
    assert response.status_code == 200
    await llm_accounting.drain_accounting()
    (row,) = rows
    provider, _, model_name = model.partition("/")
    assert (row["user_id"], row["provider"], row["model"]) == (5, provider, model_name)
    assert (row["prompt_tokens"], row["completion_tokens"]) == (21, 8)
    assert row["usage_metadata"] == NATIVE_RESPONSE["usage"]
    assert row["estimated"] is None


async def test_a_native_stream_records_one_row_with_input_and_output_tokens(rows, monkeypatch) -> None:
    """Streaming: input tokens from message_start and the final output count from message_delta."""
    row = await _stream_row(monkeypatch, rows, STREAM_FRAMES)
    assert (row["user_id"], row["provider"], row["model"]) == (5, "anthropic", "claude-3-5-haiku-latest")
    assert (row["prompt_tokens"], row["completion_tokens"]) == (17, 9)


@pytest.mark.parametrize(
    "delta_usage",
    [{"input_tokens": None, "cache_read_input_tokens": None, "output_tokens": 9}, {"input_tokens": 17, "output_tokens": 9}],
    ids=["null-fields", "cumulative"],
)
async def test_message_delta_usage_keeps_the_input_count_from_message_start(rows, monkeypatch, delta_usage) -> None:
    """A null usage field in message_delta keeps the count already read; a cumulative one restates it."""
    row = await _stream_row(monkeypatch, rows, _frames({"input_tokens": 17, "output_tokens": 1}, delta_usage))
    assert (row["prompt_tokens"], row["completion_tokens"]) == (17, 9)


@pytest.mark.parametrize(
    ("model", "usage", "prompt_tokens"),
    [
        ("anthropic/claude-sonnet-4-5", {**CACHED_USAGE, "output_tokens": 100}, 62003),
        # llama.cpp sends input_tokens = n_prompt_tokens - n_prompt_tokens_cache and no cache writes.
        ("llama.cpp/local-model", {"cache_read_input_tokens": 60000, "input_tokens": 3, "output_tokens": 100}, 60003),
    ],
    ids=["anthropic", "llama.cpp"],
)
async def test_a_cached_native_reply_counts_its_cached_input_as_prompt_tokens(
    rows, monkeypatch, model, usage, prompt_tokens
) -> None:
    """input_tokens is only the uncached remainder, so cache reads and writes count as prompt tokens too."""
    monkeypatch.setattr(messages_endpoint, "_native_post_json", _native_reply(usage))
    with scoped_context(user_id=5):
        response = await _call(model, stream=False)
    assert response.status_code == 200
    await llm_accounting.drain_accounting()
    (row,) = rows
    assert (row["prompt_tokens"], row["completion_tokens"]) == (prompt_tokens, 100)
    assert row["usage_metadata"] == usage


async def test_a_cached_native_stream_counts_its_cached_input_as_prompt_tokens(rows, monkeypatch) -> None:
    """The same for a stream: message_start's cache fields count as prompt tokens."""
    row = await _stream_row(
        monkeypatch, rows, _frames({**CACHED_USAGE, "output_tokens": 1}, {"output_tokens": 100})
    )
    assert (row["prompt_tokens"], row["completion_tokens"]) == (62003, 100)
    assert row["usage_metadata"]["cache_read_input_tokens"] == 60000


async def test_a_native_stream_closed_early_still_records_one_row(rows, monkeypatch) -> None:
    """A client that disconnects after the first frame still costs the input tokens it was sent."""
    monkeypatch.setattr(messages_endpoint, "_prepare_native_stream_iterator", _native_stream(STREAM_FRAMES))
    with scoped_context(user_id=5):
        response = await _call("anthropic/claude-3-5-haiku-latest", stream=True)
        iterator = response.body_iterator
        assert await iterator.__anext__() == STREAM_FRAMES[0]
        await iterator.aclose()
    await llm_accounting.drain_accounting()
    (row,) = rows
    assert (row["user_id"], row["prompt_tokens"]) == (5, 17)


async def test_an_invalid_native_reply_records_nothing(rows, monkeypatch) -> None:
    """A 200 that is not a valid message is reported as a 502 and, like a failed adapter call, not recorded."""

    async def _native_bad(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        """A reply with no content."""
        return {**NATIVE_RESPONSE, "content": []}

    monkeypatch.setattr(messages_endpoint, "_native_post_json", _native_bad)
    with scoped_context(user_id=5), pytest.raises(Exception) as raised:
        await _call("anthropic/claude-3-5-haiku-latest", stream=False)
    assert getattr(raised.value, "status_code", None) == 502
    await llm_accounting.drain_accounting()
    assert rows == []


async def test_with_no_user_a_native_reply_records_nothing(rows, monkeypatch) -> None:
    """No content scope and no explicit user: the call runs and nothing is written."""
    monkeypatch.setattr(messages_endpoint, "_native_post_json", _native_reply(NATIVE_RESPONSE["usage"]))
    response = await _call("anthropic/claude-3-5-haiku-latest", stream=False)
    assert response.status_code == 200
    await llm_accounting.drain_accounting()
    assert rows == []
```

- [ ] **Step 2: Run them and see them fail**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Chat/unit/test_messages_native_usage_recording.py`
Expected: `9 failed, 2 passed` (re-measured). Each failure is `ValueError: not enough values to unpack (expected 1, got 0)`: no row was written. The two "records nothing" tests are guards and pass.

- [ ] **Step 3: Teach the stream reader the Anthropic events, and count cached input**

In `tldw_Server_API/app/core/LLM_Calls/usage_accounting.py`:

1. Replace lines 135-137:

```python
                usage = data.get("usage")
                if isinstance(usage, dict) and usage:
                    self.usage = usage
```

with:

```python
                usage = data.get("usage")
                event = data.get("type")
                if event == "message_start":  # Anthropic Messages: input tokens arrive here, output in message_delta
                    message = data.get("message")
                    usage = message.get("usage") if isinstance(message, dict) else None
                if isinstance(usage, dict):
                    usage = {key: value for key, value in usage.items() if value is not None}  # null never erases a count
                    if usage:
                        self.usage = {**(self.usage or {}), **usage} if event == "message_delta" else usage
```

OpenAI-shaped chunks have no `type` key, so they keep the old replace-on-each-chunk behaviour. Dropping `None` values applies to both branches (Review I2): a `message_delta` with `"input_tokens": null` keeps the `message_start` count, and an all-null usage chunk no longer replaces a usage block already read.

2. Replace line 103, the last line of `_usage_tokens`:

```python
    return _first("prompt_tokens", "input_tokens"), _first("completion_tokens", "output_tokens")
```

with:

```python
    prompt_tokens = _first("prompt_tokens", "input_tokens")
    if "prompt_tokens" not in usage:
        # Anthropic-shaped block: input_tokens is only the uncached remainder (Anthropic, llama.cpp /v1/messages).
        prompt_tokens += _first("cache_creation_input_tokens") + _first("cache_read_input_tokens")
    return prompt_tokens, _first("completion_tokens", "output_tokens")
```

This is D1's recommendation (Review B1). The test is the absence of `prompt_tokens`, not the provider name. A block that has `prompt_tokens` already counts cached tokens in it, OpenAI's way, even when it also carries `cache_read_input_tokens` as some proxies do, so it is left alone. `_first` reads a missing, null or non-finite field as 0, so a null cache field adds nothing.

- [ ] **Step 4: Add the two public wrappers**

In the same file, change line 17 `from dataclasses import dataclass` to:

```python
from dataclasses import dataclass
from types import SimpleNamespace
```

and insert immediately above `_WRAPPERS: dict[str, Callable[[Callable[..., Any]], Callable[..., Any]]] = {`:

```python
def record_direct_response(provider: str, request: dict[str, Any], response: Any, started: float) -> None:
    """Record a successful provider call made without an adapter (the Messages API native path); never raises."""
    plan = _plan_call()
    if plan is not None:
        _record_response(SimpleNamespace(name=provider), plan, request, response, started)


def accounted_direct_stream(
    provider: str, request: dict[str, Any], stream: AsyncIterator[Any], started: float
) -> AsyncIterator[Any]:
    """Return ``stream`` wrapped to record once when it ends, or unchanged when nothing is recorded.

    For a provider stream made without an adapter. Each item must hold whole SSE events.
    """
    plan = _plan_call()
    if plan is None:
        return stream
    return _accounted_async_stream(SimpleNamespace(name=provider), plan, request, stream, started)
```

- [ ] **Step 5: Wire the native path**

In `tldw_Server_API/app/api/v1/endpoints/messages.py`:

1. Line 9: change `import os` to:

```python
import os
import time
```

2. After line 58 (`from tldw_Server_API.app.core.LLM_Calls.provider_metadata import provider_requires_api_key`) add:

```python
from tldw_Server_API.app.core.LLM_Calls.usage_accounting import accounted_direct_stream, record_direct_response
```

3. Replace the native stream block (lines 2052-2069):

```python
            stream = _extract_stream_flag(request_data.stream)
            if stream:
                stream_iter = await _prepare_native_stream_iterator(
                    url,
                    headers,
                    payload,
                    timeout=timeout,
                    provider=provider,
                    operation="messages.stream",
                )
                response = StreamingResponse(
                    _touch_on_first_stream_output(
                        _sanitize_native_messages_stream(stream_iter),
                        credential_runtime,
                        credentials,
                    ),
                    media_type="text/event-stream",
                )
```

with:

```python
            stream = _extract_stream_flag(request_data.stream)
            started = time.monotonic()
            if stream:
                stream_iter = await _prepare_native_stream_iterator(
                    url,
                    headers,
                    payload,
                    timeout=timeout,
                    provider=provider,
                    operation="messages.stream",
                )
                # This path never reaches a ChatProvider, so it records through the provider recorder (ADR-075).
                response = StreamingResponse(
                    accounted_direct_stream(
                        provider,
                        payload,
                        _touch_on_first_stream_output(
                            _sanitize_native_messages_stream(stream_iter),
                            credential_runtime,
                            credentials,
                        ),
                        started,
                    ),
                    media_type="text/event-stream",
                )
```

The wrapper goes outermost. The sanitizer re-frames the raw bytes into whole SSE events, so the reader sees whole frames. `_touch_on_first_stream_output` still sees the sanitizer's `defer_usage_until_terminal` attribute. Closing the response closes the wrapper, which closes the inner generator first, so the existing close order (source, then credential runtime) is unchanged.

4. Replace lines 2080-2082:

```python
            if not _native_message_payload_is_semantic(data):
                raise_detached_error(_messages_upstream_http_exception())
            response = JSONResponse(data)
```

with:

```python
            if not _native_message_payload_is_semantic(data):
                raise_detached_error(_messages_upstream_http_exception())
            record_direct_response(provider, payload, data, started)
            response = JSONResponse(data)
```

A row is written only for a valid reply. This matches ADR-067 ("Non-streaming calls are recorded only when they succeed") and the credential touch on the next line.

- [ ] **Step 6: Run the new tests and see them pass**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Chat/unit/test_messages_native_usage_recording.py`
Expected: `11 passed` (re-measured). With the first draft's Step 3 instead (a plain merge, and `input_tokens` alone), the prototype measured `4 failed, 7 passed`: `assert (0, 9) == (17, 9)` for the null-field `message_delta`, and `assert (3, 100) == (62003, 100)`, `(3, 100) == (60003, 100)` and `(3, 100) == (62003, 100)` for the three cached cases. So the new tests catch what Reviews I2 and B1 found.

- [ ] **Step 7: Run the suites that share this code**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Chat/unit/test_messages_endpoint_usage.py tldw_Server_API/tests/Chat/unit/test_messages_endpoint_native_errors.py tldw_Server_API/tests/Chat/integration/test_messages_endpoint.py tldw_Server_API/tests/Chat/integration/test_messages_provider_override_store_boundary.py tldw_Server_API/tests/Chat/unit/test_messages_default_provider_truthiness.py tldw_Server_API/tests/LLM_Local/test_llamacpp_messages_helpers.py tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py tldw_Server_API/tests/Usage/`
Expected: 0 failed. The prototype measured `610 passed, 1 skipped` on this set, before and after the review fixes (re-measured). The existing Messages tests set no user, so `accounted_direct_stream` hands back their streams unchanged.

`_usage_tokens` and `_StreamUsage.feed` run for every recorded provider call, so run every other test that names them or the cache fields (`git grep -lE "_usage_tokens|_StreamUsage|usage_accounting|cache_read_input_tokens|cache_creation_input_tokens" -- tldw_Server_API/tests` minus the files above and Step 6's):

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Chat/integration/test_chat_endpoint_simplified.py tldw_Server_API/tests/Admin/test_admin_usage_service.py tldw_Server_API/tests/Admin/test_llm_usage_endpoints.py tldw_Server_API/tests/AuthNZ_Postgres/test_authnz_llm_usage_log_router_columns_pg.py tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_llm_usage_log_router_columns_sqlite.py tldw_Server_API/tests/AuthNZ/unit/test_admin_usage_service_backend_selection.py tldw_Server_API/tests/LLM_Calls/test_local_cache_diagnostics.py`
Expected: 0 failed. The prototype measured `233 passed, 1 skipped` in about 8 minutes (the skip is the Postgres file, with no Postgres).

- [ ] **Step 8: Commit**

```bash
git add tldw_Server_API/app/core/LLM_Calls/usage_accounting.py tldw_Server_API/app/api/v1/endpoints/messages.py tldw_Server_API/tests/Chat/unit/test_messages_native_usage_recording.py
git commit -m "$(cat <<'EOF'
feat(usage): record the Messages API native path through the provider recorder (TASK-13610)

Anthropic and llama.cpp Messages calls go straight to the provider over
HTTP, so the ChatProvider base class never saw them. They now record one
llm_usage_log row through the same recorder: the reply's usage block when
not streaming, and message_start plus the last message_delta when
streaming. The stream reader learns the Anthropic event shape, and a null
usage field no longer erases a count. An Anthropic-shaped usage block
counts cache reads and writes as prompt tokens, because its input_tokens
is only the uncached remainder (Anthropic and llama.cpp alike).

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: An MCP tool call records and checks its LLM calls for the caller (TASK-13585 AC#1)

**Files:**
- Modify: `tldw_Server_API/app/core/MCP_unified/execution_outcomes.py:38-42` (add one catalog reason)
- Modify: `tldw_Server_API/app/core/MCP_unified/tool_execution/runtime.py:514-519`
- Modify: `tldw_Server_API/app/core/MCP_unified/tests/test_tool_execution_outcomes.py:50-55` (the catalog contract)
- Create: `tldw_Server_API/app/core/MCP_unified/tests/test_tool_llm_usage_attribution.py`

**Interfaces:**
- Consumes: `llm_call_context(**overrides)` (`Usage/llm_call_context.py:53-60`); `as_quota_user_id(value) -> int | None` (`Usage/quota_checks.py:50-55`); `LLMQuotaExceededError` (`core/exceptions.py:2075`); `refuse_llm_calls(monkeypatch)` (`tests/helpers/llm_quota.py`).
- Produces: `ExpectedToolFailureReason.LLM_QUOTA_EXCEEDED`, a tool error with `reason_code` `"llm_quota_exceeded"` and `BreakerAction.IGNORE`.

- [ ] **Step 1: Extend the catalog contract test**

In `tldw_Server_API/app/core/MCP_unified/tests/test_tool_execution_outcomes.py`, replace lines 50-55:

```python
        "DEPENDENCY_UNAVAILABLE": (
            "dependency_unavailable",
            "A required tool dependency is temporarily unavailable.",
            outcomes.BreakerAction.RECORD_FAILURE,
        ),
    }
```

with:

```python
        "DEPENDENCY_UNAVAILABLE": (
            "dependency_unavailable",
            "A required tool dependency is temporarily unavailable.",
            outcomes.BreakerAction.RECORD_FAILURE,
        ),
        "LLM_QUOTA_EXCEEDED": (
            "llm_quota_exceeded",
            "The LLM usage limit for this account has been reached.",
            outcomes.BreakerAction.IGNORE,
        ),
    }
```

- [ ] **Step 2: Write the failing attribution tests**

Create `tldw_Server_API/app/core/MCP_unified/tests/test_tool_llm_usage_attribution.py`. Its last test is D2's (Review I3):

```python
"""An MCP tool's LLM calls are recorded and quota-checked for the calling user (TASK-13585, ADR-067)."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

import pytest

from tldw_Server_API.app.core.LLM_Calls.providers.base import ChatProvider
from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleConfig
from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, RequestContext
from tldw_Server_API.app.core.Usage import llm_accounting
from tldw_Server_API.app.core.Usage.llm_call_context import llm_call_context
from tldw_Server_API.tests.helpers.llm_quota import refuse_llm_calls

pytestmark = pytest.mark.unit

TOOL = "llmtool.ask"


class _Provider(ChatProvider):
    """A provider with a native async chat that reports usage."""

    name = "mcp-fake"
    calls = 0

    def capabilities(self) -> dict[str, Any]:
        """Nothing special."""
        return {}

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Unused."""
        raise NotImplementedError

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
        """Unused."""
        return iter(())

    async def achat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Count the call and answer with usage."""
        type(self).calls += 1
        return {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 6, "completion_tokens": 2}}


class _LLMToolModule(BaseModule):
    """A module whose one tool makes an LLM call, as quizzes.generate does."""

    async def on_initialize(self) -> None:
        """Nothing to set up."""

    async def on_shutdown(self) -> None:
        """Nothing to tear down."""

    async def check_health(self) -> dict[str, bool]:
        """Always healthy."""
        return {"ok": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        """One read tool."""
        return [{"name": TOOL, "description": "", "inputSchema": {"type": "object"}, "metadata": {"category": "read"}}]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any | None = None) -> Any:
        """Ask the provider once."""
        response = await _Provider().achat({"model": "m", "messages": [{"role": "user", "content": "q"}]})
        return {"answer": response["choices"][0]["message"]["content"]}


class _AllowAllRBAC:
    """Every permission is granted."""

    async def check_permission(self, *_args: Any, **_kwargs: Any) -> bool:
        """Allow."""
        return True


class _Registry:
    """Routes every tool to one module."""

    def __init__(self, module: BaseModule) -> None:
        """Hold the module."""
        self._module = module

    async def find_module_for_tool(self, _tool_name: str) -> BaseModule:
        """The one module."""
        return self._module

    def get_module_id_for_tool(self, _tool_name: str) -> str:
        """Its name."""
        return self._module.name


@pytest.fixture
async def rows(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture what would be written, and make the test's loop the accounting loop."""
    captured: list[dict[str, Any]] = []

    async def _fake_log(**kwargs: Any) -> None:
        """Record the row instead of writing it."""
        captured.append(kwargs)

    monkeypatch.setattr("tldw_Server_API.app.core.Usage.usage_tracker.log_llm_usage", _fake_log)
    monkeypatch.setattr(llm_accounting, "_accounting_loop", asyncio.get_running_loop())
    monkeypatch.setattr(_Provider, "calls", 0)
    return captured


def _protocol(module: BaseModule) -> MCPProtocol:
    """A protocol that routes tools/call to ``module``."""
    proto = MCPProtocol()
    proto.rbac_policy = _AllowAllRBAC()
    proto.module_registry = _Registry(module)
    return proto


async def _call_tool(proto: MCPProtocol, user_id: str | None = "7") -> dict[str, Any]:
    """One tools/call, as the MCP HTTP and WebSocket transports make it: no content scope, user in the context."""
    context = RequestContext(request_id="usage-1", user_id=user_id, metadata={})
    return await proto._handle_tools_call({"name": TOOL, "arguments": {}}, context)


async def test_a_tool_call_records_one_row_for_the_calling_user(rows) -> None:
    """The MCP user is the user the tool's LLM call is recorded for."""
    result = await _call_tool(_protocol(_LLMToolModule(ModuleConfig(name="llmtool"))))
    assert result["content"][0]["json"]["answer"] == "ok"
    await llm_accounting.drain_accounting()
    (row,) = rows
    assert (row["user_id"], row["provider"], row["prompt_tokens"], row["completion_tokens"]) == (7, "mcp-fake", 6, 2)
    assert (row["operation"], row["endpoint"]) == ("llm_call", "internal")


async def test_concurrent_tool_calls_are_recorded_for_their_own_users(rows) -> None:
    """Two users' tool calls in flight at once each record for their own user."""
    proto = _protocol(_LLMToolModule(ModuleConfig(name="llmtool")))
    await asyncio.gather(_call_tool(proto, "7"), _call_tool(proto, "8"), _call_tool(proto, "7"))
    await llm_accounting.drain_accounting()
    assert sorted(row["user_id"] for row in rows) == [7, 7, 8]


async def test_a_tool_call_with_no_numeric_user_records_nothing(rows) -> None:
    """A service principal is not a user: the call runs and nothing is written."""
    result = await _call_tool(_protocol(_LLMToolModule(ModuleConfig(name="llmtool"))), user_id="svc-bot")
    assert result["content"][0]["json"]["answer"] == "ok"
    await llm_accounting.drain_accounting()
    assert rows == []


async def test_an_over_quota_user_gets_a_tool_error_that_does_not_trip_the_module_breaker(rows, monkeypatch) -> None:
    """A refusal is the user's limit, not a module outage: more refusals than the breaker threshold leave it closed."""
    refuse_llm_calls(monkeypatch)
    module = _LLMToolModule(ModuleConfig(name="llmtool", circuit_breaker_threshold=2))
    proto = _protocol(module)
    for _ in range(3):
        result = await _call_tool(proto)
        assert result["isError"] is True
        assert result["content"][0]["json"]["reason_code"] == "llm_quota_exceeded"
    assert module.is_circuit_breaker_open() is False
    assert _Provider.calls == 0


async def test_a_tool_call_inside_an_admitted_request_is_not_refused(rows, monkeypatch) -> None:
    """A tool run by an admitted /chat/completions request (tool auto-exec) keeps the request's admission (spec §4a)."""
    refuse_llm_calls(monkeypatch)
    with llm_call_context(user_id=7, prechecked=True):
        result = await _call_tool(_protocol(_LLMToolModule(ModuleConfig(name="llmtool"))))
    assert result["content"][0]["json"]["answer"] == "ok"
    assert _Provider.calls == 1


async def test_a_tool_run_by_chat_tool_auto_exec_is_recorded_for_the_user(rows, monkeypatch) -> None:
    """/chat/completions records only its own model call, so a tool it runs records the tool's LLM call."""
    refuse_llm_calls(monkeypatch)
    with llm_call_context(user_id=7, prechecked=True, records_itself=True):  # what chat.py sets around the pipeline
        result = await _call_tool(_protocol(_LLMToolModule(ModuleConfig(name="llmtool"))))
    assert result["content"][0]["json"]["answer"] == "ok"
    await llm_accounting.drain_accounting()
    (row,) = rows
    assert (row["user_id"], row["prompt_tokens"], row["completion_tokens"]) == (7, 6, 2)
```

- [ ] **Step 3: Run them and see them fail**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/app/core/MCP_unified/tests/test_tool_llm_usage_attribution.py tldw_Server_API/app/core/MCP_unified/tests/test_tool_execution_outcomes.py`
Expected: `5 failed, 6 passed` (re-measured):
- `test_a_tool_call_records_one_row_for_the_calling_user` and `test_concurrent_tool_calls_are_recorded_for_their_own_users` fail because no rows are written (`not enough values to unpack`, `[] == [7, 7, 8]`).
- `test_an_over_quota_user_gets_...` fails with `KeyError: 'isError'`: with no user, nothing is checked and the tool succeeds.
- `test_a_tool_run_by_chat_tool_auto_exec_is_recorded_for_the_user` fails with `not enough values to unpack`: under `records_itself=True` the tool's call is not recorded (Review I3).
- `test_expected_tool_failure_reason_catalog_is_fixed_and_bounded` fails because `LLM_QUOTA_EXCEEDED` is missing.
- The no-numeric-user and admitted-request tests are guards and pass.

- [ ] **Step 4: Add the catalog reason**

In `tldw_Server_API/app/core/MCP_unified/execution_outcomes.py`, after the `DEPENDENCY_UNAVAILABLE` member (lines 38-42) add:

```python
    LLM_QUOTA_EXCEEDED = (
        "llm_quota_exceeded",
        "The LLM usage limit for this account has been reached.",
        BreakerAction.IGNORE,
    )
```

- [ ] **Step 5: Name the caller and translate a refusal**

In `tldw_Server_API/app/core/MCP_unified/tool_execution/runtime.py`, replace lines 514-519:

```python
                        module_invoked = True
                        return await module.execute_tool(
                            tool_name,
                            execution_args,
                            context,
                        )
```

with:

```python
                        module_invoked = True
                        from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
                        from tldw_Server_API.app.core.Usage.llm_call_context import llm_call_context
                        from tldw_Server_API.app.core.Usage.quota_checks import as_quota_user_id

                        # MCP authentication sets no content scope, so name the caller for the
                        # tool's LLM calls (ADR-067). llm_call_context, not llm_call_root: an admitted
                        # request that runs this tool keeps its admission (LLM usage accounting spec §4a).
                        # records_itself is cleared: a caller that records itself (the chat pipeline)
                        # records only its own chat call, not this tool's calls (ADR-075).
                        llm_user = as_quota_user_id(context.user_id)
                        user = {} if llm_user is None else {"user_id": llm_user}
                        with llm_call_context(records_itself=False, **user):
                            try:
                                return await module.execute_tool(
                                    tool_name,
                                    execution_args,
                                    context,
                                )
                            except LLMQuotaExceededError:
                                # One user's limit, not a module fault: the breaker ignores it (spec §4b).
                                raise ExpectedToolFailure(
                                    ExpectedToolFailureReason.LLM_QUOTA_EXCEEDED,
                                ) from None
```

The imports are local, as in `tool_execution/security.py:1574,1645`, so `protocol.py`'s import-time graph does not grow. `ExpectedToolFailure` and `ExpectedToolFailureReason` are already imported (`runtime.py:14-18`). `llm_call_root` is not used because it sets `prechecked=False`. That would let §4b refuse a tool that an admitted `/chat/completions` request runs; the guard test in Step 2 pins this. `records_itself=False` is D2's recommendation (Review I3).

Keep the words "model", "provider" and "skills" out of this block, comments included. `test_rate_limit_categories.py::test_runtime_admission_has_no_domain_or_concrete_tool_name_branch` fails if `ToolExecutionRuntime.execute_prepared_tool_call`'s source contains any of them. A first wording of the comment above said "model call" and failed it.

**If the owner declines D2 (Review I3):** the tools that chat tool auto-execution runs stay unrecorded and unchecked, as ADR-067 says today.
- In Step 5's block, replace everything from the line `# MCP authentication sets no content scope, ...` through the line `with llm_call_context(records_itself=False, **user):` with these lines, the first draft's:

```python
                        # MCP authentication sets no content scope, so name the caller for the
                        # tool's LLM calls (ADR-067). Only the user: an admitted request that runs
                        # this tool keeps its admission (LLM usage accounting spec §4a).
                        llm_user = as_quota_user_id(context.user_id)
                        with llm_call_context(**({} if llm_user is None else {"user_id": llm_user})):
```

- In Step 2's file, replace the last test with this guard, which passes before and after:

```python
async def test_a_tool_run_by_chat_tool_auto_exec_is_not_recorded(rows, monkeypatch) -> None:
    """/chat/completions marks its pipeline records_itself, so a tool it runs is not recorded (ADR-067)."""
    refuse_llm_calls(monkeypatch)
    with llm_call_context(user_id=7, prechecked=True, records_itself=True):  # what chat.py sets around the pipeline
        result = await _call_tool(_protocol(_LLMToolModule(ModuleConfig(name="llmtool"))))
    assert result["content"][0]["json"]["answer"] == "ok"
    await llm_accounting.drain_accounting()
    assert rows == []
```

- Step 3 then expects `4 failed, 7 passed`, and Step 6 `11 passed`. The review dry-ran the first draft's overlay under this context and saw 1 provider call and 0 rows. Its Step 7 run of the first draft (this overlay, five new tests) measured `4379 passed, 16 skipped`; the guard adds one test.
- Task 5 lists the matching doc and backlog wording under "If the owner declines D2".

- [ ] **Step 6: Run the tests and see them pass**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/app/core/MCP_unified/tests/test_tool_llm_usage_attribution.py tldw_Server_API/app/core/MCP_unified/tests/test_tool_execution_outcomes.py`
Expected: `11 passed` (re-measured). The new file has `6 passed`.

- [ ] **Step 7: Run every MCP suite**

Run: `TLDW_TEST_NO_DOCKER=1 OTEL_SDK_DISABLED=true $PY -m pytest -p no:cacheprovider -q tldw_Server_API/app/core/MCP_unified/tests/ tldw_Server_API/tests/MCP_unified/ tldw_Server_API/tests/MCP/`
Expected: `4380 passed, 16 skipped` (re-measured, about 8 minutes). The review measured `4379 passed, 16 skipped` on the first draft; Review I3's test adds one. `test_extraction_contracts.py` must stay green: it scans `runtime.py` for unsafe exception logging and mutable policy reads. So must `test_rate_limit_categories.py` (see Step 5).

The overlay now clears `records_itself`, so also run every test outside these suites that names it or chat tool auto-execution (`git grep -lE "records_itself|execute_assistant_tool_calls|tool_auto_exec|ToolExecutor" -- tldw_Server_API/tests`, minus the files under `tests/Chat/unit`, `tests/LLM_Calls`, `tests/Usage` and `tests/Claims`, which Task 4 Step 7 and Task 5 Step 10 run):

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Agent_Client_Protocol/test_tool_executor.py tldw_Server_API/tests/AuthNZ_Unit/test_tools_permissions_claims.py tldw_Server_API/tests/Character_Chat_NEW/unit/test_chat_completion_precheck.py tldw_Server_API/tests/Chat/integration/test_chat_completions_marks_self_recorded.py tldw_Server_API/tests/Chat/integration/test_chat_endpoint_auto_routing.py tldw_Server_API/tests/Chat/integration/test_chat_endpoint_streaming_normalization.py tldw_Server_API/tests/Notifications/test_agent_task_jobs_consumer.py tldw_Server_API/tests/Notifications/test_automation_executors.py tldw_Server_API/tests/Setup/test_first_run_mcp_tools_service.py tldw_Server_API/tests/Skills/unit/test_skill_executor.py tldw_Server_API/tests/Tools/test_tools_endpoint_error_logs.py tldw_Server_API/tests/Tools/test_tools_routes.py`
Expected: `215 passed` (re-measured, about 4 minutes).

- [ ] **Step 8: Commit**

```bash
git add tldw_Server_API/app/core/MCP_unified/execution_outcomes.py tldw_Server_API/app/core/MCP_unified/tool_execution/runtime.py tldw_Server_API/app/core/MCP_unified/tests/test_tool_execution_outcomes.py tldw_Server_API/app/core/MCP_unified/tests/test_tool_llm_usage_attribution.py
git commit -m "$(cat <<'EOF'
feat(usage): MCP tool calls record and check LLM usage for the caller (TASK-13585)

MCP authentication sets no content scope, so a tool's LLM calls had no
user. The tool runtime now names the caller from the RequestContext
around module.execute_tool, with llm_call_context rather than a root, so
a tool run by an admitted /chat/completions request keeps its admission.
It also clears records_itself: the chat pipeline records only its own
chat call, so a tool it runs now records its own calls (ADR-075). A
quota refusal becomes the catalogued llm_quota_exceeded tool error,
which the module circuit breaker ignores; until now, a refusal on
/api/v1/tools/execute counted as a module failure for everyone.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: The claims rebuild worker survives a quota refusal (prerequisite of Task 4)

Task 4 gives rebuild tasks an owner, and from then on their LLM calls can be refused. Today `_worker_loop` would let that refusal end the only worker thread. This task lands first.

**Files:**
- Modify: `tldw_Server_API/app/core/Claims_Extraction/claims_rebuild_service.py:25` (import), `:256-264` (the `except` block, Review M3)
- Create: `tldw_Server_API/tests/Claims/test_claims_rebuild_llm_owner.py`. Its first test also checks the stored failure reason (Review M3).

**Interfaces:**
- Consumes: `LLMQuotaExceededError(*, category, current, limit, message, retry_after)` (`core/exceptions.py:2087`).
- Produces: the test module's `rows` fixture and `_until(predicate, timeout=5.0)` helper. Task 4 appends tests to this module.

- [ ] **Step 1: Write the failing test, and the claims-kept guard**

Create `tldw_Server_API/tests/Claims/test_claims_rebuild_llm_owner.py`:

```python
"""In-process claims rebuild tasks carry their owner, and their LLM calls are recorded for that owner (TASK-13585)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from tldw_Server_API.app.core.Claims_Extraction import claims_rebuild_service
from tldw_Server_API.app.core.Claims_Extraction.claims_rebuild_service import ClaimsRebuildService, ClaimsRebuildTask
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
from tldw_Server_API.app.core.Usage import llm_accounting

pytestmark = pytest.mark.unit


@pytest.fixture
async def rows(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture what would be written, make the test's loop the accounting loop, and skip health writes."""
    captured: list[dict[str, Any]] = []

    async def _fake_log(**kwargs: Any) -> None:
        """Record the row instead of writing it."""
        captured.append(kwargs)

    monkeypatch.setattr("tldw_Server_API.app.core.Usage.usage_tracker.log_llm_usage", _fake_log)
    monkeypatch.setattr(llm_accounting, "_accounting_loop", asyncio.get_running_loop())
    monkeypatch.setattr(ClaimsRebuildService, "_persist_health", lambda self, force=False: None)
    return captured


async def _until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    """Yield to the loop until ``predicate`` holds or the timeout passes."""
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        await asyncio.sleep(0.01)


async def test_a_quota_refusal_fails_the_task_and_the_worker_keeps_going(rows, monkeypatch) -> None:
    """A refused LLM call is one failed task, not a dead worker thread that strands every later task."""
    seen: list[int] = []

    def _fake_rebuild(*, db_path: str, media_id: int) -> dict[str, Any]:
        """Refuse the first media item, rebuild the second."""
        seen.append(media_id)
        if media_id == 1:
            raise LLMQuotaExceededError(
                category="llm_tokens_month", current=10, limit=10,
                message="Monthly LLM token limit reached", retry_after=60,
            )
        return {"outcome": "ok", "media_id": media_id}

    monkeypatch.setattr(claims_rebuild_service, "rebuild_claims_for_media", _fake_rebuild)
    svc = ClaimsRebuildService(worker_threads=1)
    svc.start()
    try:
        svc.submit(media_id=1, db_path=":memory:")
        svc.submit(media_id=2, db_path=":memory:")
        await _until(lambda: svc.get_stats()["processed"] == 1)
    finally:
        svc.stop()
    stats = svc.get_stats()
    assert (stats["failed"], stats["processed"], seen) == (1, 1, [1, 2])
    last_failure = svc.get_health()["last_failure"]
    assert (last_failure["media_id"], last_failure["error"]) == (1, "llm_quota_exceeded")


def test_a_refusal_during_extraction_leaves_the_media_claims_in_place(monkeypatch) -> None:
    """Extraction runs before the old claims are replaced, so a refused rebuild deletes nothing."""
    deleted: list[int] = []

    class _FakeDb:
        """A media DB with one item whose claims must survive."""

        def get_media_by_id(self, media_id: int, include_deleted: bool = False, include_trash: bool = False) -> dict:
            """The media item."""
            return {"id": media_id, "title": "Doc", "content": "A sentence."}

        def soft_delete_claims_for_media(self, media_id: int) -> int:
            """Record a deletion."""
            deleted.append(media_id)
            return 1

    @contextmanager
    def _fake_managed_media_database(*_args: Any, **_kwargs: Any) -> Iterator[_FakeDb]:
        """Hand out the fake DB."""
        yield _FakeDb()

    def _refuse(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        """Extraction's LLM call is refused."""
        raise LLMQuotaExceededError(
            category="llm_tokens_month", current=10, limit=10,
            message="Monthly LLM token limit reached", retry_after=60,
        )

    monkeypatch.setattr(claims_rebuild_service, "managed_media_database", _fake_managed_media_database)
    monkeypatch.setattr(claims_rebuild_service, "chunk_for_embedding", lambda content, file_name: [{"text": content}])
    monkeypatch.setattr(claims_rebuild_service, "resolve_claims_job_budget", lambda settings: None)
    monkeypatch.setattr(claims_rebuild_service, "extract_claims_for_chunks", _refuse)
    with pytest.raises(LLMQuotaExceededError):
        ClaimsRebuildService(worker_threads=1)._process_task(ClaimsRebuildTask(media_id=7, db_path="/tmp/m.db"))
    assert deleted == []
```

- [ ] **Step 2: Run it and see it fail**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Claims/test_claims_rebuild_llm_owner.py`
Expected: `1 failed, 1 passed` (re-measured), after about 5 s. `test_a_quota_refusal_fails_the_task_and_the_worker_keeps_going` fails with `assert (0, 0, [1]) == (1, 1, [1, 2])`: the worker thread died on task 1 and task 2 never ran. The claims-kept guard passes.

- [ ] **Step 3: Catch the refusal as a task failure**

In `tldw_Server_API/app/core/Claims_Extraction/claims_rebuild_service.py`, after line 25 (`from tldw_Server_API.app.core.DB_Management.media_db.api import managed_media_database`) add:

```python
from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError
```

and replace lines 256-264:

```python
            except _CLAIMS_REBUILD_NONCRITICAL_EXCEPTIONS as e:
                logger.error(f"Claims rebuild failed for media_id={task.media_id}: {e}")
                with self._stats_lock:
                    self._stats["failed"] += 1
                self._last_failure = {
                    "media_id": task.media_id,
                    "error": str(e),
                    "timestamp": time.time(),
                }
```

with:

```python
            # A quota refusal fails this task only; uncaught, it would end the worker thread.
            except (*_CLAIMS_REBUILD_NONCRITICAL_EXCEPTIONS, LLMQuotaExceededError) as e:
                if isinstance(e, LLMQuotaExceededError):
                    # The owner's limit, not a fault: a warning, and no quota details in the shared health row.
                    error = "llm_quota_exceeded"
                    logger.warning(f"Claims rebuild refused for media_id={task.media_id}: {error}")
                else:
                    error = str(e)
                    logger.error(f"Claims rebuild failed for media_id={task.media_id}: {e}")
                with self._stats_lock:
                    self._stats["failed"] += 1
                self._last_failure = {
                    "media_id": task.media_id,
                    "error": error,
                    "timestamp": time.time(),
                }
```

`_CLAIMS_REBUILD_NONCRITICAL_EXCEPTIONS` stays unchanged, because `stop()` and the health writes also use it.

A refusal is an expected user limit, and `str(LLMQuotaExceededError)` carries the user's quota detail (`402: {'error': 'limit_exceeded', 'current': ..., 'limit': ...}`). So it logs at warning, and the stored reason is the code `llm_quota_exceeded`. `_persist_health(force=True)` writes that reason into the claims-monitoring system user's health row as `last_failure_reason` (`:260-270`), not the detail (Review M3). The review suggested a separate `except LLMQuotaExceededError` block. This branch inside the shared `except` has the same behaviour without copying its stats, metrics and health lines.

- [ ] **Step 4: Run it and see it pass**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Claims/test_claims_rebuild_llm_owner.py tldw_Server_API/tests/Claims/test_claims_rebuild_service_failure.py`
Expected: `10 passed` (re-measured).

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Claims_Extraction/claims_rebuild_service.py tldw_Server_API/tests/Claims/test_claims_rebuild_llm_owner.py
git commit -m "$(cat <<'EOF'
fix(claims): a quota refusal fails one rebuild task instead of ending the worker (TASK-13585)

The in-process rebuild worker caught only its noncritical exceptions, so
an LLMQuotaExceededError would end its only thread and strand the queue.
Rebuild tasks are about to carry an owner, which makes that refusal
possible. A refusal is the owner's limit, not a fault: it logs as a
warning, and the health row stores llm_quota_exceeded, not the quota
details.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Claims rebuild tasks carry their owner (TASK-13585 AC#2)

**Files:**
- Modify: `tldw_Server_API/app/core/Claims_Extraction/claims_rebuild_service.py:48-51`, `:176-177`, `:274-275`, and the import added in Task 3
- Modify: `tldw_Server_API/app/core/Claims_Extraction/claims_service.py:634`, `:4599`, `:4649`
- Modify: `tldw_Server_API/app/services/startup_claims_rebuild.py:146`, `:155`
- Modify (the fakes' signatures, and assertions that now include the owner):
  - `tldw_Server_API/tests/Claims/test_claims_rebuild_stale_policy.py:64,103,147,267`
  - `tldw_Server_API/tests/Claims/test_claims_service_override_db.py:180-183,214`
  - `tldw_Server_API/tests/Services/test_startup_claims_rebuild.py:100-104,129-132`
- Test: `tldw_Server_API/tests/Claims/test_claims_rebuild_llm_owner.py` (append)

**Interfaces:**
- Consumes: `llm_call_root(user_id: Any, *, operation: str | None = None)` (`Usage/llm_roots.py:54`); the `rows` fixture and `_until` from Task 3.
- Produces:
  - `ClaimsRebuildTask(media_id: int, db_path: str, owner_user_id: int | str | None = None)`.
  - `ClaimsRebuildService.submit(media_id: int, db_path: str, owner_user_id: int | str | None = None) -> None`. Every in-app caller passes `owner_user_id=`.

- [ ] **Step 1: Write the failing tests**

In `tldw_Server_API/tests/Claims/test_claims_rebuild_llm_owner.py`:

1. Replace `from collections.abc import Callable, Iterator` with `from collections.abc import Callable, Iterable, Iterator`.
2. Replace `from tldw_Server_API.app.core.Claims_Extraction import claims_rebuild_service` with `from tldw_Server_API.app.core.Claims_Extraction import claims_rebuild_service, claims_service`.
3. After `from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError` add `from tldw_Server_API.app.core.LLM_Calls.providers.base import ChatProvider`.
4. After `pytestmark = pytest.mark.unit` add:

```python


class _Provider(ChatProvider):
    """A sync provider, as the claims extractor's adapters are."""

    name = "claims-fake"

    def capabilities(self) -> dict[str, Any]:
        """Nothing special."""
        return {}

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Answer with usage."""
        return {"choices": [{"message": {"content": "[]"}}], "usage": {"prompt_tokens": 9, "completion_tokens": 3}}

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
        """Unused."""
        return iter(())
```

5. Append at the end of the file:

```python


async def test_a_rebuild_task_records_its_llm_calls_for_its_owner(rows, monkeypatch) -> None:
    """The task's owner is the user the rebuild's LLM call is recorded for."""

    def _fake_rebuild(*, db_path: str, media_id: int) -> dict[str, Any]:
        """Make one provider call, as LLM claim extraction does."""
        _Provider().chat({"model": "m", "messages": [{"role": "user", "content": "chunk"}]})
        return {"outcome": "ok", "media_id": media_id}

    monkeypatch.setattr(claims_rebuild_service, "rebuild_claims_for_media", _fake_rebuild)
    svc = ClaimsRebuildService(worker_threads=1)
    svc.start()
    try:
        svc.submit(media_id=3, db_path=":memory:", owner_user_id="5")
        await _until(lambda: svc.get_stats()["processed"] == 1)
    finally:
        svc.stop()
    await llm_accounting.drain_accounting()
    (row,) = rows
    assert (row["user_id"], row["provider"], row["prompt_tokens"], row["completion_tokens"]) == (5, "claims-fake", 9, 3)


def test_rebuild_claims_hands_the_owner_to_the_legacy_queue(monkeypatch) -> None:
    """With Jobs off, the per-media rebuild enqueues the task with the requesting user as owner."""
    submissions: list[tuple[int, str, object]] = []

    class _FakeSvc:
        """Collects submissions."""

        def submit(self, media_id: int, db_path: str, owner_user_id: object = None) -> None:
            """Record the call."""
            submissions.append((int(media_id), db_path, owner_user_id))

    class _User:
        """A plain user."""

        id = 5
        is_admin = False

    class _Db:
        """The user's media DB."""

        db_path_str = "/tmp/user-5/Media_DB_v2.db"

    monkeypatch.setattr(claims_service.claims_jobs, "claims_jobs_enabled", lambda: False)
    result = claims_service.rebuild_claims(
        media_id=42, user_id=None, current_user=_User(), db=_Db(), rebuild_service=_FakeSvc()
    )
    assert result == {"status": "accepted", "media_id": 42}
    assert submissions == [(42, "/tmp/user-5/Media_DB_v2.db", "5")]


def test_the_rebuild_helper_hands_the_owner_to_the_legacy_queue(monkeypatch) -> None:
    """A claim edit's follow-up rebuild keeps the owner it was given when it falls back to the legacy queue."""
    submissions: list[tuple[int, str, object]] = []

    class _FakeSvc:
        """Collects submissions."""

        def submit(self, media_id: int, db_path: str, owner_user_id: object = None) -> None:
            """Record the call."""
            submissions.append((int(media_id), db_path, owner_user_id))

    monkeypatch.setattr(claims_service.claims_jobs, "claims_jobs_enabled", lambda: False)
    monkeypatch.setattr(claims_service, "get_claims_rebuild_service", lambda: _FakeSvc())
    claims_service._enqueue_claim_rebuild_if_needed(media_id=42, db_path="/tmp/u.db", owner_user_id="5")
    assert submissions == [(42, "/tmp/u.db", "5")]
```

- [ ] **Step 2: Make the existing fakes record the owner**

These fakes take the call that now passes `owner_user_id=`. Two of them also assert it.

1. `tldw_Server_API/tests/Claims/test_claims_rebuild_stale_policy.py`: at lines 64, 103, 147 and 267, change each `def submit(self, media_id: int, db_path: str):` to:

```python
        def submit(self, media_id: int, db_path: str, owner_user_id: object = None):
```

2. `tldw_Server_API/tests/Claims/test_claims_service_override_db.py`: replace lines 180-183:

```python
            self.submissions: list[tuple[int, str]] = []

        def submit(self, *, media_id: int, db_path: str) -> None:
            self.submissions.append((media_id, db_path))
```

with:

```python
            self.submissions: list[tuple[int, str, object]] = []

        def submit(self, *, media_id: int, db_path: str, owner_user_id: object = None) -> None:
            self.submissions.append((media_id, db_path, owner_user_id))
```

and line 214:

```python
    assert svc.submissions == [(5, "/tmp/user-7.db"), (8, "/tmp/user-7.db")]
```

with:

```python
    assert svc.submissions == [(5, "/tmp/user-7.db", "7"), (8, "/tmp/user-7.db", "7")]
```

3. `tldw_Server_API/tests/Services/test_startup_claims_rebuild.py`: replace lines 100-104:

```python
    submissions: list[tuple[int, str]] = []

    class _FakeService:
        def submit(self, *, media_id: int, db_path: str) -> None:
            submissions.append((media_id, db_path))
```

with:

```python
    submissions: list[tuple[int, str, object]] = []

    class _FakeService:
        def submit(self, *, media_id: int, db_path: str, owner_user_id: object = None) -> None:
            submissions.append((media_id, db_path, owner_user_id))
```

and lines 129-132:

```python
    assert submissions == [
        (101, "/tmp/media.db"),
        (202, "/tmp/media.db"),
    ]
```

with:

```python
    assert submissions == [
        (101, "/tmp/media.db", 1),
        (202, "/tmp/media.db", 1),
    ]
```

- [ ] **Step 3: Run them and see them fail**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Claims/test_claims_rebuild_llm_owner.py tldw_Server_API/tests/Claims/test_claims_rebuild_stale_policy.py tldw_Server_API/tests/Claims/test_claims_service_override_db.py tldw_Server_API/tests/Services/test_startup_claims_rebuild.py`
Expected: `5 failed, 16 passed` (re-measured):
- `test_a_rebuild_task_records_its_llm_calls_for_its_owner`: `TypeError: ClaimsRebuildService.submit() got an unexpected keyword argument 'owner_user_id'`. The task carries no owner.
- `test_rebuild_claims_hands_the_owner_to_the_legacy_queue` and `test_the_rebuild_helper_hands_the_owner_to_the_legacy_queue`: the owner recorded is `None`, not `'5'`.
- `test_rebuild_all_media_uses_override_helper_for_sqlite_admin_override`: `None != '7'`.
- `test_run_claims_rebuild_iteration_submits_selected_media_ids`: `None != 1`.

- [ ] **Step 4: Carry the owner on the task and run under it**

In `tldw_Server_API/app/core/Claims_Extraction/claims_rebuild_service.py`:

1. After `from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError` (Task 3) add:

```python
from tldw_Server_API.app.core.Usage.llm_roots import llm_call_root
```

2. Replace the dataclass (lines 48-51):

```python
@dataclass
class ClaimsRebuildTask:
    media_id: int
    db_path: str
```

with:

```python
@dataclass
class ClaimsRebuildTask:
    media_id: int
    db_path: str
    # The user the rebuild's LLM calls are recorded and quota-checked for (ADR-067).
    owner_user_id: int | str | None = None
```

3. Replace `submit`'s first two lines (176-177):

```python
    def submit(self, media_id: int, db_path: str) -> None:
        self._queue.put_nowait(ClaimsRebuildTask(media_id=int(media_id), db_path=str(db_path)))
```

with:

```python
    def submit(self, media_id: int, db_path: str, owner_user_id: int | str | None = None) -> None:
        self._queue.put_nowait(
            ClaimsRebuildTask(media_id=int(media_id), db_path=str(db_path), owner_user_id=owner_user_id)
        )
```

4. Replace `_process_task` (274-275):

```python
    def _process_task(self, task: ClaimsRebuildTask) -> None:
        rebuild_claims_for_media(db_path=task.db_path, media_id=task.media_id)
```

with:

```python
    def _process_task(self, task: ClaimsRebuildTask) -> None:
        with llm_call_root(task.owner_user_id):
            rebuild_claims_for_media(db_path=task.db_path, media_id=task.media_id)
```

`llm_call_root` sets no user for `None` or a non-numeric owner (`llm_roots.py:62-67`). The stop sentinel `ClaimsRebuildTask(media_id=-1, db_path="")` and existing callers keep working. The claims extraction pool already carries the LLM call state into its thread (`ingestion_claims.py:422-423`).

- [ ] **Step 5: Pass the owner from every caller**

In `tldw_Server_API/app/core/Claims_Extraction/claims_service.py`:
- Line 634: `svc.submit(media_id=int(media_id), db_path=str(db_path))` becomes `svc.submit(media_id=int(media_id), db_path=str(db_path), owner_user_id=owner_user_id)`.
- Line 4599: `svc.submit(media_id=media_id, db_path=db_path)` becomes `svc.submit(media_id=media_id, db_path=db_path, owner_user_id=owner_user_id)`.
- Line 4649: `svc.submit(media_id=mid, db_path=db_path)` becomes `svc.submit(media_id=mid, db_path=db_path, owner_user_id=owner_user_id)`.

In `tldw_Server_API/app/services/startup_claims_rebuild.py`:
- Line 146: `with _claims_rebuild_db_session(app_settings) as (_, db_path, db):` becomes `with _claims_rebuild_db_session(app_settings) as (user_id, db_path, db):`.
- Line 155: `service.submit(media_id=media_id, db_path=db_path)` becomes `service.submit(media_id=media_id, db_path=db_path, owner_user_id=user_id)`.

The startup loop rebuilds the `SINGLE_USER_FIXED_ID` user's media DB (`startup_claims_rebuild.py:165-167`). That user owns the media, so its rebuild calls are attributed to that user.

- [ ] **Step 6: Run them and see them pass**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Claims/test_claims_rebuild_llm_owner.py tldw_Server_API/tests/Claims/test_claims_rebuild_stale_policy.py tldw_Server_API/tests/Claims/test_claims_service_override_db.py tldw_Server_API/tests/Services/test_startup_claims_rebuild.py`
Expected: `21 passed` (re-measured). The new file has `5 passed`.

- [ ] **Step 7: Run the claims suites**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Claims/ tldw_Server_API/tests/Services/test_startup_claims_rebuild.py`
Expected: `1082 passed, 1 skipped` (re-measured, with Review M3's change; the review measured the same on the first draft). The baseline on dev is `1077 passed, 1 skipped`, and the five new tests make 1082 (Review M5: the first draft said 1081). If `test_review_metrics_payload_inclusive_window_property` fails, rerun it alone (see Measured).

- [ ] **Step 8: Commit**

```bash
git add tldw_Server_API/app/core/Claims_Extraction/claims_rebuild_service.py tldw_Server_API/app/core/Claims_Extraction/claims_service.py tldw_Server_API/app/services/startup_claims_rebuild.py tldw_Server_API/tests/Claims/test_claims_rebuild_llm_owner.py tldw_Server_API/tests/Claims/test_claims_rebuild_stale_policy.py tldw_Server_API/tests/Claims/test_claims_service_override_db.py tldw_Server_API/tests/Services/test_startup_claims_rebuild.py
git commit -m "$(cat <<'EOF'
feat(usage): claims rebuild tasks carry their owner and run under its LLM root (TASK-13585)

The in-process rebuild's tasks held only a media id and a DB path, so its
LLM calls were neither recorded nor checked. Every caller already knows
the owner: the per-media and bulk rebuilds, a claim edit's follow-up, and
the startup loop. Each now passes it on, and the worker runs the task
under llm_call_root(owner).

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: ADR-075, docs, the follow-up task, backlog close-out and the PR

**Files:**
- Create: `Docs/ADR/075-native-messages-and-mcp-tools-use-the-provider-recorder.md` (Review I1) and its mirror `Docs/Published/ADR/075-native-messages-and-mcp-tools-use-the-provider-recorder.md` (via the refresh script)
- Modify: `Docs/ADR/README.md` (an index row after ADR-071) and its mirror `Docs/Published/ADR/README.md`
- Modify: `Docs/ADR/067-llm-calls-accounted-in-provider-base-class.md` (status lines in Consequences and Follow-up) and its mirror `Docs/Published/ADR/067-llm-calls-accounted-in-provider-base-class.md`
- Modify: `Docs/Design/2026-10-09-llm-usage-accounting-design.md` (§1, §3, §4b, Review changes)
- Modify: `Docs/Operations/Usage_Quotas.md:87-88,95,100-101`
- Modify: `tldw_Server_API/app/core/MCP_unified/README.md` (the Tool Execution Failure contract, after line 75)
- Create: one backlog task, the Anthropic cache-pricing follow-up (Review B1, via backlog-py)
- Modify: `backlog/tasks/task-13585 - ...md`, `backlog/tasks/task-13610 - ...md` (via backlog-py)

- [ ] **Step 1: Write ADR-075, index it, and update ADR-067's status lines (Review I1)**

Use the ADR number from Task 1 Step 0.

1. Create `Docs/ADR/075-native-messages-and-mcp-tools-use-the-provider-recorder.md`:

```markdown
# ADR-075: The Messages native path and MCP tools record through the provider recorder

**Status:** Proposed
**Date:** 2026-10-18
**Backfilled from:** not backfilled
**Decision owner:** Repository owner (@rmusser01)
**Related task:** TASK-13585, TASK-13610
**Related spec/plan:** `Docs/Design/2026-10-09-llm-usage-accounting-design.md` (§1, §3)
**Extends:** ADR-067 (does not supersede it)

## Decision

A provider call that no `ChatProvider` method makes, today only the Messages API native path, records its one `llm_usage_log` row through the provider recorder's public wrappers `record_direct_response` and `accounted_direct_stream` (`LLM_Calls/usage_accounting.py`), and an MCP tool's LLM calls are recorded for the tool's caller even when a caller that records itself runs the tool.

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

- The pricing catalog has no Anthropic cache read or write rates, so cache reads are billed at the full prompt rate. A follow-up task, filed with TASK-13610, adds them.
```

2. In `Docs/ADR/README.md`, add this row after the ADR-071 row:

```markdown
| [ADR-075](075-native-messages-and-mcp-tools-use-the-provider-recorder.md) | Proposed | A provider call no `ChatProvider` method makes (the Messages API native path) records through the provider recorder's public wrappers, and an MCP tool's LLM calls are recorded for its caller even under a caller that records itself. |
```

3. In `Docs/ADR/067-llm-calls-accounted-in-provider-base-class.md`, update the status lines only. The first draft also inserted a "One path calls a provider without an adapter" bullet before `- **Background roots.**`; that is a new decision, so it is dropped here and lives in ADR-075. Replace:

```markdown
- **Still unrecorded,** each with its reason:
  - MCP tool calls: MCP authentication sets no content scope.
  - The in-process claims rebuild threads: their tasks carry no owner.
  - System jobs with no user, such as the scheduled quality evaluation.
```

with:

```markdown
- **MCP tool calls** get their user from the tool call's context, which MCP authentication fills in, because it sets no content scope. A tool run by an admitted request keeps the admission (ADR-075).
- **Still unrecorded:** system jobs with no user, such as the scheduled quality evaluation.
```

4. In the same file, replace `- TASK-13585: record MCP tool calls and the in-process claims rebuild.` with:

```markdown
- TASK-13585 recorded MCP tool calls and the in-process claims rebuild. TASK-13610 recorded the Messages API native path (ADR-075).
```

5. Refresh the mirror and stage the two new files:

```bash
bash Helper_Scripts/refresh_docs_published.sh
git add Docs/ADR/075-native-messages-and-mcp-tools-use-the-provider-recorder.md Docs/Published/ADR/075-native-messages-and-mcp-tools-use-the-provider-recorder.md
git status --short Docs
```

Expected (re-measured):

```
 M Docs/ADR/067-llm-calls-accounted-in-provider-base-class.md
 M Docs/ADR/README.md
A  Docs/ADR/075-native-messages-and-mcp-tools-use-the-provider-recorder.md
 M Docs/Published/ADR/067-llm-calls-accounted-in-provider-base-class.md
 M Docs/Published/ADR/README.md
A  Docs/Published/ADR/075-native-messages-and-mcp-tools-use-the-provider-recorder.md
```

The new files must be staged before Step 5: `test_refresh_output_matches_tracked_published_files` compares the refreshed tree with `git ls-files Docs/Published`.

**If the owner declines D1:** in ADR-075, delete the Consequences bullet that starts `- **Anthropic-shaped usage counts cached input.**` and the whole Follow-up section.

**If the owner declines D2:** in ADR-075, rename the file to `075-native-messages-path-uses-the-provider-recorder.md` and the title to `ADR-075: The Messages native path records through the provider recorder`. Cut `, and an MCP tool's LLM calls are recorded for the tool's caller even when a caller that records itself runs the tool` from the Decision. Delete the Context from `ADR-067 also marks` to its end, the last Alternatives row, and the last Consequences bullet. Point the index row at the new file name and cut its clause from `, and an MCP tool's LLM calls`. In ADR-067's MCP bullet, write `Only the user is set, so a tool run by an admitted request keeps the admission.` in place of its last sentence.

- [ ] **Step 2: Update the design doc**

In `Docs/Design/2026-10-09-llm-usage-accounting-design.md`:

1. After the line `  - The other sockets (ACP, prompt studio, persona, meetings, sandbox, watchlists, workflows) make no chat LLM call and are left alone.` (end of §1's roots list) insert:

```markdown
- **MCP tool calls:** MCP authentication sets no content scope, so the tool runtime sets the user from the call's `RequestContext` around `module.execute_tool` (`MCP_unified/tool_execution/runtime.py`). It uses `llm_call_context`, not `llm_call_root`, so a tool that an admitted `/chat/completions` request runs keeps the request's `prechecked` mark. It also clears `records_itself`: the chat pipeline's own row covers only its model call, so the tool's calls are recorded (ADR-075).
- **The in-process claims rebuild:** each `ClaimsRebuildTask` carries its owner, and the worker runs it under `llm_call_root(owner)`. A quota refusal fails that task and the worker goes on.
```

2. After the §3 bullet that starts `- **Streaming:** tokens come from a usage chunk` insert:

```markdown
- **No adapter:** the Messages API native path (Anthropic, llama.cpp) posts to the provider itself, so no provider method runs. It records through the same functions, `record_direct_response` and `accounted_direct_stream` (ADR-075). An Anthropic stream's input tokens come from `message_start` and its output count from the last `message_delta`. An Anthropic-shaped usage block counts `cache_creation_input_tokens` and `cache_read_input_tokens` as prompt tokens, because its `input_tokens` is only the uncached remainder.
```

3. In §4b, replace `MCP tool calls carry no user, so they are neither recorded nor checked (§1).` with:

```markdown
An MCP tool call is checked for the user its call names (§1). A refusal becomes the tool error `llm_quota_exceeded`, which the MCP module circuit breaker does not count.
```

4. At the end of the file, append:

```markdown

Revision of 2026-10-18, TASK-13585 and TASK-13610:

- **MCP tool calls and the in-process claims rebuild** are recorded and checked (§1, §4b). The Messages API native path is recorded (§3). It needs no check at the call: `require_llm_quota` admits it first, so its calls run `prechecked`.
```

**If the owner declines D2:** in the §1 MCP bullet, replace its last two sentences with `It sets only the user, with llm_call_context, not llm_call_root: a tool that an admitted /chat/completions request runs keeps the request's prechecked mark, and stays unrecorded, as ADR-067 says.` (with the code spans as above). **If the owner declines D1:** drop the last sentence of the "No adapter" bullet.

- [ ] **Step 3: Update the operator page**

In `Docs/Operations/Usage_Quotas.md`:

1. Replace line 87:

```markdown
- Recorded: calls made while serving a request; background jobs, including audio analysis and scheduled agent tasks; workflow runs; watchlist output enrichment; audio and realtime WebSocket sessions.
```

with:

```markdown
- Recorded: calls made while serving a request, including Messages API calls to Anthropic or llama.cpp; background jobs, including audio analysis and scheduled agent tasks; workflow runs; watchlist output enrichment; audio and realtime WebSocket sessions; MCP tool calls, including tools that `/chat/completions` runs; the in-process claims rebuild.
```

2. Replace line 88:

```markdown
- Not recorded: MCP tool calls, the in-process claims rebuild, and system jobs with no user, such as the scheduled quality evaluation.
```

with:

```markdown
- Not recorded: system jobs with no user, such as the scheduled quality evaluation, and MCP calls from a caller with no user account.
```

3. After line 95 (the bullet that starts `  - **A job** whose worker reads the error's`) insert:

```markdown
  - **The in-process claims rebuild** (used when claims Jobs are off) fails that media item, keeps its current claims, and goes on to the next.
```

4. Replace line 100, `  - **MCP tools** are neither recorded nor checked: they carry no user.`, with (Reviews I3 and M7):

```markdown
  - **An MCP tool** returns a tool error with `reason_code` `llm_quota_exceeded`. On `/api/v1/tools/execute` that is HTTP 200 with `ok: true` and `result.isError: true`, not a 402. The module's circuit breaker does not count it, so other users keep the tool. A tool that an admitted `/chat/completions` request runs is not refused, and its LLM calls are recorded for the user.
```

5. Delete line 101, the bullet that starts `- **Not recorded today:** Messages API calls to Anthropic or llama.cpp`.

**If the owner declines D2** (the review's wording, Review I3): in item 1, end the line with `MCP tool calls; the in-process claims rebuild.`. In item 2, use:

```markdown
- Not recorded: system jobs with no user, such as the scheduled quality evaluation; MCP calls from a caller with no user account; and MCP tools run by `/chat/completions` tool auto-execution, which are neither recorded nor checked: the chat request's own row covers only its model call (ADR-067).
```

In item 4, end the bullet with `A tool that an admitted /chat/completions request runs is not refused.` (with the code span as above).

- [ ] **Step 4: Document the new MCP tool error**

In `tldw_Server_API/app/core/MCP_unified/README.md`, after the paragraph that ends `compatibility behavior.` (line 75) insert:

```markdown

When a module's LLM call is refused because the calling user is over an LLM
quota, the call returns the `llm_quota_exceeded` tool error. It is one user's
limit, not a module fault, so the module circuit breaker does not count it.
```

- [ ] **Step 5: Run the docs tests**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Docs`
Expected: `212 passed` (re-measured, with ADR-075 staged and the mirror refreshed). With the new ADR files unstaged, `test_docs_published_refresh.py::test_refresh_output_matches_tracked_published_files` fails (`1 failed, 211 passed`).

- [ ] **Step 6: Run Bandit on the touched source**

The report goes under the system temp directory, not a hard-coded `/tmp` (ADR-006, Review M1):

```bash
BANDIT_OUT="$($PY -c 'import tempfile; print(tempfile.gettempdir())')/bandit_task13585_13610.json"
uvx bandit -q -r tldw_Server_API/app/api/v1/endpoints/messages.py tldw_Server_API/app/core/LLM_Calls/usage_accounting.py tldw_Server_API/app/core/MCP_unified/tool_execution/runtime.py tldw_Server_API/app/core/MCP_unified/execution_outcomes.py tldw_Server_API/app/core/Claims_Extraction/claims_rebuild_service.py tldw_Server_API/app/core/Claims_Extraction/claims_service.py tldw_Server_API/app/services/startup_claims_rebuild.py -f json -o "$BANDIT_OUT"
$PY -c "import json, sys; print(len(json.load(open(sys.argv[1]))['results']))" "$BANDIT_OUT"
```

Expected: `0` (re-measured). `bandit` is not installed in `.venv`, so this uses `uvx`. The `nosec encountered` warnings for `claims_service.py` are already on dev.

- [ ] **Step 7: Commit the docs**

```bash
git add Docs/ADR/067-llm-calls-accounted-in-provider-base-class.md Docs/ADR/075-native-messages-and-mcp-tools-use-the-provider-recorder.md Docs/ADR/README.md Docs/Published/ADR Docs/Design/2026-10-09-llm-usage-accounting-design.md Docs/Operations/Usage_Quotas.md tldw_Server_API/app/core/MCP_unified/README.md
git commit -m "$(cat <<'EOF'
docs(usage): ADR-075, and MCP tools, the claims rebuild and the native Messages path are recorded (TASK-13585, TASK-13610)

ADR-075 (Proposed) extends ADR-067: the Messages native path records
through the provider recorder's public wrappers, an Anthropic-shaped
usage block counts cached input as prompt tokens, and an MCP tool's LLM
calls are recorded for its caller inside the chat pipeline too. ADR-067's
status lines say where MCP gets its user, and its unrecorded list shrinks
to system jobs with no user. The spec, the operator page and the MCP
tool-error contract say what a refusal does on each path.
Docs/Published refreshed.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 8: File the cache-pricing follow-up (Review B1)**

Compute the next free id with the AGENTS.md command, then create the task:

```bash
NEXT=$(( $( { git ls-tree -r --name-only origin/dev backlog/; gh pr list --state open --limit 500 --json files --jq '.[].files[].path'; } \
  | sed -nE 's|^backlog/.*task-([0-9]+)[ .].*|\1|p' | sort -n | tail -1 ) + 1 ))
echo "TASK-$NEXT"
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "Price Anthropic prompt-cache tokens, and keep cache fields on the Anthropic adapter path" --id "TASK-$NEXT" -l quotas,backend \
  -d "TASK-13610 counts Anthropic cache reads and writes as prompt tokens on the Messages API native path (ADR-075). No provider in Config_Files/model_pricing.json has a cache_read or cache_write rate, so compute_costs (Usage/usage_tracker.py) bills cached tokens at the full prompt rate: about ten times Anthropic's cache-read price, and below its cache-write price. The Anthropic adapter (LLM_Calls/providers/anthropic_adapter.py, near lines 373-378) also builds prompt_tokens from input_tokens alone and drops cache_creation_input_tokens and cache_read_input_tokens, so a prompt-cached call through /chat/completions is under-counted the way the native path was. Found by the independent review of the TASK-13585/TASK-13610 plan." \
  --ac "model_pricing.json gives the Anthropic models cache_read and cache_write rates, and a native /v1/messages row with cache fields is costed with them, with a test" \
  --ac "The Anthropic adapter keeps the cache fields in its usage block and counts them in prompt_tokens, with a test"
git add -A backlog/tasks
```

Expected: `TASK-13858` on 2026-10-18 (the scan printed 13857; other sessions may take ids first, so use what it prints). The task file is created as `To Do`, and `tests/CI/test_backlog_task_format_ratchet.py` accepts it (`4 passed`, re-measured in the prototype). Use this id in Step 9 and Step 11. **If the owner declines D1,** skip this step.

- [ ] **Step 9: Close both tasks with backlog-py**

Use the RED/GREEN counts you observed; the counts below are the prototype's. The notes carry the ADR check (Review M2) and the owner's answers to D1 to D3; if the owner declined one, write that answer and the alternative the plan applied instead. Replace `TASK-13858` with Step 8's id.

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13610 \
  --append-notes "RED: tests/Chat/unit/test_messages_native_usage_recording.py 9 failed, 2 passed on dev (no row written; the two record-nothing tests are guards). GREEN: 11 passed. The first draft's Step 3 failed 4 of them: a null message_delta field gave (0, 9), and the cached cases (3, 100) for (62003, 100) and (60003, 100). Messages, LLM_Calls and Usage suites: 610 passed, 1 skipped. Other tests naming the recorder or the cache fields: 233 passed, 1 skipped (Postgres skipped)." \
  --append-notes "Owner decisions: D1 (count cache_creation_input_tokens and cache_read_input_tokens as prompt tokens for an Anthropic-shaped block) accepted; D3 (ADR-075) accepted. llama.cpp's /v1/messages reports input_tokens = n_prompt_tokens - n_prompt_tokens_cache with cache_read_input_tokens (llama.cpp tools/server/server-task.cpp), so the sum applies to it too. Cache rates: follow-up TASK-13858." \
  --append-notes "ADR check: ADR required: yes. ADR path: Docs/ADR/075-native-messages-and-mcp-tools-use-the-provider-recorder.md, extending ADR-067 (governing). Reason: two public recorder entry points outside ChatProvider extend ADR-067's one recording point, a durable provider-integration rule." \
  --append-notes "No check at the call on the native path: both routes into _handle_messages carry Depends(require_llm_quota) (messages.py:2221, :2252), which marks the request prechecked, so llm_quota_gate.user_to_check() returns None for any call-site check there. test_messages_refuses_an_over_quota_user_before_any_provider_call already pins the 402 for the native path, streaming and not. count_tokens generates nothing and stays unrecorded." \
  --append-notes "Bandit: uvx bandit on the touched source, 0 findings (bandit is not in .venv). Postgres-only tests skip locally; this change touches no counter or SQL. Not tested, by design: estimate rows on the native path count no Anthropic content and no system prompt (ADR-075)." \
  --check-ac 1 --check-ac 2 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "Messages API calls to Anthropic and llama.cpp now write one llm_usage_log row for the user, streaming and not, through the provider base class's recorder (record_direct_response, accounted_direct_stream in LLM_Calls/usage_accounting.py; ADR-075). Tokens come from the reply's usage block; for a stream, input tokens from message_start and the final output count from message_delta, and a null field never erases a count. Cache reads and writes count as prompt tokens, because Anthropic's input_tokens is only the uncached remainder. An invalid 200 writes nothing. The boundary check already admits every native call, so no check at the call was added. Cache pricing is follow-up TASK-13858. ADR-075, ADR-067, the spec and Usage_Quotas.md updated." \
  -s Done
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13585 \
  --append-notes "RED: MCP test_tool_llm_usage_attribution.py 4 failed, 2 passed on dev (no rows; no refusal because no user; a tool under chat tool auto-exec not recorded), plus the outcome catalog test: 5 failed, 6 passed. Claims test_claims_rebuild_llm_owner.py 4 failed, 1 passed (worker thread died on a refusal: (0, 0, [1]); submit has no owner_user_id; owners None), plus the override_db and startup fakes (owner None). GREEN: all pass; MCP suites 4380 passed, 16 skipped; Claims suites 1082 passed, 1 skipped." \
  --append-notes "MCP uses llm_call_context(user_id=..., records_itself=False) around module.execute_tool, not llm_call_root: the root clears prechecked, which would let §4b refuse a tool that an admitted /chat/completions request runs (tool auto-exec). records_itself is cleared because the chat pipeline records only its own chat call (owner decision D2, accepted; ADR-075). A refusal becomes ExpectedToolFailure(LLM_QUOTA_EXCEEDED, BreakerAction.IGNORE), so the module breaker no longer counts it; this also covers /api/v1/tools/execute, where the content scope made refusals reach the breaker since PR 2b. There the refusal is HTTP 200 with ok: true and result.isError: true." \
  --append-notes "AC #1 covers MCP tool calls over the MCP HTTP and WebSocket transports, /api/v1/tools/execute, and tools run by /chat/completions tool auto-execution, each for the caller's numeric user id. A caller with no numeric user id (a service principal) is not recorded. Direct module.execute_tool callers outside the MCP runtime (workflow MCP and LLM steps, speech chat) are covered by their own roots." \
  --append-notes "ADR check: ADR required: yes. ADR path: Docs/ADR/075-native-messages-and-mcp-tools-use-the-provider-recorder.md, extending ADR-067 (governing). Reason: clearing records_itself for MCP tools changes ADR-067's consequence that a tool calling an LLM inside the chat pipeline is recorded by nobody. The MCP user and the claims owner are new roots under ADR-067's mechanism and need no ADR." \
  --append-notes "Claims rebuild: the worker now treats LLMQuotaExceededError as one failed task; without that, the new owner would let a refusal end the only worker thread. It logs a refusal at warning and stores llm_quota_exceeded, not the quota details, as the health row's last failure. A refused rebuild keeps the media's claims (extraction runs before the soft delete). Bandit: uvx bandit, 0 findings. Flake seen once: test_claims_jobs_contracts.py::test_review_metrics_payload_inclusive_window_property, green alone, on rerun and on dev." \
  --check-ac 1 --check-ac 2 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "MCP tool calls now record and check their LLM calls for the user in the call's RequestContext, over the MCP transports and /api/v1/tools/execute; a tool that /chat/completions tool auto-execution runs is recorded too and, being admitted, never refused (ADR-075). A refusal is the tool error llm_quota_exceeded and does not trip the module circuit breaker. A caller with no numeric user id is not recorded. In-process claims rebuild tasks carry their owner from every caller (per-media and bulk rebuild, claim-edit follow-up, startup loop) and run under llm_call_root(owner); a refusal fails one task, logs a warning, and the worker continues. ADR-075, ADR-067, the spec, Usage_Quotas.md and the MCP README updated." \
  -s Done
git add -A backlog/tasks
git commit -m "$(cat <<'EOF'
chore(backlog): close TASK-13585 and TASK-13610; file the cache-pricing follow-up

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
EOF
)"
```

**If the owner declines D2,** write these instead: in the first TASK-13585 note, `3 failed, 3 passed on dev (no rows; no refusal because no user), plus the outcome catalog test: 4 failed, 7 passed`; in the second, `llm_call_context(user_id=...)` with `Only the user is set` in place of the `records_itself` sentence; in the AC #1 note, `... and /api/v1/tools/execute, each for the caller's numeric user id. Tools run by /chat/completions tool auto-execution stay unrecorded and unchecked, as ADR-067 says (owner decision D2, declined).`; drop the ADR check's reason about `records_itself`, leaving `the MCP user and the claims owner are new roots under ADR-067's mechanism`; and in the final summary, `a tool that /chat/completions tool auto-execution runs keeps its admission and stays unrecorded (ADR-067)`.

Expected: both task files show `status: Done`, every AC and DoD checked, and the notes and final summary filled. `tests/CI/test_backlog_task_format_ratchet.py` accepts backlog-py output as it stands. Re-measured: both commands, run verbatim in the prototype after Step 8, printed `(Done)` with 8 boxes checked in each file, and the ratchet gave `4 passed`.

- [ ] **Step 10: Final verification on the branch**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Chat/unit/test_messages_native_usage_recording.py tldw_Server_API/app/core/MCP_unified/tests/test_tool_llm_usage_attribution.py tldw_Server_API/app/core/MCP_unified/tests/test_tool_execution_outcomes.py tldw_Server_API/tests/Claims/test_claims_rebuild_llm_owner.py tldw_Server_API/tests/LLM_Calls tldw_Server_API/tests/Usage tldw_Server_API/tests/Chat/unit tldw_Server_API/tests/Chat/integration/test_messages_endpoint.py tldw_Server_API/tests/CI/test_backlog_task_format_ratchet.py`
Expected: `3199 passed, 4 skipped` (re-measured, about 6 minutes). The review measured `3193 passed, 4 skipped` on the first draft; the review fixes add six tests (five in Task 1, one in Task 2). `test_backlog_task_format_ratchet.py` passed (4) after a backlog-py close-out.

Run: `$PY -m ruff check tldw_Server_API/app/api/v1/endpoints/messages.py tldw_Server_API/app/core/LLM_Calls/usage_accounting.py tldw_Server_API/app/core/MCP_unified/tool_execution/runtime.py tldw_Server_API/app/core/MCP_unified/execution_outcomes.py tldw_Server_API/app/core/Claims_Extraction/claims_rebuild_service.py tldw_Server_API/app/services/startup_claims_rebuild.py tldw_Server_API/tests/Chat/unit/test_messages_native_usage_recording.py tldw_Server_API/app/core/MCP_unified/tests/test_tool_llm_usage_attribution.py tldw_Server_API/tests/Claims/test_claims_rebuild_llm_owner.py`
Expected: `All checks passed!` (re-measured). `claims_service.py` is left out because its one import-order error is already on dev.

Check the branch for trailing whitespace: `git diff --check origin/dev...HEAD` prints nothing.

- [ ] **Step 11: Review, then open the PR against dev**

1. Run `/superpowers:requesting-code-review` in a clean subagent on `origin/dev...HEAD`, and address its findings before merge.
2. If the owner accepted ADR-075 (D3), set `**Status:** Accepted` in `Docs/ADR/075-native-messages-and-mcp-tools-use-the-provider-recorder.md` and `Accepted` in its `Docs/ADR/README.md` row, run `bash Helper_Scripts/refresh_docs_published.sh`, and commit both with the `Docs/Published` diff.
3. Write the PR body under the system temp directory, not a hard-coded `/tmp` (ADR-006, Review M1), then push and open the PR:

```bash
PR_BODY="$($PY -c 'import tempfile; print(tempfile.gettempdir())')/pr_task13585_13610.md"
git push -u origin feat/llm-usage-mcp-claims-messages
gh pr create --base dev --title "feat(usage): record LLM usage for MCP tools, the claims rebuild and the Messages native path (TASK-13585, TASK-13610)" --body-file "$PR_BODY"
```

`$PR_BODY` must contain:
- A `## Change summary` section. ADR-004 applies; the owner's Change-summary waiver for LLM usage accounting PRs covers this PR, so record it as a full section. Confirm the waiver with the owner if in doubt.
- An `## ADR check` section (Review M2): `ADR required: yes. ADR path: Docs/ADR/075-native-messages-and-mcp-tools-use-the-provider-recorder.md (Proposed, or Accepted after D3), extending ADR-067. Reason: two public recorder entry points outside ChatProvider, and MCP tools recorded inside a caller that records itself, change ADR-067's one recording point and one of its consequences.`
- The owner's answers to D1, D2 and D3, and the follow-up task id from Step 8.
- A test list: the four new test files and the suites in Task 1 Step 7, Task 2 Step 7, Task 4 Step 7 and Step 10.
- It ends with:

```
🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

4. Merge through the serial queue (the owner's `pr_queue.sh`), one PR at a time; MERGE_QUEUE is unset. Wait for the required statuses after any PR-body edit.
