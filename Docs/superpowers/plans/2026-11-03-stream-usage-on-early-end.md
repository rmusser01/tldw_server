# A dropped or failed /chat/completions stream writes one usage row (TASK-13953) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `/chat/completions` stream writes exactly one `llm_usage_log` row however it ends once something was received, and its Resource Governor reservation and billing commit those tokens; a stream that ends before anything was received refunds and writes no row, as today; the stream estimate counts what was received; the provider-level recorder fills a one-sided usage block from its estimate.

**Architecture:** One recorder per stream, in `chat_service.execute_streaming_call`: a nested `_stream_usage_recorder(stream_usage, full_reply)` claims the row once (a `nonlocal` flag) and returns the write, a `partial` of the new module-level `_record_stream_usage(...)` (the usage block or the estimate, `log_llm_usage`, then `rg_commit_cb(total)` in a `finally`). Both callers hand the write to `submit_accounting` and never await it: `save_callback` at its top, before the output-policy pass, the message save and tool autoexec, so a client that closes during the save still writes the row; `_finalize_stream` on every other ending (client drop, stop, provider error) once something was received, so no await of the write runs inside a cancelled scope. When nothing was received (no usage chunk, no characters), `_finalize_stream` refunds through `rg_refund_cb` and writes nothing, as the base did: the endpoint retries a failed attempt on the same request and reservation, and its close of that attempt reaches the pipeline as a cancel without output. The handler (`StreamingResponseHandler`) gives both callbacks the `_StreamUsage` it already feeds (`stream_usage=`): the provider's usage block and the characters of content, reasoning and tool-call arguments; `create_streaming_response_with_timeout.finalize_once` forwards it to a finalize callback that accepts it. The refund callback also stays for the failures before the stream starts (`_maybe_refund_streaming_rg`). In `usage_accounting`, `_record_response` and `_record_stream` take the side a usage block does not count from their estimate and mark the row estimated only when that fill replaced a zero.

**Tech Stack:** Python 3.12, pytest (asyncio auto mode, xdist), the repo's `llm_accounting.submit_accounting`/`drain_accounting`.

**Spec:**
- `backlog/tasks/task-13953 - A-dropped-or-failed-chat-completions-stream-writes-no-usage-row.md` (its four ACs);
- the planner brief `brief_plan_13953.md` and the review `review_plan_13953.md` (planning session's plans dir; the review's findings are folded in, see the last section);
- ADR-067 (`Docs/ADR/067-llm-calls-accounted-in-provider-base-class.md`), ADR-083 (`Docs/ADR/083-llm-quotas-skip-calls-on-the-users-own-key.md`);
- the executed plans `Docs/superpowers/plans/2026-10-26-stream-usage.md` (TASK-13951) and `Docs/superpowers/plans/2026-10-26-byok-quota-exemption.md`.

**Measured on:** `origin/dev` at `576feef820` (2026-11-03, the merge of #3409). The first version of this plan was measured at `a960463a26`; #3388 landed in between and moved `chat_service.py` by four lines before `:180` (its imports), nothing else this plan touches. Everything was measured in a detached worktree, which was reverted afterwards.

The prototype is `proto_13953.patch` in the plans dir: 13 files, +699/-125. It holds every code, test and doc change in this plan (the `Docs/Published` mirror included), but not the plan copy or the task file. Every RED/GREEN count below was measured on that patch's text: first the tests on unchanged code, then the code.

`file:line` citations are for `576feef820`.

---

## What was verified (on `576feef820`)

| # | Claim | Evidence | Verdict |
|---|---|---|---|
| V1 | A stream the client drops writes no row and refunds | **Yes.** `execute_streaming_call` (`chat_service.py:4106`) runs with `records_itself=True` (`endpoints/chat.py:5692`), so only its `save_callback` (`:5010`, usage block `:5137-5190`) records. The handler calls `save_callback` only when `not is_cancelled and not error_occurred and has_output` (`streaming_utils.py:2094-2099`); a cancel or error calls `finalize_callback` instead (`:2015-2029`, `:2198-2211`), and `_finalize_stream` (`chat_service.py:5510-5538`) refunds through `rg_refund_cb(cancelled=, error=)`, which commits 0 tokens (`endpoints/chat.py:5670-5680`). Reproduced: `test_a_stream_the_client_drops_still_writes_one_row_and_commits_its_tokens` on the base: no row (`ValueError: not enough values to unpack (expected 1, got 0)`), and the refund callback was called once with `cancelled=True, error=False`. A drop *during* the save (the message write blocked, the reader cancelled) also writes nothing on the base and on a version that awaits the write at the end of `save_callback` (the review's P3): `test_a_client_drop_during_the_save_still_writes_one_row_and_commits_its_tokens`. | **fix** |
| V2 | The stream estimate counts only the reply text | **Yes.** `completion_tokens = reported_completion or max(0, len(full_reply or "") // 4)` (`chat_service.py:5149`); `full_reply` is the content only. The handler already feeds every parsed payload to `self.stream_usage.feed_event(data)` (`streaming_utils.py:1624`), whose `.chars` counts `content`, `reasoning_content` and tool-call `arguments` (`usage_accounting.py:163-175`). Reproduced: a reasoning + tool-call reply records `completion_tokens == 0` on the base. | **fix** |
| V3 | The provider-level recorder records a one-sided block as given | **Yes.** `_record_response` (`usage_accounting.py:223-241`) and `_record_stream` (`:244-262`) record `(prompt, completion)` whenever it is not `(0, 0)`. Reproduced: a `{"prompt_tokens": 50}` block records `(50, 0)`, `estimated` None. | **fix** |
| V4 | Where billing commits | `rg_commit_cb` is `_build_streaming_commit_cb(...)` (`endpoints/chat.py:5664-5669`, built at `:6526-6601`): `billing_enforcer.record_actual(tokens)`, `get_billing_enforcer().apply_usage_delta(org, LLM_TOKENS_MONTH, tokens)`, then an awaitable that writes the durable cost-units ledger and `gov.commit(handle, actuals={"tokens": tokens})`. The endpoint's own `finally` (`:6419-6432`) refunds only when an exception leaves the endpoint. So "the Resource Governor and billing commit" = call `rg_commit_cb(total)` with the recorded total for every recorded ending. Billing is additive per call and the in-memory governor pops the handle on its first commit (`governor.py:861`), so a row for an attempt the endpoint then retries on the same handle (`:5774-5900`: an OAuth refresh or a fallback provider after an error before output) would charge the prompt twice. | no change there; rule in Decisions |
| V5 | ADR-083 for a dropped own-key stream | The row's `credential_source` is `usage_credential_source` (`chat_service.py:4156`, refreshed on fallback at `:4657`), written by `log_llm_usage`; the limits skip `user` rows (ADR-083). The billing enforcer is `None` for an own-key call (`endpoints/chat.py:4409-4418`), so `rg_commit_cb` then only commits the RG reservation. The fix passes the same `credential_source` on every ending. | kept |
| V6 | Where a cancelled stream's finalize runs | In production the handler's generator is driven by a separate bounded task (`create_streaming_response_with_timeout`, `streaming_utils.py:2363`, `:2388`). A disconnect cancels the response task; its `finally` (`:2458-2466`) cancels that task once (`cancel_stream_tasks_bounded`, `:420-447`) and detaches after 50 ms, so the handler's `finally` runs to the end in that task and reaches `finalize_callback` (`:2015-2029`). `save_callback` runs in that same task, so a drop during the save cancels it too: a write awaited at its end is lost. The D7 partial settlement is detached (`start_detached_settlement`, `:989-1018`) because the `aclose()` path (`:2469`, a 50 ms deadline) and anyio's level-triggered cancellation can cut any await in that `finally`. The usage write is handed to `submit_accounting` from both callers (`Usage/llm_accounting.py:45-69`: a task on the accounting loop, strong-referenced in `_pending`, outside any cancel scope) and never awaited there. | design |
| V7 | Frozen functions | `function_size_baseline.json`: `execute_streaming_call` 1545 lines, `StreamingResponseHandler.safe_stream_generator` 774, `_execute_non_stream_call_impl` 1054. Not listed: `execute_streaming_call.tracked_streaming_generator` (295 lines on the base; a first prototype grew it to 310 and failed the ratchet), `create_streaming_response_with_timeout` (257). After the prototype: 1521, 769, 296, 255 (ratchet: `13 passed`). | constraint |
| V8 | Tests that pinned the gap | 12 existing test ids assert that a stream which fails after it starts writes no usage row, that finalize gets exactly three keywords, or the old `provider_usage` keyword and the "counts as zero" rule: `test_chat_service_fallback.py` (`:6579` `missing-finish-eof` of `..._semantic_gate_is_request_local`, `:6881`, `:7141` three cases of `..._direct_done_then_adapter_failure_...`, `:7271`), `test_streaming_utils.py:1506` (×2), `test_async_stream_task_capacity.py:511`, `:571`, `test_streaming_generation_settlement.py:110`, `test_usage_accounting.py:297`. Measured: the prototype's code with the base's test files fails exactly those 12 ids on the six focused files (`12 failed, 558 passed, 1 skipped`) and nothing else. `test_execute_streaming_call_refund_cb_still_conditional` (`:6303`) passes on the prototype as written (its fake finalize passes nothing received, so the stream refunds); Task 2 Step 6 re-points it at a stream that was read. | update (Task 2 Step 6) |

## Decisions

1. **ADR-004 Change summary.** This PR is AI-authored. The recorded waivers (memory: "ADR-004 waiver for spec 2") name many TASK-139xx PRs but not TASK-13953.
   - **Recommendation:** ask the owner to extend the waiver to this PR, as for each earlier follow-up. Otherwise the owner writes the `## Change summary` before merge.
2. **A stream that ends before anything was received refunds and writes no row, whatever the ending** (a provider error, a client drop or a stop before the first token). The review's I1 showed that recording an error before output charges the prompt twice when the endpoint retries the attempt (`endpoints/chat.py:5774-5900`, same request and RG handle); the review's rule ("an uncancelled error with nothing received refunds; a drop before output records") was folded first and failed the retry pin: the endpoint's sanitizer records the error code in its side channel (`error_state`) from the daemon thread at the raise, `_prime_provider_stream_response` (`:1599-1601`) sees it on an earlier chunk and closes the attempt before the handler has processed the failure, so the pipeline's finalize arrives as `cancelled=True, error=False` with nothing received (measured with the finalize keywords captured; under the review's rule that request wrote two rows, `(17, 0)` for the failed attempt and `(17, 2)` for the retry). The second review confirmed it with its own probe: at the pipeline, a retry's close and a client drop before the first token are the same ending. The only signal the pipeline has is "nothing received", which is also ADR-067's rule for the provider-level recorder ("a stream that is opened and never read writes no row").
   - **Consequence:** a client that drops before the first token is refunded, as on the base (the review's P5 recorded the prompt under the first plan). It gets nothing, so it does not reopen the bypass the task names (reading a reply, then dropping, is charged in every variant probed). The residual is a cost leak with no benefit to the client: the operator still pays the provider for that prompt, as today, bounded by the Resource Governor's request limits.
   - **Follow-up, filed in Task 0 as TASK-15420:** the endpoint passes a "superseded" flag to `execute_streaming_call` when it closes an attempt to retry it (about 5 lines in `endpoints/chat.py`, 3 in `_finalize_stream`), and the pipeline then records the prompt estimate for every other nothing-received ending. It changes the endpoint's contract with the pipeline, so it gets its own PR and pins: the OAuth retry pin as the guard, plus a fallback-provider retry pin (unpinned today; the second review's M10).
   - **Recommendation:** keep this rule for this PR and file the follow-up. Pinned by `test_a_stream_dropped_before_any_output_is_refunded_and_writes_no_row`, `test_a_stream_that_fails_before_any_output_is_refunded_and_writes_no_row` and the endpoint-level `test_streaming_oauth_retry_records_one_usage_row_for_the_successful_attempt`. State it in the PR body.

Not owner questions (rulings for the reviewer, each pinned by a test and named in Review Focus):
- **A failure before the stream object exists** (the factory raises, `chat_service.py:4573-4713`) still records nothing and refunds, like a failed non-streaming call in both recorders.
- **The chat message save is unchanged.** `save_callback` still saves the message only on completion, and an early end still keeps its partial reply through `partial_save_callback` (D7, `streaming_utils.py:1983`, `:2178-2184`) under the same conditions; only the usage row and the commit were added to the early endings, and the completed stream's write moved to the top of `save_callback`.
- **The row's `status` stays 200** on every ending, as the provider-level recorder writes it (`_submit_record`, `usage_accounting.py:208`); the ending is in the metrics (`stream_cancelled`/`stream_error`), not the row. An operator tells a failed stream from a completed one through `estimate_source`/`estimated` only (the review's M7).
- **`provider_usage` becomes `stream_usage`** on `save_callback` (TASK-13951's keyword, 8 days old; only `chat_service` and one handler test used it). One name for both callbacks, carrying the `_StreamUsage` object instead of a copy of its `.usage`.
- **A non-finite usage value is estimated, not zero** (ADR-067 said "count as zero"): `_usage_tokens` already reads it as 0, and AC #4 makes a 0 side come from the estimate. The test `test_non_finite_usage_never_raises_and_counts_as_zero` is renamed `..._and_is_estimated`.
- **`estimated` marks a row only when a fill replaced a zero side** (the review's M1): a block that reports a zero side the estimate agrees with (an empty reply with `completion_tokens: 0`) keeps `estimated` None in all three recorders. Pinned by `test_a_zero_side_the_estimate_agrees_with_is_not_marked_estimated`.
- **The commit runs in a `finally`** (the review's M5): a write that raises outside `_CHAT_NONCRITICAL_EXCEPTIONS` (a driver error) still commits the tokens; the detached task's failure is logged by `_on_done`. Pinned by `test_the_tokens_are_committed_even_when_the_row_write_fails`.
- **The 12 ids the new code fails are updated** to the new semantics, not deleted: 2 renamed in Task 1, 4 finalize-keyword dicts and 6 chat-level tests in Task 2 Step 6; Step 6 also re-points `..._refund_cb_still_conditional` at a read stream (7 chat-level rewrites in all): each now asserts the recorded row (`estimate_source == "stream_estimate"`) where something was received and keeps its base assertion where nothing was.

---

## Global Constraints

Carried verbatim from the brief:

- Tests make no real outbound network requests; stub the provider adapter or its transport. A `/chat/completions` test with a commercial provider calls the real API unless the adapter is stubbed.
- Function-size ratchet `tldw_Server_API/tests/lint/function_size_baseline.json`: frozen functions must not grow. `execute_streaming_call` and the handlers in `streaming_utils.py` are large; check whether they are frozen and plan the change so they do not grow (extract a helper if needed).
- Backlog:
  - edit TASK-13953 with backlog-py only (`PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task edit ...`);
  - `--append-notes`, never `--notes`; stage with `git add -A backlog/tasks`; never hand-edit task files;
  - if a new task must be filed, its explicit id comes from `zsh <plans dir>/task_next.sh` (prints the HIGHEST id in use; the new id is that plus one), after `mkdir <scratchpad>/id_lock`; release it once the PR's file list shows it.
- Branch from `origin/dev`; the PR targets `dev`. Commit trailer: `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. PR body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- Never `--no-verify`, never force-push, never `git stash`.
- Redis: never FLUSHALL or SCRIPT FLUSH. PostgreSQL only through the repo fixtures (`tldw_postgres_test`), with a session-unique `TEST_DB_NAME`; drop only your own objects.
- Mirrored docs (`Docs/Published`) need `bash Helper_Scripts/refresh_docs_published.sh` in the same PR.
- CI checks before the PR, in the last task:
  - `tests/Docs`, `tests/lint`, `tests/CI`, and backlog-py's tests;
  - shard coverage (`Helper_Scripts/ci/check_shard_coverage.py`) and `Helper_Scripts/ci/test_quality_triage.py --enforce`;
  - the route-auth ratchet and the OpenAPI export check;
  - Bandit (`/opt/homebrew/bin/uvx bandit -ll`) on touched source;
  - every test file that names a touched symbol (git grep), with a baseline on `origin/dev`; run PostgreSQL files serially.
  - Commands: see `Docs/superpowers/plans/2026-11-02-deferred-minors-3391-3392.md`'s Task 3.
  - A file listed in `Helper_Scripts/ci/scope_predicate_baseline.txt` whose lines shift fails backend-required; none of the four files above is listed today, re-check for anything else you touch.
- Agent shells don't keep the venv `PATH`: prefix commands that spawn `python` with `PATH="$(dirname $PY):$PATH"`.

Plan-specific:

- **`$PY`** is `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`. Shell variables do not survive between agent calls, so write it out or set it in each call. `<scratchpad>` is the executing session's scratchpad directory, and `<plans dir>` is the planning session's plans dir: `/Users/macbook-dev/.claude/projects/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/plans`.
- **No new task.** TASK-13953 exists; Task 0 sets it In Progress. Nothing takes the id lock.
- **Frozen functions.** After each code step, `execute_streaming_call` must be at or under 1545 lines, `safe_stream_generator` at or under 774, and the nested `tracked_streaming_generator` and `create_streaming_response_with_timeout` under 301 (they are not in the baseline, so crossing 300 fails the ratchet). Measured after Task 2: 1521, 769, 296, 255. `tracked_streaming_generator` pays for the refund branch with a one-line `error_type` expression in `_finalize_stream` (same behaviour). Do not edit `function_size_baseline.json`.
- **TDD.** Write each task's tests first, run them, and see them fail for the stated reason. Record the real counts. Four of the new tests pin behaviour the base already has (the three refund pins and the `estimated`-flag pin) and pass on the base; each fails on a version that drops that behaviour, and the plan says so where they are run.
- **Detached writes in tests.** Every usage row is written by a task the test does not await, on a completed stream too. Tests `await drain_accounting()` (from `tldw_Server_API.app.core.Usage.llm_accounting`) before asserting on it; the chat-pipeline tests of an early end poll for the row first (`_settled` in Task 2), because the stream's own task finishes after a cancelled reader. For a completed stream the write runs on the first loop iteration after its claim (a fake `log_llm_usage` never suspends), before the `[DONE]` frame reaches the reader, so the completed-stream tests in `test_chat_service_fallback.py` and elsewhere in the wider set pass without a drain (measured); the TASK-13951 tests and the new ones drain anyway.
- **Git.** Stage named files. `git diff --cached --check` must print nothing before each commit.
- **Test runs** can leave a read-only Go module cache under `tldw_Server_API/Config_Files/acp_runner_home`. Never commit it. The directory also holds a tracked file, so never `rm -rf` the directory itself. If the cache appears, run `chmod -R u+w tldw_Server_API/Config_Files/acp_runner_home` and then `git clean -ffdx -- tldw_Server_API/Config_Files/acp_runner_home` before the last `git status`, which must be empty.
- **xdist.** `tests/AuthNZ_Unit/test_provider_usage_reservations_repo.py` has a uuid4 parametrize id and fails xdist's collection check; the wider set runs it serially (the script below does).
- **`mocker`.** `app/core/MCP_unified/tests/test_model_completion_accounting.py` errors 36 times with `fixture 'mocker' not found` on the base and after this plan (pytest-mock is not in the venv). Environment, not this plan's.
- **ruff.** `chat_service.py` and `streaming_utils.py` each carry one pre-existing `I001` on the base (`chat_service.py`'s came with #3388: the `HEARTBEAT_INTERVAL as ...` imports); every other touched file is clean before and after. Do not fix them here.

## Review Focus

1. **Exactly once.** `_stream_usage_recorder` claims the row under a `nonlocal` flag; `save_callback` and `_finalize_stream` are the only callers, and both hand the write to `submit_accounting` without awaiting it. `save_callback` claims at its top, before any await, so a drop or a failure during the save (the message write, tool autoexec, the audit) cannot lose the claimed row. The handler never calls both for one stream: the save gate (`streaming_utils.py:2094-2099`) and the finalize sites (`:2015`, `:2198`) are exclusive on `is_cancelled`/`error_occurred`, and `finalize_once` (`:2298-2316`) guards finalize itself. A stop requested during the save (`_settlement_started`, `:2102`) keeps the save's row and skips finalize. Pinned: every new chat-pipeline test unpacks `(row,) = await _settled(rows)`, which waits 50 ms after the first row for a second one; the TASK-13951 tests still unpack one row on a completed stream after `drain_accounting()`; the endpoint-level retry test unpacks one row for a request with two attempts.
2. **Cancellation safety.** Neither caller awaits the write: `submit_accounting(record)` creates the task and returns. Pinned by `test_a_stream_the_client_drops_still_writes_one_row_and_commits_its_tokens` and `test_a_client_drop_during_the_save_still_writes_one_row_and_commits_its_tokens` (both cancel the task reading `response.body_iterator`, the production path through `create_streaming_response_with_timeout`; the second while `save_message_fn` is blocked) and the handler-level `test_an_early_end_passes_what_the_stream_received_to_finalize[disconnect]`. The refund of a nothing-received ending is awaited in `_finalize_stream` where the base awaited it; on a cancel it has the base's exposure (a cut await leaves the reservation to expire on its TTL, which over-counts, not under).
3. **The commit for an early end.** `rg_commit_cb(total)` runs from `_record_stream_usage`'s `finally`, so billing and the RG commit never depend on the row: not on a non-critical failure, not on a driver error outside `_CHAT_NONCRITICAL_EXCEPTIONS` (`total_tokens` is computed before the write). `rg_refund_cb` is called from `_finalize_stream` only when nothing was received, and still for every failure before the stream starts (`_maybe_refund_streaming_rg`, `chat_service.py:4556`, `:4645`, `:4705`). Pinned: `refunded == []` and `committed == [row["total_tokens"]]` in the drop and after-output failure tests; `test_the_tokens_are_committed_even_when_the_row_write_fails`; `test_execute_streaming_call_commits_a_cancelled_stream_instead_of_refunding`.
4. **Nothing received.** `_finalize_stream` refunds and returns when `stream_usage` is None or has neither a usage chunk nor characters, whatever `cancelled`/`error` say (Decision 2). Pinned at the pipeline level by `test_a_stream_that_fails_before_any_output_is_refunded_and_writes_no_row` (`refunded == [{"cancelled": False, "error": True}]`) and `test_a_stream_dropped_before_any_output_is_refunded_and_writes_no_row` (`[{"cancelled": True, "error": False}]`), and at the endpoint by `test_streaming_oauth_retry_records_one_usage_row_for_the_successful_attempt` (a stubbed `perform_chat_api_call`: an OAuth-expired first attempt that raises before output, a refreshed second attempt; one row, `completion_tokens == len("recovered") // 4`). The six pre-existing tests whose stream received nothing keep their base assertions (`usage_log.assert_not_awaited()` or one row, the healthy stream's); the capacity-rejection case (`close_and_finalize_initial_capacity_rejection`, `streaming_utils.py:2337-2360`) refunds for the same reason.
5. **Own key (ADR-083).** The dropped stream's row carries `credential_source="user"` when the call ran on the user's own key, so the limits skip it as they skip a completed one; billing is not entered for such a call at the endpoint. Pinned: `test_a_dropped_stream_on_the_users_own_key_is_recorded_as_such`.
6. **Plain-text streams.** A provider that streams bare text (no `data:` lines) was estimated from `full_reply`; `stream_usage.chars` did not see it, so the switch to `chars` would have estimated it at 0. One line in the handler's non-SSE path now adds those characters. The provider-level recorder does not count bare text (its `feed` reads `data:` lines only), a divergence in favour of counting. Pinned: `test_plain_text_chunks_count_toward_what_the_stream_received`.

## Files

| File | Task | Change |
|---|---|---|
| `tldw_Server_API/app/core/LLM_Calls/usage_accounting.py` | 1 | `_submit_record` gains `estimated`; `_record_response` and `_record_stream` fill the missing side and mark the row estimated only when a fill replaced a zero |
| `tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py` | 1 | 1 new parametrised test (2 cases) + 1 new test; the non-finite test renamed and its expectation changed |
| `tldw_Server_API/app/core/Chat/streaming_utils.py` | 2 | `_save_callback_kwargs` (new method); `stream_usage` replaces `provider_usage`; bare-text chars counted; `finalize_once` forwards `stream_usage` |
| `tldw_Server_API/app/core/Chat/chat_service.py` | 2 | `_record_stream_usage` (new, module level, commit in a `finally`); `_stream_usage_recorder` (new, nested); `save_callback` claims and submits at its top; `_finalize_stream` refunds when nothing was received, else submits; imports |
| `tldw_Server_API/tests/Chat/unit/test_streaming_generation_settlement.py` | 2 | 1 test renamed to `stream_usage`; 2 new tests (one with 3 cases) |
| `tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py` | 2 | `rg_refund_cb` on the helper; `drain_accounting()` in the four TASK-13951 tests; 9 new tests |
| `tldw_Server_API/tests/Chat/integration/test_chat_endpoint_simplified.py` | 2 | 1 new test: the OAuth-refresh retry writes one row |
| `tldw_Server_API/tests/Chat/unit/test_chat_service_fallback.py` | 2 | 5 tests updated to the new semantics (V8); two keep their base assertions |
| `tldw_Server_API/tests/Chat/unit/test_streaming_utils.py`, `tldw_Server_API/tests/Chat/unit/test_async_stream_task_capacity.py` | 2 | the finalize-keyword assertions check `isinstance(..., _StreamUsage)` |
| `Docs/ADR/067-llm-calls-accounted-in-provider-base-class.md` (+ its `Docs/Published` mirror), `Docs/Operations/Usage_Quotas.md` | 3 | the "writes no row" sentences |
| `Docs/superpowers/plans/2026-11-03-stream-usage-on-early-end.md`, `backlog/tasks/task-13953 - ....md` | 0, 3 | this plan; the task |

- **Scope baseline.** `git grep -n -E "chat_service.py|streaming_utils.py|usage_accounting.py" Helper_Scripts/ci/scope_predicate_baseline.txt` prints nothing (measured).
- **Mirror.** `Docs/ADR/067-...` is mirrored under `Docs/Published/ADR/`; `Docs/Operations/Usage_Quotas.md` is not (measured: after `bash Helper_Scripts/refresh_docs_published.sh`, `git status --short Docs/Published` lists only the ADR).
- **Open PRs.** #3388 has landed (it is in the base). #3407 (head `7cf7d21101`, 18 files) touches `chat_service.py` at `:1620` only (`build_call_params_from_request`); #3411 (head `b2b731c93f`, 45 files) touches `endpoints/chat.py`, which this plan leaves alone. This plan touches `chat_service.py` at `:207`, `:218`, `:4006` (an insertion), `:4184`, `:5010-5190` and `:5510-5538`, and neither `endpoints/chat.py` nor the baseline. No hunk overlaps; neither PR needs to land first.

## Test commands

- **Focused set** (the test files this plan edits, SQLite):
  `PATH="$(dirname $PY):$PATH" TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py tldw_Server_API/tests/Chat/unit/test_streaming_generation_settlement.py tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py tldw_Server_API/tests/Chat/unit/test_streaming_utils.py tldw_Server_API/tests/Chat/unit/test_async_stream_task_capacity.py tldw_Server_API/tests/Chat/unit/test_chat_service_fallback.py`
  plus the endpoint pin: `PATH="$(dirname $PY):$PATH" TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Chat/integration/test_chat_endpoint_simplified.py -k "streaming_oauth_retry_records_one_usage_row or streaming_lazy_openai_oauth_refreshes_once_before_output"`.
  Measured on the three files with new tests: `79 passed` on the base (unchanged tests), `95 passed` at the end; the six files `586 passed, 1 skipped` at the end; the endpoint pin `2 passed`. See each task for its own counts.
- **Wider set**: every test file that names a touched symbol. Save as `<scratchpad>/wider.sh` and run `zsh <scratchpad>/wider.sh <label>` from the worktree root:

```zsh
#!/bin/zsh
S=<scratchpad>
PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python
L=${1:-run}
git grep --untracked -l -E "execute_streaming_call|_finalize_stream|rg_refund_cb|rg_commit_cb|_build_streaming_commit_cb|_record_stream\b|_record_response\b|_submit_record|_StreamUsage|stream_estimate|safe_stream_generator|create_streaming_response_with_timeout|provider_usage|finalize_callback|usage_accounting|StreamingResponseHandler|log_llm_usage|_estimate_tokens_from_messages|_approx_tokens|submit_accounting|drain_accounting" -- 'tldw_Server_API/tests/*.py' 'tldw_Server_API/app/*/tests/*.py' | grep '/test_[^/]*\.py$' | sort -u > $S/wider_all.txt
SERIAL='AuthNZ_Unit/test_provider_usage_reservations_repo\.py$'
grep -vE '/AuthNZ_Postgres/|_pg\.py$|_postgres\.py$' $S/wider_all.txt | grep -vE "$SERIAL" > $S/wider_sqlite.txt
grep -E "$SERIAL" $S/wider_all.txt > $S/wider_serial.txt
grep -E '/AuthNZ_Postgres/|_pg\.py$|_postgres\.py$' $S/wider_all.txt > $S/wider_pg.txt
echo "files: sqlite $(wc -l < $S/wider_sqlite.txt) + serial $(wc -l < $S/wider_serial.txt), pg $(wc -l < $S/wider_pg.txt)"
PATH="/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin:$PATH" TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q --no-header -n 8 --dist loadfile -rfE $(cat $S/wider_sqlite.txt) > $S/wider_sqlite_$L.out 2>&1
echo "sqlite $L: $(tail -1 $S/wider_sqlite_$L.out)"
grep '^FAILED\|^ERROR' $S/wider_sqlite_$L.out | cut -c1-220
PATH="/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin:$PATH" TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q --no-header -rfE $(cat $S/wider_serial.txt) > $S/wider_serial_$L.out 2>&1
echo "serial $L: $(tail -1 $S/wider_serial_$L.out)"
PATH="/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin:$PATH" TEST_DB_NAME=tldw_test_13953 $PY -m pytest -p no:cacheprovider -q --no-header -rfE $(cat $S/wider_pg.txt) > $S/wider_pg_$L.out 2>&1
echo "pg $L: $(tail -1 $S/wider_pg_$L.out)"
```

  Measured: 62 files: 59 SQLite files under xdist (`test_chat_endpoint_simplified.py` among them), 1 serial, 2 PostgreSQL (`tests/AuthNZ/integration/test_mcp_completion_accounting_postgres.py`, `tests/AuthNZ/integration/test_provider_usage_reservations_postgres.py`; run serially with the Docker fixture, about 7 minutes).
  - **On `576feef820`, clean:** SQLite `1977 passed, 2 skipped, 36 errors` (about 12 minutes; the 36 errors are `mocker`, Global Constraints); serial `88 passed`; PostgreSQL `138 passed` (17 minutes).
  - **After this plan:** SQLite `1994 passed, 2 skipped, 36 errors` (the 17 new ids; about 10 minutes); serial `88 passed`; PostgreSQL `138 passed`. Any other failure is this plan's.

---

## Task 0: Branch, start the task, plan file, baseline

- [ ] **Step 1: Branch.** `git fetch origin dev --quiet && git switch -c fix/stream-usage-early-end-13953 origin/dev`.
- [ ] **Step 2: Start the task** (backlog-py only):

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-13953 -s "In Progress" --append-notes "Plan: Docs/superpowers/plans/2026-11-03-stream-usage-on-early-end.md. Owner answers: ADR-004 Change summary <answer>; Decision 2 (nothing received refunds; follow-up TASK-15420) <answer>."
```

  `<answer>` is the owner's recorded answer (the controller passes it in the dispatch); never write it literally.
- [ ] **Step 2b: File the follow-up** (Decision 2). `mkdir <scratchpad>/id_lock` (if it exists, wait; never break another session's lock), then `zsh <plans dir>/task_next.sh` from the repository root; `<FU>` is the printed value + 1, and `ls backlog/tasks backlog/completed backlog/archive/tasks | grep "^task-15420 "` must print nothing.

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task create "Charge the prompt of a /chat/completions stream the client drops before its first token" \
  --id TASK-15420 -l quotas,backend --dep TASK-13953 \
  -d "TASK-13953 records a /chat/completions stream however it ends once something was received, but a stream that ends before the first token still refunds and writes no row: at the chat pipeline, the endpoint's close of a failed attempt it is about to retry (OAuth refresh, fallback provider) and a client drop before the first token are the same ending (cancelled, nothing received), and recording it would charge the retried prompt twice. So a client that sends a prompt and drops before the first token is never charged for it, while the operator pays the provider (a cost leak with no benefit to the client, bounded by the Resource Governor's request limits). Fix: the endpoint passes a 'superseded' flag to execute_streaming_call when it closes an attempt to retry it; _finalize_stream refunds only a superseded attempt and records the prompt estimate (completion 0) for every other nothing-received ending. See Docs/superpowers/plans/2026-11-03-stream-usage-on-early-end.md, Decision 2." \
  --ac "A /chat/completions stream the client drops or stops before its first token writes one usage row from the prompt estimate, and its Resource Governor reservation and billing commit those tokens, with a test" \
  --ac "An attempt the endpoint closes to retry it (OAuth refresh, fallback provider) is still refunded and writes no row: test_streaming_oauth_retry_records_one_usage_row_for_the_successful_attempt keeps passing, and a fallback-provider retry pin is added" \
  --ac "ADR-067 and Docs/Operations/Usage_Quotas.md say a drop before the first token is charged its prompt"
```

  Measured: not run (the id is taken at execution).
- [ ] **Step 3:** Copy this plan to `Docs/superpowers/plans/2026-11-03-stream-usage-on-early-end.md`, with `TASK-15420` and `task-15420` replaced by the real id. Then:
  - `git add Docs/superpowers/plans/2026-11-03-stream-usage-on-early-end.md && git add -A backlog/tasks && git diff --cached --check` (it must print nothing);
  - commit (message 0);
  - `git push -u origin fix/stream-usage-early-end-13953`, then `gh pr create --draft --base dev --title "fix(chat): a dropped or failed /chat/completions stream writes one usage row (TASK-13953)"` with a one-line body naming the task and ending with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`;
  - when `gh pr view <n> --json files --jq '.files[].path'` lists `backlog/tasks/task-15420 - ...md`, `rmdir <scratchpad>/id_lock`.
- [ ] **Step 4: Baseline.** On the unchanged branch, run:
  - the three files with new tests, unchanged: `PATH="$(dirname $PY):$PATH" TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py tldw_Server_API/tests/Chat/unit/test_streaming_generation_settlement.py tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py`: expect `79 passed` (measured);
  - the wider set with label `base`: expect SQLite `1977 passed, 2 skipped, 36 errors` (the 36 `mocker` errors; no failure), serial `88 passed`, PostgreSQL `138 passed` (all measured on the clean base).

  Record both in the ledger. Any later failure is compared with this baseline. Never dismiss one as environment-only without it.

## Task 1: The provider-level recorder takes a missing side from its estimate (AC #4)

**Files:** `tldw_Server_API/app/core/LLM_Calls/usage_accounting.py:180-262`; `tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py`.

**Interfaces:**
- Produces: `_submit_record(..., estimate_source, estimated: bool | None = None)`; `_record_response` and `_record_stream` keep their signatures. `_approx_tokens`, `_prompt_estimate`, `_response_chars`, `_StreamUsage` are unchanged and Task 2 imports `_approx_tokens` and `_StreamUsage` from here.

- [ ] **Step 1: Write the failing tests.** In `tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py`, insert this before `async def test_the_stream_estimate_counts_reasoning_and_tool_call_arguments(rows) -> None:` (`:426`), two blank lines on each side:

```python
@pytest.mark.parametrize(
    ("usage", "expected_chat", "expected_stream"),
    [
        ({"prompt_tokens": 50}, (50, 10), (50, 2)),
        ({"completion_tokens": 7}, (usage_accounting._prompt_estimate(REQUEST), 7), (usage_accounting._prompt_estimate(REQUEST), 7)),
    ],
    ids=["prompt-only", "completion-only"],
)
async def test_a_usage_block_that_counts_one_side_takes_the_other_from_the_estimate(
    rows, usage, expected_chat, expected_stream
) -> None:
    """A one-sided block does not make the other side count as nothing, as on /chat/completions (TASK-13953)."""

    class _OneSidedProvider(_FakeProvider):
        """Reports one side of the usage, in the response and in the stream."""

        name = "fake-one-sided"

        def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
            """Return 40 characters of content with the one-sided block."""
            return {"choices": [{"message": {"content": "x" * 40}}], "usage": dict(usage)}

        def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
            """Yield 8 characters of content, then the one-sided block."""
            yield 'data: {"choices":[{"delta":{"content":"abcdefgh"}}]}\n\n'
            yield "data: " + json.dumps({"choices": [], "usage": usage}) + "\n\n"
            yield "data: [DONE]\n\n"

    with llm_call_context(user_id=7):
        _OneSidedProvider().chat(REQUEST)
        list(_OneSidedProvider().stream(REQUEST))
    chat_row, stream_row = await _settled(rows)
    assert (chat_row["prompt_tokens"], chat_row["completion_tokens"]) == expected_chat
    assert (stream_row["prompt_tokens"], stream_row["completion_tokens"]) == expected_stream
    assert chat_row["estimated"] is True and stream_row["estimated"] is True
    assert chat_row["usage_metadata"] == usage and stream_row["usage_metadata"] == usage


async def test_a_zero_side_the_estimate_agrees_with_is_not_marked_estimated(rows) -> None:
    """An empty reply with a block that counts no completion tokens keeps (prompt, 0): nothing was estimated."""
    usage = {"prompt_tokens": 50, "completion_tokens": 0}

    class _EmptyReplyProvider(_FakeProvider):
        """Returns no content, and a block that counts the prompt only."""

        name = "fake-empty-reply"

        def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
            """Return an empty reply with the block."""
            return {"choices": [{"message": {"content": ""}}], "usage": dict(usage)}

        def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
            """Yield the block and nothing else."""
            yield "data: " + json.dumps({"choices": [], "usage": usage}) + "\n\n"
            yield "data: [DONE]\n\n"

    with llm_call_context(user_id=7):
        _EmptyReplyProvider().chat(REQUEST)
        list(_EmptyReplyProvider().stream(REQUEST))
    chat_row, stream_row = await _settled(rows)
    assert (chat_row["prompt_tokens"], chat_row["completion_tokens"]) == (50, 0)
    assert (stream_row["prompt_tokens"], stream_row["completion_tokens"]) == (50, 0)
    assert chat_row["estimated"] is None and stream_row["estimated"] is None
```

  Then change `test_non_finite_usage_never_raises_and_counts_as_zero` (`:297-322`): rename it `test_non_finite_usage_never_raises_and_is_estimated`, make its docstring `"""An inf or NaN token count is skipped like a missing one, so it comes from the estimate (TASK-13953), and\n    the call is unaffected."""`, and replace its last two asserts with:

```python
    prompt_estimate = usage_accounting._prompt_estimate(REQUEST)
    assert (chat_row["prompt_tokens"], chat_row["completion_tokens"]) == (prompt_estimate, 7)
    assert (stream_row["prompt_tokens"], stream_row["completion_tokens"]) == (prompt_estimate, 2)
```

- [ ] **Step 2: Run them to verify they fail.**
  Run: `PATH="$(dirname $PY):$PATH" TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py -k "one_side or non_finite or zero_side"`
  Expected: `3 failed, 1 passed, 34 deselected` (measured): `assert (50, 0) == (50, 10)` for prompt-only, `assert (0, 7) == (15, 7)` for completion-only, and `assert (0, 7) == (15, 7)` for non-finite. The zero-side test passes on the base (it writes `(50, 0)` unmarked) and pins that the fill below does not mark it; a fill that sets `estimated` on any one-sided block fails it.
- [ ] **Step 3: Implement.** In `usage_accounting.py`:
  - `_submit_record` (`:180-190`): add the parameter `estimated: bool | None = None,` after `estimate_source: str | None,`, change the docstring to `"""Schedule one usage row; never raises. ``estimated`` marks a row whose block counts only one side."""`, and change `estimated=True if usage is None else None,` (`:213`) to `estimated=True if usage is None or estimated else None,`.
  - `_record_response` (`:230-233`): replace the four-line `_submit_record(...)` call inside `if (prompt_tokens, completion_tokens) != (0, 0):` with

```python
                # A side the block does not count (a prompt-only block, say) comes from the estimate (TASK-13953).
                counted = (prompt_tokens, completion_tokens)
                filled = (prompt_tokens or _prompt_estimate(request), completion_tokens or _approx_tokens(_response_chars(response)))
                _submit_record(
                    provider, plan, request, usage=usage, prompt_tokens=filled[0], completion_tokens=filled[1],
                    started=started, estimate_source=None, estimated=filled != counted,
                )
```

  - `_record_stream` (`:252-255`): likewise,

```python
                # A side the chunk does not count comes from the estimate, as on /chat/completions (TASK-13953).
                counted = (prompt_tokens, completion_tokens)
                filled = (prompt_tokens or _prompt_estimate(request), completion_tokens or _approx_tokens(seen.chars))
                _submit_record(
                    provider, plan, request, usage=seen.usage, prompt_tokens=filled[0], completion_tokens=filled[1],
                    started=started, estimate_source=None, estimated=filled != counted,
                )
```

- [ ] **Step 4: Run the file.** `PATH="$(dirname $PY):$PATH" TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py`. Expected: `38 passed` (35 on the base + 3). `$(dirname $PY)/ruff check tldw_Server_API/app/core/LLM_Calls/usage_accounting.py tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py`: "All checks passed!".
- [ ] **Step 5: Commit** (message 1): `git add tldw_Server_API/app/core/LLM_Calls/usage_accounting.py tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py && git diff --cached --check`.

## Task 2: One usage row per /chat/completions stream, however it ends, committed, from what was received (AC #1, #2, #3)

**Files:** `tldw_Server_API/app/core/Chat/streaming_utils.py:1213-1221, 1476-1480, 1757-1759, 2104-2113, 2259-2270, 2298-2314`; `tldw_Server_API/app/core/Chat/chat_service.py:207, 218, 4006, 4184, 5010-5018, 5137-5190, 5510-5538`; tests: `tldw_Server_API/tests/Chat/unit/test_streaming_generation_settlement.py`, `tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py`, `tldw_Server_API/tests/Chat/integration/test_chat_endpoint_simplified.py`, `tldw_Server_API/tests/Chat/unit/test_chat_service_fallback.py`, `tldw_Server_API/tests/Chat/unit/test_streaming_utils.py`, `tldw_Server_API/tests/Chat/unit/test_async_stream_task_capacity.py`.

The handler and the chat pipeline change in one task: once the handler passes `stream_usage` instead of `provider_usage`, a `save_callback` that still takes `provider_usage` would record the estimate for a reply that sent a usage chunk, and TASK-13951's tests would fail in between.

**Interfaces:**
- Consumes (Task 1): `_approx_tokens(chars) -> int`, `_StreamUsage` (`.usage: dict | None`, `.chars: int`) from `usage_accounting`; `submit_accounting(make_coro) -> bool` and `drain_accounting()` from `Usage/llm_accounting`.
- Produces:
  - `StreamingResponseHandler._save_callback_kwargs(save_callback) -> dict[str, Any]`: `{"generation": ..., "stream_usage": self.stream_usage}`, each only when the callback accepts the keyword. `save_callback` no longer receives `provider_usage`.
  - `create_streaming_response_with_timeout(..., finalize_callback=...)`: the finalize callback receives `stream_usage=handler.stream_usage` when it accepts that keyword (a `**kwargs` callback does).
  - `chat_service._record_stream_usage(*, request, provider, model, templated_llm_payload, conversation_id, llm_start_time, credential_source, stream_usage, full_reply="", rg_commit_cb) -> None` (module level; never raises; the commit runs in its `finally`).
  - Inside `execute_streaming_call`: `_stream_usage_recorder(stream_usage, full_reply="") -> Callable[[], Awaitable[None]] | None` (nested; returns None once claimed); `save_callback(..., *, generation=None, stream_usage: _StreamUsage | None = None)`; `_finalize_stream(*, success, cancelled, error, stream_usage: _StreamUsage | None = None)`.

- [ ] **Step 1: Write the failing handler tests** in `tldw_Server_API/tests/Chat/unit/test_streaming_generation_settlement.py`.
  - Add `from tldw_Server_API.app.core.Chat import streaming_utils` before `from tldw_Server_API.app.core.Chat.streaming_utils import (` (`:13`).
  - Rewrite `test_complete_reply_passes_the_whole_provider_usage_block_to_a_save_that_takes_it` (`:110-140`): rename it `test_complete_reply_passes_what_the_stream_received_to_a_save_that_takes_it`, docstring `"""The usage record needs the whole block (tier, cache counts, cost) and the characters received;\n    generation metadata keeps the counts (TASK-13951, TASK-13953)."""`, the `save` signature's last keyword `provider_usage=None` becomes `stream_usage=None` and it appends `{"generation": generation, "stream_usage": stream_usage}`, and the asserts become:

```python
    (saved,) = saves
    assert saved["stream_usage"].usage == {**usage, "service_tier": "flex"}
    assert saved["stream_usage"].chars == len("Hi")
    assert saved["generation"]["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
```

  - Append after it (before `async def test_length_finish_reason_settles_as_length():`):

```python
@pytest.mark.parametrize("ending", ["stop", "disconnect", "provider_error"])
async def test_an_early_end_passes_what_the_stream_received_to_finalize(monkeypatch, ending):
    """The usage row of a stream that ends early needs its usage chunk and character count (TASK-13953)."""
    handlers: list[StreamingResponseHandler] = []

    class _Captured(StreamingResponseHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            handlers.append(self)

    monkeypatch.setattr(streaming_utils, "StreamingResponseHandler", _Captured)
    finalized: list[dict[str, Any]] = []
    delivered = asyncio.Event()

    async def finalize(*, success, cancelled, error, stream_usage=None):
        finalized.append({"cancelled": cancelled, "error": error, "stream_usage": stream_usage})

    async def provider() -> AsyncIterator[str]:
        yield _sse({"choices": [{"delta": {"content": "Half "}}], "usage": {"prompt_tokens": 5, "completion_tokens": 1}})
        yield _content("done")
        if ending == "stop":
            handlers[0].request_stop()
            yield _content(" never")
        if ending == "provider_error":
            raise RuntimeError("upstream reset")
        await asyncio.Event().wait()

    async def consume() -> None:
        async for frame in create_streaming_response_with_timeout(
            provider(), "conv", "model", finalize_callback=finalize, heartbeat_interval=0
        ):
            if "done" in frame:
                delivered.set()

    consumer = asyncio.ensure_future(consume())
    await asyncio.wait_for(delivered.wait(), timeout=10)
    if ending == "disconnect":
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
    else:
        await asyncio.wait_for(consumer, timeout=10)
    deadline = asyncio.get_running_loop().time() + 10
    while not finalized and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.01)  # the stream's own task finishes after the consumer

    (call,) = finalized
    assert (call["cancelled"], call["error"]) == (ending != "provider_error", ending == "provider_error")
    assert call["stream_usage"].usage == {"prompt_tokens": 5, "completion_tokens": 1}
    assert call["stream_usage"].chars == len("Half done")


async def test_plain_text_chunks_count_toward_what_the_stream_received():
    """A provider that streams bare text, not SSE, is not estimated at nothing (TASK-13953)."""
    handler = StreamingResponseHandler("conv", "model")
    saves: list[Any] = []

    async def save(text, tool_calls=None, function_call=None, *, stream_usage=None):
        saves.append(stream_usage)
        return "saved-complete"

    async def provider() -> AsyncIterator[str]:
        yield "Plain "
        yield "text reply"

    async for _frame in handler.safe_stream_generator(provider(), save_callback=save):
        pass

    (received,) = saves
    assert (received.usage, received.chars) == (None, len("Plain text reply"))
```

- [ ] **Step 2: Write the failing chat-pipeline tests** in `tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py`.
  - `:9` becomes `from unittest.mock import AsyncMock, Mock`.
  - Add `from tldw_Server_API.app.core.Usage.llm_accounting import drain_accounting` after `from tldw_Server_API.app.core.Usage import usage_tracker` (`:20`).
  - In `_run_execute_streaming_call` (`:190-247`): add the parameter `rg_refund_cb=None,` after `rg_commit_cb=None,` and pass `rg_refund_cb=rg_refund_cb,` after `rg_commit_cb=rg_commit_cb,`.
  - In the four TASK-13951 tests, add `    await drain_accounting()` on its own line directly before `(row,) = stream_usage_rows`: `test_a_streamed_reply_is_recorded_from_the_usage_chunk_its_provider_sends` (`:1155`, after the blank line that follows `chunks = await _collect_sse_chunks(response)`), `test_a_streamed_reply_without_usage_is_recorded_from_the_estimate` (`:1191`), `test_a_usage_block_without_completion_tokens_counts_the_reply_from_the_estimate` (`:1225`) and `test_a_streamed_tool_call_reply_is_recorded_once_from_its_usage_chunk` (`:1263`, after `assert executed == ["c1"]`). The write is detached once Step 5 lands; see Global Constraints.
  - Append at the end of the file:

```python
# --- A stream that ends early still writes one usage row (TASK-13953) ---

_PROMPT_ESTIMATE = chat_service._estimate_tokens_from_messages([{"role": "user", "content": "hi"}])


def _content_then(*frames: dict[str, Any], then: str = "wait") -> Any:
    """A provider call that streams "hello world!" and ``frames``, then waits for a client that never comes back,
    or fails. ``then="wait_after_frames"`` signals ``read`` once every frame was read, before waiting."""
    read = asyncio.Event()

    def _call() -> Any:
        async def _stream() -> Any:
            if then != "fail_before_output":
                yield 'data: {"choices": [{"delta": {"content": "hello world!"}}]}\n\n'
            for frame in frames:
                yield f"data: {json.dumps(frame)}\n\n"
            if then.startswith("fail"):
                raise RuntimeError("upstream reset")
            read.set()
            await asyncio.Event().wait()

        return _stream()

    _call.read = read  # type: ignore[attr-defined]
    return _call


async def _drop_after(response: StreamingResponse, ready: asyncio.Event | None = None) -> None:
    """Read the stream until "hello world!" (and ``ready``), then cancel the reading task, as a disconnect does."""
    delivered = asyncio.Event()

    async def _read() -> None:
        async for chunk in response.body_iterator:
            if "hello world!" in str(chunk):
                delivered.set()

    reader = asyncio.ensure_future(_read())
    await asyncio.wait_for(delivered.wait(), timeout=10)
    if ready is not None:
        await asyncio.wait_for(ready.wait(), timeout=10)
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reader


async def _settled(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The rows once the stream's own task, which outlives a cancelled reader, has written its row."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 10
    while not rows and loop.time() < deadline:
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)  # a second row, were one written, shows up here
    await drain_accounting()
    return rows


@pytest.mark.asyncio
@pytest.mark.unit
async def test_a_stream_the_client_drops_still_writes_one_row_and_commits_its_tokens(stream_usage_rows) -> None:
    """The client read the reply and closed on it; the reply counts toward the user's limits (AC #1, AC #2)."""
    committed: list[int] = []
    refunded: list[dict[str, Any]] = []

    response = await _run_execute_streaming_call(
        llm_call_func=_content_then(),
        save_message_fn=_save_nothing,
        rg_commit_cb=committed.append,
        rg_refund_cb=lambda **kwargs: refunded.append(kwargs),
    )
    await _drop_after(response)

    (row,) = await _settled(stream_usage_rows)
    assert row["estimate_source"] == "stream_estimate"
    assert (row["prompt_tokens"], row["completion_tokens"]) == (_PROMPT_ESTIMATE, len("hello world!") // 4)
    assert committed == [row["total_tokens"]]
    assert refunded == []


@pytest.mark.asyncio
@pytest.mark.unit
async def test_a_dropped_stream_is_recorded_from_the_usage_chunk_that_arrived(stream_usage_rows) -> None:
    """The provider's count wins over the estimate when the chunk arrived before the drop (AC #1)."""
    committed: list[int] = []
    usage = {"prompt_tokens": 40, "completion_tokens": 9, "total_tokens": 49}
    llm_call_func = _content_then({"choices": [], "usage": usage})

    response = await _run_execute_streaming_call(
        llm_call_func=llm_call_func, save_message_fn=_save_nothing, rg_commit_cb=committed.append
    )
    await _drop_after(response, ready=llm_call_func.read)

    (row,) = await _settled(stream_usage_rows)
    assert row["estimate_source"] == "provider_usage"
    assert (row["prompt_tokens"], row["completion_tokens"]) == (40, 9)
    assert committed == [49]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_a_client_drop_during_the_save_still_writes_one_row_and_commits_its_tokens(stream_usage_rows) -> None:
    """The client closed on finish_reason while the message was being saved, where a close on the last frame
    lands; the reply counts all the same (AC #1, AC #2)."""
    committed: list[int] = []
    refunded: list[dict[str, Any]] = []
    saving = asyncio.Event()
    release = asyncio.Event()

    async def _blocked_save(*_args: Any, **_kwargs: Any) -> str:
        saving.set()
        await release.wait()
        return "message-1"

    response = await _run_execute_streaming_call(
        llm_call_func=_reply_then(),
        save_message_fn=_blocked_save,
        rg_commit_cb=committed.append,
        rg_refund_cb=lambda **kwargs: refunded.append(kwargs),
    )
    await _drop_after(response, ready=saving)
    release.set()

    (row,) = await _settled(stream_usage_rows)
    assert (row["prompt_tokens"], row["completion_tokens"]) == (_PROMPT_ESTIMATE, len("hello world!") // 4)
    assert committed == [row["total_tokens"]]
    assert refunded == []


@pytest.mark.asyncio
@pytest.mark.unit
async def test_a_stream_that_fails_writes_one_row_from_what_it_received(stream_usage_rows) -> None:
    """A provider error after output records the prompt and what arrived, as the provider-level recorder does;
    the tokens are committed, not refunded (AC #1, AC #2)."""
    committed: list[int] = []
    refunded: list[dict[str, Any]] = []

    response = await _run_execute_streaming_call(
        llm_call_func=_content_then(then="fail_after_output"),
        save_message_fn=_save_nothing,
        rg_commit_cb=committed.append,
        rg_refund_cb=lambda **kwargs: refunded.append(kwargs),
    )
    chunks = await _collect_sse_chunks(response)

    (row,) = await _settled(stream_usage_rows)
    assert (row["prompt_tokens"], row["completion_tokens"]) == (_PROMPT_ESTIMATE, len("hello world!") // 4)
    assert row["estimate_source"] == "stream_estimate"
    assert committed == [row["total_tokens"]]
    assert refunded == []
    assert any('"success": false' in chunk for chunk in chunks)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_a_stream_that_fails_before_any_output_is_refunded_and_writes_no_row(stream_usage_rows) -> None:
    """Nothing came back, and the endpoint retries such a failure (an OAuth refresh, a fallback provider) on the
    same request and reservation: a row here would charge the prompt twice. Refunded, as before (AC #1)."""
    committed: list[int] = []
    refunded: list[dict[str, Any]] = []

    response = await _run_execute_streaming_call(
        llm_call_func=_content_then(then="fail_before_output"),
        save_message_fn=_save_nothing,
        rg_commit_cb=committed.append,
        rg_refund_cb=lambda **kwargs: refunded.append(kwargs),
    )
    chunks = await _collect_sse_chunks(response)
    await drain_accounting()

    assert stream_usage_rows == []
    assert committed == []
    assert refunded == [{"cancelled": False, "error": True}]
    assert any('"success": false' in chunk for chunk in chunks)


@pytest.mark.asyncio
@pytest.mark.unit
async def test_a_stream_dropped_before_any_output_is_refunded_and_writes_no_row(stream_usage_rows) -> None:
    """Nothing was received, so nothing is recorded, as the provider-level recorder does for an unread stream;
    the endpoint's own close of a failed attempt before its retry reaches the pipeline as such a drop."""
    committed: list[int] = []
    refunded: list[dict[str, Any]] = []
    entered = asyncio.Event()

    def llm_call_func() -> Any:
        async def _stream() -> Any:
            entered.set()
            await asyncio.Event().wait()
            yield ""  # never reached

        return _stream()

    response = await _run_execute_streaming_call(
        llm_call_func=llm_call_func,
        save_message_fn=_save_nothing,
        rg_commit_cb=committed.append,
        rg_refund_cb=lambda **kwargs: refunded.append(kwargs),
    )
    reader = asyncio.ensure_future(_collect_sse_chunks(response))
    await asyncio.wait_for(entered.wait(), timeout=10)
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reader
    deadline = asyncio.get_running_loop().time() + 10
    while not refunded and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.01)  # the stream's own task finishes after the cancelled reader
    await drain_accounting()

    assert stream_usage_rows == []
    assert committed == []
    assert refunded == [{"cancelled": True, "error": False}]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_the_tokens_are_committed_even_when_the_row_write_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A usage write that fails with a driver error still commits to the Resource Governor and billing: a commit
    skipped on a failed write would be a quota bypass (AC #2)."""

    class _DriverError(Exception):
        """Not one of the chat pipeline's non-critical exceptions."""

    committed: list[int] = []
    monkeypatch.setattr(chat_service, "log_llm_usage", AsyncMock(side_effect=_DriverError("disk full")))

    response = await _run_execute_streaming_call(
        llm_call_func=_reply_then(), save_message_fn=_save_nothing, rg_commit_cb=committed.append
    )
    await _collect_sse_chunks(response)
    await drain_accounting()

    assert committed == [_PROMPT_ESTIMATE + len("hello world!") // 4]


@pytest.mark.asyncio
@pytest.mark.unit
async def test_a_dropped_stream_on_the_users_own_key_is_recorded_as_such(
    monkeypatch: pytest.MonkeyPatch, stream_usage_rows
) -> None:
    """The row carries credential_source 'user', which the limits skip, as a completed one does (ADR-083)."""
    monkeypatch.setattr(chat_service, "call_credential_source", lambda _args: "user")

    response = await _run_execute_streaming_call(llm_call_func=_content_then(), save_message_fn=_save_nothing)
    await _drop_after(response)

    (row,) = await _settled(stream_usage_rows)
    assert row["credential_source"] == "user"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_the_stream_estimate_counts_reasoning_and_tool_call_arguments(stream_usage_rows) -> None:
    """A tool-call-only reply is not free: its arguments and reasoning count, as the provider-level recorder
    counts them (AC #3)."""
    arguments = json.dumps({"query": "x" * 30})
    tool_delta = {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                                                         "function": {"name": "notes.search", "arguments": arguments}}]}}]}
    reasoning = {"choices": [{"delta": {"reasoning_content": "y" * 40}}]}

    def llm_call_func() -> Any:
        def _stream() -> Any:
            yield f"data: {json.dumps(reasoning)}\n\n"
            yield f"data: {json.dumps(tool_delta)}\n\n"
            yield 'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'
            yield "data: [DONE]\n\n"

        return _stream()

    response = await _run_execute_streaming_call(llm_call_func=llm_call_func, save_message_fn=_save_nothing)
    await _collect_sse_chunks(response)

    await drain_accounting()
    (row,) = stream_usage_rows
    assert row["completion_tokens"] == (len(arguments) + 40) // 4
    assert row["estimate_source"] == "stream_estimate"
```

  - **The endpoint-level retry pin**, in `tldw_Server_API/tests/Chat/integration/test_chat_endpoint_simplified.py`: insert this after `test_streaming_lazy_openai_oauth_refreshes_once_before_output` (its last line `:4056`), before `def test_real_openai_401_refreshes_oauth_once_before_stream_output(` (`:4059`), two blank lines on each side. It stubs `perform_chat_api_call`, so no request leaves the process (`chat_service`, `AsyncMock`, `ChatAuthenticationError`, `_certified_pre_dispatch` and `_credential_runtime_double` are already in the file).

```python
def test_streaming_oauth_retry_records_one_usage_row_for_the_successful_attempt(
    authenticated_client,
    mock_chacha_db,
    setup_dependencies,
):
    """The attempt that failed before any output is refunded, not recorded, so the retry on the same request and
    reservation is charged once (TASK-13953)."""
    runtime_type = _credential_runtime_double(auth_source="oauth")
    provider_manager = MagicMock()
    provider_manager.circuit_breakers = {"openai": SimpleNamespace(can_attempt_call=lambda: True)}
    request_data = ChatCompletionRequest(
        model="gpt-4o-mini",
        api_provider="openai",
        messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
        stream=True,
    )
    usage_log = AsyncMock(return_value=None)

    def expired_stream():
        if False:
            yield ""
        raise _certified_pre_dispatch(
            ChatAuthenticationError("sentinel expired oauth", provider="openai")
        )

    def refreshed_stream():
        yield 'data: {"choices":[{"delta":{"content":"recovered"}}]}\n\n'
        yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        yield "data: [DONE]\n\n"

    with (
        patch.object(chat_endpoint, "ProviderCredentialRuntime", runtime_type),
        patch.object(chat_endpoint, "get_provider_manager", return_value=provider_manager),
        patch.object(chat_endpoint, "get_request_queue", return_value=None),
        patch.object(chat_endpoint, "QUEUED_EXECUTION", False),
        patch.object(chat_endpoint, "perform_chat_api_call") as provider_call,
        patch.object(chat_service, "log_llm_usage", usage_log),
    ):
        provider_call.side_effect = [expired_stream(), refreshed_stream()]
        response = authenticated_client.post(
            "/api/v1/chat/completions",
            json=request_data.model_dump(),
        )

        assert response.status_code == status.HTTP_200_OK
        assert provider_call.call_count == 2
        (row,) = [call.kwargs for call in usage_log.await_args_list]
    assert (row["estimate_source"], row["completion_tokens"]) == ("stream_estimate", len("recovered") // 4)
```

- [ ] **Step 3: Run them to verify they fail.**
  Run: `PATH="$(dirname $PY):$PATH" TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Chat/unit/test_streaming_generation_settlement.py tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py -k "stream_received or early_end or plain_text or client_drops or drop_during_the_save or usage_chunk_that_arrived or fails_writes_one_row or fails_before_any_output or dropped_before_any_output or own_key or reasoning_and_tool_call or row_write_fails"`
  Expected: `12 failed, 2 passed, 43 deselected` (measured, about 75 s: the no-row waits run to their 10 s deadline on the base): the settlement ones with `KeyError: 'stream_usage'` / `(call,) = finalized` → `ValueError: not enough values to unpack`, the drop/failure/own-key ones with `(row,) = ...` → `ValueError: not enough values to unpack (expected 1, got 0)`, the driver-error one with `assert [] == [12]`, the reasoning one with `assert 0 == 24`. The two that pass, `..._fails_before_any_output_...` and `..._dropped_before_any_output_...`, pin the base's refund (Decision 2); a version that records every ending fails them.
  Then the endpoint pin on the unchanged code: `PATH="$(dirname $PY):$PATH" TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Chat/integration/test_chat_endpoint_simplified.py -k streaming_oauth_retry_records_one_usage_row`. Expected: `1 passed` (measured, about 40 s): the base writes the successful attempt's row only; a version that records the failed attempt writes two (`ValueError: too many values to unpack`).
- [ ] **Step 4: The handler** (`streaming_utils.py`):
  - After `generation_snapshot` (`:1213-1220`), add the method:

```python
    def _save_callback_kwargs(self, save_callback: Callable[..., Any]) -> dict[str, Any]:
        """The keywords ``save_callback`` accepts: the generation snapshot and what the stream received."""
        kwargs: dict[str, Any] = {}
        if _accepts_keyword(save_callback, "generation"):
            kwargs["generation"] = self.generation_snapshot()
        if _accepts_keyword(save_callback, "stream_usage"):
            kwargs["stream_usage"] = self.stream_usage
        return kwargs
```

  - In `safe_stream_generator`'s docstring (`:1476-1479`), replace the last two lines of the `save_callback` entry with:

```
                ``generation_snapshot()``. When it accepts ``stream_usage`` it
                receives the provider's usage block and the characters received.
```

  - In the non-SSE branch of `process_line` (`:1757-1759`), after `first_output_pending = True`, add one line at the same indentation:

```python
                    self.stream_usage.chars += len(text_piece)  # counted for the usage row, like a content delta
```

  - At the save site (`:2107-2113`), replace the seven lines from `generation_kwargs: dict[str, Any] = (` through `generation_kwargs["provider_usage"] = dict(self.stream_usage.usage)` with one line:

```python
                    generation_kwargs = self._save_callback_kwargs(save_callback)
```

  - In `create_streaming_response_with_timeout`'s docstring (`:2264`), replace the `finalize_callback` line with:

```
        finalize_callback: Optional callback invoked on error/cancel to finalize state; when it
            accepts ``stream_usage`` it receives what the stream received, for its usage row
```

  - In `finalize_once` (`:2308-2312`), replace `maybe_result = finalize_callback(success=success, cancelled=cancelled, error=error,)` (five lines) with:

```python
            kwargs: dict[str, Any] = {"success": success, "cancelled": cancelled, "error": error}
            if _accepts_keyword(finalize_callback, "stream_usage"):
                kwargs["stream_usage"] = handler.stream_usage  # what the stream received, for its usage row
            maybe_result = finalize_callback(**kwargs)
```

  Run the handler tests: `PATH="$(dirname $PY):$PATH" TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Chat/unit/test_streaming_generation_settlement.py`. Expected: `25 passed` (21 on the base + 4).
- [ ] **Step 5: The chat pipeline** (`chat_service.py`):
  - Imports: change `:207` to `from tldw_Server_API.app.core.LLM_Calls.usage_accounting import _approx_tokens, _StreamUsage, _usage_tokens` (ruff's order), and insert `from tldw_Server_API.app.core.Usage.llm_accounting import submit_accounting` before `from tldw_Server_API.app.core.Usage.llm_call_context import call_credential_source` (`:218`).
  - After `_maybe_refund_streaming_rg` (`:3991-4005`), add the module-level function:

```python
async def _record_stream_usage(
    *,
    request: Any,
    provider: str,
    model: str,
    templated_llm_payload: list[dict[str, Any]],
    conversation_id: str | None,
    llm_start_time: float,
    credential_source: str | None,
    stream_usage: _StreamUsage | None,
    full_reply: str = "",
    rg_commit_cb: Callable[[int], Any] | None,
) -> None:
    """Write the one usage row of a /chat/completions stream and commit its tokens; never raises.

    The row comes from the provider's usage chunk when one arrived, else from the prompt estimate and what
    the stream received: content, reasoning and tool-call arguments, as the provider-level recorder counts
    them. A side the chunk does not count comes from the estimate. The same tokens go to the Resource
    Governor and billing through ``rg_commit_cb``, however the stream ended (TASK-13953).
    """
    total_tokens = 0
    try:
        provider_usage = stream_usage.usage if stream_usage is not None else None
        reported_prompt, reported_completion = _usage_tokens(provider_usage) if provider_usage else (0, 0)
        reported = (reported_prompt, reported_completion) != (0, 0)
        prompt_tokens = reported_prompt
        if not prompt_tokens:
            try:
                prompt_tokens = _estimate_tokens_from_messages(templated_llm_payload)
            except _CHAT_NONCRITICAL_EXCEPTIONS:
                prompt_tokens = 0
        received_chars = stream_usage.chars if stream_usage is not None else len(full_reply or "")
        completion_tokens = reported_completion or _approx_tokens(received_chars)
        user_id = None
        api_key_id = None
        try:
            if request is not None and hasattr(request, "state"):
                user_id = getattr(request.state, "user_id", None)
                api_key_id = getattr(request.state, "api_key_id", None)
        except _CHAT_NONCRITICAL_EXCEPTIONS:
            pass
        total_tokens = int(prompt_tokens + completion_tokens)
        await log_llm_usage(
            user_id=user_id,
            key_id=api_key_id,
            request=request,
            endpoint=(f"{request.method}:{request.url.path}" if request else "POST:/api/v1/chat/completions"),
            operation="chat",
            provider=provider,
            model=model,
            status=200,
            latency_ms=int((time.time() - llm_start_time) * 1000),
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=total_tokens,
            request_id=(request.headers.get("X-Request-ID") if request else None) or (get_request_id() or None),
            conversation_id=(str(conversation_id) if conversation_id is not None else None),
            usage_metadata=provider_usage if reported else None,
            estimated=None if reported and (prompt_tokens, completion_tokens) == (reported_prompt, reported_completion) else True,
            estimate_source=None if reported else "stream_estimate",
            credential_source=credential_source,
        )
    except _CHAT_NONCRITICAL_EXCEPTIONS:
        pass
    finally:
        # Commit the recorded tokens to the Resource Governor and billing whatever the write did: a commit skipped
        # on a failed write would be a quota bypass
        try:
            if callable(rg_commit_cb):
                res = rg_commit_cb(total_tokens)  # may be async or sync
                if hasattr(res, "__await__"):
                    await res  # type: ignore[misc]
        except _CHAT_NONCRITICAL_EXCEPTIONS:
            pass
```

  - In `execute_streaming_call`, after `stream_failure_recorded = False` (`:4184`), add `stream_usage_recorded = False  # one usage row per stream, however it ends (TASK-13953)`.
  - Before `async def save_callback(` (`:5010`), add the nested helper:

```python
    def _stream_usage_recorder(
        stream_usage: _StreamUsage | None, full_reply: str = ""
    ) -> Callable[[], Awaitable[None]] | None:
        """The write of this stream's one usage row and token commit, or None once it is claimed (TASK-13953)."""
        nonlocal stream_usage_recorded
        if stream_usage_recorded:
            return None
        stream_usage_recorded = True
        return partial(
            _record_stream_usage,
            request=request,
            provider=selected_provider,
            model=model,
            templated_llm_payload=templated_llm_payload,
            conversation_id=final_conversation_id,
            llm_start_time=llm_start_time,
            credential_source=usage_credential_source,
            stream_usage=stream_usage,
            full_reply=full_reply,
            rg_commit_cb=rg_commit_cb,
        )
```

  - In `save_callback`'s signature (`:5016`), replace `provider_usage: dict[str, Any] | None = None,` with `stream_usage: _StreamUsage | None = None,`.
  - After `nonlocal stream_metrics_recorded` (`:5018`, the first line of `save_callback`'s body), insert:

```python
        # The usage row and the token commit, once per stream, detached: a client drop or a failure during the
        # save below must not lose it (an early end records in _finalize_stream)
        record = _stream_usage_recorder(stream_usage, full_reply)
        if record is not None:
            submit_accounting(record)
```

  - Delete the whole usage block of `save_callback` (`:5137-5190`: from the comment `# Usage logging after the stream completes: ...` through the `except _CHAT_NONCRITICAL_EXCEPTIONS: pass` that follows the `rg_commit_cb` call, 54 lines); `# Audit success` then follows the tool-autoexec `except` directly.
  - `_finalize_stream` (`:5510-5538`): change its signature to

```python
            async def _finalize_stream(
                *, success: bool, cancelled: bool, error: bool, stream_usage: _StreamUsage | None = None
            ) -> None:
```

    replace the six-line `if cancelled: ... error_type = "stream_incomplete"` chain (`:5515-5520`) with the one line

```python
                        error_type = "stream_cancelled" if cancelled else "stream_error" if error else "stream_incomplete"
```

    and replace its tail (`:5533-5538`, from `if success:` to the end) with:

```python
                if success:
                    return
                if not (stream_usage and (stream_usage.usage is not None or stream_usage.chars)):
                    # Nothing came back (the endpoint may be retrying this attempt): refund, as before (TASK-13953)
                    await _maybe_refund_streaming_rg(rg_refund_cb, cancelled=cancelled, error=error)
                    return
                # Recorded and committed, never refunded: the stream was read (TASK-13953). Detached, because a
                # cancelled stream's teardown may cut any await here.
                record = _stream_usage_recorder(stream_usage)
                if record is not None:
                    submit_accounting(record)
```

  - Check the sizes: `$PY - <<'EOF'` with the snippet below must print `execute_streaming_call 1521`, `tracked_streaming_generator 296`, `safe_stream_generator 769`, `create_streaming_response_with_timeout 255`:

```python
import ast
for path, names in {"tldw_Server_API/app/core/Chat/chat_service.py": {"execute_streaming_call", "tracked_streaming_generator"},
                    "tldw_Server_API/app/core/Chat/streaming_utils.py": {"safe_stream_generator", "create_streaming_response_with_timeout"}}.items():
    for node in ast.walk(ast.parse(open(path).read())):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            print(node.name, node.end_lineno - node.lineno + 1)
```

  Run Steps 1-2's tests again: the `-k` command of Step 3. Expected: `14 passed, 43 deselected`. Then the three files: `PATH="$(dirname $PY):$PATH" TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/LLM_Calls/test_usage_accounting.py tldw_Server_API/tests/Chat/unit/test_streaming_generation_settlement.py tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py`. Expected: `95 passed`. Then the endpoint pin with its sibling: `PATH="$(dirname $PY):$PATH" TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Chat/integration/test_chat_endpoint_simplified.py -k "streaming_oauth_retry_records_one_usage_row or streaming_lazy_openai_oauth_refreshes_once_before_output"`. Expected: `2 passed`.
- [ ] **Step 6: Update the tests that pinned the gap** (V8). Each asserted that a stream which fails after it starts writes no usage row, or that finalize gets exactly three keywords. The two whose stream received nothing (`test_real_queue_done_then_adapter_failure_is_atomic_and_never_marks_success`, `:6721`, and `test_real_queue_post_done_junk_is_ignored_on_normal_exhaustion`, `:6998`) keep their `usage_log.assert_not_awaited()` untouched (Decision 2).
  - `tldw_Server_API/tests/Chat/unit/test_streaming_utils.py`: add `from tldw_Server_API.app.core.LLM_Calls.usage_accounting import _StreamUsage` after `from tldw_Server_API.app.core.LLM_Calls import bounded_daemon as bounded_daemon_module` (`:27`); at `:1506-1508` replace the three-line `assert finalized == [...]` with:

```python
        (outcome,) = finalized
        assert isinstance(outcome.pop("stream_usage"), _StreamUsage)  # what the stream received (TASK-13953)
        assert outcome == {"success": False, "cancelled": False, "error": True}
```

  - `tldw_Server_API/tests/Chat/unit/test_async_stream_task_capacity.py`: add `from tldw_Server_API.app.core.LLM_Calls.usage_accounting import _StreamUsage` after `from tldw_Server_API.app.core.LLM_Calls import bounded_daemon` (`:26`); at `:511-513` and `:571-573` replace the three-line `assert finalized == [...]` with the same three lines as above.
  - `tldw_Server_API/tests/Chat/unit/test_chat_service_fallback.py`:
    - add `from tldw_Server_API.app.core.LLM_Calls.usage_accounting import _StreamUsage` and `from tldw_Server_API.app.core.Usage.llm_accounting import drain_accounting` (in that order) after `from tldw_Server_API.app.core.LLM_Calls import gateway` (`:44`);
    - `test_execute_streaming_call_refund_cb_still_conditional` (`:6303-6381`) becomes `test_execute_streaming_call_commits_a_cancelled_stream_instead_of_refunding` with the docstring `"""A cancelled stream was read, so its estimate is recorded and committed; nothing is refunded (TASK-13953)."""`. Add `committed: list[int] = []` and `usage_log = AsyncMock(return_value=None)` after `refund_calls`; in its fake `_gen()`, replace `await finalize_callback(success=False, cancelled=True, error=False)` with

```python
                received = _StreamUsage()
                received.chars = len("hi")  # the stream delivered its text before the cancel
                await finalize_callback(success=False, cancelled=True, error=False, stream_usage=received)
```

      then `monkeypatch.setattr(chat_service, "log_llm_usage", usage_log)` after the `get_request_queue` setattr, `prompt = [{"role": "user", "content": "hello there"}]` after `request = SimpleNamespace(...)`, pass `"messages_payload": prompt` and `templated_llm_payload=prompt` (both were `[]`), add `rg_commit_cb=committed.append,` before `rg_refund_cb=rg_refund_cb,`, add `await drain_accounting()` after the `finally` block that closes `agen`, and replace the first assert with:

```python
    assert refund_calls == []
    usage_log.assert_awaited_once()
    row = usage_log.await_args.kwargs
    assert (row["estimate_source"], row["prompt_tokens"]) == ("stream_estimate", chat_service._estimate_tokens_from_messages(prompt))
    assert committed == [row["total_tokens"]]
```

    - `test_concurrent_stream_semantic_gate_is_request_local` (`:6579-6582`): replace `usage_log.assert_awaited_once()` and the `conversation_id` assert with:

```python
    await drain_accounting()
    rows = {call.kwargs["conversation_id"]: call.kwargs for call in usage_log.await_args_list}
    if invalid_outcome == "missing-finish-eof":
        # The invalid stream delivered its text, so it is recorded too, from its estimate (TASK-13953).
        assert sorted(rows) == [f"stream-semantic-{invalid_outcome}-{index}" for index in range(2)]
        assert rows[f"stream-semantic-{invalid_outcome}-0"]["estimate_source"] == "stream_estimate"
    else:
        # Its only chunk is rejected before anything is counted: nothing received, no row, as before.
        assert sorted(rows) == [f"stream-semantic-{invalid_outcome}-1"]
    assert rows[f"stream-semantic-{invalid_outcome}-1"]["usage_metadata"] == {"prompt_tokens": 3, "completion_tokens": 2}
```

    - `test_real_queue_combined_output_done_late_error_never_records_provider_use` (`:6881`): replace `usage_log.assert_not_awaited()` with

```python
    await drain_accounting()
    usage_log.assert_awaited_once()  # read, so recorded, although it failed (TASK-13953)
    assert usage_log.await_args.kwargs["estimate_source"] == "stream_estimate"
```

    - `test_direct_done_then_adapter_failure_records_valid_output_without_clean_success` (`:7141`): replace `usage_log.assert_not_awaited()` with

```python
    await drain_accounting()
    if expected_content is None:
        usage_log.assert_not_awaited()  # nothing received, so nothing recorded, as before
    else:
        usage_log.assert_awaited_once()  # read, so recorded, although it failed (TASK-13953)
        assert usage_log.await_args.kwargs["estimate_source"] == "stream_estimate"
```

    - `test_concurrent_direct_failure_and_queued_success_record_provider_use_once` (`:7271-7272`): replace `usage_log.assert_awaited_once()` and the `conversation_id` assert with:

```python
    # The faulty stream was read too, so it is recorded from its estimate (TASK-13953).
    await drain_accounting()
    rows = {call.kwargs["conversation_id"]: call.kwargs for call in usage_log.await_args_list}
    assert sorted(rows) == ["direct-overlap-faulty", "direct-overlap-healthy"]
    assert rows["direct-overlap-faulty"]["estimate_source"] == "stream_estimate"
```

  Run: `PATH="$(dirname $PY):$PATH" TLDW_TEST_NO_DOCKER=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Chat/unit/test_streaming_utils.py tldw_Server_API/tests/Chat/unit/test_async_stream_task_capacity.py tldw_Server_API/tests/Chat/unit/test_chat_service_fallback.py -k "sync_bridge_sanitizes_arbitrary_daemon_failures_once or capacity_failure or commits_a_cancelled_stream or semantic_gate_is_request_local or done_then_adapter_failure or done_late_error_never_records or post_done_junk or direct_failure_and_queued_success or TestSafeStreamGenerator or TestCreateStreaming"`. Expected: `82 passed, 1 skipped, 409 deselected` (measured; 83 of 492 selected). Before Step 6 the same command fails 10 ids on the new code (V8's, less the two renames outside this selection); the rewritten tests on the unchanged code fail 11 (`11 failed, 71 passed, 1 skipped`: the 10 plus `..._commits_a_cancelled_stream_...`), both measured.
- [ ] **Step 7: Ruff and sizes.** `$(dirname $PY)/ruff check tldw_Server_API/app/core/LLM_Calls/usage_accounting.py` and the six test files: "All checks passed!". `tldw_Server_API/app/core/Chat/chat_service.py` and `tldw_Server_API/app/core/Chat/streaming_utils.py` each keep their one pre-existing finding (`I001`; Global Constraints): `git show origin/dev:<file> | $(dirname $PY)/ruff check --stdin-filename <file> -` prints "Found 1 error." before, and `$(dirname $PY)/ruff check <file>` the same after. The ratchet: `PATH="$(dirname $PY):$PATH" $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/lint/test_function_size_ratchet.py`: `13 passed`.
- [ ] **Step 8: Commit** (message 2): `git add tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/app/core/Chat/streaming_utils.py tldw_Server_API/tests/Chat/unit/test_streaming_generation_settlement.py tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py tldw_Server_API/tests/Chat/integration/test_chat_endpoint_simplified.py tldw_Server_API/tests/Chat/unit/test_chat_service_fallback.py tldw_Server_API/tests/Chat/unit/test_streaming_utils.py tldw_Server_API/tests/Chat/unit/test_async_stream_task_capacity.py && git diff --cached --check`.

## Task 3: Docs, verify, close, review, PR

- [ ] **Step 1: ADR-067** (`Docs/ADR/067-llm-calls-accounted-in-provider-base-class.md`). Replace the `- **Streams.**` bullet (`:32`) with:

```markdown
- **Streams.** A stream is recorded once by this recorder, however it ends: completion, a provider error or an early close. A stream that is opened and never read writes no row, because nothing was received. The chat pipeline's `/chat/completions` stream, which records itself, is recorded the same way since TASK-13953: once, however it ends (completion, a client drop, a stop, or a provider error once something was received), from the usage chunk when one arrived, else from the prompt estimate and what was received; its Resource Governor reservation and billing commit those tokens instead of refunding them. A stream that ends before anything was received (a provider error, a drop or a stop before the first token) still refunds and writes no row, as this recorder does for an unread stream; the endpoint retries a failed attempt on the same reservation, and its close of that attempt reaches the pipeline as such an end (TASK-15420 tracks charging the prompt of a client drop before the first token). The row is written by a detached task, so a stream dropped while the server shuts down can lose it, as this recorder's can. Before, it wrote no row when the client dropped it or it failed after it started (found by TASK-13951), so a client could read a reply and close on `finish_reason` to keep it off its limits.
```

  and replace the `- **Failures stay out of the call.**` bullet (`:36`) with these two:

```markdown
- **Failures stay out of the call.** An error in the recorder is logged and never reaches the caller. A non-finite usage value is skipped like a missing one.
- **A usage block that counts one side only** (prompt tokens only, say) takes the other side from the estimate, and the row is marked estimated, as the chat pipeline does (TASK-13953). Before, the missing side counted as zero.
```

- [ ] **Step 2: `Docs/Operations/Usage_Quotas.md`** (`:115`, the `- **Streams without a usage chunk, and router calls.**` bullet). Replace `the prompt estimate plus a quarter of the reply's characters (on `/chat/completions`, a tool-call-only reply counts no completion tokens). That includes` with `the prompt estimate plus a quarter of the characters received (content, reasoning and tool-call arguments, on `/chat/completions` as on every other caller since TASK-13953). That includes`, and replace `On `/chat/completions`, a usage chunk that counts only one side (prompt tokens only, say) takes the other side from that estimate, and the row is marked estimated. A `/chat/completions` stream that the client drops, or that fails after it starts, writes no usage row, and its Resource Governor reservation is refunded. An auto-router call` with `A usage block that counts only one side (prompt tokens only, say) takes the other side from that estimate, and the row is marked estimated, on `/chat/completions` and on every other caller. A `/chat/completions` stream that the client drops, that is stopped, or that fails after something was received writes one usage row like a completed one (from the usage chunk if it arrived, else the estimate of what was received), and its Resource Governor reservation and billing commit those tokens; only a stream that ends before the provider answers (a failure, a drop or a stop before the first token) is refunded (TASK-15420 tracks charging the prompt of a client drop before the first token). An auto-router call`.
- [ ] **Step 3: Mirror.** `bash Helper_Scripts/refresh_docs_published.sh`; `git status --short Docs/Published` must list exactly `Docs/Published/ADR/067-llm-calls-accounted-in-provider-base-class.md` (measured).
- [ ] **Step 4: Commit** (message 3): `git add Docs/ADR/067-llm-calls-accounted-in-provider-base-class.md Docs/Published/ADR/067-llm-calls-accounted-in-provider-base-class.md Docs/Operations/Usage_Quotas.md && git diff --cached --check`.
- [ ] **Step 5: Verify.** Run each of these. The expected results were measured on the prototype unless marked:
  - The focused set (Test commands): the three files with new tests `95 passed`; the six files together `586 passed, 1 skipped`; the endpoint pin with its sibling `2 passed`.
  - The wider set with label `final`, compared with Task 0's baseline: SQLite `1994 passed, 2 skipped, 36 errors` (0 failures, the 36 `mocker` errors), serial `88 passed`, PostgreSQL `138 passed`. Any other failure is this plan's.
  - `PATH="$(dirname $PY):$PATH" $PY -m pytest -p no:cacheprovider -q -n 8 tldw_Server_API/tests/Docs tldw_Server_API/tests/lint`: `501 passed` (measured, about 3.5 minutes; it includes the function-size ratchet and the `Docs/Published` mirror check). Then `PATH="$(dirname $PY):$PATH" $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/CI tools/backlog-py/tests`: `1114 passed, 5 skipped` (measured, about 3.5 minutes).
  - `PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/check_shard_coverage.py`: `new_uncovered=0` and "OK — no newly-unshared test files." (measured).
  - `PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/test_quality_triage.py --enforce`: exit 0, "OK — no offenses beyond the baseline." (measured).
  - `PATH="$(dirname $PY):$PATH" PYTHONPATH=. $PY Helper_Scripts/ci/route_auth_ratchet.py`: "208 unauthenticated routes; baseline exact." (measured).
  - `PATH="$(dirname $PY):$PATH" PYTHONPATH=. $PY Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json --check-paths apps/packages/api-types/paths.d.ts`: "OK — OpenAPI fingerprint matches the checked-in snapshot." and "OK — committed path union matches the canonical contract." (measured).
  - `git grep -n -E "chat_service.py|streaming_utils.py|usage_accounting.py|test_chat_endpoint_simplified.py" Helper_Scripts/ci/scope_predicate_baseline.txt`: nothing (measured).
  - `git diff origin/dev --stat -- tldw_Server_API/tests/lint/function_size_baseline.json tldw_Server_API/app/api/v1/endpoints/chat.py`: nothing.
  - `git diff origin/dev --check`: nothing.
- [ ] **Step 6: Bandit**, on the touched source:

```bash
/opt/homebrew/bin/uvx bandit -ll tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/app/core/Chat/streaming_utils.py tldw_Server_API/app/core/LLM_Calls/usage_accounting.py
```

  Measured: "No issues identified.", Medium 0, High 0.
- [ ] **Step 7: Close TASK-13953** (backlog-py; fill every `<...>` first):

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-13953 \
  --append-notes "How: one recorder per stream in chat_service.execute_streaming_call (_stream_usage_recorder claims the row once; _record_stream_usage writes it and calls rg_commit_cb in a finally). Both callers hand the write to submit_accounting, never awaited: save_callback at its top, before the message save, so a client that closes during the save still writes the row; _finalize_stream on a client drop, a stop or a provider error once something was received. _finalize_stream refunds and writes nothing when nothing was received (no usage chunk, no characters), whatever the ending: the endpoint retries a failed attempt on the same reservation and its close reaches the pipeline as a cancel without output. StreamingResponseHandler passes its _StreamUsage (usage chunk + chars of content, reasoning, tool-call arguments, and bare text) to save_callback and, through create_streaming_response_with_timeout.finalize_once, to the finalize callback as stream_usage (provider_usage is gone). usage_accounting._record_response/_record_stream fill a one-sided usage block from the estimate and mark the row estimated only when the fill replaced a zero; a non-finite value is estimated too." \
  --append-notes "RED/GREEN (record the real counts): Task 1 3 failed, 1 passed -> 38 passed; Task 2 12 failed, 2 passed -> 95 passed on the three files, the endpoint retry pin 1 passed on the base and after, and the 12 ids that pinned the gap updated and passing (test_chat_service_fallback.py x6 incl. parametrised, test_streaming_utils.py x2, test_async_stream_task_capacity.py x2, the two renames). Base verification of TASK-13951's finding: a drop after the first content chunk wrote no row and refunded once (cancelled=True); a drop during the save wrote nothing either. Wider set: SQLite <base> -> <after> (36 mocker errors both sides), serial <base> -> <after>, PostgreSQL <base> -> <after>. tests/Docs+lint+CI+backlog-py <n> passed. Shard coverage new_uncovered=0; test-quality ratchet OK; route-auth exact; OpenAPI unchanged. Function sizes: execute_streaming_call 1545 -> 1521, safe_stream_generator 774 -> 769, tracked_streaming_generator 296, create_streaming_response_with_timeout 255." \
  --append-notes "Decisions: a stream that ends before anything was received (error, drop or stop before the first token) refunds and writes no row, as ADR-067's recorder does for an unread stream; recording it would charge the prompt twice on the endpoint's in-request retry (OAuth refresh, fallback provider), and the pipeline cannot tell that retry's close from a client drop before the first token; TASK-15420 (filed here) tracks charging the latter's prompt. A failure before the stream object exists still records nothing and refunds. The row's status stays 200. Docs: ADR-067 (Streams; non-finite; one-sided block) + mirror, Docs/Operations/Usage_Quotas.md. Bandit: <result>." \
  --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "A /chat/completions stream now writes exactly one llm_usage_log row however it ends once something was received: on completion from save_callback, on a client drop, a stop or a provider error from _finalize_stream, in both cases written detached so cancellation cannot cut it. The row is the provider's usage chunk when it arrived, else the prompt estimate plus the characters received (content, reasoning, tool-call arguments), and its tokens are committed to the Resource Governor and billing instead of refunded; an own-key stream keeps credential_source=user. A stream that ends before anything was received still refunds and writes no row, so the endpoint's retry of a failed attempt is charged once. The provider-level recorder takes a side a usage block does not count from its estimate. ADR-067 and Usage_Quotas updated." \
  -s Done
```

  Then `git grep -n -E "<(answer|base|after|n|result)>" -- "backlog/tasks/task-13953 - *.md"` must print nothing (other task files carry such placeholders in their notes, so the grep is scoped to this task's file); `git add -A backlog/tasks && git diff --cached --check`; commit (message 4).
- [ ] **Step 8: Review before the PR.** Run `/superpowers:requesting-code-review` in a clean subagent on `origin/dev..HEAD`. Give it this plan's Review Focus. Address the findings before opening the PR, and record each ruling in the ledger.
- [ ] **Step 9: Finish the PR and merge.**
  - `git push` (the branch and the draft PR exist since Task 0). Replace the draft's body with `gh pr edit <n> --body-file <file>`, then `gh pr ready <n>`.
  - The body states: the gap (TASK-13951's finding, the cost-limit bypass, the save window); the design (one recorder, detached from both callers; what the handler hands over; the nothing-received refund and why; the recorder's one-sided fill); the Decisions (Decision 2 in full, with the residual and the follow-up TASK-15420); the Review Focus items; the measured counts; the `## Change summary` (or the waiver; Decision 1).
  - It ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
  - Merging: `gh variable get MERGE_QUEUE` (unset when this plan was written). When reviews are settled and the Change summary is in, run `QUEUE="<n>" sh <plans dir>/pr_queue3.sh` in the background. It reads the required checks from dev's ruleset (8 today), rebases server-side when BEHIND, and merges when all pass. Wait for each body edit's re-triggered license status before merging. Confirm MERGED before deleting the branch.

## Commit messages

Each ends with a blank line and `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

- 0: `chore(backlog): start TASK-13953 and add its plan`
- 1: `fix(usage): a one-sided usage block takes its missing side from the estimate (TASK-13953)`
- 2: `fix(chat): a /chat/completions stream writes one usage row however it ends, and commits it (TASK-13953)`. Body: the recorder, the detached write from both callers, the nothing-received refund and the retry it protects, the handler's `stream_usage`, the updated tests and why.
- 3: `docs(usage): a dropped /chat/completions stream is recorded and committed (TASK-13953)`
- 4: `chore(backlog): close TASK-13953`

## Not measured

- The endpoint through HTTP over a real socket: `test_streaming_oauth_retry_records_one_usage_row_for_the_successful_attempt` drives the endpoint through Starlette's `TestClient` (the app's loop in the portal thread, the real `StreamingResponse` and `create_streaming_response_with_timeout`); the chat-pipeline tests cancel the task that reads `response.body_iterator`, which is the production generator; Starlette's anyio cancel scope is exercised by the existing D7 test `test_partial_saved_when_anyio_scope_cancels_the_response_task` for the same `finally`.
- The retry through a fallback provider (`endpoints/chat.py:5857-5894`): the same refund path as the OAuth retry, through the same `_close_provider_stream_response`; only the OAuth variant is pinned.

## Folded after the plan review

Each finding of `review_plan_13953.md`, and what changed:

- **B1 (Blocker, fixed).** `save_callback` claims the record and hands it to `submit_accounting` at its top, before any await; the three lines at its end are gone. Pinned by `test_a_client_drop_during_the_save_still_writes_one_row_and_commits_its_tokens` (the review's P3: the reader cancelled while `save_message_fn` blocks), which fails on the base and on the first plan's code. `drain_accounting()` added to the four TASK-13951 tests and the reasoning test.
- **I1 (Important, fixed, rule changed).** `_finalize_stream` refunds and writes nothing when nothing was received. The review's rule (`error and not cancelled`) was tried first and failed the retry pin the controller asked for: the endpoint closes the failed attempt before the handler has processed its error, so the finalize arrives as `cancelled=True, error=False` (measured with the finalize keywords captured). The rule is now "nothing received, whatever the ending", which also refunds a drop or stop before the first token (Decision 2, flagged for the owner). The `before-output` case became `test_a_stream_that_fails_before_any_output_is_refunded_and_writes_no_row`, plus `test_a_stream_dropped_before_any_output_is_refunded_and_writes_no_row` and the endpoint-level `test_streaming_oauth_retry_records_one_usage_row_for_the_successful_attempt` (stubbed adapter, no network). Step 6 rewrites 7 chat-level ids (6 forced by the code and `commits_a_cancelled_stream`, whose fake finalize now passes a read `_StreamUsage`); the two tests whose stream received nothing keep their base assertions. `tracked_streaming_generator` is 296: the branch is paid for by a one-line `error_type` expression.
- **M1 (fixed).** `estimated` is set only when a fill replaced a zero side, in `_record_response`, `_record_stream` and `_record_stream_usage`. Pinned by `test_a_zero_side_the_estimate_agrees_with_is_not_marked_estimated`.
- **M2 (fixed).** Every `Expected:` line re-measured on `576feef820`: Task 1 Step 4 `38 passed`, Task 2 Step 4 `25 passed`, Step 6 `82 passed, 1 skipped`, and the new counts of the folds.
- **M3 (fixed).** The close-out placeholder grep is scoped to `backlog/tasks/task-13953 - *.md`.
- **M4 (fixed).** The three finalize-keyword assertions check `isinstance(outcome.pop("stream_usage"), _StreamUsage)` instead of `ANY`.
- **M5 (fixed).** The commit runs in `_record_stream_usage`'s `finally`. Pinned by `test_the_tokens_are_committed_even_when_the_row_write_fails` (a driver error outside `_CHAT_NONCRITICAL_EXCEPTIONS`; fails on the base).
- **M6 (fixed).** One clause in ADR-067's Streams bullet: the detached write of a stream dropped during a server shutdown can be lost, as the provider-level recorder's can.
- **M7 (kept).** `status` stays 200; the Decisions say why.
- **Mechanics.** Base moved to `576feef820` (#3388 landed: `chat_service.py` citations shift by −4 before `:180`, and the file now carries a pre-existing `I001`); #3407 and #3411 re-checked for overlap. The commit trailer is unchanged.

## Folded after the second review

The original reviewer re-reviewed the fold (`review2_plan_13953.md`, plans dir) on `576feef820`: 0 Blocker, 1 Important, 4 Minor. It verified B1 by effect (P3: one row, committed; the pin is RED on the base and on the first design; no double write on completion, stop or two concurrent drops), confirmed with its own endpoint probe that a retry's close and a client drop before the first token reach the pipeline as the same ending, and matched every re-measured `Expected:` line. Folded by the controller (plan text only; `proto_13953.patch` predates these and differs only in the ADR-067 and Usage_Quotas sentences that now name TASK-15420):
- **I1 (wording + follow-up):** Decision 2 gives the real reason (no pipeline signal separates the two endings), sizes the residual (a cost leak with no benefit to the client; it does not reopen the bypass), and Task 0 Step 2b files TASK-15420 (the endpoint's "superseded" flag, with the OAuth retry pin as guard and a fallback-provider retry pin). ADR-067, Usage_Quotas, the close-out note and the PR body name it.
- **M10:** the fallback-provider retry pin is an AC of TASK-15420.
- **M11:** Decision 2 says the two rows were the rejected rule's outcome; the "12 ids" line names the 2 + 4 + 6 and the re-pointed seventh chat-level test.
- **M8 (not folded):** the nothing-received refund is awaited inside `_finalize_stream` on a cancel; same exposure as the base, named in Review Focus 2.
- **M9 (not folded):** the endpoint retry pin unpacks the detached row without a drain; it passed every run (fold and both reviews). If it flakes, poll for the row as `_settled` does.
