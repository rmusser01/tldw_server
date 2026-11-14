# Benchmark runs and watchlist output summaries run on the user's stored key (TASK-15426, PR B of 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Both benchmark run routes and `POST /watchlists/outputs` with `summarize` run their LLM calls on the user's own stored key for the request's provider, and their boundaries name that provider, so an over-limit user whose calls all run on their own key is served unmarked; every other user gets the same 402 at the same point.

**Architecture:** PR A's `enforce_llm_quota_on_own_key(user_id, provider, model=None)` replaces each boundary's `enforce_llm_quota(user_id)` in one line and returns the handle for the user's own key (or None, which keeps today's server-key path byte for byte). The handle is passed down to `Summarization_General_Lib.analyze(..., provider_credentials=...)`, which already binds it on every call (`Summarization_General_Lib.py:248-352`): through `EvaluationManager.evaluate_custom_metric` and `_generate_model_response` for benchmarks (both the model's response and the judge's score, TASK-15485), and through `summarize_items_for_output` → `_summarize_single_article` → `_llm_reply` for watchlists. A benchmark request that carries a raw `api_key` names no provider: that key counts, so an over-limit user is refused, as before.

**Tech Stack:** Python 3.12, FastAPI, pytest.

**Spec:** the TASK-15426 task file (ACs #2-#3 for these two families); ADR-083 and its amendments; the brief `brief_plan_15426.md`; `plan_15426_a.md` (the helpers, the split, Decisions 2-4); TASK-15485 (#3445, the benchmark judge fix this PR builds on).

**Measured on:** `7b4cb653b4`, a simulated dev after A and the judge fix: `origin/dev` `dcdef5355d`, then `git merge --no-ff` of PR A (#3444, `feat/own-key-study-assistants-15426` at `8315fa68cf`) and of the judge fix (#3445, `fix/benchmark-judge-15485` at `4b85b81a2c`), in that order; neither merge conflicted. Since then PR A has merged (`origin/dev` `06a262b132`) and #3445 was rebased (`31da7ec215`); the second review fold measured on `3160d6392a` (`06a262b132` + `git merge --no-ff` of #3445 at `31da7ec215`, no conflict), whose tree is identical to `7b4cb653b4` (`git diff` empty), so every citation and base count below holds for both. Prototype: `proto_15426_b.patch` (plans dir; 20 files, +292/-95, 9 of them docs); it applies to `3160d6392a` and `7b4cb653b4`. `file:line` citations are for `7b4cb653b4`. History: written on `cb70f0c5d1` + A, folded on `7a020da14a` (see "Review fold"), refreshed onto this base on 2026-11-10 (see "Refresh (2026-11-10)"), and folded again the same day (see "Review fold 2 (2026-11-10)" at the end).

**Order:** after PR A (#3444, merged 2026-11-10) and the benchmark judge fix (TASK-15485, #3445) have merged (owner, 2026-11-07: A, then the judge fix, then B). Independent of C and D (disjoint source files); rebase the docs hunk on whichever of them merged first.

---

## What was verified (on `7b4cb653b4`)

| # | Question | Finding | Consequence |
|---|---|---|---|
| V1 | Benchmark routes | `/evaluations/benchmarks/{name}/run` (`evaluations_benchmarks.py:101`) and `/benchmarks/{name}/run` (`benchmark_api.py:205`) call `enforce_llm_quota(user)` with no provider (`:109`, `:226`), then `evaluation_manager.evaluate_custom_metric(..., api_name=request.api_name[, api_key=request.api_key])` per item (`:158`, `:275`), and count an item whose score is None as `unscored` (`:182-196`, `:305-319`; TASK-15485). Only the standalone route has `api_key` (`benchmark_api.py:52`). Neither function is frozen. | One provider per run, `api_name` (default `openai`). |
| V2 | The manager's calls | `evaluate_custom_metric` (`evaluation_manager.py:547`) binds the model-response call as `model_call = self._generate_model_response(...)` (`:560-564`; `_generate_model_response` `:757-778`) when an input value is `{model_response}` and awaits it in a `try` (`:565-570`), then makes the judge call (`:596-605`, `input_is_literal_text=True` at `:604`); both are `asyncio.to_thread(analyze, ...)` with the raw `api_key` or None. | Pass the handle to both: inside the `model_call` binding, and beside `input_is_literal_text=True` (keep both). |
| V3 | Watchlist summaries | `create_output` (`watchlists.py:6785`, frozen at 910 lines, measured 910) calls `enforce_llm_quota(current_user.id)` (`:7119`) inside `if payload.summarize`, then `summarize_items_for_output(items, api_name=llm_provider, ...)` (`:7130`); that runs `_summarize_single_article` per item in `loop.run_in_executor` (`outputs_service.py:692-716`), which calls `_llm_reply` → `analyze(api_name, prompt, None, ...)` (`:652-677`). | Pass the handle explicitly: a default-executor hop carries the call context in production only (`Usage/llm_roots.py:32-51`), a parameter always. |
| V4 | Output enrichment | `briefing_summary`/grouping enrichment runs later as a Scheduler task (`core/Watchlists/output_enrichment_handler.py:178-320`) through `_summarize_text_block` → `_llm_reply`; it is not part of this boundary today (only `summarize` is checked) and its calls are checked where they run. | Unchanged: it keeps the server's key (Decision 3). |
| V5 | Two LLM calls per benchmark item (TASK-15485, #3445, in the base) | The judge's input now goes to `analyze` as literal text (`input_is_literal_text=True`, `evaluation_manager.py:604`), so the formatters' `question`/`response`/`correct_answer`/`choices` input (`benchmark_utils.py:135-137`, `:507-510`) reaches the provider: an item makes the model-response call (every formatter's `response` is `{model_response}`) and then the judge call. A model or judge call that fails (an `Error:` reply) or raises leaves the item with `score: None` (`:565-579`, `:609-611`, `:681-690`), which the routes count `unscored`, in no average; a quota refusal still raises (`:567-568`, `:681-682`). Measured: Task 1 Step 1's `[benchmark-formatter]` case, the real `simple_bench` formatter (`MultipleChoiceEvaluation`), makes two adapter calls, both on the handle. | Both calls carry the handle (Task 1 Step 4; Decision 4). A failed or raised call on the user's own key is unscored the same way; this PR changes no scoring. |
| V6 | RED on the base | 9 failed, 1 passed: both custom-metric cases and the summaries test (`TypeError`: no `provider_credentials`), and every route case but `[raw-api-key]` (`asked == []`: the base names no provider, so the own-key cases also get the 402 and the standalone `[own-key]` never reaches its 404); `[raw-api-key]` passes on the base, as it must | |
| V7 | Open PRs (2026-11-10) | `watchlists.py`: #2903 [`a8544d7ed7`, draft] at `:119` (imports) and `:1454`-`:8103`, not `:59`, `:7118-7136`; #3269's `:2438` hunk is already on dev (its head is unchanged). #3450 and #3451 (TASK-13950.20 B1, B2) rewrite `:61` (the `_pagination_utils` import), two lines below this PR's `:59`: measured with `git merge-file`, the two changes merge without a conflict, but this PR's patch context includes `:61`. `outputs_service.py`: #2903 rewrites `:10-28`, `:203-262`, `:527-635`; this PR's hunks are at `:652-716`, outside them. No open PR but #3445 (in the base) touches the benchmark routes or `evaluation_manager.py`. Of this PR's tests, only #3269 [`81b0bb5836`, CONFLICTING] touches one: it adds 22 lines to `test_watchlists_api.py` at `:305`, far from the `:1919` pin (review B-2). | Either order with the open PRs; if #2903 lands first, re-apply the `outputs_service.py` hunks by hand (same function bodies); if #3450 or #3451 does, re-apply the `:59` import line by hand. |
| V8 | The standalone route's saved run (review M3) | `run_benchmark`'s `user_id: str = Depends(get_rate_limiter_dep)` (`benchmark_api.py:208`) is the AuthNZ `RateLimiter` (`auth_deps.py:391-394`), stored as the metadata's `user_id` (`:350`) when `save_results` (default true, `:54`); `store_evaluation`'s `json.dumps` (`evaluation_manager.py:343`) raises `TypeError: Object of type RateLimiter is not JSON serializable`, so every run over HTTP with the default gets `500 Failed to run benchmark` after all its LLM calls, and nothing is saved (measured: Task 1 Step 2's new test). Behind it, the route stores through the module-level `evaluation_manager = EvaluationManager()` (`:29`), which is the single-user id's database (`evaluation_manager.py:51-54`, `DatabasePaths.get_single_user_id()`): once the dump works, every user's run would land in that one user's database. The unified route makes a manager per user and stores `identity.created_by` (`evaluations_benchmarks.py:111-113`, `:217`). `evaluate_simpleqa` (`benchmark_api.py:381`) takes the same dependency but never reads it (a placeholder that stores nothing): no defect. | This PR fixes it (Decision 6): today the server's key pays for those failed calls; with this PR the user's own key would. |

## Decisions

1. **ADR-004 Change summary.** Waived for every TASK-15426 PR (owner, 2026-11-07, "Yes, waive for all"); recorded as a full `## Change summary` section.
2. **A raw `api_key` keeps counting (ruling; brief default; owner decision 2026-11-07: keep).** The standalone route's `api_key` is not the user's stored key and carries no own-key mark: the route names `None` for such a request (refused over a limit, as today) and passes no handle (the raw key is used, as today). Alternatives: ignore a raw key and resolve the stored one (as `/evaluations/geval` does, `evaluations_unified.py:388-390`) silently changes what a caller asked for; treat a raw key as the user's own would let any string lift the limit. **Answered (owner, 2026-11-07): a raw `api_key` keeps counting.**
3. **Output enrichment stays on the server's key (scope ruling).** It is a separate Scheduler task the `summarize` boundary never covered; its calls are checked where they run and it stops at a refusal (`Usage_Quotas.md`, "Watchlist output enrichment"). Moving it needs a runtime per Scheduler task, like PR D's job. Recorded in the docs. **Answered (owner, 2026-11-07): out of scope** (with `/media/ingest-web-content`, `/media/process-web-scraping` and `rag_pipeline` evaluation runs).
4. **The judge bug was fixed before this PR, in its own PR (owner, 2026-11-07): TASK-15485 (#3445), in the base.** It is pre-existing and independent of keys; fixing it changed benchmark scores, which had its own review. This PR builds on its state:
   - **Two LLM calls per benchmark item, both on the handle.** An item makes the model-response call (every formatter's `response` is `{model_response}`) and the judge call (V5). Task 1 Step 4 passes `provider_credentials` to both: inside the `model_call = self._generate_model_response(...)` binding, and to the judge's `analyze` beside `input_is_literal_text=True` (keep both).
   - **Unscored stays unscored.** With the handle, a model or judge call on the user's own key that fails or raises leaves the item with `score: None`, counted `unscored`, exactly as on the server's key; a quota refusal still raises. This PR changes no scoring, and none of its tests or Expected lines assume a failed call scores 0.0 or 0.5 (the route tests stub `evaluate_custom_metric` with a scored reply; the unit test asserts the requests).
   - **Task 1 Step 1's test** runs a custom input and the real `simple_bench` formatter (`MultipleChoiceEvaluation`), so the judge call with benchmark-shaped input is shown to carry the handle: `[("sk-users-own-key", "user")] * 2` for each.
   - **The route tests** (Task 1 Step 2) stub `evaluate_custom_metric`; they did not change.
   - **Docs:** the ADR bullet's "(every item's model response and judge call)" holds on this base.
5. **`create_output` must not grow.** The boundary line becomes `own_key = await enforce_llm_quota_on_own_key(current_user.id, llm_provider, model=payload.llm_model)` with a shorter one-line comment above it, and the summary call puts `provider_credentials=own_key` on the line with `items_for_summary` and `api_name`: measured 909 lines (ceiling 910).
6. **The standalone route's saved run is fixed here, not filed (ruling, review fold 2, 2026-11-10; review M3; V8).** This PR moves that route's calls onto the user's own key, so it must not ship a run that spends them and then returns a 500. `run_benchmark` drops the rate-limiter parameter and, as the unified route does, makes the evaluation manager for the authenticated user and stores `identity.created_by` as `user_id`. The per-user manager is part of the fix, not an extra: with only the id fixed, every user's run would start landing in the single-user id's database (V8). `evaluate_simpleqa` keeps its unused rate-limiter parameter (no defect; out of scope).

## Global Constraints

Verbatim from the brief:
- Tests make no real outbound network requests; stub the provider adapter or its transport. A test that reaches a commercial provider through an endpoint calls the real API unless the adapter or call is stubbed. Show it with `tldw_Server_API/tests/helpers/offline_network.block_live_network()` or an equivalent socket probe.
- Function-size ratchet `tldw_Server_API/tests/lint/function_size_baseline.json`: frozen functions must not grow.
- Backlog:
  - edit tasks with backlog-py only (`PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task edit ...`);
  - `--append-notes`, never `--notes`; stage with `git add -A backlog/tasks`; never hand-edit task files;
  - a new task's explicit id comes from `zsh <plans dir>/task_next.sh` (prints the HIGHEST id in use; the new id is that plus one), after `mkdir <scratchpad>/id_lock`; release it once the PR's file list shows the task. Write ids as placeholders in the plan. Do NOT allocate ids yourself now.
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
  - A file listed in `Helper_Scripts/ci/scope_predicate_baseline.txt` whose lines shift fails backend-required (regenerate with `--write-baseline` in the same PR); check every file you touch.
  - `/usr/bin/python3 Helper_Scripts/ci/check_backlog_task_ids.py .` exit 0.
  - The Frontend E2E journeys are not required checks; the PR step diffs the PR's E2E failures against a recent dev run.
- Agent shells don't keep the venv `PATH`: prefix commands that spawn `python` with `PATH="$(dirname $PY):$PATH"`.

Plan-specific:
- `$PY` is `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`; `<plans dir>` is `/Users/macbook-dev/.claude/projects/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/plans`.
- `create_output` (`watchlists.py`) is frozen at 910 lines: Decision 5 keeps it at 909.
- `scope_predicate_baseline.txt` lists none of this PR's files (checked on `7b4cb653b4`, all 15; on `3160d6392a`, all 20).
- Carry rule (owner, for B through D): the own-key handle never goes inside a copied or persisted run config or job payload. Here it is a local (`own_key`) in each route, passed only as a keyword argument down to `analyze`: the benchmark routes' `store_evaluation(..., metadata={...})` (`evaluations_benchmarks.py:211-221`, `benchmark_api.py:341-353`; with Decision 6 the standalone one stores `api_name`, `identity.created_by` and `total_samples`) and `create_output`'s output metadata (`watchlists.py:7216-7288`) and enrichment payload (`:7685`; `output_enrichment_handler.py:52`) never see it, and the summaries' executor lambda (`outputs_service.py:709-716`) only closes over it in-process.
- Every new test that drives a route stubs the layer below the provider call (`evaluate_custom_metric`, `summarize_items_for_output`) or the adapter registry; none reaches a provider.
- Every test run in this plan runs with outbound connections blocked: `PYTHONPATH=<plans dir> NETPROBE_LOG=<file> $PY -m pytest -p netprobe_15427 ...`, or inside `block_live_network()`.
- `proto_15426_b.patch` is normative for docstrings and comments the plan does not quote; every docs sentence this PR adds or rewrites is quoted in Task 3 Step 1 (Usage_Quotas, BYOK, ADR-083, the benchmark guide, the Watchlists API).
- ruff: no new finding on a touched file against the base (as PR A, Task 3 Step 3).

## Review Focus

1. **A benchmark run with a raw `api_key`, over a limit, with a stored key for `api_name`.** Expected: the 402, no lookup. Pinned by `test_run_route_names_its_provider_to_the_llm_quota[raw-api-key]` (Task 1).
2. **An over-limit user on the server's key.** Expected: the same 402 before the dataset loads / before any summary, no output written. Pinned by `[server-key]` of the three route tests (Tasks 1-2).
3. **Under a limit with a stored key.** Expected: every LLM call carries the user's handle (model response and judge, for a custom input and a real benchmark formatter's; every summary). Pinned by `test_a_custom_metric_runs_both_calls_on_the_handle_it_is_given[custom-input]` and `[benchmark-formatter]`, and `test_summaries_run_on_the_handle_they_are_given`, which go through the real `analyze` to a stub adapter (Tasks 1-2). The summaries test runs over the limit with the check at the call switched on (review B-5), so a summary without the handle would be refused there.
4. **A benchmark that does not exist, own key.** Expected: the 404 after the boundary admits (the boundary still runs before the lookup). Pinned by `[own-key]` of the standalone route test (Task 1).
5. **Watchlist summary handle and the default executor.** Expected: each per-item call in the executor thread gets the handle as a parameter, not from context. Pinned by `test_summaries_run_on_the_handle_they_are_given` (two items, two requests with the handle) (Task 2).
6. **An own-key run with no model.** The handle path takes the model only from the handle's `app_config` (`Summarization_General_Lib.py:305-309`), not the `DEFAULT_MODEL_<PROVIDER>` environment fallback the server-key path has (review B-6). On a server whose only default model comes from that variable, an own-key benchmark or watchlist summary with no explicit model fails "Model is required". Pre-existing, shared by every handle caller, unlikely; not changed here, listed for PR D's follow-up list.
7. **A benchmark item whose own-key call fails** (a revoked key, a provider error, a raised transport error). Expected: the item is unscored (`score: None`, counted `unscored`, in no average), as on the server's key; not 0.0 or 0.5. That is TASK-15485's code, which does not depend on the key, and its tests pin it (`test_a_benchmark_item_is_scored_from_the_judges_reply[judge-error]`, `test_a_failed_model_answer_leaves_the_item_unscored`); this PR adds no scoring path.
8. **A standalone `/benchmarks/{name}/run` with the default `save_results`** (Decision 6). Expected: 200, and the run is stored in the calling user's own evaluations database with `user_id` their id; before this PR, a 500 after every LLM call. Pinned by `test_a_run_saved_by_default_is_stored_for_its_user` (Task 1), which also fails if the run lands in another user's database.

## Files

- Modify: `tldw_Server_API/app/core/Evaluations/evaluation_manager.py` (`evaluate_custom_metric` `:547-605`, `_generate_model_response` `:757-778`)
- Modify: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_benchmarks.py` (`:13` import, `:107-109`, `:158-165`), `tldw_Server_API/app/api/v1/endpoints/benchmark_api.py` (`:15-19` imports, `:28-29` the module-level manager, `:208` and `:220` the rate-limiter parameter, `:225-227`, `:275-283`, `:350`)
- Modify: `tldw_Server_API/app/services/outputs_service.py` (`_llm_reply` `:652-677`, `_summarize_single_article` `:680-689`, `summarize_items_for_output` `:692-716`), `tldw_Server_API/app/api/v1/endpoints/watchlists.py` (`:59` import, `:7118-7136`)
- Modify tests: `tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_api.py`, `tldw_Server_API/tests/Evaluations/test_benchmark_api_auth.py`, `tldw_Server_API/tests/Evaluations/test_benchmark_api_error_mapping.py`, `tldw_Server_API/tests/Evaluations/unit/test_evaluation_manager.py`, `tldw_Server_API/tests/Services/test_outputs_service.py`, `tldw_Server_API/tests/Watchlists/test_watchlists_api.py`
- Docs (Task 3): `Usage_Quotas.md`, the BYOK guide, ADR-083, `Docs/User_Guides/Server/Benchmark_Creation_API_WebUI_Extension_Guide.md`, `Docs/API-related/Watchlists_API.md` (+ the four mirrors)
- Create (Task 0): `Docs/superpowers/plans/2026-11-07-own-key-benchmarks-watchlist-summaries.md`

## Test commands

`$FOCUSED_B`:

```text
tldw_Server_API/tests/Evaluations/test_benchmark_api_auth.py::test_run_route_names_its_provider_to_the_llm_quota
tldw_Server_API/tests/Evaluations/test_benchmark_api_auth.py::test_a_run_saved_by_default_is_stored_for_its_user
tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_api.py::test_benchmark_run_names_its_provider_to_the_llm_quota
tldw_Server_API/tests/Evaluations/unit/test_evaluation_manager.py::test_a_custom_metric_runs_both_calls_on_the_handle_it_is_given
tldw_Server_API/tests/Services/test_outputs_service.py::test_summaries_run_on_the_handle_they_are_given
tldw_Server_API/tests/Watchlists/test_watchlists_api.py::test_output_summaries_name_their_provider_to_the_llm_quota
```

Measured on `3160d6392a` under `-p netprobe_15427`: RED (this PR's tests, the base's source) `10 failed, 1 passed (0:35)`; GREEN `11 passed (0:37)`; 0 netprobe attempts in both. (Before review fold 2, without `test_a_run_saved_by_default_is_stored_for_its_user`: RED `9 failed, 1 passed`, GREEN `10 passed`, on `7b4cb653b4`; before the refresh, also without the `[benchmark-formatter]` case: RED `8 failed, 1 passed`, GREEN `9 passed`, on `cb70f0c5d1` and `7a020da14a`.)

**The wider set:**

```bash
git grep -l -E 'evaluate_custom_metric|_generate_model_response|benchmark_api|evaluations_benchmarks|/benchmarks/[^"]*/run|summarize_items_for_output|_summarize_single_article|_llm_reply|watchlists/outputs' -- 'tldw_Server_API/tests/*.py' | grep '/test_[^/]*\.py$'
```

22 files on `7b4cb653b4`, the same 22 as on `cb70f0c5d1` (and on the prototype). Also run `tests/Evaluations/unit/test_evals_cli_benchmark_commands.py`: the benchmark CLI is the third caller of `evaluate_custom_metric` (`benchmark_cli.py:343`; it passes no handle, server key, no HTTP boundary) and its test does not name the symbol. The fourth caller, the `eval custom` CLI command (`tldw_Server_API/cli/commands/evaluation.py:743`), is already broken (it passes `text=`, `custom_prompt=` and `save_results=`, which the method does not take; TASK-15489), has no test, and this PR's new keyword argument does not change it. Measured on `7b4cb653b4` (these 23 files, `-n 4 --dist loadfile`, none takes a PostgreSQL fixture, `-p netprobe_15427`): base `1 failed, 526 passed, 4 skipped, 1 xpassed (2:59)`; prototype, before review fold 2, `1 failed, 533 passed, 4 skipped, 1 xpassed (3:44)` (+7: this PR's new cases less the three it replaces). With review fold 2, on `3160d6392a` (the same tree as `7b4cb653b4`, so the base count stands): prototype `1 failed, 534 passed, 4 skipped, 1 xpassed (3:37)` (+8: one more new case), the same probe artifact and the same two netprobe lines. The one failure, on both, is `test_security.py::TestWebhookSecurityValidation::test_ssrf_protection_private_ip`, a probe artifact (it resolves numeric IPs; it passes without the probe; known from #3445); the netprobe log holds the same two lines on both, that test and `test_blocked_ports` (blocked DNS lookups of the test's own hosts). No new failing id.

Before the refresh (history, not re-measured): 22 files on `cb70f0c5d1`. Measured as one run of the union of the four PRs' wider sets (183 files: 176 with `-n 8 --dist loadfile`, the 7 that take a PostgreSQL fixture serially with a session-unique `TEST_DB_NAME`), on `cb70f0c5d1` and on the four prototypes stacked. Base: 3 failed, 3751 passed, 67 skipped, 1 xfailed, 1 xpassed (44:38); PostgreSQL 99 passed, 59 skipped. The base failures: `test_media_processing.py::TestProcessPdfs::test_process_pdf_upload_success[docling]` and `::test_process_pdf_multi_status_mixed[docling]` (fail on both trees), and `test_evaluations_stage1_route_and_error_regressions.py::test_response_quality_capacity_failure_drains_admitted_siblings_before_runtime_close` (flaky: passes with the prototype). Prototype (a run started before the last fixes): 51 failed, 3733 passed, 65 skipped, 3 errors (53:08); PostgreSQL 108 passed, 63 skipped. Its new failures were, and are fixed in the patches: every own-key test (a `MOCK_OPENAI_BASE_URL` left by another test in the same worker made the test helper's handle `user_unmetered`: the helper's resolution now carries no base URL), eight existing study-assistant stubs and one evaluation-identity stub (see PR A Task 2, PR C Task 1); 22 in `test_email_offline_ingestion.py` were load (the file passes alone on both trees, 42 passed on the base), and the 3 errors were environment (a PostgreSQL teardown lock, a server-e2e health-check timeout, a fixture finalizer). Every file that failed, re-run on the final prototype with `MOCK_OPENAI_BASE_URL=http://127.0.0.1:9` set for the whole run: 531 passed, 30 skipped, 1 failed (a quizzes stub, fixed after: the quizzes file then 77 passed).

---

## Task 0: Plan, branch, draft PR, baseline

- [ ] **Step 1: Branch** after PR A (#3444) and the judge fix (#3445) merge: `git fetch origin dev && git merge-base --is-ancestor origin/feat/own-key-study-assistants-15426 origin/dev && git merge-base --is-ancestor origin/fix/benchmark-judge-15485 origin/dev && git switch -c feat/own-key-benchmarks-watchlists-15426 origin/dev` (both checks exit 0; if either branch was deleted after merging, check the PR shows MERGED instead).
- [ ] **Step 2: Note and plan file.** `PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-15426 --append-notes "Plan, PR B of 4 (benchmark runs, watchlist output summaries): Docs/superpowers/plans/2026-11-07-own-key-benchmarks-watchlist-summaries.md."`; add the plan file; `check_backlog_task_ids.py` exit 0; `git add -A backlog/tasks Docs/superpowers/plans && git diff --cached --check`; commit.
- [ ] **Step 3: Draft PR** (`--base dev`), title `feat(usage): benchmark runs and watchlist summaries run on the user's stored key (TASK-15426, 2 of 4)`.
- [ ] **Step 4: Baseline** the wider set (under the outbound blocker); record counts, failing ids and netprobe lines. Expected: as measured on `7b4cb653b4` (see "The wider set"): one failure, `test_security.py::TestWebhookSecurityValidation::test_ssrf_protection_private_ip`, a probe artifact (it resolves numeric IPs; it passes without the probe), and two netprobe lines (that test and `test_blocked_ports`). A failure outside that is checked against a recent dev run before it is called flaky.

## Task 1: Benchmark runs (ACs #2, #3)

**Files:** `evaluation_manager.py`, `evaluations_benchmarks.py`, `benchmark_api.py`, four test files.

**Interfaces:**
- Consumes: `enforce_llm_quota_on_own_key`, `users_own_key_credentials`, `users_own_key` (PR A); `get_evaluation_identity` (`evaluations_auth`, as the unified route uses it).
- Produces: `EvaluationManager.evaluate_custom_metric(..., api_key=None, provider_credentials: Any = None)`; `_generate_model_response(*, input_data, api_name, api_key, provider_credentials: Any = None)`.

- [ ] **Step 1: The handle reaches both calls.** In `tests/Evaluations/unit/test_evaluation_manager.py`, at the end of the file, after TASK-15485's formatter tests (it uses their `_MC_ITEM`):

```python
@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("benchmark", [None, "simple_bench"], ids=["custom-input", "benchmark-formatter"])
async def test_a_custom_metric_runs_both_calls_on_the_handle_it_is_given(evaluation_manager, monkeypatch, benchmark):
    """The model's response and the judge's score both run on the user's own-key handle, for a custom input and
    for a benchmark formatter's (TASK-15426)."""
    # imports: SimpleNamespace, users_own_key_credentials, get_registry (Evaluations.benchmark_registry),
    # Summarization_General_Lib as summarization, call_credential_source, users_own_key
    requests: list[dict] = []

    def chat(request, timeout=None):
        requests.append(request)
        return {"choices": [{"message": {"content": "Score: 8"}}]}

    monkeypatch.setattr(
        summarization, "get_registry", lambda: SimpleNamespace(get_adapter=lambda _p: SimpleNamespace(chat=chat))
    )
    users_own_key(monkeypatch, "openai")
    handle = await users_own_key_credentials(1, "openai")
    data = (
        get_registry().create_evaluator(benchmark).format_for_custom_metric(_MC_ITEM)
        if benchmark
        else {
            "name": "m",
            "description": "d",
            "evaluation_prompt": "Rate {answer}",
            "input_data": {"question": "Which day does Cedar open?", "answer": "{model_response}"},
            "scoring_criteria": {"accuracy": "0-10"},
        }
    )

    await evaluation_manager.evaluate_custom_metric(
        metric_name=data["name"], description=data["description"], evaluation_prompt=data["evaluation_prompt"],
        input_data=data["input_data"], scoring_criteria=data["scoring_criteria"],
        api_name="openai", provider_credentials=handle,
    )

    assert [(r["api_key"], call_credential_source(r)) for r in requests] == [("sk-users-own-key", "user")] * 2
```

  (`[benchmark-formatter]` is the real `simple_bench` formatter, `MultipleChoiceEvaluation`, whose judge input reaches the provider since TASK-15485; V5. Sabotage, measured: the judge's `analyze` without `provider_credentials` fails both cases, `At index 1 diff: (None, None) != ('sk-users-own-key', 'user')`.)
- [ ] **Step 2: Flip the route pins.** Replace `test_benchmark_run_refuses_an_over_quota_user_before_loading_the_dataset` (`test_evaluations_benchmarks_api.py`) with `test_benchmark_run_names_its_provider_to_the_llm_quota`, parametrized `own_key` in `[True, False]` (ids `own-key`, `server-key`): `refuse_llm_work`; `asked = users_own_key(monkeypatch, "openai" if own_key else "anthropic")`; a registry stub whose evaluator formats one item; `load_benchmark_dataset` records the load and returns `[{"q": "q"}]`; `_get_evaluation_manager_for_user` returns a stub whose async `evaluate_custom_metric(**kwargs)` records `(kwargs["provider_credentials"].api_key, current_llm_call_context().prechecked)`. Post `{"limit": 1, "api_name": "openai", "parallel": 1, "save_results": False}`. Assert `asked == ["openai"]`; own key: 200 and `seen == [("sk-users-own-key", False)]`; server key: `assert_llm_quota_402`, `detail.error == "limit_exceeded"`, `loaded == []`. (Add `import pytest` and `from types import SimpleNamespace` to the module.)

  Replace `test_run_route_refuses_an_over_quota_user_before_any_work` (`test_benchmark_api_auth.py`) with:

```python
@pytest.mark.parametrize(
    ("body", "stored", "expected", "asks"),
    [({}, "openai", 404, ["openai"]), ({}, "anthropic", 402, ["openai"]), ({"api_key": "sk-raw"}, "openai", 402, [])],
    ids=["own-key", "server-key", "raw-api-key"],
)
def test_run_route_names_its_provider_to_the_llm_quota(monkeypatch, body, stored, expected, asks) -> None:
    """... a run on the user's own key goes on to the benchmark lookup (here a 404); a run on the server's key, or
    on a raw ``api_key`` (which counts, so the route names no provider), gets the 402 (TASK-15426)."""
    # refuse_llm_work; asked = users_own_key(monkeypatch, stored); the app and user override as today;
    # post /api/v1/benchmarks/does-not-exist/run with body
    assert asked == asks
    if expected == 402:
        assert_llm_quota_402(response)
    else:
        assert response.status_code == expected, response.text
```

  In the same file, give `_override_user` a `user_id: int = 1` parameter (the fake `User`'s `id`), and add at the end (Decision 6, V8):

```python
def test_a_run_saved_by_default_is_stored_for_its_user(monkeypatch, tmp_path) -> None:
    """A run with ``save_results`` (the default) returns 200 and is stored in the user's own evaluations database
    under their id. The route stored its rate limiter as ``user_id``, so ``json.dumps`` failed with a 500 after every
    item's LLM calls had run (TASK-15426, review of PR B)."""
    import asyncio
    import json
    from types import SimpleNamespace

    from tldw_Server_API.app.core.Evaluations.evaluation_manager import EvaluationManager

    async def _evaluate(_manager, **_kwargs):
        return {"score": 0.8, "explanation": "ok"}

    item = {
        "name": "m", "description": "d", "evaluation_prompt": "{q}", "input_data": {"q": "q"}, "scoring_criteria": {}
    }
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(
        benchmark_module,
        "get_registry",
        lambda: SimpleNamespace(
            get=lambda _name: SimpleNamespace(evaluation_type="demo"),
            create_evaluator=lambda _name: SimpleNamespace(format_for_custom_metric=lambda _item: item),
        ),
    )
    monkeypatch.setattr(benchmark_module, "load_benchmark_dataset", lambda *_a, **_k: [{"q": "q"}])
    monkeypatch.setattr(EvaluationManager, "evaluate_custom_metric", _evaluate)
    app = _make_app()
    _override_user(app, [EVALS_MANAGE], user_id=7)

    with TestClient(app) as client:
        response = client.post("/api/v1/benchmarks/demo/run", json={"limit": 1})

    assert response.status_code == 200, response.text
    stored = asyncio.run(EvaluationManager(user_id=7).get_evaluation(response.json()["evaluation_id"]))
    assert stored, "the run is not in the user's own evaluations database"
    assert json.loads(stored["metadata"])["user_id"] == "7"
```

  (It stubs `evaluate_custom_metric` on the class, so it holds whether the route uses one manager or one per user, and it reaches no provider. User 7, not the single-user id 1, so a run stored in another user's database fails the second assertion. Measured: on the base source it fails `assert 500 == 200`, `{"detail":"Failed to run benchmark"}`, and a probe wrapping `store_evaluation` showed the cause, `TypeError: Object of type RateLimiter is not JSON serializable`, `user_id` a `RateLimiter`; with the id fixed but the module-level manager kept (sabotage), it fails `the run is not in the user's own evaluations database`.)
- [ ] **Step 3: Run** the four tests. Expected on unchanged source: `7 failed, 1 passed` (measured on `3160d6392a`: both custom-metric cases fail with `TypeError: ... unexpected keyword argument 'provider_credentials'`; every route case but `[raw-api-key]` fails on `asked == []`; the saved run fails with the 500).
- [ ] **Step 4: Implement.**
  - `evaluate_custom_metric(..., provider_credentials: Any = None)`: pass it inside the `model_call = self._generate_model_response(...)` binding, and as `provider_credentials=provider_credentials` to the judge's `asyncio.to_thread(analyze, ...)` after `input_is_literal_text=True` (keep both); `_generate_model_response(..., provider_credentials: Any = None)` passes it to its `analyze`. Docstring: both calls run on the handle's key, and `score` is None when the item has no score (TASK-15485's contract, until now only an inline comment; its final review left it to this rewrite). The `try` around `await model_call` and every `score: None` path stay as they are.
  - `evaluations_benchmarks.run_benchmark`: import `enforce_llm_quota_on_own_key` (instead of `enforce_llm_quota`) and `canonical_provider_name` (`LLM_Calls.provider_identity`); the boundary becomes `own_key = await enforce_llm_quota_on_own_key(getattr(current_user, "id", None), canonical_provider_name(request.api_name))` (comment: "...and then runs to completion; on the user's own key, if any."); pass `provider_credentials=own_key` to `evaluate_custom_metric`.
  - `benchmark_api.run_benchmark`: the same, with `provider = None if request.api_key else canonical_provider_name(request.api_name)` on the line before (comment: a raw `api_key` counts, so its run names no provider).
  - `benchmark_api.run_benchmark`, the saved run (Decision 6): drop the `user_id: str = Depends(get_rate_limiter_dep)` parameter and its docstring line (the docstring names `current_user` instead); import `get_evaluation_identity` from `evaluations_auth` beside `get_eval_request_user`; open the `try` with `identity = get_evaluation_identity(current_user)` and `evaluation_manager = EvaluationManager(user_id=identity.user_scope)`, as `evaluations_benchmarks.py:111-112` does; store `"user_id": identity.created_by`; delete the module-level `evaluation_manager = EvaluationManager()` and its comment (nothing else reads it). `get_rate_limiter_dep` stays imported for `evaluate_simpleqa`.
  - `tests/Evaluations/test_benchmark_api_error_mapping.py`: its three `run_benchmark` tests called the route with `user_id="user-1"` and patched the module-level manager. Each now passes `current_user=SimpleNamespace(id=1)` and patches `benchmark_module.EvaluationManager` with `lambda **_kwargs: manager`, where `manager` is a `SimpleNamespace` holding the same `evaluate_custom_metric` (and `store_evaluation`) stubs as before; their assertions do not change.
- [ ] **Step 5: Run** them. Expected: all pass. Then the four files whole: no new failure (measured on `3160d6392a`: 64 passed; with the two Task 2 files (70 passed), the six touched test files 134 passed, 0 netprobe attempts).
- [ ] **Step 6: Commit** `feat(evaluations): benchmark runs run on the user's stored key, and the standalone route saves its run (TASK-15426)`.

## Task 2: Watchlist output summaries (ACs #2, #3)

**Files:** `outputs_service.py`, `watchlists.py`, two test files.

**Interfaces:**
- Produces: `_llm_reply(prompt, *, api_name, model_override, provider_credentials: Any = None)`, `_summarize_single_article(..., provider_credentials: Any = None)`, `summarize_items_for_output(..., provider_credentials: Any = None)`.

- [ ] **Step 1: The handle reaches every summary.** In `tests/Services/test_outputs_service.py`, before `test_generate_group_summaries_sends_text_with_braces_once`:

```python
@pytest.mark.unit
async def test_summaries_run_on_the_handle_they_are_given(monkeypatch: pytest.MonkeyPatch) -> None:
    """Over the limit, every summary request carries the user's own-key handle, so the check at each call lets it
    run: their key, recorded as theirs (TASK-15426)."""
    # imports: users_own_key_credentials, ChatProviderRegistry (LLM_Calls.adapter_registry), ChatProvider
    # (LLM_Calls.providers.base), bind_llm_call_user, call_credential_source, ensure_llm_state_executor
    # (Usage.llm_roots), refuse_llm_calls, users_own_key
    requests: list[dict] = []

    class _Provider(ChatProvider):
        """A provider below the usage-accounting wrapper, which checks each call."""

        name = "openai"

        def capabilities(self) -> dict:
            return {}

        def chat(self, request: dict, *, timeout: float | None = None) -> dict:
            requests.append(request)
            return {"choices": [{"message": {"content": "Model summary."}}]}

        def stream(self, request: dict, *, timeout: float | None = None):
            return iter(())

    registry = ChatProviderRegistry(include_defaults=False)
    registry.register_adapter("openai", _Provider)
    monkeypatch.setattr(summarization, "get_registry", lambda: registry)
    users_own_key(monkeypatch, "openai")
    handle = await users_own_key_credentials(1, "openai")
    refuse_llm_calls(monkeypatch)
    bind_llm_call_user(1)
    ensure_llm_state_executor()  # the server's default executor, which carries the call state to each summary
    items = [{"id": 1, "content": "Cedar article body."}, {"id": 2, "content": "Birch article body."}]

    await outputs_service.summarize_items_for_output(items, provider_credentials=handle, **_LLM)

    assert [(r["api_key"], call_credential_source(r)) for r in requests] == [("sk-users-own-key", "user")] * 2
```

  Review B-5 asked for `refuse_llm_calls` in this test. Measured: with the file's `adapter` fixture that is a no-op, because `_Adapter` is not a `ChatProvider`, so no usage-accounting wrapper (the check at the call) sits above it, and the summaries' default-executor hop would drop the user. So the test registers a `ChatProvider` subclass, binds user 1 and installs the carrying executor, as the server does. Sabotage (the executor lambda passing `provider_credentials=None`): 1 failed, `LLMQuotaExceededError` (402) at the call; the real code passes (`test_outputs_service.py` whole: 25 passed).

- [ ] **Step 2: Flip the route pin.** Replace `test_output_summaries_refuse_an_over_quota_user_before_any_summary` (`test_watchlists_api.py`) with `test_output_summaries_name_their_provider_to_the_llm_quota`, parametrized `own_key` (`own-key`, `server-key`): `run_id, _ = _over_quota_run(c, monkeypatch)`; `asked = users_own_key(monkeypatch, "openai" if own_key else "anthropic")`; re-stub `watchlists_endpoint.summarize_items_for_output` to record `(getattr(kwargs.get("provider_credentials"), "api_key", None), current_llm_call_context().prechecked)` and return the items; post `{"run_id": run_id, "summarize": True, "llm_provider": "openai"}`. Assert `asked == ["openai"]`; own key: 200 and `seen == [("sk-users-own-key", False)]`; server key: `assert_llm_quota_402`, `seen == []`, and the follow-up `GET /api/v1/watchlists/outputs?run_id=` totals 0.
- [ ] **Step 3: Run** both. Expected on unchanged source: `3 failed` (measured on `7b4cb653b4`).
- [ ] **Step 4: Implement.** Thread `provider_credentials` from `summarize_items_for_output` through the executor lambda to `_summarize_single_article` and `_llm_reply`, which passes it to `analyze`. In `create_output` (Decision 5):

```python
        # Usage quotas (§4a) before any summary call or output; the summaries run on the user's own key, if any.
        own_key = await enforce_llm_quota_on_own_key(current_user.id, llm_provider, model=payload.llm_model)
```

  and the call opens `items_for_summary, api_name=llm_provider, provider_credentials=own_key,` on one line. Import `enforce_llm_quota_on_own_key` in place of `enforce_llm_quota` (`:59`; `create_output` was its only user).
- [ ] **Step 5: Run** `$FOCUSED_B` → `11 passed (0:37)`, 0 netprobe attempts (measured on `3160d6392a`); `$PY -m pytest tldw_Server_API/tests/lint/test_function_size_ratchet.py -q` → passes (`create_output` 909 ≤ 910).
- [ ] **Step 6: Commit** `feat(watchlists): output summaries run on the user's stored key (TASK-15426)`.

## Task 3: Docs, verification, review, PR

- [ ] **Step 1: Docs** (one commit with the mirror):
  - `Usage_Quotas.md`: after PR A's study-assistant bullet add these two bullets (review M2: quoted, not described):

```markdown
  - benchmark runs (`POST /evaluations/benchmarks/{name}/run` and `POST /benchmarks/{name}/run`), for `api_name`, on which every item then runs. A standalone run (`POST /benchmarks/{name}/run`) that sends a raw `api_key` names no provider, because that key counts, so it is refused (`/evaluations/benchmarks` takes no `api_key`);
  - `POST /watchlists/outputs` with `summarize`, for `llm_provider`, on which every summary then runs. The output enrichment (briefing and per-group summaries, topic grouping) still runs on the server's key and is checked at its calls;
```

    In the "other boundaries" sentence, `... and a media ingest job's check when it starts, watchlist output summaries, evaluation runs and benchmark runs.` becomes `... and a media ingest job's check when it starts, and evaluation runs.`; PR A's `(the study assistants above)` becomes `(the study assistants, benchmark runs and watchlist output summaries above)`.
  - BYOK guide (Usage Quotas): add `benchmark runs, watchlist output summaries, ` to the over-limit list after `the flashcards and quizzes study assistants, `; replace PR A's sentence, as narrowed by its final commit `8315fa68cf` (`The study assistants use your own key for the reply's provider whenever it is a key that bills you as described above (not a key for a custom OpenAI-compatible endpoint, or for a provider the operator points at its own server), so those replies are recorded as yours and do not count.`), and the sentence after it (`Media analysis, watchlist summaries, evaluation runs and benchmark runs never use a key you stored: ...`) with these three (review B-1; they keep A's narrowing, not the older "whenever you stored one"):

```text
The study assistants, benchmark runs and watchlist output summaries use your own key for their provider whenever it is a key that bills you as described above (not a key for a custom OpenAI-compatible endpoint, or for a provider the operator points at its own server), so those calls are recorded as yours and do not count. A raw `api_key` you send with a standalone benchmark run (`POST /api/v1/benchmarks/{name}/run`) is not your stored key: it counts, so over a limit such a run is refused. Media analysis and evaluation runs never use a key you stored: their calls run on the server's key, or on a raw `api_key` you send with an evaluation run, which counts the same, so over a limit they still refuse.
```

  - BYOK guide (Auditing and Updates, PR A's `last_used_at` clause): the parenthesis becomes `(the study assistants, benchmark runs and watchlist output summaries)`.
  - ADR-083: append to PR A's amendment section (`## Amendment (2026-11-09)`, after its study-assistants bullet; `<date>` is this PR's date, 2026-11-10 in the prototype):

```markdown
- **Benchmark runs and watchlist output summaries** (<date>): a benchmark run's `api_name` (every item's model response and judge call), and the summaries' `llm_provider`. A standalone benchmark run (`POST /benchmarks/{name}/run`) that sends a raw `api_key` names no provider and is refused over a limit, because that key counts. Watchlist output enrichment, a Scheduler task, still runs on the server's key.
```

  - Benchmark guide (`Docs/User_Guides/Server/Benchmark_Creation_API_WebUI_Extension_Guide.md`, "402 LLM quota"; review M1): the two bullets become

```markdown
- The user is over an LLM limit: `detail.error` is `limit_exceeded`, and `Retry-After` says when to try again. The run is refused before its dataset loads, so nothing is evaluated. A run whose `api_name` runs on a key the user stored that bills them (see `Docs/User_Guides/Server/BYOK_User_Guide.md`) is not refused: every item's model response and judge call run on that key and do not count. This route takes no `api_key`; only the standalone `POST /api/v1/benchmarks/{name}/run` does, and a run there that sends one is refused, because that key counts.
- An admitted run runs to completion, even if it takes the user past the limit; no item is refused mid-run. The next run on the server's key is refused.
```

  - Watchlists API (`Docs/API-related/Watchlists_API.md`, the `summarize` bullet of "Stage 5 defensible report fields"; review M1): the text from "With `summarize`, a user over an LLM quota gets a 402" to "so a refusal there" becomes

```markdown
With `summarize`, a user over an LLM quota gets a 402 and no output is created, unless the summaries' provider (`llm_provider`, else the job's `output_prefs.llm.provider`) runs on a key the user stored that bills them (see `Docs/User_Guides/Server/BYOK_User_Guide.md`): then the summaries run on that key, do not count, and the output is created. Briefing summaries and topic grouping run after the response, on the server's key, so a refusal there
```

    (the rest of the bullet is unchanged; with Decision 3, an over-limit user on their own key who also asks for a briefing summary or topic grouping gets the output, with `enrichment_status: failed`, as the review measured).
  - `bash Helper_Scripts/refresh_docs_published.sh`; only the four mirrors change (ADR-083, BYOK, the benchmark guide, the Watchlists API; measured).
- [ ] **Step 2: Wider set** with the PR: Task 0's counts plus the new tests, no new failing id (measured on `3160d6392a`: `1 failed, 534 passed` against the base's `1 failed, 526 passed`, the same probe artifact and the same two netprobe lines).
- [ ] **Step 3: CI checks:** `PATH="$(dirname $PY):$PATH" PYTHONPATH=<plans dir> NETPROBE_LOG=<file> TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -n 4 -p netprobe_15427 tldw_Server_API/tests/Docs tldw_Server_API/tests/lint tldw_Server_API/tests/CI tools/backlog-py/tests` (review M5: the command its Expected was measured with); `$PY Helper_Scripts/ci/check_shard_coverage.py` (`new_uncovered=0`); `$PY Helper_Scripts/ci/test_quality_triage.py --enforce` (OK); `PYTHONPATH=. $PY Helper_Scripts/ci/route_auth_ratchet.py` (baseline exact); `PYTHONPATH=. $PY Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json --check-paths apps/packages/api-types/paths.d.ts` (both OK); `/usr/bin/python3 Helper_Scripts/ci/check_backlog_task_ids.py .` (exit 0); `/opt/homebrew/bin/uvx bandit -ll` on `evaluation_manager.py`, `evaluations_benchmarks.py`, `benchmark_api.py`, `outputs_service.py`, `watchlists.py`). Expected, measured on `7b4cb653b4` with this PR's prototype (B alone on A and TASK-15485) and again on `3160d6392a` after review fold 2 (same results unless noted): shard coverage `new_uncovered=0`; triage `OK — no offenses beyond the baseline.`; route-auth `208 unauthenticated routes; baseline exact.`; OpenAPI fingerprint and path union both OK; `check_backlog_task_ids.py` exit 0; `tests/Docs tests/lint tests/CI tools/backlog-py/tests` (`-n 4`, `-p netprobe_15427`, 0 attempts) base `2 failed, 1768 passed, 6 skipped, 1 error (6:25)` → prototype `2 failed, 1768 passed, 6 skipped, 1 error (11:26; 9:51 after fold 2)`, all three environment-only and on both trees (`tests/lint/test_runtime_package_discovery.py` errors and `test_jobs_postgres_exports_recorded_coverage[same-checkout]`/`[foreign-checkout]` fail: the shared venv lacks `setuptools` and `coverage`); the function-size ratchet and the `Docs/Published` mirror tests pass; Bandit `-ll` on B's five source files: one finding, pre-existing and outside this PR's lines (B608, Medium severity, Low confidence, the `# nosec B608` one line above it: `evaluation_manager.py:879` on the base, `:889` with the PR); nothing new. ruff on every touched `.py` file against the base: no new finding (the same 28 on both as the refresh counted them; after fold 2, eleven files, 26 on both with the venv's ruff 0.16.5, line numbers stripped). (The earlier run on the four prototypes stacked, A-D, on `cb70f0c5d1` found the layering ratchet `services -> api.v1.API_Deps` 19 > 18, fixed in PR D, and two triage offenses in C and D; neither involves B's files.)
- [ ] **Step 4: Network.** One run of `$FOCUSED_B` inside `block_live_network()`: `attempts == []` (measured on `3160d6392a`: `$FOCUSED_B` under `-p netprobe_15427`, 11 passed, 0 attempts; the wider set's only netprobe lines are the two pre-existing `test_security.py` ones, on both trees).
- [ ] **Step 5: Close-out notes:** `--append-notes "PR B (<url>): benchmark runs (both routes; a raw api_key names no provider) and watchlist output summaries run on the user's own key (enforce_llm_quota_on_own_key; handle through evaluate_custom_metric, both the model response and the judge call, and summarize_items_for_output). Enrichment unchanged. The standalone /benchmarks/{name}/run no longer returns 500 after its LLM calls with the default save_results: it stored its rate limiter as user_id; it now stores the run in the user's own evaluations database under their id. Built on TASK-15485 (#3445, the benchmark judge fix, merged first). Tests: <counts>. Bandit: one pre-existing finding (B608), none new. Docs updated (mirrored)."` (review B-3).
- [ ] **Step 6: Review and PR:** the clean-session `/superpowers:requesting-code-review` on the branch diff with this plan and the task file (address every finding, re-run the focused set); then the PR body (Summary; a full `## Change summary` section recording the owner's waiver of 2026-11-07, "Yes, waive for all"; Test plan with the measured counts; Review Focus; ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`), mark ready, diff the PR's Frontend E2E failures against a recent dev run, and merge through `<plans dir>/pr_queue3.sh` (check `gh variable get MERGE_QUEUE` first: unset on 2026-11-10; the script reads dev's required checks from the ruleset, 8 in #3447's run).

## Review fold (2026-11-07)

Folded from `review_plan_15426_ab.md` (B: 0 Blocker, 0 Important, 6 Minor) and the owner's decisions of 2026-11-07. The pre-fold texts are `plan_15426_b.prefold.md` and `proto_15426_b.prefold.patch`. The fold was measured on `7a020da14a` with the folded A applied, outbound connections blocked (`-p netprobe_15427`, 0 attempts in every run). No run of the fold included TASK-15485, which did not exist yet; the 2026-11-10 refresh re-measured everything on a base that does.

| Finding / decision | Where it landed |
|---|---|
| B-1: the BYOK prose existed only in the prototype | Task 3 Step 1 quotes the three sentences and the ADR bullet; Plan-specific constraint: the patch is normative for unquoted docstrings and comments. |
| B-2: V7 said no open PR touches this PR's tests | V7 corrected: #3269 adds 22 lines to `test_watchlists_api.py` at `:305`, far from the `:1919` pin. |
| B-3: the close-out note said "Bandit: no issues" | Task 3 Step 5: "one pre-existing finding (B608), none new". |
| B-4: the judge bug is worse than the follow-up said | Fixed first in its own PR (owner decision below); V5 now describes the state after that fix. |
| B-5: no end-to-end over-limit test (optional) | Task 2 Step 1 and the patch: the summaries test runs over the limit with the check at the call live (a `ChatProvider` subclass, user 1 bound, the carrying executor). The review's one-liner alone was measured to be a no-op (the file's `_Adapter` has no accounting wrapper). Sabotage: 1 failed with the 402 at the call; real code: 25 passed. |
| B-6: own-key calls skip the `DEFAULT_MODEL_<PROVIDER>` fallback | Review Focus 6 records it; listed for PR D's follow-up list. No code change. |
| A-6 (from PR A): `last_used_at` | Task 3 Step 1 extends PR A's clause to benchmark runs and watchlist output summaries (patch updated). |
| Owner: the judge fix lands before B, in its own PR | "Order", V5, V7, Decision 4, Task 0 Step 1; the fix is TASK-15485 (#3445), and the 2026-11-10 refresh folded what the fold had deferred to it (see "Refresh (2026-11-10)"). |
| Owner: a raw `api_key` keeps counting | Decision 2. |
| Owner: watchlist output enrichment out of scope | Decision 3. |
| Owner: no team or org keys; a run keeps the key resolved at its start | PR A Decisions 2-3 (one request here, so nothing to change). |
| Fold measurements | `$FOCUSED_B` 9 passed on `7a020da14a` (0 attempts); ruff on every A+B `.py` file: the same findings as on `7a020da14a`. |

## Refresh (2026-11-10)

Refreshed onto its real base, `7b4cb653b4`: `origin/dev` `dcdef5355d`, then `git merge --no-ff` of PR A (#3444 at `8315fa68cf`, rebased since the approval's `82bde371ce`, same last commit) and of the judge fix (#3445, TASK-15485 at `4b85b81a2c`); neither had merged into dev, and neither merge conflicted. The pre-refresh texts are `plan_15426_b.pre_refresh.md` and `proto_15426_b.pre_refresh.patch`. Every count was re-measured on that base, base and prototype, under `-p netprobe_15427`. No owner decision changed.

| What changed | Why |
|---|---|
| Header ("Measured on", "Order", "Spec") | The base is A + TASK-15485; the hedging about a judge fix that had not landed is gone. |
| V1, V2, V5, V6, V7 | Re-cited on `7b4cb653b4`. V2: the model call is now bound as `model_call = ...` and awaited in a `try`; the judge call carries `input_is_literal_text=True`. V5 rewritten as the state after the fix: two calls per item, both reach the provider, a failed or raised one is unscored. V6: the RED count with the new case. V7: #3450 and #3451 (new) touch `watchlists.py:61`, next to this PR's `:59` import (no merge conflict, measured; the patch context overlaps). |
| Decision 4 | Rewritten as the state after TASK-15485: two calls per item, both on the handle; unscored stays unscored. The "may change once `TASK-<JB>` lands" list, the "re-measure" notes and the `TASK-<F_B>` leftovers are gone; `TASK-<JB>` is TASK-15485 (#3445) throughout. |
| Task 1 Step 1 (and the patch) | The real-formatter case the plan deferred: the test is parametrized `[custom-input]` / `[benchmark-formatter]` (the real `simple_bench` formatter, `MultipleChoiceEvaluation`), moved to the end of the file beside TASK-15485's formatter tests (it uses their `_MC_ITEM`). The custom input uses `question`, not the `text` key the judge bug once required. Sabotage (the judge's `analyze` without the handle): both cases fail. |
| Task 1 Step 3, Step 4 (and the patch) | RED `6 failed, 1 passed` for the three tests. Step 4 names where the handle goes on the new code: inside the `model_call` binding and beside `input_is_literal_text=True` (the one known conflict, resolved keeping both). The docstring also states TASK-15485's `score: None` contract, which that PR's final review left to this rewrite. |
| Unscored interplay (Decision 4, Review Focus 7) | Checked: no test or Expected line of this PR assumes a failed call scores 0.0 or 0.5. With the handle, a failed or raised own-key call is unscored exactly as on the server's key; TASK-15485's tests pin it. |
| Carry rule (Plan-specific constraints) | Added where it holds: the handle is a route local passed as a keyword argument down to `analyze`; no stored evaluation, output metadata or enrichment payload sees it. |
| Task 3 Step 1 (and the patch) | The BYOK sentence keeps A's narrowing from `8315fa68cf` ("whenever it is a key that bills you as described above (...)"), not the older "whenever you stored one". ADR bullet dated 2026-11-10 in the prototype, under A's `## Amendment (2026-11-09)`. |
| `$FOCUSED_B`, the wider set, Task 0 Step 4, Task 2 Step 5, Task 3 Steps 2-4 | Re-measured (below). The wider set adds the benchmark CLI's test, the third caller of `evaluate_custom_metric`. |
| Task 0 Step 1, Task 3 Steps 5-6 | Branch only once both #3444 and #3445 are in dev; the close-out note names TASK-15485; the merge step checks `MERGE_QUEUE` (unset) and reads 8 required checks from the ruleset. |
| Prototype | Rebased: 15 files, +227/-62 (was +211/-63). Every source and test hunk but `evaluation_manager.py` applied unchanged (the unit test then changed as above); `Usage_Quotas.md` applied unchanged; the ADR and BYOK hunks were redone on A's final text (its amendment date, its narrowed sentence) and the mirrors regenerated with `refresh_docs_published.sh` (only the two mirrors changed). |

Measured on `7b4cb653b4` (base → prototype):
- `$FOCUSED_B`: RED `9 failed, 1 passed` → GREEN `10 passed`; 0 netprobe attempts.
- The wider set (22 files + the CLI test): `1 failed, 526 passed, 4 skipped, 1 xpassed` → `1 failed, 533 passed, 4 skipped, 1 xpassed`; the failure is the known probe artifact on both, with the same two netprobe lines.
- Function sizes: `create_output` (frozen, 910) 910 → 909. Not frozen, all under the 300-line threshold: `evaluate_custom_metric` 144 → 152, `_generate_model_response` 22 → 24, `run_benchmark` 136 → 137 (`evaluations_benchmarks.py`) and 165 → 168 (`benchmark_api.py`), `_llm_reply` 26 → 28, `_summarize_single_article` 10 → 11, `summarize_items_for_output` 24 → 27.
- Bandit `-ll` on the five source files: one finding on both, the pre-existing B608 at `evaluation_manager.py:879` → `:889`. ruff on the ten touched `.py` files: the same 28 findings on both.
- `scope_predicate_baseline.txt`: lists none of the 15 files; no regeneration.
- `tests/Docs tests/lint tests/CI tools/backlog-py/tests`: `2 failed, 1768 passed, 6 skipped, 1 error` on both, the three environment-only ones (missing `setuptools` and `coverage` in the shared venv); 0 netprobe attempts. On the prototype: shard coverage `new_uncovered=0`, triage OK, route-auth `208; baseline exact`, OpenAPI fingerprint and path union OK, `check_backlog_task_ids.py` exit 0.
- The five touched test files whole, prototype: 126 passed (the three Task 1 files 56; `test_outputs_service.py` 25), 0 netprobe attempts.

Not measured here: the union run with C and D stacked (their refreshes are separate); the PostgreSQL leg (no file in B's wider set takes a PostgreSQL fixture).

## Review fold 2 (2026-11-10)

Folded from `review_refresh_15426_b.md` (ready with fixes: 0 Blocker, 0 Important, 5 Minor) with the controller's rulings. The pre-fold texts are `plan_15426_b.pre_fold2.md` and `proto_15426_b.pre_fold2.patch`. Base: `3160d6392a` = `origin/dev` `06a262b132` (PR A #3444 merged) + `git merge --no-ff` of #3445 (`fix/benchmark-judge-15485`, rebased to `31da7ec215`; not yet in dev; no conflict). Its tree is identical to the refresh's `7b4cb653b4`, so every citation and base count carries over. Every run used `-p netprobe_15427`.

| Finding | Where it landed |
|---|---|
| M1: the benchmark guide's "402 LLM quota" and the Watchlists API's `summarize` text went stale | Task 3 Step 1 quotes the new text for both: an over-limit user whose `api_name` (the summaries' provider) runs on a key they stored that bills them is served on that key; a raw `api_key` is still refused; enrichment runs on the server's key. Four mirrors now change (measured with `refresh_docs_published.sh`); the prototype has 9 docs files. |
| M2: the Usage_Quotas bullets were described, not quoted | Task 3 Step 1 quotes the prototype's two bullets and the exact before/after of the other two edits; the Plan-specific constraint now says every docs sentence is quoted. |
| M3: the standalone `/benchmarks/{name}/run` 500s with the default `save_results` after all its LLM calls | Fixed in this PR (controller ruling; V8, Decision 6, Review Focus 8). Task 1 Step 2 adds the RED-first test `test_a_run_saved_by_default_is_stored_for_its_user`; Task 1 Step 4 drops the rate-limiter parameter and stores `identity.created_by` through a per-user manager, as the unified route does, and adapts the three direct-call tests in `test_benchmark_api_error_mapping.py`. The per-user manager goes beyond "store the user's id": the module-level manager is the single-user id's database, so with only the id fixed every user's run would start landing there (sabotage measured). `evaluate_simpleqa` also takes `user_id` from `get_rate_limiter_dep` but never reads or stores it: no defect, not changed. |
| M4: "third caller" and the carry-rule citation | The wider set names the fourth caller (`cli/commands/evaluation.py:743`, the `eval custom` command, TASK-15489); the carry rule cites `watchlists.py:7216-7288` for the output metadata. |
| M5: Task 3 Step 3's command did not match its Expected | The command is now `-n 4 -p netprobe_15427` with `PYTHONPATH=<plans dir> NETPROBE_LOG=<file>`. |

Measured on `3160d6392a` (base → prototype):
- The new test: RED on the base source, `assert 500 == 200`, `{"detail":"Failed to run benchmark"}`; a probe wrapping `store_evaluation` showed `TypeError: Object of type RateLimiter is not JSON serializable` with `user_id` a `RateLimiter`. GREEN on the prototype. Sabotage (the id fixed, the module-level manager kept): fails `the run is not in the user's own evaluations database`.
- `$FOCUSED_B`: RED `10 failed, 1 passed` → GREEN `11 passed`; 0 netprobe attempts.
- The wider set (22 files, the same 22, + the CLI test): `1 failed, 526 passed, 4 skipped, 1 xpassed` (base, the same tree as the refresh's) → `1 failed, 534 passed, 4 skipped, 1 xpassed`; the known probe artifact and the same two netprobe lines.
- The six touched test files, prototype: 134 passed (Task 1's four 64, Task 2's two 70), 0 attempts.
- Function sizes: `run_benchmark` (`benchmark_api.py`) 165 → 169 (168 before this fold; not frozen, under 300). Every other touched function as the refresh measured: `create_output` 910 → 909 (frozen at 910), `evaluate_custom_metric` 144 → 152, `_generate_model_response` 22 → 24, `run_benchmark` (`evaluations_benchmarks.py`) 136 → 137, `_llm_reply` 26 → 28, `_summarize_single_article` 10 → 11, `summarize_items_for_output` 24 → 27. `function_size_baseline.json` freezes only `create_output` of these.
- `tests/Docs tests/lint tests/CI tools/backlog-py/tests`: base `2 failed, 1768 passed, 6 skipped, 1 error` (the refresh's, same tree) → prototype `2 failed, 1768 passed, 6 skipped, 1 error (9:51)`, the same three environment-only ids; 0 netprobe attempts (the `Docs/Published` mirror tests and the function-size ratchet pass).
- Prototype checks: shard coverage `new_uncovered=0`; triage `OK — no offenses beyond the baseline.`; route-auth `208 unauthenticated routes; baseline exact.` (the run route keeps its authenticator); OpenAPI fingerprint and path union OK (the dropped dependency took no request parameter); `check_backlog_task_ids.py` exit 0; Bandit `-ll` on the five source files: the one pre-existing B608 at `evaluation_manager.py:889`; ruff on the eleven touched `.py` files: 26 findings on base and prototype, none new (venv ruff 0.16.5, line numbers stripped; the refresh's "28" for ten files was counted another way); `git diff --check` clean.
- `scope_predicate_baseline.txt` lists none of the 20 files.

Not measured here: the union run with C and D stacked; the PostgreSQL leg (no file in the wider set takes a PostgreSQL fixture); an end-to-end run of the stale-doc scenarios (M1 is text only; the review measured the behaviour).

## Review fold 3 (2026-11-10)

The scoped review of fold 2 (`review_fold2_15426_b.md`: ready with fixes, 0 Blocker, 0 Important, 1 Minor) found that the raw-`api_key` sentences were wrong for `/api/v1/evaluations/benchmarks/...`, whose request has no `api_key` field (the server drops it, so an over-limit user with a stored key is served on it). The controller scoped all four sentences to the standalone `POST /api/v1/benchmarks/{name}/run`: the Usage_Quotas bullet, the BYOK guide, the ADR bullet and the benchmark guide's "402 LLM quota". **The plan text wins over `proto_15426_b.patch` for these four sentences**; the prototype still has the fold-2 wording. Re-run `bash Helper_Scripts/refresh_docs_published.sh` after the edit, as Task 3 Step 1 says.
