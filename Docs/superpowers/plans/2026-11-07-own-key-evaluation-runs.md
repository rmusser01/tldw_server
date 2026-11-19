# Evaluation runs run on the user's stored key (TASK-15426, PR C of 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An evaluation run (`POST /evaluations/{eval_id}/runs`) names its judge provider to the LLM quota and runs every judge call on the user's own stored key for it, so an over-limit user whose run runs on their own key is started unmarked; every other user gets the same 402 at the same point.

**Architecture:** A new `run_llm_provider(evaluation, run_config) -> (provider, model)` in `eval_runner.py` resolves the judge exactly as the runner does (`_resolve_eval_provider_model_api_key`, made a `@staticmethod`), and answers `(None, None)` for a run that cannot be placed on one key: no evaluation, a type that makes no LLM call, a raw `api_key` in the spec or run config, or a `rag_pipeline` run. The route reads the evaluation (`svc.get_evaluation`) before its check, calls `enforce_llm_quota_on_own_key` (PR A, merged), and hands the returned handle to `svc.create_run(..., provider_credentials=...)`, which puts it in the in-memory `eval_config` the run's task gets (never in the stored run row). The runner's five judge paths pass `config.get("provider_credentials")` on: `run_geval`, a per-sample `RAGEvaluator` with the handle (embeddings off, as `/evaluations/rag` does with a handle), `ResponseQualityEvaluator.evaluate`, and `_call_adapter_text` for label choice and NLI fact-check.

**Tech Stack:** Python 3.12, FastAPI, pytest.

**Spec:** the TASK-15426 task file (ACs #2-#3 for this family); ADR-083; the brief (its "known weak spot": the run pin sent `"config": {}`, though the runner reads `run_config.api_name`, `eval_runner.py:183-189`); PR A's merged helpers, `users_own_key_credentials` (`usage_quota_deps.py:91-122`) and `enforce_llm_quota_on_own_key` (`:155-162`), and its test helper `users_own_key` (`tests/helpers/llm_quota.py:63-97`); `plan_15426_a.md` Decisions 2-4.

**Measured on:** `f0aab6b390`, a simulated dev after PR B: `origin/dev` `04905e02ea` (PR A #3444 and the judge fix #3445, TASK-15485, both merged), then `git merge --no-ff` of PR B (#3452, `feat/own-key-benchmarks-watchlists-15426` at `dd3e1a50d3`, open, not yet in dev); the merge did not conflict. Prototype: `proto_15426_c.patch` (plans dir; 12 files, +228/-27, 5 of them docs); it applies to `f0aab6b390`. Its source and test hunks touch no file B changes (those seven files are identical on `04905e02ea` and `f0aab6b390`), so they also apply to dev alone; its docs hunks are written on B's final text and need B. `file:line` citations are for `f0aab6b390`, and for the prototype where the code is new (said so). History: written on `cb70f0c5d1` + `proto_15426_a.patch`, folded on `7a020da14a` (see "Review fold"), refreshed onto this base on 2026-11-12 (see "Refresh (2026-11-12)" at the end), then its review folded the same day (see "Review fold (2026-11-12)").

**Order:** after PR A (#3444, merged 2026-11-10), the judge fix (#3445, merged) and PR B (#3452, open). B goes first: C's source files are disjoint from B's, but C's docs hunks rewrite B's sentences as B's branch has them (its "Review fold 3" wording), so C branches only once #3452 is in dev.

---

## What was verified (on `f0aab6b390`)

| # | Question | Finding | Consequence |
|---|---|---|---|
| V1 | The boundary | `create_run` (`evaluations_crud.py:301`) calls `enforce_llm_quota(user)` with no provider (`:328`) before the run config is read (`:331`) and `svc.create_run` (`:334`). Not frozen (64 lines). | Read the evaluation and config first, then check. |
| V2 | Where the run executes | `UnifiedEvaluationService.create_run` (`unified_evaluation_service.py:452`) stores the run with `config` only (`:493-499`), builds an in-memory `eval_config` (`:515-524`) and starts `asyncio.create_task(self._run_evaluation_async(...))` (`:527-535`): the task copies the request's context and outlives the request. | A handle resolved at the request and carried in `eval_config` outlives the request without touching the stored row. |
| V3 | One judge per run | Every LLM eval function resolves its provider with `_resolve_eval_provider_model_api_key(eval_spec, config)` (`eval_runner.py:169-218`; uses no `self`): spec `api_name`/`provider`, then run config `api_name`/`provider`, then `openai`; `evaluator_model` and `provider:model` strings may override. `target_model` is only recorded (`unified_evaluation_service.py:489`, `:510`). | One provider per run, resolved by the same function at the boundary. |
| V4 | The judge paths | `_eval_summarization` → `run_geval` in the default executor (`:1454-1463`; takes `provider_credentials`, `ms_g_eval.py:172-192`); `_eval_rag` → the shared `self.rag_evaluator` (`:1532`; `RAGEvaluator` takes the handle at construction, `rag_evaluator.py:152-180`); `_eval_response_quality` → `quality_evaluator.evaluate(...)` (`:1595`; takes `provider_credentials`, `response_quality_evaluator.py:77-88`); `_eval_label_choice` and `_eval_nli_factcheck` → `asyncio.to_thread(_call_adapter_text, ..., api_key=api_key)` (`:1918-1928`, `:2112-2122`), a direct adapter call (`:103-140`). Exact match, includes, fuzzy and propositions make no LLM call. `rag_pipeline` runs the whole RAG pipeline (`_execute_rag_pipeline_run`, `:590`, frozen at 460). | Five paths take the handle; `rag_pipeline` is not placed on one key. |
| V5 | A handle on the RAG judge | `/evaluations/rag` with a handle builds `RAGEvaluator(..., credentials_resolved=True, provider_credentials=...)` with embeddings off: "LLM credentials are not embedding credentials" (`unified_evaluation_service.py:856-881`). `RAGEvaluator` sets `credentials_resolved` itself whenever it is given a handle (`rag_evaluator.py:176-179`). | Same for runs. |
| V6 | Test stubs the route now needs | `test_evaluations_crud_create_run_api.py`'s stubs, two stubs in `test_evaluations_crud_endpoint_sanitization.py` (`_RunLookupDebugService`, `_RunRecordDebugService`) and one in `test_evaluations_identity_route_bindings.py` have no `get_evaluation`; `_ExplodingEvaluationService.get_evaluation` raises, which the route's existing sanitizer already covers. | Add `get_evaluation` to those stubs. |
| V7 | RED on the base | 8 failed, 23 passed (`$FOCUSED_C` with this PR's tests on the base source, `-n 4`, 0:30, 0 netprobe attempts): `[own-key]` and `[server-key]` of the run test (the boundary asks about nothing, so `asked == []`; the own-key run gets the 402), all five runner cases (no handle reaches a judge call: `'error': "'provider_credentials'"` for summarization, `seen` empty for RAG and quality, `[(None, None, None)]` for label choice and NLI) and the C-1 test (`DID NOT RAISE ChatConfigurationError`). `[raw-api-key]`, `[rag-pipeline]`, `[no-llm]`, the file's three other tests and the sanitization file pass on the base, as they must (the stubs' new `get_evaluation` is not called there). (On `cb70f0c5d1` + A, before the C-1 test existed: 7 failed, 23 passed.) | |
| V8 | Open PRs (2026-11-12) | `eval_runner.py`, `unified_evaluation_service.py`, `test_eval_runner.py`: only #3269 [`81b0bb5836`, unchanged since the plan] touches them; its merge base is `530e483b05`, and every hunk it shows (`eval_runner.py:14`, `:50-62`, `:295-463`, `:1303-1430`, `:1572-1590`, `:2165`, `:2384`; the service's `:165`, `:316`, `:586-620`; two tests at the end of `test_eval_runner.py`) is already on dev: its two tests are dev's `:488` and `:526`, after which this PR appends. #3433 (the service's `:94-100` and `evaluate_*` webhooks) merged on 2026-11-10: it is the -4 line shift in `unified_evaluation_service.py` since `cb70f0c5d1`. #3452 (PR B) is in the base. #3454 (the v0.1.47 release, dev to main) lists ADR-083, Usage_Quotas and the BYOK guide only because dev is ahead of main. No other open PR touches this PR's twelve files (checked with each large PR's full file list, `gh api .../pulls/<n>/files`). | Either order with #3269. |
| V9 | The judge fix (TASK-15485) and PR B's keyword | No code path of this PR reaches `evaluation_manager.py`: `eval_runner.py`, `unified_evaluation_service.py` and `evaluations_crud.py` never name `evaluate_custom_metric`, `_generate_model_response`, `EvaluationManager` or `input_is_literal_text` (git grep, empty). The run's judges are `run_geval`, `RAGEvaluator`, `ResponseQualityEvaluator` and `_call_adapter_text`. TASK-15485's unscored contract (`score: None`, counted `unscored`) is the benchmark routes' and is untouched. The runner keeps its own failure handling, which this PR does not change: a failed summarization, RAG or quality judge call is a sample error, and when it is a provider failure that `analyze` raises as `SummaryProviderError` (the RAG and quality judges call `analyze`), `_process_batch` marks the sample `provider_error` (`:1412-1420`) and the run fails at that batch's checkpoint (`:441-447`); label choice and NLI catch `_EVAL_RUNNER_NONCRITICAL_EXCEPTIONS` around their call (`eval_runner.py:1929-1931`, `:2123-2125`; the tuple `:74-84` includes `ChatConfigurationError`, `ConnectionError`, `TimeoutError`) and then score the unparsed sample 0.0 (`:1982-1995`, `:2175-2187`), while a vendor 401 (`ChatAuthenticationError`, a `ChatAPIError`) and a quota refusal (`LLMQuotaExceededError`, an `HTTPException`) are not in that tuple and become a sample error in `_process_batch` (`:1408-1420`; checked with `issubclass` on the base). PR B's `provider_credentials` keyword (`evaluate_custom_metric`, `_generate_model_response`) has the name this PR uses on `create_run` and `_call_adapter_text`; this PR neither calls nor changes it. | This PR needs neither and changes neither. Decision 3 holds: a key revoked at the vendor never falls back to the server's key; it fails each remaining sample of a summarization, label-choice or NLI run, and a RAG or response-quality run after the batch it hits (review M3). |
| V10 | The carry rule (owner, B through D) | See "Plan-specific" below: every place the run config or `eval_config` is built, copied, stored or sent, with where the handle is and is not. | The handle stays a route local and an in-memory `eval_config` key; no stored row, webhook, result or job payload sees it. |

## Decisions

1. **ADR-004 Change summary.** Waived for every TASK-15426 PR (owner, 2026-11-07); recorded as a full `## Change summary` section.
2. **Resolve at the request, hand the handle to the task (ruling).** The route resolves once (`enforce_llm_quota_on_own_key`, runtime closed at once, PR A Decision 2) and passes the handle through `create_run` into `eval_config`. Alternative: the task owns a runtime and resolves at its start; the service only knows `created_by` (a string scope, not the numeric user), the boundary would admit on a lookup the task might not repeat, and the core would import the API-layer helper. The handle is never stored: the run row gets `config`, not `eval_config` (V2, V10).
3. **A key revoked mid-run (owner decision, 2026-11-07: keep).** The run keeps the key captured at the request and never falls back to the server's key: deleting the stored key in tldw does not stop a running run. What a key revoked at the vendor (a 401) does depends on the run type (measured by the 2026-11-12 review's probe, review M3): in a summarization, label-choice or NLI run every remaining sample becomes a sample error ("Authentication failed with the chat provider.") and the run completes; in a RAG or response-quality run the 401 surfaces through `analyze` as a `SummaryProviderError`, so that batch's samples error and the run then fails at the batch checkpoint (`eval_runner.py:441-447`) with "Summary provider error: authentication (openai). Check the provider settings and availability before starting a new run." *Recommended* because a run is bounded (samples × the eval timeout), a stream admitted on a key finishes on it too, and re-resolving per sample costs a database read per sample and could move a half-done run onto the server's key. *Alternative (not taken):* re-resolve before each batch and stop the run when the key is gone. The BYOK guide says so (Task 3 Step 1, review C-4): a run already started keeps using the key it started with; cancel it (`POST /api/v1/evaluations/runs/{run_id}/cancel`, `evaluations_crud.py:447`) to stop it.
4. **Runs the boundary cannot place on one key keep the refusal (ruling).** `rag_pipeline` (its pipeline stages pick their own providers, as RAG's `_RAG_OTHER_LLM_STAGES` ruling in TASK-13956 B), a raw `api_key` in the spec or run config (it counts; brief default, as PR B), and runs of types that make no LLM call (today they are refused over a limit too; admitting them is a separate change). No handle is passed for any of them. **Owner decision, 2026-11-07:** a raw `api_key` keeps counting, and `rag_pipeline` runs stay on the server's key (out of scope).
5. **The RAG judge on a handle uses the LLM for similarity (ruling).** As `/evaluations/rag` does with a handle (V5). Own-key RAG runs score similarity with the judge instead of OpenAI embeddings; server-key runs are unchanged. This applies to every user with a stored key, not only over-limit ones, so a user's earlier and later `answer_similarity` scores are not comparable: the Usage_Quotas bullet says so (Task 3 Step 1, review C-2).
6. **Evaluation token reservations unchanged** (owner, 2026-10-29): `limits.evaluation_tokens_per_day` keeps counting own-key calls.

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
- `$PY` is `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`; `<plans dir>` is `/Users/macbook-dev/.claude/projects/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/plans`; `<scratchpad>` is `/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/scratchpad`.
- `_execute_rag_pipeline_run` is frozen at 460 lines and is not touched (460 on base and prototype). `_execute_evaluation`, `create_run` (route and service), the eval functions and `_call_adapter_text` are not frozen.
- `scope_predicate_baseline.txt` lists none of this PR's files (checked on `f0aab6b390`, all 12).
- **Carry rule** (owner, for B through D): the own-key handle never goes inside a copied or persisted run config, stored evaluation metadata, or a job payload. Where it holds for each path this PR touches (lines on `f0aab6b390`, prototype lines where new):
  - **The route.** `own_key` is a local of `create_run` (prototype `evaluations_crud.py:334`), passed only as the `provider_credentials=` keyword to `svc.create_run` (prototype `:345`). The request's `config` is read before it (`:331`) and never gets the handle; the idempotency record stores the run id only (`:344-345` on the base, prototype `:349`).
  - **The service.** `UnifiedEvaluationService.create_run` stores the request's `config` in the run row (`unified_evaluation_service.py:493-499`; `Evaluations_DB.create_run` `json.dumps(config)`, `Evaluations_DB.py:1949`), audits run, evaluation and target model (`:485-490`) and sends the started webhook with `eval_id`, `target_model` and `eval_type` (`:501-512`); none of them sees the handle. The handle is the top-level key `eval_config["provider_credentials"]` (prototype `:528`), never inside `eval_config["config"]`. The runner reads it only from the top level (`config.get("provider_credentials")`, where the eval functions' `config` is `eval_config`), so a client that sends `config.provider_credentials` in the run's JSON never supplies a handle.
  - **The task.** `asyncio.create_task(self._run_evaluation_async(..., eval_config=eval_config))` (`:527-535`) passes the dict by reference through the circuit breaker (`circuit_breaker.py:179-196`, no copy) to `run_evaluation` and `_execute_evaluation` (`eval_runner.py:253-277`, `:278-494`). What they store is built from sample results, not from `eval_config`: progress (`:321`, `:337`), checkpoints (`:425-437`), results (`:357`, `:471`), the failure message (`:483`); the run webhooks (`:360`, `:476`, `:488`; `_send_webhook` `:2417-2458`; the service's `:569-581`, `:597-602`, `:617-622`) carry run id, status, results or an error string. `_process_batch` deep-copies the samples only (`:1354`). No resume path re-reads a stored run.
  - **The results.** No judge path puts the handle in its sample result: summarization and RAG return scores; quality's `grader` records provider and model (`:1630-1634`) and `metric_results` keeps only `score`, `raw_score`, `rationale` and `rubric` (`:1635-1639`); label choice and NLI return the prediction and labels.
  - **The backstop.** `ProviderCallCredentials` has a redacted `repr` (`provider_credential_runtime.py:269`) and refuses pickle, copy, deepcopy and pydantic serialization (`:272-290`); `json.dumps` raises on it. An accidental copy or store raises instead of leaking, and an error string that includes it shows `credentials=[REDACTED]`.
  - **Jobs.** `POST /evaluations/{eval_id}/runs` is an asyncio task, not a Jobs job. The Jobs-backed evaluation paths, recipe runs (`evaluations_recipes.py:253`, `recipe_runs_jobs.py:77`) and embeddings A/B tests (`evaluations_embeddings_abtest.py:277`), are untouched and never see a handle.
  - **Could any path carry it?** One receives it without using it: the `rag_pipeline` branch (`_execute_rag_pipeline_run`, `eval_runner.py:590`) gets `eval_config`, but no handle is issued for a `rag_pipeline` run; it would reach that branch only if the evaluation were edited to `rag_pipeline` between the boundary's read and the task's (the review's PATCH race). That branch reads only `config`, `created_by` and `user_id` from `eval_config` (`:605`, `:645`, `:1235-1262`), never `provider_credentials`, and stores none of it, so its calls stay on the server's key, checked at the call. No path stores or queues the handle.
- Every test run in this plan runs with outbound connections blocked: `PYTHONPATH=<plans dir> NETPROBE_LOG=<file> $PY -m pytest -p netprobe_15427 ...`, or inside `block_live_network()`.
- `proto_15426_c.patch` is normative for docstrings and comments the plan does not quote; the docs prose is quoted in Task 3.
- ruff: no new finding on a touched file against the base (as PR A, Task 3 Step 3).

## Review Focus

1. **The judge named only by the run config** (`{"config": {"api_name": "openai"}}`, the spec silent). Expected: the boundary asks about `openai` and the run gets the handle. Pinned by `test_create_run_names_its_judge_provider_to_the_llm_quota[own-key]` (Task 1).
2. **An over-limit user on the server's key.** Expected: the same 402 and the run never created. Pinned by `[server-key]` (Task 1).
3. **A raw `api_key` in the run config, a `rag_pipeline` run, a run with no LLM call.** Expected: the 402 with no lookup. Pinned by `[raw-api-key]`, `[rag-pipeline]`, `[no-llm]` (Task 1).
4. **Every judge path of a run carries the handle.** Expected: summarization, RAG, response quality, label choice and NLI all hand it on; the adapter request for the last two carries the user's key, recorded `user`. Pinned by `test_every_judge_evaluator_of_a_run_gets_the_users_handle[*]` and `test_a_runs_direct_judge_calls_carry_the_users_key[*]` (Task 2).
5. **A run without a handle.** Expected: the shared evaluators and today's raw-key path, unchanged. Pinned by the existing `test_eval_runner.py` tests, which pass `{}` as the config (wider set).
6. **A handle for another provider reaches a direct judge call** (an evaluation edited, `PATCH`, between the boundary's read and the task's first read). Expected: `_call_adapter_text` raises `ChatConfigurationError` before any adapter, as `ms_g_eval._call_adapter_text` does; no call runs on any key. Before review C-1 only the adapter's own bind stopped it. Pinned by `test_a_direct_judge_call_refuses_a_handle_for_another_provider` (Task 2). (Label choice and NLI catch that error around their call and score the sample 0.0, as they do any `ChatConfigurationError` today; V9. `TASK-<FJ>`, filed in Task 0, makes it a sample error.)
7. **The handle is never stored** (carry rule). Expected: the run row's `config` is the request's, and nothing built from `eval_config` is stored, sent or queued. Shown by reading (V10, the "Carry rule" constraint); no test stores a run on the real service with a handle, and `ProviderCallCredentials` raises on any copy or serialization.

## Files

- Modify: `tldw_Server_API/app/core/Evaluations/eval_runner.py` (imports; `_call_adapter_text` `:103-140`; new `run_llm_provider` above `class EvaluationRunner` `:146`; `@staticmethod` on `:169`; `:1454-1462`, `:1531-1532`, `:1595-1602`, `:1925-1926`, `:2119-2120`)
- Modify: `tldw_Server_API/app/core/Evaluations/unified_evaluation_service.py` (`create_run` `:452-524`)
- Modify: `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_crud.py` (`:12` import, `:326-342`)
- Modify tests: `tldw_Server_API/tests/Evaluations/test_evaluations_crud_create_run_api.py`, `tldw_Server_API/tests/Evaluations/unit/test_eval_runner.py`, `tldw_Server_API/tests/Evaluations/test_evaluations_crud_endpoint_sanitization.py`, `tldw_Server_API/tests/Evaluations/unit/test_evaluations_identity_route_bindings.py`
- Docs (Task 3): `Docs/Operations/Usage_Quotas.md`, `Docs/User_Guides/Server/BYOK_User_Guide.md`, `Docs/ADR/083-llm-quotas-skip-calls-on-the-users-own-key.md` (+ the two mirrors). Not the benchmark guide or the Watchlists API, which PR B edits: neither mentions evaluation runs (checked).
- Create (Task 0): `Docs/superpowers/plans/2026-11-07-own-key-evaluation-runs.md`; with backlog-py, `TASK-<FJ>`'s file and a note on TASK-15426 (`backlog/tasks/`)

## Test commands

`$FOCUSED_C`:

```text
tldw_Server_API/tests/Evaluations/test_evaluations_crud_create_run_api.py
tldw_Server_API/tests/Evaluations/unit/test_eval_runner.py::test_every_judge_evaluator_of_a_run_gets_the_users_handle
tldw_Server_API/tests/Evaluations/unit/test_eval_runner.py::test_a_runs_direct_judge_calls_carry_the_users_key
tldw_Server_API/tests/Evaluations/unit/test_eval_runner.py::test_a_direct_judge_call_refuses_a_handle_for_another_provider
tldw_Server_API/tests/Evaluations/test_evaluations_crud_endpoint_sanitization.py
```

Measured on `f0aab6b390` under `-p netprobe_15427` (`-n 4`): RED (this PR's tests, the base's source) `8 failed, 23 passed (0:30)`; GREEN `31 passed (0:23)`; 0 netprobe attempts in both. (Before the refresh: RED 7 failed, 23 passed on `cb70f0c5d1` + A, without the C-1 test; GREEN 31 passed in the fold on `7a020da14a`.)

**The wider set:**

```bash
git grep -l -E 'eval_runner|EvaluationRunner|_call_adapter_text|UnifiedEvaluationService\(|unified_evaluation_service|evaluations_crud|/evaluations/[^"]*/runs' -- 'tldw_Server_API/tests/*.py' | grep '/test_[^/]*\.py$'
```

60 files on `f0aab6b390` and on the prototype (61 on `cb70f0c5d1`: `test_eval_test_mode_truthiness.py` was deleted on dev with #3433's `_await_webhook_inline_in_test_mode`). Three take a PostgreSQL fixture (`pg_eval_params`, `pg_database_config`, `pg_restricted_backend`): `test_evaluations_postgres_crud.py`, `test_run_progress_response.py` and `test_eval_runner.py`. Run them serially with a session-unique `TEST_DB_NAME`, and the other 57 with `-n 4 --dist loadfile`. The grep misses no caller: a second grep for the other touched or new names (`_resolve_eval_provider_model_api_key`, `run_llm_provider`, `get_unified_evaluation_service_for_user`, and run paths written as `{eval_id}/runs` or ending `/runs"`) adds three files, none of which reaches an evaluation run (`AuthNZ/integration/test_prompt_studio_run_history.py` is Prompt Studio's runs; `Services/test_openapi_contracts.py` and `Services/test_router_groups_contract.py` name other routers' `/runs`). In the app, the route is the only caller of `UnifiedEvaluationService.create_run`, and the service is the only builder of `EvaluationRunner`. PR B's wider set is not needed: this PR touches none of B's code (V9).

Measured on `f0aab6b390` under `-p netprobe_15427` (base → prototype): the 57 files `821 passed, 42 skipped (11:03)` → `1 failed, 824 passed, 42 skipped (15:34)`; the three PostgreSQL files, serially with `TEST_DB_NAME=tldw_test_refreshc` (PostgreSQL up, every case ran), `84 passed` → `90 passed`; 0 netprobe attempts on both trees. The net change is +4 cases in the 57 (the route pin becomes five cases) and +6 in `test_eval_runner.py`. The prototype's one failure, `test_evaluations_stage1_route_and_error_regressions.py::test_geval_capacity_exhaustion_returns_detached_503_and_closes_runtime` (`assert 200 == 503`), is load: its file passes alone on the prototype (30 passed), the route it drives (`/evaluations/geval`) is not touched here, and the machine's load average was about 30. Before the refresh (history, not re-measured): the four PRs' union run on `cb70f0c5d1`, in `plan_15426_c.pre_refresh.md`.

---

## Task 0: Plan, branch, draft PR, baseline

- [ ] **Step 1: Branch** after PR B (#3452) merges (PR A and the judge fix are already in dev): `git fetch origin dev && git merge-base --is-ancestor origin/feat/own-key-benchmarks-watchlists-15426 origin/dev && git switch -c feat/own-key-evaluation-runs-15426 origin/dev` (the check exits 0; if B's branch was deleted after merging, check `gh pr view 3452 --json state` shows `MERGED` instead).
- [ ] **Step 2: Follow-up task, note and plan file** (controller ruling, 2026-11-12 fold: this PR files the label-choice/NLI follow-up the refresh surfaced, "For the owner" at the end).
  - (a) **Id.** `mkdir <scratchpad>/id_lock` (if it exists, another session holds the lock: wait and retry; never remove a lock you did not create). `git fetch origin dev --quiet`, then `zsh <plans dir>/task_next.sh` from the repository root: it prints the HIGHEST id in use on `origin/dev` and in every open PR, so `<FJ>` = printed + 1. Check that `ls backlog/tasks backlog/completed backlog/archive/tasks | grep "^task-<FJ> "` prints nothing. From here on write the id literally in every command.
  - (b) **File TASK-<FJ>** (`task create` ignores `--append-notes`, `tools/backlog-py/src/backlog_py/cli/main.py:85-97`, so the facts go in `-d` and the cross-reference in (c)):

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task create "Label-choice and NLI evaluation runs score a failed judge call 0.0" \
  --id TASK-<FJ> -l evaluations,bug \
  -d "Found by TASK-15426 PR C's plan refresh and review (2026-11-12). Pre-existing: the same on the server's key. A label-choice or NLI fact-check evaluation run (POST /api/v1/evaluations/{eval_id}/runs) scores a sample whose judge call failed as a wrong answer. _eval_label_choice and _eval_nli_factcheck wrap their _call_adapter_text call in except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS and set the reply to an empty string (core/Evaluations/eval_runner.py:1929-1931 and :2123-2125; lines on 04905e02ea). That tuple (:74-84) holds ChatConfigurationError, ConnectionError and TimeoutError, as well as OSError, ValueError, TypeError, KeyError, RuntimeError, AttributeError and the httpx and aiohttp client errors. The empty reply parses to no label, so the sample scores accuracy 0.0 with passed False (:1982-1995, :2175-2187). A vendor 401, 403 or 429 and a quota refusal are not in the tuple: they already become a sample error in _process_batch (:1408-1420). With TASK-15426 PR C the same 0.0 also covers a handle refused for another provider (an evaluation edited between the run route's read and the task's) and a ConnectionError or TimeoutError on the user's own key. Intended, following the owner's 2026-11-10 ruling for benchmark items (TASK-15485: a failed judge call is not a score; neither a success nor a failure, left out of every average): such a sample becomes a sample error, not 0.0. Note: the run's aggregate scores every sample error 0.0 today, for every run type (_calculate_aggregate_results, :2319-2322, and pass_rate divides by every sample, :2348), while by_metric leaves it out (_calculate_metric_stats), so a sample error alone does not leave mean_score." \
  --ac "A label-choice run whose judge call raises an error _eval_label_choice catches today (ChatConfigurationError, ConnectionError, TimeoutError) records that sample as a sample error (its error set, counted in the run's failed_samples, no accuracy score), not accuracy 0.0, with a test" \
  --ac "The same for an NLI fact-check run, with a test" \
  --ac "The run's aggregate leaves such a sample out of mean_score and pass_rate, as by_metric already does, with a test. This changes the aggregate of every run type with a sample error; the owner confirmed it on 2026-11-12 (\"yes\")"
```

  - (c) **Note on TASK-15426:** `PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-15426 --append-notes "Plan, PR C of 4 (evaluation runs): Docs/superpowers/plans/2026-11-07-own-key-evaluation-runs.md. Filed TASK-<FJ>: label-choice and NLI evaluation runs score a failed judge call 0.0 (pre-existing; the owner's 2026-11-10 benchmark ruling applied to runs)."` (the task id is required: a bare `15426` raises `KeyError: 'Task not found: 15426'`).
  - (d) **Plan file and commit.** Copy this plan to `Docs/superpowers/plans/2026-11-07-own-key-evaluation-runs.md` unchanged (it keeps the `<FJ>` placeholder and ends without a blank line). `git add Docs/superpowers/plans/2026-11-07-own-key-evaluation-runs.md && git add -A backlog/tasks && git diff --cached --check` must print nothing; `git grep -n "TASK-<FJ>" -- ':!Docs/superpowers/plans'` must print nothing; `/usr/bin/python3 Helper_Scripts/ci/check_backlog_task_ids.py .` exit 0. Commit `chore(backlog): start TASK-15426 PR C, file TASK-<FJ> and add its plan`.

  Dry-run on `a57208cfd3` with a throwaway id (TASK-99901, removed afterwards): (b) and (c) succeed, the new file has the two labels and three ACs, `check_backlog_task_ids.py` exits 0 and `git diff --check` on `backlog/tasks` is clean.
- [ ] **Step 3: Draft PR** (`--base dev`), title `feat(evaluations): evaluation runs run on the user's stored key (TASK-15426, 3 of 4)`. Then release the id lock: once `gh pr view <n> --json files --jq '.files[].path' | grep 'task-<FJ> '` prints the new task's file, `rmdir <scratchpad>/id_lock`.
- [ ] **Step 4: Baseline** the wider set (under the outbound blocker; PostgreSQL files serially); record counts, failing ids and netprobe lines. Expected, as measured on `f0aab6b390`: the 57 files `821 passed, 42 skipped`, the three PostgreSQL files `84 passed` (their PostgreSQL cases skip if PostgreSQL is down), no netprobe line. Three tests are load-sensitive: two capacity tests in `test_evaluations_stage1_route_and_error_regressions.py`, `test_geval_capacity_exhaustion_returns_detached_503_and_closes_runtime` (failed once under load in the refresh, passes alone) and `test_response_quality_capacity_failure_drains_admitted_siblings_before_runtime_close` (flaked in the pre-refresh base run), and `test_embeddings_abtest_multi_user_api.py::test_dedicated_abtest_create_uses_string_tenant_scope_db` (`get_abtest(...)` is None; failed in the 2026-11-12 review's prototype run and in the C/D review's base run, passes alone). A failure outside these is checked against a recent dev run before it is called flaky.

## Task 1: The run route names its judge (AC #3)

**Files:** `eval_runner.py` (`run_llm_provider`, the staticmethod), `unified_evaluation_service.py`, `evaluations_crud.py`, `test_evaluations_crud_create_run_api.py`, `test_evaluations_crud_endpoint_sanitization.py`, `test_evaluations_identity_route_bindings.py`.

**Interfaces:**
- Consumes: `enforce_llm_quota_on_own_key` (`usage_quota_deps.py:155`) and, in tests, `users_own_key` (`tests/helpers/llm_quota.py:63`); both PR A, merged.
- Produces: `def run_llm_provider(evaluation: Optional[Mapping[str, Any]], run_config: Optional[Mapping[str, Any]]) -> tuple[Optional[str], Optional[str]]` in `eval_runner.py`; `UnifiedEvaluationService.create_run(..., provider_credentials: Any = None)`; `eval_config["provider_credentials"]` (the handle or None).

- [ ] **Step 1: Flip the pin.** Replace `test_create_run_refuses_an_over_quota_user_before_the_run_starts` with:

```python
_GRADED = {"eval_type": "model_graded", "eval_spec": {"sub_type": "summarization"}}


@pytest.mark.integration
@pytest.mark.parametrize(
    ("evaluation", "config", "stored", "admitted", "asks"),
    [
        (_GRADED, {"api_name": "openai"}, "openai", True, ["openai"]),
        (_GRADED, {"api_name": "openai"}, "anthropic", False, ["openai"]),
        (_GRADED, {"api_name": "openai", "api_key": "sk-raw"}, "openai", False, []),
        ({"eval_type": "model_graded", "eval_spec": {"sub_type": "rag_pipeline"}}, {}, "openai", False, []),
        ({"eval_type": "exact_match", "eval_spec": {}}, {}, "openai", False, []),
    ],
    ids=["own-key", "server-key", "raw-api-key", "rag-pipeline", "no-llm"],
)
def test_create_run_names_its_judge_provider_to_the_llm_quota(
    evals_crud_client, monkeypatch, evaluation, config, stored, admitted, asks
):
```

  Body: a `_SvcStub` with `async def get_evaluation(self, eval_id, **_kwargs): return evaluation` and `async def create_run(self, eval_id, target_model, provider_credentials=None, **_kwargs)` that records `(getattr(provider_credentials, "api_key", None), current_llm_call_context().prechecked)` and returns the run dict as today; `refuse_llm_work`; `asked = users_own_key(monkeypatch, stored)`; post `{"target_model": "gpt-4o-mini", "config": config}`. Assert `asked == asks`; admitted: 202 and `started == [("sk-users-own-key", False)]`; else `assert_llm_quota_402(r)` and `started == []`. The file has three other tests; two of them define a `_SvcStub` (`test_create_run_uses_typed_model` and `test_create_run_returns_503_on_mandatory_audit_failure`). Give both stubs `get_evaluation` returning None (and `test_create_run_uses_typed_model`'s stub a `provider_credentials=None` parameter); give `_RunLookupDebugService` and `_RunRecordDebugService` (`test_evaluations_crud_endpoint_sanitization.py`) `async def get_evaluation(self, *_args, **_kwargs) -> None: return None`; and in `test_evaluations_identity_route_bindings.py::test_create_run_uses_canonical_identity_for_service_and_webhook_owner` give `_Service` `async def get_evaluation(self, eval_id, *, created_by)` that records `captured["evaluation_owner"] = created_by` and returns None, add `provider_credentials=None` to its `create_run`, and assert `captured["evaluation_owner"] == "tenant-created"` (the route reads the evaluation as the canonical owner). Measured: without them the wider set fails that test (`AttributeError: '_Service' object has no attribute 'get_evaluation'`).
- [ ] **Step 2: Run** the two files `test_evaluations_crud_create_run_api.py` and `test_evaluations_crud_endpoint_sanitization.py` (8 + 17 = 25 tests). Expected on unchanged source: 2 failed, 23 passed (`[own-key]`, `[server-key]`; measured on `f0aab6b390`, again in the 2026-11-12 fold).
- [ ] **Step 3: Implement.**
  - `eval_runner.py`: `from collections.abc import Mapping` (in sorted position, before `from contextlib import suppress`: placed after `datetime`, as the first prototype had it, ruff reports a new `I001`, measured); `from tldw_Server_API.app.core.LLM_Calls.provider_identity import canonical_provider_name` (the module `unified_evaluation_service.py:93` imports it from, as the prototype does; it re-exports `core.Utils.provider_identity`'s); `_resolve_eval_provider_model_api_key` gets `@staticmethod` (drop `self`; call sites unchanged); above the class:

```python
_LLM_RUN_TYPES = frozenset({"model_graded", "label_choice", "nli_factcheck"})


def run_llm_provider(evaluation, run_config):  # annotated as in Interfaces
    """The provider and model every LLM call of an evaluation run uses, as the runner resolves them (TASK-15426).

    ``(None, None)`` when the run cannot be placed on one key: no evaluation, a type that makes no LLM
    call, a raw ``api_key`` in the spec or the run config (it counts), or a ``rag_pipeline`` run, whose
    pipeline stages pick their own providers.
    """
    spec = (evaluation or {}).get("eval_spec") or {}
    if (evaluation or {}).get("eval_type") not in _LLM_RUN_TYPES or spec.get("sub_type") == "rag_pipeline":
        return None, None
    api_name, model, api_key = EvaluationRunner._resolve_eval_provider_model_api_key(spec, {"config": run_config})
    return (None, None) if api_key else (canonical_provider_name(api_name), model)
```

  - `UnifiedEvaluationService.create_run(..., webhook_user_id=None, provider_credentials: Any = None)`: add `"provider_credentials": provider_credentials,` to `eval_config`; docstring: the handle runs every judge call on its key and is never stored.
  - `evaluations_crud.create_run`: import `enforce_llm_quota_on_own_key` (in place of `enforce_llm_quota`) and `run_llm_provider`; move the `target_model` and `config` lines above the check, and replace the check with

```python
        # Usage quotas (LLM usage accounting spec §4a): a run's samples are LLM calls, made in a task
        # started from this request, so the run is checked here, naming its judge, and then runs to completion
        # on the user's own key when they stored one (TASK-15426).
        provider, model = run_llm_provider(await svc.get_evaluation(eval_id, created_by=identity.created_by), config)
        own_key = await enforce_llm_quota_on_own_key(getattr(current_user, "id", None), provider, model=model)
```

    and pass `provider_credentials=own_key` to `svc.create_run`.
- [ ] **Step 4: Run** the two files of Step 2 and `test_evaluations_identity_route_bindings.py`, the third file this task edits. Expected: all pass (30 passed, measured in the 2026-11-12 fold).
- [ ] **Step 5: Commit** `feat(evaluations): the run route names its judge provider (TASK-15426)`.

## Task 2: Every judge call of the run carries the handle (AC #2)

**Files:** `eval_runner.py` (the five paths, `_call_adapter_text`), `tests/Evaluations/unit/test_eval_runner.py`.

**Interfaces:**
- Consumes: `eval_config["provider_credentials"]` (Task 1).
- Produces: `_call_adapter_text(..., provider_credentials: Any = None, **extra_kwargs)`.

- [ ] **Step 1: Write the failing tests** at the end of `test_eval_runner.py` (two tests, so none stacks four `monkeypatch` calls: the file is integration-marked and `test_quality_triage.py --enforce` flags `mock_density` there; measured on the first prototype, one five-way test gave `mock_density=1 (new)`). Shared: `_SAMPLE` (one sample with every input key the five functions read: `source_text`, `summary`, `query`, `contexts`, `response`, `prompt`, `question`, `claim`, `evidence`; `expected` `"A"`), `_SPEC = {"allowed_labels": ["A", "B"]}`, and

```python
async def _own_key_run(monkeypatch, tmp_path):
    """A runner and a run config carrying the user's own-key handle for OpenAI (TASK-15426)."""
    users_own_key(monkeypatch, "openai")
    handle = await users_own_key_credentials(1, "openai")
    runner = EvaluationRunner(db_path=str(tmp_path / "evals.db"))
    return runner, handle, {"config": {"api_name": "openai"}, "provider_credentials": handle}
```

  `test_every_judge_evaluator_of_a_run_gets_the_users_handle`, parametrized `["summarization", "rag", "response_quality"]`: a fake evaluator class recording `provider_credentials` at `__init__` and at `evaluate` whenever it is given; patch `eval_runner.run_geval` (record `kwargs["provider_credentials"]`), `eval_runner.RAGEvaluator` and `eval_runner.ResponseQualityEvaluator` (the fake; the runner builds its quality evaluator lazily from that name, so do not assign `runner._quality_evaluator`, which counts as a stub injection). Call the function; `assert "error" not in result` and `assert seen and all(item is handle for item in seen)`.

  `test_a_runs_direct_judge_calls_carry_the_users_key`, parametrized `["label_choice", "nli_factcheck"]`: patch `eval_runner.get_adapter_or_raise` with an object whose `chat` records the request and returns `{"choices": [{"message": {"content": "A"}}]}`; call the function; assert no error and (the key imported from `provider_credential_runtime`, not read through `eval_runner`, which imports it only in Step 3, and read with `.get`, so RED fails on the behaviour, review C-3)

```python
    from tldw_Server_API.app.core.Provider_Credentials.provider_credential_runtime import (
        PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
    )
    ...
    assert [(r["api_key"], call_credential_source(r), r.get(PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY)) for r in requests] == [
        ("sk-users-own-key", "user", handle)
    ]
```

  `test_a_direct_judge_call_refuses_a_handle_for_another_provider` (review C-1):

```python
@pytest.mark.asyncio
async def test_a_direct_judge_call_refuses_a_handle_for_another_provider(monkeypatch, tmp_path):
    """A handle issued for another provider fails the call before any adapter, so it never runs on a key (TASK-15426)."""
    from types import SimpleNamespace

    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
    from tldw_Server_API.app.core.Evaluations import eval_runner

    _runner, handle, _config = await _own_key_run(monkeypatch, tmp_path)
    adapters: list[str] = []

    def _adapter(provider):
        adapters.append(provider)
        return SimpleNamespace(chat=lambda _request, timeout=None: {"choices": [{"message": {"content": "A"}}]})

    monkeypatch.setattr(eval_runner, "get_adapter_or_raise", _adapter)

    with pytest.raises(ChatConfigurationError) as refused:
        eval_runner._call_adapter_text(
            api_endpoint="anthropic", messages_payload=[{"role": "user", "content": "A or B?"}], model="claude-test",
            provider_credentials=handle,
        )
    assert (refused.value.provider, refused.value.message) == ("anthropic", "Provider credential context is invalid.")
    assert adapters == []
```

  The explicit `model` matters: without it the call fails "Model is required" before the adapter whatever the handle, so the test passed without the fix (measured). The assertion on the refusal's provider and message matters too: without it the test asserts only its own stub collector (`pytest.raises` is not an `assert`), and `test_quality_triage.py --enforce` reports `tautology_suspect=1 (new)` for `test_eval_runner.py` (2026-11-12 review M1).

- [ ] **Step 2: Run them.** Expected on unchanged source: 6 failed (measured on `f0aab6b390`, the base source): summarization: `run_geval` gets no `provider_credentials`, so the sample errors (`'error': "'provider_credentials'"`); RAG and quality: `seen` empty; label choice and NLI: `[(None, None, None)] != [("sk-users-own-key", "user", handle)]` (no server key is configured in the test environment, so the server's key is None); the C-1 test: `DID NOT RAISE ChatConfigurationError` (the call reached the `anthropic` adapter carrying the user's OpenAI key). With Step 3 but without its C-1 line, only the C-1 test fails (measured in the fold on `7a020da14a`: 1 failed, the same reason).
- [ ] **Step 3: Implement** (all reads are `config.get("provider_credentials")`, None for every caller that does not set it):
  - `_call_adapter_text`: new keyword `provider_credentials: Any = None`; when given, first check its provider as `ms_g_eval._call_adapter_text` does (review C-1), then copy its key:

```python
    if provider_credentials is not None:  # the user's own key, recorded as theirs (TASK-15426)
        if not is_runtime_issued_provider_call_credentials(provider_credentials, provider=provider):
            raise ChatConfigurationError(provider=provider, message="Provider credential context is invalid.")
        api_key, app_config = provider_credentials.api_key, provider_credentials.app_config
        extra_kwargs.update({PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY: provider_credentials, "credentials_resolved": True})
```

    Import `PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY` and `is_runtime_issued_provider_call_credentials` from `provider_credential_runtime`.
  - `_eval_summarization`: `provider_credentials=config.get("provider_credentials")` in the `run_geval` partial.
  - `_eval_rag`: `handle = config.get("provider_credentials")`; `evaluator = self.rag_evaluator if handle is None else RAGEvaluator(embedding_provider=None, embedding_model=None, provider_credentials=handle)` (comment: the similarity is the LLM's too, as on `/evaluations/rag`); `await evaluator.evaluate(...)`.
  - `_eval_response_quality`: `provider_credentials=config.get("provider_credentials")` in `quality_evaluator.evaluate`.
  - `_eval_label_choice`, `_eval_nli_factcheck`: `provider_credentials=config.get("provider_credentials")` after `model=model_override,` in their `_call_adapter_text` calls.
- [ ] **Step 4: Run** `$FOCUSED_C` → 31 passed (measured on `f0aab6b390`: 0:23, `-n 4`, 0 netprobe attempts); `test_eval_runner.py` whole, serially with the wider set's two other PostgreSQL files: no new failure (measured: 90 passed, PostgreSQL up, so its three `backend_kind=postgresql` cases ran; the fold, with PostgreSQL down, had skipped them).
- [ ] **Step 5: Commit** `feat(evaluations): a run's judge calls run on the user's stored key (TASK-15426)`.

## Task 3: Docs, verification, review, PR

- [ ] **Step 1: Docs** (one commit with the mirror). Every edit starts from PR B's final text, as #3452 merges it (its "Review fold 3" wording: the raw-`api_key` sentences name the standalone `POST /benchmarks/{name}/run` only), and A's narrowed BYOK sentence.
  - `Usage_Quotas.md`, after PR B's `POST /watchlists/outputs` bullet (reviews C-2, C-5):

```text
  - evaluation runs (`POST /evaluations/{eval_id}/runs`), for the judge provider the runner resolves from the evaluation's spec and the run's `config` (`api_name`, `provider`, `evaluator_model`, a `provider/model` model string; OpenAI when unset), on which every judge call of the run then runs. On the user's own key a RAG run scores answer similarity with the judge, not with embeddings, as `/evaluations/rag` does. A run that cannot be placed on one key is refused: a `rag_pipeline` run, whose pipeline stages pick their own providers, a run with a raw `api_key` in its spec or config, and a run of a type that makes no LLM call;
```

    In the "other boundaries" sentence, `... and a media ingest job's check when it starts, and evaluation runs.` becomes `... and a media ingest job's check when it starts.` PR A's sentence, as B left it, `The features TASK-15426 moved onto a stored key (the study assistants, benchmark runs and watchlist output summaries above) run on the user's own key whenever it serves, over a limit or not: ...` becomes `The features TASK-15426 moved onto a stored key (the study assistants, benchmark runs, watchlist output summaries and evaluation runs above) run on the user's own key whenever it serves, over a limit or not, except the runs and requests the list above says are refused: ...` (the rest unchanged; review C-5: without the qualifier it is untrue for `rag_pipeline` runs and runs with a raw `api_key`).
  - BYOK guide (Usage Quotas), both sentences review C-5 names, plus the owner's revoked-key sentence (review C-4):
    - the over-limit list: `benchmark runs, watchlist output summaries, the judge-model evaluations` becomes `benchmark runs, watchlist output summaries, evaluation runs, the judge-model evaluations`;
    - `The study assistants, benchmark runs and watchlist output summaries use your own key for their provider whenever it is a key that bills you as described above (...)` becomes `The study assistants, benchmark runs, watchlist output summaries and evaluation runs use your own key for their provider whenever it is a key that bills you as described above (...)` (A's narrowing and the rest of the sentence unchanged);
    - PR B's last two sentences, as its branch has them (`dd3e1a50d3`):

```text
A raw `api_key` you send with a standalone benchmark run (`POST /api/v1/benchmarks/{name}/run`) is not your stored key: it counts, so over a limit such a run is refused. Media analysis and evaluation runs never use a key you stored: their calls run on the server's key, or on a raw `api_key` you send with an evaluation run, which counts the same, so over a limit they still refuse.
```

      become (B's standalone scoping kept; review M4: the over-limit list names evaluation runs, so the refused kinds are named too, as Usage_Quotas and ADR-083 do; the four types are the runner's no-LLM ones, `eval_runner.py:1322-1332`, and `[no-llm]` pins the refusal):

```text
A raw `api_key` you send with a standalone benchmark run (`POST /api/v1/benchmarks/{name}/run`) or in an evaluation's spec or run config is not your stored key: it counts, so over a limit such a run is refused, and so are a RAG-pipeline evaluation run and an evaluation run of a type that makes no LLM call (exact match, includes, fuzzy match, proposition extraction). An evaluation run already started keeps using the key it started with if you delete the key; cancel the run (`POST /api/v1/evaluations/runs/{run_id}/cancel`) to stop it. Media analysis never uses a key you stored: its calls run on the server's key, so over a limit it still refuses.
```

  - BYOK guide (Auditing and Updates, PR A's `last_used_at` clause as B extended it): the parenthesis `(the study assistants, benchmark runs and watchlist output summaries)` becomes `(the study assistants, benchmark runs, watchlist output summaries and evaluation runs)`.
  - ADR-083, appended to PR A's amendment section (`## Amendment (2026-11-09)`), after PR B's `**Benchmark runs and watchlist output summaries**` bullet (`<date>` is this PR's date, 2026-11-12 in the prototype):

```markdown
- **Evaluation runs** (<date>): the judge provider the runner resolves from the evaluation's spec and the run's config. The handle is resolved at the request and handed to the run's task, never stored. A `rag_pipeline` run, a run with a raw `api_key` and a run of a type that makes no LLM call name no provider and are refused over a limit, as before.
```

  - `bash Helper_Scripts/refresh_docs_published.sh`; only the two mirrors change (ADR-083 and the BYOK guide; `Usage_Quotas.md` is not mirrored; measured on `f0aab6b390`).
  - PR B also edits the benchmark guide's "402 LLM quota" and the Watchlists API's `summarize` text. This PR leaves both alone: neither mentions evaluation runs (checked on `f0aab6b390`), and no other doc outside `Docs/Published` and design history says evaluation runs stay on the server's key (git grep).
- [ ] **Step 2: Wider set** with the PR: Task 0's counts plus the new tests, no new failing id (measured on `f0aab6b390`: `1 failed, 824 passed, 42 skipped` and `90 passed` against the base's `821 passed, 42 skipped` and `84 passed`; the one failure is the load-sensitive capacity test of Task 0 Step 4, which passes alone; 0 netprobe attempts).
- [ ] **Step 3: CI checks:** `PATH="$(dirname $PY):$PATH" PYTHONPATH=<plans dir> NETPROBE_LOG=<file> TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -n 4 -p netprobe_15427 tldw_Server_API/tests/Docs tldw_Server_API/tests/lint tldw_Server_API/tests/CI tools/backlog-py/tests`; `$PY Helper_Scripts/ci/check_shard_coverage.py` (`new_uncovered=0`); `$PY Helper_Scripts/ci/test_quality_triage.py --enforce` (OK); `PYTHONPATH=. $PY Helper_Scripts/ci/route_auth_ratchet.py` (baseline exact); `PYTHONPATH=. $PY Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json --check-paths apps/packages/api-types/paths.d.ts` (both OK); `/usr/bin/python3 Helper_Scripts/ci/check_backlog_task_ids.py .` (exit 0); `/opt/homebrew/bin/uvx bandit -ll` on `eval_runner.py`, `unified_evaluation_service.py`, `evaluations_crud.py`. Expected, measured on `f0aab6b390` with this PR's prototype: shard coverage `OK — no newly-unshared test files.`; triage `OK — no offenses beyond the baseline.` (the runner tests stay split, so no test stacks four `monkeypatch` calls, and the C-1 test asserts the refusal's provider and message; re-measured in the 2026-11-12 fold with the plan's C-1 text, exit 0); route-auth `208 unauthenticated routes; baseline exact.`; OpenAPI fingerprint and path union both OK (no route signature changes); `check_backlog_task_ids.py` exit 0; `tests/Docs` with `tests/lint/test_function_size_ratchet.py` (`-n 4`, `-p netprobe_15427`): `227 passed` on base and prototype, 0 attempts (the `Docs/Published` mirror tests pass with the two refreshed mirrors); Bandit `-ll` on C's three source files: `No issues identified.` on base and prototype (one `#nosec`-disabled finding skipped on both). Not re-measured in the refresh: `tests/lint` and `tests/CI` whole and backlog-py's tests; on B's base they had three environment-only failures (`tests/lint/test_runtime_package_discovery.py` errors, `test_jobs_postgres_exports_recorded_coverage` 2 cases; the shared venv lacks `setuptools` and `coverage`), which are expected here too. Function sizes: `function_size_baseline.json` freezes only `_execute_rag_pipeline_run` of C's files (460 → 460, untouched); the touched functions, none frozen, all under the 300-line threshold: `_call_adapter_text` 40 → 46, `run_llm_provider` new (14), `_resolve_eval_provider_model_api_key` 50 → 49, `_eval_summarization` 73 → 74, `_eval_rag` 69 → 74, `_eval_response_quality` 66 → 67, `_eval_label_choice` 195 → 196, `_eval_nli_factcheck` 191 → 192, `UnifiedEvaluationService.create_run` 92 → 97, the route's `create_run` 64 → 67. `scope_predicate_baseline.txt` lists none of the 12 files: no regeneration. ruff on every touched `.py` file against the base: no new finding (measured on `f0aab6b390`, venv ruff 0.16.5, line numbers stripped: the same 6 findings on base and prototype, `eval_runner.py`'s `I001` and `TRY203`, `evaluations_crud.py`'s `I001`, and `test_evaluations_crud_create_run_api.py`'s `I001`, `UP006`, `UP035`; nothing new once `Mapping` is in sorted position).
- [ ] **Step 4: Network.** `$FOCUSED_C` inside `block_live_network()`: `attempts == []` (measured on `f0aab6b390` under `-p netprobe_15427`: 0 attempts for `$FOCUSED_C`, RED and GREEN, and for the wider set on both trees).
- [ ] **Step 5: Close-out notes:** `PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-15426 --append-notes "PR C (<url>): evaluation runs name their judge provider (run_llm_provider, the runner's own resolution) and run every judge call on the user's own key (handle in eval_config, never stored; summarization, RAG, response quality, label choice, NLI; a direct judge call checks the handle's provider). rag_pipeline, raw api_key and no-LLM runs still refused over a limit. A run already started keeps its key (owner, 2026-11-07); the BYOK guide says to cancel it. Built on PR B (#3452). Follow-up: TASK-<FJ> (label choice and NLI score a failed judge call 0.0). Tests: <counts>. Bandit: no issues. Docs updated (mirrored)."` (the id written literally).
- [ ] **Step 6: Review and PR:** the clean-session `/superpowers:requesting-code-review` on the branch diff with this plan and the task file (address every finding, re-run the focused set); then the PR body (Summary; a full `## Change summary` section recording the owner's waiver of 2026-11-07, "Yes, waive for all"; Test plan with the measured counts; Review Focus; ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`), mark ready, diff the PR's Frontend E2E failures against a recent dev run, and merge through `<plans dir>/pr_queue3.sh` (check `gh variable get MERGE_QUEUE` first: not set on 2026-11-12, so rebase only this PR when it is next, wait for the required statuses, then merge; the script reads dev's required checks from the ruleset).

## Review fold (2026-11-07)

Folded from `review_plan_15426_cd.md` (C: 0 Blocker, 0 Important, 5 Minor), the A+B review's A-6, and the owner's decisions of 2026-11-07. The pre-fold texts are `plan_15426_c.prefold.md` and `proto_15426_c.prefold.patch`. The fold was measured on `7a020da14a` with the folded A and B applied, outbound connections blocked (`-p netprobe_15427`, 0 attempts in every run); PostgreSQL was down, so the `backend_kind=postgresql` cases of `test_eval_runner.py` (3) were skipped, not run.

| Finding / decision | Where it landed |
|---|---|
| C-1: `_call_adapter_text` trusted the handle without checking its provider | Task 2 Step 3 and the patch: the `is_runtime_issued_provider_call_credentials(handle, provider=provider)` check, as in `ms_g_eval`; new test `test_a_direct_judge_call_refuses_a_handle_for_another_provider` (Task 2 Step 1, `$FOCUSED_C`, Review Focus 6). RED without the line: `DID NOT RAISE ChatConfigurationError`; GREEN with it. |
| C-2: Decision 5 changes RAG similarity for every user with a stored key | Decision 5 and the Usage_Quotas bullet (Task 3 Step 1, patch updated). |
| C-3: the label-choice/NLI RED failed on a missing name | Task 2 Step 1: the key imported from `provider_credential_runtime` and read with `.get`; Task 2 Step 2's RED reason is now `[(None, None, None)] != [("sk-users-own-key", "user", handle)]` (measured). |
| C-4: a revoked key mid-run; add the cancel note | Decision 3 and Task 3 Step 1: the BYOK sentence (with the owner's decision). |
| C-5: the plan named only one of the two BYOK sentences; the Usage_Quotas "whenever it serves" sentence was untrue for refused runs | Task 3 Step 1 quotes every BYOK change and the qualifier "except the runs and requests the list above says are refused" (patch updated; PR D keeps the qualifier). |
| Dry-run note: "the file's two other stubs" was easy to misread | Task 1 Step 1 names the two tests whose stubs change. |
| A-6 (from PR A): `last_used_at` | Task 3 Step 1 extends PR A's clause to evaluation runs (patch updated). |
| Owner: a run keeps the key it resolved at its start; cancel it to stop | Decision 3; Task 3 Step 1 adds the BYOK sentence with the cancel route. |
| Owner: a raw `api_key` keeps counting; `rag_pipeline` runs stay out of scope | Decision 4. |
| Owner: no team or org keys; the judge fix before B | PR A Decision 3; PR B (nothing here). |
| Found by the fold's ruff check: `from collections.abc import Mapping` after `datetime` adds an `I001` | Task 1 Step 3 and the patch: sorted position (measured: no new ruff finding on C's files). |
| Fold measurements | Task 2 Step 2 RED 6 failed; `$FOCUSED_C` GREEN 31 passed; `test_eval_runner.py` and the identity-binding file whole 46 passed, 3 skipped (PostgreSQL). |

## Refresh (2026-11-12)

Refreshed onto its real base, `f0aab6b390`: `origin/dev` `04905e02ea` (PR A #3444 and the judge fix #3445, TASK-15485, both merged), then `git merge --no-ff` of PR B (#3452, `feat/own-key-benchmarks-watchlists-15426` at `dd3e1a50d3`, open, not in dev); the merge did not conflict. The pre-refresh texts are `plan_15426_c.pre_refresh.md` and `proto_15426_c.pre_refresh.patch`. Every count was re-measured on that base, base and prototype, under `-p netprobe_15427`. No owner decision or ruling changed.

| What changed | Why |
|---|---|
| Header ("Spec", "Measured on", "Order"), Architecture | PR A's helpers are cited as merged code (`usage_quota_deps.py:91-122`, `:155-162`; `tests/helpers/llm_quota.py:63-97`), not `proto_15426_a.patch`. The base is dev + B, and B goes first: C's source files are disjoint from B's, but its docs rewrite B's final sentences. |
| V2, V3, V5 | Re-cited. `unified_evaluation_service.py` moved 4 lines up (#3433 removed `_await_webhook_inline_in_test_mode`); `eval_runner.py`, `evaluations_crud.py` and the four test files are unchanged since `cb70f0c5d1`, so their citations stand. V5 adds where `RAGEvaluator` sets `credentials_resolved` from the handle. |
| V7 | RED re-measured with the C-1 test: 8 failed, 23 passed. |
| V8 | #3433 merged; #3269 is unchanged and every hunk it shows is already on dev; #3454 is the dev-to-main release. No open PR conflicts. |
| V9 (new) | The brief's interplay check. No code path of C calls into `evaluation_manager.py`, so TASK-15485's `input_is_literal_text=True`, its bound-then-awaited model call and its unscored contract are untouched, and C needs and changes neither them nor B's `provider_credentials` keyword. The runner's own failure handling is recorded (see "For the owner"). |
| V10 and the "Carry rule" constraint (new) | The owner's carry rule, checked for every path that builds, copies, stores, sends or queues the run config or `eval_config`, with lines; one path receives the handle without using it (the `rag_pipeline` branch after a PATCH race) and stores nothing. Evaluation runs are an asyncio task, not a Jobs job; the Jobs-backed evaluation paths are untouched. |
| Decisions 2-3 | Point at V10 and the cancel route's line (`evaluations_crud.py:447`); text and rulings unchanged. Decision 3 checked against V9: a vendor 401 still fails the remaining samples as errors. |
| Review Focus 6, 7 | 6 notes that label choice and NLI score the C-1 refusal 0.0 (pre-existing handling, V9). 7 (new) is the carry rule, shown by reading. |
| Files | The service's lines; the docs by path, and why the benchmark guide and the Watchlists API, which B edits, are not touched. |
| `$FOCUSED_C`, the wider set, Task 0 Step 4, Task 1 Step 2, Task 2 Steps 2 and 4, Task 3 Steps 2-4 | Re-measured (below). The wider set is 60 files (`test_eval_test_mode_truthiness.py` was deleted on dev); a second grep for the other touched names found no missed caller; PostgreSQL was up, so the three PostgreSQL files ran in full, serially, with `TEST_DB_NAME=tldw_test_refreshc`. The old four-PR union run is history (pre-refresh copy). |
| Task 0 Step 1 | Branch only once #3452 is in dev (`merge-base --is-ancestor`, or `gh pr view 3452` MERGED). |
| Task 1 Interfaces | Consumes the merged helpers by `file:line`. |
| Task 3 Step 1 (and the patch) | Every docs edit starts from B's final text. BYOK: the over-limit list; A's narrowed sentence (`whenever it is a key that bills you as described above (...)`) gains evaluation runs, instead of the older `whenever you stored one` the plan quoted; B's last two sentences become C's three, keeping B's review-fold-3 scoping to the standalone `POST /api/v1/benchmarks/{name}/run`. Usage_Quotas: the bullet goes after B's watchlist bullet; the two other edits are as before. ADR-083: after B's bullet, dated 2026-11-12 in the prototype. Mirrors regenerated: the same two. |
| Task 3 Steps 3, 5, 6 | The CI command runs with `-n 4 -p netprobe_15427` (as B's review M5); its Expected is this base's; the close-out note names #3452; the merge step checks `MERGE_QUEUE` (not set on 2026-11-12). |
| Prototype | Rebased: 12 files, +228/-27, as before. The source and test hunks applied unchanged (the service hunk at offset -4); the docs hunks were redone on B's text. Against the pre-refresh patch, only the BYOK paragraph (both copies) and the ADR date differ; every source and test line is the same. |

Measured on `f0aab6b390` (base → prototype), every run under `-p netprobe_15427`:
- `$FOCUSED_C`: RED `8 failed, 23 passed` → GREEN `31 passed`; 0 netprobe attempts.
- The wider set (60 files): 57 files with `-n 4 --dist loadfile` `821 passed, 42 skipped` → `1 failed, 824 passed, 42 skipped`; the three PostgreSQL files serially `84 passed` → `90 passed`; 0 netprobe attempts on both. The one failure is load (`test_geval_capacity_exhaustion_returns_detached_503_and_closes_runtime`, an untouched route; its file passes alone, 30 passed).
- Function sizes (`function_size_baseline.json` freezes only `_execute_rag_pipeline_run` of C's files: 460 → 460, untouched): `_call_adapter_text` 40 → 46, `run_llm_provider` new (14), `_resolve_eval_provider_model_api_key` 50 → 49, `_eval_summarization` 73 → 74, `_eval_rag` 69 → 74, `_eval_response_quality` 66 → 67, `_eval_label_choice` 195 → 196, `_eval_nli_factcheck` 191 → 192, `UnifiedEvaluationService.create_run` 92 → 97, the route's `create_run` 64 → 67; all under 300.
- Bandit `-ll` on the three source files: `No issues identified.` on both (one `#nosec`-disabled finding skipped on both). ruff on the seven touched `.py` files: the same 6 findings on both.
- `scope_predicate_baseline.txt`: lists none of the 12 files; no regeneration.
- `tests/Docs` with the function-size ratchet: `227 passed` → `227 passed`, 0 attempts, with the two mirrors refreshed by `refresh_docs_published.sh`.
- Prototype checks: shard coverage OK, triage OK, route-auth `208; baseline exact`, OpenAPI fingerprint and path union OK, `check_backlog_task_ids.py` exit 0.

Not measured here: `tests/lint` and `tests/CI` whole and backlog-py's tests (only `tests/Docs` and the size ratchet); the union run with D stacked; the C-1 sabotage (the code is unchanged since the fold measured it); the Frontend E2E journeys.

For the owner (surfaced by the refresh, not decided in this plan):
- **Label choice and NLI score a caught judge-call failure 0.0.** Their call is wrapped in `except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS` (`eval_runner.py:1929-1931`, `:2123-2125`), which includes `ChatConfigurationError`, `ConnectionError` and `TimeoutError`; the empty reply then scores the sample `accuracy: 0.0` (`:1982-1994`, `:2175-2186`). This is pre-existing and the same on the server's key, and it is the evaluation-run analogue of what TASK-15485 fixed for benchmark items (unscored, not 0.0). With this PR it also covers the C-1 refusal in the PATCH race and any `ConnectionError` or `TimeoutError` that reaches that handler from a call on the user's own key. A vendor 401 (a revoked key) and a quota refusal are not caught there and become sample errors, so Decision 3 holds. **Recommendation:** keep this PR as planned (it changes no scoring) and file a follow-up, its own PR as TASK-15485 was, since it changes scores: a failed label-choice or NLI call becomes a sample error. (Ruled in the 2026-11-12 fold: filed as `TASK-<FJ>` in Task 0 Step 2.)

## Review fold (2026-11-12)

Folded from `review_refresh_15426_c.md` (ready with fixes: 0 Blocker, 0 Important, 5 Minor) and one controller ruling. The pre-fold texts are `plan_15426_c.pre_fold.md` and `proto_15426_c.pre_fold.patch`. Measured on `a57208cfd3`: `origin/dev` `04905e02ea`, then `git merge --no-ff` of PR B (`dd3e1a50d3`; #3452 still open, not in dev); no conflict, and the tree is identical to `f0aab6b390`, so every `f0aab6b390` citation stands. Every run under `-p netprobe_15427`, 0 netprobe attempts.

| Finding / ruling | Change |
|---|---|
| M1: the quoted C-1 test failed the test-quality ratchet (`tautology_suspect`) | Task 2 Step 1: `as refused` and the assertion on the refusal's provider and message, from the prototype (the plan's C-1 block is now byte-identical to the prototype's); one sentence says why. Task 3 Step 3 notes the re-measure. |
| M2: the plan ended with a blank line | The file ends with one newline; no line has trailing whitespace or a tab (embedded code included). Task 0 Step 2(d) says the copy keeps that. |
| M3: a vendor 401 fails RAG and quality runs, not each sample | Decision 3 and V9 say what happens per run type (summarization, label choice, NLI: each remaining sample errors, the run completes; RAG, quality: the batch's samples error, then the run fails at the checkpoint, `eval_runner.py:441-447`). V9 also re-cites the tuple (`:74-84`, was `:80-90`) and the 0.0 returns (`:1982-1995`, `:2175-2187`). No code or docs change. |
| M4: the BYOK guide did not say no-LLM runs are refused over a limit | Task 3 Step 1 and the prototype: "…and so are a RAG-pipeline evaluation run and an evaluation run of a type that makes no LLM call (exact match, includes, fuzzy match, proposition extraction)." "Proposition extraction", not the review's "propositions": the type is `proposition_extraction`. True against the code: those four are the runner's only no-LLM types (`eval_runner.py:1322-1332`; proposition extraction matches by TF-IDF or Jaccard, `Chunking/utils/proposition_eval.py`), `run_llm_provider` names no provider for them, and `[no-llm]` pins the 402. Mirrors regenerated: still only the two. |
| M5: three ambiguities | Task 1 Step 3 names `tldw_Server_API.app.core.LLM_Calls.provider_identity`; Task 0 Step 2(c) and Task 3 Step 5 give the full `task edit TASK-15426 --append-notes` command; Task 1 Step 2 names the two files (8 + 17 = 25 tests), and Step 4 adds the identity-binding file (30). |
| Review's dry-run note: the abtest flake | Task 0 Step 4 lists it with the load-sensitive tests. |
| Controller ruling: file the label-choice/NLI follow-up | Task 0 Step 2(a)-(d): id under `<scratchpad>/id_lock` with `task_next.sh` (+1), `task create` for `TASK-<FJ>` (labels evaluations, bug; three ACs), the note on TASK-15426, the placeholder check; Step 3 releases the lock once the PR lists the file. The description adds one fact the ruling's intent needs: the run's aggregate scores every sample error 0.0 (`_calculate_aggregate_results`, `eval_runner.py:2319-2322`, `:2348`), so AC #3, leaving it out of `mean_score` and `pass_rate`, changes every run type's aggregate and waits for the owner. Review Focus 6, Files and the close-out note name it. |

Measured (base `a57208cfd3` → prototype):
- `$FOCUSED_C`: RED `8 failed, 23 passed` → GREEN `31 passed`; the C-1 RED reason is still `DID NOT RAISE ChatConfigurationError`.
- Task 1: the two files RED `2 failed, 23 passed`; the three files GREEN `30 passed`.
- `test_quality_triage.py --enforce` on the prototype, whose C-1 test is the plan's text: `OK — no offenses beyond the baseline.`, exit 0. `tests/CI/test_test_quality_ratchet.py` passes (41).
- `tests/Docs` with the function-size ratchet: `227 passed`, with the mirrors refreshed by `refresh_docs_published.sh`. `git diff --check` on the prototype tree: clean.
- Prototype: 12 files, +228/-27, as before; only the two BYOK hunks changed. It applies to `a57208cfd3`.
- No source or test line changed, so function sizes, Bandit, ruff, shard coverage, route-auth, OpenAPI and the wider set stand as the refresh measured them (not re-run).
