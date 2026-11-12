# Benchmark items with no score are "unscored", not successful (a fold into PR #3445, TASK-15485) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A benchmark item with no score is counted as `unscored`: neither `successful` nor `failed`, and in no average. "No score" means the model's own answer call failed (it returned an error or raised), the judge call failed or raised, or the judge's reply held no readable score. Today these score 0.0 or 0.5 and count as `successful`, except a raised model call, which escapes the method: the routes count it `failed` and the CLI run stops.

**Architecture:** `EvaluationManager.evaluate_custom_metric` returns `"score": None` on its no-score paths (it returned 0.0 or 0.5). A model call that raises is caught as the judge's is (a quota refusal still re-raises) and takes the same unscored path. Its three callers handle `None`:
- `/api/v1/benchmarks/{name}/run` (`benchmark_api.py`) and `/api/v1/evaluations/benchmarks/{name}/run` (`evaluations_benchmarks.py`) count `unscored` in the summary. `successful` holds only scored items, so the averages, `by_category` and the stored `scores` drop the rest without further edits.
- The benchmark CLI (`benchmark_cli._save_results`, `benchmark_utils.format_benchmark_summary`, `calculate_accuracy_by_category`) skips `None` and prints an `Unscored:` count.

No new abstraction: each summary block gains two lines.

**Tech Stack:** Python 3.12, pytest (pytest-asyncio), FastAPI TestClient, click CliRunner, backlog-py.

**Spec:**
- Owner decisions (2026-11-10), binding:
  - "no; and it shouldnt be a success either": a failed judge call is neither `failed` nor `successful`. It is a third count, `unscored`, left out of every average.
  - Also `unscored` (the recommended option, both): the model's own answer call failing (`_generate_model_response` returns `Error:`), and a judge reply with no readable score (today the 0.5 default, `Could not parse valid score`).
  - Folded into #3445 (it answers that PR's own Review Focus 2), not a separate PR.
- Controller ruling on the plan review's Minor 2 (binding): a model answer call that fails with a provider error, timeout or rate limit is unscored, also when it raises rather than returning `Error:`. `LLMQuotaExceededError` re-raises; any other exception returns the unscored result, logged as the model call's failure. `failed` keeps meaning "the item's evaluation raised"; the guide says what can still land there.
- The plan review `review_plan_unscored.md` (ready with fixes, 0 Blocker, 0 Important, 7 Minor; all seven folded, see "Review fold").
- TASK-15485 and its plan on the branch, `Docs/superpowers/plans/2026-11-07-benchmark-judge-reaches-the-provider.md`.
- The planner brief `brief_plan_unscored.md` (plans dir).

**Measured on:** `origin/fix/benchmark-judge-15485` at `5916bd586f`, in a detached worktree. The controller's local head `f0acbf2713` adds only `backlog/tasks/task-15488 - ...md` (47 lines), so every count below holds there. `file:line` citations are for `5916bd586f` unless marked "PR".

**Prototype:** `proto_unscored.patch` (plans dir), against `5916bd586f`: 12 files, +252/-45. Before the review fold: `proto_unscored.prefold.patch` (+235/-41) and `plan_unscored.prefold.md`. It holds every code, test and docs change in this plan, the `Docs/Published` mirror included. It does not hold the plan copy or the backlog edits. It applies cleanly to `f0acbf2713` too.

---

## What produces no score today (verified on `5916bd586f`)

| # | Path in `evaluate_custom_metric` | Today | After |
|---|---|---|---|
| U1 | The model's own answer failed: `_generate_model_response` returned `Error: ...` (`evaluation_manager.py:565-572`) | `"score": 0.0` (`:569`), counted `successful` | `None`, with the `Error:` text |
| U1r | The model's own answer call raised (`:560-564`, outside any `try`). `analyze` lets `NetworkError` (e.g. a local server that is down) and `EgressPolicyError` through (`Summarization_General_Lib.py:113-117`, `:427-428`) | escapes the method: the routes count the item `failed`; the CLI's `gather` (no `return_exceptions`) stops the whole run | caught (`LLMQuotaExceededError` re-raises), turned into `Error: <exc>`, then U1's path: `None`, logged `Model response generation failed` |
| U2 | The judge call failed, G1 (`:600-604`): provider error, timeout, rate limit, empty or truncated reply | `"score": 0.0` (`:604`) | `None`, with the `Error:` text |
| U3 | The judge call raised (`except Exception`, `:677-684`): `analyze` raised, or the parse did | `"score": 0.0` (`:681`), "Evaluation failed: ..." | `None`, same explanation |
| U4 | The reply held no readable score (`:663-666`), e.g. "I cannot judge this.", `{"score": [1, 2]}`, `Score: NaN`, a JSON number outside 0-10 | `score = 0.5` (`:666`), warning `Could not parse valid score` | `None`, same warning |

`LLMQuotaExceededError` still propagates from both calls (`:675-676` for the judge's; a new clause for the model's). In the routes' `gather(..., return_exceptions=True)` it would land in `failed`, but both routes check the quota at the boundary and mark the run prechecked (`usage_quota_deps.py:111`), so no item is refused mid-run. **What can still land in `failed`:** an exception outside the two calls, e.g. item data that cannot be built into a prompt (`dict(input_data)` at `:558`, `scoring_criteria.items()` at `:584`, both before the judge's `try`; by code).

**Representation: `"score": None`** (the brief's recommendation; no reason found to prefer a flag).
- A flag beside a float (`"score": 0.0, "scored": False`) keeps a number that any consumer which ignores the flag averages silently. That is the bug class TASK-15485 fixes (the 0.5 placeholders).
- With `None`, a consumer that does arithmetic on it fails loudly. The base proves it: each new route and CLI test fails on `5916bd586f` with a `TypeError` from `sum()`, `float()` or `>=` over the `None` (Task 1 Step 2).
- Every consumer is known (next section), and each one that can see a score is changed here.

## Callers and what reads the score

`git grep -n "evaluate_custom_metric" -- tldw_Server_API apps` (app code, workflows, Prompt Studio, the unified evaluation service, MCP):

| Caller | Reads | Change |
|---|---|---|
| `app/api/v1/endpoints/benchmark_api.py:275` (`run_benchmark`) | `result.get("score", 0.0)` (`:301`, keeps `None`), then `successful_results` (`:306`), `scores` (`:307`), the summary (`:309-316`), `by_category` (`:318-331`), stored `scores` (`:343`) | `successful_results` keeps only scored items; `failed` counts raised items; a new `unscored` key |
| `app/api/v1/endpoints/evaluations/evaluations_benchmarks.py:158` (`run_benchmark`) | `result.get("score", 0.0)` (`:177`), `successful_results` (`:182`), `float(...)` (`:183`, `:198`), stored `scores` (`:210`) | the same |
| `app/core/Evaluations/cli/benchmark_cli.py:343` (`_evaluate_batch`) | results go to `format_benchmark_summary` (`:202`) and `_save_results` (`:370-383`); `--resume-from` reads saved results back (`:140-151`) | the three helpers skip `None` |
| `tldw_Server_API/cli/commands/evaluation.py:743` (`_run_custom_evaluation`) | calls it with `text=`, `custom_prompt=`, `save_results=`, which the method does not take, so it raises `TypeError` before any score; reads only `result.get("metrics")` | none (pre-existing, unreachable score) |

No other app caller: Prompt Studio has its own `evaluation_manager.py`; the unified evaluation service, workflows and MCP do not name it.

**The CLI's resume path.** `--resume-from` loads the saved `results`, now with `"score": null` for unscored items, and passes them to the same `format_benchmark_summary` and `_save_results`, which skip `None`. Resume does not retry unscored items: it skips completed items by index, since the results carry no item id (pre-existing, out of scope).

**What reads a stored result or the summary:**
- **Stored rows.** `save_results` writes `{"summary", "scores"}` to `internal_evaluations`. Nothing a user sees reads them back: `EvaluationManager.get_history` / `compare_evaluations` / `get_evaluation` have no route caller (TASK-15485's notes). Their trend and comparison code reads `results["metrics"]` and the `evaluation_metrics` table, which benchmark rows never fill. So the new `unscored` key and the shorter `scores` list change nothing they show.
- **The WebUI** Runs tab prints the run's response as raw JSON (`apps/packages/ui/src/components/Option/Evaluations/tabs/RunsTab.tsx:619-621`). The new `unscored` key appears there as a line; nothing parses `results_summary`.
- **Exports** (Chatbooks) read the unified `evaluations` table, not `internal_evaluations`.
- **A run with no scored item** keeps the existing empty-scores values: `successful: 0`, `average_score`/`min_score`/`max_score` 0.0, no `by_category`. That is what an all-`failed` run shows today. The guide says these zeros are not a measured zero. Returning `null` instead would change the all-`failed` case too, for no consumer that needs it.

---

## Global Constraints

Verbatim from the brief:
- Tests make no real outbound network requests; stub the provider adapter or its transport; run every test you touch with the probe plugin (`PYTHONPATH=<plans dir> NETPROBE_LOG=<file> ... -p netprobe_15427`).
- Function-size ratchet `tldw_Server_API/tests/lint/function_size_baseline.json`: frozen functions must not grow. Check `evaluate_custom_metric` and both route functions.
- `tldw_Server_API/tests/lint/scope_predicate_baseline.txt` pins line numbers: shifting lines in a listed file fails backend-required unless regenerated (`--write-baseline`).
- Backlog: edit tasks with backlog-py only (`PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task edit ...`); `--append-notes`, never `--notes`; stage with `git add -A backlog/tasks`; never hand-edit task files.
- The PR targets `dev`. Commit trailer: `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. PR body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- Never `--no-verify`, never force-push, never `git stash`.
- Redis: never FLUSHALL or SCRIPT FLUSH. PostgreSQL only through the repo fixtures, with a session-unique `TEST_DB_NAME`; drop only your own objects; never restart the server.
- Mirrored docs (`Docs/Published`) need `bash Helper_Scripts/refresh_docs_published.sh` in the same PR.
- CI checks before the push, in the last task: `tests/Docs`, `tests/lint`, `tests/CI`, backlog-py's tests; shard coverage; `test_quality_triage.py --enforce`; route-auth ratchet; OpenAPI export check; Bandit on touched source; `check_backlog_task_ids.py` exit 0.
- Agent shells don't keep the venv `PATH`: prefix commands that spawn `python` with `PATH="$(dirname $PY):$PATH"`.
- Known environment-only failures in the shared venv (no `setuptools`, no `coverage`): `tests/lint/test_runtime_package_discovery.py` errors and `test_jobs_postgres_exports_recorded_coverage` (2 cases) fails. Do not modify the shared venv.

Plan-specific:
- **Paths.** `$PY` is `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`; write it out in each call. `<plans dir>` is `/Users/macbook-dev/.claude/projects/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/plans`. `<scratchpad>` is the executing session's scratchpad.
- **`<pytest>`** stands for `PATH="$(dirname $PY):$PATH" PYTHONPATH=<plans dir> NETPROBE_LOG=<scratchpad>/netprobe_<label>.log TLDW_TEST_NO_DOCKER=1 TEST_DB_NAME=tldw_test_unscored_<session> $PY -m pytest -p no:cacheprovider -p netprobe_15427`. After each run, check the log: empty, or only the two base lines named below.
- **Probe artifact.** `test_security.py::TestWebhookSecurityValidation::test_ssrf_protection_private_ip` fails under the probe on the base and the fold alike (it resolves numeric IPs); its log line and `test_blocked_ports`'s (`example.com`) are the two base lines.
- **Function sizes and scope baseline (measured).**
  - `function_size_baseline.json` has no entry for any touched function (all under 300 lines, so none is frozen): `evaluate_custom_metric` 138 → 144 lines; `benchmark_api.run_benchmark` 162 → 165; `evaluations_benchmarks.run_benchmark` 132 → 136; `_save_results` 14 → 16; `calculate_accuracy_by_category` 43 → 45; `format_benchmark_summary` 46 → 48.
  - `Helper_Scripts/ci/scope_predicate_baseline.txt` (the file the ratchet reads; `tests/lint/scope_predicate_baseline.txt` does not exist) names none of the touched files: `grep -c -E "evaluation_manager|benchmark_api|evaluations_benchmarks|benchmark_cli|benchmark_utils|Benchmark_Creation"` prints `0`.
- **Branch.** Work on `fix/benchmark-judge-15485` at the controller's head `f0acbf2713`. Do not touch `backlog/tasks/task-15488 - ...md`.
- **Git.** Stage named files. `git diff --cached --check` must print nothing before each commit.
- **Go cache.** If a run leaves a read-only Go module cache: `chmod -R u+w tldw_Server_API/Config_Files/acp_runner_home && git clean -ffdx -- tldw_Server_API/Config_Files/acp_runner_home`. Never `rm -rf` it. The last `git status --short` must be empty.
- **Load.** Prefer `-n 4`. A timing failure that passes alone is load.

## Review Focus

1. **`successful + failed` no longer adds up to `total_evaluated`.** A client that derived one from the other now misses `unscored`. Expected: `total_evaluated` = `successful` + `failed` + `unscored`. Pinned by both route tests (3 = 1 + 1 + 1). No in-repo client derives it (the WebUI prints raw JSON).
2. **`failed` keeps its meaning** (the item's evaluation raised). Expected: a raised item is `failed`, not `unscored`; but a model or judge call that raises no longer raises out of the method (it is unscored), so `failed` now holds only errors outside the two calls, and a quota refusal, which the boundary precheck keeps from happening mid-run. Pinned by the route tests' raised item, the unchanged `test_run_benchmark_sanitizes_item_evaluation_failure_log` (`failed == 1`) and `test_run_benchmark_sanitizes_per_item_failure_log`, and, for the model call, `test_a_failed_model_answer_leaves_the_item_unscored[raises]` and `test_evaluate_custom_metric_lets_an_llm_quota_refusal_through[model-call]`.
3. **A run with nothing scored** shows `successful: 0` and 0.0 for the three scores, as an all-failed run does today. The guide says so. Not changed (see "What reads").
4. **A reply with an out-of-range or malformed score is unscored, not 0.5.** Pinned by `test_security.py` with exact values per input. Parser gaps that were 0.5 and are now unscored or misread, all pre-existing and noted on TASK-15487 (Task 2 Step 3), not changed here: a bare-number reply `"8"` is `None` (`json.loads` parses it as an int, so the regex branch never runs); `{"score": "8"}` is `None`; `Score: 1e308` reads its leading 1 and scores 0.0 (pinned as is).
5. **Accuracy in the CLI summary** is now correct/scored, not correct/total, and the saved file's `average_score` is over scored items. Pinned by the CLI test (`Accuracy: 100.0%` with one 0.8 and one unscored).
6. **Stored results** written before the fold hold 0.0 for failed calls and 0.5 for unreadable replies inside `scores`. Nothing reads them back; not rewritten (as TASK-15485 decided for the 0.5 placeholders).

---

## Files

| File | Task | Change |
|---|---|---|
| `tldw_Server_API/app/core/Evaluations/evaluation_manager.py` | 1 | `evaluate_custom_metric`: `None` on U1-U4; a raised model call caught (U1r); one comment line stating the `None` contract |
| `tldw_Server_API/app/api/v1/endpoints/benchmark_api.py` | 1 | `run_benchmark` summary: scored-only `successful_results`, `failed`, `unscored` |
| `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_benchmarks.py` | 1 | the same |
| `tldw_Server_API/app/core/Evaluations/cli/benchmark_cli.py` | 1 | `_save_results`: scored-only average, `unscored` |
| `tldw_Server_API/app/core/Evaluations/benchmark_utils.py` | 1 | `calculate_accuracy_by_category` skips `None`; `format_benchmark_summary` skips `None`, prints `Unscored:` |
| `tldw_Server_API/tests/Evaluations/unit/test_evaluation_manager.py` | 1 | `[no-score]`, `[judge-error]` → `None`; `test_handle_llm_failure` → `None`; `test_evaluate_custom_metric_lets_an_llm_quota_refusal_through` gains a `[model-call]` row; new `test_a_failed_model_answer_leaves_the_item_unscored` (`[returns-error]`, `[raises]`) |
| `tldw_Server_API/tests/Evaluations/test_security.py` | 1 | `test_malicious_score_injection`, `test_json_parsing_security`: exact value per input; the `float` type check kept for scored rows |
| `tldw_Server_API/tests/Evaluations/test_benchmark_api_error_mapping.py` | 1 | new `test_run_benchmark_counts_an_unscored_item_apart` |
| `tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_api.py` | 1 | new `test_benchmark_run_counts_an_unscored_item_apart`, inserted after `test_benchmark_run_endpoint_exists` (away from TASK-15426 B's hunks) |
| `tldw_Server_API/tests/Evaluations/unit/test_evals_cli_benchmark_commands.py` | 1 | new `test_benchmark_run_summary_leaves_an_unscored_item_out` |
| `Docs/User_Guides/Server/Benchmark_Creation_API_WebUI_Extension_Guide.md` + `Docs/Published` mirror | 1 | response example gains `unscored`; "Run failures in summary"; "Every item scored 0.5" last bullet |
| `Docs/superpowers/plans/2026-11-10-benchmark-items-with-no-score-are-unscored.md` | 1 | this plan |
| `backlog/tasks/task-15485 - ...md`, `task-15487 - ...md` | 2 | notes, TASK-15485's AC #2 and final summary |

Not edited: `EVALS_DEVELOPER_GUIDE.md` (lists only the signature, `-> Dict[str, Any]`; no score contract); `Docs/API-related/Evaluations_API_Unified_Reference.md` (its `successful`/`failed` example is the batch evaluation API, not benchmarks); the 2026-11-07 judge plan (a record).

## Test commands

- **New and changed tests** (24 cases):

```zsh
E=tldw_Server_API/tests/Evaluations
<pytest> -q --tb=line -rfE \
  "$E/unit/test_evaluation_manager.py::test_a_benchmark_item_is_scored_from_the_judges_reply" \
  "$E/unit/test_evaluation_manager.py::test_a_failed_model_answer_leaves_the_item_unscored" \
  "$E/unit/test_evaluation_manager.py::TestErrorHandling::test_handle_llm_failure" \
  "$E/unit/test_evaluation_manager.py::test_evaluate_custom_metric_lets_an_llm_quota_refusal_through" \
  "$E/test_security.py::TestScoreParsingSecurity" \
  "$E/test_evaluations_benchmarks_api.py::test_benchmark_run_counts_an_unscored_item_apart" \
  "$E/test_benchmark_api_error_mapping.py::test_run_benchmark_counts_an_unscored_item_apart" \
  "$E/unit/test_evals_cli_benchmark_commands.py::test_benchmark_run_summary_leaves_an_unscored_item_out"
```

- **Focused** (the five touched test files): `<pytest> -q -n 4 --tb=no -rfE $E/unit/test_evaluation_manager.py $E/test_security.py $E/test_evaluations_benchmarks_api.py $E/test_benchmark_api_error_mapping.py $E/unit/test_evals_cli_benchmark_commands.py`.
- **Wider** (every test file that names a touched symbol; 35 files, about 80 s):

```zsh
git grep -l -E 'evaluate_custom_metric|format_for_custom_metric|benchmark_utils|format_benchmark_summary|calculate_accuracy_by_category|benchmark_cli|evals_cli|benchmark_registry|EvaluationManager|evaluation_manager|benchmark_api|evaluations_benchmarks|/benchmarks' \
  -- 'tldw_Server_API/tests/*.py' | grep '/test_[^/]*\.py$' | sort -u > <scratchpad>/wider.txt
<pytest> -q -n 4 --tb=no -rfE $(cat <scratchpad>/wider.txt)
```

| Set | Base `5916bd586f` | Prototype |
|---|---|---|
| New and changed (24) | `10 failed, 14 passed` | `24 passed` |
| Focused (5 files) | `1 failed, 65 passed` | `1 failed, 71 passed` |
| Wider (35 files) | `1 failed, 706 passed` | `1 failed, 712 passed` |

The one failure in focused and wider is the probe artifact. Probe log: the same 2 lines on both trees (wider and focused); empty for the 24. The "base" column for the 24 is the fold's tests against the base's `tldw_Server_API/app`.

---

## Task 1: Unscored items, in the manager, both routes and the CLI (owner decisions 1-3)

**Files:** the twelve code, test and docs files above (the guide and its mirror are two), and the plan copy.

**Interfaces:** `evaluate_custom_metric(...) -> {"metric_name", "score": float | None, "explanation", "raw_output"}`. The route summary gains `"unscored": int`. The CLI's saved `summary` gains `"unscored": int`.

- [ ] **Step 1: Start.** On `fix/benchmark-judge-15485`, `git log --oneline -1` shows `f0acbf2713` (or the controller's newer head; `git diff --stat 5916bd586f HEAD -- tldw_Server_API Docs` must be empty, or re-measure). `git status --short` is empty. Copy this plan to `Docs/superpowers/plans/2026-11-10-benchmark-items-with-no-score-are-unscored.md` (Step 9's `git diff --cached --check` checks it for trailing whitespace).

- [ ] **Step 2: RED.** Make these test edits.

`tldw_Server_API/tests/Evaluations/unit/test_evaluation_manager.py`, in `TestErrorHandling.test_handle_llm_failure`, replace

```python
            # When LLM fails, it should return a result with score 0.0 and error message
            assert result["score"] == 0.0
```

with

```python
            # When LLM fails, the item is unscored (None, owner 2026-11-10), with the error message
            assert result["score"] is None
```

In the parametrize list of `test_a_benchmark_item_is_scored_from_the_judges_reply`, replace the `[no-score]` row and the `[judge-error]` row and its comment:

```python
        # no readable score: unscored (None), not the 0.5 default (owner, 2026-11-10)
        ("simple_bench", _MC_ITEM, "I cannot judge this.", _MC_ASKS, None),
```

```python
        # a failed judge call: unscored (None) with the error, not a 0.0 counted as a success
        ("simple_bench", _MC_ITEM, RuntimeError("429 rate limited"), _MC_ASKS, None),
```

Keep `assert result["score"] == pytest.approx(expected)`: pytest (9.1.1 here; since 6.1) compares a non-numeric `approx` by equality, so `None == approx(None)` holds and `0.0 == approx(None)` and `0.5 == approx(None)` do not (checked); the failure message reads `assert 0.5 == None`.

In `test_evaluate_custom_metric_lets_an_llm_quota_refusal_through`, add a model-call row (pins the new clause that re-raises a quota refusal from the model's call). Replace

```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_evaluate_custom_metric_lets_an_llm_quota_refusal_through(evaluation_manager):
    """A refusal is not scored 0.0 (TASK-13638)."""
```

with

```python
@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["a", "{model_response}"], ids=["judge-call", "model-call"])
async def test_evaluate_custom_metric_lets_an_llm_quota_refusal_through(evaluation_manager, answer):
    """A refusal is not scored 0.0 (TASK-13638), nor unscored, on the judge's call or the model's."""
```

and in its call, `input_data={"answer": "a"},` → `input_data={"answer": answer},`.

Append at the end of the file:

```python
@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("raises", [False, True], ids=["returns-error", "raises"])
async def test_a_failed_model_answer_leaves_the_item_unscored(evaluation_manager, monkeypatch, raises):
    """The model's own call failed, so there is no answer to judge: the item is unscored (None), not a 0.0
    counted as a success, and the judge is not called (owner, 2026-11-10; TASK-15485). analyze turns most
    adapter errors into "Error: ..."; a transport failure (a local server that is down) it lets through."""
    from types import SimpleNamespace

    from tldw_Server_API.app.core.Evaluations.benchmark_registry import get_registry
    from tldw_Server_API.app.core.exceptions import NetworkError
    from tldw_Server_API.app.core.LLM_Calls import Summarization_General_Lib as summarization

    prompts: list[str] = []

    def chat(request, timeout=None):
        prompts.append(request["messages"][-1]["content"])
        raise NetworkError("local server down") if raises else RuntimeError("503 model unavailable")

    monkeypatch.setattr(
        summarization, "get_registry", lambda: SimpleNamespace(get_adapter=lambda _p: SimpleNamespace(chat=chat))
    )
    data = get_registry().create_evaluator("simple_bench").format_for_custom_metric(_MC_ITEM)
    result = await evaluation_manager.evaluate_custom_metric(
        metric_name=data["name"],
        description=data["description"],
        evaluation_prompt=data["evaluation_prompt"],
        input_data=data["input_data"],
        scoring_criteria=data["scoring_criteria"],
        api_name="openai",
        api_key="sk-test",
    )
    assert len(prompts) == 1, prompts  # the model's call only
    assert result["score"] is None
    assert result["explanation"].startswith("Error:")
```

`tldw_Server_API/tests/Evaluations/test_security.py`, `TestScoreParsingSecurity`. The expected values were measured on the base (`analyze` patched): the six unparseable replies and the two non-numeric JSON scores give 0.5 today; `{"score": 8}` 0.778, `{"score": 7}` 0.667, `Score: 1e308` 0.0. In `test_malicious_score_injection`, keep the line `# Mock the analyze function to return malicious responses`. Replace everything from the line after it through the `for response in malicious_responses:` header, that is

```python
        malicious_responses = [
            "999999999999999999999999",  # Huge number
            "-999999",  # Negative number
            "Score: <script>alert('xss')</script>",  # XSS attempt
            '{"score": "\'; DROP TABLE evaluations; --"}',  # SQL injection in JSON
            "Score: NaN",  # Not a number
            "Score: Infinity",  # Infinity
            "Score: 1e308",  # Very large number
        ]

        for response in malicious_responses:
```

with

```python
        # No readable 1-10 score leaves the item unscored (None), never a value outside 0-1
        malicious_responses = [
            ("999999999999999999999999", None),  # Huge number
            ("-999999", None),  # Negative number
            ("Score: <script>alert('xss')</script>", None),  # XSS attempt
            ('{"score": "\'; DROP TABLE evaluations; --"}', None),  # SQL injection in JSON
            ("Score: NaN", None),  # Not a number
            ("Score: Infinity", None),  # Infinity
            ("Score: 1e308", 0.0),  # Very large number: the pattern reads its leading 1, the worst rating
        ]

        for response, expected in malicious_responses:
```

and replace the comment and the two asserts at the loop's end,

```python
                # Score should be safely bounded between 0 and 1
                assert 0 <= result["score"] <= 1
                assert isinstance(result["score"], float)
```

with

```python
                # Unscored (None), or a float in 0-1
                assert result["score"] == expected, response
                assert expected is None or isinstance(result["score"], float)
```

In `test_json_parsing_security`, replace the list and the loop header,

```python
        json_responses = [
            '{"score": 8, "__proto__": {"isAdmin": true}}',  # Prototype pollution attempt
            '{"score": 7, "constructor": {"prototype": {"isAdmin": true}}}',  # Constructor attack
            '{"score": [1,2,3,4,5,6,7,8,9,10]}',  # Array instead of number
            '{"score": {"value": 8}}',  # Nested object
        ]

        for response in json_responses:
```

with

```python
        json_responses = [
            ('{"score": 8, "__proto__": {"isAdmin": true}}', 7 / 9),  # Prototype pollution attempt
            ('{"score": 7, "constructor": {"prototype": {"isAdmin": true}}}', 6 / 9),  # Constructor attack
            ('{"score": [1,2,3,4,5,6,7,8,9,10]}', None),  # Array instead of number: unscored
            ('{"score": {"value": 8}}', None),  # Nested object: unscored
        ]

        for response, expected in json_responses:
```

and the comment and assert at the loop's end, `# Should handle gracefully` / `assert 0 <= result["score"] <= 1`, with

```python
                # Should handle gracefully: the 1-10 score, or unscored (None)
                assert result["score"] == pytest.approx(expected), response
```

`tldw_Server_API/tests/Evaluations/test_benchmark_api_error_mapping.py`: insert after `test_run_benchmark_sanitizes_item_evaluation_failure_log` (uses the file's `SimpleNamespace`, `AsyncMock`, `pytest`, `benchmark_module`):

```python
@pytest.mark.asyncio
async def test_run_benchmark_counts_an_unscored_item_apart(monkeypatch):
    """A scored, an unscored (score None) and a raised item: only the scored one is a success, and only it
    reaches the averages, by_category and the stored scores (owner, 2026-11-10; TASK-15485)."""
    replies = {
        "scored": {"score": 0.8, "explanation": "ok"},
        "unscored": {"score": None, "explanation": "Error: 429 rate limited"},
        "raised": RuntimeError("benchmark item exploded"),
    }

    async def _evaluate(**kwargs):
        reply = replies[kwargs["input_data"]["question"]]
        if isinstance(reply, Exception):
            raise reply
        return reply

    class _Registry:
        def get(self, name):
            return SimpleNamespace(evaluation_type="demo-eval")

        def create_evaluator(self, name):
            return SimpleNamespace(
                format_for_custom_metric=lambda item: {
                    "name": "demo-metric",
                    "description": "demo",
                    "evaluation_prompt": "{question}",
                    "input_data": {"question": item["question"]},
                    "scoring_criteria": {"accuracy": "high"},
                }
            )

    store = AsyncMock(return_value="eval-1")
    monkeypatch.setattr(benchmark_module, "get_registry", lambda: _Registry())
    monkeypatch.setattr(
        benchmark_module,
        "load_benchmark_dataset",
        lambda *args, **kwargs: [{"question": question, "category": "math"} for question in replies],
    )
    monkeypatch.setattr(benchmark_module.evaluation_manager, "evaluate_custom_metric", _evaluate)
    monkeypatch.setattr(benchmark_module.evaluation_manager, "store_evaluation", store)

    request = benchmark_module.BenchmarkRunRequest(limit=3, api_name="openai", parallel=3, save_results=True)
    response = await benchmark_module.run_benchmark("demo-benchmark", request, user_id="user-1")

    summary = response.results_summary
    assert [summary[k] for k in ("total_evaluated", "successful", "failed", "unscored")] == [3, 1, 1, 1]
    assert [summary[k] for k in ("average_score", "min_score", "max_score")] == [0.8, 0.8, 0.8]
    assert summary["by_category"] == {"math": {"count": 1, "average": 0.8}}
    assert store.await_args.kwargs["results"]["scores"] == [0.8]
```

`tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_api.py`: insert directly after `test_benchmark_run_endpoint_exists` (no new import, so TASK-15426 B's import hunk and its rewrite of the file's last test stay clear):

```python
def test_benchmark_run_counts_an_unscored_item_apart(monkeypatch):
    """A scored, an unscored (score None) and a raised item: only the scored one is a success, and only it
    reaches the averages, by_category and the stored scores (owner, 2026-11-10; TASK-15485)."""
    replies = {
        "scored": {"score": 0.8, "explanation": "ok"},
        "unscored": {"score": None, "explanation": "Error: 429 rate limited"},
        "raised": RuntimeError("benchmark item exploded"),
    }

    async def _evaluate(**kwargs):
        reply = replies[kwargs["input_data"]["question"]]
        if isinstance(reply, Exception):
            raise reply
        return reply

    class _Reg:
        def get(self, name):
            return type("Cfg", (), {"evaluation_type": "bullshit_detection"})()

        def create_evaluator(self, _name):
            class _Eval:
                def format_for_custom_metric(self, item):
                    return {
                        "name": "bullshit_detection",
                        "description": "d",
                        "evaluation_prompt": "{question}",
                        "input_data": {"question": item["question"]},
                        "scoring_criteria": {"incoherence_identification": "x"},
                    }

            return _Eval()

    class _Manager:
        evaluate_custom_metric = staticmethod(_evaluate)
        store_evaluation = AsyncMock(return_value="eval_1")

    monkeypatch.setattr(benchmarks_ep, "get_registry", lambda: _Reg())
    monkeypatch.setattr(
        benchmarks_ep,
        "load_benchmark_dataset",
        lambda *args, **kwargs: [{"question": question, "domain": "d"} for question in replies],
    )
    monkeypatch.setattr(benchmarks_ep, "_get_evaluation_manager_for_user", lambda _user: _Manager())

    app = _build_app()
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/evaluations/benchmarks/bullshit_benchmark/run",
            json={"limit": 3, "api_name": "openai", "parallel": 3, "save_results": True},
        )
    assert response.status_code == 200, response.text
    summary = response.json()["results_summary"]
    assert [summary[k] for k in ("total_evaluated", "successful", "failed", "unscored")] == [3, 1, 1, 1]
    assert [summary[k] for k in ("average_score", "min_score", "max_score")] == [0.8, 0.8, 0.8]
    assert summary["by_category"] == {"d": {"count": 1, "average": 0.8}}
    assert _Manager.store_evaluation.await_args.kwargs["results"]["scores"] == [0.8]
```

`tldw_Server_API/tests/Evaluations/unit/test_evals_cli_benchmark_commands.py`: insert after `test_benchmark_run_command_executes`:

```python
def test_benchmark_run_summary_leaves_an_unscored_item_out(monkeypatch, tmp_path):
    """An unscored item (score None) is counted apart and left out of the average and accuracy, on screen and in
    the saved results (owner, 2026-11-10; TASK-15485)."""
    import json

    class _Registry:
        def get(self, name):
            return type("Cfg", (), {"evaluation_type": "bullshit_detection"})()

        def create_evaluator(self, name):
            return type("Eval", (), {"format_for_custom_metric": lambda self, item: {}})()

    cli = "tldw_Server_API.app.core.Evaluations.cli.benchmark_cli"
    monkeypatch.setattr(f"{cli}.get_registry", lambda: _Registry())
    monkeypatch.setattr(f"{cli}.load_benchmark_dataset", lambda *a, **k: [{"id": "q1"}, {"id": "q2"}])
    # "category" sends the summary through calculate_accuracy_by_category too
    monkeypatch.setattr(
        f"{cli}.run_async_safely",
        lambda coro: coro.close() or [
            {"score": 0.8, "explanation": "ok", "category": "c"},
            {"score": None, "explanation": "Error: 429 rate limited", "category": "c"},
        ],
    )
    out = tmp_path / "results.json"

    result = CliRunner().invoke(main, ["benchmark", "run", "bullshit_benchmark", "--output", str(out)])

    assert result.exit_code == 0, result.output
    for line in ("Unscored: 1", "Accuracy: 100.0%", "Average Score: 0.800", "c: 100.0% (1/1)"):
        assert line in result.output
    assert json.loads(out.read_text())["summary"] == {"total": 2, "unscored": 1, "average_score": 0.8}
```

Run the new and changed tests (label `red`).

Expected (measured, the 24 against the base's `tldw_Server_API/app`): `10 failed, 14 passed`, empty probe log. Each for the right reason:
- `[no-score]`: `assert 0.5 == None`; `[judge-error]`: `assert 0.0 == None`;
- `test_a_failed_model_answer_leaves_the_item_unscored[returns-error]`, `test_handle_llm_failure`: `assert 0.0 is None`;
- `test_a_failed_model_answer_leaves_the_item_unscored[raises]`: `NetworkError: local server down` escapes the method;
- `test_malicious_score_injection`: `AssertionError: 999999999999999999999999` (0.5, not None); `test_json_parsing_security`: `AssertionError: {"score": [1,2,3,4,5,6,7,8,9,10]}`;
- `/evaluations/benchmarks` route: `AssertionError: {"detail":"Failed to run benchmark: ..."}` (500; `TypeError` from `float(None)`);
- `/benchmarks` route: `HTTPException: 500: Failed to run benchmark` (`TypeError: unsupported operand type(s) for +: 'float' and 'NoneType'` in `sum`);
- CLI: exit code 1, `TypeError("'>=' not supported between instances of 'NoneType' and 'float'")`.

The 14 that pass are the other 12 rows of the judge's-reply test and both rows of the quota test (the base has no `try` around the model call, so a refusal already escapes; the `[model-call]` row pins that the new `try` keeps it so).

- [ ] **Step 3: The manager.** In `tldw_Server_API/app/core/Evaluations/evaluation_manager.py`, `evaluate_custom_metric`:
  - U1 and U1r (`:560-572`): replace

```python
            model_response = await self._generate_model_response(
                input_data=runtime_input_data,
                api_name=api_name,
                api_key=api_key,
            )
            if isinstance(model_response, str) and model_response.startswith("Error:"):
                logger.error(f"Model response generation failed for metric '{metric_name}': {model_response}")
                return {
                    "metric_name": metric_name,
                    "score": 0.0,
```

  with

```python
            model_call = self._generate_model_response(
                input_data=runtime_input_data,
                api_name=api_name,
                api_key=api_key,
            )
            try:
                model_response = await model_call
            except LLMQuotaExceededError:
                raise  # the user's limit is not an unscored item
            except Exception as e:  # a model call that raised (provider error, timeout, rate limit) is unscored too
                model_response = f"Error: {e}"
            if isinstance(model_response, str) and model_response.startswith("Error:"):
                logger.error(f"Model response generation failed for metric '{metric_name}': {model_response}")
                # score None is "no score": the callers count the item unscored, in no average (owner, 2026-11-10)
                return {
                    "metric_name": metric_name,
                    "score": None,
```

  The call is bound first and awaited inside the `try`, so the four lines of the call itself, which B's first hunk edits, keep their indentation. The comment line is the method's `None` contract (the docstring stays as is, Interplay).
  - U2 (`:600-604`): replace

```python
            # A failed judge call (provider error, timeout, rate limit, truncated or empty reply)
            # is a 0.0 with the error, as for the model's own answer above; not a silent 0.5.
            if isinstance(response, str) and response.startswith("Error:"):
                logger.error(f"Judge call failed for metric '{metric_name}': {response}")
                return {"metric_name": metric_name, "score": 0.0, "explanation": response, "raw_output": None}
```

  with

```python
            # A failed judge call (provider error, timeout, rate limit, truncated or empty reply)
            # is unscored with the error, as for the model's own answer above; not a silent 0.5.
            if isinstance(response, str) and response.startswith("Error:"):
                logger.error(f"Judge call failed for metric '{metric_name}': {response}")
                return {"metric_name": metric_name, "score": None, "explanation": response, "raw_output": None}
```

  - U4 (`:663-666`): replace

```python
            # Default to 0.5 if no valid score found
            if score is None:
                logger.warning(f"Could not parse valid score from response: {response[:100]}...")
                score = 0.5
```

  with

```python
            # No valid score found: the item is unscored (None), not a 0.5 placeholder
            if score is None:
                logger.warning(f"Could not parse valid score from response: {response[:100]}...")
```

  - U3 (`:681`), in the closing `except Exception` return: `"score": 0.0,` → `"score": None,`.
  - Leave the docstring and the signature alone: TASK-15426 B rewrites both (Interplay).

- [ ] **Step 4: Both routes.** In `tldw_Server_API/app/api/v1/endpoints/benchmark_api.py` (`:305-312`), replace

```python
        # Calculate summary statistics
        successful_results = [r for r in results if "error" not in r]
        scores = [r["score"] for r in successful_results]

        summary = {
            "total_evaluated": len(results),
            "successful": len(successful_results),
            "failed": len(results) - len(successful_results),
```

with

```python
        # Calculate summary statistics. An item with no score (its model or judge call failed, or the reply held
        # none) is neither successful nor failed: it is unscored, outside every average (owner, 2026-11-10).
        successful_results = [r for r in results if "error" not in r and r["score"] is not None]
        failed = sum("error" in r for r in results)
        scores = [r["score"] for r in successful_results]

        summary = {
            "total_evaluated": len(results),
            "successful": len(successful_results),
            "failed": failed,
            "unscored": len(results) - len(successful_results) - failed,
```

In `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_benchmarks.py` (`:182-188`), replace

```python
        successful_results = [r for r in results if "error" not in r]
        scores = [float(r.get("score", 0.0)) for r in successful_results]

        summary: dict[str, Any] = {
            "total_evaluated": len(results),
            "successful": len(successful_results),
            "failed": len(results) - len(successful_results),
```

with

```python
        # An item with no score (its model or judge call failed, or the reply held none) is neither successful
        # nor failed: it is unscored, outside every average (owner, 2026-11-10).
        successful_results = [r for r in results if "error" not in r and r["score"] is not None]
        failed = sum("error" in r for r in results)
        scores = [float(r.get("score", 0.0)) for r in successful_results]

        summary: dict[str, Any] = {
            "total_evaluated": len(results),
            "successful": len(successful_results),
            "failed": failed,
            "unscored": len(results) - len(successful_results) - failed,
```

`by_category` and the stored `scores` already iterate `successful_results`, so they need no edit. `r["score"]` always exists: both routes build each non-error row with `result.get("score", 0.0)`.

- [ ] **Step 5: The CLI helpers.** In `tldw_Server_API/app/core/Evaluations/cli/benchmark_cli.py`, `_save_results`, replace

```python
    """Save evaluation results to file."""
    output_data = {
        "benchmark": benchmark_name,
        "num_samples": len(results),
        "results": results,
        "summary": {
            "total": len(results),
            "average_score": sum(r.get('score', 0) for r in results) / len(results) if results else 0
        }
    }
```

with

```python
    """Save evaluation results to file."""
    scores = [r["score"] for r in results if r.get("score") is not None]  # an unscored item is in no average
    output_data = {
        "benchmark": benchmark_name,
        "num_samples": len(results),
        "results": results,
        "summary": {
            "total": len(results),
            "unscored": len(results) - len(scores),
            "average_score": sum(scores) / len(scores) if scores else 0
        }
    }
```

In `tldw_Server_API/app/core/Evaluations/benchmark_utils.py`, `calculate_accuracy_by_category` (`:1063`), after `score = result.get("score", 0.0)` add:

```python
        if score is None:
            continue  # unscored: in no accuracy or average
```

`format_benchmark_summary` (`:1116-1129`): after `scores = [r.get("score", 0.0) for r in results]` add

```python
    scores = [s for s in scores if s is not None]  # an unscored item is in no accuracy or average
```

replace

```python
    accuracy = correct / total
    avg_score = sum(scores) / len(scores)
```

with

```python
    accuracy = correct / len(scores) if scores else 0.0
    avg_score = sum(scores) / len(scores) if scores else 0.0
```

and after the `  Total Questions: {total}` line of the f-string add `  Unscored: {total - len(scores)}`.

- [ ] **Step 6: GREEN.** Run the new and changed tests (label `green`). Expected (measured): `24 passed`, empty probe log.
  - Ablations (measured): with `calculate_accuracy_by_category`'s guard removed, the CLI test fails (exit code 1); with the model call's `except LLMQuotaExceededError: raise` removed, `[model-call]` fails (`DID NOT RAISE LLMQuotaExceededError`). Each other edit is the only fix for a RED listed in Step 2.

- [ ] **Step 7: Focused set** (label `focused`). Expected (measured): `1 failed, 71 passed` (base `1 failed, 65 passed`; +6 new cases). The failure is the probe artifact; the log holds the 2 base lines.

- [ ] **Step 8: Docs.** In `Docs/User_Guides/Server/Benchmark_Creation_API_WebUI_Extension_Guide.md`:
  - In "Typical response shape", after `"failed": 0,` add `"unscored": 0,` (same indent).
  - In "Run failures in summary", replace the first bullet with:

```markdown
- `failed > 0` counts items whose evaluation raised outside the model and judge calls, such as an item whose data could not be built into a prompt. A failed model or judge call is never `failed`: it is `unscored`.
- `unscored > 0` counts items with no score: the model's own answer call failed or raised (provider error, timeout, rate limit, or a local server that is down), the judge call failed or raised (the same, or an empty or truncated reply), or the judge's reply held no readable score. They are neither `successful` nor `failed`, and they are left out of `average_score`, `min_score`, `max_score`, `by_category` and the stored scores. The server logs `Model response generation failed`, `Judge call failed for metric`, `Custom metric evaluation failed` or `Could not parse valid score`.
- If every item is unscored or failed, `successful` is 0 and the three scores read 0.0; they are not a measured zero.
```

  - In "Every item scored 0.5", replace the last bullet with:

```markdown
- Each item's score is the judge's reply mapped to 0-1: a 1-10 score (10 is 1.0, 1 is 0.0). A reply with no score, a failed judge call or a failed model answer leaves the item unscored (see "Run failures in summary"). The run's response does not show per-item explanations, so look in the server log; the CLI's saved results keep a `null` score and the explanation (the `Error:` text for a failed call). Re-running such an item may succeed.
```

  - `bash Helper_Scripts/refresh_docs_published.sh`. Expected (measured): only `Docs/Published/User_Guides/Server/Benchmark_Creation_API_WebUI_Extension_Guide.md` changes.

- [ ] **Step 9: Commit.** `git add` these twelve files and the plan copy by name; `git diff --cached --check` prints nothing; `git diff --cached --stat` lists 13 files:
  - `tldw_Server_API/app/core/Evaluations/evaluation_manager.py`
  - `tldw_Server_API/app/api/v1/endpoints/benchmark_api.py`
  - `tldw_Server_API/app/api/v1/endpoints/evaluations/evaluations_benchmarks.py`
  - `tldw_Server_API/app/core/Evaluations/cli/benchmark_cli.py`
  - `tldw_Server_API/app/core/Evaluations/benchmark_utils.py`
  - `tldw_Server_API/tests/Evaluations/unit/test_evaluation_manager.py`
  - `tldw_Server_API/tests/Evaluations/test_security.py`
  - `tldw_Server_API/tests/Evaluations/test_benchmark_api_error_mapping.py`
  - `tldw_Server_API/tests/Evaluations/test_evaluations_benchmarks_api.py`
  - `tldw_Server_API/tests/Evaluations/unit/test_evals_cli_benchmark_commands.py`
  - `Docs/User_Guides/Server/Benchmark_Creation_API_WebUI_Extension_Guide.md`
  - `Docs/Published/User_Guides/Server/Benchmark_Creation_API_WebUI_Extension_Guide.md` (without it `tests/Docs` fails)
  - `Docs/superpowers/plans/2026-11-10-benchmark-items-with-no-score-are-unscored.md`

  Message:

```text
fix(evaluations): a benchmark item with no score is unscored, not successful (TASK-15485)

Owner, 2026-11-10, on #3445's Review Focus 2: a failed judge call is neither failed nor a success. The
failed model answer and a reply with no readable score are unscored too.

- evaluate_custom_metric returns score None on all four no-score paths (was 0.0, 0.0, 0.0 and 0.5),
  and a model call that raises is caught as the judge's is (a quota refusal still re-raises).
- Both benchmark routes count them as unscored; successful, the averages, by_category and the stored
  scores hold scored items only; failed still counts items whose evaluation raised.
- The benchmark CLI shows an Unscored count and leaves them out of accuracy and averages.
- The benchmark guide (and its Docs/Published mirror) says so.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
```

## Task 2: Verify, close out, review, push

- [ ] **Step 1: Wider set** (label `final`). Expected (measured on the prototype): `1 failed, 712 passed` (base `1 failed, 706 passed`), the probe artifact, the same 2 log lines.

- [ ] **Step 2: CI checks.** Each expected result was measured on the prototype (without the plan copy and backlog edits):

| Command | Expected |
|---|---|
| `<pytest> -q -n 4 --tb=no -rfE tldw_Server_API/tests/Docs tldw_Server_API/tests/lint tldw_Server_API/tests/CI tools/backlog-py/tests` | `2 failed, 1750 passed, 6 skipped, 1 error` (about 2.5 minutes), empty probe log. The failures and the error are the known environment-only ones: `test_jobs_postgres_exports_recorded_coverage[same-checkout]` and `[foreign-checkout]`, and `tests/lint/test_runtime_package_discovery.py` (ImportError). The function-size and scope-predicate ratchets pass. With the plan copy and task files the passed count may move by the Docs tests that scan them; any new failure is real. |
| `PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/check_shard_coverage.py` | `new_uncovered=0`, "OK — no newly-unshared test files." |
| `PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/test_quality_triage.py --enforce` | exit 0, "OK — no offenses beyond the baseline." |
| `PATH="$(dirname $PY):$PATH" PYTHONPATH=. $PY Helper_Scripts/ci/route_auth_ratchet.py` | exit 0, "208 unauthenticated routes; baseline exact." (Once, run right after the pytest set under load, it printed a `libc++abi ... recursive_mutex lock failed` abort at interpreter exit; alone it gave exit 0 and the line above. This PR mounts no route.) |
| `PATH="$(dirname $PY):$PATH" PYTHONPATH=. $PY Helper_Scripts/export_openapi_schema.py --check apps/packages/api-types/openapi.fingerprint.json --check-paths apps/packages/api-types/paths.d.ts` | "OK — OpenAPI fingerprint matches the checked-in snapshot." and "OK — committed path union matches the canonical contract." (`results_summary` is `dict[str, Any]`, so the new key changes no schema.) |
| `/opt/homebrew/bin/uvx bandit -q -ll` on the five touched source files | One finding, pre-existing: B608 Medium at `evaluation_manager.py:873` on the base, `:879` with the fold. Nothing new. |
| `$PY -m ruff check` on the ten touched Python files | 28 findings on the base and 28 with the fold (all pre-existing: 13 in `test_evaluation_manager.py`, 14 in `test_security.py`, 1 I001 in `evaluations_benchmarks.py`); none on a new line. |
| `/usr/bin/python3 Helper_Scripts/ci/check_backlog_task_ids.py .` | exit 0 |
| `grep -c -E "evaluation_manager\|benchmark_api\|evaluations_benchmarks\|benchmark_cli\|benchmark_utils\|Benchmark_Creation" Helper_Scripts/ci/scope_predicate_baseline.txt` | `0` |
| `git diff origin/fix/benchmark-judge-15485 --check` | nothing |

- [ ] **Step 3: Backlog.** TASK-15485's AC #2 and final summary state the old values ("a reply with neither still scores 0.5; a failed judge call ... scores 0.0"), so both are replaced; TASK-15487's note says "SCORE: 70% falls to 0.5", which is now unscored, and the parser gaps the plan review found go on TASK-15487 too. Fill `<...>` with the measured counts first:

```bash
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-15485 \
  --remove-ac 2 \
  --ac "The item's score is read from the judge's reply in the format its formatter asks for (multiple choice on 1-10: 10 gives 1.0, 1 gives 0.0, also when the judge restates the rubric or writes markdown labels; a SimpleQA grade: correct 1.0, otherwise 0.0), with tests; a reply with neither, a failed or raised judge call (provider error, timeout, rate limit, truncated or empty reply) or a failed or raised model answer call leaves the item unscored (score None, with the error text for a failed call): neither successful nor failed in the route summaries, and in no average; a quota refusal still raises (owner, 2026-11-10)" \
  --check-ac 3 \
  --append-notes "Owner decision (2026-11-10), folded into PR #3445 (it answers its Review Focus 2): an item with no score is neither failed nor successful but unscored, left out of every average. evaluate_custom_metric returns score None for a failed model answer (was 0.0), a failed judge call (G1, was 0.0), a raised judge call (the closing except, was 0.0) and a reply with no readable score (was 0.5). A model call that raises (analyze lets NetworkError and EgressPolicyError through; it escaped the method, so the routes counted the item failed and the CLI run stopped) is now caught the same way, on the controller's ruling of 2026-11-10 on the plan review's Minor 2; LLMQuotaExceededError still propagates from both calls. Both routes' summaries gain unscored: successful counts scored items, failed still counts items whose evaluation raised (now only errors outside the two calls), total_evaluated all three; average_score, min_score, max_score, by_category and the stored scores use scored items only, and a run with none scored keeps the 0.0 values with successful 0. The benchmark CLI shows Unscored: N and leaves those items out of accuracy and averages, on screen and in its saved file. AC #2 was reworded to match and is now AC #3. Plan: Docs/superpowers/plans/2026-11-10-benchmark-items-with-no-score-are-unscored.md." \
  --append-notes "Fold RED/GREEN: the 24 new and changed tests <10 failed, 14 passed> against 5916bd586f's app code (each a 0.0 or 0.5 where None is due, a NetworkError escaping from the model call, or a TypeError over a None score in the routes and the CLI) -> <24 passed>. Touched test files <1 failed, 65 passed> -> <1 failed, 71 passed>; wider set (35 files, under the socket probe) <1 failed, 706 passed> -> <1 failed, 712 passed>, the one failure the probe artifact test_ssrf_protection_private_ip, the same two probe lines. tests/Docs + lint + CI + backlog-py <n>; shard coverage, test-quality triage, route-auth ratchet and OpenAPI unchanged; Bandit one pre-existing B608." \
  --final-summary "The benchmark judge now reaches the provider: evaluate_custom_metric passes its JSON input to analyze as literal text, so a formatter's question/choices/response keys no longer empty it. Its reply is read as each formatter asks: the score pattern ignores case ('SCORE:'), the multiple-choice formatter asks for the manager's 1-10 scale (it asked 1.0/0.0, and 1 is the worst rating on 1-10), and a SimpleQA grade decides (correct 1.0, else 0.0). One parametrized test (14 cases) drives simple_bench, mmlu_pro, simpleqa_verified and bullshit_benchmark through their own formatters, the real manager and analyze to a stubbed adapter. Results stored before the fix are 0.5 placeholders; the benchmark guide says so. A judge that restates the rubric or writes markdown labels is read right. An item with no score (a failed or raised model answer or judge call, or a reply with no readable score) is unscored: neither successful nor failed, in no average, and counted apart in the route and CLI summaries (owner, 2026-11-10). Follow-ups filed: TASK-15486 (tests that reach real hosts) and TASK-15487 (benchmarks on correctly shaped data)."
PYTHONPATH=tools/backlog-py/src /Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python -m backlog_py --cwd . task edit TASK-15487 \
  --append-notes "Since TASK-15485's 2026-11-10 fold, a judge reply with no readable score leaves the item unscored (score None, counted apart from successful and failed, in no average), not 0.5. So the require_reasoning rubric's 'SCORE: 70%' leaves the item unscored rather than scoring 0.5. Parser gaps in evaluate_custom_metric that a human would read as a score, found by the fold's plan review (2026-11-10), in scope when the formatters move to 1-10 (AC #4): a bare-number reply '8' is unscored (json.loads parses it as an int, which is not a dict, so the regex branch never runs and its '^(\\d+)\\s*$' pattern cannot apply); {\"score\": \"8\"} (a numeric string) is unscored; 'Score: 1e308' reads its leading 1 and scores 0.0 (test_security pins that value as is)."
```

  - Expected: TASK-15485 still `Done`; its ACs read #1 (checked), #2 (the old #3, checked), #3 (the new text, checked); the two notes appended; the final summary replaced; TASK-15487 gains one note; `check_backlog_task_ids.py` exit 0. Measured with these flags in the planner's worktree, then reverted.
  - `git grep -n -E "<(10 failed|24 passed|1 failed|n)" -- 'backlog/tasks/task-15485 *'` prints nothing.
  - `git grep -c -F '^(\d+)\s*$' -- 'backlog/tasks/task-15487 *'` prints the file with `:1` (the shell kept the pattern's backslashes; measured with this exact command under bash, then reverted).
  - `git add -A backlog/tasks && git diff --cached --check`; `git diff --cached --stat` names only the two task files. Commit: `chore(backlog): TASK-15485 records the owner's unscored decision; TASK-15487 note` with the trailer.

- [ ] **Step 4: Review before the push.** Run `/superpowers:requesting-code-review` in a clean subagent on `f0acbf2713..HEAD` (the fold). Give it this plan's "What produces no score today", "Callers and what reads the score" and Review Focus. Address every finding, re-run the new and changed tests, and record each ruling in the ledger (`.superpowers/sdd/2026-11-07-benchmark-judge-reaches-the-provider/progress.md`).

- [ ] **Step 5: Push and the PR body.**
  - `git push` (fast-forward; never force).
  - Edit `<plans dir>/body_judge.md` with the section replacements below, then `gh pr edit 3445 --body-file <plans dir>/body_judge.md`. The body edit re-triggers the license status: wait for it.
  - #3445 merges after #3444 (owner order). When both have settled: `gh variable get MERGE_QUEUE` (unset when this plan was written), then `QUEUE="3445:fix/benchmark-judge-15485" sh <plans dir>/pr_queue3.sh` in the background. dev requires 8 statuses. Confirm MERGED before deleting the branch; then tell the TASK-15426 B executor to rebase.

### PR body (`body_judge.md`): sections that change

- **What changes.** In "Three guards", G1 becomes: "**G1:** a failed judge call (provider error, timeout, rate limit, empty or truncated reply) is unscored with the error (below), not a silent 0.5." After "Final-review fix", add:

```markdown
- **Unscored items (owner, 2026-11-10, answering Review Focus 2):** an item with no score is neither `successful` nor `failed`.
  - `evaluate_custom_metric` returns `score: None` when the model's own answer failed, the judge call failed (G1), the judge call raised, or the reply held no readable score (it returned 0.0, 0.0, 0.0 and 0.5). A model call that raises (a local server that is down) is now caught the same way; it used to escape, so the routes counted it `failed` and the CLI run stopped. A quota refusal still raises.
  - Both routes count these as `unscored`, a new summary key. `successful` counts scored items; `failed` still counts items whose evaluation raised, now only errors outside the two calls; `total_evaluated` is all three. `average_score`, `min_score`, `max_score`, `by_category` and the stored `scores` use scored items only.
  - The benchmark CLI prints `Unscored: N` and leaves those items out of accuracy and averages, on screen and in its saved file.
```

- **Docs.** Replace the bullet "It says where a failed judge call shows: ..." with: "It explains the new `unscored` count: which items it holds (a failed or raised model answer or judge call, a reply with no score), that they are in no average, and the server log line for each; and what can still be `failed`. The response example shows `unscored`. A run with nothing scored shows 0.0 scores with `successful: 0`; the guide says those are not a measured zero."
- **Measured.** Add a bullet:

```markdown
- **The unscored fold** (base `5916bd586f`):
  - the 24 new and changed tests, against the base's app code: `10 failed, 14 passed` (a 0.0 or 0.5 where `None` is due; a `NetworkError` escaping the model call; a `TypeError` over a `None` score in both routes and the CLI) → `24 passed`;
  - the five touched test files: `1 failed, 65 passed` → `1 failed, 71 passed`;
  - wider set (35 files): `1 failed, 706 passed` → `1 failed, 712 passed`, the same probe artifact and the same 2 probe lines.
```

- **Review Focus 2** becomes: "**An item with no score is `unscored`** (owner, 2026-11-10: a failed judge call is 'neither failed nor a success'; the failed model answer, raised or not, and the unreadable reply too). `failed` keeps its meaning (the evaluation raised), which now leaves only errors outside the model and judge calls. `successful + failed` no longer equals `total_evaluated`; `unscored` is the rest. A run with nothing scored shows `successful: 0` and 0.0 scores, as an all-failed run does."
- **Review Focus 4** becomes: "**Every caller of `evaluate_custom_metric` is a benchmark path, and each handles `score: None`:** both routes and the benchmark CLI. `cli/commands/evaluation.py` also names it, with arguments it does not take, so it raises before any score (pre-existing, unchanged)."
- **Reviews.** Add: "**Unscored fold:** plan independently reviewed and dry-run: ready with fixes, 0 Blocker, 0 Important, 7 Minor, all folded and re-measured (a model call that raises is unscored too, on the controller's ruling); clean-session review <result>."
- Unchanged: Why, Root cause, Filed here, Merge order, Change summary (the 2026-11-07 waiver is for this PR; the fold implements the owner's own answer).

---

## Interplay with TASK-15426 PR B

`proto_15426_b.patch` shares five files with this fold. Measured by `git apply --3way` of B's hunks for those five files, onto `5916bd586f` and onto the folded prototype commit (after the review fold): **the same result on both.** Four files apply cleanly; `evaluation_manager.py` has the one known conflict at the judge's `analyze` call (`input_is_literal_text=True` against `provider_credentials=provider_credentials`; keep both). **B still applies; the fold adds no conflict.** B's hunks, by file:
- `evaluation_manager.py`: B adds `provider_credentials` to the signature, rewrites the docstring, and passes the handle to `_generate_model_response` and to the judge's `analyze` call. The fold leaves the signature and the docstring alone. U1r changes only the call's first line (`model_response = await ...` → `model_call = ...`) and adds the `try` after its closing `)`, so the call's argument lines keep their indentation: B's `provider_credentials=provider_credentials,` merges into `model_call = self._generate_model_response(...)` cleanly (checked in the merged file). U2 sits below the judge call that is already B's one known conflict.
- `benchmark_api.py`, `evaluations_benchmarks.py`: B edits the imports, the quota line and the `evaluate_custom_metric(...)` call; the fold edits the summary block 20 lines lower.
- `test_evaluations_benchmarks_api.py`: B adds two imports and rewrites the file's last test; the fold's test sits after `test_benchmark_run_endpoint_exists`, with no new import.
- `test_evaluation_manager.py`: B adds a test inside `TestCustomMetrics` (`:327`); the fold edits `TestErrorHandling`, the quota test below it and the file's end.
- B's own counts change by the fold's +6 cases in the files it re-measures; its behaviour does not. With B, a raised model call on the user's own key is unscored the same way.

## Conflicts with open PRs

`gh pr list --base dev --state open` (42 PRs), each PR's file list matched against the fold's files: only #3445 itself, and #3269, whose `prompt_studio/evaluation_manager.py` is a different file. No conflict.

## Not measured

- PostgreSQL cases: none of the touched tests uses PostgreSQL (`TLDW_TEST_NO_DOCKER=1`).
- A real provider: every judge and model call is a stubbed adapter or a stubbed `evaluate_custom_metric`.
- The WebUI was not run; it prints the response as raw JSON (by code).
- `tldw_Server_API/cli/commands/evaluation.py`'s custom evaluation raising `TypeError` is by code (it passes `text=`, `custom_prompt=`, `save_results=`), not run.

## Decisions left for the owner

- None on the code. The representation (`None`), the all-unscored values (0.0 with `successful: 0`, as all-failed today) and the CLI's accuracy over scored items follow the owner's rule "left out of every average"; the raised model call follows the controller's ruling on Minor 2; the TASK-15485 AC #2 rewording records the owner's decision.
- ADR-004 (the plan review's recommendation): the plan leaves #3445's Change summary as the 2026-11-07 waiver, since the fold is the owner's own answer. The standing rule is to ask every time; recommend a one-line confirmation from the owner before the merge.

## Review fold (2026-11-10)

`review_plan_unscored.md`: ready with fixes, 0 Blocker, 0 Important, 7 Minor. All seven folded into this plan and `proto_unscored.patch` (prefold copies kept); every changed count re-measured.

| # | Finding | Fold |
|---|---|---|
| M1 | The guide's log lines missed `Custom metric evaluation failed` (U3) | Step 8: added to the `unscored` bullet |
| M2 | A model call that raises counted `failed`, the judge's `unscored` | Controller ruling (reviewer's docs-only option rejected): Step 3 catches it (U1r; `LLMQuotaExceededError` re-raises); Step 2 adds `[raises]` (RED: `NetworkError` escapes) and the quota test's `[model-call]` row (ablation-pinned); Step 8's `failed` bullet says what can still land there |
| M3 | Parser gaps now unscored or misread (`"8"`, `{"score": "8"}`, `1e308`) | Task 2 Step 3: added to the TASK-15487 note; Review Focus 4; no code change |
| M4 | `test_security.py` edit ambiguous about two comments; the `float` check dropped | Step 2 shows exact before and after for both tests; `assert expected is None or isinstance(result["score"], float)` kept |
| M5 | Step 9 said "eleven files" | Step 9 lists all twelve plus the plan copy (13 staged) |
| M6 | Step 1's `git diff --check` on an untracked file checks nothing | dropped; Step 9's `git diff --cached --check` is the check |
| M7 | The `None` contract was only in the guide | Step 3: one comment line at U1's return; the docstring stays (B rewrites it) |
