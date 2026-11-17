# Evaluation runner and CLI fixes: TASK-15489, TASK-15491, TASK-15492 (plan evals_15489)

> **For agentic workers:** use superpowers:executing-plans or superpowers:subagent-driven-development. Steps use checkbox syntax.

**Goal:** One PR, three commit sets, one per backlog task. (1) `eval custom` and a batch `type: custom` spec call `evaluate_custom_metric` with its real parameters and print an item with no score as unscored. (2) A label-choice or NLI run whose judge call fails records a sample error, and every run's aggregate leaves a sample with no score out of `mean_score` and `pass_rate`. (3) An evaluation's `evaluator_model` is a provider only when it names one; a model name is a model, `provider:model` and `provider/model` give both, and the run route names the provider the runner uses.

**Architecture:** Three small, independent changes.
- TASK-15492 changes `EvaluationRunner._resolve_eval_provider_model_api_key` (`eval_runner.py`), which `run_llm_provider` (the run route's quota provider, #3460) calls. Fixing the resolver fixes the route with no change to the route (AC #3 holds by construction, and a test pins it). Two module helpers: `_is_known_provider` (membership in `ChatProviderRegistry.DEFAULT_ADAPTERS` after `canonical_provider_name`) and `_split_provider_prefix`. A `prefix:` or `prefix/` is split off `evaluator_model` or the model string only when no provider was named explicitly through `api_name` or `provider` (spec or run config); the `"openai"` default does not count as named (review I1: `anthropic/claude-3.5-sonnet` with `api_name: openrouter` is an OpenRouter model id).
- TASK-15491 deletes the `try/except ... resp = ""` around `_call_adapter_text` in `_eval_label_choice` and `_eval_nli_factcheck`, so the call's exception reaches the function's existing outer `except`, which returns `{"sample_id", "error"}` (the same shape `_process_batch` already gives a vendor 401/429). `_calculate_aggregate_results` skips a sample with no `avg_score` and divides `pass_rate` by the scored count.
- TASK-15489 rewrites `_run_custom_evaluation` (CLI) to call `evaluate_custom_metric(metric_name, description, evaluation_prompt, input_data, scoring_criteria, api_name, api_key)`, builds `metrics` from the returned `score` and `explanation`, drops the `--save/--no-save` option from `eval custom`, and teaches `print_evaluation_results` and `_display_batch_summary` the "unscored" state.

**Tech Stack:** Python 3.12, FastAPI backend, click CLI (`tldw_Server_API/cli`), pytest with `CliRunner`, backlog-py.

**Spec:** `backlog/tasks/task-15489 - The-eval-custom-CLI-command-calls-evaluate-custom-metric-with-arguments-it-does-not-take.md`, `backlog/tasks/task-15491 - Label-choice-and-NLI-evaluation-runs-score-a-failed-judge-call-0.0.md`, `backlog/tasks/task-15492 - An-evaluation-s-evaluator-model-set-to-a-model-name-is-used-as-the-provider-name.md`.

**Fold (independent review, 2026-11-16):** the review's I1 (a named provider stops the prefix split), I2 (sibling CLI runners confirmed broken: follow-up task, not this PR) and M1-M5 are folded in below. Tests are inline in the steps, verbatim from the prototype, so the plan can be dry-run alone.

**Base:** `origin/dev` at `3883a73712` (fetched 2026-11-16). Prototype: `plans/proto_evals_15489.patch` (apply with `git apply`; it holds source, tests and the doc edit, and the refreshed `Docs/Published` mirror).

**Why one PR:** the three share one area (the evaluation runner and its CLI), none depends on another's code, and the shared file `eval_runner.py` is touched in disjoint functions (resolver `:200-255`; `_eval_label_choice` `:1955`; `_eval_nli_factcheck` `:2148`; `_calculate_aggregate_results` `:2331`). Each task has its own commit set and closes on its own evidence. Splitting would only add two serial merges. If review wants a split, 15489 (CLI only, no shared file) is the one to peel off.

**Shared files with the TASK-15487 planner (benchmarks):** this plan touches NO file of `evaluation_manager.py`, `benchmark_utils.py` or `benchmark_cli.py`. It calls `EvaluationManager.evaluate_custom_metric` from the CLI but does not edit it. Files this plan edits that another plan might: `tldw_Server_API/app/core/Evaluations/eval_runner.py` and `tldw_Server_API/tests/Evaluations/unit/test_eval_runner.py` (see "Open PRs" in the Decisions section: only stale #3269 also touches them).

## Global Constraints

- Tests make no real outbound network requests. Every pytest run uses the network probe plugin: `PYTHONPATH=<plans dir>:<plans dir>/pyext NETPROBE_LOG=<file> python -m pytest -p netprobe_15427 ...`. It blocks and logs every outbound DNS lookup and connection; a run must end with 0 log lines for the tests the plan adds or changes. (`pyext` holds the real `pytest-mock`, which the shared venv lacks and the root `conftest.py` loads.)
- A test that drives `/chat/completions`, an LLM adapter or a judge must stub the adapter or the provider call. A commercial provider otherwise calls the real API.
- A failed judge or model call, or a reply with no readable score, is unscored: `score: None`, a sample error, and out of every average. Never 0.0, never a success. This is the owner's rule, recorded 2026-11-10 and 2026-11-12.
- Backlog edits use backlog-py only (`PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task edit ...`). Use `--append-notes`, never `--notes`. Stage with `git add -A backlog/tasks`. Never hand-edit task files. A new follow-up task takes its id under the session's id lock (the controller does that; write `TASK-<NEW>`).
- Branch from `origin/dev`; the PR targets `dev`.
  - Commit trailer: `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
  - The PR body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- Never `--no-verify`, never force-push, never `git stash`.
- PostgreSQL only through the repo fixtures, with a session-unique `TEST_DB_NAME`. Never restart the server. Redis: never FLUSHALL or SCRIPT FLUSH.
- A mirrored doc (ADRs, the BYOK guide, `Env_Vars.md` and the other files under `Docs/Published`) needs `bash Helper_Scripts/refresh_docs_published.sh`, with the `Docs/Published` diff in the same PR.
- Shifting lines in a file listed in `Helper_Scripts/ci/scope_predicate_baseline.txt` fails backend-required. Regenerate it with the script's `--write-baseline`, in the same PR. (None of the six files this plan edits is listed; measured.)
- Size-capped functions (the function-size ratchet in `tests/lint`) must not grow past their caps. Measure each touched function before and after (Task 4 gives the measured numbers; no touched function is frozen).
- A plan with embedded diffs fails trailing-whitespace when it is committed: keep the plan free of trailing spaces.
- Plan-specific:
  - The shared venv lacks `pytest-mock`; always add `<plans dir>/pyext` to `PYTHONPATH`.
  - New tests in the label-choice and NLI cases stub `eval_runner.get_adapter_or_raise` (or `_call_adapter_text`); the CLI tests stub `evaluation_manager.analyze`. No test calls a provider.
  - The PostgreSQL-backed cases in the wider set (`test_run_progress_response.py` and `test_eval_runner.py` use `pg_*` fixtures) run only through the repo fixtures, serially, with a session-unique `TEST_DB_NAME`; see Task 4.

## Review Focus

1. **Aggregate semantics (TASK-15491 AC #3).** `mean_score`, `std_dev`, `min_score`, `max_score` and `pass_rate` now cover the scored samples only, for EVERY run type. `pass_rate` was `passed / len(results)`; it is now `passed / scored`. `failed_samples` still counts every sample with no numeric `avg_score` (unchanged). `total_samples` is unchanged. A run in which no sample is scored reports `mean_score 0`, `pass_rate 0.0`, `failed_samples == total_samples` (the existing branch, now reachable). Check that no other reader divides by `total_samples` to recompute a pass rate.
2. **Exception reach (TASK-15491).** With the inner `try` gone, a failed judge call raises out of `_eval_label_choice` into its own outer `except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS`, which returns the error dict; an exception outside that tuple goes to `_process_batch` (unchanged path). An EMPTY or unparseable reply from a call that succeeded is still accuracy 0.0 (see Decision 1).
3. **Known-provider test and the named-provider rule (TASK-15492).** `ChatProviderRegistry.DEFAULT_ADAPTERS` is the class dict of built-in adapters (including `custom-openai-api-N`). A provider registered only at runtime is not in it; ceiling noted in Decision 2. Before this plan, any prefix of the model string made a provider (`meta-llama/llama-3` with `api_name: openrouter` made `meta-llama` the provider). With the known-provider test alone, a prefix that IS a provider still hijacked an explicit one (`anthropic/claude-3.5-sonnet` + `openrouter` gave `anthropic`; `mistral:7b` + `ollama` gave `mistral`); the `named` guard closes that (Decision 5). Check the three rows `evaluator_model: openai/gpt-4o` + `api_name: openrouter`, `mistral:7b` + `ollama` and `anthropic:claude-3-opus` (no provider named) in the test.
4. **CLI (TASK-15489).** `--save/--no-save` is removed from `eval custom` (Decision 3); `_run_custom_evaluation` loses its `save` parameter and `_run_single_evaluation` stops passing it; a batch spec's `save` key is ignored for `type: custom`. `--model` is still accepted and shown, but `evaluate_custom_metric` has no model parameter, so it is not sent (pre-existing; out of scope).
5. **Exit code.** An unscored `eval custom` exits 1 with `produced no score: <explanation>` (never "completed successfully"); in a batch it is a `status: unscored` row counted in its own `Unscored` line, not in `Successful`.

---

## Task 0: Start (backlog, branch, draft PR, baseline)

**Files:** `backlog/tasks/task-15489 ...md`, `task-15491 ...md`, `task-15492 ...md` (via backlog-py only).

- [ ] **Step 1: Branch.** From the worktree: `git fetch origin dev && git checkout -b fix/evals-runner-cli-15489 origin/dev`.
- [ ] **Step 2: In Progress.** For each id: `PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-<id> -s "In Progress"` (15489, 15491, 15492). `git add -A backlog/tasks && git commit -m "chore(backlog): start TASK-15489, TASK-15491, TASK-15492"` plus the trailer.
- [ ] **Step 3: Draft PR.** `git push -u origin HEAD`, then `gh pr create --draft --base dev --title "fix(evaluations): eval custom CLI, unscored label-choice/NLI judge failures, evaluator_model provider (TASK-15489, 15491, 15492)" --body "<one paragraph + the three task ids>"` ending with the Claude Code line.
- [ ] **Step 4: Baseline.** On the base tree, with the netprobe plugin: `$PY -m pytest -q -p netprobe_15427 tldw_Server_API/tests/Evaluations/unit/test_eval_runner.py`. Expected: `44 passed` (derived: 61 on the prototype less this plan's 17 new cases; re-measure on base). The wider set (Task 4 Step 3) is 284 passed on base (derived: 307 less this plan's 23 new cases).

## Task 1: TASK-15489, the `eval custom` CLI calls `evaluate_custom_metric` with its real parameters

**Files:**
- Modify: `tldw_Server_API/cli/commands/evaluation.py` (`custom_eval` `:291-335`, `_run_custom_evaluation` `:731-764`, `_run_single_evaluation` `:881-885`, `_display_batch_summary` `:906`)
- Modify: `tldw_Server_API/cli/utils/output.py` (`print_evaluation_results` `:248`)
- Create: `tldw_Server_API/tests/Evaluations/unit/test_evals_cli_custom_command.py`

**Interfaces:**
- Consumes: `EvaluationManager.evaluate_custom_metric(metric_name, description, evaluation_prompt, input_data, scoring_criteria, api_name="openai", api_key=None, provider_credentials=None) -> {"metric_name", "score" (float|None), "explanation", "raw_output"}`.
- Produces: `_run_custom_evaluation(metric_name, text, prompt: Optional[str], provider, model, api_key, config) -> dict` with `status` (`"completed"` or `"unscored"`) and `metrics={metric_name: {"score", "explanation"}}`. The `save` parameter is gone.

- [ ] **Step 1 (RED): the tests.** Create `tldw_Server_API/tests/Evaluations/unit/test_evals_cli_custom_command.py` with the content below (the `judge` fixture stubs `evaluation_manager.analyze`, the provider call, and points the manager's database at `tmp_path`). Run `$PY -m pytest -q -p netprobe_15427 tldw_Server_API/tests/Evaluations/unit/test_evals_cli_custom_command.py`. Expected on base: `6 failed` (measured: `TypeError` from `evaluate_custom_metric() got an unexpected keyword argument 'text'`; the no-save test fails because `--save` is in the help).

```python
"""`eval custom` and a batch `type: custom` spec call evaluate_custom_metric with its real parameters (TASK-15489)."""

import json

import pytest
from click.testing import CliRunner

from tldw_Server_API.app.core.Evaluations import evaluation_manager
from tldw_Server_API.cli.commands import evaluation as eval_commands
from tldw_Server_API.cli.evals_cli import main


@pytest.fixture
def judge(monkeypatch, tmp_path):
    """Stub the provider call (`analyze`) and keep the manager's database in tmp_path; `reply` is the judge's text."""
    state = {"reply": "SCORE: 10\nA vivid story.", "calls": []}

    def _analyze(api_name, input_data, custom_prompt, api_key, **kwargs):
        state["calls"].append({"api_name": api_name, "input_data": input_data, "prompt": custom_prompt})
        return state["reply"]

    monkeypatch.setattr(evaluation_manager, "analyze", _analyze)
    real_init = evaluation_manager.EvaluationManager.__init__
    monkeypatch.setattr(
        evaluation_manager.EvaluationManager, "__init__",
        lambda self, db_path=None, **kw: real_init(self, db_path=tmp_path / "evals.db", **kw),
    )
    return state


def test_eval_custom_scores_the_text_with_the_prompt(judge):
    result = CliRunner().invoke(main, ["eval", "custom", "creativity", "Once upon a time", "--prompt", "Rate creativity"])

    assert result.exit_code == 0, result.output
    assert "1.000" in result.output and "A vivid story." in result.output
    (call,) = judge["calls"]
    assert call["api_name"] == "openai"
    assert json.loads(call["input_data"]) == {"text": "Once upon a time"}
    assert call["prompt"].startswith("Rate creativity")


def test_eval_custom_without_a_prompt_names_the_metric(judge):
    result = CliRunner().invoke(main, ["eval", "custom", "clarity", "Some text", "--format", "json"])

    assert result.exit_code == 0, result.output
    assert "'clarity'" in judge["calls"][0]["prompt"]


@pytest.mark.parametrize("reply", ["Error: provider timed out", "I cannot rate this."])
def test_eval_custom_prints_a_failed_judge_as_unscored_not_zero_or_success(judge, reply):
    judge["reply"] = reply

    result = CliRunner().invoke(main, ["eval", "custom", "creativity", "text"])

    assert result.exit_code == 1, result.output
    assert "unscored" in result.output and "produced no score" in result.output
    assert "0.000" not in result.output and "completed successfully" not in result.output


def test_eval_custom_has_no_save_option():
    """evaluate_custom_metric stores nothing, so the command does not offer to (TASK-15489)."""
    assert "--save" not in CliRunner().invoke(main, ["eval", "custom", "--help"]).output


def test_a_batch_custom_spec_runs_and_an_unscored_one_is_counted_apart(judge, monkeypatch):
    spec = {"type": "custom", "metric_name": "creativity", "text": "t", "prompt": "Rate it"}
    scored = eval_commands._run_single_evaluation(spec, "openai", None, None, {})
    judge["reply"] = "Error: boom"
    unscored = eval_commands._run_single_evaluation(spec, "openai", None, None, {})
    rows: list = []
    monkeypatch.setattr(eval_commands, "print_table", lambda data, title=None: rows.extend(data))
    monkeypatch.setattr(eval_commands, "print_info", lambda *_a, **_k: None)

    eval_commands._display_batch_summary([scored, unscored])

    assert scored["metrics"]["creativity"]["score"] == 1.0 and scored["status"] == "completed"
    assert unscored["metrics"]["creativity"]["score"] is None and unscored["status"] == "unscored"
    assert {row["Metric"]: row["Count"] for row in rows if row["Metric"] != "Success Rate"} == {
        "Total Evaluations": 2, "Successful": 1, "Failed": 0, "Unscored": 1,
    }
```

- [ ] **Step 2 (GREEN): the CLI.** In `evaluation.py`:
  - delete the `--save/--no-save` option line and the `save` parameter of `custom_eval`; pass `cli_context.config` right after `api_key` in the `_run_custom_evaluation(...)` call;
  - after the display, add:
    ```python
    if result['status'] == 'unscored':  # the judge call failed or its reply held no score: not a success
        print_error(f"Custom evaluation '{metric_name}' produced no score: {result['details'].get('explanation')}")
        sys.exit(1)
    ```
    before the existing `print_success(...)`;
  - `_run_custom_evaluation(metric_name, text, prompt: Optional[str], provider, model, api_key, config)`: replace the call and return with
    ```python
    result = asyncio.run(eval_manager.evaluate_custom_metric(
        metric_name=metric_name,
        description=metric_name,
        evaluation_prompt=prompt or f"Evaluate the text below on the metric '{metric_name}'.\n\n{{text}}",
        input_data={'text': text},
        scoring_criteria={metric_name: prompt or 'Overall quality on this metric'},
        api_name=provider,
        api_key=api_key,
    ))
    score = result.get('score')  # None: no score, kept apart from a 0.0 (TASK-15485)

    return {
        'evaluation_type': 'custom',
        'status': 'unscored' if score is None else 'completed',
        'metric_name': metric_name,
        'provider': provider,
        'model': model,
        'timestamp': format_timestamp(None),
        'metrics': {metric_name: {'score': score, 'explanation': result.get('explanation', '')}},
        'details': result
    }
    ```
  - `_run_single_evaluation`: drop `save` from the `type == 'custom'` call;
  - `_display_batch_summary`: `unscored = sum(1 for r in results if r.get('status') == 'unscored')`, `successful = sum(1 for r in results if r.get('status') not in ('failed', 'unscored'))`, `failed = len(results) - successful - unscored`, and a new `{'Metric': 'Unscored', 'Count': unscored}` row after `Failed`.
  In `output.py` `print_evaluation_results`, before `elif isinstance(score, (int, float)):` insert `if score is None: score_display = "[dim]unscored[/dim]"` (turning the existing `if isinstance` into `elif`).
- [ ] **Step 3: Run.** Same command as Step 1. Expected: `6 passed`, netprobe log 0 lines. Also run `tldw_Server_API/tests/Evaluations/unit/test_evals_cli_benchmark_commands.py` and `test_evals_cli_recipe_commands.py` (same CLI group). Expected: pass.
- [ ] **Step 4: Commit.** `git add -A tldw_Server_API && git commit -m "fix(cli): eval custom calls evaluate_custom_metric with its real parameters and prints no score as unscored (TASK-15489)"` plus trailer.

## Task 2: TASK-15491, a failed label-choice or NLI judge call is a sample error; the aggregate leaves it out

**Files:**
- Modify: `tldw_Server_API/app/core/Evaluations/eval_runner.py` (`_eval_label_choice` `:1955-1971`, `_eval_nli_factcheck` `:2148-2164`, `_calculate_aggregate_results` `:2331-2386`)
- Modify: `tldw_Server_API/tests/Evaluations/unit/test_eval_runner.py` (append)
- Modify (docs): `Docs/API-related/Evaluations_API_Reference.md` line 357, then the mirror

**Interfaces:** `_eval_label_choice` and `_eval_nli_factcheck` return `{"sample_id", "error"}` when the judge call raises (no `scores`, no `avg_score`, no `passed`). `_calculate_aggregate_results(results, metrics, threshold)` keeps its signature and keys.

- [ ] **Step 1 (RED): tests.** Append to `tldw_Server_API/tests/Evaluations/unit/test_eval_runner.py` (it reuses the file's `_SAMPLE` and `_SPEC`):

```python
@pytest.mark.asyncio
@pytest.mark.parametrize("eval_type", ["label_choice", "nli_factcheck"])
@pytest.mark.parametrize("failure", [ConnectionError("down"), TimeoutError("slow")])
async def test_a_failed_judge_call_is_a_sample_error_not_accuracy_zero(monkeypatch, tmp_path, eval_type, failure):
    """A label-choice or NLI judge call that raises leaves the sample unscored: an error, no score (TASK-15491)."""
    from types import SimpleNamespace

    from tldw_Server_API.app.core.Evaluations import eval_runner

    def _chat(_request, timeout=None):
        raise failure

    monkeypatch.setattr(eval_runner, "get_adapter_or_raise", lambda _provider: SimpleNamespace(chat=_chat))
    runner = EvaluationRunner(db_path=str(tmp_path / "evals.db"))
    fn = {"label_choice": runner._eval_label_choice, "nli_factcheck": runner._eval_nli_factcheck}[eval_type]

    result = await fn(_SAMPLE, _SPEC, {"config": {"api_name": "openai", "model": "m"}}, "sample_000001")

    assert result == {"sample_id": "sample_000001", "error": str(failure)}


@pytest.mark.asyncio
@pytest.mark.parametrize("eval_type", ["label_choice", "nli_factcheck"])
async def test_a_judge_config_error_is_a_sample_error(monkeypatch, tmp_path, eval_type):
    """ChatConfigurationError (a refused handle, no model) is a sample error too (TASK-15491)."""
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
    from tldw_Server_API.app.core.Evaluations import eval_runner

    def _refused(**_kwargs):
        raise ChatConfigurationError(provider="openai", message="Provider credential context is invalid.")

    monkeypatch.setattr(eval_runner, "_call_adapter_text", _refused)
    runner = EvaluationRunner(db_path=str(tmp_path / "evals.db"))
    fn = {"label_choice": runner._eval_label_choice, "nli_factcheck": runner._eval_nli_factcheck}[eval_type]

    result = await fn(_SAMPLE, _SPEC, {"config": {"api_name": "openai", "model": "m"}}, "s1")

    assert "error" in result and "scores" not in result and "avg_score" not in result, result


def test_the_aggregate_leaves_a_sample_error_out_of_mean_and_pass_rate(tmp_path):
    runner = EvaluationRunner(db_path=str(tmp_path / "evals.db"))
    results = [{"avg_score": 1.0, "passed": True}, {"error": "judge down"}, {"avg_score": 0.0, "passed": False}]

    aggregate = runner._calculate_aggregate_results(results, None, 0.7)

    assert (aggregate["mean_score"], aggregate["pass_rate"], aggregate["min_score"]) == (0.5, 0.5, 0.0)
    assert (aggregate["total_samples"], aggregate["failed_samples"]) == (3, 1)


def test_an_all_unscored_run_reports_every_sample_failed(tmp_path):
    runner = EvaluationRunner(db_path=str(tmp_path / "evals.db"))

    aggregate = runner._calculate_aggregate_results([{"error": "a"}, {"error": "b"}], None, 0.7)

    assert (aggregate["mean_score"], aggregate["pass_rate"], aggregate["failed_samples"]) == (0, 0.0, 2)
```

  Then the pin for review M2 (the call is the system under test, not a judge: an unparseable reply from a call that succeeded stays a wrong answer; it passes on base and keeps Decision 1 from changing by accident). The reply is `xyz` because the label matcher is a substring test, so a reply such as `banana` would match the label `A`:

```python
@pytest.mark.asyncio
@pytest.mark.parametrize("eval_type", ["label_choice", "nli_factcheck"])
async def test_an_unparseable_reply_from_a_successful_call_is_a_wrong_answer(monkeypatch, tmp_path, eval_type):
    """The call is the system under test, not a judge: a reply that names no allowed label is a wrong answer.

    Accuracy 0.0 fits the owner's unscored rule, which is about a grader's reply with no readable score (TASK-15491).
    """
    from tldw_Server_API.app.core.Evaluations import eval_runner

    monkeypatch.setattr(eval_runner, "_call_adapter_text", lambda **_kwargs: "xyz")
    runner = EvaluationRunner(db_path=str(tmp_path / "evals.db"))
    fn = {"label_choice": runner._eval_label_choice, "nli_factcheck": runner._eval_nli_factcheck}[eval_type]

    result = await fn(_SAMPLE, _SPEC, {"config": {"api_name": "openai", "model": "m"}}, "s1")

    assert (result["scores"], result["passed"], result["avg_score"]) == ({"accuracy": 0.0}, False, 0.0)
```

  Run `$PY -m pytest -q -p netprobe_15427 tldw_Server_API/tests/Evaluations/unit/test_eval_runner.py -k "failed_judge or judge_config_error or aggregate_leaves or all_unscored or unparseable"`. Expected on base (measured on the earlier prototype for the first four; the `unparseable` cases measured against the pre-fold source): `7 failed, 3 passed` (the 4 failed-judge cases, 2 config-error cases and the aggregate case fail; the all-unscored guard and the 2 unparseable pins pass).

- [ ] **Step 2 (GREEN): the runner.**
  - In both functions replace the `try: resp = await asyncio.to_thread(_call_adapter_text, ...) except _EVAL_RUNNER_NONCRITICAL_EXCEPTIONS as ce: logger.error(...); resp = ""` block with the bare call, dedented one level, under the comment `# A failed judge call raises to the sample-error handler below: unscored, not accuracy 0.0 (TASK-15491)`.
  - In `_calculate_aggregate_results` replace the score loop with
    ```python
    # A sample with no score (an error, a failed judge call) is unscored: in no mean or pass rate
    all_scores = []
    passed_count = 0
    failed_samples = 0

    for result in results:
        try:
            score_val = float(result.get("avg_score"))
        except (TypeError, ValueError):
            failed_samples += 1
            continue
        all_scores.append(score_val)
        if result.get("passed", False):
            passed_count += 1
    ```
    and `"pass_rate": passed_count / len(all_scores)` (the `if not all_scores:` branch above it is unchanged).
- [ ] **Step 3: Run.** The command of Step 1 without `-k`. Expected: `70 passed` in `test_eval_runner.py` (44 base + 26 new, of which 10 belong to this task and 16 to Task 3). Measured on the folded prototype in a scratch clone: `test_eval_runner.py` plus the CLI file `76 passed`, netprobe log 0 lines.
- [ ] **Step 4: Doc.** In `Docs/API-related/Evaluations_API_Reference.md` line 357 change `"failed_samples": 25` to `"failed_samples": 25   // Unscored (no score; a failed judge call): in no mean_score or pass_rate`. Do the refresh once, in Task 3's doc step, so the mirror diff is one change.
- [ ] **Step 5: Commit** the runner, the tests and the doc line: `fix(evaluations): a failed label-choice or NLI judge call is a sample error, and the aggregate leaves unscored samples out (TASK-15491)` plus trailer.

## Task 3: TASK-15492, `evaluator_model` is a provider only when it names one

**Files:**
- Modify: `tldw_Server_API/app/core/Evaluations/eval_runner.py` (imports `:12` and `:47`; new helpers above `class EvaluationRunner` `:178`; `_resolve_eval_provider_model_api_key` `:240-251`)
- Modify: `tldw_Server_API/tests/Evaluations/unit/test_eval_runner.py` (append)
- Modify (docs): `Docs/API-related/Evaluations_API_Reference.md` line 55, and the `Docs/Published` mirror

**Interfaces:**
- Produces: `_is_known_provider(name: str) -> bool`; `_split_provider_prefix(api_name: str, value: str) -> tuple[str, str]` (module level in `eval_runner.py`).
- Unchanged: `_resolve_eval_provider_model_api_key(eval_spec, eval_config) -> (api_name, model_override, api_key)` and `run_llm_provider`, which calls it.

- [ ] **Step 1 (RED): the test.** Append to `test_eval_runner.py` (each row asserts the resolver's `(api_name, model)` AND that `run_llm_provider` names the same pair, AC #3: the route names the runner's provider). The first two rows are AC #1 (a regression guard that passes on base); the rows below the comment are review I1:

```python
@pytest.mark.parametrize("spec,want", [
    ({"evaluator_model": "openai"}, ("openai", None)),
    ({"evaluator_model": "Anthropic"}, ("anthropic", None)),
    ({"evaluator_model": "gpt-4"}, ("openai", "gpt-4")),
    ({"evaluator_model": "claude-2", "api_name": "anthropic"}, ("anthropic", "claude-2")),
    ({"evaluator_model": "gpt-4", "model": "gpt-4o"}, ("openai", "gpt-4o")),
    ({"evaluator_model": "anthropic:claude-3-opus"}, ("anthropic", "claude-3-opus")),
    ({"evaluator_model": "openrouter/meta-llama/llama-3"}, ("openrouter", "meta-llama/llama-3")),
    ({"evaluator_model": "ft:gpt-3.5-turbo:org"}, ("openai", "ft:gpt-3.5-turbo:org")),
    ({"model": "meta-llama/llama-3", "api_name": "openrouter"}, ("openrouter", "meta-llama/llama-3")),
    # a provider named in api_name/provider stops the split: these are the ids of OpenRouter, Together and Ollama
    ({"model": "anthropic/claude-3.5-sonnet", "api_name": "openrouter"}, ("openrouter", "anthropic/claude-3.5-sonnet")),
    ({"model": "google/gemma-2-9b", "provider": "openrouter"}, ("openrouter", "google/gemma-2-9b")),
    ({"model": "Qwen/Qwen2-72B-Instruct", "api_name": "together"}, ("together", "Qwen/Qwen2-72B-Instruct")),
    ({"model": "cohere/command-r", "api_name": "openrouter"}, ("openrouter", "cohere/command-r")),
    ({"model": "mistral:7b", "api_name": "ollama"}, ("ollama", "mistral:7b")),
    ({"model": "qwen:14b", "provider": "ollama"}, ("ollama", "qwen:14b")),
    ({"evaluator_model": "openai/gpt-4o", "api_name": "openrouter"}, ("openrouter", "openai/gpt-4o")),
])
def test_evaluator_model_is_a_provider_only_when_it_names_one(spec, want):
    """A model name is a model, a provider name a provider, "provider:model" both; the run route names the same one."""
    from tldw_Server_API.app.core.Evaluations.eval_runner import run_llm_provider

    api_name, model, _key = EvaluationRunner._resolve_eval_provider_model_api_key(spec, {})

    assert (api_name, model) == want
    assert run_llm_provider({"eval_type": "label_choice", "eval_spec": spec}, {}) == (api_name, model)
```

  Expected on base: the 7 I1 rows and rows 3-9 of the first nine fail (14 failed, 2 passed; to re-measure at execution). Measured against the PRE-FOLD source (without the `named` guard): the 7 I1 rows fail, so the guard is what they test (`7 failed` of the 16 rows, with the 2 unparseable pins passing).

- [ ] **Step 2 (GREEN): the resolver.** Add `import re` and `from tldw_Server_API.app.core.LLM_Calls.adapter_registry import ChatProviderRegistry`. Above `class EvaluationRunner` add

    ```python
    def _is_known_provider(name: str) -> bool:
        return bool(name) and canonical_provider_name(name) in ChatProviderRegistry.DEFAULT_ADAPTERS


    def _split_provider_prefix(api_name: str, value: str) -> tuple[str, str]:
        """``("openai", "gpt-4")`` for ``"openai:gpt-4"`` or ``"openai/gpt-4"``; ``(api_name, value)`` when the prefix is no provider."""
        match = re.fullmatch(r"([^:/]+)[:/](.+)", value, re.DOTALL)
        return (normalize_provider(match[1]), match[2]) if match and _is_known_provider(match[1]) else (api_name, value)
    ```

  In the resolver replace the `api_name = (...)` / `api_name = str(api_name).strip() or "openai"` pair with
    ```python
    named = str(
        eval_spec.get("api_name")
        or eval_spec.get("provider")
        or run_config.get("api_name")
        or run_config.get("provider")
        or ""
    ).strip()
    api_name = named or "openai"
    ```
  and replace the two blocks from `# Backward compatibility: evaluator_model ...` to `return api_name, model_override, api_key` with
    ```python
    # evaluator_model names a provider, a model, or "provider:model" / "provider/model". Only a
    # registered provider counts as one, so a model name such as "gpt-4" stays a model (TASK-15492).
    evaluator_model = str(eval_spec.get("evaluator_model") or "").strip()
    if _is_known_provider(evaluator_model):
        api_name = normalize_provider(evaluator_model)
    elif evaluator_model:
        if not named:  # a provider named in api_name/provider keeps "anthropic/claude-3.5-sonnet" a model id
            api_name, evaluator_model = _split_provider_prefix(api_name, evaluator_model)
        model_override = model_override or evaluator_model

    # Allow model strings like "provider:model" or "provider/model" to override provider, unless one is named
    if not named and isinstance(model_override, str) and model_override:
        api_name, model_override = _split_provider_prefix(api_name, model_override)

    return api_name, model_override, api_key
    ```

- [ ] **Step 3: Run.** The new test, then the whole `test_eval_runner.py` (`70 passed`), then `test_evaluations_crud_create_run_api.py`, `test_evaluations_crud_endpoint_sanitization.py` and `unit/test_evaluations_identity_route_bindings.py` (the route files that name `run_llm_provider`). Expected: all pass; netprobe 0 lines.
- [ ] **Step 4: Doc.** Line 55 of `Docs/API-related/Evaluations_API_Reference.md`: change the comment to `// A provider ("openai"), a model ("gpt-4") or "provider:model"; a model runs on api_name/provider, else OpenAI`. Then `bash Helper_Scripts/refresh_docs_published.sh`; expected: the only `Docs/Published` change is `Docs/Published/API-related/Evaluations_API_Reference.md` (both doc edits, lines 55 and 357).
- [ ] **Step 5: Commit** `fix(evaluations): evaluator_model is a provider only when it names one; provider:model gives both (TASK-15492)` plus trailer, including the doc and mirror.

## Task 4: Verify, close out, review, PR

- [ ] **Step 1: Function sizes** (none touched function is in `function_size_baseline.json`; only `_execute_rag_pipeline_run` of these files is, and it is untouched). Measured by `ast` (lines, before -> after): `_resolve_eval_provider_model_api_key` 49 -> 42; `_eval_label_choice` 196 -> 193; `_eval_nli_factcheck` 192 -> 189; `_calculate_aggregate_results` 60 -> 56; new `_is_known_provider` (2) and `_split_provider_prefix` (4); CLI: `_run_custom_evaluation` 35 -> 37, `custom_eval` 43 -> 46, `_display_batch_summary` 16 -> 18. All far under the 300-line ratchet threshold. Run `$PY -m pytest -q -p netprobe_15427 tldw_Server_API/tests/lint/test_function_size_ratchet.py tldw_Server_API/tests/lint/test_http_usage_guard.py` (both name the edited files). `direct_llm_adapter_calls_baseline.json` lists `eval_runner.py` with `references: 1`; this plan adds none (the `get_adapter_or_raise` call count is unchanged).
- [ ] **Step 2: Wider set** (every test file that names a symbol or route this plan touches; the `git grep` lists are in the measured note below). Run serially, no `-n`: `test_evaluations_core_hardening.py`, `test_evaluations_stage2_user_isolation_and_usage_accounting.py`, `test_evaluations_stage3_batch_failfast_and_metrics_none.py`, `test_evaluations_unified.py`, `test_rag_pipeline_runner.py`, `test_run_progress_response.py`, `test_evaluations_crud_create_run_api.py`, `test_evaluations_crud_endpoint_sanitization.py`, `test_unified_evaluation_service_strict_audit.py`, and under `unit/`: `test_eval_runner.py`, `test_evals_cli_custom_command.py`, `test_evals_cli_benchmark_commands.py`, `test_evals_cli_recipe_commands.py`, `test_persona_chat_judge_review_command.py`, `test_rag_evaluator_alias_scoring.py`, `test_scoring_normalization.py`, `test_unified_evaluation_service_mapping.py`, `test_evaluations_identity_route_bindings.py`. Expected: `316 passed` on the folded prototype (307 measured on the pre-fold prototype, plus the 9 cases the fold added; the 9 are derived, to re-measure at execution), 284 on base (derived); netprobe 0 lines. None of these tests uses the `mocker` fixture (grep), so the stub-vs-real `pytest-mock` difference does not touch them; the two files this plan adds or changes were re-run with the real plugin: `23 passed` for the new cases.
- [ ] **Step 3: PostgreSQL files.** Two files in the set take a PostgreSQL fixture: `test_eval_runner.py` (`pg_database_config`, the 3 `postgresql` cases of `test_two_row_exact_match_run_retains_scores_and_dataset`) and `test_run_progress_response.py` (`pg_restricted_backend`, 2 cases). `test_evaluations_postgres_crud.py` names none of this plan's symbols and is not in the set. Run them serially through the repo fixtures with a session-unique name (`TEST_DB_NAME=evals15489_<unique>`), and record the counts. In the planning pass (`TLDW_TEST_NO_DOCKER=1`, no `TEST_DB_NAME`) the 3 `postgresql` cases of `test_eval_runner.py` ran and passed (`3 passed`, no skip reported), and the wide set reported `307 passed` with no skip; the executor must still re-measure with a session-unique `TEST_DB_NAME`.
- [ ] **Step 4: CI checks.** `$PY -m pytest -q -p netprobe_15427 tldw_Server_API/tests/Docs tldw_Server_API/tests/lint tldw_Server_API/tests/CI` (expect only the environment-only failures: `test_runtime_package_discovery.py` errors, `test_jobs_postgres_exports_recorded_coverage` 2 cases; docling absent); `$PY Helper_Scripts/ci/check_shard_coverage.py` (the new file `tests/Evaluations/unit/test_evals_cli_custom_command.py` must be in a shard: `tests/Evaluations/unit` is a covered directory, expected `new_uncovered=0`); `$PY Helper_Scripts/ci/test_quality_triage.py --enforce` (expect no offense beyond baseline; the new tests use at most two `monkeypatch` calls each and no `time.sleep`); `/usr/bin/python3 Helper_Scripts/ci/check_backlog_task_ids.py .` (exit 0); `/opt/homebrew/bin/uvx bandit -ll` on `eval_runner.py`, `cli/commands/evaluation.py`, `cli/utils/output.py`; `git diff --check`. Measured in the planning pass on the prototype: shard coverage `shards=910 test_files=5494 ignored=4 baseline=44 new_uncovered=0`, `OK`; `check_backlog_task_ids.py` exit 0; `git diff --check` exit 0; Bandit `-ll` on the three source files `No issues identified.`; ruff on the five touched `.py` files: 2 findings, `eval_runner.py` `I001` (import block) and `TRY203` (`:2483`), the same two the base file has (the new import sits in the already-unsorted block; not fixed, out of scope); `test_quality_triage.py --enforce` exit 0 on the prototype; the `tests/Docs`, `tests/lint`, `tests/CI` directories were NOT run whole in the planning pass (the executor runs them; the mirror test in `tests/Docs` is the one this plan's doc edit exercises).
- [ ] **Step 4b: Follow-up task for the sibling runners (review I2, confirmed).** Under the controller's id lock create `TASK-<NEW>`, "eval rag and eval quality CLI runners call their evaluators with arguments they do not take" (labels `evaluations`, `cli`, `bug`), with ACs mirroring TASK-15489 AC 1 (`eval rag`/`eval quality` and a batch spec of that type call the evaluator with its real parameters and print the score; a CLI test with a stubbed provider pins it), AC 2 (an item with no score prints as unscored) and one AC for the `--save` flag on those commands. Description, verified by reading the signatures: `_run_rag_evaluation` raises `TypeError: __init__() got an unexpected keyword argument 'api_name'` on `RAGEvaluator(api_key=, api_name=)`, and its `evaluate(save_results=)` has no such parameter; `_run_quality_evaluation` raises `TypeError: ResponseQualityEvaluator() takes no arguments`, and the real `evaluate` parameters are `prompt, response, expected_format, custom_criteria, api_name, api_key`; `_run_geval_assessment` is fine. Not in this PR (it needs its own decisions on `--save`, on `RAGEvaluator`'s embedding provider and on the result shape). Link it from the three tasks' notes and the PR body.
- [ ] **Step 4c: After TASK-15487 merges,** re-run `tldw_Server_API/tests/Evaluations/unit/test_evals_cli_custom_command.py`: 15487 widens `evaluate_custom_metric`'s score parser (a bare number now scores), and the CLI tests' unscored replies (`Error: ...`, `I cannot rate this.`) hold no number, so they should keep passing; confirm.
- [ ] **Step 5: Close-out.** For each task: check its ACs (`--check-ac 1 --check-ac 2 ...`; 15489 has 3, 15491 has 3, 15492 has 3) and DoD (`--check-dod 1..6`), `--final-summary` with the counts above, and `--append-notes` with the evidence per AC:
  - TASK-15489: AC1 the CLI tests (`6 failed` on base, `6 passed`); AC2 the `unscored` print and the exit-1 test; AC3 `--save/--no-save` removed (the command never stored anything). Note: `--model` is accepted but not sent, because `evaluate_custom_metric` takes no model. DoD #3: no doc names `eval custom --save`.
  - TASK-15491: AC1/AC2 the four failed-call cases plus two config-error cases; AC3 the aggregate test; owner confirmation 2026-11-12 cited. Note that an unparseable reply from a successful call stays accuracy 0.0 (Decision 1).
  - TASK-15492: AC1 the two provider-name cases (a regression guard: they pass on base); AC2 the model cases; AC3 the `provider:model`, `provider/model` and the route-equality assertion. Note the model-string split fix (`meta-llama/llama-3`).
  Then status Done: `... task edit TASK-<id> -s Done` for each. `git add -A backlog/tasks`; one commit per task `chore(backlog): close TASK-<id>` plus trailer.
- [ ] **Step 6: Final review.** In a clean subagent (the controller dispatches it; this plan does not): `/superpowers:requesting-code-review` on the PR diff, with Review Focus above in the brief. Fold findings, re-run Step 2.
- [ ] **Step 7: PR.** Mark the draft ready; the body lists the three tasks, the counts, the Docs/Published diff, Decisions 1-3 and ends with the Claude Code line. Merge through the controller's queue (MERGE_QUEUE unset: rebase only the PR about to merge, wait for the required statuses). Clean the Go cache: `chmod -R u+w tldw_Server_API/Config_Files/acp_runner_home && git clean -ffdx -- tldw_Server_API/Config_Files/acp_runner_home`.

## Decisions (decided by the controller on the review; recorded, none blocks the plan)

1. **Unparseable reply from a label-choice or NLI call that SUCCEEDED: keep accuracy 0.0 (decided).** Evidence (review): the call is the system under test, not a judge. `_eval_label_choice` and `_eval_nli_factcheck` call the model only when no prediction was supplied and `generate_predictions` is set; the prompt is the question itself; the score is `pred == gold`, where `gold` comes from the dataset's `expected`. The owner's rule is about a grader's reply with no readable score, so a non-answer from the model under test is a wrong answer, and a call that RAISES has no prediction at all and is unscored. Pinned by `test_an_unparseable_reply_from_a_successful_call_is_a_wrong_answer`.
2. **Known provider = the built-in adapter table (decided).** `ChatProviderRegistry.DEFAULT_ADAPTERS` plus the aliases of `canonical_provider_name`. A provider registered only at runtime would read as a model name; none exists today. `evaluator_model: gemini-1.5-pro` is now a model on OpenAI instead of a provider named `gemini` (both fail, differently; the provider is `google`).
3. **Remove `--save/--no-save` from `eval custom` (decided).** `evaluate_custom_metric` stores nothing, and nothing in Docs (minus the mirror), `Helper_Scripts` or the tests passes the flag. Keeping it would mean calling `store_evaluation`, which writes `score` with a `0.0` default for a missing key, and editing `evaluation_manager.py`, which TASK-15487 also edits.
4. **An unscored `eval custom` exits 1 (decided).** A script treating exit 0 as "scored" would otherwise average nothing.
5. **An explicitly named provider stops the `prefix:` / `prefix/` split (decided, review I1).** Without it, OpenRouter, Together and Ollama ids (`anthropic/claude-3.5-sonnet`, `Qwen/Qwen2-72B-Instruct`, `mistral:7b`) would run on the wrong provider with the user's other provider's key. `provider:model` still selects both where no provider is named, which is the case AC #3 describes. In Task 3's tests.
6. **The broken `eval rag` and `eval quality` runners get a follow-up task, not this PR (decided, review I2).** Task 4 Step 4b.
7. **ADR-004 Change summary waiver:** not yet answered for this PR; the controller asks.

## Notes (no change made)

- **M3: an empty-content reply scores 0.0.** A reasoning model that spends its `max_tokens=16` thinking returns empty content, which `_call_adapter_text` turns into `str(response)`; the parser finds no label and the sample scores 0.0 although nothing was answered. This is the only place the unscored alternative to Decision 1 would matter; not worth doing now.
- **M4: `--model` on `eval custom` is accepted and not used** (`evaluate_custom_metric` has no model parameter), so the result table prints a model that was never sent. Pre-existing; left as is.
- **M5: a `NaN` `avg_score` crashes `statistics.stdev`** in `_calculate_aggregate_results` for a run with two or more scores (`AttributeError: 'float' object has no attribute 'numerator'`), as on base. An optional `math.isfinite` guard in the new `try` would close it; not added.
- **Label matcher quirk (seen while writing the M2 test):** a reply is matched to a label by substring (`lab in up`), so with labels `A` and `B` a reply such as `banana` is read as `A`. Pre-existing; not touched.

## Measured notes and what was not measured

- Base: `origin/dev` `3883a73712`. Folded prototype: 7 files changed, `+271 -82`.
- RED on base (pre-fold prototype): the 17 runner cases of the first plan: `14 failed, 3 passed`; the 6 CLI cases: `6 failed`. Fold: against the pre-fold source, the 7 new I1 rows fail and the 2 unparseable pins pass (`7 failed, 11 passed` in `-k "evaluator_model_is or unparseable"`). Against the pristine base the I1 rows are expected to fail too (to re-measure at execution).
- GREEN (folded prototype, in a scratch clone of `3883a73712` under the scratchpad, no branch of the repo touched): `test_eval_runner.py` and `test_evals_cli_custom_command.py` together `76 passed` (70 + 6), netprobe log 0 lines, real `pytest-mock` on the path. Not re-run after the fold: the wider set (derived `316`, to re-measure at execution), the PostgreSQL cases, `tests/Docs`, `tests/lint`, `tests/CI`, ruff and Bandit on the new resolver lines.
- Review's own measurements on `61ad5280ca` (reproduced the plan's numbers): the wider set `307 passed` with the PostgreSQL cases; shard coverage `new_uncovered=0`; triage exit 0; `check_backlog_task_ids.py` exit 0; Bandit no issues; ruff the same two pre-existing findings. The review could not finish `tests/Docs`, `tests/lint` or `tests/CI` whole on the loaded machine (only `test_function_size_ratchet.py` `13 passed`); the executor runs them.
- Overlap with open PRs: only #3269 (stale, CONFLICTING) touches `eval_runner.py` and `test_eval_runner.py`. TASK-15487's plan edits `evaluation_manager.py` and `test_evaluation_manager.py`, which this plan does not touch; see Task 4 Step 4c for the semantic interaction.
- Function sizes (measured on the pre-fold prototype): resolver 49 -> 42 before the `named` guard; the guard adds about 6 lines, to re-measure at execution (the ratchet threshold is 300).
