# TASK-15496 Implementation Plan: own-key analyze calls use the DEFAULT_MODEL_<PROVIDER> fallback

> For agentic workers: execute task by task, RED before GREEN.

**Goal:** An own-key call through `analyze` reaches `DEFAULT_MODEL_<PROVIDER>` when the handle's config has no model,
and that model meets the provider-override model policy.

**Architecture:** One change in the shared `_summarize_via_adapter`
(`tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py`). When a handle is present and the handle's config
yields no model, fall back to `resolve_provider_model(provider, effective_app_config)` (the same function the
server-key path uses; the config section is empty by then, so it reads only the environment and MLX path). A fallback
model is then checked with `capture_provider_override_call_snapshot(provider).enforce(model)`, imported inside a small
helper `_model_allowed_by_override`. A denial, or an unreadable policy (`ByokResolutionError`), fails the call with the
existing `configuration` code (legacy text `Error: Provider credential context is invalid.`), before any adapter call.
No other file in `app/` changes. Every handle caller (study assistants, benchmarks, watchlists, evaluation runs, media)
reaches `_summarize_via_adapter` through `_dispatch_to_api`, so the one fix covers them.

**Tech Stack:** Python, pytest.

**Spec:** `backlog/tasks/task-15496 - Own-key-calls-through-analyze-skip-the-DEFAULT-MODEL-PROVIDER-fallback.md`

## Global Constraints

- Tests make no real outbound network requests. Every pytest run uses the network probe plugin:
  `PYTHONPATH=<plans dir> NETPROBE_LOG=<file> python -m pytest -p netprobe_15427 ...`. It blocks and logs every
  outbound DNS lookup and connection; a run must end with 0 log lines for the tests the plan adds or changes.
- A test that drives `/chat/completions`, an LLM adapter or a judge must stub the adapter or the provider call. A
  commercial provider otherwise calls the real API.
- A failed judge or model call, or a reply with no readable score, is unscored: `score: None`, a sample error, and
  out of every average. Never 0.0, never a success. This is the owner's rule, recorded 2026-11-10 and 2026-11-12.
- Backlog edits use backlog-py only (`PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task edit ...`).
  Use `--append-notes`, never `--notes`. Stage with `git add -A backlog/tasks`. Never hand-edit task files. A
  new follow-up task takes its id under the session's id lock (the controller does that; write `TASK-<NEW>`).
- Branch from `origin/dev`; the PR targets `dev`.
  - Commit trailer: `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
  - The PR body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- Never `--no-verify`, never force-push, never `git stash`.
- PostgreSQL only through the repo fixtures, with a session-unique `TEST_DB_NAME`. Never restart the server.
  Redis: never FLUSHALL or SCRIPT FLUSH.
- A mirrored doc (ADRs, the BYOK guide, `Env_Vars.md` and the other files under `Docs/Published`) needs
  `bash Helper_Scripts/refresh_docs_published.sh`, with the `Docs/Published` diff in the same PR. (This plan touches no
  mirrored doc.)
- Shifting lines in a file listed in `Helper_Scripts/ci/scope_predicate_baseline.txt` fails backend-required.
  Regenerate it with the script's `--write-baseline`, in the same PR. Measured: `Summarization_General_Lib.py` is not
  in the baseline files list checked here; re-check in Task 2.
- Size-capped functions (the function-size ratchet in `tests/lint`) must not grow past their caps. Measured:
  `_summarize_via_adapter` is not in `function_size_baseline.json` (threshold 300 lines; it is about 200 and grows
  by 12).
- A plan with embedded diffs fails trailing-whitespace when it is committed: keep the plan free of trailing spaces.
- Extra: the import must stay inside the helper. A module-level import of `llm_provider_overrides` would add AuthNZ
  modules to the load set that `tests/lint/test_llm_calls_import_isolation.py` ratchets.

## Review Focus

1. The fallback applies only when a handle is present and the handle's config has no model; a configured model and a
   `model_override` take the old paths unchanged (the override is checked at resolve time, as before).
2. A model chosen at dispatch is enforced against the override snapshot, and the failure is closed: a denied model, a
   disabled provider and an unhealthy override cache all stop the call before the adapter.
3. The `configuration` code and its text are reused, so no new public error code or doc change.
4. The server-key path (no handle) is untouched.

## Task 0: Setup

- [ ] Mark In Progress: `PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task edit 15496 -s "In Progress"`
  (use the exact flags of the repo's backlog-py usage; stage with `git add -A backlog/tasks`).
- [ ] `git fetch origin dev`, then `git switch -c fix/own-key-default-model-15496 origin/dev`.
- [ ] Push the branch and open a draft PR with base `dev`.
- [ ] Baseline (serial, `-n 0`), from the repo root with the repo venv and the probe plugin:
  `tests/LLM_Calls/test_provider_adapter_runtime_boundary.py`, `tests/LLM_Calls/test_summarization_adapter.py`
  (127 tests; its `test_explicit_missing_summary_model_does_not_use_default_model_environment` is the one Step 4 changes),
  `tests/LLM_Calls/test_summarization_runtime_credentials.py`,
  `tests/Chat/unit/test_authoritative_adapter_translation.py`, `tests/Media/test_media_llm_quota.py`,
  `tests/Media/test_audio_summary_service_prompt.py`, `tests/Media/test_email_summary_service_prompt.py`,
  `tests/MediaIngestion_NEW/unit/test_media_ingest_job_llm_quota_refusal.py`,
  `tests/lint/test_function_size_ratchet.py`, `tests/lint/test_llm_calls_import_isolation.py`.
  Note: the shared venv in this environment had no `pytest_mock`; the planner ran with an empty stub module on
  `PYTHONPATH`. If the executor's venv has it, no stub is needed.

## Task 1: RED tests, then the fix

**Files:**
- Modify: `tldw_Server_API/tests/LLM_Calls/test_provider_adapter_runtime_boundary.py` (add before
  `test_event_gated_cloud_calls_keep_rotated_credentials_and_timeouts_isolated`)
- Modify: `tldw_Server_API/app/core/LLM_Calls/Summarization_General_Lib.py`
- Modify: `tldw_Server_API/tests/LLM_Calls/test_summarization_adapter.py` (invert one test, add one; Step 4)

**Interfaces:** `_model_allowed_by_override(provider: str, model: str) -> bool` (private). No public signature
changes. `_summarize_via_adapter` returns `"Error: Provider credential context is invalid."` (or raises
`SummaryProviderError(code="configuration")` when `raise_on_error`) for a denied fallback model.

- [ ] **Step 1: tests.** `proto_15496.patch` holds them. In `test_provider_adapter_runtime_boundary.py`, the helper
  `_own_key_summary` (a genuine handle from the file's `_issue_credentials`, a recording adapter registry,
  `DEFAULT_MODEL_OPENAI` set or deleted through `monkeypatch`, an override cache set through
  `set_llm_provider_overrides_cache_for_tests(overrides, healthy=...)` and cleared in `finally`), and 4 tests, 7 cases:
  - `test_own_key_summary_falls_back_to_the_default_model_environment_variable` (AC 1): result `"summary"`, adapter
    saw `["gpt-env-default"]`.
  - `test_own_key_summary_default_model_environment_variable_meets_the_override_policy` (AC 2), 3 cases:
    `model-not-allowed` (`allowed_models=["gpt-allowed"]`), `provider-disabled` (`is_enabled=False`),
    `override-cache-unhealthy` (`healthy=False`). The adapter is never called and the result is
    `Error: Provider credential context is invalid.`.
  - `test_own_key_summary_named_model_beats_the_environment_default_and_skips_the_fallback_policy`, 2 cases: a config
    model `cfg-m` and a `model_override="ovr-m"` each reach the adapter even with an allowlist of `["x"]`
    (control: only the fallback model is newly checked).
  - `test_own_key_summary_with_an_allowlist_and_no_default_model_stays_missing_model`: no env default, allowlist set:
    `Error: Model is required for provider 'openai'`, adapter never called (control; Decision 2).
- [ ] **Step 2: RED on `origin/dev`.**
  `... -p netprobe_15427 -q tldw_Server_API/tests/LLM_Calls/test_provider_adapter_runtime_boundary.py -k own_key_summary`
  Expected: the independent review measured, for 6 new cases (fallback, 3 policy cases, config/override control,
  missing-model control), 4 failed and 2 passed; the 2 passes are the two controls. The failures read
  `Error: Model is required for provider 'openai'` (the planner measured the first two tests failing that way).
  The `test_summarization_adapter.py` pair (Step 4) fails on dev the other way round: the inverted test fails, the
  no-env test passes.
- [ ] **Step 3: the change.** In `_summarize_via_adapter`, in the `elif explicit_credentials:` branch, after the
  mlx lines:

```python
        if not model:
            # Same fallback as the server-key path (DEFAULT_MODEL_<PROVIDER>), but this model was
            # not seen at resolve time, so it meets the provider-override policy here (TASK-15496).
            model = resolve_provider_model(provider, effective_app_config)
            if model and not _model_allowed_by_override(provider, model):
                return _summary_failure(
                    code="configuration",
                    provider=provider,
                    raise_on_error=raise_on_error,
                    legacy_message="Error: Provider credential context is invalid.",
                )
```

  and above `_summarize_via_adapter` the helper `_model_allowed_by_override` (function-local imports of
  `ByokResolutionError` and `capture_provider_override_call_snapshot`; `enforce(model)` in a `try`; return False on
  `ByokResolutionError`). The exact text is in `proto_15496.patch`; apply with `git apply proto_15496.patch`.
- [ ] **Step 4: GREEN, and the test the fix inverts.** Same command as Step 2. Expected: all `own_key_summary`
  cases pass, netprobe log absent or 0 lines. Then, in `tests/LLM_Calls/test_summarization_adapter.py`, replace
  `test_explicit_missing_summary_model_does_not_use_default_model_environment` (it expected `missing_model` with
  `DEFAULT_MODEL_MOONSHOT` set and an empty-config handle; it fails with the fix, measured by the review: 1 failed, 126
  passed). It came from aca6ed7b32 "keep explicit credential policy visible", which kept the env fallback out of a handle
  call because the fallback was unchecked. The new override-policy check removes that reason, so the test now pins the
  new behaviour. The patch has:
  - `test_explicit_missing_summary_model_uses_default_model_environment_under_the_override_policy`: the env model
    reaches the adapter; a model the policy denies (stubbed `_model_allowed_by_override`) raises
    `SummaryProviderError("configuration")` with no second adapter request;
  - `test_explicit_missing_summary_model_without_default_model_environment_is_missing_model`: the old assertion, with
    the env var removed;
  - a shared helper `_analyze_with_empty_moonshot_handle`.
  Expected: both pass.
- [ ] **Step 5: wider set** (Task 0 list, one file per invocation, `-n 0`). Measured by the review with the change
  applied: `test_provider_adapter_runtime_boundary.py` 90 passed (84 existing + 6 new; the patch's 7 cases are one more
  than the review's 6), `test_summarization_runtime_credentials.py` 10, `test_summarization_adapter.py` 126 passed plus
  the 1 failure Step 4 fixes, `test_authoritative_adapter_translation.py` 23, `test_media_ingest_job_llm_quota_refusal.py`
  8, `test_study_assistant_adapter_contract.py` 73, `test_study_assistant_service.py` 7,
  `test_evaluations_benchmarks_api.py` 6, `test_eval_runner.py` 44, `test_output_enrichment_scheduler.py` 5,
  `test_llm_calls_import_isolation.py` 7, `test_function_size_ratchet.py` 13. `tests/Media/test_media_llm_quota.py`
  hung at its first `TestClient` test under load (19 earlier tests passed): run it alone with a long timeout and record
  it, or record it as not measurable.
- [ ] **Step 6: commit** `fix(llm): own-key summaries use the DEFAULT_MODEL_<PROVIDER> fallback under the override policy (TASK-15496)`.

## Task 2: Verification, close-out, review, PR

- [ ] Re-run Task 1 Step 5, plus `tests/Docs`, `tests/lint`, `tests/CI`.
- [ ] `Helper_Scripts/ci/test_quality_triage.py --enforce`; shard coverage (no new test file is added, so no shard
  change); `check_backlog_task_ids.py` exit 0; Bandit on `Summarization_General_Lib.py` with `uvx bandit` if available (the shared
  venv has none), otherwise record it as not run in the task notes (DoD 4); `git diff --check`.
  Known environment-only failures: `test_runtime_package_discovery.py` (no setuptools), 2 jobs-postgres coverage cases
  (no coverage), docling absent.
- [ ] Backlog (backlog-py only): check ACs 1 and 2 and DoD 1-6, `--append-notes` with the measured results, a Final
  Summary, status Done. Stage with `git add -A backlog/tasks`. Docs: none needed (no mirrored doc changes, no new
  config); DoD 3 recorded as not applicable.
- [ ] Clean-session final review (`/superpowers:requesting-code-review`), address findings, then mark the PR ready,
  base `dev`. The merge goes through the controller's queue.

## Measured (planner worktree, base origin/dev 3883a73712; review base 61ad5280ca)

- RED: 2 failed on unmodified source, both `Error: Model is required for provider 'openai'` where the second expected
  the configuration text. GREEN: 2 passed; netprobe log not created (0 lines).
- Independent review (applied before this plan was final): results are in Task 1 Step 5. The review also found
  `tests/CI` 3 failed: the 2 known coverage cases and `test_land_required_gates.py::test_release_helper_requires_success_for_the_current_head[gate_pending]`
  (not in the known list; it tests a script this diff does not touch; take a dev baseline before dismissing it).
  `tests/Docs` 214 passed; `tests/lint` 313 passed.
- Planner's own wider set (Task 0 list, plus the lint files): NOT measured by the planner. Whole-file runs under the heavily loaded
  machine did not finish in 10+ minutes (the two-test `-k own_key_summary` run took 22 s and printed `2 passed`, then
  aborted at interpreter exit with a libc++abi recursive_mutex error). The executor must run each file separately and
  record results in Task 1 Step 5; the shared venv also lacked `pytest_mock`, so the planner used an empty stub module.
  Static review: the helper imports inside the function, so the AuthNZ load ratchet is not changed;
  `_summarize_via_adapter` is not in the function-size baseline.

## Decisions

1. **Failure code for a denied fallback model** (the independent review agrees: reuse `configuration`; the fallback is
   operator config, not user input). Options: (a) reuse `configuration` (recommended, no new public code,
   matches how resolve-time policy denials surface as an invalid credential context); (b) add a `model_not_allowed`
   code to `_SUMMARY_ERROR_CODES` (cost: callers' error maps and docs change). Recommend (a).
2. **An allowlisted provider with no configured model and no env default** (the review agrees: leave it, consistent with
   the workflow LLM adapter, which enforces with `require_model=True` after `resolve_provider_model`). Today it still fails `missing_model`;
   this plan does not guess a model from the allowlist (the chat path's `get_default_model_for_provider` does).
   Recommend leaving it; widening is a separate product choice.
3. ADR-004 Change summary waiver: the controller asks.

## AC corrections

None. The description's line numbers (303-311) are for an older commit; the branch is now around lines 321-330 after
this change.
