# PR 2426 Local CI Review Rebase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebase PR 2426 on latest `dev` and address all open review comments on the local CI runner.

**Architecture:** Keep the runner as a stdlib-first command-line tool while adding narrow tests around argument parsing, changed-file detection, pytest command/env parity, and venv re-exec behavior. Use Loguru only for runner-owned status/error reporting and continue to let subprocesses stream their own output.

**Tech Stack:** Python stdlib, pytest, Loguru, Git, Backlog.md.

---

## Stage 1: Rebase And Review Inventory

**Goal:** Confirm the branch is based on latest `origin/dev` and all current PR review threads are understood.
**Success Criteria:** Rebase reports clean/up-to-date; review inventory covers Gemini and Qodo threads.
**Tests:** `git rebase --autostash origin/dev`, PR review thread query.
**Status:** Complete

- [x] Create Backlog task `TASK-2396` for this PR work.
- [x] Fetch latest `dev` and PR refs.
- [x] Rebase `local-ci-tooling` onto `origin/dev`.
- [x] Collect open review threads and map each to a code/test change.

## Stage 2: Failing Runner Tests

**Goal:** Add focused tests that fail against the reviewed runner behavior.
**Success Criteria:** Tests demonstrate current failures for quoted pytest args, recursive changed-file detection, CI-like pytest env/xdist loading, and Windows re-exec exit propagation.
**Tests:** `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/CI/test_run_local_ci.py -q`
**Status:** Complete

- [x] Add `tldw_Server_API/tests/CI/test_run_local_ci.py` with direct function-level coverage.
- [x] Run the new tests and confirm the behavior tests fail before implementation.

## Stage 3: Runner Fixes

**Goal:** Implement the minimal runner changes required by the review feedback.
**Success Criteria:** The runner preserves quoted pytest args, detects nested changed `.py` files, uses CI-like pytest env defaults, loads xdist explicitly when jobs are enabled, propagates Windows re-exec exit status, uses Loguru for runner-owned messages, and has docstrings for new classes/functions.
**Tests:** New CI runner unit tests plus syntax/import checks.
**Status:** Complete

- [x] Add docstrings to runner dataclasses and functions.
- [x] Replace direct `print()` status/error output with Loguru-backed helper output.
- [x] Extend `_run()` with optional environment overrides.
- [x] Filter git diff output in Python instead of relying on non-recursive `*.py` pathspecs.
- [x] Parse `--pytest-args` with `shlex.split(...)`.
- [x] Add CI-like pytest environment defaults and explicit `xdist.plugin` loading.
- [x] Use `subprocess.call(...)` on Windows venv re-exec and exit with its return code.

## Stage 4: Documentation, Backlog, And Verification

**Goal:** Update tracking/docs as needed and verify the changed scope.
**Success Criteria:** Tests and Bandit pass on touched scope; Backlog task records verification and final summary; PR branch is pushed.
**Tests:** Targeted pytest, `compileall`, Bandit on `Helper_Scripts/ci/run_local_ci.py`.
**Status:** In Progress

- [x] Update Local CI docs only if behavior or quoting guidance changes.
- [x] Run targeted pytest for the new runner tests.
- [x] Run syntax/compile verification for the runner.
- [x] Run Bandit on the touched Python runner.
- [x] Update `TASK-2396` with verification and final summary.
- [x] Commit and push the rebased PR branch.

## Stage 5: Follow-Up CodeRabbit Review

**Goal:** Address the new CodeRabbit comments posted after the first review-fix push.
**Success Criteria:** The pre-push hook uses a cross-platform Python launcher and full local CI runs send `app/` to the syntax guard even when changed files exist.
**Tests:** `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/CI/test_run_local_ci.py -q`
**Status:** Complete

- [x] Add failing tests for `local-ci-fast` launcher portability and full-tier guard syntax scope.
- [x] Change `local-ci-fast` from `python3` to `python`.
- [x] Thread `full` context into `phase_guards`.
- [x] Rerun targeted pytest, Ruff, compileall, Bandit, and local CI smoke verification.
