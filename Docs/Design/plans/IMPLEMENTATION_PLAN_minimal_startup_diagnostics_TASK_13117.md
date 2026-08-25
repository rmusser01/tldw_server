# Minimal Startup Diagnostics Implementation Plan

> **For agentic workers:** Execute this plan inline with test-driven development. Multi-agent delegation is not authorized for this task.

**Goal:** Make the minimal single-user setup path non-TTY-safe, respect the configured startup log level, and document the maintained deployment and recovery flows.

**Architecture:** Keep the existing AuthNZ and startup logging boundaries. Handle `EOFError` in the shared prompt helper, normalize `LOG_LEVEL` in the side-effect-free startup logging helper, and wire that value into `main.py`; preserve the existing fail-closed database invariant.

**Tech Stack:** Python 3.11, pytest, Loguru, Markdown, Docker Compose.

**Spec:** Approved bounded design in the Codex task on 2026-08-24; tracked by `TASK-13117`.

## Global Constraints

- Do not deactivate users, revoke keys, or replace databases automatically.
- Do not add dependencies or new configuration keys.
- Use the project virtual environment for Python, pytest, and Bandit.
- Run Bandit on every touched Python source path before completion.

---

## Stage 1: Isolated Baseline and Tracking

**Goal:** Establish an isolated `origin/dev` baseline and required Backlog tracking.

**Success Criteria:** Worktree is on `codex/minimal-startup-diagnostics`; `TASK-13117` is In Progress; focused baseline tests pass.

**Tests:** `python -m pytest -q tldw_Server_API/tests/AuthNZ/unit/test_initialize_mcp_secrets.py tldw_Server_API/tests/Docs/test_onboarding_guides_structure.py tldw_Server_API/tests/Logging/test_loguru_placeholder_style.py`

**Status:** Complete

- [x] Create the worktree from `origin/dev`.
- [x] Search Backlog.md for overlapping work and create `TASK-13117`.
- [x] Run the 33-test baseline and confirm zero failures.

## Stage 2: Non-TTY AuthNZ Prompts

**Goal:** Use each prompt's declared default when stdin is closed.

**Success Criteria:** `_prompt_yes_no()` catches `EOFError`, prints a concise default-selection notice, and returns `default_yes` without changing explicit interactive answers.

**Tests:** Add two unit cases to `tldw_Server_API/tests/AuthNZ/unit/test_initialize_mcp_secrets.py` and run that file.

**Status:** Complete

- [x] Add a failing test proving EOF returns `True` for a yes-default prompt.
- [x] Add a failing test proving EOF returns `False` for a no-default prompt.
- [x] Run both tests and confirm they fail because `EOFError` escapes.
- [x] Add the minimal `except EOFError` branch to `_prompt_yes_no()`.
- [x] Run the AuthNZ initializer unit file and confirm it passes.

## Stage 3: Configurable Startup Log Level

**Goal:** Replace the hard-coded DEBUG sink threshold with validated `LOG_LEVEL` behavior.

**Success Criteria:** Recognized Loguru levels are honored case-insensitively; missing or invalid values use INFO; importing `main.py` with `LOG_LEVEL=WARNING` resolves the sink threshold to WARNING.

**Tests:** Extend `tldw_Server_API/tests/Config/test_startup_api_key_logging.py`, add `tldw_Server_API/tests/Logging/test_main_log_level.py`, and run both files.

**Status:** Complete

- [x] Add failing normalization tests with literal expected values.
- [x] Add a failing subprocess test for the real `main.py` wiring.
- [x] Run the tests and confirm current DEBUG behavior fails the wiring case.
- [x] Add `normalize_startup_log_level(value)` to `startup_logging.py` and use it in `main.py`.
- [x] Run the logging tests and confirm they pass.

## Stage 4: Deployment Guide and Verification

**Goal:** Replace obsolete setup instructions and document observable, reversible recovery.

**Success Criteria:** The guide uses current Make/wizard and Compose paths, distinguishes one-time setup from server start, shows log capture, and explains backup-first invariant recovery without automatic cleanup.

**Tests:** Add a guide contract test to `tldw_Server_API/tests/Docs/test_onboarding_guides_structure.py`; run focused tests, `git diff --check`, Bandit, and the relevant combined suites.

**Status:** In Progress

- [x] Add the failing guide contract test.
- [x] Rewrite `Docs/Deployment/minimal-deploy.md` around the supported local and Docker profiles.
- [x] Run focused AuthNZ, Config, Logging, and Docs tests (43 passed).
- [x] Run Bandit on `initialize.py`, `startup_logging.py`, and `main.py` (0 findings).
- [ ] Review the diff, update `TASK-13117`, remove this completed plan file, commit, push, and open a draft PR against `dev`.
