# Persona Chat Judge Review Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an offline Persona Chat judge review command that compares already-produced candidate judge outputs with the checked-in V1 contract fixture and emits a bounded JSON report.

**Architecture:** Add a small command group to the existing unified `tldw-evals` CLI instead of creating a parallel executable. Keep the command as a thin file/JSON adapter over `build_persona_chat_judge_report()`, with no provider/model imports, no database persistence, no Jobs worker, no API endpoint, and no runtime Persona Chat gating.

**Tech Stack:** Python 3.11, Click, pytest `CliRunner`, existing Persona Chat judge harness and contract fixture.

---

### Task 1: CLI Tests

**Files:**
- Create: `tldw_Server_API/tests/Evaluations/unit/test_persona_chat_judge_review_command.py`
- Read: `tldw_Server_API/tests/fixtures/persona_chat_judge_contract_cases.json`
- Read: `tldw_Server_API/app/core/Evaluations/persona_chat_judge_harness.py`

- [x] **Step 1: Write failing tests for the review command**

Add tests that exercise the public unified CLI via `tldw_Server_API.cli.evals_cli.main` and verify:
- `persona-chat-judge review --candidates <path>` prints a JSON report with `offline_only: true`, total case counts, agreement metrics, and no prompt/assistant text.
- `--output <path>` writes the same JSON report to an explicit file.
- Missing candidate file paths fail cleanly.
- Malformed candidate JSON fails cleanly.
- Non-object candidate JSON roots fail cleanly before calling the harness.

- [x] **Step 2: Run tests to verify RED**

Run:
`/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Evaluations/unit/test_persona_chat_judge_review_command.py -q`

Expected: fail because the `persona-chat-judge` command group does not exist.

### Task 2: CLI Command Group

**Files:**
- Create: `tldw_Server_API/app/core/Evaluations/cli/persona_chat_judge_cli.py`
- Modify: `tldw_Server_API/cli/evals_cli.py`
- Test: `tldw_Server_API/tests/Evaluations/unit/test_persona_chat_judge_review_command.py`

- [x] **Step 1: Add a focused command module**

Create a module-level docstring explaining this is an offline review utility for Persona Chat judge harness reports. Define:
- `PERSONA_CHAT_JUDGE_FIXTURE_PACKAGE`
- `PERSONA_CHAT_JUDGE_FIXTURE_RESOURCE`
- `persona_chat_judge_group`
- `review_persona_chat_judge_candidates`

The command should accept:
- required `--candidates` path
- optional `--fixture` path overriding the packaged V1 contract fixture
- optional `--output` path

- [x] **Step 2: Implement JSON loading and validation**

Load fixture and candidate files with explicit UTF-8 decoding. Convert JSON errors and non-object roots into `click.ClickException` messages. Require the candidate payload root to be a JSON object so malformed envelopes are rejected before report construction.

- [x] **Step 3: Emit bounded report JSON**

Call `build_persona_chat_judge_report(fixture_payload, candidate_payload).to_dict()`. Serialize with stable indentation and sorted keys. Print to stdout, and when `--output` is provided, write the same JSON plus trailing newline to that explicit path.

- [x] **Step 4: Register the command group**

Add the command group to `tldw_Server_API/cli/evals_cli.py` as `persona-chat-judge`.

- [x] **Step 5: Run tests to verify GREEN**

Run:
`/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Evaluations/unit/test_persona_chat_judge_review_command.py -q`

Expected: pass.

### Task 3: Documentation And Tracker Updates

**Files:**
- Modify: `Docs/Reviews/PERSONA_CHAT_JUDGE_EVAL_CONTRACT_2026_05_11.md`
- Modify: `backlog/tasks/task-257.1 - Add-offline-Persona-Chat-judge-review-command.md`

- [x] **Step 1: Document command usage and boundaries**

Add a short review-command section with an example command and state that the output is explicit file-based offline persistence only. Repeat non-goals: no providers, DB persistence, Jobs, endpoint, WebUI, runtime gating, or response mutation.

- [x] **Step 2: Update Backlog progress**

Record implementation notes and check acceptance criteria as they are satisfied.

### Task 4: Verification And Packaging

**Files:**
- Touch all changed files from Tasks 1-3.

- [x] **Step 1: Run closeout checks**

Run:
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Evaluations/unit/test_persona_chat_judge_review_command.py tldw_Server_API/tests/Evaluations/test_persona_chat_judge_harness.py tldw_Server_API/tests/Evaluations/test_persona_chat_judge_contract.py -q`
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Evaluations/cli/persona_chat_judge_cli.py tldw_Server_API/tests/Evaluations/unit/test_persona_chat_judge_review_command.py -f json -o /tmp/bandit_persona_chat_judge_review_command.json`
- `rg -n "TO[D]O|TB[D]|FIX[M]E|PLACE[H]OLDER|\\?\\?" Docs/Reviews/PERSONA_CHAT_JUDGE_EVAL_CONTRACT_2026_05_11.md Docs/superpowers/plans/2026-05-12-persona-chat-judge-review-command.md tldw_Server_API/app/core/Evaluations/cli/persona_chat_judge_cli.py tldw_Server_API/tests/Evaluations/unit/test_persona_chat_judge_review_command.py`
- `git diff --check`

- [x] **Step 2: Commit and open PR**

Commit the task, plan, CLI, tests, and docs together. Open a PR against `dev` and link GitHub issue #1579 plus parent trackers #1566, #1543, and #1510.
