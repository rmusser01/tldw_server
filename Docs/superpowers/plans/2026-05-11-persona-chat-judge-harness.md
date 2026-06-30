# Persona Chat Judge Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic offline Persona Chat judge harness that replays the V1 contract fixture and reports calibration mismatches without provider calls or runtime Persona Chat changes.

**Architecture:** Keep the harness as a pure core helper under `app/core/Evaluations`, with no database, Jobs worker, endpoint, or model dependency. The helper accepts already-produced candidate judge outputs keyed by `PC-JUDGE-###`, validates the output envelope, compares verdict and flags against the fixture expectations, and returns a bounded report safe for docs or future CLI/API layers.

**Tech Stack:** Python 3.11, dataclasses, pytest, existing Persona Chat judge contract fixture.

---

### Task 1: Harness Tests

**Files:**
- Create: `tldw_Server_API/tests/Evaluations/test_persona_chat_judge_harness.py`
- Read: `tldw_Server_API/tests/fixtures/persona_chat_judge_contract_cases.json`

- [x] **Step 1: Write failing tests for the offline harness**

Add tests that import future helpers from `tldw_Server_API.app.core.Evaluations.persona_chat_judge_harness` and verify:
- `expected_candidate_outputs_from_fixture()` extracts candidate outputs from the existing contract fixture.
- `build_persona_chat_judge_report()` returns full agreement when candidates match expected outputs.
- A verdict mismatch produces a bounded mismatch entry and does not copy assistant response text into the report.
- A flag mismatch is counted separately from verdict agreement.
- Missing score axes and malformed labels produce invalid candidate results.

- [x] **Step 2: Run tests to verify RED**

Run:
`/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Evaluations/test_persona_chat_judge_harness.py -q`

Expected: fail because `persona_chat_judge_harness.py` does not exist.

### Task 2: Core Harness Helper

**Files:**
- Create: `tldw_Server_API/app/core/Evaluations/persona_chat_judge_harness.py`
- Test: `tldw_Server_API/tests/Evaluations/test_persona_chat_judge_harness.py`

- [x] **Step 1: Add module constants and dataclasses**

Define:
- `REQUIRED_SCORE_NAMES`
- `ALLOWED_VERDICTS`
- `PersonaChatJudgeCaseResult`
- `PersonaChatJudgeHarnessReport`

The dataclasses should expose `to_dict()` methods so future API/CLI layers can reuse the report shape without coupling to dataclass internals.

- [x] **Step 2: Add candidate validation helpers**

Implement strict envelope validation:
- verdict is one of `pass`, `fail`, `inconclusive`
- `expected_flags` is a list of unique `PC-*` labels
- scores contain exactly all required score axes
- score values are `None` or numeric `int`/`float`, excluding `bool`, in `[0.0, 1.0]`

- [x] **Step 3: Add report builder**

Implement:
- `expected_candidate_outputs_from_fixture(fixture_payload)`
- `build_persona_chat_judge_report(fixture_payload, candidate_outputs_by_case_id)`

Report fields should include:
- `schema_version`
- `offline_only`
- `total_cases`
- `matched_cases`
- `mismatched_cases`
- `missing_candidate_count`
- `invalid_candidate_count`
- `extra_candidate_ids`
- `verdict_agreement`
- `flag_agreement`
- per-case bounded mismatch data

- [x] **Step 4: Run tests to verify GREEN**

Run:
`/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Evaluations/test_persona_chat_judge_harness.py -q`

Expected: pass.

### Task 3: Docs And Tracker Updates

**Files:**
- Modify: `Docs/Reviews/PERSONA_CHAT_JUDGE_EVAL_CONTRACT_2026_05_11.md`
- Modify: `backlog/tasks/task-241.1.1 - Add-offline-Persona-Chat-judge-harness.md`

- [x] **Step 1: Document the offline harness boundary**

Add a short section to the judge contract explaining that the offline harness:
- compares already-produced candidate judge outputs to contract fixtures
- does not call providers
- does not persist evaluation runs
- does not gate Persona Chat responses
- is intended as the calibration report substrate for later judge adapters

- [x] **Step 2: Update Backlog with verification evidence**

Record focused pytest, Bandit, placeholder scan, and diff hygiene results after verification.

### Task 4: Verification And Packaging

**Files:**
- Touch all changed files from Tasks 1-3

- [x] **Step 1: Run closeout checks**

Run:
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Evaluations/test_persona_chat_judge_harness.py tldw_Server_API/tests/Evaluations/test_persona_chat_judge_contract.py -q`
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Evaluations/persona_chat_judge_harness.py tldw_Server_API/tests/Evaluations/test_persona_chat_judge_harness.py -f json -o /tmp/bandit_persona_chat_judge_harness.json`
- `rg -n "TO[D]O|TB[D]|FIX[M]E|PLACE[H]OLDER|\\?\\?" Docs/Reviews/PERSONA_CHAT_JUDGE_EVAL_CONTRACT_2026_05_11.md Docs/superpowers/plans/2026-05-11-persona-chat-judge-harness.md`
- `git diff --check`

- [x] **Step 2: Commit and open PR**

Run:
```bash
git add Docs/Reviews/PERSONA_CHAT_JUDGE_EVAL_CONTRACT_2026_05_11.md Docs/superpowers/plans/2026-05-11-persona-chat-judge-harness.md tldw_Server_API/app/core/Evaluations/persona_chat_judge_harness.py tldw_Server_API/tests/Evaluations/test_persona_chat_judge_harness.py "backlog/tasks/task-241.1.1 - Add-offline-Persona-Chat-judge-harness.md"
git commit -m "Add offline Persona Chat judge harness"
```

Open a PR against `dev` referencing #1572 and #1566.
