# Persona Chat Judge Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional/offline calibrated Persona Chat judge contract tied to deterministic quality fixtures without changing runtime chat behavior.

**Architecture:** Add a small core Evaluations helper that defines narrow binary judge dimensions, converts existing fixture records into judge inputs, builds structured Pass/Fail judge prompts, and compares predicted judge outputs against expected fixture labels. Keep execution offline and deterministic in tests; no API endpoint or live LLM call is required for this slice.

**Tech Stack:** Python dataclasses, pytest, existing Persona Chat quality fixtures, existing Evaluations package, Markdown docs.

---

### Task 1: Contract and Calibration Tests

**Files:**
- Create: `tldw_Server_API/tests/Evaluations/test_persona_chat_judge.py`
- Read: `tldw_Server_API/tests/fixtures/persona_chat_quality_cases.json`
- Read: `tldw_Server_API/tests/Persona/persona_chat_quality_cases.py`

- [x] **Step 1: Write failing tests for fixture-derived judge inputs**

Add tests that load existing Persona Chat quality cases and assert that `build_persona_chat_judge_input()` preserves the fixture contract: `case_id`, assistant identity, memory mode, user input, expected context, response observation, labels, and expected evidence.

- [x] **Step 2: Write failing tests for binary judge prompt shape**

Add tests that assert the prompt builder uses one binary dimension at a time, includes Pass/Fail definitions, requires critique before verdict, includes structured JSON output, and does not use Likert/rating language.

- [x] **Step 3: Write failing tests for calibration comparisons**

Add tests with at least one expected-pass and one expected-fail fixture case. Assert that calibration returns confusion counts, TPR/TNR, and a warning when sample size is too small for production calibration.

- [x] **Step 4: Run the focused tests and confirm they fail because the module does not exist**

Run: `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/Evaluations/test_persona_chat_judge.py -v`

Expected: FAIL with an import/module error for the new helper.

### Task 2: Minimal Evaluations Helper

**Files:**
- Create: `tldw_Server_API/app/core/Evaluations/persona_chat_judge.py`
- Modify only if needed: `tldw_Server_API/app/core/Evaluations/__init__.py`

- [x] **Step 1: Implement typed judge data structures**

Add small frozen dataclasses for judge dimensions, judge input, judge prediction, per-dimension calibration metrics, and calibration result. Keep fields explicit and serializable.

- [x] **Step 2: Implement fixture normalization**

Implement `build_persona_chat_judge_input(case)` and `build_persona_chat_judge_inputs(cases)` using defensive copies so tests can mutate returned data without corrupting fixture state.

- [x] **Step 3: Implement judge prompt generation**

Implement `build_persona_chat_judge_prompt(judge_input, dimension_key)` with binary Pass/Fail definitions and JSON output instructions. Do not call an LLM.

- [x] **Step 4: Implement calibration comparison**

Implement `calibrate_persona_chat_judge_predictions(inputs, predictions)` to derive expected failures from fixture labels, compare predictions per dimension, compute TPR/TNR, record missing prediction/unknown dimension errors, and mark results as not production calibrated when sample counts are below the documented threshold.

- [x] **Step 5: Run focused tests and iterate to green**

Run: `source ../../.venv/bin/activate && python -m pytest tldw_Server_API/tests/Evaluations/test_persona_chat_judge.py -v`

Expected: PASS.

### Task 3: Documentation and Verification

**Files:**
- Create: `Docs/Reviews/PERSONA_CHAT_JUDGE_EVALUATION_CONTRACT_2026_05_11.md`
- Modify: `backlog/tasks/task-257 - Add-optional-calibrated-Persona-Chat-judge-evaluation.md`

- [x] **Step 1: Document the contract**

Document the offline judge input, binary dimensions, calibration requirements, minimum-data warning, and non-goals. Explicitly state that deterministic fixtures remain primary and no runtime chat gating changes are introduced.

- [x] **Step 2: Run focused verification**

Run focused pytest for the new tests. Run Bandit on `tldw_Server_API/app/core/Evaluations/persona_chat_judge.py`.

- [x] **Step 3: Inspect diff hygiene**

Run `git diff --check` and `git status --short`.

- [x] **Step 4: Update Backlog task**

Mark acceptance criteria complete only after verification evidence is recorded in `TASK-257`.
