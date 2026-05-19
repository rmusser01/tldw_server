# Persona Chat Judge Adapter Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an offline Persona Chat judge execution boundary that converts strict JSON judge responses into bounded `PersonaChatJudgePrediction` records without runtime Persona Chat gating.

**Architecture:** Add a small execution module beside the existing Persona Chat judge helpers. The module builds prompts with `build_persona_chat_judge_prompt()`, calls an explicit injected completion callable, parses strict JSON into sanitized predictions, and records bounded failures without persisting or calling providers directly. This feeds `calibrate_persona_chat_judge_predictions()`; the existing harness/review-command/policy path remains the report-review path.

**Tech Stack:** Python dataclasses, existing Persona Chat judge helpers, pytest, Bandit.

---

### Task 1: Define Adapter Success Contract

**Files:**
- Create: `tldw_Server_API/app/core/Evaluations/persona_chat_judge_execution.py`
- Test: `tldw_Server_API/tests/Evaluations/test_persona_chat_judge_execution.py`

- [x] **Step 1: Write the failing valid-response test**

Add a test that builds fixture-derived judge inputs, uses a fake completion callable returning:

```json
{"critique":"uses persona memory mode","result":"Fail","evidence":["persona_memory_mode","assistant_text"]}
```

Assert the executor returns one `PersonaChatJudgePrediction` with the expected case id, dimension key, result, sanitized evidence, and a redacted critique field.

- [x] **Step 2: Run the test and verify RED**

Run:

```bash
python -m pytest tldw_Server_API/tests/Evaluations/test_persona_chat_judge_execution.py::test_execute_persona_chat_judge_collects_valid_prediction -q
```

Expected: import failure for the missing execution module.

- [x] **Step 3: Implement the minimal success path**

Create the execution module with:

- `PersonaChatJudgeExecutionFailure`
- `PersonaChatJudgeExecutionResult`
- `execute_persona_chat_judge()`

The executor should accept explicit inputs, dimension keys, provider/model metadata, and a callable that receives the prompt plus bounded metadata. It must return predictions and failures, not write files or call providers itself.

- [x] **Step 4: Run the success-path test and verify GREEN**

Run the same focused pytest command. Expected: pass.

### Task 2: Fail Closed For Malformed Responses

**Files:**
- Modify: `tldw_Server_API/app/core/Evaluations/persona_chat_judge_execution.py`
- Test: `tldw_Server_API/tests/Evaluations/test_persona_chat_judge_execution.py`

- [x] **Step 1: Write malformed-response tests**

Add tests for malformed JSON, non-object JSON, missing `result`, invalid `result`, unregistered dimension, provider-call exception, and evidence entries that are not allowed field references.

- [x] **Step 2: Run tests and verify RED**

Run:

```bash
python -m pytest tldw_Server_API/tests/Evaluations/test_persona_chat_judge_execution.py -q
```

Expected: failures for missing bounded error handling.

- [x] **Step 3: Implement bounded failures and evidence allowlisting**

Failures must include only `case_id`, `dimension_key`, `provider`, `model`, and an error key such as `malformed_json`, `invalid_response_shape`, `missing_result`, `invalid_result`, `invalid_evidence`, `unknown_dimension`, or `provider_call_failed`. They must not include raw prompts, raw model responses, exception messages, paths, or fixture text.

- [x] **Step 4: Run execution tests and verify GREEN**

Run the execution test file. Expected: all pass.

### Task 3: Integrate With Existing Calibration

**Files:**
- Modify: `tldw_Server_API/tests/Evaluations/test_persona_chat_judge_execution.py`
- Modify: `tldw_Server_API/app/core/Evaluations/persona_chat_judge_execution.py`

- [x] **Step 1: Write calibration integration test**

Add a test that passes executor predictions into `calibrate_persona_chat_judge_predictions()` and asserts calibration metrics are produced without any runtime Persona Chat behavior changes.

- [x] **Step 2: Run tests and verify RED/GREEN**

Run:

```bash
python -m pytest tldw_Server_API/tests/Evaluations/test_persona_chat_judge_execution.py tldw_Server_API/tests/Evaluations/test_persona_chat_judge.py -q
```

Expected after implementation: pass.

- [x] **Step 3: Export public helpers**

Add the execution types/functions to `__all__`. Do not wire them into API endpoints, Jobs, DB persistence, WebUI state, or runtime chat paths.

### Task 4: Docs, Backlog, And Verification

**Files:**
- Modify: `Docs/Reviews/PERSONA_CHAT_JUDGE_EVALUATION_CONTRACT_2026_05_11.md`
- Modify: `Docs/Reviews/PERSONA_CHAT_JUDGE_EVAL_CONTRACT_2026_05_11.md`
- Modify: `backlog/tasks/task-257.3 - Add-Persona-Chat-judge-executable-adapter-boundary.md`

- [x] **Step 1: Document execution boundary**

Document that the executable adapter is offline/explicit, completion-callable driven, privacy preserving, and feeds calibration only. Clarify that the harness/review-command/policy report path remains separate.

- [x] **Step 2: Run focused verification**

Run:

```bash
python -m pytest tldw_Server_API/tests/Evaluations/test_persona_chat_judge_execution.py tldw_Server_API/tests/Evaluations/test_persona_chat_judge.py tldw_Server_API/tests/Evaluations/test_persona_chat_judge_harness.py tldw_Server_API/tests/Evaluations/test_persona_chat_judge_policy.py tldw_Server_API/tests/Evaluations/unit/test_persona_chat_judge_review_command.py -q
python -m bandit -r tldw_Server_API/app/core/Evaluations/persona_chat_judge_execution.py tldw_Server_API/tests/Evaluations/test_persona_chat_judge_execution.py -s B101 -f json -o /tmp/bandit_persona_chat_judge_execution.json
git diff --check
```

Expected: tests pass, Bandit has no new findings, diff hygiene is clean.

- [x] **Step 3: Update Backlog and commit**

Record verification results, known residual risks, and final summary in `TASK-257.3`, then commit the full slice.
