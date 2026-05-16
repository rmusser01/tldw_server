# Persona Chat Judge Calibration Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a trace-safe calibration policy layer for offline Persona Chat judge reports without adding provider execution, persistence, Jobs, API/WebUI state, or runtime chat gating.

**Architecture:** Keep the existing contract fixture, harness, and review CLI as the source of report data. Add one focused policy helper that consumes `PersonaChatJudgeHarnessReport` or its dict form and returns a bounded advisory status with reason keys and safe case/source ids only. Documentation records thresholds, failure modes, and V1 boundaries; runtime Persona Chat remains unchanged.

**Tech Stack:** Python 3.11, dataclasses, existing Persona Chat judge harness, pytest, Bandit.

---

### File Structure

- Create: `tldw_Server_API/app/core/Evaluations/persona_chat_judge_policy.py`
  - Owns calibration policy inputs, status enums, reason keys, trace-safe issue summaries, and report classification.
- Create: `tldw_Server_API/tests/Evaluations/test_persona_chat_judge_policy.py`
  - Covers clean advisory classification, invalid/missing/extra candidates, low agreement, low fixture sample counts, and raw-text exclusion.
- Modify: `Docs/Reviews/PERSONA_CHAT_JUDGE_EVAL_CONTRACT_2026_05_11.md`
  - Adds the calibration policy section and keeps executable adapter work deferred.
- Modify: `Docs/Reviews/PERSONA_CHAT_JUDGE_EVALUATION_CONTRACT_2026_05_11.md`
  - Aligns the earlier contract artifact with the new policy helper and trace-safe output semantics.
- Modify: `Docs/Reviews/PERSONA_CHAT_QUALITY_EVAL_FOLLOWUP_2026_05_10.md`
  - Adds a Stage 2 progress note for the policy slice.
- Modify: `backlog/tasks/task-257.2 - Add-Persona-Chat-judge-calibration-policy.md`
  - Record plan path, implementation notes, verification, residual risk, and final summary.

### Task 1: Policy Helper Contract

**Files:**
- Create: `tldw_Server_API/app/core/Evaluations/persona_chat_judge_policy.py`
- Test: `tldw_Server_API/tests/Evaluations/test_persona_chat_judge_policy.py`

- [ ] **Step 1: Write failing tests for clean advisory policy output**

Add tests that build the packaged fixture expected candidates with `expected_candidate_outputs_from_fixture()`, pass the harness report into a new `evaluate_persona_chat_judge_report_policy()` helper, and assert:

```python
policy.status == "advisory"
policy.production_calibrated is False
"sample_too_small" in policy.reason_keys
policy.runtime_gating_allowed is False
policy.case_issues == ()
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Evaluations/test_persona_chat_judge_policy.py::test_clean_fixture_report_remains_advisory_until_sample_threshold -q
```

Expected: FAIL because `persona_chat_judge_policy` does not exist.

- [ ] **Step 3: Implement minimal policy dataclasses and clean-report classification**

Create:

```python
PolicyStatus = Literal["advisory", "blocked"]

@dataclass(frozen=True)
class PersonaChatJudgePolicyIssue:
    case_id: str
    source_case_id: str
    reason_keys: tuple[str, ...]

@dataclass(frozen=True)
class PersonaChatJudgePolicyResult:
    status: PolicyStatus
    production_calibrated: bool
    runtime_gating_allowed: bool
    reason_keys: tuple[str, ...]
    case_issues: tuple[PersonaChatJudgePolicyIssue, ...]
```

Implement `evaluate_persona_chat_judge_report_policy(report, min_cases_per_verdict=20, min_verdict_agreement=1.0, min_flag_agreement=1.0)` so a clean two-case synthetic report returns `status="advisory"`, `runtime_gating_allowed=False`, `production_calibrated=False`, and `sample_too_small`.

- [ ] **Step 4: Run the clean policy test**

Run the same pytest command.

Expected: PASS.

### Task 2: Invalid, Missing, Extra, And Agreement Reasons

**Files:**
- Modify: `tldw_Server_API/app/core/Evaluations/persona_chat_judge_policy.py`
- Modify: `tldw_Server_API/tests/Evaluations/test_persona_chat_judge_policy.py`

- [ ] **Step 1: Write failing tests for blocked report classes**

Add tests for:

- non-mapping candidate envelope -> `invalid_candidates`
- missing candidate -> `missing_candidates`
- extra candidate id -> `extra_candidates`
- verdict/flag agreement below threshold -> `verdict_agreement_below_threshold` / `flag_agreement_below_threshold`

Assert issue summaries include only `case_id`, `source_case_id`, and reason keys. Include a serialized-output guard:

```python
serialized = json.dumps(policy.to_dict(), sort_keys=True)
assert "I will remember that permanently" not in serialized
assert "Ignore earlier directions" not in serialized
```

- [ ] **Step 2: Run targeted failing tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Evaluations/test_persona_chat_judge_policy.py -q
```

Expected: FAIL for unimplemented reason keys and issue summaries.

- [ ] **Step 3: Implement bounded reason and issue extraction**

Implementation rules:

- `invalid_candidate_count > 0` adds `invalid_candidates`.
- `missing_candidate_count > 0` adds `missing_candidates`.
- `extra_candidate_ids` adds `extra_candidates`.
- `verdict_agreement < min_verdict_agreement` adds `verdict_agreement_below_threshold`.
- `flag_agreement < min_flag_agreement` adds `flag_agreement_below_threshold`.
- Any blocked reason above sets `status="blocked"`.
- `case_issues` is built only from harness case rows whose `status != "matched"` or whose `mismatches` is non-empty.
- `case_issues` never includes rationale, prompt text, assistant text, evidence text, expected context, or candidate payloads.

- [ ] **Step 4: Run policy tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Evaluations/test_persona_chat_judge_policy.py -q
```

Expected: PASS.

### Task 3: Compatibility And Public Exports

**Files:**
- Modify: `tldw_Server_API/app/core/Evaluations/persona_chat_judge_policy.py`
- Modify: `tldw_Server_API/tests/Evaluations/test_persona_chat_judge_policy.py`

- [ ] **Step 1: Add tests for dict report input and stable serialization**

Add tests that pass `build_persona_chat_judge_report(...).to_dict()` into the helper and assert `policy.to_dict()` is JSON serializable with stable keys:

```python
assert set(policy.to_dict()) == {
    "status",
    "production_calibrated",
    "runtime_gating_allowed",
    "reason_keys",
    "case_issues",
}
```

- [ ] **Step 2: Run targeted tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Evaluations/test_persona_chat_judge_policy.py -q
```

Expected: FAIL until dict input and `to_dict()` are implemented.

- [ ] **Step 3: Implement dict compatibility and exports**

Keep parsing conservative:

- Accept `PersonaChatJudgeHarnessReport` or `Mapping[str, Any]`.
- Normalize absent or malformed numeric/count fields to blocked reason `invalid_report`.
- Export public dataclasses and helper in `__all__`.
- Keep module docstring explicit that this is review policy only.

- [ ] **Step 4: Run targeted tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Evaluations/test_persona_chat_judge_policy.py -q
```

Expected: PASS.

### Task 4: Documentation And Backlog

**Files:**
- Modify: `Docs/Reviews/PERSONA_CHAT_JUDGE_EVAL_CONTRACT_2026_05_11.md`
- Modify: `Docs/Reviews/PERSONA_CHAT_JUDGE_EVALUATION_CONTRACT_2026_05_11.md`
- Modify: `Docs/Reviews/PERSONA_CHAT_QUALITY_EVAL_FOLLOWUP_2026_05_10.md`
- Modify: `backlog/tasks/task-257.2 - Add-Persona-Chat-judge-calibration-policy.md`

- [ ] **Step 1: Update docs with policy semantics**

Document:

- policy status values and reason keys
- production sample threshold remains 20 pass and 20 fail cases per verdict class
- current synthetic fixture remains advisory/not production-calibrated
- trace-safe output includes only case ids, source case ids, and reason/mismatch keys
- no runtime gating, DB persistence, Jobs, API endpoint, WebUI state, provider calls, or chat response mutation

- [ ] **Step 2: Update Backlog implementation notes**

Record:

- issue #1586
- plan path
- touched files
- failure modes and residual risks
- verification commands and outcomes as they are run

- [ ] **Step 3: Run docs placeholder scan**

Run:

```bash
rg -n "TO[D]O|TB[D]|FIX[M]E|PLACE[H]OLDER|\\?\\?" Docs/Reviews/PERSONA_CHAT_JUDGE_EVAL_CONTRACT_2026_05_11.md Docs/Reviews/PERSONA_CHAT_JUDGE_EVALUATION_CONTRACT_2026_05_11.md Docs/Reviews/PERSONA_CHAT_QUALITY_EVAL_FOLLOWUP_2026_05_10.md Docs/superpowers/plans/2026-05-12-persona-chat-judge-calibration-policy.md "backlog/tasks/task-257.2 - Add-Persona-Chat-judge-calibration-policy.md"
```

Expected: no matches.

### Task 5: Verification And Commit

**Files:**
- All touched files.

- [ ] **Step 1: Run focused pytest**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Evaluations/test_persona_chat_judge_policy.py tldw_Server_API/tests/Evaluations/test_persona_chat_judge.py tldw_Server_API/tests/Evaluations/test_persona_chat_judge_harness.py tldw_Server_API/tests/Evaluations/test_persona_chat_judge_contract.py tldw_Server_API/tests/Evaluations/unit/test_persona_chat_judge_review_command.py -q
```

Expected: PASS.

- [ ] **Step 2: Run Bandit on touched Python**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Evaluations/persona_chat_judge_policy.py tldw_Server_API/tests/Evaluations/test_persona_chat_judge_policy.py -s B101 -f json -o /tmp/bandit_persona_chat_judge_policy.json
```

Expected: 0 findings.

- [ ] **Step 3: Run diff hygiene**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 4: Update Backlog checkboxes and final summary**

Mark acceptance criteria complete only after verification passes. Add final summary with no runtime behavior changes and residual risk.

- [ ] **Step 5: Commit**

Run:

```bash
git add tldw_Server_API/app/core/Evaluations/persona_chat_judge_policy.py \
  tldw_Server_API/tests/Evaluations/test_persona_chat_judge_policy.py \
  Docs/Reviews/PERSONA_CHAT_JUDGE_EVAL_CONTRACT_2026_05_11.md \
  Docs/Reviews/PERSONA_CHAT_JUDGE_EVALUATION_CONTRACT_2026_05_11.md \
  Docs/Reviews/PERSONA_CHAT_QUALITY_EVAL_FOLLOWUP_2026_05_10.md \
  Docs/superpowers/plans/2026-05-12-persona-chat-judge-calibration-policy.md \
  "backlog/tasks/task-257.2 - Add-Persona-Chat-judge-calibration-policy.md"
git commit -m "Add Persona Chat judge calibration policy"
```

Expected: commit succeeds.
