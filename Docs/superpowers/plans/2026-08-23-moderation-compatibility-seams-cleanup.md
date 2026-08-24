# Moderation Compatibility Seams Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove eight repository-unused private `ModerationService` evaluator shims while preserving public behavior, active dynamic dispatch, `policy_types()` compatibility, and all moderation semantics.

**Architecture:** Keep `ModerationService` as the supported facade and `PolicyEvaluator` as the canonical owner of pure evaluation helpers. Delete only direct private evaluator forwarding methods proven to have no production call sites; retain `_evaluate_text_core()`, `_evaluate_action_internal()`, public methods, compiler delegates, immutable limit snapshots, and evaluator implementation unchanged.

**Tech Stack:** Python 3, pytest, standard-library `inspect`, Black, Ruff, Bandit, Git, Backlog.md MCP, and the project virtual environment at `/Users/appledev/Documents/GitHub/tldw_server/.venv`.

---

## Approved Inputs

- Design: `Docs/superpowers/specs/2026-08-23-moderation-compatibility-seams-cleanup-design.md`
- Backlog: `TASK-13112`
- Initial base: `origin/dev` at `2c3589fa09`
- First verification base after initial rebase (historical): `origin/dev` at
  `83fa300fc1b0e77e81219af2abe5b4ddc2c85069`
- Task 3 pre-tracking HEAD (historical):
  `83f489876e7828e7e1fbd42cd043f6d6ed95b57d`
- Final verification base after second rebase: `origin/dev` at
  `21aed4cc0d1e9e2e2a34fc84307bbd1d3b879871`
- Task 4 pre-tracking HEAD: `8b51d70c642a3f597ccd5c35726296bbd5ba9529`
- Worktree: `.worktrees/moderation-compatibility-seams`
- Branch: `codex/moderation-compatibility-seams`

## Non-Negotiable Boundaries

- Remove exactly these eight `ModerationService` methods:
  - `_effective_rule_categories()`
  - `_rule_applies_to_phase()`
  - `_rule_matches_enabled_categories()`
  - `_build_sanitized_snippet()`
  - `_iter_scan_chunks()`
  - `_find_match_span()`
  - `_collect_rule_matches()`
  - `_apply_rule_redactions()`
- Retain `_evaluate_text_core()` and `_evaluate_action_internal()` unchanged.
- Retain all public `ModerationService` methods and signatures unchanged.
- Retain `PolicyCompiler.policy_types()` and `PolicyEvaluator.policy_types()` unchanged.
- Do not modify `policy_evaluator.py`, regex execution, rule ordering, phase or
  category behavior, scan geometry, limits, replacement behavior, persistence,
  logging, or endpoint schemas.
- Treat removal as an intentional undocumented private callable-surface break,
  not as a public API migration.

## File Map

- Modify: `tldw_Server_API/app/core/Moderation/moderation_service.py`
  - Delete eight direct private evaluator shims only.
- Modify: `tldw_Server_API/tests/unit/test_moderation_policy_evaluator_delegation.py`
  - Add class-local absence invariants.
  - Delete tests for removed forwarding methods.
  - Keep public delegation, limit snapshot, signature, and active private
    dispatch coverage.
- Modify: `tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py`
  - Delete service-private scan characterization duplicated by direct evaluator
    tests.
  - Keep all public service behavior and dynamic-dispatch characterization.
- Modify: `tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py`
  - Preserve the exact numeric-string scan-limit geometry formerly asserted by
    the deleted service-private characterization.
- Modify: `Docs/superpowers/specs/2026-08-23-moderation-compatibility-seams-cleanup-design.md`
  - Record approved status only.
- Modify: `backlog/tasks/task-13112 - Clean-up-Moderation-evaluator-compatibility-seams.md`
  - Record execution, verification, review, and final summary through Backlog.md
    MCP operations.
- Do not modify: `tldw_Server_API/app/core/Moderation/policy_evaluator.py`
- Do not modify: caller, endpoint, Workflow, Chat, Guardian, or Audio source and
  test files; they are verification-only gates.

## Stages

### Stage 1: Remove Stateless Policy Helper Shims

**Goal:** Remove five class/static direct evaluator delegates and their obsolete wrapper tests.

**Success Criteria:** The five names are absent from `ModerationService.__dict__`; retained public and active private methods keep their signatures and dispatch.

**Tests:** Focused red/green absence test plus the evaluator delegation and direct evaluator suites.

**Status:** Complete

### Stage 2: Remove Instance Scan Helper Shims

**Goal:** Remove three instance scan delegates and service-private characterization already covered on `PolicyEvaluator`.

**Success Criteria:** The three names are absent from `ModerationService.__dict__`; direct scan and public service behavior tests pass.

**Tests:** Focused red/green absence test plus evaluator, characterization, and delegation suites.

**Status:** Complete

### Stage 3: Stability And Security Verification

**Goal:** Prove the structural cleanup preserved supported runtime behavior and introduced no security or formatting regression.

**Success Criteria:** Compilation, formatting, lint, full Moderation, Guardian, Chat, Workflow, Audio, Bandit, scope, and whitespace gates pass.

**Tests:** Exact pre-change matrix repeated after implementation.

**Status:** Complete

### Stage 4: Independent Review And PR Readiness

**Goal:** Resolve independent review findings, finalize tracking, and leave a clean reviewable branch.

**Success Criteria:** Spec and quality reviews approve, task evidence is complete, and the branch is ready for PR preparation without uncommitted files.

**Tests:** Rerun every affected gate after any review fix and rerun the complete final matrix before readiness is claimed.

**Status:** Complete

---

### Task 1: Remove Stateless Policy Helper Delegates

**Files:**
- Modify: `tldw_Server_API/tests/unit/test_moderation_policy_evaluator_delegation.py`
- Modify: `tldw_Server_API/app/core/Moderation/moderation_service.py`
- Modify through Backlog.md MCP: `backlog/tasks/task-13112 - Clean-up-Moderation-evaluator-compatibility-seams.md`

- [x] **Step 1: Add the failing class-local absence invariant**

Add this constant near the delegation test helpers and add the test after the
constructor ownership test:

```python
_OBSOLETE_POLICY_HELPER_DELEGATES = (
    "_effective_rule_categories",
    "_rule_applies_to_phase",
    "_rule_matches_enabled_categories",
    "_build_sanitized_snippet",
    "_apply_rule_redactions",
)


def test_obsolete_policy_helper_delegates_are_not_class_local():
    for name in _OBSOLETE_POLICY_HELPER_DELEGATES:
        assert name not in ModerationService.__dict__
```

Do not remove production methods or old delegation tests yet. The first run must
demonstrate that the new surface contract fails against the baseline.

- [x] **Step 2: Run the focused test and verify RED**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_delegation.py::test_obsolete_policy_helper_delegates_are_not_class_local \
  -q
```

Expected: FAIL because `_effective_rule_categories` is present in
`ModerationService.__dict__`.

- [x] **Step 3: Delete the five direct production delegates**

Delete these complete method blocks from `ModerationService` and make no other
production changes:

```python
@classmethod
def _effective_rule_categories(cls, rule: PatternRule) -> set[str]:
    return PolicyEvaluator.effective_rule_categories(rule)

@staticmethod
def _rule_applies_to_phase(
    rule: PatternRule,
    phase: str | None,
) -> bool:
    return PolicyEvaluator.rule_applies_to_phase(rule, phase)

@classmethod
def _rule_matches_enabled_categories(
    cls,
    rule: PatternRule,
    categories_enabled: set[str] | None,
) -> bool:
    return PolicyEvaluator.rule_matches_enabled_categories(
        rule,
        categories_enabled,
    )

@staticmethod
def _build_sanitized_snippet(
    text: str,
    match_span: tuple[int, int],
    replacement: str,
) -> str | None:
    return PolicyEvaluator.build_sanitized_snippet_for_replacement(
        text,
        match_span,
        replacement,
    )

@staticmethod
def _apply_rule_redactions(
    text: str,
    matches: list[re.Match],
    replacement: str,
) -> str:
    return PolicyEvaluator.apply_rule_redactions(
        text,
        matches,
        replacement,
    )
```

Keep the public `build_sanitized_snippet()` implementation unchanged. Do not
remove the `re` import; `moderation_service.py` still compiles regular
expressions in other paths.

- [x] **Step 4: Compile the production module immediately**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m py_compile \
  tldw_Server_API/app/core/Moderation/moderation_service.py
```

Expected: exit 0 with no output. Stop and correct compilation before editing
the wrapper-specific tests.

- [x] **Step 5: Delete obsolete wrapper-only delegation assertions**

Delete these complete tests from
`test_moderation_policy_evaluator_delegation.py`:

```text
test_effective_rule_categories_delegates_exactly_once
test_rule_applies_to_phase_delegates_exactly_once
test_rule_matches_enabled_categories_delegates_exactly_once
test_build_sanitized_snippet_for_replacement_delegates_exactly_once
test_apply_rule_redactions_delegates_exactly_once
test_private_helper_descriptors_are_preserved
```

In `test_service_method_parameter_names_and_kinds_are_preserved()`, remove only
this obsolete entry from `expected`:

```python
"_apply_rule_redactions": ("text", "matches", "replacement"),
```

Keep the public `test_build_sanitized_snippet_delegates_exactly_once()` test and
all retained signature checks.

- [x] **Step 6: Run the focused test and verify GREEN**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_delegation.py::test_obsolete_policy_helper_delegates_are_not_class_local \
  -q
```

Expected: 1 passed.

- [x] **Step 7: Run the affected evaluator suites**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_delegation.py \
  -q
```

Expected: PASS with zero failures.

- [x] **Step 8: Record Task 1 evidence in Backlog.md**

Use `backlog.task_edit` for `TASK-13112` and append notes containing:

```text
Task 1 removed the five class/static direct PolicyEvaluator shims. The new
ModerationService.__dict__ absence test failed before deletion and passed after
deletion. Direct evaluator and service delegation suites passed.
```

- [x] **Step 9: Commit Task 1**

```bash
git add \
  tldw_Server_API/app/core/Moderation/moderation_service.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_delegation.py \
  'backlog/tasks/task-13112 - Clean-up-Moderation-evaluator-compatibility-seams.md'
git commit -m "refactor(moderation): remove obsolete policy helper shims"
```

Expected: one commit containing only Task 1 production, tests, and tracking.

### Task 2: Remove Instance Scan Helper Delegates

**Files:**
- Modify: `tldw_Server_API/tests/unit/test_moderation_policy_evaluator_delegation.py`
- Modify: `tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py`
- Modify: `tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py`
- Modify: `tldw_Server_API/app/core/Moderation/moderation_service.py`
- Modify through Backlog.md MCP: `backlog/tasks/task-13112 - Clean-up-Moderation-evaluator-compatibility-seams.md`

- [x] **Step 1: Add the failing class-local scan absence invariant**

Add this constant beside the Task 1 obsolete-name tuple and add the test beside
the Task 1 absence test:

```python
_OBSOLETE_SCAN_HELPER_DELEGATES = (
    "_iter_scan_chunks",
    "_find_match_span",
    "_collect_rule_matches",
)


def test_obsolete_scan_helper_delegates_are_not_class_local():
    for name in _OBSOLETE_SCAN_HELPER_DELEGATES:
        assert name not in ModerationService.__dict__
```

- [x] **Step 2: Run the focused test and verify RED**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_delegation.py::test_obsolete_scan_helper_delegates_are_not_class_local \
  -q
```

Expected: FAIL because `_iter_scan_chunks` is present in
`ModerationService.__dict__`.

- [x] **Step 3: Delete the three instance scan delegates**

Delete these complete method blocks from `ModerationService`:

```python
def _iter_scan_chunks(
    self,
    text: str,
) -> Iterator[tuple[int, int]]:
    yield from self._policy_evaluator.iter_scan_chunks(
        text,
        self._evaluation_limits(),
    )

def _find_match_span(
    self,
    pat: re.Pattern,
    text: str,
) -> tuple[int, int] | None:
    return self._policy_evaluator.find_match_span(
        pat,
        text,
        self._evaluation_limits(),
    )

def _collect_rule_matches(
    self,
    text: str,
    pat: re.Pattern,
) -> list[re.Match]:
    """Collect non-overlapping matches across scan chunks for soft-capped redaction."""
    return self._policy_evaluator.collect_rule_matches(
        text,
        pat,
        self._evaluation_limits(),
    )
```

Keep `Iterator` imported because blocklist line readers still use it. Keep
`_evaluation_limits()` unchanged because public evaluator delegation requires a
fresh locked snapshot.

- [x] **Step 4: Compile the production module immediately**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m py_compile \
  tldw_Server_API/app/core/Moderation/moderation_service.py
```

Expected: exit 0 with no output. Stop and correct compilation before removing
wrapper-specific tests.

- [x] **Step 5: Delete obsolete scan delegation tests and signature entries**

Delete these complete tests from
`test_moderation_policy_evaluator_delegation.py`:

```text
test_iter_scan_chunks_delegates_exactly_once_with_one_snapshot
test_find_match_span_delegates_exactly_once_with_one_snapshot
test_collect_rule_matches_delegates_exactly_once_with_one_snapshot
```

Remove only these obsolete entries from the `expected` mapping in
`test_service_method_parameter_names_and_kinds_are_preserved()`:

```python
"_iter_scan_chunks": ("self", "text"),
"_find_match_span": ("self", "pat", "text"),
"_collect_rule_matches": ("self", "text", "pat"),
```

Do not remove `_evaluate_text_core` or `_evaluate_action_internal` from the
signature mapping or its required-phase loop.

- [x] **Step 6: Delete duplicated service-private scan characterization**

Delete `_RecordingPattern` and these complete tests from
`test_moderation_policy_evaluator_characterization.py`:

```text
test_chunk_geometry_and_search_bounds_are_literal
test_original_string_search_preserves_lookbehind_before_pos
test_chunk_search_does_not_turn_mid_text_anchor_into_start_anchor
test_full_text_fallback_limit_is_inclusive_and_guarded
test_evaluation_max_scan_coercion_errors_are_literal
test_numeric_string_max_scan_is_coerced_for_evaluation
test_long_evaluation_limit_coercion_errors_are_literal
test_numeric_string_window_and_fallback_limits_are_coerced
```

These behaviors remain directly covered in
`test_moderation_policy_evaluator.py` by:

```text
test_direct_scan_geometry_matches_characterized_behavior
test_direct_original_string_search_bounds_are_literal
test_direct_lookbehind_anchor_and_fallback_behavior_is_literal
test_direct_max_scan_coercion_errors_are_literal
test_direct_long_limit_coercion_errors_are_literal
test_direct_numeric_string_limits_are_coerced_for_evaluation
```

Keep `test_redaction_path_comparison_does_not_coerce_max_scan()` and all public
redaction tests. Keep the `re` import if any remaining characterization test
uses it; run Ruff to determine whether import cleanup is required rather than
guessing.

Quality review identified that the deleted numeric-string characterization used
`max_scan_chars="2"` and asserted exact chunk geometry, while the direct test
used `"1"` and only asserted the evaluation result. Preserve that behavior in
`test_direct_numeric_string_limits_are_coerced_for_evaluation()` with:

```python
limits = EvaluationLimits(max_scan_chars="2")
evaluator = PolicyEvaluator()
assert list(evaluator.iter_scan_chunks("xx", limits)) == [(0, 2)]
```

Keep the existing no-match evaluation assertion with the same evaluator and
limits.

- [x] **Step 7: Run the focused test and verify GREEN**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_delegation.py::test_obsolete_scan_helper_delegates_are_not_class_local \
  -q
```

Expected: 1 passed.

- [x] **Step 8: Run all evaluator boundary suites**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_delegation.py \
  -q
```

Expected: PASS with zero failures. A lower collected-test count is expected only
because wrapper-specific duplicate tests were deliberately deleted.

- [x] **Step 9: Prove retained dispatch points remain covered**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py::test_check_and_evaluate_dispatch_through_evaluate_text_core \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py::test_action_wrappers_dispatch_through_public_evaluate_text \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py::test_evaluation_dispatches_through_public_redact_text \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_delegation.py::test_service_method_parameter_names_and_kinds_are_preserved \
  tldw_Server_API/tests/unit/test_moderation_models_characterization.py::test_policy_type_descriptors_and_tuples_are_literal
```

Expected: 5 passed.

- [x] **Step 10: Record Task 2 evidence in Backlog.md**

Use `backlog.task_edit` for `TASK-13112` and append notes containing:

```text
Task 2 removed the three instance scan shims and wrapper-only duplicate
characterization. The class-local absence test completed a red/green cycle.
Direct evaluator, public service dispatch, signature, and policy_types()
compatibility tests passed.
```

- [x] **Step 11: Commit Task 2**

```bash
git add \
  tldw_Server_API/app/core/Moderation/moderation_service.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_delegation.py \
  'backlog/tasks/task-13112 - Clean-up-Moderation-evaluator-compatibility-seams.md'
git commit -m "refactor(moderation): remove obsolete scan helper shims"
```

Expected: one commit containing only Task 2 production, tests, and tracking.

### Task 3: Run Stability, Quality, And Security Gates

**Files:**
- Verify: `tldw_Server_API/app/core/Moderation/moderation_service.py`
- Verify: `tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py`
- Verify: `tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py`
- Verify: `tldw_Server_API/tests/unit/test_moderation_policy_evaluator_delegation.py`
- Modify through Backlog.md MCP: `backlog/tasks/task-13112 - Clean-up-Moderation-evaluator-compatibility-seams.md`

- [x] **Step 1: Verify production method scope exactly**

Run:

```bash
rg -n '^    def (_effective_rule_categories|_rule_applies_to_phase|_rule_matches_enabled_categories|_build_sanitized_snippet|_iter_scan_chunks|_find_match_span|_collect_rule_matches|_apply_rule_redactions)\b' \
  tldw_Server_API/app/core/Moderation/moderation_service.py
```

Expected: no matches and exit status 1.

Run:

```bash
rg -n '^    def (_evaluate_text_core|_evaluate_action_internal)\b' \
  tldw_Server_API/app/core/Moderation/moderation_service.py
```

Expected: exactly two matches, one for each retained private dispatch method.

- [x] **Step 2: Compile touched Python files**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m py_compile \
  tldw_Server_API/app/core/Moderation/moderation_service.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_delegation.py
```

Expected: exit 0 with no output.

- [x] **Step 3: Run formatting and lint checks**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m black --check \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_delegation.py
python -m ruff check \
  tldw_Server_API/app/core/Moderation/moderation_service.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_delegation.py
```

Expected: Black reports the three changed test files unchanged and Ruff reports
`All checks passed!`. Run Black against `moderation_service.py` separately. If
it still reports that the production file would be reformatted, compare the
same check against the `origin/dev` version and document the pre-existing
parity; do not mass-format the production module in this structural cleanup. If
Ruff identifies an import made unused solely by the approved deletion, remove
that import, rerun compilation, then rerun this step.

- [x] **Step 4: Run the complete Moderation unit gate**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest tldw_Server_API/tests/unit/test_moderation*.py -q
```

Expected: PASS with zero failures. Compare with the pre-change baseline of 318
tests, accounting only for explicitly deleted wrapper-specific tests and the two
new absence tests.

- [x] **Step 5: Run Guardian, Chat, Workflow, And Audio caller gates**

Run each command independently:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Guardian/test_supervised_policy.py \
  -q
```

Expected: 89 passed.

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py \
  -q
```

Expected: 16 passed.

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Workflows/adapters/test_llm_adapters.py \
  -k moderation_adapter \
  -q
```

Expected: 12 passed, 45 deselected.

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Audio/test_audio_transcription_retention_and_redaction.py::test_audio_transcriptions_redacts_text_and_segments_when_stt_redaction_enabled \
  -q
```

Expected: 1 passed.

- [x] **Step 6: Run Bandit on touched production scope**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m bandit \
  -r tldw_Server_API/app/core/Moderation/moderation_service.py \
  -f json \
  -o /tmp/bandit_task_13112.json
```

Expected: exit 0 and no new findings. Read the JSON summary; do not infer a
clean result from file creation alone.

- [x] **Step 7: Run whitespace and scope checks**

Run:

```bash
git diff --check origin/dev...HEAD
git diff --name-only origin/dev...HEAD
git status --short
```

Expected:

```text
Docs/superpowers/plans/2026-08-23-moderation-compatibility-seams-cleanup.md
Docs/superpowers/specs/2026-08-23-moderation-compatibility-seams-cleanup-design.md
backlog/tasks/task-13112 - Clean-up-Moderation-evaluator-compatibility-seams.md
tldw_Server_API/app/core/Moderation/moderation_service.py
tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py
tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py
tldw_Server_API/tests/unit/test_moderation_policy_evaluator_delegation.py
```

`git diff --check` must emit no output. `git status --short` may show only the
Backlog task update produced while recording these gates; no generated database,
cache, report, or unrelated file may be staged.

- [x] **Step 8: Record verification in Backlog.md**

Use `backlog.task_edit` for `TASK-13112`. Append exact command results, Bandit
summary, test counts, and any warnings or skips. Do not mark acceptance criteria
or Definition of Done items complete until the corresponding evidence exists.

- [x] **Step 9: Commit verification tracking**

```bash
git add 'backlog/tasks/task-13112 - Clean-up-Moderation-evaluator-compatibility-seams.md'
git commit -m "chore(backlog): record moderation cleanup verification"
```

Expected: a tracking-only commit after every technical gate passes.

### Task 4: Independent Review And Final Readiness

**Files:**
- Review all files in `git diff origin/dev...HEAD`
- Modify only files required to resolve validated findings
- Modify through Backlog.md MCP: `backlog/tasks/task-13112 - Clean-up-Moderation-evaluator-compatibility-seams.md`

- [x] **Step 1: Request spec compliance review**

Provide an independent reviewer with the approved design, this plan, base SHA,
head SHA, and exact eight-method list. Require a finding-first response that
checks:

```text
- exactly eight approved private evaluator shims are absent
- _evaluate_text_core and _evaluate_action_internal remain unchanged
- public method signatures and dynamic dispatch remain unchanged
- policy_types hooks remain unchanged
- no evaluator semantics or caller files changed
- wrapper-only test deletion did not remove unique behavior coverage
```

Expected: APPROVE or actionable findings with file and line references.

- [x] **Step 2: Resolve every validated spec finding**

For each finding:

1. verify it against current code and the approved design
2. add or adjust the smallest focused test when behavior or compatibility is at
   risk
3. make the minimum scoped correction
4. rerun the focused test and the affected Task 3 gate
5. request re-review

Do not accept suggestions that remove retained dispatch methods, change public
behavior, alter `policy_types()`, or harden regex behavior in this PR.

- [x] **Step 3: Request code quality and security review**

Require an independent reviewer to inspect the full branch diff for stale
imports, accidental test-coverage loss, signature changes, out-of-scope edits,
security impact, and maintainability. Resolve all critical and important
findings and re-request review until approved.

- [x] **Step 4: Repeat the complete Task 3 verification matrix**

Rerun Task 3 Steps 1 through 7 after the final review fix. Previous results are
not substitutes. Record fresh outputs and counts in `TASK-13112`.

- [x] **Step 5: Finalize Backlog.md tracking**

Use `backlog.task_edit` to:

- check all five acceptance criteria after evidence exists
- check all applicable Definition of Done items
- add final modified files
- add independent review outcomes
- add a final summary stating what changed and why
- document any known skip or residual private-surface compatibility risk
- set status to `Done` only after all required work is complete

- [x] **Step 6: Commit final review and tracking changes**

If review required code or test changes, stage only the validated files plus the
Backlog task. Otherwise stage only the Backlog task.

```bash
git add \
  tldw_Server_API/app/core/Moderation/moderation_service.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_delegation.py \
  'backlog/tasks/task-13112 - Clean-up-Moderation-evaluator-compatibility-seams.md'
git commit -m "chore(moderation): finalize compatibility cleanup"
```

When only Backlog changed, use:

```bash
git add 'backlog/tasks/task-13112 - Clean-up-Moderation-evaluator-compatibility-seams.md'
git commit -m "chore(backlog): close moderation compatibility cleanup"
```

- [x] **Step 7: Verify branch cleanliness and summarize PR readiness**

Run:

```bash
git status --short
git log --oneline origin/dev..HEAD
git diff --stat origin/dev...HEAD
git diff --check origin/dev...HEAD
```

Expected: clean status, only scoped commits, expected files only, and no
whitespace errors. Do not create or merge a PR until the human requester provides
the required human-written `Change summary` explaining what changed and why.

## Final Self-Review Checklist

- [x] Exactly eight approved private service delegates are removed.
- [x] `_evaluate_text_core()` remains and still dispatches redaction through
  public `redact_text()`.
- [x] `_evaluate_action_internal()` remains and still dispatches through public
  `evaluate_text()`.
- [x] Public method signatures and tuple ordering are unchanged.
- [x] `PolicyCompiler.policy_types()` and `PolicyEvaluator.policy_types()` are
  unchanged.
- [x] No production file except `moderation_service.py` changed.
- [x] Direct evaluator tests retain every deleted service-private scan behavior.
- [x] Two absence tests completed documented red/green cycles.
- [x] Compilation, changed-test Black, Ruff, pytest, Bandit, scope, and
  whitespace gates pass, with any production-file Black result compared to the
  base revision and documented.
- [x] Independent spec and quality reviews approve the final diff.
- [x] `TASK-13112` contains exact evidence, residual risk, and final summary.
- [x] Human-written PR `Change summary` gate is respected.
