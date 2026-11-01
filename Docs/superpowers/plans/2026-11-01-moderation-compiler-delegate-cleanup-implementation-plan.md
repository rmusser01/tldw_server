# Moderation Compiler Delegate Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (selected previously) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove five unused private service compiler delegates without changing supported moderation behavior.

**Architecture:** PolicyCompiler keeps all compilation implementations. ModerationService keeps active parsing/validation delegates, report logging, public APIs, persistence, and evaluation dispatch. Replace three test-only calls before removing the unused methods.

**Tech Stack:** Python, pytest, Loguru, Ruff, Black, Bandit; existing project virtual environment and test fixtures.

**Spec:** `Docs/superpowers/specs/2026-11-01-moderation-compiler-delegate-cleanup-design.md`

**Tracking:** TASK-15205 covers the approved design and this plan. Create a linked implementation task before source/test edits.

## Global Constraints

- Remove only `_split_action_directive`, `_resolve_categories_override`, `_parse_categories_override`, `_compile_user_rule`, and `_coalesce_bool` from ModerationService.
- Keep their PolicyCompiler implementations and every public API unchanged.
- Retain active parsing, boolean parsing, regex checks, blocklist loading/building, constants, warning sanitization, and dispatch.
- No regex hardening, policy semantics changes, persistence changes, caller migrations, dependency additions, or unrelated formatting.
- Accept only the documented private callable-surface break; external private callers cannot be ruled out.
- Activate `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate` before every Python command.
- Keep Backlog records and stage statuses current; do not create or merge a PR in this execution without authorization.

## Review Focus

- Invalid `is_regex`, dangerous regex, and malformed regex must still produce the three sanitized service warnings (Stage 1).
- Rule IDs and filesystem paths must never appear in those warnings (Stage 1).
- Literal user rules must retain case-insensitive matching, action, phase, and wildcard categories (Stage 1).
- Compiler rejection must retain sanitized issue source, reason, and supplied index (Stage 1).
- Active parser/validator dispatch and public evaluation/redaction overrides must remain intact (Stages 2 and 3).

## Task 1: Preserve Coverage Through Its Owners (Stage 1)

**Goal:** Eliminate test-only dependencies on the removed delegates before deletion.
**Success Criteria:** The replacement tests pass against unchanged production code and retain every old assertion.
**Tests:** Three compiler/blocklist/user-override suites and the downstream matrix below.
**Status:** Not Started

**Files:** Modify `tldw_Server_API/tests/unit/test_moderation_policy_compiler.py`, `test_moderation_blocklist_parse.py`, and `test_moderation_user_override_validation.py` in the same unit-test directory.
**Interfaces:** Consume `PolicyCompiler.compile_user_rule(raw_rule: object, report: PolicyCompilationReport, index: int) -> PatternRule | None` and `ModerationService.get_effective_policy(user_id: str | None) -> ModerationPolicy`. Produce no new application API.

- [ ] Create a linked implementation Backlog task, read the approved spec, and repeat repository-wide literal-name and quoted-name searches. Stop for new production callers rather than widening scope.
- [ ] Compile the service/compiler and run the complete Stage 3 regression matrix before editing. Record counts; investigate failures before attributing them to this cleanup. Capture scoped Bandit baseline findings.
- [ ] Add `test_compile_user_rule_rejects_dangerous_regex_with_sanitized_issue() -> None` to the compiler suite. Use `{id: "private-rule-id-92741", pattern: "(a+)+$", is_regex: True, action: "block", phase: "both"}`, a fresh report, and index `7`. Assert result is `None`, issues equal `[PolicyCompilationIssue(source="user_rule", reason="dangerous_regex", index=7)]`, and report repr excludes the regex and rule ID.
- [ ] Add `test_compile_user_rule_preserves_phase_action_and_literal_matching() -> None`. Compile `{id: "warn-1", pattern: "heads up", is_regex: False, action: "warn", phase: "input"}`. Assert canonical PatternRule, `action == "warn"`, `phase == "input"`, `categories == {"*"}`, `replacement is None`, an empty report, and matching `"please HEADS UP now"`.
- [ ] Replace `test_compile_user_rule_sanitizes_rule_id_warning_logs` with `test_effective_policy_compilation_sanitizes_rule_warning_logs() -> None`. Keep the existing three invalid rule dictionaries and all warning/non-disclosure assertions. Create a normal service before adding the warning sink; set a controlled `ModerationPolicy(enabled=True, per_user_overrides=True, block_patterns=[])` and raw `_user_overrides = {"test-user": {"rules": rules}}`; call public `get_effective_policy("test-user")` under the sink and assert the effective policy has no rules. Restore the sink in `finally`. Do not call a setter or sanitizer that removes the invalid rules before compilation.
- [ ] Run the two new compiler tests and the replacement service-log test against unchanged production code; expect PASS. This is coverage migration, not a behavioral red/green change.
- [ ] Remove only the two superseded tests from `test_moderation_blocklist_parse.py`; retain the active parser delegation tests. Add concise docstrings and annotations to new/modified tests.
- [ ] Compile touched test files, run the three complete focused suites, run scoped lint/security checks, and commit the green coverage migration with its Backlog update.

## Task 2: Remove Exactly Five Unused Delegates (Stage 2)

**Goal:** Retire the redundant service surface with a demonstrated structural red/green cycle.
**Success Criteria:** Five removal cases pass; compiler and service behavior tests stay green; no other production edits.
**Tests:** New surface test plus the Stage 1 focused suites.
**Status:** Not Started

**Files:** Create `tldw_Server_API/tests/unit/test_moderation_compiler_compatibility_surface.py`; modify `tldw_Server_API/app/core/Moderation/moderation_service.py`.
**Interfaces:** No new supported interface; the five private methods cease to exist in the service class only.

- [ ] Add module imports for pytest and ModerationService, `pytestmark = pytest.mark.unit`, and this parameterized unit test with a docstring:

```python
@pytest.mark.parametrize("name", [
    "_split_action_directive", "_resolve_categories_override",
    "_parse_categories_override", "_compile_user_rule", "_coalesce_bool",
])
def test_service_omits_unused_compiler_delegate(name: str) -> None:
    """Keep unused compiler forwarding methods out of the service facade."""
    assert name not in ModerationService.__dict__
```

- [ ] Run `python -m pytest tldw_Server_API/tests/unit/test_moderation_compiler_compatibility_surface.py -q`; expect exactly five assertion failures because the delegates still exist.
- [ ] Delete only the five method definitions and their decorators/docstrings from the service. Do not change imports, constants, adjacent bodies, PolicyCompiler, or active delegates.
- [ ] Compile the service and new test first. Run the surface test plus all three focused suites; expect PASS. Run the new module with `-m unit` to prove all five cases are selected. Re-run the existing `test_service_parser_wrappers_delegate_to_policy_compiler` case explicitly.
- [ ] Confirm remaining references to removed names outside historical docs/tracking are only the new surface test; compiler method names without the private service prefix remain supported. Run lint, formatting of the new file, scoped security checks, and diff checks, then commit the green removal.

## Task 3: Verify And Review The Complete Change (Stage 3)

**Goal:** Demonstrate unchanged supported behavior and obtain independent review.
**Success Criteria:** The regression matrix and quality gates pass with no new findings; review issues are resolved.
**Tests:** Complete matrix below.
**Status:** Not Started

**Files:** Only task records and this plan for status/results, unless review identifies an in-scope correction.
**Interfaces:** Deliver a reviewed local branch; no PR creation or merge is implied.

- [ ] Activate the project virtual environment, compile every touched Python file, then run:

```bash
python -m pytest tldw_Server_API/tests/unit/test_moderation*.py -q
python -m pytest tldw_Server_API/tests/Guardian/test_supervised_policy.py -q
python -m pytest tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py -q
python -m pytest tldw_Server_API/tests/Workflows/adapters/test_llm_adapters.py -k moderation_adapter -q
python -m pytest tldw_Server_API/tests/Audio/test_audio_transcription_retention_and_redaction.py::test_audio_transcriptions_redacts_text_and_segments_when_stt_redaction_enabled -q
```

- [ ] Run Ruff on touched Python files. Run Black on the new surface test; format only newly edited test functions elsewhere to avoid whole-file churn. Compare any existing formatter/lint failures against the baseline rather than hiding them.
- [ ] Run Bandit on touched production and test paths, recording JSON under `/tmp`; exclude expected test-assert B101 only for test scans. Compare existing findings to baseline and fix any new in-scope findings. Run `git diff --check` and inspect `git diff origin/dev...HEAD` for scope.
- [ ] Have fresh subagents review spec compliance and code quality for each implementation task, then obtain a fresh whole-branch review, following the selected subagent-driven workflow. Review must specifically check preservation of warning-log coverage, active dispatch, canonical imports, and compiler issue semantics.
- [ ] Fix verified in-scope findings and rerun affected checks. Record commands/counts, residual private-call compatibility risk, and review outcomes in the implementation task; update stage statuses and commit only the related records.
- [ ] Report local branch readiness. A future AI-authored PR still requires the human-written Change summary; do not reuse the previous PR's summary for this different change.

## Plan Self-Review

The exact-name scope and retained helpers map to Stage 2; all three migrated tests
and five review-focus conditions map to Stages 1-3. Signatures match the current
compiler/service code. Production changes are deletions only; the plan adds no
engine, configuration, model, public API, or behavior changes. Implementation is
awaiting review of this plan; subagent-driven execution is the preserved method.
