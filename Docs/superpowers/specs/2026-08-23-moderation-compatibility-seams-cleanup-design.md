# Moderation Compatibility Seams Cleanup Design

**Backlog task:** TASK-13112
**Status:** Approved
**Base:** `origin/dev` at `2c3589fa09`

## Purpose

Remove eight obsolete private `ModerationService` evaluator delegates left
behind by the `PolicyEvaluator` extraction. The change reduces duplicate
callable surface without changing moderation decisions, public service
behavior, active extension dispatch, policy compilation, or regex/redaction
semantics. Removing these undocumented private names is an intentional private
callable-surface break; runtime behavior and supported interfaces remain stable.

This is the next structural unit after the shared Moderation models extraction.
Behavior-changing hardening remains a separate follow-up.

## Context And Evidence

`PolicyEvaluator` now owns the rule-category helpers, phase/category gates,
snippet construction helper, scan helpers, match collection, and redaction
application. `ModerationService` still exposes eight private methods that only
forward to those evaluator operations.

Repository-wide usage on current `dev` shows no production call sites for the
candidate private service methods. Remaining repository references are the
method definitions, historical design documents, and compatibility or
characterization tests written to protect the earlier extraction. Equivalent
evaluator behavior is covered by `test_moderation_policy_evaluator.py`, while
public service dispatch remains covered by the service characterization and
delegation suites.

Repository analysis cannot prove that third-party consumers never called these
private names. They are not documented public APIs or active service extension
points, so this design accepts their removal as a deliberate private-surface
break. `_evaluate_action_internal()` is different: it dispatches through the
public, overridable `evaluate_text()` boundary and was explicitly retained by
the prior evaluator design. It remains in this PR.

The unchanged pre-implementation baseline on the starting commit is:

- 318 tests passed across every `tests/unit/test_moderation*.py` file
- 89 Guardian supervised-policy tests passed
- 16 tests passed in the complete Chat moderation integration suite
- 12 selected Workflow moderation-adapter tests passed
- 1 targeted Audio transcription redaction test passed

## Considered Approaches

### 1. Remove all transitional compatibility seams

Remove the private evaluator delegates, `_evaluate_text_core()`, and both
`policy_types()` hooks in one PR.

This produces the smallest surface immediately, but it also changes tested
subclass dispatch and model-factory extension behavior. That exceeds a strict
structural cleanup and makes regressions harder to localize.

### 2. Remove only repository-unused private evaluator delegates

Remove the forwarding methods that have no production or required extension
role. Keep public methods, `_evaluate_text_core()`, compiler compatibility
methods, and `policy_types()` unchanged.

This is the recommended approach. It removes proven redundancy while preserving
the boundaries that still participate in service behavior or have not completed
an external compatibility review.

### 3. Deprecate the private delegates in place

Keep every delegate and add warnings or documentation before later removal.

This adds runtime and maintenance noise for private methods with no repository
consumers. There is no established public deprecation contract for these names,
so this approach provides little practical value.

## Exact Scope

Remove these private `ModerationService` methods:

- `_effective_rule_categories()`
- `_rule_applies_to_phase()`
- `_rule_matches_enabled_categories()`
- `_build_sanitized_snippet()`
- `_iter_scan_chunks()`
- `_find_match_span()`
- `_collect_rule_matches()`
- `_apply_rule_redactions()`

Retain unchanged:

- `check_text()` and `evaluate_text()` dispatch through `_evaluate_text_core()`
- `_evaluate_text_core()` and its dispatch through public `redact_text()`
- `build_sanitized_snippet()`
- `redact_text()` and `redact_text_with_count()`
- `evaluate_action()` and `evaluate_action_with_match()`
- `_evaluate_action_internal()` and its dispatch through public
  `evaluate_text()`
- `_evaluation_limits()` and per-call immutable limit snapshots
- compiler-owned service delegates, including `_compile_user_rule()`,
  `_coalesce_bool()`, and `_parse_bool_value()`
- `PolicyCompiler.policy_types()` and `PolicyEvaluator.policy_types()`

## Test Design

First add a focused surface test that lists the eight obsolete names and expects
each name to be absent from `ModerationService.__dict__`. This class-local check
is the intended structural invariant and will not make assumptions about future
base classes. It must fail against the baseline. Then remove the methods and
their wrapper-specific delegation/signature tests.

Characterization tests that call removed scan helpers through the service will
be removed only where direct `PolicyEvaluator` tests already cover the same
literal behavior. Public facade and dynamic-dispatch tests remain. The action
wrapper characterization will continue to verify `_evaluate_action_internal()`,
`evaluate_action()`, and `evaluate_action_with_match()` through overridden
public `evaluate_text()`.

Verification will cover:

- the focused compatibility surface test with a red/green cycle
- every `tldw_Server_API/tests/unit/test_moderation*.py` test
- `tldw_Server_API/tests/Guardian/test_supervised_policy.py`
- the complete `tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py`
  suite
- `tldw_Server_API/tests/Workflows/adapters/test_llm_adapters.py` selected with
  `-k moderation_adapter`
- `test_audio_transcriptions_redacts_text_and_segments_when_stt_redaction_enabled`
  from
  `tldw_Server_API/tests/Audio/test_audio_transcription_retention_and_redaction.py`
- Python compilation of touched modules and tests
- Bandit on touched Python paths
- Ruff, `git diff --check`, and import self-review

The same pytest matrix will run before implementation and after the final code
change so results remain directly comparable.

## Compatibility And Risk Controls

The removed methods are private, but absence is still a callable-surface change
and external usage cannot be ruled out. Risk is accepted because the methods are
undocumented direct evaluator shims with no repository consumers. It is
controlled by the narrow exact-name list and retention of every public and
active extension boundary, including `_evaluate_action_internal()`.

No evaluator implementation moves in this PR. No rule ordering, phase/category
logic, scan geometry, fallback limit, replacement count, redaction sequencing,
exception behavior, or state snapshot behavior may change.

## Non-Goals

- removing or redesigning `policy_types()`
- removing `_evaluate_text_core()`
- removing `_evaluate_action_internal()`
- changing public `ModerationService` signatures
- migrating callers to new public APIs
- moving `EvaluationLimits` or `_ResolvedModerationServiceState`
- changing policy compilation or compiler delegates
- hardening long-text regex or redaction execution
- adding deprecation warnings

## Follow-Up Work

Separate reviewed tasks may:

1. audit and retire `policy_types()` if external compatibility permits
2. assess compiler-side private delegate cleanup
3. harden the complete long-text regex and redaction execution path
