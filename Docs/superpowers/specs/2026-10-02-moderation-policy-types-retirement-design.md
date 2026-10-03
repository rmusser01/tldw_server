# Moderation `policy_types()` Retirement Design

**Backlog task:** TASK-13421
**Date:** 2026-10-02
**Status:** Approved for written-spec review
**Predecessor:** TASK-13112, merged by PR #2929

## Purpose

Retire `PolicyCompiler.policy_types()` and `PolicyEvaluator.policy_types()` now
that `Moderation/models.py` canonically owns `ModerationPolicy`, `PatternRule`,
and `ModerationEvaluationResult`. Compiler and evaluator runtime logic will use
their existing private canonical model aliases directly.

This is a narrow structural refactor. Moderation decisions, policy assembly,
scan geometry, redaction, exception behavior, model identity, service imports,
and public endpoint contracts remain unchanged. The one intentional
compatibility break is removal of the undocumented model-substitution hook:
direct callers can no longer call `policy_types()`, and subclasses can no
longer replace model classes by overriding it.

## Context And Audit Evidence

`policy_types()` was introduced while the policy compiler and evaluator could
not import model classes without creating a cycle through
`moderation_service.py`. The shared-model extraction moved all three classes to
the neutral `Moderation/models.py` module and changed both hooks to return
private aliases imported from that module. The original cycle-breaking reason
no longer exists.

Current repository usage on `origin/dev` shows:

- production references are limited to each hook's definition and internal
  calls within `policy_compiler.py` and `policy_evaluator.py`
- no endpoint, service caller, integration module, user documentation, or
  public API documentation calls either hook
- tests deliberately preserve staticmethod descriptors, tuple identity,
  evaluator tuple caching, and subclass model substitution
- clean-process tests use the hooks to prove that compiler/evaluator imports do
  not load `moderation_service.py`
- the service compatibility exports still resolve to the exact canonical model
  classes

Repository analysis cannot prove that external Python consumers never called
or overrode these methods. The methods are not documented as public APIs, and
their only original purpose was import-cycle avoidance. This design accepts
their removal as an intentional callable and subclass-extension surface break.

The current focused baseline is 119 passing tests across the compiler,
evaluator, model characterization, canonical-model, and model-import suites.

## Goals

1. Remove both `policy_types()` methods and every internal call to them.
2. Bind compiler/evaluator runtime behavior directly to the canonical private
   model aliases already imported from `Moderation/models.py`.
3. Preserve exact canonical class identity and legacy imports through
   `moderation_service.py`.
4. Preserve compiler/evaluator runtime namespaces: public model names remain
   type-checking-only and absent at runtime.
5. Preserve import isolation without relying on the retired hooks.
6. Preserve all moderation policy, evaluation, redaction, exception, and
   caller behavior.
7. Keep the change small enough for one reviewed implementation pull request.

## Non-Goals

- No changes to `ModerationService` methods or signatures.
- No removal of compiler-side private service delegates.
- No migration of established callers to canonical model imports.
- No changes to model fields, defaults, factories, annotations, mutability,
  equality, serialization, or `__module__` values.
- No changes to policy precedence, rule parsing, category handling, scanning,
  action ranking, snippets, redaction, or replacement counting.
- No regex or ReDoS hardening.
- No new dependency-injection or model-factory abstraction.
- No deprecation period or runtime warning for the retired hooks.

## Approaches Considered

### Immediate retirement

Remove both methods, replace their internal uses with the existing private
canonical aliases, and replace hook-specific tests with behavioral and import
isolation coverage.

This is the selected approach. It removes speculative generality and the last
model-factory indirection introduced solely for the former import cycle.

### Deprecation period

Retain both methods, emit warnings, and remove them in a later release. This
would add runtime noise and another migration stage for undocumented internal
hooks with no repository or documentation consumers. It is not selected.

### Retain the hooks

Keep the current methods as supported model factories. This preserves the
tested subclass seam but leaves compiler/evaluator correctness dependent on
replaceable concrete domain classes without a current product requirement. It
is not selected.

## Production Design

### PolicyCompiler

In `policy_compiler.py`:

- remove `PolicyCompiler.policy_types()`
- replace policy construction through a local value returned by
  `self.policy_types()` with direct `_ModerationPolicy(...)` construction
- replace runtime `PatternRule` checks or construction selected through
  `self.policy_types()` with `_PatternRule`
- retain the existing `TYPE_CHECKING` imports so annotations and source-level
  type readability remain unchanged
- retain private runtime aliases so `ModerationPolicy` and `PatternRule` do not
  become public runtime module attributes

Compilation inputs, reports, outputs, method signatures, and dispatch among
actual compiler operations remain unchanged.

### PolicyEvaluator

In `policy_evaluator.py`:

- remove `PolicyEvaluator.policy_types()`
- remove the now-unused `functools.lru_cache` import
- replace runtime `PatternRule` selection with `_PatternRule`
- construct every evaluator result with `_ModerationEvaluationResult`
- retain the existing `TYPE_CHECKING` imports and private runtime aliases

Evaluation, redaction, snippet, scan, match, and count paths remain literal
apart from canonical class selection. No helper dispatch or algorithm changes
are permitted.

### Models And Service Facade

`models.py` and `moderation_service.py` receive no production changes.
`moderation_service.py` continues re-exporting the exact canonical classes:

```python
moderation_service.ModerationPolicy is models.ModerationPolicy
moderation_service.PatternRule is models.PatternRule
moderation_service.ModerationEvaluationResult is models.ModerationEvaluationResult
```

## Exact Implementation Scope

Expected production changes are limited to:

- `tldw_Server_API/app/core/Moderation/policy_compiler.py`
- `tldw_Server_API/app/core/Moderation/policy_evaluator.py`

Expected test changes are limited to the hook and import contracts in:

- `tldw_Server_API/tests/unit/test_moderation_models_characterization.py`
- `tldw_Server_API/tests/unit/test_moderation_models_imports.py`
- `tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py`

The Backlog task, this specification, and the later implementation plan are
the only expected tracking/documentation changes. Any need to modify
`models.py`, `moderation_service.py`, an endpoint, schema, configuration file,
or downstream caller requires stopping and revising the design before
continuing.

## Compatibility Boundary

The following behavior is intentionally removed:

- calling `PolicyCompiler.policy_types()`
- calling `PolicyEvaluator.policy_types()`
- overriding either method to substitute model classes
- relying on the evaluator hook's cached tuple object identity

After implementation, compiler/evaluator internals always use the canonical
classes imported from `Moderation/models.py`. A subclass may still override
actual compiler or evaluator operations through ordinary Python method
dispatch, but model-class substitution is no longer an extension point.

The following remain supported and unchanged:

- all public `ModerationService` methods and return shapes
- legacy model imports from `moderation_service.py`
- canonical model imports from `models.py`
- service evaluation and redaction dynamic dispatch
- compiler/evaluator operational subclassing unrelated to `policy_types()`
- complete model identity across service and canonical import paths

The pull request description must call out the removed direct-call and
subclass-hook behavior. No silent claim of complete callable-surface
compatibility is permitted.

## Import And Runtime Namespace Contracts

The neutral dependency direction remains:

```text
models.py
  ^       ^
  |       |
policy_compiler.py   policy_evaluator.py
          ^           ^
          |           |
          moderation_service.py
```

Removing the hooks must not weaken proof that the compiler and evaluator are
independent of the service module. The existing clean-process hook calls will
be replaced by representative operations:

1. import `PolicyCompiler`, compile a minimal global policy, assert the output
   is the canonical `ModerationPolicy`, and confirm `moderation_service` was
   never loaded
2. import `PolicyEvaluator`, evaluate a disabled or empty canonical policy with
   explicit `EvaluationLimits`, assert the output is the canonical
   `ModerationEvaluationResult`, and confirm `moderation_service` was never
   loaded

Existing import-order identity tests remain. Runtime namespace tests continue
asserting that public model names are absent from the compiler and evaluator
modules. Runtime type-hint behavior remains unchanged: public annotation names
stay type-checking-only rather than becoming runtime globals.

## Test Design

### Remove hook-specific tests

Delete tests whose only contract is the retired seam:

- staticmethod descriptor and signature assertions
- literal `policy_types()` tuple assertions
- evaluator tuple cache identity

Do not replace these tests with `__dict__`, `hasattr()`, or other durable
private-layout assertions. Hook absence is verified during review with source
search and diff inspection, not encoded as a behavioral contract.

Replace the two existing subclass model-substitution tests with the behavioral
inverse: subclasses may define a legacy-named `policy_types()` method, but
representative compiler and evaluator operations must ignore it and return the
canonical model classes. These tests fail against the current dynamic dispatch
and pass only after internal construction and checks use canonical aliases.
They lock the intended compatibility decision without asserting whether a
particular attribute exists on the base class.

### Preserve and strengthen behavioral coverage

Retain or adapt tests to prove:

- compiler global and user-policy outputs use the canonical policy class
- compiled rules use the canonical rule class
- evaluator empty, pass, warn, redact, and block paths return the canonical
  result class
- canonical identities survive legacy service imports and service-export
  rebinding
- compiler/evaluator representative operations do not load the service module
- all existing policy and evaluator literal behavior expectations remain
  unchanged
- compiler/evaluator runtime modules still exclude public canonical model names

Service-export rebinding coverage will execute representative compiler and
evaluator operations and assert canonical output identities instead of calling
the retired hooks.

### TDD sequence

1. Change the existing subclass tests to require canonical output even when a
   subclass defines the legacy hook; run them and record the expected failure.
2. Add or adapt clean-process operation tests while the current implementation
   is still present; run them green to establish the replacement import oracle.
3. Remove both hooks and switch internal type selection to canonical aliases.
4. Remove descriptor, tuple, and cache-only assertions that no longer describe
   supported behavior.
5. Run compilation before the focused red/green suite, then proceed through the
   complete verification matrix.

## Verification Gates

Verification is compilation-first.

1. Run `py_compile` over the two production modules and every changed test.
2. Run focused compiler, evaluator, characterization, canonical-model, and
   import-isolation tests.
3. Run every `tldw_Server_API/tests/unit/test_moderation*.py` test.
4. Run Guardian supervised-policy tests.
5. Run the complete Chat moderation integration suite.
6. Run Workflow moderation-adapter tests.
7. Run the exact Audio transcription redaction integration test.
8. Run Ruff on all touched Python files.
9. Run Black checks without mass-formatting unrelated existing code.
10. Run Bandit over the touched Moderation production modules.
11. Run `git diff --check` and a source audit proving no production or test
    references to the retired hooks remain.
12. Fetch current `origin/dev`, verify the branch merge-base is current, and
    rerun affected gates after any rebase.

No gate may be weakened merely because the production diff is small.

## Security, Reliability, And Performance

This change introduces no input, output, authorization, persistence, file,
network, regex, or logging path. It changes no exception boundary and adds no
mutable state.

Removing the evaluator's one-entry tuple cache removes only cached class-tuple
allocation; evaluator operations already read the cached tuple per call. Direct
private aliases avoid that lookup and do not add meaningful runtime cost.

Concurrency behavior is unchanged because canonical classes are module-level
immutable bindings for this purpose, and the retired tuple cache contains no
request-specific state.

## Rollout And Rollback

No feature flag or data migration is required. The change ships as one focused
implementation pull request after the design and implementation plan are
approved.

Rollback is a normal pull-request revert. Restoring the two methods and their
internal dispatch restores direct-call and subclass model-substitution
compatibility without database or configuration repair.

## Risks And Mitigations

### Unknown external subclass usage

Risk: an external consumer may override `policy_types()` or call it directly.

Mitigation: make the break explicit in the design and PR description, keep the
production diff narrowly limited, and retain a one-commit revert path. The
project has no documented or repository caller to migrate.

### Import-cycle regression

Risk: deleting hook-focused tests could remove the proof that compiler and
evaluator imports remain service-independent.

Mitigation: replace those tests with stronger clean-process tests that execute
representative compiler/evaluator behavior and inspect `sys.modules`.

### Runtime namespace expansion

Risk: replacing the hooks with normal public imports could expose model names
from compiler/evaluator modules or change type-hint resolution.

Mitigation: use the existing underscore-prefixed runtime aliases, retain
`TYPE_CHECKING` imports, and keep runtime namespace and type-hint tests.

### Tautological identity coverage

Risk: merely asserting imported aliases could miss an internal construction
site that still selects a substitute type.

Mitigation: assert canonical identity on actual compiler and evaluator outputs
across representative paths and run the full moderation regression matrix.

## Follow-Up Work

After this slice, separate reviewed tasks may:

1. audit and remove repository-unused compiler-side private
   `ModerationService` delegates
2. harden the complete long-text regex and redaction execution path as an
   explicitly behavior-changing security/reliability change

Neither follow-up belongs in this pull request.
