# Moderation PolicyEvaluator Refactor Design

Task: TASK-12984
Date: 2026-07-23
Status: Approved

## Purpose

Extract moderation rule evaluation and redaction from `ModerationService` into a mostly pure, stateless `PolicyEvaluator` without changing observable behavior. `ModerationService` remains the public facade and continues to own configuration, policy compilation, I/O, persistence, locking, and logging.

This is the next structural slice after the `PolicyCompiler` refactor merged in PR #2528. The implementation is intentionally behavior-preserving. Any verified behavior or security changes belong in separate follow-up pull requests.

## Current Problem

`PolicyCompiler` now owns deterministic policy assembly, but `moderation_service.py` still owns two separate clusters of rule-execution logic:

- decision evaluation, including phase/category gating, match scanning, action ranking, category selection, snippets, and result construction
- redaction, including sequential substitutions, replacement limits, long-text match collection, and replacement counting

This keeps `ModerationService` responsible for policy lifecycle and policy execution. It also makes the evaluation behavior difficult to test independently from service configuration and state.

## Goals

- Preserve all public `ModerationService` method signatures and return types.
- Preserve dynamic dispatch through the public `ModerationService` evaluation and redaction methods.
- Preserve current evaluation, scanning, ranking, snippet, and redaction behavior, including known quirks.
- Add a stateless `PolicyEvaluator` with explicit text, policy, phase, and limit inputs.
- Make runtime evaluation limits explicit through an immutable per-call snapshot.
- Keep one owner for evaluation and redaction logic while retaining service compatibility delegates.
- Keep the evaluator free of file I/O, persistence, configuration lookup, logging, and mutable runtime state.
- Add characterization and direct evaluator tests that make current behavior explicit.

## Non-Goals

- Do not change moderation endpoint schemas or response fields.
- Do not remove or bypass a public `ModerationService` extension point.
- Do not change action precedence, category semantics, phase semantics, scanning algorithms, or redaction behavior.
- Do not harden the long-text redaction scan in this pull request.
- Do not move `ModerationPolicy`, `PatternRule`, or `ModerationEvaluationResult` out of `moderation_service.py`.
- Do not remove public or private service compatibility methods.
- Do not fold supervised/guardian behavior into `PolicyEvaluator`.
- Do not add a feature flag, new diagnostics, or new logging.
- Do not add remote moderation providers or asynchronous evaluation.

## Approaches Considered

### Stateless evaluator with explicit inputs

`ModerationService` passes text, policy, phase, and an immutable `EvaluationLimits` snapshot into a stateless evaluator. This keeps configuration changes visible immediately and avoids hidden evaluator state.

This is the selected approach.

### Configured evaluator instance

The evaluator could retain scan and replacement limits and be rebuilt during service reloads. Calls would be shorter, but direct limit changes and partial reload behavior could leave stale evaluator state. This is not selected.

### Functional helper module

Top-level functions would be mechanically simple, but they would not provide the requested class boundary and would leave a weaker dependency seam for service delegation and focused tests. This is not selected.

## Proposed Structure

Create `tldw_Server_API/app/core/Moderation/policy_evaluator.py` containing:

- `EvaluationLimits`: a frozen snapshot of the service's current evaluation limits
- `PolicyEvaluator`: stateless evaluation, scanning, ranking, snippet, and redaction behavior
- an evaluator-local noncritical exception tuple matching the existing moved catch behavior

Modify `moderation_service.py` to:

- construct one stateless `PolicyEvaluator` instance
- build a fresh, coherent `EvaluationLimits` for each evaluator delegation
- preserve the current public evaluation/redaction call chain while delegating the logic bodies
- retain current public and private method descriptors, signatures, and outputs
- remove only duplicated logic bodies after delegates are in place

No endpoint, schema, workflow, Chat, STT, guardian, or persistence module requires a contract change.

## Data Contracts

### EvaluationLimits

`EvaluationLimits` is a frozen dataclass with the service's supported runtime types:

```python
@dataclass(frozen=True)
class EvaluationLimits:
    max_scan_chars: int
    match_window_chars: int
    max_fallback_scan_chars: int
    max_replacements_per_pattern: int | None
```

The annotations describe supported service state; they do not introduce runtime validation. `ModerationService._evaluation_limits()` copies the current attributes without clamping, defaulting, coercing, or otherwise normalizing them. Unsupported values introduced through direct private-attribute mutation are therefore also copied unchanged, and the evaluator preserves the current operation-level coercion and exception behavior. Characterization tests may deliberately inject unsupported values with an explicit type-check suppression.

`_evaluation_limits()` acquires the service's existing `RLock` while reading all four attributes. Config resolution reached from `reload()` and `update_settings()` can rewrite `max_scan_chars`, `match_window_chars`, and `max_replacements_per_pattern` while that lock is held. `max_fallback_scan_chars` is initialized from the environment and is not currently reloaded. The lock prevents a snapshot from observing config resolution while it is in progress. A completed resolution may still retain an older field value when that field's conversion was independently suppressed; preserving that mixed retained state is current behavior. Unsynchronized direct mutation of private attributes remains outside the concurrency guarantee.

Each service-to-evaluator delegation creates one snapshot. A direct `PolicyEvaluator.evaluate_text(..., include_redacted_text=True)` call reuses its supplied snapshot for evaluator-owned redaction. `ModerationService` evaluation instead preserves its existing public dispatch through `self.redact_text()`; the built-in public redaction method creates its own snapshot. This retains public override behavior and the current possibility that runtime limits change between decision evaluation and redaction.

### Existing policy and result types

`ModerationPolicy`, `PatternRule`, and `ModerationEvaluationResult` remain defined and publicly importable from `moderation_service.py`. `policy_evaluator.py` uses `TYPE_CHECKING` and deferred runtime imports, matching the established `PolicyCompiler` pattern and avoiding a module import cycle.

The evaluator continues accepting both `PatternRule` values and raw compiled regex values in `ModerationPolicy.block_patterns`.

## Evaluator Interface

The primary operations are:

```python
evaluate_text(
    text,
    policy,
    phase,
    limits,
    *,
    include_redacted_text,
) -> ModerationEvaluationResult

redact_text(text, policy, phase, limits) -> str

redact_text_with_count(text, policy, phase, limits) -> tuple[str, int]

build_sanitized_snippet(
    text,
    policy,
    match_span,
    pattern=None,
) -> str | None
```

Service-facing helper operations cover:

- effective rule category normalization
- phase eligibility
- category eligibility
- sanitized snippet construction for a known replacement
- scan-chunk iteration
- first-match lookup
- redaction-match collection
- applying precomputed redactions

Evaluator helpers are static where they do not require an evaluator instance. `ModerationService` retains each existing helper's class, static, or instance descriptor and delegates to the corresponding evaluator operation.

## Service Delegation

The service delegation flow is:

1. Keep `check_text()` and `evaluate_text()` dispatching through `self._evaluate_text_core()`.
2. `_evaluate_text_core()` takes one locked limits snapshot and calls evaluator decision evaluation with `include_redacted_text=False`.
3. When its existing flag requests redacted output and the decision is `redact`, `_evaluate_text_core()` calls the public `self.redact_text()` method and uses `dataclasses.replace()` to return the decision with that redacted text.
4. The built-in `redact_text()` and `redact_text_with_count()` methods take their own locked snapshots and delegate their logic to `PolicyEvaluator`.
5. Return the existing service result shapes unchanged.

Specific mappings are:

- `check_text()` retains its call to `self._evaluate_text_core(..., include_redacted_text=False)` and returns the current `(flagged, sample)` tuple.
- `evaluate_text()` retains its call to `self._evaluate_text_core(..., include_redacted_text=True)`.
- `_evaluate_text_core()` remains a compatibility delegate, uses the evaluator for the decision, and preserves dynamic dispatch through `self.redact_text()` when its existing flag requires redacted output.
- `evaluate_action()`, `evaluate_action_with_match()`, and `_evaluate_action_internal()` keep their current service call chain and tuple ordering.
- `redact_text()` and `redact_text_with_count()` delegate with one per-call limit snapshot.
- `build_sanitized_snippet()` delegates without changing pattern-to-replacement lookup behavior.
- existing private helper names remain callable delegates.

All public service methods retain dynamic dispatch, including evaluation-triggered calls to an overridden or monkeypatched public `redact_text()`. `check_text()` and `evaluate_text()` also retain their current dispatch through `_evaluate_text_core()`.

Once execution enters a private compatibility delegate, evaluator-internal calls do not route back through overridden or monkeypatched private scan/category/snippet helpers. Repository search found no caller that relies on interception at that private-helper level. This is an explicit unsupported-private-extension boundary, not a change to public dispatch.

## Evaluation Semantics To Preserve

### Entry behavior

- Empty text returns a default pass result.
- A disabled policy returns a default pass result.
- `phase="input"` respects `policy.input_enabled` and uses `policy.input_action` as the default action.
- `phase="output"` respects `policy.output_enabled` and uses `policy.output_action` as the default action.
- `phase=None` or any other value bypasses rule phase filtering and uses `warn` as the default action.

Unknown phase does not mean that only rules marked `both` run. It means no phase filter is applied, so input-only, output-only, and both-phase rules are all eligible.

### Rule eligibility

- Phase and category checks apply to `PatternRule` values.
- Raw compiled regex values bypass `PatternRule` phase and category metadata checks.
- A falsy enabled-category filter allows all categories.
- `*` in either enabled categories or rule categories acts as a wildcard.
- A rule with no effective categories uses `uncategorized`.

### Matching and ranking

- Evaluation scans the original input text for every eligible rule.
- Action rank remains `block` over `redact` over `warn`.
- A missing or falsy effective action falls back to `warn`.
- A string action is lowercased; an unsupported string falls back to `warn`.
- Evaluation calls `.lower()` on the effective action without pre-normalizing its type, then tests the returned value for membership in the allowed-action set.
- A missing `.lower()` propagates `AttributeError`; a hashable unsupported result such as lowercased `bytes` falls back to `warn`; a valid string result is accepted; and an unhashable result propagates `TypeError`.
- For equal action rank, the earliest match position wins.
- For equal rank and equal position, the first encountered rule remains selected.
- The selected pattern remains the regex pattern string.
- Category selection retains the current normalization, intersection, generic `pii` removal, and lexical ordering behavior.

### Result construction

- No selected match returns a default pass result.
- Sanitized samples use the selected rule replacement or policy replacement and retain current bounds/truncation.
- Redacted output is computed only when requested and the selected action is `redact`.
- `include_redacted_text=False` must not invoke redaction.

## Scan Semantics To Preserve

Evaluation and redaction remain separate internal scan paths because their current behavior differs.

### Evaluation first-match scan

- `chunk_size` is exactly `max(1, int(max_scan_chars))`.
- Text whose length is less than or equal to `chunk_size` uses one `pat.search(text)`.
- For longer text, overlap is exactly `min(1024, max(32, chunk_size // 10))`, capped to `max(0, chunk_size - 1)`.
- Chunk step is `chunk_size - overlap` when `chunk_size > overlap`, otherwise `chunk_size`.
- Long-text matching searches the original text with `pat.search(text, start, window_end)`; it does not search sliced chunk strings.
- `window_end` is `min(len(text), end + max(0, int(match_window_chars)))`.
- A match found in the extended window is accepted only when `match.start() < end`.
- The full-text fallback limit is exactly `max(1, int(max_fallback_scan_chars))`; fallback runs only when input length is less than or equal to that value.
- Regex errors return no match.

### Redaction match scan

- Short text uses `re.sub` or `re.subn` with the current per-pattern count semantics.
- Long text uses the current full-text `finditer` collection path.
- Long-text match collection skips zero-length matches; short-text substitution behavior remains unchanged.
- A non-positive replacement cap remains unlimited.
- The replacement cap resets for each rule.

The inaccurate existing implication that long-text redaction is chunk-scanned must not drive an implementation change in this pull request.

## Redaction Semantics To Preserve

- Direct redaction does not check `policy.enabled`.
- Direct redaction returns the original text for empty text, no patterns, or a disabled requested input/output phase.
- Redaction applies eligible rules sequentially, so later rules see text already changed by earlier rules.
- Direct redaction ignores each eligible rule's action and therefore substitutes matching `warn`, `block`, and `redact` rules alike.
- Rule replacement overrides the policy replacement when truthy.
- Replacement strings remain literal and are not interpreted as backreferences.
- `redact_text_with_count()` returns the current total count across sequential rules.
- Regex failures skip the current rule where they do today.

Direct `PolicyEvaluator` evaluation uses evaluator-owned redaction and the same supplied `EvaluationLimits`. Evaluation through `ModerationService` deliberately calls back through the public `self.redact_text()` method to preserve the existing extension point; the built-in public method then delegates the redaction logic to the evaluator.

## Snippet Semantics To Preserve

- Match spans are clamped to the input bounds.
- The snippet keeps up to 16 characters on each side of the replacement.
- Empty replacement falls back to `[REDACTED]` where it does today.
- Snippets longer than 80 characters are truncated to 77 characters plus `...`.
- Public snippet construction finds the first `PatternRule` with the same pattern string and uses its truthy replacement.
- Snippet replacement lookup does not add phase or category filtering.

## Error Handling

`policy_evaluator.py` owns an evaluation-specific noncritical exception tuple that initially mirrors the exception types currently caught by the moved broad fallback sites:

- `OSError`
- `ValueError`
- `TypeError`
- `KeyError`
- `RuntimeError`
- `AttributeError`
- `ConnectionError`
- `TimeoutError`
- `json.JSONDecodeError`
- `re.error`

This tuple is used only where moved evaluation code currently catches `_MODERATION_NONCRITICAL_EXCEPTIONS`. Existing narrow `re.error` catches remain narrow. Unexpected failures continue propagating where they do today.

The evaluator does not add logging, diagnostics, sanitized reports, or public errors.

## State And Concurrency

`PolicyEvaluator` owns no mutable instance state and introduces no lock. It is reentrant when callers do not mutate its inputs during an operation.

Policies, rule objects, category sets, and pattern collections are borrowed mutable values. They may be service-owned, caller-owned, or shared: `get_effective_policy()` can return the service's global policy by identity, and user-policy compilation preserves base-policy identity when no override exists and otherwise shallow-copies the rule list.

The evaluator preserves these identity and shallow-alias relationships and does not mutate any borrowed input. It does not claim to make concurrent mutation of shared policy inputs safe.

## Compatibility Boundaries

The following remain unchanged:

- public `ModerationService` method signatures and tuple/result ordering
- public evaluation/redaction dynamic dispatch, including overridden `redact_text()`
- `ModerationPolicy`, `PatternRule`, and `ModerationEvaluationResult` identities and import paths
- endpoint schemas and response behavior
- Chat input, streaming output, non-streaming output, and persistence call sites
- Workflow moderation adapter behavior
- STT redaction behavior
- supervised/guardian policy composition
- service config, environment, reload, persistence, blocklist, and user-override behavior

`PolicyCompiler` remains the policy assembly owner. `PolicyEvaluator` consumes its compatible `ModerationPolicy` output and has no compiler responsibility.

## Testing Strategy

### Characterization before extraction

Add table-driven service tests with literal expected outputs before moving logic. These tests are the independent behavior oracle and cover:

- disabled policy versus direct redaction
- input, output, `None`, and unknown phases
- empty, wildcard, uncategorized, and intersecting categories
- raw regex and `PatternRule` values
- block/redact/warn precedence and earliest-match tie-breaking
- falsy and unsupported-string effective actions
- representative truthy non-string actions whose `.lower()` is missing, returns `bytes`, returns a valid action string, or returns an unhashable value, with literal fallback/exception expectations
- equal-rank/equal-position first-rule selection
- rule and policy replacements
- sanitized snippets and pattern-to-replacement lookup
- sequential redaction and replacement counting
- direct redaction of eligible `warn`, `block`, and `redact` rules
- positive, zero, negative, `None`, numeric-string, and malformed-string replacement limits on both short and long paths
- `None`, numeric-string, and malformed-string `max_scan_chars` values across evaluation, direct chunk iteration, and redaction path selection
- `None`, numeric-string, and malformed-string `match_window_chars` and `max_fallback_scan_chars` values on long evaluation paths that reach their respective coercion sites
- short and long scan paths
- exact overlap/step behavior, original-string `pos`/`endpos` scanning, anchored/lookbehind patterns, boundary-spanning matches, and fallback limits
- bounded long redaction proving full-text `finditer` behavior with exact text and replacement count
- short/long zero-length match behavior
- malformed raw rules and current exception propagation
- `check_text()` and `_evaluate_text_core(..., include_redacted_text=False)` avoiding the current public redaction call
- service evaluation preserving a subclassed or monkeypatched public `redact_text()` override
- global-policy identity and shallow rule-alias behavior remaining unchanged
- no mutation of policy, rules, or categories

Characterization expectations must be literal. Tests must not derive expected values by invoking another path that will later delegate to the same evaluator.

### Direct evaluator tests

Create `tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py` with direct tests for the same behavior matrix and explicit `EvaluationLimits` values.

Direct evaluator tests additionally prove:

- `include_redacted_text=False` never invokes evaluator redaction, using a redaction spy or a regex double whose search succeeds and whose substitution methods raise
- `include_redacted_text=True` passes the identical supplied limits object into nested evaluator redaction
- `EvaluationLimits` is frozen and evaluator operations do not mutate it
- evaluator operations do not mutate borrowed policies, rules, category sets, or pattern collections

Supplemental service/evaluator parity tests may compare outputs after delegation, but they are not substitutes for literal characterization expectations.

### Delegation invariant tests

Post-extraction service tests prove:

- `_evaluation_limits()` copies all four raw values without normalization
- snapshot construction holds the service lock and observes either the state before a controlled config reload or its completed state, never the in-progress assignments
- built-in public evaluation and redaction create separate snapshots
- `check_text()` still dispatches through `_evaluate_text_core()` but does not invoke redaction
- `evaluate_text()` still dispatches through `_evaluate_text_core()` with `include_redacted_text=True`
- `_evaluate_action_internal()`, `evaluate_action()`, and `evaluate_action_with_match()` still dynamically dispatch through an overridden or monkeypatched public `evaluate_text()`
- full evaluation still dispatches through an overridden or monkeypatched public `redact_text()`

### Service and caller regressions

Run and extend as needed:

- all `tldw_Server_API/tests/unit/test_moderation*.py` tests
- real `ModerationService` and moderation test-endpoint coverage
- `tldw_Server_API/tests/Guardian/test_supervised_policy.py`
- `tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py`, adding at least one production-mode case backed by a real configured `ModerationService`
- `tldw_Server_API/tests/Workflows/adapters/test_llm_adapters.py`, adding production-mode real-service check and redaction/count cases
- `tldw_Server_API/tests/Audio/test_audio_transcription_retention_and_redaction.py`, including its real STT redaction integration case

Stubbed caller tests prove the downstream contract only. They do not count as real evaluator integration coverage.

### Bounded performance safety

Long-text characterization uses deterministic bounded inputs and explicit timeouts. Extraction tests must not use intentionally catastrophic regular expressions. ReDoS hardening remains a separate task.

## Implementation Sequence

1. Add literal characterization tests against current `ModerationService` behavior.
2. Add `EvaluationLimits`, `PolicyEvaluator`, and direct unit tests.
3. Move helper logic while retaining exact evaluation and redaction algorithm differences.
4. Delegate service helpers and public methods while preserving the existing public dynamic-dispatch chain.
5. Remove duplicated logic bodies only; retain all compatibility delegates.
6. Run focused and caller-level regressions.
7. Run final compilation, security, diff, and mergeability checks.

Each implementation stage should follow TDD and be committed only when its focused tests pass.

## Verification Gates

Before the implementation pull request is ready:

- run Python compilation checks for touched Moderation modules and tests
- run the direct evaluator and characterization suites
- run all moderation unit tests
- run supervised-policy regressions
- run the named real-service moderation endpoint, Chat, Workflow, and STT cases plus their stubbed contract suites
- run `git diff --check`
- run Bandit over `tldw_Server_API/app/core/Moderation`
- fetch current `origin/dev`
- verify clean mergeability against current `origin/dev`
- confirm the diff contains no endpoint/schema contract changes and no unrelated refactor

## Rollout And Rollback

No feature flag is required because this is an internal structural extraction with unchanged public contracts.

Rollback is a normal pull-request revert. `ModerationService` remains the external facade, so reverting does not require caller migration or data repair.

## Follow-Up Work

Create separate reviewed tasks for behavior-changing work after the structural extraction:

- harden long-text redaction so the fallback guardrail covers the full regex execution path
- assess moving shared moderation policy/result models into a neutral module while preserving import compatibility
- reassess and remove private service delegates only after repository and external usage are understood

These follow-ups must not be folded into the first `PolicyEvaluator` implementation pull request.

## Risks And Mitigations

### Accidental semantic unification

Risk: shared helpers could make evaluation and redaction use the same preflight or scan behavior.

Mitigation: preserve separate operation paths and lock their differences with literal characterization tests.

### Circular imports

Risk: evaluator imports service-owned policy/result classes while the service imports the evaluator.

Mitigation: use postponed annotations, `TYPE_CHECKING`, and deferred runtime imports, following `PolicyCompiler`.

### Stale limits

Risk: a configured evaluator could retain old limits after service changes.

Mitigation: keep the evaluator stateless, build each snapshot while holding the service reload lock, and pass it explicitly. Public evaluation and redaction retain separate snapshots because they remain separate dynamically dispatched service calls.

### Tautological tests

Risk: service/evaluator parity passes because both paths call the same implementation.

Mitigation: write literal expected characterization cases before delegation and treat parity as supplemental.

### Hidden downstream breakage

Risk: callers depend on tuple ordering or fallback method availability.

Mitigation: retain all service signatures and run real-service plus stubbed caller-contract regressions.

### Public extension-point bypass

Risk: evaluator-owned nested redaction could bypass an overridden public `ModerationService.redact_text()` and return insufficiently transformed output.

Mitigation: service evaluation requests a decision-only evaluator result, then dynamically dispatches through `self.redact_text()` when redacted output is required. A characterization test locks this behavior before extraction.
