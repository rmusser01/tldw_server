# Moderation PolicyEvaluator Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract moderation decision evaluation and redaction into a stateless `PolicyEvaluator` behind `ModerationService` without changing supported service behavior or caller contracts.

**Architecture:** Add `policy_evaluator.py` with immutable `EvaluationLimits` and explicit-input evaluation, scan, snippet, and redaction operations. Keep policy/result dataclasses, configuration, locking, persistence, logging, and every public service call chain in `ModerationService`; service methods become compatibility delegates and evaluation-triggered redaction continues dispatching through public `self.redact_text()`.

**Tech Stack:** Python 3, dataclasses, `re`, threading, pytest, pytest-timeout, Loguru, Bandit, existing `ModerationPolicy`/`PatternRule`/`ModerationEvaluationResult` types, existing project virtualenv at `/Users/appledev/Documents/GitHub/tldw_server/.venv`.

## Global Constraints

- This pull request is a strict structural extraction; any behavior or security change is a separate follow-up pull request.
- Preserve every public `ModerationService` signature, return type, tuple order, and dynamic-dispatch path.
- Keep `ModerationPolicy`, `PatternRule`, and `ModerationEvaluationResult` defined and publicly importable from `moderation_service.py`.
- Keep all existing private compatibility helper names callable with their current class, static, or instance descriptors.
- Build a fresh lossless `EvaluationLimits` snapshot under the existing service `RLock` for each service-to-evaluator delegation.
- Keep evaluation scanning and redaction scanning as separate literal algorithms, including current unsupported-value and exception behavior.
- `PolicyEvaluator` owns no mutable state and performs no configuration lookup, file I/O, persistence, logging, or mutation of borrowed policy inputs.
- Do not change endpoint schemas, caller contracts, Guardian behavior, configuration persistence, logging, diagnostics, or regex hardening.
- Add literal characterization tests before production extraction and retain distinct direct-evaluator, service-delegation, and real-caller coverage.
- Track implementation and verification evidence in `TASK-12992`; run compilation, focused regressions, Bandit, diff, and current-`dev` mergeability gates before PR preparation.

---

## Source And Tracking

Design: `Docs/superpowers/specs/2026-07-20-moderation-policy-evaluator-refactor-design.md`

Design Backlog: `TASK-12990`

Plan Backlog: `TASK-12991`

Implementation Backlog: `TASK-12992`

## File Structure

- Create: `tldw_Server_API/app/core/Moderation/policy_evaluator.py`
  - Owns `EvaluationLimits`, `PolicyEvaluator`, and the exception tuple used by moved evaluation fallbacks.
  - Owns category/phase eligibility, snippet construction, exact chunk geometry, first-match lookup, action ranking, result construction, full-text redaction match collection, and sequential redaction.
  - Uses `TYPE_CHECKING` plus deferred runtime imports for service-owned dataclasses.
  - Performs no configuration lookup, file I/O, persistence, logging, or mutation of caller/service-owned inputs.
- Modify: `tldw_Server_API/app/core/Moderation/moderation_service.py`
  - Constructs one stateless evaluator.
  - Creates locked, lossless `EvaluationLimits` snapshots.
  - Retains all existing public and private method signatures and descriptor forms.
  - Preserves `check_text()`/`evaluate_text()` dispatch through `_evaluate_text_core()`.
  - Preserves evaluation-triggered dynamic dispatch through public `self.redact_text()`.
- Create: `tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py`
  - Literal pre-extraction service behavior oracle; must pass before production extraction starts.
- Create: `tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py`
  - Direct evaluator behavior and input-mutation tests.
- Create: `tldw_Server_API/tests/unit/test_moderation_policy_evaluator_delegation.py`
  - Service snapshot, descriptor, and dynamic-dispatch invariants.
- Modify: `tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py`
  - Adds one production-mode endpoint path backed by a real configured `ModerationService`.
- Modify: `tldw_Server_API/tests/Workflows/adapters/test_llm_adapters.py`
  - Adds production-mode real-service check and redaction/count cases.
- Verify unchanged: `tldw_Server_API/tests/unit/test_moderation_test_endpoint_sample.py`
  - Existing moderation endpoint uses the real service and canonical result.
- Verify unchanged: `tldw_Server_API/tests/Guardian/test_supervised_policy.py`
  - Existing policy-overlay compatibility suite.
- Verify unchanged: `tldw_Server_API/tests/Audio/test_audio_transcription_retention_and_redaction.py`
  - Existing STT test uses real moderation redaction with a stubbed STT provider.
- Modify: `backlog/tasks/task-12987 - Implement-Moderation-PolicyEvaluator-refactor.md`
  - Tracks task progress, exact touched files, review results, verification evidence, and final summary.

## Conventions

Run Python commands from the implementation worktree after activating the project virtualenv:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
```

Before each task commit:

1. Update `TASK-12992` through Backlog MCP with the task status, files, and focused test result.
2. Run the task's focused tests.
3. Run `git diff --check`.
4. Review the staged diff for behavior changes or unrelated edits.

Do not move shared policy/result dataclasses, harden regex behavior, remove private wrappers, alter endpoint schemas, or normalize currently unsupported values in this implementation.

---

### Task 1: Add Literal Decision And Dispatch Characterization

**Files:**
- Create: `tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py`
- Modify: `backlog/tasks/task-12987 - Implement-Moderation-PolicyEvaluator-refactor.md`

**Interfaces:**
- Consumes: existing `ModerationService`, `ModerationPolicy`, `PatternRule`, and `ModerationEvaluationResult` public and private behavior.
- Produces: a pre-extraction literal decision, snippet, policy-aliasing, and public-dispatch oracle that Task 5 must keep green without changing expected values.

- [ ] **Step 1: Add the shared service/policy fixtures**

Create the characterization file with fixtures that bypass configuration I/O while exercising the current service implementation:

```python
from __future__ import annotations

import re
import threading
from typing import Any

import pytest

from tldw_Server_API.app.core.Moderation.moderation_service import (
    ModerationEvaluationResult,
    ModerationPolicy,
    ModerationService,
    PatternRule,
)
from tldw_Server_API.app.core.Moderation.policy_compiler import PolicyCompiler


def _service(
    *,
    max_scan_chars: Any = 200_000,
    match_window_chars: Any = 4_096,
    max_fallback_scan_chars: Any = 800_000,
    max_replacements_per_pattern: Any = 1_000,
    service_type: type[ModerationService] = ModerationService,
) -> ModerationService:
    service = service_type.__new__(service_type)
    service._lock = threading.RLock()
    service._max_scan_chars = max_scan_chars
    service._match_window_chars = match_window_chars
    service._max_fallback_scan_chars = max_fallback_scan_chars
    service._max_replacements_per_pattern = max_replacements_per_pattern
    return service


def _rule(
    pattern: str,
    *,
    action: Any = None,
    replacement: str | None = None,
    categories: set[str] | None = None,
    phase: str = "both",
) -> PatternRule:
    return PatternRule(
        regex=re.compile(pattern),
        action=action,
        replacement=replacement,
        categories=categories,
        phase=phase,
    )


def _policy(
    *rules: PatternRule | re.Pattern[str],
    enabled: bool = True,
    input_enabled: bool = True,
    output_enabled: bool = True,
    input_action: Any = "block",
    output_action: Any = "redact",
    replacement: str = "[REDACTED]",
    categories_enabled: set[str] | None = None,
) -> ModerationPolicy:
    return ModerationPolicy(
        enabled=enabled,
        input_enabled=input_enabled,
        output_enabled=output_enabled,
        input_action=input_action,
        output_action=output_action,
        redact_replacement=replacement,
        per_user_overrides=False,
        block_patterns=list(rules),
        categories_enabled=categories_enabled,
    )
```

- [ ] **Step 2: Add literal phase, metadata, ranking, and category tests**

Append:

```python
@pytest.mark.parametrize(
    ("phase", "input_enabled", "output_enabled", "expected_action"),
    [
        ("input", True, True, "block"),
        ("output", True, True, "redact"),
        ("input", False, True, "pass"),
        ("output", True, False, "pass"),
        (None, True, True, "warn"),
        ("unknown", True, True, "warn"),
    ],
)
def test_service_evaluate_text_phase_characterization(
    phase,
    input_enabled,
    output_enabled,
    expected_action,
):
    policy = _policy(
        _rule("secret", phase="both"),
        input_enabled=input_enabled,
        output_enabled=output_enabled,
    )

    result = _service().evaluate_text("contains secret", policy, phase)

    assert result.action == expected_action
    assert result.matched_pattern == ("secret" if expected_action != "pass" else None)


@pytest.mark.parametrize("phase", [None, "unknown"])
@pytest.mark.parametrize("rule_phase", ["input", "output"])
def test_unknown_phase_bypasses_rule_phase_metadata(phase, rule_phase):
    result = _service().evaluate_text(
        "secret",
        _policy(_rule("secret", phase=rule_phase)),
        phase,
    )

    assert result.action == "warn"
    assert result.matched_pattern == "secret"


def test_raw_regex_bypasses_phase_and_category_metadata():
    policy = _policy(
        re.compile("secret"),
        categories_enabled={"allowed-only"},
    )

    result = _service().evaluate_text("secret", policy, "input")

    assert result.action == "block"
    assert result.category is None
    assert result.match_span == (0, 6)


def test_action_rank_then_position_then_rule_order_is_literal():
    policy = _policy(
        _rule("later", action="warn", categories={"zeta"}),
        _rule("early", action="block", categories={"pii", "confidential"}),
        _rule("early", action="block", categories={"other"}),
    )

    result = _service().evaluate_text("early ... later", policy, "input")

    assert result == ModerationEvaluationResult(
        action="block",
        redacted_text=None,
        matched_pattern="early",
        category="confidential",
        match_span=(0, 5),
        sample="[REDACTED] ... later",
    )


def test_equal_rank_prefers_earliest_match():
    policy = _policy(
        _rule("later", action="block"),
        _rule("early", action="block"),
    )

    result = _service().evaluate_text("early then later", policy, "input")

    assert result.matched_pattern == "early"
    assert result.match_span == (0, 5)


def test_redact_outranks_an_earlier_warn_match():
    result = _service().evaluate_text(
        "warn first, redact later",
        _policy(
            _rule("warn", action="warn"),
            _rule("later", action="redact"),
        ),
        "input",
    )

    assert result.action == "redact"
    assert result.matched_pattern == "later"
    assert result.match_span == (19, 24)


def test_uncategorized_and_wildcard_category_behavior_is_literal():
    uncategorized = _policy(
        _rule("first", action="warn"),
        categories_enabled={"uncategorized"},
    )
    wildcard = _policy(
        _rule("second", action="warn", categories={"*"}),
        categories_enabled={"restricted"},
    )

    assert _service().evaluate_text("first", uncategorized, "input").category == "uncategorized"
    assert _service().evaluate_text("second", wildcard, "input").action == "warn"


def test_enabled_category_wildcard_allows_specific_rule_category():
    result = _service().evaluate_text(
        "secret",
        _policy(
            _rule(
                "secret",
                action="warn",
                categories={"restricted"},
            ),
            categories_enabled={"*"},
        ),
        "input",
    )

    assert result.action == "warn"
    assert result.category is None


@pytest.mark.parametrize("categories_enabled", [None, set()])
def test_falsy_category_filters_allow_all_rules(categories_enabled):
    result = _service().evaluate_text(
        "secret",
        _policy(
            _rule(
                "secret",
                action="warn",
                categories={"confidential"},
            ),
            categories_enabled=categories_enabled,
        ),
        "input",
    )

    assert result.action == "warn"
    assert result.category == "confidential"


def test_enabled_categories_intersect_before_lexical_selection():
    result = _service().evaluate_text(
        "secret",
        _policy(
            _rule(
                "secret",
                action="warn",
                categories={"pii", "confidential", "financial"},
            ),
            categories_enabled={"pii", "financial"},
        ),
        "input",
    )

    assert result.category == "financial"


def test_disabled_policy_passes_but_direct_redaction_still_applies():
    policy = _policy(
        _rule("secret", action="redact", replacement="[R]"),
        enabled=False,
    )
    service = _service()

    assert service.evaluate_text("secret", policy, "input") == (
        ModerationEvaluationResult()
    )
    assert service.redact_text("secret", policy, "input") == "[R]"
```

- [ ] **Step 3: Add literal unsupported-action behavior**

Append:

```python
class _MissingLower:
    pass


class _LowerBytes:
    def lower(self):
        return b"block"


class _LowerBlock:
    def lower(self):
        return "block"


class _LowerUnhashable:
    def lower(self):
        return []


@pytest.mark.parametrize(
    ("action", "expected_action"),
    [
        (None, "block"),
        ("", "block"),
        ("unsupported", "warn"),
    ],
)
def test_falsy_and_unsupported_string_actions_are_literal(
    action,
    expected_action,
):
    result = _service().evaluate_text(
        "secret",
        _policy(_rule("secret", action=action)),
        "input",
    )

    assert result.action == expected_action


def test_effective_action_lower_result_behavior_is_literal():
    service = _service()

    bytes_result = service.evaluate_text(
        "secret",
        _policy(_rule("secret", action=_LowerBytes())),
        "input",
    )
    block_result = service.evaluate_text(
        "secret",
        _policy(_rule("secret", action=_LowerBlock())),
        "input",
    )

    assert bytes_result.action == "warn"
    assert block_result.action == "block"

    with pytest.raises(AttributeError):
        service.evaluate_text(
            "secret",
            _policy(_rule("secret", action=_MissingLower())),
            "input",
        )

    with pytest.raises(TypeError):
        service.evaluate_text(
            "secret",
            _policy(_rule("secret", action=_LowerUnhashable())),
            "input",
        )
```

- [ ] **Step 4: Add public dispatch characterization**

Append:

```python
def test_check_and_evaluate_dispatch_through_evaluate_text_core():
    calls = []

    class _DispatchService(ModerationService):
        def _evaluate_text_core(self, text, policy, phase, *, include_redacted_text):
            calls.append((text, phase, include_redacted_text))
            return ModerationEvaluationResult(action="warn", sample="[SAFE]")

    service = _DispatchService.__new__(_DispatchService)
    policy = _policy()

    assert service.check_text("probe", policy, "input") == (True, "[SAFE]")
    assert service.evaluate_text("probe", policy, "output").action == "warn"
    assert calls == [
        ("probe", "input", False),
        ("probe", "output", True),
    ]


def test_action_wrappers_dispatch_through_public_evaluate_text():
    calls = []

    class _DispatchService(ModerationService):
        def evaluate_text(self, text, policy, phase=None):
            calls.append((text, phase))
            return ModerationEvaluationResult(
                action="redact",
                redacted_text="[R]",
                matched_pattern="secret",
                category="confidential",
                match_span=(0, 6),
            )

    service = _DispatchService.__new__(_DispatchService)
    policy = _policy()

    assert service._evaluate_action_internal("secret", policy, "input") == (
        "redact", "[R]", "secret", "confidential", (0, 6)
    )
    assert service.evaluate_action("secret", policy, "input") == (
        "redact", "[R]", "secret", "confidential"
    )
    assert service.evaluate_action_with_match("secret", policy, "input") == (
        "redact", "[R]", "secret", "confidential", (0, 6)
    )
    assert calls == [
        ("secret", "input"),
        ("secret", "input"),
        ("secret", "input"),
    ]


def test_check_and_decision_only_core_do_not_invoke_public_redaction():
    class _NoRedactionService(ModerationService):
        def redact_text(self, text, policy, phase=None):
            raise AssertionError("redaction must not run")

    service = _service(service_type=_NoRedactionService)
    policy = _policy(_rule("secret", action="redact", replacement="[R]"))

    assert service.check_text("secret", policy, "input") == (True, "[R]")
    decision = service._evaluate_text_core(
        "secret",
        policy,
        "input",
        include_redacted_text=False,
    )

    assert decision.action == "redact"
    assert decision.redacted_text is None


def test_evaluation_dispatches_through_public_redact_text():
    class _DispatchService(ModerationService):
        def redact_text(self, text, policy, phase=None):
            return "[PUBLIC REDACTION]"

    service = _service(service_type=_DispatchService)

    result = service.evaluate_text(
        "secret",
        _policy(_rule("secret", action="redact")),
        "input",
    )

    assert result.redacted_text == "[PUBLIC REDACTION]"


def test_public_snippet_uses_first_matching_rule_replacement():
    policy = _policy(
        _rule("secret", replacement="[FIRST]"),
        _rule("secret", replacement="[SECOND]"),
        replacement="[POLICY]",
    )

    assert _service().build_sanitized_snippet(
        "before secret after",
        policy,
        (7, 13),
        pattern="secret",
    ) == "before [FIRST] after"


def test_effective_policy_identity_and_rule_aliasing_are_unchanged():
    base_rule = _rule("secret")
    policy = _policy(base_rule)
    policy.per_user_overrides = True
    service = _service()
    service._global_policy = policy
    service._user_overrides = {}

    assert service.get_effective_policy("user-1") is policy

    service._policy_compiler = PolicyCompiler()
    service._user_overrides = {
        "user-1": {"input_action": "warn"},
    }
    overlaid = service.get_effective_policy("user-1")

    assert overlaid is not policy
    assert overlaid.block_patterns is not policy.block_patterns
    assert overlaid.block_patterns[0] is base_rule
```

- [ ] **Step 5: Run the baseline decision suite**

Format and lint the new test file:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m black \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py
python -m ruff check \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py
```

Expected: Black completes successfully and Ruff reports `All checks passed!`.

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py \
  -q
```

Expected: PASS against the current pre-extraction `ModerationService`.

- [ ] **Step 6: Record and commit Task 1**

Update `TASK-12992`, then run:

```bash
git add \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py \
  'backlog/tasks/task-12987 - Implement-Moderation-PolicyEvaluator-refactor.md'
git commit -m "test: characterize moderation evaluation behavior"
```

---

### Task 2: Add Literal Scan, Redaction, And Limit Characterization

**Files:**
- Modify: `tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py`
- Modify: `backlog/tasks/task-12987 - Implement-Moderation-PolicyEvaluator-refactor.md`

**Interfaces:**
- Consumes: Task 1's `_service()`, `_rule()`, and `_policy()` characterization fixtures.
- Produces: the complete literal service oracle for scan geometry, limit coercion, redaction sequencing/counts, malformed raw rules, and borrowed-input immutability.

- [ ] **Step 1: Add direct-redaction and sequential-count behavior**

Append:

```python
@pytest.mark.parametrize("action", ["warn", "block", "redact"])
def test_direct_redaction_ignores_policy_enabled_and_rule_action(action):
    policy = _policy(
        _rule("secret", action=action, replacement="[RULE]"),
        enabled=False,
    )

    redacted, count = _service().redact_text_with_count(
        "secret and secret",
        policy,
        "input",
    )

    assert redacted == "[RULE] and [RULE]"
    assert count == 2


def test_sequential_redaction_applies_later_rules_to_changed_text():
    policy = _policy(
        _rule("secret", action="warn", replacement="token"),
        _rule("token", action="block", replacement="[FINAL]"),
    )

    redacted, count = _service().redact_text_with_count("secret", policy)

    assert redacted == "[FINAL]"
    assert count == 2


def test_replacement_text_is_literal_not_a_backreference():
    policy = _policy(
        _rule(r"(secret)", action="redact", replacement=r"\1-literal"),
    )

    assert _service().redact_text("secret", policy) == r"\1-literal"
```

- [ ] **Step 2: Add short/long replacement-limit asymmetry**

Append:

```python
@pytest.mark.parametrize(
    ("limit", "expected_text", "expected_count"),
    [
        (None, "[R] [R] [R]", 3),
        (0, "[R] [R] [R]", 3),
        (-1, "[R] [R] [R]", 3),
        ("2", "[R] [R] x", 2),
        ("bad", "[R] [R] [R]", 3),
    ],
)
def test_short_redaction_replacement_limit_characterization(
    limit,
    expected_text,
    expected_count,
):
    service = _service(
        max_scan_chars=100,
        max_replacements_per_pattern=limit,
    )
    policy = _policy(_rule("x", replacement="[R]"))

    assert service.redact_text_with_count("x x x", policy) == (
        expected_text,
        expected_count,
    )


@pytest.mark.parametrize(
    ("limit", "expected_text", "expected_count"),
    [
        (None, "[R] [R] [R]", 3),
        (0, "[R] [R] [R]", 3),
        (-1, "[R] [R] [R]", 3),
        (2, "[R] [R] x", 2),
    ],
)
def test_long_redaction_supported_limit_characterization(
    limit,
    expected_text,
    expected_count,
):
    service = _service(
        max_scan_chars=3,
        max_replacements_per_pattern=limit,
    )
    policy = _policy(_rule("x", replacement="[R]"))

    assert service.redact_text_with_count("x x x", policy) == (
        expected_text,
        expected_count,
    )


@pytest.mark.parametrize(
    ("limit", "error_type"),
    [
        ("2", TypeError),
        ("bad", ValueError),
    ],
)
def test_long_redaction_unsupported_limit_exception_characterization(
    limit,
    error_type,
):
    service = _service(
        max_scan_chars=3,
        max_replacements_per_pattern=limit,
    )

    with pytest.raises(error_type):
        service.redact_text_with_count(
            "x x x",
            _policy(_rule("x", replacement="[R]")),
        )
```

- [ ] **Step 3: Add exact chunk geometry and original-string search tests**

Append:

```python
class _RecordingPattern:
    pattern = "never"

    def __init__(self):
        self.bounds = []

    def search(self, _text, *bounds):
        self.bounds.append(bounds)
        return None


def test_chunk_geometry_and_search_bounds_are_literal():
    service = _service(
        max_scan_chars=10,
        match_window_chars=5,
        max_fallback_scan_chars=100,
    )
    chunks = list(service._iter_scan_chunks("x" * 25))
    pattern = _RecordingPattern()

    assert chunks[:3] == [(0, 10), (1, 11), (2, 12)]
    assert chunks[-1] == (15, 25)
    assert len(chunks) == 16

    assert service._find_match_span(pattern, "x" * 20) is None
    assert pattern.bounds == [
        (0, 15),
        (1, 16),
        (2, 17),
        (3, 18),
        (4, 19),
        (5, 20),
        (6, 20),
        (7, 20),
        (8, 20),
        (9, 20),
        (10, 20),
        (),
    ]


def test_original_string_search_preserves_lookbehind_before_pos():
    service = _service(
        max_scan_chars=1,
        match_window_chars=5,
        max_fallback_scan_chars=1,
    )

    assert service._find_match_span(
        re.compile(r"(?<=A)needle"),
        ("x" * 9) + "Aneedle",
    ) == (10, 16)


def test_chunk_search_does_not_turn_mid_text_anchor_into_start_anchor():
    service = _service(
        max_scan_chars=2,
        match_window_chars=10,
        max_fallback_scan_chars=100,
    )

    assert service._find_match_span(re.compile(r"^needle"), "xxneedle") is None


def test_full_text_fallback_limit_is_inclusive_and_guarded():
    pattern = re.compile(r"^needle$")

    assert _service(
        max_scan_chars=3,
        match_window_chars=0,
        max_fallback_scan_chars=6,
    )._find_match_span(pattern, "needle") == (0, 6)
    assert _service(
        max_scan_chars=3,
        match_window_chars=0,
        max_fallback_scan_chars=5,
    )._find_match_span(pattern, "needle") is None
```

- [ ] **Step 4: Add raw limit coercion characterization for all four fields**

Append:

```python
@pytest.mark.parametrize(
    ("raw", "error_type"),
    [
        (None, TypeError),
        ("bad", ValueError),
    ],
)
def test_evaluation_max_scan_coercion_errors_are_literal(raw, error_type):
    service = _service(max_scan_chars=raw)
    with pytest.raises(error_type):
        service._find_match_span(re.compile("x"), "x")
    with pytest.raises(error_type):
        list(service._iter_scan_chunks("x"))


def test_numeric_string_max_scan_is_coerced_for_evaluation():
    service = _service(max_scan_chars="2")
    assert service._find_match_span(re.compile("x"), "xx") == (0, 1)
    assert list(service._iter_scan_chunks("xx")) == [(0, 2)]


@pytest.mark.parametrize("raw", [None, "2", "bad"])
def test_redaction_path_comparison_does_not_coerce_max_scan(raw):
    service = _service(max_scan_chars=raw)
    with pytest.raises(TypeError):
        service.redact_text("x", _policy(_rule("x")))


@pytest.mark.parametrize(
    ("field", "raw", "error_type"),
    [
        ("_match_window_chars", None, TypeError),
        ("_match_window_chars", "bad", ValueError),
        ("_max_fallback_scan_chars", None, TypeError),
        ("_max_fallback_scan_chars", "bad", ValueError),
    ],
)
def test_long_evaluation_limit_coercion_errors_are_literal(
    field,
    raw,
    error_type,
):
    service = _service(max_scan_chars=1)
    setattr(service, field, raw)
    with pytest.raises(error_type):
        service._find_match_span(re.compile("never"), "long text")


def test_numeric_string_window_and_fallback_limits_are_coerced():
    service = _service(
        max_scan_chars=1,
        match_window_chars="2",
        max_fallback_scan_chars="20",
    )
    assert service._find_match_span(re.compile("never"), "long text") is None
```

- [ ] **Step 5: Add full-text long redaction and zero-length path differences**

Append:

```python
@pytest.mark.timeout(2)
def test_long_redaction_uses_bounded_full_text_finditer():
    service = _service(
        max_scan_chars=3,
        match_window_chars=0,
        max_replacements_per_pattern=10,
    )
    policy = _policy(_rule("ABCDE", replacement="[R]"))

    assert service.redact_text_with_count("xxABCDEyy", policy) == (
        "xx[R]yy",
        1,
    )


def test_zero_length_matches_differ_between_short_and_long_redaction():
    policy = _policy(_rule(r"(?=a)", replacement="[R]"))

    short = _service(max_scan_chars=10).redact_text_with_count("a", policy)
    long = _service(max_scan_chars=1).redact_text_with_count("aa", policy)

    assert short == ("[R]a", 1)
    assert long == ("aa", 0)


def test_malformed_raw_rule_exceptions_propagate():
    policy = _policy()
    policy.block_patterns = [None]  # type: ignore[list-item]
    service = _service()

    with pytest.raises(AttributeError):
        service.evaluate_text("secret", policy, "input")
    with pytest.raises(AttributeError):
        service.redact_text("secret", policy, "input")


class _RegexErrorPattern:
    pattern = "broken"

    def search(self, *_args, **_kwargs):
        raise re.error("broken")

    def finditer(self, *_args, **_kwargs):
        raise re.error("broken")

    def sub(self, *_args, **_kwargs):
        raise re.error("broken")

    def subn(self, *_args, **_kwargs):
        raise re.error("broken")


def test_regex_errors_keep_current_no_match_and_skip_behavior():
    policy = _policy()
    policy.block_patterns = [_RegexErrorPattern()]  # type: ignore[list-item]
    service = _service()

    assert service.evaluate_text("secret", policy, "input") == (
        ModerationEvaluationResult()
    )
    assert service.redact_text("secret", policy, "input") == "secret"
    assert service.redact_text_with_count(
        "secret",
        policy,
        "input",
    ) == ("secret", 0)


def test_replacement_lookup_regex_error_remains_inside_rule_boundary():
    class _ReplacementErrorPolicy:
        block_patterns = [re.compile("secret")]
        input_enabled = True
        output_enabled = True
        categories_enabled = None

        @property
        def redact_replacement(self):
            raise re.error("replacement lookup failed")

    service = _service()
    policy = _ReplacementErrorPolicy()

    assert service.redact_text(
        "secret",
        policy,  # type: ignore[arg-type]
    ) == "secret"
    assert service.redact_text_with_count(
        "secret",
        policy,  # type: ignore[arg-type]
    ) == ("secret", 0)


def test_service_evaluation_and_redaction_do_not_mutate_inputs():
    categories = {"confidential"}
    rule = _rule(
        "secret",
        action="redact",
        replacement="[R]",
        categories=categories,
    )
    rules = [rule]
    policy = _policy(*rules, categories_enabled={"confidential"})
    pattern_collection = policy.block_patterns
    service = _service()

    service.evaluate_text("secret", policy, "input")
    service.redact_text("secret", policy, "input")

    assert policy.block_patterns is pattern_collection
    assert policy.block_patterns[0] is rule
    assert rule.categories is categories
    assert categories == {"confidential"}
```

- [ ] **Step 6: Run the complete pre-extraction characterization suite**

Reformat and lint the completed characterization file:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m black \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py
python -m ruff check \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py
```

Expected: Black completes successfully and Ruff reports `All checks passed!`.

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py \
  -q
```

Expected: PASS. Do not begin evaluator implementation until every literal baseline case is green.

- [ ] **Step 7: Record and commit Task 2**

Update `TASK-12992`, then run:

```bash
git add \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py \
  'backlog/tasks/task-12987 - Implement-Moderation-PolicyEvaluator-refactor.md'
git commit -m "test: characterize moderation scan and redaction behavior"
```

---

### Task 3: Add EvaluationLimits And Direct Decision Evaluation

**Files:**
- Create: `tldw_Server_API/app/core/Moderation/policy_evaluator.py`
- Create: `tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py`
- Modify: `backlog/tasks/task-12987 - Implement-Moderation-PolicyEvaluator-refactor.md`

**Interfaces:**
- Consumes: service-owned `ModerationPolicy`, `PatternRule`, and `ModerationEvaluationResult` through deferred runtime imports, plus Task 2's literal oracle.
- Produces: frozen `EvaluationLimits`; stateless `PolicyEvaluator.evaluate_text(text, policy, phase, limits, *, include_redacted_text)`; category, phase, snippet, chunk, and first-match helpers used by Tasks 4 and 5.

- [ ] **Step 1: Add failing direct evaluator tests**

Create the direct test file:

```python
from __future__ import annotations

import re
from dataclasses import FrozenInstanceError

import pytest

from tldw_Server_API.app.core.Moderation.moderation_service import (
    ModerationEvaluationResult,
    ModerationPolicy,
    PatternRule,
)
from tldw_Server_API.app.core.Moderation.policy_evaluator import (
    EvaluationLimits,
    PolicyEvaluator,
)


LIMITS = EvaluationLimits(
    max_scan_chars=10,
    match_window_chars=5,
    max_fallback_scan_chars=100,
    max_replacements_per_pattern=10,
)


def _policy(*rules, **overrides):
    values = {
        "enabled": True,
        "input_enabled": True,
        "output_enabled": True,
        "input_action": "block",
        "output_action": "redact",
        "redact_replacement": "[REDACTED]",
        "per_user_overrides": False,
        "block_patterns": list(rules),
        "categories_enabled": None,
    }
    values.update(overrides)
    return ModerationPolicy(**values)


def _rule(
    pattern,
    *,
    action=None,
    replacement=None,
    categories=None,
    phase="both",
):
    return PatternRule(
        regex=re.compile(pattern),
        action=action,
        replacement=replacement,
        categories=categories,
        phase=phase,
    )


class _MissingLower:
    pass


class _LowerBytes:
    def lower(self):
        return b"block"


class _LowerBlock:
    def lower(self):
        return "block"


class _LowerUnhashable:
    def lower(self):
        return []


def test_evaluation_limits_are_frozen():
    with pytest.raises(FrozenInstanceError):
        LIMITS.max_scan_chars = 99


def test_direct_decision_evaluation_has_literal_result():
    policy = _policy(
        PatternRule(
            regex=re.compile("secret"),
            action="block",
            replacement="[RULE]",
            categories={"pii", "confidential"},
            phase="input",
        )
    )

    result = PolicyEvaluator().evaluate_text(
        "secret here",
        policy,
        "input",
        LIMITS,
        include_redacted_text=False,
    )

    assert result == ModerationEvaluationResult(
        action="block",
        redacted_text=None,
        matched_pattern="secret",
        category="confidential",
        match_span=(0, 6),
        sample="[RULE] here",
    )


@pytest.mark.parametrize(
    ("phase", "overrides", "expected_action"),
    [
        ("input", {}, "block"),
        ("output", {}, "redact"),
        ("input", {"input_enabled": False}, "pass"),
        ("output", {"output_enabled": False}, "pass"),
        (None, {}, "warn"),
        ("unknown", {}, "warn"),
    ],
)
def test_direct_phase_behavior_is_literal(
    phase,
    overrides,
    expected_action,
):
    result = PolicyEvaluator().evaluate_text(
        "secret",
        _policy(_rule("secret"), **overrides),
        phase,
        LIMITS,
        include_redacted_text=False,
    )

    assert result.action == expected_action
    assert result.matched_pattern == (
        "secret" if expected_action != "pass" else None
    )


@pytest.mark.parametrize("phase", [None, "unknown"])
@pytest.mark.parametrize("rule_phase", ["input", "output"])
def test_direct_unknown_phase_bypasses_rule_phase_metadata(
    phase,
    rule_phase,
):
    result = PolicyEvaluator().evaluate_text(
        "secret",
        _policy(_rule("secret", phase=rule_phase)),
        phase,
        LIMITS,
        include_redacted_text=False,
    )

    assert result.action == "warn"
    assert result.matched_pattern == "secret"


def test_direct_disabled_policy_and_raw_regex_behavior_are_literal():
    evaluator = PolicyEvaluator()
    disabled = evaluator.evaluate_text(
        "secret",
        _policy(_rule("secret"), enabled=False),
        "input",
        LIMITS,
        include_redacted_text=False,
    )
    raw = evaluator.evaluate_text(
        "secret",
        _policy(
            re.compile("secret"),
            categories_enabled={"restricted"},
        ),
        "input",
        LIMITS,
        include_redacted_text=False,
    )

    assert disabled == ModerationEvaluationResult()
    assert raw == ModerationEvaluationResult(
        action="block",
        redacted_text=None,
        matched_pattern="secret",
        category=None,
        match_span=(0, 6),
        sample="[REDACTED]",
    )


def test_direct_category_filter_ranking_and_tie_behavior_are_literal():
    policy = _policy(
        _rule(
            "later",
            action="warn",
            categories={"unselected"},
        ),
        _rule(
            "early",
            action="block",
            replacement="[FIRST]",
            categories={"pii", "financial", "confidential"},
        ),
        _rule(
            "early",
            action="block",
            replacement="[SECOND]",
            categories={"other"},
        ),
        categories_enabled={"pii", "confidential", "financial", "other"},
    )

    result = PolicyEvaluator().evaluate_text(
        "early ... later",
        policy,
        "input",
        LIMITS,
        include_redacted_text=False,
    )

    assert result == ModerationEvaluationResult(
        action="block",
        redacted_text=None,
        matched_pattern="early",
        category="confidential",
        match_span=(0, 5),
        sample="[FIRST] ... later",
    )


def test_direct_redact_rank_and_enabled_category_wildcard_are_literal():
    result = PolicyEvaluator().evaluate_text(
        "warn first, redact later",
        _policy(
            _rule(
                "warn",
                action="warn",
                categories={"first"},
            ),
            _rule(
                "later",
                action="redact",
                categories={"restricted"},
            ),
            categories_enabled={"*"},
        ),
        "input",
        LIMITS,
        include_redacted_text=False,
    )

    assert result.action == "redact"
    assert result.matched_pattern == "later"
    assert result.category is None
    assert result.match_span == (19, 24)


@pytest.mark.parametrize(
    ("action", "expected_action"),
    [
        (None, "block"),
        ("", "block"),
        ("unsupported", "warn"),
    ],
)
def test_direct_falsy_and_unsupported_string_actions_are_literal(
    action,
    expected_action,
):
    result = PolicyEvaluator().evaluate_text(
        "secret",
        _policy(_rule("secret", action=action)),
        "input",
        LIMITS,
        include_redacted_text=False,
    )

    assert result.action == expected_action


def test_direct_effective_action_lower_result_behavior_is_literal():
    evaluator = PolicyEvaluator()

    bytes_result = evaluator.evaluate_text(
        "secret",
        _policy(_rule("secret", action=_LowerBytes())),
        "input",
        LIMITS,
        include_redacted_text=False,
    )
    block_result = evaluator.evaluate_text(
        "secret",
        _policy(_rule("secret", action=_LowerBlock())),
        "input",
        LIMITS,
        include_redacted_text=False,
    )

    assert bytes_result.action == "warn"
    assert block_result.action == "block"

    with pytest.raises(AttributeError):
        evaluator.evaluate_text(
            "secret",
            _policy(_rule("secret", action=_MissingLower())),
            "input",
            LIMITS,
            include_redacted_text=False,
        )
    with pytest.raises(TypeError):
        evaluator.evaluate_text(
            "secret",
            _policy(_rule("secret", action=_LowerUnhashable())),
            "input",
            LIMITS,
            include_redacted_text=False,
        )


def test_direct_public_snippet_lookup_is_literal():
    policy = _policy(
        _rule("secret", replacement="[FIRST]"),
        _rule("secret", replacement="[SECOND]"),
        redact_replacement="[POLICY]",
    )

    assert PolicyEvaluator().build_sanitized_snippet(
        "before secret after",
        policy,
        (7, 13),
        pattern="secret",
    ) == "before [FIRST] after"


def test_direct_snippet_bounds_fallback_and_truncation_are_literal():
    evaluator = PolicyEvaluator()

    assert evaluator.build_sanitized_snippet_for_replacement(
        "secret",
        (-3, 99),
        "",
    ) == "[REDACTED]"

    long_snippet = evaluator.build_sanitized_snippet_for_replacement(
        ("a" * 20) + "secret" + ("b" * 20),
        (20, 26),
        "R" * 100,
    )

    assert long_snippet is not None
    assert len(long_snippet) == 80
    assert long_snippet.endswith("...")


def test_direct_scan_geometry_matches_characterized_behavior():
    evaluator = PolicyEvaluator()
    limits = EvaluationLimits(
        max_scan_chars=10,
        match_window_chars=5,
        max_fallback_scan_chars=100,
        max_replacements_per_pattern=10,
    )

    chunks = list(evaluator.iter_scan_chunks("x" * 25, limits))

    assert chunks[:3] == [(0, 10), (1, 11), (2, 12)]
    assert chunks[-1] == (15, 25)
    assert len(chunks) == 16


class _RecordingPattern:
    pattern = "never"

    def __init__(self):
        self.bounds = []

    def search(self, _text, *bounds):
        self.bounds.append(bounds)
        return None


class _RegexErrorPattern:
    pattern = "broken"

    def search(self, *_args, **_kwargs):
        raise re.error("broken")

    def finditer(self, *_args, **_kwargs):
        raise re.error("broken")

    def sub(self, *_args, **_kwargs):
        raise re.error("broken")

    def subn(self, *_args, **_kwargs):
        raise re.error("broken")


def test_direct_original_string_search_bounds_are_literal():
    evaluator = PolicyEvaluator()
    pattern = _RecordingPattern()

    assert evaluator.find_match_span(
        pattern,
        "x" * 20,
        LIMITS,
    ) is None
    assert pattern.bounds == [
        (0, 15),
        (1, 16),
        (2, 17),
        (3, 18),
        (4, 19),
        (5, 20),
        (6, 20),
        (7, 20),
        (8, 20),
        (9, 20),
        (10, 20),
        (),
    ]


def test_direct_lookbehind_anchor_and_fallback_behavior_is_literal():
    evaluator = PolicyEvaluator()
    lookbehind_limits = EvaluationLimits(1, 5, 1, 10)
    anchor_limits = EvaluationLimits(2, 10, 100, 10)

    assert evaluator.find_match_span(
        re.compile(r"(?<=A)needle"),
        ("x" * 9) + "Aneedle",
        lookbehind_limits,
    ) == (10, 16)
    assert evaluator.find_match_span(
        re.compile(r"^needle"),
        "xxneedle",
        anchor_limits,
    ) is None
    assert evaluator.find_match_span(
        re.compile(r"^needle$"),
        "needle",
        EvaluationLimits(3, 0, 6, 10),
    ) == (0, 6)
    assert evaluator.find_match_span(
        re.compile(r"^needle$"),
        "needle",
        EvaluationLimits(3, 0, 5, 10),
    ) is None


@pytest.mark.parametrize(
    ("raw", "error_type"),
    [
        (None, TypeError),
        ("bad", ValueError),
    ],
)
def test_direct_max_scan_coercion_errors_are_literal(raw, error_type):
    limits = EvaluationLimits(
        max_scan_chars=raw,  # type: ignore[arg-type]
        match_window_chars=5,
        max_fallback_scan_chars=100,
        max_replacements_per_pattern=10,
    )
    evaluator = PolicyEvaluator()

    with pytest.raises(error_type):
        evaluator.find_match_span(re.compile("x"), "x", limits)
    with pytest.raises(error_type):
        list(evaluator.iter_scan_chunks("x", limits))


@pytest.mark.parametrize(
    ("field", "raw", "error_type"),
    [
        ("match_window_chars", None, TypeError),
        ("match_window_chars", "bad", ValueError),
        ("max_fallback_scan_chars", None, TypeError),
        ("max_fallback_scan_chars", "bad", ValueError),
    ],
)
def test_direct_long_limit_coercion_errors_are_literal(
    field,
    raw,
    error_type,
):
    values = {
        "max_scan_chars": 1,
        "match_window_chars": 2,
        "max_fallback_scan_chars": 20,
        "max_replacements_per_pattern": 10,
    }
    values[field] = raw
    limits = EvaluationLimits(**values)  # type: ignore[arg-type]

    with pytest.raises(error_type):
        PolicyEvaluator().find_match_span(
            re.compile("never"),
            "long text",
            limits,
        )


def test_direct_numeric_string_limits_are_coerced_for_evaluation():
    limits = EvaluationLimits(
        max_scan_chars="1",  # type: ignore[arg-type]
        match_window_chars="2",  # type: ignore[arg-type]
        max_fallback_scan_chars="20",  # type: ignore[arg-type]
        max_replacements_per_pattern=10,
    )

    assert PolicyEvaluator().find_match_span(
        re.compile("never"),
        "long text",
        limits,
    ) is None


def test_direct_malformed_raw_rule_exception_propagates():
    policy = _policy(None)

    with pytest.raises(AttributeError):
        PolicyEvaluator().evaluate_text(
            "secret",
            policy,
            "input",
            LIMITS,
            include_redacted_text=False,
        )


def test_direct_empty_text_and_regex_error_behavior_are_literal():
    evaluator = PolicyEvaluator()
    policy = _policy(_rule("secret"))
    regex_error_policy = _policy(_RegexErrorPattern())

    assert evaluator.evaluate_text(
        "",
        policy,
        "input",
        LIMITS,
        include_redacted_text=False,
    ) == ModerationEvaluationResult()
    assert evaluator.evaluate_text(
        "secret",
        regex_error_policy,
        "input",
        LIMITS,
        include_redacted_text=False,
    ) == ModerationEvaluationResult()


def test_direct_evaluator_does_not_mutate_borrowed_inputs():
    categories = {"confidential"}
    rule = PatternRule(
        regex=re.compile("secret"),
        action="warn",
        categories=categories,
    )
    rules = [rule]
    policy = _policy(*rules, categories_enabled={"confidential"})
    pattern_collection = policy.block_patterns
    limits_before = EvaluationLimits(**vars(LIMITS))

    PolicyEvaluator().evaluate_text(
        "secret",
        policy,
        "input",
        LIMITS,
        include_redacted_text=False,
    )

    assert policy.block_patterns is pattern_collection
    assert policy.block_patterns == rules
    assert rule.categories is categories
    assert categories == {"confidential"}
    assert LIMITS == limits_before
```

- [ ] **Step 2: Run direct tests to verify they fail**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py \
  -q
```

Expected: collection FAIL with `ModuleNotFoundError` for `policy_evaluator`.

- [ ] **Step 3: Add evaluator imports, limits, lazy service types, and helpers**

Create `policy_evaluator.py`:

```python
"""Stateless moderation policy evaluation and redaction."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Moderation.moderation_service import (
        ModerationEvaluationResult,
        ModerationPolicy,
        PatternRule,
    )


_EVALUATION_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
    json.JSONDecodeError,
    re.error,
)


@dataclass(frozen=True)
class EvaluationLimits:
    max_scan_chars: int
    match_window_chars: int
    max_fallback_scan_chars: int
    max_replacements_per_pattern: int | None


class PolicyEvaluator:
    """Evaluate and redact text using explicit policy and limit inputs."""

    _UNCATEGORIZED_CATEGORY = "uncategorized"

    @staticmethod
    def policy_types() -> tuple[
        type[ModerationPolicy],
        type[PatternRule],
        type[ModerationEvaluationResult],
    ]:
        """Load service-owned dataclasses lazily to avoid import cycles."""

        from tldw_Server_API.app.core.Moderation.moderation_service import (
            ModerationEvaluationResult,
            ModerationPolicy,
            PatternRule,
        )

        return ModerationPolicy, PatternRule, ModerationEvaluationResult

    @classmethod
    def effective_rule_categories(cls, rule: PatternRule) -> set[str]:
        cats = rule.categories or set()
        normalized = {str(c).strip().lower() for c in cats if str(c).strip()}
        return normalized if normalized else {cls._UNCATEGORIZED_CATEGORY}

    @staticmethod
    def rule_applies_to_phase(rule: PatternRule, phase: str | None) -> bool:
        if phase not in {"input", "output"}:
            return True
        rule_phase = str(getattr(rule, "phase", "both") or "both").strip().lower()
        if rule_phase not in {"input", "output", "both"}:
            rule_phase = "both"
        return rule_phase in {"both", phase}

    @classmethod
    def rule_matches_enabled_categories(
        cls,
        rule: PatternRule,
        categories_enabled: set[str] | None,
    ) -> bool:
        if not categories_enabled or "*" in categories_enabled:
            return True
        rule_categories = cls.effective_rule_categories(rule)
        if "*" in rule_categories:
            return True
        return bool(rule_categories & categories_enabled)

    @staticmethod
    def build_sanitized_snippet_for_replacement(
        text: str,
        match_span: tuple[int, int],
        replacement: str,
    ) -> str | None:
        if not text or not match_span:
            return None
        start, end = match_span
        if start < 0:
            start = 0
        if end < start:
            end = start
        if start > len(text):
            start = len(text)
        if end > len(text):
            end = len(text)
        left_start = max(0, start - 16)
        right_end = min(len(text), end + 16)
        snippet = (
            text[left_start:start]
            + (replacement or "[REDACTED]")
            + text[end:right_end]
        ).strip()
        return snippet[:77] + "..." if len(snippet) > 80 else snippet

    def build_sanitized_snippet(
        self,
        text: str,
        policy: ModerationPolicy,
        match_span: tuple[int, int] | None,
        pattern: str | None = None,
    ) -> str | None:
        if not text or not match_span:
            return None
        _, PatternRule, _ = self.policy_types()
        replacement = policy.redact_replacement or "[REDACTED]"
        if pattern and policy.block_patterns:
            for rule in policy.block_patterns:
                if not isinstance(rule, PatternRule):
                    continue
                try:
                    if getattr(rule.regex, "pattern", None) == pattern:
                        if rule.replacement:
                            replacement = rule.replacement
                        break
                except _EVALUATION_NONCRITICAL_EXCEPTIONS:
                    continue
        return self.build_sanitized_snippet_for_replacement(
            text,
            match_span,
            replacement,
        )

    @staticmethod
    def iter_scan_chunks(
        text: str,
        limits: EvaluationLimits,
    ) -> Iterator[tuple[int, int]]:
        if not text:
            return
        chunk_size = max(1, int(limits.max_scan_chars))
        if len(text) <= chunk_size:
            yield 0, len(text)
            return
        overlap = min(1024, max(32, chunk_size // 10))
        if overlap >= chunk_size:
            overlap = max(0, chunk_size - 1)
        step = chunk_size - overlap if chunk_size > overlap else chunk_size
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(text_len, start + chunk_size)
            yield start, end
            if end == text_len:
                break
            start += step

    def find_match_span(
        self,
        pattern: re.Pattern[str],
        text: str,
        limits: EvaluationLimits,
    ) -> tuple[int, int] | None:
        try:
            chunk_limit = max(1, int(limits.max_scan_chars))
            if len(text) <= chunk_limit:
                match = pattern.search(text)
                return (match.start(), match.end()) if match else None
            text_len = len(text)
            window = max(0, int(limits.match_window_chars))
            for start, end in self.iter_scan_chunks(text, limits):
                window_end = min(text_len, end + window)
                match = pattern.search(text, start, window_end)
                if match and match.start() < end:
                    return match.start(), match.end()
            fallback_limit = max(1, int(limits.max_fallback_scan_chars))
            if len(text) <= fallback_limit:
                match = pattern.search(text)
                if match:
                    return match.start(), match.end()
            return None
        except re.error:
            return None
```

- [ ] **Step 4: Add exact decision evaluation**

Continue the class:

```python
    def evaluate_text(
        self,
        text: str,
        policy: ModerationPolicy,
        phase: str | None,
        limits: EvaluationLimits,
        *,
        include_redacted_text: bool,
    ) -> ModerationEvaluationResult:
        _, PatternRule, ModerationEvaluationResult = self.policy_types()
        if not text or not policy.enabled:
            return ModerationEvaluationResult()
        enabled_phase = True
        if phase == "input":
            enabled_phase = policy.input_enabled
        elif phase == "output":
            enabled_phase = policy.output_enabled
        if not enabled_phase:
            return ModerationEvaluationResult()
        default_action = "warn"
        if phase == "input":
            default_action = policy.input_action
        elif phase == "output":
            default_action = policy.output_action

        best_action = "pass"
        best_rank = 0
        best_pattern = None
        best_category = None
        best_match_pos = None
        best_match_span = None
        best_replacement = None
        for rule in policy.block_patterns or []:
            pattern = rule.regex if isinstance(rule, PatternRule) else rule
            if isinstance(rule, PatternRule) and not self.rule_applies_to_phase(rule, phase):
                continue
            if (
                isinstance(rule, PatternRule)
                and not self.rule_matches_enabled_categories(
                    rule,
                    policy.categories_enabled,
                )
            ):
                continue
            match_span = self.find_match_span(pattern, text, limits)
            if not match_span:
                continue
            action = (
                rule.action
                if isinstance(rule, PatternRule) and rule.action
                else default_action
            )
            action = (action or "warn").lower()
            if action not in {"block", "redact", "warn"}:
                action = "warn"
            rank = {"warn": 1, "redact": 2, "block": 3}.get(action, 1)
            match_pos = match_span[0]
            if rank > best_rank or (
                rank == best_rank
                and (best_match_pos is None or match_pos < best_match_pos)
            ):
                best_action = action
                best_rank = rank
                best_match_pos = match_pos
                best_match_span = match_span
                best_pattern = pattern.pattern
                best_replacement = (
                    rule.replacement
                    if isinstance(rule, PatternRule) and rule.replacement
                    else policy.redact_replacement
                )
                if isinstance(rule, PatternRule):
                    try:
                        categories = self.effective_rule_categories(rule)
                        if policy.categories_enabled:
                            categories &= set(policy.categories_enabled)
                        if categories and "pii" in categories and len(categories) > 1:
                            categories = {c for c in categories if c != "pii"}
                        best_category = (
                            sorted(categories)[0] if categories else None
                        )
                    except _EVALUATION_NONCRITICAL_EXCEPTIONS:
                        best_category = None
                else:
                    best_category = None

        if best_action == "pass" or best_match_span is None:
            return ModerationEvaluationResult()
        sample = self.build_sanitized_snippet_for_replacement(
            text,
            best_match_span,
            best_replacement or policy.redact_replacement or "[REDACTED]",
        )
        redacted_text = None
        if include_redacted_text and best_action == "redact":
            redacted_text = self.redact_text(text, policy, phase, limits)
        return ModerationEvaluationResult(
            action=best_action,
            redacted_text=redacted_text,
            matched_pattern=best_pattern,
            category=best_category,
            match_span=best_match_span,
            sample=sample,
        )
```

- [ ] **Step 5: Run direct decision and baseline characterization tests**

Format and lint the new evaluator and direct test:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m black \
  tldw_Server_API/app/core/Moderation/policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py
python -m ruff check \
  tldw_Server_API/app/core/Moderation/policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py
```

Expected: Black completes successfully and Ruff reports `All checks passed!`.

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py \
  -q
```

Expected: direct decision tests PASS; baseline service characterization remains PASS.

- [ ] **Step 6: Record and commit Task 3**

Update `TASK-12992`, then run:

```bash
git add \
  tldw_Server_API/app/core/Moderation/policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py \
  'backlog/tasks/task-12987 - Implement-Moderation-PolicyEvaluator-refactor.md'
git commit -m "refactor: add moderation policy decision evaluator"
```

---

### Task 4: Add Direct Evaluator Redaction And Nested-Limit Tests

**Files:**
- Modify: `tldw_Server_API/app/core/Moderation/policy_evaluator.py`
- Modify: `tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py`
- Modify: `backlog/tasks/task-12987 - Implement-Moderation-PolicyEvaluator-refactor.md`

**Interfaces:**
- Consumes: Task 3's `EvaluationLimits`, evaluator type loader, eligibility helpers, and direct decision evaluator.
- Produces: `PolicyEvaluator.redact_text(...) -> str`, `redact_text_with_count(...) -> tuple[str, int]`, `collect_rule_matches(...)`, and `apply_rule_redactions(...)` with literal service behavior.

- [ ] **Step 1: Add failing redaction and nested-call tests**

Append:

```python
def test_evaluate_without_redacted_text_never_invokes_redaction():
    class _NoRedactionEvaluator(PolicyEvaluator):
        def redact_text(self, *_args, **_kwargs):
            raise AssertionError("redaction must not run")

    result = _NoRedactionEvaluator().evaluate_text(
        "secret",
        _policy(
            PatternRule(
                regex=re.compile("secret"),
                action="redact",
                replacement="[R]",
            )
        ),
        "input",
        LIMITS,
        include_redacted_text=False,
    )

    assert result.action == "redact"
    assert result.redacted_text is None


def test_nested_redaction_receives_identical_limits_object():
    seen = []

    class _RecordingEvaluator(PolicyEvaluator):
        def redact_text(self, text, policy, phase, limits):
            seen.append(limits)
            return "[R]"

    result = _RecordingEvaluator().evaluate_text(
        "secret",
        _policy(
            PatternRule(regex=re.compile("secret"), action="redact")
        ),
        "input",
        LIMITS,
        include_redacted_text=True,
    )

    assert result.redacted_text == "[R]"
    assert seen == [LIMITS]
    assert seen[0] is LIMITS


def test_direct_redaction_is_sequential_and_action_agnostic():
    policy = _policy(
        PatternRule(
            regex=re.compile("secret"),
            action="warn",
            replacement="token",
        ),
        PatternRule(
            regex=re.compile("token"),
            action="block",
            replacement="[FINAL]",
        ),
        enabled=False,
    )

    assert PolicyEvaluator().redact_text_with_count(
        "secret",
        policy,
        None,
        LIMITS,
    ) == ("[FINAL]", 2)


@pytest.mark.timeout(2)
def test_direct_long_redaction_uses_full_text_finditer():
    limits = EvaluationLimits(
        max_scan_chars=3,
        match_window_chars=0,
        max_fallback_scan_chars=3,
        max_replacements_per_pattern=10,
    )
    policy = _policy(
        PatternRule(
            regex=re.compile("ABCDE"),
            action="warn",
            replacement="[R]",
        )
    )

    assert PolicyEvaluator().redact_text_with_count(
        "xxABCDEyy",
        policy,
        "input",
        limits,
    ) == ("xx[R]yy", 1)


@pytest.mark.parametrize(
    ("limit", "expected_text", "expected_count"),
    [
        (None, "[R] [R] [R]", 3),
        (0, "[R] [R] [R]", 3),
        (-1, "[R] [R] [R]", 3),
        ("2", "[R] [R] x", 2),
        ("bad", "[R] [R] [R]", 3),
    ],
)
def test_direct_short_redaction_limit_behavior_is_literal(
    limit,
    expected_text,
    expected_count,
):
    limits = EvaluationLimits(
        max_scan_chars=100,
        match_window_chars=5,
        max_fallback_scan_chars=100,
        max_replacements_per_pattern=limit,
    )

    assert PolicyEvaluator().redact_text_with_count(
        "x x x",
        _policy(_rule("x", replacement="[R]")),
        None,
        limits,
    ) == (expected_text, expected_count)


@pytest.mark.parametrize(
    ("limit", "expected_text", "expected_count"),
    [
        (None, "[R] [R] [R]", 3),
        (0, "[R] [R] [R]", 3),
        (-1, "[R] [R] [R]", 3),
        (2, "[R] [R] x", 2),
    ],
)
def test_direct_long_redaction_supported_limit_behavior_is_literal(
    limit,
    expected_text,
    expected_count,
):
    limits = EvaluationLimits(
        max_scan_chars=3,
        match_window_chars=5,
        max_fallback_scan_chars=100,
        max_replacements_per_pattern=limit,
    )

    assert PolicyEvaluator().redact_text_with_count(
        "x x x",
        _policy(_rule("x", replacement="[R]")),
        None,
        limits,
    ) == (expected_text, expected_count)


@pytest.mark.parametrize(
    ("limit", "error_type"),
    [
        ("2", TypeError),
        ("bad", ValueError),
    ],
)
def test_direct_long_redaction_unsupported_limit_errors_are_literal(
    limit,
    error_type,
):
    limits = EvaluationLimits(
        max_scan_chars=3,
        match_window_chars=5,
        max_fallback_scan_chars=100,
        max_replacements_per_pattern=limit,
    )

    with pytest.raises(error_type):
        PolicyEvaluator().redact_text_with_count(
            "x x x",
            _policy(_rule("x", replacement="[R]")),
            None,
            limits,
        )


def test_direct_short_and_long_zero_length_behavior_is_literal():
    policy = _policy(_rule(r"(?=a)", replacement="[R]"))

    assert PolicyEvaluator().redact_text_with_count(
        "a",
        policy,
        None,
        EvaluationLimits(10, 5, 100, 10),
    ) == ("[R]a", 1)
    assert PolicyEvaluator().redact_text_with_count(
        "aa",
        policy,
        None,
        EvaluationLimits(1, 5, 100, 10),
    ) == ("aa", 0)


@pytest.mark.parametrize("raw", [None, "2", "bad"])
def test_direct_redaction_path_does_not_coerce_max_scan(raw):
    limits = EvaluationLimits(
        max_scan_chars=raw,  # type: ignore[arg-type]
        match_window_chars=5,
        max_fallback_scan_chars=100,
        max_replacements_per_pattern=10,
    )

    with pytest.raises(TypeError):
        PolicyEvaluator().redact_text(
            "x",
            _policy(_rule("x")),
            None,
            limits,
        )


def test_direct_redaction_phase_gate_and_malformed_rule_are_literal():
    evaluator = PolicyEvaluator()
    policy = _policy(
        _rule("secret", replacement="[R]"),
        input_enabled=False,
    )

    assert evaluator.redact_text("secret", policy, "input", LIMITS) == "secret"

    malformed = _policy(None)
    with pytest.raises(AttributeError):
        evaluator.redact_text("secret", malformed, None, LIMITS)


def test_direct_redaction_empty_literal_replacement_and_regex_errors():
    evaluator = PolicyEvaluator()
    literal_policy = _policy(
        _rule(r"(secret)", replacement=r"\1-literal"),
    )
    regex_error_policy = _policy(_RegexErrorPattern())

    assert evaluator.redact_text("", literal_policy, None, LIMITS) == ""
    assert evaluator.redact_text(
        "secret",
        _policy(),
        None,
        LIMITS,
    ) == "secret"
    assert evaluator.redact_text(
        "secret",
        literal_policy,
        None,
        LIMITS,
    ) == r"\1-literal"
    assert evaluator.redact_text(
        "secret",
        regex_error_policy,
        None,
        LIMITS,
    ) == "secret"
    assert evaluator.redact_text_with_count(
        "secret",
        regex_error_policy,
        None,
        LIMITS,
    ) == ("secret", 0)


def test_direct_replacement_lookup_regex_error_is_skipped():
    class _ReplacementErrorPolicy:
        block_patterns = [re.compile("secret")]
        input_enabled = True
        output_enabled = True
        categories_enabled = None

        @property
        def redact_replacement(self):
            raise re.error("replacement lookup failed")

    evaluator = PolicyEvaluator()
    policy = _ReplacementErrorPolicy()

    assert evaluator.redact_text(
        "secret",
        policy,  # type: ignore[arg-type]
        None,
        LIMITS,
    ) == "secret"
    assert evaluator.redact_text_with_count(
        "secret",
        policy,  # type: ignore[arg-type]
        None,
        LIMITS,
    ) == ("secret", 0)


def test_direct_redaction_does_not_mutate_inputs_or_limits():
    categories = {"confidential"}
    rule = _rule(
        "secret",
        action="warn",
        replacement="[R]",
        categories=categories,
    )
    policy = _policy(rule, categories_enabled={"confidential"})
    pattern_collection = policy.block_patterns
    limits_before = EvaluationLimits(**vars(LIMITS))

    assert PolicyEvaluator().redact_text(
        "secret",
        policy,
        "input",
        LIMITS,
    ) == "[R]"

    assert policy.block_patterns is pattern_collection
    assert policy.block_patterns[0] is rule
    assert rule.categories is categories
    assert categories == {"confidential"}
    assert LIMITS == limits_before
```

- [ ] **Step 2: Run the redaction tests to verify they fail**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py \
  -q
```

Expected: FAIL because `PolicyEvaluator.redact_text()` and `redact_text_with_count()` are not implemented.

- [ ] **Step 3: Implement redaction and match application exactly**

Add to `PolicyEvaluator`:

```python
    @staticmethod
    def collect_rule_matches(
        text: str,
        pattern: re.Pattern[str],
        limits: EvaluationLimits,
    ) -> list[re.Match[str]]:
        if not text:
            return []
        limit = limits.max_replacements_per_pattern
        if limit is not None and int(limit) <= 0:
            limit = None
        matches = []
        try:
            for match in pattern.finditer(text):
                start, end = match.span()
                if start == end:
                    continue
                matches.append(match)
                if limit is not None and len(matches) >= limit:
                    break
        except re.error:
            return []
        return matches

    @staticmethod
    def apply_rule_redactions(
        text: str,
        matches: list[re.Match[str]],
        replacement: str,
    ) -> str:
        if not matches:
            return text
        parts = []
        last = 0
        for match in matches:
            start, end = match.span()
            if start < last:
                continue
            parts.append(text[last:start])
            parts.append(replacement)
            last = end
        parts.append(text[last:])
        return "".join(parts)

    def redact_text(
        self,
        text: str,
        policy: ModerationPolicy,
        phase: str | None,
        limits: EvaluationLimits,
    ) -> str:
        _, PatternRule, _ = self.policy_types()
        if not text or not policy.block_patterns:
            return text
        if phase == "input" and not policy.input_enabled:
            return text
        if phase == "output" and not policy.output_enabled:
            return text
        redacted = text
        for rule in policy.block_patterns:
            if isinstance(
                rule,
                PatternRule,
            ) and not self.rule_applies_to_phase(rule, phase):
                continue
            if isinstance(
                rule,
                PatternRule,
            ) and not self.rule_matches_enabled_categories(
                rule,
                policy.categories_enabled,
            ):
                continue
            pattern = rule.regex if isinstance(rule, PatternRule) else rule
            replacement_override = None
            if isinstance(rule, PatternRule) and rule.replacement:
                replacement_override = rule.replacement
            try:
                replacement = (
                    replacement_override or policy.redact_replacement
                )
                limit_raw = limits.max_replacements_per_pattern
                try:
                    limit = int(limit_raw) if limit_raw is not None else 0
                except _EVALUATION_NONCRITICAL_EXCEPTIONS:
                    limit = 0
                if limit <= 0:
                    limit = 0
                if len(redacted) <= limits.max_scan_chars:
                    redacted = pattern.sub(
                        lambda _match, value=replacement: value,
                        redacted,
                        count=limit,
                    )
                else:
                    matches = self.collect_rule_matches(
                        redacted,
                        pattern,
                        limits,
                    )
                    if matches:
                        redacted = self.apply_rule_redactions(
                            redacted,
                            matches,
                            replacement,
                        )
            except re.error:
                continue
        return redacted

    def redact_text_with_count(
        self,
        text: str,
        policy: ModerationPolicy,
        phase: str | None,
        limits: EvaluationLimits,
    ) -> tuple[str, int]:
        _, PatternRule, _ = self.policy_types()
        if not text or not policy.block_patterns:
            return text, 0
        if phase == "input" and not policy.input_enabled:
            return text, 0
        if phase == "output" and not policy.output_enabled:
            return text, 0
        redacted = text
        total_count = 0
        for rule in policy.block_patterns:
            if isinstance(rule, PatternRule) and not self.rule_applies_to_phase(
                rule,
                phase,
            ):
                continue
            if (
                isinstance(rule, PatternRule)
                and not self.rule_matches_enabled_categories(
                    rule,
                    policy.categories_enabled,
                )
            ):
                continue
            pattern = rule.regex if isinstance(rule, PatternRule) else rule
            replacement_override = None
            if isinstance(rule, PatternRule) and rule.replacement:
                replacement_override = rule.replacement
            try:
                replacement = (
                    replacement_override or policy.redact_replacement
                )
                limit_raw = limits.max_replacements_per_pattern
                try:
                    limit = int(limit_raw) if limit_raw is not None else 0
                except _EVALUATION_NONCRITICAL_EXCEPTIONS:
                    limit = 0
                if limit <= 0:
                    limit = 0
                if len(redacted) <= limits.max_scan_chars:
                    redacted, count = pattern.subn(
                        lambda _match, value=replacement: value,
                        redacted,
                        count=limit,
                    )
                else:
                    matches = self.collect_rule_matches(
                        redacted,
                        pattern,
                        limits,
                    )
                    count = len(matches)
                    if matches:
                        redacted = self.apply_rule_redactions(
                            redacted,
                            matches,
                            replacement,
                        )
                total_count += count
            except re.error:
                continue
        return redacted, total_count
```

Keep the two public evaluator redaction bodies separate. This deliberately preserves the current short-path `sub()` versus `subn()` calls, long-path count handling, and narrow `re.error` exception scope without introducing an internal counting abstraction.

- [ ] **Step 4: Run direct evaluator and characterization suites**

Reformat and lint the evaluator and direct test after redaction is added:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m black \
  tldw_Server_API/app/core/Moderation/policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py
python -m ruff check \
  tldw_Server_API/app/core/Moderation/policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py
```

Expected: Black completes successfully and Ruff reports `All checks passed!`.

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py \
  -q
```

Expected: PASS.

- [ ] **Step 5: Record and commit Task 4**

Update `TASK-12992`, then run:

```bash
git add \
  tldw_Server_API/app/core/Moderation/policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py \
  'backlog/tasks/task-12987 - Implement-Moderation-PolicyEvaluator-refactor.md'
git commit -m "refactor: add moderation policy redaction evaluator"
```

---

### Task 5: Delegate ModerationService While Preserving Dispatch

**Files:**
- Modify: `tldw_Server_API/app/core/Moderation/moderation_service.py`
- Modify: `tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py`
- Create: `tldw_Server_API/tests/unit/test_moderation_policy_evaluator_delegation.py`
- Modify: `backlog/tasks/task-12987 - Implement-Moderation-PolicyEvaluator-refactor.md`

**Interfaces:**
- Consumes: the complete stateless evaluator from Tasks 3-4 and the unchanged service oracle from Tasks 1-2.
- Produces: `ModerationService._evaluation_limits() -> EvaluationLimits`, one service-owned evaluator instance, public/service helper delegates, locked lossless snapshots, and preserved public dynamic dispatch.

- [ ] **Step 1: Add failing delegation and descriptor tests**

First update the characterization test setup now that Task 3 has created the evaluator module:

```python
from tldw_Server_API.app.core.Moderation.policy_evaluator import PolicyEvaluator
```

In `_service()`, add:

```python
    service._policy_evaluator = PolicyEvaluator()
```

Because every characterization service now comes from `_service()`, this
single setup-only change keeps all literal expectations untouched.

Create:

```python
from __future__ import annotations

import inspect
import re
import threading
from unittest.mock import Mock

import tldw_Server_API.app.core.Moderation.moderation_service as moderation_service_module
from tldw_Server_API.app.core.Moderation.moderation_service import (
    ModerationEvaluationResult,
    ModerationPolicy,
    ModerationService,
    PatternRule,
)
from tldw_Server_API.app.core.Moderation.policy_evaluator import (
    EvaluationLimits,
    PolicyEvaluator,
)


def _service() -> ModerationService:
    service = ModerationService.__new__(ModerationService)
    service._lock = threading.RLock()
    service._max_scan_chars = 10
    service._match_window_chars = 5
    service._max_fallback_scan_chars = 100
    service._max_replacements_per_pattern = 2
    service._policy_evaluator = PolicyEvaluator()
    return service


def _policy(action="redact"):
    return ModerationPolicy(
        enabled=True,
        input_action="block",
        output_action="redact",
        per_user_overrides=False,
        block_patterns=[
            PatternRule(
                regex=re.compile("secret"),
                action=action,
                replacement="[R]",
            )
        ],
    )


def test_evaluation_limits_copy_raw_values():
    service = _service()
    service._max_scan_chars = "10"
    service._match_window_chars = None
    service._max_fallback_scan_chars = object()
    service._max_replacements_per_pattern = "bad"

    limits = service._evaluation_limits()

    assert limits.max_scan_chars == "10"
    assert limits.match_window_chars is None
    assert limits.max_fallback_scan_chars is service._max_fallback_scan_chars
    assert limits.max_replacements_per_pattern == "bad"


def test_evaluation_limits_wait_for_service_lock():
    service = _service()
    started = threading.Event()
    completed = threading.Event()

    def snapshot():
        started.set()
        service._evaluation_limits()
        completed.set()

    with service._lock:
        thread = threading.Thread(target=snapshot)
        thread.start()
        assert started.wait(timeout=1)
        assert not completed.wait(timeout=0.05)
    thread.join(timeout=1)

    assert completed.is_set()


def test_evaluation_limits_never_observe_reload_partial_assignments(
    monkeypatch,
):
    service = _service()
    service._global_policy = _policy("block")
    service._user_overrides = {}
    partial_assignment = threading.Event()
    release_reload = threading.Event()
    snapshot_complete = threading.Event()
    observed = []
    load_calls = 0

    def controlled_load_global_policy():
        nonlocal load_calls
        load_calls += 1
        service._max_scan_chars = 20
        if load_calls == 1:
            partial_assignment.set()
            release_reload.wait(timeout=1)
        service._match_window_chars = 6
        service._max_fallback_scan_chars = 200
        service._max_replacements_per_pattern = 3
        return service._global_policy

    service._load_global_policy = controlled_load_global_policy
    service._load_runtime_overrides_file = lambda: None
    service._load_user_overrides = lambda: {}
    monkeypatch.setattr(
        moderation_service_module,
        "load_and_log_configs",
        lambda: {},
    )

    reload_thread = threading.Thread(target=service.reload)
    reload_thread.start()
    assert partial_assignment.wait(timeout=1)

    def snapshot():
        observed.append(service._evaluation_limits())
        snapshot_complete.set()

    snapshot_thread = threading.Thread(target=snapshot)
    snapshot_thread.start()
    assert not snapshot_complete.wait(timeout=0.05)
    release_reload.set()
    reload_thread.join(timeout=1)
    snapshot_thread.join(timeout=1)

    assert not reload_thread.is_alive()
    assert not snapshot_thread.is_alive()
    assert observed == [EvaluationLimits(20, 6, 200, 3)]


def test_service_evaluation_and_redaction_use_separate_snapshots():
    first = EvaluationLimits(10, 5, 100, 2)
    second = EvaluationLimits(20, 6, 200, 3)
    evaluator = Mock()
    evaluator.evaluate_text.return_value = ModerationEvaluationResult(
        action="redact",
        matched_pattern="secret",
        match_span=(0, 6),
    )
    evaluator.redact_text.return_value = "[R]"
    service = _service()
    service._policy_evaluator = evaluator
    service._evaluation_limits = Mock(side_effect=[first, second])

    result = service.evaluate_text("secret", _policy(), "input")

    assert result.redacted_text == "[R]"
    assert evaluator.evaluate_text.call_args.args[3] is first
    assert evaluator.evaluate_text.call_args.kwargs == {
        "include_redacted_text": False,
    }
    assert evaluator.redact_text.call_args.args[3] is second


def test_check_and_decision_only_core_do_not_invoke_public_redaction():
    service = _service()
    service.redact_text = Mock(
        side_effect=AssertionError("redaction must not run"),
    )
    policy = _policy("redact")

    assert service.check_text("secret", policy, "input") == (True, "[R]")
    decision = service._evaluate_text_core(
        "secret",
        policy,
        "input",
        include_redacted_text=False,
    )

    assert decision.action == "redact"
    assert decision.redacted_text is None
    service.redact_text.assert_not_called()


def test_private_helper_descriptors_are_preserved():
    assert isinstance(
        inspect.getattr_static(ModerationService, "_effective_rule_categories"),
        classmethod,
    )
    assert isinstance(
        inspect.getattr_static(
            ModerationService,
            "_rule_matches_enabled_categories",
        ),
        classmethod,
    )
    for name in (
        "_rule_applies_to_phase",
        "_build_sanitized_snippet",
        "_apply_rule_redactions",
    ):
        assert isinstance(
            inspect.getattr_static(ModerationService, name),
            staticmethod,
        )


def test_service_method_parameter_names_and_kinds_are_preserved():
    expected = {
        "check_text": ("self", "text", "policy", "phase"),
        "build_sanitized_snippet": (
            "self",
            "text",
            "policy",
            "match_span",
            "pattern",
        ),
        "redact_text": ("self", "text", "policy", "phase"),
        "redact_text_with_count": ("self", "text", "policy", "phase"),
        "evaluate_text": ("self", "text", "policy", "phase"),
        "_evaluate_text_core": (
            "self",
            "text",
            "policy",
            "phase",
            "include_redacted_text",
        ),
        "_evaluate_action_internal": ("self", "text", "policy", "phase"),
        "evaluate_action": ("self", "text", "policy", "phase"),
        "evaluate_action_with_match": ("self", "text", "policy", "phase"),
        "_iter_scan_chunks": ("self", "text"),
        "_find_match_span": ("self", "pat", "text"),
        "_collect_rule_matches": ("self", "text", "pat"),
        "_apply_rule_redactions": ("text", "matches", "replacement"),
    }

    for name, parameter_names in expected.items():
        descriptor = inspect.getattr_static(ModerationService, name)
        target = (
            descriptor.__func__
            if isinstance(descriptor, staticmethod)
            else descriptor
        )
        signature = inspect.signature(
            target,
        )
        assert tuple(signature.parameters) == parameter_names

    core_signature = inspect.signature(
        ModerationService._evaluate_text_core,
    )
    assert (
        core_signature.parameters["include_redacted_text"].kind
        is inspect.Parameter.KEYWORD_ONLY
    )
    for method_name, parameter_name in (
        ("check_text", "phase"),
        ("build_sanitized_snippet", "pattern"),
        ("redact_text", "phase"),
        ("redact_text_with_count", "phase"),
        ("evaluate_text", "phase"),
    ):
        signature = inspect.signature(
            getattr(ModerationService, method_name),
        )
        assert signature.parameters[parameter_name].default is None

    for method_name in (
        "_evaluate_action_internal",
        "evaluate_action",
        "evaluate_action_with_match",
    ):
        signature = inspect.signature(
            getattr(ModerationService, method_name),
        )
        assert (
            signature.parameters["phase"].default
            is inspect.Parameter.empty
        )
```

- [ ] **Step 2: Run delegation tests to verify they fail**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_delegation.py \
  -q
```

Expected: FAIL because `_evaluation_limits()` and service evaluator delegation are absent.

- [ ] **Step 3: Import and construct the evaluator**

In `moderation_service.py`:

```python
from dataclasses import dataclass, field, replace

from tldw_Server_API.app.core.Moderation.policy_evaluator import (
    EvaluationLimits,
    PolicyEvaluator,
)
```

After constructing `_lock` in `__init__`, construct one evaluator:

```python
self._policy_evaluator = PolicyEvaluator()
```

Alias the compatibility constant:

```python
class ModerationService:
    _UNCATEGORIZED_CATEGORY = PolicyEvaluator._UNCATEGORIZED_CATEGORY
```

- [ ] **Step 4: Add locked lossless snapshot construction**

Add:

```python
def _evaluation_limits(self) -> EvaluationLimits:
    with self._lock:
        return EvaluationLimits(
            max_scan_chars=self._max_scan_chars,
            match_window_chars=self._match_window_chars,
            max_fallback_scan_chars=self._max_fallback_scan_chars,
            max_replacements_per_pattern=self._max_replacements_per_pattern,
        )
```

Do not coerce values here. The dataclass annotations describe supported state but do not enforce runtime types.

- [ ] **Step 5: Replace service logic bodies with exact delegates**

Keep existing signatures and descriptors:

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

def build_sanitized_snippet(
    self,
    text: str,
    policy: ModerationPolicy,
    match_span: tuple[int, int] | None,
    pattern: str | None = None,
) -> str | None:
    return self._policy_evaluator.build_sanitized_snippet(
        text,
        policy,
        match_span,
        pattern,
    )

def redact_text(
    self,
    text: str,
    policy: ModerationPolicy,
    phase: str | None = None,
) -> str:
    return self._policy_evaluator.redact_text(
        text,
        policy,
        phase,
        self._evaluation_limits(),
    )

def redact_text_with_count(
    self,
    text: str,
    policy: ModerationPolicy,
    phase: str | None = None,
) -> tuple[str, int]:
    return self._policy_evaluator.redact_text_with_count(
        text,
        policy,
        phase,
        self._evaluation_limits(),
    )
```

Preserve the public evaluation call chain:

```python
def check_text(
    self,
    text: str,
    policy: ModerationPolicy,
    phase: str | None = None,
) -> tuple[bool, str | None]:
    result = self._evaluate_text_core(
        text,
        policy,
        phase,
        include_redacted_text=False,
    )
    return result.action != "pass", result.sample

def evaluate_text(
    self,
    text: str,
    policy: ModerationPolicy,
    phase: str | None = None,
) -> ModerationEvaluationResult:
    return self._evaluate_text_core(
        text,
        policy,
        phase,
        include_redacted_text=True,
    )

def _evaluate_text_core(
    self,
    text: str,
    policy: ModerationPolicy,
    phase: str | None,
    *,
    include_redacted_text: bool,
) -> ModerationEvaluationResult:
    decision = self._policy_evaluator.evaluate_text(
        text,
        policy,
        phase,
        self._evaluation_limits(),
        include_redacted_text=False,
    )
    if include_redacted_text and decision.action == "redact":
        return replace(
            decision,
            redacted_text=self.redact_text(text, policy, phase=phase),
        )
    return decision
```

Keep `_evaluate_action_internal()`, `evaluate_action()`, and `evaluate_action_with_match()` calling public `self.evaluate_text()` exactly as they do before extraction.

Delegate scan/redaction helpers without changing descriptors:

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
    return self._policy_evaluator.collect_rule_matches(
        text,
        pat,
        self._evaluation_limits(),
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

- [ ] **Step 6: Run delegation, direct, characterization, and existing moderation tests**

Format all new Python files and lint the touched Moderation scope:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m black \
  tldw_Server_API/app/core/Moderation/policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_delegation.py
python -m ruff check \
  tldw_Server_API/app/core/Moderation/policy_evaluator.py \
  tldw_Server_API/app/core/Moderation/moderation_service.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_delegation.py
```

Expected: Black completes successfully and Ruff reports `All checks passed!`.
Do not run Black across the existing `moderation_service.py`; it is not
currently Black-clean, and repository-wide formatting churn is outside scope.

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_delegation.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py \
  tldw_Server_API/tests/unit/test_moderation_check_text_snippet.py \
  tldw_Server_API/tests/unit/test_moderation_redact_categories.py \
  tldw_Server_API/tests/unit/test_moderation_blocklist_parse.py \
  -q
```

Expected: PASS.

- [ ] **Step 7: Record and commit Task 5**

Update `TASK-12992`, then run:

```bash
git add \
  tldw_Server_API/app/core/Moderation/moderation_service.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_delegation.py \
  'backlog/tasks/task-12987 - Implement-Moderation-PolicyEvaluator-refactor.md'
git commit -m "refactor: delegate moderation service policy evaluation"
```

---

### Task 6: Add Real-Service Chat And Workflow Regressions

**Files:**
- Modify: `tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py`
- Modify: `tldw_Server_API/tests/Workflows/adapters/test_llm_adapters.py`
- Modify: `backlog/tasks/task-12987 - Implement-Moderation-PolicyEvaluator-refactor.md`

**Interfaces:**
- Consumes: Task 5's unchanged `ModerationService` facade and existing Chat/Workflow dependency seams.
- Produces: production-mode Chat output-redaction coverage and Workflow check plus redaction/count coverage backed by a real configured service/evaluator.

- [ ] **Step 1: Add a real-service Chat endpoint test**

Add these imports:

```python
import threading

from tldw_Server_API.app.core.Moderation.moderation_service import (
    ModerationPolicy,
    ModerationService,
    PatternRule,
)
from tldw_Server_API.app.core.Moderation.policy_evaluator import PolicyEvaluator
```

Add this local helper:

```python
def _real_moderation_service(policy: ModerationPolicy) -> ModerationService:
    service = ModerationService.__new__(ModerationService)
    service._lock = threading.RLock()
    service._policy_evaluator = PolicyEvaluator()
    service._max_scan_chars = 200_000
    service._match_window_chars = 4_096
    service._max_fallback_scan_chars = 800_000
    service._max_replacements_per_pattern = 1_000
    service._global_policy = policy
    service._user_overrides = {}
    return service
```

Add:

```python
@pytest.mark.unit
def test_output_redaction_non_streaming_with_real_moderation_service(
    monkeypatch,
    credentialed_test_client_factory,
):
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TLDW_TEST_MODE", raising=False)
    db, db_path = _make_test_db()
    policy = ModerationPolicy(
        enabled=True,
        input_action="warn",
        output_action="redact",
        redact_replacement="[REDACTED]",
        per_user_overrides=False,
        block_patterns=[
            PatternRule(
                regex=re.compile("secret", re.IGNORECASE),
                action="redact",
                replacement="[RULE]",
                categories={"confidential"},
                phase="output",
            )
        ],
    )
    service = _real_moderation_service(policy)
    reply = {
        "id": "chatcmpl-real-moderation",
        "object": "chat.completion",
        "created": 123,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "this has secret token",
                },
                "finish_reason": "stop",
            }
        ],
    }
    try:
        app.dependency_overrides[get_chacha_db_for_user] = lambda: db
        with patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.get_moderation_service",
            return_value=service,
        ), patch(
            "tldw_Server_API.app.api.v1.endpoints.chat.perform_chat_api_call",
            return_value=reply,
        ):
            with credentialed_test_client_factory(app) as client:
                response = client.get("/api/v1/health")
                client.csrf_token = response.cookies.get("csrf_token", "")
                result = _post_with_csrf(
                    client,
                    "/api/v1/chat/completions",
                    json={
                        "api_provider": "openai",
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": False,
                    },
                    headers=_auth_headers(client),
                )
        assert result.status_code == 200
        assert (
            result.json()["choices"][0]["message"]["content"]
            == "this has [RULE] token"
        )
    finally:
        _cleanup_db_artifacts(db_path)
        app.dependency_overrides.pop(get_chacha_db_for_user, None)
```

- [ ] **Step 2: Add real-service Workflow adapter tests**

In the Workflow test file, add these imports:

```python
import re
import threading

from tldw_Server_API.app.core.Moderation.moderation_service import (
    ModerationPolicy,
    ModerationService,
    PatternRule,
)
from tldw_Server_API.app.core.Moderation.policy_evaluator import PolicyEvaluator
```

Add this local helper:

```python
def _real_moderation_service(policy: ModerationPolicy) -> ModerationService:
    service = ModerationService.__new__(ModerationService)
    service._lock = threading.RLock()
    service._policy_evaluator = PolicyEvaluator()
    service._max_scan_chars = 200_000
    service._match_window_chars = 4_096
    service._max_fallback_scan_chars = 800_000
    service._max_replacements_per_pattern = 1_000
    service._global_policy = policy
    service._user_overrides = {}
    return service
```

Add:

```python
@pytest.mark.asyncio
async def test_moderation_adapter_check_with_real_service(monkeypatch):
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TLDW_TEST_MODE", raising=False)
    policy = ModerationPolicy(
        enabled=True,
        input_action="block",
        output_action="redact",
        per_user_overrides=False,
        block_patterns=[
            PatternRule(
                regex=re.compile("secret"),
                action="block",
                categories={"confidential"},
                phase="input",
            )
        ],
    )
    service = _real_moderation_service(policy)
    import tldw_Server_API.app.core.Moderation.moderation_service as service_module
    monkeypatch.setattr(
        service_module,
        "get_moderation_service",
        lambda: service,
    )

    result = await run_moderation_adapter(
        {"action": "check", "text": "secret"},
        {"user_id": "1"},
    )

    assert result["allowed"] is False
    assert result["reason"] == "matched:confidential"
    assert result["matched_rules"] == ["secret"]
    assert result["action_recommended"] == "block"


@pytest.mark.asyncio
async def test_moderation_adapter_redact_with_real_service_counts_replacements(
    monkeypatch,
):
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TLDW_TEST_MODE", raising=False)
    policy = ModerationPolicy(
        enabled=False,
        redact_replacement="[REDACTED]",
        per_user_overrides=False,
        block_patterns=[
            PatternRule(
                regex=re.compile("secret"),
                action="warn",
                replacement="[RULE]",
            )
        ],
    )
    service = _real_moderation_service(policy)
    import tldw_Server_API.app.core.Moderation.moderation_service as service_module
    monkeypatch.setattr(
        service_module,
        "get_moderation_service",
        lambda: service,
    )

    result = await run_moderation_adapter(
        {"action": "redact", "text": "secret and secret"},
        {"user_id": "1"},
    )

    assert result["redacted_text"] == "[RULE] and [RULE]"
    assert result["text"] == "[RULE] and [RULE]"
    assert result["redaction_count"] == 2
```

- [ ] **Step 3: Run focused real-service caller tests**

Lint the modified caller tests without rewriting their pre-existing style:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m ruff check \
  tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py
python -m ruff check \
  --ignore I001,F401 \
  tldw_Server_API/tests/Workflows/adapters/test_llm_adapters.py
```

Expected: both commands report `All checks passed!`. `I001` and `F401` are
ignored only for the Workflow file because those findings already exist in
untouched lines; manually review the added import block for ordering and use.

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py::test_output_redaction_non_streaming_with_real_moderation_service \
  tldw_Server_API/tests/Workflows/adapters/test_llm_adapters.py::test_moderation_adapter_check_with_real_service \
  tldw_Server_API/tests/Workflows/adapters/test_llm_adapters.py::test_moderation_adapter_redact_with_real_service_counts_replacements \
  tldw_Server_API/tests/Audio/test_audio_transcription_retention_and_redaction.py::test_audio_transcriptions_redacts_text_and_segments_when_stt_redaction_enabled \
  tldw_Server_API/tests/unit/test_moderation_test_endpoint_sample.py::test_moderation_test_sample_matches_selected_rule
```

Expected: PASS.

- [ ] **Step 4: Run stubbed caller-contract suites**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py \
  -q
python -m pytest \
  tldw_Server_API/tests/Workflows/adapters/test_llm_adapters.py \
  -k moderation_adapter \
  -q
```

Expected: PASS. These tests remain contract coverage; they do not replace the real-service cases.

- [ ] **Step 5: Record and commit Task 6**

Update `TASK-12992`, then run:

```bash
git add \
  tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py \
  tldw_Server_API/tests/Workflows/adapters/test_llm_adapters.py \
  'backlog/tasks/task-12987 - Implement-Moderation-PolicyEvaluator-refactor.md'
git commit -m "test: cover moderation evaluator caller integration"
```

---

### Task 7: Final Verification, Review, And Scope Audit

**Files:**
- Modify: `backlog/tasks/task-12987 - Implement-Moderation-PolicyEvaluator-refactor.md`

**Interfaces:**
- Consumes: all implementation commits, literal/direct/delegation tests, and real/stubbed caller suites.
- Produces: compilation, regression, Bandit, scope, independent-review, and current-`dev` mergeability evidence recorded in finalized `TASK-12992`.

- [x] **Step 1: Compile every touched Python file**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m py_compile \
  tldw_Server_API/app/core/Moderation/policy_evaluator.py \
  tldw_Server_API/app/core/Moderation/moderation_service.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_delegation.py \
  tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py \
  tldw_Server_API/tests/Workflows/adapters/test_llm_adapters.py
```

Expected: exit 0 with no output.

- [x] **Step 2: Run final formatting and lint gates**

Run Black in check mode for the new files:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m black --check \
  tldw_Server_API/app/core/Moderation/policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_delegation.py
```

Run Ruff over every clean touched file:

```bash
python -m ruff check \
  tldw_Server_API/app/core/Moderation/policy_evaluator.py \
  tldw_Server_API/app/core/Moderation/moderation_service.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_delegation.py \
  tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py
python -m ruff check \
  --ignore I001,F401 \
  tldw_Server_API/tests/Workflows/adapters/test_llm_adapters.py
```

Expected: Black reports four unchanged files; both Ruff commands report
`All checks passed!`. The Workflow ignores cover only pre-existing import
findings; manually confirm the new imports are ordered and used.

- [x] **Step 3: Run evaluator and all moderation unit suites**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/unit/test_moderation*.py \
  -q
```

Expected: PASS with zero failures.

- [x] **Step 4: Run endpoint, Guardian, Chat, Workflow, and STT gates**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/unit/test_moderation_test_endpoint_sample.py \
  tldw_Server_API/tests/Guardian/test_supervised_policy.py \
  -q
python -m pytest \
  tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py \
  -q
python -m pytest \
  tldw_Server_API/tests/Workflows/adapters/test_llm_adapters.py \
  -k moderation_adapter \
  -q
python -m pytest \
  tldw_Server_API/tests/Audio/test_audio_transcription_retention_and_redaction.py::test_audio_transcriptions_redacts_text_and_segments_when_stt_redaction_enabled \
  -q
```

Expected: every command PASS with zero failures.

- [x] **Step 5: Run Bandit on the touched production scope**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m bandit \
  -r tldw_Server_API/app/core/Moderation \
  -f json \
  -o /tmp/bandit_moderation_policy_evaluator.json
```

Expected: exit 0 and no new findings in touched code. Record the JSON summary in `TASK-12992`.

- [x] **Step 6: Run diff and scope checks**

Run:

```bash
git diff --check
git status --short
git diff --name-status origin/dev...HEAD
```

Expected:

- no whitespace errors
- no unstaged implementation files
- only the evaluator, service, named tests, design/plan/task records, and no endpoint/schema production files

Review the final diff and confirm:

- `ModerationPolicy`, `PatternRule`, and `ModerationEvaluationResult` remain in `moderation_service.py`
- public method signatures and tuple ordering are unchanged
- `check_text()` and `evaluate_text()` retain `_evaluate_text_core()` dispatch
- action wrappers retain public `evaluate_text()` dispatch
- evaluation-triggered redaction retains public `self.redact_text()` dispatch
- private compatibility methods retain their descriptors
- evaluation and redaction scan paths remain distinct
- no logging, diagnostics, normalization, regex hardening, or shared-model move was added

- [x] **Step 7: Verify current-dev mergeability**

Run:

```bash
git fetch origin dev
git merge-tree --write-tree HEAD origin/dev
```

Expected: fetch succeeds; `git merge-tree` exits 0 and prints one tree object ID with no conflicts.

- [x] **Step 8: Request final code review**

Dispatch an independent code-review subagent with:

- approved design and this plan
- `origin/dev` as the base
- current `HEAD` as the review target
- explicit focus on behavioral drift, public dispatch, exception scopes, scan geometry, tautological tests, and missing caller coverage

Resolve every actionable finding and rerun the affected verification command.

- [x] **Step 9: Commit review fixes and rerun final gates**

When review requires implementation or test changes, stage the complete
touched scope and commit the resolved findings:

```bash
git add \
  tldw_Server_API/app/core/Moderation/policy_evaluator.py \
  tldw_Server_API/app/core/Moderation/moderation_service.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_characterization.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_delegation.py \
  tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py \
  tldw_Server_API/tests/Workflows/adapters/test_llm_adapters.py \
  'backlog/tasks/task-12987 - Implement-Moderation-PolicyEvaluator-refactor.md'
git commit -m "fix: address PolicyEvaluator review findings"
```

If the review is clean, record that result in `TASK-12992` and do not create
an empty commit. In either case, rerun Steps 1-7 on the final `HEAD`; the final
diff/scope and mergeability results must therefore include every review fix.

- [x] **Step 10: Finalize TASK-12992 and commit verification**

Update `TASK-12992` with:

- completed acceptance criteria and Definition of Done
- exact test counts and commands
- Bandit summary
- review findings and resolutions
- mergeability result
- modified files
- final human-authored change-summary requirement for the PR

Then run:

```bash
git add 'backlog/tasks/task-12987 - Implement-Moderation-PolicyEvaluator-refactor.md'
git commit -m "docs: record PolicyEvaluator verification"
```

## Self-Review Checklist

- [x] Every approved design requirement maps to a numbered task.
- [x] Characterization tests run green before production extraction.
- [x] Direct evaluator and service delegation tests are separate and non-tautological.
- [x] Exact scan geometry, original-string search, full-text redaction, zero-length differences, and unsupported limit/action behavior are represented.
- [x] Public and private compatibility boundaries match the approved design.
- [x] Real-service Chat, Workflow, endpoint, and STT coverage is distinct from stubbed caller contracts.
- [x] Every created or modified file has one clear responsibility.
- [x] Every implementation step includes exact code, command, expected result, and commit boundary.
- [x] Deferred-instruction scan is clean.
- [x] Type names, method signatures, task IDs, plan paths, and test selectors are consistent throughout.
- [x] Scope remains a strict structural extraction; behavior changes stay in follow-up tasks.
