# Moderation Policy Type Hook Retirement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the obsolete `PolicyCompiler.policy_types()` and `PolicyEvaluator.policy_types()` hooks while preserving supported Moderation behavior, canonical model identity, import isolation, and runtime namespace contracts.

**Architecture:** Keep `Moderation/models.py` as the neutral owner of the policy, rule, and evaluation-result dataclasses. Bind compiler and evaluator runtime operations directly to their existing underscore-prefixed model aliases, leaving `TYPE_CHECKING` imports and the `ModerationService` facade unchanged. Treat direct hook calls, subclass model substitution through the hook, tuple identity, and private alias rebinding as intentionally unsupported.

**Tech Stack:** Python 3.11+, dataclasses, `pytest`, Ruff, Black, Bandit, Git, Backlog.md

**Design:** `Docs/superpowers/specs/2026-10-02-moderation-policy-types-retirement-design.md`

**Backlog implementation task:** `TASK-13436`

---

## File Map

- Modify `tldw_Server_API/app/core/Moderation/policy_compiler.py`: remove the compiler hook and use `_ModerationPolicy` / `_PatternRule` at the four current runtime selection sites.
- Modify `tldw_Server_API/app/core/Moderation/policy_evaluator.py`: remove the evaluator hook, its cache import, and its now-unused `_ModerationPolicy` runtime import, then use `_PatternRule` / `_ModerationEvaluationResult` at the four current runtime selection sites.
- Modify `tldw_Server_API/tests/unit/test_moderation_models_characterization.py`: remove hook-shape assertions and invert the two subclass fixtures so legacy-named hooks must be ignored.
- Modify `tldw_Server_API/tests/unit/test_moderation_models_imports.py`: replace hook calls with representative operations, retain import-order and namespace contracts, and adapt service-export rebinding coverage.
- Modify `tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py`: remove the evaluator hook/cache-only characterization test and its now-unused `inspect` import.
- Update `backlog/tasks/task-13436 - Implement-Moderation-policy-types-retirement.md`: record execution status, verification evidence, touched files, and final summary.
- Do not modify `tldw_Server_API/app/core/Moderation/models.py`, `tldw_Server_API/app/core/Moderation/moderation_service.py`, endpoints, schemas, configuration, or downstream callers. Stop and revise the design if any such change appears necessary.

### Task 1: Establish The Tracked Baseline

**Files:**
- Update through Backlog MCP: `backlog/tasks/task-13436 - Implement-Moderation-policy-types-retirement.md`
- Read: `Docs/superpowers/specs/2026-10-02-moderation-policy-types-retirement-design.md`
- Read: `Docs/superpowers/plans/2026-10-03-moderation-policy-types-retirement-implementation-plan.md`

- [ ] **Step 1: Mark the implementation task in progress**

Use the Backlog MCP `task_edit` operation with project root `/Users/appledev/Documents/GitHub/tldw_server/.worktrees/moderation-policy-types-retirement`, task id `TASK-13436`, and status `In Progress`.

Expected: the task remains dependent on `TASK-13435`, and all six acceptance criteria remain unchecked.

- [ ] **Step 2: Confirm the branch and worktree are the intended ones**

Run:

```bash
git branch --show-current
git status --short --branch
```

Expected: branch `codex/moderation-policy-types-retirement`; only planning/Backlog changes already created by this workstream may be present before implementation begins.

- [ ] **Step 3: Run the approved focused baseline before editing runtime code**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/unit/test_moderation_policy_compiler.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_models_characterization.py \
  tldw_Server_API/tests/unit/test_moderation_models_canonical.py \
  tldw_Server_API/tests/unit/test_moderation_models_imports.py \
  -q
```

Expected: PASS. The approved-design baseline is 119 tests; if current `dev` has legitimately changed the count, record the new passing count in `TASK-13436` before proceeding.

### Task 2: Establish The Retired-Hook Contract With Tests

**Files:**
- Modify: `tldw_Server_API/tests/unit/test_moderation_models_characterization.py`
- Modify: `tldw_Server_API/tests/unit/test_moderation_models_imports.py`
- Modify: `tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py`

- [ ] **Step 1: Replace the hook descriptor and subclass-substitution tests with inverse regression fixtures**

In `tldw_Server_API/tests/unit/test_moderation_models_characterization.py`, delete `test_policy_type_descriptors_and_tuples_are_literal`, `test_compiler_uses_overridden_policy_types`, and `test_evaluator_uses_overridden_policy_types`. Insert these two tests in their place:

```python
def test_compiler_ignores_legacy_model_type_hook():
    class ReplacementPolicy:
        def __init__(self, **values):
            self.values = values

    class LegacyReplacementCompiler(PolicyCompiler):
        @staticmethod
        def policy_types():
            return ReplacementPolicy, PatternRule

    compiler = LegacyReplacementCompiler()
    global_result = compiler.compile_global(
        PolicyCompilationInput(
            config=ResolvedModerationConfig(),
            runtime_override={},
            blocklist_lines=["secret -> block #confidential"],
            pii_rules=[],
        )
    )

    assert type(global_result.policy) is ModerationPolicy
    assert type(global_result.policy.block_patterns[0]) is PatternRule

    user_result = compiler.compile_user_policy(
        global_result.policy,
        {
            "rules": [
                {
                    "pattern": "token",
                    "is_regex": False,
                    "action": "warn",
                    "phase": "input",
                }
            ]
        },
    )

    assert type(user_result.policy) is ModerationPolicy
    assert all(type(rule) is PatternRule for rule in user_result.policy.block_patterns)


def test_evaluator_ignores_legacy_model_type_hook():
    class ReplacementResult:
        def __init__(self, **values):
            self.values = values

    class LegacyReplacementEvaluator(PolicyEvaluator):
        @staticmethod
        def policy_types():
            return ModerationPolicy, PatternRule, ReplacementResult

    result = LegacyReplacementEvaluator().evaluate_text(
        "secret",
        ModerationPolicy(
            enabled=True,
            block_patterns=[
                PatternRule(
                    regex=re.compile("secret"),
                    action="block",
                    categories={"confidential"},
                    phase="input",
                )
            ],
        ),
        "input",
        _LIMITS,
        include_redacted_text=False,
    )

    assert type(result) is ModerationEvaluationResult
    assert result.action == "block"
```

These fixtures intentionally keep only local, legacy-named `policy_types()` methods. They specify that actual compiler/evaluator operations ignore that old extension seam without asserting private class layout.

- [ ] **Step 2: Run the two inverse tests and prove they are red**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/unit/test_moderation_models_characterization.py::test_compiler_ignores_legacy_model_type_hook \
  tldw_Server_API/tests/unit/test_moderation_models_characterization.py::test_evaluator_ignores_legacy_model_type_hook \
  -q
```

Expected: FAIL for both tests. The compiler returns `ReplacementPolicy`; the evaluator returns `ReplacementResult`. A collection error or unrelated exception is not an acceptable red state.

- [ ] **Step 3: Replace clean-process hook probes with real operations**

In `tldw_Server_API/tests/unit/test_moderation_models_imports.py`, remove `import inspect`. Replace the `script` parameter values and test name at the top of the file with:

```python
@pytest.mark.parametrize(
    "script",
    [
        f"""
import sys
from tldw_Server_API.app.core.Moderation.models import ModerationPolicy
from tldw_Server_API.app.core.Moderation.policy_compiler import (
    PolicyCompilationInput,
    PolicyCompiler,
    ResolvedModerationConfig,
)
assert {_SERVICE_MODULE!r} not in sys.modules
result = PolicyCompiler().compile_global(
    PolicyCompilationInput(config=ResolvedModerationConfig())
)
assert type(result.policy) is ModerationPolicy
assert {_SERVICE_MODULE!r} not in sys.modules
from tldw_Server_API.app.core.Moderation import models, moderation_service
assert moderation_service.ModerationPolicy is models.ModerationPolicy
assert moderation_service.PatternRule is models.PatternRule
""",
        f"""
import sys
from tldw_Server_API.app.core.Moderation.models import (
    ModerationEvaluationResult,
    ModerationPolicy,
)
from tldw_Server_API.app.core.Moderation.policy_evaluator import (
    EvaluationLimits,
    PolicyEvaluator,
)
assert {_SERVICE_MODULE!r} not in sys.modules
limits = EvaluationLimits(
    max_scan_chars=1024,
    match_window_chars=128,
    max_fallback_scan_chars=4096,
    max_replacements_per_pattern=None,
)
result = PolicyEvaluator().evaluate_text(
    "",
    ModerationPolicy(enabled=False),
    "input",
    limits,
    include_redacted_text=False,
)
assert type(result) is ModerationEvaluationResult
assert {_SERVICE_MODULE!r} not in sys.modules
from tldw_Server_API.app.core.Moderation import models, moderation_service
assert moderation_service.ModerationPolicy is models.ModerationPolicy
assert moderation_service.PatternRule is models.PatternRule
assert moderation_service.ModerationEvaluationResult is models.ModerationEvaluationResult
""",
    ],
)
def test_representative_operations_do_not_load_service(script):
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
```

- [ ] **Step 4: Remove staticmethod-only import tests and adapt service rebinding coverage**

Delete these tests from `tldw_Server_API/tests/unit/test_moderation_models_imports.py`:

```python
def test_compiler_policy_types_remains_staticmethod():
    assert isinstance(
        inspect.getattr_static(policy_compiler.PolicyCompiler, "policy_types"),
        staticmethod,
    )


def test_evaluator_policy_types_remains_staticmethod():
    assert isinstance(
        inspect.getattr_static(policy_evaluator.PolicyEvaluator, "policy_types"),
        staticmethod,
    )
```

Replace `test_service_export_rebinding_does_not_replace_canonical_policy_types` with:

```python
def test_service_export_rebinding_does_not_replace_canonical_runtime_outputs(monkeypatch):
    monkeypatch.setattr(moderation_service, "ModerationPolicy", type("Policy", (), {}))
    monkeypatch.setattr(moderation_service, "PatternRule", type("Rule", (), {}))
    monkeypatch.setattr(
        moderation_service,
        "ModerationEvaluationResult",
        type("Result", (), {}),
    )

    compiled = policy_compiler.PolicyCompiler().compile_global(
        policy_compiler.PolicyCompilationInput(
            config=policy_compiler.ResolvedModerationConfig()
        )
    )
    evaluated = policy_evaluator.PolicyEvaluator().evaluate_text(
        "",
        ModerationPolicy(enabled=False),
        "input",
        policy_evaluator.EvaluationLimits(
            max_scan_chars=1024,
            match_window_chars=128,
            max_fallback_scan_chars=4096,
            max_replacements_per_pattern=None,
        ),
        include_redacted_text=False,
    )

    assert type(compiled.policy) is ModerationPolicy
    assert type(evaluated) is ModerationEvaluationResult
```

Keep the complete import-order identity tests, public-runtime-namespace exclusion tests, and unresolved-runtime-type-hint tests unchanged.

- [ ] **Step 5: Remove evaluator hook/cache characterization**

In `tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py`, delete `import inspect` and delete this complete test:

```python
def test_direct_policy_type_loader_and_evaluator_shape_are_literal():
    descriptor = inspect.getattr_static(PolicyEvaluator, "policy_types")
    evaluator = PolicyEvaluator()

    assert isinstance(descriptor, staticmethod)
    assert evaluator.policy_types() == (
        ModerationPolicy,
        PatternRule,
        ModerationEvaluationResult,
    )
    assert evaluator.policy_types() is evaluator.policy_types()
    assert vars(evaluator) == {}
```

Do not add `hasattr`, `__dict__`, descriptor, signature, tuple-value, or cache-identity assertions for the retired methods.

- [ ] **Step 6: Prove the replacement import oracle is green before production changes**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/unit/test_moderation_models_imports.py \
  -q
```

Expected: PASS on the pre-retirement production implementation. This separates import-isolation proof from the red subclass behavior.

### Task 3: Remove The Hooks And Bind Canonical Private Aliases

**Files:**
- Modify: `tldw_Server_API/app/core/Moderation/policy_compiler.py`
- Modify: `tldw_Server_API/app/core/Moderation/policy_evaluator.py`
- Test: `tldw_Server_API/tests/unit/test_moderation_models_characterization.py`
- Test: `tldw_Server_API/tests/unit/test_moderation_models_imports.py`
- Test: `tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py`

- [ ] **Step 1: Remove the compiler hook and replace its four internal lookups**

Apply these exact changes in `tldw_Server_API/app/core/Moderation/policy_compiler.py`:

```diff
 class PolicyCompiler:
     """Compile moderation policies from resolved config and rule inputs."""

     _ALLOWED_REGEX_FLAGS = {"i", "m", "s", "x"}
     _ALLOWED_ACTIONS = {"block", "redact", "warn"}

-    @staticmethod
-    def policy_types() -> tuple[type[ModerationPolicy], type[PatternRule]]:
-        """Return the canonical policy dataclasses without loading the service."""
-
-        return _ModerationPolicy, _PatternRule
-
     def compile_global(self, data: PolicyCompilationInput) -> PolicyCompilationResult:
         """Compile the global moderation policy from config and blocklist input."""

         report = PolicyCompilationReport()
         config = data.config
-        ModerationPolicy, _ = self.policy_types()
         categories_enabled = self.resolve_runtime_categories(
             data.runtime_override,
             config.categories_enabled,
         )
         pii_enabled = self.resolve_runtime_pii(data.runtime_override, config.pii_enabled)
         block_patterns = self.compile_blocklist_lines(data.blocklist_lines, report)
         if pii_enabled:
             block_patterns.extend(list(data.pii_rules or []))

-        policy = ModerationPolicy(
+        policy = _ModerationPolicy(
@@
     def compile_user_policy(
@@
         report = PolicyCompilationReport()
         if not override:
             return PolicyCompilationResult(policy=base_policy, report=report)
-        ModerationPolicy, _ = self.policy_types()
-        policy = ModerationPolicy(
+        policy = _ModerationPolicy(
@@
     def compile_rule_expression(
@@
-        _, PatternRule = self.policy_types()
         try:
@@
-        return PatternRule(
+        return _PatternRule(
@@
     def compile_user_rule(
@@
-        _, PatternRule = self.policy_types()
         try:
@@
-        return PatternRule(
+        return _PatternRule(
```

Retain both `TYPE_CHECKING` imports and the two private runtime alias imports exactly as they are. Do not rename public annotations to underscore-prefixed names.

- [ ] **Step 2: Remove the evaluator hook/cache and replace its four internal lookups**

Apply these exact changes in `tldw_Server_API/app/core/Moderation/policy_evaluator.py`:

```diff
 import json
 import re
 from collections.abc import Iterator
 from dataclasses import dataclass
-from functools import lru_cache
 from typing import TYPE_CHECKING
@@
-from tldw_Server_API.app.core.Moderation.models import (
-    ModerationPolicy as _ModerationPolicy,
-)
 @@
 class PolicyEvaluator:
     """Evaluate and redact text using explicit policy and limit inputs."""

     _UNCATEGORIZED_CATEGORY = "uncategorized"

-    @staticmethod
-    @lru_cache(maxsize=1)
-    def policy_types() -> tuple[
-        type[ModerationPolicy],
-        type[PatternRule],
-        type[ModerationEvaluationResult],
-    ]:
-        """Return canonical policy dataclasses without loading the service."""
-
-        return _ModerationPolicy, _PatternRule, _ModerationEvaluationResult
-
     @classmethod
@@
     def build_sanitized_snippet(
@@
-        _, PatternRule, _ = self.policy_types()
         replacement = policy.redact_replacement or "[REDACTED]"
         if pattern and policy.block_patterns:
             for rule in policy.block_patterns:
-                if not isinstance(rule, PatternRule):
+                if not isinstance(rule, _PatternRule):
@@
     def redact_text(
@@
-        _, PatternRule, _ = self.policy_types()
         if not text or not policy.block_patterns:
@@
-                PatternRule,
+                _PatternRule,
@@
-                PatternRule,
+                _PatternRule,
@@
-            pattern = rule.regex if isinstance(rule, PatternRule) else rule
+            pattern = rule.regex if isinstance(rule, _PatternRule) else rule
             replacement_override = None
-            if isinstance(rule, PatternRule) and rule.replacement:
+            if isinstance(rule, _PatternRule) and rule.replacement:
@@
     def redact_text_with_count(
@@
-        _, PatternRule, _ = self.policy_types()
         if not text or not policy.block_patterns:
@@
-            if isinstance(rule, PatternRule) and not self.rule_applies_to_phase(
+            if isinstance(rule, _PatternRule) and not self.rule_applies_to_phase(
@@
-            if isinstance(rule, PatternRule) and not self.rule_matches_enabled_categories(
+            if isinstance(rule, _PatternRule) and not self.rule_matches_enabled_categories(
@@
-            pattern = rule.regex if isinstance(rule, PatternRule) else rule
+            pattern = rule.regex if isinstance(rule, _PatternRule) else rule
             replacement_override = None
-            if isinstance(rule, PatternRule) and rule.replacement:
+            if isinstance(rule, _PatternRule) and rule.replacement:
@@
     def evaluate_text(
@@
-        _, PatternRule, ModerationEvaluationResult = self.policy_types()
         if not text or not policy.enabled:
-            return ModerationEvaluationResult()
+            return _ModerationEvaluationResult()
@@
         if not enabled_phase:
-            return ModerationEvaluationResult()
+            return _ModerationEvaluationResult()
@@
         best_replacement = None
         for rule in policy.block_patterns or []:
-            pattern = rule.regex if isinstance(rule, PatternRule) else rule
-            if isinstance(rule, PatternRule) and not self.rule_applies_to_phase(rule, phase):
+            pattern = rule.regex if isinstance(rule, _PatternRule) else rule
+            if isinstance(rule, _PatternRule) and not self.rule_applies_to_phase(rule, phase):
                 continue
-            if isinstance(rule, PatternRule) and not self.rule_matches_enabled_categories(
+            if isinstance(rule, _PatternRule) and not self.rule_matches_enabled_categories(
@@
-            action = rule.action if isinstance(rule, PatternRule) and rule.action else default_action
+            action = rule.action if isinstance(rule, _PatternRule) and rule.action else default_action
@@
-                    if isinstance(rule, PatternRule) and rule.replacement
+                    if isinstance(rule, _PatternRule) and rule.replacement
@@
-                if isinstance(rule, PatternRule):
+                if isinstance(rule, _PatternRule):
@@
         if best_action == "pass" or best_match_span is None:
-            return ModerationEvaluationResult()
+            return _ModerationEvaluationResult()
@@
-        return ModerationEvaluationResult(
+        return _ModerationEvaluationResult(
```

Retain the `TYPE_CHECKING` import of `ModerationPolicy` so source annotations remain unchanged. The underscore-prefixed runtime import is removed only because deleting the hook makes it unused; `_PatternRule` and `_ModerationEvaluationResult` remain the evaluator's direct canonical runtime bindings. Do not alter scan, matching, ranking, redaction, snippet, or exception logic.

- [ ] **Step 3: Compile every changed Python file before running tests**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m py_compile \
  tldw_Server_API/app/core/Moderation/policy_compiler.py \
  tldw_Server_API/app/core/Moderation/policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_models_characterization.py \
  tldw_Server_API/tests/unit/test_moderation_models_imports.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py
```

Expected: exit 0 with no output.

- [ ] **Step 4: Re-run the inverse tests and prove they are green**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/unit/test_moderation_models_characterization.py::test_compiler_ignores_legacy_model_type_hook \
  tldw_Server_API/tests/unit/test_moderation_models_characterization.py::test_evaluator_ignores_legacy_model_type_hook \
  -q
```

Expected: 2 passed.

- [ ] **Step 5: Run the complete focused suite**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/unit/test_moderation_policy_compiler.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_models_characterization.py \
  tldw_Server_API/tests/unit/test_moderation_models_canonical.py \
  tldw_Server_API/tests/unit/test_moderation_models_imports.py \
  -q
```

Expected: PASS. Hook-only test count may be lower than the 119-test baseline because descriptor, tuple, and cache assertions were intentionally removed; every remaining and replacement test must pass.

- [ ] **Step 6: Record the red/green evidence in Backlog**

Append notes to `TASK-13436` through Backlog MCP containing:

```text
TDD evidence: the two legacy subclass-hook regression tests failed against the original hook dispatch because compiler/evaluator returned the replacement classes. After direct canonical-alias binding, both passed. Clean-process representative-operation tests passed before and after production changes, preserving the import-isolation oracle.
```

- [ ] **Step 7: Commit the focused implementation**

Run:

```bash
git add \
  tldw_Server_API/app/core/Moderation/policy_compiler.py \
  tldw_Server_API/app/core/Moderation/policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_models_characterization.py \
  tldw_Server_API/tests/unit/test_moderation_models_imports.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py \
  "backlog/tasks/task-13436 - Implement-Moderation-policy-types-retirement.md"
git commit -m "refactor(moderation): retire policy type hooks"
```

Expected: commit succeeds without bypassing hooks.

### Task 4: Run Downstream Moderation Verification

**Files:**
- Test: `tldw_Server_API/tests/unit/test_moderation*.py`
- Test: `tldw_Server_API/tests/Guardian/test_supervised_policy.py`
- Test: `tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py`
- Test: `tldw_Server_API/tests/Workflows/adapters/test_llm_adapters.py`
- Test: `tldw_Server_API/tests/Audio/test_audio_transcription_retention_and_redaction.py`

- [ ] **Step 1: Run all Moderation unit tests**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest tldw_Server_API/tests/unit/test_moderation*.py -q
```

Expected: PASS with no failures or errors.

- [ ] **Step 2: Run Guardian supervised-policy coverage**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Guardian/test_supervised_policy.py -q
```

Expected: PASS.

- [ ] **Step 3: Run Chat moderation integration coverage**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py -q
```

Expected: PASS.

- [ ] **Step 4: Run Workflow moderation-adapter coverage**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Workflows/adapters/test_llm_adapters.py \
  -k moderation_adapter \
  -q
```

Expected: all selected moderation-adapter tests pass; unrelated tests are deselected.

- [ ] **Step 5: Run the exact Audio redaction integration test**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Audio/test_audio_transcription_retention_and_redaction.py::test_audio_transcriptions_redacts_text_and_segments_when_stt_redaction_enabled \
  -q
```

Expected: 1 passed.

### Task 5: Run Quality, Security, And Scope Gates

**Files:**
- Check: `tldw_Server_API/app/core/Moderation/policy_compiler.py`
- Check: `tldw_Server_API/app/core/Moderation/policy_evaluator.py`
- Check: `tldw_Server_API/tests/unit/test_moderation_models_characterization.py`
- Check: `tldw_Server_API/tests/unit/test_moderation_models_imports.py`
- Check: `tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py`

- [ ] **Step 1: Run Ruff on every touched Python file**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m ruff check \
  tldw_Server_API/app/core/Moderation/policy_compiler.py \
  tldw_Server_API/app/core/Moderation/policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_models_characterization.py \
  tldw_Server_API/tests/unit/test_moderation_models_imports.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py
```

Expected: `All checks passed!`

- [ ] **Step 2: Run Black in check mode on the touched scope**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m black --check \
  tldw_Server_API/app/core/Moderation/policy_compiler.py \
  tldw_Server_API/app/core/Moderation/policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_models_characterization.py \
  tldw_Server_API/tests/unit/test_moderation_models_imports.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py
```

Expected: all five files would be left unchanged. If Black reports only touched-file formatting differences, run the same command without `--check`, inspect the diff, and rerun Tasks 3 Step 3, Task 3 Step 5, and this check. Do not format unrelated files.

- [ ] **Step 3: Run Bandit on the changed production scope**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/Moderation/policy_compiler.py \
  tldw_Server_API/app/core/Moderation/policy_evaluator.py \
  -f json \
  -o /tmp/bandit_TASK-13436.json
```

Expected: exit 0 with no findings in the changed production files. Record the result and report path in `TASK-13436`.

- [ ] **Step 4: Check whitespace and inspect the exact branch diff**

Run:

```bash
git diff --check origin/dev...HEAD
git diff --stat origin/dev...HEAD
git diff origin/dev...HEAD -- \
  tldw_Server_API/app/core/Moderation/policy_compiler.py \
  tldw_Server_API/app/core/Moderation/policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_models_characterization.py \
  tldw_Server_API/tests/unit/test_moderation_models_imports.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py
```

Expected: `git diff --check` is silent; production changes are limited to hook/cache removal and private-alias use; test changes match Task 2; no supported behavior or unrelated formatting changed.

- [ ] **Step 5: Audit retired-hook references**

Run the production audit:

```bash
rg -n "policy_types" \
  tldw_Server_API/app/core/Moderation/policy_compiler.py \
  tldw_Server_API/app/core/Moderation/policy_evaluator.py
```

Expected: no output and `rg` exit status 1, meaning no production definition or call remains.

Run the changed-test audit:

```bash
rg -n "policy_types" \
  tldw_Server_API/tests/unit/test_moderation_models_characterization.py \
  tldw_Server_API/tests/unit/test_moderation_models_imports.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py
```

Expected: exactly two matches, both local `def policy_types():` declarations inside `LegacyReplacementCompiler` and `LegacyReplacementEvaluator`. There must be no base-class call, descriptor assertion, tuple assertion, or cache assertion.

- [ ] **Step 6: Confirm runtime public namespaces and service facade were not changed**

Run:

```bash
git diff --name-only origin/dev...HEAD
```

Expected runtime/test scope: the two production modules and three focused test files, plus approved design/plan and Backlog records. `models.py`, `moderation_service.py`, endpoints, schemas, configuration, and downstream caller files must not appear.

### Task 6: Rebase Onto Current Dev And Re-Verify

**Files:**
- Rebase target: `origin/dev`
- Verify: all files and suites from Tasks 3 through 5

- [ ] **Step 1: Fetch the current target branch**

Run:

```bash
git fetch origin dev
git rev-parse origin/dev
```

Expected: fetch succeeds and prints the current `origin/dev` commit.

- [ ] **Step 2: Verify the branch already contains current dev**

Run:

```bash
git merge-base --is-ancestor origin/dev HEAD
```

Expected: exit 0. If it exits 1, run `git rebase origin/dev`, resolve only conflicts within this task's approved scope, and rerun every command in Tasks 3 Step 3 through Task 5 Step 6.

- [ ] **Step 3: Re-run compilation and focused tests even when no rebase was needed**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m py_compile \
  tldw_Server_API/app/core/Moderation/policy_compiler.py \
  tldw_Server_API/app/core/Moderation/policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_models_characterization.py \
  tldw_Server_API/tests/unit/test_moderation_models_imports.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py
python -m pytest \
  tldw_Server_API/tests/unit/test_moderation_policy_compiler.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_models_characterization.py \
  tldw_Server_API/tests/unit/test_moderation_models_canonical.py \
  tldw_Server_API/tests/unit/test_moderation_models_imports.py \
  -q
```

Expected: compilation exits 0 and the focused suite passes.

- [ ] **Step 4: Re-run all downstream and quality gates after a rebase**

If Step 2 required `git rebase origin/dev`, rerun all five commands in Task 4 and all six checks in Task 5. If no rebase was required, retain the fresh Task 4 and Task 5 evidence already collected.

Expected: every gate remains green against current `dev`.

### Task 7: Finalize Tracking And Prepare Review

**Files:**
- Update through Backlog MCP: `backlog/tasks/task-13436 - Implement-Moderation-policy-types-retirement.md`
- Review: complete branch diff against `origin/dev`

- [ ] **Step 1: Perform a final code-review pass**

Review `git diff origin/dev...HEAD` for:

```text
1. No policy, parsing, scanning, ranking, redaction, exception, or service dispatch changes.
2. No public runtime model imports added to compiler/evaluator modules.
3. Every removed hook lookup has one direct private-alias replacement.
4. Legacy service imports retain exact canonical identity.
5. The only retained policy_types references are the two intentional subclass regression fixtures.
6. The PR explicitly documents the direct-call and subclass-hook compatibility break.
```

Expected: no unaddressed correctness, compatibility, security, or test-coverage issue.

- [ ] **Step 2: Complete the Backlog acceptance criteria and Definition of Done**

Use Backlog MCP `task_edit` for `TASK-13436` to:

```text
- check acceptance criteria 1 through 6
- check Definition of Done items 1 through 6
- append exact test counts and command outcomes
- append Bandit result and /tmp/bandit_TASK-13436.json
- add the final implementation commit hash
- set status to Done
- add a final summary stating that the two hooks and cache-only surface were removed, eight internal selection sites now use canonical private aliases, supported Moderation behavior/import isolation remain covered, and direct hook calls/subclass model substitution are intentionally unsupported
```

- [ ] **Step 3: Commit the completed task record**

Run:

```bash
git add "backlog/tasks/task-13436 - Implement-Moderation-policy-types-retirement.md"
git commit -m "chore(backlog): complete TASK-13436"
```

Expected: commit succeeds without bypassing hooks.

- [ ] **Step 4: Verify the final tree and commit range**

Run:

```bash
git status --short --branch
git log --oneline origin/dev..HEAD
git diff --check origin/dev...HEAD
```

Expected: clean worktree, the planning and implementation commits appear above current `origin/dev`, and the diff check is silent.

- [ ] **Step 5: Prepare the pull request facts without weakening the compatibility statement**

Use this technical summary in the PR body alongside the requester’s required human-written `Change summary`:

```markdown
## Technical details

- removes `PolicyCompiler.policy_types()` and `PolicyEvaluator.policy_types()`
- binds compiler/evaluator runtime operations to the canonical private model aliases from `Moderation/models.py`
- replaces hook-based import probes with clean-process compile/evaluate operations
- preserves supported `ModerationService` APIs, model identity, import isolation, runtime namespaces, and moderation semantics

## Compatibility

Direct calls to `policy_types()`, subclass model substitution through those methods, and evaluator tuple-cache identity are intentionally no longer supported. Private runtime alias rebinding remains outside the compatibility contract.

## Verification

- compilation-first focused Moderation tests
- complete Moderation unit suite
- Guardian, Chat, Workflow, and Audio downstream tests
- Ruff, Black check, Bandit, diff check, and source audit
```

Do not describe the change as preserving the removed callable/subclass surface. Do not merge an AI-materially-authored PR until the human requester supplies a `Change summary` that explains both what changed and why these implementation choices were made.
