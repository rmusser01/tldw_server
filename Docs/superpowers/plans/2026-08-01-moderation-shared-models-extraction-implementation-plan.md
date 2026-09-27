# Moderation Shared Models Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Moderation.models` the canonical owner of the three shared policy/result dataclasses without changing supported moderation behavior or established service imports.

**Architecture:** Add a standard-library-only `models.py`, then turn `moderation_service.py` into the compatibility facade for the exact canonical class objects. `PolicyCompiler` and `PolicyEvaluator` use private runtime aliases from the neutral module while retaining their static `policy_types()` descriptors and subclass dispatch; subprocess tests prove those methods no longer load the service.

**Tech Stack:** Python 3, standard-library dataclasses/AST/importlib/subprocess tools, pytest, Black, Ruff, Bandit, Loguru only in the existing service, and the project virtualenv at `/Users/appledev/Documents/GitHub/tldw_server/.venv`.

## Global Constraints

- This is a strict structural extraction. Moderation decisions, scanning, snippets, redaction, configuration, persistence, and endpoint behavior must not change.
- `models.py` canonically owns exactly `ModerationPolicy`, `PatternRule`, and `ModerationEvaluationResult` and imports only `__future__`, `dataclasses`, `json`, and `re`.
- `moderation_service.py` continues exporting all three names as the exact canonical objects; established production callers remain on that facade.
- Preserve dataclass constructors, field order, defaults, factories, annotations, equality, mutability, borrowed list/set/regex identity, and the exact mapping returned by `ModerationPolicy.to_dict()`.
- The three canonical classes use their natural `tldw_Server_API.app.core.Moderation.models` `__module__`; do not override it.
- Existing pickles naming the service facade remain resolvable. New pickle bytes and loading new payloads on older releases are outside compatibility and are not exercised in tests.
- `PolicyCompiler.policy_types()` and `PolicyEvaluator.policy_types()` remain `staticmethod` descriptors with unchanged signatures, tuple order, and internal `self.policy_types()` dispatch.
- Compiler/evaluator runtime references use underscore-prefixed aliases. Public model names remain absent from those module namespaces and runtime `typing.get_type_hints()` behavior remains unchanged.
- Rebinding service facade exports and monkeypatching private service globals are outside compatibility. Subclass overrides of `policy_types()` remain supported.
- `Moderation/__init__.py`, endpoint/schema files, compiler-specific dataclasses, `EvaluationLimits`, and `_ResolvedModerationServiceState` do not move.
- Do not add regex hardening, validation, normalization, wrappers, duplicate class identities, package re-exports, or caller migrations.
- `moderation_service.py` and `policy_compiler.py` have pre-existing Black debt at the recorded predecessor. Do not format either whole file in this PR; enforce Black on new files and the already-clean evaluator, and enforce Ruff plus diff review on all touched Python files.
- Track implementation in `TASK-13011`. The stacked branch boundary is predecessor `285773ea6b05318512f4b004375e394e969e425d`.
- The implementation PR must not target `dev` until `TASK-12992` is merged and the post-predecessor commits are transplanted onto current `origin/dev`.

---

## Source And Tracking

- Design: `Docs/superpowers/specs/2026-08-01-moderation-shared-models-extraction-design.md`
- Design task: `TASK-13010`
- Implementation task: `TASK-13011`
- Predecessor task: `TASK-12992`
- Recorded predecessor head: `285773ea6b05318512f4b004375e394e969e425d`

## File Structure

- Create: `tldw_Server_API/app/core/Moderation/models.py`
  - Owns exactly the three canonical dataclasses and model-local private constants used by `ModerationPolicy.to_dict()`.
  - Has no project, service, compiler, evaluator, configuration, or Loguru imports.
- Modify: `tldw_Server_API/app/core/Moderation/moderation_service.py`
  - Removes the three class bodies and imports their exact canonical definitions at module scope.
  - Retains `_ResolvedModerationServiceState`, service exception handling, configuration, locking, persistence, logging, and all service methods.
- Modify: `tldw_Server_API/app/core/Moderation/policy_compiler.py`
  - Imports canonical classes under private runtime aliases and under normal names only for `TYPE_CHECKING`.
  - Keeps `policy_types()` and every internal dynamic call site.
- Modify: `tldw_Server_API/app/core/Moderation/policy_evaluator.py`
  - Imports canonical classes under private runtime aliases and under normal names only for `TYPE_CHECKING`.
  - Keeps `EvaluationLimits`, `policy_types()`, and every internal dynamic call site.
- Create: `tldw_Server_API/tests/unit/test_moderation_models_characterization.py`
  - Pre-move oracle for dataclass, aliasing, `to_dict()`, descriptor, tuple, and subclass-dispatch behavior.
- Create: `tldw_Server_API/tests/unit/test_moderation_models_canonical.py`
  - Canonical ownership, service identity, natural metadata, source import allowlist, and parent-package runtime isolation.
- Create: `tldw_Server_API/tests/unit/test_moderation_models_imports.py`
  - Fresh-process deferred-edge removal and compiler/evaluator public namespace contracts.
- Modify throughout execution: `backlog/tasks/task-13011 - Implement-Moderation-shared-models-extraction.md`
  - Records per-task files, focused test results, reviews, verification, rollout state, and final summary.

## Command Conventions

Run commands from the isolated implementation worktree. Use the project interpreter directly so every command uses the configured environment:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest --version
```

Before each task commit:

1. Update `TASK-13011` through Backlog MCP with status, touched files, and focused verification.
2. Run the task's focused tests and `git diff --check`.
3. Review staged files against the recorded predecessor boundary.
4. Commit only the task's files and Backlog record.

---

### Task 1: Add Pre-Move Model And Dispatch Characterization

**Files:**
- Create: `tldw_Server_API/tests/unit/test_moderation_models_characterization.py`
- Modify: `backlog/tasks/task-13011 - Implement-Moderation-shared-models-extraction.md`

**Interfaces:**
- Consumes: service-owned `ModerationPolicy`, `PatternRule`, `ModerationEvaluationResult`, `PolicyCompiler.policy_types()`, and `PolicyEvaluator.policy_types()` as they exist before extraction.
- Produces: a green, implementation-independent oracle that Tasks 2 and 3 must preserve except for separately approved canonical metadata and service-export rebinding boundaries.

- [ ] **Step 1: Add exact dataclass fixtures and field contracts**

Create the characterization file with these imports, constants, and helper classes:

```python
from __future__ import annotations

import inspect
import re
import typing
from dataclasses import fields

import pytest

from tldw_Server_API.app.core.Moderation.moderation_service import (
    ModerationEvaluationResult,
    ModerationPolicy,
    PatternRule,
)
from tldw_Server_API.app.core.Moderation.policy_compiler import (
    PolicyCompilationInput,
    PolicyCompiler,
    ResolvedModerationConfig,
)
from tldw_Server_API.app.core.Moderation.policy_evaluator import (
    EvaluationLimits,
    PolicyEvaluator,
)

pytestmark = pytest.mark.unit

_LIMITS = EvaluationLimits(
    max_scan_chars=100,
    match_window_chars=16,
    max_fallback_scan_chars=200,
    max_replacements_per_pattern=10,
)


class _BrokenPattern:
    @property
    def pattern(self):
        raise ValueError("broken pattern")


class _UnexpectedPattern:
    @property
    def pattern(self):
        raise ZeroDivisionError("unexpected pattern failure")
```

Append exact signature, field, and annotation assertions:

```python
@pytest.mark.parametrize(
    ("model_type", "expected_signature", "expected_fields", "expected_annotations"),
    [
        (
            ModerationPolicy,
            "(enabled: 'bool' = False, input_enabled: 'bool' = True, "
            "output_enabled: 'bool' = True, input_action: 'str' = 'block', "
            "output_action: 'str' = 'redact', redact_replacement: 'str' = "
            "'[REDACTED]', per_user_overrides: 'bool' = True, "
            "block_patterns: 'list[PatternRule]' = <factory>, "
            "categories_enabled: 'set[str] | None' = None) -> None",
            (
                "enabled",
                "input_enabled",
                "output_enabled",
                "input_action",
                "output_action",
                "redact_replacement",
                "per_user_overrides",
                "block_patterns",
                "categories_enabled",
            ),
            {
                "enabled": "bool",
                "input_enabled": "bool",
                "output_enabled": "bool",
                "input_action": "str",
                "output_action": "str",
                "redact_replacement": "str",
                "per_user_overrides": "bool",
                "block_patterns": "list[PatternRule]",
                "categories_enabled": "set[str] | None",
            },
        ),
        (
            PatternRule,
            "(regex: 're.Pattern', action: 'str | None' = None, "
            "replacement: 'str | None' = None, categories: 'set[str] | None' "
            "= None, phase: 'str' = 'both') -> None",
            ("regex", "action", "replacement", "categories", "phase"),
            {
                "regex": "re.Pattern",
                "action": "str | None",
                "replacement": "str | None",
                "categories": "set[str] | None",
                "phase": "str",
            },
        ),
        (
            ModerationEvaluationResult,
            "(action: 'str' = 'pass', redacted_text: 'str | None' = None, "
            "matched_pattern: 'str | None' = None, category: 'str | None' "
            "= None, match_span: 'tuple[int, int] | None' = None, sample: "
            "'str | None' = None) -> None",
            (
                "action",
                "redacted_text",
                "matched_pattern",
                "category",
                "match_span",
                "sample",
            ),
            {
                "action": "str",
                "redacted_text": "str | None",
                "matched_pattern": "str | None",
                "category": "str | None",
                "match_span": "tuple[int, int] | None",
                "sample": "str | None",
            },
        ),
    ],
)
def test_model_declarations_are_literal(
    model_type,
    expected_signature,
    expected_fields,
    expected_annotations,
):
    assert str(inspect.signature(model_type)) == expected_signature
    assert tuple(field.name for field in fields(model_type)) == expected_fields
    assert model_type.__annotations__ == expected_annotations
```

- [ ] **Step 2: Characterize defaults, borrowing, equality, and mutability**

Append:

```python
def test_model_defaults_and_borrowed_values_are_literal():
    first = ModerationPolicy()
    second = ModerationPolicy()
    supplied_rules = []
    supplied_categories = {"pii"}
    regex = re.compile("secret")
    rule_categories = {"confidential"}
    rule = PatternRule(regex=regex, categories=rule_categories)
    policy = ModerationPolicy(
        block_patterns=supplied_rules,
        categories_enabled=supplied_categories,
    )

    assert first == second
    assert first.block_patterns == []
    assert first.block_patterns is not second.block_patterns
    assert policy.block_patterns is supplied_rules
    assert policy.categories_enabled is supplied_categories
    assert rule.regex is regex
    assert rule.categories is rule_categories

    result = ModerationEvaluationResult()
    result.action = "warn"
    assert result.action == "warn"


def test_resolved_model_type_hints_are_literal():
    policy_hints = typing.get_type_hints(ModerationPolicy)
    rule_hints = typing.get_type_hints(PatternRule)
    result_hints = typing.get_type_hints(ModerationEvaluationResult)

    assert policy_hints["block_patterns"] == list[PatternRule]
    assert policy_hints["categories_enabled"] == set[str] | None
    assert rule_hints["regex"] is re.Pattern
    assert rule_hints["categories"] == set[str] | None
    assert result_hints["match_span"] == tuple[int, int] | None
```

- [ ] **Step 3: Characterize the exact `to_dict()` mapping and fallbacks**

Append:

```python
def test_policy_to_dict_returns_literal_mapping():
    policy = ModerationPolicy(
        enabled=True,
        block_patterns=[
            PatternRule(
                regex=re.compile("secret"),
                action="redact",
                replacement="[RULE]",
                categories=None,
                phase="input",
            )
        ],
        categories_enabled={"pii", "confidential"},
    )

    assert policy.to_dict() == {
        "enabled": True,
        "input_enabled": True,
        "output_enabled": True,
        "input_action": "block",
        "output_action": "redact",
        "redact_replacement": "[REDACTED]",
        "per_user_overrides": True,
        "blocklist_count": 1,
        "block_patterns": ["secret"],
        "rules": [
            {
                "pattern": "secret",
                "action": "redact",
                "replacement": "[RULE]",
                "phase": "input",
                "categories": "uncategorized",
            }
        ],
        "categories_enabled": ["confidential", "pii"],
    }


def test_policy_to_dict_preserves_legacy_regex_shape():
    policy = ModerationPolicy(block_patterns=[re.compile("legacy")])

    snapshot = policy.to_dict()

    assert snapshot["block_patterns"] == ["legacy"]
    assert snapshot["rules"] == [
        {
            "pattern": "legacy",
            "action": "",
            "replacement": "",
            "phase": "both",
            "categories": "",
        }
    ]


def test_policy_to_dict_preserves_noncritical_fallbacks():
    policy = ModerationPolicy(block_patterns=[_BrokenPattern()])

    snapshot = policy.to_dict()

    assert snapshot["blocklist_count"] == 0
    assert snapshot["block_patterns"] == []
    assert snapshot["rules"] == []


def test_policy_to_dict_does_not_swallow_unlisted_exceptions():
    policy = ModerationPolicy(block_patterns=[_UnexpectedPattern()])

    with pytest.raises(ZeroDivisionError, match="unexpected pattern failure"):
        policy.to_dict()
```

- [ ] **Step 4: Characterize descriptors, tuples, and subclass dispatch**

Append:

```python
def test_policy_type_descriptors_and_tuples_are_literal():
    assert isinstance(inspect.getattr_static(PolicyCompiler, "policy_types"), staticmethod)
    assert isinstance(inspect.getattr_static(PolicyEvaluator, "policy_types"), staticmethod)
    assert str(inspect.signature(PolicyCompiler.policy_types)) == (
        "() -> 'tuple[type[ModerationPolicy], type[PatternRule]]'"
    )
    assert str(inspect.signature(PolicyEvaluator.policy_types)) == (
        "() -> 'tuple[type[ModerationPolicy], type[PatternRule], "
        "type[ModerationEvaluationResult]]'"
    )
    assert PolicyCompiler.policy_types() == (ModerationPolicy, PatternRule)
    assert PolicyEvaluator.policy_types() == (
        ModerationPolicy,
        PatternRule,
        ModerationEvaluationResult,
    )


def test_compiler_uses_overridden_policy_types():
    class ReplacementPolicy:
        def __init__(self, **values):
            self.values = values

    class ReplacementCompiler(PolicyCompiler):
        @staticmethod
        def policy_types():
            return ReplacementPolicy, PatternRule

    result = ReplacementCompiler().compile_global(
        PolicyCompilationInput(
            config=ResolvedModerationConfig(),
            runtime_override={},
            blocklist_lines=[],
            pii_rules=[],
        )
    )

    assert isinstance(result.policy, ReplacementPolicy)
    assert result.policy.values["block_patterns"] == []


def test_evaluator_uses_overridden_policy_types():
    class ReplacementResult:
        pass

    class ReplacementEvaluator(PolicyEvaluator):
        @staticmethod
        def policy_types():
            return ModerationPolicy, PatternRule, ReplacementResult

    result = ReplacementEvaluator().evaluate_text(
        "",
        ModerationPolicy(enabled=False),
        "input",
        _LIMITS,
        include_redacted_text=False,
    )

    assert isinstance(result, ReplacementResult)
```

- [ ] **Step 5: Run the pre-move oracle and confirm it is green**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/unit/test_moderation_models_characterization.py -q
```

Expected: all tests pass against the service-owned classes before `models.py` exists.

- [ ] **Step 6: Record and commit the characterization**

Update `TASK-13011` with the new test file and passing count, then run:

```bash
git diff --check
git add tldw_Server_API/tests/unit/test_moderation_models_characterization.py "backlog/tasks/task-13011 - Implement-Moderation-shared-models-extraction.md"
git commit -m "test: characterize Moderation shared models"
```

---

### Task 2: Create Canonical Models And Preserve The Service Facade

**Files:**
- Create: `tldw_Server_API/tests/unit/test_moderation_models_canonical.py`
- Create: `tldw_Server_API/app/core/Moderation/models.py`
- Modify: `tldw_Server_API/app/core/Moderation/moderation_service.py`
- Modify: `backlog/tasks/task-13011 - Implement-Moderation-shared-models-extraction.md`

**Interfaces:**
- Consumes: Task 1's literal model and `to_dict()` oracle.
- Produces: canonical `models.ModerationPolicy`, `models.PatternRule`, and `models.ModerationEvaluationResult`, plus exact service facade aliases consumed by Task 3.

- [ ] **Step 1: Write failing canonical ownership and import-boundary tests**

Create `test_moderation_models_canonical.py`:

```python
from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from dataclasses import is_dataclass
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Moderation import models, moderation_service

pytestmark = pytest.mark.unit

_MODEL_NAMES = (
    "ModerationPolicy",
    "PatternRule",
    "ModerationEvaluationResult",
)
_FORBIDDEN_MODULES = (
    "tldw_Server_API.app.core.config",
    "tldw_Server_API.app.core.Moderation.moderation_service",
    "tldw_Server_API.app.core.Moderation.policy_compiler",
    "tldw_Server_API.app.core.Moderation.policy_evaluator",
)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_MODELS_PATH = (
    Path(__file__).resolve().parents[2]
    / "app"
    / "core"
    / "Moderation"
    / "models.py"
)


def test_service_facade_exports_exact_canonical_classes():
    for name in _MODEL_NAMES:
        canonical = getattr(models, name)
        assert getattr(moderation_service, name) is canonical
        assert canonical.__module__ == models.__name__


def test_models_module_owns_exactly_three_dataclass_types():
    owned_dataclasses = {
        name
        for name, value in vars(models).items()
        if isinstance(value, type)
        and is_dataclass(value)
        and value.__module__ == models.__name__
    }

    assert owned_dataclasses == set(_MODEL_NAMES)


def test_legacy_qualified_names_resolve_to_canonical_classes():
    legacy = importlib.import_module(
        "tldw_Server_API.app.core.Moderation.moderation_service"
    )

    for name in _MODEL_NAMES:
        assert getattr(legacy, name) is getattr(models, name)


def test_models_source_imports_only_approved_standard_library_modules():
    tree = ast.parse(_MODELS_PATH.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])

    assert roots == {"__future__", "dataclasses", "json", "re"}


def test_models_import_adds_no_moderation_or_config_dependencies():
    script = f"""
import importlib
import sys

forbidden = {repr(_FORBIDDEN_MODULES)}
importlib.import_module("tldw_Server_API.app.core.Moderation")
assert not [name for name in forbidden if name in sys.modules]
importlib.import_module("tldw_Server_API.app.core.Moderation.models")
assert not [name for name in forbidden if name in sys.modules]
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
```

- [ ] **Step 2: Run the canonical tests and confirm the missing-module failure**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/unit/test_moderation_models_canonical.py -q
```

Expected: collection fails with `ModuleNotFoundError` or `ImportError` for `tldw_Server_API.app.core.Moderation.models`.

- [ ] **Step 3: Add the literal canonical models module**

Create `models.py` with the current declarations and only the approved service-global substitutions:

```python
"""Shared canonical data models for Moderation policy compilation and evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

_MODEL_NONCRITICAL_EXCEPTIONS = (
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
_UNCATEGORIZED_CATEGORY = "uncategorized"


@dataclass
class ModerationPolicy:
    enabled: bool = False
    input_enabled: bool = True
    output_enabled: bool = True
    input_action: str = "block"  # block | redact | warn
    output_action: str = "redact"  # redact | block | warn (block only applies to non-streaming)
    redact_replacement: str = "[REDACTED]"
    per_user_overrides: bool = True
    # Compiled rules; each rule includes the regex and optional per-pattern action/replacement
    block_patterns: list[PatternRule] = field(default_factory=list)
    # Enabled categories filter (None or empty means allow all)
    categories_enabled: set[str] | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable snapshot of the policy (without raw regex objects)."""
        patterns: list[str] = []
        try:
            if self.block_patterns:
                # Backward-friendly: expose raw patterns as strings
                tmp: list[str] = []
                for p in self.block_patterns:
                    pat = getattr(p, "pattern", None)
                    if pat is None and isinstance(p, PatternRule):
                        pat = getattr(p.regex, "pattern", "")
                    tmp.append(pat or "")
                patterns = tmp
        except _MODEL_NONCRITICAL_EXCEPTIONS:
            patterns = []
        # Provide richer rule view
        rules: list[dict[str, str]] = []
        try:
            if self.block_patterns:
                for p in self.block_patterns:
                    if isinstance(p, PatternRule):
                        cats = p.categories if p.categories else {_UNCATEGORIZED_CATEGORY}
                        rules.append({
                            "pattern": p.regex.pattern,
                            "action": p.action or "",
                            "replacement": p.replacement or "",
                            "phase": p.phase or "both",
                            "categories": ",".join(sorted(cats)) if cats else "",
                        })
                    else:
                        rules.append(
                            {
                                "pattern": getattr(p, "pattern", ""),
                                "action": "",
                                "replacement": "",
                                "phase": "both",
                                "categories": "",
                            }
                        )
        except _MODEL_NONCRITICAL_EXCEPTIONS:
            rules = []
        return {
            "enabled": self.enabled,
            "input_enabled": self.input_enabled,
            "output_enabled": self.output_enabled,
            "input_action": self.input_action,
            "output_action": self.output_action,
            "redact_replacement": self.redact_replacement,
            "per_user_overrides": self.per_user_overrides,
            "blocklist_count": len(patterns),
            "block_patterns": patterns,
            "rules": rules,
            "categories_enabled": sorted(self.categories_enabled) if self.categories_enabled else [],
        }


@dataclass
class PatternRule:
    regex: re.Pattern
    action: str | None = None  # block | redact | warn | None
    replacement: str | None = None  # only used when action=redact
    categories: set[str] | None = None  # e.g., {"pii", "confidential"}
    phase: str = "both"  # input | output | both


@dataclass
class ModerationEvaluationResult:
    """Canonical moderation evaluation result."""

    action: str = "pass"
    redacted_text: str | None = None
    matched_pattern: str | None = None
    category: str | None = None
    match_span: tuple[int, int] | None = None
    sample: str | None = None
```

Do not improve exception handling, move class order, normalize category values, copy borrowed objects, add `__all__`, or override `__module__`.

- [ ] **Step 4: Replace service-owned class bodies with canonical imports**

In `moderation_service.py`:

1. Change `from dataclasses import dataclass, field, replace` to `from dataclasses import dataclass, replace`.
2. Add this module-level import before compiler/evaluator imports:

```python
from tldw_Server_API.app.core.Moderation.models import (
    ModerationEvaluationResult,
    ModerationPolicy,
    PatternRule,
)
```

3. Delete only the three class bodies from `@dataclass class ModerationPolicy` through the end of `ModerationEvaluationResult`.
4. Keep `_MODERATION_NONCRITICAL_EXCEPTIONS`, `_ResolvedModerationServiceState`, and every service method unchanged.

- [ ] **Step 5: Run canonical and pre-move characterization tests**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest \
  tldw_Server_API/tests/unit/test_moderation_models_characterization.py \
  tldw_Server_API/tests/unit/test_moderation_models_canonical.py \
  -q
```

Expected: both files pass; service and canonical paths have exact identity, and model import does not load service/config/compiler/evaluator.

- [ ] **Step 6: Record and commit canonical ownership**

Update `TASK-13011` with production/test files and focused results, then run:

```bash
git diff --check
git add tldw_Server_API/app/core/Moderation/models.py tldw_Server_API/app/core/Moderation/moderation_service.py tldw_Server_API/tests/unit/test_moderation_models_canonical.py "backlog/tasks/task-13011 - Implement-Moderation-shared-models-extraction.md"
git commit -m "refactor: extract Moderation shared models"
```

---

### Task 3: Remove Deferred Service Edges Without Expanding Public Namespaces

**Files:**
- Create: `tldw_Server_API/tests/unit/test_moderation_models_imports.py`
- Modify: `tldw_Server_API/app/core/Moderation/policy_compiler.py`
- Modify: `tldw_Server_API/app/core/Moderation/policy_evaluator.py`
- Modify: `backlog/tasks/task-13011 - Implement-Moderation-shared-models-extraction.md`

**Interfaces:**
- Consumes: Task 2's exact canonical classes and service facade aliases.
- Produces: service-independent `policy_types()` tuples, unchanged static descriptors and subclass dispatch, and unchanged compiler/evaluator public model-name behavior.

- [ ] **Step 1: Write failing fresh-process and rebinding-boundary tests**

Create `test_moderation_models_imports.py`:

```python
from __future__ import annotations

import inspect
import subprocess
import sys
import typing
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Moderation import (
    moderation_service,
    policy_compiler,
    policy_evaluator,
)
from tldw_Server_API.app.core.Moderation.models import (
    ModerationEvaluationResult,
    ModerationPolicy,
    PatternRule,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SERVICE_MODULE = "tldw_Server_API.app.core.Moderation.moderation_service"


@pytest.mark.parametrize(
    "script",
    [
        f"""
import sys
from tldw_Server_API.app.core.Moderation.policy_compiler import PolicyCompiler
assert {_SERVICE_MODULE!r} not in sys.modules
types = PolicyCompiler.policy_types()
assert [item.__name__ for item in types] == ["ModerationPolicy", "PatternRule"]
assert all(item.__module__ == "tldw_Server_API.app.core.Moderation.models" for item in types)
assert {_SERVICE_MODULE!r} not in sys.modules
from tldw_Server_API.app.core.Moderation import models, moderation_service
assert moderation_service.ModerationPolicy is models.ModerationPolicy
assert moderation_service.PatternRule is models.PatternRule
""",
        f"""
import sys
from tldw_Server_API.app.core.Moderation.policy_evaluator import PolicyEvaluator
assert {_SERVICE_MODULE!r} not in sys.modules
types = PolicyEvaluator.policy_types()
assert [item.__name__ for item in types] == ["ModerationPolicy", "PatternRule", "ModerationEvaluationResult"]
assert all(item.__module__ == "tldw_Server_API.app.core.Moderation.models" for item in types)
assert {_SERVICE_MODULE!r} not in sys.modules
from tldw_Server_API.app.core.Moderation import models, moderation_service
assert moderation_service.ModerationPolicy is models.ModerationPolicy
assert moderation_service.PatternRule is models.PatternRule
assert moderation_service.ModerationEvaluationResult is models.ModerationEvaluationResult
""",
    ],
)
def test_policy_types_do_not_load_service(script):
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "module_order",
    [
        (
            "tldw_Server_API.app.core.Moderation.models",
            "tldw_Server_API.app.core.Moderation.moderation_service",
            "tldw_Server_API.app.core.Moderation.policy_compiler",
            "tldw_Server_API.app.core.Moderation.policy_evaluator",
        ),
        (
            "tldw_Server_API.app.core.Moderation.moderation_service",
            "tldw_Server_API.app.core.Moderation.policy_compiler",
            "tldw_Server_API.app.core.Moderation.policy_evaluator",
        ),
    ],
)
def test_complete_import_orders_resolve_exact_identity(module_order):
    script = f"""
import importlib

for module_name in {module_order!r}:
    importlib.import_module(module_name)
models = importlib.import_module("tldw_Server_API.app.core.Moderation.models")
service = importlib.import_module("tldw_Server_API.app.core.Moderation.moderation_service")
for name in ("ModerationPolicy", "PatternRule", "ModerationEvaluationResult"):
    assert getattr(service, name) is getattr(models, name)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_policy_type_descriptors_and_public_namespaces_remain_literal():
    assert isinstance(inspect.getattr_static(policy_compiler.PolicyCompiler, "policy_types"), staticmethod)
    assert isinstance(inspect.getattr_static(policy_evaluator.PolicyEvaluator, "policy_types"), staticmethod)
    assert not hasattr(policy_compiler, "ModerationPolicy")
    assert not hasattr(policy_compiler, "PatternRule")
    assert not hasattr(policy_evaluator, "ModerationPolicy")
    assert not hasattr(policy_evaluator, "PatternRule")
    assert not hasattr(policy_evaluator, "ModerationEvaluationResult")

    with pytest.raises(NameError):
        typing.get_type_hints(policy_compiler.PolicyCompiler.compile_user_policy)
    with pytest.raises(NameError):
        typing.get_type_hints(policy_evaluator.PolicyEvaluator.evaluate_text)


def test_service_export_rebinding_does_not_replace_canonical_policy_types(monkeypatch):
    monkeypatch.setattr(moderation_service, "ModerationPolicy", type("Policy", (), {}))
    monkeypatch.setattr(moderation_service, "PatternRule", type("Rule", (), {}))
    monkeypatch.setattr(
        moderation_service,
        "ModerationEvaluationResult",
        type("Result", (), {}),
    )

    assert policy_compiler.PolicyCompiler.policy_types() == (
        ModerationPolicy,
        PatternRule,
    )
    assert policy_evaluator.PolicyEvaluator.policy_types() == (
        ModerationPolicy,
        PatternRule,
        ModerationEvaluationResult,
    )
```

- [ ] **Step 2: Run the dependency tests and confirm the intended failures**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/unit/test_moderation_models_imports.py -q
```

Expected before implementation:

- both subprocess cases fail because calling `policy_types()` loads `moderation_service`
- the service-export rebinding test fails because current deferred lookups return rebound service names
- public namespace and runtime type-hint assertions already pass

- [ ] **Step 3: Switch `PolicyCompiler` to private canonical aliases**

Replace its service type imports with:

```python
from typing import TYPE_CHECKING

from tldw_Server_API.app.core.Moderation.models import (
    ModerationPolicy as _ModerationPolicy,
    PatternRule as _PatternRule,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Moderation.models import (
        ModerationPolicy,
        PatternRule,
    )
```

Keep the descriptor and signature, but replace the deferred body:

```python
@staticmethod
def policy_types() -> tuple[type[ModerationPolicy], type[PatternRule]]:
    """Return the canonical policy dataclasses without loading the service."""
    return _ModerationPolicy, _PatternRule
```

Do not replace any existing internal `self.policy_types()` calls with module aliases.

- [ ] **Step 4: Switch `PolicyEvaluator` to private canonical aliases**

Replace its service type imports with:

```python
from typing import TYPE_CHECKING

from tldw_Server_API.app.core.Moderation.models import (
    ModerationEvaluationResult as _ModerationEvaluationResult,
    ModerationPolicy as _ModerationPolicy,
    PatternRule as _PatternRule,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Moderation.models import (
        ModerationEvaluationResult,
        ModerationPolicy,
        PatternRule,
    )
```

Keep the descriptor and signature, but replace the deferred body:

```python
@staticmethod
def policy_types() -> tuple[
    type[ModerationPolicy],
    type[PatternRule],
    type[ModerationEvaluationResult],
]:
    """Return canonical policy dataclasses without loading the service."""
    return _ModerationPolicy, _PatternRule, _ModerationEvaluationResult
```

Do not replace any existing internal `self.policy_types()` calls with module aliases.

- [ ] **Step 5: Run all model, compiler, and evaluator focused tests**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest \
  tldw_Server_API/tests/unit/test_moderation_models_characterization.py \
  tldw_Server_API/tests/unit/test_moderation_models_canonical.py \
  tldw_Server_API/tests/unit/test_moderation_models_imports.py \
  tldw_Server_API/tests/unit/test_moderation_policy_compiler.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_policy_evaluator_delegation.py \
  -q
```

Expected: all tests pass, including Task 1 subclass-dispatch tests and Task 3 service-absence checks.

- [ ] **Step 6: Record and commit dependency removal**

Update `TASK-13011` with focused results, then run:

```bash
git diff --check
git add tldw_Server_API/app/core/Moderation/policy_compiler.py tldw_Server_API/app/core/Moderation/policy_evaluator.py tldw_Server_API/tests/unit/test_moderation_models_imports.py "backlog/tasks/task-13011 - Implement-Moderation-shared-models-extraction.md"
git commit -m "refactor: decouple Moderation policy types"
```

---

### Task 4: Run Full Verification And Independent Review

**Files:**
- Verify unchanged: `tldw_Server_API/app/api/v1/endpoints/moderation.py`
- Verify unchanged: `tldw_Server_API/app/api/v1/schemas/moderation_schemas.py`
- Verify unchanged: `tldw_Server_API/app/core/Moderation/supervised_policy.py`
- Verify unchanged: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_policy.py`
- Modify: `backlog/tasks/task-13011 - Implement-Moderation-shared-models-extraction.md`

**Interfaces:**
- Consumes: the complete three-commit structural extraction from Tasks 1 through 3.
- Produces: review and verification evidence that the stacked implementation is ready to transplant after the predecessor merges.

- [ ] **Step 1: Compile every touched Python file before broader tests**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m py_compile \
  tldw_Server_API/app/core/Moderation/models.py \
  tldw_Server_API/app/core/Moderation/moderation_service.py \
  tldw_Server_API/app/core/Moderation/policy_compiler.py \
  tldw_Server_API/app/core/Moderation/policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_models_characterization.py \
  tldw_Server_API/tests/unit/test_moderation_models_canonical.py \
  tldw_Server_API/tests/unit/test_moderation_models_imports.py
```

Expected: exit code 0 and no output.

- [ ] **Step 2: Format new/clean files and lint every touched Python file**

Format only the new files and the already-Black-clean evaluator, then prove they are clean:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m black \
  tldw_Server_API/app/core/Moderation/models.py \
  tldw_Server_API/app/core/Moderation/policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_models_characterization.py \
  tldw_Server_API/tests/unit/test_moderation_models_canonical.py \
  tldw_Server_API/tests/unit/test_moderation_models_imports.py
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m black --check \
  tldw_Server_API/app/core/Moderation/models.py \
  tldw_Server_API/app/core/Moderation/policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_models_characterization.py \
  tldw_Server_API/tests/unit/test_moderation_models_canonical.py \
  tldw_Server_API/tests/unit/test_moderation_models_imports.py
```

Do not run Black in write mode over `moderation_service.py` or
`policy_compiler.py`; both fail Black before this extraction and whole-file
formatting would contaminate the structural diff. Run Ruff over every touched
Python file:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m ruff check \
  tldw_Server_API/app/core/Moderation/models.py \
  tldw_Server_API/app/core/Moderation/moderation_service.py \
  tldw_Server_API/app/core/Moderation/policy_compiler.py \
  tldw_Server_API/app/core/Moderation/policy_evaluator.py \
  tldw_Server_API/tests/unit/test_moderation_models_characterization.py \
  tldw_Server_API/tests/unit/test_moderation_models_canonical.py \
  tldw_Server_API/tests/unit/test_moderation_models_imports.py
```

Expected: Black passes for all new/previously clean files and Ruff passes for
the complete touched set. Do not add new ignores. Record the predecessor Black
baseline for the two excluded files in `TASK-13011`.

- [ ] **Step 3: Run the exact focused regression suites**

Run each command independently and record its pass count:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/unit/test_moderation*.py -q
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/unit/test_moderation_test_endpoint_sample.py tldw_Server_API/tests/Guardian/test_supervised_policy.py -q
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py -q
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Workflows/adapters/test_llm_adapters.py -k moderation_adapter -q
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m pytest tldw_Server_API/tests/Audio/test_audio_transcription_retention_and_redaction.py::test_audio_transcriptions_redacts_text_and_segments_when_stt_redaction_enabled -q
```

Expected: every command passes with no changed production caller imports.

- [ ] **Step 4: Run the required security scan**

Run:

```bash
/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/python -m bandit \
  -r tldw_Server_API/app/core/Moderation \
  -f json \
  -o /tmp/bandit_moderation_shared_models.json
```

Expected: exit code 0 and zero new findings. Inspect the JSON result and record total lines scanned, findings, skips, and `nosec` counts in `TASK-13011`.

- [ ] **Step 5: Audit scope and whitespace against the predecessor boundary**

Run:

```bash
git diff --check 285773ea6b05318512f4b004375e394e969e425d..HEAD
git diff --name-status 285773ea6b05318512f4b004375e394e969e425d..HEAD
git log --oneline 285773ea6b05318512f4b004375e394e969e425d..HEAD
```

The production file list must contain only:

```text
tldw_Server_API/app/core/Moderation/models.py
tldw_Server_API/app/core/Moderation/moderation_service.py
tldw_Server_API/app/core/Moderation/policy_compiler.py
tldw_Server_API/app/core/Moderation/policy_evaluator.py
```

Permitted non-production additions are the approved design, this plan, `TASK-13010`, `TASK-13011`, and the three focused model test files. Stop and remove any unrelated file before review.

- [ ] **Step 6: Request an independent whole-branch review**

Give the reviewer the design path, predecessor SHA, current head SHA, exact production scope, and these review questions:

1. Are class identity, metadata, defaults, aliases, and `to_dict()` fallbacks literal?
2. Can model/compiler/evaluator imports load service or config unexpectedly?
3. Are `policy_types()` descriptors, tuple order, subclass dispatch, and public module namespaces preserved?
4. Did any endpoint/schema/caller migration, regex hardening, or unsupported compatibility mechanism enter the diff?
5. Are pickle and monkeypatch boundaries represented exactly as approved?

Resolve each valid finding with a focused test and rerun the affected gate before continuing.

- [ ] **Step 7: Record the verified stacked implementation**

Update `TASK-13011` with all command results, touched files, independent review findings, known predecessor/Change-summary blockers, and the current head. Commit only the tracking update:

```bash
git add "backlog/tasks/task-13011 - Implement-Moderation-shared-models-extraction.md"
git commit -m "chore: verify Moderation shared models extraction"
```

Do not mark `TASK-13011` Done while the predecessor transplant and current-`dev` rerun remain outstanding.

---

### Task 5: Transplant Onto Current Dev And Prepare The Pull Request

**Files:**
- Modify only if verification metadata changes: `backlog/tasks/task-13011 - Implement-Moderation-shared-models-extraction.md`

**Interfaces:**
- Consumes: the verified stacked branch from Task 4 and the merged `TASK-12992` predecessor.
- Produces: a branch based directly on current `origin/dev`, with only post-predecessor commits and complete PR-readiness evidence.

- [ ] **Step 1: Verify rollout prerequisites**

Confirm both conditions before changing history:

1. `TASK-12992` is merged into `dev` through PR #2770.
2. The requester is ready to provide the repository-required human-written `Change summary` explaining what changed and why these choices were made.

If either condition is false, keep `TASK-13011` In Progress, record the blocker, and stop Task 5 without creating a PR.

- [ ] **Step 2: Fetch current dev and preserve a local recovery ref**

Run:

```bash
git fetch origin dev
git worktree add .worktrees/moderation-shared-models-pr -b codex/moderation-shared-models-dev origin/dev
```

Expected: `origin/dev` updates, the original stacked branch remains unchanged as the recovery reference, and the PR branch starts from current `origin/dev`.

- [ ] **Step 3: Replay only post-predecessor commits**

Run:

```bash
git cherry-pick --no-commit da8c4669c2 e53854594f 1a2857666c 9ec5993dfe 8b05563b3f 8c09793fec 9ce47749f8 1038073f9b 5d33b21ca4
```

Resolve conflicts only by reapplying the approved models extraction onto the merged predecessor. Do not reintroduce predecessor-only diffs or broaden production scope.

- [ ] **Step 4: Prove branch ancestry and scope**

Run:

```bash
git merge-base HEAD origin/dev
git rev-parse origin/dev
git log --oneline origin/dev..HEAD
git diff --name-status origin/dev...HEAD
git diff --check origin/dev...HEAD
```

Expected:

- `git merge-base HEAD origin/dev` equals `git rev-parse origin/dev`
- the log contains only design, planning, characterization, extraction, dependency-removal, verification, and correction commits created after the recorded predecessor
- production scope remains the four approved Moderation files

- [ ] **Step 5: Rerun all Task 4 gates on the transplanted head**

Repeat Task 4 Steps 1 through 6 exactly against current `origin/dev`. Record fresh compilation, Black, Ruff, all five pytest commands, Bandit, scope, whitespace, and independent review results. Prior stacked-branch results are not substitutes.

- [ ] **Step 6: Finalize tracking and request the human Change summary**

Update `TASK-13011` with:

- transplanted head SHA and current `origin/dev` SHA
- fresh verification results
- final production/test/documentation file list
- independent review disposition
- explicit statement that PR creation is waiting for the requester's own Change summary

Ask the requester for that summary. Do not draft, paraphrase, or infer it on their behalf.

- [ ] **Step 7: Create the PR only after receiving the requester-owned summary**

Publish the verified rebased branch first:

```bash
git push -u origin codex/moderation-shared-models-dev
```

Use the exact requester-provided Change summary in the PR body, add the verified test/security evidence beneath it, target `dev`, and mark the PR ready rather than draft only if every Task 5 gate remains green.

After successful PR creation, add its URL to `TASK-13011`, check all acceptance criteria and Definition of Done items, add the final summary, set the task to Done, and commit the task update:

```bash
git add "backlog/tasks/task-13011 - Implement-Moderation-shared-models-extraction.md"
git commit -m "chore: finalize Moderation shared models task"
git push origin codex/moderation-shared-models-dev
```

If task-finalization changes are committed after the first push, push the branch again before reporting completion.
