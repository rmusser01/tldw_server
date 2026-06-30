# Moderation PolicyCompiler Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract deterministic moderation policy assembly into `PolicyCompiler` while preserving `ModerationService`, `ModerationPolicy`, and current endpoint/caller behavior.

**Architecture:** Add a focused `policy_compiler.py` module that receives already-loaded inputs and returns compatible `ModerationPolicy` objects plus sanitized internal diagnostics. Keep file I/O, path resolution, locking, logging, persistence, and public service methods in `ModerationService`. Preserve lint output and service helper compatibility through delegating wrappers.

**Tech Stack:** Python 3, dataclasses, `re`, Loguru, pytest, existing `ModerationPolicy`/`PatternRule` types, existing project virtualenv at `/Users/appledev/Documents/GitHub/tldw_server/.venv`.

---

## Source Spec

Design: `Docs/superpowers/specs/2026-06-24-moderation-policy-compiler-refactor-design.md`

Plan Backlog: `TASK-2431`
Implementation Backlog: `TASK-2432`

## File Structure

- Create: `tldw_Server_API/app/core/Moderation/policy_compiler.py`
  - Owns `ResolvedModerationConfig`, `PolicyCompilationInput`, `PolicyCompilationIssue`, `PolicyCompilationReport`, `PolicyCompilationResult`, and `PolicyCompiler`.
  - Owns shared parsing, regex flag, dangerous-regex, category, boolean, and rule compilation helpers.
  - Must not import `ModerationPolicy` or `PatternRule` from `moderation_service.py` at module import time. Use `TYPE_CHECKING` plus a local runtime import helper to avoid a circular import when `ModerationService` imports `PolicyCompiler`.
- Modify: `tldw_Server_API/app/core/Moderation/moderation_service.py`
  - Keeps all I/O, path resolution, persistence, locks, logging, and public methods.
  - Builds `ResolvedModerationConfig`.
  - Reads blocklist lines and PII rules, then passes loaded inputs to `PolicyCompiler`.
  - Keeps compatibility wrappers for `_parse_rule_line()`, `_load_block_patterns()`, and `_build_block_patterns()`.
  - Keeps `lint_blocklist_lines()` endpoint output behavior while delegating parser logic where useful.
- Create: `tldw_Server_API/tests/unit/test_moderation_policy_compiler.py`
  - Focused unit tests for compiler-only behavior.
- Modify: `tldw_Server_API/tests/unit/test_moderation_blocklist_parse.py`
  - Add/adjust compatibility tests for wrappers and lint output.
- Modify: `tldw_Server_API/tests/unit/test_moderation_effective_settings.py`
  - Add/adjust settings and PII effective-policy compatibility tests.
- Modify: `tldw_Server_API/tests/Guardian/test_supervised_policy.py`
  - Add/adjust overlay regression coverage only if compiler integration affects policy construction tests.
- Modify: `backlog/tasks/task-2432 - Implement-Moderation-PolicyCompiler-refactor.md`
  - Track implementation progress, modified files, and verification evidence.

## Conventions

Run Python commands from the implementation worktree after activating the main virtualenv:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
```

Use focused pytest commands during tasks, then run the broader verification command in Task 7.

---

### Task 1: Add Compiler Types And Base Global Compilation

**Files:**
- Create: `tldw_Server_API/app/core/Moderation/policy_compiler.py`
- Create: `tldw_Server_API/tests/unit/test_moderation_policy_compiler.py`

- [ ] **Step 1: Write failing compiler smoke tests**

Add this test file:

```python
import re

from tldw_Server_API.app.core.Moderation.moderation_service import ModerationPolicy, PatternRule
from tldw_Server_API.app.core.Moderation.policy_compiler import (
    PolicyCompilationInput,
    PolicyCompiler,
    ResolvedModerationConfig,
)


def _config(**overrides):
    values = {
        "enabled": True,
        "input_enabled": True,
        "output_enabled": True,
        "input_action": "block",
        "output_action": "redact",
        "redact_replacement": "[REDACTED]",
        "per_user_overrides": True,
        "categories_enabled": None,
        "pii_enabled": False,
    }
    values.update(overrides)
    return ResolvedModerationConfig(**values)


def test_compile_global_policy_uses_resolved_defaults():
    result = PolicyCompiler().compile_global(
        PolicyCompilationInput(
            config=_config(categories_enabled={"pii", "confidential"}),
            runtime_override={},
            blocklist_lines=[],
            pii_rules=[],
        )
    )

    assert isinstance(result.policy, ModerationPolicy)
    assert result.policy.enabled is True
    assert result.policy.input_action == "block"
    assert result.policy.output_action == "redact"
    assert result.policy.categories_enabled == {"pii", "confidential"}
    assert result.report.issues == []


def test_compile_global_policy_copies_pii_rules_when_enabled():
    pii_rule = PatternRule(
        regex=re.compile("email", re.IGNORECASE),
        action="redact",
        replacement="[PII]",
        categories={"pii", "pii_email"},
    )

    result = PolicyCompiler().compile_global(
        PolicyCompilationInput(
            config=_config(pii_enabled=True),
            runtime_override={},
            blocklist_lines=[],
            pii_rules=[pii_rule],
        )
    )

    assert result.policy.block_patterns == [pii_rule]
```

- [ ] **Step 2: Run smoke tests to verify they fail**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest tldw_Server_API/tests/unit/test_moderation_policy_compiler.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `policy_compiler`.

- [ ] **Step 3: Add minimal compiler types and global compile path**

Create `tldw_Server_API/app/core/Moderation/policy_compiler.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Moderation.moderation_service import (
        ModerationPolicy,
        PatternRule,
    )


@dataclass(frozen=True)
class ResolvedModerationConfig:
    enabled: bool = False
    input_enabled: bool = True
    output_enabled: bool = True
    input_action: str = "block"
    output_action: str = "redact"
    redact_replacement: str = "[REDACTED]"
    per_user_overrides: bool = True
    categories_enabled: set[str] | None = None
    pii_enabled: bool = False


@dataclass(frozen=True)
class PolicyCompilationIssue:
    source: str
    reason: str
    index: int | None = None
    detail: str | None = None


@dataclass
class PolicyCompilationReport:
    issues: list[PolicyCompilationIssue] = field(default_factory=list)

    def add(self, source: str, reason: str, *, index: int | None = None, detail: str | None = None) -> None:
        self.issues.append(PolicyCompilationIssue(source=source, reason=reason, index=index, detail=detail))


@dataclass
class PolicyCompilationInput:
    config: ResolvedModerationConfig
    runtime_override: dict[str, object] = field(default_factory=dict)
    blocklist_lines: list[str] = field(default_factory=list)
    user_override: dict[str, object] | None = None
    pii_rules: list[PatternRule] = field(default_factory=list)


@dataclass
class PolicyCompilationResult:
    policy: ModerationPolicy
    report: PolicyCompilationReport


class PolicyCompiler:
    @staticmethod
    def policy_types() -> tuple[type[ModerationPolicy], type[PatternRule]]:
        from tldw_Server_API.app.core.Moderation.moderation_service import (
            ModerationPolicy,
            PatternRule,
        )

        return ModerationPolicy, PatternRule

    def compile_global(self, data: PolicyCompilationInput) -> PolicyCompilationResult:
        report = PolicyCompilationReport()
        config = data.config
        ModerationPolicy, PatternRule = self.policy_types()
        categories_enabled = self.resolve_runtime_categories(
            data.runtime_override,
            config.categories_enabled,
        )
        pii_enabled = self.resolve_runtime_pii(data.runtime_override, config.pii_enabled)
        block_patterns: list[PatternRule] = []
        if pii_enabled:
            block_patterns.extend(list(data.pii_rules or []))

        policy = ModerationPolicy(
            enabled=config.enabled,
            input_enabled=config.input_enabled,
            output_enabled=config.output_enabled,
            input_action=str(config.input_action).lower(),
            output_action=str(config.output_action).lower(),
            redact_replacement=config.redact_replacement,
            per_user_overrides=config.per_user_overrides,
            block_patterns=block_patterns,
            categories_enabled=categories_enabled,
        )
        return PolicyCompilationResult(policy=policy, report=report)

    @staticmethod
    def resolve_runtime_pii(runtime_override: dict[str, object], default: bool) -> bool:
        if "pii_enabled" in runtime_override:
            return bool(runtime_override.get("pii_enabled"))
        return bool(default)

    @staticmethod
    def resolve_runtime_categories(
        runtime_override: dict[str, object],
        default: set[str] | None,
    ) -> set[str] | None:
        if "categories_enabled" not in runtime_override:
            return set(default) if default is not None else None
        raw = runtime_override.get("categories_enabled") or []
        if isinstance(raw, (set, list, tuple)):
            parsed = {str(c).strip().lower() for c in raw if str(c).strip()}
            return parsed or None
        if isinstance(raw, str):
            parsed = {c.strip().lower() for c in raw.split(",") if c.strip()}
            return parsed or None
        return set(default) if default is not None else None
```

- [ ] **Step 4: Run smoke tests to verify they pass**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest tldw_Server_API/tests/unit/test_moderation_policy_compiler.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add tldw_Server_API/app/core/Moderation/policy_compiler.py tldw_Server_API/tests/unit/test_moderation_policy_compiler.py
git commit -m "Add moderation policy compiler skeleton"
```

---

### Task 2: Move Blocklist Parsing And Safe Rule Compilation Into PolicyCompiler

**Files:**
- Modify: `tldw_Server_API/app/core/Moderation/policy_compiler.py`
- Modify: `tldw_Server_API/tests/unit/test_moderation_policy_compiler.py`

- [ ] **Step 1: Add failing parser and report tests**

Append tests:

```python
def test_compile_global_policy_compiles_literal_and_regex_blocklist_rules():
    result = PolicyCompiler().compile_global(
        PolicyCompilationInput(
            config=_config(),
            runtime_override={},
            blocklist_lines=[
                "secret -> block #confidential",
                r"/leak\\d+/i -> redact:[MASK] #pii",
            ],
            pii_rules=[],
        )
    )

    rules = result.policy.block_patterns
    assert len(rules) == 2
    assert rules[0].regex.search("SECRET")
    assert rules[0].action == "block"
    assert rules[0].categories == {"confidential"}
    assert rules[1].regex.search("leak123")
    assert rules[1].action == "redact"
    assert rules[1].replacement == "[MASK]"
    assert rules[1].categories == {"pii"}


def test_compile_global_policy_reports_invalid_lines_without_raw_regex():
    result = PolicyCompiler().compile_global(
        PolicyCompilationInput(
            config=_config(),
            runtime_override={},
            blocklist_lines=[
                "secret -> invalid_action",
                "/(a+)+$/ -> block",
                "/(unclosed/ -> block",
            ],
            pii_rules=[],
        )
    )

    assert result.policy.block_patterns == []
    reasons = [issue.reason for issue in result.report.issues]
    assert reasons == ["invalid_action", "dangerous_regex", "invalid_regex"]
    rendered = repr(result.report.issues)
    assert "(a+)+$" not in rendered
    assert "(unclosed" not in rendered
```

- [ ] **Step 2: Run parser tests to verify they fail**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest tldw_Server_API/tests/unit/test_moderation_policy_compiler.py::test_compile_global_policy_compiles_literal_and_regex_blocklist_rules tldw_Server_API/tests/unit/test_moderation_policy_compiler.py::test_compile_global_policy_reports_invalid_lines_without_raw_regex -q
```

Expected: FAIL because blocklist lines are not compiled yet.

- [ ] **Step 3: Add parser, regex, and report helpers**

Update `PolicyCompiler` with these helpers and call `compile_blocklist_lines()` before appending PII rules:

```python
import re


class PolicyCompiler:
    _ALLOWED_REGEX_FLAGS = {"i", "m", "s", "x"}
    _ALLOWED_ACTIONS = {"block", "redact", "warn"}

    def compile_global(self, data: PolicyCompilationInput) -> PolicyCompilationResult:
        report = PolicyCompilationReport()
        config = data.config
        ModerationPolicy, _ = self.policy_types()
        categories_enabled = self.resolve_runtime_categories(data.runtime_override, config.categories_enabled)
        pii_enabled = self.resolve_runtime_pii(data.runtime_override, config.pii_enabled)
        block_patterns = self.compile_blocklist_lines(data.blocklist_lines, report)
        if pii_enabled:
            block_patterns.extend(list(data.pii_rules or []))
        policy = ModerationPolicy(
            enabled=config.enabled,
            input_enabled=config.input_enabled,
            output_enabled=config.output_enabled,
            input_action=str(config.input_action).lower(),
            output_action=str(config.output_action).lower(),
            redact_replacement=config.redact_replacement,
            per_user_overrides=config.per_user_overrides,
            block_patterns=block_patterns,
            categories_enabled=categories_enabled,
        )
        return PolicyCompilationResult(policy=policy, report=report)

    def compile_blocklist_lines(
        self,
        lines: list[str],
        report: PolicyCompilationReport | None = None,
    ) -> list[PatternRule]:
        compiled: list[PatternRule] = []
        active_report = report or PolicyCompilationReport()
        for idx, raw in enumerate(lines or []):
            line = str(raw).strip()
            if not line or line.startswith("#"):
                continue
            expr, action, replacement, categories = self.parse_rule_line(line)
            if expr is None:
                active_report.add("blocklist", "empty_pattern", index=idx)
                continue
            if action and not self.is_valid_action(action):
                active_report.add("blocklist", "invalid_action", index=idx)
                continue
            rule = self.compile_rule_expression(
                expr,
                action=action,
                replacement=replacement,
                categories=categories,
                phase="both",
                report=active_report,
                source="blocklist",
                index=idx,
            )
            if rule is not None:
                compiled.append(rule)
        return compiled

    @classmethod
    def parse_rule_line(cls, text: str) -> tuple[str | None, str | None, str | None, set[str] | None]:
        if not text:
            return None, None, None, None
        line = text
        action = None
        replacement = None
        categories: set[str] | None = None
        if "#" in line:
            cut_index = cls._find_category_suffix(line)
            if cut_index != -1:
                after = line[cut_index + 1:]
                cats = {c.strip().lower() for c in after.split(",") if c.strip()}
                if cats:
                    categories = cats
                    line = line[:cut_index].strip()
        if "->" in line:
            lhs, rhs = cls.split_action_directive(line)
            if rhs is not None:
                line = lhs
                if rhs:
                    rhs_lower = rhs.lower()
                    if rhs_lower.startswith("redact:"):
                        action = "redact"
                        replacement = rhs[len("redact:"):].strip()
                    elif rhs_lower in cls._ALLOWED_ACTIONS:
                        action = rhs_lower
                    else:
                        action = rhs
        return line, action, replacement, categories
```

Add these helper methods to `PolicyCompiler`:

```python
    @staticmethod
    def _find_category_suffix(text: str) -> int:
        if "#" not in text:
            return -1
        for i in range(len(text) - 1, -1, -1):
            if text[i] != "#":
                continue
            backslash_count = 0
            j = i - 1
            while j >= 0 and text[j] == "\\":
                backslash_count += 1
                j -= 1
            escaped = backslash_count % 2 == 1
            previous = text[i - 1] if i > 0 else ""
            if not escaped and (i == 0 or previous.isspace()):
                return i
        return -1

    @staticmethod
    def split_action_directive(text: str) -> tuple[str, str | None]:
        if "->" not in text:
            return text, None
        in_regex = False
        escape = False
        for i in range(len(text) - 1):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == "/" and i == 0:
                in_regex = True
                continue
            if ch == "/" and in_regex:
                in_regex = False
                continue
            if not in_regex and text[i:i + 2] == "->":
                backslash_count = 0
                j = i - 1
                while j >= 0 and text[j] == "\\":
                    backslash_count += 1
                    j -= 1
                if backslash_count % 2 == 1:
                    continue
                return text[:i].strip(), text[i + 2:].strip()
        return text, None

    @classmethod
    def parse_regex_expr(cls, expr: str) -> tuple[str, str] | None:
        if not expr or not expr.startswith("/"):
            return None
        last_slash = expr.rfind("/")
        if last_slash <= 0:
            return None
        flags = expr[last_slash + 1:]
        if flags:
            lowered = flags.lower()
            if any(ch not in cls._ALLOWED_REGEX_FLAGS for ch in lowered):
                return None
        raw = expr[1:last_slash]
        if raw == "":
            return None
        return raw, flags

    @classmethod
    def regex_flags(cls, flags: str | None) -> int:
        value = re.IGNORECASE
        lowered = (flags or "").lower()
        if "i" in lowered:
            value |= re.IGNORECASE
        if "m" in lowered:
            value |= re.MULTILINE
        if "s" in lowered:
            value |= re.DOTALL
        if "x" in lowered:
            value |= re.VERBOSE
        return value

    @classmethod
    def is_valid_action(cls, action: str) -> bool:
        return str(action).strip().lower() in cls._ALLOWED_ACTIONS

    @staticmethod
    def has_nested_quantifiers(expr: str) -> bool:
        try:
            return bool(re.search(r"\((?:[^)(]|\([^)(]*\))*[+*][^)]*\)\s*[+*]", expr))
        except (TypeError, ValueError, re.error):
            return False

    @staticmethod
    def too_many_groups(expr: str, limit: int = 100) -> bool:
        try:
            return expr.count("(") - expr.count("\\(") > limit
        except (TypeError, ValueError):
            return False

    def is_regex_dangerous(self, expr: str) -> bool:
        if not expr:
            return True
        if len(expr) > 2000:
            return True
        if self.has_nested_quantifiers(expr):
            return True
        return self.too_many_groups(expr)
```

Use this rule compilation body:

```python
    def compile_rule_expression(
        self,
        expr: str,
        *,
        action: str | None,
        replacement: str | None,
        categories: set[str] | None,
        phase: str,
        report: PolicyCompilationReport,
        source: str,
        index: int | None,
    ) -> PatternRule | None:
        _, PatternRule = self.policy_types()
        try:
            regex_parts = self.parse_regex_expr(expr)
            if regex_parts:
                raw, flags_str = regex_parts
                if self.is_regex_dangerous(raw):
                    report.add(source, "dangerous_regex", index=index)
                    return None
                flags = self.regex_flags(flags_str)
                regex = re.compile(raw, flags=flags)
            else:
                regex = re.compile(re.escape(expr.replace("\\#", "#")), flags=re.IGNORECASE)
        except re.error:
            report.add(source, "invalid_regex", index=index)
            return None
        return PatternRule(
            regex=regex,
            action=action or None,
            replacement=replacement or None,
            categories=categories or None,
            phase=phase,
        )
```

- [ ] **Step 4: Run parser tests to verify they pass**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest tldw_Server_API/tests/unit/test_moderation_policy_compiler.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add tldw_Server_API/app/core/Moderation/policy_compiler.py tldw_Server_API/tests/unit/test_moderation_policy_compiler.py
git commit -m "Move moderation rule compilation into policy compiler"
```

---

### Task 3: Preserve ModerationService Parser Wrappers And Public Lint Output

**Files:**
- Modify: `tldw_Server_API/app/core/Moderation/moderation_service.py`
- Modify: `tldw_Server_API/tests/unit/test_moderation_blocklist_parse.py`

- [ ] **Step 1: Add failing compatibility tests for wrappers and lint output**

Add or adjust tests:

```python
def test_service_parser_wrappers_delegate_to_policy_compiler():
    svc = ModerationService()

    expr, action, repl, cats = svc._parse_rule_line("/leak\\d+/ -> redact:[MASK] #pii")

    assert expr == "/leak\\d+/"
    assert action == "redact"
    assert repl == "[MASK]"
    assert cats == {"pii"}


def test_lint_blocklist_lines_keeps_public_response_shape():
    svc = ModerationService()

    result = svc.lint_blocklist_lines(["/foo/z", "secret -> block #confidential"])

    assert set(result) == {"items", "valid_count", "invalid_count"}
    invalid = result["items"][0]
    valid = result["items"][1]
    assert invalid["line"] == "/foo/z"
    assert invalid["ok"] is True
    assert invalid["pattern_type"] == "literal"
    assert invalid["warning"] == "invalid regex flags; treating as literal"
    assert valid["line"] == "secret -> block #confidential"
    assert valid["ok"] is True
    assert valid["pattern_type"] == "literal"
    assert valid["sample"] == "secret"
    assert valid["categories"] == ["confidential"]
```

- [ ] **Step 2: Run compatibility tests**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest tldw_Server_API/tests/unit/test_moderation_blocklist_parse.py::test_service_parser_wrappers_delegate_to_policy_compiler tldw_Server_API/tests/unit/test_moderation_blocklist_parse.py::test_lint_blocklist_lines_keeps_public_response_shape -q
```

Expected: The wrapper test may pass before refactor; lint shape should pass before and after. Keep both tests as regression coverage.

- [ ] **Step 3: Add PolicyCompiler dependency and wrapper delegation**

In `moderation_service.py`, import the compiler:

```python
from tldw_Server_API.app.core.Moderation.policy_compiler import (
    PolicyCompilationInput,
    PolicyCompilationReport,
    PolicyCompiler,
    ResolvedModerationConfig,
)
```

In `ModerationService.__init__`, create one compiler:

```python
self._policy_compiler = PolicyCompiler()
```

Change wrapper methods to delegate:

```python
def _parse_rule_line(self, s: str) -> tuple[str | None, str | None, str | None, set[str] | None]:
    return self._policy_compiler.parse_rule_line(s)

@staticmethod
def _split_action_directive(text: str) -> tuple[str, str | None]:
    return PolicyCompiler.split_action_directive(text)

@classmethod
def _parse_regex_expr(cls, expr: str) -> tuple[str, str] | None:
    return PolicyCompiler.parse_regex_expr(expr)

def _is_regex_dangerous(self, expr: str) -> bool:
    return self._policy_compiler.is_regex_dangerous(expr)
```

Keep `lint_blocklist_lines()` response assembly in `ModerationService`. It may call compiler parser helpers, but it must keep existing `line`, `sample`, `error`, `warning`, `valid_count`, and `invalid_count` behavior.

- [ ] **Step 4: Run blocklist parse tests**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest tldw_Server_API/tests/unit/test_moderation_blocklist_parse.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add tldw_Server_API/app/core/Moderation/moderation_service.py tldw_Server_API/tests/unit/test_moderation_blocklist_parse.py
git commit -m "Preserve moderation parser and lint compatibility"
```

---

### Task 4: Integrate Global Policy Compilation Into ModerationService

**Files:**
- Modify: `tldw_Server_API/app/core/Moderation/moderation_service.py`
- Modify: `tldw_Server_API/tests/unit/test_moderation_policy_compiler.py`
- Modify: `tldw_Server_API/tests/unit/test_moderation_effective_settings.py`

- [ ] **Step 1: Add failing service integration tests**

Add tests that assert service-built policies match existing behavior:

```python
def test_service_global_policy_uses_compiler_without_leaking_paths(tmp_path, monkeypatch):
    blocklist = tmp_path / "blocklist.txt"
    blocklist.write_text("secret -> block #confidential\n", encoding="utf-8")

    svc = ModerationService()
    svc._blocklist_path = str(blocklist)
    svc._runtime_override = {}
    svc._policy_compiler = PolicyCompiler()

    policy = svc._compile_global_policy_from_resolved_config(
        ResolvedModerationConfig(
            enabled=True,
            input_enabled=True,
            output_enabled=True,
            input_action="block",
            output_action="redact",
            redact_replacement="[REDACTED]",
            per_user_overrides=True,
            categories_enabled=None,
            pii_enabled=False,
        )
    )

    assert policy.enabled is True
    assert len(policy.block_patterns) == 1
    assert policy.block_patterns[0].categories == {"confidential"}
```

Keep the helper name `_compile_global_policy_from_resolved_config()` so this integration test can target a stable service boundary in the implementation branch.

- [ ] **Step 2: Run integration test to verify it fails**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest tldw_Server_API/tests/unit/test_moderation_policy_compiler.py::test_service_global_policy_uses_compiler_without_leaking_paths -q
```

Expected: FAIL because service integration helper is not implemented yet.

- [ ] **Step 3: Build `ResolvedModerationConfig` in the service**

Refactor `_load_global_policy()` into three internal steps:

```python
def _load_global_policy(self) -> ModerationPolicy:
    mod_cfg = self._load_moderation_config_section()
    resolved = self._resolve_moderation_config(mod_cfg)
    self._user_overrides_path = resolved.user_overrides_path
    self._blocklist_path = resolved.blocklist_path
    self._runtime_overrides_path = resolved.runtime_overrides_path
    return self._compile_global_policy_from_resolved_config(resolved.compiler_config)
```

Add this service-only dataclass near the existing moderation dataclasses:

```python
@dataclass(frozen=True)
class _ResolvedModerationServiceState:
    compiler_config: ResolvedModerationConfig
    blocklist_path: str | None
    user_overrides_path: str | None
    runtime_overrides_path: str | None
```

Have `_resolve_moderation_config()` return `_ResolvedModerationServiceState`. Do not put paths in `ResolvedModerationConfig`. Keep runtime overrides out of `_resolve_moderation_config()`; the compiler applies runtime category and PII overrides so the service can load explicit PII rules from the effective PII state.

Implement config loading and resolution with existing defaults and env fallback semantics:

```python
def _load_moderation_config_section(self) -> dict[str, object]:
    mod_cfg = (self._config.get("moderation") or {}) if isinstance(self._config, dict) else {}
    if mod_cfg:
        return dict(mod_cfg)
    try:
        parser = load_comprehensive_config()
        if parser and parser.has_section("Moderation"):
            return dict(parser.items("Moderation"))
    except _MODERATION_NONCRITICAL_EXCEPTIONS:
        return {}
    return {}


def _resolve_moderation_config(self, mod_cfg: dict[str, object]) -> _ResolvedModerationServiceState:
    def _b(key: str, default: bool) -> bool:
        val = str(mod_cfg.get(key, default)).strip().lower()
        return is_truthy(val)

    def _anchor(path_value: str) -> str:
        try:
            from pathlib import Path as _Path
            path = _Path(str(path_value))
            if path.is_absolute():
                return str(path)
            from tldw_Server_API.app.core.Utils.Utils import get_project_root
            return str((_Path(get_project_root()) / path).resolve())
        except _MODERATION_NONCRITICAL_EXCEPTIONS:
            return str(path_value)

    blocklist_path = (
        mod_cfg.get("blocklist_file")
        or os.getenv("MODERATION_BLOCKLIST_FILE")
        or "tldw_Server_API/Config_Files/moderation_blocklist.txt"
    )
    user_overrides_path = (
        mod_cfg.get("user_overrides_file")
        or os.getenv("MODERATION_USER_OVERRIDES_FILE")
        or "tldw_Server_API/Config_Files/moderation_user_overrides.json"
    )
    runtime_overrides_path = (
        mod_cfg.get("runtime_overrides_file")
        or os.getenv("MODERATION_RUNTIME_OVERRIDES_FILE")
        or "tldw_Server_API/Config_Files/moderation_runtime_overrides.json"
    )

    with contextlib.suppress(_MODERATION_NONCRITICAL_EXCEPTIONS):
        self._max_scan_chars = int(mod_cfg.get("max_scan_chars", self._max_scan_chars))
    with contextlib.suppress(_MODERATION_NONCRITICAL_EXCEPTIONS):
        self._max_replacements_per_pattern = int(
            mod_cfg.get("max_replacements_per_pattern", self._max_replacements_per_pattern)
        )
    with contextlib.suppress(_MODERATION_NONCRITICAL_EXCEPTIONS):
        self._match_window_chars = int(mod_cfg.get("match_window_chars", self._match_window_chars))
    with contextlib.suppress(_MODERATION_NONCRITICAL_EXCEPTIONS):
        if "blocklist_write_debounce_ms" in mod_cfg:
            self._write_debounce_ms = int(mod_cfg.get("blocklist_write_debounce_ms", self._write_debounce_ms) or 0)

    cats_val = mod_cfg.get("categories_enabled") if "categories_enabled" in mod_cfg else os.getenv("MODERATION_CATEGORIES_ENABLED", "")
    categories_enabled: set[str] = set()
    if isinstance(cats_val, (list, set, tuple)):
        categories_enabled = {str(c).strip().lower() for c in cats_val if str(c).strip()}
    elif isinstance(cats_val, str) and cats_val.strip():
        categories_enabled = {c.strip().lower() for c in cats_val.split(",") if c.strip()}
    elif cats_val:
        logger.warning("Invalid moderation categories_enabled type")

    pii_enabled = is_truthy(
        str(mod_cfg.get("pii_enabled", os.getenv("MODERATION_PII_ENABLED", "false"))).strip().lower()
    )

    compiler_config = ResolvedModerationConfig(
        enabled=_b("enabled", False),
        input_enabled=_b("input_enabled", True),
        output_enabled=_b("output_enabled", True),
        input_action=str(mod_cfg.get("input_action", "block")).lower(),
        output_action=str(mod_cfg.get("output_action", "redact")).lower(),
        redact_replacement=str(mod_cfg.get("redact_replacement", "[REDACTED]")),
        per_user_overrides=_b("per_user_overrides", True),
        categories_enabled=categories_enabled or None,
        pii_enabled=bool(pii_enabled),
    )
    return _ResolvedModerationServiceState(
        compiler_config=compiler_config,
        blocklist_path=_anchor(blocklist_path) if blocklist_path else None,
        user_overrides_path=_anchor(user_overrides_path) if user_overrides_path else None,
        runtime_overrides_path=_anchor(runtime_overrides_path) if runtime_overrides_path else None,
    )
```

- [ ] **Step 4: Read blocklist lines and pass explicit PII rules**

Add a service helper:

```python
def _read_blocklist_lines_for_compile(self, path: str | None) -> list[str]:
    if not path:
        return []
    if not os.path.exists(path):
        logger.warning("Moderation blocklist file not found")
        return []
    try:
        return self._read_blocklist_lines_from_path(path)
    except _MODERATION_NONCRITICAL_EXCEPTIONS:
        logger.error("Failed to load moderation blocklist")
        return []
```

Compile globally with explicit PII rules:

```python
def _compile_global_policy_from_resolved_config(self, config: ResolvedModerationConfig) -> ModerationPolicy:
    effective_pii_enabled = self._policy_compiler.resolve_runtime_pii(
        self._runtime_override,
        config.pii_enabled,
    )
    self._pii_enabled = bool(effective_pii_enabled)
    pii_rules = self._load_builtin_pii_rules() if effective_pii_enabled else []
    lines = self._read_blocklist_lines_for_compile(getattr(self, "_blocklist_path", None))
    result = self._policy_compiler.compile_global(
        PolicyCompilationInput(
            config=config,
            runtime_override=self._runtime_override,
            blocklist_lines=lines,
            pii_rules=pii_rules,
        )
    )
    self._log_compilation_report(result.report)
    return result.policy
```

Log sanitized report reasons only:

```python
def _log_compilation_report(self, report: PolicyCompilationReport) -> None:
    for issue in report.issues:
        if issue.reason == "invalid_action":
            logger.warning("Invalid moderation action in blocklist; skipping line")
        elif issue.reason == "dangerous_regex":
            logger.warning("Skipped dangerous regex in blocklist")
        elif issue.reason == "invalid_regex":
            logger.warning("Invalid blocklist pattern; skipping line")
```

- [ ] **Step 5: Keep `_load_block_patterns()` and `_build_block_patterns()` wrappers**

Keep wrappers for tests and adjacent code:

```python
def _load_block_patterns(self, path: str | None) -> list[PatternRule]:
    report = PolicyCompilationReport()
    lines = self._read_blocklist_lines_for_compile(path)
    rules = self._policy_compiler.compile_blocklist_lines(lines, report)
    self._log_compilation_report(report)
    return rules

def _build_block_patterns(self, path: str | None) -> list[PatternRule]:
    patterns = self._load_block_patterns(path)
    if self._pii_enabled:
        try:
            patterns.extend(self._load_builtin_pii_rules())
        except _MODERATION_NONCRITICAL_EXCEPTIONS:
            logger.warning("Failed to load builtin PII rules")
    return patterns
```

- [ ] **Step 6: Run global integration tests**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest tldw_Server_API/tests/unit/test_moderation_policy_compiler.py tldw_Server_API/tests/unit/test_moderation_effective_settings.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 4**

Run:

```bash
git add tldw_Server_API/app/core/Moderation/moderation_service.py tldw_Server_API/app/core/Moderation/policy_compiler.py tldw_Server_API/tests/unit/test_moderation_policy_compiler.py tldw_Server_API/tests/unit/test_moderation_effective_settings.py
git commit -m "Integrate moderation global policy compiler"
```

---

### Task 5: Move Per-User Effective Policy Assembly Into PolicyCompiler

**Files:**
- Modify: `tldw_Server_API/app/core/Moderation/policy_compiler.py`
- Modify: `tldw_Server_API/app/core/Moderation/moderation_service.py`
- Modify: `tldw_Server_API/tests/unit/test_moderation_policy_compiler.py`
- Modify: `tldw_Server_API/tests/unit/test_moderation_blocklist_parse.py`

- [ ] **Step 1: Add failing user override compiler tests**

Append tests:

```python
def test_compile_user_policy_preserves_empty_category_override():
    base = ModerationPolicy(
        enabled=True,
        input_enabled=True,
        output_enabled=True,
        input_action="block",
        output_action="redact",
        redact_replacement="[REDACTED]",
        per_user_overrides=True,
        block_patterns=[],
        categories_enabled={"pii"},
    )

    result = PolicyCompiler().compile_user_policy(
        base,
        {
            "enabled": True,
            "categories_enabled": "",
            "rules": [],
        },
    )

    assert result.policy.categories_enabled == set()


def test_compile_user_policy_adds_wildcard_quick_rules():
    base = ModerationPolicy(enabled=True, block_patterns=[], categories_enabled={"pii"})

    result = PolicyCompiler().compile_user_policy(
        base,
        {
            "rules": [
                {
                    "id": "r1",
                    "pattern": "secret",
                    "is_regex": False,
                    "action": "warn",
                    "phase": "input",
                }
            ]
        },
    )

    assert len(result.policy.block_patterns) == 1
    rule = result.policy.block_patterns[0]
    assert rule.regex.search("secret")
    assert rule.action == "warn"
    assert rule.categories == {"*"}
    assert rule.phase == "input"


def test_compile_user_policy_accepts_legacy_bool_like_is_regex_values():
    base = ModerationPolicy(enabled=True, block_patterns=[])

    result = PolicyCompiler().compile_user_policy(
        base,
        {
            "rules": [
                {
                    "id": "r1",
                    "pattern": r"token-\d+",
                    "is_regex": "yes",
                    "action": "block",
                    "phase": "input",
                },
                {
                    "id": "r2",
                    "pattern": "literal.*",
                    "is_regex": "false",
                    "action": "warn",
                    "phase": "output",
                },
            ]
        },
    )

    assert len(result.policy.block_patterns) == 2
    assert result.policy.block_patterns[0].regex.search("token-42")
    assert result.policy.block_patterns[1].regex.search("literal.*")
    assert not result.policy.block_patterns[1].regex.search("literalabc")
    assert result.report.issues == []
```

- [ ] **Step 2: Run user tests to verify they fail**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/unit/test_moderation_policy_compiler.py::test_compile_user_policy_preserves_empty_category_override \
  tldw_Server_API/tests/unit/test_moderation_policy_compiler.py::test_compile_user_policy_adds_wildcard_quick_rules \
  tldw_Server_API/tests/unit/test_moderation_policy_compiler.py::test_compile_user_policy_accepts_legacy_bool_like_is_regex_values \
  -q
```

Expected: FAIL because `compile_user_policy()` does not exist.

- [ ] **Step 3: Implement compiler user policy assembly**

Add to `PolicyCompiler`:

```python
    def compile_user_policy(
        self,
        base_policy: ModerationPolicy,
        override: dict[str, object] | None,
    ) -> PolicyCompilationResult:
        report = PolicyCompilationReport()
        if not override:
            return PolicyCompilationResult(policy=base_policy, report=report)
        policy = ModerationPolicy(
            enabled=self.coalesce_bool(override.get("enabled"), base_policy.enabled),
            input_enabled=self.coalesce_bool(override.get("input_enabled"), base_policy.input_enabled),
            output_enabled=self.coalesce_bool(override.get("output_enabled"), base_policy.output_enabled),
            input_action=str(override.get("input_action", base_policy.input_action)).lower(),
            output_action=str(override.get("output_action", base_policy.output_action)).lower(),
            redact_replacement=str(override.get("redact_replacement", base_policy.redact_replacement)),
            per_user_overrides=base_policy.per_user_overrides,
            block_patterns=list(base_policy.block_patterns or []),
            categories_enabled=self.resolve_categories_override(
                override,
                base_policy.categories_enabled,
            ),
        )
        rules_raw = override.get("rules")
        if isinstance(rules_raw, list):
            for idx, raw_rule in enumerate(rules_raw):
                compiled = self.compile_user_rule(raw_rule, report, idx)
                if compiled is not None:
                    policy.block_patterns.append(compiled)
        return PolicyCompilationResult(policy=policy, report=report)
```

Add helper implementations using current service semantics:

```python
    @staticmethod
    def coalesce_bool(value: object, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}

    @staticmethod
    def parse_bool_value(value: object) -> bool | None:
        if isinstance(value, bool):
            return value
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"1", "true", "yes", "on", "y"}:
                return True
            if text in {"0", "false", "no", "n", "off"}:
                return False
        return None

    def resolve_categories_override(
        self,
        override: dict[str, object],
        default_categories: set[str] | None,
    ) -> set[str] | None:
        if "categories_enabled" not in override:
            return default_categories
        parsed = self.parse_categories_override(override.get("categories_enabled"))
        return parsed if parsed is not None else default_categories

    @staticmethod
    def parse_categories_override(value: object | None) -> set[str] | None:
        if value is None:
            return None
        if isinstance(value, list):
            return {str(x).strip().lower() for x in value if str(x).strip()}
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return set()
            return {c.strip().lower() for c in text.split(",") if c.strip()}
        return None
```

Compile quick rules with existing wildcard categories:

```python
    def compile_user_rule(
        self,
        raw_rule: object,
        report: PolicyCompilationReport,
        index: int,
    ) -> PatternRule | None:
        if not isinstance(raw_rule, dict):
            report.add("user_rule", "invalid_rule", index=index)
            return None
        pattern = str(raw_rule.get("pattern", "")).strip()
        action = str(raw_rule.get("action", "")).strip().lower()
        phase = str(raw_rule.get("phase", "both")).strip().lower()
        is_regex = self.parse_bool_value(raw_rule.get("is_regex", False))
        if is_regex is None:
            report.add("user_rule", "invalid_is_regex", index=index)
            return None
        if not pattern or action not in {"block", "warn"}:
            report.add("user_rule", "invalid_rule", index=index)
            return None
        if phase not in {"input", "output", "both"}:
            phase = "both"
        try:
            if is_regex:
                if self.is_regex_dangerous(pattern):
                    report.add("user_rule", "dangerous_regex", index=index)
                    return None
                regex = re.compile(pattern, flags=re.IGNORECASE)
            else:
                regex = re.compile(re.escape(pattern), flags=re.IGNORECASE)
        except re.error:
            report.add("user_rule", "invalid_regex", index=index)
            return None
        return PatternRule(regex=regex, action=action, replacement=None, categories={"*"}, phase=phase)
```

- [ ] **Step 4: Wire `get_effective_policy()` to compiler user assembly**

In `ModerationService.get_effective_policy()`:

```python
def get_effective_policy(self, user_id: str | None) -> ModerationPolicy:
    policy = self._global_policy
    if not policy.per_user_overrides or not user_id:
        return policy
    override = self._user_overrides.get(str(user_id))
    if not override:
        return policy
    result = self._policy_compiler.compile_user_policy(policy, override)
    self._log_compilation_report(result.report)
    return result.policy
```

Keep `_parse_categories_override()`, `_resolve_categories_override()`, `_coalesce_bool()`, and `_compile_user_rule()` as delegating wrappers if tests or adjacent code call them directly.

- [ ] **Step 5: Run user override tests**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest tldw_Server_API/tests/unit/test_moderation_policy_compiler.py tldw_Server_API/tests/unit/test_moderation_blocklist_parse.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 5**

Run:

```bash
git add tldw_Server_API/app/core/Moderation/policy_compiler.py tldw_Server_API/app/core/Moderation/moderation_service.py tldw_Server_API/tests/unit/test_moderation_policy_compiler.py tldw_Server_API/tests/unit/test_moderation_blocklist_parse.py
git commit -m "Move moderation user policy assembly into compiler"
```

---

### Task 6: Preserve Service Behavior Across Reload, Settings, Blocklist, And Supervised Overlay

**Files:**
- Modify: `tldw_Server_API/tests/unit/test_moderation_blocklist_parse.py`
- Modify: `tldw_Server_API/tests/unit/test_moderation_effective_settings.py`
- Modify: `tldw_Server_API/tests/Guardian/test_supervised_policy.py`
- Modify: `tldw_Server_API/app/core/Moderation/moderation_service.py` only if tests expose integration drift.

- [ ] **Step 1: Add regression tests for recompile triggers**

Add the settings regression to `test_moderation_effective_settings.py` and the blocklist write regression to `test_moderation_blocklist_parse.py`. If both files need `_tmp_moderation_config()`, define the helper locally in each file or reuse an equivalent helper that already exists in that same file; do not import test helpers across test modules.

```python
def _tmp_moderation_config(tmp_path, blocklist_path):
    return {
        "moderation": {
            "enabled": "true",
            "blocklist_file": str(blocklist_path),
            "user_overrides_file": str(tmp_path / "moderation_user_overrides.json"),
            "runtime_overrides_file": str(tmp_path / "moderation_runtime_overrides.json"),
            "categories_enabled": "pii",
        }
    }


def test_update_settings_recompiles_global_policy_with_runtime_categories(monkeypatch, tmp_path):
    blocklist = tmp_path / "moderation_blocklist.txt"
    blocklist.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        moderation_service_module,
        "load_and_log_configs",
        lambda: _tmp_moderation_config(tmp_path, blocklist),
    )

    svc = ModerationService()
    svc._runtime_override = {}

    result = svc.update_settings(categories_enabled=["confidential"], persist=False)

    assert result["categories_enabled"] == ["confidential"]
    assert svc._global_policy.categories_enabled == {"confidential"}


def test_set_blocklist_lines_recompiles_policy_from_file(monkeypatch, tmp_path):
    blocklist = tmp_path / "moderation_blocklist.txt"
    monkeypatch.setattr(
        moderation_service_module,
        "load_and_log_configs",
        lambda: _tmp_moderation_config(tmp_path, blocklist),
    )

    svc = ModerationService()

    assert svc.set_blocklist_lines(["secret -> block #confidential"]) is True

    policy = svc.get_effective_policy(None)
    assert any(rule.regex.search("secret") for rule in policy.block_patterns)
```

Add `import tldw_Server_API.app.core.Moderation.moderation_service as moderation_service_module` to `test_moderation_effective_settings.py` if it is not already present. `test_moderation_blocklist_parse.py` already has this import. Keep `persist=False` in the settings test so it does not write runtime override files unless the test is explicitly asserting persistence.

- [ ] **Step 2: Add supervised overlay regression if not already covered**

Add or keep this coverage in `test_supervised_policy.py`:

```python
def test_supervised_overlay_still_accepts_compiler_policy(db):
    base_policy = ModerationPolicy(
        enabled=True,
        input_enabled=True,
        output_enabled=True,
        input_action="block",
        output_action="redact",
        block_patterns=[],
        categories_enabled={"pii"},
    )
    engine = SupervisedPolicyEngine(db)

    merged = engine.overlay_policy("child1", base_policy)

    assert isinstance(merged, ModerationPolicy)
    assert merged.input_enabled is base_policy.input_enabled
    assert merged.output_enabled is base_policy.output_enabled
```

If existing fixtures require policy creation before overlaying, use the established fixture pattern in the same file and assert the returned type and base settings are preserved.

- [ ] **Step 3: Run behavior regression tests**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest tldw_Server_API/tests/unit/test_moderation_blocklist_parse.py tldw_Server_API/tests/unit/test_moderation_effective_settings.py tldw_Server_API/tests/Guardian/test_supervised_policy.py -q
```

Expected: PASS.

- [ ] **Step 4: Fix integration drift exposed by tests**

If reload/settings/blocklist tests fail because `_load_global_policy()` no longer sees persisted paths or runtime overrides, update service wiring so:

```python
def reload(self) -> None:
    with self._lock:
        self._config = load_and_log_configs() or {}
        self._global_policy = self._load_global_policy()
        try:
            self._load_runtime_overrides_file()
            self._global_policy = self._load_global_policy()
        except _MODERATION_NONCRITICAL_EXCEPTIONS:
            pass
        self._user_overrides = self._load_user_overrides()
```

The method shape should remain behavior-compatible. Do not move file reads into `PolicyCompiler`.

- [ ] **Step 5: Re-run behavior regression tests**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest tldw_Server_API/tests/unit/test_moderation_blocklist_parse.py tldw_Server_API/tests/unit/test_moderation_effective_settings.py tldw_Server_API/tests/Guardian/test_supervised_policy.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit Task 6**

Run:

```bash
git add tldw_Server_API/app/core/Moderation/moderation_service.py tldw_Server_API/tests/unit/test_moderation_blocklist_parse.py tldw_Server_API/tests/unit/test_moderation_effective_settings.py tldw_Server_API/tests/Guardian/test_supervised_policy.py
git commit -m "Preserve moderation service behavior with compiler integration"
```

---

### Task 7: Final Verification And Cleanup

**Files:**
- Modify: `backlog/tasks/task-2432 - Implement-Moderation-PolicyCompiler-refactor.md`
- Modify: `Docs/superpowers/plans/2026-06-24-moderation-policy-compiler-refactor-implementation-plan.md` only if execution notes reveal plan gaps.

- [ ] **Step 1: Run compile checks**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m py_compile \
  tldw_Server_API/app/core/Moderation/policy_compiler.py \
  tldw_Server_API/app/core/Moderation/moderation_service.py \
  tldw_Server_API/app/core/Moderation/supervised_policy.py \
  tldw_Server_API/tests/unit/test_moderation_policy_compiler.py \
  tldw_Server_API/tests/unit/test_moderation_blocklist_parse.py \
  tldw_Server_API/tests/unit/test_moderation_effective_settings.py \
  tldw_Server_API/tests/Guardian/test_supervised_policy.py
```

Expected: exit code 0.

- [ ] **Step 2: Run targeted pytest suite**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/unit/test_moderation_policy_compiler.py \
  tldw_Server_API/tests/unit/test_moderation_blocklist_parse.py \
  tldw_Server_API/tests/unit/test_moderation_effective_settings.py \
  tldw_Server_API/tests/unit/test_moderation_check_text_snippet.py \
  tldw_Server_API/tests/unit/test_moderation_redact_categories.py \
  tldw_Server_API/tests/Guardian/test_supervised_policy.py \
  -q
```

Expected: PASS.

- [ ] **Step 3: Run whitespace diff check**

Run:

```bash
git diff --check
```

Expected: no output and exit code 0.

- [ ] **Step 4: Run Bandit on touched Moderation code**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m bandit -r tldw_Server_API/app/core/Moderation -f json -o /tmp/bandit_moderation_policy_compiler.json
```

Expected: exit code 0 and zero new findings in touched Moderation code.

- [ ] **Step 5: Inspect git diff for scope**

Run:

```bash
git status --short
git diff --stat
```

Expected: only planned Moderation code/tests, plan docs, and Backlog task files are changed.

- [ ] **Step 6: Update Backlog implementation task**

Record:

- exact pytest result summary
- py_compile result
- Bandit result and output path
- `git diff --check` result
- any known skips or blockers

- [ ] **Step 7: Commit final verification notes**

Run:

```bash
git add 'backlog/tasks/task-2432 - Implement-Moderation-PolicyCompiler-refactor.md'
git commit -m "Record moderation policy compiler verification"
```

If no task file changed after verification, skip this commit and record the evidence in the final response.

---

## Self-Review Checklist

- [x] Spec coverage: every design requirement maps to one or more tasks.
- [x] Placeholder scan: plan contains no unfinished markers.
- [x] Type consistency: names match across tasks: `ResolvedModerationConfig`, `PolicyCompilationInput`, `PolicyCompilationIssue`, `PolicyCompilationReport`, `PolicyCompilationResult`, `PolicyCompiler`.
- [x] Scope check: no `PolicyEvaluator` implementation appears in this plan.
- [x] Boundary check: no file I/O is assigned to `PolicyCompiler`.
- [x] Compatibility check: public service methods and existing policy types remain intact.
