from __future__ import annotations

import ast
import asyncio
import dataclasses
import inspect
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SELECTORS_ROOT = REPO_ROOT / "tldw_Server_API" / "app" / "core" / "Web_Scraping" / "selectors"
FETCHERS_PATH = REPO_ROOT / "tldw_Server_API" / "app" / "core" / "Watchlists" / "fetchers.py"
ENDPOINT_PATH = REPO_ROOT / "tldw_Server_API" / "app" / "api" / "v1" / "endpoints" / "watchlists.py"
ARTICLE_PATH = REPO_ROOT / "tldw_Server_API" / "app" / "core" / "Web_Scraping" / "Article_Extractor_Lib.py"
HANDLERS_PATH = REPO_ROOT / "tldw_Server_API" / "app" / "core" / "Web_Scraping" / "handlers.py"
ENHANCED_SCRAPER_PATH = REPO_ROOT / "tldw_Server_API" / "app" / "core" / "Web_Scraping" / "enhanced_web_scraping.py"
EXTRACTION_ROOT = REPO_ROOT / "tldw_Server_API" / "app" / "core" / "Web_Scraping" / "extraction"
EXTRACTION_DEPENDENCIES_PATH = (
    REPO_ROOT / "tldw_Server_API" / "app" / "core" / "Web_Scraping" / "extraction" / "dependencies.py"
)
METRICS_PATH = EXTRACTION_ROOT / "metrics.py"

EXPECTED_IMPORTS = {
    "__init__.py": {".caches", ".schema"},
    "caches.py": {"__future__", "collections", "os", "threading", "typing"},
    "engine.py": {
        ".caches",
        "__future__",
        "collections.abc",
        "cssselect",
        "functools",
        "loguru",
        "lxml.cssselect",
        "lxml.etree",
        "lxml.html",
        "os",
        "re",
        "typing",
    },
    "schema.py": {
        "..safe_regex",
        ".engine",
        "__future__",
        "codecs",
        "collections.abc",
        "contextlib",
        "contextvars",
        "dataclasses",
        "dateutil",
        "datetime",
        "email.utils",
        "lxml",
        "lxml.html",
        "re",
        "string",
        "typing",
        "urllib.parse",
    },
}

BANNED_UPWARD_IMPORT_PARTS = {
    "Article_Extractor_Lib",
    "Enhanced_Web_Scraping",
    "Watchlists",
    "WebSearch",
    "enhanced_web_scraping",
    "extraction",
    "handlers",
    "orchestration",
    "policy",
    "preflight",
    "routing",
}

_BUILTIN_BASE_EXCEPTION = "builtins.BaseException"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imports(path: Path) -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add("." * node.level + (node.module or ""))
    return imported


def _defined_functions(path: Path) -> set[str]:
    return {node.name for node in _tree(path).body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _imported_names(path: Path, module: str) -> set[str]:
    names: set[str] = set()
    for node in _tree(path).body:
        if isinstance(node, ast.ImportFrom) and node.module == module:
            names.update(alias.name for alias in node.names)
    return names


def _imported_names_at_any_scope(path: Path, module: str) -> set[str]:
    return {
        alias.name
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.ImportFrom) and node.module == module
        for alias in node.names
    }


def _function(path: Path, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    return next(
        node
        for node in _tree(path).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )


def _called_names(node: ast.AST) -> set[str]:
    return {
        child.func.id for child in ast.walk(node) if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
    }


@dataclasses.dataclass(frozen=True)
class _CancellationBinding:
    name: str
    order: int
    owner: _CancellationScope
    kind: str
    value: ast.expr | None = None
    imported_name: str | None = None
    operation: ast.operator | None = None


class _CancellationScope:
    def __init__(self, parent: _CancellationScope | None = None, *, kind: str) -> None:
        self.parent = parent
        self.kind = kind
        self.bindings: dict[str, list[_CancellationBinding]] = {}
        self.local_names: set[str] = set()

    def bind(
        self,
        name: str,
        order: int,
        *,
        kind: str,
        value: ast.expr | None = None,
        imported_name: str | None = None,
        operation: ast.operator | None = None,
    ) -> None:
        self.bindings.setdefault(name, []).append(
            _CancellationBinding(
                name=name,
                order=order,
                owner=self,
                kind=kind,
                value=value,
                imported_name=imported_name,
                operation=operation,
            )
        )

    def resolve_name(self, name: str, *, before_order: int | None) -> _CancellationBinding | None:
        for binding in reversed(self.bindings.get(name, [])):
            if before_order is None or binding.order < before_order:
                return binding
        if self.kind == "function" and name in self.local_names:
            return None
        if self.parent is None:
            return None
        # Function free variables resolve from live closure cells or globals when called.
        runtime_parent = self.kind == "function" and self.parent.kind in {"function", "module"}
        parent_order = None if runtime_parent else before_order
        return self.parent.resolve_name(name, before_order=parent_order)


@dataclasses.dataclass(frozen=True)
class _CancellationContext:
    scope: _CancellationScope
    order: int


class _FunctionLocalCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)

    def visit_Import(self, node: ast.Import) -> None:
        self.names.update(imported.asname or imported.name.split(".")[0] for imported in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.names.update(imported.asname or imported.name for imported in node.names if imported.name != "*")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_ListComp(self, node: ast.ListComp) -> None:
        return

    visit_SetComp = visit_ListComp
    visit_DictComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.names.add(node.name)
        self.generic_visit(node)


def _parameter_names(arguments: ast.arguments) -> list[str]:
    parameters = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
    names = [parameter.arg for parameter in parameters]
    if arguments.vararg:
        names.append(arguments.vararg.arg)
    if arguments.kwarg:
        names.append(arguments.kwarg.arg)
    return names


class _CancellationScopeBuilder(ast.NodeVisitor):
    def __init__(self) -> None:
        self.order = 0
        self.scope = _CancellationScope(kind="module")
        self.scope.bind(
            "BaseException",
            self.order,
            kind="import",
            imported_name=_BUILTIN_BASE_EXCEPTION,
        )
        self.contexts: dict[ast.ExceptHandler, _CancellationContext] = {}

    def next_order(self) -> int:
        self.order += 1
        return self.order

    def visit_Import(self, node: ast.Import) -> None:
        for imported in node.names:
            bound_name = imported.asname or imported.name.split(".")[0]
            imported_name = imported.name if imported.asname else imported.name.split(".")[0]
            self.scope.bind(bound_name, self.next_order(), kind="import", imported_name=imported_name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ""
        for imported in node.names:
            if imported.name == "*":
                continue
            imported_name = f"{module}.{imported.name}" if module else imported.name
            self.scope.bind(
                imported.asname or imported.name,
                self.next_order(),
                kind="import",
                imported_name=imported_name,
            )

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.scope.bind(target.id, self.next_order(), kind="assignment", value=node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name) and node.value is not None:
            self.scope.bind(node.target.id, self.next_order(), kind="assignment", value=node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            self.scope.bind(
                node.target.id,
                self.next_order(),
                kind="augmented_assignment",
                value=node.value,
                operation=node.op,
            )

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.scope.bind(node.name, self.next_order(), kind="shadow")
        lexical_parent = self.scope
        while lexical_parent.kind == "class" and lexical_parent.parent is not None:
            lexical_parent = lexical_parent.parent
        function_scope = _CancellationScope(lexical_parent, kind="function")
        collector = _FunctionLocalCollector()
        for statement in node.body:
            collector.visit(statement)
        parameter_names = _parameter_names(node.args)
        function_scope.local_names.update(collector.names | set(parameter_names))

        previous_scope = self.scope
        self.scope = function_scope
        for name in parameter_names:
            self.scope.bind(name, self.next_order(), kind="shadow")
        for statement in node.body:
            self.visit(statement)
        self.scope = previous_scope

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.bind(node.name, self.next_order(), kind="shadow")
        class_scope = _CancellationScope(self.scope, kind="class")
        previous_scope = self.scope
        self.scope = class_scope
        for statement in node.body:
            self.visit(statement)
        self.scope = previous_scope

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        context_order = self.next_order()
        self.contexts[node] = _CancellationContext(self.scope, context_order)
        if node.name:
            self.scope.bind(node.name, self.next_order(), kind="shadow")
        for statement in node.body:
            self.visit(statement)


def _scope_for_nodes(tree: ast.Module) -> dict[ast.ExceptHandler, _CancellationContext]:
    builder = _CancellationScopeBuilder()
    builder.visit(tree)
    return builder.contexts


def _dotted_name(
    expression: ast.expr,
    scope: _CancellationScope,
    before_order: int,
    seen: set[int],
) -> str | None:
    if isinstance(expression, ast.Name):
        binding = scope.resolve_name(expression.id, before_order=before_order)
        if binding is None or id(binding) in seen:
            return None
        if binding.imported_name is not None:
            return binding.imported_name
        if binding.kind == "assignment" and binding.value is not None:
            return _dotted_name(binding.value, binding.owner, binding.order, seen | {id(binding)})
        return None
    if not isinstance(expression, ast.Attribute):
        return None
    parent = _dotted_name(expression.value, scope, before_order, seen)
    return f"{parent}.{expression.attr}" if parent else None


def _resolve_exception_expression(
    expression: ast.expr,
    scope: _CancellationScope,
    before_order: int,
    seen: set[int],
) -> tuple[bool, bool, bool]:
    """Return cancellation, tuple-shape, and built-in BaseException resolution."""
    dotted = _dotted_name(expression, scope, before_order, seen)
    if dotted in {"asyncio.CancelledError", "asyncio.exceptions.CancelledError"}:
        return True, False, False
    if dotted == _BUILTIN_BASE_EXCEPTION:
        return True, False, True
    if isinstance(expression, ast.Name):
        binding = scope.resolve_name(expression.id, before_order=before_order)
        if binding is not None and id(binding) not in seen:
            binding_seen = seen | {id(binding)}
            if binding.kind == "assignment" and binding.value is not None:
                return _resolve_exception_expression(binding.value, binding.owner, binding.order, binding_seen)
            if (
                binding.kind == "augmented_assignment"
                and isinstance(binding.operation, ast.Add)
                and binding.value is not None
            ):
                previous = binding.owner.resolve_name(binding.name, before_order=binding.order)
                previous_resolution = (
                    _resolve_exception_expression(
                        ast.Name(id=binding.name, ctx=ast.Load()),
                        binding.owner,
                        binding.order,
                        binding_seen,
                    )
                    if previous is not None
                    else (False, False, False)
                )
                added_resolution = _resolve_exception_expression(
                    binding.value,
                    binding.owner,
                    binding.order,
                    binding_seen,
                )
                return (
                    previous_resolution[0] or added_resolution[0],
                    True,
                    previous_resolution[2] or added_resolution[2],
                )
    if isinstance(expression, ast.Starred):
        contains_cancelled, _, contains_base_exception = _resolve_exception_expression(
            expression.value,
            scope,
            before_order,
            seen,
        )
        return contains_cancelled, True, contains_base_exception
    if isinstance(expression, ast.Tuple):
        resolutions = [_resolve_exception_expression(item, scope, before_order, seen) for item in expression.elts]
        return (
            any(resolution[0] for resolution in resolutions),
            True,
            any(resolution[2] for resolution in resolutions),
        )
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
        left_resolution = _resolve_exception_expression(expression.left, scope, before_order, seen)
        right_resolution = _resolve_exception_expression(expression.right, scope, before_order, seen)
        return (
            left_resolution[0] or right_resolution[0],
            True,
            left_resolution[2] or right_resolution[2],
        )
    return False, False, False


def _is_unconditional_bare_reraise(handler: ast.ExceptHandler) -> bool:
    return (
        len(handler.body) == 1
        and isinstance(handler.body[0], ast.Raise)
        and handler.body[0].exc is None
        and handler.body[0].cause is None
    )


def _import_binding_name(alias: ast.alias) -> str:
    return alias.asname or alias.name.split(".")[0]


def _direct_module_reference(expression: ast.expr, module_names: set[str]) -> bool:
    return isinstance(expression, ast.Name) and expression.id in module_names


def _direct_cancellation_reference(
    expression: ast.expr,
    module_names: set[str],
    exception_names: set[str],
) -> bool:
    if isinstance(expression, ast.Name):
        return expression.id in exception_names
    attributes: list[str] = []
    current = expression
    while isinstance(current, ast.Attribute):
        attributes.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name) or current.id not in module_names:
        return False
    return tuple(reversed(attributes)) in {("CancelledError",), ("exceptions", "CancelledError")}


def _protected_cancellation_names(tree: ast.Module) -> tuple[set[str], set[str]]:
    module_names: set[str] = set()
    exception_names = {"BaseException"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            module_names.update(
                _import_binding_name(alias) for alias in node.names if alias.name in {"asyncio", "asyncio.exceptions"}
            )
        elif isinstance(node, ast.ImportFrom) and node.module in {"asyncio", "asyncio.exceptions"}:
            for alias in node.names:
                bound_name = alias.asname or alias.name
                if alias.name == "CancelledError":
                    exception_names.add(bound_name)
                elif node.module == "asyncio" and alias.name == "exceptions":
                    module_names.add(bound_name)

    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            value: ast.expr | None = None
            target: ast.expr | None = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target, value = node.targets[0], node.value
            elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
                target, value = node.target, node.value
            if not isinstance(target, ast.Name) or value is None or target.id == "BaseException":
                continue
            if _direct_module_reference(value, module_names) and target.id not in module_names:
                module_names.add(target.id)
                changed = True
            if (
                _direct_cancellation_reference(value, module_names, exception_names)
                and target.id not in exception_names
            ):
                exception_names.add(target.id)
                changed = True
    return module_names, exception_names


def _cancellation_shadow_violations(tree: ast.Module) -> list[str]:
    """Reject competing bindings for cancellation roots and direct aliases."""
    module_names, exception_names = _protected_cancellation_names(tree)
    protected_names = module_names | exception_names
    safe_targets: set[int] = set()
    shadows: set[str] = set()

    for node in ast.walk(tree):
        value: ast.expr | None = None
        target: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):
            target, value = node.target, node.value
        if isinstance(target, ast.Name) and value is not None and target.id != "BaseException":
            if target.id in module_names and _direct_module_reference(value, module_names):
                safe_targets.add(id(target))
            if target.id in exception_names and _direct_cancellation_reference(
                value,
                module_names,
                exception_names,
            ):
                safe_targets.add(id(target))

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and node.id in protected_names
            and not isinstance(node.ctx, ast.Load)
            and id(node) not in safe_targets
        ):
            shadows.add(node.id)
        elif isinstance(node, ast.arg) and node.arg in protected_names:
            shadows.add(node.arg)
        elif isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and node.name in protected_names:
            shadows.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest in protected_names:
            shadows.add(node.rest)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in protected_names:
            shadows.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bound_name = _import_binding_name(alias)
                if bound_name in protected_names and (
                    alias.name not in {"asyncio", "asyncio.exceptions"} or bound_name == "BaseException"
                ):
                    shadows.add(bound_name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound_name = alias.asname or alias.name
                is_cancellation_import = node.module in {"asyncio", "asyncio.exceptions"} and (
                    alias.name == "CancelledError" or node.module == "asyncio" and alias.name == "exceptions"
                )
                if bound_name in protected_names and (not is_cancellation_import or bound_name == "BaseException"):
                    shadows.add(bound_name)
    return [f"shadow {name}" for name in sorted(shadows)]


def _recoverable_cancelled_error_violations(tree: ast.Module) -> list[str]:
    shadow_violations = _cancellation_shadow_violations(tree)
    if shadow_violations:
        return shadow_violations
    contexts = _scope_for_nodes(tree)
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        if node.type is None:
            if not _is_unconditional_bare_reraise(node):
                violations.append("bare except")
            continue
        context = contexts[node]
        contains_cancelled, is_tuple, contains_base_exception = _resolve_exception_expression(
            node.type,
            context.scope,
            context.order,
            set(),
        )
        if contains_cancelled and (
            (is_tuple and not contains_base_exception) or not _is_unconditional_bare_reraise(node)
        ):
            violations.append(ast.unparse(node.type))
    return violations


_RAW_METRIC_SINK_NAMES = {"increment_counter", "log_counter", "observe_histogram"}
_RAW_METRICS_MODULE_PREFIX = "tldw_Server_API.app.core.Metrics"


def _metric_boundary_bypasses(paths: list[Path]) -> list[str]:
    """Find extraction modules that emit metrics without the canonical boundary."""
    violations: list[str] = []
    for path in paths:
        if path == METRICS_PATH:
            continue
        tree = _tree(path)
        raw_aliases: set[str] = set()
        callable_aliases: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(_RAW_METRICS_MODULE_PREFIX):
                raw_aliases.update(alias.asname or alias.name for alias in node.names)
                violations.append(f"{path.name}: raw metrics import")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(_RAW_METRICS_MODULE_PREFIX):
                        raw_aliases.add(alias.asname or alias.name.split(".")[0])
                        violations.append(f"{path.name}: raw metrics import")
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if isinstance(value, ast.Attribute) and value.attr in _RAW_METRIC_SINK_NAMES:
                    callable_aliases.update(target.id for target in targets if isinstance(target, ast.Name))
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr in _RAW_METRIC_SINK_NAMES:
                    violations.append(f"{path.name}: direct {node.func.attr} call")
                elif isinstance(node.func, ast.Name) and node.func.id in raw_aliases | callable_aliases:
                    violations.append(f"{path.name}: aliased metric sink call")
    return violations


def test_selector_package_has_the_approved_files() -> None:
    assert {path.name for path in SELECTORS_ROOT.glob("*.py")} == set(EXPECTED_IMPORTS)


def test_selector_files_have_an_explicit_dependency_inventory() -> None:
    actual = {path.name: _imports(path) for path in sorted(SELECTORS_ROOT.glob("*.py"))}

    assert actual == EXPECTED_IMPORTS


def test_selector_package_has_no_upward_application_dependencies() -> None:
    violations: list[str] = []
    for path in sorted(SELECTORS_ROOT.glob("*.py")):
        for imported in sorted(_imports(path)):
            if any(part in imported.split(".") for part in BANNED_UPWARD_IMPORT_PARTS):
                violations.append(f"{path.name}: {imported}")

    assert violations == []


def test_cache_state_has_one_canonical_owner() -> None:
    cache_tokens = {
        "_CSS_SELECTOR_CACHE",
        "_SELECTOR_CACHE_LOCK",
        "_XPATH_SELECTOR_CACHE",
    }
    owners: dict[str, list[str]] = {token: [] for token in cache_tokens}
    inspected = [*SELECTORS_ROOT.glob("*.py"), FETCHERS_PATH, ARTICLE_PATH]
    for path in inspected:
        text = path.read_text(encoding="utf-8")
        for token in cache_tokens:
            if token in text:
                owners[token].append(path.name)

    assert owners == {token: ["caches.py"] for token in cache_tokens}


def test_watchlists_has_direct_canonical_imports_without_selector_bodies() -> None:
    forbidden_definitions = {
        "clear_selector_caches",
        "extract_schema_fields",
        "get_selector_cache_stats",
        "reload_selector_guardrails_from_env",
        "validate_selector_rules",
    }
    assert _defined_functions(FETCHERS_PATH).isdisjoint(forbidden_definitions)

    facade_module = "tldw_Server_API.app.core.Web_Scraping.selectors"
    engine_module = f"{facade_module}.engine"
    schema_module = f"{facade_module}.schema"
    assert _imported_names(FETCHERS_PATH, facade_module) == {
        "clear_selector_caches",
        "extract_schema_fields",
        "get_selector_cache_stats",
        "validate_selector_rules",
    }
    assert _imported_names(FETCHERS_PATH, engine_module) == {
        "coerce_value",
        "ensure_sequence",
        "extract_value",
        "reload_selector_guardrails_from_env",
        "select_nodes",
    }
    assert _imported_names(FETCHERS_PATH, schema_module) == {"normalize_datetime"}

    private_imports = {
        alias.name
        for node in ast.walk(_tree(FETCHERS_PATH))
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(facade_module)
        for alias in node.names
        if alias.name.startswith("_")
    }
    assert private_imports == set()


def test_article_selector_responsibilities_import_the_canonical_facade() -> None:
    facade_module = "tldw_Server_API.app.core.Web_Scraping.selectors"
    article_selector_names = {
        "clear_selector_caches",
        "get_selector_cache_stats",
    }
    extraction_selector_names = {
        "extract_schema_fields",
        "validate_selector_rules",
    }

    assert _imported_names(ARTICLE_PATH, facade_module) == article_selector_names
    assert _imported_names_at_any_scope(EXTRACTION_DEPENDENCIES_PATH, facade_module) == extraction_selector_names

    violations = []
    for path in (ARTICLE_PATH, EXTRACTION_DEPENDENCIES_PATH):
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = node.module or ""
            imported = {alias.name for alias in node.names}
            if ".Watchlists" in module and imported & (article_selector_names | extraction_selector_names):
                violations.append({"module": module, "names": sorted(imported)})
    assert violations == []


def test_phase4b_moved_consumers_import_canonical_content_and_extraction_facades() -> None:
    content_module = "tldw_Server_API.app.core.Web_Scraping.content"
    extraction_module = "tldw_Server_API.app.core.Web_Scraping.extraction"
    legacy_module = "tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib"

    assert _imported_names_at_any_scope(HANDLERS_PATH, content_module) == {"convert_html_to_markdown"}
    assert _imported_names_at_any_scope(HANDLERS_PATH, extraction_module) == {"extract_article_data_from_html"}
    assert _imported_names_at_any_scope(ENHANCED_SCRAPER_PATH, content_module) == {"convert_html_to_markdown"}
    assert _imported_names_at_any_scope(ENHANCED_SCRAPER_PATH, extraction_module) == {"extract_article_with_pipeline"}
    assert _imported_names_at_any_scope(HANDLERS_PATH, legacy_module) == set()
    generic_handler = _function(HANDLERS_PATH, "handle_generic_html")
    handler_imports = {
        node.module: {alias.name for alias in node.names}
        for node in generic_handler.body
        if isinstance(node, ast.ImportFrom) and node.module in {content_module, extraction_module}
    }
    assert handler_imports == {
        content_module: {"convert_html_to_markdown"},
        extraction_module: {"extract_article_data_from_html"},
    }
    canonical_handler_import_locations = {
        (node.lineno, node.module, tuple(alias.name for alias in node.names))
        for node in ast.walk(_tree(HANDLERS_PATH))
        if isinstance(node, ast.ImportFrom) and node.module in {content_module, extraction_module}
    }
    assert len(canonical_handler_import_locations) == 2
    assert canonical_handler_import_locations == {
        (node.lineno, node.module, tuple(alias.name for alias in node.names))
        for node in generic_handler.body
        if isinstance(node, ast.ImportFrom) and node.module in {content_module, extraction_module}
    }
    enhanced_tree = _tree(ENHANCED_SCRAPER_PATH)
    legacy_imports = [
        node for node in ast.walk(enhanced_tree) if isinstance(node, ast.ImportFrom) and node.module == legacy_module
    ]
    assert len(legacy_imports) == 1
    assert [alias.name for alias in legacy_imports[0].names] == ["scrape_article"]
    manager = next(
        node for node in enhanced_tree.body if isinstance(node, ast.ClassDef) and node.name == "ScrapingJobQueue"
    )
    execute_job = next(
        node for node in manager.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "_execute_job"
    )
    parent_scraper_branch = next(
        node
        for node in execute_job.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and isinstance(node.test.value, ast.Name)
        and node.test.value.id == "self"
        and node.test.attr == "parent_scraper"
    )
    fallback_imports = [node for node in parent_scraper_branch.orelse if isinstance(node, ast.ImportFrom)]
    assert fallback_imports == legacy_imports
    fallback_calls = [
        statement.value
        for statement in parent_scraper_branch.orelse
        if isinstance(statement, ast.Return)
        and isinstance(statement.value, ast.Await)
        and isinstance(statement.value.value, ast.Call)
        and isinstance(statement.value.value.func, ast.Name)
        and statement.value.value.func.id == "scrape_article"
    ]
    assert len(fallback_calls) == 1
    fallback_call = fallback_calls[0].value
    assert len(fallback_call.args) == 1
    assert isinstance(fallback_call.args[0], ast.Attribute)
    assert isinstance(fallback_call.args[0].value, ast.Name)
    assert fallback_call.args[0].value.id == "job"
    assert fallback_call.args[0].attr == "url"
    assert [keyword.arg for keyword in fallback_call.keywords] == ["custom_cookies", "allow_llm_extraction"]
    expected_metadata_gets = {
        "custom_cookies": ("custom_cookies", None),
        "allow_llm_extraction": ("allow_llm_extraction", True),
    }
    for keyword in fallback_call.keywords:
        assert keyword.arg is not None
        assert isinstance(keyword.value, ast.Call)
        assert isinstance(keyword.value.func, ast.Attribute)
        assert keyword.value.func.attr == "get"
        assert isinstance(keyword.value.func.value, ast.Attribute)
        assert keyword.value.func.value.attr == "metadata"
        assert isinstance(keyword.value.func.value.value, ast.Name)
        assert keyword.value.func.value.value.id == "job"
        expected_key, expected_default = expected_metadata_gets[keyword.arg]
        assert len(keyword.value.args) == 1 + (expected_default is not None)
        assert isinstance(keyword.value.args[0], ast.Constant)
        assert keyword.value.args[0].value == expected_key
        if expected_default is not None:
            assert isinstance(keyword.value.args[1], ast.Constant)
            assert keyword.value.args[1].value is expected_default


def test_phase4b_enhanced_no_parent_fallback_forwards_legacy_job_arguments(monkeypatch) -> None:
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as article
    from tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping import ScrapingJob, ScrapingJobQueue

    calls: list[tuple[str, dict[str, Any]]] = []

    async def fallback(url: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((url, kwargs))
        return {"url": url, "extraction_successful": True}

    monkeypatch.setattr(article, "scrape_article", fallback)
    job = ScrapingJob(
        job_id="fallback-job",
        url="https://example.com/fallback",
        method="auto",
        metadata={"custom_cookies": {"session": "value"}, "allow_llm_extraction": False},
    )

    result = asyncio.run(ScrapingJobQueue(parent_scraper=None)._execute_job(job))

    assert calls == [
        (
            "https://example.com/fallback",
            {"custom_cookies": {"session": "value"}, "allow_llm_extraction": False},
        )
    ]
    assert result == {"url": "https://example.com/fallback", "extraction_successful": True}


def test_phase4b_crawl_bound_article_helper_keeps_its_async_surface_and_canonical_dependency() -> None:
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as article

    extraction_module = "tldw_Server_API.app.core.Web_Scraping.extraction"
    helper = _function(ARTICLE_PATH, "scrape_article_async")

    assert isinstance(helper, ast.AsyncFunctionDef)
    assert inspect.signature(article.scrape_article_async) == inspect.Signature(
        parameters=[
            inspect.Parameter("context", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("url", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
            inspect.Parameter(
                "allow_llm_extraction",
                inspect.Parameter.KEYWORD_ONLY,
                default=True,
                annotation=bool,
            ),
        ],
        return_annotation=dict[str, Any],
    )
    assert inspect.get_annotations(article.scrape_article_async, eval_str=True) == {
        "url": str,
        "allow_llm_extraction": bool,
        "return": dict[str, Any],
    }
    assert "extract_article_with_pipeline" in _imported_names(ARTICLE_PATH, extraction_module)
    assert "extract_article_with_pipeline" in _called_names(helper)


def test_phase4b_crawl_bound_article_helper_forwards_to_canonical_extraction_and_closes_page(
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as article

    class Page:
        closed = False

        async def goto(self, url: str) -> None:
            assert url == "https://example.com/article"

        async def wait_for_load_state(self, state: str) -> None:
            assert state == "networkidle"

        async def title(self) -> str:
            return "Browser title"

        async def content(self) -> str:
            return "<article>Body</article>"

        async def close(self) -> None:
            self.closed = True

    class Context:
        def __init__(self) -> None:
            self.page = Page()

        async def new_page(self) -> Page:
            return self.page

    calls: list[tuple[str, str, dict[str, bool]]] = []

    def extract(html: str, url: str, **kwargs: bool) -> dict[str, Any]:
        calls.append((html, url, kwargs))
        return {"title": "N/A", "content": "Body", "extraction_successful": True}

    monkeypatch.setattr(article, "extract_article_with_pipeline", extract)
    monkeypatch.setattr(article, "convert_html_to_markdown", lambda content: f"markdown:{content}")
    context = Context()

    result = asyncio.run(
        article.scrape_article_async(context, "https://example.com/article", allow_llm_extraction=False)
    )

    assert calls == [("<article>Body</article>", "https://example.com/article", {"allow_llm_extraction": False})]
    assert result == {"title": "Browser title", "content": "markdown:Body", "extraction_successful": True}
    assert context.page.closed is True


def test_phase4b_crawl_bound_article_helper_returns_legacy_failure_shape_and_closes_page(monkeypatch) -> None:
    from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as article

    class Page:
        closed = False

        async def goto(self, _url: str) -> None:
            raise RuntimeError("legacy failure")

        async def close(self) -> None:
            self.closed = True

    class Context:
        def __init__(self) -> None:
            self.page = Page()

        async def new_page(self) -> Page:
            return self.page

    context = Context()
    result = asyncio.run(article.scrape_article_async(context, "https://example.com/article"))

    assert result == {
        "url": "https://example.com/article",
        "extraction_successful": False,
        "error": "legacy failure",
    }
    assert context.page.closed is True


def test_phase4b_handler_import_is_lazy_and_cycle_free() -> None:
    script = """
import json
import sys

import tldw_Server_API.app.core.Web_Scraping.handlers

blocked = (
    "tldw_Server_API.app.core.Web_Scraping.Article_Extractor_Lib",
    "tldw_Server_API.app.core.Web_Scraping.enhanced_web_scraping",
    "tldw_Server_API.app.core.Web_Scraping.extraction",
    "tldw_Server_API.app.core.Metrics",
    "tldw_Server_API.app.core.Metrics.metrics_logger",
)
print("HANDLER_IMPORTS=" + json.dumps([name for name in blocked if name in sys.modules]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    marker = next(line for line in result.stdout.splitlines() if line.startswith("HANDLER_IMPORTS="))

    assert json.loads(marker.removeprefix("HANDLER_IMPORTS=")) == []


def test_phase4b_metric_contract_rejects_sensitive_or_high_cardinality_values() -> None:
    from tldw_Server_API.app.core.Web_Scraping.extraction import metrics

    invalid_events = [
        ("extraction_strategy_total", {"strategy": "https://example.com", "status": "success"}),
        ("extraction_retry_total", {"strategy": "regex", "attempt": "secret-token"}),
        ("extraction_cluster_total", {"status": "RuntimeError: raw error"}),
        ("extraction_cluster_cache_total", {"cache": "embedding", "url": "https://example.com"}),
        ("llm_tokens_used_total", {"provider": "api-key-secret", "model": "payload-hash", "type": "prompt"}),
        (
            "llm_tokens_used_total_by_operation",
            {"provider": "openai", "model": "configured", "type": "prompt", "operation": "https://example.com"},
        ),
    ]
    for name, labels in invalid_events:
        with pytest.raises(ValueError):
            metrics.validate_metric(name, labels=labels)


def test_phase4b_llm_provider_labels_have_one_frozen_contract() -> None:
    from tldw_Server_API.app.core.Web_Scraping.extraction import metrics
    from tldw_Server_API.app.core.Web_Scraping.extraction.strategies import llm

    contract = metrics.METRIC_LABEL_CONTRACT
    provider_labels = metrics.LLM_PROVIDER_LABEL_VALUES

    assert contract["llm_tokens_used_total"]["provider"] is provider_labels
    assert contract["llm_tokens_used_total_by_operation"]["provider"] is provider_labels
    assert llm.LLM_PROVIDER_LABEL_VALUES is provider_labels
    assert {llm._metric_provider(provider) for provider in provider_labels} == provider_labels
    assert llm._metric_provider("future-provider") == "other"

    with pytest.raises(TypeError):
        contract["future_metric"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        contract["llm_tokens_used_total"]["provider"] = frozenset({"other"})  # type: ignore[index]


def test_phase4b_only_the_canonical_metric_boundary_owns_metric_sinks() -> None:
    assert METRICS_PATH.is_file()
    assert _metric_boundary_bypasses(sorted(EXTRACTION_ROOT.rglob("*.py"))) == []


def test_phase4b_metric_boundary_rejects_unlisted_emitters_and_bypasses(tmp_path: Path) -> None:
    cases = {
        "new_emitter.py": """
from tldw_Server_API.app.core.Metrics import increment_counter
increment_counter("future_metric", labels={"url": "https://example.com"})
""",
        "alias.py": """
def emit(dependencies):
    sink = dependencies.increment_counter
    sink("article_extracted", labels={"success": "true"})
""",
        "wrapper.py": """
def emit(dependencies):
    dependencies.observe_histogram(
        "extraction_strategy_duration_seconds", 1.0, labels={"strategy": "jsonld", "status": "success"}
    )
""",
    }
    for filename, source in cases.items():
        path = tmp_path / filename
        path.write_text(source, encoding="utf-8")
        assert _metric_boundary_bypasses([path])


def test_phase4b_metric_contract_covers_every_production_emission_and_allowed_value(monkeypatch) -> None:
    from tldw_Server_API.app.core.Web_Scraping.extraction import caches, metrics, pipeline
    from tldw_Server_API.app.core.Web_Scraping.extraction.dependencies import build_default_dependencies
    from tldw_Server_API.app.core.Web_Scraping.extraction.strategies import cluster, llm, trafilatura

    events: list[tuple[str, float | None, dict[str, str]]] = []

    def record(name: str, value: float | None = None, labels: dict[str, str] | None = None) -> None:
        events.append((name, value, dict(labels or {})))

    dependencies = dataclasses.replace(
        build_default_dependencies(),
        increment_counter=record,
        log_counter=record,
        observe_histogram=record,
    )

    contract = metrics.METRIC_LABEL_CONTRACT
    strategies = contract["extraction_strategy_total"]["strategy"]
    statuses = contract["extraction_strategy_total"]["status"]
    for strategy in strategies:
        for status in statuses:
            pipeline._trace_entry(dependencies, strategy, status, "test")
            pipeline._record_strategy_metrics(
                dependencies,
                strategy,
                status,
                0.0,
                {"content": "metric coverage"},
            )

    def always_fails() -> dict[str, Any]:
        raise RuntimeError("retry")

    monkeypatch.setattr(pipeline, "_extractor_retry_settings", lambda: (4, 0.0, 0.0))
    pipeline._run_with_retries(always_fails, strategy="unrecognized", dependencies=dependencies)
    monkeypatch.setattr(pipeline, "_extractor_retry_settings", lambda: (1, 0.0, 0.0))
    for strategy in strategies - {"unknown"}:
        pipeline._run_with_retries(always_fails, strategy=strategy, dependencies=dependencies)

    def failing_provider(**_kwargs: Any) -> None:
        raise RuntimeError("provider retry")

    retrying_dependencies = dataclasses.replace(dependencies, perform_chat_api_call=failing_provider)
    monkeypatch.setattr(llm, "_retry_settings", lambda: (4, 0.0, 0.0))
    llm.call_llm_provider(
        provider="openai",
        settings={},
        messages=[],
        app_config=None,
        dependencies=retrying_dependencies,
        stage="extraction",
        url="https://example.com",
    )

    caches.clear_extraction_caches()
    assert caches._cluster_cache_get("metric-cache", increment_counter=record) is None
    caches._cluster_cache_put("metric-cache", [1.0])
    assert caches._cluster_cache_get("metric-cache", increment_counter=record) == [1.0]
    for status in contract["extraction_cluster_total"]["status"]:
        cluster._increment_counter(dependencies, "extraction_cluster_total", labels={"status": status})

    monkeypatch.setattr(trafilatura.trafilatura, "extract_metadata", lambda _html: None)
    monkeypatch.setattr(trafilatura, "log_counter", record)
    monkeypatch.setattr(trafilatura.trafilatura, "extract", lambda _html, **_kwargs: "article body")
    trafilatura.extract_with_trafilatura("<html></html>", "https://example.com/success")
    monkeypatch.setattr(trafilatura.trafilatura, "extract", lambda _html, **_kwargs: None)
    trafilatura.extract_with_trafilatura("<html></html>", "https://example.com/failure")

    providers = contract["llm_tokens_used_total"]["provider"]
    for provider in providers:
        input_provider = provider if provider != "other" else "https://user:secret@example.com"
        llm.record_llm_usage_metrics(
            {"prompt_tokens": 1, "completion_tokens": 1},
            provider=input_provider,
            model="unbounded-model-payload",
            dependencies=dependencies,
        )

    assert {name for name, _value, _labels in events} == set(contract)
    for name, value, labels in events:
        metrics.validate_metric(name, value=value, labels=labels)
        if value is not None:
            assert math.isfinite(value)
    for metric_name, labels_contract in contract.items():
        metric_events = [labels for name, _value, labels in events if name == metric_name]
        for label_name, allowed_values in labels_contract.items():
            assert {labels[label_name] for labels in metric_events} == allowed_values


def test_phase4b_metric_boundary_rejects_uncontracted_and_nonfinite_values() -> None:
    from tldw_Server_API.app.core.Web_Scraping.extraction import metrics

    with pytest.raises(ValueError):
        metrics.validate_metric("future_uncontracted_metric", labels={})
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            metrics.validate_metric(
                "extraction_strategy_duration_seconds",
                value=value,
                labels={"strategy": "jsonld", "status": "success"},
            )


def test_phase4b_metric_boundary_blocks_an_unsafe_real_pipeline_branch() -> None:
    from tldw_Server_API.app.core.Web_Scraping.extraction import metrics, pipeline
    from tldw_Server_API.app.core.Web_Scraping.extraction.dependencies import build_default_dependencies

    events: list[tuple[str, float | None, dict[str, str]]] = []

    def record(name: str, value: float | None = None, labels: dict[str, str] | None = None) -> None:
        events.append((name, value, dict(labels or {})))

    dependencies = dataclasses.replace(build_default_dependencies(), log_counter=record)
    pipeline._trace_entry(dependencies, "jsonld", "https://example.com/raw-error", "test")

    assert events == []
    with pytest.raises(ValueError):
        metrics.validate_metric(
            "extraction_strategy_total",
            labels={"strategy": "jsonld", "status": "https://example.com/raw-error"},
        )


def test_phase4b_extraction_never_recovers_cancelled_error_in_exception_tuples() -> None:
    violations: list[str] = []
    for path in EXTRACTION_ROOT.rglob("*.py"):
        violations.extend(
            f"{path.relative_to(EXTRACTION_ROOT)}: {violation}"
            for violation in _recoverable_cancelled_error_violations(_tree(path))
        )

    assert violations == []


def test_phase4b_cancellation_guard_rejects_recoverable_handlers() -> None:
    cases = {
        "bare_except_bare_reraise": (
            """
try:
    pass
except:
    raise
""",
            [],
        ),
        "bare_except_swallow": (
            """
try:
    pass
except:
    pass
""",
            ["bare except"],
        ),
        "bare_except_cleanup_before_reraise": (
            """
try:
    pass
except:
    cleanup()
    raise
""",
            ["bare except"],
        ),
        "base_exception_bare_reraise": (
            """
try:
    pass
except BaseException:
    raise
""",
            [],
        ),
        "base_exception_swallow": (
            """
try:
    pass
except BaseException:
    pass
""",
            ["BaseException"],
        ),
        "base_exception_explicit_reraise": (
            """
try:
    pass
except BaseException as exc:
    raise exc
""",
            ["BaseException"],
        ),
        "base_exception_alias_tuple_bare_reraise": (
            """
ROOT_EXCEPTION = BaseException
_RECOVERABLE = (ValueError, ROOT_EXCEPTION)
try:
    pass
except _RECOVERABLE:
    raise
""",
            [],
        ),
        "base_exception_alias_tuple_swallow": (
            """
ROOT_EXCEPTION = BaseException
_RECOVERABLE = (ValueError, ROOT_EXCEPTION)
try:
    pass
except _RECOVERABLE:
    pass
""",
            ["_RECOVERABLE"],
        ),
        "base_exception_augmented_tuple_swallow": (
            """
_RECOVERABLE = (ValueError,)
_RECOVERABLE += (BaseException,)
try:
    pass
except _RECOVERABLE:
    pass
""",
            ["_RECOVERABLE"],
        ),
        "shadowed_base_exception": (
            """
BaseException = ValueError
try:
    pass
except BaseException:
    pass
""",
            ["shadow BaseException"],
        ),
        "shadowed_base_exception_alias_tuple": (
            """
BaseException = ValueError
_RECOVERABLE = (TypeError, BaseException)
try:
    pass
except _RECOVERABLE:
    pass
""",
            ["shadow BaseException"],
        ),
        "base_exception_rebound_after_handler": (
            """
try:
    pass
except BaseException:
    pass
BaseException = ValueError
""",
            ["shadow BaseException"],
        ),
        "function_parameter_shadows_base_exception": (
            """
def extract(BaseException):
    try:
        pass
    except BaseException:
        pass
""",
            ["shadow BaseException"],
        ),
        "function_runtime_global_shadows_base_exception": (
            """
def extract():
    try:
        operation()
    except BaseException:
        recover()

BaseException = ValueError
extract()
""",
            ["shadow BaseException"],
        ),
        "closure_runtime_binding_shadows_base_exception": (
            """
def outer():
    def extract():
        try:
            operation()
        except BaseException:
            recover()

    BaseException = ValueError
    extract()

outer()
""",
            ["shadow BaseException"],
        ),
        "method_skips_class_base_exception_shadow": (
            """
class Extractor:
    BaseException = ValueError

    def extract(self):
        try:
            pass
        except BaseException:
            pass
""",
            ["shadow BaseException"],
        ),
        "function_called_before_runtime_global_rebind": (
            """
def extract():
    try:
        operation()
    except BaseException:
        recover()

extract()
BaseException = ValueError
""",
            ["shadow BaseException"],
        ),
        "conditional_base_exception_rebind": (
            """
if False:
    BaseException = ValueError
try:
    operation()
except BaseException:
    recover()
""",
            ["shadow BaseException"],
        ),
        "global_base_exception_rebind": (
            """
def extract():
    global BaseException
    try:
        operation()
    except BaseException:
        recover()
    BaseException = ValueError
""",
            ["shadow BaseException"],
        ),
        "deleted_exception_target_rebind": (
            """
try:
    operation()
except ValueError as BaseException:
    recover()
try:
    operation()
except BaseException:
    recover()
""",
            ["shadow BaseException"],
        ),
        "composite_assignment_rebind": (
            """
(BaseException,) = (ValueError,)
try:
    operation()
except BaseException:
    recover()
""",
            ["shadow BaseException"],
        ),
        "match_capture_rebind": (
            """
def extract(value):
    match value:
        case BaseException:
            pass
    try:
        operation()
    except BaseException:
        recover()
""",
            ["shadow BaseException"],
        ),
        "match_star_capture_rebind": (
            """
def extract(value):
    match value:
        case [*BaseException]:
            pass
    try:
        operation()
    except BaseException:
        recover()
""",
            ["shadow BaseException"],
        ),
        "match_mapping_rest_rebind": (
            """
def extract(value):
    match value:
        case {"error": _, **BaseException}:
            pass
    try:
        operation()
    except BaseException:
        recover()
""",
            ["shadow BaseException"],
        ),
        "qualified_bare_reraise": (
            """
import asyncio
try:
    pass
except asyncio.CancelledError:
    raise
""",
            [],
        ),
        "imported_alias_bare_reraise": (
            """
from asyncio import CancelledError as Cancelled
try:
    pass
except Cancelled:
    raise
""",
            [],
        ),
        "exceptions_module_alias_bare_reraise": (
            """
import asyncio.exceptions as ae
try:
    pass
except ae.CancelledError:
    raise
""",
            [],
        ),
        "exceptions_class_alias_bare_reraise": (
            """
from asyncio.exceptions import CancelledError as CE
try:
    pass
except CE:
    raise
""",
            [],
        ),
        "nested_attribute_bare_reraise": (
            """
import asyncio as aio
try:
    pass
except aio.exceptions.CancelledError:
    raise
""",
            [],
        ),
        "imported_exceptions_module_bare_reraise": (
            """
from asyncio import exceptions as aio_exceptions
try:
    pass
except aio_exceptions.CancelledError:
    raise
""",
            [],
        ),
        "qualified_swallow": (
            """
import asyncio
try:
    pass
except asyncio.CancelledError:
    pass
""",
            ["asyncio.CancelledError"],
        ),
        "alias_swallow": (
            """
from asyncio import CancelledError as Cancelled
try:
    pass
except Cancelled:
    return None
""",
            ["Cancelled"],
        ),
        "exceptions_module_alias_swallow": (
            """
import asyncio.exceptions as ae
try:
    pass
except ae.CancelledError:
    pass
""",
            ["ae.CancelledError"],
        ),
        "exceptions_class_alias_swallow": (
            """
from asyncio.exceptions import CancelledError as CE
try:
    pass
except CE:
    return None
""",
            ["CE"],
        ),
        "nested_attribute_swallow": (
            """
import asyncio
try:
    pass
except asyncio.exceptions.CancelledError:
    pass
""",
            ["asyncio.exceptions.CancelledError"],
        ),
        "imported_exceptions_module_swallow": (
            """
from asyncio import exceptions as aio_exceptions
try:
    pass
except aio_exceptions.CancelledError:
    pass
""",
            ["aio_exceptions.CancelledError"],
        ),
        "explicit_exception": (
            """
import asyncio
try:
    pass
except asyncio.CancelledError:
    raise RuntimeError("cancelled")
""",
            ["asyncio.CancelledError"],
        ),
        "failure_construction": (
            """
import asyncio
try:
    pass
except asyncio.CancelledError:
    result = {"error": "cancelled"}
    return result
""",
            ["asyncio.CancelledError"],
        ),
        "fallthrough": (
            """
import asyncio
try:
    pass
except asyncio.CancelledError:
    cleanup()
""",
            ["asyncio.CancelledError"],
        ),
        "retry": (
            """
from asyncio import CancelledError
try:
    pass
except CancelledError:
    retry_request()
""",
            ["CancelledError"],
        ),
        "named_starred_tuple": (
            """
import asyncio
_BASE = (asyncio.CancelledError,)
_RECOVERABLE = (*_BASE, ValueError)
try:
    pass
except _RECOVERABLE:
    pass
""",
            ["_RECOVERABLE"],
        ),
        "composed_tuple": (
            """
from asyncio import CancelledError as Cancelled
_BASE = (ValueError,)
_RECOVERABLE = _BASE + (Cancelled,)
try:
    pass
except _RECOVERABLE:
    pass
""",
            ["_RECOVERABLE"],
        ),
        "function_local_composed_tuple": (
            """
import asyncio
def extract():
    base = (ValueError,)
    recoverable = base + (asyncio.CancelledError,)
    try:
        pass
    except recoverable:
        pass
""",
            ["recoverable"],
        ),
        "function_local_starred_tuple": (
            """
from asyncio.exceptions import CancelledError as CE
def extract():
    base = (CE,)
    recoverable = (*base, ValueError)
    try:
        pass
    except recoverable:
        pass
""",
            ["recoverable"],
        ),
        "augmented_tuple": (
            """
import asyncio
_RECOVERABLE = (ValueError,)
_RECOVERABLE += (asyncio.CancelledError,)
try:
    pass
except _RECOVERABLE:
    pass
""",
            ["_RECOVERABLE"],
        ),
        "annotated_tuple": (
            """
from asyncio import CancelledError
_RECOVERABLE: tuple[type[BaseException], ...] = (ValueError, CancelledError)
try:
    pass
except _RECOVERABLE:
    pass
""",
            ["_RECOVERABLE"],
        ),
        "module_annotation_without_value_does_not_rebind": (
            """
from asyncio import CancelledError as CE
CE: type[BaseException]
try:
    pass
except CE:
    pass
""",
            ["shadow CE"],
        ),
        "function_annotation_without_value_shadows_global": (
            """
from asyncio import CancelledError as CE
def extract():
    CE: type[BaseException]
    try:
        pass
    except CE:
        pass
""",
            ["shadow CE"],
        ),
        "import_rebound_before_handler": (
            """
import asyncio
asyncio = object()
try:
    pass
except asyncio.CancelledError:
    pass
""",
            ["shadow asyncio"],
        ),
        "class_rebound_before_handler": (
            """
from asyncio import CancelledError
CancelledError = ValueError
try:
    pass
except CancelledError:
    pass
""",
            ["shadow CancelledError"],
        ),
        "alias_captures_import_before_rebinding": (
            """
import asyncio
_RECOVERABLE = (asyncio.CancelledError,)
asyncio = object()
try:
    pass
except _RECOVERABLE:
    pass
""",
            ["shadow asyncio"],
        ),
        "rebound_after_handler": (
            """
from asyncio import CancelledError as CE
try:
    pass
except CE:
    pass
CE = ValueError
""",
            ["shadow CE"],
        ),
        "function_global_imported_after_definition": (
            """
def extract():
    try:
        operation()
    except CE:
        recover()

from asyncio import CancelledError as CE
extract()
""",
            ["CE"],
        ),
        "function_global_rebound_after_definition": (
            """
from asyncio import CancelledError as CE
def extract():
    try:
        operation()
    except CE:
        recover()

CE = ValueError
extract()
""",
            ["shadow CE"],
        ),
        "closure_imported_after_inner_definition": (
            """
def outer():
    def extract():
        try:
            operation()
        except CE:
            recover()

    from asyncio import CancelledError as CE
    extract()

outer()
""",
            ["CE"],
        ),
        "closure_rebound_after_inner_definition": (
            """
def outer():
    from asyncio import CancelledError as CE

    def extract():
        try:
            operation()
        except CE:
            recover()

    CE = ValueError
    extract()

outer()
""",
            ["shadow CE"],
        ),
        "absent_asyncio_import": (
            """
try:
    pass
except asyncio.CancelledError:
    pass
""",
            [],
        ),
        "parameter_shadows_module": (
            """
import asyncio
def extract(asyncio):
    try:
        pass
    except asyncio.CancelledError:
        pass
""",
            ["shadow asyncio"],
        ),
        "parameter_shadows_imported_class": (
            """
from asyncio import CancelledError as CE
def extract(CE):
    try:
        pass
    except CE:
        pass
""",
            ["shadow CE"],
        ),
        "method_skips_class_only_alias": (
            """
class Extractor:
    from asyncio import CancelledError as CE

    def extract(self):
        try:
            pass
        except CE:
            pass
""",
            [],
        ),
        "method_skips_class_shadow_for_module_alias": (
            """
from asyncio import CancelledError as CE
class Extractor:
    CE = ValueError

    def extract(self):
        try:
            pass
        except CE:
            pass
""",
            ["shadow CE"],
        ),
        "conditional_imported_alias_rebind": (
            """
from asyncio import CancelledError as CE
if False:
    CE = ValueError
try:
    operation()
except CE:
    recover()
""",
            ["shadow CE"],
        ),
        "conditional_asyncio_root_rebind": (
            """
import asyncio
if False:
    asyncio = object()
try:
    operation()
except asyncio.CancelledError:
    recover()
""",
            ["shadow asyncio"],
        ),
        "function_called_before_imported_alias_rebind": (
            """
from asyncio import CancelledError as CE
def extract():
    try:
        operation()
    except CE:
        recover()

extract()
CE = ValueError
""",
            ["shadow CE"],
        ),
        "composite_imported_alias_rebind": (
            """
from asyncio import CancelledError as CE
(CE,) = (ValueError,)
try:
    operation()
except CE:
    recover()
""",
            ["shadow CE"],
        ),
        "direct_cancellation_alias_rebind": (
            """
from asyncio import CancelledError as CE
Cancellation = CE
if False:
    Cancellation = ValueError
try:
    operation()
except Cancellation:
    recover()
""",
            ["shadow Cancellation"],
        ),
        "assigned_asyncio_root_alias_rebind": (
            """
import asyncio
aio = asyncio
if False:
    aio = object()
try:
    operation()
except aio.CancelledError:
    recover()
""",
            ["shadow aio"],
        ),
        "inline_tuple": (
            """
import asyncio as aio
try:
    pass
except (ValueError, aio.CancelledError):
    pass
""",
            ["(ValueError, aio.CancelledError)"],
        ),
    }

    for source, expected in cases.values():
        assert _recoverable_cancelled_error_violations(ast.parse(source)) == expected


def test_article_cache_stats_do_not_load_watchlists_in_a_fresh_process() -> None:
    script = """
import json
import sys

from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as article

article.get_extraction_cache_stats()
loaded = sorted(
    name
    for name in sys.modules
    if name == "tldw_Server_API.app.core.Watchlists"
    or name.startswith("tldw_Server_API.app.core.Watchlists.")
)
print("SELECTOR_ARCH=" + json.dumps(loaded))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    marker = next(line for line in result.stdout.splitlines() if line.startswith("SELECTOR_ARCH="))

    assert json.loads(marker.removeprefix("SELECTOR_ARCH=")) == []


def test_watchlists_endpoint_imports_validation_from_canonical_facade() -> None:
    facade_module = "tldw_Server_API.app.core.Web_Scraping.selectors"
    fetchers_module = "tldw_Server_API.app.core.Watchlists.fetchers"

    assert _imported_names(ENDPOINT_PATH, facade_module) == {"validate_selector_rules"}
    assert _imported_names(ENDPOINT_PATH, fetchers_module) == {
        "fetch_rss_feed",
        "fetch_site_items_with_rules",
    }
