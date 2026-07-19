"""Architecture contracts for the governed Phase 3 preflight package."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
from dataclasses import dataclass
from pathlib import Path

import pytest
import tomllib
from packaging.requirements import Requirement

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
APP_ROOT = REPO_ROOT / "tldw_Server_API/app"
WEB_SCRAPING_ROOT = APP_ROOT / "core/Web_Scraping"
PREFLIGHT_ROOT = WEB_SCRAPING_ROOT / "preflight"
PREFLIGHT_ANALYZERS = PREFLIGHT_ROOT / "analyzers"
RUNTIME_ROOT = WEB_SCRAPING_ROOT / "runtime"
SCRAPER_ANALYZERS_ROOT = WEB_SCRAPING_ROOT / "scraper_analyzers"

WEB_SCRAPING_PACKAGE = "tldw_Server_API.app.core.Web_Scraping"
PREFLIGHT_PACKAGE = f"{WEB_SCRAPING_PACKAGE}.preflight"
LEGACY_PACKAGE = f"{WEB_SCRAPING_PACKAGE}.scraper_analyzers"

FORBIDDEN_ANALYZER_IMPORTS = {
    "asyncio.subprocess",
    "aiohttp",
    "curl_cffi",
    "http.client",
    "httpx",
    "playwright",
    "requests",
    "subprocess",
    "tldw_Server_API.app.core.http_client",
    "tldw_Server_API.app.core.Security.egress",
    "urllib.request",
    "urllib3",
}
FORBIDDEN_ANALYZER_PROCESS_CALLS = {
    "asyncio.create_subprocess_exec",
    "asyncio.create_subprocess_shell",
    "os.popen",
    "os.posix_spawn",
    "os.posix_spawnp",
    "os.system",
}
FORBIDDEN_PREFLIGHT_CONSUMERS = {
    f"{WEB_SCRAPING_PACKAGE}.Article_Extractor_Lib",
    f"{WEB_SCRAPING_PACKAGE}.enhanced_web_scraping",
}
CONSUMER_PATHS = (
    WEB_SCRAPING_ROOT / "Article_Extractor_Lib.py",
    WEB_SCRAPING_ROOT / "enhanced_web_scraping.py",
)

# These are the only application imports of scraper_analyzers recorded by the
# Phase 0 inventory. Consumer-specific checks below reject their reintroduction.
PHASE0_APPLICATION_LEGACY_IMPORT_ALLOWLIST = {
    (
        "tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py",
        LEGACY_PACKAGE,
        "run_analysis",
    ),
    (
        "tldw_Server_API/app/core/Web_Scraping/enhanced_web_scraping.py",
        LEGACY_PACKAGE,
        "run_analysis",
    ),
}

PHASE0_LEGACY_MODULES = {
    LEGACY_PACKAGE,
    f"{LEGACY_PACKAGE}.analyzers",
    f"{LEGACY_PACKAGE}.analyzers.behavioral_detector",
    f"{LEGACY_PACKAGE}.analyzers.captcha_detector",
    f"{LEGACY_PACKAGE}.analyzers.fingerprint_analyzer",
    f"{LEGACY_PACKAGE}.analyzers.integrity_analyzer",
    f"{LEGACY_PACKAGE}.analyzers.js_detector",
    f"{LEGACY_PACKAGE}.analyzers.rate_limit_profiler",
    f"{LEGACY_PACKAGE}.analyzers.robots_checker",
    f"{LEGACY_PACKAGE}.analyzers.tls_analyzer",
    f"{LEGACY_PACKAGE}.analyzers.waf_detector",
    f"{LEGACY_PACKAGE}.recommendations.recommender",
    f"{LEGACY_PACKAGE}.runner",
    f"{LEGACY_PACKAGE}.scoring.scoring_engine",
    f"{LEGACY_PACKAGE}.utils.browser_identities",
    f"{LEGACY_PACKAGE}.utils.impersonate_target",
    f"{LEGACY_PACKAGE}.utils.waf_result_parser",
}
ALLOWED_CHILD_SHIM_IMPORTS = {
    "analyzers/__init__.py": frozenset(
        {
            "behavioral_detector",
            "captcha_detector",
            "fingerprint_analyzer",
            "integrity_analyzer",
            "js_detector",
            "rate_limit_profiler",
            "robots_checker",
            "tls_analyzer",
            "waf_detector",
        }
    ),
}


@dataclass(frozen=True)
class ImportReference:
    path: Path
    line: int
    module: str
    imported_name: str | None

    @property
    def imported_path(self) -> str:
        if self.imported_name is None:
            return self.module
        return f"{self.module}.{self.imported_name}" if self.module else self.imported_name

    def display(self) -> str:
        return f"{_display_path(self.path)}:{self.line} imports {self.imported_path}"


@dataclass(frozen=True)
class CanonicalReexport:
    path: Path
    legacy_module: str
    canonical_module: str
    imported_name: str
    bound_name: str


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _module_name_for_path(path: Path) -> str:
    module_path = path.relative_to(REPO_ROOT).with_suffix("")
    parts = list(module_path.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_from_module(path: Path, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""

    current_parts = _module_name_for_path(path).split(".")
    package_parts = current_parts if path.name == "__init__.py" else current_parts[:-1]
    levels_up = node.level - 1
    if levels_up:
        package_parts = package_parts[:-levels_up]
    if node.module:
        package_parts.extend(node.module.split("."))
    return ".".join(package_parts)


def _import_references(path: Path) -> list[ImportReference]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    references: list[ImportReference] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            references.extend(
                ImportReference(path=path, line=node.lineno, module=alias.name, imported_name=None)
                for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            module = _resolve_from_module(path, node)
            references.extend(
                ImportReference(path=path, line=node.lineno, module=module, imported_name=alias.name)
                for alias in node.names
            )
    return references


def _python_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.py"))


def _matches_module(candidate: str, module: str) -> bool:
    return candidate == module or candidate.startswith(f"{module}.")


def _reference_matches(reference: ImportReference, module: str) -> bool:
    return _matches_module(reference.module, module) or _matches_module(reference.imported_path, module)


def assert_no_imports(root: Path, forbidden_modules: set[str]) -> None:
    violations: list[str] = []
    for path in _python_files(root):
        for reference in _import_references(path):
            matches = sorted(module for module in forbidden_modules if _reference_matches(reference, module))
            if matches:
                violations.append(f"{reference.display()} (forbidden: {', '.join(matches)})")
    assert violations == []


def _dotted_expression_name(expression: ast.expr) -> str | None:
    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Attribute):
        parent = _dotted_expression_name(expression.value)
        if parent is not None:
            return f"{parent}.{expression.attr}"
    return None


def _import_bindings(tree: ast.AST) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound_name = alias.asname or alias.name.split(".", maxsplit=1)[0]
                bindings[bound_name] = alias.name if alias.asname else bound_name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            for alias in node.names:
                if alias.name == "*":
                    continue
                bindings[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return bindings


def _resolve_bound_expression(expression: ast.expr, bindings: dict[str, str]) -> str | None:
    dotted_name = _dotted_expression_name(expression)
    if dotted_name is None:
        return None
    root_name, separator, remainder = dotted_name.partition(".")
    imported_root = bindings.get(root_name)
    if imported_root is None:
        return dotted_name
    return f"{imported_root}.{remainder}" if separator else imported_root


def _is_forbidden_process_call(call_name: str) -> bool:
    if call_name in FORBIDDEN_ANALYZER_PROCESS_CALLS:
        return True
    if call_name.startswith("subprocess."):
        return True
    return call_name.startswith("os.exec") or call_name.startswith("os.spawn")


def assert_analyzer_dependencies_are_governed(root: Path) -> None:
    violations: list[str] = []
    for path in _python_files(root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for reference in _import_references(path):
            matches = sorted(module for module in FORBIDDEN_ANALYZER_IMPORTS if _reference_matches(reference, module))
            if matches:
                violations.append(f"{reference.display()} (forbidden: {', '.join(matches)})")

        bindings = _import_bindings(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = _resolve_bound_expression(node.func, bindings)
            if call_name is not None and _is_forbidden_process_call(call_name):
                violations.append(f"{_display_path(path)}:{node.lineno} calls ungoverned process API {call_name}")

    assert violations == []


def _is_docstring(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _is_explicit_all_assignment(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return False
    target = statement.targets[0]
    if not isinstance(target, ast.Name) or target.id != "__all__":
        return False
    if not isinstance(statement.value, (ast.List, ast.Tuple)):
        return False
    return all(isinstance(item, ast.Constant) and isinstance(item.value, str) for item in statement.value.elts)


def _explicit_all_names(statement: ast.stmt) -> list[str] | None:
    if not _is_explicit_all_assignment(statement):
        return None
    assert isinstance(statement, ast.Assign)
    assert isinstance(statement.value, (ast.List, ast.Tuple))
    return [item.value for item in statement.value.elts if isinstance(item, ast.Constant)]


def _shim_module_name(path: Path, root: Path) -> str:
    relative_path = path.relative_to(root).with_suffix("")
    parts = list(relative_path.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join((LEGACY_PACKAGE, *parts))


def _resolve_from_module_name(module_name: str, is_package: bool, node: ast.ImportFrom) -> str:
    if node.level == 0:
        return node.module or ""
    package_parts = module_name.split(".") if is_package else module_name.split(".")[:-1]
    levels_up = node.level - 1
    if levels_up:
        package_parts = package_parts[:-levels_up]
    if node.module:
        package_parts.extend(node.module.split("."))
    return ".".join(package_parts)


def _inspect_shim(
    path: Path,
    root: Path,
) -> tuple[list[CanonicalReexport], set[str], list[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    relative_path = path.relative_to(root).as_posix()
    legacy_module = _shim_module_name(path, root)
    allowed_children = ALLOWED_CHILD_SHIM_IMPORTS.get(relative_path)
    canonical_exports: list[CanonicalReexport] = []
    child_exports: set[str] = set()
    imported_bindings: list[str] = []
    all_names: list[str] | None = None
    all_assignments = 0
    violations: list[str] = []

    for index, statement in enumerate(tree.body):
        if index == 0 and _is_docstring(statement):
            continue
        if isinstance(statement, ast.ImportFrom) and statement.module == "__future__":
            continue
        explicit_all = _explicit_all_names(statement)
        if explicit_all is not None:
            all_assignments += 1
            all_names = explicit_all
            continue
        if isinstance(statement, ast.Import):
            violations.append(f"{_display_path(path)}:{statement.lineno} imports a noncanonical or side-effect module")
            continue
        if isinstance(statement, ast.ImportFrom):
            if any(alias.name == "*" for alias in statement.names):
                violations.append(f"{_display_path(path)}:{statement.lineno} uses a wildcard re-export")
                continue

            source_module = _resolve_from_module_name(
                legacy_module,
                path.name == "__init__.py",
                statement,
            )
            is_child_import = statement.level == 1 and statement.module is None
            if allowed_children is not None and is_child_import:
                for alias in statement.names:
                    bound_name = alias.asname or alias.name
                    if alias.asname is not None or alias.name not in allowed_children:
                        violations.append(
                            f"{_display_path(path)}:{statement.lineno} imports unexpected child shim " f"{alias.name}"
                        )
                    child_exports.add(bound_name)
                    imported_bindings.append(bound_name)
                continue
            if allowed_children is not None:
                violations.append(
                    f"{_display_path(path)}:{statement.lineno} package aggregator may import only its "
                    "known relative child shims"
                )
                continue
            if not _matches_module(source_module, PREFLIGHT_PACKAGE):
                violations.append(
                    f"{_display_path(path)}:{statement.lineno} imports noncanonical source {source_module}"
                )
                continue

            for alias in statement.names:
                bound_name = alias.asname or alias.name
                imported_bindings.append(bound_name)
                canonical_exports.append(
                    CanonicalReexport(
                        path=path,
                        legacy_module=legacy_module,
                        canonical_module=source_module,
                        imported_name=alias.name,
                        bound_name=bound_name,
                    )
                )
            continue
        violations.append(
            f"{_display_path(path)}:{getattr(statement, 'lineno', '?')} "
            f"contains disallowed {type(statement).__name__}"
        )

    if all_assignments != 1:
        violations.append(f"{_display_path(path)} must contain exactly one explicit __all__ assignment")
    elif all_names is not None:
        duplicate_exports = sorted(name for name in set(all_names) if all_names.count(name) > 1)
        if duplicate_exports:
            violations.append(f"{_display_path(path)} repeats __all__ exports: {', '.join(duplicate_exports)}")
        missing_exports = sorted(set(imported_bindings) - set(all_names))
        unexpected_exports = sorted(set(all_names) - set(imported_bindings))
        if missing_exports or unexpected_exports:
            violations.append(
                f"{_display_path(path)} __all__ disagrees with imported bindings "
                f"(missing: {missing_exports}; unexpected: {unexpected_exports})"
            )

    if allowed_children is not None and child_exports != allowed_children:
        violations.append(
            f"{_display_path(path)} child shim set differs "
            f"(missing: {sorted(allowed_children - child_exports)}; "
            f"unexpected: {sorted(child_exports - allowed_children)})"
        )

    return canonical_exports, child_exports, violations


def _write_python_fixture(root: Path, relative_path: str, source: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_preflight_dependency_direction() -> None:
    assert_no_imports(PREFLIGHT_ROOT, FORBIDDEN_PREFLIGHT_CONSUMERS)
    assert_analyzer_dependencies_are_governed(PREFLIGHT_ANALYZERS)
    assert_no_imports(
        RUNTIME_ROOT,
        {
            PREFLIGHT_PACKAGE,
            f"{WEB_SCRAPING_PACKAGE}.policy",
        },
    )


@pytest.mark.parametrize(
    "source",
    (
        "import aiohttp as http\n\nasync def probe():\n    return http.ClientSession()\n",
        "from httpx import AsyncClient as Client\n\nasync def probe():\n    return Client()\n",
        "import asyncio as aio\n\nasync def probe():\n    return await aio.create_subprocess_exec('tool')\n",
        "from asyncio import create_subprocess_shell as spawn\n\nasync def probe():\n    return await spawn('tool')\n",
        "import os as operating_system\n\ndef probe():\n    return operating_system.system('tool')\n",
        "from os import posix_spawn as spawn\n\ndef probe():\n    return spawn('tool', ['tool'], {})\n",
        "import subprocess as process\n\ndef probe():\n    return process.Popen(['tool'])\n",
    ),
)
def test_analyzer_guard_rejects_direct_ungoverned_resource_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    _write_python_fixture(tmp_path, "mutated_analyzer.py", source)
    monkeypatch.setitem(globals(), "PREFLIGHT_ANALYZERS", tmp_path)

    with pytest.raises(AssertionError, match="mutated_analyzer.py"):
        test_preflight_dependency_direction()


def test_analyzer_guard_allows_asyncio_timing_and_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_python_fixture(
        tmp_path,
        "safe_analyzer.py",
        """import asyncio as aio

async def wait_for_result(awaitable):
    try:
        return await aio.wait_for(awaitable, timeout=1)
    except aio.CancelledError:
        raise
""",
    )
    monkeypatch.setitem(globals(), "PREFLIGHT_ANALYZERS", tmp_path)

    test_preflight_dependency_direction()


def test_scrape_consumers_import_only_the_package_level_preflight_facade() -> None:
    violations: list[str] = []
    for path in CONSUMER_PATHS:
        canonical_imports: list[ImportReference] = []
        for reference in _import_references(path):
            if _reference_matches(reference, LEGACY_PACKAGE):
                violations.append(f"{reference.display()} (legacy analyzer package)")
            if _reference_matches(reference, PREFLIGHT_PACKAGE):
                canonical_imports.append(reference)
                if reference.imported_path != PREFLIGHT_PACKAGE:
                    violations.append(f"{reference.display()} (consumer must import package-level preflight)")
        if not canonical_imports:
            violations.append(f"{path.relative_to(REPO_ROOT).as_posix()} does not import {PREFLIGHT_PACKAGE}")

    assert violations == []


def test_application_has_no_new_phase0_legacy_analyzer_imports() -> None:
    violations: list[str] = []
    for path in _python_files(APP_ROOT):
        if path.is_relative_to(SCRAPER_ANALYZERS_ROOT):
            continue
        for reference in _import_references(path):
            if not _reference_matches(reference, LEGACY_PACKAGE):
                continue
            record = (
                path.relative_to(REPO_ROOT).as_posix(),
                reference.module,
                reference.imported_name,
            )
            if record not in PHASE0_APPLICATION_LEGACY_IMPORT_ALLOWLIST:
                violations.append(f"{reference.display()} (not present in the Phase 0 allowlist)")

    assert violations == []


def test_every_phase0_legacy_module_resolves() -> None:
    unresolved = sorted(module for module in PHASE0_LEGACY_MODULES if importlib.util.find_spec(module) is None)
    assert unresolved == []


def test_legacy_shims_contain_only_explicit_reexports() -> None:
    violations: list[str] = []
    for path in _python_files(SCRAPER_ANALYZERS_ROOT):
        _, _, shim_violations = _inspect_shim(path, SCRAPER_ANALYZERS_ROOT)
        violations.extend(shim_violations)

    assert violations == []


@pytest.mark.parametrize(
    ("relative_path", "source"),
    (
        (
            "runner.py",
            '"""Mutated shim."""\n\nfrom os import environ\n\n__all__ = ["environ"]\n',
        ),
        (
            "runner.py",
            '"""Mutated shim."""\n\nfrom ..preflight.runner import gather_analysis\n\n' '__all__ = ["run_analysis"]\n',
        ),
        (
            "runner.py",
            '"""Mutated shim."""\n\nfrom ..preflight.runner import gather_analysis\n\n__all__ = []\n',
        ),
        (
            "analyzers/__init__.py",
            '"""Mutated shim."""\n\nfrom . import behavioral_detector, surprise\n\n'
            '__all__ = ["behavioral_detector", "surprise"]\n',
        ),
    ),
)
def test_shim_guard_rejects_noncanonical_or_incomplete_reexports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
    source: str,
) -> None:
    _write_python_fixture(tmp_path, relative_path, source)
    monkeypatch.setitem(globals(), "SCRAPER_ANALYZERS_ROOT", tmp_path)

    with pytest.raises(AssertionError, match=Path(relative_path).name):
        test_legacy_shims_contain_only_explicit_reexports()


def test_legacy_public_exports_match_canonical_contracts() -> None:
    violations: list[str] = []
    canonical_exports: list[CanonicalReexport] = []
    for path in _python_files(SCRAPER_ANALYZERS_ROOT):
        path_exports, _, shim_violations = _inspect_shim(path, SCRAPER_ANALYZERS_ROOT)
        canonical_exports.extend(path_exports)
        violations.extend(shim_violations)

    for export in canonical_exports:
        legacy_module = importlib.import_module(export.legacy_module)
        canonical_module = importlib.import_module(export.canonical_module)
        legacy_value = getattr(legacy_module, export.bound_name)
        canonical_value = getattr(canonical_module, export.imported_name)
        export_label = f"{export.legacy_module}.{export.bound_name}"
        if legacy_value is not canonical_value:
            violations.append(f"{export_label} is not the canonical export")

        if inspect.isfunction(canonical_value) or inspect.ismethod(canonical_value):
            try:
                canonical_signature = inspect.signature(canonical_value)
                legacy_signature = inspect.signature(legacy_value)
            except (TypeError, ValueError):
                pass
            else:
                if legacy_signature != canonical_signature:
                    violations.append(f"{export_label} has a different signature")
            if inspect.iscoroutinefunction(legacy_value) != inspect.iscoroutinefunction(canonical_value):
                violations.append(f"{export_label} changed coroutine classification")

    for relative_path, expected_children in ALLOWED_CHILD_SHIM_IMPORTS.items():
        path = SCRAPER_ANALYZERS_ROOT / relative_path
        legacy_module_name = _shim_module_name(path, SCRAPER_ANALYZERS_ROOT)
        legacy_module = importlib.import_module(legacy_module_name)
        actual_children = set(legacy_module.__all__)
        if actual_children != expected_children:
            violations.append(
                f"{legacy_module_name} child set differs "
                f"(missing: {sorted(expected_children - actual_children)}; "
                f"unexpected: {sorted(actual_children - expected_children)})"
            )
        for child_name in expected_children:
            legacy_child_name = f"{legacy_module_name}.{child_name}"
            canonical_child_name = f"{PREFLIGHT_PACKAGE}.analyzers.{child_name}"
            exported_child = getattr(legacy_module, child_name)
            legacy_child = importlib.import_module(legacy_child_name)
            canonical_child = importlib.import_module(canonical_child_name)
            if exported_child is not legacy_child:
                violations.append(f"{legacy_module_name}.{child_name} is not its legacy child module")
            if exported_child is canonical_child:
                violations.append(f"{legacy_module_name}.{child_name} unexpectedly aliases the canonical module")

    assert violations == []


@pytest.mark.parametrize(
    ("legacy_name", "export_name"),
    (
        (f"{LEGACY_PACKAGE}.analyzers.behavioral_detector", "HONEYPOT_THRESHOLD"),
        (f"{LEGACY_PACKAGE}.analyzers.behavioral_detector", "ScanDepth"),
        (f"{LEGACY_PACKAGE}.runner", "AnalysisOutput"),
        (f"{LEGACY_PACKAGE}.utils", "MODERN_BROWSER_IDENTITIES"),
    ),
)
def test_legacy_export_identity_guard_covers_constants_and_callable_types(
    monkeypatch: pytest.MonkeyPatch,
    legacy_name: str,
    export_name: str,
) -> None:
    legacy_module = importlib.import_module(legacy_name)
    monkeypatch.setattr(legacy_module, export_name, object())

    with pytest.raises(AssertionError, match=export_name):
        test_legacy_public_exports_match_canonical_contracts()


def test_all_playwright_dependency_floors_are_exactly_1_48() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    requirement_groups = {
        "base": project["dependencies"],
        "web_research": project["optional-dependencies"]["web_research"],
        "scrape-analyzers": project["optional-dependencies"]["scrape-analyzers"],
    }
    violations: list[str] = []
    for group_name, requirement_strings in requirement_groups.items():
        playwright_requirements = [
            Requirement(requirement)
            for requirement in requirement_strings
            if Requirement(requirement).name.lower() == "playwright"
        ]
        if len(playwright_requirements) != 1:
            violations.append(f"{group_name} has {len(playwright_requirements)} Playwright requirements")
            continue
        specifier = str(playwright_requirements[0].specifier)
        if specifier != ">=1.48.0":
            violations.append(f"{group_name} pins Playwright as {specifier!r}, expected '>=1.48.0'")

    assert violations == []
