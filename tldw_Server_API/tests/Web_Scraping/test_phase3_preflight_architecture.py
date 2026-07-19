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
    "curl_cffi",
    "playwright",
    "subprocess",
    "tldw_Server_API.app.core.http_client",
    "tldw_Server_API.app.core.Security.egress",
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

PUBLIC_CALLABLE_PAIRS = {
    LEGACY_PACKAGE: (PREFLIGHT_PACKAGE, ("gather_analysis", "run_analysis")),
    f"{LEGACY_PACKAGE}.runner": (
        f"{PREFLIGHT_PACKAGE}.runner",
        (
            "analyze_fingerprinting",
            "analyze_function_integrity",
            "analyze_js_rendering",
            "analyze_tls_fingerprint",
            "check_robots_txt",
            "detect_captcha",
            "detect_honeypots",
            "detect_waf",
            "gather_analysis",
            "profile_rate_limits",
            "run_analysis",
        ),
    ),
    f"{LEGACY_PACKAGE}.analyzers.behavioral_detector": (
        f"{PREFLIGHT_PACKAGE}.analyzers.behavioral_detector",
        ("detect_honeypots",),
    ),
    f"{LEGACY_PACKAGE}.analyzers.captcha_detector": (
        f"{PREFLIGHT_PACKAGE}.analyzers.captcha_detector",
        ("detect_captcha",),
    ),
    f"{LEGACY_PACKAGE}.analyzers.fingerprint_analyzer": (
        f"{PREFLIGHT_PACKAGE}.analyzers.fingerprint_analyzer",
        ("analyze_fingerprinting",),
    ),
    f"{LEGACY_PACKAGE}.analyzers.integrity_analyzer": (
        f"{PREFLIGHT_PACKAGE}.analyzers.integrity_analyzer",
        ("analyze_function_integrity",),
    ),
    f"{LEGACY_PACKAGE}.analyzers.js_detector": (
        f"{PREFLIGHT_PACKAGE}.analyzers.js_detector",
        ("analyze_js_rendering",),
    ),
    f"{LEGACY_PACKAGE}.analyzers.rate_limit_profiler": (
        f"{PREFLIGHT_PACKAGE}.analyzers.rate_limit_profiler",
        ("profile_rate_limits",),
    ),
    f"{LEGACY_PACKAGE}.analyzers.robots_checker": (
        f"{PREFLIGHT_PACKAGE}.analyzers.robots_checker",
        ("check_robots_txt",),
    ),
    f"{LEGACY_PACKAGE}.analyzers.tls_analyzer": (
        f"{PREFLIGHT_PACKAGE}.analyzers.tls_analyzer",
        ("analyze_tls_fingerprint",),
    ),
    f"{LEGACY_PACKAGE}.analyzers.waf_detector": (
        f"{PREFLIGHT_PACKAGE}.analyzers.waf_detector",
        ("detect_waf",),
    ),
    f"{LEGACY_PACKAGE}.recommendations.recommender": (
        f"{PREFLIGHT_PACKAGE}.recommendations.recommender",
        ("generate_recommendations",),
    ),
    f"{LEGACY_PACKAGE}.recommendations": (
        f"{PREFLIGHT_PACKAGE}.recommendations",
        ("generate_recommendations",),
    ),
    f"{LEGACY_PACKAGE}.scoring.scoring_engine": (
        f"{PREFLIGHT_PACKAGE}.scoring.scoring_engine",
        ("calculate_difficulty_score",),
    ),
    f"{LEGACY_PACKAGE}.scoring": (
        f"{PREFLIGHT_PACKAGE}.scoring",
        ("calculate_difficulty_score",),
    ),
    f"{LEGACY_PACKAGE}.utils.impersonate_target": (
        f"{PREFLIGHT_PACKAGE}.utils.impersonate_target",
        ("get_impersonate_target",),
    ),
    f"{LEGACY_PACKAGE}.utils.waf_result_parser": (
        f"{PREFLIGHT_PACKAGE}.utils.waf_result_parser",
        ("clean_text", "parse_wafw00f_output"),
    ),
    f"{LEGACY_PACKAGE}.utils": (
        f"{PREFLIGHT_PACKAGE}.utils",
        ("get_impersonate_target", "parse_wafw00f_output"),
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
        relative_path = self.path.relative_to(REPO_ROOT).as_posix()
        return f"{relative_path}:{self.line} imports {self.imported_path}"


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


def test_preflight_dependency_direction() -> None:
    assert_no_imports(PREFLIGHT_ROOT, FORBIDDEN_PREFLIGHT_CONSUMERS)
    assert_no_imports(PREFLIGHT_ANALYZERS, FORBIDDEN_ANALYZER_IMPORTS)
    assert_no_imports(
        RUNTIME_ROOT,
        {
            PREFLIGHT_PACKAGE,
            f"{WEB_SCRAPING_PACKAGE}.policy",
        },
    )


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
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        all_assignments = 0
        for index, statement in enumerate(tree.body):
            if index == 0 and _is_docstring(statement):
                continue
            if isinstance(statement, ast.ImportFrom) and statement.module == "__future__":
                continue
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                if any(alias.name == "*" for alias in statement.names):
                    violations.append(f"{path.relative_to(REPO_ROOT)}:{statement.lineno} uses a wildcard re-export")
                continue
            if _is_explicit_all_assignment(statement):
                all_assignments += 1
                continue
            violations.append(
                f"{path.relative_to(REPO_ROOT)}:{getattr(statement, 'lineno', '?')} "
                f"contains disallowed {type(statement).__name__}"
            )
        if all_assignments != 1:
            violations.append(f"{path.relative_to(REPO_ROOT)} must contain exactly one explicit __all__ assignment")

    assert violations == []


def test_legacy_public_callables_match_canonical_contracts() -> None:
    violations: list[str] = []
    for legacy_name, (canonical_name, callable_names) in PUBLIC_CALLABLE_PAIRS.items():
        legacy_module = importlib.import_module(legacy_name)
        canonical_module = importlib.import_module(canonical_name)
        for callable_name in callable_names:
            legacy_callable = getattr(legacy_module, callable_name)
            canonical_callable = getattr(canonical_module, callable_name)
            if legacy_callable is not canonical_callable:
                violations.append(f"{legacy_name}.{callable_name} is not the canonical callable")
            if inspect.signature(legacy_callable) != inspect.signature(canonical_callable):
                violations.append(f"{legacy_name}.{callable_name} has a different signature")
            if inspect.iscoroutinefunction(legacy_callable) != inspect.iscoroutinefunction(canonical_callable):
                violations.append(f"{legacy_name}.{callable_name} changed coroutine classification")

    assert violations == []


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
