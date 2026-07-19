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


@dataclass(frozen=True)
class ExpectedReexport:
    canonical_module: str
    imported_name: str
    bound_name: str


@dataclass(frozen=True)
class ExpectedShim:
    canonical_reexports: tuple[ExpectedReexport, ...]
    child_modules: frozenset[str] = frozenset()

    @property
    def all_exports(self) -> frozenset[str]:
        return frozenset(export.bound_name for export in self.canonical_reexports) | self.child_modules


ALLOWED_ANALYZER_IMPORT_MODULES = {
    "__future__",
    "asyncio",
    "bs4",
    "collections.abc",
    "json",
    "math",
    "typing",
    "urllib.parse",
}
FORBIDDEN_ASYNCIO_DIRECT_IO_APIS = {
    "create_connection",
    "create_datagram_endpoint",
    "create_subprocess_exec",
    "create_subprocess_shell",
    "create_unix_connection",
    "open_connection",
    "open_unix_connection",
    "sock_connect",
    "subprocess_exec",
    "subprocess_shell",
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

EXPECTED_SHIM_MANIFEST = {
    "__init__.py": ExpectedShim(
        canonical_reexports=(
            ExpectedReexport(f"{PREFLIGHT_PACKAGE}.runner", "gather_analysis", "gather_analysis"),
            ExpectedReexport(f"{PREFLIGHT_PACKAGE}.runner", "run_analysis", "run_analysis"),
        ),
    ),
    "analyzers/__init__.py": ExpectedShim(
        canonical_reexports=(),
        child_modules=frozenset(
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
    ),
    "analyzers/behavioral_detector.py": ExpectedShim(
        canonical_reexports=(
            ExpectedReexport(
                f"{PREFLIGHT_PACKAGE}.analyzers.behavioral_detector",
                "HONEYPOT_THRESHOLD",
                "HONEYPOT_THRESHOLD",
            ),
            ExpectedReexport(f"{PREFLIGHT_PACKAGE}.analyzers.behavioral_detector", "ScanDepth", "ScanDepth"),
            ExpectedReexport(
                f"{PREFLIGHT_PACKAGE}.analyzers.behavioral_detector",
                "detect_honeypots",
                "detect_honeypots",
            ),
        ),
    ),
    "analyzers/captcha_detector.py": ExpectedShim(
        canonical_reexports=(
            ExpectedReexport(
                f"{PREFLIGHT_PACKAGE}.analyzers.captcha_detector",
                "CAPTCHA_FINGERPRINTS",
                "CAPTCHA_FINGERPRINTS",
            ),
            ExpectedReexport(
                f"{PREFLIGHT_PACKAGE}.analyzers.captcha_detector",
                "detect_captcha",
                "detect_captcha",
            ),
        ),
    ),
    "analyzers/fingerprint_analyzer.py": ExpectedShim(
        canonical_reexports=(
            ExpectedReexport(
                f"{PREFLIGHT_PACKAGE}.analyzers.fingerprint_analyzer",
                "JS_PROBE_SCRIPT",
                "JS_PROBE_SCRIPT",
            ),
            ExpectedReexport(
                f"{PREFLIGHT_PACKAGE}.analyzers.fingerprint_analyzer",
                "KNOWN_BOT_DETECTION_SCRIPTS",
                "KNOWN_BOT_DETECTION_SCRIPTS",
            ),
            ExpectedReexport(
                f"{PREFLIGHT_PACKAGE}.analyzers.fingerprint_analyzer",
                "KNOWN_BOT_GLOBAL_OBJECTS",
                "KNOWN_BOT_GLOBAL_OBJECTS",
            ),
            ExpectedReexport(
                f"{PREFLIGHT_PACKAGE}.analyzers.fingerprint_analyzer",
                "analyze_fingerprinting",
                "analyze_fingerprinting",
            ),
        ),
    ),
    "analyzers/integrity_analyzer.py": ExpectedShim(
        canonical_reexports=(
            ExpectedReexport(
                f"{PREFLIGHT_PACKAGE}.analyzers.integrity_analyzer",
                "FUNCTION_SUSPICION_MAP",
                "FUNCTION_SUSPICION_MAP",
            ),
            ExpectedReexport(
                f"{PREFLIGHT_PACKAGE}.analyzers.integrity_analyzer",
                "FUNCTIONS_TO_CHECK",
                "FUNCTIONS_TO_CHECK",
            ),
            ExpectedReexport(
                f"{PREFLIGHT_PACKAGE}.analyzers.integrity_analyzer",
                "analyze_function_integrity",
                "analyze_function_integrity",
            ),
        ),
    ),
    "analyzers/js_detector.py": ExpectedShim(
        canonical_reexports=(
            ExpectedReexport(
                f"{PREFLIGHT_PACKAGE}.analyzers.js_detector",
                "analyze_js_rendering",
                "analyze_js_rendering",
            ),
        ),
    ),
    "analyzers/rate_limit_profiler.py": ExpectedShim(
        canonical_reexports=(
            ExpectedReexport(
                f"{PREFLIGHT_PACKAGE}.analyzers.rate_limit_profiler",
                "BLOCKING_STATUS_CODES",
                "BLOCKING_STATUS_CODES",
            ),
            ExpectedReexport(
                f"{PREFLIGHT_PACKAGE}.analyzers.rate_limit_profiler",
                "BURST_COUNT",
                "BURST_COUNT",
            ),
            ExpectedReexport(
                f"{PREFLIGHT_PACKAGE}.analyzers.rate_limit_profiler",
                "DEFAULT_DELAY",
                "DEFAULT_DELAY",
            ),
            ExpectedReexport(
                f"{PREFLIGHT_PACKAGE}.analyzers.rate_limit_profiler",
                "GENTLE_PROBE_COUNT",
                "GENTLE_PROBE_COUNT",
            ),
            ExpectedReexport(
                f"{PREFLIGHT_PACKAGE}.analyzers.rate_limit_profiler",
                "profile_rate_limits",
                "profile_rate_limits",
            ),
        ),
    ),
    "analyzers/robots_checker.py": ExpectedShim(
        canonical_reexports=(
            ExpectedReexport(
                f"{PREFLIGHT_PACKAGE}.analyzers.robots_checker",
                "check_robots_txt",
                "check_robots_txt",
            ),
        ),
    ),
    "analyzers/tls_analyzer.py": ExpectedShim(
        canonical_reexports=(
            ExpectedReexport(
                f"{PREFLIGHT_PACKAGE}.analyzers.tls_analyzer",
                "analyze_tls_fingerprint",
                "analyze_tls_fingerprint",
            ),
        ),
    ),
    "analyzers/waf_detector.py": ExpectedShim(
        canonical_reexports=(
            ExpectedReexport(f"{PREFLIGHT_PACKAGE}.analyzers.waf_detector", "detect_waf", "detect_waf"),
        ),
    ),
    "recommendations/__init__.py": ExpectedShim(
        canonical_reexports=(
            ExpectedReexport(
                f"{PREFLIGHT_PACKAGE}.recommendations.recommender",
                "generate_recommendations",
                "generate_recommendations",
            ),
        ),
    ),
    "recommendations/recommender.py": ExpectedShim(
        canonical_reexports=(
            ExpectedReexport(
                f"{PREFLIGHT_PACKAGE}.recommendations.recommender",
                "generate_recommendations",
                "generate_recommendations",
            ),
        ),
    ),
    "runner.py": ExpectedShim(
        canonical_reexports=tuple(
            ExpectedReexport(f"{PREFLIGHT_PACKAGE}.runner", name, name)
            for name in (
                "AnalysisOutput",
                "ScanDepth",
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
            )
        ),
    ),
    "scoring/__init__.py": ExpectedShim(
        canonical_reexports=(
            ExpectedReexport(
                f"{PREFLIGHT_PACKAGE}.scoring.scoring_engine",
                "calculate_difficulty_score",
                "calculate_difficulty_score",
            ),
        ),
    ),
    "scoring/scoring_engine.py": ExpectedShim(
        canonical_reexports=(
            ExpectedReexport(
                f"{PREFLIGHT_PACKAGE}.scoring.scoring_engine",
                "calculate_difficulty_score",
                "calculate_difficulty_score",
            ),
        ),
    ),
    "utils/__init__.py": ExpectedShim(
        canonical_reexports=(
            ExpectedReexport(
                f"{PREFLIGHT_PACKAGE}.utils.browser_identities",
                "MODERN_BROWSER_IDENTITIES",
                "MODERN_BROWSER_IDENTITIES",
            ),
            ExpectedReexport(
                f"{PREFLIGHT_PACKAGE}.utils.impersonate_target",
                "get_impersonate_target",
                "get_impersonate_target",
            ),
            ExpectedReexport(
                f"{PREFLIGHT_PACKAGE}.utils.waf_result_parser",
                "parse_wafw00f_output",
                "parse_wafw00f_output",
            ),
        ),
    ),
    "utils/browser_identities.py": ExpectedShim(
        canonical_reexports=(
            ExpectedReexport(
                f"{PREFLIGHT_PACKAGE}.utils.browser_identities",
                "MODERN_BROWSER_IDENTITIES",
                "MODERN_BROWSER_IDENTITIES",
            ),
        ),
    ),
    "utils/impersonate_target.py": ExpectedShim(
        canonical_reexports=(
            ExpectedReexport(
                f"{PREFLIGHT_PACKAGE}.utils.impersonate_target",
                "get_impersonate_target",
                "get_impersonate_target",
            ),
        ),
    ),
    "utils/waf_result_parser.py": ExpectedShim(
        canonical_reexports=(
            ExpectedReexport(f"{PREFLIGHT_PACKAGE}.utils.waf_result_parser", "ANSI_RE", "ANSI_RE"),
            ExpectedReexport(
                f"{PREFLIGHT_PACKAGE}.utils.waf_result_parser",
                "GENERIC_PHRASES",
                "GENERIC_PHRASES",
            ),
            ExpectedReexport(f"{PREFLIGHT_PACKAGE}.utils.waf_result_parser", "clean_text", "clean_text"),
            ExpectedReexport(
                f"{PREFLIGHT_PACKAGE}.utils.waf_result_parser",
                "parse_wafw00f_output",
                "parse_wafw00f_output",
            ),
        ),
    ),
}


def _legacy_module_from_manifest_path(relative_path: str) -> str:
    parts = list(Path(relative_path).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join((LEGACY_PACKAGE, *parts))


PHASE0_LEGACY_MODULES = frozenset(
    _legacy_module_from_manifest_path(relative_path) for relative_path in EXPECTED_SHIM_MANIFEST
)


@dataclass(frozen=True)
class ImportReference:
    path: Path
    line: int
    module: str
    imported_name: str | None
    is_relative: bool = False

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
                ImportReference(
                    path=path,
                    line=node.lineno,
                    module=module,
                    imported_name=alias.name,
                    is_relative=node.level > 0,
                )
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


def _analyzer_import_is_allowed(reference: ImportReference) -> bool:
    if reference.is_relative:
        return _matches_module(reference.module, PREFLIGHT_PACKAGE)
    return reference.module in ALLOWED_ANALYZER_IMPORT_MODULES


def _forbidden_asyncio_direct_io_references(path: Path, tree: ast.AST) -> list[str]:
    # Conservative by design: spelling any direct-I/O constructor requires review. This
    # avoids maintaining a partial evaluator for Python's binding semantics.
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in FORBIDDEN_ASYNCIO_DIRECT_IO_APIS:
                    violations.append(
                        f"{_display_path(path)}:{node.lineno} imports forbidden asyncio direct-I/O API {alias.name}"
                    )
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_ASYNCIO_DIRECT_IO_APIS:
            violations.append(
                f"{_display_path(path)}:{node.lineno} references forbidden asyncio direct-I/O API {node.id}"
            )
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ASYNCIO_DIRECT_IO_APIS:
            violations.append(
                f"{_display_path(path)}:{node.lineno} references forbidden asyncio direct-I/O API {node.attr}"
            )
    return violations


def assert_analyzer_dependencies_are_governed(root: Path) -> None:
    violations: list[str] = []
    for path in _python_files(root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for reference in _import_references(path):
            if not _analyzer_import_is_allowed(reference):
                violations.append(f"{reference.display()} (not in the analyzer import allowlist)")
        violations.extend(_forbidden_asyncio_direct_io_references(path, tree))

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
    shim_label = f"scraper_analyzers/{relative_path}"
    legacy_module = _shim_module_name(path, root)
    expected_shim = EXPECTED_SHIM_MANIFEST.get(relative_path)
    allowed_children = expected_shim.child_modules if expected_shim is not None else frozenset()
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
            violations.append(f"{shim_label}:{statement.lineno} imports a noncanonical or side-effect module")
            continue
        if isinstance(statement, ast.ImportFrom):
            if any(alias.name == "*" for alias in statement.names):
                violations.append(f"{shim_label}:{statement.lineno} uses a wildcard re-export")
                continue

            source_module = _resolve_from_module_name(
                legacy_module,
                path.name == "__init__.py",
                statement,
            )
            is_child_import = statement.level == 1 and statement.module is None
            if allowed_children and is_child_import:
                for alias in statement.names:
                    bound_name = alias.asname or alias.name
                    if alias.asname is not None or alias.name not in allowed_children:
                        violations.append(f"{shim_label}:{statement.lineno} imports unexpected child shim {alias.name}")
                    child_exports.add(bound_name)
                    imported_bindings.append(bound_name)
                continue
            if allowed_children:
                violations.append(
                    f"{shim_label}:{statement.lineno} package aggregator may import only its "
                    "known relative child shims"
                )
                continue
            if not _matches_module(source_module, PREFLIGHT_PACKAGE):
                violations.append(f"{shim_label}:{statement.lineno} imports noncanonical source {source_module}")
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
            f"{shim_label}:{getattr(statement, 'lineno', '?')} " f"contains disallowed {type(statement).__name__}"
        )

    if all_assignments != 1:
        violations.append(f"{shim_label} must contain exactly one explicit __all__ assignment")
    elif all_names is not None:
        duplicate_exports = sorted(name for name in set(all_names) if all_names.count(name) > 1)
        if duplicate_exports:
            violations.append(f"{shim_label} repeats __all__ exports: {', '.join(duplicate_exports)}")
        missing_exports = sorted(set(imported_bindings) - set(all_names))
        unexpected_exports = sorted(set(all_names) - set(imported_bindings))
        if missing_exports or unexpected_exports:
            violations.append(
                f"{shim_label} __all__ disagrees with imported bindings "
                f"(missing: {missing_exports}; unexpected: {unexpected_exports})"
            )

    if child_exports != allowed_children:
        violations.append(
            f"{shim_label} child shim set differs "
            f"(missing: {sorted(allowed_children - child_exports)}; "
            f"unexpected: {sorted(child_exports - allowed_children)})"
        )

    if expected_shim is None:
        violations.append(f"{shim_label} is not present in the expected shim manifest")
    else:
        actual_reexports = sorted(
            (export.canonical_module, export.imported_name, export.bound_name) for export in canonical_exports
        )
        expected_reexports = sorted(
            (export.canonical_module, export.imported_name, export.bound_name)
            for export in expected_shim.canonical_reexports
        )
        if actual_reexports != expected_reexports:
            violations.append(
                f"{shim_label} canonical re-export surface differs "
                f"(expected count: {len(expected_reexports)}; actual: {actual_reexports})"
            )
        if all_names is not None and set(all_names) != expected_shim.all_exports:
            violations.append(
                f"{shim_label} public export surface differs "
                f"(missing: {sorted(expected_shim.all_exports - set(all_names))}; "
                f"unexpected: {sorted(set(all_names) - expected_shim.all_exports)})"
            )

    return canonical_exports, child_exports, violations


def _inspect_shim_tree(root: Path) -> tuple[list[CanonicalReexport], list[str]]:
    paths = _python_files(root)
    actual_relative_paths = {path.relative_to(root).as_posix() for path in paths}
    expected_relative_paths = set(EXPECTED_SHIM_MANIFEST)
    violations: list[str] = []
    canonical_exports: list[CanonicalReexport] = []
    for path in paths:
        path_exports, _, path_violations = _inspect_shim(path, root)
        canonical_exports.extend(path_exports)
        violations.extend(path_violations)

    missing_paths = sorted(expected_relative_paths - actual_relative_paths)
    unexpected_paths = sorted(actual_relative_paths - expected_relative_paths)
    if missing_paths:
        violations.append(f"{_display_path(root)} is missing expected shims: {missing_paths}")
    if unexpected_paths:
        violations.append(f"{_display_path(root)} has unexpected shims: {unexpected_paths}")
    return canonical_exports, violations


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
        "import asyncio as aio\n\nspawn = aio.create_subprocess_exec\n\n"
        "async def probe():\n    return await spawn('tool')\n",
        "import asyncio as aio\n\ndef outer():\n    spawn = aio.create_subprocess_shell\n\n"
        "    async def probe():\n        return await spawn('tool')\n\n    return probe\n",
        "import asyncio as aio\n\nasync def probe():\n    return await aio.create_subprocess_exec('tool')\n\n"
        "def unrelated():\n    import types as aio\n    return aio\n",
        "import os\n\ndef probe():\n    return os.fork()\n",
        "from os import forkpty as create_child\n\ndef probe():\n    return create_child()\n",
        "import asyncio\n\nasync def probe(spawn=asyncio.create_subprocess_exec):\n" "    return await spawn('tool')\n",
        "import asyncio\n\nprobe = lambda spawn=asyncio.create_subprocess_shell: spawn\n",
        "import asyncio\n\nasync def probe():\n" "    return await (spawn := asyncio.create_subprocess_exec)('tool')\n",
        "import pathlib\n\ndef probe():\n    return pathlib.Path('result.json')\n",
        "import asyncio\n\nasync def probe(protocol_factory):\n"
        "    return await asyncio.get_running_loop().subprocess_exec(protocol_factory, 'tool')\n",
        "import asyncio\n\nasync def probe(protocol_factory):\n"
        "    return await asyncio.get_running_loop().subprocess_shell(protocol_factory, 'tool')\n",
        "import asyncio\n\nasync def probe():\n" "    return await asyncio.open_connection('example.test', 443)\n",
        "import asyncio\n\nasync def probe():\n" "    return await asyncio.open_unix_connection('/tmp/service.sock')\n",
        "import asyncio\n\nasync def probe(protocol_factory):\n"
        "    return await asyncio.get_running_loop().create_connection(protocol_factory, 'example.test', 443)\n",
        "import asyncio\n\nasync def probe(protocol_factory):\n"
        "    return await asyncio.get_running_loop().create_unix_connection(protocol_factory, '/tmp/service.sock')\n",
        "import asyncio\n\nasync def probe(protocol_factory):\n"
        "    return await asyncio.get_running_loop().create_datagram_endpoint("
        "protocol_factory, remote_addr=('example.test', 53))\n",
        "import asyncio\n\nasync def probe(sock):\n"
        "    return await asyncio.get_running_loop().sock_connect(sock, ('example.test', 443))\n",
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
    _, violations = _inspect_shim_tree(SCRAPER_ANALYZERS_ROOT)

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
        (
            "runner.py",
            '"""Mutated shim."""\n\nfrom __future__ import annotations\n\n__all__ = []\n',
        ),
        (
            "runner.py",
            '"""Mutated shim."""\n\nfrom ..preflight.options import ScanDepth\n\n' '__all__ = ["ScanDepth"]\n',
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
    canonical_exports, violations = _inspect_shim_tree(SCRAPER_ANALYZERS_ROOT)
    assert violations == []

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

    for relative_path, expected_shim in EXPECTED_SHIM_MANIFEST.items():
        expected_children = expected_shim.child_modules
        if not expected_children:
            continue
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
