from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

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


def _metric_label_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    for node in ast.walk(_tree(path)):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if keyword.arg != "labels" or not isinstance(keyword.value, ast.Dict):
                continue
            for key in keyword.value.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.add(key.value)
    return keys


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
    assert _imported_names_at_any_scope(ENHANCED_SCRAPER_PATH, legacy_module) == {"scrape_article"}


def test_phase4b_crawl_bound_article_helper_keeps_its_async_surface_and_canonical_dependency() -> None:
    extraction_module = "tldw_Server_API.app.core.Web_Scraping.extraction"
    helper = _function(ARTICLE_PATH, "scrape_article_async")

    assert isinstance(helper, ast.AsyncFunctionDef)
    assert [argument.arg for argument in helper.args.args[:2]] == ["context", "url"]
    assert "extract_article_with_pipeline" in _imported_names(ARTICLE_PATH, extraction_module)
    assert "extract_article_with_pipeline" in _called_names(helper)


def test_phase4b_extraction_metrics_use_only_safe_label_keys() -> None:
    forbidden = {"url", "base_url", "host", "error", "pattern", "payload", "secret", "hash"}
    metric_label_keys = set().union(*(_metric_label_keys(path) for path in EXTRACTION_ROOT.rglob("*.py")))

    assert metric_label_keys.isdisjoint(forbidden)


def test_phase4b_extraction_never_recovers_cancelled_error_in_exception_tuples() -> None:
    violations: list[str] = []
    for path in EXTRACTION_ROOT.rglob("*.py"):
        for node in ast.walk(_tree(path)):
            if not isinstance(node, ast.ExceptHandler) or not isinstance(node.type, ast.Tuple):
                continue
            for exception_type in node.type.elts:
                if (
                    isinstance(exception_type, ast.Attribute)
                    and isinstance(exception_type.value, ast.Name)
                    and exception_type.value.id == "asyncio"
                    and exception_type.attr == "CancelledError"
                ):
                    violations.append(str(path.relative_to(EXTRACTION_ROOT)))

    assert violations == []


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
